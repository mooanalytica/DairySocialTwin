"""Strict, resumable S05 provisional-evidence review-video renderer.

Each video concatenates only a source-end window and a target-start window;
the long temporal gap is never decoded or rendered.  The module is
proposal-only and has no identity-changing API or encoder fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import stat
import subprocess
import time
from typing import Any, Callable, Mapping, Sequence
import uuid

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from cowtrack.config import ContractError
from cowtrack.linking.dataset_contract import (
    EXPECTED_CLIP_ORDER,
    EXPECTED_SEQUENCE_ID,
)
from cowtrack.linking.long_proposal_config import load_long_proposal_config
from cowtrack.linking.s04_runtime import S04FinalizedBundle, load_s04_finalized
from cowtrack.qa.ffprobe import QaMp4Metadata, validate_qa_mp4
from cowtrack.qa.nvenc import NvencVideoWriter
from cowtrack.qa.s05_review_config import S05ReviewConfig, load_s05_review_config
from cowtrack.qa.s05_review_plan import (
    CANDIDATE_COLUMNS,
    PURPLE_STABLE_POLICY,
    S05ReviewCase,
    S05ReviewPlan,
    build_s05_review_plan,
)
from cowtrack.qa.s05_review_render import (
    INTERMEDIATE_BGR,
    SOURCE_BGR,
    TARGET_BGR,
    UNRELATED_BGR,
    render_s05_review_frame,
)
from cowtrack.schemas.detections import DETECTIONS_SCHEMA
from cowtrack.schemas.frames import FRAMES_SCHEMA
from cowtrack.schemas.s05_proposals import (
    LONG_CANDIDATE_EDGES_SCHEMA,
    LONG_LINK_PROPOSALS_SCHEMA,
)
from cowtrack.video import open_raw_video_capture


LogFn = Callable[[str], None]
CaptureFactory = Callable[[Path], cv2.VideoCapture]

EXPECTED_FRAME_RATE = Fraction(30_000, 1_001)
EXPECTED_FRAME_PERIOD_SEC = float(1 / EXPECTED_FRAME_RATE)
EXPECTED_CLIP_IDS = EXPECTED_CLIP_ORDER
RAW_WIDTH = 3_840
RAW_HEIGHT = 2_160
MANIFEST_NAME = "review_manifest.json"
SUCCESS_NAME = "_SUCCESS.json"
MANIFEST_SCHEMA_VERSION = "cowtrack.s05-video-review.v1"
SUCCESS_SCHEMA_VERSION = "cowtrack.s05-video-review-success.v1"

_PROPOSAL_OUTPUTS = (
    "long_candidate_edges.parquet",
    "long_link_proposals.parquet",
    "s05_review_manifest.json",
    "s05_proposal_report.json",
    "effective_config.json",
)
_PROPOSAL_MARKER_KEYS = {
    "schema_version",
    "stage",
    "config_hash",
    "execution_mode",
    "automatic_merge_allowed",
    "confirmed_links_allowed",
    "solver_used",
    "path_cover_used",
    "num_merges",
    "input_fingerprints",
    "output_fingerprints",
    "stats",
    "elapsed_sec",
}
_PROPOSAL_STATS_KEYS = {
    "num_candidates",
    "num_proposals",
    "num_rejects",
    "num_confirmed",
    "num_solver_selected",
    "num_merges",
}


def log(message: str) -> None:
    print(message, flush=True)


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value}")


@dataclass(frozen=True)
class CandidateEndpoint:
    candidate_id: str
    source_stable_id: int
    target_stable_id: int
    source_start_global_frame: int
    source_end_global_frame: int
    target_start_global_frame: int
    target_end_global_frame: int
    source_start_time_sec: float
    source_end_time_sec: float
    target_start_time_sec: float
    target_end_time_sec: float
    source_end_clip_id: str
    target_start_clip_id: str


@dataclass(frozen=True)
class RenderCase:
    case: S05ReviewCase
    endpoint: CandidateEndpoint
    competing_stable_ids: tuple[int, ...]
    source_window_start_global_frame: int
    source_window_end_global_frame: int
    target_window_start_global_frame: int
    target_window_end_global_frame: int
    output_relative_path: str

    @property
    def source_frame_count(self) -> int:
        return (
            self.source_window_end_global_frame
            - self.source_window_start_global_frame
            + 1
        )

    @property
    def target_frame_count(self) -> int:
        return (
            self.target_window_end_global_frame
            - self.target_window_start_global_frame
            + 1
        )

    @property
    def expected_frame_count(self) -> int:
        return self.source_frame_count + self.target_frame_count

    @property
    def frame_sequence(self) -> tuple[int, ...]:
        return tuple(
            range(
                self.source_window_start_global_frame,
                self.source_window_end_global_frame + 1,
            )
        ) + tuple(
            range(
                self.target_window_start_global_frame,
                self.target_window_end_global_frame + 1,
            )
        )


@dataclass(frozen=True)
class LoadedS05ReviewData:
    frames: Mapping[str, np.ndarray]
    detections: Mapping[str, np.ndarray]
    stable_id_by_detection_position: np.ndarray
    valid_detection_positions_by_frame: np.ndarray
    valid_detection_frame_offsets: np.ndarray
    candidates_by_id: Mapping[str, CandidateEndpoint]
    video_paths: Mapping[str, Path]
    plan: S05ReviewPlan
    input_fingerprints: tuple[dict[str, Any], ...]


def _read_json(path: Path, label: str) -> Any:
    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ContractError(f"cannot read {label} {path}: {exc}") from exc


def _atomic_write_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    except (OSError, TypeError, ValueError) as exc:
        temporary.unlink(missing_ok=True)
        raise ContractError(f"cannot atomically write S05 review JSON {path}: {exc}") from exc


def _sha256(
    path: Path,
    *,
    progress_interval_sec: float | None = None,
    logger: LogFn | None = None,
) -> str:
    digest = hashlib.sha256()
    completed = 0
    last_report = time.monotonic()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
                completed += len(block)
                now = time.monotonic()
                if (
                    logger is not None
                    and progress_interval_sec is not None
                    and now - last_report >= progress_interval_sec
                ):
                    logger(
                        f"[s05-review] fingerprint {path.name}: "
                        f"{completed / 1024**3:.1f} GiB"
                    )
                    last_report = now
    except OSError as exc:
        raise ContractError(f"cannot fingerprint S05 review file {path}: {exc}") from exc
    return digest.hexdigest()


def _fingerprint(
    path: Path,
    *,
    progress_interval_sec: float | None = None,
    logger: LogFn | None = None,
) -> dict[str, Any]:
    path = path.resolve()
    try:
        before = path.stat()
    except OSError as exc:
        raise ContractError(f"required S05 review file does not exist: {path}: {exc}") from exc
    if not stat.S_ISREG(before.st_mode):
        raise ContractError(f"required S05 review path is not a regular file: {path}")
    digest = _sha256(
        path,
        progress_interval_sec=progress_interval_sec,
        logger=logger,
    )
    try:
        after = path.stat()
    except OSError as exc:
        raise ContractError(f"cannot restat S05 review file {path}: {exc}") from exc
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ContractError(f"S05 review file changed while fingerprinting: {path}")
    return {
        "path": str(path),
        "size_bytes": int(before.st_size),
        "sha256": digest,
    }


def _relative_fingerprint(path: Path, output_dir: Path) -> dict[str, Any]:
    current = _fingerprint(path)
    current["path"] = str(path.resolve().relative_to(output_dir.resolve()))
    return current


def _canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _json_exact(left: Any, right: Any) -> bool:
    """Compare JSON structures without Python's bool/int equality aliasing."""

    try:
        return _canonical_hash(left) == _canonical_hash(right)
    except (TypeError, ValueError):
        return False


def _normalize_fingerprints(
    records: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    result: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, Mapping):
            raise ContractError("S05 review fingerprint record must be a mapping")
        path, size, digest = (
            record.get("path"),
            record.get("size_bytes"),
            record.get("sha256"),
        )
        if (
            not isinstance(path, str)
            or not Path(path).is_absolute()
            or str(Path(path).resolve()) != path
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ContractError("S05 review fingerprint record is invalid")
        normalized = {"path": path, "size_bytes": size, "sha256": digest}
        if path in result and result[path] != normalized:
            raise ContractError(f"S05 review fingerprint path is inconsistent: {path}")
        result[path] = normalized
    return tuple(result[path] for path in sorted(result))


def _verify_fingerprints_unchanged(
    records: Sequence[Mapping[str, Any]],
    *,
    progress_interval_sec: float,
    logger: LogFn,
) -> None:
    for record in records:
        current = _fingerprint(
            Path(str(record["path"])),
            progress_interval_sec=progress_interval_sec,
            logger=logger if int(record["size_bytes"]) >= 1024**3 else None,
        )
        if current != dict(record):
            raise ContractError(
                f"immutable S05 review input changed: {record['path']}"
            )


def _read_parquet_table(path: Path, schema: pa.Schema, label: str) -> pa.Table:
    try:
        parquet = pq.ParquetFile(path)
        if not parquet.schema_arrow.equals(schema, check_metadata=False):
            raise ContractError(f"{label} schema mismatch: {path}")
        return pq.read_table(path)
    except ContractError:
        raise
    except (OSError, pa.ArrowException, TypeError, ValueError) as exc:
        raise ContractError(f"cannot read {label} {path}: {exc}") from exc


def _verified_marker_outputs(
    directory: Path,
    marker: Mapping[str, Any],
    required_names: Sequence[str],
    label: str,
    *,
    exact_set: bool = True,
) -> list[dict[str, Any]]:
    records = marker.get("output_fingerprints")
    if not isinstance(records, list):
        raise ContractError(f"{label} marker lacks output_fingerprints")
    by_name: dict[str, Mapping[str, Any]] = {}
    for record in records:
        if not isinstance(record, Mapping) or not isinstance(record.get("path"), str):
            raise ContractError(f"{label} output fingerprint record is invalid")
        name = str(record["path"])
        if name in by_name:
            raise ContractError(f"{label} output fingerprint paths are duplicated")
        by_name[name] = record
    if (exact_set and set(by_name) != set(required_names)) or not set(
        required_names
    ).issubset(by_name):
        raise ContractError(f"{label} output fingerprint artifact set differs")
    observed: list[dict[str, Any]] = []
    for name in required_names:
        current = _fingerprint(directory / name)
        recorded = by_name[name]
        if (
            current["size_bytes"] != recorded.get("size_bytes")
            or current["sha256"] != recorded.get("sha256")
        ):
            raise ContractError(f"completed {label} artifact changed: {name}")
        observed.append(current)
    observed.append(_fingerprint(directory / SUCCESS_NAME))
    return observed


def _recorded_input_snapshot(marker: Mapping[str, Any], label: str) -> list[dict[str, Any]]:
    values = marker.get("input_fingerprints")
    if not isinstance(values, list) or not values:
        raise ContractError(f"{label} marker input_fingerprints must be non-empty")
    observed: list[dict[str, Any]] = []
    paths: set[str] = set()
    for value in values:
        if not isinstance(value, Mapping) or set(value) != {
            "path",
            "size_bytes",
            "sha256",
        }:
            raise ContractError(f"{label} input fingerprint record is invalid")
        raw_path = value["path"]
        if not isinstance(raw_path, str) or raw_path in paths:
            raise ContractError(f"{label} input fingerprint paths are invalid")
        path = Path(raw_path)
        if not path.is_absolute() or str(path.resolve()) != raw_path:
            raise ContractError(f"{label} input path is not canonical: {raw_path}")
        current = _fingerprint(path)
        if (
            current["size_bytes"] != value["size_bytes"]
            or current["sha256"] != value["sha256"]
        ):
            raise ContractError(f"recorded {label} input changed: {path}")
        observed.append(current)
        paths.add(raw_path)
    return observed


def _load_s00(
    ingest_dir: Path,
    *,
    progress_interval_sec: float,
    logger: LogFn,
) -> tuple[
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, Path],
    list[dict[str, Any]],
]:
    marker = _read_json(ingest_dir / SUCCESS_NAME, "S00 marker")
    if not isinstance(marker, dict) or marker.get("stage") != "S00":
        raise ContractError("S05 review requires a completed S00 ingest directory")
    observed = _verified_marker_outputs(
        ingest_dir,
        marker,
        ("frames.parquet", "detections.parquet", "resolved_manifest.json"),
        "S00",
        exact_set=False,
    )
    frames_table = _read_parquet_table(
        ingest_dir / "frames.parquet", FRAMES_SCHEMA, "S00 frames"
    )
    detections_table = _read_parquet_table(
        ingest_dir / "detections.parquet", DETECTIONS_SCHEMA, "S00 detections"
    )
    frames = {
        name: frames_table[name].combine_chunks().to_numpy(zero_copy_only=False)
        for name in FRAMES_SCHEMA.names
    }
    detections = {
        name: detections_table[name].combine_chunks().to_numpy(zero_copy_only=False)
        for name in DETECTIONS_SCHEMA.names
    }
    _validate_frames(frames)
    _validate_detections(frames, detections)

    resolved = _read_json(ingest_dir / "resolved_manifest.json", "S00 resolved manifest")
    if not isinstance(resolved, list) or len(resolved) != len(EXPECTED_CLIP_IDS):
        raise ContractError("S05 review requires the complete resolved S00 clip sequence")
    input_records = {
        str(record.get("path")): record
        for record in marker.get("input_fingerprints", [])
        if isinstance(record, dict) and isinstance(record.get("path"), str)
    }
    paths: dict[str, Path] = {}
    for order, row in enumerate(resolved):
        if not isinstance(row, dict):
            raise ContractError("S00 resolved manifest row must be an object")
        if (
            row.get("sequence_id") != EXPECTED_SEQUENCE_ID
            or row.get("clip_id") != EXPECTED_CLIP_IDS[order]
            or row.get("clip_order") != order
            or not isinstance(row.get("video_path"), str)
        ):
            raise ContractError("S00 resolved manifest sequence/clip order differs")
        clip_id = EXPECTED_CLIP_IDS[order]
        video_path = Path(str(row["video_path"])).resolve()
        recorded = input_records.get(str(video_path))
        if recorded is None:
            raise ContractError(f"source video was not fingerprinted by S00: {video_path}")
        current = _fingerprint(
            video_path,
            progress_interval_sec=progress_interval_sec,
            logger=logger,
        )
        if (
            current["size_bytes"] != recorded.get("size_bytes")
            or current["sha256"] != recorded.get("sha256")
        ):
            raise ContractError(f"source video changed since S00: {video_path}")
        paths[clip_id] = video_path
        observed.append(current)
    return frames, detections, paths, observed


def _validate_frames(frames: Mapping[str, np.ndarray]) -> None:
    count = len(frames["global_frame"])
    global_frames = np.asarray(frames["global_frame"], dtype=np.int64)
    if count == 0 or not np.array_equal(global_frames, np.arange(count)):
        raise ContractError("S05 review frames.global_frame must be contiguous from zero")
    times = np.asarray(frames["global_time_sec"], dtype=np.float64)
    if not np.all(np.isfinite(times)) or not np.allclose(
        times,
        np.arange(count, dtype=np.float64) * EXPECTED_FRAME_PERIOD_SEC,
        rtol=0.0,
        atol=1e-9,
    ):
        raise ContractError("S05 review requires the exact 30000/1001 S00 timeline")
    if not np.all(np.asarray(frames["width"]) == RAW_WIDTH) or not np.all(
        np.asarray(frames["height"]) == RAW_HEIGHT
    ):
        raise ContractError("S05 review raw frames must be exactly 3840x2160")
    if set(map(str, np.unique(frames["sequence_id"]))) != {EXPECTED_SEQUENCE_ID}:
        raise ContractError("S05 review S00 sequence ID differs")
    clip_ids = np.asarray(frames["clip_id"], dtype=object)
    clip_orders = np.asarray(frames["clip_order"], dtype=np.int64)
    local_frames = np.asarray(frames["local_frame"], dtype=np.int64)
    pts = np.asarray(frames["pts_sec"], dtype=np.float64)
    observed_order: list[str] = []
    for clip_id in map(str, clip_ids):
        if not observed_order or observed_order[-1] != clip_id:
            if clip_id in observed_order:
                raise ContractError("S00 clip appears in disjoint frame ranges")
            observed_order.append(clip_id)
    if tuple(observed_order) != EXPECTED_CLIP_IDS:
        raise ContractError("S05 review S00 clip order differs")
    for clip_order, clip_id in enumerate(EXPECTED_CLIP_IDS):
        positions = np.flatnonzero(clip_ids == clip_id)
        if (
            not len(positions)
            or not np.array_equal(local_frames[positions], np.arange(len(positions)))
            or not np.all(clip_orders[positions] == clip_order)
            or not np.allclose(
                pts[positions],
                np.arange(len(positions), dtype=np.float64)
                * EXPECTED_FRAME_PERIOD_SEC,
                rtol=0.0,
                atol=1e-9,
            )
        ):
            raise ContractError(f"S00 local frames are not contiguous for {clip_id}")


def _validate_detections(
    frames: Mapping[str, np.ndarray], detections: Mapping[str, np.ndarray]
) -> None:
    det_ids = np.asarray(detections["det_id"], dtype=np.int64)
    if len(np.unique(det_ids)) != len(det_ids):
        raise ContractError("S05 review S00 detection IDs are duplicated")
    valid = np.asarray(detections["valid"])
    if valid.dtype.kind != "b":
        raise ContractError("S05 review S00 detections.valid must be boolean")
    det_frames = np.asarray(detections["global_frame"], dtype=np.int64)
    if np.any(det_frames < 0) or np.any(det_frames >= len(frames["global_frame"])):
        raise ContractError("S05 review detection references an invalid frame")
    frame_positions = det_frames
    if (
        not np.all(
            np.asarray(detections["sequence_id"], dtype=object)
            == np.asarray(frames["sequence_id"], dtype=object)[frame_positions]
        )
        or not np.all(
            np.asarray(detections["clip_id"], dtype=object)
            == np.asarray(frames["clip_id"], dtype=object)[frame_positions]
        )
        or not np.array_equal(
            np.asarray(detections["local_frame"], dtype=np.int64),
            np.asarray(frames["local_frame"], dtype=np.int64)[frame_positions],
        )
        or not np.allclose(
            np.asarray(detections["global_time_sec"], dtype=np.float64),
            np.asarray(frames["global_time_sec"], dtype=np.float64)[frame_positions],
            rtol=0.0,
            atol=1e-9,
        )
    ):
        raise ContractError("S05 review detection/frame metadata differs")
    positions = np.flatnonzero(valid)
    boxes = np.column_stack(
        [
            np.asarray(detections[name], dtype=np.float64)[positions]
            for name in ("x1", "y1", "x2", "y2")
        ]
    )
    if len(boxes) and (
        not np.all(np.isfinite(boxes))
        or np.any(boxes[:, 0] < 0.0)
        or np.any(boxes[:, 1] < 0.0)
        or np.any(boxes[:, 0] >= boxes[:, 2])
        or np.any(boxes[:, 1] >= boxes[:, 3])
        or np.any(boxes[:, 2] > RAW_WIDTH)
        or np.any(boxes[:, 3] > RAW_HEIGHT)
    ):
        raise ContractError("S05 review S00 valid bbox geometry is invalid")


def _validate_proposal_marker(marker: Any) -> Mapping[str, Any]:
    if not isinstance(marker, dict) or set(marker) != _PROPOSAL_MARKER_KEYS:
        raise ContractError("S05_PROPOSE marker fields differ")
    if (
        marker.get("schema_version") != "1.0"
        or marker.get("stage") != "S05_PROPOSE"
        or marker.get("execution_mode") != "long_proposal_only"
        or marker.get("automatic_merge_allowed") is not False
        or marker.get("confirmed_links_allowed") is not False
        or marker.get("solver_used") is not False
        or marker.get("path_cover_used") is not False
        or type(marker.get("num_merges")) is not int
        or marker.get("num_merges") != 0
    ):
        raise ContractError("S05_PROPOSE marker violates proposal-only policy")
    config_hash = marker.get("config_hash")
    if (
        not isinstance(config_hash, str)
        or len(config_hash) != 64
        or any(character not in "0123456789abcdef" for character in config_hash)
    ):
        raise ContractError("S05_PROPOSE marker config hash is invalid")
    elapsed = marker.get("elapsed_sec")
    if (
        isinstance(elapsed, bool)
        or not isinstance(elapsed, (int, float))
        or not math.isfinite(float(elapsed))
        or float(elapsed) < 0.0
    ):
        raise ContractError("S05_PROPOSE marker elapsed_sec is invalid")
    stats = marker.get("stats")
    if not isinstance(stats, dict) or set(stats) != _PROPOSAL_STATS_KEYS:
        raise ContractError("S05_PROPOSE marker stats fields differ")
    for name in _PROPOSAL_STATS_KEYS:
        value = stats[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ContractError(f"S05_PROPOSE marker stat {name} is invalid")
    if any(stats[name] != 0 for name in ("num_confirmed", "num_solver_selected", "num_merges")):
        raise ContractError("S05_PROPOSE marker contains identity-changing results")
    if stats["num_rejects"] != stats["num_candidates"] - stats["num_proposals"]:
        raise ContractError("S05_PROPOSE candidate/proposal/reject counts disagree")

    inputs = marker.get("input_fingerprints")
    if not isinstance(inputs, list) or not inputs:
        raise ContractError("S05_PROPOSE input_fingerprints must be non-empty")
    input_paths: list[str] = []
    for record in inputs:
        if not isinstance(record, dict) or set(record) != {
            "path",
            "size_bytes",
            "sha256",
        }:
            raise ContractError("S05_PROPOSE input fingerprint is invalid")
        path, size, digest = (
            record["path"],
            record["size_bytes"],
            record["sha256"],
        )
        if (
            not isinstance(path, str)
            or not Path(path).is_absolute()
            or str(Path(path).resolve()) != path
            or type(size) is not int
            or size < 0
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ContractError("S05_PROPOSE input fingerprint fields are invalid")
        input_paths.append(path)
    if input_paths != sorted(set(input_paths)):
        raise ContractError("S05_PROPOSE input fingerprints are not canonical")

    outputs = marker.get("output_fingerprints")
    if not isinstance(outputs, list):
        raise ContractError("S05_PROPOSE output_fingerprints must be a list")
    output_paths: list[str] = []
    for record in outputs:
        if not isinstance(record, dict) or set(record) != {
            "path",
            "size_bytes",
            "sha256",
        }:
            raise ContractError("S05_PROPOSE output fingerprint is invalid")
        path, size, digest = (
            record["path"],
            record["size_bytes"],
            record["sha256"],
        )
        if (
            not isinstance(path, str)
            or Path(path).is_absolute()
            or len(Path(path).parts) != 1
            or type(size) is not int
            or size < 0
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ContractError("S05_PROPOSE output fingerprint fields are invalid")
        output_paths.append(path)
    if output_paths != sorted(_PROPOSAL_OUTPUTS):
        raise ContractError("S05_PROPOSE output fingerprints are not canonical")
    return marker


def _load_proposals(
    proposals_dir: Path,
    config: S05ReviewConfig,
) -> tuple[
    S05ReviewPlan,
    dict[str, dict[str, Any]],
    list[dict[str, Any]],
]:
    entries = list(proposals_dir.iterdir())
    if any(path.is_symlink() or not path.is_file() for path in entries):
        raise ContractError("completed S05_PROPOSE artifact tree differs")
    actual_names = {path.name for path in entries}
    if actual_names != {*_PROPOSAL_OUTPUTS, SUCCESS_NAME}:
        raise ContractError("completed S05_PROPOSE artifact tree differs")
    marker = _validate_proposal_marker(
        _read_json(proposals_dir / SUCCESS_NAME, "S05_PROPOSE marker")
    )
    observed = _verified_marker_outputs(
        proposals_dir, marker, _PROPOSAL_OUTPUTS, "S05_PROPOSE"
    )
    observed.extend(_recorded_input_snapshot(marker, "S05_PROPOSE"))
    proposal_config, _, proposal_hash = load_long_proposal_config(
        proposals_dir / "effective_config.json"
    )
    if proposal_hash != marker["config_hash"]:
        raise ContractError("S05_PROPOSE effective config/hash differs")
    if (
        proposal_config.global_merge_allowed
        or proposal_config.solver_allowed
        or proposal_config.path_cover_allowed
        or proposal_config.confirmed_links_allowed
        or proposal_config.human_labels_applied
    ):
        raise ContractError("S05_PROPOSE effective config permits identity changes")

    candidate_table = _read_parquet_table(
        proposals_dir / "long_candidate_edges.parquet",
        LONG_CANDIDATE_EDGES_SCHEMA,
        "S05 long candidate edges",
    )
    proposal_table = _read_parquet_table(
        proposals_dir / "long_link_proposals.parquet",
        LONG_LINK_PROPOSALS_SCHEMA,
        "S05 long link proposals",
    )
    candidates = candidate_table.to_pylist()
    proposals = proposal_table.to_pylist()
    stats = marker["stats"]
    if len(candidates) != stats["num_candidates"] or len(proposals) != stats[
        "num_proposals"
    ]:
        raise ContractError("S05_PROPOSE Parquet row counts differ from marker")
    candidate_by_id: dict[str, dict[str, Any]] = {}
    proposed_rows: list[dict[str, Any]] = []
    observed_candidate_order: list[str] = []
    observed_pairs: set[tuple[int, int]] = set()
    for row in candidates:
        candidate_id = row.get("candidate_id")
        if not isinstance(candidate_id, str) or not candidate_id or candidate_id in candidate_by_id:
            raise ContractError("S05 candidate IDs are blank or duplicated")
        source_id, target_id = row.get("source_stable_id"), row.get("target_stable_id")
        if (
            isinstance(source_id, bool)
            or not isinstance(source_id, int)
            or source_id < 0
            or isinstance(target_id, bool)
            or not isinstance(target_id, int)
            or target_id < 0
            or source_id == target_id
            or candidate_id != f"s05c-{source_id:06d}-{target_id:06d}"
            or (source_id, target_id) in observed_pairs
        ):
            raise ContractError("S05 candidate ID/pair identity is invalid")
        observed_candidate_order.append(candidate_id)
        observed_pairs.add((source_id, target_id))
        candidate_by_id[candidate_id] = row
        if (
            row.get("selected_by_solver") is not False
            or row.get("confirmed") is not False
            or row.get("merge_applied") is not False
            or row.get("decision") == "confirmed"
        ):
            raise ContractError("S05 proposal candidate contains identity-changing state")
        if row.get("decision") not in {"reject", "provisional"}:
            raise ContractError("S05 candidate decision is not reject/provisional")
        proposed = row.get("decision") == "provisional"
        if row.get("proposed_for_review") is not proposed:
            raise ContractError("S05 candidate proposed_for_review flag differs")
        if proposed:
            proposed_rows.append(row)
    if observed_candidate_order != sorted(observed_candidate_order):
        raise ContractError("S05 candidates are not in canonical candidate_id order")
    proposal_ids: set[str] = set()
    proposal_candidate_ids: set[str] = set()
    observed_proposal_order: list[str] = []
    shared_fields = set(LONG_CANDIDATE_EDGES_SCHEMA.names).intersection(
        LONG_LINK_PROPOSALS_SCHEMA.names
    )
    for row in proposals:
        proposal_id, candidate_id = row.get("proposal_id"), row.get("candidate_id")
        if (
            not isinstance(proposal_id, str)
            or not proposal_id
            or proposal_id in proposal_ids
            or not isinstance(candidate_id, str)
            or candidate_id in proposal_candidate_ids
            or row.get("review_status") != "pending"
            or proposal_id != f"s05p-{candidate_id[5:]}"
        ):
            raise ContractError("S05 proposal IDs/status are invalid")
        source = candidate_by_id.get(candidate_id)
        if source is None or source.get("proposed_for_review") is not True:
            raise ContractError("S05 proposal references a non-provisional candidate")
        for name in shared_fields:
            if row.get(name) != source.get(name):
                raise ContractError(f"S05 proposal/candidate field differs: {name}")
        expected_evidence_status = (
            "selected_gate_uncertified"
            if source.get("passes_selected_gate") is True
            else "provisional"
        )
        if row.get("evidence_status") != expected_evidence_status:
            raise ContractError("S05 proposal evidence_status differs")
        proposal_ids.add(proposal_id)
        proposal_candidate_ids.add(candidate_id)
        observed_proposal_order.append(proposal_id)
    if observed_proposal_order != sorted(observed_proposal_order):
        raise ContractError("S05 proposals are not in canonical proposal_id order")
    if proposal_candidate_ids != {str(row["candidate_id"]) for row in proposed_rows}:
        raise ContractError("S05 provisional candidate/proposal sets differ")
    if not candidates:
        raise ContractError("S05 review requires at least one scored candidate")
    thresholds = {float(row["provisional_threshold"]) for row in candidates}
    if len(thresholds) != 1:
        raise ContractError("S05 candidates use different provisional thresholds")
    selected_indices = [
        index for index, row in enumerate(candidates) if row["proposed_for_review"]
    ]
    projection = {
        name: np.asarray(
            [candidates[index][name] for index in selected_indices], dtype=object
        )
        for name in CANDIDATE_COLUMNS
    }
    plan = build_s05_review_plan(
        projection,
        provisional_threshold=next(iter(thresholds)),
        random_seed=config.random_seed,
        maximum_cases=config.maximum_cases,
        high_score_count=config.high_score_count,
        threshold_near_count=config.threshold_near_count,
        ambiguous_count=config.ambiguous_count,
        random_count=config.random_count,
        threshold_probability_band=config.threshold_probability_band,
        ambiguity_margin_upper=config.ambiguity_margin_upper,
        ambiguous_mutual_rank_above=config.ambiguous_mutual_rank_above,
    )
    return plan, candidate_by_id, observed


def _build_detection_indices(
    frames: Mapping[str, np.ndarray],
    detections: Mapping[str, np.ndarray],
    stable: S04FinalizedBundle,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Join S00 valid detections to the immutable S04 stable mapping by det_id."""

    det_ids = np.asarray(detections["det_id"], dtype=np.int64)
    valid = np.asarray(detections["valid"], dtype=np.bool_)
    valid_positions = np.flatnonzero(valid)
    stable_det_ids = np.asarray(stable.det_ids, dtype=np.int64)
    stable_ids = np.asarray(stable.det_stable_ids, dtype=np.int64)
    if len(stable_det_ids) != len(stable_ids) or len(valid_positions) != len(
        stable_det_ids
    ):
        raise ContractError("S00 valid detection count differs from finalized S04")
    if len(stable_det_ids) and (
        len(np.unique(stable_det_ids)) != len(stable_det_ids)
        or np.any(stable_det_ids[1:] <= stable_det_ids[:-1])
    ):
        raise ContractError("finalized S04 det_id mapping is not unique/canonical")

    valid_det_ids = det_ids[valid_positions]
    order = np.argsort(valid_det_ids, kind="stable")
    if not np.array_equal(valid_det_ids[order], stable_det_ids):
        raise ContractError("S00 valid det_id set differs from finalized S04")
    stable_id_by_position = np.full(len(det_ids), -1, dtype=np.int64)
    stable_id_by_position[valid_positions[order]] = stable_ids

    det_frames = np.asarray(detections["global_frame"], dtype=np.int64)
    canonical = valid_positions[
        np.lexsort((valid_det_ids, det_frames[valid_positions]))
    ]
    frame_count = len(frames["global_frame"])
    counts = np.bincount(det_frames[canonical], minlength=frame_count)
    if len(counts) != frame_count:
        raise ContractError("S05 review valid detection frame index differs")
    offsets = np.r_[0, np.cumsum(counts, dtype=np.int64)]
    stable_id_by_position.setflags(write=False)
    canonical.setflags(write=False)
    offsets.setflags(write=False)
    return stable_id_by_position, canonical, offsets


def _same_float(left: Any, right: Any) -> bool:
    try:
        return math.isclose(
            float(left), float(right), rel_tol=0.0, abs_tol=1e-9
        )
    except (TypeError, ValueError, OverflowError):
        return False


def _validate_tracklet_projection(
    row: Mapping[str, Any],
    prefix: str,
    stable_id: int,
    stable: S04FinalizedBundle,
) -> None:
    tracklet = stable.stable_tracklets.get(stable_id)
    constituents = stable.stable_to_micros.get(stable_id)
    if tracklet is None or constituents is None:
        raise ContractError(
            f"S05 candidate references unknown finalized stable ID {stable_id}"
        )
    expected: dict[str, Any] = {
        f"{prefix}_constituent_micro_ids": list(constituents),
        f"{prefix}_start_det_id": tracklet.start_det_id,
        f"{prefix}_end_det_id": tracklet.end_det_id,
        f"{prefix}_start_clip_id": tracklet.start_clip_id,
        f"{prefix}_end_clip_id": tracklet.end_clip_id,
        f"{prefix}_start_global_frame": tracklet.start_global_frame,
        f"{prefix}_end_global_frame": tracklet.end_global_frame,
        f"{prefix}_num_detections": tracklet.num_detections,
    }
    for name, value in expected.items():
        observed = row.get(name)
        if name.endswith("_constituent_micro_ids"):
            try:
                observed = list(map(int, observed))
            except (TypeError, ValueError, OverflowError):
                pass
        if observed != value:
            raise ContractError(
                f"S05 candidate {row.get('candidate_id')} field {name} differs "
                "from finalized S04"
            )
    for suffix, value in (
        ("start_time_sec", tracklet.start_time_sec),
        ("end_time_sec", tracklet.end_time_sec),
    ):
        name = f"{prefix}_{suffix}"
        if not _same_float(row.get(name), value):
            raise ContractError(
                f"S05 candidate {row.get('candidate_id')} field {name} differs "
                "from finalized S04"
            )

    appearance = stable.stable_appearance.get(stable_id)
    if appearance is None:
        raise ContractError(
            f"finalized S04 lacks appearance provenance for stable ID {stable_id}"
        )
    present_name = f"{prefix}_gallery_present"
    if row.get(present_name) is not appearance.appearance_usable:
        raise ContractError(
            f"S05 candidate {row.get('candidate_id')} field {present_name} differs "
            "from finalized S04"
        )
    gallery_fields = (
        "sample_ids",
        "gallery_det_ids",
        "embedding_rows",
        "medoid_sample_id",
        "appearance_quality",
        "internal_cosine_p10",
        "internal_cosine_p50",
        "internal_cosine_min",
        "gallery_num_input_samples",
        "gallery_num_overlap_rejected",
        "gallery_num_review_excluded",
        "gallery_num_local_outliers",
        "gallery_max_other_bbox_iou",
        "gallery_max_clean_other_bbox_iou",
    )
    if appearance.appearance_usable:
        expected_gallery: dict[str, Any] = {
            "sample_ids": list(appearance.clean_sample_ids),
            "gallery_det_ids": list(appearance.clean_det_ids),
            "embedding_rows": list(appearance.clean_embedding_rows),
            "medoid_sample_id": appearance.medoid_sample_id,
            "appearance_quality": appearance.appearance_quality,
            "internal_cosine_p10": appearance.internal_cosine_p10,
            "internal_cosine_p50": appearance.internal_cosine_p50,
            "internal_cosine_min": appearance.internal_cosine_min,
            "gallery_num_input_samples": appearance.num_input_samples,
            "gallery_num_overlap_rejected": appearance.num_overlap_rejected,
            "gallery_num_review_excluded": appearance.num_review_excluded,
            "gallery_num_local_outliers": appearance.num_local_outliers,
            "gallery_max_other_bbox_iou": appearance.max_other_bbox_iou,
            "gallery_max_clean_other_bbox_iou": (
                appearance.max_clean_other_bbox_iou
            ),
        }
    else:
        expected_gallery = {name: None for name in gallery_fields}
    float_fields = {
        "appearance_quality",
        "internal_cosine_p10",
        "internal_cosine_p50",
        "internal_cosine_min",
        "gallery_max_other_bbox_iou",
        "gallery_max_clean_other_bbox_iou",
    }
    list_fields = {"sample_ids", "gallery_det_ids", "embedding_rows"}
    for suffix in gallery_fields:
        name = f"{prefix}_{suffix}"
        observed, expected_value = row.get(name), expected_gallery[suffix]
        if suffix in list_fields and observed is not None:
            try:
                observed = list(map(int, observed))
            except (TypeError, ValueError, OverflowError):
                pass
        agrees = (
            _same_float(observed, expected_value)
            if suffix in float_fields and expected_value is not None
            else observed == expected_value
        )
        if not agrees:
            raise ContractError(
                f"S05 candidate {row.get('candidate_id')} field {name} differs "
                "from finalized S04 gallery provenance"
            )


def _candidate_endpoints(
    candidate_by_id: Mapping[str, Mapping[str, Any]],
    stable: S04FinalizedBundle,
    frames: Mapping[str, np.ndarray],
    detections: Mapping[str, np.ndarray],
    stable_id_by_detection_position: np.ndarray,
    plan: S05ReviewPlan,
) -> dict[str, CandidateEndpoint]:
    frame_count = len(frames["global_frame"])
    frame_clip = np.asarray(frames["clip_id"], dtype=object)
    frame_times = np.asarray(frames["global_time_sec"], dtype=np.float64)
    det_ids = np.asarray(detections["det_id"], dtype=np.int64)
    det_frames = np.asarray(detections["global_frame"], dtype=np.int64)
    det_order = np.argsort(det_ids, kind="stable")
    sorted_det_ids = det_ids[det_order]

    def validate_endpoint_detection(
        *, det_id: int, stable_id: int, global_frame: int, label: str
    ) -> None:
        position = int(np.searchsorted(sorted_det_ids, det_id))
        if position >= len(sorted_det_ids) or int(sorted_det_ids[position]) != det_id:
            raise ContractError(f"S05 {label} det_id is absent from S00: {det_id}")
        detection_position = int(det_order[position])
        if (
            int(stable_id_by_detection_position[detection_position]) != stable_id
            or int(det_frames[detection_position]) != global_frame
        ):
            raise ContractError(
                f"S05 {label} detection/stable/frame endpoint differs"
            )

    result: dict[str, CandidateEndpoint] = {}
    for evidence in plan.candidates:
        candidate_id = evidence.candidate_id
        row = candidate_by_id.get(candidate_id)
        if row is None:
            raise ContractError(f"S05 review plan references unknown {candidate_id}")
        source_id = evidence.source_stable_id
        target_id = evidence.target_stable_id
        if (
            row.get("source_stable_id") != source_id
            or row.get("target_stable_id") != target_id
        ):
            raise ContractError(f"S05 review plan/candidate IDs differ: {candidate_id}")
        _validate_tracklet_projection(row, "source", source_id, stable)
        _validate_tracklet_projection(row, "target", target_id, stable)

        if row.get("appearance_present") is not bool(
            row.get("source_gallery_present") is True
            and row.get("target_gallery_present") is True
        ):
            raise ContractError(
                f"S05 candidate {candidate_id} appearance-present flags differ"
            )

        source = stable.stable_tracklets[source_id]
        target = stable.stable_tracklets[target_id]
        gap = target.start_time_sec - source.end_time_sec
        if (
            source_id == target_id
            or source.end_global_frame >= target.start_global_frame
            or not math.isfinite(gap)
            or gap <= 5.0
            or not _same_float(row.get("temporal_gap_sec"), gap)
            or row.get("temporally_nonoverlapping") is not True
        ):
            raise ContractError(
                f"S05 candidate {candidate_id} is not a valid >5-second future edge"
            )
        endpoint_frames = (
            (source.start_global_frame, source.start_clip_id, source.start_time_sec),
            (source.end_global_frame, source.end_clip_id, source.end_time_sec),
            (target.start_global_frame, target.start_clip_id, target.start_time_sec),
            (target.end_global_frame, target.end_clip_id, target.end_time_sec),
        )
        for global_frame, clip_id, global_time in endpoint_frames:
            if (
                global_frame < 0
                or global_frame >= frame_count
                or str(frame_clip[global_frame]) != clip_id
                or not _same_float(frame_times[global_frame], global_time)
            ):
                raise ContractError(
                    f"S05 candidate {candidate_id} endpoint differs from S00 timeline"
                )
        validate_endpoint_detection(
            det_id=source.start_det_id,
            stable_id=source_id,
            global_frame=source.start_global_frame,
            label="source-start",
        )
        validate_endpoint_detection(
            det_id=source.end_det_id,
            stable_id=source_id,
            global_frame=source.end_global_frame,
            label="source-end",
        )
        validate_endpoint_detection(
            det_id=target.start_det_id,
            stable_id=target_id,
            global_frame=target.start_global_frame,
            label="target-start",
        )
        validate_endpoint_detection(
            det_id=target.end_det_id,
            stable_id=target_id,
            global_frame=target.end_global_frame,
            label="target-end",
        )
        result[candidate_id] = CandidateEndpoint(
            candidate_id=candidate_id,
            source_stable_id=source_id,
            target_stable_id=target_id,
            source_start_global_frame=source.start_global_frame,
            source_end_global_frame=source.end_global_frame,
            target_start_global_frame=target.start_global_frame,
            target_end_global_frame=target.end_global_frame,
            source_start_time_sec=source.start_time_sec,
            source_end_time_sec=source.end_time_sec,
            target_start_time_sec=target.start_time_sec,
            target_end_time_sec=target.end_time_sec,
            source_end_clip_id=source.end_clip_id,
            target_start_clip_id=target.start_clip_id,
        )
    if set(result) != {item.candidate_id for item in plan.candidates}:
        raise ContractError("S05 review candidate endpoint set differs")
    return result


def _load_data(
    proposals_dir: Path,
    ingest_dir: Path,
    stable_dir: Path,
    config_path: Path,
    config: S05ReviewConfig,
    logger: LogFn,
    *,
    config_fingerprint: Mapping[str, Any],
) -> LoadedS05ReviewData:
    frames, detections, video_paths, observed = _load_s00(
        ingest_dir,
        progress_interval_sec=config.progress_interval_sec,
        logger=logger,
    )
    stable = load_s04_finalized(stable_dir)
    stable_by_detection, valid_positions, offsets = _build_detection_indices(
        frames, detections, stable
    )
    plan, candidate_by_id, proposal_observed = _load_proposals(
        proposals_dir, config
    )
    endpoints = _candidate_endpoints(
        candidate_by_id,
        stable,
        frames,
        detections,
        stable_by_detection,
        plan,
    )
    fingerprints = _normalize_fingerprints(
        [
            *observed,
            *(item.as_dict() for item in stable.input_fingerprints),
            *proposal_observed,
            dict(config_fingerprint),
        ]
    )
    return LoadedS05ReviewData(
        frames=frames,
        detections=detections,
        stable_id_by_detection_position=stable_by_detection,
        valid_detection_positions_by_frame=valid_positions,
        valid_detection_frame_offsets=offsets,
        candidates_by_id=endpoints,
        video_paths=video_paths,
        plan=plan,
        input_fingerprints=fingerprints,
    )


def _render_cases(
    data: LoadedS05ReviewData, config: S05ReviewConfig
) -> tuple[RenderCase, ...]:
    times = np.asarray(data.frames["global_time_sec"], dtype=np.float64)
    result: list[RenderCase] = []
    seen_paths: set[str] = set()
    for case in data.plan.cases:
        endpoint = data.candidates_by_id.get(case.candidate.candidate_id)
        if endpoint is None:
            raise ContractError(
                f"S05 review lacks endpoint for {case.candidate.candidate_id}"
            )
        source_start = max(
            0,
            int(
                np.searchsorted(
                    times,
                    endpoint.source_end_time_sec - config.source_tail_sec,
                    side="left",
                )
            ),
        )
        target_end = min(
            len(times) - 1,
            int(
                np.searchsorted(
                    times,
                    endpoint.target_start_time_sec + config.target_head_sec,
                    side="right",
                )
                - 1
            ),
        )
        if not (
            0 <= source_start <= endpoint.source_end_global_frame
            < endpoint.target_start_global_frame <= target_end < len(times)
        ):
            raise ContractError(
                f"S05 candidate {case.candidate.candidate_id} has invalid review windows"
            )
        output_path = f"videos/{case.suggested_filename}"
        if output_path in seen_paths:
            raise ContractError(f"duplicate S05 review output path: {output_path}")
        seen_paths.add(output_path)
        result.append(
            RenderCase(
                case=case,
                endpoint=endpoint,
                competing_stable_ids=data.plan.competing_stable_ids(case),
                source_window_start_global_frame=source_start,
                source_window_end_global_frame=endpoint.source_end_global_frame,
                target_window_start_global_frame=endpoint.target_start_global_frame,
                target_window_end_global_frame=target_end,
                output_relative_path=output_path,
            )
        )
    if len(result) > config.maximum_cases:
        raise ContractError("S05 review plan exceeds maximum_cases")
    return tuple(result)


class RawFrameReader:
    """Exact seek/sequential decoder for raw S00 frames with autorotate disabled."""

    def __init__(
        self,
        video_paths: Mapping[str, Path],
        *,
        raw_width: int,
        raw_height: int,
        capture_factory: CaptureFactory = open_raw_video_capture,
    ) -> None:
        self._video_paths = dict(video_paths)
        self._raw_width = raw_width
        self._raw_height = raw_height
        self._capture_factory = capture_factory
        self._capture: cv2.VideoCapture | None = None
        self._clip_id: str | None = None
        self._next_local_frame: int | None = None

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
        self._capture = None
        self._clip_id = None
        self._next_local_frame = None

    def __enter__(self) -> RawFrameReader:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool:
        self.close()
        return False

    def read(self, clip_id: str, local_frame: int, expected_pts_sec: float) -> np.ndarray:
        if clip_id not in self._video_paths:
            raise ContractError(f"frame references unknown S00 clip_id: {clip_id}")
        if self._capture is None or self._clip_id != clip_id:
            self.close()
            self._capture = self._capture_factory(self._video_paths[clip_id])
            self._clip_id = clip_id
            self._next_local_frame = None
        assert self._capture is not None
        if self._next_local_frame != local_frame and not self._capture.set(
            cv2.CAP_PROP_POS_FRAMES, int(local_frame)
        ):
            raise ContractError(f"cannot seek {clip_id} local frame {local_frame}")
        ok, frame = self._capture.read()
        if not ok or frame is None:
            raise ContractError(f"cannot decode {clip_id} local frame {local_frame}")
        if frame.shape != (self._raw_height, self._raw_width, 3) or frame.dtype != np.uint8:
            raise ContractError(
                f"decoded raw frame geometry/type mismatch at {clip_id}:{local_frame}"
            )
        position_after = int(round(self._capture.get(cv2.CAP_PROP_POS_FRAMES)))
        if position_after != local_frame + 1:
            raise ContractError(f"OpenCV frame-position mismatch at {clip_id}:{local_frame}")
        if not hasattr(cv2, "CAP_PROP_PTS"):
            raise ContractError("OpenCV lacks CAP_PROP_PTS for exact S05 review decoding")
        decoded_pts = int(round(self._capture.get(cv2.CAP_PROP_PTS)))
        if decoded_pts != local_frame:
            raise ContractError(f"OpenCV PTS-frame mismatch at {clip_id}:{local_frame}")
        decoded_msec = float(self._capture.get(cv2.CAP_PROP_POS_MSEC))
        if not math.isclose(
            decoded_msec, expected_pts_sec * 1000.0, rel_tol=0.0, abs_tol=0.1
        ):
            raise ContractError(f"OpenCV PTS-time mismatch at {clip_id}:{local_frame}")
        self._next_local_frame = local_frame + 1
        return frame


def _preflight_nvenc(config: S05ReviewConfig) -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
        raise ContractError(
            "S05 review requires CUDA_VISIBLE_DEVICES exactly equal to '1'"
        )
    if shutil.which(config.ffmpeg_binary) is None:
        raise ContractError(f"FFmpeg executable not found: {config.ffmpeg_binary}")
    if shutil.which(config.ffprobe_binary) is None:
        raise ContractError(f"ffprobe executable not found: {config.ffprobe_binary}")
    command = [
        config.ffmpeg_binary,
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-f",
        "rawvideo",
        "-pixel_format",
        "bgr24",
        "-video_size",
        f"{config.output_width}x{config.output_height}",
        "-framerate",
        f"{EXPECTED_FRAME_RATE.numerator}/{EXPECTED_FRAME_RATE.denominator}",
        "-i",
        "pipe:0",
        "-frames:v",
        "1",
        "-c:v",
        "h264_nvenc",
        "-gpu",
        str(config.logical_gpu),
        "-preset",
        config.preset,
        "-tune",
        "hq",
        "-rc:v",
        "vbr",
        "-cq:v",
        str(config.cq),
        "-b:v",
        "0",
        "-pix_fmt",
        config.pixel_format,
        "-f",
        "null",
        "-",
    ]
    try:
        completed = subprocess.run(
            command,
            input=bytes(config.output_width * config.output_height * 3),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            check=False,
            shell=False,
        )
    except OSError as exc:
        raise ContractError(f"cannot start S05 NVENC preflight: {exc}") from exc
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise ContractError(
            f"GPU 1 h264_nvenc preflight failed ({completed.returncode}): "
            f"{detail or 'no FFmpeg diagnostics'}"
        )


def _frame_role_boxes(
    data: LoadedS05ReviewData,
    render_case: RenderCase,
    global_frame: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    start = int(data.valid_detection_frame_offsets[global_frame])
    stop = int(data.valid_detection_frame_offsets[global_frame + 1])
    positions = data.valid_detection_positions_by_frame[start:stop]
    stable_ids = data.stable_id_by_detection_position[positions]
    source_mask = stable_ids == render_case.case.candidate.source_stable_id
    target_mask = stable_ids == render_case.case.candidate.target_stable_id
    if render_case.competing_stable_ids:
        competing_mask = np.isin(
            stable_ids,
            np.asarray(render_case.competing_stable_ids, dtype=np.int64),
        )
    else:
        competing_mask = np.zeros(len(positions), dtype=np.bool_)
    competing_mask &= ~(source_mask | target_mask)
    unrelated_mask = ~(source_mask | target_mask | competing_mask)
    boxes = np.column_stack(
        [
            np.asarray(data.detections[name], dtype=np.float64)[positions]
            for name in ("x1", "y1", "x2", "y2")
        ]
    )
    return (
        boxes[unrelated_mask],
        boxes[source_mask],
        boxes[target_mask],
        boxes[competing_mask],
    )


def _video_validation_payload(metadata: QaMp4Metadata) -> dict[str, Any]:
    return {
        "codec_name": metadata.codec_name,
        "width": metadata.width,
        "height": metadata.height,
        "average_frame_rate": str(metadata.average_frame_rate),
        "num_frames": metadata.frame_count,
    }


def _render_one_case(
    data: LoadedS05ReviewData,
    render_case: RenderCase,
    config: S05ReviewConfig,
    output_dir: Path,
    *,
    logger: LogFn,
) -> dict[str, Any]:
    output_path = output_dir / render_case.output_relative_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames = data.frames
    raw_width = int(np.asarray(frames["width"])[0])
    raw_height = int(np.asarray(frames["height"])[0])
    last_report = time.monotonic()
    with RawFrameReader(
        data.video_paths, raw_width=raw_width, raw_height=raw_height
    ) as reader, NvencVideoWriter(
        output_path,
        width=config.output_width,
        height=config.output_height,
        fps=EXPECTED_FRAME_RATE,
        expected_frame_count=render_case.expected_frame_count,
        ffmpeg_binary=config.ffmpeg_binary,
        preset=config.preset,
        cq=config.cq,
        logical_gpu=config.logical_gpu,
        pixel_format=config.pixel_format,
    ) as writer:
        for completed, global_frame in enumerate(render_case.frame_sequence, start=1):
            raw = reader.read(
                str(frames["clip_id"][global_frame]),
                int(frames["local_frame"][global_frame]),
                float(frames["pts_sec"][global_frame]),
            )
            unrelated, source, target, competing = _frame_role_boxes(
                data, render_case, global_frame
            )
            try:
                annotated = render_s05_review_frame(
                    raw,
                    output_width=config.output_width,
                    output_height=config.output_height,
                    unrelated_boxes=unrelated,
                    source_boxes=source,
                    target_boxes=target,
                    intermediate_boxes=competing,
                )
            except (TypeError, ValueError, cv2.error) as exc:
                raise ContractError(
                    f"cannot annotate {render_case.case.candidate.candidate_id} "
                    f"global frame {global_frame}: {exc}"
                ) from exc
            writer.write(np.ascontiguousarray(annotated))
            now = time.monotonic()
            if now - last_report >= config.progress_interval_sec:
                logger(
                    f"[s05-review] {render_case.case.candidate.candidate_id}: "
                    f"{completed:,}/{render_case.expected_frame_count:,} frames"
                )
                last_report = now
    metadata = validate_qa_mp4(
        output_path,
        expected_frame_rate=EXPECTED_FRAME_RATE,
        expected_frame_count=render_case.expected_frame_count,
        ffprobe_binary=config.ffprobe_binary,
    )
    fingerprint = _relative_fingerprint(output_path, output_dir)
    return {
        "status": "completed",
        "completed_at_unix_sec": time.time(),
        "output_fingerprint": fingerprint,
        "video_validation": _video_validation_payload(metadata),
    }


def _window_record(
    data: LoadedS05ReviewData, start: int, end: int
) -> dict[str, Any]:
    frames = data.frames
    return {
        "start_clip_id": str(frames["clip_id"][start]),
        "end_clip_id": str(frames["clip_id"][end]),
        "start_local_frame": int(frames["local_frame"][start]),
        "end_local_frame": int(frames["local_frame"][end]),
        "start_global_frame": start,
        "end_global_frame": end,
        "start_global_time_sec": float(frames["global_time_sec"][start]),
        "end_global_time_sec": float(frames["global_time_sec"][end]),
        "frame_count": end - start + 1,
    }


def _case_record(
    data: LoadedS05ReviewData, render_case: RenderCase
) -> dict[str, Any]:
    record = render_case.case.to_manifest_record()
    endpoint = render_case.endpoint
    record.update(
        {
            "source_endpoint": {
                "stable_id": endpoint.source_stable_id,
                "start_global_frame": endpoint.source_start_global_frame,
                "end_global_frame": endpoint.source_end_global_frame,
                "start_global_time_sec": endpoint.source_start_time_sec,
                "end_global_time_sec": endpoint.source_end_time_sec,
                "end_clip_id": endpoint.source_end_clip_id,
            },
            "target_endpoint": {
                "stable_id": endpoint.target_stable_id,
                "start_global_frame": endpoint.target_start_global_frame,
                "end_global_frame": endpoint.target_end_global_frame,
                "start_global_time_sec": endpoint.target_start_time_sec,
                "end_global_time_sec": endpoint.target_end_time_sec,
                "start_clip_id": endpoint.target_start_clip_id,
            },
            "source_window": _window_record(
                data,
                render_case.source_window_start_global_frame,
                render_case.source_window_end_global_frame,
            ),
            "target_window": _window_record(
                data,
                render_case.target_window_start_global_frame,
                render_case.target_window_end_global_frame,
            ),
            "concatenation_order": ["source_window", "target_window"],
            "long_gap_frames_rendered": 0,
            "expected_frame_count": render_case.expected_frame_count,
            "output_path": render_case.output_relative_path,
            "competing_stable_ids": list(render_case.competing_stable_ids),
            "purple_box_policy": PURPLE_STABLE_POLICY,
            "render": {"status": "pending"},
        }
    )
    return record


def _validate_completed_video(
    record: Mapping[str, Any], output_dir: Path, config: S05ReviewConfig
) -> None:
    candidate_id = record.get("candidate_id")
    render = record.get("render")
    if not isinstance(render, dict) or set(render) != {
        "status",
        "completed_at_unix_sec",
        "output_fingerprint",
        "video_validation",
    }:
        raise ContractError(f"S05 review case render is malformed: {candidate_id}")
    completed_at = render.get("completed_at_unix_sec")
    if (
        render.get("status") != "completed"
        or isinstance(completed_at, bool)
        or not isinstance(completed_at, (int, float))
        or not math.isfinite(float(completed_at))
        or float(completed_at) < 0.0
    ):
        raise ContractError(f"S05 review case is not complete: {candidate_id}")
    relative = record.get("output_path")
    if (
        not isinstance(relative, str)
        or Path(relative).is_absolute()
        or len(Path(relative).parts) != 2
        or Path(relative).parts[0] != "videos"
        or Path(relative).suffix != ".mp4"
    ):
        raise ContractError(f"S05 review output path is invalid: {relative!r}")
    output_path = output_dir / relative
    current = _relative_fingerprint(output_path, output_dir)
    expected = render.get("output_fingerprint")
    if not isinstance(expected, dict) or set(expected) != {
        "path",
        "size_bytes",
        "sha256",
    }:
        raise ContractError(f"S05 completed fingerprint is malformed: {relative}")
    if (
        expected.get("path") != relative
        or type(expected.get("size_bytes")) is not int
        or expected["size_bytes"] < 0
        or not isinstance(expected.get("sha256"), str)
        or len(expected["sha256"]) != 64
        or any(
            character not in "0123456789abcdef"
            for character in expected["sha256"]
        )
    ):
        raise ContractError(f"S05 completed fingerprint fields are invalid: {relative}")
    if current != expected:
        raise ContractError(f"completed S05 review video changed: {output_path}")
    expected_count = record.get("expected_frame_count")
    if (
        isinstance(expected_count, bool)
        or not isinstance(expected_count, int)
        or expected_count <= 0
    ):
        raise ContractError(f"S05 expected frame count is invalid: {candidate_id}")
    metadata = validate_qa_mp4(
        output_path,
        expected_frame_rate=EXPECTED_FRAME_RATE,
        expected_frame_count=expected_count,
        ffprobe_binary=config.ffprobe_binary,
    )
    if render.get("video_validation") != _video_validation_payload(metadata):
        raise ContractError(f"completed S05 video metadata changed: {output_path}")


def _effective_config_output_record(
    output_dir: Path, config_payload: Mapping[str, Any]
) -> dict[str, Any]:
    path = output_dir / "effective_config.json"
    if not _json_exact(
        _read_json(path, "S05 review effective config"), config_payload
    ):
        raise ContractError("S05 review effective_config.json changed")
    return _relative_fingerprint(path, output_dir)


def _validate_output_tree(
    output_dir: Path,
    cases: Sequence[Mapping[str, Any]],
    *,
    success_expected: bool,
) -> None:
    if not output_dir.is_dir():
        raise ContractError(f"S05 review output is not a directory: {output_dir}")
    expected_root_files = {MANIFEST_NAME, "effective_config.json"}
    if success_expected:
        expected_root_files.add(SUCCESS_NAME)
    root_files: set[str] = set()
    root_dirs: set[str] = set()
    for path in output_dir.iterdir():
        if path.is_symlink():
            raise ContractError(f"S05 review output cannot contain symlinks: {path}")
        if path.is_file():
            root_files.add(path.name)
        elif path.is_dir():
            root_dirs.add(path.name)
        else:
            raise ContractError(f"S05 review output contains a special path: {path}")
    if root_files != expected_root_files or root_dirs != {"videos"}:
        raise ContractError("S05 review root artifact tree differs")

    required_videos: set[str] = set()
    allowed_videos: set[str] = set()
    for record in cases:
        render = record.get("render")
        relative = record.get("output_path")
        if not isinstance(render, Mapping) or not isinstance(relative, str):
            raise ContractError("S05 review manifest case is malformed")
        path = Path(relative)
        if len(path.parts) != 2 or path.parts[0] != "videos":
            raise ContractError("S05 review manifest video path is malformed")
        allowed_videos.add(path.name)
        if render.get("status") == "completed":
            required_videos.add(path.name)
        elif render != {"status": "pending"}:
            raise ContractError("S05 review render state is invalid")
    videos_dir = output_dir / "videos"
    actual_videos: set[str] = set()
    for path in videos_dir.iterdir():
        if path.is_symlink() or not path.is_file():
            raise ContractError(f"S05 videos directory contains an invalid path: {path}")
        actual_videos.add(path.name)
    if (
        not required_videos.issubset(actual_videos)
        or not actual_videos.issubset(allowed_videos)
        or success_expected
        and actual_videos != required_videos
    ):
        raise ContractError("S05 review video artifact set differs from manifest")


def _initialize_output_tree(
    output_dir: Path,
    config_payload: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> None:
    """Publish the initial config/manifest/videos tree with one directory rename."""

    try:
        output_dir.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ContractError(
            f"cannot create S05 review output parent {output_dir.parent}: {exc}"
        ) from exc
    staging = output_dir.with_name(
        f".{output_dir.name}.staging-{uuid.uuid4().hex}"
    )
    try:
        staging.mkdir()
        (staging / "videos").mkdir()
        _atomic_write_json(staging / "effective_config.json", config_payload)
        _atomic_write_json(staging / MANIFEST_NAME, manifest)
        if output_dir.exists():
            raise ContractError(
                f"S05 review output appeared during initialization: {output_dir}"
            )
        os.rename(staging, output_dir)
    except ContractError:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    except OSError as exc:
        shutil.rmtree(staging, ignore_errors=True)
        raise ContractError(
            f"cannot atomically initialize S05 review output {output_dir}: {exc}"
        ) from exc


def _immutable_case_records(cases: Any) -> list[dict[str, Any]]:
    if not isinstance(cases, list) or not all(isinstance(item, dict) for item in cases):
        raise ContractError("S05 review manifest cases must be a list of objects")
    return [
        {key: value for key, value in item.items() if key != "render"}
        for item in cases
    ]


def _success_stats(
    data: LoadedS05ReviewData, render_cases: Sequence[RenderCase]
) -> dict[str, int]:
    source_frames = sum(item.source_frame_count for item in render_cases)
    target_frames = sum(item.target_frame_count for item in render_cases)
    return {
        "num_available_candidates": len(data.plan.candidates),
        "num_rendered_cases": len(render_cases),
        "num_source_window_frames": source_frames,
        "num_target_window_frames": target_frames,
        "num_rendered_frames": source_frames + target_frames,
        "num_long_gap_frames": 0,
        "num_merges": 0,
    }


def run_s05_review(
    proposals_dir: Path,
    ingest_dir: Path,
    stable_dir: Path,
    config_path: Path,
    output_dir: Path,
    *,
    logger: LogFn = log,
) -> dict[str, Any]:
    """Render bounded S05 provisional evidence without changing identities."""

    config_path = config_path.resolve()
    config_snapshot = _fingerprint(config_path)
    config, config_payload, config_hash = load_s05_review_config(config_path)
    if _fingerprint(config_path) != config_snapshot:
        raise ContractError("S05 review config changed while being loaded")
    proposals_dir = proposals_dir.resolve()
    ingest_dir = ingest_dir.resolve()
    stable_dir = stable_dir.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists() and not output_dir.is_dir():
        raise ContractError(f"S05 review output path is not a directory: {output_dir}")
    for label, input_dir in (
        ("S05 proposals", proposals_dir),
        ("S00 ingest", ingest_dir),
        ("S04 finalized", stable_dir),
    ):
        if not input_dir.is_dir():
            raise ContractError(f"required {label} directory does not exist: {input_dir}")
        if (
            output_dir == input_dir
            or output_dir.is_relative_to(input_dir)
            or input_dir.is_relative_to(output_dir)
        ):
            raise ContractError(
                f"S05 review output cannot overlap immutable {label}: {input_dir}"
            )
    if config_path.is_relative_to(output_dir):
        raise ContractError("S05 review config cannot be inside its output directory")

    _preflight_nvenc(config)
    logger("[s05-review] GPU 1 h264_nvenc preflight passed; validating inputs")
    data = _load_data(
        proposals_dir,
        ingest_dir,
        stable_dir,
        config_path,
        config,
        logger,
        config_fingerprint=config_snapshot,
    )
    _verify_fingerprints_unchanged(
        data.input_fingerprints,
        progress_interval_sec=config.progress_interval_sec,
        logger=logger,
    )
    render_cases = _render_cases(data, config)
    plan_payload = data.plan.to_manifest_payload()
    case_records = [_case_record(data, item) for item in render_cases]
    immutable_cases = _immutable_case_records(case_records)
    identity = {
        "config_hash": config_hash,
        "input_fingerprints": list(data.input_fingerprints),
        "plan_hash": _canonical_hash(
            {"selection_plan": plan_payload, "cases": immutable_cases}
        ),
    }
    video_contract = {
        "input_coordinate_system": "raw_3840x2160_encoded_landscape_no_autorotate",
        "bbox_coordinates": "original_raw_frame_before_whole_frame_resize",
        "raw_width": RAW_WIDTH,
        "raw_height": RAW_HEIGHT,
        "output_width": config.output_width,
        "output_height": config.output_height,
        "average_frame_rate": str(EXPECTED_FRAME_RATE),
        "frame_selection": {
            "source_tail_sec": config.source_tail_sec,
            "target_head_sec": config.target_head_sec,
            "concatenation_order": ["source_window", "target_window"],
            "full_long_gap_rendered": False,
        },
        "encoder": {
            "codec": "h264_nvenc",
            "physical_gpu": 1,
            "cuda_visible_devices": "1",
            "logical_gpu": config.logical_gpu,
            "cpu_encoder_fallback": False,
            "opencv_encoder_fallback": False,
            "pixel_format": config.pixel_format,
        },
        "frame_contents": "original_image_and_real_s00_bboxes_only",
        "text": False,
        "ids": False,
        "scores": False,
        "legend": False,
        "panels": False,
        "inset": False,
        "trajectory": False,
        "keypoints": False,
        "synthetic_boxes": False,
        "colors_bgr": {
            "unrelated_gray": list(UNRELATED_BGR),
            "source_yellow": list(SOURCE_BGR),
            "target_green": list(TARGET_BGR),
            "competing_stable_purple": list(INTERMEDIATE_BGR),
        },
        "purple_box_policy": PURPLE_STABLE_POLICY,
    }
    manifest_path = output_dir / MANIFEST_NAME
    manifest_keys = {
        "schema_version",
        "stage",
        "review_purpose",
        "candidate_semantics",
        "automatic_merge_allowed",
        "confirmed_links_allowed",
        "solver_used",
        "num_merges",
        "per_item_human_labels_expected",
        "identity",
        "effective_config",
        "selection_plan",
        "video_contract",
        "cases",
    }
    if manifest_path.is_file():
        manifest = _read_json(manifest_path, "S05 review manifest")
        if not isinstance(manifest, dict) or set(manifest) != manifest_keys:
            raise ContractError("existing S05 review manifest fields differ")
        if (
            manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION
            or manifest.get("stage") != "S05_REVIEW"
            or manifest.get("review_purpose") != "provisional_candidate_evidence"
            or manifest.get("candidate_semantics")
            != "provisional_review_evidence_only"
            or manifest.get("automatic_merge_allowed") is not False
            or manifest.get("confirmed_links_allowed") is not False
            or manifest.get("solver_used") is not False
            or type(manifest.get("num_merges")) is not int
            or manifest.get("num_merges") != 0
            or manifest.get("per_item_human_labels_expected") is not True
            or not _json_exact(manifest.get("identity"), identity)
            or not _json_exact(manifest.get("effective_config"), config_payload)
            or not _json_exact(manifest.get("selection_plan"), plan_payload)
            or not _json_exact(manifest.get("video_contract"), video_contract)
            or not _json_exact(
                _immutable_case_records(manifest.get("cases")), immutable_cases
            )
        ):
            raise ContractError("existing S05 review output belongs to another plan")
    else:
        if output_dir.exists() and any(output_dir.iterdir()):
            raise ContractError(
                f"S05 review output is non-empty without {MANIFEST_NAME}: {output_dir}"
            )
        if output_dir.exists():
            try:
                output_dir.rmdir()
            except OSError as exc:
                raise ContractError(
                    f"cannot replace empty S05 review output {output_dir}: {exc}"
                ) from exc
        manifest = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "stage": "S05_REVIEW",
            "review_purpose": "provisional_candidate_evidence",
            "candidate_semantics": "provisional_review_evidence_only",
            "automatic_merge_allowed": False,
            "confirmed_links_allowed": False,
            "solver_used": False,
            "num_merges": 0,
            "per_item_human_labels_expected": True,
            "identity": identity,
            "effective_config": config_payload,
            "selection_plan": plan_payload,
            "video_contract": video_contract,
            "cases": case_records,
        }
        _initialize_output_tree(output_dir, config_payload, manifest)

    effective_output = _effective_config_output_record(output_dir, config_payload)
    success_path = output_dir / SUCCESS_NAME
    _validate_output_tree(
        output_dir,
        manifest["cases"],
        success_expected=success_path.is_file(),
    )
    expected_stats = _success_stats(data, render_cases)
    success_keys = {
        "schema_version",
        "stage",
        "review_purpose",
        "candidate_semantics",
        "automatic_merge_allowed",
        "confirmed_links_allowed",
        "solver_used",
        "num_merges",
        "identity",
        "manifest_sha256",
        "stats",
        "output_fingerprints",
    }
    if success_path.is_file():
        success = _read_json(success_path, "S05 review success marker")
        if (
            not isinstance(success, dict)
            or set(success) != success_keys
            or success.get("schema_version") != SUCCESS_SCHEMA_VERSION
            or success.get("stage") != "S05_REVIEW"
            or success.get("review_purpose") != "provisional_candidate_evidence"
            or success.get("candidate_semantics")
            != "provisional_review_evidence_only"
            or success.get("automatic_merge_allowed") is not False
            or success.get("confirmed_links_allowed") is not False
            or success.get("solver_used") is not False
            or type(success.get("num_merges")) is not int
            or success.get("num_merges") != 0
            or not _json_exact(success.get("identity"), identity)
            or success.get("manifest_sha256") != _sha256(manifest_path)
            or not isinstance(success.get("stats"), dict)
            or set(success["stats"]) != set(expected_stats)
            or any(
                type(value) is not int or value < 0
                for value in success["stats"].values()
            )
            or not _json_exact(success.get("stats"), expected_stats)
        ):
            raise ContractError("existing S05 review success marker differs")
        for record in manifest["cases"]:
            _validate_completed_video(record, output_dir, config)
        expected_outputs = [effective_output] + [
            record["render"]["output_fingerprint"] for record in manifest["cases"]
        ]
        if not _json_exact(success.get("output_fingerprints"), expected_outputs):
            raise ContractError("S05 review success output fingerprints changed")
        _verify_fingerprints_unchanged(
            data.input_fingerprints,
            progress_interval_sec=config.progress_interval_sec,
            logger=logger,
        )
        logger(f"[s05-review] already complete: {success_path}")
        return manifest

    by_candidate = {
        item.case.candidate.candidate_id: item for item in render_cases
    }
    for index, record in enumerate(manifest["cases"], start=1):
        if record.get("render") == {"status": "pending"}:
            candidate_id = str(record["candidate_id"])
            render_case = by_candidate.get(candidate_id)
            if render_case is None:
                raise ContractError(
                    f"S05 manifest references unknown render case {candidate_id}"
                )
            logger(
                f"[s05-review] render {index}/{len(render_cases)}: "
                f"{candidate_id} -> {record['output_path']}"
            )
            record["render"] = _render_one_case(
                data,
                render_case,
                config,
                output_dir,
                logger=logger,
            )
            _atomic_write_json(manifest_path, manifest)
        else:
            _validate_completed_video(record, output_dir, config)

    _verify_fingerprints_unchanged(
        data.input_fingerprints,
        progress_interval_sec=config.progress_interval_sec,
        logger=logger,
    )
    effective_output = _effective_config_output_record(output_dir, config_payload)
    for record in manifest["cases"]:
        _validate_completed_video(record, output_dir, config)
    _validate_output_tree(output_dir, manifest["cases"], success_expected=False)
    output_fingerprints = [effective_output] + [
        record["render"]["output_fingerprint"] for record in manifest["cases"]
    ]
    success = {
        "schema_version": SUCCESS_SCHEMA_VERSION,
        "stage": "S05_REVIEW",
        "review_purpose": "provisional_candidate_evidence",
        "candidate_semantics": "provisional_review_evidence_only",
        "automatic_merge_allowed": False,
        "confirmed_links_allowed": False,
        "solver_used": False,
        "num_merges": 0,
        "identity": identity,
        "manifest_sha256": _sha256(manifest_path),
        "stats": expected_stats,
        "output_fingerprints": output_fingerprints,
    }
    _atomic_write_json(success_path, success)
    _validate_output_tree(output_dir, manifest["cases"], success_expected=True)
    logger(f"[s05-review] complete: {success_path}")
    return manifest


__all__ = [
    "EXPECTED_FRAME_RATE",
    "CandidateEndpoint",
    "LoadedS05ReviewData",
    "RawFrameReader",
    "RenderCase",
    "run_s05_review",
]
