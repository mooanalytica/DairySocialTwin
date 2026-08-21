from __future__ import annotations

import csv
import atexit
import fcntl
import hashlib
import json
import math
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Mapping

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from cowtrack.config import ContractError
from cowtrack.qa.ffprobe import validate_qa_mp4
from cowtrack.qa.nvenc import NvencVideoWriter
from cowtrack.qa.review_config import S01ReviewConfig, load_s01_review_config
from cowtrack.qa.s01_review_plan import (
    DETECTION_COLUMNS,
    DET_EDGE_COLUMNS,
    DET_TO_MICRO_COLUMNS,
    MICROTRACKLET_COLUMNS,
    ReviewCase,
    ReviewPlan,
    build_review_plan,
)
from cowtrack.qa.s01_review_render import render_review_frame
from cowtrack.schemas.detections import DETECTIONS_SCHEMA
from cowtrack.schemas.edges import DET_EDGES_SCHEMA
from cowtrack.schemas.frames import FRAMES_SCHEMA
from cowtrack.schemas.tracklets import DET_TO_MICRO_SCHEMA, MICROTRACKLETS_SCHEMA
from cowtrack.video import open_raw_video_capture


LogFn = Callable[[str], None]
CaptureFactory = Callable[[Path], cv2.VideoCapture]

EXPECTED_FRAME_RATE = Fraction(30_000, 1_001)
EXPECTED_FRAME_PERIOD_SEC = float(1 / EXPECTED_FRAME_RATE)
RAW_WIDTH = 3_840
RAW_HEIGHT = 2_160
MANIFEST_NAME = "review_manifest.json"
LABELS_NAME = "review_labels.csv"
SUCCESS_NAME = "_SUCCESS.json"
LOCK_NAME = ".s01-review.lock"
MANIFEST_SCHEMA_VERSION = "cowtrack.s01-video-review.v2"
SUCCESS_SCHEMA_VERSION = "cowtrack.s01-video-review-success.v2"
COMPLETION_SCHEMA_VERSION = "cowtrack.s01-video-review-completion.v2"
LABEL_COLUMNS = (
    "case_id",
    "case_kind",
    "reasons",
    "video_path",
    "verdict",
    "reviewer",
    "notes",
)
ALLOWED_VERDICTS = {"", "PASS", "WARN", "FAIL", "UNJUDGABLE"}
TRUE_EVENT_REASONS = frozenset(
    {"low_purity", "large_center_jump", "legacy_id_transition"}
)
EVENT_HIGHLIGHT_RADIUS_FRAMES = int(round(0.5 * float(EXPECTED_FRAME_RATE)))


def log(message: str) -> None:
    print(message, flush=True)


@dataclass(frozen=True)
class LoadedReviewData:
    frames: dict[str, np.ndarray]
    detections: dict[str, np.ndarray]
    det_to_micro: dict[str, np.ndarray]
    microtracklets: dict[str, np.ndarray]
    det_edges: dict[str, np.ndarray]
    video_paths: dict[str, Path]
    input_fingerprints: tuple[dict[str, Any], ...]
    valid_detection_positions_by_frame: np.ndarray
    valid_detection_frame_offsets: np.ndarray
    micro_paths: dict[int, np.ndarray]

    @property
    def sequence_id(self) -> str:
        values = set(map(str, np.unique(self.frames["sequence_id"])))
        if len(values) != 1:
            raise ContractError("review data does not contain exactly one sequence_id")
        return next(iter(values))

    @property
    def num_frames(self) -> int:
        return len(self.frames["global_frame"])


@dataclass(frozen=True)
class RenderCase:
    case: ReviewCase
    start_global_frame: int
    end_global_frame: int
    output_relative_path: str

    @property
    def expected_frame_count(self) -> int:
        return self.end_global_frame - self.start_global_frame + 1


@dataclass
class _OutputLock:
    path: Path
    descriptor: int
    token: str
    released: bool = False


def _read_json(path: Path, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read {label}: {path}: {exc}") from exc


def _atomic_write_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    except OSError as exc:
        temporary.unlink(missing_ok=True)
        raise ContractError(f"cannot atomically write JSON {path}: {exc}") from exc


def _release_output_lock(lock: _OutputLock) -> None:
    if lock.released:
        return
    try:
        fcntl.flock(lock.descriptor, fcntl.LOCK_UN)
    except OSError:
        pass
    try:
        os.close(lock.descriptor)
    except OSError:
        pass
    lock.released = True


def _acquire_output_lock(output_dir: Path) -> _OutputLock:
    lock_path = output_dir / LOCK_NAME
    token = os.urandom(16).hex()
    payload = {
        "pid": os.getpid(),
        "token": token,
        "created_at_unix_sec": time.time(),
    }
    encoded = (json.dumps(payload, sort_keys=True) + "\n").encode("utf-8")
    try:
        descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    except OSError as exc:
        raise ContractError(f"cannot open review output lock: {exc}") from exc
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        os.close(descriptor)
        raise ContractError(
            f"another S01 review process is active: {lock_path}"
        ) from exc
    except OSError as exc:
        os.close(descriptor)
        raise ContractError(f"cannot acquire review output lock: {exc}") from exc
    try:
        os.ftruncate(descriptor, 0)
        os.lseek(descriptor, 0, os.SEEK_SET)
        total = 0
        while total < len(encoded):
            written = os.write(descriptor, encoded[total:])
            if written <= 0:
                raise OSError("short write to review output lock")
            total += written
        os.fsync(descriptor)
    except OSError as exc:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)
        raise ContractError(f"cannot write review output lock: {exc}") from exc
    lock = _OutputLock(lock_path, descriptor, token)
    atexit.register(_release_output_lock, lock)
    return lock


def _remove_stale_encoder_parts(output_dir: Path) -> None:
    stale_paths = {
        *output_dir.rglob(".*.part.mp4"),
        *output_dir.rglob(".*.tmp"),
    }
    for path in sorted(stale_paths):
        try:
            path.unlink()
        except OSError as exc:
            raise ContractError(f"cannot remove stale encoder part {path}: {exc}") from exc


def _sha256(
    path: Path,
    *,
    progress_interval_sec: float | None = None,
    logger: LogFn | None = None,
) -> str:
    digest = hashlib.sha256()
    try:
        total_size = path.stat().st_size
    except OSError as exc:
        raise ContractError(f"cannot stat fingerprint input {path}: {exc}") from exc
    processed = 0
    last_report = time.monotonic()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
                processed += len(block)
                now = time.monotonic()
                if (
                    logger is not None
                    and progress_interval_sec is not None
                    and now - last_report >= progress_interval_sec
                ):
                    percent = 100.0 * processed / total_size if total_size else 100.0
                    logger(
                        f"[s01-review] fingerprint {path.name}: "
                        f"{processed / 1024**3:.2f}/{total_size / 1024**3:.2f} GiB "
                        f"({percent:.1f}%)"
                    )
                    last_report = now
    except OSError as exc:
        raise ContractError(f"cannot fingerprint {path}: {exc}") from exc
    if logger is not None:
        logger(
            f"[s01-review] fingerprint {path.name}: verified "
            f"{processed / 1024**3:.2f} GiB"
        )
    return digest.hexdigest()


def _fingerprint(
    path: Path,
    *,
    progress_interval_sec: float | None = None,
    logger: LogFn | None = None,
) -> dict[str, Any]:
    try:
        stat = path.stat()
    except OSError as exc:
        raise ContractError(f"required input does not exist: {path}: {exc}") from exc
    if not path.is_file():
        raise ContractError(f"required input is not a file: {path}")
    sha256 = _sha256(
        path,
        progress_interval_sec=progress_interval_sec,
        logger=logger,
    )
    try:
        after = path.stat()
    except OSError as exc:
        raise ContractError(f"cannot restat fingerprint input {path}: {exc}") from exc
    if (stat.st_size, stat.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ContractError(f"input changed while fingerprinting: {path}")
    return {
        "path": str(path.resolve()),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": sha256,
    }


def _canonical_hash(payload: Any) -> str:
    canonical = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _case_commitment(record: dict[str, Any]) -> str:
    return _canonical_hash(
        {key: value for key, value in record.items() if key != "render"}
    )


def _completion_journal_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.name}.completion.json")


def _remove_completion_journal(output_path: Path) -> None:
    journal_path = _completion_journal_path(output_path)
    try:
        journal_path.unlink(missing_ok=True)
    except OSError as exc:
        raise ContractError(f"cannot remove completion journal {journal_path}: {exc}") from exc


def _parquet_columns(
    path: Path,
    *,
    expected_schema: pa.Schema,
    expected_rows: int,
    columns: tuple[str, ...] | list[str],
    label: str,
) -> dict[str, np.ndarray]:
    try:
        parquet = pq.ParquetFile(path)
    except (OSError, pa.ArrowException) as exc:
        raise ContractError(f"cannot open {label}: {path}: {exc}") from exc
    if not parquet.schema_arrow.equals(expected_schema, check_metadata=False):
        raise ContractError(f"{label} schema mismatch: {path}")
    if parquet.metadata.num_rows != expected_rows:
        raise ContractError(
            f"{label} row count mismatch: {parquet.metadata.num_rows} != {expected_rows}"
        )
    try:
        table = pq.read_table(path, columns=list(columns))
    except (OSError, pa.ArrowException) as exc:
        raise ContractError(f"cannot read {label}: {path}: {exc}") from exc
    return {
        name: table[name].combine_chunks().to_numpy(zero_copy_only=False)
        for name in columns
    }


def _validated_stage_files(
    directory: Path,
    *,
    expected_stage: str,
    required_names: tuple[str, ...],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    success_path = directory / SUCCESS_NAME
    payload = _read_json(success_path, f"{expected_stage} success marker")
    if not isinstance(payload, dict) or payload.get("stage") != expected_stage:
        raise ContractError(f"{success_path} is not a completed {expected_stage} output")
    records = payload.get("output_fingerprints")
    if not isinstance(records, list):
        raise ContractError(f"{expected_stage} success marker lacks output fingerprints")
    by_name = {
        str(record.get("path")): record
        for record in records
        if isinstance(record, dict)
    }
    missing = sorted(set(required_names) - set(by_name))
    if missing:
        raise ContractError(f"{expected_stage} success marker lacks artifacts: {missing}")
    observed: list[dict[str, Any]] = []
    for name in required_names:
        path = (directory / name).resolve()
        record = by_name[name]
        current = _fingerprint(path)
        for key in ("size_bytes", "sha256"):
            if current[key] != record.get(key):
                raise ContractError(
                    f"completed {expected_stage} artifact changed: {name} ({key})"
                )
        observed.append(current)
    observed.append(_fingerprint(success_path.resolve()))
    return payload, observed


def _required_stat(
    stats: Mapping[str, Any], stage: str, key: str, *, minimum: int = 0
) -> int:
    value = stats.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ContractError(
            f"{stage} success marker has invalid {key}: {value!r}"
        )
    return value


def _validate_upstream_stats(
    s00_success: Mapping[str, Any],
    s01_success: Mapping[str, Any],
    *,
    manifest_num_clips: int,
) -> dict[str, int]:
    s00_stats = s00_success.get("stats")
    s01_stats = s01_success.get("stats")
    if not isinstance(s00_stats, dict):
        raise ContractError("S00 success marker lacks stats")
    if not isinstance(s01_stats, dict):
        raise ContractError("S01 success marker lacks stats")

    num_clips = _required_stat(s00_stats, "S00", "num_clips", minimum=1)
    num_frames = _required_stat(s00_stats, "S00", "num_frames", minimum=1)
    num_detections = _required_stat(
        s00_stats, "S00", "num_input_boxes", minimum=1
    )
    num_valid = _required_stat(
        s00_stats, "S00", "num_valid_boxes", minimum=1
    )
    s01_num_frames = _required_stat(s01_stats, "S01", "num_frames", minimum=1)
    s01_num_valid = _required_stat(
        s01_stats, "S01", "num_valid_detections", minimum=1
    )
    num_microtracklets = _required_stat(
        s01_stats, "S01", "num_microtracklets", minimum=1
    )
    num_edges = _required_stat(s01_stats, "S01", "num_accepted_edges")

    if num_clips != manifest_num_clips:
        raise ContractError("S00 clip count differs from resolved manifest")
    if num_detections < num_valid:
        raise ContractError("S00 valid detections exceed input detections")
    if num_frames != s01_num_frames:
        raise ContractError("S00/S01 frame counts differ")
    if num_valid != s01_num_valid:
        raise ContractError("S00/S01 valid detection counts differ")
    if num_microtracklets > num_valid:
        raise ContractError("S01 microtracklets exceed valid detections")
    if num_edges != num_valid - num_microtracklets:
        raise ContractError(
            "S01 accepted-edge count does not partition detections into microtracklets"
        )
    return {
        "num_clips": num_clips,
        "num_frames": num_frames,
        "num_detections": num_detections,
        "num_valid_detections": num_valid,
        "num_microtracklets": num_microtracklets,
        "num_accepted_edges": num_edges,
    }


def _resolved_manifest_contract(
    resolved_manifest_path: Path,
) -> tuple[str, dict[str, Path]]:
    rows = _read_json(resolved_manifest_path, "S00 resolved manifest")
    if not isinstance(rows, list) or not rows:
        raise ContractError("S01 review requires a non-empty resolved manifest")
    sequence_id: str | None = None
    video_paths: dict[str, Path] = {}
    for expected_order, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ContractError("resolved manifest rows must be mappings")
        row_sequence_id = str(row.get("sequence_id", ""))
        if not row_sequence_id:
            raise ContractError("resolved manifest sequence_id is blank")
        if sequence_id is None:
            sequence_id = row_sequence_id
        elif row_sequence_id != sequence_id:
            raise ContractError("resolved manifest contains multiple sequence_id values")
        clip_order = row.get("clip_order")
        if (
            isinstance(clip_order, bool)
            or not isinstance(clip_order, int)
            or clip_order != expected_order
        ):
            raise ContractError(
                "resolved manifest clip_order must be contiguous from zero"
            )
        clip_id = str(row.get("clip_id", ""))
        if not clip_id or clip_id in video_paths:
            raise ContractError("resolved manifest clip_id is blank or duplicated")
        raw_path = row.get("video_path")
        if not isinstance(raw_path, str) or not raw_path:
            raise ContractError(f"resolved manifest has no video path for {clip_id}")
        video_paths[clip_id] = Path(raw_path).resolve()
    if sequence_id is None:
        raise ContractError("resolved manifest sequence_id is missing")
    return sequence_id, video_paths


def _validate_upstream_reports(
    ingest_report_path: Path,
    microtrack_report_path: Path,
    *,
    sequence_id: str,
    video_paths: Mapping[str, Path],
    counts: Mapping[str, int],
) -> None:
    coordinate_system = "raw_encoded_landscape_no_autorotate"
    ingest = _read_json(ingest_report_path, "S00 ingest report")
    if not isinstance(ingest, dict):
        raise ContractError("S00 ingest report must be an object")
    if ingest.get("sequence_id") != sequence_id:
        raise ContractError("S00 ingest report sequence_id differs from manifest")
    if ingest.get("coordinate_system") != coordinate_system:
        raise ContractError("S00 ingest report coordinate system differs")
    for key, expected in {
        "num_clips": counts["num_clips"],
        "num_frames": counts["num_frames"],
        "num_input_boxes": counts["num_detections"],
        "num_valid_boxes": counts["num_valid_detections"],
    }.items():
        if ingest.get(key) != expected:
            raise ContractError(f"S00 ingest report/success stat mismatch: {key}")

    boundaries = ingest.get("clip_boundaries")
    if not isinstance(boundaries, list) or len(boundaries) != len(video_paths):
        raise ContractError("S00 clip-boundary count differs from resolved manifest")
    expected_start = 0
    for expected_order, (clip_id, boundary) in enumerate(
        zip(video_paths, boundaries, strict=True)
    ):
        if not isinstance(boundary, dict):
            raise ContractError("S00 clip boundary must be an object")
        clip_frames = boundary.get("num_frames")
        start = boundary.get("start_global_frame")
        end = boundary.get("end_global_frame_inclusive")
        if (
            boundary.get("clip_id") != clip_id
            or boundary.get("clip_order") != expected_order
            or isinstance(clip_frames, bool)
            or not isinstance(clip_frames, int)
            or clip_frames <= 0
            or start != expected_start
            or end != expected_start + clip_frames - 1
        ):
            raise ContractError(f"S00 clip boundary differs for {clip_id}")
        if boundary.get("opencv_auto_rotate") is not False:
            raise ContractError("S00 clip boundary enables OpenCV autorotation")
        expected_start += clip_frames
    if expected_start != counts["num_frames"]:
        raise ContractError("S00 clip boundaries do not cover the frame timeline")

    micro = _read_json(microtrack_report_path, "S01 microtrack report")
    if not isinstance(micro, dict):
        raise ContractError("S01 microtrack report must be an object")
    if micro.get("sequence_id") != sequence_id:
        raise ContractError("S01 microtrack report sequence_id differs from manifest")
    if micro.get("input_coordinate_system") != coordinate_system:
        raise ContractError("S01 microtrack report coordinate system differs")
    if micro.get("input_rows_modified") is not False:
        raise ContractError("S01 microtrack report says input rows were modified")
    micro_stats = micro.get("stats")
    if not isinstance(micro_stats, dict):
        raise ContractError("S01 microtrack report lacks stats")
    for key in (
        "num_frames",
        "num_valid_detections",
        "num_microtracklets",
        "num_accepted_edges",
    ):
        if micro_stats.get(key) != counts[key]:
            raise ContractError(f"S01 microtrack report/success stat mismatch: {key}")


def _load_video_paths(
    resolved_manifest_path: Path,
    s00_success: dict[str, Any],
    *,
    progress_interval_sec: float,
    logger: LogFn,
) -> tuple[str, dict[str, Path], list[dict[str, Any]]]:
    sequence_id, video_paths = _resolved_manifest_contract(resolved_manifest_path)
    s00_stats = s00_success.get("stats")
    if not isinstance(s00_stats, dict):
        raise ContractError("S00 success marker lacks stats")
    if _required_stat(s00_stats, "S00", "num_clips", minimum=1) != len(
        video_paths
    ):
        raise ContractError("S00 clip count differs from resolved manifest")
    recorded_inputs = {
        str(Path(record.get("path", "")).resolve()): record
        for record in s00_success.get("input_fingerprints", [])
        if isinstance(record, dict) and record.get("path")
    }
    fingerprints: list[dict[str, Any]] = []
    for clip_id, video_path in video_paths.items():
        recorded = recorded_inputs.get(str(video_path))
        if recorded is None:
            raise ContractError(
                f"video was not fingerprinted by completed S00: {video_path}"
            )
        observed = _fingerprint(
            video_path,
            progress_interval_sec=progress_interval_sec,
            logger=logger,
        )
        for key in ("size_bytes", "mtime_ns", "sha256"):
            if observed[key] != recorded.get(key):
                raise ContractError(f"source video changed since S00: {video_path} ({key})")
        fingerprints.append(observed)
    return sequence_id, video_paths, fingerprints


def _validate_frames(
    frames: dict[str, np.ndarray],
    video_paths: dict[str, Path],
    *,
    expected_sequence_id: str,
    expected_num_frames: int,
) -> None:
    global_frame = np.asarray(frames["global_frame"], dtype=np.int64)
    if not np.array_equal(
        global_frame, np.arange(expected_num_frames, dtype=np.int64)
    ):
        raise ContractError("review frame timeline is not contiguous 0..N-1")
    if set(map(str, np.unique(frames["sequence_id"]))) != {
        expected_sequence_id
    }:
        raise ContractError("review frames sequence_id mismatch")
    widths = np.asarray(frames["width"], dtype=np.int64)
    heights = np.asarray(frames["height"], dtype=np.int64)
    if not np.all(widths == RAW_WIDTH) or not np.all(heights == RAW_HEIGHT):
        raise ContractError("review frames must remain raw 3840x2160 landscape")
    global_time = np.asarray(frames["global_time_sec"], dtype=np.float64)
    if not np.all(np.isfinite(global_time)) or not np.allclose(
        np.diff(global_time), EXPECTED_FRAME_PERIOD_SEC, rtol=0.0, atol=1e-9
    ):
        raise ContractError("review global timeline is not exact 30000/1001 cadence")

    clip_ids = np.asarray(frames["clip_id"], dtype=object)
    local_frames = np.asarray(frames["local_frame"], dtype=np.int64)
    pts = np.asarray(frames["pts_sec"], dtype=np.float64)
    observed_clip_order: list[str] = []
    for clip_id in map(str, clip_ids):
        if not observed_clip_order or observed_clip_order[-1] != clip_id:
            if clip_id in observed_clip_order:
                raise ContractError("a clip appears in multiple disjoint frame ranges")
            observed_clip_order.append(clip_id)
    if observed_clip_order != list(video_paths):
        raise ContractError("frames clip order differs from resolved manifest")
    for clip_id in observed_clip_order:
        positions = np.flatnonzero(clip_ids == clip_id)
        if not np.array_equal(
            local_frames[positions], np.arange(len(positions), dtype=np.int64)
        ):
            raise ContractError(f"{clip_id} local frames are not contiguous from zero")
        if not np.all(np.isfinite(pts[positions])) or not np.allclose(
            np.diff(pts[positions]), EXPECTED_FRAME_PERIOD_SEC, rtol=0.0, atol=1e-9
        ):
            raise ContractError(f"{clip_id} PTS cadence is not exact 30000/1001")


def _validate_detection_frame_foreign_keys(
    frames: dict[str, np.ndarray],
    detections: dict[str, np.ndarray],
    *,
    expected_valid_detections: int | None = None,
) -> None:
    det_frames = np.asarray(detections["global_frame"], dtype=np.int64)
    if np.any(det_frames < 0) or np.any(det_frames >= len(frames["global_frame"])):
        raise ContractError("a detection global_frame is outside frames.parquet")
    checks = (
        ("sequence_id", np.asarray(detections["sequence_id"], dtype=object)),
        ("clip_id", np.asarray(detections["clip_id"], dtype=object)),
        ("local_frame", np.asarray(detections["local_frame"], dtype=np.int64)),
        (
            "global_time_sec",
            np.asarray(detections["global_time_sec"], dtype=np.float64),
        ),
    )
    for name, observed in checks:
        expected = np.asarray(frames[name])[det_frames]
        if not np.array_equal(observed, expected):
            mismatch = int(np.flatnonzero(observed != expected)[0])
            raise ContractError(
                f"detection-to-frame foreign key mismatch for {name} at "
                f"detection row {mismatch}"
            )
    valid = np.asarray(detections["valid"], dtype=np.bool_)
    if expected_valid_detections is not None and int(
        np.count_nonzero(valid)
    ) != expected_valid_detections:
        raise ContractError("review detections valid count mismatch")
    boxes = np.column_stack(
        [
            np.asarray(detections[name], dtype=np.float64)[valid]
            for name in ("x1", "y1", "x2", "y2")
        ]
    )
    if not np.all(np.isfinite(boxes)):
        raise ContractError("a valid review bbox is non-finite")
    if np.any(
        (boxes[:, 0] < 0.0)
        | (boxes[:, 1] < 0.0)
        | (boxes[:, 0] >= boxes[:, 2])
        | (boxes[:, 1] >= boxes[:, 3])
        | (boxes[:, 2] > RAW_WIDTH)
        | (boxes[:, 3] > RAW_HEIGHT)
    ):
        raise ContractError("a valid review bbox is outside raw frame geometry")


def _build_indices(
    frames: dict[str, np.ndarray],
    detections: dict[str, np.ndarray],
    mapping: dict[str, np.ndarray],
    microtracklets: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, dict[int, np.ndarray]]:
    det_ids = np.asarray(detections["det_id"], dtype=np.int64)
    det_sort = np.argsort(det_ids, kind="stable")
    sorted_det_ids = det_ids[det_sort]
    if np.any(np.diff(sorted_det_ids) == 0):
        raise ContractError("review detections.det_id is not unique")
    mapping_det_ids = np.asarray(mapping["det_id"], dtype=np.int64)
    valid = np.asarray(detections["valid"], dtype=np.bool_)
    valid_positions = np.flatnonzero(valid)
    if (
        len(mapping_det_ids) != len(valid_positions)
        or len(np.unique(mapping_det_ids)) != len(mapping_det_ids)
    ):
        raise ContractError(
            "review mapping is not a one-to-one mapping of valid detections"
        )
    locations = np.searchsorted(sorted_det_ids, mapping_det_ids)
    clipped = np.minimum(locations, len(sorted_det_ids) - 1)
    if not np.all(
        (locations < len(sorted_det_ids)) & (sorted_det_ids[clipped] == mapping_det_ids)
    ):
        raise ContractError("review mapping contains unknown det_id")
    det_position_by_mapping_row = det_sort[locations]

    if not np.all(valid[det_position_by_mapping_row]):
        raise ContractError("review mapping contains invalid detections")
    if not np.array_equal(
        np.sort(det_position_by_mapping_row), valid_positions
    ):
        raise ContractError("review mapping does not cover every valid detection")
    det_frames = np.asarray(detections["global_frame"], dtype=np.int64)
    context_order = np.lexsort((det_ids[valid_positions], det_frames[valid_positions]))
    valid_by_frame = valid_positions[context_order]
    counts = np.bincount(det_frames[valid_by_frame], minlength=len(frames["global_frame"]))
    frame_offsets = np.empty(len(counts) + 1, dtype=np.int64)
    frame_offsets[0] = 0
    np.cumsum(counts, out=frame_offsets[1:])

    micro_ids = np.asarray(mapping["micro_id"], dtype=np.int64)
    orders = np.asarray(mapping["order_in_micro"], dtype=np.int64)
    order = np.lexsort((mapping_det_ids, orders, micro_ids))
    sorted_micro = micro_ids[order]
    starts = np.flatnonzero(np.r_[True, sorted_micro[1:] != sorted_micro[:-1]])
    stops = np.r_[starts[1:], len(order)]
    paths: dict[int, np.ndarray] = {}
    for start, stop in zip(starts, stops, strict=True):
        rows = order[int(start) : int(stop)]
        micro_id = int(micro_ids[rows[0]])
        actual_orders = orders[rows]
        if not np.array_equal(actual_orders, np.arange(len(rows), dtype=np.int64)):
            raise ContractError(f"microtrack {micro_id} orders are not contiguous")
        positions = det_position_by_mapping_row[rows]
        if np.any(np.diff(det_frames[positions]) <= 0):
            raise ContractError(f"microtrack {micro_id} is not time ordered")
        paths[micro_id] = positions

    summary_ids = np.asarray(microtracklets["micro_id"], dtype=np.int64)
    if len(np.unique(summary_ids)) != len(summary_ids):
        raise ContractError("microtracklets.micro_id is not unique")
    summary_id_set = set(map(int, summary_ids))
    if set(paths) != summary_id_set:
        raise ContractError("mapping and microtrack summary ID sets differ")
    summary_rows = {
        int(micro_id): row for row, micro_id in enumerate(summary_ids)
    }
    for micro_id, path in paths.items():
        if int(microtracklets["num_detections"][summary_rows[micro_id]]) != len(
            path
        ):
            raise ContractError(
                f"microtrack {micro_id} summary length differs from mapping"
            )
    return valid_by_frame, frame_offsets, paths


def _load_review_data(
    ingest_dir: Path,
    microtrack_dir: Path,
    config_path: Path,
    *,
    progress_interval_sec: float = 10.0,
    logger: LogFn = log,
) -> LoadedReviewData:
    s00_success, s00_fingerprints = _validated_stage_files(
        ingest_dir,
        expected_stage="S00",
        required_names=(
            "frames.parquet",
            "detections.parquet",
            "resolved_manifest.json",
            "ingest_report.json",
        ),
    )
    s01_success, s01_fingerprints = _validated_stage_files(
        microtrack_dir,
        expected_stage="S01",
        required_names=(
            "det_to_micro.parquet",
            "microtracklets.parquet",
            "det_edges.parquet",
            "microtrack_report.json",
        ),
    )
    sequence_id, video_paths, video_fingerprints = _load_video_paths(
        ingest_dir / "resolved_manifest.json",
        s00_success,
        progress_interval_sec=progress_interval_sec,
        logger=logger,
    )
    counts = _validate_upstream_stats(
        s00_success,
        s01_success,
        manifest_num_clips=len(video_paths),
    )
    _validate_upstream_reports(
        ingest_dir / "ingest_report.json",
        microtrack_dir / "microtrack_report.json",
        sequence_id=sequence_id,
        video_paths=video_paths,
        counts=counts,
    )
    frame_columns = tuple(field.name for field in FRAMES_SCHEMA)
    frames = _parquet_columns(
        ingest_dir / "frames.parquet",
        expected_schema=FRAMES_SCHEMA,
        expected_rows=counts["num_frames"],
        columns=frame_columns,
        label="S00 frames",
    )
    detections = _parquet_columns(
        ingest_dir / "detections.parquet",
        expected_schema=DETECTIONS_SCHEMA,
        expected_rows=counts["num_detections"],
        columns=list(dict.fromkeys((*DETECTION_COLUMNS, "x1", "y1", "x2", "y2"))),
        label="S00 detections",
    )
    mapping = _parquet_columns(
        microtrack_dir / "det_to_micro.parquet",
        expected_schema=DET_TO_MICRO_SCHEMA,
        expected_rows=counts["num_valid_detections"],
        columns=list(DET_TO_MICRO_COLUMNS),
        label="S01 det_to_micro",
    )
    microtracklets = _parquet_columns(
        microtrack_dir / "microtracklets.parquet",
        expected_schema=MICROTRACKLETS_SCHEMA,
        expected_rows=counts["num_microtracklets"],
        columns=list(MICROTRACKLET_COLUMNS),
        label="S01 microtracklets",
    )
    edges = _parquet_columns(
        microtrack_dir / "det_edges.parquet",
        expected_schema=DET_EDGES_SCHEMA,
        expected_rows=counts["num_accepted_edges"],
        columns=list(dict.fromkeys((*DET_EDGE_COLUMNS, "accepted"))),
        label="S01 det_edges",
    )
    _validate_frames(
        frames,
        video_paths,
        expected_sequence_id=sequence_id,
        expected_num_frames=counts["num_frames"],
    )
    _validate_detection_frame_foreign_keys(
        frames,
        detections,
        expected_valid_detections=counts["num_valid_detections"],
    )
    indices = _build_indices(frames, detections, mapping, microtracklets)
    input_fingerprints = sorted(
        [
            *s00_fingerprints,
            *s01_fingerprints,
            *video_fingerprints,
            _fingerprint(config_path),
        ],
        key=lambda record: record["path"],
    )
    return LoadedReviewData(
        frames=frames,
        detections=detections,
        det_to_micro=mapping,
        microtracklets=microtracklets,
        det_edges=edges,
        video_paths=video_paths,
        input_fingerprints=tuple(input_fingerprints),
        valid_detection_positions_by_frame=indices[0],
        valid_detection_frame_offsets=indices[1],
        micro_paths=indices[2],
    )


def _plan(data: LoadedReviewData, config: S01ReviewConfig) -> ReviewPlan:
    frame_times = np.asarray(data.frames["global_time_sec"], dtype=np.float64)
    return build_review_plan(
        detections=data.detections,
        det_to_micro=data.det_to_micro,
        microtracklets=data.microtracklets,
        det_edges=data.det_edges,
        window_before_sec=config.window_before_sec,
        window_after_sec=config.window_after_sec,
        long_track_min_detections=config.long_track_min_detections,
        long_track_chunk_sec=config.long_track_chunk_sec,
        long_track_chunk_overlap_sec=config.long_track_chunk_overlap_sec,
        low_purity_threshold=config.low_purity_threshold,
        high_jump_threshold=config.high_jump_threshold,
        requested_quality_samples=config.good_sample_count,
        quality_min_detections=config.good_min_detections,
        quality_max_detections=config.good_max_detections,
        quality_min_purity=config.good_min_purity,
        quality_max_jump=config.good_max_jump,
        timeline_start_time_sec=float(frame_times[0]),
        timeline_end_time_sec=float(frame_times[-1]),
    )


def _true_trigger_frames(case: ReviewCase) -> set[int]:
    return {
        int(event["global_frame"])
        for fields in case.events
        for event in (dict(fields),)
        if event.get("reason") in TRUE_EVENT_REASONS and "global_frame" in event
    }


def _render_cases(data: LoadedReviewData, plan: ReviewPlan) -> tuple[RenderCase, ...]:
    frame_times = np.asarray(data.frames["global_time_sec"], dtype=np.float64)
    det_frames = np.asarray(data.detections["global_frame"], dtype=np.int64)
    result: list[RenderCase] = []
    paths_seen: set[str] = set()
    for case in plan.cases:
        start = int(np.searchsorted(frame_times, case.window_start_time_sec, side="left"))
        end = int(np.searchsorted(frame_times, case.window_end_time_sec, side="right") - 1)
        start = max(0, min(start, len(frame_times) - 1))
        end = max(0, min(end, len(frame_times) - 1))
        if end < start or not start <= case.anchor_global_frame <= end:
            raise ContractError(f"case {case.case_id} has an invalid frame window")
        path = data.micro_paths[case.micro_id]
        in_window = path[(det_frames[path] >= start) & (det_frames[path] <= end)]
        if len(in_window) == 0:
            raise ContractError(f"case {case.case_id} window contains no target detection")
        folder = "risk" if case.case_kind == "risk" else "quality_reference"
        relative_path = f"{folder}/{case.suggested_filename}"
        if relative_path in paths_seen:
            raise ContractError(f"duplicate review output path: {relative_path}")
        paths_seen.add(relative_path)
        result.append(RenderCase(case, start, end, relative_path))
    return tuple(result)


def _validate_plan(plan: ReviewPlan, render_cases: tuple[RenderCase, ...]) -> None:
    if len(render_cases) != len(plan.cases):
        raise ContractError("S01 review plan/render case counts differ")
    case_ids = [case.case_id for case in plan.cases]
    if any(not case_id for case_id in case_ids) or len(set(case_ids)) != len(case_ids):
        raise ContractError("S01 review plan case_id values are blank or duplicated")
    if any(
        case.case_kind not in {"risk", "quality_reference"}
        for case in plan.cases
    ):
        raise ContractError("S01 review plan contains an unknown case kind")
    if tuple(item.case for item in render_cases) != plan.cases:
        raise ContractError("S01 review render cases differ from the plan order")
    if any(item.expected_frame_count <= 0 for item in render_cases):
        raise ContractError("S01 review plan contains an empty render window")
    output_paths = [item.output_relative_path for item in render_cases]
    if len(set(output_paths)) != len(output_paths):
        raise ContractError("S01 review plan repeats an output path")

    risk_microtracks = tuple(sorted(set(plan.risk_microtracks)))
    if risk_microtracks != plan.risk_microtracks:
        raise ContractError("S01 review risk_microtracks are not sorted and unique")
    observed_risk_microtracks = tuple(
        sorted({case.micro_id for case in plan.risk_cases})
    )
    if observed_risk_microtracks != risk_microtracks:
        raise ContractError("S01 review risk cases and risk_microtracks differ")
    if any(case.micro_id in risk_microtracks for case in plan.quality_cases):
        raise ContractError("S01 review quality case also belongs to a risk microtrack")
    if len(plan.quality_cases) > plan.requested_quality_samples:
        raise ContractError("S01 review selected too many quality-reference cases")

    true_events: set[tuple[int, str, int, int, int]] = set()
    for case in plan.risk_cases:
        for fields in case.events:
            event = dict(fields)
            reason = str(event.get("reason", ""))
            if reason not in TRUE_EVENT_REASONS:
                continue
            try:
                key = (
                    case.micro_id,
                    reason,
                    int(event["global_frame"]),
                    int(event["src_det_id"]),
                    int(event["dst_det_id"]),
                )
            except (KeyError, TypeError, ValueError, OverflowError) as exc:
                raise ContractError("S01 review true event lacks edge provenance") from exc
            if key in true_events:
                raise ContractError("S01 review true event is duplicated")
            true_events.add(key)


class _RawFrameReader:
    def __init__(
        self,
        video_paths: dict[str, Path],
        *,
        capture_factory: CaptureFactory = open_raw_video_capture,
    ) -> None:
        self._video_paths = video_paths
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

    def __enter__(self) -> _RawFrameReader:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool:
        self.close()
        return False

    def read(self, clip_id: str, local_frame: int, expected_pts_sec: float) -> np.ndarray:
        if clip_id not in self._video_paths:
            raise ContractError(f"frame references unknown clip_id: {clip_id}")
        if self._capture is None or self._clip_id != clip_id:
            self.close()
            self._capture = self._capture_factory(self._video_paths[clip_id])
            self._clip_id = clip_id
            self._next_local_frame = None
        assert self._capture is not None
        if self._next_local_frame != local_frame:
            if not self._capture.set(cv2.CAP_PROP_POS_FRAMES, int(local_frame)):
                raise ContractError(f"cannot seek {clip_id} local frame {local_frame}")
        ok, frame = self._capture.read()
        if not ok or frame is None:
            raise ContractError(f"cannot decode {clip_id} local frame {local_frame}")
        if frame.shape != (RAW_HEIGHT, RAW_WIDTH, 3) or frame.dtype != np.uint8:
            raise ContractError(
                f"decoded raw frame geometry/type mismatch at {clip_id}:{local_frame}"
            )
        position_after = int(round(self._capture.get(cv2.CAP_PROP_POS_FRAMES)))
        if position_after != local_frame + 1:
            raise ContractError(
                f"OpenCV seek/decode mismatch at {clip_id}: {position_after - 1} != {local_frame}"
            )
        if not hasattr(cv2, "CAP_PROP_PTS"):
            raise ContractError("OpenCV lacks CAP_PROP_PTS for exact review decoding")
        decoded_pts_frame = int(round(self._capture.get(cv2.CAP_PROP_PTS)))
        if decoded_pts_frame != local_frame:
            raise ContractError(
                f"OpenCV PTS frame mismatch at {clip_id}: {decoded_pts_frame} != {local_frame}"
            )
        decoded_msec = float(self._capture.get(cv2.CAP_PROP_POS_MSEC))
        if not math.isclose(
            decoded_msec, expected_pts_sec * 1000.0, rel_tol=0.0, abs_tol=0.1
        ):
            raise ContractError(
                f"OpenCV PTS time mismatch at {clip_id}:{local_frame}: "
                f"{decoded_msec} != {expected_pts_sec * 1000.0} ms"
            )
        self._next_local_frame = local_frame + 1
        return frame


def _preflight_nvenc(config: S01ReviewConfig) -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
        raise ContractError(
            "S01 review requires CUDA_VISIBLE_DEVICES exactly equal to '1'"
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
        raise ContractError(f"cannot start NVENC preflight: {exc}") from exc
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise ContractError(
            f"GPU 1 NVENC preflight failed ({completed.returncode}): "
            f"{detail or 'no FFmpeg diagnostics'}"
        )


def _context_boxes(data: LoadedReviewData, global_frame: int) -> np.ndarray:
    start = int(data.valid_detection_frame_offsets[global_frame])
    stop = int(data.valid_detection_frame_offsets[global_frame + 1])
    positions = data.valid_detection_positions_by_frame[start:stop]
    detections = data.detections
    return np.column_stack(
        [
            np.asarray(detections[name], dtype=np.float64)[positions]
            for name in ("x1", "y1", "x2", "y2")
        ]
    )


def _target_bbox_at_frame(
    data: LoadedReviewData, micro_id: int, global_frame: int
) -> np.ndarray | None:
    path = data.micro_paths[micro_id]
    path_frames = np.asarray(data.detections["global_frame"], dtype=np.int64)[path]
    location = int(np.searchsorted(path_frames, global_frame))
    if location >= len(path) or path_frames[location] != global_frame:
        return None
    position = int(path[location])
    return np.asarray(
        [data.detections[name][position] for name in ("x1", "y1", "x2", "y2")],
        dtype=np.float64,
    )


def _render_one_case(
    data: LoadedReviewData,
    render_case: RenderCase,
    config: S01ReviewConfig,
    output_dir: Path,
    *,
    logger: LogFn,
) -> dict[str, Any]:
    case = render_case.case
    output_path = output_dir / render_case.output_relative_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    trigger_frames = _true_trigger_frames(case)
    event_highlight_frames = {
        frame
        for trigger in trigger_frames
        for frame in range(
            trigger - EVENT_HIGHLIGHT_RADIUS_FRAMES,
            trigger + EVENT_HIGHLIGHT_RADIUS_FRAMES + 1,
        )
    }
    frames = data.frames
    last_report = time.monotonic()
    with _RawFrameReader(data.video_paths) as reader, NvencVideoWriter(
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
        for completed, global_frame in enumerate(
            range(render_case.start_global_frame, render_case.end_global_frame + 1),
            start=1,
        ):
            clip_id = str(frames["clip_id"][global_frame])
            local_frame = int(frames["local_frame"][global_frame])
            raw = reader.read(clip_id, local_frame, float(frames["pts_sec"][global_frame]))
            target_box = _target_bbox_at_frame(
                data, case.micro_id, global_frame
            )
            context = (
                _context_boxes(data, global_frame)
                if config.draw_context_boxes
                else np.empty((0, 4), dtype=np.float64)
            )
            try:
                annotated = render_review_frame(
                    raw,
                    output_width=config.output_width,
                    output_height=config.output_height,
                    context_boxes=context,
                    target_bbox=target_box,
                    case_kind=case.case_kind,
                    event_frame=global_frame in event_highlight_frames,
                )
            except (TypeError, ValueError, cv2.error) as exc:
                raise ContractError(
                    f"cannot annotate {case.case_id} global frame {global_frame}: {exc}"
                ) from exc
            writer.write(np.ascontiguousarray(annotated))
            now = time.monotonic()
            if now - last_report >= config.progress_interval_sec:
                logger(
                    f"[s01-review] {case.case_id}: {completed:,}/"
                    f"{render_case.expected_frame_count:,} frames"
                )
                last_report = now
    metadata = validate_qa_mp4(
        output_path,
        expected_frame_rate=EXPECTED_FRAME_RATE,
        expected_frame_count=render_case.expected_frame_count,
        ffprobe_binary=config.ffprobe_binary,
    )
    fingerprint = _fingerprint(output_path)
    logger(
        f"[s01-review] completed {case.case_id}: "
        f"{render_case.expected_frame_count:,} frames, {fingerprint['size_bytes'] / 1024**2:.1f} MiB"
    )
    render_record = {
        "status": "completed",
        "completed_at_unix_sec": time.time(),
        "output_fingerprint": {
            "path": render_case.output_relative_path,
            "size_bytes": fingerprint["size_bytes"],
            "mtime_ns": fingerprint["mtime_ns"],
            "sha256": fingerprint["sha256"],
        },
        "video_validation": {
            "codec_name": metadata.codec_name,
            "width": metadata.width,
            "height": metadata.height,
            "average_frame_rate": str(metadata.average_frame_rate),
            "num_frames": metadata.frame_count,
        },
    }
    journal = {
        "schema_version": COMPLETION_SCHEMA_VERSION,
        "case_id": case.case_id,
        "case_commitment": _case_commitment(_case_record(render_case)),
        "expected_frame_count": render_case.expected_frame_count,
        "output_path": render_case.output_relative_path,
        "render": render_record,
    }
    _atomic_write_json(_completion_journal_path(output_path), journal)
    return render_record


def _case_record(render_case: RenderCase) -> dict[str, Any]:
    record = render_case.case.to_manifest_record()
    record.update(
        {
            "start_global_frame": render_case.start_global_frame,
            "end_global_frame": render_case.end_global_frame,
            "expected_frame_count": render_case.expected_frame_count,
            "output_path": render_case.output_relative_path,
            "encoded_duration_sec": (
                render_case.expected_frame_count / float(EXPECTED_FRAME_RATE)
            ),
            "render": {"status": "pending"},
        }
    )
    return record


def _write_or_validate_labels(path: Path, case_records: list[dict[str, Any]]) -> None:
    expected_ids = [str(record["case_id"]) for record in case_records]
    if path.is_file():
        try:
            with path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                if tuple(reader.fieldnames or ()) != LABEL_COLUMNS:
                    raise ContractError(f"{LABELS_NAME} column contract changed")
                rows = list(reader)
        except OSError as exc:
            raise ContractError(f"cannot read {LABELS_NAME}: {exc}") from exc
        if [row["case_id"] for row in rows] != expected_ids:
            raise ContractError(f"{LABELS_NAME} case order/IDs differ from review plan")
        verdicts: set[str] = set()
        for row, record in zip(rows, case_records, strict=True):
            immutable_expected = {
                "case_kind": str(record["case_kind"]),
                "reasons": ";".join(record["reasons"]),
                "video_path": str(record["output_path"]),
            }
            for key, expected in immutable_expected.items():
                if row.get(key) != expected:
                    raise ContractError(
                        f"{LABELS_NAME} immutable field changed for "
                        f"{record['case_id']}: {key}"
                    )
            raw_verdict = row.get("verdict")
            if not isinstance(raw_verdict, str):
                raise ContractError(f"{LABELS_NAME} contains a malformed row")
            verdicts.add(raw_verdict.strip().upper())
        invalid = sorted(verdicts - ALLOWED_VERDICTS)
        if invalid:
            raise ContractError(f"{LABELS_NAME} contains invalid verdicts: {invalid}")
        return

    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=LABEL_COLUMNS)
            writer.writeheader()
            for record in case_records:
                writer.writerow(
                    {
                        "case_id": record["case_id"],
                        "case_kind": record["case_kind"],
                        "reasons": ";".join(record["reasons"]),
                        "video_path": record["output_path"],
                        "verdict": "",
                        "reviewer": "",
                        "notes": "",
                    }
                )
        os.replace(temporary, path)
    except OSError as exc:
        temporary.unlink(missing_ok=True)
        raise ContractError(f"cannot create {LABELS_NAME}: {exc}") from exc


def _manifest_identity(
    *,
    config_hash: str,
    input_fingerprints: tuple[dict[str, Any], ...],
    plan_payload: dict[str, Any],
    case_records: list[dict[str, Any]],
) -> dict[str, Any]:
    immutable_cases = [
        {key: value for key, value in record.items() if key != "render"}
        for record in case_records
    ]
    identity = {
        "config_hash": config_hash,
        "input_fingerprints": list(input_fingerprints),
        "plan_hash": _canonical_hash(
            {"selection": plan_payload, "render_cases": immutable_cases}
        ),
    }
    return identity


def _validate_completed_record(
    record: dict[str, Any], output_dir: Path, config: S01ReviewConfig
) -> None:
    render = record.get("render")
    if not isinstance(render, dict) or render.get("status") != "completed":
        raise ContractError(f"case is not completed: {record.get('case_id')}")
    output_path = output_dir / str(record["output_path"])
    expected = render.get("output_fingerprint")
    if not isinstance(expected, dict):
        raise ContractError(f"completed case has no fingerprint: {record['case_id']}")
    if expected.get("path") != record.get("output_path"):
        raise ContractError(f"completed case fingerprint path mismatch: {record['case_id']}")
    current = _fingerprint(output_path)
    for key in ("size_bytes", "sha256"):
        if current[key] != expected.get(key):
            raise ContractError(f"completed review video changed: {record['output_path']} ({key})")
    metadata = validate_qa_mp4(
        output_path,
        expected_frame_rate=EXPECTED_FRAME_RATE,
        expected_frame_count=int(record["expected_frame_count"]),
        ffprobe_binary=config.ffprobe_binary,
    )
    expected_validation = {
        "codec_name": metadata.codec_name,
        "width": metadata.width,
        "height": metadata.height,
        "average_frame_rate": str(metadata.average_frame_rate),
        "num_frames": metadata.frame_count,
    }
    if render.get("video_validation") != expected_validation:
        raise ContractError(f"completed case video validation changed: {record['case_id']}")


def _adopt_uncommitted_video(
    record: dict[str, Any], output_dir: Path, config: S01ReviewConfig
) -> dict[str, Any]:
    output_path = output_dir / str(record["output_path"])
    journal_path = _completion_journal_path(output_path)
    journal = _read_json(journal_path, "case completion journal")
    if not isinstance(journal, dict):
        raise ContractError(f"completion journal is not an object: {journal_path}")
    if set(journal) != {
        "schema_version",
        "case_id",
        "case_commitment",
        "expected_frame_count",
        "output_path",
        "render",
    }:
        raise ContractError(f"completion journal keys mismatch: {journal_path}")
    expected_header = {
        "schema_version": COMPLETION_SCHEMA_VERSION,
        "case_id": str(record["case_id"]),
        "case_commitment": _case_commitment(record),
        "expected_frame_count": int(record["expected_frame_count"]),
        "output_path": str(record["output_path"]),
    }
    for key, expected in expected_header.items():
        if journal.get(key) != expected:
            raise ContractError(
                f"completion journal does not match case {record['case_id']}: {key}"
            )
    render = journal.get("render")
    if not isinstance(render, dict) or render.get("status") != "completed":
        raise ContractError(f"completion journal has no completed render: {journal_path}")
    candidate = dict(record)
    candidate["render"] = render
    _validate_completed_record(candidate, output_dir, config)
    recovered = dict(render)
    recovered["recovered_after_interruption"] = True
    return recovered


def _initial_or_resumed_manifest(
    output_dir: Path,
    *,
    config_payload: dict[str, Any],
    identity: dict[str, Any],
    plan_payload: dict[str, Any],
    case_records: list[dict[str, Any]],
    config: S01ReviewConfig,
) -> dict[str, Any]:
    manifest_path = output_dir / MANIFEST_NAME
    output_video_contract = {
        "codec": "h264_nvenc",
        "physical_gpu": 1,
        "logical_gpu": config.logical_gpu,
        "width": config.output_width,
        "height": config.output_height,
        "average_frame_rate": str(EXPECTED_FRAME_RATE),
        "rotation": None,
        "cpu_encoder_fallback": False,
        "render_style": "bbox_colors_only_v1",
        "text": False,
        "inset": False,
        "trajectory": False,
        "markers": False,
        "bbox_colors_bgr": {
            "context": [150, 150, 150],
            "risk_target": [0, 165, 255],
            "quality_target": [45, 205, 70],
            "event_target": [40, 40, 235],
        },
        "true_event_highlight_radius_frames": EVENT_HIGHLIGHT_RADIUS_FRAMES,
        "synthetic_long_and_quality_anchors_are_not_events": True,
    }
    selection_warning = (
        "Risk reasons select material for human review; they are not automatic FAIL labels. "
        "legacy_track_id is QA-only and is not ground truth."
    )
    if manifest_path.is_file():
        existing = _read_json(manifest_path, "S01 review manifest")
        if not isinstance(existing, dict):
            raise ContractError("existing review manifest is not a JSON object")
        if existing.get("schema_version") != MANIFEST_SCHEMA_VERSION:
            raise ContractError("existing review manifest schema mismatch")
        if existing.get("identity") != identity:
            raise ContractError("existing review output belongs to different inputs/config/plan")
        expected_top_level = {
            "coordinate_system": "raw_encoded_landscape_no_autorotate",
            "output_video_contract": output_video_contract,
            "selection_warning": selection_warning,
            "effective_config": config_payload,
            "selection_plan": plan_payload,
        }
        for key, expected in expected_top_level.items():
            if existing.get(key) != expected:
                raise ContractError(f"existing review manifest changed: {key}")
        existing_cases = existing.get("cases")
        if not isinstance(existing_cases, list):
            raise ContractError("existing review manifest has no cases list")
        expected_immutable = [
            {key: value for key, value in record.items() if key != "render"}
            for record in case_records
        ]
        observed_immutable = [
            {key: value for key, value in record.items() if key != "render"}
            for record in existing_cases
            if isinstance(record, dict)
        ]
        if observed_immutable != expected_immutable:
            raise ContractError("existing review manifest case contract changed")
        expected_video_paths = {
            str(record["output_path"]) for record in existing_cases
        }
        observed_video_paths = {
            str(path.relative_to(output_dir))
            for path in output_dir.rglob("*.mp4")
            if not path.name.endswith(".part.mp4")
        }
        unexpected_videos = sorted(observed_video_paths - expected_video_paths)
        if unexpected_videos:
            raise ContractError(
                f"review output contains videos absent from the plan: {unexpected_videos}"
            )
        expected_journal_paths = {
            f"{record['output_path']}.completion.json" for record in existing_cases
        }
        observed_journal_paths = {
            str(path.relative_to(output_dir))
            for path in output_dir.rglob("*.mp4.completion.json")
        }
        unexpected_journals = sorted(
            observed_journal_paths - expected_journal_paths
        )
        if unexpected_journals:
            raise ContractError(
                "review output contains completion journals absent from the plan: "
                f"{unexpected_journals}"
            )
        recovered = False
        recovered_outputs: list[Path] = []
        last_report = time.monotonic()
        for record in existing_cases:
            render = record.get("render", {})
            output_path = output_dir / str(record.get("output_path", ""))
            journal_path = _completion_journal_path(output_path)
            if isinstance(render, dict) and render.get("status") == "completed":
                _validate_completed_record(record, output_dir, config)
                _remove_completion_journal(output_path)
            elif output_path.is_file():
                if journal_path.is_file():
                    record["render"] = _adopt_uncommitted_video(
                        record, output_dir, config
                    )
                    recovered = True
                    recovered_outputs.append(output_path)
                else:
                    try:
                        output_path.unlink()
                    except OSError as exc:
                        raise ContractError(
                            f"cannot remove uncommitted review video {output_path}: {exc}"
                        ) from exc
                    log(
                        "[s01-review] removed uncommitted video without a trusted "
                        f"completion journal: {output_path}"
                    )
            elif journal_path.is_file():
                _remove_completion_journal(output_path)
            now = time.monotonic()
            if now - last_report >= config.progress_interval_sec:
                log("[s01-review] validating completed videos for resume")
                last_report = now
        if recovered:
            _atomic_write_json(manifest_path, existing)
            for output_path in recovered_outputs:
                _remove_completion_journal(output_path)
        return existing

    allowed_initial = {"risk", "quality_reference", LOCK_NAME}
    existing_names = {path.name for path in output_dir.iterdir()}
    if existing_names - allowed_initial:
        raise ContractError(
            f"review output is non-empty without {MANIFEST_NAME}: {output_dir}"
        )
    for folder_name in ("quality_reference", "risk"):
        folder = output_dir / folder_name
        if not folder.is_dir() or any(folder.iterdir()):
            raise ContractError(
                f"review output has orphan content without {MANIFEST_NAME}: {folder}"
            )
    payload = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "coordinate_system": "raw_encoded_landscape_no_autorotate",
        "output_video_contract": output_video_contract,
        "selection_warning": selection_warning,
        "identity": identity,
        "effective_config": config_payload,
        "selection_plan": plan_payload,
        "cases": case_records,
    }
    _atomic_write_json(manifest_path, payload)
    return payload


def _verify_inputs_unchanged(
    fingerprints: tuple[dict[str, Any], ...],
    *,
    progress_interval_sec: float,
    logger: LogFn,
) -> None:
    for fingerprint in fingerprints:
        path = Path(str(fingerprint["path"]))
        progress_logger = (
            logger if int(fingerprint["size_bytes"]) >= 1024**3 else None
        )
        current = _fingerprint(
            path,
            progress_interval_sec=progress_interval_sec,
            logger=progress_logger,
        )
        for key in ("size_bytes", "mtime_ns", "sha256"):
            if current[key] != fingerprint[key]:
                raise ContractError(
                    f"review input changed during rendering: {path} ({key})"
                )


def run_s01_review(
    ingest_dir: Path,
    microtrack_dir: Path,
    config_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Render strict, resumable S01 human-review videos without mutating inputs."""

    ingest_dir = ingest_dir.resolve()
    microtrack_dir = microtrack_dir.resolve()
    config_path = config_path.resolve()
    output_dir = output_dir.resolve()
    for label, input_dir in (
        ("S00 ingest", ingest_dir),
        ("S01 microtrack", microtrack_dir),
    ):
        if (
            output_dir == input_dir
            or output_dir.is_relative_to(input_dir)
            or input_dir.is_relative_to(output_dir)
        ):
            raise ContractError(
                f"review --output must not overlap the immutable {label} tree: "
                f"output={output_dir}, input={input_dir}"
            )
    config, config_payload, config_hash = load_s01_review_config(config_path)
    _preflight_nvenc(config)
    logger = log
    logger("[s01-review] GPU 1 NVENC preflight passed; loading immutable S00/S01 inputs")
    data = _load_review_data(
        ingest_dir,
        microtrack_dir,
        config_path,
        progress_interval_sec=config.progress_interval_sec,
        logger=logger,
    )
    logger("[s01-review] immutable inputs and fingerprints validated; building review plan")
    plan = _plan(data, config)
    render_cases = _render_cases(data, plan)
    _validate_plan(plan, render_cases)
    logger("[s01-review] review-plan consistency checks passed")
    plan_payload = plan.to_manifest_payload()
    case_records = [_case_record(render_case) for render_case in render_cases]
    identity = _manifest_identity(
        config_hash=config_hash,
        input_fingerprints=data.input_fingerprints,
        plan_payload=plan_payload,
        case_records=case_records,
    )

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ContractError(f"cannot create review output directory: {exc}") from exc
    output_lock = _acquire_output_lock(output_dir)
    _remove_stale_encoder_parts(output_dir)
    try:
        (output_dir / "risk").mkdir(exist_ok=True)
        (output_dir / "quality_reference").mkdir(exist_ok=True)
    except OSError as exc:
        raise ContractError(f"cannot create review output directories: {exc}") from exc
    manifest = _initial_or_resumed_manifest(
        output_dir,
        config_payload=config_payload,
        identity=identity,
        plan_payload=plan_payload,
        case_records=case_records,
        config=config,
    )
    _write_or_validate_labels(output_dir / LABELS_NAME, manifest["cases"])
    success_path = output_dir / SUCCESS_NAME
    if success_path.is_file():
        existing_success = _read_json(success_path, "S01 review success marker")
        if not isinstance(existing_success, dict):
            raise ContractError("existing S01 review success marker is not an object")
        if any(
            not isinstance(record.get("render"), dict)
            or record["render"].get("status") != "completed"
            for record in manifest["cases"]
        ):
            raise ContractError(
                "review success marker exists while manifest has pending cases"
            )
        expected_success_fields = {
            "schema_version": SUCCESS_SCHEMA_VERSION,
            "stage": "S01_HUMAN_VIDEO_REVIEW",
            "identity": identity,
            "manifest_sha256": _sha256(output_dir / MANIFEST_NAME),
            "stats": {
                "num_cases": len(manifest["cases"]),
                "num_risk_cases": len(plan.risk_cases),
                "num_quality_reference_cases": len(plan.quality_cases),
                "num_frames_rendered": sum(
                    item.expected_frame_count for item in render_cases
                ),
            },
            "output_fingerprints": [
                record["render"]["output_fingerprint"]
                for record in manifest["cases"]
            ],
            "human_labels_file": LABELS_NAME,
            "human_labels_are_mutable_and_not_fingerprinted": True,
        }
        for key, expected in expected_success_fields.items():
            if existing_success.get(key) != expected:
                raise ContractError(
                    f"existing S01 review success marker mismatch: {key}"
                )
        last_report = time.monotonic()
        for index, record in enumerate(manifest["cases"], start=1):
            _validate_completed_record(record, output_dir, config)
            now = time.monotonic()
            if now - last_report >= config.progress_interval_sec:
                logger(
                    f"[s01-review] revalidate completed output: {index:,}/"
                    f"{len(manifest['cases']):,} videos"
                )
                last_report = now
        logger(f"[s01-review] already complete and fully revalidated: {success_path}")
        _release_output_lock(output_lock)
        return manifest

    logger(
        f"[s01-review] plan: {len(plan.risk_cases):,} risk videos, "
        f"{len(plan.quality_cases):,} quality references"
    )
    render_by_id = {item.case.case_id: item for item in render_cases}
    for index, record in enumerate(manifest["cases"], start=1):
        render = record.get("render", {})
        if isinstance(render, dict) and render.get("status") == "completed":
            logger(
                f"[s01-review] resume {index:,}/{len(render_cases):,}: "
                f"already complete {record['case_id']}"
            )
            continue
        logger(
            f"[s01-review] render {index:,}/{len(render_cases):,}: "
            f"{record['case_id']} -> {record['output_path']}"
        )
        record["render"] = _render_one_case(
            data,
            render_by_id[str(record["case_id"])],
            config,
            output_dir,
            logger=logger,
        )
        _atomic_write_json(output_dir / MANIFEST_NAME, manifest)
        _remove_completion_journal(
            output_dir / str(record["output_path"])
        )

    logger("[s01-review] re-fingerprinting immutable inputs before final commit")
    _verify_inputs_unchanged(
        data.input_fingerprints,
        progress_interval_sec=config.progress_interval_sec,
        logger=logger,
    )
    last_report = time.monotonic()
    for index, record in enumerate(manifest["cases"], start=1):
        _validate_completed_record(record, output_dir, config)
        now = time.monotonic()
        if now - last_report >= config.progress_interval_sec:
            logger(
                f"[s01-review] final validation: {index:,}/"
                f"{len(manifest['cases']):,} videos"
            )
            last_report = now
    success = {
        "schema_version": SUCCESS_SCHEMA_VERSION,
        "stage": "S01_HUMAN_VIDEO_REVIEW",
        "identity": identity,
        "manifest_sha256": _sha256(output_dir / MANIFEST_NAME),
        "stats": {
            "num_cases": len(manifest["cases"]),
            "num_risk_cases": len(plan.risk_cases),
            "num_quality_reference_cases": len(plan.quality_cases),
            "num_frames_rendered": sum(item.expected_frame_count for item in render_cases),
        },
        "output_fingerprints": [
            record["render"]["output_fingerprint"] for record in manifest["cases"]
        ],
        "human_labels_file": LABELS_NAME,
        "human_labels_are_mutable_and_not_fingerprinted": True,
    }
    _atomic_write_json(success_path, success)
    logger(f"[s01-review] complete: {success_path}")
    _release_output_lock(output_lock)
    return manifest


__all__ = [
    "EXPECTED_FRAME_RATE",
    "LoadedReviewData",
    "RenderCase",
    "run_s01_review",
]
