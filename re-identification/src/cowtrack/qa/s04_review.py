"""Strict, resumable S04 provisional-link review-video renderer."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any, Callable

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from cowtrack.config import ContractError
from cowtrack.linking.dataset_contract import (
    EXPECTED_CLIP_ORDER,
    EXPECTED_SEQUENCE_ID,
)
from cowtrack.qa.ffprobe import validate_qa_mp4
from cowtrack.qa.nvenc import NvencVideoWriter
from cowtrack.qa.s04_review_config import S04ReviewConfig, load_s04_review_config
from cowtrack.qa.s04_review_plan import (
    PURPLE_BOX_POLICY,
    S04ReviewCase,
    S04ReviewPlan,
    build_s04_review_plan,
)
from cowtrack.qa.s04_review_render import (
    INTERMEDIATE_BGR,
    SOURCE_BGR,
    TARGET_BGR,
    UNRELATED_BGR,
    render_s04_review_frame,
)
from cowtrack.schemas.detections import DETECTIONS_SCHEMA
from cowtrack.schemas.frames import FRAMES_SCHEMA
from cowtrack.schemas.s04 import (
    SHORT_CANDIDATE_EDGES_SCHEMA,
    SHORT_LINK_PROPOSALS_SCHEMA,
)
from cowtrack.schemas.tracklets import DET_TO_MICRO_SCHEMA, MICROTRACKLETS_SCHEMA
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
MANIFEST_SCHEMA_VERSION = "cowtrack.s04-video-review.v1"
SUCCESS_SCHEMA_VERSION = "cowtrack.s04-video-review-success.v1"


def log(message: str) -> None:
    print(message, flush=True)


@dataclass(frozen=True)
class CandidateEndpoint:
    edge_id: str
    source_micro_id: int
    target_micro_id: int
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
    probability: float
    high_overlap: bool


@dataclass(frozen=True)
class RenderCase:
    case: S04ReviewCase
    endpoint: CandidateEndpoint
    intermediate_micro_ids: tuple[int, ...]
    start_global_frame: int
    end_global_frame: int
    output_relative_path: str

    @property
    def expected_frame_count(self) -> int:
        return self.end_global_frame - self.start_global_frame + 1


@dataclass(frozen=True)
class LoadedS04ReviewData:
    frames: dict[str, np.ndarray]
    detections: dict[str, np.ndarray]
    micro_id_by_detection_position: np.ndarray
    valid_detection_positions_by_frame: np.ndarray
    valid_detection_frame_offsets: np.ndarray
    candidates_by_edge_id: dict[str, CandidateEndpoint]
    video_paths: dict[str, Path]
    plan: S04ReviewPlan
    input_fingerprints: tuple[dict[str, Any], ...]


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
                        f"[s04-review] fingerprint {path.name}: "
                        f"{completed / 1024**3:.1f} GiB"
                    )
                    last_report = now
    except OSError as exc:
        raise ContractError(f"cannot fingerprint {path}: {exc}") from exc
    return digest.hexdigest()


def _fingerprint(
    path: Path,
    *,
    progress_interval_sec: float | None = None,
    logger: LogFn | None = None,
) -> dict[str, Any]:
    try:
        before = path.stat()
    except OSError as exc:
        raise ContractError(f"required review input does not exist: {path}: {exc}") from exc
    if not path.is_file():
        raise ContractError(f"required review input is not a file: {path}")
    digest = _sha256(
        path,
        progress_interval_sec=progress_interval_sec,
        logger=logger,
    )
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ContractError(f"review input changed while fingerprinting: {path}")
    return {
        "path": str(path.resolve()),
        "size_bytes": int(before.st_size),
        "mtime_ns": int(before.st_mtime_ns),
        "sha256": digest,
    }


def _validated_stage_artifacts(
    directory: Path,
    *,
    expected_stages: frozenset[str],
    required_names: tuple[str, ...],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    success_path = directory / SUCCESS_NAME
    success = _read_json(success_path, "completed stage marker")
    if not isinstance(success, dict) or success.get("stage") not in expected_stages:
        raise ContractError(
            f"{success_path} stage must be one of {sorted(expected_stages)}"
        )
    records = success.get("output_fingerprints")
    if not isinstance(records, list):
        raise ContractError(f"completed marker lacks output fingerprints: {success_path}")
    by_path = {
        str(record.get("path")): record
        for record in records
        if isinstance(record, dict)
    }
    missing = sorted(set(required_names) - set(by_path))
    if missing:
        raise ContractError(f"completed marker lacks artifacts: {missing}")
    observed: list[dict[str, Any]] = []
    for name in required_names:
        current = _fingerprint((directory / name).resolve())
        recorded = by_path[name]
        for key in ("size_bytes", "sha256"):
            if current[key] != recorded.get(key):
                raise ContractError(f"completed artifact changed: {name} ({key})")
        observed.append(current)
    observed.append(_fingerprint(success_path.resolve()))
    return success, observed


def _read_parquet_columns(
    path: Path,
    *,
    expected_schema: pa.Schema,
    columns: tuple[str, ...],
    label: str,
) -> dict[str, np.ndarray]:
    try:
        parquet = pq.ParquetFile(path)
    except (OSError, pa.ArrowException) as exc:
        raise ContractError(f"cannot open {label}: {path}: {exc}") from exc
    if not parquet.schema_arrow.equals(expected_schema, check_metadata=False):
        raise ContractError(f"{label} schema mismatch: {path}")
    try:
        table = pq.read_table(path, columns=list(columns))
    except (OSError, pa.ArrowException) as exc:
        raise ContractError(f"cannot read {label}: {path}: {exc}") from exc
    return {
        name: table[name].combine_chunks().to_numpy(zero_copy_only=False)
        for name in columns
    }


def _load_video_paths(
    ingest_dir: Path,
    s00_success: dict[str, Any],
    *,
    progress_interval_sec: float,
    logger: LogFn,
) -> tuple[dict[str, Path], list[dict[str, Any]], list[dict[str, Any]]]:
    rows = _read_json(ingest_dir / "resolved_manifest.json", "S00 resolved manifest")
    if not isinstance(rows, list) or len(rows) != len(EXPECTED_CLIP_IDS):
        raise ContractError("S04 review requires the complete resolved clip sequence")
    recorded_inputs = {
        str(Path(str(record.get("path"))).resolve()): record
        for record in s00_success.get("input_fingerprints", [])
        if isinstance(record, dict) and record.get("path")
    }
    paths: dict[str, Path] = {}
    provenance: list[dict[str, Any]] = []
    stats: list[dict[str, Any]] = []
    sequence_ids: set[str] = set()
    for expected_order, row in enumerate(rows):
        if not isinstance(row, dict) or int(row.get("clip_order", -1)) != expected_order:
            raise ContractError("S00 resolved manifest clip order is not contiguous")
        sequence_ids.add(str(row.get("sequence_id", "")))
        clip_id = str(row.get("clip_id", ""))
        raw_path = row.get("video_path")
        if clip_id != EXPECTED_CLIP_IDS[expected_order]:
            raise ContractError("S04 review resolved clip order differs")
        if clip_id in paths or not isinstance(raw_path, str):
            raise ContractError("S00 resolved manifest clip/video identifiers are invalid")
        path = Path(raw_path).resolve()
        recorded = recorded_inputs.get(str(path))
        if recorded is None:
            raise ContractError(f"source video was not fingerprinted by S00: {path}")
        observed = _fingerprint(
            path,
            progress_interval_sec=progress_interval_sec,
            logger=logger,
        )
        for key in ("size_bytes", "mtime_ns", "sha256"):
            if observed[key] != recorded.get(key):
                raise ContractError(f"source video changed since S00: {path} ({key})")
        paths[clip_id] = path
        provenance.append(observed)
        stats.append(
            {
                "path": str(path),
                "size_bytes": observed["size_bytes"],
                "mtime_ns": observed["mtime_ns"],
            }
        )
    if sequence_ids != {EXPECTED_SEQUENCE_ID}:
        raise ContractError("S04 review sequence_id differs from S00 contract")
    return paths, provenance, stats


def _validate_frames(
    frames: dict[str, np.ndarray], video_paths: dict[str, Path]
) -> None:
    count = len(frames["global_frame"])
    global_frames = np.asarray(frames["global_frame"], dtype=np.int64)
    if count == 0 or not np.array_equal(global_frames, np.arange(count)):
        raise ContractError("S04 review frames.global_frame must be contiguous from zero")
    times = np.asarray(frames["global_time_sec"], dtype=np.float64)
    if not np.all(np.isfinite(times)) or not np.allclose(
        np.diff(times), EXPECTED_FRAME_PERIOD_SEC, rtol=0.0, atol=1e-9
    ):
        raise ContractError("S04 review requires the exact 30000/1001 S00 timeline")
    widths = np.asarray(frames["width"], dtype=np.int64)
    heights = np.asarray(frames["height"], dtype=np.int64)
    if not np.all(widths == RAW_WIDTH) or not np.all(heights == RAW_HEIGHT):
        raise ContractError("fixed S04 review raw frames must be exactly 3840x2160")
    clip_ids = np.asarray(frames["clip_id"], dtype=object)
    local = np.asarray(frames["local_frame"], dtype=np.int64)
    observed_order: list[str] = []
    for value in map(str, clip_ids):
        if not observed_order or observed_order[-1] != value:
            if value in observed_order:
                raise ContractError("S00 clip appears in disjoint frame ranges")
            observed_order.append(value)
    if observed_order != list(video_paths):
        raise ContractError("S00 frames clip order differs from resolved manifest")
    for clip_id in observed_order:
        positions = np.flatnonzero(clip_ids == clip_id)
        if not np.array_equal(local[positions], np.arange(len(positions))):
            raise ContractError(f"S00 local frames are not contiguous for {clip_id}")


def _build_detection_indices(
    frames: dict[str, np.ndarray],
    detections: dict[str, np.ndarray],
    mapping: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    det_ids = np.asarray(detections["det_id"], dtype=np.int64)
    if len(np.unique(det_ids)) != len(det_ids):
        raise ContractError("S00 detections.det_id must be unique")
    valid = np.asarray(detections["valid"])
    if valid.dtype.kind != "b":
        raise ContractError("S00 detections.valid must be boolean")
    det_frames = np.asarray(detections["global_frame"], dtype=np.int64)
    if np.any(det_frames < 0) or np.any(det_frames >= len(frames["global_frame"])):
        raise ContractError("S00 detection references an invalid global_frame")
    boxes = np.column_stack(
        [np.asarray(detections[name], dtype=np.float64) for name in ("x1", "y1", "x2", "y2")]
    )
    widths = np.asarray(frames["width"], dtype=np.float64)[det_frames]
    heights = np.asarray(frames["height"], dtype=np.float64)[det_frames]
    valid_boxes = boxes[valid]
    if not np.all(np.isfinite(valid_boxes)) or np.any(
        (valid_boxes[:, 0] < 0)
        | (valid_boxes[:, 1] < 0)
        | (valid_boxes[:, 0] >= valid_boxes[:, 2])
        | (valid_boxes[:, 1] >= valid_boxes[:, 3])
        | (valid_boxes[:, 2] > widths[valid])
        | (valid_boxes[:, 3] > heights[valid])
    ):
        raise ContractError("S00 valid detection has invalid raw bbox geometry")

    order = np.argsort(det_ids, kind="stable")
    sorted_ids = det_ids[order]
    mapping_ids = np.asarray(mapping["det_id"], dtype=np.int64)
    locations = np.searchsorted(sorted_ids, mapping_ids)
    clipped = np.minimum(locations, max(0, len(sorted_ids) - 1))
    if len(sorted_ids) == 0 or not np.all(
        (locations < len(sorted_ids)) & (sorted_ids[clipped] == mapping_ids)
    ):
        raise ContractError("S01 det_to_micro references unknown S00 det_id")
    positions = order[locations]
    if len(positions) != int(np.count_nonzero(valid)) or not np.all(valid[positions]):
        raise ContractError("S01 det_to_micro must cover every S00 valid detection once")
    if len(np.unique(positions)) != len(positions):
        raise ContractError("S01 det_to_micro contains duplicate det_id")
    micro_by_position = np.full(len(det_ids), -1, dtype=np.int64)
    micro_by_position[positions] = np.asarray(mapping["micro_id"], dtype=np.int64)

    valid_positions = np.flatnonzero(valid)
    frame_order = np.lexsort((det_ids[valid_positions], det_frames[valid_positions]))
    valid_by_frame = valid_positions[frame_order]
    counts = np.bincount(det_frames[valid_by_frame], minlength=len(frames["global_frame"]))
    offsets = np.empty(len(counts) + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(counts, out=offsets[1:])
    return micro_by_position, valid_by_frame, offsets


def _candidate_endpoints(
    candidates: dict[str, np.ndarray],
    microtracklets: dict[str, np.ndarray],
    frames: dict[str, np.ndarray],
) -> dict[str, CandidateEndpoint]:
    micro_ids = np.asarray(microtracklets["micro_id"], dtype=np.int64)
    if len(np.unique(micro_ids)) != len(micro_ids):
        raise ContractError("S01 microtracklets.micro_id must be unique")
    micro_row = {int(value): row for row, value in enumerate(micro_ids)}
    result: dict[str, CandidateEndpoint] = {}
    count = len(candidates["edge_id"])
    for row in range(count):
        if str(candidates["decision"][row]) != "provisional" or not bool(
            candidates["proposed_for_review"][row]
        ):
            continue
        edge_id = str(candidates["edge_id"][row])
        if not edge_id or edge_id in result:
            raise ContractError("provisional candidate edge_id is blank or duplicated")
        source = int(candidates["source_micro_id"][row])
        target = int(candidates["target_micro_id"][row])
        if source not in micro_row or target not in micro_row:
            raise ContractError(f"candidate {edge_id} references an unknown microtrack")
        source_row = micro_row[source]
        target_row = micro_row[target]
        summary_checks = (
            ("source_start_global_frame", "start_global_frame", source_row),
            ("source_end_global_frame", "end_global_frame", source_row),
            ("target_start_global_frame", "start_global_frame", target_row),
            ("target_end_global_frame", "end_global_frame", target_row),
        )
        for candidate_name, summary_name, summary_row in summary_checks:
            if int(candidates[candidate_name][row]) != int(
                microtracklets[summary_name][summary_row]
            ):
                raise ContractError(
                    f"candidate {edge_id} endpoint differs from immutable S01 summary"
                )
        summary_time_checks = (
            ("source_start_time_sec", "start_time_sec", source_row),
            ("source_end_time_sec", "end_time_sec", source_row),
            ("target_start_time_sec", "start_time_sec", target_row),
            ("target_end_time_sec", "end_time_sec", target_row),
        )
        for candidate_name, summary_name, summary_row in summary_time_checks:
            if not math.isclose(
                float(candidates[candidate_name][row]),
                float(microtracklets[summary_name][summary_row]),
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                raise ContractError(
                    f"candidate {edge_id} time differs from immutable S01 summary"
                )
        probability_value = candidates["probability"][row]
        if probability_value is None:
            raise ContractError(f"provisional candidate {edge_id} has null probability")
        endpoint = CandidateEndpoint(
            edge_id=edge_id,
            source_micro_id=source,
            target_micro_id=target,
            source_start_global_frame=int(candidates["source_start_global_frame"][row]),
            source_end_global_frame=int(candidates["source_end_global_frame"][row]),
            target_start_global_frame=int(candidates["target_start_global_frame"][row]),
            target_end_global_frame=int(candidates["target_end_global_frame"][row]),
            source_start_time_sec=float(candidates["source_start_time_sec"][row]),
            source_end_time_sec=float(candidates["source_end_time_sec"][row]),
            target_start_time_sec=float(candidates["target_start_time_sec"][row]),
            target_end_time_sec=float(candidates["target_end_time_sec"][row]),
            source_end_clip_id=str(candidates["source_end_clip_id"][row]),
            target_start_clip_id=str(candidates["target_start_clip_id"][row]),
            probability=float(probability_value),
            high_overlap=bool(candidates["high_overlap"][row]),
        )
        if endpoint.source_end_global_frame >= endpoint.target_start_global_frame:
            raise ContractError(f"candidate {edge_id} endpoint order is not forward")
        for label, frame, expected_time in (
            (
                "source_start",
                endpoint.source_start_global_frame,
                endpoint.source_start_time_sec,
            ),
            (
                "source_end",
                endpoint.source_end_global_frame,
                endpoint.source_end_time_sec,
            ),
            (
                "target_start",
                endpoint.target_start_global_frame,
                endpoint.target_start_time_sec,
            ),
            (
                "target_end",
                endpoint.target_end_global_frame,
                endpoint.target_end_time_sec,
            ),
        ):
            if not 0 <= frame < len(frames["global_frame"]):
                raise ContractError(f"candidate {edge_id} {label} frame is out of range")
            if not math.isclose(
                float(frames["global_time_sec"][frame]),
                expected_time,
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                raise ContractError(
                    f"candidate {edge_id} {label} time differs from S00 timeline"
                )
        if (
            str(frames["clip_id"][endpoint.source_end_global_frame])
            != endpoint.source_end_clip_id
            or str(frames["clip_id"][endpoint.target_start_global_frame])
            != endpoint.target_start_clip_id
        ):
            raise ContractError(f"candidate {edge_id} clip provenance differs from S00")
        result[edge_id] = endpoint
    return result


def _validate_proposal_candidate_join(
    plan: S04ReviewPlan, endpoints: dict[str, CandidateEndpoint]
) -> None:
    proposal_edge_ids = {proposal.edge_id for proposal in plan.proposals}
    if set(endpoints) != proposal_edge_ids:
        missing = sorted(proposal_edge_ids - set(endpoints))[:5]
        extra = sorted(set(endpoints) - proposal_edge_ids)[:5]
        raise ContractError(
            "provisional candidate/proposal edge_id sets are not identical; "
            f"missing={missing}, extra={extra}"
        )
    for proposal in plan.proposals:
        endpoint = endpoints[proposal.edge_id]
        if (
            endpoint.source_micro_id != proposal.source_micro_id
            or endpoint.target_micro_id != proposal.target_micro_id
            or endpoint.high_overlap != proposal.high_overlap
            or endpoint.probability != proposal.probability
        ):
            raise ContractError(
                f"proposal {proposal.proposal_id} differs from candidate {proposal.edge_id}"
            )


def _load_data(
    proposals_dir: Path,
    ingest_dir: Path,
    microtrack_dir: Path,
    config_path: Path,
    config: S04ReviewConfig,
    logger: LogFn,
    *,
    plan_factory: Callable[[dict[str, np.ndarray]], Any] | None = None,
    extra_input_paths: tuple[Path, ...] = (),
) -> LoadedS04ReviewData:
    s00, s00_fingerprints = _validated_stage_artifacts(
        ingest_dir,
        expected_stages=frozenset({"S00"}),
        required_names=("frames.parquet", "detections.parquet", "resolved_manifest.json"),
    )
    _, s01_fingerprints = _validated_stage_artifacts(
        microtrack_dir,
        expected_stages=frozenset({"S01"}),
        required_names=("det_to_micro.parquet", "microtracklets.parquet"),
    )
    proposal_success, proposal_fingerprints = _validated_stage_artifacts(
        proposals_dir,
        expected_stages=frozenset({"S04_PROPOSE"}),
        required_names=(
            "short_candidate_edges.parquet",
            "short_link_proposals.parquet",
            "review_manifest.json",
            "review_labels.csv",
            "s04_proposal_report.json",
            "effective_config.json",
        ),
    )
    required_proposal_policy = {
        "execution_mode": "proposal_only",
        "accepted_decision_for_review": "provisional",
        "automatic_merge_allowed": False,
        "human_labels_applied": False,
        "num_confirmed_edges": 0,
        "num_automatic_merges": 0,
    }
    for key, expected in required_proposal_policy.items():
        if proposal_success.get(key) != expected:
            raise ContractError(
                f"completed S04_PROPOSE marker violates proposal-only policy: {key}"
            )
    expected_proposal_artifacts = {
        "short_candidate_edges.parquet",
        "short_link_proposals.parquet",
        "review_manifest.json",
        "review_labels.csv",
        "s04_proposal_report.json",
        "effective_config.json",
    }
    recorded_proposal_artifacts = {
        str(item.get("path"))
        for item in proposal_success.get("output_fingerprints", [])
        if isinstance(item, dict)
    }
    actual_proposal_artifacts = {
        str(path.relative_to(proposals_dir))
        for path in proposals_dir.rglob("*")
        if path.is_file() and path.name != SUCCESS_NAME
    }
    if (
        recorded_proposal_artifacts != expected_proposal_artifacts
        or actual_proposal_artifacts != expected_proposal_artifacts
    ):
        raise ContractError("completed S04_PROPOSE artifact set differs")
    video_paths, video_provenance, _ = _load_video_paths(
        ingest_dir,
        s00,
        progress_interval_sec=config.progress_interval_sec,
        logger=logger,
    )
    input_fingerprints = (
        s00_fingerprints
        + s01_fingerprints
        + proposal_fingerprints
        + video_provenance
        + [_fingerprint(config_path.resolve())]
        + [_fingerprint(path.resolve()) for path in extra_input_paths]
    )

    frame_columns = tuple(field.name for field in FRAMES_SCHEMA)
    frames = _read_parquet_columns(
        ingest_dir / "frames.parquet",
        expected_schema=FRAMES_SCHEMA,
        columns=frame_columns,
        label="S00 frames",
    )
    detection_columns = (
        "det_id",
        "global_frame",
        "x1",
        "y1",
        "x2",
        "y2",
        "valid",
    )
    detections = _read_parquet_columns(
        ingest_dir / "detections.parquet",
        expected_schema=DETECTIONS_SCHEMA,
        columns=detection_columns,
        label="S00 detections",
    )
    mapping = _read_parquet_columns(
        microtrack_dir / "det_to_micro.parquet",
        expected_schema=DET_TO_MICRO_SCHEMA,
        columns=("det_id", "micro_id", "order_in_micro"),
        label="S01 det_to_micro",
    )
    microtracklets = _read_parquet_columns(
        microtrack_dir / "microtracklets.parquet",
        expected_schema=MICROTRACKLETS_SCHEMA,
        columns=(
            "micro_id",
            "start_global_frame",
            "end_global_frame",
            "start_time_sec",
            "end_time_sec",
        ),
        label="S01 microtracklets",
    )
    proposal_columns = tuple(field.name for field in SHORT_LINK_PROPOSALS_SCHEMA)
    proposals = _read_parquet_columns(
        proposals_dir / "short_link_proposals.parquet",
        expected_schema=SHORT_LINK_PROPOSALS_SCHEMA,
        columns=proposal_columns,
        label="S04 short-link proposals",
    )
    candidate_columns = (
        "edge_id",
        "source_micro_id",
        "target_micro_id",
        "source_end_clip_id",
        "target_start_clip_id",
        "source_start_global_frame",
        "source_end_global_frame",
        "target_start_global_frame",
        "target_end_global_frame",
        "source_start_time_sec",
        "source_end_time_sec",
        "target_start_time_sec",
        "target_end_time_sec",
        "probability",
        "high_overlap",
        "decision",
        "proposed_for_review",
    )
    candidates = _read_parquet_columns(
        proposals_dir / "short_candidate_edges.parquet",
        expected_schema=SHORT_CANDIDATE_EDGES_SCHEMA,
        columns=candidate_columns,
        label="S04 short candidate edges",
    )
    _validate_frames(frames, video_paths)
    micro_by_position, valid_by_frame, offsets = _build_detection_indices(
        frames, detections, mapping
    )
    if plan_factory is None:
        plan = build_s04_review_plan(
            proposals,
            random_seed=config.random_seed,
            top_count=config.top_count,
            bottom_count=config.bottom_count,
            random_count=config.random_count,
        )
    else:
        plan = plan_factory(proposals)
        for attribute in (
            "proposals",
            "cases",
            "intermediate_micro_ids",
            "to_manifest_payload",
        ):
            if not hasattr(plan, attribute):
                raise ContractError(
                    f"S04 review plan factory result lacks {attribute}"
                )
    endpoints = _candidate_endpoints(candidates, microtracklets, frames)
    _validate_proposal_candidate_join(plan, endpoints)
    return LoadedS04ReviewData(
        frames=frames,
        detections=detections,
        micro_id_by_detection_position=micro_by_position,
        valid_detection_positions_by_frame=valid_by_frame,
        valid_detection_frame_offsets=offsets,
        candidates_by_edge_id=endpoints,
        video_paths=video_paths,
        plan=plan,
        input_fingerprints=tuple(input_fingerprints),
    )


def _render_cases(
    data: LoadedS04ReviewData, config: S04ReviewConfig
) -> tuple[RenderCase, ...]:
    times = np.asarray(data.frames["global_time_sec"], dtype=np.float64)
    result: list[RenderCase] = []
    seen_paths: set[str] = set()
    for case in data.plan.cases:
        endpoint = data.candidates_by_edge_id[case.proposal.edge_id]
        start_time = endpoint.source_end_time_sec - config.source_tail_sec
        end_time = endpoint.target_start_time_sec + config.target_head_sec
        start = max(0, int(np.searchsorted(times, start_time, side="left")))
        end = min(
            len(times) - 1,
            int(np.searchsorted(times, end_time, side="right") - 1),
        )
        if not (
            0 <= start <= endpoint.source_end_global_frame
            < endpoint.target_start_global_frame <= end < len(times)
        ):
            raise ContractError(
                f"proposal {case.proposal.proposal_id} has an invalid review window"
            )
        output_path = f"videos/{case.suggested_filename}"
        if output_path in seen_paths:
            raise ContractError(f"duplicate S04 review output path: {output_path}")
        seen_paths.add(output_path)
        result.append(
            RenderCase(
                case=case,
                endpoint=endpoint,
                intermediate_micro_ids=data.plan.intermediate_micro_ids(case),
                start_global_frame=start,
                end_global_frame=end,
                output_relative_path=output_path,
            )
        )
    return tuple(result)


class RawFrameReader:
    """Exact sequential/seek reader for raw, no-autorotate S00 clip frames."""

    def __init__(
        self,
        video_paths: dict[str, Path],
        *,
        raw_width: int,
        raw_height: int,
        capture_factory: CaptureFactory = open_raw_video_capture,
    ) -> None:
        self._video_paths = video_paths
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
            raise ContractError("OpenCV lacks CAP_PROP_PTS for exact S04 review decoding")
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


def _preflight_nvenc(config: S04ReviewConfig) -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
        raise ContractError(
            "S04 review requires CUDA_VISIBLE_DEVICES exactly equal to '1'"
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
        raise ContractError(f"cannot start S04 NVENC preflight: {exc}") from exc
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise ContractError(
            f"GPU 1 h264_nvenc preflight failed ({completed.returncode}): "
            f"{detail or 'no FFmpeg diagnostics'}"
        )


def _frame_role_boxes(
    data: LoadedS04ReviewData,
    render_case: RenderCase,
    global_frame: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    start = int(data.valid_detection_frame_offsets[global_frame])
    stop = int(data.valid_detection_frame_offsets[global_frame + 1])
    positions = data.valid_detection_positions_by_frame[start:stop]
    micro_ids = data.micro_id_by_detection_position[positions]
    source_mask = micro_ids == render_case.case.proposal.source_micro_id
    target_mask = micro_ids == render_case.case.proposal.target_micro_id
    inside_gap = (
        render_case.endpoint.source_end_global_frame
        < global_frame
        < render_case.endpoint.target_start_global_frame
    )
    if inside_gap and render_case.intermediate_micro_ids:
        intermediate_mask = np.isin(
            micro_ids, np.asarray(render_case.intermediate_micro_ids, dtype=np.int64)
        )
    else:
        intermediate_mask = np.zeros(len(positions), dtype=np.bool_)
    intermediate_mask &= ~(source_mask | target_mask)
    unrelated_mask = ~(source_mask | target_mask | intermediate_mask)
    all_boxes = np.column_stack(
        [
            np.asarray(data.detections[name], dtype=np.float64)[positions]
            for name in ("x1", "y1", "x2", "y2")
        ]
    )
    return (
        all_boxes[unrelated_mask],
        all_boxes[source_mask],
        all_boxes[target_mask],
        all_boxes[intermediate_mask],
    )


def _render_one_case(
    data: LoadedS04ReviewData,
    render_case: RenderCase,
    config: S04ReviewConfig,
    output_dir: Path,
    *,
    logger: LogFn,
    log_prefix: str = "s04-review",
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
        for completed, global_frame in enumerate(
            range(render_case.start_global_frame, render_case.end_global_frame + 1),
            start=1,
        ):
            raw = reader.read(
                str(frames["clip_id"][global_frame]),
                int(frames["local_frame"][global_frame]),
                float(frames["pts_sec"][global_frame]),
            )
            unrelated, source, target, intermediate = _frame_role_boxes(
                data, render_case, global_frame
            )
            try:
                annotated = render_s04_review_frame(
                    raw,
                    output_width=config.output_width,
                    output_height=config.output_height,
                    unrelated_boxes=unrelated,
                    source_boxes=source,
                    target_boxes=target,
                    intermediate_boxes=intermediate,
                )
            except (TypeError, ValueError, cv2.error) as exc:
                raise ContractError(
                    f"cannot annotate {render_case.case.proposal.proposal_id} "
                    f"global frame {global_frame}: {exc}"
                ) from exc
            writer.write(np.ascontiguousarray(annotated))
            now = time.monotonic()
            if now - last_report >= config.progress_interval_sec:
                logger(
                    f"[{log_prefix}] {render_case.case.proposal.proposal_id}: "
                    f"{completed:,}/{render_case.expected_frame_count:,} frames"
                )
                last_report = now
    metadata = validate_qa_mp4(
        output_path,
        expected_frame_rate=EXPECTED_FRAME_RATE,
        expected_frame_count=render_case.expected_frame_count,
        ffprobe_binary=config.ffprobe_binary,
    )
    fingerprint = _fingerprint(output_path)
    return {
        "status": "completed",
        "completed_at_unix_sec": time.time(),
        "output_fingerprint": {
            "path": render_case.output_relative_path,
            "size_bytes": fingerprint["size_bytes"],
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


def _case_record(render_case: RenderCase) -> dict[str, Any]:
    record = render_case.case.to_manifest_record()
    endpoint = render_case.endpoint
    record.update(
        {
            "source_end_clip_id": endpoint.source_end_clip_id,
            "target_start_clip_id": endpoint.target_start_clip_id,
            "source_start_global_frame": endpoint.source_start_global_frame,
            "source_end_global_frame": endpoint.source_end_global_frame,
            "target_start_global_frame": endpoint.target_start_global_frame,
            "target_end_global_frame": endpoint.target_end_global_frame,
            "source_start_time_sec": endpoint.source_start_time_sec,
            "source_end_time_sec": endpoint.source_end_time_sec,
            "target_start_time_sec": endpoint.target_start_time_sec,
            "target_end_time_sec": endpoint.target_end_time_sec,
            "review_start_global_frame": render_case.start_global_frame,
            "review_end_global_frame": render_case.end_global_frame,
            "expected_frame_count": render_case.expected_frame_count,
            "output_path": render_case.output_relative_path,
            "intermediate_micro_ids": list(render_case.intermediate_micro_ids),
            "purple_box_policy": PURPLE_BOX_POLICY,
            "render": {"status": "pending"},
        }
    )
    return record


def _canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _verify_inputs_unchanged(
    data: LoadedS04ReviewData,
    *,
    progress_interval_sec: float,
    logger: LogFn,
) -> None:
    for recorded in data.input_fingerprints:
        path = Path(str(recorded["path"]))
        current = _fingerprint(
            path,
            progress_interval_sec=progress_interval_sec,
            logger=logger if int(recorded["size_bytes"]) >= 1024**3 else None,
        )
        for key in ("size_bytes", "mtime_ns", "sha256"):
            if current[key] != recorded[key]:
                raise ContractError(
                    f"immutable input changed during S04 review: {path} ({key})"
                )


def _validate_completed_video(
    record: dict[str, Any], output_dir: Path, config: S04ReviewConfig
) -> None:
    render = record.get("render")
    if not isinstance(render, dict) or render.get("status") != "completed":
        raise ContractError(f"S04 review case is not complete: {record.get('proposal_id')}")
    output_path = output_dir / str(record.get("output_path", ""))
    current = _fingerprint(output_path)
    expected = render.get("output_fingerprint")
    if not isinstance(expected, dict) or expected.get("path") != record.get(
        "output_path"
    ):
        raise ContractError(f"completed review fingerprint is malformed: {output_path}")
    if current["size_bytes"] != expected.get("size_bytes") or current[
        "sha256"
    ] != expected.get("sha256"):
        raise ContractError(f"completed review video changed: {output_path}")
    metadata = validate_qa_mp4(
        output_path,
        expected_frame_rate=EXPECTED_FRAME_RATE,
        expected_frame_count=int(record["expected_frame_count"]),
        ffprobe_binary=config.ffprobe_binary,
    )
    observed_validation = {
        "codec_name": metadata.codec_name,
        "width": metadata.width,
        "height": metadata.height,
        "average_frame_rate": str(metadata.average_frame_rate),
        "num_frames": metadata.frame_count,
    }
    if render.get("video_validation") != observed_validation:
        raise ContractError(f"completed review video metadata changed: {output_path}")


def _effective_config_output_record(
    output_dir: Path, config_payload: dict[str, Any]
) -> dict[str, Any]:
    path = output_dir / "effective_config.json"
    if _read_json(path, "S04 review effective config") != config_payload:
        raise ContractError("S04 review effective_config.json changed")
    fingerprint = _fingerprint(path)
    return {
        "path": "effective_config.json",
        "size_bytes": fingerprint["size_bytes"],
        "sha256": fingerprint["sha256"],
    }


def run_s04_video_review(
    *,
    proposals_dir: Path,
    ingest_dir: Path,
    microtrack_dir: Path,
    config_path: Path,
    output_dir: Path,
    config: S04ReviewConfig,
    config_payload: dict[str, Any],
    config_hash: str,
    plan_factory: Callable[[dict[str, np.ndarray]], Any] | None = None,
    extra_input_paths: tuple[Path, ...] = (),
    manifest_schema_version: str = MANIFEST_SCHEMA_VERSION,
    success_schema_version: str = SUCCESS_SCHEMA_VERSION,
    stage: str = "S04_REVIEW",
    review_purpose: str = "aggregate_quality_only",
    log_prefix: str = "s04-review",
    success_policy: dict[str, Any] | None = None,
    logger: LogFn = log,
) -> dict[str, Any]:
    """Render a deterministic read-only selection from completed S04 proposals."""

    proposals_dir = proposals_dir.resolve()
    ingest_dir = ingest_dir.resolve()
    microtrack_dir = microtrack_dir.resolve()
    config_path = config_path.resolve()
    output_dir = output_dir.resolve()
    resolved_success_policy = (
        dict(success_policy)
        if success_policy is not None
        else {"aggregate_quality_feedback_only": True}
    )
    for label, input_dir in (
        ("S04 proposals", proposals_dir),
        ("S00 ingest", ingest_dir),
        ("S01 microtrack", microtrack_dir),
    ):
        if (
            output_dir == input_dir
            or output_dir.is_relative_to(input_dir)
            or input_dir.is_relative_to(output_dir)
        ):
            raise ContractError(
                f"S04 review output cannot overlap immutable {label}: {input_dir}"
            )
    _preflight_nvenc(config)
    logger(
        f"[{log_prefix}] GPU 1 h264_nvenc preflight passed; validating inputs"
    )
    data = _load_data(
        proposals_dir,
        ingest_dir,
        microtrack_dir,
        config_path,
        config,
        logger,
        plan_factory=plan_factory,
        extra_input_paths=extra_input_paths,
    )
    render_cases = _render_cases(data, config)
    plan_payload = data.plan.to_manifest_payload()
    case_records = [_case_record(case) for case in render_cases]
    immutable_cases = [
        {key: value for key, value in record.items() if key != "render"}
        for record in case_records
    ]
    identity = {
        "config_hash": config_hash,
        "input_fingerprints": list(data.input_fingerprints),
        "plan_hash": _canonical_hash(
            {"selection_plan": plan_payload, "cases": immutable_cases}
        ),
    }
    video_contract = {
        "coordinate_system": "raw_encoded_landscape_no_autorotate",
        "codec": "h264_nvenc",
        "physical_gpu": 1,
        "logical_gpu": 0,
        "cpu_or_opencv_encoder_fallback": False,
        "width": config.output_width,
        "height": config.output_height,
        "average_frame_rate": str(EXPECTED_FRAME_RATE),
        "frame_contents": "original_image_and_colored_bboxes_only",
        "text": False,
        "ids": False,
        "score_overlay": False,
        "legend": False,
        "panels": False,
        "inset": False,
        "trajectory": False,
        "keypoints": False,
        "colors_bgr": {
            "unrelated_gray": list(UNRELATED_BGR),
            "source_old_yellow": list(SOURCE_BGR),
            "target_new_green": list(TARGET_BGR),
            "intermediate_unstable_purple": list(INTERMEDIATE_BGR),
        },
        "purple_box_policy": PURPLE_BOX_POLICY,
    }
    manifest_path = output_dir / MANIFEST_NAME
    if manifest_path.is_file():
        manifest = _read_json(manifest_path, "S04 review manifest")
        if not isinstance(manifest, dict):
            raise ContractError("existing S04 review manifest is not an object")
        if (
            manifest.get("schema_version") != manifest_schema_version
            or manifest.get("identity") != identity
            or manifest.get("selection_plan") != plan_payload
            or manifest.get("video_contract") != video_contract
            or [
                {key: value for key, value in record.items() if key != "render"}
                for record in manifest.get("cases", [])
                if isinstance(record, dict)
            ]
            != immutable_cases
        ):
            raise ContractError("existing S04 review output belongs to another plan")
    else:
        if output_dir.exists() and any(output_dir.iterdir()):
            raise ContractError(
                f"S04 review output is non-empty without {MANIFEST_NAME}: {output_dir}"
            )
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "videos").mkdir()
        _atomic_write_json(output_dir / "effective_config.json", config_payload)
        manifest = {
            "schema_version": manifest_schema_version,
            "stage": stage,
            "review_purpose": review_purpose,
            "per_item_human_labels_expected": False,
            "identity": identity,
            "effective_config": config_payload,
            "selection_plan": plan_payload,
            "video_contract": video_contract,
            "cases": case_records,
        }
        _atomic_write_json(manifest_path, manifest)

    effective_output_record = _effective_config_output_record(
        output_dir, config_payload
    )
    success_path = output_dir / SUCCESS_NAME
    if success_path.is_file():
        success = _read_json(success_path, "S04 review success marker")
        if (
            not isinstance(success, dict)
            or success.get("schema_version") != success_schema_version
            or success.get("stage") != stage
            or success.get("identity") != identity
            or success.get("manifest_sha256") != _sha256(manifest_path)
            or any(
                success.get(key) != value
                for key, value in resolved_success_policy.items()
            )
        ):
            raise ContractError("existing S04 review success marker identity mismatch")
        for record in manifest["cases"]:
            _validate_completed_video(record, output_dir, config)
        expected_outputs = [effective_output_record] + [
            record["render"]["output_fingerprint"]
            for record in manifest["cases"]
        ]
        if success.get("output_fingerprints") != expected_outputs:
            raise ContractError("S04 review success output fingerprints changed")
        _verify_inputs_unchanged(
            data,
            progress_interval_sec=config.progress_interval_sec,
            logger=logger,
        )
        logger(f"[{log_prefix}] already complete: {success_path}")
        return manifest

    by_proposal = {
        item.case.proposal.proposal_id: item for item in render_cases
    }
    for index, record in enumerate(manifest["cases"], start=1):
        if record.get("render", {}).get("status") == "completed":
            _validate_completed_video(record, output_dir, config)
            continue
        proposal_id = str(record["proposal_id"])
        logger(
            f"[{log_prefix}] render {index}/{len(render_cases)}: "
            f"{proposal_id} -> {record['output_path']}"
        )
        record["render"] = _render_one_case(
            data,
            by_proposal[proposal_id],
            config,
            output_dir,
            logger=logger,
            log_prefix=log_prefix,
        )
        _atomic_write_json(manifest_path, manifest)

    _verify_inputs_unchanged(
        data,
        progress_interval_sec=config.progress_interval_sec,
        logger=logger,
    )
    output_fingerprints = [effective_output_record] + [
        record["render"]["output_fingerprint"] for record in manifest["cases"]
    ]
    success = {
        "schema_version": success_schema_version,
        "stage": stage,
        "identity": identity,
        "manifest_sha256": _sha256(manifest_path),
        "stats": {
            "num_available_proposals": len(data.plan.proposals),
            "num_rendered_cases": len(render_cases),
            "num_rendered_frames": sum(
                case.expected_frame_count for case in render_cases
            ),
        },
        "output_fingerprints": output_fingerprints,
        "per_item_human_labels_expected": False,
    }
    success.update(resolved_success_policy)
    _atomic_write_json(success_path, success)
    logger(f"[{log_prefix}] complete: {success_path}")
    return manifest


def run_s04_review(
    *,
    proposals_dir: Path,
    ingest_dir: Path,
    microtrack_dir: Path,
    config_path: Path,
    output_dir: Path,
    logger: LogFn = log,
) -> dict[str, Any]:
    """Render the fixed aggregate-quality sample from completed S04 proposals."""

    config, config_payload, config_hash = load_s04_review_config(config_path)
    return run_s04_video_review(
        proposals_dir=proposals_dir,
        ingest_dir=ingest_dir,
        microtrack_dir=microtrack_dir,
        config_path=config_path,
        output_dir=output_dir,
        config=config,
        config_payload=config_payload,
        config_hash=config_hash,
        logger=logger,
    )


__all__ = [
    "EXPECTED_FRAME_RATE",
    "LoadedS04ReviewData",
    "RawFrameReader",
    "RenderCase",
    "run_s04_review",
    "run_s04_video_review",
]
