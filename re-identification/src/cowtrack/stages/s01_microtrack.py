from __future__ import annotations

import hashlib
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from cowtrack.config import (
    ContractError,
    MicrotrackConfig,
    load_microtrack_config,
)
from cowtrack.schemas.detections import DETECTIONS_SCHEMA
from cowtrack.schemas.edges import DET_EDGES_SCHEMA
from cowtrack.schemas.frames import FRAMES_SCHEMA
from cowtrack.schemas.tracklets import (
    DET_TO_MICRO_SCHEMA,
    MICROTRACKLETS_SCHEMA,
    MICROTRACKLET_STATUS_SHORT_FRAGMENT,
    MICROTRACKLET_STATUS_VALID,
)
from cowtrack.tracking.microtrack import (
    AssignmentResult,
    DetectionBatch,
    LinkResult,
    MicrotrackSettings,
    MotionPrior,
    assign_microtracks,
    build_motion_prior,
    infer_auto_max_time_gap,
    link_microtracks,
)


LogFn = Callable[[str], None]

S00_REQUIRED_ARTIFACT_NAMES = {
    "frames.parquet",
    "detections.parquet",
    "resolved_manifest.json",
    "ingest_report.json",
}
EXPECTED_S01_ARTIFACT_NAMES = {
    "det_edges.parquet",
    "det_to_micro.parquet",
    "microtracklets.parquet",
    "motion_prior.npz",
    "effective_config.json",
    "microtrack_report.json",
}

# This projection is an explicit dependency boundary. clip_id/local_frame are
# used only to validate the S00 timeline and are not forwarded to the tracker.
# It excludes keypoints, legacy_track_id, identity, and every CSV-only field.
S01_DETECTION_COLUMNS = (
    "det_id",
    "sequence_id",
    "clip_id",
    "local_frame",
    "global_frame",
    "global_time_sec",
    "x1",
    "y1",
    "x2",
    "y2",
    "cx_norm",
    "cy_norm",
    "w_norm",
    "h_norm",
    "valid",
)


def log(message: str) -> None:
    print(message, flush=True)


@dataclass(frozen=True)
class ClipBoundary:
    clip_id: str
    clip_order: int
    num_frames: int
    start_global_frame: int
    end_global_frame: int
    width: int
    height: int


@dataclass(frozen=True)
class S00Contract:
    success: dict[str, Any]
    sequence_id: str
    clip_ids: tuple[str, ...]
    boundaries: tuple[ClipBoundary, ...]
    num_frames: int
    num_input_detections: int
    num_valid_detections: int


@dataclass(frozen=True)
class LoadedInput:
    sequence_id: str
    clip_ids: tuple[str, ...]
    frame_times: np.ndarray
    frame_period_sec: float
    batch: DetectionBatch


def _atomic_write_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _file_fingerprint(
    path: Path,
    *,
    progress_interval_sec: float,
    logger: LogFn,
) -> dict[str, Any]:
    before = path.stat()
    digest = hashlib.sha256()
    processed = 0
    last_report = time.monotonic()
    with path.open("rb") as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
            processed += len(block)
            now = time.monotonic()
            if now - last_report >= progress_interval_sec:
                percent = 100.0 * processed / before.st_size if before.st_size else 100.0
                logger(
                    f"[s01] fingerprint {path.name}: {processed / (1024**2):.1f}/"
                    f"{before.st_size / (1024**2):.1f} MiB ({percent:.1f}%)"
                )
                last_report = now
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ContractError(f"input changed while fingerprinting: {path}")
    return {
        "path": str(path.resolve()),
        "size_bytes": int(before.st_size),
        "mtime_ns": int(before.st_mtime_ns),
        "sha256": digest.hexdigest(),
    }


def _output_fingerprint(path: Path, output_dir: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return {
        "path": str(path.relative_to(output_dir)),
        "size_bytes": int(path.stat().st_size),
        "sha256": digest.hexdigest(),
    }


def _read_json(path: Path, context: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read {context}: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ContractError(f"{context} must be a JSON object: {path}")
    return payload


def _read_json_list(path: Path, context: str) -> list[Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read {context}: {path}: {exc}") from exc
    if not isinstance(payload, list):
        raise ContractError(f"{context} must be a JSON list: {path}")
    return payload


def _contract_int(
    mapping: dict[str, Any],
    key: str,
    context: str,
    *,
    minimum: int = 0,
) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ContractError(
            f"{context} {key} must be an integer >= {minimum}: {value!r}"
        )
    return value


def _validate_s00_contract(input_dir: Path) -> S00Contract:
    success_path = input_dir / "_SUCCESS.json"
    if not success_path.is_file():
        raise ContractError(f"S00 _SUCCESS.json does not exist: {success_path}")
    success = _read_json(success_path, "S00 success marker")
    if success.get("stage") != "S00":
        raise ContractError("S01 input _SUCCESS.json is not from S00")
    missing_artifacts = sorted(
        name
        for name in S00_REQUIRED_ARTIFACT_NAMES
        if not (input_dir / name).is_file()
    )
    if missing_artifacts:
        raise ContractError(f"S01 is missing required S00 artifacts: {missing_artifacts}")

    stats = success.get("stats")
    if not isinstance(stats, dict):
        raise ContractError("S00 _SUCCESS.json has no stats mapping")
    num_clips = _contract_int(stats, "num_clips", "S00 stats", minimum=1)
    num_frames = _contract_int(stats, "num_frames", "S00 stats", minimum=2)
    num_input = _contract_int(stats, "num_input_boxes", "S00 stats", minimum=1)
    num_valid = _contract_int(stats, "num_valid_boxes", "S00 stats", minimum=1)
    num_invalid = _contract_int(stats, "num_invalid_boxes", "S00 stats")
    num_duplicates = _contract_int(
        stats, "num_duplicate_boxes_removed", "S00 stats"
    )
    num_clamped = _contract_int(stats, "num_clamped_boxes", "S00 stats")
    if num_input != num_valid + num_invalid:
        raise ContractError("S00 stats input boxes do not partition into valid/invalid")
    if num_duplicates > num_invalid or num_clamped > num_input:
        raise ContractError("S00 duplicate/clamped stats exceed their possible totals")

    resolved_path = input_dir / "resolved_manifest.json"
    report_path = input_dir / "ingest_report.json"
    if not resolved_path.is_file() or not report_path.is_file():
        raise ContractError(
            "S01 requires S00 resolved_manifest.json and ingest_report.json"
        )
    resolved = _read_json_list(resolved_path, "S00 resolved manifest")
    if len(resolved) != num_clips:
        raise ContractError("S00 resolved manifest clip count differs from S00 stats")
    sequence_id: str | None = None
    clip_ids: list[str] = []
    for expected_order, row in enumerate(resolved):
        if not isinstance(row, dict):
            raise ContractError("S00 resolved manifest entries must be JSON objects")
        row_sequence = row.get("sequence_id")
        clip_id = row.get("clip_id")
        if not isinstance(row_sequence, str) or not row_sequence:
            raise ContractError("S00 resolved manifest has an invalid sequence_id")
        if not isinstance(clip_id, str) or not clip_id:
            raise ContractError("S00 resolved manifest has an invalid clip_id")
        if sequence_id is None:
            sequence_id = row_sequence
        elif row_sequence != sequence_id:
            raise ContractError("S00 resolved manifest contains multiple sequences")
        if clip_id in clip_ids:
            raise ContractError("S00 resolved manifest contains duplicate clip_id values")
        if row.get("clip_order") != expected_order:
            raise ContractError("S00 resolved manifest clip_order is not contiguous")
        if row.get("frame_index_base") != 0 or row.get("bbox_format") != "xywh":
            raise ContractError("S00 resolved manifest coordinate convention changed")
        clip_ids.append(clip_id)
    assert sequence_id is not None

    report = _read_json(report_path, "S00 ingest report")
    if report.get("sequence_id") != sequence_id:
        raise ContractError("S00 ingest report sequence_id differs from resolved manifest")
    for key in (
        "num_clips",
        "num_frames",
        "num_input_boxes",
        "num_valid_boxes",
        "num_invalid_boxes",
        "num_duplicate_boxes_removed",
        "num_clamped_boxes",
    ):
        if report.get(key) != stats[key]:
            raise ContractError(
                f"S00 ingest report and success marker disagree for {key}"
            )
    if report.get("source_rows_retained") is not True:
        raise ContractError("S00 ingest report does not guarantee source row retention")
    if report.get("keypoints_used") is not False:
        raise ContractError("S00 ingest report unexpectedly used keypoints")
    if report.get("legacy_track_id_used") is not False:
        raise ContractError("S00 ingest report unexpectedly used legacy tracking IDs")
    if report.get("coordinate_system") != "raw_encoded_landscape_no_autorotate":
        raise ContractError("S00 ingest report coordinate system changed")

    boundary_rows = report.get("clip_boundaries")
    if not isinstance(boundary_rows, list) or len(boundary_rows) != num_clips:
        raise ContractError("S00 ingest report clip boundaries do not match clip count")
    boundaries: list[ClipBoundary] = []
    next_global_frame = 0
    previous_end_time: float | None = None
    for clip_order, (clip_id, row) in enumerate(zip(clip_ids, boundary_rows)):
        if not isinstance(row, dict):
            raise ContractError("S00 clip boundary entries must be JSON objects")
        if row.get("clip_id") != clip_id or row.get("clip_order") != clip_order:
            raise ContractError("S00 clip boundary order differs from resolved manifest")
        count = _contract_int(row, "num_frames", "S00 clip boundary", minimum=1)
        start = _contract_int(row, "start_global_frame", "S00 clip boundary")
        end = _contract_int(row, "end_global_frame_inclusive", "S00 clip boundary")
        width = _contract_int(row, "width", "S00 clip boundary", minimum=1)
        height = _contract_int(row, "height", "S00 clip boundary", minimum=1)
        if start != next_global_frame or end != start + count - 1:
            raise ContractError("S00 clip boundary global frames are not contiguous")
        try:
            start_time = float(row["start_global_time_sec"])
            end_time = float(row["end_global_time_sec_inclusive"])
            end_time_exclusive = float(row["end_global_time_sec_exclusive"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ContractError("S00 clip boundary has invalid global times") from exc
        if not all(math.isfinite(value) for value in (start_time, end_time, end_time_exclusive)):
            raise ContractError("S00 clip boundary global times are non-finite")
        if end_time < start_time or end_time_exclusive <= end_time:
            raise ContractError("S00 clip boundary global times are not increasing")
        if previous_end_time is not None and not math.isclose(
            start_time, previous_end_time, rel_tol=0.0, abs_tol=1e-9
        ):
            raise ContractError("S00 clip boundary global times are not contiguous")
        boundaries.append(
            ClipBoundary(
                clip_id=clip_id,
                clip_order=clip_order,
                num_frames=count,
                start_global_frame=start,
                end_global_frame=end,
                width=width,
                height=height,
            )
        )
        next_global_frame = end + 1
        previous_end_time = end_time_exclusive
    if next_global_frame != num_frames:
        raise ContractError("S00 clip boundary frame total differs from S00 stats")
    try:
        sequence_end_time = float(report["sequence_end_time_sec_exclusive"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ContractError("S00 ingest report has invalid sequence end time") from exc
    if previous_end_time is None or not math.isclose(
        sequence_end_time, previous_end_time, rel_tol=0.0, abs_tol=1e-9
    ):
        raise ContractError("S00 sequence end time differs from final clip boundary")

    per_clip = report.get("per_clip")
    if not isinstance(per_clip, list) or len(per_clip) != num_clips:
        raise ContractError("S00 per-clip report does not match clip count")
    per_clip_totals = {
        "num_input_boxes": 0,
        "num_valid_boxes": 0,
        "num_invalid_boxes": 0,
        "num_duplicate_boxes_removed": 0,
        "num_clamped_boxes": 0,
    }
    for clip_id, boundary, row in zip(clip_ids, boundaries, per_clip):
        if not isinstance(row, dict) or row.get("clip_id") != clip_id:
            raise ContractError("S00 per-clip report order differs from resolved manifest")
        values = {
            key: _contract_int(row, key, f"S00 per-clip report {clip_id}")
            for key in per_clip_totals
        }
        frames_with = _contract_int(
            row, "num_frames_with_boxes", f"S00 per-clip report {clip_id}"
        )
        frames_without = _contract_int(
            row, "num_frames_without_boxes", f"S00 per-clip report {clip_id}"
        )
        if values["num_input_boxes"] != (
            values["num_valid_boxes"] + values["num_invalid_boxes"]
        ):
            raise ContractError(f"S00 per-clip boxes do not partition for {clip_id}")
        if values["num_duplicate_boxes_removed"] > values["num_invalid_boxes"]:
            raise ContractError(f"S00 per-clip duplicate count is impossible for {clip_id}")
        if values["num_clamped_boxes"] > values["num_input_boxes"]:
            raise ContractError(f"S00 per-clip clamped count is impossible for {clip_id}")
        if frames_with + frames_without != boundary.num_frames:
            raise ContractError(f"S00 per-clip frame counts differ for {clip_id}")
        for key, value in values.items():
            per_clip_totals[key] += value
    for key, total in per_clip_totals.items():
        if total != stats[key]:
            raise ContractError(f"S00 per-clip totals disagree for {key}")

    recorded = success.get("output_fingerprints")
    if not isinstance(recorded, list):
        raise ContractError("S00 _SUCCESS.json has no output_fingerprints list")
    recorded_by_name: dict[str, dict[str, Any]] = {}
    for item in recorded:
        if not isinstance(item, dict):
            raise ContractError("S00 output fingerprint entries must be JSON objects")
        name = str(item.get("path"))
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ContractError(f"unsafe S00 artifact path: {name}")
        if name in recorded_by_name:
            raise ContractError(f"duplicate S00 artifact fingerprint: {name}")
        recorded_by_name[name] = item
    if not S00_REQUIRED_ARTIFACT_NAMES.issubset(recorded_by_name):
        raise ContractError("S00 success marker does not fingerprint all S01 inputs")
    for name in S00_REQUIRED_ARTIFACT_NAMES:
        item = recorded_by_name[name]
        size = item.get("size_bytes")
        digest = item.get("sha256")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ContractError(f"S00 success marker has invalid {name} size_bytes")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ContractError(f"S00 success marker has invalid {name} sha256")
        try:
            int(digest, 16)
        except ValueError as exc:
            raise ContractError(
                f"S00 success marker has invalid {name} sha256"
            ) from exc
    return S00Contract(
        success=success,
        sequence_id=sequence_id,
        clip_ids=tuple(clip_ids),
        boundaries=tuple(boundaries),
        num_frames=num_frames,
        num_input_detections=num_input,
        num_valid_detections=num_valid,
    )


def _verify_s00_fingerprints(
    contract: S00Contract, input_fingerprints: list[dict[str, Any]]
) -> None:
    recorded = {
        str(item["path"]): item
        for item in contract.success["output_fingerprints"]
        if str(item.get("path")) in S00_REQUIRED_ARTIFACT_NAMES
    }
    current = {
        Path(item["path"]).name: item
        for item in input_fingerprints
        if Path(item["path"]).name in S00_REQUIRED_ARTIFACT_NAMES
    }
    if set(current) != set(recorded):
        raise ContractError("S01 input fingerprint set differs from S00 outputs")
    for name, item in current.items():
        for key in ("size_bytes", "sha256"):
            if item[key] != recorded[name].get(key):
                raise ContractError(f"S01 input {name} differs from completed S00 ({key})")


def _verify_inputs_unchanged(fingerprints: list[dict[str, Any]]) -> None:
    for fingerprint in fingerprints:
        current = Path(fingerprint["path"]).stat()
        if (
            current.st_size != fingerprint["size_bytes"]
            or current.st_mtime_ns != fingerprint["mtime_ns"]
        ):
            raise ContractError(f"input changed during S01: {fingerprint['path']}")


def _column_numpy(table: pa.Table, name: str, dtype: np.dtype[Any]) -> np.ndarray:
    return np.asarray(table[name].to_numpy(zero_copy_only=False), dtype=dtype)


def _load_inputs(input_dir: Path, contract: S00Contract) -> LoadedInput:
    frames_path = input_dir / "frames.parquet"
    detections_path = input_dir / "detections.parquet"
    if not frames_path.is_file() or not detections_path.is_file():
        raise ContractError("S01 requires frames.parquet and detections.parquet")
    frames_file = pq.ParquetFile(frames_path)
    detections_file = pq.ParquetFile(detections_path)
    if not frames_file.schema_arrow.equals(FRAMES_SCHEMA, check_metadata=False):
        raise ContractError("S01 frames.parquet schema mismatch")
    if not detections_file.schema_arrow.equals(DETECTIONS_SCHEMA, check_metadata=False):
        raise ContractError("S01 detections.parquet schema mismatch")
    if frames_file.metadata.num_rows != contract.num_frames:
        raise ContractError("S01 frames.parquet row count mismatch")
    if detections_file.metadata.num_rows != contract.num_input_detections:
        raise ContractError("S01 detections.parquet row count mismatch")

    frames = pq.read_table(
        frames_path,
        columns=[
            "sequence_id",
            "clip_id",
            "clip_order",
            "local_frame",
            "global_frame",
            "global_time_sec",
            "width",
            "height",
        ],
    )
    sequence_values = pc.unique(frames["sequence_id"]).to_pylist()
    if sequence_values != [contract.sequence_id]:
        raise ContractError(f"unexpected S01 sequence_id values: {sequence_values}")
    global_frames = _column_numpy(frames, "global_frame", np.dtype(np.int64))
    frame_times = _column_numpy(frames, "global_time_sec", np.dtype(np.float64))
    if not np.array_equal(
        global_frames, np.arange(contract.num_frames, dtype=np.int64)
    ):
        raise ContractError("S01 frame timeline is not contiguous 0..N-1")
    differences = np.diff(frame_times)
    if not np.all(np.isfinite(frame_times)) or not np.all(differences > 0.0):
        raise ContractError("S01 frame times are not finite and strictly increasing")
    clip_orders = _column_numpy(frames, "clip_order", np.dtype(np.int16))
    local_frames = _column_numpy(frames, "local_frame", np.dtype(np.int32))
    widths = _column_numpy(frames, "width", np.dtype(np.int32))
    heights = _column_numpy(frames, "height", np.dtype(np.int32))
    for boundary in contract.boundaries:
        start = boundary.start_global_frame
        stop = boundary.end_global_frame + 1
        clip_slice = frames["clip_id"].slice(start, boundary.num_frames)
        if not bool(pc.all(pc.equal(clip_slice, boundary.clip_id)).as_py()):
            raise ContractError(
                f"S01 frames clip_id mapping differs for {boundary.clip_id}"
            )
        if not np.all(clip_orders[start:stop] == boundary.clip_order):
            raise ContractError(
                f"S01 frames clip_order mapping differs for {boundary.clip_id}"
            )
        if not np.array_equal(
            local_frames[start:stop],
            np.arange(boundary.num_frames, dtype=np.int32),
        ):
            raise ContractError(
                f"S01 local frame timeline differs for {boundary.clip_id}"
            )
        if not np.all(widths[start:stop] == boundary.width) or not np.all(
            heights[start:stop] == boundary.height
        ):
            raise ContractError(
                f"S01 frame dimensions differ from S00 report for {boundary.clip_id}"
            )
    frame_period = float(np.median(differences))

    detections = pq.read_table(
        detections_path,
        columns=list(S01_DETECTION_COLUMNS),
        filters=[("valid", "=", True)],
    )
    if detections.num_rows != contract.num_valid_detections:
        raise ContractError("S01 valid detection count mismatch")
    if not bool(pc.all(detections["valid"]).as_py()):
        raise ContractError("S01 valid=true filter contract failed")
    detection_sequences = pc.unique(detections["sequence_id"]).to_pylist()
    if detection_sequences != [contract.sequence_id]:
        raise ContractError(f"unexpected detection sequence_id values: {detection_sequences}")

    det_id = _column_numpy(detections, "det_id", np.dtype(np.int64))
    det_local_frame = _column_numpy(
        detections, "local_frame", np.dtype(np.int32)
    )
    det_frame = _column_numpy(detections, "global_frame", np.dtype(np.int64))
    det_time = _column_numpy(detections, "global_time_sec", np.dtype(np.float64))
    if np.unique(det_id).size != contract.num_valid_detections:
        raise ContractError("S01 valid detections contain duplicate det_id values")
    if np.any(det_frame < 0) or np.any(det_frame >= contract.num_frames):
        raise ContractError("S01 detection global_frame is outside frames.parquet")
    if not np.array_equal(det_time, frame_times[det_frame]):
        raise ContractError("S01 detection time does not exactly join frames.parquet")
    validated_detections = 0
    for boundary in contract.boundaries:
        in_clip = (det_frame >= boundary.start_global_frame) & (
            det_frame <= boundary.end_global_frame
        )
        positions = np.flatnonzero(in_clip)
        validated_detections += len(positions)
        if len(positions) == 0:
            continue
        detection_clip_ids = detections["clip_id"].take(
            pa.array(positions, type=pa.int64())
        )
        if not bool(
            pc.all(pc.equal(detection_clip_ids, boundary.clip_id)).as_py()
        ):
            raise ContractError(
                f"S01 detection clip_id mapping differs for {boundary.clip_id}"
            )
        if not np.array_equal(
            det_local_frame[positions],
            det_frame[positions] - boundary.start_global_frame,
        ):
            raise ContractError(
                f"S01 detection local_frame mapping differs for {boundary.clip_id}"
            )
    if validated_detections != contract.num_valid_detections:
        raise ContractError("S01 detections do not map to exactly one clip boundary")
    order = np.lexsort((det_id, det_frame))

    def ordered(name: str, dtype: np.dtype[Any]) -> np.ndarray:
        return _column_numpy(detections, name, dtype)[order]

    batch = DetectionBatch(
        det_id=det_id[order],
        global_frame=det_frame[order],
        global_time_sec=det_time[order],
        x1=ordered("x1", np.dtype(np.float64)),
        y1=ordered("y1", np.dtype(np.float64)),
        x2=ordered("x2", np.dtype(np.float64)),
        y2=ordered("y2", np.dtype(np.float64)),
        cx_norm=ordered("cx_norm", np.dtype(np.float64)),
        cy_norm=ordered("cy_norm", np.dtype(np.float64)),
        w_norm=ordered("w_norm", np.dtype(np.float64)),
        h_norm=ordered("h_norm", np.dtype(np.float64)),
    )
    return LoadedInput(
        contract.sequence_id,
        contract.clip_ids,
        frame_times,
        frame_period,
        batch,
    )


def _settings(config: MicrotrackConfig, frame_times: np.ndarray) -> MicrotrackSettings:
    max_gap = config.max_time_gap_sec
    if max_gap is None:
        max_gap = infer_auto_max_time_gap(frame_times, config.max_time_gap_multiplier)
    return MicrotrackSettings(
        center_distance_weight=config.center_distance_weight,
        iou_weight=config.iou_weight,
        size_weight=config.size_weight,
        max_time_gap_sec=max_gap,
        center_distance_gate=config.center_distance_gate,
        max_area_ratio=config.max_area_ratio,
        min_iou=config.min_iou,
        alternate_center_gate=config.alternate_center_gate,
        ambiguity_margin=config.ambiguity_margin,
        velocity_history_detections=config.velocity_history_detections,
        grid_width=config.fisheye_grid_width,
        grid_height=config.fisheye_grid_height,
        motion_prior_gate_floor=config.motion_prior_gate_floor,
        motion_prior_min_edges_per_cell=config.motion_prior_min_edges_per_cell,
    )


def _progress_logger(
    round_name: str, interval_sec: float, logger: LogFn
) -> Callable[[str, int, int], None]:
    last_report = {"forward": 0.0, "backward": 0.0}

    def report(direction: str, completed: int, total: int) -> None:
        if direction == "fixed_point":
            logger(
                f"[s01] {round_name} fixed-point prune: "
                f"retained={completed:,}/{total:,} edges"
            )
            return
        now = time.monotonic()
        if completed == total or now - last_report[direction] >= interval_sec:
            logger(
                f"[s01] {round_name} {direction}: {completed:,}/{total:,} frame pairs"
            )
            last_report[direction] = now

    return report


def _write_motion_prior(path: Path, prior: MotionPrior, settings: MicrotrackSettings) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            schema_version=np.asarray("1.0"),
            grid_width=np.asarray(settings.grid_width, dtype=np.int32),
            grid_height=np.asarray(settings.grid_height, dtype=np.int32),
            min_edges_per_cell=np.asarray(
                settings.motion_prior_min_edges_per_cell, dtype=np.int32
            ),
            applied_percentile=np.asarray(99.0, dtype=np.float32),
            center_gate_floor=np.asarray(
                settings.motion_prior_gate_floor, dtype=np.float32
            ),
            count=prior.count,
            residual_p50=prior.residual_p50,
            residual_p95=prior.residual_p95,
            residual_p99=prior.residual_p99,
            scale_p50=prior.scale_p50,
            scale_p95=prior.scale_p95,
            scale_p99=prior.scale_p99,
        )
    os.replace(temporary, path)


def _write_parquet(path: Path, table: pa.Table, compression: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    pq.write_table(table, temporary, compression=compression)
    os.replace(temporary, path)


def _edge_table(batch: DetectionBatch, links: LinkResult) -> pa.Table:
    count = links.num_edges
    delta_time = (
        batch.global_time_sec[links.dst_index] - batch.global_time_sec[links.src_index]
    ).astype(np.float32)
    return pa.Table.from_arrays(
        [
            pa.array(batch.det_id[links.src_index], type=pa.int64()),
            pa.array(batch.det_id[links.dst_index], type=pa.int64()),
            pa.array(delta_time, type=pa.float32()),
            pa.array(links.forward_cost, type=pa.float32()),
            pa.array(links.backward_cost, type=pa.float32()),
            pa.array(links.forward_rank, type=pa.int16()),
            pa.array(links.backward_rank, type=pa.int16()),
            pa.array(np.ones(count, dtype=np.bool_), type=pa.bool_()),
            pa.array(np.ones(count, dtype=np.bool_), type=pa.bool_()),
            pa.array(["accepted"] * count, type=pa.string()),
        ],
        schema=DET_EDGES_SCHEMA,
    )


def _mapping_table(batch: DetectionBatch, assignment: AssignmentResult) -> pa.Table:
    order = np.lexsort((assignment.order_in_micro, assignment.micro_id))
    scores = assignment.incoming_edge_score[order]
    return pa.Table.from_arrays(
        [
            pa.array(batch.det_id[order], type=pa.int64()),
            pa.array(assignment.micro_id[order], type=pa.int64()),
            pa.array(assignment.order_in_micro[order], type=pa.int32()),
            pa.array(scores, mask=np.isnan(scores), type=pa.float32()),
        ],
        schema=DET_TO_MICRO_SCHEMA,
    )


def _endpoint_velocity(
    batch: DetectionBatch, indices: np.ndarray
) -> tuple[float, float]:
    if len(indices) < 2:
        return 0.0, 0.0
    times = batch.global_time_sec[indices].astype(np.float64, copy=False)
    centered = times - float(np.mean(times))
    denominator = float(np.dot(centered, centered))
    if denominator <= np.finfo(np.float64).eps:
        return 0.0, 0.0
    vx = float(np.dot(centered, batch.cx_norm[indices]) / denominator)
    vy = float(np.dot(centered, batch.cy_norm[indices]) / denominator)
    return vx, vy


def _actual_center_jump(batch: DetectionBatch, source: int, destination: int) -> float:
    distance = float(
        np.hypot(
            batch.cx_norm[destination] - batch.cx_norm[source],
            batch.cy_norm[destination] - batch.cy_norm[source],
        )
    )
    diagonal_source = float(np.hypot(batch.w_norm[source], batch.h_norm[source]))
    diagonal_destination = float(
        np.hypot(batch.w_norm[destination], batch.h_norm[destination])
    )
    return distance / (
        0.5 * (diagonal_source + diagonal_destination) + np.finfo(np.float64).eps
    )


def _tracklet_table(
    batch: DetectionBatch,
    links: LinkResult,
    assignment: AssignmentResult,
    *,
    frame_period_sec: float,
    min_length_detections: int,
    velocity_history_detections: int,
    progress_interval_sec: float = 10.0,
    logger: LogFn | None = None,
) -> pa.Table:
    incoming_cost = np.full(len(batch.det_id), np.nan, dtype=np.float64)
    for edge_index, destination in enumerate(links.dst_index):
        incoming_cost[int(destination)] = 0.5 * (
            float(links.forward_cost[edge_index])
            + float(links.backward_cost[edge_index])
        )
    rows: list[dict[str, Any]] = []
    last_report = time.monotonic()
    for micro_id, path in enumerate(assignment.paths):
        start = int(path[0])
        end = int(path[-1])
        start_velocity = _endpoint_velocity(
            batch, path[:velocity_history_detections]
        )
        end_velocity = _endpoint_velocity(
            batch, path[-velocity_history_detections:]
        )
        internal_costs = incoming_cost[path[1:]]
        if len(path) > 1:
            if not np.all(np.isfinite(internal_costs)):
                raise ContractError("micro-tracklet internal edge lacks a finite cost")
            median_cost = float(np.median(internal_costs))
            maximum_cost = float(np.max(internal_costs))
            jumps = [
                _actual_center_jump(batch, int(source), int(destination))
                for source, destination in zip(path, path[1:])
            ]
            max_jump = max(jumps)
            local_purity = float(math.exp(-maximum_cost))
            bidirectional_agreement = 1.0
        else:
            median_cost = 0.0
            max_jump = 0.0
            local_purity = 0.0
            bidirectional_agreement = 0.0
        status = (
            MICROTRACKLET_STATUS_VALID
            if len(path) >= min_length_detections
            else MICROTRACKLET_STATUS_SHORT_FRAGMENT
        )
        rows.append(
            {
                "micro_id": micro_id,
                "start_global_frame": int(batch.global_frame[start]),
                "end_global_frame": int(batch.global_frame[end]),
                "start_time_sec": float(batch.global_time_sec[start]),
                "end_time_sec": float(batch.global_time_sec[end]),
                "num_detections": len(path),
                "duration_sec": float(
                    batch.global_time_sec[end]
                    - batch.global_time_sec[start]
                    + frame_period_sec
                ),
                "start_x1": float(batch.x1[start]),
                "start_y1": float(batch.y1[start]),
                "start_x2": float(batch.x2[start]),
                "start_y2": float(batch.y2[start]),
                "end_x1": float(batch.x1[end]),
                "end_y1": float(batch.y1[end]),
                "end_x2": float(batch.x2[end]),
                "end_y2": float(batch.y2[end]),
                "end_vx_norm_per_sec": end_velocity[0],
                "end_vy_norm_per_sec": end_velocity[1],
                "start_vx_norm_per_sec": start_velocity[0],
                "start_vy_norm_per_sec": start_velocity[1],
                "max_internal_center_jump": max_jump,
                "median_internal_cost": median_cost,
                "bidirectional_agreement": bidirectional_agreement,
                "local_purity_score": local_purity,
                "status": status,
            }
        )
        now = time.monotonic()
        if logger is not None and now - last_report >= progress_interval_sec:
            logger(
                f"[s01] summarize micro-tracklets: {micro_id + 1:,}/"
                f"{len(assignment.paths):,}"
            )
            last_report = now
    if logger is not None:
        logger(
            f"[s01] summarize micro-tracklets: {len(assignment.paths):,}/"
            f"{len(assignment.paths):,}"
        )
    return pa.Table.from_pylist(rows, schema=MICROTRACKLETS_SCHEMA)


def _validate_results(
    loaded: LoadedInput,
    first_pass: LinkResult,
    second_pass: LinkResult,
    prior: MotionPrior,
    assignment: AssignmentResult,
    edge_table: pa.Table,
    mapping_table: pa.Table,
    tracklet_table: pa.Table,
    settings: MicrotrackSettings,
) -> None:
    batch = loaded.batch
    if not second_pass.edge_pairs().issubset(first_pass.edge_pairs()):
        raise ContractError("motion-prior second pass created an edge absent from first pass")
    if edge_table.schema != DET_EDGES_SCHEMA:
        raise ContractError("det_edges in-memory schema mismatch")
    if mapping_table.schema != DET_TO_MICRO_SCHEMA:
        raise ContractError("det_to_micro in-memory schema mismatch")
    if tracklet_table.schema != MICROTRACKLETS_SCHEMA:
        raise ContractError("microtracklets in-memory schema mismatch")
    if mapping_table.num_rows != len(batch.det_id):
        raise ContractError("not every valid detection appears in det_to_micro")
    mapped_ids = mapping_table["det_id"].to_numpy(zero_copy_only=False)
    if np.unique(mapped_ids).size != len(batch.det_id):
        raise ContractError("det_to_micro contains duplicate det_id values")
    if not np.array_equal(
        np.sort(np.asarray(mapped_ids, dtype=np.int64)),
        np.sort(batch.det_id.astype(np.int64, copy=False)),
    ):
        raise ContractError("det_to_micro det_id set differs from valid S00 detections")
    if sum(len(path) for path in assignment.paths) != len(batch.det_id):
        raise ContractError("micro-tracklet paths do not partition valid detections")
    for path in assignment.paths:
        frames = batch.global_frame[path]
        if np.unique(frames).size != len(path):
            raise ContractError("a micro-tracklet contains multiple detections in one frame")
        if len(path) > 1 and not np.all(np.diff(frames) == 1):
            raise ContractError("a micro-tracklet bridges a missing global frame")
    if second_pass.num_edges != len(batch.det_id) - len(assignment.paths):
        raise ContractError("accepted edge count does not form a disjoint path cover")
    expected_shape = (settings.grid_height, settings.grid_width)
    for array in (
        prior.count,
        prior.residual_p50,
        prior.residual_p95,
        prior.residual_p99,
        prior.scale_p50,
        prior.scale_p95,
        prior.scale_p99,
    ):
        if array.shape != expected_shape:
            raise ContractError("motion prior grid shape mismatch")
    if int(np.sum(prior.count)) != first_pass.num_edges:
        raise ContractError("motion prior did not account for every first-pass edge")
    reliable = prior.count >= settings.motion_prior_min_edges_per_cell
    if np.any(reliable & ~np.isfinite(prior.residual_p99)):
        raise ContractError("reliable motion-prior cell has non-finite p99")


def _validate_output_files(output_dir: Path, expected_stats: dict[str, Any]) -> None:
    expected = {
        "det_edges.parquet": (DET_EDGES_SCHEMA, int(expected_stats["num_accepted_edges"])),
        "det_to_micro.parquet": (
            DET_TO_MICRO_SCHEMA,
            int(expected_stats["num_valid_detections"]),
        ),
        "microtracklets.parquet": (
            MICROTRACKLETS_SCHEMA,
            int(expected_stats["num_microtracklets"]),
        ),
    }
    for filename, (schema, rows) in expected.items():
        path = output_dir / filename
        if not path.is_file():
            raise ContractError(f"completed S01 output is missing: {filename}")
        parquet = pq.ParquetFile(path)
        if not parquet.schema_arrow.equals(schema, check_metadata=False):
            raise ContractError(f"completed S01 schema mismatch: {filename}")
        if parquet.metadata.num_rows != rows:
            raise ContractError(f"completed S01 row count mismatch: {filename}")
    prior_path = output_dir / "motion_prior.npz"
    if not prior_path.is_file():
        raise ContractError("completed S01 output is missing: motion_prior.npz")
    with np.load(prior_path, allow_pickle=False) as prior:
        required_keys = {
            "schema_version",
            "grid_width",
            "grid_height",
            "min_edges_per_cell",
            "applied_percentile",
            "center_gate_floor",
            "count",
            "residual_p50",
            "residual_p95",
            "residual_p99",
            "scale_p50",
            "scale_p95",
            "scale_p99",
        }
        if set(prior.files) != required_keys:
            raise ContractError("completed motion_prior.npz key set mismatch")
        if str(prior["schema_version"].item()) != "1.0":
            raise ContractError("completed motion prior schema_version mismatch")
        grid_width = int(prior["grid_width"].item())
        grid_height = int(prior["grid_height"].item())
        if grid_width != int(expected_stats["motion_grid_width"]) or grid_height != int(
            expected_stats["motion_grid_height"]
        ):
            raise ContractError("completed motion prior grid dimensions mismatch")
        if int(prior["min_edges_per_cell"].item()) != int(
            expected_stats["motion_prior_min_edges_per_cell"]
        ):
            raise ContractError("completed motion prior support threshold mismatch")
        if not math.isclose(
            float(prior["center_gate_floor"].item()),
            float(expected_stats["motion_prior_gate_floor"]),
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            raise ContractError("completed motion prior center gate floor mismatch")
        if not math.isclose(
            float(prior["applied_percentile"].item()),
            99.0,
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            raise ContractError("completed motion prior percentile mismatch")
        shape = (grid_height, grid_width)
        count = prior["count"]
        if count.shape != shape or not np.issubdtype(count.dtype, np.integer):
            raise ContractError("completed motion prior count array mismatch")
        if np.any(count < 0):
            raise ContractError("completed motion prior has negative cell count")
        if int(np.sum(count)) != int(expected_stats["num_first_pass_accepted_edges"]):
            raise ContractError("completed motion prior count total mismatch")
        percentile_arrays: dict[str, np.ndarray] = {}
        for key in (
            "residual_p50",
            "residual_p95",
            "residual_p99",
            "scale_p50",
            "scale_p95",
            "scale_p99",
        ):
            values = prior[key]
            percentile_arrays[key] = values
            if values.shape != shape or not np.issubdtype(values.dtype, np.floating):
                raise ContractError(f"completed motion prior array mismatch: {key}")
            if np.any((count > 0) & ~np.isfinite(values)):
                raise ContractError(f"populated motion prior cell is non-finite: {key}")
            if np.any((count == 0) & ~np.isnan(values)):
                raise ContractError(f"empty motion prior cell is not NaN: {key}")
        populated = count > 0
        if np.any(
            populated
            & (
                (percentile_arrays["residual_p50"] > percentile_arrays["residual_p95"])
                | (percentile_arrays["residual_p95"] > percentile_arrays["residual_p99"])
                | (percentile_arrays["scale_p50"] > percentile_arrays["scale_p95"])
                | (percentile_arrays["scale_p95"] > percentile_arrays["scale_p99"])
            )
        ):
            raise ContractError("completed motion prior percentile ordering mismatch")


def _validate_completed_output(
    output_dir: Path,
    existing: dict[str, Any],
    input_fingerprints: list[dict[str, Any]],
    config_hash: str,
) -> None:
    if existing.get("stage") != "S01":
        raise ContractError("existing _SUCCESS.json is not from S01")
    if existing.get("config_hash") != config_hash:
        raise ContractError("existing S01 output belongs to a different config")
    recorded_inputs = {
        str(Path(item["path"]).resolve()): item
        for item in existing.get("input_fingerprints", [])
    }
    current_inputs = {str(Path(item["path"]).resolve()): item for item in input_fingerprints}
    if set(recorded_inputs) != set(current_inputs):
        raise ContractError("completed S01 input fingerprint path set changed")
    for path, current in current_inputs.items():
        recorded = recorded_inputs[path]
        for key in ("size_bytes", "sha256"):
            if current[key] != recorded.get(key):
                raise ContractError(f"completed S01 input changed: {path} ({key})")
    recorded_outputs = existing.get("output_fingerprints")
    if not isinstance(recorded_outputs, list) or not recorded_outputs:
        raise ContractError("completed S01 has no output fingerprints")
    recorded_names = [str(item.get("path")) for item in recorded_outputs]
    if len(recorded_names) != len(set(recorded_names)):
        raise ContractError("completed S01 has duplicate output fingerprint paths")
    if set(recorded_names) != EXPECTED_S01_ARTIFACT_NAMES:
        raise ContractError("completed S01 output fingerprint set mismatch")
    for name in recorded_names:
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or len(relative.parts) != 1:
            raise ContractError(f"unsafe completed S01 artifact path: {name}")
    actual_names = {
        str(path.relative_to(output_dir))
        for path in output_dir.rglob("*")
        if path.is_file() and path.name != "_SUCCESS.json"
    }
    if actual_names != EXPECTED_S01_ARTIFACT_NAMES:
        raise ContractError("completed S01 artifact set mismatch")
    for recorded in recorded_outputs:
        artifact = output_dir / str(recorded["path"])
        if not artifact.is_file():
            raise ContractError(f"completed S01 artifact is missing: {recorded['path']}")
        current = _output_fingerprint(artifact, output_dir)
        for key in ("size_bytes", "sha256"):
            if current[key] != recorded.get(key):
                raise ContractError(
                    f"completed S01 artifact changed: {recorded['path']} ({key})"
                )
    stats = existing.get("stats")
    if not isinstance(stats, dict):
        raise ContractError("completed S01 success marker has no stats")
    _validate_output_files(output_dir, stats)


def run_s01(input_dir: Path, config_path: Path, output_dir: Path) -> dict[str, Any]:
    input_dir = input_dir.resolve()
    config_path = config_path.resolve()
    output_dir = output_dir.resolve()
    config, config_payload, config_hash = load_microtrack_config(config_path)
    s00_contract = _validate_s00_contract(input_dir)
    s00_input_paths = [
        input_dir / "_SUCCESS.json",
        input_dir / "frames.parquet",
        input_dir / "detections.parquet",
        input_dir / "resolved_manifest.json",
        input_dir / "ingest_report.json",
    ]
    input_paths = [*s00_input_paths, config_path]
    input_fingerprints = [
        _file_fingerprint(
            path,
            progress_interval_sec=config.progress_interval_sec,
            logger=log,
        )
        for path in input_paths
    ]
    _verify_s00_fingerprints(
        s00_contract, input_fingerprints[: len(s00_input_paths)]
    )

    success_path = output_dir / "_SUCCESS.json"
    if success_path.is_file():
        existing = _read_json(success_path, "S01 success marker")
        _validate_completed_output(output_dir, existing, input_fingerprints, config_hash)
        log(f"[s01] already complete and fully revalidated: {success_path}")
        return existing
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ContractError(
            f"output directory is non-empty without _SUCCESS.json: {output_dir}"
        )

    final_output_dir = output_dir
    final_output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = final_output_dir.parent / (
        f".{final_output_dir.name}.staging-{os.getpid()}"
    )
    if staging_dir.exists():
        raise ContractError(f"S01 staging directory already exists: {staging_dir}")
    staging_dir.mkdir(parents=False)
    output_dir = staging_dir

    log("[s01] loading and validating S00 Parquet inputs")
    loaded = _load_inputs(input_dir, s00_contract)
    settings = _settings(config, loaded.frame_times)
    log(
        f"[s01] loaded {len(loaded.batch.det_id):,} valid detections; "
        f"auto max_dt={settings.max_time_gap_sec:.9f}s"
    )

    first_pass = link_microtracks(
        loaded.batch,
        settings,
        progress=_progress_logger("round1", config.progress_interval_sec, log),
    )
    log(f"[s01] round1 accepted edges: {first_pass.num_edges:,}")
    prior = build_motion_prior(loaded.batch, first_pass, settings)
    reliable_cells = int(
        np.count_nonzero(prior.count >= settings.motion_prior_min_edges_per_cell)
    )
    log(
        f"[s01] motion prior: reliable cells={reliable_cells}/"
        f"{settings.grid_width * settings.grid_height}"
    )
    _write_motion_prior(output_dir / "motion_prior.npz", prior, settings)

    second_pass = link_microtracks(
        loaded.batch,
        settings,
        prior=prior,
        allowed_edge_pairs=first_pass.edge_pairs(),
        progress=_progress_logger("round2", config.progress_interval_sec, log),
    )
    if not second_pass.edge_pairs().issubset(first_pass.edge_pairs()):
        raise ContractError("round2 created a replacement edge instead of pruning")
    log(f"[s01] round2 accepted edges: {second_pass.num_edges:,}")
    assignment = assign_microtracks(loaded.batch, second_pass)

    edges = _edge_table(loaded.batch, second_pass)
    mapping = _mapping_table(loaded.batch, assignment)
    tracklets = _tracklet_table(
        loaded.batch,
        second_pass,
        assignment,
        frame_period_sec=loaded.frame_period_sec,
        min_length_detections=config.min_length_detections,
        velocity_history_detections=config.velocity_history_detections,
        progress_interval_sec=config.progress_interval_sec,
        logger=log,
    )
    _validate_results(
        loaded,
        first_pass,
        second_pass,
        prior,
        assignment,
        edges,
        mapping,
        tracklets,
        settings,
    )
    _write_parquet(output_dir / "det_edges.parquet", edges, config.parquet_compression)
    _write_parquet(
        output_dir / "det_to_micro.parquet", mapping, config.parquet_compression
    )
    _write_parquet(
        output_dir / "microtracklets.parquet", tracklets, config.parquet_compression
    )
    _atomic_write_json(output_dir / "effective_config.json", config_payload)

    statuses = tracklets["status"].to_pylist()
    stats = {
        "num_clips": len(loaded.clip_ids),
        "num_frames": len(loaded.frame_times),
        "num_valid_detections": len(loaded.batch.det_id),
        "num_first_pass_accepted_edges": first_pass.num_edges,
        "num_accepted_edges": second_pass.num_edges,
        "num_second_pass_accepted_edges": second_pass.num_edges,
        "num_edges_pruned_by_motion_prior": first_pass.num_edges - second_pass.num_edges,
        "num_microtracklets": len(assignment.paths),
        "num_valid_microtracklets": statuses.count(MICROTRACKLET_STATUS_VALID),
        "num_short_fragments": statuses.count(MICROTRACKLET_STATUS_SHORT_FRAGMENT),
        "num_reliable_motion_cells": reliable_cells,
        "motion_grid_width": settings.grid_width,
        "motion_grid_height": settings.grid_height,
        "motion_prior_min_edges_per_cell": settings.motion_prior_min_edges_per_cell,
        "motion_prior_gate_floor": settings.motion_prior_gate_floor,
    }
    report = {
        "stage": "S01",
        "schema_version": config.schema_version,
        "sequence_id": loaded.sequence_id,
        "clip_ids": list(loaded.clip_ids),
        "input_coordinate_system": "raw_encoded_landscape_no_autorotate",
        "input_rows_modified": False,
        "keypoints_used": False,
        "legacy_track_id_used": False,
        "bridge_missing_detections": False,
        "bidirectional_edges_only": True,
        "second_pass_only_prunes_first_pass": True,
        "empty_or_sparse_motion_cell_policy": "reject_without_fallback",
        "max_time_gap_sec": settings.max_time_gap_sec,
        "frame_period_sec": loaded.frame_period_sec,
        "duration_definition": "end_time_sec - start_time_sec + frame_period_sec",
        "metric_definitions": {
            "incoming_edge_score": "exp(-mean(forward_cost, backward_cost))",
            "local_purity_score": "exp(-maximum_internal_mean_directional_cost)",
            "max_internal_center_jump": "observed center displacement divided by mean bbox diagonal",
            "bidirectional_agreement": "1 for linked paths; 0 for singleton fragments",
        },
        "status_policy": {
            "valid": f"num_detections >= {config.min_length_detections}",
            "short_fragment": f"num_detections < {config.min_length_detections}",
            "junk_candidate": "not assigned automatically from S00 valid=true boxes",
        },
        "stats": stats,
    }
    _atomic_write_json(output_dir / "microtrack_report.json", report)
    _validate_output_files(output_dir, stats)
    _verify_inputs_unchanged(input_fingerprints)

    artifact_paths = sorted(
        path
        for path in output_dir.rglob("*")
        if path.is_file() and path.name != "_SUCCESS.json"
    )
    output_fingerprints = [
        _output_fingerprint(path, output_dir) for path in artifact_paths
    ]
    success = {
        "stage": "S01",
        "schema_version": config.schema_version,
        "config_hash": config_hash,
        "program_commit_hash": None,
        "input_fingerprints": input_fingerprints,
        "output_fingerprints": output_fingerprints,
        "stats": stats,
    }
    _atomic_write_json(output_dir / "_SUCCESS.json", success)

    if final_output_dir.exists():
        if any(final_output_dir.iterdir()):
            raise ContractError(
                f"final S01 output became non-empty during staging: {final_output_dir}"
            )
        final_output_dir.rmdir()
    os.replace(staging_dir, final_output_dir)
    success_path = final_output_dir / "_SUCCESS.json"
    log(f"[s01] complete: {success_path}")
    return success
