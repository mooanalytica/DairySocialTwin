from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class ContractError(RuntimeError):
    """Raised when an input cannot satisfy a fixed pipeline-stage contract."""


@dataclass(frozen=True)
class ClipManifest:
    sequence_id: str
    clip_order: int
    clip_id: str
    video_path: Path
    bbox_csv_path: Path
    frame_index_base: int
    bbox_format: str


@dataclass(frozen=True)
class InputCSVConfig:
    video_column: str
    frame_column: str
    x_column: str
    y_column: str
    width_column: str
    height_column: str
    confidence_column: str
    legacy_track_id_column: str


@dataclass(frozen=True)
class IngestConfig:
    schema_version: str
    random_seed: int
    input_csv: InputCSVConfig
    ffprobe_binary: str
    opencv_auto_rotate: bool
    clamp_boxes: bool
    retain_invalid_rows: bool
    minimum_bbox_area_pixels: float
    minimum_retained_area_fraction: float
    duplicate_iou_threshold: float
    duplicate_area_similarity_min: float
    parquet_compression: str
    parquet_batch_rows: int
    progress_interval_sec: float
    create_overlay_samples: bool
    overlay_jpeg_quality: int
    acceptance: dict[str, Any]


@dataclass(frozen=True)
class MicrotrackConfig:
    """Fixed S01 configuration, independent from the completed S00 config."""

    schema_version: str
    random_seed: int
    center_distance_weight: float
    iou_weight: float
    size_weight: float
    max_time_gap_sec: float | None
    max_time_gap_multiplier: float
    center_distance_gate: float
    max_area_ratio: float
    min_iou: float
    alternate_center_gate: float
    ambiguity_margin: float
    bidirectional_edges_only: bool
    bridge_missing_detections: bool
    min_length_detections: int
    velocity_history_detections: int
    fisheye_grid_width: int
    fisheye_grid_height: int
    motion_prior_percentile: float
    motion_prior_gate_floor: float
    motion_prior_min_edges_per_cell: int
    parquet_compression: str
    progress_interval_sec: float


def _required(mapping: dict[str, Any], key: str, context: str) -> Any:
    if key not in mapping:
        raise ContractError(f"missing required config key: {context}.{key}")
    return mapping[key]


def _required_bool(mapping: dict[str, Any], key: str, context: str) -> bool:
    value = _required(mapping, key, context)
    if not isinstance(value, bool):
        raise ContractError(f"{context}.{key} must be a boolean")
    return value


def _required_finite_float(
    mapping: dict[str, Any], key: str, context: str
) -> float:
    value = _required(mapping, key, context)
    if isinstance(value, bool):
        raise ContractError(f"{context}.{key} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ContractError(f"{context}.{key} must be a finite number") from exc
    if not math.isfinite(result):
        raise ContractError(f"{context}.{key} must be a finite number")
    return result


def _required_int(mapping: dict[str, Any], key: str, context: str) -> int:
    value = _required(mapping, key, context)
    if isinstance(value, bool):
        raise ContractError(f"{context}.{key} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ContractError(f"{context}.{key} must be an integer") from exc
    if isinstance(value, float) and not value.is_integer():
        raise ContractError(f"{context}.{key} must be an integer")
    return result


def load_microtrack_config(
    path: Path,
) -> tuple[MicrotrackConfig, dict[str, Any], str]:
    """Load the strict, standalone S01 configuration."""

    path = path.resolve()
    if not path.is_file():
        raise ContractError(f"microtrack config does not exist: {path}")
    raw_bytes = path.read_bytes()
    payload = yaml.safe_load(raw_bytes)
    if not isinstance(payload, dict):
        raise ContractError("microtrack config root must be a mapping")
    expected_sections = {"pipeline", "microtrack"}
    if set(payload) != expected_sections:
        raise ContractError(
            "microtrack config must contain exactly pipeline and microtrack sections"
        )
    pipeline = _required(payload, "pipeline", "config")
    microtrack = _required(payload, "microtrack", "config")
    if not isinstance(pipeline, dict) or not isinstance(microtrack, dict):
        raise ContractError("pipeline and microtrack must be mappings")
    if _required_bool(pipeline, "use_keypoints", "pipeline"):
        raise ContractError("S01 requires pipeline.use_keypoints=false")
    if _required_bool(pipeline, "use_legacy_tracking_id", "pipeline"):
        raise ContractError("S01 requires pipeline.use_legacy_tracking_id=false")

    weights = _required(microtrack, "cost_weights", "microtrack")
    if not isinstance(weights, dict):
        raise ContractError("microtrack.cost_weights must be a mapping")
    grid = _required(microtrack, "fisheye_motion_grid", "microtrack")
    if (
        not isinstance(grid, list)
        or len(grid) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) for value in grid)
    ):
        raise ContractError("microtrack.fisheye_motion_grid must be [width, height]")

    max_gap_value = _required(microtrack, "max_time_gap_sec", "microtrack")
    if isinstance(max_gap_value, str) and max_gap_value == "auto":
        max_time_gap_sec: float | None = None
    elif isinstance(max_gap_value, bool):
        raise ContractError("microtrack.max_time_gap_sec must be auto or a positive number")
    else:
        try:
            max_time_gap_sec = float(max_gap_value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ContractError(
                "microtrack.max_time_gap_sec must be auto or a positive number"
            ) from exc
        if not math.isfinite(max_time_gap_sec) or max_time_gap_sec <= 0.0:
            raise ContractError(
                "microtrack.max_time_gap_sec must be auto or a positive number"
            )

    result = MicrotrackConfig(
        schema_version=str(_required(pipeline, "schema_version", "pipeline")),
        random_seed=_required_int(pipeline, "random_seed", "pipeline"),
        center_distance_weight=_required_finite_float(
            weights, "center_distance", "microtrack.cost_weights"
        ),
        iou_weight=_required_finite_float(weights, "iou", "microtrack.cost_weights"),
        size_weight=_required_finite_float(weights, "size", "microtrack.cost_weights"),
        max_time_gap_sec=max_time_gap_sec,
        max_time_gap_multiplier=_required_finite_float(
            microtrack, "max_time_gap_multiplier", "microtrack"
        ),
        center_distance_gate=_required_finite_float(
            microtrack, "center_distance_gate", "microtrack"
        ),
        max_area_ratio=_required_finite_float(
            microtrack, "max_area_ratio", "microtrack"
        ),
        min_iou=_required_finite_float(microtrack, "min_iou", "microtrack"),
        alternate_center_gate=_required_finite_float(
            microtrack, "alternate_center_gate", "microtrack"
        ),
        ambiguity_margin=_required_finite_float(
            microtrack, "ambiguity_margin", "microtrack"
        ),
        bidirectional_edges_only=_required_bool(
            microtrack, "bidirectional_edges_only", "microtrack"
        ),
        bridge_missing_detections=_required_bool(
            microtrack, "bridge_missing_detections", "microtrack"
        ),
        min_length_detections=_required_int(
            microtrack, "min_length_detections", "microtrack"
        ),
        velocity_history_detections=_required_int(
            microtrack, "velocity_history_detections", "microtrack"
        ),
        fisheye_grid_width=int(grid[0]),
        fisheye_grid_height=int(grid[1]),
        motion_prior_percentile=_required_finite_float(
            microtrack, "motion_prior_percentile", "microtrack"
        ),
        motion_prior_gate_floor=_required_finite_float(
            microtrack, "motion_prior_gate_floor", "microtrack"
        ),
        motion_prior_min_edges_per_cell=_required_int(
            microtrack, "motion_prior_min_edges_per_cell", "microtrack"
        ),
        parquet_compression=str(
            _required(microtrack, "parquet_compression", "microtrack")
        ),
        progress_interval_sec=_required_finite_float(
            microtrack, "progress_interval_sec", "microtrack"
        ),
    )

    weights_tuple = (
        result.center_distance_weight,
        result.iou_weight,
        result.size_weight,
    )
    if result.schema_version != "1.0":
        raise ContractError("fixed S01 requires pipeline.schema_version=1.0")
    if any(not math.isfinite(value) or value < 0.0 for value in weights_tuple):
        raise ContractError("microtrack cost weights must be finite and non-negative")
    if not math.isclose(sum(weights_tuple), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ContractError("microtrack cost weights must sum to 1")
    if result.max_time_gap_multiplier <= 0.0:
        raise ContractError("microtrack.max_time_gap_multiplier must be positive")
    if result.center_distance_gate <= 0.0:
        raise ContractError("microtrack.center_distance_gate must be positive")
    if result.max_area_ratio < 1.0:
        raise ContractError("microtrack.max_area_ratio must be at least 1")
    if not 0.0 <= result.min_iou <= 1.0:
        raise ContractError("microtrack.min_iou must be in [0, 1]")
    if not 0.0 < result.alternate_center_gate <= result.center_distance_gate:
        raise ContractError(
            "microtrack.alternate_center_gate must be in (0, center_distance_gate]"
        )
    if result.ambiguity_margin < 0.0:
        raise ContractError("microtrack.ambiguity_margin must be non-negative")
    if not result.bidirectional_edges_only:
        raise ContractError("S01 requires bidirectional_edges_only=true")
    if result.bridge_missing_detections:
        raise ContractError("S01 requires bridge_missing_detections=false")
    if result.min_length_detections < 1:
        raise ContractError("microtrack.min_length_detections must be positive")
    if not 2 <= result.velocity_history_detections <= 4:
        raise ContractError("velocity_history_detections must be in [2, 4]")
    if result.fisheye_grid_width <= 0 or result.fisheye_grid_height <= 0:
        raise ContractError("fisheye motion grid dimensions must be positive")
    if not 0.0 < result.motion_prior_percentile <= 100.0:
        raise ContractError("motion_prior_percentile must be in (0, 100]")
    if not math.isclose(
        result.motion_prior_percentile, 99.0, rel_tol=0.0, abs_tol=1e-9
    ):
        raise ContractError("fixed S01 requires motion_prior_percentile=99")
    if not 0.0 < result.motion_prior_gate_floor <= result.center_distance_gate:
        raise ContractError(
            "motion_prior_gate_floor must be in (0, center_distance_gate]"
        )
    if result.motion_prior_min_edges_per_cell < 1:
        raise ContractError("motion_prior_min_edges_per_cell must be positive")
    if not result.parquet_compression:
        raise ContractError("microtrack.parquet_compression cannot be blank")
    if result.progress_interval_sec <= 0.0:
        raise ContractError("microtrack.progress_interval_sec must be positive")

    canonical = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return result, payload, hashlib.sha256(canonical).hexdigest()


def load_config(path: Path) -> tuple[IngestConfig, dict[str, Any], str]:
    path = path.resolve()
    if not path.is_file():
        raise ContractError(f"config does not exist: {path}")
    raw_bytes = path.read_bytes()
    payload = yaml.safe_load(raw_bytes)
    if not isinstance(payload, dict):
        raise ContractError("config root must be a mapping")

    pipeline = _required(payload, "pipeline", "config")
    input_csv = _required(payload, "input_csv", "config")
    ingest = _required(payload, "ingest", "config")
    acceptance = _required(payload, "acceptance", "config")
    if not all(
        isinstance(item, dict) for item in (pipeline, input_csv, ingest, acceptance)
    ):
        raise ContractError("pipeline, input_csv, ingest, and acceptance must be mappings")
    if bool(_required(pipeline, "use_keypoints", "pipeline")):
        raise ContractError("S00 requires pipeline.use_keypoints=false")
    if bool(_required(pipeline, "use_legacy_tracking_id", "pipeline")):
        raise ContractError("S00 requires pipeline.use_legacy_tracking_id=false")

    csv_config = InputCSVConfig(
        video_column=str(_required(input_csv, "video_column", "input_csv")),
        frame_column=str(_required(input_csv, "frame_column", "input_csv")),
        x_column=str(_required(input_csv, "x_column", "input_csv")),
        y_column=str(_required(input_csv, "y_column", "input_csv")),
        width_column=str(_required(input_csv, "width_column", "input_csv")),
        height_column=str(_required(input_csv, "height_column", "input_csv")),
        confidence_column=str(_required(input_csv, "confidence_column", "input_csv")),
        legacy_track_id_column=str(
            _required(input_csv, "legacy_track_id_column", "input_csv")
        ),
    )
    result = IngestConfig(
        schema_version=str(_required(pipeline, "schema_version", "pipeline")),
        random_seed=int(_required(pipeline, "random_seed", "pipeline")),
        input_csv=csv_config,
        ffprobe_binary=str(_required(ingest, "ffprobe_binary", "ingest")),
        opencv_auto_rotate=bool(_required(ingest, "opencv_auto_rotate", "ingest")),
        clamp_boxes=bool(_required(ingest, "clamp_boxes", "ingest")),
        retain_invalid_rows=bool(_required(ingest, "retain_invalid_rows", "ingest")),
        minimum_bbox_area_pixels=float(
            _required(ingest, "minimum_bbox_area_pixels", "ingest")
        ),
        minimum_retained_area_fraction=float(
            _required(ingest, "minimum_retained_area_fraction", "ingest")
        ),
        duplicate_iou_threshold=float(
            _required(ingest, "duplicate_iou_threshold", "ingest")
        ),
        duplicate_area_similarity_min=float(
            _required(ingest, "duplicate_area_similarity_min", "ingest")
        ),
        parquet_compression=str(_required(ingest, "parquet_compression", "ingest")),
        parquet_batch_rows=int(_required(ingest, "parquet_batch_rows", "ingest")),
        progress_interval_sec=float(
            _required(ingest, "progress_interval_sec", "ingest")
        ),
        create_overlay_samples=bool(
            _required(ingest, "create_overlay_samples", "ingest")
        ),
        overlay_jpeg_quality=int(_required(ingest, "overlay_jpeg_quality", "ingest")),
        acceptance={str(key): value for key, value in acceptance.items()},
    )
    if result.opencv_auto_rotate:
        raise ContractError(
            "opencv_auto_rotate must be false: Stage1 bbox coordinates use raw 3840x2160 frames"
        )
    if not result.clamp_boxes or not result.retain_invalid_rows:
        raise ContractError("S00 requires clamp_boxes=true and retain_invalid_rows=true")
    if not 0.0 < result.minimum_retained_area_fraction <= 1.0:
        raise ContractError("minimum_retained_area_fraction must be in (0, 1]")
    if not 0.0 < result.duplicate_iou_threshold <= 1.0:
        raise ContractError("duplicate_iou_threshold must be in (0, 1]")
    if not 0.0 < result.duplicate_area_similarity_min <= 1.0:
        raise ContractError("duplicate_area_similarity_min must be in (0, 1]")
    if result.minimum_bbox_area_pixels <= 0 or result.parquet_batch_rows <= 0:
        raise ContractError("bbox area and parquet batch size must be positive")
    if not 1 <= result.overlay_jpeg_quality <= 100:
        raise ContractError("overlay_jpeg_quality must be in [1, 100]")
    canonical = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return result, payload, hashlib.sha256(canonical).hexdigest()


def load_manifest(path: Path) -> tuple[list[ClipManifest], str]:
    path = path.resolve()
    if not path.is_file():
        raise ContractError(f"manifest does not exist: {path}")
    raw_bytes = path.read_bytes()
    required = {
        "sequence_id",
        "clip_order",
        "clip_id",
        "video_path",
        "bbox_csv_path",
        "frame_index_base",
        "bbox_format",
    }
    rows: list[ClipManifest] = []
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            missing = sorted(required - set(reader.fieldnames or []))
            raise ContractError(f"manifest missing columns: {missing}")
        for row_number, row in enumerate(reader, start=2):
            try:
                video_path = Path(row["video_path"]).expanduser().resolve()
                bbox_path = Path(row["bbox_csv_path"]).expanduser().resolve()
                clip = ClipManifest(
                    sequence_id=row["sequence_id"].strip(),
                    clip_order=int(row["clip_order"]),
                    clip_id=row["clip_id"].strip(),
                    video_path=video_path,
                    bbox_csv_path=bbox_path,
                    frame_index_base=int(row["frame_index_base"]),
                    bbox_format=row["bbox_format"].strip().lower(),
                )
            except Exception as exc:
                raise ContractError(f"invalid manifest row {row_number}: {exc}") from exc
            if not clip.sequence_id or not clip.clip_id:
                raise ContractError(f"blank sequence_id or clip_id at manifest row {row_number}")
            if "|" in clip.sequence_id or "|" in clip.clip_id:
                raise ContractError("sequence_id and clip_id cannot contain '|' (det_id delimiter)")
            if clip.bbox_format not in {"xywh"}:
                raise ContractError(
                    f"this fixed S00 dataset requires bbox_format=xywh, got {clip.bbox_format}"
                )
            if clip.frame_index_base != 0:
                raise ContractError("this fixed S00 dataset requires frame_index_base=0")
            if not clip.video_path.is_file():
                raise ContractError(f"video does not exist: {clip.video_path}")
            if not clip.bbox_csv_path.is_file():
                raise ContractError(f"bbox CSV does not exist: {clip.bbox_csv_path}")
            rows.append(clip)
    if not rows:
        raise ContractError("manifest has no clips")
    rows.sort(key=lambda item: item.clip_order)
    if [item.clip_order for item in rows] != list(range(len(rows))):
        raise ContractError("clip_order must be unique and contiguous from zero")
    if len({item.sequence_id for item in rows}) != 1:
        raise ContractError("all clips must share exactly one sequence_id")
    for label, values in {
        "clip_id": [item.clip_id for item in rows],
        "video_path": [str(item.video_path) for item in rows],
        "bbox_csv_path": [str(item.bbox_csv_path) for item in rows],
    }.items():
        if len(values) != len(set(values)):
            raise ContractError(f"manifest {label} values must be unique")
    return rows, hashlib.sha256(raw_bytes).hexdigest()
