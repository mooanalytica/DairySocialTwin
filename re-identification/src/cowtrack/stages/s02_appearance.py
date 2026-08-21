from __future__ import annotations

import hashlib
import json
import math
import os
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence, cast

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from cowtrack.appearance.config import (
    LOCKED_ENCODER_PROFILE,
    LOCKED_SELECTION_MODE,
    AppearanceConfig,
    load_appearance_config,
)
from cowtrack.appearance.encoders import (
    EncoderProfile,
    build_encoder,
    embed_four_views,
    production_encoder_profiles,
)
from cowtrack.appearance.prototypes import build_tracklet_prototypes
from cowtrack.appearance.quality import (
    bbox_area_percentiles,
    combine_crop_quality,
    crop_with_padding,
    laplacian_blur_score,
    maximum_other_bbox_iou,
    soft_border_mask,
)
from cowtrack.appearance.sampling import (
    select_candidate_pool_indices,
    select_representative_indices,
)
from cowtrack.config import ClipManifest, ContractError, load_manifest
from cowtrack.schemas.appearance import (
    APPEARANCE_EXCLUSIONS_SCHEMA,
    APPEARANCE_SAMPLES_SCHEMA,
    MICRO_APPEARANCE_SCHEMA,
    MICRO_CONTEXT_SCHEMA,
)
from cowtrack.schemas.detections import DETECTIONS_SCHEMA
from cowtrack.schemas.frames import FRAMES_SCHEMA
from cowtrack.schemas.tracklets import (
    DET_TO_MICRO_SCHEMA,
    MICROTRACKLETS_SCHEMA,
    MICROTRACKLET_STATUSES,
)
from cowtrack.video import open_raw_video_capture


LogFn = Callable[[str], None]

EXPECTED_COORDINATE_SYSTEM = "raw_encoded_landscape_no_autorotate"
RAW_WIDTH = 3_840
RAW_HEIGHT = 2_160
TRUE_EVENT_REASONS = frozenset(
    {"low_purity", "large_center_jump", "legacy_id_transition"}
)
REVIEW_SCHEMA_VERSION = "cowtrack.s01-video-review.v2"
DEFAULT_EVENT_RADIUS_FRAMES = 15


def log(message: str) -> None:
    print(message, flush=True)


@dataclass(frozen=True)
class LoadedAppearanceData:
    sequence_id: str
    frames: dict[str, np.ndarray]
    detections: dict[str, np.ndarray]
    microtracklets: dict[str, np.ndarray]
    micro_ids_by_detection: np.ndarray
    order_in_micro_by_detection: np.ndarray
    paths: dict[int, np.ndarray]
    positions_by_frame: np.ndarray
    frame_offsets: np.ndarray
    video_paths: dict[str, Path]
    input_fingerprints: tuple[dict[str, Any], ...]

    @property
    def num_frames(self) -> int:
        return len(self.frame_offsets) - 1

    @property
    def num_valid_detections(self) -> int:
        return len(self.detections["det_id"])

    @property
    def num_microtracklets(self) -> int:
        return len(self.microtracklets["micro_id"])


@dataclass(frozen=True)
class ExclusionResult:
    excluded: np.ndarray
    reasons: tuple[str, ...]
    case_ids: tuple[str, ...]
    trigger_frames: tuple[str, ...]
    num_true_events: int


@dataclass(frozen=True)
class CropRecord:
    detection_position: int
    micro_id: int
    det_id: int
    clip_id: str
    global_frame: int
    global_time_sec: float
    crop_quality: float
    other_bbox_max_iou: float
    clipped_fraction: float
    bbox_area_percentile: float
    blur_score: float
    distance_to_image_boundary: float
    selection_reason: str


class _EmbeddingSession:
    def __init__(
        self,
        profiles: Sequence[EncoderProfile],
        config: AppearanceConfig,
        *,
        device: str,
        encoder_factory: Callable[..., Any] = build_encoder,
        logger: LogFn = log,
    ) -> None:
        if device != config.device:
            raise ContractError(
                f"S02 CLI device {device!r} differs from config device {config.device!r}"
            )
        if device != "cuda:0":
            raise ContractError(
                "fixed S02 requires CUDA_VISIBLE_DEVICES=1 with logical device cuda:0"
            )
        self.config = config
        self.profiles = tuple(profiles)
        configured_names = tuple(profile.name for profile in config.model_profiles)
        runtime_names = tuple(profile.name for profile in self.profiles)
        if configured_names != (LOCKED_ENCODER_PROFILE,):
            raise ContractError("S02 config does not contain only the locked encoder")
        if runtime_names != configured_names:
            raise ContractError("S02 runtime encoder differs from the locked encoder")
        if config.selection_mode != LOCKED_SELECTION_MODE:
            raise ContractError("S02 encoder selection mode is not locked")
        self.profile = self.profiles[0]
        self.encoder = encoder_factory(self.profile, device=device)
        self.logger = logger
        self.records: list[CropRecord] = []
        self._final_chunks: list[np.ndarray] = []
        self.winner_name = self.profile.name
        self.separation_warning = False
        self.benchmark_report: dict[str, Any] = {
            "schema_version": config.schema_version,
            "selection_mode": config.selection_mode,
            "selected_profile": self.winner_name,
            "evaluation_mode": LOCKED_SELECTION_MODE,
            "target_tpr": None,
            "num_embedding_rows": 0,
            "num_production_embedding_rows": 0,
            "pair_counts": {"positive": 0, "negative": 0},
            "split_pair_counts": {
                "calibration_positive": 0,
                "calibration_negative": 0,
                "validation_positive": 0,
                "validation_negative": 0,
            },
            "profiles": {self.winner_name: {}},
            "selection": {
                "winner": self.winner_name,
                "priority": [],
                "basis": LOCKED_SELECTION_MODE,
            },
            "quality_gate": {
                "evaluated": False,
                "minimum_separation_margin": None,
                "observed_winner_margin": None,
                "passed": None,
                "policy": "not_applicable_locked_existing_winner",
            },
        }
        self.logger(
            f"[s02] encoder locked to {self.winner_name}; "
            "multi-model bake-off disabled"
        )

    def consume(
        self,
        raw_crops: Sequence[np.ndarray],
        masked_crops: Sequence[np.ndarray],
        records: Sequence[CropRecord],
    ) -> None:
        if not records:
            return
        if not (len(raw_crops) == len(masked_crops) == len(records)):
            raise ContractError("S02 crop batch lengths differ")
        embedded = embed_four_views(
            self.encoder,
            raw_crops,
            masked_crops,
            batch_size=self.config.batch_size,
        )
        self._final_chunks.append(embedded)
        self.records.extend(records)

    def finalize(self) -> tuple[np.ndarray, str, dict[str, Any], bool]:
        if not self._final_chunks:
            raise ContractError("S02 found no high-quality crops for the locked encoder")
        embeddings = np.concatenate(self._final_chunks, axis=0).astype(
            np.float32, copy=False
        )
        if len(embeddings) != len(self.records):
            raise ContractError("S02 embedding/sample row count mismatch")
        if not np.isfinite(embeddings).all():
            raise ContractError("S02 final embeddings contain non-finite values")
        norms = np.linalg.norm(embeddings, axis=1)
        if not np.allclose(norms, 1.0, rtol=0.0, atol=1e-4):
            raise ContractError("S02 final embeddings are not L2-normalized")
        self.benchmark_report["num_production_embedding_rows"] = int(len(embeddings))
        return (
            embeddings,
            self.winner_name,
            self.benchmark_report,
            self.separation_warning,
        )


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


def _read_json(path: Path, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read {label}: {path}: {exc}") from exc


def _fingerprint(
    path: Path,
    *,
    progress_interval_sec: float | None = None,
    logger: LogFn | None = None,
) -> dict[str, Any]:
    try:
        before = path.stat()
    except OSError as exc:
        raise ContractError(f"required input does not exist: {path}: {exc}") from exc
    if not path.is_file():
        raise ContractError(f"required input is not a file: {path}")
    digest = hashlib.sha256()
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
                    percent = (
                        100.0 * processed / before.st_size if before.st_size else 100.0
                    )
                    logger(
                        f"[s02] fingerprint {path.name}: "
                        f"{processed / 1024**3:.2f}/{before.st_size / 1024**3:.2f} GiB "
                        f"({percent:.1f}%)"
                    )
                    last_report = now
    except OSError as exc:
        raise ContractError(f"cannot fingerprint {path}: {exc}") from exc
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


def _parquet_columns(
    path: Path,
    *,
    expected_schema: pa.Schema,
    expected_rows: int,
    columns: Sequence[str],
    label: str,
    filters: list[tuple[str, str, Any]] | None = None,
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
        table = pq.read_table(path, columns=list(columns), filters=filters)
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
    progress_interval_sec: float,
    logger: LogFn,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    success_path = directory / "_SUCCESS.json"
    success = _read_json(success_path, f"{expected_stage} success marker")
    if not isinstance(success, dict) or success.get("stage") != expected_stage:
        raise ContractError(f"{success_path} is not completed {expected_stage} output")
    records = success.get("output_fingerprints")
    if not isinstance(records, list):
        raise ContractError(f"{expected_stage} success marker lacks fingerprints")
    by_name = {
        str(item.get("path")): item for item in records if isinstance(item, dict)
    }
    missing = sorted(set(required_names) - set(by_name))
    if missing:
        raise ContractError(f"{expected_stage} success marker lacks artifacts: {missing}")
    observed: list[dict[str, Any]] = []
    for name in required_names:
        current = _fingerprint(
            directory / name,
            progress_interval_sec=progress_interval_sec,
            logger=logger,
        )
        recorded = by_name[name]
        for key in ("size_bytes", "sha256"):
            if current[key] != recorded.get(key):
                raise ContractError(
                    f"completed {expected_stage} artifact changed: {name} ({key})"
                )
        observed.append(current)
    observed.append(_fingerprint(success_path))
    return success, observed


def _manifest_rows(rows: Sequence[ClipManifest]) -> list[dict[str, Any]]:
    return [
        {
            "sequence_id": row.sequence_id,
            "clip_order": row.clip_order,
            "clip_id": row.clip_id,
            "video_path": str(row.video_path),
            "bbox_csv_path": str(row.bbox_csv_path),
            "frame_index_base": row.frame_index_base,
            "bbox_format": row.bbox_format,
        }
        for row in rows
    ]


def _validate_manifest_bijection(
    manifest: Sequence[ClipManifest], resolved_manifest_path: Path
) -> dict[str, Path]:
    if not manifest:
        raise ContractError("S02 manifest has no clips")
    clip_ids = [row.clip_id for row in manifest]
    if len(set(clip_ids)) != len(clip_ids):
        raise ContractError("S02 manifest clip_id values are not unique")
    if [row.clip_order for row in manifest] != list(range(len(manifest))):
        raise ContractError("S02 manifest clip order is not contiguous from zero")
    if len({row.sequence_id for row in manifest}) != 1:
        raise ContractError("S02 manifest must contain exactly one sequence_id")
    resolved = _read_json(resolved_manifest_path, "S00 resolved manifest")
    if not isinstance(resolved, list):
        raise ContractError("S00 resolved manifest must be a JSON list")
    expected = _manifest_rows(manifest)
    if resolved != expected:
        raise ContractError("live manifest and immutable S00 resolved manifest differ")
    return {row.clip_id: row.video_path for row in manifest}


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
    num_input_detections = _required_stat(
        s00_stats, "S00", "num_input_boxes", minimum=1
    )
    num_valid_detections = _required_stat(
        s00_stats, "S00", "num_valid_boxes", minimum=1
    )
    s01_num_frames = _required_stat(s01_stats, "S01", "num_frames", minimum=1)
    s01_num_valid = _required_stat(
        s01_stats, "S01", "num_valid_detections", minimum=1
    )
    num_microtracklets = _required_stat(
        s01_stats, "S01", "num_microtracklets", minimum=1
    )

    if num_clips != manifest_num_clips:
        raise ContractError(
            "S00 clip count differs from the live/resolved manifest: "
            f"{num_clips} != {manifest_num_clips}"
        )
    if num_input_detections < num_valid_detections:
        raise ContractError("S00 valid detection count exceeds input detection count")
    if s01_num_frames != num_frames:
        raise ContractError("S00/S01 frame counts differ")
    if s01_num_valid != num_valid_detections:
        raise ContractError("S00/S01 valid detection counts differ")
    if num_microtracklets > num_valid_detections:
        raise ContractError("S01 microtracklet count exceeds valid detections")
    return {
        "num_clips": num_clips,
        "num_frames": num_frames,
        "num_input_detections": num_input_detections,
        "num_valid_detections": num_valid_detections,
        "num_microtracklets": num_microtracklets,
    }


def _validate_clip_boundaries(
    manifest: Sequence[ClipManifest],
    boundaries: Any,
    *,
    num_frames: int,
) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(boundaries, list) or len(boundaries) != len(manifest):
        raise ContractError(
            "S02 S00 clip-boundary count differs from the resolved manifest"
        )
    validated: list[Mapping[str, Any]] = []
    expected_start = 0
    for clip, boundary in zip(manifest, boundaries, strict=True):
        if not isinstance(boundary, dict):
            raise ContractError("S02 S00 clip boundary must be an object")
        if boundary.get("clip_id") != clip.clip_id or boundary.get(
            "clip_order"
        ) != clip.clip_order:
            raise ContractError(
                f"S02 S00 clip boundary order differs for {clip.clip_id}"
            )
        clip_frames = boundary.get("num_frames")
        start = boundary.get("start_global_frame")
        end = boundary.get("end_global_frame_inclusive")
        if (
            isinstance(clip_frames, bool)
            or not isinstance(clip_frames, int)
            or clip_frames <= 0
            or isinstance(start, bool)
            or not isinstance(start, int)
            or isinstance(end, bool)
            or not isinstance(end, int)
        ):
            raise ContractError(f"S02 S00 clip boundary is malformed: {clip.clip_id}")
        if start != expected_start or end != start + clip_frames - 1:
            raise ContractError(
                f"S02 S00 clip boundary is not contiguous: {clip.clip_id}"
            )
        if boundary.get("opencv_auto_rotate") is not False:
            raise ContractError("S02 requires every S00 clip to disable autorotation")
        if boundary.get("width") != RAW_WIDTH or boundary.get("height") != RAW_HEIGHT:
            raise ContractError(
                f"S02 requires raw {RAW_WIDTH}x{RAW_HEIGHT} clips: {clip.clip_id}"
            )
        expected_start += clip_frames
        validated.append(boundary)
    if expected_start != num_frames:
        raise ContractError("S02 S00 clip boundaries do not cover the frame timeline")
    return tuple(validated)


def _build_indices(
    detections: dict[str, np.ndarray],
    mapping: dict[str, np.ndarray],
    microtracklets: dict[str, np.ndarray],
    *,
    num_frames: int,
) -> tuple[np.ndarray, np.ndarray, dict[int, np.ndarray], np.ndarray, np.ndarray]:
    det_ids = np.asarray(detections["det_id"], dtype=np.int64)
    det_frames = np.asarray(detections["global_frame"], dtype=np.int64)
    det_times = np.asarray(detections["global_time_sec"], dtype=np.float64)
    det_sort = np.argsort(det_ids, kind="stable")
    sorted_ids = det_ids[det_sort]
    if np.any(np.diff(sorted_ids) == 0):
        raise ContractError("S02 detections.det_id is not unique")
    mapping_ids = np.asarray(mapping["det_id"], dtype=np.int64)
    locations = np.searchsorted(sorted_ids, mapping_ids)
    if np.any(locations == len(sorted_ids)):
        raise ContractError("S02 mapping contains unknown det_id")
    positions = det_sort[locations]
    if not np.array_equal(det_ids[positions], mapping_ids):
        raise ContractError("S02 mapping-to-detection join is not bijective")
    if len(np.unique(positions)) != len(positions) or len(positions) != len(det_ids):
        raise ContractError("S02 mapping is not a complete one-to-one detection mapping")

    micro_by_position = np.empty(len(det_ids), dtype=np.int64)
    order_by_position = np.empty(len(det_ids), dtype=np.int32)
    micro_by_position[positions] = np.asarray(mapping["micro_id"], dtype=np.int64)
    order_by_position[positions] = np.asarray(mapping["order_in_micro"], dtype=np.int32)

    path_order = np.lexsort((det_ids, order_by_position, micro_by_position))
    sorted_micro = micro_by_position[path_order]
    starts = np.flatnonzero(np.r_[True, sorted_micro[1:] != sorted_micro[:-1]])
    stops = np.r_[starts[1:], len(path_order)]
    paths: dict[int, np.ndarray] = {}
    for start, stop in zip(starts, stops, strict=True):
        path = path_order[int(start) : int(stop)]
        micro_id = int(micro_by_position[path[0]])
        if not np.array_equal(
            order_by_position[path], np.arange(len(path), dtype=np.int32)
        ):
            raise ContractError(f"microtrack {micro_id} order is not contiguous")
        if np.any(np.diff(det_frames[path]) <= 0):
            raise ContractError(f"microtrack {micro_id} is not time ordered")
        paths[micro_id] = path
    summary_id_array = np.asarray(microtracklets["micro_id"], dtype=np.int64)
    if len(np.unique(summary_id_array)) != len(summary_id_array):
        raise ContractError("S02 microtrack summary contains duplicate micro_id")
    summary_ids = set(map(int, summary_id_array))
    if set(paths) != summary_ids:
        raise ContractError("S02 mapping and microtrack summary ID sets differ")
    required_summary_columns = {
        "start_global_frame",
        "end_global_frame",
        "start_time_sec",
        "end_time_sec",
        "num_detections",
        "status",
    }
    if not required_summary_columns.issubset(microtracklets):
        raise ContractError("S02 microtrack summary columns are incomplete")
    summary_row = {
        int(micro_id): row for row, micro_id in enumerate(summary_id_array)
    }
    for micro_id, path in paths.items():
        row = summary_row[micro_id]
        if int(microtracklets["num_detections"][row]) != len(path):
            raise ContractError(
                f"microtrack {micro_id} summary num_detections differs from path"
            )
        status = str(microtracklets["status"][row])
        if status not in MICROTRACKLET_STATUSES:
            raise ContractError(f"microtrack {micro_id} has unknown status {status!r}")
        if int(microtracklets["start_global_frame"][row]) != int(det_frames[path[0]]):
            raise ContractError(f"microtrack {micro_id} summary start frame differs")
        if int(microtracklets["end_global_frame"][row]) != int(det_frames[path[-1]]):
            raise ContractError(f"microtrack {micro_id} summary end frame differs")
        if not math.isclose(
            float(microtracklets["start_time_sec"][row]),
            float(det_times[path[0]]),
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ContractError(f"microtrack {micro_id} summary start time differs")
        if not math.isclose(
            float(microtracklets["end_time_sec"][row]),
            float(det_times[path[-1]]),
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ContractError(f"microtrack {micro_id} summary end time differs")

    frame_order = np.lexsort((det_ids, det_frames))
    counts = np.bincount(det_frames[frame_order], minlength=num_frames)
    offsets = np.empty(num_frames + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(counts, out=offsets[1:])
    return micro_by_position, order_by_position, paths, frame_order, offsets


def _load_data(
    manifest_path: Path,
    ingest_dir: Path,
    microtrack_dir: Path,
    review_manifest_path: Path,
    config_path: Path,
    config: AppearanceConfig,
    *,
    logger: LogFn,
) -> LoadedAppearanceData:
    manifest, _ = load_manifest(manifest_path)
    s00_success, s00_fingerprints = _validated_stage_files(
        ingest_dir,
        expected_stage="S00",
        required_names=(
            "frames.parquet",
            "detections.parquet",
            "resolved_manifest.json",
            "ingest_report.json",
        ),
        progress_interval_sec=config.progress_interval_sec,
        logger=logger,
    )
    s01_success, s01_fingerprints = _validated_stage_files(
        microtrack_dir,
        expected_stage="S01",
        required_names=(
            "det_to_micro.parquet",
            "microtracklets.parquet",
            "microtrack_report.json",
        ),
        progress_interval_sec=config.progress_interval_sec,
        logger=logger,
    )
    video_paths = _validate_manifest_bijection(
        manifest, ingest_dir / "resolved_manifest.json"
    )
    counts = _validate_upstream_stats(
        s00_success,
        s01_success,
        manifest_num_clips=len(manifest),
    )
    sequence_id = manifest[0].sequence_id
    ingest_report = _read_json(ingest_dir / "ingest_report.json", "S00 ingest report")
    if not isinstance(ingest_report, dict) or ingest_report.get(
        "coordinate_system"
    ) != EXPECTED_COORDINATE_SYSTEM:
        raise ContractError("S02 S00 coordinate-system contract mismatch")
    if ingest_report.get("sequence_id") != sequence_id:
        raise ContractError("S02 S00 report sequence_id differs from manifest")
    for key, expected in {
        "num_clips": counts["num_clips"],
        "num_frames": counts["num_frames"],
        "num_input_boxes": counts["num_input_detections"],
        "num_valid_boxes": counts["num_valid_detections"],
    }.items():
        if ingest_report.get(key) != expected:
            raise ContractError(f"S02 S00 report/success stat mismatch: {key}")
    clip_boundaries = _validate_clip_boundaries(
        manifest,
        ingest_report.get("clip_boundaries"),
        num_frames=counts["num_frames"],
    )
    microtrack_report = _read_json(
        microtrack_dir / "microtrack_report.json", "S01 microtrack report"
    )
    if not isinstance(microtrack_report, dict) or microtrack_report.get(
        "input_coordinate_system"
    ) != EXPECTED_COORDINATE_SYSTEM:
        raise ContractError("S02 S01 coordinate-system contract mismatch")
    if microtrack_report.get("input_rows_modified") is not False:
        raise ContractError("S02 requires S01 to preserve its input rows")
    if microtrack_report.get("sequence_id") != sequence_id:
        raise ContractError("S02 S01 report sequence_id differs from manifest")
    microtrack_report_stats = microtrack_report.get("stats")
    if not isinstance(microtrack_report_stats, dict):
        raise ContractError("S02 S01 report lacks stats")
    for key, expected in {
        "num_frames": counts["num_frames"],
        "num_valid_detections": counts["num_valid_detections"],
        "num_microtracklets": counts["num_microtracklets"],
    }.items():
        if microtrack_report_stats.get(key) != expected:
            raise ContractError(f"S02 S01 report/success stat mismatch: {key}")

    frames = _parquet_columns(
        ingest_dir / "frames.parquet",
        expected_schema=FRAMES_SCHEMA,
        expected_rows=counts["num_frames"],
        columns=tuple(field.name for field in FRAMES_SCHEMA),
        label="S00 frames",
    )
    detection_columns = (
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
        "valid",
    )
    detections = _parquet_columns(
        ingest_dir / "detections.parquet",
        expected_schema=DETECTIONS_SCHEMA,
        expected_rows=counts["num_input_detections"],
        columns=detection_columns,
        filters=[("valid", "=", True)],
        label="S00 detections",
    )
    if len(detections["det_id"]) != counts["num_valid_detections"]:
        raise ContractError("S02 valid detection count mismatch")
    mapping = _parquet_columns(
        microtrack_dir / "det_to_micro.parquet",
        expected_schema=DET_TO_MICRO_SCHEMA,
        expected_rows=counts["num_valid_detections"],
        columns=("det_id", "micro_id", "order_in_micro"),
        label="S01 det_to_micro",
    )
    microtracklets = _parquet_columns(
        microtrack_dir / "microtracklets.parquet",
        expected_schema=MICROTRACKLETS_SCHEMA,
        expected_rows=counts["num_microtracklets"],
        columns=(
            "micro_id",
            "start_global_frame",
            "end_global_frame",
            "start_time_sec",
            "end_time_sec",
            "num_detections",
            "max_internal_center_jump",
            "local_purity_score",
            "status",
        ),
        label="S01 microtracklets",
    )

    global_frames = np.asarray(frames["global_frame"], dtype=np.int64)
    if not np.array_equal(global_frames, np.arange(counts["num_frames"])):
        raise ContractError("S02 frame timeline is not contiguous")
    if set(map(str, np.unique(frames["sequence_id"]))) != {sequence_id}:
        raise ContractError("S02 frame sequence_id mismatch")
    if not np.all(np.asarray(frames["width"]) == RAW_WIDTH) or not np.all(
        np.asarray(frames["height"]) == RAW_HEIGHT
    ):
        raise ContractError("S02 requires raw 3840x2160 frames")
    frame_clip_ids = np.asarray(frames["clip_id"], dtype=object)
    frame_local = np.asarray(frames["local_frame"], dtype=np.int64)
    for clip, boundary in zip(manifest, clip_boundaries, strict=True):
        start = int(boundary["start_global_frame"])
        stop = int(boundary["end_global_frame_inclusive"]) + 1
        if not np.all(frame_clip_ids[start:stop] == clip.clip_id):
            raise ContractError(f"S02 frame clip block differs: {clip.clip_id}")
        if not np.array_equal(
            frame_local[start:stop], np.arange(stop - start, dtype=np.int64)
        ):
            raise ContractError(f"S02 local frame timeline differs: {clip.clip_id}")
    det_frames = np.asarray(detections["global_frame"], dtype=np.int64)
    if np.any(det_frames < 0) or np.any(det_frames >= counts["num_frames"]):
        raise ContractError("S02 detection global_frame is outside frames")
    for name in ("sequence_id", "clip_id", "local_frame", "global_time_sec"):
        observed = np.asarray(detections[name])
        expected = np.asarray(frames[name])[det_frames]
        if not np.array_equal(observed, expected):
            raise ContractError(f"S02 detection-to-frame mismatch: {name}")
    boxes = np.column_stack(
        [np.asarray(detections[name], dtype=np.float64) for name in ("x1", "y1", "x2", "y2")]
    )
    if not np.all(np.isfinite(boxes)) or np.any(
        (boxes[:, 0] < 0.0)
        | (boxes[:, 1] < 0.0)
        | (boxes[:, 0] >= boxes[:, 2])
        | (boxes[:, 1] >= boxes[:, 3])
        | (boxes[:, 2] > RAW_WIDTH)
        | (boxes[:, 3] > RAW_HEIGHT)
    ):
        raise ContractError("S02 valid bbox contract failed")

    indices = _build_indices(
        detections,
        mapping,
        microtracklets,
        num_frames=counts["num_frames"],
    )
    fingerprint_paths = [manifest_path, review_manifest_path, config_path]
    review_success_path = review_manifest_path.with_name("_SUCCESS.json")
    if review_success_path.is_file():
        fingerprint_paths.append(review_success_path)
    for row in manifest:
        fingerprint_paths.append(row.video_path)
    for profile in config.model_profiles:
        fingerprint_paths.append(profile.checkpoint_path)
        model_dir = profile.checkpoint_path.parent
        fingerprint_paths.extend((model_dir / "config.json", model_dir / "README.md"))
        fingerprint_paths.append(
            model_dir
            / ".cache"
            / "huggingface"
            / "download"
            / f"{profile.checkpoint_path.name}.metadata"
        )
    unique_paths = sorted({path.resolve() for path in fingerprint_paths}, key=str)
    extra_fingerprints = [
        _fingerprint(
            path,
            progress_interval_sec=config.progress_interval_sec,
            logger=logger,
        )
        for path in unique_paths
    ]
    input_fingerprints = sorted(
        [*s00_fingerprints, *s01_fingerprints, *extra_fingerprints],
        key=lambda item: item["path"],
    )
    return LoadedAppearanceData(
        sequence_id=sequence_id,
        frames=frames,
        detections=detections,
        microtracklets=microtracklets,
        micro_ids_by_detection=indices[0],
        order_in_micro_by_detection=indices[1],
        paths=indices[2],
        positions_by_frame=indices[3],
        frame_offsets=indices[4],
        video_paths=video_paths,
        input_fingerprints=tuple(input_fingerprints),
    )


def _case_contract(case: Mapping[str, Any]) -> dict[str, Any]:
    keys = ("case_id", "case_kind", "micro_id", "reasons", "events", "anchor", "microtrack")
    return {key: case.get(key) for key in keys}


def _load_review_exclusions(
    review_manifest_path: Path, data: LoadedAppearanceData
) -> ExclusionResult:
    manifest = _read_json(review_manifest_path, "S01 review manifest")
    if not isinstance(manifest, dict):
        raise ContractError("S01 review manifest must be a JSON object")
    if manifest.get("schema_version") != REVIEW_SCHEMA_VERSION:
        raise ContractError("S01 review manifest schema_version mismatch")
    if manifest.get("coordinate_system") != EXPECTED_COORDINATE_SYSTEM:
        raise ContractError("S01 review coordinate system mismatch")
    contract = manifest.get("output_video_contract")
    if not isinstance(contract, dict):
        raise ContractError("S01 review manifest lacks output_video_contract")
    try:
        radius = int(contract.get("true_event_highlight_radius_frames", -1))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ContractError("review event radius must be an integer") from exc
    if radius != DEFAULT_EVENT_RADIUS_FRAMES:
        raise ContractError("fixed S02 requires review event radius=15 frames")
    if contract.get("synthetic_long_and_quality_anchors_are_not_events") is not True:
        raise ContractError("review synthetic-anchor event policy mismatch")

    cases = manifest.get("cases")
    selection_plan = manifest.get("selection_plan")
    planned_cases = selection_plan.get("cases") if isinstance(selection_plan, dict) else None
    if not isinstance(cases, list) or not isinstance(planned_cases, list):
        raise ContractError("review manifest lacks rendered and planned cases")
    rendered_by_id: dict[str, Mapping[str, Any]] = {}
    for case in cases:
        if not isinstance(case, dict):
            raise ContractError("review rendered case must be an object")
        case_id = str(case.get("case_id", ""))
        if not case_id or case_id in rendered_by_id:
            raise ContractError("review rendered case_id is blank or duplicated")
        rendered_by_id[case_id] = case
    planned_by_id: dict[str, Mapping[str, Any]] = {}
    for case in planned_cases:
        if not isinstance(case, dict):
            raise ContractError("review planned case must be an object")
        case_id = str(case.get("case_id", ""))
        if not case_id or case_id in planned_by_id:
            raise ContractError("review planned case_id is blank or duplicated")
        planned_by_id[case_id] = case
    if set(rendered_by_id) != set(planned_by_id):
        raise ContractError("review rendered/planned case ID sets differ")
    for case_id in rendered_by_id:
        if _case_contract(rendered_by_id[case_id]) != _case_contract(
            planned_by_id[case_id]
        ):
            raise ContractError(f"review rendered/planned case differs: {case_id}")

    success_path = review_manifest_path.with_name("_SUCCESS.json")
    if success_path.is_file():
        success = _read_json(success_path, "S01 review success marker")
        if not isinstance(success, dict):
            raise ContractError("S01 review success marker must be an object")
        manifest_hash = hashlib.sha256(review_manifest_path.read_bytes()).hexdigest()
        if success.get("manifest_sha256") != manifest_hash:
            raise ContractError("review manifest differs from its success marker")

    det_frames = np.asarray(data.detections["global_frame"], dtype=np.int64)
    det_times = np.asarray(data.detections["global_time_sec"], dtype=np.float64)
    det_ids = np.asarray(data.detections["det_id"], dtype=np.int64)
    det_clips = np.asarray(data.detections["clip_id"], dtype=object)
    det_position_by_id = {int(det_id): row for row, det_id in enumerate(det_ids)}
    if len(det_position_by_id) != len(det_ids):
        raise ContractError("S02 review validation requires unique det_id values")
    summary_ids = np.asarray(data.microtracklets["micro_id"], dtype=np.int64)
    summary_row_by_id = {
        int(micro_id): row for row, micro_id in enumerate(summary_ids)
    }
    details: dict[int, dict[str, set[str] | set[int]]] = {}
    event_keys: set[tuple[int, str, int]] = set()
    for case_id in sorted(rendered_by_id):
        case = rendered_by_id[case_id]
        try:
            micro_id = int(case.get("micro_id"))
        except (TypeError, ValueError, OverflowError) as exc:
            raise ContractError(f"review case {case_id} has invalid micro_id") from exc
        path = data.paths.get(micro_id)
        if path is None:
            raise ContractError(f"review case references unknown micro_id: {micro_id}")
        summary_row = summary_row_by_id.get(micro_id)
        case_summary = case.get("microtrack")
        if summary_row is None or not isinstance(case_summary, dict):
            raise ContractError(f"review case {case_id} lacks microtrack summary")
        if int(case_summary.get("num_detections", -1)) != int(
            data.microtracklets["num_detections"][summary_row]
        ):
            raise ContractError(
                f"review case {case_id} num_detections differs from S01 summary"
            )
        if str(case_summary.get("status", "")) != str(
            data.microtracklets["status"][summary_row]
        ):
            raise ContractError(
                f"review case {case_id} status differs from S01 summary"
            )
        events = case.get("events")
        if not isinstance(events, list):
            raise ContractError(f"review case {case_id} events must be a list")
        path_frames = det_frames[path]
        for event in events:
            if not isinstance(event, dict):
                raise ContractError(f"review case {case_id} event must be an object")
            reason = str(event.get("reason", ""))
            if reason not in TRUE_EVENT_REASONS:
                continue
            try:
                trigger = int(event["global_frame"])
            except (KeyError, TypeError, ValueError, OverflowError) as exc:
                raise ContractError(
                    f"review true event lacks integer global_frame: {case_id}"
                ) from exc
            key = (micro_id, reason, trigger)
            if key in event_keys:
                raise ContractError(f"duplicate review true event: {key}")
            event_keys.add(key)
            try:
                src_det_id = int(event["src_det_id"])
                dst_det_id = int(event["dst_det_id"])
                event_time = float(event["global_time_sec"])
            except (KeyError, TypeError, ValueError, OverflowError) as exc:
                raise ContractError(
                    f"review true event lacks edge provenance: {case_id}/{trigger}"
                ) from exc
            src_position = det_position_by_id.get(src_det_id)
            dst_position = det_position_by_id.get(dst_det_id)
            if src_position is None or dst_position is None:
                raise ContractError(
                    f"review true event references unknown edge detection: "
                    f"{case_id}/{trigger}"
                )
            if (
                int(data.micro_ids_by_detection[src_position]) != micro_id
                or int(data.micro_ids_by_detection[dst_position]) != micro_id
                or int(data.order_in_micro_by_detection[dst_position])
                != int(data.order_in_micro_by_detection[src_position]) + 1
            ):
                raise ContractError(
                    f"review true event edge is not consecutive on target micro: "
                    f"{case_id}/{trigger}"
                )
            if int(det_frames[dst_position]) != trigger or not math.isclose(
                float(det_times[dst_position]),
                event_time,
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                raise ContractError(
                    f"review true event trigger differs from dst detection: "
                    f"{case_id}/{trigger}"
                )
            event_clip = event.get("clip_id")
            if event_clip is not None and str(event_clip) != str(det_clips[dst_position]):
                raise ContractError(
                    f"review true event clip differs from dst detection: "
                    f"{case_id}/{trigger}"
                )
            left = int(np.searchsorted(path_frames, trigger - radius, side="left"))
            right = int(np.searchsorted(path_frames, trigger + radius, side="right"))
            if left == right:
                raise ContractError(
                    f"review true event maps to no target detection: {case_id}/{trigger}"
                )
            for position in map(int, path[left:right]):
                detail = details.setdefault(
                    position,
                    {"reasons": set(), "case_ids": set(), "trigger_frames": set()},
                )
                reasons = cast(set[str], detail["reasons"])
                case_ids = cast(set[str], detail["case_ids"])
                trigger_frames = cast(set[int], detail["trigger_frames"])
                reasons.add(reason)
                case_ids.add(case_id)
                trigger_frames.add(trigger)
    excluded = np.zeros(len(det_frames), dtype=np.bool_)
    reasons_out = [""] * len(det_frames)
    case_ids_out = [""] * len(det_frames)
    triggers_out = [""] * len(det_frames)
    for position, detail in details.items():
        excluded[position] = True
        reasons_out[position] = ";".join(sorted(map(str, detail["reasons"])))
        case_ids_out[position] = ";".join(sorted(map(str, detail["case_ids"])))
        triggers_out[position] = ";".join(
            map(str, sorted(map(int, detail["trigger_frames"])))
        )
    return ExclusionResult(
        excluded=excluded,
        reasons=tuple(reasons_out),
        case_ids=tuple(case_ids_out),
        trigger_frames=tuple(triggers_out),
        num_true_events=len(event_keys),
    )


def _boundary_distances(data: LoadedAppearanceData) -> np.ndarray:
    x1 = np.asarray(data.detections["x1"], dtype=np.float64)
    y1 = np.asarray(data.detections["y1"], dtype=np.float64)
    x2 = np.asarray(data.detections["x2"], dtype=np.float64)
    y2 = np.asarray(data.detections["y2"], dtype=np.float64)
    distance = np.minimum.reduce((x1, y1, RAW_WIDTH - x2, RAW_HEIGHT - y2))
    short_side = np.minimum(x2 - x1, y2 - y1)
    scale = np.maximum(0.5 * short_side, np.finfo(np.float64).eps)
    return np.clip(distance / scale, 0.0, 1.0).astype(np.float32)


def _resolve_encoder_profiles(
    config: AppearanceConfig,
    input_fingerprints: Sequence[Mapping[str, Any]],
) -> tuple[EncoderProfile, ...]:
    runtime_profiles = {profile.name: profile for profile in production_encoder_profiles()}
    config_names = tuple(profile.name for profile in config.model_profiles)
    if config_names != (LOCKED_ENCODER_PROFILE,):
        raise ContractError("S02 config must contain only the locked encoder profile")
    if LOCKED_ENCODER_PROFILE not in runtime_profiles:
        raise ContractError("S02 locked runtime encoder profile is unavailable")
    fingerprint_by_path = {
        str(Path(item["path"]).resolve()): item for item in input_fingerprints
    }
    result: list[EncoderProfile] = []
    for configured in config.model_profiles:
        runtime = runtime_profiles[configured.name]
        checks = {
            "architecture": (configured.architecture, runtime.timm_model_name),
            "checkpoint_path": (
                configured.checkpoint_path.resolve(),
                runtime.checkpoint_path.resolve(),
            ),
            "checkpoint_filter": (
                configured.checkpoint_filter,
                runtime.checkpoint_filter,
            ),
            "input_size": (configured.input_size, runtime.input_size),
            "resize_mode": (configured.resize_mode, runtime.resize_mode),
            "crop_pct": (configured.crop_pct, runtime.crop_pct),
            "interpolation": (configured.interpolation, runtime.interpolation),
        }
        for label, (observed, expected) in checks.items():
            if observed != expected:
                raise ContractError(
                    f"encoder profile {configured.name} {label} differs from runtime"
                )
        if configured.normalization_profile == "half":
            expected_mean = expected_std = (0.5, 0.5, 0.5)
        elif configured.normalization_profile == "imagenet":
            expected_mean = (0.485, 0.456, 0.406)
            expected_std = (0.229, 0.224, 0.225)
        else:
            raise ContractError(
                f"unsupported normalization profile: {configured.normalization_profile}"
            )
        if runtime.mean != expected_mean or runtime.std != expected_std:
            raise ContractError(
                f"encoder profile {configured.name} normalization differs from runtime"
            )
        fingerprint = fingerprint_by_path.get(str(runtime.checkpoint_path.resolve()))
        if fingerprint is None:
            raise ContractError(
                f"encoder checkpoint was not fingerprinted: {runtime.checkpoint_path}"
            )
        if fingerprint["sha256"] != configured.expected_weight_sha256:
            raise ContractError(
                f"encoder checkpoint hash mismatch: {runtime.checkpoint_path}"
            )
        metadata_path = (
            runtime.checkpoint_path.parent
            / ".cache"
            / "huggingface"
            / "download"
            / f"{runtime.checkpoint_path.name}.metadata"
        )
        if not metadata_path.is_file():
            raise ContractError(f"encoder revision metadata is missing: {metadata_path}")
        try:
            revision = metadata_path.read_text(encoding="utf-8").splitlines()[0]
        except (OSError, IndexError) as exc:
            raise ContractError(
                f"cannot read encoder revision metadata: {metadata_path}"
            ) from exc
        if revision != configured.expected_revision:
            raise ContractError(
                f"encoder revision mismatch for {configured.name}: {revision}"
            )
        result.append(runtime)
    return tuple(result)


def _selection_reasons(
    data: LoadedAppearanceData,
    selected: np.ndarray,
    excluded: np.ndarray,
) -> dict[int, str]:
    selected_set = set(map(int, selected))
    result: dict[int, str] = {}
    for micro_id in sorted(data.paths):
        path = data.paths[micro_id]
        eligible = path[~excluded[path]]
        selected_path = [int(position) for position in eligible if int(position) in selected_set]
        if not selected_path:
            continue
        eligible_order = {int(position): index for index, position in enumerate(eligible)}
        for position in selected_path:
            order = eligible_order[position]
            reasons: list[str] = []
            if order < 3:
                reasons.append("head")
            if order >= len(eligible) - 3:
                reasons.append("tail")
            if not reasons:
                reasons.append("periodic")
            result[position] = ";".join(reasons)
    if set(result) != selected_set:
        raise ContractError("S02 could not classify every representative selection")
    return result


def _selected_positions(
    data: LoadedAppearanceData,
    exclusions: ExclusionResult,
    other_iou: np.ndarray,
    config: AppearanceConfig,
) -> np.ndarray:
    selected = select_candidate_pool_indices(
        data.micro_ids_by_detection,
        np.asarray(data.detections["global_time_sec"], dtype=np.float64),
        np.asarray(data.detections["global_frame"], dtype=np.int64),
        other_iou,
        exclusions.excluded,
        period_sec=config.candidate_pool_period_sec,
        max_samples=config.candidate_pool_max_samples_per_microtrack,
        endpoint_samples=config.candidate_pool_endpoint_samples,
        preferred_max_iou=config.max_other_bbox_iou_preferred,
    )
    if len(selected) == 0:
        raise ContractError("S02 candidate-pool selection produced no detections")
    if np.any(exclusions.excluded[selected]):
        raise ContractError("S02 candidate pool contains an excluded detection")
    return selected


def _finalize_representative_samples(
    records: Sequence[CropRecord],
    embeddings: np.ndarray,
    config: AppearanceConfig,
) -> tuple[list[CropRecord], np.ndarray]:
    """Apply the public 3/3, 0.75-second, 24-sample contract after quality."""

    if len(records) != len(embeddings):
        raise ContractError("S02 quality candidate/embedding row count mismatch")
    if not records:
        raise ContractError("S02 found no high-quality candidate crops")
    micro_ids = np.asarray([row.micro_id for row in records], dtype=np.int64)
    times = np.asarray([row.global_time_sec for row in records], dtype=np.float64)
    frames = np.asarray([row.global_frame for row in records], dtype=np.int64)
    overlaps = np.asarray(
        [row.other_bbox_max_iou for row in records], dtype=np.float64
    )
    selected = select_representative_indices(
        micro_ids,
        times,
        frames,
        overlaps,
        np.zeros(len(records), dtype=np.bool_),
        period_sec=config.representative_period_sec,
        max_samples=config.max_samples_per_microtrack,
        endpoint_samples=config.head_samples,
        preferred_max_iou=config.max_other_bbox_iou_preferred,
    )
    if not len(selected):
        raise ContractError("S02 final representative selection is empty")

    selected_set = set(map(int, selected))
    reasons: dict[int, str] = {}
    for micro_id in sorted(set(map(int, micro_ids))):
        group = np.flatnonzero(micro_ids == micro_id)
        order = np.lexsort(
            (
                np.asarray([records[int(i)].det_id for i in group], dtype=np.int64),
                frames[group],
                times[group],
            )
        )
        ordered = group[order]
        selected_group = [int(index) for index in ordered if int(index) in selected_set]
        if not selected_group:
            continue
        position_by_index = {
            int(index): offset for offset, index in enumerate(map(int, ordered))
        }
        for index in selected_group:
            offset = position_by_index[index]
            labels: list[str] = []
            if offset < config.head_samples:
                labels.append("head")
            if offset >= len(ordered) - config.tail_samples:
                labels.append("tail")
            if not labels:
                labels.append("periodic")
            reasons[index] = ";".join(labels)
    if set(reasons) != selected_set:
        raise ContractError("S02 could not classify every final representative sample")

    final_records = [
        replace(records[int(index)], selection_reason=reasons[int(index)])
        for index in selected
    ]
    final_embeddings = np.asarray(embeddings[selected], dtype=np.float32)
    if len(final_records) != len(final_embeddings):
        raise ContractError("S02 final representative row alignment failed")
    return final_records, final_embeddings


def _flush_crop_batch(
    raw_crops: list[np.ndarray],
    masked_crops: list[np.ndarray],
    metadata: list[dict[str, Any]],
    session: _EmbeddingSession,
    config: AppearanceConfig,
) -> int:
    if not metadata:
        return 0
    blur_size = config.quality_blur_measurement_size
    blur = np.asarray(
        [
            laplacian_blur_score(
                np.ascontiguousarray(
                    cv2.resize(
                        crop,
                        (blur_size, blur_size),
                        interpolation=cv2.INTER_AREA,
                    )
                )
            )
            for crop in raw_crops
        ],
        dtype=np.float64,
    )
    other_iou = np.asarray([item["other_iou"] for item in metadata], dtype=np.float64)
    clipped = np.asarray([item["clipped"] for item in metadata], dtype=np.float64)
    area = np.asarray([item["area_percentile"] for item in metadata], dtype=np.float64)
    boundary = np.asarray([item["boundary"] for item in metadata], dtype=np.float64)
    quality = combine_crop_quality(
        other_iou,
        clipped,
        area,
        blur,
        boundary,
        weights=tuple(config.quality_weights.values()),
        blur_scale=config.quality_blur_scale,
    )
    keep = np.flatnonzero(quality >= config.min_crop_quality)
    kept_raw: list[np.ndarray] = []
    kept_masked: list[np.ndarray] = []
    records: list[CropRecord] = []
    for index in map(int, keep):
        item = metadata[index]
        kept_raw.append(raw_crops[index])
        kept_masked.append(masked_crops[index])
        records.append(
            CropRecord(
                detection_position=int(item["position"]),
                micro_id=int(item["micro_id"]),
                det_id=int(item["det_id"]),
                clip_id=str(item["clip_id"]),
                global_frame=int(item["global_frame"]),
                global_time_sec=float(item["global_time_sec"]),
                crop_quality=float(quality[index]),
                other_bbox_max_iou=float(other_iou[index]),
                clipped_fraction=float(clipped[index]),
                bbox_area_percentile=float(area[index]),
                blur_score=float(blur[index]),
                distance_to_image_boundary=float(boundary[index]),
                selection_reason=str(item["selection_reason"]),
            )
        )
    if records:
        session.consume(kept_raw, kept_masked, records)
    rejected = len(metadata) - len(records)
    raw_crops.clear()
    masked_crops.clear()
    metadata.clear()
    return rejected


def _decode_selected_crops(
    data: LoadedAppearanceData,
    selected: np.ndarray,
    other_iou: np.ndarray,
    area_percentile: np.ndarray,
    config: AppearanceConfig,
    session: _EmbeddingSession,
    *,
    logger: LogFn,
) -> int:
    clip_ids = np.asarray(data.detections["clip_id"], dtype=object)
    local_frames = np.asarray(data.detections["local_frame"], dtype=np.int64)
    global_frames = np.asarray(data.detections["global_frame"], dtype=np.int64)
    times = np.asarray(data.detections["global_time_sec"], dtype=np.float64)
    det_ids = np.asarray(data.detections["det_id"], dtype=np.int64)
    boxes = np.column_stack(
        [
            np.asarray(data.detections[name], dtype=np.float64)
            for name in ("x1", "y1", "x2", "y2")
        ]
    )
    rejected = 0
    raw_batch: list[np.ndarray] = []
    masked_batch: list[np.ndarray] = []
    metadata_batch: list[dict[str, Any]] = []
    decoded_total = 0
    last_report = time.monotonic()

    for clip_id, video_path in data.video_paths.items():
        clip_positions = selected[np.asarray(clip_ids[selected] == clip_id)]
        if len(clip_positions) == 0:
            raise ContractError(f"S02 selected no crop from required clip {clip_id}")
        by_local: dict[int, list[int]] = {}
        for position in map(int, clip_positions):
            by_local.setdefault(int(local_frames[position]), []).append(position)
        last_local = max(by_local)
        capture = open_raw_video_capture(video_path)
        try:
            for local_frame in range(last_local + 1):
                ok, frame_bgr = capture.read()
                if not ok or frame_bgr is None:
                    raise ContractError(
                        f"S02 cannot decode {clip_id} local frame {local_frame}"
                    )
                if frame_bgr.shape != (RAW_HEIGHT, RAW_WIDTH, 3) or frame_bgr.dtype != np.uint8:
                    raise ContractError(
                        f"S02 decoded unexpected frame geometry for {clip_id}: "
                        f"{frame_bgr.shape}/{frame_bgr.dtype}"
                    )
                decoded_total += 1
                for position in sorted(
                    by_local.get(local_frame, ()),
                    key=lambda item: (int(data.micro_ids_by_detection[item]), int(det_ids[item])),
                ):
                    crop_bgr, clipped, boundary = crop_with_padding(
                        frame_bgr,
                        boxes[position],
                        config.bbox_padding_ratio,
                    )
                    crop_rgb = np.ascontiguousarray(crop_bgr[:, :, ::-1])
                    raw_batch.append(crop_rgb)
                    masked_batch.append(soft_border_mask(crop_rgb))
                    metadata_batch.append(
                        {
                            "position": position,
                            "micro_id": int(data.micro_ids_by_detection[position]),
                            "det_id": int(det_ids[position]),
                            "clip_id": clip_id,
                            "global_frame": int(global_frames[position]),
                            "global_time_sec": float(times[position]),
                            "other_iou": float(other_iou[position]),
                            "clipped": float(clipped),
                            "area_percentile": float(area_percentile[position]),
                            "boundary": float(boundary),
                            "selection_reason": "candidate_pool",
                        }
                    )
                    if len(metadata_batch) >= config.batch_size:
                        rejected += _flush_crop_batch(
                            raw_batch,
                            masked_batch,
                            metadata_batch,
                            session,
                            config,
                        )
                now = time.monotonic()
                if now - last_report >= config.progress_interval_sec:
                    logger(
                        f"[s02] decode {clip_id}: {local_frame + 1:,}/{last_local + 1:,} "
                        f"frames; embedded samples={len(session.records):,}"
                    )
                    last_report = now
        finally:
            capture.release()
        logger(
            f"[s02] decoded {clip_id} once in source order through frame "
            f"{last_local:,}"
        )
    rejected += _flush_crop_batch(
        raw_batch,
        masked_batch,
        metadata_batch,
        session,
        config,
    )
    if decoded_total <= 0:
        raise ContractError("S02 decoded no video frames")
    return rejected


def _table_from_pydict(payload: Mapping[str, Any], schema: pa.Schema) -> pa.Table:
    if set(payload) != set(schema.names):
        raise ContractError(
            f"S02 table columns differ from schema: {sorted(set(payload) ^ set(schema.names))}"
        )
    try:
        table = pa.Table.from_pydict(dict(payload), schema=schema)
    except (pa.ArrowException, TypeError, ValueError) as exc:
        raise ContractError(f"cannot construct S02 Arrow table: {exc}") from exc
    if not table.schema.equals(schema, check_metadata=False):
        raise ContractError("constructed S02 Arrow table schema mismatch")
    return table


def _exclusion_table(
    data: LoadedAppearanceData, exclusions: ExclusionResult
) -> pa.Table:
    positions = np.flatnonzero(exclusions.excluded)
    det_ids = np.asarray(data.detections["det_id"], dtype=np.int64)
    frames = np.asarray(data.detections["global_frame"], dtype=np.int64)
    order = np.lexsort(
        (det_ids[positions], data.micro_ids_by_detection[positions], frames[positions])
    )
    positions = positions[order]

    def split_text(value: str) -> list[str]:
        return value.split(";") if value else []

    def split_frames(value: str) -> list[int]:
        return [int(item) for item in value.split(";")] if value else []

    return _table_from_pydict(
        {
            "det_id": det_ids[positions],
            "micro_id": data.micro_ids_by_detection[positions],
            "global_frame": frames[positions],
            "appearance_excluded": np.ones(len(positions), dtype=np.bool_),
            "crop_quality": np.zeros(len(positions), dtype=np.float32),
            "exclusion_reasons": [
                split_text(exclusions.reasons[int(position)]) for position in positions
            ],
            "review_case_ids": [
                split_text(exclusions.case_ids[int(position)]) for position in positions
            ],
            "trigger_global_frames": [
                split_frames(exclusions.trigger_frames[int(position)])
                for position in positions
            ],
        },
        APPEARANCE_EXCLUSIONS_SCHEMA,
    )


def _sample_table(
    data: LoadedAppearanceData,
    records: Sequence[CropRecord],
    prototype_inlier: np.ndarray,
    prototype_outlier: np.ndarray,
) -> pa.Table:
    clip_ids = np.asarray(data.detections["clip_id"], dtype=object)
    count = len(records)
    prototype_inlier = np.asarray(prototype_inlier, dtype=np.bool_)
    prototype_outlier = np.asarray(prototype_outlier, dtype=np.bool_)
    if (
        prototype_inlier.shape != (count,)
        or prototype_outlier.shape != (count,)
        or np.any(prototype_inlier & prototype_outlier)
        or np.any(~(prototype_inlier | prototype_outlier))
    ):
        raise ContractError("S02 per-sample prototype masks are inconsistent")
    return _table_from_pydict(
        {
            "sample_id": np.arange(count, dtype=np.int64),
            "micro_id": np.asarray([row.micro_id for row in records], dtype=np.int64),
            "det_id": np.asarray([row.det_id for row in records], dtype=np.int64),
            "clip_id": [str(clip_ids[row.detection_position]) for row in records],
            "global_frame": np.asarray(
                [row.global_frame for row in records], dtype=np.int64
            ),
            "global_time_sec": np.asarray(
                [row.global_time_sec for row in records], dtype=np.float64
            ),
            "crop_quality": np.asarray(
                [row.crop_quality for row in records], dtype=np.float32
            ),
            "other_bbox_max_iou": np.asarray(
                [row.other_bbox_max_iou for row in records], dtype=np.float32
            ),
            "clipped_fraction": np.asarray(
                [row.clipped_fraction for row in records], dtype=np.float32
            ),
            "bbox_area_percentile": np.asarray(
                [row.bbox_area_percentile for row in records], dtype=np.float32
            ),
            "blur_score": np.asarray(
                [row.blur_score for row in records], dtype=np.float32
            ),
            "distance_to_image_boundary": np.asarray(
                [row.distance_to_image_boundary for row in records], dtype=np.float32
            ),
            "selection_reason": [row.selection_reason for row in records],
            "prototype_inlier": prototype_inlier,
            "prototype_outlier": prototype_outlier,
            "embedding_row": np.arange(count, dtype=np.int64),
        },
        APPEARANCE_SAMPLES_SCHEMA,
    )


def _micro_context_table(
    data: LoadedAppearanceData,
    other_iou: np.ndarray,
    boundary_distance: np.ndarray,
    all_micro_ids: np.ndarray,
) -> pa.Table:
    det_ids = np.asarray(data.detections["det_id"], dtype=np.int64)
    clip_ids = np.asarray(data.detections["clip_id"], dtype=object)
    frames = np.asarray(data.detections["global_frame"], dtype=np.int64)
    starts = np.asarray([data.paths[int(micro_id)][0] for micro_id in all_micro_ids])
    ends = np.asarray([data.paths[int(micro_id)][-1] for micro_id in all_micro_ids])
    return _table_from_pydict(
        {
            "micro_id": all_micro_ids,
            "start_det_id": det_ids[starts],
            "end_det_id": det_ids[ends],
            "start_clip_id": [str(value) for value in clip_ids[starts]],
            "end_clip_id": [str(value) for value in clip_ids[ends]],
            "start_max_other_iou": other_iou[starts].astype(np.float32),
            "end_max_other_iou": other_iou[ends].astype(np.float32),
            "start_boundary_distance": boundary_distance[starts].astype(np.float32),
            "end_boundary_distance": boundary_distance[ends].astype(np.float32),
            "start_global_frame": frames[starts],
            "end_global_frame": frames[ends],
        },
        MICRO_CONTEXT_SCHEMA,
    )


def _micro_appearance_table(
    prototypes: Any,
    *,
    selected_encoder: str,
) -> pa.Table:
    count = len(prototypes.micro_ids)
    return _table_from_pydict(
        {
            "micro_id": prototypes.micro_ids,
            "prototype_row": np.arange(count, dtype=np.int64),
            "num_appearance_samples": prototypes.num_samples.astype(np.int16),
            "appearance_usable": np.any(
                prototypes.prototype_mask, axis=1
            ).astype(np.bool_),
            "appearance_quality": prototypes.appearance_quality.astype(np.float32),
            "internal_cosine_p10": prototypes.internal_cosine_p10.astype(np.float32),
            "internal_cosine_p50": prototypes.internal_cosine_p50.astype(np.float32),
            "internal_cosine_min": prototypes.internal_cosine_min.astype(np.float32),
            "appearance_outlier_count": prototypes.outlier_count.astype(np.int16),
            "selected_encoder": [selected_encoder] * count,
        },
        MICRO_APPEARANCE_SCHEMA,
    )


def _write_parquet(path: Path, table: pa.Table, compression: str) -> None:
    try:
        pq.write_table(table, path, compression=compression, version="2.6")
    except (OSError, pa.ArrowException) as exc:
        raise ContractError(f"cannot write S02 Parquet {path}: {exc}") from exc


def _write_npy(path: Path, array: np.ndarray) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("wb") as handle:
            np.save(handle, array, allow_pickle=False)
        os.replace(temporary, path)
    except OSError as exc:
        temporary.unlink(missing_ok=True)
        raise ContractError(f"cannot atomically write S02 array {path}: {exc}") from exc


def _verify_inputs_unchanged(fingerprints: Sequence[Mapping[str, Any]]) -> None:
    for fingerprint in fingerprints:
        path = Path(str(fingerprint["path"]))
        current = path.stat()
        if (
            current.st_size != fingerprint["size_bytes"]
            or current.st_mtime_ns != fingerprint["mtime_ns"]
        ):
            raise ContractError(f"input changed during S02: {path}")


def _expected_artifact_names(config: AppearanceConfig) -> set[str]:
    artifacts = config.artifacts
    return {
        artifacts.appearance_samples,
        artifacts.sample_embeddings,
        artifacts.appearance_exclusions,
        artifacts.micro_prototypes,
        artifacts.micro_prototype_mask,
        artifacts.micro_appearance,
        artifacts.micro_context,
        artifacts.encoder_choice,
        artifacts.encoder_benchmark,
        artifacts.effective_config,
        artifacts.appearance_report,
    }


def _validate_output_files(
    output_dir: Path,
    config: AppearanceConfig,
    stats: Mapping[str, Any],
) -> None:
    artifacts = config.artifacts
    expected_microtracklets = _required_stat(
        stats, "S02", "num_microtracklets", minimum=1
    )
    choice = _read_json(output_dir / artifacts.encoder_choice, "S02 encoder choice")
    benchmark = _read_json(
        output_dir / artifacts.encoder_benchmark, "S02 encoder benchmark"
    )
    configured_profile = config.model_profiles[0]
    if (
        not isinstance(choice, dict)
        or choice.get("selection_mode") != LOCKED_SELECTION_MODE
        or choice.get("selected_profile") != LOCKED_ENCODER_PROFILE
        or choice.get("embedding_dim") != int(stats["embedding_dim"])
        or choice.get("checkpoint_sha256")
        != configured_profile.expected_weight_sha256
        or choice.get("no_runtime_fallback") is not True
        or not isinstance(choice.get("selection_metrics"), dict)
    ):
        raise ContractError("completed S02 locked encoder choice contract differs")
    if (
        not isinstance(benchmark, dict)
        or benchmark.get("selection_mode") != LOCKED_SELECTION_MODE
        or benchmark.get("selected_profile") != LOCKED_ENCODER_PROFILE
        or benchmark.get("evaluation_mode") != LOCKED_SELECTION_MODE
        or benchmark.get("num_embedding_rows") != 0
        or benchmark.get("num_production_embedding_rows")
        != int(stats["num_high_quality_candidates"])
        or benchmark.get("profiles") != {LOCKED_ENCODER_PROFILE: {}}
    ):
        raise ContractError("completed S02 locked encoder benchmark contract differs")
    parquet_contracts = (
        (
            artifacts.appearance_samples,
            APPEARANCE_SAMPLES_SCHEMA,
            int(stats["num_appearance_samples"]),
        ),
        (
            artifacts.appearance_exclusions,
            APPEARANCE_EXCLUSIONS_SCHEMA,
            int(stats["num_appearance_exclusions"]),
        ),
        (
            artifacts.micro_appearance,
            MICRO_APPEARANCE_SCHEMA,
            expected_microtracklets,
        ),
        (
            artifacts.micro_context,
            MICRO_CONTEXT_SCHEMA,
            expected_microtracklets,
        ),
    )
    for name, schema, rows in parquet_contracts:
        path = output_dir / name
        try:
            parquet = pq.ParquetFile(path)
        except (OSError, pa.ArrowException) as exc:
            raise ContractError(f"cannot validate completed S02 Parquet {name}: {exc}") from exc
        if not parquet.schema_arrow.equals(schema, check_metadata=False):
            raise ContractError(f"completed S02 schema mismatch: {name}")
        if parquet.metadata.num_rows != rows:
            raise ContractError(f"completed S02 row count mismatch: {name}")

    try:
        embeddings = np.load(output_dir / artifacts.sample_embeddings, mmap_mode="r")
        prototypes = np.load(output_dir / artifacts.micro_prototypes, mmap_mode="r")
        mask = np.load(output_dir / artifacts.micro_prototype_mask, mmap_mode="r")
    except (OSError, ValueError) as exc:
        raise ContractError(f"cannot load completed S02 arrays: {exc}") from exc
    expected_samples = int(stats["num_appearance_samples"])
    embedding_dim = int(stats["embedding_dim"])
    if embeddings.shape != (expected_samples, embedding_dim) or embeddings.dtype != np.float16:
        raise ContractError("completed S02 sample embedding shape/dtype mismatch")
    if prototypes.shape != (
        expected_microtracklets,
        config.prototypes_per_tracklet,
        embedding_dim,
    ) or prototypes.dtype != np.float16:
        raise ContractError("completed S02 prototype shape/dtype mismatch")
    if mask.shape != (
        expected_microtracklets,
        config.prototypes_per_tracklet,
    ) or mask.dtype != np.bool_:
        raise ContractError("completed S02 prototype mask shape/dtype mismatch")
    if not np.isfinite(embeddings).all() or not np.isfinite(prototypes).all():
        raise ContractError("completed S02 arrays contain non-finite values")
    if len(embeddings):
        norms = np.linalg.norm(np.asarray(embeddings, dtype=np.float32), axis=1)
        if not np.allclose(norms, 1.0, rtol=0.0, atol=2e-3):
            raise ContractError("completed S02 sample embeddings are not normalized")
    if np.any(np.asarray(prototypes)[~np.asarray(mask)] != 0.0):
        raise ContractError("completed S02 unused prototype slots are not zero")
    valid_prototypes = np.asarray(prototypes, dtype=np.float32)[np.asarray(mask)]
    if len(valid_prototypes):
        norms = np.linalg.norm(valid_prototypes, axis=1)
        if not np.allclose(norms, 1.0, rtol=0.0, atol=2e-3):
            raise ContractError("completed S02 prototypes are not normalized")

    try:
        samples = pq.read_table(
            output_dir / artifacts.appearance_samples,
            columns=[
                "sample_id",
                "micro_id",
                "det_id",
                "crop_quality",
                "selection_reason",
                "prototype_inlier",
                "prototype_outlier",
                "embedding_row",
            ],
        )
        micro = pq.read_table(
            output_dir / artifacts.micro_appearance,
            columns=[
                "micro_id",
                "prototype_row",
                "num_appearance_samples",
                "appearance_usable",
                "appearance_outlier_count",
            ],
        )
    except (OSError, pa.ArrowException, TypeError, ValueError) as exc:
        raise ContractError(f"cannot cross-validate completed S02 tables: {exc}") from exc

    def column_numpy(table: pa.Table, name: str) -> np.ndarray:
        return table[name].combine_chunks().to_numpy(zero_copy_only=False)

    sample_ids = np.asarray(column_numpy(samples, "sample_id"), dtype=np.int64)
    embedding_rows = np.asarray(
        column_numpy(samples, "embedding_row"), dtype=np.int64
    )
    expected_rows = np.arange(expected_samples, dtype=np.int64)
    if not np.array_equal(sample_ids, expected_rows) or not np.array_equal(
        embedding_rows, expected_rows
    ):
        raise ContractError("completed S02 sample/embedding row mapping is not identity")
    sample_det_ids = np.asarray(column_numpy(samples, "det_id"), dtype=np.int64)
    if len(np.unique(sample_det_ids)) != len(sample_det_ids):
        raise ContractError("completed S02 appearance samples repeat a detection")
    sample_quality = np.asarray(
        column_numpy(samples, "crop_quality"), dtype=np.float32
    )
    if np.any(sample_quality < config.min_crop_quality) or np.any(
        sample_quality > 1.0
    ):
        raise ContractError("completed S02 samples violate the crop-quality gate")
    selection_reason = set(map(str, column_numpy(samples, "selection_reason")))
    if not selection_reason.issubset({"head", "tail", "head;tail", "periodic"}):
        raise ContractError("completed S02 sample has an unknown selection reason")
    inlier = np.asarray(column_numpy(samples, "prototype_inlier"), dtype=np.bool_)
    outlier = np.asarray(column_numpy(samples, "prototype_outlier"), dtype=np.bool_)
    if np.any(inlier & outlier) or np.any(~(inlier | outlier)):
        raise ContractError("completed S02 per-sample prototype masks are inconsistent")

    micro_ids = np.asarray(column_numpy(micro, "micro_id"), dtype=np.int64)
    prototype_rows = np.asarray(
        column_numpy(micro, "prototype_row"), dtype=np.int64
    )
    if (
        len(np.unique(micro_ids)) != expected_microtracklets
        or np.any(np.diff(micro_ids) <= 0)
        or not np.array_equal(
            prototype_rows, np.arange(expected_microtracklets, dtype=np.int64)
        )
    ):
        raise ContractError("completed S02 micro/prototype row mapping is invalid")
    usable = np.asarray(column_numpy(micro, "appearance_usable"), dtype=np.bool_)
    if not np.array_equal(usable, np.any(np.asarray(mask), axis=1)):
        raise ContractError("completed S02 appearance_usable differs from prototype mask")
    sample_micro_ids = np.asarray(
        column_numpy(samples, "micro_id"), dtype=np.int64
    )
    locations = np.searchsorted(micro_ids, sample_micro_ids)
    if (
        len(locations)
        and (
            np.any(locations >= len(micro_ids))
            or not np.array_equal(micro_ids[locations], sample_micro_ids)
        )
    ):
        raise ContractError("completed S02 sample references an unknown micro_id")
    sample_counts = np.zeros(expected_microtracklets, dtype=np.int64)
    outlier_counts = np.zeros(expected_microtracklets, dtype=np.int64)
    np.add.at(sample_counts, locations, 1)
    np.add.at(outlier_counts, locations, outlier.astype(np.int64))
    if not np.array_equal(
        sample_counts,
        np.asarray(column_numpy(micro, "num_appearance_samples"), dtype=np.int64),
    ):
        raise ContractError("completed S02 sample counts differ across artifacts")
    if not np.array_equal(
        outlier_counts,
        np.asarray(column_numpy(micro, "appearance_outlier_count"), dtype=np.int64),
    ):
        raise ContractError("completed S02 outlier counts differ across artifacts")


def _validate_completed_output(
    output_dir: Path,
    existing: Mapping[str, Any],
    input_fingerprints: Sequence[Mapping[str, Any]],
    config_hash: str,
    config: AppearanceConfig,
    data: LoadedAppearanceData,
    exclusions: ExclusionResult,
) -> None:
    if existing.get("stage") != "S02":
        raise ContractError("existing _SUCCESS.json is not from S02")
    if existing.get("config_hash") != config_hash:
        raise ContractError("existing S02 output belongs to a different config")
    recorded_inputs = {
        str(Path(str(item["path"])).resolve()): item
        for item in existing.get("input_fingerprints", [])
        if isinstance(item, dict) and item.get("path")
    }
    current_inputs = {
        str(Path(str(item["path"])).resolve()): item for item in input_fingerprints
    }
    if set(recorded_inputs) != set(current_inputs):
        raise ContractError("completed S02 input fingerprint path set changed")
    for path, current in current_inputs.items():
        for key in ("size_bytes", "sha256"):
            if current[key] != recorded_inputs[path].get(key):
                raise ContractError(f"completed S02 input changed: {path} ({key})")
    records = existing.get("output_fingerprints")
    if not isinstance(records, list):
        raise ContractError("completed S02 lacks output fingerprints")
    by_name = {str(item.get("path")): item for item in records if isinstance(item, dict)}
    expected_names = _expected_artifact_names(config)
    if set(by_name) != expected_names or len(by_name) != len(records):
        raise ContractError("completed S02 output fingerprint set mismatch")
    actual_names = {
        str(path.relative_to(output_dir))
        for path in output_dir.rglob("*")
        if path.is_file() and path.name != config.artifacts.success
    }
    if actual_names != expected_names:
        raise ContractError("completed S02 artifact set mismatch")
    for name, recorded in by_name.items():
        current = _output_fingerprint(output_dir / name, output_dir)
        for key in ("size_bytes", "sha256"):
            if current[key] != recorded.get(key):
                raise ContractError(f"completed S02 artifact changed: {name} ({key})")
    stats = existing.get("stats")
    if not isinstance(stats, dict):
        raise ContractError("completed S02 success marker lacks stats")
    for key, expected in {
        "num_frames": data.num_frames,
        "num_valid_detections": data.num_valid_detections,
        "num_microtracklets": data.num_microtracklets,
        "num_true_review_events": exclusions.num_true_events,
        "num_appearance_exclusions": int(
            np.count_nonzero(exclusions.excluded)
        ),
    }.items():
        if stats.get(key) != expected:
            raise ContractError(f"completed S02 input-derived stat differs: {key}")
    _validate_output_files(output_dir, config, stats)


def run_s02(
    manifest_path: Path,
    ingest_dir: Path,
    microtrack_dir: Path,
    review_manifest_path: Path,
    config_path: Path,
    device: str,
    output_dir: Path,
    *,
    logger: LogFn = log,
    encoder_factory: Callable[..., Any] = build_encoder,
) -> dict[str, Any]:
    """Run the fixed S02 appearance stage.

    Video frames are decoded once per source clip, in source order. Every
    retained crop is embedded by the locked existing winner; no encoder
    bake-off is run.
    """

    manifest_path = manifest_path.resolve()
    ingest_dir = ingest_dir.resolve()
    microtrack_dir = microtrack_dir.resolve()
    review_manifest_path = review_manifest_path.resolve()
    config_path = config_path.resolve()
    output_dir = output_dir.resolve()
    config, config_payload, config_hash = load_appearance_config(config_path)
    if device != config.device:
        raise ContractError(f"S02 requires --device {config.device}")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
        raise ContractError(
            "fixed S02 requires the command environment CUDA_VISIBLE_DEVICES=1"
        )

    logger("[s02] validating immutable S00/S01/review/model inputs")
    data = _load_data(
        manifest_path,
        ingest_dir,
        microtrack_dir,
        review_manifest_path,
        config_path,
        config,
        logger=logger,
    )
    profiles = _resolve_encoder_profiles(config, data.input_fingerprints)
    exclusions = _load_review_exclusions(review_manifest_path, data)

    success_path = output_dir / config.artifacts.success
    if success_path.is_file():
        existing = _read_json(success_path, "S02 success marker")
        if not isinstance(existing, dict):
            raise ContractError("S02 success marker must be a JSON object")
        _validate_completed_output(
            output_dir,
            existing,
            data.input_fingerprints,
            config_hash,
            config,
            data,
            exclusions,
        )
        logger(f"[s02] already complete and fully revalidated: {success_path}")
        return dict(existing)
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
        raise ContractError(f"S02 staging directory already exists: {staging_dir}")
    staging_dir.mkdir(parents=False)
    output_dir = staging_dir

    boxes = np.column_stack(
        [
            np.asarray(data.detections[name], dtype=np.float64)
            for name in ("x1", "y1", "x2", "y2")
        ]
    )
    det_frames = np.asarray(data.detections["global_frame"], dtype=np.int64)
    logger("[s02] computing deterministic bbox-only quality context")
    other_iou = maximum_other_bbox_iou(boxes, det_frames)
    area_percentile = bbox_area_percentiles(boxes)
    boundary_distance = _boundary_distances(data)
    selected = _selected_positions(data, exclusions, other_iou, config)
    logger(
        f"[s02] quality candidate pool={len(selected):,}; "
        f"red-window exclusions={int(np.count_nonzero(exclusions.excluded)):,}"
    )

    session = _EmbeddingSession(
        profiles,
        config,
        device=device,
        encoder_factory=encoder_factory,
        logger=logger,
    )
    rejected_low_quality = _decode_selected_crops(
        data,
        selected,
        other_iou,
        area_percentile,
        config,
        session,
        logger=logger,
    )
    embeddings, winner, benchmark_report, separation_warning = session.finalize()
    if len(embeddings) == 0:
        raise ContractError("S02 retained no appearance embeddings")
    num_high_quality_candidates = len(session.records)
    final_records, embeddings = _finalize_representative_samples(
        session.records, embeddings, config
    )
    session.records = final_records
    logger(
        f"[s02] final high-quality representatives={len(final_records):,} "
        f"from {num_high_quality_candidates:,} retained candidates"
    )

    all_micro_ids = np.sort(
        np.asarray(data.microtracklets["micro_id"], dtype=np.int64)
    )
    sample_micro_ids = np.asarray(
        [record.micro_id for record in session.records], dtype=np.int64
    )
    crop_quality = np.asarray(
        [record.crop_quality for record in session.records], dtype=np.float32
    )
    prototype_batch = build_tracklet_prototypes(
        embeddings,
        sample_micro_ids,
        crop_quality,
        all_micro_ids,
        min_samples=config.min_samples_per_microtrack,
        max_prototypes=config.prototypes_per_tracklet,
        outlier_medoid_cosine=config.prototype_outlier_cosine_min,
        outlier_support_cosine=config.prototype_outlier_support_cosine,
        new_prototype_cosine=1.0 - config.prototype_cluster_min_separation,
    )

    sample_table = _sample_table(
        data,
        session.records,
        prototype_batch.sample_inlier_mask,
        prototype_batch.sample_outlier_mask,
    )
    exclusion_table = _exclusion_table(data, exclusions)
    micro_appearance = _micro_appearance_table(
        prototype_batch, selected_encoder=winner
    )
    micro_context = _micro_context_table(
        data, other_iou, boundary_distance, all_micro_ids
    )

    artifacts = config.artifacts
    _write_parquet(
        output_dir / artifacts.appearance_samples,
        sample_table,
        config.parquet_compression,
    )
    _write_parquet(
        output_dir / artifacts.appearance_exclusions,
        exclusion_table,
        config.parquet_compression,
    )
    _write_parquet(
        output_dir / artifacts.micro_appearance,
        micro_appearance,
        config.parquet_compression,
    )
    _write_parquet(
        output_dir / artifacts.micro_context,
        micro_context,
        config.parquet_compression,
    )
    _write_npy(output_dir / artifacts.sample_embeddings, embeddings.astype(np.float16))
    _write_npy(
        output_dir / artifacts.micro_prototypes,
        prototype_batch.prototypes.astype(np.float16),
    )
    _write_npy(
        output_dir / artifacts.micro_prototype_mask,
        prototype_batch.prototype_mask.astype(np.bool_),
    )

    selected_profile = next(profile for profile in profiles if profile.name == winner)
    configured_profile = next(
        profile for profile in config.model_profiles if profile.name == winner
    )
    choice = {
        "schema_version": config.schema_version,
        "selection_mode": config.selection_mode,
        "selected_profile": winner,
        "model_id": selected_profile.model_id,
        "timm_model_name": selected_profile.timm_model_name,
        "checkpoint_path": str(selected_profile.checkpoint_path),
        "checkpoint_filter": selected_profile.checkpoint_filter,
        "checkpoint_sha256": configured_profile.expected_weight_sha256,
        "model_revision": configured_profile.expected_revision,
        "preprocessing_id": selected_profile.preprocessing_id,
        "input_size": selected_profile.input_size,
        "resize_mode": selected_profile.resize_mode,
        "crop_pct": selected_profile.crop_pct,
        "interpolation": selected_profile.interpolation,
        "normalization_profile": configured_profile.normalization_profile,
        "mean": list(selected_profile.mean),
        "std": list(selected_profile.std),
        "embedding_dim": selected_profile.embedding_dim,
        "selection_metrics": benchmark_report["profiles"][winner],
        "selection_metrics_status": "not_recomputed_locked_existing_winner",
        "no_runtime_fallback": True,
    }
    _atomic_write_json(output_dir / artifacts.encoder_choice, choice)
    _atomic_write_json(output_dir / artifacts.encoder_benchmark, benchmark_report)
    _atomic_write_json(output_dir / artifacts.effective_config, config_payload)

    samples_per_micro = prototype_batch.num_samples.astype(np.int64)
    stats = {
        "num_frames": data.num_frames,
        "num_valid_detections": data.num_valid_detections,
        "num_microtracklets": data.num_microtracklets,
        "num_true_review_events": exclusions.num_true_events,
        "num_appearance_exclusions": int(exclusion_table.num_rows),
        "num_quality_candidate_pool_detections": int(len(selected)),
        "num_low_quality_candidates_rejected": int(rejected_low_quality),
        "num_high_quality_candidates": int(num_high_quality_candidates),
        "num_appearance_samples": int(len(embeddings)),
        "num_bakeoff_crops": 0,
        "num_microtracklets_below_min_samples": int(
            np.count_nonzero(samples_per_micro < config.min_samples_per_microtrack)
        ),
        "num_valid_prototypes": int(np.count_nonzero(prototype_batch.prototype_mask)),
        "embedding_dim": int(embeddings.shape[1]),
        "selected_encoder": winner,
        "encoder_selection_mode": config.selection_mode,
        "hard_negative_separation_warning": bool(separation_warning),
    }
    report = {
        "stage": "S02",
        "schema_version": config.schema_version,
        "sequence_id": data.sequence_id,
        "coordinate_system": EXPECTED_COORDINATE_SYSTEM,
        "source_rows_modified": False,
        "s00_s01_parquet_modified": False,
        "keypoints_used": False,
        "legacy_tracking_id_used_as_identity_feature": False,
        "review_legacy_transition_used_only_for_crop_exclusion": True,
        "video_decode_policy": "one_sequential_pass_per_clip_no_random_seek",
        "encoder_selection": {
            "mode": config.selection_mode,
            "selected_profile": winner,
            "bakeoff_run": False,
        },
        "sampling_policy": {
            "candidate_pool": {
                "period_sec": config.candidate_pool_period_sec,
                "max_samples_per_microtrack": (
                    config.candidate_pool_max_samples_per_microtrack
                ),
                "endpoint_samples": config.candidate_pool_endpoint_samples,
            },
            "quality_gate_before_final_selection": True,
            "final_representatives": {
                "period_sec": config.representative_period_sec,
                "max_samples_per_microtrack": config.max_samples_per_microtrack,
                "head_samples": config.head_samples,
                "tail_samples": config.tail_samples,
            },
        },
        "appearance_exclusion_policy": {
            "target_micro_only": True,
            "event_reasons": list(config.review_event_reasons),
            "event_half_window_frames": config.review_event_half_window_frames,
            "overlapping_windows": "union",
        },
        "warnings": (
            ["hard_negative_separation_below_configured_margin"]
            if separation_warning
            else []
        ),
        "stats": stats,
    }
    _atomic_write_json(output_dir / artifacts.appearance_report, report)
    _validate_output_files(output_dir, config, stats)
    _verify_inputs_unchanged(data.input_fingerprints)

    artifact_paths = sorted(
        path
        for path in output_dir.rglob("*")
        if path.is_file() and path.name != artifacts.success
    )
    if {path.name for path in artifact_paths} != _expected_artifact_names(config):
        raise ContractError("S02 staged artifact set mismatch")
    output_fingerprints = [
        _output_fingerprint(path, output_dir) for path in artifact_paths
    ]
    success = {
        "stage": "S02",
        "schema_version": config.schema_version,
        "config_hash": config_hash,
        "program_commit_hash": None,
        "input_fingerprints": list(data.input_fingerprints),
        "output_fingerprints": output_fingerprints,
        "stats": stats,
    }
    _atomic_write_json(output_dir / artifacts.success, success)

    if final_output_dir.exists():
        if any(final_output_dir.iterdir()):
            raise ContractError(
                f"final S02 output became non-empty during staging: {final_output_dir}"
            )
        final_output_dir.rmdir()
    os.replace(staging_dir, final_output_dir)
    final_success = final_output_dir / artifacts.success
    logger(f"[s02] complete: {final_success}")
    return success


__all__ = ["run_s02"]
