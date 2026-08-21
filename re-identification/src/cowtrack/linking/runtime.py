"""Public, read-only S00/S01/S02 inputs for calibrated link inference.

This module owns the exact joins and fingerprint checks shared by S03
calibration and downstream production link stages.  It never writes upstream
artifacts and never accepts row-order joins or whole-micro S02 prototypes.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from cowtrack.appearance.quality import maximum_other_bbox_iou
from cowtrack.config import ContractError
from cowtrack.linking.config import LinkCalibrationConfig, load_link_calibration_config
from cowtrack.linking.dataset_contract import EXPECTED_CLIP_ORDER
from cowtrack.linking.pseudo_pairs import CalibrationInput
from cowtrack.schemas.appearance import (
    APPEARANCE_EXCLUSIONS_SCHEMA,
    APPEARANCE_SAMPLES_SCHEMA,
    MICRO_APPEARANCE_SCHEMA,
)
from cowtrack.schemas.detections import DETECTIONS_SCHEMA
from cowtrack.schemas.frames import FRAMES_SCHEMA
from cowtrack.schemas.tracklets import DET_TO_MICRO_SCHEMA, MICROTRACKLETS_SCHEMA


LogFn = Callable[[str], None]
EXPECTED_CLIP_IDS = EXPECTED_CLIP_ORDER


@dataclass(frozen=True)
class FileFingerprint:
    """Content identity of one consumed file."""

    path: str
    size_bytes: int
    sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class MicroEndpoint:
    """Canonical full-tracklet endpoints in original encoded coordinates."""

    micro_id: int
    start_position: int
    end_position: int
    start_det_id: int
    end_det_id: int
    start_clip_id: str
    end_clip_id: str
    start_global_frame: int
    end_global_frame: int
    start_global_time_sec: float
    end_global_time_sec: float
    start_bbox_xyxy: tuple[float, float, float, float]
    end_bbox_xyxy: tuple[float, float, float, float]


@dataclass(frozen=True)
class CanonicalDetectionMetadata:
    """Canonical valid detections, aligned by array position."""

    det_ids: np.ndarray
    micro_ids: np.ndarray
    order_in_micro: np.ndarray
    clip_ids: np.ndarray
    global_frames: np.ndarray
    global_time_sec: np.ndarray
    x1: np.ndarray
    y1: np.ndarray
    x2: np.ndarray
    y2: np.ndarray
    other_bbox_max_iou: np.ndarray
    boundary_distance: np.ndarray
    review_excluded: np.ndarray


@dataclass(frozen=True)
class ProductionInputBundle:
    """Strict production input plus endpoint/path and input-byte provenance."""

    calibration_input: CalibrationInput
    detections: CanonicalDetectionMetadata
    micro_paths: Mapping[int, np.ndarray]
    endpoints: Mapping[int, MicroEndpoint]
    micro_appearance: Mapping[str, np.ndarray]
    appearance_report: Mapping[str, Any]
    encoder_choice: Mapping[str, Any]
    consumed_paths: tuple[Path, ...]
    input_fingerprints: tuple[FileFingerprint, ...]


@dataclass(frozen=True)
class S03RuntimeConfig:
    """Strictly reloaded S03 effective config and its marker contract."""

    config: LinkCalibrationConfig
    payload: Mapping[str, Any]
    config_hash: str
    effective_config_fingerprint: FileFingerprint
    success_marker: Mapping[str, Any]


_UPSTREAM_ARTIFACTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("S00", ("frames.parquet", "detections.parquet")),
    ("S01", ("det_to_micro.parquet", "microtracklets.parquet")),
    (
        "S02",
        (
            "appearance_samples.parquet",
            "sample_embeddings.f16.npy",
            "appearance_exclusions.parquet",
            "micro_appearance.parquet",
            "appearance_report.json",
            "encoder_choice.json",
            "effective_config.json",
        ),
    ),
)


def production_input_paths(
    ingest_dir: Path, microtrack_dir: Path, appearance_dir: Path
) -> tuple[Path, ...]:
    """Return the exact ordered files consumed by link feature reconstruction."""

    ingest = ingest_dir.resolve()
    micro = microtrack_dir.resolve()
    appearance = appearance_dir.resolve()
    return (
        ingest / "_SUCCESS.json",
        ingest / "frames.parquet",
        ingest / "detections.parquet",
        micro / "_SUCCESS.json",
        micro / "det_to_micro.parquet",
        micro / "microtracklets.parquet",
        appearance / "_SUCCESS.json",
        appearance / "appearance_samples.parquet",
        appearance / "sample_embeddings.f16.npy",
        appearance / "appearance_exclusions.parquet",
        appearance / "micro_appearance.parquet",
        appearance / "appearance_report.json",
        appearance / "encoder_choice.json",
        appearance / "effective_config.json",
    )


def fingerprint_file(path: Path) -> FileFingerprint:
    """Hash one required file or raise a link contract failure."""

    resolved = path.resolve()
    if not resolved.is_file():
        raise ContractError(f"required link input does not exist: {resolved}")
    digest = hashlib.sha256()
    try:
        with resolved.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
        size = int(resolved.stat().st_size)
    except OSError as exc:
        raise ContractError(f"cannot fingerprint link input {resolved}: {exc}") from exc
    return FileFingerprint(str(resolved), size, digest.hexdigest())


def _read_json(path: Path, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read {label} {path}: {exc}") from exc


def _column(table: pa.Table, name: str) -> np.ndarray:
    return table[name].combine_chunks().to_numpy(zero_copy_only=False)


def _read_parquet(
    path: Path,
    *,
    schema: pa.Schema,
    columns: Sequence[str] | None = None,
    label: str,
) -> pa.Table:
    try:
        parquet = pq.ParquetFile(path)
        if not parquet.schema_arrow.equals(schema, check_metadata=False):
            raise ContractError(f"{label} schema mismatch: {path}")
        return pq.read_table(path, columns=list(columns) if columns is not None else None)
    except ContractError:
        raise
    except (OSError, pa.ArrowException, TypeError, ValueError) as exc:
        raise ContractError(f"cannot read {label} {path}: {exc}") from exc


def _locate_ids(known_ids: np.ndarray, requested: np.ndarray, label: str) -> np.ndarray:
    known = np.asarray(known_ids, dtype=np.int64)
    query = np.asarray(requested, dtype=np.int64)
    order = np.argsort(known, kind="stable")
    sorted_ids = known[order]
    locations = np.searchsorted(sorted_ids, query)
    if len(locations) and (
        np.any(locations >= len(sorted_ids))
        or not np.array_equal(sorted_ids[locations], query)
    ):
        raise ContractError(f"link input {label} references an unknown det_id")
    return order[locations]


def _boundary_distance(
    x1: np.ndarray,
    y1: np.ndarray,
    x2: np.ndarray,
    y2: np.ndarray,
    width: np.ndarray,
    height: np.ndarray,
) -> np.ndarray:
    distance = np.minimum.reduce((x1, y1, width - x2, height - y2))
    short_side = np.minimum(x2 - x1, y2 - y1)
    scale = np.maximum(0.5 * short_side, np.finfo(np.float64).eps)
    result = np.clip(distance / scale, 0.0, 1.0)
    if not np.all(np.isfinite(result)):
        raise ContractError("link endpoint boundary distance is non-finite")
    return result.astype(np.float32)


def _verify_upstream_marker(
    directory: Path,
    stage: str,
    required_names: Sequence[str],
    current: Mapping[str, FileFingerprint],
) -> None:
    success = _read_json(directory / "_SUCCESS.json", f"{stage} success marker")
    if not isinstance(success, dict) or success.get("stage") != stage:
        raise ContractError(f"{directory} is not a completed {stage} output")
    records = success.get("output_fingerprints")
    if not isinstance(records, list):
        raise ContractError(f"{stage} success marker lacks output fingerprints")
    by_name = {
        str(item.get("path")): item for item in records if isinstance(item, dict)
    }
    if len(by_name) != len(records):
        raise ContractError(f"{stage} success marker has duplicate/invalid fingerprints")
    missing = sorted(set(required_names) - set(by_name))
    if missing:
        raise ContractError(f"{stage} success marker lacks artifacts: {missing}")
    for name in required_names:
        actual = current.get(str((directory / name).resolve()))
        if actual is None:
            raise ContractError(f"link loader did not fingerprint {stage} artifact: {name}")
        recorded = by_name[name]
        if actual.size_bytes != recorded.get("size_bytes"):
            raise ContractError(f"completed {stage} artifact changed: {name} (size_bytes)")
        if actual.sha256 != recorded.get("sha256"):
            raise ContractError(f"completed {stage} artifact changed: {name} (sha256)")


def _build_calibration_input(data: Mapping[str, Any]) -> CalibrationInput:
    summary = data["micro_summary"]
    parent_ids = np.asarray(summary["micro_id"], dtype=np.int64)
    appearance = data["micro_appearance"]
    appearance_ids = np.asarray(appearance["micro_id"], dtype=np.int64)
    appearance_order = np.argsort(appearance_ids, kind="stable")
    sorted_ids = appearance_ids[appearance_order]
    locations = np.searchsorted(sorted_ids, parent_ids)
    if np.any(locations >= len(sorted_ids)) or not np.array_equal(
        sorted_ids[locations], parent_ids
    ):
        raise ContractError("link loader cannot align parent and appearance summaries")
    appearance_rows = appearance_order[locations]
    frame_clips = np.asarray(data["frame_clip_id"], dtype=object)
    frame_times = np.asarray(data["frame_global_time_sec"], dtype=np.float64)
    timeline_clips = np.asarray(tuple(dict.fromkeys(map(str, frame_clips))), dtype=object)
    timeline_starts = np.asarray(
        [np.min(frame_times[frame_clips == clip]) for clip in timeline_clips],
        dtype=np.float64,
    )
    timeline_ends = np.asarray(
        [np.max(frame_times[frame_clips == clip]) for clip in timeline_clips],
        dtype=np.float64,
    )
    return CalibrationInput(
        timeline_clip_ids=timeline_clips,
        timeline_clip_start_time_sec=timeline_starts,
        timeline_clip_end_time_sec=timeline_ends,
        parent_micro_ids=parent_ids,
        parent_status=np.asarray(summary["status"], dtype=object),
        parent_num_detections=np.asarray(summary["num_detections"], dtype=np.int64),
        parent_local_purity_score=np.asarray(summary["local_purity_score"], dtype=np.float64),
        parent_bidirectional_agreement=np.asarray(summary["bidirectional_agreement"], dtype=np.float64),
        parent_internal_cosine_p10=np.asarray(
            appearance["internal_cosine_p10"], dtype=np.float64
        )[appearance_rows],
        det_ids=np.asarray(data["det_id"], dtype=np.int64),
        det_micro_ids=np.asarray(data["micro_id_by_detection"], dtype=np.int64),
        det_order_in_micro=np.asarray(data["order_in_micro"], dtype=np.int64),
        det_clip_ids=np.asarray(data["clip_id"], dtype=object),
        det_global_frames=np.asarray(data["global_frame"], dtype=np.int64),
        det_global_time_sec=np.asarray(data["global_time_sec"], dtype=np.float64),
        det_cx_norm=np.asarray(data["cx_norm"], dtype=np.float64),
        det_cy_norm=np.asarray(data["cy_norm"], dtype=np.float64),
        det_w_norm=np.asarray(data["w_norm"], dtype=np.float64),
        det_h_norm=np.asarray(data["h_norm"], dtype=np.float64),
        det_other_bbox_max_iou=np.asarray(data["other_bbox_max_iou"], dtype=np.float32),
        det_boundary_distance=np.asarray(data["boundary_distance"], dtype=np.float32),
        det_review_excluded=np.asarray(data["review_excluded"], dtype=np.bool_),
        sample_ids=np.asarray(data["sample_id"], dtype=np.int64),
        sample_micro_ids=np.asarray(data["sample_micro_id"], dtype=np.int64),
        sample_det_ids=np.asarray(data["sample_det_id"], dtype=np.int64),
        sample_crop_quality=np.asarray(data["sample_crop_quality"], dtype=np.float32),
        sample_other_bbox_max_iou=np.asarray(
            data["sample_other_bbox_max_iou"], dtype=np.float32
        ),
        sample_s02_inlier=np.asarray(data["sample_s02_inlier"], dtype=np.bool_),
        sample_embedding_rows=np.asarray(data["sample_embedding_row"], dtype=np.int64),
        embeddings=np.asarray(data["sample_embeddings"]),
    )


def _load_canonical_data(
    ingest_dir: Path,
    microtrack_dir: Path,
    appearance_dir: Path,
    *,
    logger: LogFn,
) -> dict[str, Any]:
    """Load and validate canonical columnar data without row-order assumptions."""

    frames = _read_parquet(
        ingest_dir / "frames.parquet",
        schema=FRAMES_SCHEMA,
        columns=("clip_id", "global_frame", "global_time_sec", "width", "height"),
        label="link frames",
    )
    frame_number = np.asarray(_column(frames, "global_frame"), dtype=np.int64)
    if not np.array_equal(frame_number, np.arange(len(frame_number), dtype=np.int64)):
        raise ContractError("link frames.global_frame must be contiguous and ordered")
    frame_time = np.asarray(_column(frames, "global_time_sec"), dtype=np.float64)
    if not len(frame_time) or not np.all(np.isfinite(frame_time)) or not np.all(np.diff(frame_time) > 0.0):
        raise ContractError("link frame time must be finite and strictly increasing")
    frame_width = np.asarray(_column(frames, "width"), dtype=np.float64)
    frame_height = np.asarray(_column(frames, "height"), dtype=np.float64)
    frame_clip = np.asarray(frames["clip_id"].to_pylist(), dtype=object)
    observed_clips = tuple(dict.fromkeys(map(str, frame_clip)))
    transitions = np.flatnonzero(frame_clip[1:] != frame_clip[:-1])
    if (
        observed_clips != EXPECTED_CLIP_IDS
        or len(transitions) != len(EXPECTED_CLIP_IDS) - 1
    ):
        raise ContractError("link input requires the complete contiguous 11-clip order")

    detections = _read_parquet(
        ingest_dir / "detections.parquet",
        schema=DETECTIONS_SCHEMA,
        columns=(
            "det_id", "clip_id", "global_frame", "global_time_sec", "x1", "y1",
            "x2", "y2", "cx_norm", "cy_norm", "w_norm", "h_norm", "valid",
        ),
        label="link detections",
    )
    valid = np.asarray(_column(detections, "valid"), dtype=np.bool_)
    if not np.any(valid):
        raise ContractError("link input has no valid detections")
    raw_ids = np.asarray(_column(detections, "det_id"), dtype=np.int64)[valid]
    raw_frames = np.asarray(_column(detections, "global_frame"), dtype=np.int64)[valid]
    canonical = np.lexsort((raw_ids, raw_frames))

    def numeric(name: str, dtype: Any) -> np.ndarray:
        values = np.asarray(_column(detections, name), dtype=dtype)[valid][canonical]
        if values.dtype.kind == "f" and not np.all(np.isfinite(values)):
            raise ContractError(f"link detection {name} contains non-finite values")
        return values

    det_id = raw_ids[canonical]
    global_frame = raw_frames[canonical]
    global_time = numeric("global_time_sec", np.float64)
    clip_id = np.asarray(detections["clip_id"].to_pylist(), dtype=object)[valid][canonical]
    x1, y1, x2, y2 = (numeric(name, np.float64) for name in ("x1", "y1", "x2", "y2"))
    cx, cy = numeric("cx_norm", np.float64), numeric("cy_norm", np.float64)
    width_norm, height_norm = numeric("w_norm", np.float64), numeric("h_norm", np.float64)
    if len(np.unique(det_id)) != len(det_id):
        raise ContractError("link valid det_id values are not unique")
    if np.any(global_frame < 0) or np.any(global_frame >= len(frame_time)):
        raise ContractError("link detection global_frame is out of range")
    if not np.array_equal(global_time, frame_time[global_frame]) or not np.array_equal(
        clip_id, frame_clip[global_frame]
    ):
        raise ContractError("link detection/frame join differs")
    if np.any(x2 <= x1) or np.any(y2 <= y1):
        raise ContractError("link detection boxes must have positive area")

    logger("[link-input] computing same-frame overlap and endpoint context")
    other_iou = maximum_other_bbox_iou(
        np.column_stack((x1, y1, x2, y2)),
        global_frame,
        progress=lambda done, total: logger(
            f"[link-input] overlap context: {done:,}/{total:,} frames"
        ),
        progress_interval_sec=10.0,
    )
    boundary = _boundary_distance(
        x1, y1, x2, y2, frame_width[global_frame], frame_height[global_frame]
    )

    mapping = _read_parquet(
        microtrack_dir / "det_to_micro.parquet",
        schema=DET_TO_MICRO_SCHEMA,
        columns=("det_id", "micro_id", "order_in_micro"),
        label="link det-to-micro",
    )
    mapping_det = np.asarray(_column(mapping, "det_id"), dtype=np.int64)
    mapping_micro = np.asarray(_column(mapping, "micro_id"), dtype=np.int64)
    mapping_order = np.asarray(_column(mapping, "order_in_micro"), dtype=np.int64)
    det_positions = _locate_ids(det_id, mapping_det, "det-to-micro")
    if len(det_positions) != len(det_id) or len(np.unique(det_positions)) != len(det_id):
        raise ContractError("link det-to-micro is not a bijection over valid detections")
    micro_by_detection = np.full(len(det_id), -1, dtype=np.int64)
    order_in_micro = np.full(len(det_id), -1, dtype=np.int64)
    micro_by_detection[det_positions], order_in_micro[det_positions] = mapping_micro, mapping_order
    if np.any(micro_by_detection < 0) or np.any(order_in_micro < 0):
        raise ContractError("link valid detection lacks a micro assignment")
    paths: dict[int, np.ndarray] = {}
    mapping_sort = np.lexsort((det_id[det_positions], mapping_order, mapping_micro))
    sorted_micro, sorted_position = mapping_micro[mapping_sort], det_positions[mapping_sort]
    sorted_order = mapping_order[mapping_sort]
    starts = np.flatnonzero(np.r_[True, sorted_micro[1:] != sorted_micro[:-1]])
    stops = np.r_[starts[1:], len(sorted_micro)]
    for start, stop in zip(starts, stops, strict=True):
        micro_id = int(sorted_micro[start])
        if not np.array_equal(sorted_order[start:stop], np.arange(stop - start)):
            raise ContractError(f"link micro {micro_id} order_in_micro is not contiguous")
        path = sorted_position[start:stop]
        if np.any(np.diff(global_time[path]) <= 0.0):
            raise ContractError(f"link micro {micro_id} is not strictly time ordered")
        immutable_path = np.asarray(path, dtype=np.int64)
        immutable_path.setflags(write=False)
        paths[micro_id] = immutable_path

    micros = _read_parquet(
        microtrack_dir / "microtracklets.parquet",
        schema=MICROTRACKLETS_SCHEMA,
        label="link microtracklets",
    )
    micro_ids = np.asarray(_column(micros, "micro_id"), dtype=np.int64)
    if len(np.unique(micro_ids)) != len(micro_ids) or set(map(int, micro_ids)) != set(paths):
        raise ContractError("link micro summary/path ID sets differ")
    micro_summary = {
        name: np.asarray(micros[name].to_pylist(), dtype=object)
        if pa.types.is_string(micros.schema.field(name).type)
        else np.asarray(_column(micros, name))
        for name in micros.column_names
    }

    samples = _read_parquet(
        appearance_dir / "appearance_samples.parquet",
        schema=APPEARANCE_SAMPLES_SCHEMA,
        label="link appearance samples",
    )
    sample_ids = np.asarray(_column(samples, "sample_id"), dtype=np.int64)
    embedding_rows = np.asarray(_column(samples, "embedding_row"), dtype=np.int64)
    sample_det_ids = np.asarray(_column(samples, "det_id"), dtype=np.int64)
    sample_micro_ids = np.asarray(_column(samples, "micro_id"), dtype=np.int64)
    expected_rows = np.arange(len(samples), dtype=np.int64)
    if not np.array_equal(sample_ids, expected_rows) or not np.array_equal(embedding_rows, expected_rows):
        raise ContractError("link sample/embedding row mappings are not identity")
    if len(np.unique(sample_det_ids)) != len(sample_det_ids):
        raise ContractError("link appearance samples repeat a detection")
    sample_positions = _locate_ids(det_id, sample_det_ids, "appearance sample")
    if not np.array_equal(micro_by_detection[sample_positions], sample_micro_ids):
        raise ContractError("link appearance sample micro join differs")
    sample_clips = np.asarray(samples["clip_id"].to_pylist(), dtype=object)
    sample_frames = np.asarray(_column(samples, "global_frame"), dtype=np.int64)
    sample_times = np.asarray(_column(samples, "global_time_sec"), dtype=np.float64)
    sample_quality = np.asarray(_column(samples, "crop_quality"), dtype=np.float32)
    sample_overlap = np.asarray(_column(samples, "other_bbox_max_iou"), dtype=np.float32)
    sample_inlier = np.asarray(_column(samples, "prototype_inlier"), dtype=np.bool_)
    sample_outlier = np.asarray(_column(samples, "prototype_outlier"), dtype=np.bool_)
    if not np.array_equal(sample_clips, clip_id[sample_positions]) or not np.array_equal(
        sample_frames, global_frame[sample_positions]
    ) or not np.array_equal(sample_times, global_time[sample_positions]):
        raise ContractError("link appearance sample detection join differs")
    if not np.allclose(sample_overlap, np.asarray(other_iou, dtype=np.float32)[sample_positions], rtol=0.0, atol=1e-6):
        raise ContractError("link appearance sample overlap join differs")
    if not np.all(np.isfinite(sample_quality)) or np.any((sample_quality < 0.0) | (sample_quality > 1.0)):
        raise ContractError("link appearance sample crop quality is invalid")
    if np.any(sample_inlier & sample_outlier) or np.any(~(sample_inlier | sample_outlier)):
        raise ContractError("link appearance sample masks are inconsistent")
    try:
        embeddings = np.load(appearance_dir / "sample_embeddings.f16.npy", mmap_mode="r")
    except (OSError, ValueError) as exc:
        raise ContractError(f"cannot load link sample embeddings: {exc}") from exc
    if embeddings.ndim != 2 or embeddings.shape[0] != len(samples) or embeddings.dtype != np.float16 or not np.isfinite(embeddings).all():
        raise ContractError("link sample embedding shape/dtype/value contract differs")
    if len(embeddings):
        norms = np.linalg.norm(np.asarray(embeddings, dtype=np.float32), axis=1)
        if not np.allclose(norms, 1.0, rtol=0.0, atol=2e-3):
            raise ContractError("link sample embeddings are not L2-normalized")

    exclusions = _read_parquet(
        appearance_dir / "appearance_exclusions.parquet",
        schema=APPEARANCE_EXCLUSIONS_SCHEMA,
        columns=("det_id", "appearance_excluded"),
        label="link appearance exclusions",
    )
    excluded_ids = np.asarray(_column(exclusions, "det_id"), dtype=np.int64)
    excluded_values = np.asarray(_column(exclusions, "appearance_excluded"), dtype=np.bool_)
    if not np.all(excluded_values) or len(np.unique(excluded_ids)) != len(excluded_ids):
        raise ContractError("link appearance exclusion rows are inconsistent")
    excluded_positions = _locate_ids(det_id, excluded_ids, "appearance exclusion")
    review_excluded = np.zeros(len(det_id), dtype=np.bool_)
    review_excluded[excluded_positions] = True
    if np.any(review_excluded[sample_positions]):
        raise ContractError("link appearance samples contain a review-excluded detection")

    micro_appearance = _read_parquet(
        appearance_dir / "micro_appearance.parquet",
        schema=MICRO_APPEARANCE_SCHEMA,
        label="link micro appearance",
    )
    appearance_ids = np.asarray(_column(micro_appearance, "micro_id"), dtype=np.int64)
    if len(appearance_ids) != len(paths) or len(np.unique(appearance_ids)) != len(appearance_ids) or set(map(int, appearance_ids)) != set(paths):
        raise ContractError("link micro appearance ID set differs")
    prototype_rows = np.asarray(_column(micro_appearance, "prototype_row"), dtype=np.int64)
    if not np.array_equal(np.sort(prototype_rows), np.arange(len(prototype_rows))):
        raise ContractError("link micro appearance prototype rows are not a bijection")
    micro_appearance_values = {
        name: np.asarray(micro_appearance[name].to_pylist(), dtype=object)
        if pa.types.is_string(micro_appearance.schema.field(name).type)
        else np.asarray(_column(micro_appearance, name))
        for name in micro_appearance.column_names
    }
    report_payloads: dict[str, dict[str, Any]] = {}
    for name in ("appearance_report.json", "encoder_choice.json", "effective_config.json"):
        payload = _read_json(appearance_dir / name, f"S02 {name}")
        if not isinstance(payload, dict):
            raise ContractError(f"S02 {name} must contain a JSON object")
        report_payloads[name] = payload

    return {
        "det_id": det_id, "global_frame": global_frame, "global_time_sec": global_time,
        "clip_id": clip_id, "x1": x1, "y1": y1, "x2": x2, "y2": y2,
        "cx_norm": cx, "cy_norm": cy, "w_norm": width_norm, "h_norm": height_norm,
        "micro_id_by_detection": micro_by_detection, "order_in_micro": order_in_micro,
        "other_bbox_max_iou": np.asarray(other_iou, dtype=np.float32),
        "boundary_distance": boundary, "review_excluded": review_excluded,
        "paths": paths, "micro_summary": micro_summary, "sample_id": sample_ids,
        "sample_det_id": sample_det_ids, "sample_micro_id": sample_micro_ids,
        "sample_crop_quality": sample_quality, "sample_other_bbox_max_iou": sample_overlap,
        "sample_s02_inlier": sample_inlier, "sample_embedding_row": embedding_rows,
        "sample_embeddings": embeddings, "micro_appearance": micro_appearance_values,
        "frame_clip_id": frame_clip, "frame_global_time_sec": frame_time,
        "appearance_report": report_payloads["appearance_report.json"],
        "encoder_choice": report_payloads["encoder_choice.json"],
    }


def load_production_inputs(
    ingest_dir: Path,
    microtrack_dir: Path,
    appearance_dir: Path,
    *,
    logger: LogFn = lambda _message: None,
) -> ProductionInputBundle:
    """Strict-load immutable upstream inputs and return public runtime metadata."""

    ingest, micro, appearance = (
        ingest_dir.resolve(), microtrack_dir.resolve(), appearance_dir.resolve()
    )
    paths = production_input_paths(ingest, micro, appearance)
    before = tuple(fingerprint_file(path) for path in paths)
    current = {item.path: item for item in before}
    for directory, (stage, names) in zip(
        (ingest, micro, appearance), _UPSTREAM_ARTIFACTS, strict=True
    ):
        _verify_upstream_marker(directory, stage, names, current)
    data = _load_canonical_data(ingest, micro, appearance, logger=logger)
    after = tuple(fingerprint_file(path) for path in paths)
    if before != after:
        raise ContractError("link production inputs changed while being loaded")
    calibration = _build_calibration_input(data)
    detections = CanonicalDetectionMetadata(
        det_ids=calibration.det_ids,
        micro_ids=calibration.det_micro_ids,
        order_in_micro=calibration.det_order_in_micro,
        clip_ids=calibration.det_clip_ids,
        global_frames=calibration.det_global_frames,
        global_time_sec=calibration.det_global_time_sec,
        x1=np.asarray(data["x1"], dtype=np.float64),
        y1=np.asarray(data["y1"], dtype=np.float64),
        x2=np.asarray(data["x2"], dtype=np.float64),
        y2=np.asarray(data["y2"], dtype=np.float64),
        other_bbox_max_iou=calibration.det_other_bbox_max_iou,
        boundary_distance=calibration.det_boundary_distance,
        review_excluded=calibration.det_review_excluded,
    )
    endpoints: dict[int, MicroEndpoint] = {}
    for micro_id, positions in data["paths"].items():
        start, end = int(positions[0]), int(positions[-1])
        endpoints[int(micro_id)] = MicroEndpoint(
            micro_id=int(micro_id), start_position=start, end_position=end,
            start_det_id=int(calibration.det_ids[start]), end_det_id=int(calibration.det_ids[end]),
            start_clip_id=str(calibration.det_clip_ids[start]), end_clip_id=str(calibration.det_clip_ids[end]),
            start_global_frame=int(calibration.det_global_frames[start]), end_global_frame=int(calibration.det_global_frames[end]),
            start_global_time_sec=float(calibration.det_global_time_sec[start]), end_global_time_sec=float(calibration.det_global_time_sec[end]),
            start_bbox_xyxy=tuple(float(data[name][start]) for name in ("x1", "y1", "x2", "y2")),
            end_bbox_xyxy=tuple(float(data[name][end]) for name in ("x1", "y1", "x2", "y2")),
        )
    return ProductionInputBundle(
        calibration_input=calibration,
        detections=detections,
        micro_paths=MappingProxyType(dict(data["paths"])),
        endpoints=MappingProxyType(endpoints),
        micro_appearance=MappingProxyType(dict(data["micro_appearance"])),
        appearance_report=MappingProxyType(dict(data["appearance_report"])),
        encoder_choice=MappingProxyType(dict(data["encoder_choice"])),
        consumed_paths=paths,
        input_fingerprints=before,
    )


def load_s03_runtime_config(calibration_dir: Path) -> S03RuntimeConfig:
    """Verify and strict-load S03 ``effective_config.json`` against its marker."""

    directory = calibration_dir.resolve()
    success = _read_json(directory / "_SUCCESS.json", "S03 success marker")
    if not isinstance(success, dict) or success.get("stage") != "S03":
        raise ContractError("calibration directory is not a completed S03 output")
    recorded_hash = success.get("config_hash")
    if not isinstance(recorded_hash, str) or len(recorded_hash) != 64 or any(
        char not in "0123456789abcdef" for char in recorded_hash
    ):
        raise ContractError("S03 success marker has an invalid config hash")
    records = success.get("output_fingerprints")
    if not isinstance(records, list):
        raise ContractError("S03 success marker lacks output fingerprints")
    by_name = {str(item.get("path")): item for item in records if isinstance(item, dict)}
    if len(by_name) != len(records) or "effective_config.json" not in by_name:
        raise ContractError("S03 success marker lacks a unique effective config fingerprint")
    fingerprint = fingerprint_file(directory / "effective_config.json")
    recorded = by_name["effective_config.json"]
    if fingerprint.size_bytes != recorded.get("size_bytes") or fingerprint.sha256 != recorded.get("sha256"):
        raise ContractError("completed S03 runtime artifact changed: effective_config.json")
    config, payload, computed_hash = load_link_calibration_config(
        directory / "effective_config.json"
    )
    if computed_hash != recorded_hash:
        raise ContractError("S03 effective config canonical hash differs from success marker")
    return S03RuntimeConfig(
        config=config,
        payload=MappingProxyType(payload),
        config_hash=computed_hash,
        effective_config_fingerprint=fingerprint,
        success_marker=MappingProxyType(success),
    )


def validate_s03_input_fingerprints(
    calibration_dir: Path, bundle: ProductionInputBundle
) -> None:
    """Prove S04's supplied S00/S01/S02 bytes equal those consumed by S03."""

    if not isinstance(bundle, ProductionInputBundle):
        raise ContractError("S03 input validation requires ProductionInputBundle")
    runtime = load_s03_runtime_config(calibration_dir)
    records = runtime.success_marker.get("input_fingerprints")
    if not isinstance(records, list):
        raise ContractError("S03 success marker lacks input fingerprints")
    # S03 records its source config first, then the exact ordered upstream set.
    expected_names = [path.name for path in bundle.consumed_paths]
    if len(records) != len(expected_names) + 1:
        raise ContractError("S03 input fingerprint count differs from runtime inputs")
    upstream = records[1:]
    if any(not isinstance(item, dict) for item in upstream) or [
        Path(str(item.get("path"))).name for item in upstream
    ] != expected_names:
        raise ContractError("S03 input fingerprint order/path contract differs")
    for current, recorded in zip(bundle.input_fingerprints, upstream, strict=True):
        if current.size_bytes != recorded.get("size_bytes") or current.sha256 != recorded.get("sha256"):
            raise ContractError(f"runtime input differs from S03 marker bytes: {current.path}")


__all__ = [
    "CanonicalDetectionMetadata",
    "FileFingerprint",
    "MicroEndpoint",
    "ProductionInputBundle",
    "S03RuntimeConfig",
    "fingerprint_file",
    "load_production_inputs",
    "load_s03_runtime_config",
    "production_input_paths",
    "validate_s03_input_fingerprints",
]
