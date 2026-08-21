"""S03 unsupervised short/long link-scorer calibration.

The stage is intentionally metadata-only: it never decodes video and never
runs an appearance encoder.  Internal split geometry is reconstructed by an
explicit det_id join across immutable S00/S01/S02 artifacts.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from cowtrack.appearance.quality import maximum_other_bbox_iou
from cowtrack.config import ContractError
from cowtrack.linking.config import LinkCalibrationConfig, load_link_calibration_config
from cowtrack.linking.dataset_contract import EXPECTED_CLIP_ORDER
from cowtrack.linking.features import LONG_FEATURE_SCHEMA, SHORT_FEATURE_SCHEMA
from cowtrack.linking.model import (
    LinkModelArtifact,
    LinkScorer,
    calibrate_mode_thresholds,
    clopper_pearson_upper,
    disabled_link_model,
    fit_link_model,
    load_link_model,
    save_link_model,
)
from cowtrack.linking.runtime import load_production_inputs
from cowtrack.schemas.appearance import (
    APPEARANCE_EXCLUSIONS_SCHEMA,
    APPEARANCE_SAMPLES_SCHEMA,
    MICRO_APPEARANCE_SCHEMA,
)
from cowtrack.schemas.calibration import PSEUDO_PAIRS_SCHEMA, S03_ALL_FEATURES
from cowtrack.schemas.detections import DETECTIONS_SCHEMA
from cowtrack.schemas.frames import FRAMES_SCHEMA
from cowtrack.schemas.tracklets import DET_TO_MICRO_SCHEMA, MICROTRACKLETS_SCHEMA


LogFn = Callable[[str], None]
EXPECTED_CLIP_IDS = EXPECTED_CLIP_ORDER


def log(message: str) -> None:
    print(message, flush=True)


def _read_json(path: Path, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read {label} {path}: {exc}") from exc


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    except OSError as exc:
        temporary.unlink(missing_ok=True)
        raise ContractError(f"cannot atomically write S03 JSON {path}: {exc}") from exc


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise ContractError(f"cannot fingerprint S03 input {path}: {exc}") from exc
    return digest.hexdigest()


def _fingerprint(path: Path, *, relative_to: Path | None = None) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise ContractError(f"required S03 file does not exist: {path}")
    display = str(path.relative_to(relative_to)) if relative_to is not None else str(path)
    return {
        "path": display,
        "size_bytes": int(path.stat().st_size),
        "sha256": _sha256(path),
    }


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


def _require_success(directory: Path, stage: str) -> Path:
    path = directory / "_SUCCESS.json"
    payload = _read_json(path, f"{stage} success marker")
    if not isinstance(payload, dict) or payload.get("stage") != stage:
        raise ContractError(f"{directory} is not a completed {stage} output")
    return path


def _verify_upstream_fingerprints(
    directory: Path,
    stage: str,
    required_names: Sequence[str],
    current_fingerprints: Mapping[str, Mapping[str, Any]],
) -> None:
    """Prove each consumed artifact still matches its upstream success marker."""

    success_path = directory / "_SUCCESS.json"
    success = _read_json(success_path, f"{stage} success marker")
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
        path = str((directory / name).resolve())
        current = current_fingerprints.get(path)
        if current is None:
            raise ContractError(f"S03 did not fingerprint consumed {stage} artifact: {name}")
        for key in ("size_bytes", "sha256"):
            if current[key] != by_name[name].get(key):
                raise ContractError(
                    f"completed {stage} artifact changed: {name} ({key})"
                )


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
        raise ContractError(f"S03 {label} references an unknown det_id")
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
        raise ContractError("S03 endpoint boundary distance is non-finite")
    return result.astype(np.float32)


def _stage_input_paths(
    ingest_dir: Path, microtrack_dir: Path, appearance_dir: Path
) -> list[Path]:
    return [
        ingest_dir / "_SUCCESS.json",
        ingest_dir / "frames.parquet",
        ingest_dir / "detections.parquet",
        microtrack_dir / "_SUCCESS.json",
        microtrack_dir / "det_to_micro.parquet",
        microtrack_dir / "microtracklets.parquet",
        appearance_dir / "_SUCCESS.json",
        appearance_dir / "appearance_samples.parquet",
        appearance_dir / "sample_embeddings.f16.npy",
        appearance_dir / "appearance_exclusions.parquet",
        appearance_dir / "micro_appearance.parquet",
        appearance_dir / "appearance_report.json",
        appearance_dir / "encoder_choice.json",
        appearance_dir / "effective_config.json",
    ]


def _load_stage_data(
    ingest_dir: Path,
    microtrack_dir: Path,
    appearance_dir: Path,
    *,
    logger: LogFn,
) -> tuple[dict[str, Any], list[Path]]:
    """Load canonical columnar inputs without accepting row-order assumptions."""

    _require_success(ingest_dir, "S00")
    _require_success(microtrack_dir, "S01")
    _require_success(appearance_dir, "S02")

    frames = _read_parquet(
        ingest_dir / "frames.parquet",
        schema=FRAMES_SCHEMA,
        columns=("clip_id", "global_frame", "global_time_sec", "width", "height"),
        label="S03 frames",
    )
    frame_number = np.asarray(_column(frames, "global_frame"), dtype=np.int64)
    if not np.array_equal(frame_number, np.arange(len(frame_number), dtype=np.int64)):
        raise ContractError("S03 frames.global_frame must be contiguous and ordered")
    frame_time = np.asarray(_column(frames, "global_time_sec"), dtype=np.float64)
    if not len(frame_time) or not np.all(np.diff(frame_time) > 0.0):
        raise ContractError("S03 frame time must be strictly increasing")
    frame_width = np.asarray(_column(frames, "width"), dtype=np.float64)
    frame_height = np.asarray(_column(frames, "height"), dtype=np.float64)
    frame_clip = np.asarray(frames["clip_id"].to_pylist(), dtype=object)
    observed_clips = tuple(dict.fromkeys(map(str, frame_clip)))
    clip_transitions = np.flatnonzero(frame_clip[1:] != frame_clip[:-1])
    if (
        observed_clips != EXPECTED_CLIP_IDS
        or len(clip_transitions) != len(EXPECTED_CLIP_IDS) - 1
    ):
        raise ContractError(
            "S03 requires the complete contiguous 11-clip manifest order"
        )

    detections = _read_parquet(
        ingest_dir / "detections.parquet",
        schema=DETECTIONS_SCHEMA,
        columns=(
            "det_id",
            "clip_id",
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
        ),
        label="S03 detections",
    )
    valid = np.asarray(_column(detections, "valid"), dtype=np.bool_)
    if not np.any(valid):
        raise ContractError("S03 has no valid detections")
    raw_det_id = np.asarray(_column(detections, "det_id"), dtype=np.int64)[valid]
    raw_frame = np.asarray(_column(detections, "global_frame"), dtype=np.int64)[valid]
    canonical = np.lexsort((raw_det_id, raw_frame))

    def detection_numeric(name: str, dtype: Any) -> np.ndarray:
        return np.asarray(_column(detections, name), dtype=dtype)[valid][canonical]

    det_id = raw_det_id[canonical]
    global_frame = raw_frame[canonical]
    global_time = detection_numeric("global_time_sec", np.float64)
    clip_id = np.asarray(detections["clip_id"].to_pylist(), dtype=object)[valid][canonical]
    x1 = detection_numeric("x1", np.float64)
    y1 = detection_numeric("y1", np.float64)
    x2 = detection_numeric("x2", np.float64)
    y2 = detection_numeric("y2", np.float64)
    cx = detection_numeric("cx_norm", np.float64)
    cy = detection_numeric("cy_norm", np.float64)
    width_norm = detection_numeric("w_norm", np.float64)
    height_norm = detection_numeric("h_norm", np.float64)
    if len(np.unique(det_id)) != len(det_id):
        raise ContractError("S03 valid detection det_id values are not unique")
    if not np.array_equal(global_time, frame_time[global_frame]):
        raise ContractError("S03 detection/frame time join differs")
    if not np.array_equal(clip_id, frame_clip[global_frame]):
        raise ContractError("S03 detection/frame clip join differs")

    boxes = np.column_stack((x1, y1, x2, y2))
    logger("[s03] computing same-frame overlap and endpoint boundary context")
    other_iou = maximum_other_bbox_iou(
        boxes,
        global_frame,
        progress=lambda completed, total: logger(
            f"[s03] overlap context: {completed:,}/{total:,} frames"
        ),
        progress_interval_sec=10.0,
    )
    boundary = _boundary_distance(
        x1,
        y1,
        x2,
        y2,
        frame_width[global_frame],
        frame_height[global_frame],
    )

    mapping = _read_parquet(
        microtrack_dir / "det_to_micro.parquet",
        schema=DET_TO_MICRO_SCHEMA,
        columns=("det_id", "micro_id", "order_in_micro"),
        label="S03 det-to-micro",
    )
    mapping_det = np.asarray(_column(mapping, "det_id"), dtype=np.int64)
    mapping_micro = np.asarray(_column(mapping, "micro_id"), dtype=np.int64)
    mapping_order = np.asarray(_column(mapping, "order_in_micro"), dtype=np.int64)
    det_positions = _locate_ids(det_id, mapping_det, "det-to-micro")
    if len(det_positions) != len(det_id) or len(np.unique(det_positions)) != len(det_id):
        raise ContractError("S03 det-to-micro is not a bijection over valid detections")
    micro_by_detection = np.full(len(det_id), -1, dtype=np.int64)
    order_in_micro = np.full(len(det_id), -1, dtype=np.int64)
    micro_by_detection[det_positions] = mapping_micro
    order_in_micro[det_positions] = mapping_order
    if np.any(micro_by_detection < 0) or np.any(order_in_micro < 0):
        raise ContractError("S03 valid detection lacks micro assignment")

    paths: dict[int, np.ndarray] = {}
    mapping_sort = np.lexsort((det_id[det_positions], mapping_order, mapping_micro))
    sorted_micro = mapping_micro[mapping_sort]
    sorted_position = det_positions[mapping_sort]
    sorted_order = mapping_order[mapping_sort]
    starts = np.flatnonzero(np.r_[True, sorted_micro[1:] != sorted_micro[:-1]])
    stops = np.r_[starts[1:], len(sorted_micro)]
    for start, stop in zip(starts, stops, strict=True):
        micro_id = int(sorted_micro[start])
        observed = sorted_order[start:stop]
        if not np.array_equal(observed, np.arange(stop - start, dtype=np.int64)):
            raise ContractError(f"S03 micro {micro_id} order_in_micro is not contiguous")
        path = sorted_position[start:stop]
        if np.any(np.diff(global_time[path]) <= 0.0):
            raise ContractError(f"S03 micro {micro_id} is not strictly time ordered")
        paths[micro_id] = path

    micros = _read_parquet(
        microtrack_dir / "microtracklets.parquet",
        schema=MICROTRACKLETS_SCHEMA,
        label="S03 microtracklets",
    )
    micro_ids = np.asarray(_column(micros, "micro_id"), dtype=np.int64)
    if len(np.unique(micro_ids)) != len(micro_ids) or set(map(int, micro_ids)) != set(paths):
        raise ContractError("S03 micro summary/path ID sets differ")
    micro_summary = {
        name: (
            np.asarray(micros[name].to_pylist(), dtype=object)
            if pa.types.is_string(micros.schema.field(name).type)
            else np.asarray(_column(micros, name))
        )
        for name in micros.column_names
    }

    samples = _read_parquet(
        appearance_dir / "appearance_samples.parquet",
        schema=APPEARANCE_SAMPLES_SCHEMA,
        label="S03 appearance samples",
    )
    sample_ids = np.asarray(_column(samples, "sample_id"), dtype=np.int64)
    embedding_rows = np.asarray(_column(samples, "embedding_row"), dtype=np.int64)
    sample_det_ids = np.asarray(_column(samples, "det_id"), dtype=np.int64)
    sample_micro_ids = np.asarray(_column(samples, "micro_id"), dtype=np.int64)
    expected_sample_rows = np.arange(len(samples), dtype=np.int64)
    if not np.array_equal(sample_ids, expected_sample_rows) or not np.array_equal(
        embedding_rows, expected_sample_rows
    ):
        raise ContractError("S03 sample/embedding row mappings are not identity")
    if len(np.unique(sample_det_ids)) != len(sample_det_ids):
        raise ContractError("S03 appearance samples repeat a detection")
    sample_det_positions = _locate_ids(det_id, sample_det_ids, "appearance sample")
    if not np.array_equal(micro_by_detection[sample_det_positions], sample_micro_ids):
        raise ContractError("S03 appearance sample micro join differs")
    sample_order_in_micro = order_in_micro[sample_det_positions]
    sample_clip_ids = np.asarray(samples["clip_id"].to_pylist(), dtype=object)
    if not np.array_equal(sample_clip_ids, clip_id[sample_det_positions]):
        raise ContractError("S03 appearance sample clip join differs")
    sample_frames = np.asarray(_column(samples, "global_frame"), dtype=np.int64)
    sample_times = np.asarray(_column(samples, "global_time_sec"), dtype=np.float64)
    sample_quality = np.asarray(_column(samples, "crop_quality"), dtype=np.float32)
    sample_overlap = np.asarray(
        _column(samples, "other_bbox_max_iou"), dtype=np.float32
    )
    sample_inlier = np.asarray(_column(samples, "prototype_inlier"), dtype=np.bool_)
    sample_outlier = np.asarray(_column(samples, "prototype_outlier"), dtype=np.bool_)
    if not np.array_equal(sample_frames, global_frame[sample_det_positions]):
        raise ContractError("S03 appearance sample frame join differs")
    if not np.array_equal(sample_times, global_time[sample_det_positions]):
        raise ContractError("S03 appearance sample time join differs")
    if not np.allclose(
        sample_overlap,
        np.asarray(other_iou, dtype=np.float32)[sample_det_positions],
        rtol=0.0,
        atol=1e-6,
    ):
        raise ContractError("S03 appearance sample overlap join differs")
    if (
        not np.all(np.isfinite(sample_quality))
        or np.any(sample_quality < 0.0)
        or np.any(sample_quality > 1.0)
    ):
        raise ContractError("S03 appearance sample crop quality is invalid")
    if np.any(sample_inlier & sample_outlier) or np.any(~(sample_inlier | sample_outlier)):
        raise ContractError("S03 appearance sample inlier/outlier masks are inconsistent")

    try:
        embeddings = np.load(
            appearance_dir / "sample_embeddings.f16.npy", mmap_mode="r"
        )
    except (OSError, ValueError) as exc:
        raise ContractError(f"cannot load S03 sample embeddings: {exc}") from exc
    if (
        embeddings.ndim != 2
        or embeddings.shape[0] != len(samples)
        or embeddings.dtype != np.float16
        or not np.isfinite(embeddings).all()
    ):
        raise ContractError("S03 sample embedding shape/dtype/value contract differs")
    if len(embeddings):
        embedding_norms = np.linalg.norm(
            np.asarray(embeddings, dtype=np.float32), axis=1
        )
        if not np.allclose(embedding_norms, 1.0, rtol=0.0, atol=2e-3):
            raise ContractError("S03 sample embeddings are not L2-normalized")

    exclusions = _read_parquet(
        appearance_dir / "appearance_exclusions.parquet",
        schema=APPEARANCE_EXCLUSIONS_SCHEMA,
        columns=("det_id", "appearance_excluded"),
        label="S03 appearance exclusions",
    )
    excluded_det_ids = np.asarray(_column(exclusions, "det_id"), dtype=np.int64)
    excluded_values = np.asarray(_column(exclusions, "appearance_excluded"), dtype=np.bool_)
    if not np.all(excluded_values) or len(np.unique(excluded_det_ids)) != len(excluded_det_ids):
        raise ContractError("S03 appearance exclusion rows are inconsistent")
    excluded_positions = _locate_ids(det_id, excluded_det_ids, "appearance exclusion")
    review_excluded = np.zeros(len(det_id), dtype=np.bool_)
    review_excluded[excluded_positions] = True
    if np.any(review_excluded[sample_det_positions]):
        raise ContractError("S03 appearance samples contain a review-excluded detection")

    micro_appearance = _read_parquet(
        appearance_dir / "micro_appearance.parquet",
        schema=MICRO_APPEARANCE_SCHEMA,
        label="S03 micro appearance",
    )
    appearance_micro_ids = np.asarray(_column(micro_appearance, "micro_id"), dtype=np.int64)
    if (
        len(appearance_micro_ids) != len(paths)
        or len(np.unique(appearance_micro_ids)) != len(appearance_micro_ids)
        or set(map(int, appearance_micro_ids)) != set(paths)
    ):
        raise ContractError("S03 micro appearance ID set differs")
    prototype_rows = np.asarray(
        _column(micro_appearance, "prototype_row"), dtype=np.int64
    )
    if not np.array_equal(
        np.sort(prototype_rows), np.arange(len(prototype_rows), dtype=np.int64)
    ):
        raise ContractError("S03 micro appearance prototype rows are not a bijection")
    micro_appearance_values = {
        name: (
            np.asarray(micro_appearance[name].to_pylist(), dtype=object)
            if pa.types.is_string(micro_appearance.schema.field(name).type)
            else np.asarray(_column(micro_appearance, name))
        )
        for name in micro_appearance.column_names
    }

    appearance_report = _read_json(
        appearance_dir / "appearance_report.json", "S02 appearance report"
    )
    encoder_choice = _read_json(
        appearance_dir / "encoder_choice.json", "S02 encoder choice"
    )
    if not isinstance(appearance_report, dict) or not isinstance(encoder_choice, dict):
        raise ContractError("S03 S02 reports must be JSON objects")

    data = {
        "det_id": det_id,
        "global_frame": global_frame,
        "global_time_sec": global_time,
        "clip_id": clip_id,
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "cx_norm": cx,
        "cy_norm": cy,
        "w_norm": width_norm,
        "h_norm": height_norm,
        "micro_id_by_detection": micro_by_detection,
        "order_in_micro": order_in_micro,
        "other_bbox_max_iou": np.asarray(other_iou, dtype=np.float32),
        "boundary_distance": boundary,
        "review_excluded": review_excluded,
        "paths": paths,
        "micro_summary": micro_summary,
        "sample_id": sample_ids,
        "sample_det_id": sample_det_ids,
        "sample_det_position": sample_det_positions,
        "sample_micro_id": sample_micro_ids,
        "sample_order_in_micro": sample_order_in_micro,
        "sample_global_frame": sample_frames,
        "sample_global_time_sec": sample_times,
        "sample_crop_quality": sample_quality,
        "sample_other_bbox_max_iou": sample_overlap,
        "sample_s02_inlier": sample_inlier,
        "sample_s02_outlier": sample_outlier,
        "sample_embedding_row": embedding_rows,
        "sample_embeddings": embeddings,
        "micro_appearance": micro_appearance_values,
        "frame_clip_id": frame_clip,
        "frame_global_time_sec": frame_time,
        "appearance_report": appearance_report,
        "encoder_choice": encoder_choice,
    }
    paths_used = _stage_input_paths(ingest_dir, microtrack_dir, appearance_dir)
    return data, paths_used


def _feature_names(mode: str) -> tuple[str, ...]:
    if mode == "short":
        return SHORT_FEATURE_SCHEMA
    if mode == "long":
        return LONG_FEATURE_SCHEMA
    raise ContractError(f"unsupported S03 mode: {mode}")


def _rows_for(
    rows: Sequence[dict[str, Any]], *, mode: str, split: str | None = None
) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if row["mode"] == mode and (split is None or row["split"] == split)
    ]


def _matrix(rows: Sequence[dict[str, Any]], names: Sequence[str]) -> np.ndarray:
    values = np.asarray(
        [[float(row[name]) for name in names] for row in rows], dtype=np.float64
    )
    if values.shape != (len(rows), len(names)) or not np.all(np.isfinite(values)):
        raise ContractError("S03 pseudo-pair feature matrix is incomplete or non-finite")
    return values


def _labels(rows: Sequence[dict[str, Any]]) -> np.ndarray:
    return np.asarray([bool(row["label"]) for row in rows], dtype=np.int8)


def _effective_sample_requirements(
    config: LinkCalibrationConfig,
    clip_ids: Sequence[str],
) -> dict[str, Any]:
    """Return dataset-level evidence floors scaled by the input clip count.

    Pair generation is allowed to be sparse within any individual clip because
    a frame without a detection row is unobserved, not a negative observation.
    The model is nevertheless calibrated on a dataset-level evidence budget
    that grows with the number of clips.  This prevents a future, larger input
    collection from passing on the same small fixed pool of evidence while
    avoiding a false requirement that every clip contribute every class in
    every time partition.
    """

    normalized_clip_ids = tuple(map(str, clip_ids))
    if (
        not normalized_clip_ids
        or any(not clip_id for clip_id in normalized_clip_ids)
        or len(set(normalized_clip_ids)) != len(normalized_clip_ids)
    ):
        raise ContractError("S03 calibration requires unique non-blank clip IDs")
    clip_count = len(normalized_clip_ids)

    configured = {
        "train": (
            config.min_train_positive_per_model,
            config.min_train_hard_negative_per_model,
        ),
        "calibration": (
            config.min_calibration_positive_per_model,
            config.min_calibration_hard_negative_per_model,
        ),
        "audit": (
            config.min_audit_positive_per_model,
            config.min_audit_hard_negative_per_model,
        ),
    }
    advisory_per_clip = {
        "train": 1,
        "calibration": 1,
        "audit": config.min_audit_pairs_per_class_per_clip,
    }
    splits: dict[str, dict[str, int]] = {}
    for split, (base_positive, base_negative) in configured.items():
        clip_scaled_floor = advisory_per_clip[split] * clip_count
        splits[split] = {
            "positive": max(base_positive, clip_scaled_floor),
            "hard_negative": max(base_negative, clip_scaled_floor),
            "configured_positive": base_positive,
            "configured_hard_negative": base_negative,
            "clip_scaled_floor_per_class": clip_scaled_floor,
            "advisory_per_clip_per_class": advisory_per_clip[split],
        }

    role_fractions = {
        "threshold_selection": config.calibration_selection_fraction,
        "certification": config.calibration_certification_fraction,
    }
    calibration_roles: dict[str, dict[str, int]] = {}
    for role, fraction in role_fractions.items():
        configured_positive = int(
            math.ceil(config.min_calibration_positive_per_model * fraction)
        )
        configured_negative = int(
            math.ceil(config.min_calibration_hard_negative_per_model * fraction)
        )
        calibration_roles[role] = {
            "positive": max(configured_positive, clip_count),
            "hard_negative": max(configured_negative, clip_count),
            "configured_fractional_positive": configured_positive,
            "configured_fractional_hard_negative": configured_negative,
            "clip_scaled_floor_per_class": clip_count,
            "advisory_per_clip_per_class": 1,
        }

    return {
        "evidence_scope": "aggregate_observed_rows",
        "clip_count": clip_count,
        "per_clip_shortfall_action": "warning_only",
        "splits": splits,
        "calibration_roles": calibration_roles,
        "parent_groups_per_split": max(
            config.min_parent_groups_per_split, clip_count
        ),
        "configured_parent_groups_per_split": config.min_parent_groups_per_split,
        "clip_scaled_parent_group_floor": clip_count,
    }


def _minimum_failure_reason(
    rows: Sequence[dict[str, Any]],
    mode: str,
    config: LinkCalibrationConfig,
    clip_ids: Sequence[str],
) -> str | None:
    requirements = _effective_sample_requirements(config, clip_ids)
    for split, minimums in requirements["splits"].items():
        minimum_positive = int(minimums["positive"])
        minimum_negative = int(minimums["hard_negative"])
        selected = _rows_for(rows, mode=mode, split=split)
        positive = sum(bool(row["label"]) for row in selected)
        negative = len(selected) - positive
        if positive < minimum_positive or negative < minimum_negative:
            return (
                f"{split}_samples_below_minimum:positive={positive}/"
                f"{minimum_positive},hard_negative={negative}/{minimum_negative}"
            )
        parent_groups = {int(row["parent_micro_id"]) for row in selected}
        minimum_parent_groups = int(requirements["parent_groups_per_split"])
        if len(parent_groups) < minimum_parent_groups:
            return (
                f"{split}_parent_groups_below_minimum:{len(parent_groups)}/"
                f"{minimum_parent_groups}"
            )
    calibration = _rows_for(rows, mode=mode, split="calibration")
    for role, minimums in requirements["calibration_roles"].items():
        role_rows = [
            row for row in calibration if row["calibration_role"] == role
        ]
        positive = sum(bool(row["label"]) for row in role_rows)
        negative = len(role_rows) - positive
        minimum_positive = int(minimums["positive"])
        minimum_negative = int(minimums["hard_negative"])
        if positive < minimum_positive or negative < minimum_negative:
            return (
                f"calibration_{role}_samples_below_minimum:positive={positive}/"
                f"{minimum_positive},hard_negative={negative}/{minimum_negative}"
            )
    return None


def _candidate_margins(rows: Sequence[dict[str, Any]]) -> np.ndarray:
    margins = np.empty(len(rows), dtype=np.float64)
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[str(row["candidate_group_id"])].append(index)
    for group_id, indices in groups.items():
        if len(indices) < 2:
            raise ContractError(f"S03 candidate group lacks a negative: {group_id}")
        scores = np.asarray(
            [float(rows[index]["model_probability"]) for index in indices],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(scores)):
            raise ContractError("S03 cannot compute a margin from non-finite scores")
        for local, index in enumerate(indices):
            competitors = np.delete(scores, local)
            margins[index] = float(scores[local] - np.max(competitors))
    return margins


def _disabled_thresholds(
    mode: str, reason: str, config: LinkCalibrationConfig
) -> dict[str, Any]:
    return {
        "mode": mode,
        "model_enabled": False,
        "disabled_reason": reason,
        "confirmed_enabled": False,
        "confirmed_threshold": None,
        "provisional_threshold": None,
        "appearance_margin_threshold": None,
        "confirmed_far_target": config.confirmed_false_accept_rate_target,
        "confidence_level": config.threshold_confidence_level,
        "provisional_tpr_target": config.provisional_tpr_target,
        "calibration_positive_count": 0,
        "calibration_hard_negative_count": 0,
        "confirmed_false_accepts": 0,
        "confirmed_false_accept_upper": None,
    }


def _fit_and_score(
    rows: list[dict[str, Any]],
    config: LinkCalibrationConfig,
    clip_ids: Sequence[str],
    *,
    logger: LogFn | None = None,
) -> tuple[dict[str, LinkModelArtifact], dict[str, dict[str, Any]]]:
    models: dict[str, LinkModelArtifact] = {}
    threshold_by_mode: dict[str, dict[str, Any]] = {}
    for mode in ("short", "long"):
        names = _feature_names(mode)
        all_mode = _rows_for(rows, mode=mode)
        reason = _minimum_failure_reason(rows, mode, config, clip_ids)
        if reason is not None:
            if logger is not None:
                logger(f"[s03] {mode} model disabled fail-closed: {reason}")
            model = disabled_link_model(
                names, mode=mode, random_seed=config.random_seed, reason=reason
            )
            models[mode] = model
            disabled_thresholds = _disabled_thresholds(mode, reason, config)
            calibration_rows = _rows_for(
                rows, mode=mode, split="calibration"
            )
            calibration_positive = sum(
                bool(row["label"]) for row in calibration_rows
            )
            calibration_negative = len(calibration_rows) - calibration_positive
            disabled_thresholds["calibration_positive_count"] = calibration_positive
            disabled_thresholds[
                "calibration_hard_negative_count"
            ] = calibration_negative
            if calibration_negative:
                disabled_thresholds[
                    "confirmed_false_accept_upper"
                ] = clopper_pearson_upper(
                    0,
                    calibration_negative,
                    config.threshold_confidence_level,
                )
            threshold_by_mode[mode] = disabled_thresholds
            for row in all_mode:
                row["model_probability"] = None
                row["model_raw_score"] = None
                row["candidate_margin"] = None
                row["decision"] = "reject"
            continue

        train = _rows_for(rows, mode=mode, split="train")
        calibration = _rows_for(rows, mode=mode, split="calibration")
        threshold_selection = [
            row
            for row in calibration
            if row["calibration_role"] == "threshold_selection"
        ]
        certification = [
            row
            for row in calibration
            if row["calibration_role"] == "certification"
        ]
        if logger is not None:
            logger(f"[s03] fitting {mode} LogisticRegression on {len(train):,} train pairs")
        model = fit_link_model(
            _matrix(train, names),
            _labels(train),
            names,
            mode=mode,
            random_seed=config.random_seed,
            logistic_c=config.model_regularization_c,
            max_iter=config.model_max_iterations,
            minimum_per_class=min(
                config.min_train_positive_per_model,
                config.min_train_hard_negative_per_model,
            ),
            tolerance=config.model_tolerance,
            fit_intercept=config.model_fit_intercept,
        )
        models[mode] = model
        probability = model.probabilities(_matrix(all_mode, names))
        raw_score = model.raw_scores(_matrix(all_mode, names))
        for row, probability_value, raw_value in zip(
            all_mode, probability, raw_score, strict=True
        ):
            row["model_probability"] = float(probability_value)
            row["model_raw_score"] = float(raw_value)
        margins = _candidate_margins(all_mode)
        for row, margin in zip(all_mode, margins, strict=True):
            row["candidate_margin"] = float(margin)

        selection_probability = np.asarray(
            [row["model_probability"] for row in threshold_selection],
            dtype=np.float64,
        )
        selection_labels = _labels(threshold_selection)
        certification_probability = np.asarray(
            [row["model_probability"] for row in certification], dtype=np.float64
        )
        certification_labels = _labels(certification)
        calibrated = calibrate_mode_thresholds(
            selection_probability,
            selection_labels,
            selection_labels == 0,
            mode=mode,
            confirmed_far_target=config.confirmed_false_accept_rate_target,
            confidence_level=config.threshold_confidence_level,
            provisional_tpr_target=config.provisional_tpr_target,
            candidate_margins=(
                np.asarray(
                    [row["candidate_margin"] for row in threshold_selection],
                    dtype=np.float64,
                )
                if mode == "long"
                else None
            ),
            certification_probabilities=certification_probability,
            certification_labels=certification_labels,
            certification_hard_negative_mask=certification_labels == 0,
            certification_candidate_margins=(
                np.asarray(
                    [row["candidate_margin"] for row in certification],
                    dtype=np.float64,
                )
                if mode == "long"
                else None
            ),
        )
        calibrated["model_enabled"] = True
        calibrated["disabled_reason"] = None
        threshold_by_mode[mode] = calibrated
        if logger is not None:
            logger(
                f"[s03] calibrated {mode}: confirmed_enabled="
                f"{calibrated['confirmed_enabled']} provisional="
                f"{calibrated['provisional_threshold']:.6f}"
            )
        scorer = LinkScorer(model, calibrated)
        for row in all_mode:
            feature_values = {name: float(row[name]) for name in names}
            result = scorer.score_features(
                feature_values,
                appearance_present=bool(row["appearance_present"]),
                high_overlap=bool(row["high_overlap"]),
                candidate_margin=(
                    float(row["candidate_margin"]) if mode == "long" else None
                ),
            )
            row["decision"] = result.decision
    return models, threshold_by_mode


def _stratified_counts(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    dimensions = (
        "split",
        "calibration_role",
        "mode",
        "source_clip_id",
        "target_clip_id",
        "stratum",
        "decision",
    )
    for dimension in dimensions:
        result[dimension] = dict(
            sorted(Counter(str(row[dimension]) for row in rows).items())
        )
    result["label"] = {
        "positive": sum(bool(row["label"]) for row in rows),
        "hard_negative": sum(not bool(row["label"]) for row in rows),
    }
    return result


def _audit_metrics(
    rows: Sequence[dict[str, Any]],
    threshold_by_mode: Mapping[str, Mapping[str, Any]],
    config: LinkCalibrationConfig,
    clip_ids: Sequence[str],
) -> dict[str, Any]:
    evidence_requirements = _effective_sample_requirements(config, clip_ids)

    def decision_metrics(selected: Sequence[dict[str, Any]]) -> dict[str, Any]:
        labels = np.asarray([bool(row["label"]) for row in selected], dtype=np.bool_)
        decisions = np.asarray([row["decision"] for row in selected], dtype=object)
        negatives = ~labels
        confirmed = decisions == "confirmed"
        provisional_or_better = confirmed | (decisions == "provisional")
        values: dict[str, Any] = {
            "num_pairs": len(selected),
            "num_positive": int(np.count_nonzero(labels)),
            "num_hard_negative": int(np.count_nonzero(negatives)),
            "confirmed_true_accepts": int(np.count_nonzero(confirmed & labels)),
            "confirmed_false_accepts": int(np.count_nonzero(confirmed & negatives)),
            "provisional_or_better_true_accepts": int(
                np.count_nonzero(provisional_or_better & labels)
            ),
            "provisional_or_better_false_accepts": int(
                np.count_nonzero(provisional_or_better & negatives)
            ),
        }
        if np.count_nonzero(negatives):
            values["confirmed_false_accept_upper_95"] = clopper_pearson_upper(
                values["confirmed_false_accepts"],
                values["num_hard_negative"],
                config.threshold_confidence_level,
            )
        return values

    report: dict[str, Any] = {}
    for mode in ("short", "long"):
        mode_report: dict[str, Any] = {
            "evidence_scope": evidence_requirements["evidence_scope"],
            "aggregate_evidence_requirements": evidence_requirements,
            "aggregate_evidence": {},
            "per_clip_evidence_warnings": [],
        }
        threshold = threshold_by_mode[mode]
        per_split_per_clip: dict[str, Any] = {}
        for split in ("train", "calibration", "audit"):
            selected = _rows_for(rows, mode=mode, split=split)
            aggregate_metrics = decision_metrics(selected)
            mode_report[split] = aggregate_metrics
            minimums = evidence_requirements["splits"][split]
            mode_report["aggregate_evidence"][split] = {
                "positive": aggregate_metrics["num_positive"],
                "hard_negative": aggregate_metrics["num_hard_negative"],
                "required_positive": minimums["positive"],
                "required_hard_negative": minimums["hard_negative"],
                "meets_minimum": bool(
                    aggregate_metrics["num_positive"] >= minimums["positive"]
                    and aggregate_metrics["num_hard_negative"]
                    >= minimums["hard_negative"]
                ),
            }
            per_split_per_clip[split] = {}
            for clip_id in clip_ids:
                clip_rows = [
                    row
                    for row in selected
                    if row["source_clip_id"] == clip_id
                    or row["target_clip_id"] == clip_id
                ]
                strata = sorted({str(row["stratum"]) for row in clip_rows})
                clip_metrics = decision_metrics(clip_rows)
                advisory_minimum = minimums["advisory_per_clip_per_class"]
                meets_advisory = bool(
                    clip_metrics["num_positive"] >= advisory_minimum
                    and clip_metrics["num_hard_negative"] >= advisory_minimum
                )
                per_split_per_clip[split][clip_id] = {
                    "counts": _stratified_counts(clip_rows),
                    "metrics": clip_metrics,
                    "metrics_by_stratum": {
                        stratum: decision_metrics(
                            [row for row in clip_rows if row["stratum"] == stratum]
                        )
                        for stratum in strata
                    },
                    "evidence": {
                        "observed": bool(clip_rows),
                        "positive": clip_metrics["num_positive"],
                        "hard_negative": clip_metrics["num_hard_negative"],
                        "advisory_minimum_per_class": advisory_minimum,
                        "meets_advisory_minimum": meets_advisory,
                        "shortfall_action": "warning_only",
                    },
                }
                if not meets_advisory:
                    mode_report["per_clip_evidence_warnings"].append(
                        {
                            "code": "per_clip_evidence_below_advisory",
                            "mode": mode,
                            "split": split,
                            "calibration_role": "not_applicable",
                            "clip_id": clip_id,
                            "positive": clip_metrics["num_positive"],
                            "hard_negative": clip_metrics["num_hard_negative"],
                            "advisory_minimum_per_class": advisory_minimum,
                            "model_enablement_effect": "none_when_aggregate_minimum_is_met",
                        }
                    )
        mode_report["per_split_per_clip"] = per_split_per_clip
        mode_report["audit_per_clip"] = per_split_per_clip["audit"]
        calibration = _rows_for(rows, mode=mode, split="calibration")
        calibration_role_evidence: dict[str, Any] = {}
        for role, minimums in evidence_requirements["calibration_roles"].items():
            role_rows = [
                row for row in calibration if row["calibration_role"] == role
            ]
            role_metrics = decision_metrics(role_rows)
            calibration_role_evidence[role] = {
                "positive": role_metrics["num_positive"],
                "hard_negative": role_metrics["num_hard_negative"],
                "required_positive": minimums["positive"],
                "required_hard_negative": minimums["hard_negative"],
                "meets_minimum": bool(
                    role_metrics["num_positive"] >= minimums["positive"]
                    and role_metrics["num_hard_negative"]
                    >= minimums["hard_negative"]
                ),
            }
            advisory_minimum = minimums["advisory_per_clip_per_class"]
            for clip_id in clip_ids:
                clip_role_rows = [
                    row
                    for row in role_rows
                    if row["source_clip_id"] == clip_id
                    or row["target_clip_id"] == clip_id
                ]
                role_positive = sum(bool(row["label"]) for row in clip_role_rows)
                role_negative = len(clip_role_rows) - role_positive
                if (
                    role_positive < advisory_minimum
                    or role_negative < advisory_minimum
                ):
                    mode_report["per_clip_evidence_warnings"].append(
                        {
                            "code": "per_clip_calibration_role_evidence_below_advisory",
                            "mode": mode,
                            "split": "calibration",
                            "calibration_role": role,
                            "clip_id": clip_id,
                            "positive": role_positive,
                            "hard_negative": role_negative,
                            "advisory_minimum_per_class": advisory_minimum,
                            "model_enablement_effect": "none_when_aggregate_minimum_is_met",
                        }
                    )
        mode_report["calibration_role_evidence"] = calibration_role_evidence
        audit = _rows_for(rows, mode=mode, split="audit")
        mode_report["audit_by_stratum"] = {
            stratum: decision_metrics(
                [row for row in audit if row["stratum"] == stratum]
            )
            for stratum in sorted({str(row["stratum"]) for row in audit})
        }
        mode_report["thresholds"] = dict(threshold)
        mode_report["confirmed_gate"] = {
            "fail_closed": True,
            "evidence_scope": "aggregate_observed_rows_independent_certification",
            "enabled": bool(threshold["confirmed_enabled"]),
            "disabled_reason": threshold.get("confirmed_disabled_reason"),
        }
        report[mode] = mode_report
    return report


def _pair_table(rows: Sequence[dict[str, Any]]) -> pa.Table:
    normalized: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        for name in S03_ALL_FEATURES:
            item.setdefault(name, None)
        item.setdefault("model_probability", None)
        item.setdefault("model_raw_score", None)
        item.setdefault("candidate_margin", None)
        item.setdefault("decision", "reject")
        normalized.append(item)
    try:
        return pa.Table.from_pylist(normalized, schema=PSEUDO_PAIRS_SCHEMA)
    except (pa.ArrowException, TypeError, ValueError) as exc:
        raise ContractError(f"cannot construct S03 pseudo-pair table: {exc}") from exc


def _write_parquet(path: Path, rows: Sequence[dict[str, Any]], compression: str) -> None:
    table = _pair_table(rows)
    try:
        pq.write_table(table, path, compression=compression, version="2.6")
    except (OSError, pa.ArrowException) as exc:
        raise ContractError(f"cannot write S03 Parquet {path}: {exc}") from exc


def _feature_schema_payload() -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "short": {"ordered_features": list(SHORT_FEATURE_SCHEMA)},
        "long": {"ordered_features": list(LONG_FEATURE_SCHEMA)},
        "numeric_dtype": "float64_fit_float32_artifact",
        "missing_appearance_policy": "reject_without_scoring",
        "high_overlap_policy": "at_most_provisional",
        "identity_feature_policy": "anonymous_clean_appearance_and_relative_geometry_only",
    }


def _threshold_payload(
    threshold_by_mode: Mapping[str, Mapping[str, Any]],
    config: LinkCalibrationConfig,
) -> dict[str, Any]:
    short = threshold_by_mode["short"]
    long = threshold_by_mode["long"]
    return {
        "schema_version": "1.0",
        "short_model_enabled": bool(short["model_enabled"]),
        "short_confirmed_enabled": bool(short["confirmed_enabled"]),
        "short_confirmed_threshold": short["confirmed_threshold"],
        "short_provisional_threshold": short["provisional_threshold"],
        "long_model_enabled": bool(long["model_enabled"]),
        "long_confirmed_enabled": bool(long["confirmed_enabled"]),
        "long_confirmed_threshold": long["confirmed_threshold"],
        "long_provisional_threshold": long["provisional_threshold"],
        "appearance_margin_threshold": long["appearance_margin_threshold"],
        "aggressive_global_merge_allowed": False,
        "missing_appearance_decision": "reject",
        "high_overlap_max_decision": "provisional",
        "short": dict(short),
        "long": dict(long),
        "confirmed_false_accept_rate_target": config.confirmed_false_accept_rate_target,
        "confidence_level": config.threshold_confidence_level,
    }


def _expected_artifact_names(config: LinkCalibrationConfig) -> set[str]:
    artifacts = config.artifacts
    return {
        artifacts.link_model_short,
        artifacts.link_model_long,
        artifacts.thresholds,
        artifacts.pair_feature_schema,
        artifacts.calibration_report,
        artifacts.pseudo_pairs_train,
        artifacts.pseudo_pairs_calibration,
        artifacts.pseudo_pairs_audit,
        artifacts.effective_config,
    }


def _validate_completed_output(
    output_dir: Path,
    success: Mapping[str, Any],
    input_fingerprints: Sequence[Mapping[str, Any]],
    config_hash: str,
    config: LinkCalibrationConfig,
) -> None:
    if success.get("stage") != "S03" or success.get("config_hash") != config_hash:
        raise ContractError("completed S03 marker belongs to a different stage/config")
    recorded_inputs = {
        str(item.get("path")): item
        for item in success.get("input_fingerprints", [])
        if isinstance(item, dict)
    }
    current_inputs = {str(item["path"]): item for item in input_fingerprints}
    if set(recorded_inputs) != set(current_inputs):
        raise ContractError("completed S03 input fingerprint path set changed")
    for path, current in current_inputs.items():
        for key in ("size_bytes", "sha256"):
            if recorded_inputs[path].get(key) != current[key]:
                raise ContractError(f"completed S03 input changed: {path} ({key})")
    records = success.get("output_fingerprints")
    if not isinstance(records, list):
        raise ContractError("completed S03 marker lacks output fingerprints")
    recorded_outputs = {
        str(item.get("path")): item for item in records if isinstance(item, dict)
    }
    expected = _expected_artifact_names(config)
    if set(recorded_outputs) != expected or len(recorded_outputs) != len(records):
        raise ContractError("completed S03 output fingerprint set differs")
    actual = {
        str(path.relative_to(output_dir))
        for path in output_dir.rglob("*")
        if path.is_file() and path.name != config.artifacts.success
    }
    if actual != expected:
        raise ContractError("completed S03 artifact file set differs")
    for name, recorded in recorded_outputs.items():
        current = _fingerprint(output_dir / name, relative_to=output_dir)
        for key in ("size_bytes", "sha256"):
            if current[key] != recorded.get(key):
                raise ContractError(f"completed S03 artifact changed: {name} ({key})")
    loaded_models: dict[str, LinkModelArtifact] = {}
    for mode, name in (
        ("short", config.artifacts.link_model_short),
        ("long", config.artifacts.link_model_long),
    ):
        model = load_link_model(output_dir / name)
        if model.mode != mode or model.feature_names != _feature_names(mode):
            raise ContractError(f"completed S03 {mode} model schema differs")
        loaded_models[mode] = model
    for split, name in (
        ("train", config.artifacts.pseudo_pairs_train),
        ("calibration", config.artifacts.pseudo_pairs_calibration),
        ("audit", config.artifacts.pseudo_pairs_audit),
    ):
        parquet = pq.ParquetFile(output_dir / name)
        if not parquet.schema_arrow.equals(PSEUDO_PAIRS_SCHEMA, check_metadata=False):
            raise ContractError(f"completed S03 {split} pair schema differs")
        table = pq.read_table(
            output_dir / name, columns=["split", "calibration_role"]
        )
        observed_splits = set(map(str, table["split"].to_pylist()))
        if not observed_splits.issubset({split}):
            raise ContractError(f"completed S03 {split} file contains another split")
        observed_roles = set(map(str, table["calibration_role"].to_pylist()))
        expected_roles = (
            {"threshold_selection", "certification"}
            if split == "calibration"
            else {"not_applicable"}
        )
        if not observed_roles.issubset(expected_roles):
            raise ContractError(f"completed S03 {split} calibration role differs")
    threshold_payload = _read_json(
        output_dir / config.artifacts.thresholds, "S03 thresholds"
    )
    persisted_feature_schema = _read_json(
        output_dir / config.artifacts.pair_feature_schema,
        "S03 pair feature schema",
    )
    report = _read_json(
        output_dir / config.artifacts.calibration_report, "S03 calibration report"
    )
    if (
        not isinstance(threshold_payload, dict)
        or threshold_payload.get("aggressive_global_merge_allowed") is not False
        or not isinstance(report, dict)
        or report.get("aggressive_global_merge_allowed") is not False
    ):
        raise ContractError("completed S03 aggressive-merge policy differs")
    if persisted_feature_schema != _feature_schema_payload():
        raise ContractError("completed S03 pair feature schema payload differs")
    for mode in ("short", "long"):
        mode_thresholds = threshold_payload.get(mode)
        if not isinstance(mode_thresholds, dict):
            raise ContractError(f"completed S03 thresholds lack {mode} policy")
        LinkScorer(loaded_models[mode], mode_thresholds)


def _verify_inputs_unchanged(fingerprints: Sequence[Mapping[str, Any]]) -> None:
    for expected in fingerprints:
        current = _fingerprint(Path(str(expected["path"])))
        for key in ("size_bytes", "sha256"):
            if current[key] != expected[key]:
                raise ContractError(
                    f"S03 input changed during calibration: {expected['path']} ({key})"
                )


def _canonical_pair_rows(pairs: Sequence[Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pair in pairs:
        if not hasattr(pair, "as_row"):
            raise ContractError("S03 pseudo-pair object lacks as_row()")
        row = dict(pair.as_row())
        if float(row.get("gap_sec", -1.0)) <= 0.0:
            raise ContractError("S03 pseudo-pair contains a non-positive gap")
        if row.get("appearance_present") is not True:
            raise ContractError("S03 fitted pseudo-pair lacks clean appearance")
        split = str(row.get("split"))
        calibration_role = str(row.get("calibration_role"))
        if (
            split == "calibration"
            and calibration_role not in {"threshold_selection", "certification"}
        ) or (
            split in {"train", "audit"}
            and calibration_role != "not_applicable"
        ):
            raise ContractError("S03 pseudo-pair split/calibration role is inconsistent")
        rows.append(row)
    rows.sort(
        key=lambda row: (
            str(row["split"]),
            str(row["calibration_role"]),
            str(row["mode"]),
            str(row["candidate_group_id"]),
            -int(bool(row["label"])),
            int(row["hard_negative_rank"]),
            str(row["pair_id"]),
        )
    )
    if not rows:
        return []
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    parent_assignments: dict[int, set[tuple[str, str]]] = defaultdict(set)
    for row in rows:
        groups[str(row["candidate_group_id"])].append(row)
        parent_assignments[int(row["parent_micro_id"])].add(
            (str(row["split"]), str(row["calibration_role"]))
        )
    for parent_micro_id, assignments in parent_assignments.items():
        if len(assignments) != 1:
            raise ContractError(
                f"S03 parent {parent_micro_id} crosses split/calibration role"
            )
    for group_id, group_rows in groups.items():
        if sum(bool(row["label"]) for row in group_rows) != 1 or not any(
            not bool(row["label"]) for row in group_rows
        ):
            raise ContractError(f"S03 candidate group is not 1-positive/N-negative: {group_id}")
        gaps = np.asarray([float(row["gap_sec"]) for row in group_rows])
        if float(np.max(gaps) - np.min(gaps)) > 1e-6:
            raise ContractError(f"S03 positive/negative gap mismatch in group: {group_id}")
        parent_splits = {
            (
                int(row["parent_micro_id"]),
                str(row["split"]),
                str(row["calibration_role"]),
            )
            for row in group_rows
        }
        if len(parent_splits) != 1:
            raise ContractError(f"S03 parent group crosses split: {group_id}")
    return rows


def _build_calibration_input(data: Mapping[str, Any]) -> Any:
    """Bridge the strict stage loader to the columnar pseudo-pair generator."""

    from cowtrack.linking.pseudo_pairs import CalibrationInput

    summary = data["micro_summary"]
    parent_ids = np.asarray(summary["micro_id"], dtype=np.int64)
    appearance = data["micro_appearance"]
    appearance_ids = np.asarray(appearance["micro_id"], dtype=np.int64)
    appearance_order = np.argsort(appearance_ids, kind="stable")
    sorted_appearance_ids = appearance_ids[appearance_order]
    locations = np.searchsorted(sorted_appearance_ids, parent_ids)
    if np.any(locations >= len(sorted_appearance_ids)) or not np.array_equal(
        sorted_appearance_ids[locations], parent_ids
    ):
        raise ContractError("S03 cannot align parent and appearance summaries")
    appearance_rows = appearance_order[locations]
    frame_clips = np.asarray(data["frame_clip_id"], dtype=object)
    frame_times = np.asarray(data["frame_global_time_sec"], dtype=np.float64)
    timeline_clips = np.asarray(
        tuple(dict.fromkeys(map(str, frame_clips))), dtype=object
    )
    timeline_starts = np.asarray(
        [np.min(frame_times[frame_clips == clip_id]) for clip_id in timeline_clips],
        dtype=np.float64,
    )
    timeline_ends = np.asarray(
        [np.max(frame_times[frame_clips == clip_id]) for clip_id in timeline_clips],
        dtype=np.float64,
    )
    return CalibrationInput(
        timeline_clip_ids=timeline_clips,
        timeline_clip_start_time_sec=timeline_starts,
        timeline_clip_end_time_sec=timeline_ends,
        parent_micro_ids=parent_ids,
        parent_status=np.asarray(summary["status"], dtype=object),
        parent_num_detections=np.asarray(summary["num_detections"], dtype=np.int64),
        parent_local_purity_score=np.asarray(
            summary["local_purity_score"], dtype=np.float64
        ),
        parent_bidirectional_agreement=np.asarray(
            summary["bidirectional_agreement"], dtype=np.float64
        ),
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
        det_other_bbox_max_iou=np.asarray(
            data["other_bbox_max_iou"], dtype=np.float32
        ),
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


def run_s03(
    ingest_dir: Path,
    microtrack_dir: Path,
    appearance_dir: Path,
    config_path: Path,
    output_dir: Path,
    *,
    logger: LogFn = log,
) -> dict[str, Any]:
    """Build S03 pseudo pairs, models, thresholds, and held-out audit reports."""

    started = time.monotonic()
    ingest_dir = ingest_dir.resolve()
    microtrack_dir = microtrack_dir.resolve()
    appearance_dir = appearance_dir.resolve()
    config_path = config_path.resolve()
    output_dir = output_dir.resolve()
    config, config_payload, config_hash = load_link_calibration_config(config_path)

    logger("[s03] fingerprinting immutable S00/S01/S02 inputs")
    paths_used = _stage_input_paths(ingest_dir, microtrack_dir, appearance_dir)
    input_fingerprints = [_fingerprint(path) for path in [config_path, *paths_used]]
    fingerprints_by_path = {
        str(Path(str(item["path"])).resolve()): item for item in input_fingerprints
    }
    _verify_upstream_fingerprints(
        ingest_dir,
        "S00",
        ("frames.parquet", "detections.parquet"),
        fingerprints_by_path,
    )
    _verify_upstream_fingerprints(
        microtrack_dir,
        "S01",
        ("det_to_micro.parquet", "microtracklets.parquet"),
        fingerprints_by_path,
    )
    _verify_upstream_fingerprints(
        appearance_dir,
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
        fingerprints_by_path,
    )
    success_path = output_dir / config.artifacts.success
    if success_path.is_file():
        existing = _read_json(success_path, "S03 success marker")
        if not isinstance(existing, dict):
            raise ContractError("S03 success marker must be a JSON object")
        _validate_completed_output(
            output_dir, existing, input_fingerprints, config_hash, config
        )
        logger(f"[s03] already complete and fully revalidated: {success_path}")
        return dict(existing)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ContractError(
            f"output directory is non-empty without _SUCCESS.json: {output_dir}"
        )

    logger("[s03] validating and loading S00/S01/S02 tables")
    production_inputs = load_production_inputs(
        ingest_dir, microtrack_dir, appearance_dir, logger=logger
    )
    if [path.resolve() for path in production_inputs.consumed_paths] != [
        path.resolve() for path in paths_used
    ]:
        raise ContractError("S03 internal consumed-input path set changed")

    from cowtrack.linking.pseudo_pairs import generate_pseudo_pairs

    logger("[s03] assigning parent/time-block splits and generating pseudo pairs")
    calibration_input = production_inputs.calibration_input
    clean_gallery_status: dict[int, bool] = {}
    pairs = generate_pseudo_pairs(
        calibration_input,
        config,
        logger=logger,
        progress_interval_sec=config.progress_interval_sec,
        clean_gallery_status=clean_gallery_status,
    )
    if set(clean_gallery_status) != set(
        map(int, calibration_input.parent_micro_ids)
    ):
        raise ContractError("S03 clean-gallery population is incomplete")
    rows = _canonical_pair_rows(pairs)
    clip_ids = list(map(str, calibration_input.timeline_clip_ids))
    if tuple(clip_ids) != EXPECTED_CLIP_IDS:
        raise ContractError(
            f"fixed S03 requires clips {list(EXPECTED_CLIP_IDS)}, got {clip_ids}"
        )
    logger(f"[s03] generated {len(rows):,} leakage-safe pseudo pairs")

    evidence_requirements = _effective_sample_requirements(config, clip_ids)
    audit_requirements = evidence_requirements["splits"]["audit"]
    logger(
        "[s03] calibration evidence policy: "
        f"scope={evidence_requirements['evidence_scope']}, "
        f"clips={evidence_requirements['clip_count']}, "
        f"audit_positive>={audit_requirements['positive']}, "
        f"audit_hard_negative>={audit_requirements['hard_negative']}; "
        "per-clip shortfalls are warnings"
    )

    models, threshold_by_mode = _fit_and_score(
        rows, config, clip_ids, logger=logger
    )
    thresholds = _threshold_payload(threshold_by_mode, config)
    audit = _audit_metrics(rows, threshold_by_mode, config, clip_ids)
    appearance_usable = np.asarray(
        production_inputs.micro_appearance["appearance_usable"], dtype=np.bool_
    )
    appearance_ids = np.asarray(
        production_inputs.micro_appearance["micro_id"], dtype=np.int64
    )
    usable_by_micro = {
        int(micro_id): bool(usable)
        for micro_id, usable in zip(appearance_ids, appearance_usable, strict=True)
    }
    s02_appearance_by_clip: dict[str, dict[str, int]] = {}
    clean_gallery_by_clip: dict[str, dict[str, int]] = {}
    det_micro = np.asarray(production_inputs.detections.micro_ids, dtype=np.int64)
    det_clip = np.asarray(production_inputs.detections.clip_ids, dtype=object)
    for clip_id in clip_ids:
        clip_micros = set(map(int, np.unique(det_micro[det_clip == clip_id])))
        present = sum(usable_by_micro[micro_id] for micro_id in clip_micros)
        s02_appearance_by_clip[clip_id] = {
            "present": int(present),
            "missing": int(len(clip_micros) - present),
        }
        clean_present = sum(clean_gallery_status[micro_id] for micro_id in clip_micros)
        clean_gallery_by_clip[clip_id] = {
            "present": int(clean_present),
            "missing": int(len(clip_micros) - clean_present),
        }
    report = {
        "schema_version": config.schema_version,
        "stage": "S03",
        "aggressive_global_merge_allowed": False,
        "input_coordinate_system": "raw_encoded_landscape_no_autorotate",
        "video_decoded": False,
        "encoder_run": False,
        "keypoints_used": False,
        "legacy_tracking_id_used": False,
        "review_metadata_used_as_identity_label": False,
        "split_policy": {
            "strategy": config.split_strategy,
            "fractions": config.split_fractions,
            "calibration_subsplit_fractions": {
                "threshold_selection": config.calibration_selection_fraction,
                "certification": config.calibration_certification_fraction,
            },
            "group_by_parent_micro": True,
            "hard_negative_mining_per_split": True,
            "hard_negative_mining_per_calibration_subsplit": True,
            "crossing_parent_assignment": "most_held_out_touched_time_block",
        },
        "evidence_policy": _effective_sample_requirements(config, clip_ids),
        "appearance_policy": {
            "clean_max_other_bbox_iou_exclusive": config.clean_max_other_bbox_iou,
            "minimum_clean_inliers_per_side": config.clean_min_samples_per_side,
            "require_s02_whole_track_prototype_inlier_prefilter": (
                config.require_s02_prototype_inlier
            ),
            "side_local_outlier_recomputed_after_prefilter": True,
            "missing_appearance_decision": "reject",
            "high_overlap_max_decision": "provisional",
        },
        "s02_warning_forwarded": {
            "warnings": production_inputs.appearance_report.get("warnings", []),
            "hard_negative_separation_warning": production_inputs.appearance_report
            .get("stats", {})
            .get("hard_negative_separation_warning"),
            "encoder_selection_metrics": production_inputs.encoder_choice.get(
                "selection_metrics", {}
            ),
        },
        "s02_appearance_population": {
            "present": int(np.count_nonzero(appearance_usable)),
            "missing": int(np.count_nonzero(~appearance_usable)),
            "per_clip": s02_appearance_by_clip,
        },
        "s03_clean_gallery_population": {
            "present": int(sum(clean_gallery_status.values())),
            "missing": int(
                len(clean_gallery_status) - sum(clean_gallery_status.values())
            ),
            "per_clip": clean_gallery_by_clip,
        },
        "pair_counts": _stratified_counts(rows),
        "metrics": audit,
        "thresholds": thresholds,
    }

    final_output = output_dir
    final_output.parent.mkdir(parents=True, exist_ok=True)
    staging = final_output.parent / f".{final_output.name}.staging-{os.getpid()}"
    if staging.exists():
        raise ContractError(f"S03 staging directory already exists: {staging}")
    staging.mkdir(parents=False)
    try:
        artifacts = config.artifacts
        save_link_model(staging / artifacts.link_model_short, models["short"])
        save_link_model(staging / artifacts.link_model_long, models["long"])
        _write_json(staging / artifacts.thresholds, thresholds)
        _write_json(staging / artifacts.pair_feature_schema, _feature_schema_payload())
        _write_json(staging / artifacts.calibration_report, report)
        _write_json(staging / artifacts.effective_config, config_payload)
        for split, artifact_name in (
            ("train", artifacts.pseudo_pairs_train),
            ("calibration", artifacts.pseudo_pairs_calibration),
            ("audit", artifacts.pseudo_pairs_audit),
        ):
            split_rows = [row for row in rows if row["split"] == split]
            _write_parquet(
                staging / artifact_name, split_rows, config.parquet_compression
            )

        _verify_inputs_unchanged(input_fingerprints)
        output_fingerprints = [
            _fingerprint(staging / name, relative_to=staging)
            for name in sorted(_expected_artifact_names(config))
        ]
        stats = {
            "num_pairs": len(rows),
            "num_train_pairs": sum(row["split"] == "train" for row in rows),
            "num_calibration_pairs": sum(
                row["split"] == "calibration" for row in rows
            ),
            "num_audit_pairs": sum(row["split"] == "audit" for row in rows),
            "num_positive_pairs": sum(bool(row["label"]) for row in rows),
            "num_hard_negative_pairs": sum(not bool(row["label"]) for row in rows),
            "short_model_enabled": models["short"].pipeline is not None,
            "long_model_enabled": models["long"].pipeline is not None,
        }
        success = {
            "schema_version": config.schema_version,
            "stage": "S03",
            "config_hash": config_hash,
            "input_fingerprints": input_fingerprints,
            "output_fingerprints": output_fingerprints,
            "stats": stats,
            "elapsed_sec": float(time.monotonic() - started),
        }
        _write_json(staging / artifacts.success, success)
        if final_output.exists():
            try:
                final_output.rmdir()
            except OSError as exc:
                raise ContractError(
                    f"S03 output directory cannot be atomically committed: {final_output}"
                ) from exc
        os.replace(staging, final_output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    logger(f"[s03] complete: {final_output}")
    return success


__all__ = ["run_s03"]
