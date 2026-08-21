"""Rebuild clean appearance galleries for finalized stable components.

The builder deliberately accepts raw S02 sample embeddings, never S02 micro
prototypes. Constituent samples are pooled per finalized component and passed
through the exact S03 clean-gallery policy.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
import time
from typing import Any, Callable

import numpy as np

from cowtrack.config import ContractError
from cowtrack.linking.config import LinkCalibrationConfig
from cowtrack.linking.features import CleanGallery, build_clean_gallery
from cowtrack.linking.pseudo_pairs import CalibrationInput


@dataclass(frozen=True)
class StableAppearanceResult:
    """Dense tensors and schema-ready rows in stable-ID order."""

    stable_ids: np.ndarray
    stable_prototypes: np.ndarray
    stable_prototype_mask: np.ndarray
    stable_appearance_rows: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class _PreparedAppearanceInput:
    parent_micro_ids: np.ndarray
    det_ids: np.ndarray
    det_micro_ids: np.ndarray
    det_review_excluded: np.ndarray
    sample_ids: np.ndarray
    sample_micro_ids: np.ndarray
    sample_det_ids: np.ndarray
    sample_crop_quality: np.ndarray
    sample_other_bbox_max_iou: np.ndarray
    sample_s02_inlier: np.ndarray
    sample_embedding_rows: np.ndarray
    sample_review_excluded: np.ndarray
    embeddings: np.ndarray


def _integers(values: object, *, name: str, length: int | None = None) -> np.ndarray:
    result = np.asarray(values)
    if result.ndim != 1 or result.dtype.kind not in "iu":
        raise ContractError(f"stable appearance {name} must be an integer vector")
    if length is not None and len(result) != length:
        raise ContractError(f"stable appearance {name} has an inconsistent length")
    if result.dtype.kind == "u" and len(result):
        if int(np.max(result)) > np.iinfo(np.int64).max:
            raise ContractError(f"stable appearance {name} exceeds signed int64")
    return result.astype(np.int64, copy=False)


def _unique(values: np.ndarray, *, name: str) -> None:
    if len(np.unique(values)) != len(values):
        raise ContractError(f"stable appearance {name} values must be unique")


def _floats(values: object, *, name: str, length: int) -> np.ndarray:
    result = np.asarray(values)
    if result.ndim != 1 or len(result) != length or result.dtype.kind not in "fiu":
        raise ContractError(f"stable appearance {name} must be a numeric vector")
    result = result.astype(np.float32, copy=False)
    if not np.all(np.isfinite(result)):
        raise ContractError(f"stable appearance {name} must be finite")
    return result


def _booleans(values: object, *, name: str, length: int) -> np.ndarray:
    result = np.asarray(values)
    if result.ndim != 1 or len(result) != length or result.dtype != np.bool_:
        raise ContractError(f"stable appearance {name} must be a boolean vector")
    return result


def _validate_policy(config: LinkCalibrationConfig) -> None:
    if not isinstance(config, LinkCalibrationConfig):
        raise ContractError("stable appearance config must be LinkCalibrationConfig")
    fixed = (
        math.isclose(
            float(config.clean_max_other_bbox_iou),
            0.25,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        and config.clean_iou_comparison == "less_than"
        and config.clean_min_samples_per_side == 3
        and config.require_s02_prototype_inlier is True
        and config.exclude_review_detections is True
        and config.prototypes_per_side == 3
        and config.missing_appearance_decision == "reject"
    )
    if not fixed:
        raise ContractError(
            "stable appearance requires the exact fixed S03 clean policy"
        )
    thresholds = (
        config.outlier_medoid_cosine,
        config.outlier_support_cosine,
        config.new_prototype_cosine,
        config.min_internal_cosine_p10,
    )
    try:
        thresholds_are_valid = all(
            not isinstance(value, bool)
            and math.isfinite(float(value))
            and -1.0 <= float(value) <= 1.0
            for value in thresholds
        )
    except (TypeError, ValueError, OverflowError):
        thresholds_are_valid = False
    if not thresholds_are_valid:
        raise ContractError("stable appearance S03 cosine thresholds are invalid")


def _prepare(data: CalibrationInput) -> _PreparedAppearanceInput:
    if not isinstance(data, CalibrationInput):
        raise ContractError("stable appearance input must be CalibrationInput")
    parent_ids = _integers(data.parent_micro_ids, name="parent_micro_ids")
    _unique(parent_ids, name="parent_micro_ids")
    if not len(parent_ids):
        raise ContractError("stable appearance requires at least one parent micro")

    det_ids = _integers(data.det_ids, name="det_ids")
    det_count = len(det_ids)
    _unique(det_ids, name="det_ids")
    det_micro_ids = _integers(
        data.det_micro_ids, name="det_micro_ids", length=det_count
    )
    det_review = _booleans(
        data.det_review_excluded,
        name="det_review_excluded",
        length=det_count,
    )
    unknown_det_micros = sorted(
        set(map(int, det_micro_ids)) - set(map(int, parent_ids))
    )
    if unknown_det_micros:
        raise ContractError(
            "stable appearance detections reference unknown parent micros: "
            f"{unknown_det_micros[:5]}"
        )

    sample_ids = _integers(data.sample_ids, name="sample_ids")
    sample_count = len(sample_ids)
    _unique(sample_ids, name="sample_ids")
    sample_micro_ids = _integers(
        data.sample_micro_ids, name="sample_micro_ids", length=sample_count
    )
    sample_det_ids = _integers(
        data.sample_det_ids, name="sample_det_ids", length=sample_count
    )
    _unique(sample_det_ids, name="sample_det_ids")
    sample_quality = _floats(
        data.sample_crop_quality, name="sample_crop_quality", length=sample_count
    )
    sample_iou = _floats(
        data.sample_other_bbox_max_iou,
        name="sample_other_bbox_max_iou",
        length=sample_count,
    )
    sample_inlier = _booleans(
        data.sample_s02_inlier, name="sample_s02_inlier", length=sample_count
    )
    embedding_rows = _integers(
        data.sample_embedding_rows,
        name="sample_embedding_rows",
        length=sample_count,
    )
    _unique(embedding_rows, name="sample_embedding_rows")
    if np.any((sample_quality < 0.0) | (sample_quality > 1.0)):
        raise ContractError("stable appearance sample crop quality must be in [0, 1]")
    if np.any((sample_iou < 0.0) | (sample_iou > 1.0)):
        raise ContractError("stable appearance sample overlap must be in [0, 1]")

    embeddings = np.asarray(data.embeddings)
    if (
        embeddings.ndim != 2
        or embeddings.shape[1] <= 0
        or embeddings.dtype.kind != "f"
        or len(embeddings) != sample_count
    ):
        raise ContractError(
            "stable appearance embeddings must be floating [num_samples, D]"
        )
    embeddings = embeddings.astype(np.float32, copy=False)
    if not np.all(np.isfinite(embeddings)):
        raise ContractError("stable appearance embeddings must be finite")
    if len(embeddings):
        norms = np.linalg.norm(embeddings, axis=1)
        if not np.allclose(norms, 1.0, rtol=0.0, atol=2e-3):
            raise ContractError("stable appearance embeddings must be L2-normalized")
    if not np.array_equal(
        np.sort(embedding_rows), np.arange(sample_count, dtype=np.int64)
    ):
        raise ContractError(
            "stable appearance sample/embedding rows must be a bijection"
        )

    det_row = {int(det_id): row for row, det_id in enumerate(det_ids)}
    sample_review = np.empty(sample_count, dtype=np.bool_)
    parent_set = set(map(int, parent_ids))
    for row, (det_id, micro_id) in enumerate(
        zip(sample_det_ids, sample_micro_ids, strict=True)
    ):
        det_position = det_row.get(int(det_id))
        if det_position is None:
            raise ContractError(
                f"stable appearance sample references unknown det_id {int(det_id)}"
            )
        if int(micro_id) not in parent_set or int(det_micro_ids[det_position]) != int(
            micro_id
        ):
            raise ContractError(
                "stable appearance sample micro/detection join is inconsistent"
            )
        sample_review[row] = det_review[det_position]

    return _PreparedAppearanceInput(
        parent_micro_ids=parent_ids,
        det_ids=det_ids,
        det_micro_ids=det_micro_ids,
        det_review_excluded=det_review,
        sample_ids=sample_ids,
        sample_micro_ids=sample_micro_ids,
        sample_det_ids=sample_det_ids,
        sample_crop_quality=sample_quality,
        sample_other_bbox_max_iou=sample_iou,
        sample_s02_inlier=sample_inlier,
        sample_embedding_rows=embedding_rows,
        sample_review_excluded=sample_review,
        embeddings=embeddings,
    )


def _stable_mapping(
    parent_micro_ids: np.ndarray,
    micro_to_stable: Mapping[int, int],
) -> tuple[dict[int, int], np.ndarray]:
    if not isinstance(micro_to_stable, Mapping):
        raise ContractError("stable appearance micro_to_stable must be a mapping")
    result: dict[int, int] = {}
    int64 = np.iinfo(np.int64)
    for raw_micro, raw_stable in micro_to_stable.items():
        if (
            isinstance(raw_micro, (bool, np.bool_))
            or not isinstance(raw_micro, (int, np.integer))
            or isinstance(raw_stable, (bool, np.bool_))
            or not isinstance(raw_stable, (int, np.integer))
        ):
            raise ContractError("stable appearance mapping IDs must be integers")
        micro_id = int(raw_micro)
        stable_id = int(raw_stable)
        if not int64.min <= micro_id <= int64.max or not 0 <= stable_id <= int64.max:
            raise ContractError("stable appearance mapping IDs exceed int64 bounds")
        result[micro_id] = stable_id
    expected = set(map(int, parent_micro_ids))
    if set(result) != expected or len(result) != len(parent_micro_ids):
        raise ContractError(
            "stable appearance mapping keys must exactly cover parent micro IDs"
        )
    stable_ids = np.asarray(sorted(set(result.values())), dtype=np.int64)
    if not np.array_equal(stable_ids, np.arange(len(stable_ids), dtype=np.int64)):
        raise ContractError("stable appearance stable IDs must be dense from zero")
    return result, stable_ids


def _gallery_row(
    *,
    stable_id: int,
    constituent_micro_ids: tuple[int, ...],
    positions: np.ndarray,
    data: _PreparedAppearanceInput,
    gallery: CleanGallery | None,
    usable: bool,
    missing_reason: str | None,
    clean_candidate_mask: np.ndarray,
) -> dict[str, Any]:
    overlap_rejected = (
        data.sample_s02_inlier[positions]
        & ~data.sample_review_excluded[positions]
        & (
            data.sample_other_bbox_max_iou[positions]
            >= np.float32(0.25)
        )
    )
    max_iou = (
        float(np.max(data.sample_other_bbox_max_iou[positions]))
        if len(positions)
        else 0.0
    )
    if gallery is None:
        clean_sample_ids: list[int] = []
        clean_det_ids: list[int] = []
        clean_embedding_rows: list[int] = []
        medoid_sample_id: int | None = None
        appearance_quality: float | None = None
        cosine_p10: float | None = None
        cosine_p50: float | None = None
        cosine_min: float | None = None
        local_outliers: int | None = None
        max_clean_iou: float | None = None
    else:
        clean_sample_ids = list(map(int, gallery.sample_ids))
        clean_det_ids = list(map(int, gallery.det_ids))
        clean_embedding_rows = list(map(int, gallery.embedding_rows))
        medoid_sample_id = int(gallery.medoid_sample_id)
        appearance_quality = float(gallery.appearance_quality)
        cosine_p10 = float(gallery.internal_cosine_p10)
        cosine_p50 = float(gallery.internal_cosine_p50)
        cosine_min = float(gallery.internal_cosine_min)
        local_outliers = int(gallery.num_local_outliers)
        max_clean_iou = float(gallery.max_clean_other_bbox_iou)
    return {
        "stable_id": stable_id,
        "prototype_row": stable_id,
        "constituent_micro_ids": list(constituent_micro_ids),
        "num_input_samples": len(positions),
        "num_s02_inliers": int(
            np.count_nonzero(data.sample_s02_inlier[positions])
        ),
        "num_clean_candidates": int(np.count_nonzero(clean_candidate_mask)),
        "num_clean_inliers": len(clean_sample_ids),
        "num_valid_prototypes": (
            len(gallery.prototypes)
            if usable and gallery is not None
            else 0
        ),
        "appearance_usable": usable,
        "missing_reason": missing_reason,
        "clean_sample_ids": clean_sample_ids,
        "clean_det_ids": clean_det_ids,
        "clean_embedding_rows": clean_embedding_rows,
        "medoid_sample_id": medoid_sample_id,
        "appearance_quality": appearance_quality,
        "internal_cosine_p10": cosine_p10,
        "internal_cosine_p50": cosine_p50,
        "internal_cosine_min": cosine_min,
        "num_overlap_rejected": int(np.count_nonzero(overlap_rejected)),
        "num_review_excluded": int(
            np.count_nonzero(data.sample_review_excluded[positions])
        ),
        "num_local_outliers": local_outliers,
        "max_other_bbox_iou": max_iou,
        "max_clean_other_bbox_iou": max_clean_iou,
    }


def build_stable_appearance(
    data: CalibrationInput,
    micro_to_stable: Mapping[int, int],
    config: LinkCalibrationConfig,
    *,
    progress_callback: Callable[[int, int], None] | None = None,
    progress_interval_sec: float = 10.0,
) -> StableAppearanceResult:
    """Pool constituent S02 samples and rebuild every stable clean gallery."""

    if (
        isinstance(progress_interval_sec, bool)
        or not isinstance(progress_interval_sec, (int, float))
        or not math.isfinite(float(progress_interval_sec))
        or float(progress_interval_sec) <= 0.0
    ):
        raise ContractError("stable appearance progress interval must be positive")
    _validate_policy(config)
    prepared = _prepare(data)
    mapping, stable_ids = _stable_mapping(
        prepared.parent_micro_ids, micro_to_stable
    )
    embedding_dim = prepared.embeddings.shape[1]
    prototypes = np.zeros(
        (len(stable_ids), 3, embedding_dim), dtype=np.float16
    )
    prototype_mask = np.zeros((len(stable_ids), 3), dtype=np.bool_)
    rows: list[dict[str, Any]] = []

    stable_by_sample = np.asarray(
        [mapping[int(micro_id)] for micro_id in prepared.sample_micro_ids],
        dtype=np.int64,
    )
    last_progress = time.monotonic()
    total = len(stable_ids)
    for completed, stable_id in enumerate(map(int, stable_ids), start=1):
        constituents = tuple(
            sorted(
                micro_id
                for micro_id, assigned in mapping.items()
                if assigned == stable_id
            )
        )
        positions = np.flatnonzero(stable_by_sample == stable_id)
        local_inlier = prepared.sample_s02_inlier[positions]
        local_review = prepared.sample_review_excluded[positions]
        local_iou = prepared.sample_other_bbox_max_iou[positions]
        local_quality = prepared.sample_crop_quality[positions]
        clean_candidate_mask = (
            local_inlier
            & ~local_review
            & (local_iou < np.float32(config.clean_max_other_bbox_iou))
            & (local_quality > 0.0)
        )
        gallery = build_clean_gallery(
            prepared.embeddings[prepared.sample_embedding_rows[positions]],
            sample_ids=prepared.sample_ids[positions],
            det_ids=prepared.sample_det_ids[positions],
            embedding_rows=prepared.sample_embedding_rows[positions],
            crop_quality=local_quality,
            other_bbox_max_iou=local_iou,
            s02_inlier_mask=local_inlier,
            review_excluded_mask=local_review,
            max_other_bbox_iou=config.clean_max_other_bbox_iou,
            min_clean_inliers=config.clean_min_samples_per_side,
            outlier_medoid_cosine=config.outlier_medoid_cosine,
            outlier_support_cosine=config.outlier_support_cosine,
            new_prototype_cosine=config.new_prototype_cosine,
        )
        if gallery is None:
            usable = False
            missing_reason = "clean_gallery_missing"
        elif gallery.internal_cosine_p10 < config.min_internal_cosine_p10:
            usable = False
            missing_reason = "internal_cosine_p10_below_minimum"
        else:
            usable = True
            missing_reason = None
            count = len(gallery.prototypes)
            prototypes[stable_id, :count] = gallery.prototypes.astype(
                np.float16
            )
            prototype_mask[stable_id, :count] = True
        rows.append(
            _gallery_row(
                stable_id=stable_id,
                constituent_micro_ids=constituents,
                positions=positions,
                data=prepared,
                gallery=gallery,
                usable=usable,
                missing_reason=missing_reason,
                clean_candidate_mask=clean_candidate_mask,
            )
        )
        now = time.monotonic()
        if progress_callback is not None and (
            completed == total
            or now - last_progress >= float(progress_interval_sec)
        ):
            progress_callback(completed, total)
            last_progress = now

    if prototypes.dtype != np.float16 or prototype_mask.dtype != np.bool_:
        raise ContractError("stable appearance tensor dtypes changed")
    if not np.all(np.isfinite(prototypes)):
        raise ContractError("stable appearance prototypes must be finite")
    if np.any(prototypes[~prototype_mask] != 0.0):
        raise ContractError("stable appearance unused prototype slots must be zero")
    valid = prototypes[prototype_mask].astype(np.float32, copy=False)
    if len(valid) and not np.allclose(
        np.linalg.norm(valid, axis=1), 1.0, rtol=0.0, atol=2e-3
    ):
        raise ContractError("stable appearance prototypes must be L2-normalized")
    if [row["stable_id"] for row in rows] != list(range(len(stable_ids))):
        raise ContractError("stable appearance rows lost dense stable-ID order")
    return StableAppearanceResult(
        stable_ids=stable_ids,
        stable_prototypes=prototypes,
        stable_prototype_mask=prototype_mask,
        stable_appearance_rows=tuple(rows),
    )


__all__ = ["StableAppearanceResult", "build_stable_appearance"]
