"""Fail-closed, deterministic S03 pair feature construction.

This module deliberately accepts only anonymous appearance and geometry data.
Legacy tracking IDs, keypoints, and raw absolute position are neither accepted
nor emitted by its public feature builders.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

from cowtrack.appearance.prototypes import build_tracklet_prototypes
from cowtrack.config import ContractError


LinkMode = Literal["short", "long"]

SHORT_MAX_GAP_SEC = 5.0

APPEARANCE_FEATURE_NAMES = (
    "prototype_cosine_max",
    "prototype_cosine_top3_mean",
    "medoid_cosine",
    "mutual_prototype_score",
    "appearance_quality_min",
    "appearance_quality_mean",
)

BASE_MOTION_FEATURE_NAMES = (
    "gap_sec",
    "log1p_gap_sec",
    "log_width_ratio",
    "log_height_ratio",
    "log_area_ratio",
)

SHORT_MOTION_FEATURE_NAMES = (
    "gap_sec",
    "log1p_gap_sec",
    "predicted_center_residual",
    "predicted_iou",
    "log_width_ratio",
    "log_height_ratio",
    "log_area_ratio",
)

LONG_MOTION_FEATURE_NAMES = BASE_MOTION_FEATURE_NAMES

CONTEXT_FEATURE_NAMES = (
    "src_end_max_other_iou",
    "dst_start_max_other_iou",
    "src_end_boundary_distance",
    "dst_start_boundary_distance",
    "is_clip_boundary",
)

SHORT_FEATURE_SCHEMA = (
    APPEARANCE_FEATURE_NAMES
    + SHORT_MOTION_FEATURE_NAMES
    + CONTEXT_FEATURE_NAMES
)
LONG_FEATURE_SCHEMA = (
    APPEARANCE_FEATURE_NAMES
    + LONG_MOTION_FEATURE_NAMES
    + CONTEXT_FEATURE_NAMES
)


@dataclass(frozen=True)
class CleanGallery:
    """A side-local gallery rebuilt only from clean S02 sample embeddings.

    The three masks retain provenance in the caller's input row order.  The
    sample arrays themselves contain only the final side-local inliers and are
    sorted by stable embedding/sample IDs, making prototype construction
    independent of input table order.
    """

    sample_ids: np.ndarray
    det_ids: np.ndarray
    embedding_rows: np.ndarray
    embeddings: np.ndarray
    prototypes: np.ndarray
    medoid_embedding: np.ndarray
    medoid_sample_id: int
    clean_candidate_mask: np.ndarray
    clean_inlier_mask: np.ndarray
    local_outlier_mask: np.ndarray
    appearance_quality: float
    internal_cosine_p10: float
    internal_cosine_p50: float
    internal_cosine_min: float
    num_input_samples: int
    num_overlap_rejected: int
    num_review_excluded: int
    num_local_outliers: int
    max_other_bbox_iou: float
    max_clean_other_bbox_iou: float


@dataclass(frozen=True)
class EndpointGeometry:
    """Normalized geometry at one segment endpoint."""

    time_sec: float
    cx_norm: float
    cy_norm: float
    w_norm: float
    h_norm: float


@dataclass(frozen=True)
class MotionHistory:
    """Source-side observations used for a constant-velocity prediction."""

    time_sec: np.ndarray
    cx_norm: np.ndarray
    cy_norm: np.ndarray
    w_norm: np.ndarray
    h_norm: np.ndarray


@dataclass(frozen=True)
class EndpointContext:
    """Appearance-contamination and boundary metadata for one endpoint."""

    max_other_iou: float
    boundary_distance: float
    clip_id: str


def _one_dimensional_integer(
    values: np.ndarray, *, name: str, length: int
) -> np.ndarray:
    result = np.asarray(values)
    if result.ndim != 1 or len(result) != length or result.dtype.kind not in "iu":
        raise ContractError(f"S03 {name} must be a length-N integer array")
    result = result.astype(np.int64, copy=False)
    if np.unique(result).size != len(result):
        raise ContractError(f"S03 {name} values must be unique")
    return result


def _boolean_mask(values: np.ndarray, *, name: str, length: int) -> np.ndarray:
    result = np.asarray(values)
    if result.ndim != 1 or len(result) != length or result.dtype != np.bool_:
        raise ContractError(f"S03 {name} must be a length-N boolean mask")
    return result


def _unit_embeddings(embeddings: np.ndarray, *, context: str) -> np.ndarray:
    result = np.asarray(embeddings)
    if result.ndim != 2 or result.shape[1] <= 0 or result.dtype.kind != "f":
        raise ContractError(f"S03 {context} must be a floating [N, D] array")
    result = result.astype(np.float32, copy=False)
    if not np.all(np.isfinite(result)):
        raise ContractError(f"S03 {context} must be finite")
    if len(result):
        norms = np.linalg.norm(result, axis=1)
        if not np.allclose(norms, 1.0, rtol=0.0, atol=2e-3):
            raise ContractError(f"S03 {context} must be L2-normalized")
    return result


def _finite_vector(
    values: np.ndarray,
    *,
    name: str,
    length: int,
    lower: float | None = None,
    upper: float | None = None,
) -> np.ndarray:
    result = np.asarray(values)
    if result.ndim != 1 or len(result) != length or result.dtype.kind != "f":
        raise ContractError(f"S03 {name} must be a length-N floating array")
    result = result.astype(np.float32, copy=False)
    if not np.all(np.isfinite(result)):
        raise ContractError(f"S03 {name} must be finite")
    if lower is not None and np.any(result < lower):
        raise ContractError(f"S03 {name} must be at least {lower}")
    if upper is not None and np.any(result > upper):
        raise ContractError(f"S03 {name} must be at most {upper}")
    return result


def _cosine_threshold(value: float | None, *, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not math.isfinite(float(value)):
        raise ContractError(f"S03 {name} must be a finite cosine threshold")
    result = float(value)
    if result < -1.0 or result > 1.0:
        raise ContractError(f"S03 {name} must be in [-1, 1]")
    return result


def build_clean_gallery(
    sample_embeddings: np.ndarray,
    *,
    sample_ids: np.ndarray,
    det_ids: np.ndarray,
    embedding_rows: np.ndarray,
    crop_quality: np.ndarray,
    other_bbox_max_iou: np.ndarray,
    s02_inlier_mask: np.ndarray,
    review_excluded_mask: np.ndarray | None = None,
    max_other_bbox_iou: float = 0.25,
    min_clean_inliers: int = 3,
    outlier_medoid_cosine: float = 0.65,
    outlier_support_cosine: float | None = 0.70,
    new_prototype_cosine: float = 0.92,
) -> CleanGallery | None:
    """Rebuild a clean, side-local gallery or fail closed with ``None``.

    Rows must already refer to exactly one temporal side (A or B).  Original
    S02 outliers, review exclusions, zero-quality crops, and crops at or above
    the overlap threshold cannot enter the local outlier/prototype fit.
    """

    embeddings = _unit_embeddings(sample_embeddings, context="sample embeddings")
    count = len(embeddings)
    sample_ids64 = _one_dimensional_integer(
        sample_ids, name="sample_ids", length=count
    )
    det_ids64 = _one_dimensional_integer(det_ids, name="det_ids", length=count)
    embedding_rows64 = _one_dimensional_integer(
        embedding_rows, name="embedding_rows", length=count
    )
    quality = _finite_vector(
        crop_quality, name="crop_quality", length=count, lower=0.0, upper=1.0
    )
    overlap = _finite_vector(
        other_bbox_max_iou,
        name="other_bbox_max_iou",
        length=count,
        lower=0.0,
        upper=1.0,
    )
    original_inlier = _boolean_mask(
        s02_inlier_mask, name="s02_inlier_mask", length=count
    )
    if review_excluded_mask is None:
        review_excluded = np.zeros(count, dtype=np.bool_)
    else:
        review_excluded = _boolean_mask(
            review_excluded_mask, name="review_excluded_mask", length=count
        )

    if (
        isinstance(max_other_bbox_iou, bool)
        or not math.isfinite(float(max_other_bbox_iou))
        or not 0.0 <= float(max_other_bbox_iou) <= 1.0
    ):
        raise ContractError("S03 max_other_bbox_iou must be finite and in [0, 1]")
    if (
        isinstance(min_clean_inliers, bool)
        or not isinstance(min_clean_inliers, (int, np.integer))
        or int(min_clean_inliers) != 3
    ):
        raise ContractError("S03 min_clean_inliers is fixed at 3")
    medoid_threshold = _cosine_threshold(
        outlier_medoid_cosine, name="outlier_medoid_cosine"
    )
    support_threshold = _cosine_threshold(
        outlier_support_cosine, name="outlier_support_cosine"
    )
    prototype_threshold = _cosine_threshold(
        new_prototype_cosine, name="new_prototype_cosine"
    )
    if medoid_threshold is None or prototype_threshold is None:
        raise ContractError("S03 medoid and prototype cosine thresholds are required")

    threshold = float(max_other_bbox_iou)
    candidate_mask = (
        original_inlier
        & ~review_excluded
        & (overlap < threshold)
        & (quality > 0.0)
    )
    candidate_indices = np.flatnonzero(candidate_mask)
    if len(candidate_indices) < int(min_clean_inliers):
        return None

    # Stable artifact identifiers, rather than input row numbers, break every
    # tie in the reused prototype builder.
    order = np.lexsort(
        (
            sample_ids64[candidate_indices],
            det_ids64[candidate_indices],
            embedding_rows64[candidate_indices],
        )
    )
    canonical_indices = candidate_indices[order]
    local = build_tracklet_prototypes(
        embeddings[canonical_indices],
        np.zeros(len(canonical_indices), dtype=np.int64),
        quality[canonical_indices],
        np.asarray([0], dtype=np.int64),
        min_samples=3,
        max_prototypes=3,
        outlier_medoid_cosine=medoid_threshold,
        outlier_support_cosine=support_threshold,
        new_prototype_cosine=prototype_threshold,
    )

    local_inliers = local.sample_inlier_mask
    if int(np.count_nonzero(local_inliers)) < int(min_clean_inliers):
        return None
    prototype_mask = local.prototype_mask[0]
    if not np.any(prototype_mask):
        return None

    inlier_indices = canonical_indices[local_inliers]
    inlier_embeddings = embeddings[inlier_indices]
    similarity = np.clip(
        inlier_embeddings @ inlier_embeddings.T, -1.0, 1.0
    ).astype(np.float32, copy=False)
    medoid_scores = np.mean(similarity, axis=1, dtype=np.float64)
    best_score = float(np.max(medoid_scores))
    tied = np.flatnonzero(
        np.isclose(medoid_scores, best_score, rtol=0.0, atol=1e-12)
    )
    medoid_local = int(tied[0])

    clean_inlier_mask = np.zeros(count, dtype=np.bool_)
    clean_inlier_mask[inlier_indices] = True
    local_outlier_mask = np.zeros(count, dtype=np.bool_)
    local_outlier_mask[canonical_indices[local.sample_outlier_mask]] = True
    overlap_rejected = original_inlier & ~review_excluded & (overlap >= threshold)

    return CleanGallery(
        sample_ids=sample_ids64[inlier_indices].copy(),
        det_ids=det_ids64[inlier_indices].copy(),
        embedding_rows=embedding_rows64[inlier_indices].copy(),
        embeddings=inlier_embeddings.copy(),
        prototypes=local.prototypes[0, prototype_mask].copy(),
        medoid_embedding=inlier_embeddings[medoid_local].copy(),
        medoid_sample_id=int(sample_ids64[inlier_indices[medoid_local]]),
        clean_candidate_mask=candidate_mask.copy(),
        clean_inlier_mask=clean_inlier_mask,
        local_outlier_mask=local_outlier_mask,
        appearance_quality=float(local.appearance_quality[0]),
        internal_cosine_p10=float(local.internal_cosine_p10[0]),
        internal_cosine_p50=float(local.internal_cosine_p50[0]),
        internal_cosine_min=float(local.internal_cosine_min[0]),
        num_input_samples=count,
        num_overlap_rejected=int(np.count_nonzero(overlap_rejected)),
        num_review_excluded=int(np.count_nonzero(review_excluded)),
        num_local_outliers=int(local.outlier_count[0]),
        max_other_bbox_iou=float(np.max(overlap)) if count else 0.0,
        max_clean_other_bbox_iou=float(np.max(overlap[inlier_indices])),
    )


def _validate_gallery(gallery: CleanGallery, *, side: str) -> None:
    if not isinstance(gallery, CleanGallery):
        raise ContractError(f"S03 {side} appearance gallery is missing")
    count = len(gallery.sample_ids)
    if count < 3:
        raise ContractError(f"S03 {side} gallery has fewer than 3 clean inliers")
    _one_dimensional_integer(gallery.sample_ids, name=f"{side} sample_ids", length=count)
    _one_dimensional_integer(gallery.det_ids, name=f"{side} det_ids", length=count)
    _one_dimensional_integer(
        gallery.embedding_rows, name=f"{side} embedding_rows", length=count
    )
    embeddings = _unit_embeddings(
        gallery.embeddings, context=f"{side} clean embeddings"
    )
    if len(embeddings) != count:
        raise ContractError(f"S03 {side} gallery arrays have inconsistent lengths")
    prototypes = _unit_embeddings(
        gallery.prototypes, context=f"{side} clean prototypes"
    )
    if not 1 <= len(prototypes) <= 3:
        raise ContractError(f"S03 {side} gallery must have 1 to 3 prototypes")
    medoid = np.asarray(gallery.medoid_embedding)
    if medoid.ndim != 1 or medoid.shape[0] != embeddings.shape[1]:
        raise ContractError(f"S03 {side} medoid has the wrong shape")
    _unit_embeddings(medoid[None, :], context=f"{side} medoid")
    if prototypes.shape[1] != embeddings.shape[1]:
        raise ContractError(f"S03 {side} prototype dimension does not match samples")
    if not math.isfinite(float(gallery.appearance_quality)) or not (
        0.0 <= float(gallery.appearance_quality) <= 1.0
    ):
        raise ContractError(f"S03 {side} appearance quality must be in [0, 1]")


def validate_disjoint_sides(source: CleanGallery, target: CleanGallery) -> None:
    """Reject any A/B sample, detection, or embedding-row leakage."""

    _validate_gallery(source, side="source")
    _validate_gallery(target, side="target")
    checks = (
        (source.sample_ids, target.sample_ids, "sample_id"),
        (source.det_ids, target.det_ids, "det_id"),
        (source.embedding_rows, target.embedding_rows, "embedding_row"),
    )
    for left, right, identifier in checks:
        if np.intersect1d(left, right, assume_unique=True).size:
            raise ContractError(f"S03 A/B galleries share {identifier}")


def appearance_pair_features(
    source: CleanGallery, target: CleanGallery
) -> dict[str, float]:
    """Compute the fixed S03 appearance feature schema."""

    validate_disjoint_sides(source, target)
    if source.prototypes.shape[1] != target.prototypes.shape[1]:
        raise ContractError("S03 source/target appearance dimensions differ")
    similarity = np.clip(
        source.prototypes @ target.prototypes.T, -1.0, 1.0
    ).astype(np.float64, copy=False)
    flattened = np.sort(similarity, axis=None)[::-1]
    top_count = min(3, len(flattened))
    medoid_cosine = float(
        np.clip(source.medoid_embedding @ target.medoid_embedding, -1.0, 1.0)
    )
    quality_source = float(source.appearance_quality)
    quality_target = float(target.appearance_quality)
    values = {
        "prototype_cosine_max": float(flattened[0]),
        "prototype_cosine_top3_mean": float(np.mean(flattened[:top_count])),
        "medoid_cosine": medoid_cosine,
        "mutual_prototype_score": float(
            0.5
            * (
                float(np.mean(np.max(similarity, axis=1)))
                + float(np.mean(np.max(similarity, axis=0)))
            )
        ),
        "appearance_quality_min": min(quality_source, quality_target),
        "appearance_quality_mean": 0.5 * (quality_source + quality_target),
    }
    if tuple(values) != APPEARANCE_FEATURE_NAMES or not all(
        math.isfinite(value) for value in values.values()
    ):
        raise ContractError("S03 appearance feature construction failed")
    return values


def _validate_endpoint(endpoint: EndpointGeometry, *, name: str) -> None:
    if not isinstance(endpoint, EndpointGeometry):
        raise ContractError(f"S03 {name} endpoint geometry is missing")
    values = (
        endpoint.time_sec,
        endpoint.cx_norm,
        endpoint.cy_norm,
        endpoint.w_norm,
        endpoint.h_norm,
    )
    if not all(math.isfinite(float(value)) for value in values):
        raise ContractError(f"S03 {name} endpoint geometry must be finite")
    if not 0.0 <= float(endpoint.cx_norm) <= 1.0 or not 0.0 <= float(
        endpoint.cy_norm
    ) <= 1.0:
        raise ContractError(f"S03 {name} endpoint center must be normalized")
    if not 0.0 < float(endpoint.w_norm) <= 1.0 or not 0.0 < float(
        endpoint.h_norm
    ) <= 1.0:
        raise ContractError(f"S03 {name} endpoint size must be normalized and positive")


def _history_state(
    history: MotionHistory, source_end: EndpointGeometry
) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(history, MotionHistory):
        raise ContractError("S03 short motion requires source history arrays")
    arrays = (
        np.asarray(history.time_sec),
        np.asarray(history.cx_norm),
        np.asarray(history.cy_norm),
        np.asarray(history.w_norm),
        np.asarray(history.h_norm),
    )
    lengths = {len(array) for array in arrays if array.ndim == 1}
    if any(array.ndim != 1 or array.dtype.kind not in "fiu" for array in arrays):
        raise ContractError("S03 source motion history must use one-dimensional numeric arrays")
    if len(lengths) != 1 or not lengths or next(iter(lengths)) == 0:
        raise ContractError("S03 source motion history arrays have inconsistent lengths")
    numeric = tuple(array.astype(np.float64, copy=False) for array in arrays)
    if not all(np.all(np.isfinite(array)) for array in numeric):
        raise ContractError("S03 source motion history must be finite")
    times, cx, cy, width, height = numeric
    if np.any((cx < 0.0) | (cx > 1.0)) or np.any((cy < 0.0) | (cy > 1.0)):
        raise ContractError("S03 source motion history centers must be normalized")
    if np.any((width <= 0.0) | (width > 1.0)) or np.any(
        (height <= 0.0) | (height > 1.0)
    ):
        raise ContractError(
            "S03 source motion history sizes must be normalized and positive"
        )
    order = np.argsort(times, kind="stable")
    times = times[order]
    if len(times) > 1 and np.any(np.diff(times) <= 0.0):
        raise ContractError("S03 source motion history times must be unique")
    cx = cx[order]
    cy = cy[order]
    width = width[order]
    height = height[order]
    expected_endpoint = np.asarray(
        [
            source_end.time_sec,
            source_end.cx_norm,
            source_end.cy_norm,
            source_end.w_norm,
            source_end.h_norm,
        ],
        dtype=np.float64,
    )
    actual_endpoint = np.asarray(
        [times[-1], cx[-1], cy[-1], width[-1], height[-1]], dtype=np.float64
    )
    if not np.allclose(
        actual_endpoint, expected_endpoint, rtol=0.0, atol=2e-6
    ):
        raise ContractError("S03 source history does not end at source endpoint")
    state = np.column_stack((cx, cy, np.log(width), np.log(height)))
    return times, state


def _predict_state(
    history: MotionHistory, source_end: EndpointGeometry, target_time: float
) -> np.ndarray:
    times, states = _history_state(history, source_end)
    if len(times) < 2:
        return states[-1].copy()
    centered = times - float(np.mean(times))
    denominator = float(np.dot(centered, centered))
    if denominator <= np.finfo(np.float64).eps:
        raise ContractError("S03 source history cannot define a motion slope")
    slopes = centered @ states / denominator
    prediction = np.mean(states, axis=0) + slopes * (
        float(target_time) - float(np.mean(times))
    )
    # Match the fixed S01 state extrapolator.  Width/height may transiently
    # predict outside the image, but clipping their log-state prevents
    # numerical explosions while leaving that mismatch visible to the model.
    prediction[2:] = np.clip(prediction[2:], -20.0, 2.0)
    if not np.all(np.isfinite(prediction)):
        raise ContractError("S03 motion prediction is non-finite")
    return prediction


def _predicted_iou(prediction: np.ndarray, target: EndpointGeometry) -> float:
    pred_width, pred_height = np.exp(prediction[2:])
    pred_x1 = float(prediction[0] - 0.5 * pred_width)
    pred_y1 = float(prediction[1] - 0.5 * pred_height)
    pred_x2 = float(prediction[0] + 0.5 * pred_width)
    pred_y2 = float(prediction[1] + 0.5 * pred_height)
    dst_x1 = float(target.cx_norm - 0.5 * target.w_norm)
    dst_y1 = float(target.cy_norm - 0.5 * target.h_norm)
    dst_x2 = float(target.cx_norm + 0.5 * target.w_norm)
    dst_y2 = float(target.cy_norm + 0.5 * target.h_norm)
    intersection = max(0.0, min(pred_x2, dst_x2) - max(pred_x1, dst_x1)) * max(
        0.0, min(pred_y2, dst_y2) - max(pred_y1, dst_y1)
    )
    union = float(
        pred_width * pred_height
        + target.w_norm * target.h_norm
        - intersection
    )
    return intersection / union if union > 0.0 else 0.0


def motion_pair_features(
    source_end: EndpointGeometry,
    target_start: EndpointGeometry,
    *,
    mode: LinkMode,
    source_history: MotionHistory | None = None,
) -> dict[str, float]:
    """Compute motion features, omitting constant velocity for long links."""

    _validate_endpoint(source_end, name="source")
    _validate_endpoint(target_start, name="target")
    if mode not in ("short", "long"):
        raise ContractError(f"S03 unsupported link mode: {mode}")
    gap = float(target_start.time_sec - source_end.time_sec)
    if not math.isfinite(gap) or gap <= 0.0:
        raise ContractError("S03 candidate gap_sec must be positive")
    if mode == "short" and gap > SHORT_MAX_GAP_SEC:
        raise ContractError("S03 short link gap_sec must be <= 5")
    if mode == "long" and gap <= SHORT_MAX_GAP_SEC:
        raise ContractError("S03 long link gap_sec must be > 5")

    width_ratio = float(target_start.w_norm / source_end.w_norm)
    height_ratio = float(target_start.h_norm / source_end.h_norm)
    base = {
        "gap_sec": gap,
        "log1p_gap_sec": math.log1p(gap),
    }
    if mode == "short":
        if source_history is None:
            raise ContractError("S03 short motion requires source history arrays")
        prediction = _predict_state(source_history, source_end, target_start.time_sec)
        pred_width, pred_height = np.exp(prediction[2:])
        denominator = 0.5 * (
            float(np.hypot(pred_width, pred_height))
            + float(np.hypot(target_start.w_norm, target_start.h_norm))
        )
        residual = float(
            np.hypot(
                target_start.cx_norm - prediction[0],
                target_start.cy_norm - prediction[1],
            )
            / (denominator + np.finfo(np.float64).eps)
        )
        base["predicted_center_residual"] = residual
        base["predicted_iou"] = float(_predicted_iou(prediction, target_start))
    base.update(
        {
            "log_width_ratio": math.log(width_ratio),
            "log_height_ratio": math.log(height_ratio),
            "log_area_ratio": math.log(width_ratio * height_ratio),
        }
    )
    expected = SHORT_MOTION_FEATURE_NAMES if mode == "short" else LONG_MOTION_FEATURE_NAMES
    if tuple(base) != expected or not all(math.isfinite(value) for value in base.values()):
        raise ContractError("S03 motion feature construction failed")
    return base


def _validate_context(context: EndpointContext, *, name: str) -> None:
    if not isinstance(context, EndpointContext):
        raise ContractError(f"S03 {name} endpoint context is missing")
    if not isinstance(context.clip_id, str) or not context.clip_id:
        raise ContractError(f"S03 {name} clip_id must be a non-empty string")
    for value, field in (
        (context.max_other_iou, "max_other_iou"),
        (context.boundary_distance, "boundary_distance"),
    ):
        if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
            raise ContractError(f"S03 {name} {field} must be finite and in [0, 1]")


def context_pair_features(
    source_end: EndpointContext, target_start: EndpointContext
) -> dict[str, float]:
    """Preserve raw endpoint contamination metadata without a sentinel."""

    _validate_context(source_end, name="source")
    _validate_context(target_start, name="target")
    values = {
        "src_end_max_other_iou": float(source_end.max_other_iou),
        "dst_start_max_other_iou": float(target_start.max_other_iou),
        "src_end_boundary_distance": float(source_end.boundary_distance),
        "dst_start_boundary_distance": float(target_start.boundary_distance),
        "is_clip_boundary": float(source_end.clip_id != target_start.clip_id),
    }
    if tuple(values) != CONTEXT_FEATURE_NAMES:
        raise ContractError("S03 context feature construction failed")
    return values


def has_high_endpoint_overlap(
    source_end: EndpointContext,
    target_start: EndpointContext,
    *,
    threshold: float = 0.25,
) -> bool:
    """Return the explicit gating flag for contaminated pair endpoints."""

    _validate_context(source_end, name="source")
    _validate_context(target_start, name="target")
    if (
        isinstance(threshold, bool)
        or not math.isfinite(float(threshold))
        or not 0.0 <= float(threshold) <= 1.0
    ):
        raise ContractError("S03 overlap gate threshold must be in [0, 1]")
    return bool(
        source_end.max_other_iou >= float(threshold)
        or target_start.max_other_iou >= float(threshold)
    )


def feature_schema(mode: LinkMode) -> tuple[str, ...]:
    if mode == "short":
        return SHORT_FEATURE_SCHEMA
    if mode == "long":
        return LONG_FEATURE_SCHEMA
    raise ContractError(f"S03 unsupported link mode: {mode}")


def pair_features(
    source_gallery: CleanGallery,
    target_gallery: CleanGallery,
    source_end_geometry: EndpointGeometry,
    target_start_geometry: EndpointGeometry,
    source_end_context: EndpointContext,
    target_start_context: EndpointContext,
    *,
    mode: LinkMode,
    source_history: MotionHistory | None = None,
) -> dict[str, float]:
    """Build one complete, schema-ordered SHORT or LONG feature mapping."""

    values = appearance_pair_features(source_gallery, target_gallery)
    values.update(
        motion_pair_features(
            source_end_geometry,
            target_start_geometry,
            mode=mode,
            source_history=source_history,
        )
    )
    values.update(context_pair_features(source_end_context, target_start_context))
    expected = feature_schema(mode)
    if tuple(values) != expected:
        raise ContractError("S03 pair feature order does not match its schema")
    return values


__all__ = [
    "APPEARANCE_FEATURE_NAMES",
    "BASE_MOTION_FEATURE_NAMES",
    "CONTEXT_FEATURE_NAMES",
    "LONG_FEATURE_SCHEMA",
    "LONG_MOTION_FEATURE_NAMES",
    "SHORT_FEATURE_SCHEMA",
    "SHORT_MAX_GAP_SEC",
    "SHORT_MOTION_FEATURE_NAMES",
    "CleanGallery",
    "EndpointContext",
    "EndpointGeometry",
    "LinkMode",
    "MotionHistory",
    "appearance_pair_features",
    "build_clean_gallery",
    "context_pair_features",
    "feature_schema",
    "has_high_endpoint_overlap",
    "motion_pair_features",
    "pair_features",
    "validate_disjoint_sides",
]
