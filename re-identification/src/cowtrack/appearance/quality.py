"""Deterministic bbox-only crop extraction and quality measurements."""

from __future__ import annotations

import math
import time
from typing import Callable, Sequence

import numpy as np

from cowtrack.config import ContractError


# Fixed production quality mapping.  The public combiner accepts explicit
# overrides so the stage can persist the complete numerical contract in config.
DEFAULT_QUALITY_WEIGHTS = (0.30, 0.25, 0.15, 0.20, 0.10)
DEFAULT_OVERLAP_ZERO_IOU = 0.50
DEFAULT_CLIPPING_ZERO_FRACTION = 0.25
DEFAULT_AREA_FULL_PERCENTILE = 0.50
DEFAULT_BLUR_ZERO_VARIANCE = 20.0
DEFAULT_BLUR_SCALE = 380.0
SOFT_BORDER_WIDTH_FRACTION = 0.20
SOFT_BORDER_MINIMUM_WEIGHT = 0.25


def _boxes_array(boxes: np.ndarray) -> np.ndarray:
    array = np.asarray(boxes, dtype=np.float64)
    if array.ndim != 2 or array.shape[1:] != (4,):
        raise ContractError("S02 boxes must have shape [N, 4]")
    if not np.all(np.isfinite(array)):
        raise ContractError("S02 boxes must be finite")
    if len(array) and not np.all(
        (array[:, 2] > array[:, 0]) & (array[:, 3] > array[:, 1])
    ):
        raise ContractError("S02 boxes must have positive width and height")
    return array


def maximum_other_bbox_iou(
    boxes: np.ndarray,
    frame_ids: np.ndarray,
    *,
    progress: Callable[[int, int], None] | None = None,
    progress_interval_sec: float = 10.0,
) -> np.ndarray:
    """Return each box's maximum IoU with another box in the same frame."""

    boxes = _boxes_array(boxes)
    frame_ids = np.asarray(frame_ids)
    if frame_ids.ndim != 1 or len(frame_ids) != len(boxes):
        raise ContractError("S02 frame_ids must be one-dimensional and match boxes")
    if frame_ids.dtype.kind not in "iu":
        raise ContractError("S02 frame_ids must be an integer array")
    result = np.zeros(len(boxes), dtype=np.float32)
    if not len(boxes):
        return result

    order = np.argsort(frame_ids, kind="stable")
    ordered_frames = frame_ids[order]
    starts = np.flatnonzero(
        np.r_[True, ordered_frames[1:] != ordered_frames[:-1]]
    )
    stops = np.r_[starts[1:], len(order)]
    if not math.isfinite(float(progress_interval_sec)) or progress_interval_sec <= 0.0:
        raise ContractError("S02 IoU progress interval must be positive")
    last_report = time.monotonic()
    total_groups = len(starts)
    for group_index, (start, stop) in enumerate(
        zip(starts, stops, strict=True), start=1
    ):
        now = time.monotonic()
        if progress is not None and now - last_report >= progress_interval_sec:
            progress(group_index - 1, total_groups)
            last_report = now
        group = order[int(start) : int(stop)]
        if len(group) < 2:
            continue
        group_boxes = boxes[group]
        x1 = np.maximum(group_boxes[:, None, 0], group_boxes[None, :, 0])
        y1 = np.maximum(group_boxes[:, None, 1], group_boxes[None, :, 1])
        x2 = np.minimum(group_boxes[:, None, 2], group_boxes[None, :, 2])
        y2 = np.minimum(group_boxes[:, None, 3], group_boxes[None, :, 3])
        intersection = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
        areas = (group_boxes[:, 2] - group_boxes[:, 0]) * (
            group_boxes[:, 3] - group_boxes[:, 1]
        )
        union = areas[:, None] + areas[None, :] - intersection
        if np.any(union <= 0.0):
            raise ContractError("S02 IoU encountered a non-positive union")
        iou = intersection / union
        np.fill_diagonal(iou, -np.inf)
        result[group] = np.max(iou, axis=1).astype(np.float32)
    if progress is not None:
        progress(total_groups, total_groups)
    return result


def bbox_area_percentiles(boxes: np.ndarray) -> np.ndarray:
    """Return deterministic midrank area percentiles in ``[0, 1]``.

    Equal-area boxes receive the same percentile.  A singleton receives 1.0,
    so a sequence containing one usable detection is not penalized merely for
    lacking a comparison population.
    """

    boxes = _boxes_array(boxes)
    count = len(boxes)
    if count == 0:
        return np.empty(0, dtype=np.float32)
    if count == 1:
        return np.ones(1, dtype=np.float32)
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    ordered = np.sort(areas, kind="stable")
    left = np.searchsorted(ordered, areas, side="left").astype(np.float64)
    right = np.searchsorted(ordered, areas, side="right").astype(np.float64) - 1.0
    percentiles = ((left + right) * 0.5) / float(count - 1)
    return percentiles.astype(np.float32)


def _validate_frame(frame: np.ndarray) -> np.ndarray:
    frame = np.asarray(frame)
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ContractError("S02 crop frame must have shape [H, W, 3]")
    if frame.dtype != np.uint8:
        raise ContractError("S02 crop frame must have dtype uint8")
    if frame.shape[0] <= 0 or frame.shape[1] <= 0:
        raise ContractError("S02 crop frame must be non-empty")
    return frame


def _one_box(bbox: np.ndarray | Sequence[float]) -> np.ndarray:
    box = np.asarray(bbox, dtype=np.float64)
    if box.shape != (4,) or not np.all(np.isfinite(box)):
        raise ContractError("S02 crop bbox must contain four finite coordinates")
    if box[2] <= box[0] or box[3] <= box[1]:
        raise ContractError("S02 crop bbox must have positive width and height")
    return box


def _normalized_boundary_distance(
    box: np.ndarray, *, frame_width: int, frame_height: int
) -> float:
    gaps = np.asarray(
        [box[0], box[1], frame_width - box[2], frame_height - box[3]],
        dtype=np.float64,
    )
    if np.any(gaps < -1e-9):
        raise ContractError("S02 crop bbox must be clamped to the input frame")
    short_side = min(float(box[2] - box[0]), float(box[3] - box[1]))
    scale = max(short_side * 0.5, np.finfo(np.float64).eps)
    return float(np.clip(np.min(gaps) / scale, 0.0, 1.0))


def crop_with_padding(
    frame: np.ndarray,
    bbox: np.ndarray | Sequence[float],
    padding_ratio: float,
) -> tuple[np.ndarray, float, float]:
    """Extract a padded RGB crop and report clipping/boundary measurements.

    ``clipped_fraction`` is the fraction of the requested padded rectangle that
    fell outside the image.  ``boundary_distance`` is the closest source-box
    edge distance, normalized to reach 1.0 at half the bbox's shorter side.
    Pixel bounds use floor for the leading edge and ceil for the trailing edge.
    """

    frame = _validate_frame(frame)
    box = _one_box(bbox)
    if not math.isfinite(padding_ratio) or padding_ratio < 0.0:
        raise ContractError("S02 bbox padding_ratio must be finite and non-negative")
    height, width = frame.shape[:2]
    if box[0] < 0.0 or box[1] < 0.0 or box[2] > width or box[3] > height:
        raise ContractError("S02 crop bbox must be clamped to the input frame")

    box_width = float(box[2] - box[0])
    box_height = float(box[3] - box[1])
    requested = np.asarray(
        [
            box[0] - box_width * padding_ratio,
            box[1] - box_height * padding_ratio,
            box[2] + box_width * padding_ratio,
            box[3] + box_height * padding_ratio,
        ],
        dtype=np.float64,
    )
    clipped = np.asarray(
        [
            np.clip(requested[0], 0.0, float(width)),
            np.clip(requested[1], 0.0, float(height)),
            np.clip(requested[2], 0.0, float(width)),
            np.clip(requested[3], 0.0, float(height)),
        ],
        dtype=np.float64,
    )
    requested_area = (requested[2] - requested[0]) * (requested[3] - requested[1])
    retained_area = (clipped[2] - clipped[0]) * (clipped[3] - clipped[1])
    if requested_area <= 0.0 or retained_area <= 0.0:
        raise ContractError("S02 padded crop has non-positive area")
    clipped_fraction = float(np.clip(1.0 - retained_area / requested_area, 0.0, 1.0))
    x1 = max(0, min(width, int(math.floor(clipped[0]))))
    y1 = max(0, min(height, int(math.floor(clipped[1]))))
    x2 = max(0, min(width, int(math.ceil(clipped[2]))))
    y2 = max(0, min(height, int(math.ceil(clipped[3]))))
    if x2 <= x1 or y2 <= y1:
        raise ContractError("S02 padded crop is empty after integer conversion")
    crop = np.ascontiguousarray(frame[y1:y2, x1:x2])
    boundary_distance = _normalized_boundary_distance(
        box, frame_width=width, frame_height=height
    )
    return crop, clipped_fraction, boundary_distance


def soft_border_mask(crop: np.ndarray) -> np.ndarray:
    """Attenuate crop borders and corners while preserving central pixels."""

    crop = _validate_frame(crop)
    height, width = crop.shape[:2]
    y_distance = np.minimum(
        np.arange(height, dtype=np.float32) + 0.5,
        np.arange(height, 0, -1, dtype=np.float32) - 0.5,
    ) / max(float(height) * SOFT_BORDER_WIDTH_FRACTION, 1.0)
    x_distance = np.minimum(
        np.arange(width, dtype=np.float32) + 0.5,
        np.arange(width, 0, -1, dtype=np.float32) - 0.5,
    ) / max(float(width) * SOFT_BORDER_WIDTH_FRACTION, 1.0)
    distance = np.minimum(y_distance[:, None], x_distance[None, :])
    transition = np.clip(distance, 0.0, 1.0)
    transition = transition * transition * (3.0 - 2.0 * transition)
    weights = SOFT_BORDER_MINIMUM_WEIGHT + (
        1.0 - SOFT_BORDER_MINIMUM_WEIGHT
    ) * transition
    masked = np.rint(crop.astype(np.float32) * weights[:, :, None])
    return np.clip(masked, 0.0, 255.0).astype(np.uint8)


def laplacian_blur_score(crop: np.ndarray) -> float:
    """Return RGB grayscale Laplacian variance; larger means sharper."""

    crop = _validate_frame(crop)
    if crop.shape[0] < 3 or crop.shape[1] < 3:
        raise ContractError("S02 crop must be at least 3x3 for blur measurement")
    rgb = crop.astype(np.float32)
    gray = rgb[:, :, 0] * 0.299 + rgb[:, :, 1] * 0.587 + rgb[:, :, 2] * 0.114
    laplacian = (
        gray[:-2, 1:-1]
        + gray[2:, 1:-1]
        + gray[1:-1, :-2]
        + gray[1:-1, 2:]
        - 4.0 * gray[1:-1, 1:-1]
    )
    score = float(np.var(laplacian, dtype=np.float64))
    if not math.isfinite(score):
        raise ContractError("S02 Laplacian blur score is non-finite")
    return score


def _quality_vector(name: str, values: np.ndarray) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim != 1:
        raise ContractError(f"S02 {name} must be one-dimensional")
    if not np.all(np.isfinite(vector)):
        raise ContractError(f"S02 {name} must be finite")
    return vector


def combine_crop_quality(
    other_bbox_max_iou: np.ndarray,
    clipped_fraction: np.ndarray,
    bbox_area_percentile: np.ndarray,
    blur_score: np.ndarray,
    distance_to_image_boundary: np.ndarray,
    *,
    weights: Sequence[float] = DEFAULT_QUALITY_WEIGHTS,
    overlap_zero_iou: float = DEFAULT_OVERLAP_ZERO_IOU,
    clipping_zero_fraction: float = DEFAULT_CLIPPING_ZERO_FRACTION,
    area_full_percentile: float = DEFAULT_AREA_FULL_PERCENTILE,
    blur_zero_variance: float = DEFAULT_BLUR_ZERO_VARIANCE,
    blur_scale: float = DEFAULT_BLUR_SCALE,
) -> np.ndarray:
    """Combine five interpretable measurements into ``crop_quality``.

    Components are linearly clipped except blur, which is mapped in log space.
    The default weighted formula is overlap/clipping/area/blur/boundary =
    0.30/0.25/0.15/0.20/0.10.
    """

    overlap = _quality_vector("other_bbox_max_iou", other_bbox_max_iou)
    count = len(overlap)
    clipping = _quality_vector("clipped_fraction", clipped_fraction)
    area = _quality_vector("bbox_area_percentile", bbox_area_percentile)
    blur = _quality_vector("blur_score", blur_score)
    boundary = _quality_vector(
        "distance_to_image_boundary", distance_to_image_boundary
    )
    if any(len(vector) != count for vector in (clipping, area, blur, boundary)):
        raise ContractError("S02 crop-quality columns have inconsistent lengths")
    if not np.all((overlap >= 0.0) & (overlap <= 1.0)):
        raise ContractError("S02 other_bbox_max_iou must be in [0, 1]")
    if not np.all((clipping >= 0.0) & (clipping <= 1.0)):
        raise ContractError("S02 clipped_fraction must be in [0, 1]")
    if not np.all((area >= 0.0) & (area <= 1.0)):
        raise ContractError("S02 bbox_area_percentile must be in [0, 1]")
    if np.any(blur < 0.0):
        raise ContractError("S02 blur_score must be non-negative")
    if not np.all((boundary >= 0.0) & (boundary <= 1.0)):
        raise ContractError("S02 distance_to_image_boundary must be in [0, 1]")

    weights_array = np.asarray(weights, dtype=np.float64)
    if weights_array.shape != (5,) or not np.all(np.isfinite(weights_array)):
        raise ContractError("S02 crop-quality weights must contain five finite values")
    if np.any(weights_array < 0.0) or not np.isclose(np.sum(weights_array), 1.0):
        raise ContractError("S02 crop-quality weights must be non-negative and sum to 1")
    thresholds = (
        overlap_zero_iou,
        clipping_zero_fraction,
        area_full_percentile,
        blur_zero_variance,
        blur_scale,
    )
    if not all(math.isfinite(value) for value in thresholds):
        raise ContractError("S02 crop-quality thresholds must be finite")
    if overlap_zero_iou <= 0.0 or clipping_zero_fraction <= 0.0:
        raise ContractError("S02 overlap/clipping zero thresholds must be positive")
    if area_full_percentile <= 0.0:
        raise ContractError("S02 area_full_percentile must be positive")
    if blur_zero_variance < 0.0 or blur_scale <= 0.0:
        raise ContractError("S02 blur variance thresholds are invalid")

    overlap_component = 1.0 - np.clip(overlap / overlap_zero_iou, 0.0, 1.0)
    clipping_component = 1.0 - np.clip(
        clipping / clipping_zero_fraction, 0.0, 1.0
    )
    area_component = np.clip(area / area_full_percentile, 0.0, 1.0)
    blur_component = np.clip(
        (np.log1p(blur) - math.log1p(blur_zero_variance))
        / (
            math.log1p(blur_zero_variance + blur_scale)
            - math.log1p(blur_zero_variance)
        ),
        0.0,
        1.0,
    )
    components = np.column_stack(
        (
            overlap_component,
            clipping_component,
            area_component,
            blur_component,
            boundary,
        )
    )
    quality = components @ weights_array
    return np.clip(quality, 0.0, 1.0).astype(np.float32)
