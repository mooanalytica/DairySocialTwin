"""Deterministic S02 appearance sampling and aggregation primitives."""

from cowtrack.appearance.prototypes import (
    PrototypeBatch,
    build_micro_prototypes,
    build_tracklet_prototypes,
)
from cowtrack.appearance.quality import (
    bbox_area_percentiles,
    combine_crop_quality,
    crop_with_padding,
    laplacian_blur_score,
    maximum_other_bbox_iou,
    soft_border_mask,
)
from cowtrack.appearance.sampling import select_representative_indices

__all__ = [
    "PrototypeBatch",
    "bbox_area_percentiles",
    "build_micro_prototypes",
    "build_tracklet_prototypes",
    "combine_crop_quality",
    "crop_with_padding",
    "laplacian_blur_score",
    "maximum_other_bbox_iou",
    "select_representative_indices",
    "soft_border_mask",
]
