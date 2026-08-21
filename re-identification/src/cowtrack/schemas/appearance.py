"""Stable Arrow schemas for S02 appearance and downstream context artifacts."""

from __future__ import annotations

import pyarrow as pa


APPEARANCE_SAMPLES_SCHEMA = pa.schema(
    [
        pa.field("sample_id", pa.int64(), nullable=False),
        pa.field("micro_id", pa.int64(), nullable=False),
        pa.field("det_id", pa.int64(), nullable=False),
        pa.field("clip_id", pa.string(), nullable=False),
        pa.field("global_frame", pa.int64(), nullable=False),
        pa.field("global_time_sec", pa.float64(), nullable=False),
        pa.field("crop_quality", pa.float32(), nullable=False),
        pa.field("other_bbox_max_iou", pa.float32(), nullable=False),
        pa.field("clipped_fraction", pa.float32(), nullable=False),
        pa.field("bbox_area_percentile", pa.float32(), nullable=False),
        pa.field("blur_score", pa.float32(), nullable=False),
        pa.field("distance_to_image_boundary", pa.float32(), nullable=False),
        pa.field("selection_reason", pa.string(), nullable=False),
        pa.field("prototype_inlier", pa.bool_(), nullable=False),
        pa.field("prototype_outlier", pa.bool_(), nullable=False),
        pa.field("embedding_row", pa.int64(), nullable=False),
    ]
)


APPEARANCE_EXCLUSIONS_SCHEMA = pa.schema(
    [
        pa.field("det_id", pa.int64(), nullable=False),
        pa.field("micro_id", pa.int64(), nullable=False),
        pa.field("global_frame", pa.int64(), nullable=False),
        pa.field("appearance_excluded", pa.bool_(), nullable=False),
        pa.field("crop_quality", pa.float32(), nullable=False),
        pa.field("exclusion_reasons", pa.list_(pa.string()), nullable=False),
        pa.field("review_case_ids", pa.list_(pa.string()), nullable=False),
        pa.field("trigger_global_frames", pa.list_(pa.int64()), nullable=False),
    ]
)


MICRO_APPEARANCE_SCHEMA = pa.schema(
    [
        pa.field("micro_id", pa.int64(), nullable=False),
        pa.field("prototype_row", pa.int64(), nullable=False),
        pa.field("num_appearance_samples", pa.int16(), nullable=False),
        pa.field("appearance_usable", pa.bool_(), nullable=False),
        pa.field("appearance_quality", pa.float32(), nullable=False),
        pa.field("internal_cosine_p10", pa.float32(), nullable=False),
        pa.field("internal_cosine_p50", pa.float32(), nullable=False),
        pa.field("internal_cosine_min", pa.float32(), nullable=False),
        pa.field("appearance_outlier_count", pa.int16(), nullable=False),
        pa.field("selected_encoder", pa.string(), nullable=False),
    ]
)


MICRO_CONTEXT_SCHEMA = pa.schema(
    [
        pa.field("micro_id", pa.int64(), nullable=False),
        pa.field("start_det_id", pa.int64(), nullable=False),
        pa.field("end_det_id", pa.int64(), nullable=False),
        pa.field("start_clip_id", pa.string(), nullable=False),
        pa.field("end_clip_id", pa.string(), nullable=False),
        pa.field("start_max_other_iou", pa.float32(), nullable=False),
        pa.field("end_max_other_iou", pa.float32(), nullable=False),
        pa.field("start_boundary_distance", pa.float32(), nullable=False),
        pa.field("end_boundary_distance", pa.float32(), nullable=False),
        pa.field("start_global_frame", pa.int64(), nullable=False),
        pa.field("end_global_frame", pa.int64(), nullable=False),
    ]
)
