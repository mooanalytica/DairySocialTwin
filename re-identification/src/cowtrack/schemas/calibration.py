"""Stable Arrow schema for S03 pseudo-link pairs and scored diagnostics."""

from __future__ import annotations

import pyarrow as pa


S03_APPEARANCE_FEATURES = (
    "prototype_cosine_max",
    "prototype_cosine_top3_mean",
    "medoid_cosine",
    "mutual_prototype_score",
    "appearance_quality_min",
    "appearance_quality_mean",
)

S03_MOTION_FEATURES = (
    "gap_sec",
    "log1p_gap_sec",
    "predicted_center_residual",
    "predicted_iou",
    "log_width_ratio",
    "log_height_ratio",
    "log_area_ratio",
)

S03_CONTEXT_FEATURES = (
    "src_end_max_other_iou",
    "dst_start_max_other_iou",
    "src_end_boundary_distance",
    "dst_start_boundary_distance",
    "is_clip_boundary",
)

S03_ALL_FEATURES = (
    *S03_APPEARANCE_FEATURES,
    *S03_MOTION_FEATURES,
    *S03_CONTEXT_FEATURES,
)


PSEUDO_PAIRS_SCHEMA = pa.schema(
    [
        pa.field("pair_id", pa.string(), nullable=False),
        pa.field("candidate_group_id", pa.string(), nullable=False),
        pa.field("parent_group_id", pa.string(), nullable=False),
        pa.field("split", pa.string(), nullable=False),
        pa.field("calibration_role", pa.string(), nullable=False),
        pa.field("mode", pa.string(), nullable=False),
        pa.field("label", pa.bool_(), nullable=False),
        pa.field("pair_kind", pa.string(), nullable=False),
        pa.field("stratum", pa.string(), nullable=False),
        pa.field("parent_micro_id", pa.int64(), nullable=False),
        pa.field("source_micro_id", pa.int64(), nullable=False),
        pa.field("target_micro_id", pa.int64(), nullable=False),
        pa.field("source_start_det_id", pa.int64(), nullable=False),
        pa.field("source_end_det_id", pa.int64(), nullable=False),
        pa.field("target_start_det_id", pa.int64(), nullable=False),
        pa.field("target_end_det_id", pa.int64(), nullable=False),
        pa.field("source_end_global_frame", pa.int64(), nullable=False),
        pa.field("target_start_global_frame", pa.int64(), nullable=False),
        pa.field("source_start_global_frame", pa.int64(), nullable=False),
        pa.field("target_end_global_frame", pa.int64(), nullable=False),
        pa.field("source_start_time_sec", pa.float64(), nullable=False),
        pa.field("source_end_time_sec", pa.float64(), nullable=False),
        pa.field("target_start_time_sec", pa.float64(), nullable=False),
        pa.field("target_end_time_sec", pa.float64(), nullable=False),
        pa.field("source_clip_id", pa.string(), nullable=False),
        pa.field("target_clip_id", pa.string(), nullable=False),
        pa.field("source_segment_start_clip_id", pa.string(), nullable=False),
        pa.field("target_segment_end_clip_id", pa.string(), nullable=False),
        pa.field("source_num_detections", pa.int32(), nullable=False),
        pa.field("target_num_detections", pa.int32(), nullable=False),
        pa.field("source_sample_ids", pa.list_(pa.int64()), nullable=False),
        pa.field("target_sample_ids", pa.list_(pa.int64()), nullable=False),
        pa.field("source_gallery_det_ids", pa.list_(pa.int64()), nullable=False),
        pa.field("target_gallery_det_ids", pa.list_(pa.int64()), nullable=False),
        pa.field("source_embedding_rows", pa.list_(pa.int64()), nullable=False),
        pa.field("target_embedding_rows", pa.list_(pa.int64()), nullable=False),
        pa.field("source_medoid_sample_id", pa.int64(), nullable=False),
        pa.field("target_medoid_sample_id", pa.int64(), nullable=False),
        pa.field("source_appearance_quality", pa.float32(), nullable=False),
        pa.field("target_appearance_quality", pa.float32(), nullable=False),
        pa.field("source_internal_cosine_p10", pa.float32(), nullable=False),
        pa.field("target_internal_cosine_p10", pa.float32(), nullable=False),
        pa.field("source_internal_cosine_p50", pa.float32(), nullable=False),
        pa.field("target_internal_cosine_p50", pa.float32(), nullable=False),
        pa.field("source_internal_cosine_min", pa.float32(), nullable=False),
        pa.field("target_internal_cosine_min", pa.float32(), nullable=False),
        pa.field("source_gallery_num_input_samples", pa.int16(), nullable=False),
        pa.field("target_gallery_num_input_samples", pa.int16(), nullable=False),
        pa.field("source_gallery_num_overlap_rejected", pa.int16(), nullable=False),
        pa.field("target_gallery_num_overlap_rejected", pa.int16(), nullable=False),
        pa.field("source_gallery_num_review_excluded", pa.int16(), nullable=False),
        pa.field("target_gallery_num_review_excluded", pa.int16(), nullable=False),
        pa.field("source_gallery_num_local_outliers", pa.int16(), nullable=False),
        pa.field("target_gallery_num_local_outliers", pa.int16(), nullable=False),
        pa.field("source_gallery_max_other_bbox_iou", pa.float32(), nullable=False),
        pa.field("target_gallery_max_other_bbox_iou", pa.float32(), nullable=False),
        pa.field("source_gallery_max_clean_other_bbox_iou", pa.float32(), nullable=False),
        pa.field("target_gallery_max_clean_other_bbox_iou", pa.float32(), nullable=False),
        pa.field("gap_target_sec", pa.float32(), nullable=False),
        pa.field("gap_error_sec", pa.float32(), nullable=False),
        pa.field("hard_negative_rank", pa.int16(), nullable=False),
        pa.field("appearance_present", pa.bool_(), nullable=False),
        pa.field("high_overlap", pa.bool_(), nullable=False),
        *[
            pa.field(
                name,
                pa.float32(),
                nullable=name in {"predicted_center_residual", "predicted_iou"},
            )
            for name in S03_ALL_FEATURES
        ],
        pa.field("model_probability", pa.float32(), nullable=True),
        pa.field("model_raw_score", pa.float32(), nullable=True),
        pa.field("candidate_margin", pa.float32(), nullable=True),
        pa.field("decision", pa.string(), nullable=False),
    ]
)


__all__ = [
    "PSEUDO_PAIRS_SCHEMA",
    "S03_ALL_FEATURES",
    "S03_APPEARANCE_FEATURES",
    "S03_CONTEXT_FEATURES",
    "S03_MOTION_FEATURES",
]
