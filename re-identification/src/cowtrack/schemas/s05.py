"""Exact Arrow schema for S05A stable-path long calibration rows."""

from __future__ import annotations

import pyarrow as pa

from cowtrack.linking.features import LONG_FEATURE_SCHEMA


def _segment_fields(prefix: str) -> list[pa.Field]:
    return [
        pa.field(f"{prefix}_stable_id", pa.int64(), nullable=False),
        pa.field(
            f"{prefix}_constituent_micro_ids",
            pa.list_(pa.int64()),
            nullable=False,
        ),
        pa.field(f"{prefix}_start_det_id", pa.int64(), nullable=False),
        pa.field(f"{prefix}_end_det_id", pa.int64(), nullable=False),
        pa.field(f"{prefix}_start_global_frame", pa.int64(), nullable=False),
        pa.field(f"{prefix}_end_global_frame", pa.int64(), nullable=False),
        pa.field(f"{prefix}_start_time_sec", pa.float64(), nullable=False),
        pa.field(f"{prefix}_end_time_sec", pa.float64(), nullable=False),
        pa.field(f"{prefix}_start_clip_id", pa.string(), nullable=False),
        pa.field(f"{prefix}_end_clip_id", pa.string(), nullable=False),
        pa.field(f"{prefix}_num_detections", pa.int32(), nullable=False),
    ]


def _gallery_fields(prefix: str) -> list[pa.Field]:
    return [
        pa.field(f"{prefix}_sample_ids", pa.list_(pa.int64()), nullable=False),
        pa.field(f"{prefix}_gallery_det_ids", pa.list_(pa.int64()), nullable=False),
        pa.field(f"{prefix}_embedding_rows", pa.list_(pa.int64()), nullable=False),
        pa.field(f"{prefix}_medoid_sample_id", pa.int64(), nullable=False),
        pa.field(f"{prefix}_appearance_quality", pa.float32(), nullable=False),
        pa.field(f"{prefix}_internal_cosine_p10", pa.float32(), nullable=False),
        pa.field(f"{prefix}_internal_cosine_p50", pa.float32(), nullable=False),
        pa.field(f"{prefix}_internal_cosine_min", pa.float32(), nullable=False),
        pa.field(f"{prefix}_gallery_num_input_samples", pa.int32(), nullable=False),
        pa.field(
            f"{prefix}_gallery_num_overlap_rejected", pa.int32(), nullable=False
        ),
        pa.field(
            f"{prefix}_gallery_num_review_excluded", pa.int32(), nullable=False
        ),
        pa.field(f"{prefix}_gallery_num_local_outliers", pa.int32(), nullable=False),
        pa.field(
            f"{prefix}_gallery_max_other_bbox_iou", pa.float32(), nullable=False
        ),
        pa.field(
            f"{prefix}_gallery_max_clean_other_bbox_iou",
            pa.float32(),
            nullable=False,
        ),
    ]


LONG_CALIBRATION_PAIRS_SCHEMA = pa.schema(
    [
        pa.field("pair_id", pa.string(), nullable=False),
        pa.field("candidate_group_id", pa.string(), nullable=False),
        pa.field("parent_group_id", pa.string(), nullable=False),
        pa.field("partition", pa.string(), nullable=False),
        pa.field("split", pa.string(), nullable=False),
        pa.field("calibration_role", pa.string(), nullable=False),
        pa.field("mode", pa.string(), nullable=False),
        pa.field("label", pa.bool_(), nullable=False),
        pa.field("pair_kind", pa.string(), nullable=False),
        pa.field("parent_stable_id", pa.int64(), nullable=False),
        pa.field("source_clip_id", pa.string(), nullable=False),
        pa.field("target_clip_id", pa.string(), nullable=False),
        pa.field("hard_negative_rank", pa.int32(), nullable=False),
        pa.field("appearance_present", pa.bool_(), nullable=False),
        pa.field("high_overlap", pa.bool_(), nullable=False),
        pa.field("cooccurrence_clip_id", pa.string(), nullable=True),
        pa.field("cooccurrence_global_frame", pa.int64(), nullable=True),
        pa.field("cooccurrence_parent_det_id", pa.int64(), nullable=True),
        pa.field("cooccurrence_other_det_id", pa.int64(), nullable=True),
        *_segment_fields("source"),
        *_segment_fields("target"),
        *_gallery_fields("source"),
        *_gallery_fields("target"),
        *[
            pa.field(name, pa.float64(), nullable=False)
            for name in LONG_FEATURE_SCHEMA
        ],
        pa.field("model_probability", pa.float64(), nullable=True),
        pa.field("model_raw_score", pa.float64(), nullable=True),
        pa.field("candidate_margin", pa.float64(), nullable=True),
        pa.field("decision", pa.string(), nullable=False),
    ]
)


__all__ = ["LONG_CALIBRATION_PAIRS_SCHEMA"]
