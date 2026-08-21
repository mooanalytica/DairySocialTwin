"""Stable Arrow schema for S04 finalized-component appearance provenance."""

from __future__ import annotations

import pyarrow as pa


STABLE_APPEARANCE_SCHEMA = pa.schema(
    [
        pa.field("stable_id", pa.int64(), nullable=False),
        pa.field("prototype_row", pa.int64(), nullable=False),
        pa.field("constituent_micro_ids", pa.list_(pa.int64()), nullable=False),
        pa.field("num_input_samples", pa.int32(), nullable=False),
        pa.field("num_s02_inliers", pa.int32(), nullable=False),
        pa.field("num_clean_candidates", pa.int32(), nullable=False),
        pa.field("num_clean_inliers", pa.int32(), nullable=False),
        pa.field("num_valid_prototypes", pa.int8(), nullable=False),
        pa.field("appearance_usable", pa.bool_(), nullable=False),
        pa.field("missing_reason", pa.string(), nullable=True),
        pa.field("clean_sample_ids", pa.list_(pa.int64()), nullable=False),
        pa.field("clean_det_ids", pa.list_(pa.int64()), nullable=False),
        pa.field("clean_embedding_rows", pa.list_(pa.int64()), nullable=False),
        pa.field("medoid_sample_id", pa.int64(), nullable=True),
        pa.field("appearance_quality", pa.float32(), nullable=True),
        pa.field("internal_cosine_p10", pa.float32(), nullable=True),
        pa.field("internal_cosine_p50", pa.float32(), nullable=True),
        pa.field("internal_cosine_min", pa.float32(), nullable=True),
        pa.field("num_overlap_rejected", pa.int32(), nullable=False),
        pa.field("num_review_excluded", pa.int32(), nullable=False),
        pa.field("num_local_outliers", pa.int32(), nullable=True),
        pa.field("max_other_bbox_iou", pa.float32(), nullable=False),
        pa.field("max_clean_other_bbox_iou", pa.float32(), nullable=True),
    ]
)


__all__ = ["STABLE_APPEARANCE_SCHEMA"]
