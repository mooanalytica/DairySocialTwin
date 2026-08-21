"""Arrow contracts for the operator-forced appearance consolidation audit."""

from __future__ import annotations

import pyarrow as pa


GRADED_STABLE_APPEARANCE_SCHEMA = pa.schema(
    [
        pa.field("stable_id", pa.int64(), nullable=False),
        pa.field("evidence_grade", pa.string(), nullable=False),
        pa.field("descriptor_usable", pa.bool_(), nullable=False),
        pa.field("selection_policy", pa.string(), nullable=True),
        pa.field("num_valid_prototypes", pa.int8(), nullable=False),
        pa.field("num_input_samples", pa.int32(), nullable=False),
        pa.field("num_selected_samples", pa.int32(), nullable=False),
        pa.field("input_sample_ids", pa.list_(pa.int64()), nullable=False),
        pa.field("selected_sample_ids", pa.list_(pa.int64()), nullable=False),
        pa.field("selected_quality_min", pa.float32(), nullable=True),
        pa.field("selected_quality_mean", pa.float32(), nullable=True),
        pa.field("selected_max_other_bbox_iou", pa.float32(), nullable=True),
        pa.field("used_high_overlap", pa.bool_(), nullable=False),
        pa.field("used_s02_outlier", pa.bool_(), nullable=False),
        pa.field("missing_reason", pa.string(), nullable=True),
    ]
)


RESCUE_SAMPLES_SCHEMA = pa.schema(
    [
        pa.field("stable_id", pa.int64(), nullable=False),
        pa.field("micro_id", pa.int64(), nullable=False),
        pa.field("det_id", pa.int64(), nullable=False),
        pa.field("clip_id", pa.string(), nullable=False),
        pa.field("local_frame", pa.int32(), nullable=False),
        pa.field("global_frame", pa.int64(), nullable=False),
        pa.field("global_time_sec", pa.float64(), nullable=False),
        pa.field("x1", pa.float32(), nullable=False),
        pa.field("y1", pa.float32(), nullable=False),
        pa.field("x2", pa.float32(), nullable=False),
        pa.field("y2", pa.float32(), nullable=False),
        pa.field("crop_quality", pa.float32(), nullable=False),
        pa.field("other_bbox_max_iou", pa.float32(), nullable=False),
        pa.field("clipped_fraction", pa.float32(), nullable=False),
        pa.field("bbox_area_percentile", pa.float32(), nullable=False),
        pa.field("blur_score", pa.float32(), nullable=False),
        pa.field("distance_to_image_boundary", pa.float32(), nullable=False),
        pa.field("review_excluded", pa.bool_(), nullable=False),
        pa.field("quality_gate_passed", pa.bool_(), nullable=False),
        pa.field("selected_for_descriptor", pa.bool_(), nullable=False),
        pa.field("selection_reason", pa.string(), nullable=False),
        pa.field("embedding_row", pa.int64(), nullable=True),
    ]
)


FORCED_CANDIDATE_EDGES_SCHEMA = pa.schema(
    [
        pa.field("candidate_id", pa.string(), nullable=False),
        pa.field("source_stable_id", pa.int64(), nullable=False),
        pa.field("target_stable_id", pa.int64(), nullable=False),
        pa.field("source_end_clip_id", pa.string(), nullable=False),
        pa.field("target_start_clip_id", pa.string(), nullable=False),
        pa.field("source_end_global_frame", pa.int64(), nullable=False),
        pa.field("target_start_global_frame", pa.int64(), nullable=False),
        pa.field("source_end_time_sec", pa.float64(), nullable=False),
        pa.field("target_start_time_sec", pa.float64(), nullable=False),
        pa.field("temporal_gap_sec", pa.float64(), nullable=False),
        pa.field("strictly_nonoverlapping", pa.bool_(), nullable=False),
        pa.field("appearance_cosine", pa.float32(), nullable=False),
        pa.field("source_evidence_grade", pa.string(), nullable=False),
        pa.field("target_evidence_grade", pa.string(), nullable=False),
        pa.field("selected_by_source_topk", pa.bool_(), nullable=False),
        pa.field("selected_by_target_topk", pa.bool_(), nullable=False),
        pa.field("temporal_backbone", pa.bool_(), nullable=False),
        pa.field("prior_global_link", pa.bool_(), nullable=False),
        pa.field("appearance_cost_int", pa.int64(), nullable=False),
        pa.field("solver_cost_int", pa.int64(), nullable=False),
        pa.field("selected_by_solver", pa.bool_(), nullable=False),
        pa.field("global_link_id", pa.string(), nullable=True),
        pa.field("authorization_basis", pa.string(), nullable=False),
        pa.field("id_status", pa.string(), nullable=False),
    ]
)


__all__ = [
    "FORCED_CANDIDATE_EDGES_SCHEMA",
    "GRADED_STABLE_APPEARANCE_SCHEMA",
    "RESCUE_SAMPLES_SCHEMA",
]
