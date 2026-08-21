"""Stable Arrow schemas for S04 short-link candidates and proposals."""

from __future__ import annotations

import pyarrow as pa

from cowtrack.linking.features import SHORT_FEATURE_SCHEMA


def _gallery_fields(prefix: str) -> list[pa.Field]:
    int64_list = pa.list_(pa.field("element", pa.int64()))
    return [
        pa.field(f"{prefix}_sample_ids", int64_list, nullable=True),
        pa.field(f"{prefix}_gallery_det_ids", int64_list, nullable=True),
        pa.field(f"{prefix}_embedding_rows", int64_list, nullable=True),
        pa.field(f"{prefix}_medoid_sample_id", pa.int64(), nullable=True),
        pa.field(f"{prefix}_appearance_quality", pa.float32(), nullable=True),
        pa.field(f"{prefix}_internal_cosine_p10", pa.float32(), nullable=True),
        pa.field(f"{prefix}_internal_cosine_p50", pa.float32(), nullable=True),
        pa.field(f"{prefix}_internal_cosine_min", pa.float32(), nullable=True),
        pa.field(f"{prefix}_gallery_num_input_samples", pa.int32(), nullable=True),
        pa.field(f"{prefix}_gallery_num_overlap_rejected", pa.int32(), nullable=True),
        pa.field(f"{prefix}_gallery_num_review_excluded", pa.int32(), nullable=True),
        pa.field(f"{prefix}_gallery_num_local_outliers", pa.int32(), nullable=True),
        pa.field(
            f"{prefix}_gallery_max_other_bbox_iou", pa.float32(), nullable=True
        ),
        pa.field(
            f"{prefix}_gallery_max_clean_other_bbox_iou",
            pa.float32(),
            nullable=True,
        ),
    ]


SHORT_CANDIDATE_EDGES_SCHEMA = pa.schema(
    [
        pa.field("edge_id", pa.string(), nullable=False),
        pa.field("source_micro_id", pa.int64(), nullable=False),
        pa.field("target_micro_id", pa.int64(), nullable=False),
        pa.field("source_start_det_id", pa.int64(), nullable=False),
        pa.field("source_end_det_id", pa.int64(), nullable=False),
        pa.field("target_start_det_id", pa.int64(), nullable=False),
        pa.field("target_end_det_id", pa.int64(), nullable=False),
        pa.field("source_start_clip_id", pa.string(), nullable=False),
        pa.field("source_end_clip_id", pa.string(), nullable=False),
        pa.field("target_start_clip_id", pa.string(), nullable=False),
        pa.field("target_end_clip_id", pa.string(), nullable=False),
        pa.field("source_start_global_frame", pa.int64(), nullable=False),
        pa.field("source_end_global_frame", pa.int64(), nullable=False),
        pa.field("target_start_global_frame", pa.int64(), nullable=False),
        pa.field("target_end_global_frame", pa.int64(), nullable=False),
        pa.field("source_start_time_sec", pa.float64(), nullable=False),
        pa.field("source_end_time_sec", pa.float64(), nullable=False),
        pa.field("target_start_time_sec", pa.float64(), nullable=False),
        pa.field("target_end_time_sec", pa.float64(), nullable=False),
        pa.field("appearance_present", pa.bool_(), nullable=False),
        pa.field("high_overlap", pa.bool_(), nullable=False),
        pa.field("motion_history_present", pa.bool_(), nullable=True),
        pa.field("source_gallery_present", pa.bool_(), nullable=False),
        pa.field("source_gallery_reason", pa.string(), nullable=True),
        pa.field("target_gallery_present", pa.bool_(), nullable=False),
        pa.field("target_gallery_reason", pa.string(), nullable=True),
        *_gallery_fields("source"),
        *_gallery_fields("target"),
        *[
            pa.field(
                name,
                pa.float64() if name == "gap_sec" else pa.float32(),
                nullable=True,
            )
            for name in SHORT_FEATURE_SCHEMA
        ],
        pa.field("probability", pa.float32(), nullable=True),
        pa.field("raw_score", pa.float32(), nullable=True),
        pa.field("decision", pa.string(), nullable=False),
        pa.field("decision_reason", pa.string(), nullable=False),
        pa.field("proposed_for_review", pa.bool_(), nullable=False),
    ]
)


SHORT_LINK_PROPOSALS_SCHEMA = pa.schema(
    [
        pa.field("proposal_id", pa.string(), nullable=False),
        pa.field("edge_id", pa.string(), nullable=False),
        pa.field("source_micro_id", pa.int64(), nullable=False),
        pa.field("target_micro_id", pa.int64(), nullable=False),
        pa.field("probability", pa.float32(), nullable=False),
        pa.field("high_overlap", pa.bool_(), nullable=False),
        pa.field("source_rank", pa.int32(), nullable=False),
        pa.field("target_rank", pa.int32(), nullable=False),
        pa.field("conflict_degree", pa.int32(), nullable=False),
        pa.field("conflict_group_id", pa.string(), nullable=False),
        pa.field("conflict_group_edge_count", pa.int32(), nullable=False),
        pa.field("conflict_group_node_count", pa.int32(), nullable=False),
        pa.field("review_status", pa.string(), nullable=False),
    ]
)


MICRO_TO_STABLE_SCHEMA = pa.schema(
    [
        pa.field("micro_id", pa.int64(), nullable=False),
        pa.field("stable_id", pa.int64(), nullable=False),
        pa.field("order_in_stable", pa.int32(), nullable=False),
        pa.field("predecessor_micro_id", pa.int64(), nullable=True),
        pa.field("predecessor_edge_id", pa.string(), nullable=True),
        pa.field("predecessor_link_probability", pa.float32(), nullable=True),
        pa.field("component_num_microtracklets", pa.int32(), nullable=False),
        pa.field("component_num_proposal_edges", pa.int32(), nullable=False),
        pa.field("component_min_proposal_probability", pa.float32(), nullable=True),
        pa.field("component_mean_proposal_probability", pa.float32(), nullable=True),
        pa.field("component_max_proposal_probability", pa.float32(), nullable=True),
    ]
)


STABLE_TRACKLETS_SCHEMA = pa.schema(
    [
        pa.field("stable_id", pa.int64(), nullable=False),
        pa.field("first_micro_id", pa.int64(), nullable=False),
        pa.field("last_micro_id", pa.int64(), nullable=False),
        pa.field("start_det_id", pa.int64(), nullable=False),
        pa.field("end_det_id", pa.int64(), nullable=False),
        pa.field("start_clip_id", pa.string(), nullable=False),
        pa.field("end_clip_id", pa.string(), nullable=False),
        pa.field("start_global_frame", pa.int64(), nullable=False),
        pa.field("end_global_frame", pa.int64(), nullable=False),
        pa.field("start_time_sec", pa.float64(), nullable=False),
        pa.field("end_time_sec", pa.float64(), nullable=False),
        pa.field("num_microtracklets", pa.int32(), nullable=False),
        pa.field("num_detections", pa.int64(), nullable=False),
        pa.field("num_proposal_edges", pa.int32(), nullable=False),
        pa.field("min_proposal_probability", pa.float32(), nullable=True),
        pa.field("mean_proposal_probability", pa.float32(), nullable=True),
        pa.field("max_proposal_probability", pa.float32(), nullable=True),
        pa.field("is_singleton", pa.bool_(), nullable=False),
    ]
)


DET_TO_STABLE_SCHEMA = pa.schema(
    [
        pa.field("det_id", pa.int64(), nullable=False),
        pa.field("micro_id", pa.int64(), nullable=False),
        pa.field("stable_id", pa.int64(), nullable=False),
        pa.field("order_in_stable", pa.int32(), nullable=False),
        pa.field("order_in_micro", pa.int32(), nullable=False),
        pa.field("order_in_stable_detection", pa.int64(), nullable=False),
    ]
)


__all__ = [
    "DET_TO_STABLE_SCHEMA",
    "MICRO_TO_STABLE_SCHEMA",
    "SHORT_CANDIDATE_EDGES_SCHEMA",
    "SHORT_LINK_PROPOSALS_SCHEMA",
    "STABLE_TRACKLETS_SCHEMA",
]
