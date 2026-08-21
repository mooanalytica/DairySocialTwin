"""Arrow schemas for operator-approved S05 global path-cover outputs.

The current fixed-data run is authorized by an explicit operator blanket
strategy over every persisted provisional proposal.  That authorization is
kept separate from the original proposal evidence: ``proposal_confirmed`` and
the proposal evidence/status fields must be copied verbatim and must never be
rewritten to imply independent certification.
"""

from __future__ import annotations

import pyarrow as pa


GLOBAL_CANDIDATE_EDGES_SCHEMA = pa.schema(
    [
        # Stable proposal identity and temporal endpoints.
        pa.field("candidate_id", pa.string(), nullable=False),
        pa.field("proposal_id", pa.string(), nullable=True),
        pa.field("source_stable_id", pa.int64(), nullable=False),
        pa.field("target_stable_id", pa.int64(), nullable=False),
        pa.field("source_end_clip_id", pa.string(), nullable=False),
        pa.field("target_start_clip_id", pa.string(), nullable=False),
        pa.field("source_end_global_frame", pa.int64(), nullable=False),
        pa.field("target_start_global_frame", pa.int64(), nullable=False),
        pa.field("source_end_time_sec", pa.float64(), nullable=False),
        pa.field("target_start_time_sec", pa.float64(), nullable=False),
        pa.field("temporal_gap_sec", pa.float64(), nullable=False),
        pa.field("temporally_nonoverlapping", pa.bool_(), nullable=False),
        # Compact retrieval/model evidence copied from S05_PROPOSE.
        pa.field("appearance_present", pa.bool_(), nullable=False),
        pa.field("high_overlap", pa.bool_(), nullable=False),
        pa.field("selected_by_appearance_topk", pa.bool_(), nullable=False),
        pa.field("selected_by_temporal_nearest", pa.bool_(), nullable=False),
        pa.field("appearance_rank_out", pa.int32(), nullable=True),
        pa.field("appearance_rank_in", pa.int32(), nullable=True),
        pa.field("best_margin_out", pa.float64(), nullable=True),
        pa.field("best_margin_in", pa.float64(), nullable=True),
        pa.field("gallery_score_mutual", pa.float64(), nullable=True),
        pa.field("model_probability", pa.float64(), nullable=True),
        pa.field("model_raw_score", pa.float64(), nullable=True),
        pa.field("candidate_margin", pa.float64(), nullable=True),
        pa.field("provisional_threshold", pa.float64(), nullable=False),
        pa.field(
            "selected_probability_threshold", pa.float64(), nullable=False
        ),
        pa.field("selected_margin_threshold", pa.float64(), nullable=False),
        pa.field("passes_provisional_threshold", pa.bool_(), nullable=False),
        pa.field(
            "passes_selected_probability_gate", pa.bool_(), nullable=False
        ),
        pa.field("passes_selected_margin_gate", pa.bool_(), nullable=False),
        pa.field("passes_selected_gate", pa.bool_(), nullable=False),
        # Original proposal-stage state.  These fields are provenance, not the
        # operator authorization or final solver decision.
        pa.field("proposal_decision", pa.string(), nullable=False),
        pa.field("proposal_decision_reason", pa.string(), nullable=False),
        pa.field("proposal_evidence_status", pa.string(), nullable=True),
        pa.field("proposal_review_status", pa.string(), nullable=True),
        pa.field("proposal_selected_by_solver", pa.bool_(), nullable=False),
        pa.field("proposal_confirmed", pa.bool_(), nullable=False),
        pa.field("proposal_merge_applied", pa.bool_(), nullable=False),
        # Explicit operator authorization and independent path-cover result.
        pa.field("operator_approved", pa.bool_(), nullable=False),
        pa.field("approval_strategy", pa.string(), nullable=False),
        pa.field("authorization_basis", pa.string(), nullable=False),
        pa.field("eligible_for_solver", pa.bool_(), nullable=False),
        pa.field("selected_by_solver", pa.bool_(), nullable=False),
        pa.field("solver_iteration", pa.int8(), nullable=False),
        pa.field("solver_cost_int", pa.int64(), nullable=True),
        pa.field("global_link_id", pa.string(), nullable=True),
        pa.field("final_decision", pa.string(), nullable=False),
        pa.field("final_decision_reason", pa.string(), nullable=False),
    ]
)


STABLE_TO_GLOBAL_SCHEMA = pa.schema(
    [
        pa.field("stable_id", pa.int64(), nullable=False),
        pa.field("global_track_id", pa.int64(), nullable=False),
        pa.field("global_track_uuid", pa.string(), nullable=False),
        pa.field("display_global_id", pa.string(), nullable=False),
        pa.field("order_in_global_path", pa.int32(), nullable=False),
        pa.field("predecessor_stable_id", pa.int64(), nullable=True),
        pa.field("predecessor_candidate_id", pa.string(), nullable=True),
        pa.field("predecessor_proposal_id", pa.string(), nullable=True),
        pa.field("predecessor_global_link_id", pa.string(), nullable=True),
        pa.field("predecessor_link_probability", pa.float64(), nullable=True),
        pa.field("predecessor_link_margin", pa.float64(), nullable=True),
        pa.field(
            "predecessor_authorization_basis", pa.string(), nullable=True
        ),
        pa.field("link_type", pa.string(), nullable=False),
        pa.field("cross_clip_boundary", pa.bool_(), nullable=False),
        pa.field("component_num_stable_tracklets", pa.int32(), nullable=False),
        pa.field("component_num_detections", pa.int64(), nullable=False),
        pa.field("component_num_long_links", pa.int32(), nullable=False),
        pa.field(
            "component_min_link_probability", pa.float64(), nullable=True
        ),
        pa.field(
            "component_mean_link_probability", pa.float64(), nullable=True
        ),
        pa.field(
            "component_max_link_probability", pa.float64(), nullable=True
        ),
        pa.field("identity_basis", pa.string(), nullable=False),
        pa.field("id_status", pa.string(), nullable=False),
    ]
)


GLOBAL_TRACKS_SCHEMA = pa.schema(
    [
        pa.field("global_track_id", pa.int64(), nullable=False),
        pa.field("global_track_uuid", pa.string(), nullable=False),
        pa.field("display_global_id", pa.string(), nullable=False),
        pa.field("sequence_id", pa.string(), nullable=False),
        pa.field("first_stable_id", pa.int64(), nullable=False),
        pa.field("last_stable_id", pa.int64(), nullable=False),
        pa.field("start_det_id", pa.int64(), nullable=False),
        pa.field("end_det_id", pa.int64(), nullable=False),
        pa.field("start_clip_id", pa.string(), nullable=False),
        pa.field("end_clip_id", pa.string(), nullable=False),
        pa.field("clip_ids", pa.list_(pa.field("element", pa.string())), nullable=False),
        pa.field("start_global_frame", pa.int64(), nullable=False),
        pa.field("end_global_frame", pa.int64(), nullable=False),
        pa.field("start_time_sec", pa.float64(), nullable=False),
        pa.field("end_time_sec", pa.float64(), nullable=False),
        pa.field("num_stable_tracklets", pa.int32(), nullable=False),
        pa.field("num_microtracklets", pa.int32(), nullable=False),
        pa.field("num_detections", pa.int64(), nullable=False),
        pa.field("num_long_links", pa.int32(), nullable=False),
        pa.field("duration_visible_sec", pa.float64(), nullable=False),
        pa.field("min_link_probability", pa.float64(), nullable=True),
        pa.field("p10_link_probability", pa.float64(), nullable=True),
        pa.field("mean_link_probability", pa.float64(), nullable=True),
        pa.field("max_link_probability", pa.float64(), nullable=True),
        pa.field("min_link_margin", pa.float64(), nullable=True),
        pa.field("appearance_consistency", pa.float64(), nullable=True),
        pa.field("identity_basis", pa.string(), nullable=False),
        pa.field("id_status", pa.string(), nullable=False),
        pa.field("spans_multiple_clips", pa.bool_(), nullable=False),
        pa.field("population_warning", pa.bool_(), nullable=False),
    ]
)


DET_TO_GLOBAL_SCHEMA = pa.schema(
    [
        pa.field("det_id", pa.int64(), nullable=False),
        pa.field("sequence_id", pa.string(), nullable=False),
        pa.field("clip_id", pa.string(), nullable=False),
        pa.field("clip_order", pa.int16(), nullable=False),
        pa.field("local_frame", pa.int32(), nullable=False),
        pa.field("global_frame", pa.int64(), nullable=False),
        pa.field("global_time_sec", pa.float64(), nullable=False),
        pa.field("valid", pa.bool_(), nullable=False),
        pa.field("micro_id", pa.int64(), nullable=False),
        pa.field("stable_id", pa.int64(), nullable=False),
        pa.field("global_track_id", pa.int64(), nullable=False),
        pa.field("global_track_uuid", pa.string(), nullable=False),
        pa.field("display_global_id", pa.string(), nullable=False),
        pa.field("order_in_micro", pa.int32(), nullable=False),
        pa.field("order_in_stable", pa.int32(), nullable=False),
        pa.field("order_in_stable_detection", pa.int64(), nullable=False),
        pa.field("order_in_global_stable", pa.int32(), nullable=False),
        pa.field("order_in_global_detection", pa.int64(), nullable=False),
        pa.field("identity_basis", pa.string(), nullable=False),
        pa.field("id_status", pa.string(), nullable=False),
    ]
)


__all__ = [
    "DET_TO_GLOBAL_SCHEMA",
    "GLOBAL_CANDIDATE_EDGES_SCHEMA",
    "GLOBAL_TRACKS_SCHEMA",
    "STABLE_TO_GLOBAL_SCHEMA",
]
