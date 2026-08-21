"""Arrow contracts for the fixed S06 forced-provisional QA/export stage."""

from __future__ import annotations

import pyarrow as pa


# This is an audit-friendly superset of the S06 specification.  The legacy
# (clip-local) ID is intentionally adjacent to the global ID columns.  Every
# identity/mapping field is nullable because invalid S00 rows remain in the
# output and must not be assigned a fabricated identity.
DETECTIONS_WITH_GLOBAL_ID_SCHEMA = pa.schema(
    [
        pa.field("sequence_id", pa.string(), nullable=False),
        pa.field("clip_id", pa.string(), nullable=False),
        pa.field("clip_order", pa.int16(), nullable=False),
        pa.field("det_id", pa.int64(), nullable=False),
        pa.field("csv_row_index", pa.int64(), nullable=False),
        pa.field("legacy_track_id", pa.string(), nullable=True),
        pa.field("global_track_id", pa.int64(), nullable=True),
        pa.field("global_track_uuid", pa.string(), nullable=True),
        pa.field("display_global_id", pa.string(), nullable=True),
        pa.field("id_status", pa.string(), nullable=True),
        pa.field("identity_basis", pa.string(), nullable=True),
        pa.field("local_frame", pa.int32(), nullable=False),
        pa.field("global_frame", pa.int64(), nullable=False),
        pa.field("global_time_sec", pa.float64(), nullable=False),
        pa.field("x1", pa.float32(), nullable=False),
        pa.field("y1", pa.float32(), nullable=False),
        pa.field("x2", pa.float32(), nullable=False),
        pa.field("y2", pa.float32(), nullable=False),
        pa.field("bbox_confidence", pa.float32(), nullable=True),
        pa.field("valid", pa.bool_(), nullable=False),
        pa.field("qa_flags", pa.uint32(), nullable=False),
        pa.field("invalid_reason", pa.string(), nullable=True),
        pa.field("micro_id", pa.int64(), nullable=True),
        pa.field("stable_id", pa.int64(), nullable=True),
        pa.field("order_in_micro", pa.int32(), nullable=True),
        pa.field("order_in_stable", pa.int32(), nullable=True),
        pa.field("order_in_stable_detection", pa.int64(), nullable=True),
        pa.field("order_in_global_stable", pa.int32(), nullable=True),
        pa.field("order_in_global_detection", pa.int64(), nullable=True),
        pa.field("local_purity_score", pa.float32(), nullable=True),
        # Forced appearance cosine is not a calibrated probability.  These
        # fields therefore remain null for this fixed export.
        pa.field("assignment_confidence", pa.float64(), nullable=True),
        pa.field("incoming_link_probability", pa.float64(), nullable=True),
        pa.field("outgoing_link_probability", pa.float64(), nullable=True),
    ]
)


GLOBAL_TRACK_SUMMARY_SCHEMA = pa.schema(
    [
        pa.field("global_track_id", pa.int64(), nullable=False),
        pa.field("global_track_uuid", pa.string(), nullable=False),
        pa.field("display_global_id", pa.string(), nullable=False),
        pa.field("sequence_id", pa.string(), nullable=False),
        pa.field("id_status", pa.string(), nullable=False),
        pa.field("identity_basis", pa.string(), nullable=False),
        pa.field("authorization_basis", pa.string(), nullable=False),
        pa.field("certification_claimed", pa.bool_(), nullable=False),
        pa.field("start_clip_id", pa.string(), nullable=False),
        pa.field("end_clip_id", pa.string(), nullable=False),
        # Pipe-separated in CSV, ordered by the fixed clip order.
        pa.field("clip_ids", pa.string(), nullable=False),
        pa.field("start_global_frame", pa.int64(), nullable=False),
        pa.field("end_global_frame", pa.int64(), nullable=False),
        pa.field("start_time_sec", pa.float64(), nullable=False),
        pa.field("end_time_sec", pa.float64(), nullable=False),
        pa.field("duration_visible_sec", pa.float64(), nullable=False),
        pa.field("num_stable_tracklets", pa.int32(), nullable=False),
        pa.field("num_microtracklets", pa.int32(), nullable=False),
        pa.field("num_detections", pa.int64(), nullable=False),
        pa.field("selected_link_count", pa.int32(), nullable=False),
        pa.field("spans_multiple_clips", pa.bool_(), nullable=False),
        pa.field("selected_link_cosine_min", pa.float64(), nullable=True),
        pa.field("selected_link_cosine_p10", pa.float64(), nullable=True),
        pa.field("selected_link_cosine_mean", pa.float64(), nullable=True),
        pa.field("selected_link_cosine_max", pa.float64(), nullable=True),
        pa.field("longest_gap_sec", pa.float64(), nullable=True),
        pa.field("num_links_cosine_below_0_3", pa.int32(), nullable=False),
        pa.field("num_links_cosine_below_0_4", pa.int32(), nullable=False),
        pa.field("num_links_cosine_below_0_5", pa.int32(), nullable=False),
        pa.field("num_links_cosine_below_0_6", pa.int32(), nullable=False),
        pa.field("num_link_endpoints_grade_a_clean", pa.int32(), nullable=False),
        pa.field(
            "num_link_endpoints_grade_b_existing_degraded",
            pa.int32(),
            nullable=False,
        ),
        pa.field(
            "num_link_endpoints_grade_c_reencoded_degraded",
            pa.int32(),
            nullable=False,
        ),
        pa.field("num_links_selected_by_source_topk", pa.int32(), nullable=False),
        pa.field("num_links_selected_by_target_topk", pa.int32(), nullable=False),
        pa.field("num_links_temporal_backbone", pa.int32(), nullable=False),
        pa.field("num_links_prior_global", pa.int32(), nullable=False),
        # Explicit nullable probability fields preserve the generic S06 shape
        # without relabelling appearance cosine as confidence.
        pa.field("min_link_probability", pa.float64(), nullable=True),
        pa.field("p10_link_probability", pa.float64(), nullable=True),
        pa.field("mean_link_probability", pa.float64(), nullable=True),
        pa.field("max_link_probability", pa.float64(), nullable=True),
        pa.field("probability_not_available_reason", pa.string(), nullable=False),
        pa.field("population_soft_max", pa.int16(), nullable=False),
        pa.field("population_overflow", pa.int16(), nullable=False),
        pa.field("population_warning", pa.bool_(), nullable=False),
    ]
)


# Scalar evidence used to render the deterministic low-confidence HTML.  Crop
# file paths are presentation assets and deliberately are not identity scores.
LOW_CONFIDENCE_LINKS_SCHEMA = pa.schema(
    [
        pa.field("report_order", pa.int32(), nullable=False),
        pa.field("priority_group", pa.int8(), nullable=False),
        pa.field("global_link_id", pa.string(), nullable=False),
        pa.field("global_track_id", pa.int64(), nullable=False),
        pa.field("display_global_id", pa.string(), nullable=False),
        pa.field("source_stable_id", pa.int64(), nullable=False),
        pa.field("target_stable_id", pa.int64(), nullable=False),
        pa.field("source_end_clip_id", pa.string(), nullable=False),
        pa.field("target_start_clip_id", pa.string(), nullable=False),
        pa.field("source_end_global_frame", pa.int64(), nullable=False),
        pa.field("target_start_global_frame", pa.int64(), nullable=False),
        pa.field("temporal_gap_sec", pa.float64(), nullable=False),
        pa.field("appearance_cosine", pa.float32(), nullable=False),
        pa.field("source_candidate_rank", pa.int32(), nullable=False),
        pa.field("target_candidate_rank", pa.int32(), nullable=False),
        pa.field("source_second_best_cosine_margin", pa.float32(), nullable=True),
        pa.field("target_second_best_cosine_margin", pa.float32(), nullable=True),
        pa.field("conservative_cosine_margin", pa.float32(), nullable=True),
        pa.field("source_evidence_grade", pa.string(), nullable=False),
        pa.field("target_evidence_grade", pa.string(), nullable=False),
        pa.field("selected_by_source_topk", pa.bool_(), nullable=False),
        pa.field("selected_by_target_topk", pa.bool_(), nullable=False),
        pa.field("temporal_backbone", pa.bool_(), nullable=False),
        pa.field("prior_global_link", pa.bool_(), nullable=False),
        pa.field("authorization_basis", pa.string(), nullable=False),
        pa.field("id_status", pa.string(), nullable=False),
    ]
)


__all__ = [
    "DETECTIONS_WITH_GLOBAL_ID_SCHEMA",
    "GLOBAL_TRACK_SUMMARY_SCHEMA",
    "LOW_CONFIDENCE_LINKS_SCHEMA",
]
