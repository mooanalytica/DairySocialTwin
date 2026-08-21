"""Exact Arrow schemas for proposal-only S05 long-link retrieval.

These schemas deliberately stop at review proposals.  They contain no path
cover, solver assignment, or global-identity fields.  Directional retrieval
diagnostics are retained so a proposal can be audited without re-running the
exact cosine search.
"""

from __future__ import annotations

import pyarrow as pa

from cowtrack.linking.features import LONG_FEATURE_SCHEMA


def _path_fields(prefix: str) -> list[pa.Field]:
    int64_list = pa.list_(pa.field("element", pa.int64()))
    return [
        pa.field(f"{prefix}_constituent_micro_ids", int64_list, nullable=False),
        pa.field(f"{prefix}_start_det_id", pa.int64(), nullable=False),
        pa.field(f"{prefix}_end_det_id", pa.int64(), nullable=False),
        pa.field(f"{prefix}_start_clip_id", pa.string(), nullable=False),
        pa.field(f"{prefix}_end_clip_id", pa.string(), nullable=False),
        pa.field(f"{prefix}_start_global_frame", pa.int64(), nullable=False),
        pa.field(f"{prefix}_end_global_frame", pa.int64(), nullable=False),
        pa.field(f"{prefix}_start_time_sec", pa.float64(), nullable=False),
        pa.field(f"{prefix}_end_time_sec", pa.float64(), nullable=False),
        pa.field(f"{prefix}_num_detections", pa.int64(), nullable=False),
    ]


def _retrieval_fields(*, nullable_values: bool) -> list[pa.Field]:
    return [
        pa.field("selected_by_appearance_topk", pa.bool_(), nullable=False),
        pa.field("selected_by_temporal_nearest", pa.bool_(), nullable=False),
        # Temporal-nearest union candidates need not occur inside the retained
        # appearance top-k.  A null rank records that condition explicitly.
        pa.field("appearance_rank_out", pa.int32(), nullable=nullable_values),
        pa.field("appearance_rank_in", pa.int32(), nullable=nullable_values),
        # A direction has no runner-up when its eligible pool has size one.
        # Null must be retained rather than an infinite/fabricated margin.
        pa.field("best_margin_out", pa.float64(), nullable=True),
        pa.field("best_margin_in", pa.float64(), nullable=True),
    ]


def _gallery_score_fields(*, nullable: bool) -> list[pa.Field]:
    return [
        pa.field("gallery_score_max", pa.float64(), nullable=nullable),
        pa.field("gallery_score_top3", pa.float64(), nullable=nullable),
        pa.field("gallery_score_src_to_dst", pa.float64(), nullable=nullable),
        pa.field("gallery_score_dst_to_src", pa.float64(), nullable=nullable),
        pa.field("gallery_score_mutual", pa.float64(), nullable=nullable),
    ]


def _gallery_provenance_fields(prefix: str, *, nullable: bool) -> list[pa.Field]:
    """Persist the exact clean S02 samples behind every scored path gallery."""

    return [
        pa.field(f"{prefix}_sample_ids", pa.list_(pa.int64()), nullable=nullable),
        pa.field(
            f"{prefix}_gallery_det_ids", pa.list_(pa.int64()), nullable=nullable
        ),
        pa.field(
            f"{prefix}_embedding_rows", pa.list_(pa.int64()), nullable=nullable
        ),
        pa.field(f"{prefix}_medoid_sample_id", pa.int64(), nullable=nullable),
        pa.field(f"{prefix}_appearance_quality", pa.float64(), nullable=nullable),
        pa.field(f"{prefix}_internal_cosine_p10", pa.float64(), nullable=nullable),
        pa.field(f"{prefix}_internal_cosine_p50", pa.float64(), nullable=nullable),
        pa.field(f"{prefix}_internal_cosine_min", pa.float64(), nullable=nullable),
        pa.field(
            f"{prefix}_gallery_num_input_samples", pa.int32(), nullable=nullable
        ),
        pa.field(
            f"{prefix}_gallery_num_overlap_rejected", pa.int32(), nullable=nullable
        ),
        pa.field(
            f"{prefix}_gallery_num_review_excluded", pa.int32(), nullable=nullable
        ),
        pa.field(
            f"{prefix}_gallery_num_local_outliers", pa.int32(), nullable=nullable
        ),
        pa.field(
            f"{prefix}_gallery_max_other_bbox_iou", pa.float64(), nullable=nullable
        ),
        pa.field(
            f"{prefix}_gallery_max_clean_other_bbox_iou",
            pa.float64(),
            nullable=nullable,
        ),
    ]


def _model_fields(*, nullable_values: bool) -> list[pa.Field]:
    return [
        *[
            pa.field(name, pa.float64(), nullable=nullable_values)
            for name in LONG_FEATURE_SCHEMA
        ],
        pa.field(
            "model_probability", pa.float64(), nullable=nullable_values
        ),
        pa.field("model_raw_score", pa.float64(), nullable=nullable_values),
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
        pa.field("decision", pa.string(), nullable=False),
        pa.field("decision_reason", pa.string(), nullable=False),
        # S05 proposal-only must persist these as false.  Their presence makes
        # accidental promotion to an identity-changing artifact detectable.
        pa.field("selected_by_solver", pa.bool_(), nullable=False),
        pa.field("confirmed", pa.bool_(), nullable=False),
        pa.field("merge_applied", pa.bool_(), nullable=False),
    ]


LONG_CANDIDATE_EDGES_SCHEMA = pa.schema(
    [
        pa.field("candidate_id", pa.string(), nullable=False),
        pa.field("source_stable_id", pa.int64(), nullable=False),
        pa.field("target_stable_id", pa.int64(), nullable=False),
        # Kept separate from the calibrated LONG_FEATURE_SCHEMA gap_sec so a
        # missing-appearance rejection still retains its structural gap while
        # every unavailable model feature remains null.
        pa.field("temporal_gap_sec", pa.float64(), nullable=False),
        *_path_fields("source"),
        *_path_fields("target"),
        pa.field("temporally_nonoverlapping", pa.bool_(), nullable=False),
        pa.field("appearance_present", pa.bool_(), nullable=False),
        pa.field("source_gallery_present", pa.bool_(), nullable=False),
        pa.field("target_gallery_present", pa.bool_(), nullable=False),
        pa.field(
            "source_endpoint_review_excluded", pa.bool_(), nullable=False
        ),
        pa.field(
            "target_endpoint_review_excluded", pa.bool_(), nullable=False
        ),
        pa.field("high_overlap", pa.bool_(), nullable=False),
        *_gallery_provenance_fields("source", nullable=True),
        *_gallery_provenance_fields("target", nullable=True),
        *_retrieval_fields(nullable_values=True),
        *_gallery_score_fields(nullable=True),
        *_model_fields(nullable_values=True),
        pa.field("proposed_for_review", pa.bool_(), nullable=False),
    ]
)


LONG_LINK_PROPOSALS_SCHEMA = pa.schema(
    [
        pa.field("proposal_id", pa.string(), nullable=False),
        pa.field("candidate_id", pa.string(), nullable=False),
        pa.field("source_stable_id", pa.int64(), nullable=False),
        pa.field("target_stable_id", pa.int64(), nullable=False),
        pa.field("temporal_gap_sec", pa.float64(), nullable=False),
        *_path_fields("source"),
        *_path_fields("target"),
        pa.field("temporally_nonoverlapping", pa.bool_(), nullable=False),
        pa.field("appearance_present", pa.bool_(), nullable=False),
        pa.field("source_gallery_present", pa.bool_(), nullable=False),
        pa.field("target_gallery_present", pa.bool_(), nullable=False),
        pa.field(
            "source_endpoint_review_excluded", pa.bool_(), nullable=False
        ),
        pa.field(
            "target_endpoint_review_excluded", pa.bool_(), nullable=False
        ),
        pa.field("high_overlap", pa.bool_(), nullable=False),
        *_gallery_provenance_fields("source", nullable=False),
        *_gallery_provenance_fields("target", nullable=False),
        *_retrieval_fields(nullable_values=False),
        *_gallery_score_fields(nullable=False),
        *_model_fields(nullable_values=False),
        pa.field("evidence_status", pa.string(), nullable=False),
        pa.field("review_status", pa.string(), nullable=False),
    ]
)


__all__ = ["LONG_CANDIDATE_EDGES_SCHEMA", "LONG_LINK_PROPOSALS_SCHEMA"]
