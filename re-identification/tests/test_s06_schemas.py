from __future__ import annotations

import pyarrow as pa

from cowtrack.schemas.s06 import (
    DETECTIONS_WITH_GLOBAL_ID_SCHEMA,
    GLOBAL_TRACK_SUMMARY_SCHEMA,
    LOW_CONFIDENCE_LINKS_SCHEMA,
)


def test_s06_detection_csv_schema_is_unique_and_audit_friendly() -> None:
    schema = DETECTIONS_WITH_GLOBAL_ID_SCHEMA
    assert len(schema.names) == len(set(schema.names))
    assert schema.names[:11] == [
        "sequence_id",
        "clip_id",
        "clip_order",
        "det_id",
        "csv_row_index",
        "legacy_track_id",
        "global_track_id",
        "global_track_uuid",
        "display_global_id",
        "id_status",
        "identity_basis",
    ]
    assert schema.field("legacy_track_id").type == pa.string()
    assert schema.field("global_track_id").type == pa.int64()
    assert schema.field("display_global_id").type == pa.string()
    assert schema.field("bbox_confidence").type == pa.float32()
    assert schema.field("qa_flags").type == pa.uint32()
    assert schema.field("invalid_reason").type == pa.string()


def test_s06_invalid_rows_can_only_have_null_identity_mapping() -> None:
    schema = DETECTIONS_WITH_GLOBAL_ID_SCHEMA
    nullable_identity_fields = {
        "global_track_id",
        "global_track_uuid",
        "display_global_id",
        "id_status",
        "identity_basis",
        "micro_id",
        "stable_id",
        "order_in_micro",
        "order_in_stable",
        "order_in_stable_detection",
        "order_in_global_stable",
        "order_in_global_detection",
        "local_purity_score",
        "assignment_confidence",
        "incoming_link_probability",
        "outgoing_link_probability",
    }
    assert all(schema.field(name).nullable for name in nullable_identity_fields)
    assert not schema.field("valid").nullable
    assert not schema.field("csv_row_index").nullable
    assert not schema.field("det_id").nullable


def test_s06_probability_fields_are_explicitly_nullable_not_cosine_aliases() -> None:
    detection = DETECTIONS_WITH_GLOBAL_ID_SCHEMA
    for name in (
        "assignment_confidence",
        "incoming_link_probability",
        "outgoing_link_probability",
    ):
        assert detection.field(name).nullable
        assert detection.field(name).type == pa.float64()

    summary = GLOBAL_TRACK_SUMMARY_SCHEMA
    for name in (
        "min_link_probability",
        "p10_link_probability",
        "mean_link_probability",
        "max_link_probability",
    ):
        assert summary.field(name).nullable
    assert not summary.field("probability_not_available_reason").nullable


def test_s06_global_summary_has_required_forced_provisional_evidence() -> None:
    schema = GLOBAL_TRACK_SUMMARY_SCHEMA
    assert len(schema.names) == len(set(schema.names))
    required = {
        "global_track_id",
        "display_global_id",
        "id_status",
        "authorization_basis",
        "certification_claimed",
        "num_stable_tracklets",
        "num_microtracklets",
        "num_detections",
        "duration_visible_sec",
        "spans_multiple_clips",
        "selected_link_cosine_min",
        "selected_link_cosine_p10",
        "selected_link_cosine_mean",
        "selected_link_cosine_max",
        "longest_gap_sec",
        "num_links_cosine_below_0_3",
        "num_links_cosine_below_0_4",
        "num_links_cosine_below_0_5",
        "num_links_cosine_below_0_6",
        "num_link_endpoints_grade_a_clean",
        "num_link_endpoints_grade_b_existing_degraded",
        "num_link_endpoints_grade_c_reencoded_degraded",
        "population_soft_max",
        "population_overflow",
        "population_warning",
    }
    assert required <= set(schema.names)
    assert schema.field("selected_link_cosine_min").nullable
    assert schema.field("longest_gap_sec").nullable


def test_s06_low_confidence_schema_names_cosine_rank_and_margin_precisely() -> None:
    schema = LOW_CONFIDENCE_LINKS_SCHEMA
    assert len(schema.names) == len(set(schema.names))
    assert schema.field("appearance_cosine").type == pa.float32()
    assert schema.field("source_candidate_rank").type == pa.int32()
    assert schema.field("target_candidate_rank").type == pa.int32()
    assert schema.field("source_second_best_cosine_margin").nullable
    assert schema.field("target_second_best_cosine_margin").nullable
    assert schema.field("conservative_cosine_margin").nullable
    assert not any("probability_margin" in name for name in schema.names)
