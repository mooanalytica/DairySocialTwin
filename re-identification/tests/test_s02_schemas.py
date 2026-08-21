from __future__ import annotations

import pyarrow as pa

from cowtrack.schemas.appearance import (
    APPEARANCE_EXCLUSIONS_SCHEMA,
    APPEARANCE_SAMPLES_SCHEMA,
    MICRO_APPEARANCE_SCHEMA,
    MICRO_CONTEXT_SCHEMA,
)


def test_appearance_samples_schema_is_stable() -> None:
    assert APPEARANCE_SAMPLES_SCHEMA.names == [
        "sample_id",
        "micro_id",
        "det_id",
        "clip_id",
        "global_frame",
        "global_time_sec",
        "crop_quality",
        "other_bbox_max_iou",
        "clipped_fraction",
        "bbox_area_percentile",
        "blur_score",
        "distance_to_image_boundary",
        "selection_reason",
        "prototype_inlier",
        "prototype_outlier",
        "embedding_row",
    ]
    assert APPEARANCE_SAMPLES_SCHEMA.field("sample_id").type == pa.int64()
    assert APPEARANCE_SAMPLES_SCHEMA.field("crop_quality").type == pa.float32()
    assert APPEARANCE_SAMPLES_SCHEMA.field("prototype_inlier").type == pa.bool_()
    assert APPEARANCE_SAMPLES_SCHEMA.field("embedding_row").type == pa.int64()


def test_appearance_exclusions_schema_persists_union_provenance() -> None:
    assert APPEARANCE_EXCLUSIONS_SCHEMA.names == [
        "det_id",
        "micro_id",
        "global_frame",
        "appearance_excluded",
        "crop_quality",
        "exclusion_reasons",
        "review_case_ids",
        "trigger_global_frames",
    ]
    assert APPEARANCE_EXCLUSIONS_SCHEMA.field("appearance_excluded").type == pa.bool_()
    assert APPEARANCE_EXCLUSIONS_SCHEMA.field("exclusion_reasons").type == pa.list_(
        pa.string()
    )
    assert APPEARANCE_EXCLUSIONS_SCHEMA.field("trigger_global_frames").type == pa.list_(
        pa.int64()
    )


def test_micro_appearance_schema_matches_prototype_rows() -> None:
    assert MICRO_APPEARANCE_SCHEMA.names == [
        "micro_id",
        "prototype_row",
        "num_appearance_samples",
        "appearance_usable",
        "appearance_quality",
        "internal_cosine_p10",
        "internal_cosine_p50",
        "internal_cosine_min",
        "appearance_outlier_count",
        "selected_encoder",
    ]
    assert MICRO_APPEARANCE_SCHEMA.field("num_appearance_samples").type == pa.int16()
    assert MICRO_APPEARANCE_SCHEMA.field("appearance_usable").type == pa.bool_()
    assert MICRO_APPEARANCE_SCHEMA.field("selected_encoder").type == pa.string()


def test_micro_context_closes_s03_endpoint_contract() -> None:
    assert MICRO_CONTEXT_SCHEMA.names == [
        "micro_id",
        "start_det_id",
        "end_det_id",
        "start_clip_id",
        "end_clip_id",
        "start_max_other_iou",
        "end_max_other_iou",
        "start_boundary_distance",
        "end_boundary_distance",
        "start_global_frame",
        "end_global_frame",
    ]
    assert MICRO_CONTEXT_SCHEMA.field("start_global_frame").type == pa.int64()
    assert MICRO_CONTEXT_SCHEMA.field("end_boundary_distance").type == pa.float32()


def test_s02_parquet_schemas_have_no_nullable_fields() -> None:
    for schema in (
        APPEARANCE_SAMPLES_SCHEMA,
        APPEARANCE_EXCLUSIONS_SCHEMA,
        MICRO_APPEARANCE_SCHEMA,
        MICRO_CONTEXT_SCHEMA,
    ):
        assert [field.name for field in schema if field.nullable] == []
