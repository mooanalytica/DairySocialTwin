from __future__ import annotations

import pyarrow as pa

from cowtrack.schemas.stable_appearance import STABLE_APPEARANCE_SCHEMA


def test_stable_appearance_schema_retains_gallery_and_missing_provenance() -> None:
    assert STABLE_APPEARANCE_SCHEMA.names == [
        "stable_id",
        "prototype_row",
        "constituent_micro_ids",
        "num_input_samples",
        "num_s02_inliers",
        "num_clean_candidates",
        "num_clean_inliers",
        "num_valid_prototypes",
        "appearance_usable",
        "missing_reason",
        "clean_sample_ids",
        "clean_det_ids",
        "clean_embedding_rows",
        "medoid_sample_id",
        "appearance_quality",
        "internal_cosine_p10",
        "internal_cosine_p50",
        "internal_cosine_min",
        "num_overlap_rejected",
        "num_review_excluded",
        "num_local_outliers",
        "max_other_bbox_iou",
        "max_clean_other_bbox_iou",
    ]
    assert STABLE_APPEARANCE_SCHEMA.field("stable_id").type == pa.int64()
    assert STABLE_APPEARANCE_SCHEMA.field("prototype_row").type == pa.int64()
    assert STABLE_APPEARANCE_SCHEMA.field("constituent_micro_ids").type == pa.list_(
        pa.int64()
    )
    assert STABLE_APPEARANCE_SCHEMA.field("appearance_usable").type == pa.bool_()
    assert STABLE_APPEARANCE_SCHEMA.field("missing_reason").nullable
    assert STABLE_APPEARANCE_SCHEMA.field("appearance_quality").nullable


def test_only_missing_or_gallery_dependent_fields_are_nullable() -> None:
    nullable = {
        field.name for field in STABLE_APPEARANCE_SCHEMA if field.nullable
    }
    assert nullable == {
        "missing_reason",
        "medoid_sample_id",
        "appearance_quality",
        "internal_cosine_p10",
        "internal_cosine_p50",
        "internal_cosine_min",
        "num_local_outliers",
        "max_clean_other_bbox_iou",
    }
