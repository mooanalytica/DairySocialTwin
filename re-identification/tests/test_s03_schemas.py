from __future__ import annotations

from cowtrack.linking.features import LONG_FEATURE_SCHEMA, SHORT_FEATURE_SCHEMA
from cowtrack.schemas.calibration import PSEUDO_PAIRS_SCHEMA, S03_ALL_FEATURES


def test_pair_schema_persists_clean_gallery_and_split_provenance() -> None:
    required = {
        "candidate_group_id",
        "parent_group_id",
        "split",
        "calibration_role",
        "source_sample_ids",
        "target_sample_ids",
        "source_gallery_det_ids",
        "target_gallery_det_ids",
        "source_embedding_rows",
        "target_embedding_rows",
        "source_internal_cosine_p10",
        "target_internal_cosine_p10",
        "stratum",
        "high_overlap",
        "candidate_margin",
        "decision",
    }
    assert required.issubset(PSEUDO_PAIRS_SCHEMA.names)
    assert set(SHORT_FEATURE_SCHEMA).issubset(PSEUDO_PAIRS_SCHEMA.names)
    assert set(LONG_FEATURE_SCHEMA).issubset(PSEUDO_PAIRS_SCHEMA.names)


def test_model_feature_names_exclude_forbidden_identity_inputs() -> None:
    lowered = {name.lower() for name in S03_ALL_FEATURES}
    assert not any("legacy" in name for name in lowered)
    assert not any("keypoint" in name for name in lowered)
    assert not any("absolute" in name for name in lowered)


def test_only_inapplicable_or_unavailable_scores_are_nullable() -> None:
    nullable = {field.name for field in PSEUDO_PAIRS_SCHEMA if field.nullable}
    assert nullable == {
        "predicted_center_residual",
        "predicted_iou",
        "model_probability",
        "model_raw_score",
        "candidate_margin",
    }
