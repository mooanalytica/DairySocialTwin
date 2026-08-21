from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pytest
import yaml

from cowtrack.config import ContractError
from cowtrack.linking.features import LONG_FEATURE_SCHEMA
from cowtrack.linking.long_proposal_config import load_long_proposal_config
from cowtrack.schemas.s05_proposals import (
    LONG_CANDIDATE_EDGES_SCHEMA,
    LONG_LINK_PROPOSALS_SCHEMA,
)


CONFIG = Path("configs/s05_proposals.yaml")


def _payload() -> dict[str, object]:
    result = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert isinstance(result, dict)
    return result


def _write(tmp_path: Path, payload: object) -> Path:
    path = tmp_path / "s05_proposals.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def test_repository_s05_proposal_config_is_fixed_and_proposal_only() -> None:
    config, payload, config_hash = load_long_proposal_config(CONFIG)

    assert payload["pipeline"]["execution_mode"] == "long_proposal_only"
    assert config.retrieval_method == "exact_cosine"
    assert config.approximate_index_allowed is False
    assert config.appearance_topk == 20
    assert config.temporal_nearest_k == 5
    assert config.min_gap_sec_exclusive == 5.0
    assert config.require_non_overlapping is True
    assert config.max_representative_prototypes == 3
    assert config.ordered_features == LONG_FEATURE_SCHEMA
    assert config.global_merge_allowed is False
    assert config.solver_allowed is False
    assert config.path_cover_allowed is False
    assert config.confirmed_links_allowed is False
    assert config.confirmed_decision_allowed is False
    assert config.artifacts.candidate_edges == "long_candidate_edges.parquet"
    assert len(config_hash) == 64


def test_s05_proposal_schemas_retain_provenance_scores_and_gate_evidence() -> None:
    required = {
        "source_constituent_micro_ids",
        "target_constituent_micro_ids",
        "source_start_det_id",
        "source_end_det_id",
        "target_start_det_id",
        "target_end_det_id",
        "source_start_global_frame",
        "source_end_global_frame",
        "target_start_global_frame",
        "target_end_global_frame",
        "source_start_time_sec",
        "source_end_time_sec",
        "target_start_time_sec",
        "target_end_time_sec",
        "selected_by_appearance_topk",
        "selected_by_temporal_nearest",
        "appearance_rank_out",
        "appearance_rank_in",
        "best_margin_out",
        "best_margin_in",
        "gallery_score_max",
        "gallery_score_top3",
        "gallery_score_src_to_dst",
        "gallery_score_dst_to_src",
        "gallery_score_mutual",
        "model_probability",
        "model_raw_score",
        "candidate_margin",
        "provisional_threshold",
        "selected_probability_threshold",
        "selected_margin_threshold",
        "source_gallery_present",
        "target_gallery_present",
        "source_endpoint_review_excluded",
        "target_endpoint_review_excluded",
        "passes_provisional_threshold",
        "passes_selected_probability_gate",
        "passes_selected_margin_gate",
        "passes_selected_gate",
        "decision",
        "decision_reason",
        "selected_by_solver",
        "confirmed",
        "merge_applied",
        *LONG_FEATURE_SCHEMA,
    }
    for schema in (LONG_CANDIDATE_EDGES_SCHEMA, LONG_LINK_PROPOSALS_SCHEMA):
        assert len(schema.names) == len(set(schema.names))
        assert required <= set(schema.names)
        assert "global_id" not in schema.names
        assert "global_track_uuid" not in schema.names
        assert schema.field("source_constituent_micro_ids").type == pa.list_(
            pa.field("element", pa.int64())
        )

    nullable_candidate_values = {
        "appearance_rank_out",
        "appearance_rank_in",
        "gallery_score_max",
        "gallery_score_top3",
        "gallery_score_src_to_dst",
        "gallery_score_dst_to_src",
        "gallery_score_mutual",
        "model_probability",
        "model_raw_score",
        *LONG_FEATURE_SCHEMA,
    }
    assert all(
        LONG_CANDIDATE_EDGES_SCHEMA.field(name).nullable
        for name in nullable_candidate_values
    )
    assert all(
        not LONG_LINK_PROPOSALS_SCHEMA.field(name).nullable
        for name in nullable_candidate_values
    )
    for schema in (LONG_CANDIDATE_EDGES_SCHEMA, LONG_LINK_PROPOSALS_SCHEMA):
        assert schema.field("best_margin_out").nullable
        assert schema.field("best_margin_in").nullable
        assert schema.field("candidate_margin").nullable


@pytest.mark.parametrize(
    ("section", "name", "value"),
    [
        ("pipeline", "global_merge_allowed", True),
        ("pipeline", "solver_allowed", True),
        ("pipeline", "path_cover_allowed", True),
        ("pipeline", "confirmed_links_allowed", True),
        ("scoring", "confirmed_decision_allowed", True),
        ("scoring", "short_model_fallback_allowed", True),
        ("scoring", "raw_cosine_decision_fallback_allowed", True),
        ("scoring", "motion_only_fallback_allowed", True),
    ],
)
def test_s05_proposal_config_rejects_safety_policy_substitution(
    tmp_path: Path, section: str, name: str, value: object
) -> None:
    payload = _payload()
    payload[section][name] = value
    with pytest.raises(ContractError):
        load_long_proposal_config(_write(tmp_path, payload))


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("method", "approximate_cosine"),
        ("approximate_index_allowed", True),
        ("appearance_topk", 19),
        ("temporal_nearest_k", 6),
        ("min_gap_sec_exclusive", 5.001),
        ("require_non_overlapping", False),
        ("max_representative_prototypes", 8),
    ],
)
def test_s05_proposal_config_rejects_retrieval_substitution(
    tmp_path: Path, name: str, value: object
) -> None:
    payload = _payload()
    payload["retrieval"][name] = value
    with pytest.raises(ContractError):
        load_long_proposal_config(_write(tmp_path, payload))


def test_s05_proposal_config_rejects_unknown_feature_artifact_and_bool_type(
    tmp_path: Path,
) -> None:
    payload = _payload()
    payload["unexpected"] = {}
    with pytest.raises(ContractError):
        load_long_proposal_config(_write(tmp_path, payload))

    payload = _payload()
    payload["scoring"]["ordered_features"] = list(reversed(LONG_FEATURE_SCHEMA))
    with pytest.raises(ContractError):
        load_long_proposal_config(_write(tmp_path, payload))

    payload = _payload()
    payload["artifacts"]["proposals"] = "unsafe.parquet"
    with pytest.raises(ContractError):
        load_long_proposal_config(_write(tmp_path, payload))

    payload = _payload()
    payload["pipeline"]["solver_allowed"] = 0
    with pytest.raises(ContractError):
        load_long_proposal_config(_write(tmp_path, payload))
