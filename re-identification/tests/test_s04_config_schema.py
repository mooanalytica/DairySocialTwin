from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from cowtrack.config import ContractError
from cowtrack.linking.dataset_contract import EXPECTED_CLIP_ORDER
from cowtrack.linking.features import SHORT_FEATURE_SCHEMA
from cowtrack.linking.proposal_config import load_short_proposal_config
from cowtrack.schemas.s04 import SHORT_CANDIDATE_EDGES_SCHEMA


CONFIG = Path(__file__).parents[1] / "configs" / "s04_proposals.yaml"


def test_fixed_s04_config_loads() -> None:
    config, payload, digest = load_short_proposal_config(CONFIG)
    assert config.execution_mode == "proposal_only"
    assert config.automatic_merge_allowed is False
    assert config.human_labels_applied is False
    assert config.accepted_decision_for_review == "provisional"
    assert config.clip_order == EXPECTED_CLIP_ORDER
    assert len(digest) == 64
    assert payload["pipeline"]["random_seed"] == 20260710


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("pipeline", "execution_mode", "automatic"),
        ("pipeline", "automatic_merge_allowed", True),
        ("pipeline", "human_labels_applied", True),
        ("proposals", "accepted_decision_for_review", "confirmed"),
        ("proposals", "max_gap_sec", 5.1),
        ("proposals", "clip_order", list(reversed(EXPECTED_CLIP_ORDER))),
    ],
)
def test_fixed_s04_config_rejects_policy_changes(
    tmp_path: Path, section: str, key: str, value: object
) -> None:
    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    payload[section][key] = value
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ContractError):
        load_short_proposal_config(path)


def test_candidate_feature_columns_are_contiguous_and_in_s03_order() -> None:
    names = SHORT_CANDIDATE_EDGES_SCHEMA.names
    first = names.index(SHORT_FEATURE_SCHEMA[0])
    assert tuple(names[first : first + len(SHORT_FEATURE_SCHEMA)]) == SHORT_FEATURE_SCHEMA
    assert "source_gallery_present" in names
    assert "target_gallery_reason" in names
    assert SHORT_CANDIDATE_EDGES_SCHEMA.field("gap_sec").nullable is True
    assert (
        SHORT_CANDIDATE_EDGES_SCHEMA.field("source_sample_ids")
        .type.value_field.name
        == "element"
    )
