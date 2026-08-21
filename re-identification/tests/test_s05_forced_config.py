from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from cowtrack.config import ContractError
from cowtrack.linking.forced_appearance_config import load_forced_appearance_config


CONFIG = Path("configs/s05_force_appearance.yaml")


def test_forced_appearance_config_is_exact_62_contract() -> None:
    config, payload, digest = load_forced_appearance_config(CONFIG)
    assert config.target_global_track_count == 62
    assert config.expected_stable_track_count is None
    assert config.expected_microtrack_count is None
    assert config.expected_clean_appearance_count is None
    assert config.expected_prior_link_count is None
    assert len(config.clip_order) == 11
    assert config.selected_encoder == "megadescriptor_l_384_imagenet"
    assert config.device == "cuda:0"
    assert config.source_top_k == config.target_top_k == 32
    assert payload["appearance"]["allow_best_degraded_crop"] is True
    assert payload["solver"]["algorithm"] == (
        "deterministic_full_sequence_fixed_cardinality_min_cost_flow"
    )
    assert payload["solver"]["objective"] == (
        "exact_62_global_paths_then_minimum_full_sequence_appearance_cost"
    )
    assert len(digest) == 64


def test_forced_appearance_config_rejects_policy_drift(tmp_path: Path) -> None:
    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    payload["solver"]["target_global_track_count"] = 57
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    with pytest.raises(ContractError, match="target_global_track_count=62"):
        load_forced_appearance_config(path)


def test_forced_appearance_config_rejects_extra_keys(tmp_path: Path) -> None:
    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    payload["solver"]["fallback"] = "forbidden"
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    with pytest.raises(ContractError, match="keys mismatch"):
        load_forced_appearance_config(path)
