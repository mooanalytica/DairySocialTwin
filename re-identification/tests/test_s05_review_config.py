from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from cowtrack.config import ContractError
from cowtrack.qa.s05_review_config import load_s05_review_config


CONFIG = Path(__file__).parents[1] / "configs" / "s05_review.yaml"


def test_loads_fixed_s05_review_config_and_json_roundtrips(tmp_path: Path) -> None:
    config, payload, digest = load_s05_review_config(CONFIG)

    assert config.random_seed == 20260710
    assert config.maximum_cases == 50
    assert (
        config.high_score_count,
        config.threshold_near_count,
        config.ambiguous_count,
        config.random_count,
    ) == (15, 15, 10, 10)
    assert config.threshold_probability_band == 0.03
    assert config.ambiguity_margin_upper == 0.08
    assert (config.output_width, config.output_height) == (1920, 1080)
    assert payload["pipeline"]["review_semantics"] == "provisional_evidence_only"
    assert len(digest) == 64

    effective = tmp_path / "effective_config.json"
    effective.write_text(json.dumps(payload), encoding="utf-8")
    roundtrip, roundtrip_payload, roundtrip_digest = load_s05_review_config(effective)
    assert roundtrip == config
    assert roundtrip_payload == payload
    assert roundtrip_digest == digest


def test_rejects_changed_selection_or_semantics(tmp_path: Path) -> None:
    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    payload["selection"]["maximum_cases"] = 49
    path = tmp_path / "changed.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ContractError, match="fixed S05 review"):
        load_s05_review_config(path)

    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    payload["pipeline"]["review_semantics"] = "automatic_merge"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ContractError, match="provisional-evidence-only"):
        load_s05_review_config(path)


def test_rejects_cpu_encoder_or_wrong_gpu_mapping(tmp_path: Path) -> None:
    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    path = tmp_path / "changed.yaml"
    payload["encoder"]["codec"] = "libx264"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ContractError, match="h264_nvenc"):
        load_s05_review_config(path)

    payload["encoder"]["codec"] = "h264_nvenc"
    payload["encoder"]["logical_gpu"] = 1
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ContractError, match="logical_gpu=0"):
        load_s05_review_config(path)
