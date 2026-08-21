from __future__ import annotations

from pathlib import Path

import pytest

from cowtrack.config import ContractError
from cowtrack.qa.review_config import load_s01_review_config


ROOT = Path(__file__).resolve().parents[1]


def test_fixed_review_config_loads_with_strict_nvenc_contract() -> None:
    config, payload, config_hash = load_s01_review_config(
        ROOT / "configs" / "s01_review.yaml"
    )

    assert config.output_width == 1920
    assert config.output_height == 1080
    assert config.long_track_min_detections == 10_000
    assert config.good_sample_count == 20
    assert config.logical_gpu == 0
    assert payload["encoder"]["codec"] == "h264_nvenc"
    assert len(config_hash) == 64


def test_review_config_rejects_unknown_keys(tmp_path: Path) -> None:
    source = (ROOT / "configs" / "s01_review.yaml").read_text(encoding="utf-8")
    path = tmp_path / "review.yaml"
    path.write_text(source + "  cpu_fallback: true\n", encoding="utf-8")

    with pytest.raises(ContractError, match="encoder config keys mismatch"):
        load_s01_review_config(path)


def test_review_config_wraps_malformed_yaml(tmp_path: Path) -> None:
    path = tmp_path / "review.yaml"
    path.write_text("pipeline: [\n", encoding="utf-8")

    with pytest.raises(ContractError, match="cannot read S01 review config"):
        load_s01_review_config(path)
