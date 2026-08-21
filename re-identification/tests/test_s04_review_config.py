from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from cowtrack.config import ContractError
from cowtrack.qa.s04_review_config import load_s04_review_config


CONFIG = Path(__file__).parents[1] / "configs" / "s04_review.yaml"


def test_loads_fixed_s04_review_config() -> None:
    config, payload, digest = load_s04_review_config(CONFIG)

    assert (config.top_count, config.bottom_count, config.random_count) == (15, 15, 20)
    assert config.random_seed == 20260710
    assert (config.source_tail_sec, config.target_head_sec) == (2.0, 2.0)
    assert (config.output_width, config.output_height) == (1920, 1080)
    assert payload["encoder"]["codec"] == "h264_nvenc"
    assert len(digest) == 64


def test_rejects_changed_sampling_contract(tmp_path: Path) -> None:
    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    payload["selection"]["top_probability_count"] = 14
    payload["selection"]["bottom_probability_count"] = 16
    path = tmp_path / "changed.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ContractError, match="top=15"):
        load_s04_review_config(path)


def test_rejects_changed_random_seed(tmp_path: Path) -> None:
    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    payload["pipeline"]["random_seed"] = 7
    path = tmp_path / "changed.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ContractError, match="20260710"):
        load_s04_review_config(path)


def test_rejects_non_nvenc_or_non_gpu1_logical_mapping(tmp_path: Path) -> None:
    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    payload["encoder"]["codec"] = "libx264"
    path = tmp_path / "changed.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ContractError, match="h264_nvenc"):
        load_s04_review_config(path)

    payload["encoder"]["codec"] = "h264_nvenc"
    payload["encoder"]["logical_gpu"] = 1
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ContractError, match="logical_gpu=0"):
        load_s04_review_config(path)
