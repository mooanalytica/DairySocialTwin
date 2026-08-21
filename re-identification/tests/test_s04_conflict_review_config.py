from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from cowtrack.config import ContractError
from cowtrack.qa.s04_conflict_review_config import (
    load_s04_conflict_review_config,
)


CONFIG = Path(__file__).parents[1] / "configs" / "s04_conflict_review.yaml"


def test_loads_fixed_s04_conflict_review_config() -> None:
    config, payload, digest = load_s04_conflict_review_config(CONFIG)

    assert config.random_seed == 20260710
    assert (
        config.largest_group_count,
        config.random_group_count,
        config.max_edges_per_group,
    ) == (3, 3, 3)
    assert (config.source_tail_sec, config.target_head_sec) == (2.0, 2.0)
    assert (config.output_width, config.output_height) == (1920, 1080)
    assert payload["pipeline"]["schema_version"] == "1.0"
    assert payload["encoder"]["codec"] == "h264_nvenc"
    assert len(digest) == 64


@pytest.mark.parametrize(
    ("section", "key", "value", "message"),
    [
        ("pipeline", "random_seed", 7, "20260710"),
        ("selection", "largest_group_count", 2, "largest_groups=3"),
        ("selection", "random_group_count", 2, "random_groups=3"),
        ("selection", "max_edges_per_group", 4, "max_edges_per_group=3"),
    ],
)
def test_rejects_changed_fixed_contract(
    tmp_path: Path,
    section: str,
    key: str,
    value: int,
    message: str,
) -> None:
    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    payload[section][key] = value
    path = tmp_path / "changed.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ContractError, match=message):
        load_s04_conflict_review_config(path)


def test_rejects_extra_config_key(tmp_path: Path) -> None:
    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    payload["selection"]["unexpected"] = True
    path = tmp_path / "changed.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ContractError, match="keys mismatch"):
        load_s04_conflict_review_config(path)


def test_rejects_non_nvenc_or_non_gpu1_logical_mapping(tmp_path: Path) -> None:
    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    payload["encoder"]["codec"] = "libx264"
    path = tmp_path / "changed.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ContractError, match="h264_nvenc"):
        load_s04_conflict_review_config(path)

    payload["encoder"]["codec"] = "h264_nvenc"
    payload["encoder"]["logical_gpu"] = 1
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ContractError, match="logical_gpu=0"):
        load_s04_conflict_review_config(path)
