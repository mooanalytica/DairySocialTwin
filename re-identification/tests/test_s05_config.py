from pathlib import Path

import pytest
import yaml

from cowtrack.config import ContractError
from cowtrack.linking.long_calibration_config import load_long_calibration_config


CONFIG = Path("configs/s05_long_calibration.yaml")


def test_repository_s05_long_config_is_fixed() -> None:
    config, payload, config_hash = load_long_calibration_config(CONFIG)
    assert payload["pipeline"]["execution_mode"] == "long_calibration_only"
    assert config.global_merge_allowed is False
    assert config.min_gap_sec_exclusive == 5.0
    assert config.artifacts.pairs_certification == (
        "long_pairs_calibration_certification.parquet"
    )
    assert len(config_hash) == 64


def test_s05_config_rejects_unknown_or_changed_safety_policy(tmp_path: Path) -> None:
    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    payload["pipeline"]["global_merge_allowed"] = True
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ContractError):
        load_long_calibration_config(path)

    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    payload["split"]["certification_fraction"] = 0.11
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ContractError):
        load_long_calibration_config(path)

    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    payload["unexpected"] = {}
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ContractError):
        load_long_calibration_config(path)


def test_s05_config_rejects_artifact_or_split_substitution(tmp_path: Path) -> None:
    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    payload["artifacts"]["link_model_long"] = "different.joblib"
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ContractError):
        load_long_calibration_config(path)


def test_s05_config_rejects_seed_codec_and_boolean_type_substitution(
    tmp_path: Path,
) -> None:
    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    payload["pipeline"]["random_seed"] += 1
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ContractError):
        load_long_calibration_config(path)

    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    payload["runtime"]["parquet_compression"] = "snappy"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ContractError):
        load_long_calibration_config(path)

    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    payload["pipeline"]["global_merge_allowed"] = 0
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ContractError):
        load_long_calibration_config(path)
