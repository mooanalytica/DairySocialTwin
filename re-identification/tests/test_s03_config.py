from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from cowtrack.config import ContractError
from cowtrack.linking.config import (
    load_calibration_config,
    load_link_calibration_config,
)


REPOSITORY_CONFIG = (
    Path(__file__).resolve().parents[1] / "configs" / "s03_calibration.yaml"
)


def _payload() -> dict[str, object]:
    payload = yaml.safe_load(REPOSITORY_CONFIG.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _write(path: Path, payload: dict[str, object], *, sort_keys: bool = False) -> None:
    path.write_text(
        yaml.safe_dump(payload, sort_keys=sort_keys), encoding="utf-8"
    )


def _section(payload: dict[str, object], name: str) -> dict[str, object]:
    section = payload[name]
    assert isinstance(section, dict)
    return section


def test_repository_s03_config_loads_approved_contract() -> None:
    config, payload, config_hash = load_link_calibration_config(REPOSITORY_CONFIG)

    assert config.schema_version == "1.0"
    assert config.random_seed == 20260710
    assert config.use_keypoints is False
    assert config.use_legacy_tracking_id is False
    assert config.split_fractions == {
        "train": 0.60,
        "calibration": 0.20,
        "audit": 0.20,
    }
    assert config.calibration_selection_fraction == 0.50
    assert config.calibration_certification_fraction == 0.50
    assert config.split_group_by_parent_micro is True
    assert config.split_group_cross_clip_micro is True
    assert config.split_hard_negative_mining_per_split is True
    assert config.short_gap_targets_sec == (0.1, 0.5, 1.0, 2.0, 5.0)
    assert config.long_gap_targets_sec == (10.0, 30.0, 60.0)
    assert config.min_local_purity_score == 0.99
    assert config.min_bidirectional_agreement == 1.0
    assert config.min_internal_cosine_p10 == 0.70
    assert config.clean_max_other_bbox_iou == 0.25
    assert config.clean_iou_comparison == "less_than"
    assert config.clean_min_samples_per_side == 3
    assert config.outlier_medoid_cosine == 0.65
    assert config.outlier_support_cosine == 0.70
    assert config.new_prototype_cosine == 0.92
    assert config.missing_appearance_decision == "reject"
    assert config.include_constant_velocity_short is True
    assert config.include_constant_velocity_long is False
    assert config.hard_negative_per_split_independent is True
    assert config.model_estimator == "logistic_regression"
    assert config.model_penalty == "l2"
    assert config.model_regularization_c == 1.0
    assert config.threshold_confidence_level == 0.95
    assert config.confirmed_false_accept_rate_target == 1e-4
    assert config.provisional_tpr_target == 0.95
    assert config.disable_confirmed_when_unproven is True
    assert config.aggressive_global_merge_allowed is False
    assert config.insufficient_sample_action == "disable_affected_model"
    assert config.artifacts.link_model_short == "link_model_short.joblib"
    assert config.artifacts.link_model_long == "link_model_long.joblib"
    assert config.artifacts.pseudo_pairs_calibration == (
        "pseudo_pairs_calibration.parquet"
    )
    assert payload["runtime"]["log_flush"] is True  # type: ignore[index]
    assert len(config_hash) == 64
    assert int(config_hash, 16) >= 0


def test_loader_alias_returns_identical_contract() -> None:
    direct = load_link_calibration_config(REPOSITORY_CONFIG)
    alias = load_calibration_config(REPOSITORY_CONFIG)

    assert alias == direct


@pytest.mark.parametrize(
    ("section", "key"),
    [
        ("pipeline", "random_seed"),
        ("split", "audit_fraction"),
        ("pseudo_positive", "short_gap_targets_sec"),
        ("appearance", "clean_iou_comparison"),
        ("motion", "velocity_history_detections"),
        ("hard_negative", "strategy"),
        ("model", "regularization_c"),
        ("thresholds", "confirmed_method"),
        ("minimum_samples", "audit_positive_per_model"),
        ("runtime", "progress_interval_sec"),
        ("artifacts", "pseudo_pairs_audit"),
    ],
)
def test_s03_config_rejects_missing_key(
    tmp_path: Path, section: str, key: str
) -> None:
    payload = _payload()
    del _section(payload, section)[key]
    path = tmp_path / "missing.yaml"
    _write(path, payload)

    with pytest.raises(ContractError, match="missing="):
        load_link_calibration_config(path)


def test_s03_config_rejects_unknown_section(tmp_path: Path) -> None:
    payload = _payload()
    payload["fallback"] = {"enabled": True}
    path = tmp_path / "extra-section.yaml"
    _write(path, payload)

    with pytest.raises(ContractError, match="extra=.*fallback"):
        load_link_calibration_config(path)


def test_s03_config_rejects_unknown_nested_key(tmp_path: Path) -> None:
    payload = _payload()
    _section(payload, "model")["motion_only_fallback"] = True
    path = tmp_path / "extra-key.yaml"
    _write(path, payload)

    with pytest.raises(ContractError, match="extra=.*motion_only_fallback"):
        load_link_calibration_config(path)


@pytest.mark.parametrize(
    ("section", "key", "bad_value", "message"),
    [
        ("pipeline", "use_keypoints", True, "use_keypoints=False"),
        (
            "pipeline",
            "use_legacy_tracking_id",
            True,
            "use_legacy_tracking_id=False",
        ),
        ("split", "train_fraction", 0.50, "train_fraction=0.6"),
        (
            "pseudo_positive",
            "short_gap_targets_sec",
            [0.1, 5.0],
            "short_gap_targets_sec",
        ),
        (
            "appearance",
            "clean_iou_comparison",
            "less_than_or_equal",
            "clean_iou_comparison='less_than'",
        ),
        (
            "appearance",
            "missing_appearance_decision",
            "provisional",
            "missing_appearance_decision='reject'",
        ),
        (
            "motion",
            "include_constant_velocity_long",
            True,
            "include_constant_velocity_long=False",
        ),
        (
            "hard_negative",
            "require_positive_gap",
            False,
            "require_positive_gap=True",
        ),
        ("model", "estimator", "random_forest", "logistic_regression"),
        (
            "thresholds",
            "confirmed_false_accept_rate_target",
            0.001,
            "confirmed_false_accept_rate_target=0.0001",
        ),
        (
            "thresholds",
            "disable_confirmed_when_unproven",
            False,
            "disable_confirmed_when_unproven=True",
        ),
    ],
)
def test_s03_config_rejects_changes_to_approved_policy(
    tmp_path: Path,
    section: str,
    key: str,
    bad_value: object,
    message: str,
) -> None:
    payload = _payload()
    _section(payload, section)[key] = bad_value
    path = tmp_path / "changed.yaml"
    _write(path, payload)

    with pytest.raises(ContractError, match=message):
        load_link_calibration_config(path)


@pytest.mark.parametrize(
    ("section", "key"),
    [
        ("pseudo_positive", "min_local_purity_score"),
        ("appearance", "outlier_medoid_cosine"),
        ("model", "regularization_c"),
        ("thresholds", "confidence_level"),
        ("runtime", "progress_interval_sec"),
    ],
)
def test_s03_config_rejects_nonfinite_number(
    tmp_path: Path, section: str, key: str
) -> None:
    payload = _payload()
    _section(payload, section)[key] = float("nan")
    path = tmp_path / "nan.yaml"
    _write(path, payload)

    with pytest.raises(ContractError, match="finite number"):
        load_link_calibration_config(path)


def test_s03_config_rejects_boolean_as_integer(tmp_path: Path) -> None:
    payload = _payload()
    _section(payload, "runtime")["worker_count"] = True
    path = tmp_path / "bool.yaml"
    _write(path, payload)

    with pytest.raises(ContractError, match="worker_count must be an integer"):
        load_link_calibration_config(path)


def test_s03_config_rejects_bad_minimum_sample_count(tmp_path: Path) -> None:
    payload = _payload()
    _section(payload, "minimum_samples")["audit_pairs_per_class_per_clip"] = 0
    path = tmp_path / "minimum.yaml"
    _write(path, payload)

    with pytest.raises(ContractError, match="minimum_samples counts must be positive"):
        load_link_calibration_config(path)


def test_s03_config_rejects_artifact_substitution(tmp_path: Path) -> None:
    payload = _payload()
    _section(payload, "artifacts")["thresholds"] = "unsafe.json"
    path = tmp_path / "artifact.yaml"
    _write(path, payload)

    with pytest.raises(ContractError, match="artifact filenames are fixed"):
        load_link_calibration_config(path)


def test_s03_config_hash_is_canonical_over_mapping_order(tmp_path: Path) -> None:
    payload = _payload()
    reordered = tmp_path / "reordered.yaml"
    _write(reordered, payload, sort_keys=True)

    _, loaded_payload, expected_hash = load_link_calibration_config(
        REPOSITORY_CONFIG
    )
    _, reordered_payload, reordered_hash = load_link_calibration_config(reordered)

    assert reordered_payload == loaded_payload
    assert reordered_hash == expected_hash
