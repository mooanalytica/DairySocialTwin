from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from cowtrack.appearance.config import load_appearance_config
from cowtrack.config import ContractError


REPOSITORY_CONFIG = (
    Path(__file__).resolve().parents[1] / "configs" / "s02_appearance.yaml"
)


def _payload() -> dict[str, object]:
    payload = yaml.safe_load(REPOSITORY_CONFIG.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _write(path: Path, payload: dict[str, object]) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def test_repository_appearance_config_loads_fixed_contract() -> None:
    config, payload, config_hash = load_appearance_config(REPOSITORY_CONFIG)

    assert [profile.name for profile in config.model_profiles] == [
        "megadescriptor_l_384_imagenet",
    ]
    assert config.selection_mode == "locked_existing_winner"
    assert [profile.input_size for profile in config.model_profiles] == [384]
    assert [profile.resize_mode for profile in config.model_profiles] == [
        "shortest_edge_center_crop",
    ]
    assert [profile.crop_pct for profile in config.model_profiles] == [0.9]
    assert [profile.interpolation for profile in config.model_profiles] == [
        "bicubic",
    ]
    assert config.device == "cuda:0"
    assert config.head_samples == config.tail_samples == 3
    assert config.representative_period_sec == 0.75
    assert config.prototypes_per_tracklet == 3
    assert config.prototype_outlier_cosine_min == 0.65
    assert config.prototype_outlier_support_cosine == 0.70
    assert config.prototype_cluster_min_separation == 0.08
    assert config.candidate_pool_max_samples_per_microtrack == 48
    assert config.quality_blur_measurement_size == 256
    assert config.review_event_half_window_frames == 15
    assert config.artifacts.micro_prototype_mask == "micro_prototype_mask.npy"
    assert config.artifacts.micro_context == "micro_context.parquet"
    assert config.artifacts.effective_config == "effective_config.json"
    assert config.artifacts.appearance_report == "appearance_report.json"
    assert payload["runtime"]["device"] == "cuda:0"  # type: ignore[index]
    assert len(config_hash) == 64


@pytest.mark.parametrize(
    ("section", "key"),
    [
        ("sampling", "representative_period_sec"),
        ("quality", "blur_scale"),
        ("quality", "blur_measurement_size"),
        ("prototypes", "outlier_cosine_min"),
        ("prototypes", "outlier_support_cosine"),
        ("models", "selection_mode"),
        ("runtime", "batch_size"),
        ("artifacts", "micro_context"),
        ("artifacts", "effective_config"),
    ],
)
def test_appearance_config_rejects_missing_required_key(
    tmp_path: Path, section: str, key: str
) -> None:
    payload = _payload()
    section_payload = payload[section]
    assert isinstance(section_payload, dict)
    del section_payload[key]
    path = tmp_path / "missing.yaml"
    _write(path, payload)

    with pytest.raises(ContractError, match="missing="):
        load_appearance_config(path)


def test_appearance_config_rejects_unknown_key(tmp_path: Path) -> None:
    payload = _payload()
    sampling = payload["sampling"]
    assert isinstance(sampling, dict)
    sampling["allow_fallback"] = True
    path = tmp_path / "extra.yaml"
    _write(path, payload)

    with pytest.raises(ContractError, match="extra=.*allow_fallback"):
        load_appearance_config(path)


@pytest.mark.parametrize(
    ("section", "key", "bad_value", "message"),
    [
        ("runtime", "device", "cpu", "runtime.device=cuda:0"),
        ("sampling", "horizontal_flip", True, "horizontal_flip=false"),
        ("sampling", "rotations", [0], "rotations=\\[0, 180\\]"),
        ("review_exclusion", "event_half_window_frames", 14, "frames=15"),
        (
            "models",
            "selection_mode",
            "benchmark",
            "selection_mode=locked_existing_winner",
        ),
    ],
)
def test_appearance_config_rejects_changes_to_fixed_contract(
    tmp_path: Path,
    section: str,
    key: str,
    bad_value: object,
    message: str,
) -> None:
    payload = _payload()
    section_payload = payload[section]
    assert isinstance(section_payload, dict)
    section_payload[key] = bad_value
    path = tmp_path / "changed.yaml"
    _write(path, payload)

    with pytest.raises(ContractError, match=message):
        load_appearance_config(path)


def test_appearance_config_rejects_model_path_substitution(tmp_path: Path) -> None:
    payload = _payload()
    models = payload["models"]
    assert isinstance(models, dict)
    profiles = models["profiles"]
    assert isinstance(profiles, list)
    profile = profiles[0]
    assert isinstance(profile, dict)
    profile["checkpoint_path"] = str(tmp_path / "other.bin")
    path = tmp_path / "model.yaml"
    _write(path, payload)

    with pytest.raises(ContractError, match="checkpoint_path must be fixed"):
        load_appearance_config(path)


def test_appearance_config_rejects_any_additional_encoder_profile(
    tmp_path: Path,
) -> None:
    payload = _payload()
    models = payload["models"]
    assert isinstance(models, dict)
    profiles = models["profiles"]
    assert isinstance(profiles, list)
    profiles.append(dict(profiles[0]))
    path = tmp_path / "multiple-models.yaml"
    _write(path, payload)

    with pytest.raises(ContractError, match="only the locked MegaDescriptor profile"):
        load_appearance_config(path)


def test_appearance_config_rejects_nonfinite_quality_value(tmp_path: Path) -> None:
    payload = _payload()
    quality = payload["quality"]
    assert isinstance(quality, dict)
    quality["blur_scale"] = float("nan")
    path = tmp_path / "nan.yaml"
    _write(path, payload)

    with pytest.raises(ContractError, match="finite number"):
        load_appearance_config(path)


def test_appearance_config_rejects_bad_quality_weight_sum(tmp_path: Path) -> None:
    payload = _payload()
    quality = payload["quality"]
    assert isinstance(quality, dict)
    weights = quality["metric_weights"]
    assert isinstance(weights, dict)
    weights["blur_score"] = 0.10
    path = tmp_path / "weights.yaml"
    _write(path, payload)

    with pytest.raises(ContractError, match="quality metric weights must be"):
        load_appearance_config(path)


@pytest.mark.parametrize("bad_value", [-1.01, 1.01])
def test_appearance_config_rejects_out_of_range_support_cosine(
    tmp_path: Path, bad_value: float
) -> None:
    payload = _payload()
    prototypes = payload["prototypes"]
    assert isinstance(prototypes, dict)
    prototypes["outlier_support_cosine"] = bad_value
    path = tmp_path / "support.yaml"
    _write(path, payload)

    with pytest.raises(ContractError, match="outlier_support_cosine=0.70"):
        load_appearance_config(path)
