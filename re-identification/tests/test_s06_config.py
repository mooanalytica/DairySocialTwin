from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from cowtrack.config import ContractError
from cowtrack.linking.dataset_contract import (
    EXPECTED_CLIP_ORDER,
    EXPECTED_FRAME_COUNT,
    EXPECTED_FRAME_COUNTS,
)
from cowtrack.qa.s06_config import load_s06_export_config


CONFIG = Path(__file__).parents[1] / "configs" / "s06_export.yaml"


def _payload() -> dict[str, object]:
    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _write(tmp_path: Path, payload: object) -> Path:
    path = tmp_path / "s06_export.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def test_repository_s06_config_is_fixed_forced_provisional_export() -> None:
    config, payload, digest = load_s06_export_config(CONFIG)

    assert config.required_upstream_stage == "S05_FORCE_APPEARANCE"
    assert config.identity_source == "05_forced_appearance_only"
    assert config.authorization_basis == "operator_forced_appearance_exact_62"
    assert config.id_status == "forced_provisional"
    assert config.certification_claimed is False
    assert config.probability_from_cosine_allowed is False
    assert config.clip_order == EXPECTED_CLIP_ORDER
    assert config.expected_frame_count == EXPECTED_FRAME_COUNT
    assert config.frame_counts_by_clip == EXPECTED_FRAME_COUNTS
    assert config.expected_detection_count == 3_322_909
    assert config.expected_valid_detection_count == 3_318_113
    assert config.expected_invalid_detection_count == 4_796
    assert len(config.expected_total_detections_by_clip) == 11
    assert sum(config.expected_total_detections_by_clip) == 3_322_909
    assert sum(config.expected_valid_detections_by_clip) == 3_318_113
    assert sum(config.expected_invalid_detections_by_clip) == 4_796
    assert config.expected_microtrack_count is None
    assert config.expected_stable_track_count is None
    assert config.expected_selected_link_count is None
    assert config.expected_global_track_count == 62
    assert config.expected_candidate_edge_count is None
    assert config.expected_rescue_candidate_count is None
    assert config.expected_rescue_embedding_count is None
    assert config.expected_grade_a_clean_count is None
    assert config.expected_grade_b_existing_degraded_count is None
    assert config.expected_grade_c_reencoded_degraded_count is None
    assert config.num_confirmed_ids == 0
    assert config.num_provisional_ids == 62
    assert config.population_soft_max == 57
    assert config.population_overflow == 5
    assert config.reference_mode == "observed_only"
    assert config.low_confidence_thresholds == (0.3, 0.4, 0.5, 0.6)
    assert "expected_low_confidence_counts" not in payload["qa"]
    assert not any(
        name.startswith("expected_selected_cosine_") for name in payload["qa"]
    )
    assert "expected_c_grade_selected_high_overlap_crop_count" not in payload["qa"]
    assert "expected_c_grade_high_overlap_stable_ids" not in payload["qa"]
    assert "provenance_warning_required" not in payload["qa"]
    assert (config.output_width, config.output_height) == (1920, 1080)
    assert (config.fps_numerator, config.fps_denominator) == (30000, 1001)
    assert config.bbox_scale == 0.5
    assert config.codec == "h264_nvenc"
    assert config.required_cuda_visible_devices == "1"
    assert config.logical_gpu == 0
    assert (config.preset, config.cq, config.pixel_format) == ("p4", 21, "yuv420p")
    assert config.artifacts.videos_by_clip == tuple(
        f"qa/videos/{clip}_tracked.mp4" for clip in EXPECTED_CLIP_ORDER
    )
    assert payload["runtime"]["row_sort_keys"] == ["clip_order", "csv_row_index"]
    assert len(digest) == 64


@pytest.mark.parametrize(
    ("section", "name", "value"),
    [
        ("pipeline", "required_upstream_stage", "S05"),
        ("pipeline", "identity_source", "05_global_link"),
        ("pipeline", "id_status", "confirmed"),
        ("pipeline", "certification_claimed", True),
        ("pipeline", "probability_from_cosine_allowed", True),
        ("qa", "reference_mode", "historical_snapshot"),
        ("inputs", "expected_total_detection_count", 3_322_908),
        ("inputs", "expected_global_track_count", 57),
        ("identity", "population_overflow", 0),
        ("render", "fps_denominator", 1000),
        ("render", "autorotate", True),
        ("encoder", "codec", "libx264"),
        ("encoder", "logical_gpu", 1),
        ("encoder", "preset", "p5"),
        ("encoder", "cq", 20),
    ],
)
def test_s06_config_rejects_fixed_policy_or_snapshot_drift(
    tmp_path: Path,
    section: str,
    name: str,
    value: object,
) -> None:
    payload = _payload()
    payload[section][name] = value  # type: ignore[index]
    with pytest.raises(ContractError, match="fixed S06 export"):
        load_s06_export_config(_write(tmp_path, payload))


def test_s06_config_rejects_extra_and_missing_keys(tmp_path: Path) -> None:
    payload = _payload()
    payload["render"]["fallback_encoder"] = "opencv"  # type: ignore[index]
    with pytest.raises(ContractError, match="keys mismatch"):
        load_s06_export_config(_write(tmp_path, payload))

    payload = _payload()
    del payload["qa"]["reference_mode"]  # type: ignore[index]
    with pytest.raises(ContractError, match="keys mismatch"):
        load_s06_export_config(_write(tmp_path, payload))

    payload = _payload()
    del payload["qa"]["margin_semantics"]  # type: ignore[index]
    with pytest.raises(ContractError, match="keys mismatch"):
        load_s06_export_config(_write(tmp_path, payload))


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("expected_low_confidence_counts", {"below_0_5": 133}),
        ("expected_selected_cosine_mean", 0.8604075520997553),
        ("expected_c_grade_selected_high_overlap_crop_count", 25),
        ("expected_c_grade_high_overlap_stable_ids", [2720, 2780]),
        ("provenance_warning_required", True),
    ],
)
def test_s06_config_rejects_historical_qa_reference_fields(
    tmp_path: Path, name: str, value: object
) -> None:
    payload = _payload()
    payload["qa"][name] = value  # type: ignore[index]
    with pytest.raises(ContractError, match="keys mismatch"):
        load_s06_export_config(_write(tmp_path, payload))


def test_s06_config_rejects_count_and_artifact_substitution(tmp_path: Path) -> None:
    payload = _payload()
    payload["inputs"]["invalid_detections_by_clip"]["GX050006"] = 304  # type: ignore[index]
    with pytest.raises(ContractError):
        load_s06_export_config(_write(tmp_path, payload))

    payload = _payload()
    payload["artifacts"]["detections_csv"] = "../detections.csv"  # type: ignore[index]
    with pytest.raises(ContractError):
        load_s06_export_config(_write(tmp_path, payload))


def test_s06_config_rejects_bool_in_integer_field(tmp_path: Path) -> None:
    payload = _payload()
    payload["inputs"]["expected_global_track_count"] = True  # type: ignore[index]
    with pytest.raises(ContractError):
        load_s06_export_config(_write(tmp_path, payload))


def test_s06_config_rejects_bool_in_optional_upstream_count(tmp_path: Path) -> None:
    payload = _payload()
    payload["inputs"]["expected_candidate_edge_count"] = True  # type: ignore[index]
    with pytest.raises(ContractError, match="integer"):
        load_s06_export_config(_write(tmp_path, payload))
