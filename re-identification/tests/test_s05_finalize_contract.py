from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pytest
import yaml

import cowtrack.linking.s05_finalize_config as finalize_config_module
from cowtrack.config import ContractError
from cowtrack.linking.dataset_contract import (
    EXPECTED_CLIP_ORDER,
    EXPECTED_FRAME_COUNTS,
)
from cowtrack.linking.s05_finalize_config import load_s05_finalize_config
from cowtrack.schemas.s05_finalize import (
    DET_TO_GLOBAL_SCHEMA,
    GLOBAL_CANDIDATE_EDGES_SCHEMA,
    GLOBAL_TRACKS_SCHEMA,
    STABLE_TO_GLOBAL_SCHEMA,
)


CONFIG = Path("configs/s05_finalize.yaml")


def _payload() -> dict[str, object]:
    result = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert isinstance(result, dict)
    return result


def _write(tmp_path: Path, payload: object) -> Path:
    path = tmp_path / "s05_finalize.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def test_repository_s05_finalize_config_is_fixed_operator_approval() -> None:
    config, payload, config_hash = load_s05_finalize_config(CONFIG)

    assert payload["pipeline"]["operator_approved"] is True
    assert config.execution_mode == "operator_approved_global_path_cover"
    assert config.operator_approved is True
    assert config.global_merge_allowed is True
    assert config.solver_allowed is True
    assert config.path_cover_allowed is True
    assert config.certification_claim_allowed is False
    assert config.approval_strategy == "all_provisional_proposals"
    assert config.authorization_basis == "operator_blanket_strategy"
    assert config.review_output_consumed is False
    assert config.review_labels_consumed is False
    assert config.require_proposal_confirmed_false is True
    assert config.preserve_proposal_evidence_status is True
    assert config.objective == "maximum_cardinality_then_evidence_cost"
    assert config.evidence_cost == "deterministic_evidence_ordinal"
    assert config.candidate_policy == "all_provisional_proposals"
    assert config.population_policy == "warning_only"
    assert config.population_warning_threshold == 62
    assert config.expected_python_version == "3.14.4"
    assert config.expected_scipy_version == "1.18.0"
    assert config.population_affects_solver is False
    assert config.force_merge_allowed is False
    assert config.threshold_adaptation_allowed is False
    assert len(config_hash) == 64


@pytest.mark.parametrize(
    ("runtime_name", "runtime_value"),
    [("python", "3.14.3"), ("scipy", "1.17.0")],
)
def test_s05_finalize_config_rejects_solver_runtime_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    runtime_name: str,
    runtime_value: str,
) -> None:
    if runtime_name == "python":
        monkeypatch.setattr(
            finalize_config_module.platform, "python_version", lambda: runtime_value
        )
    else:
        monkeypatch.setattr(finalize_config_module.scipy, "__version__", runtime_value)

    with pytest.raises(
        ContractError,
        match=r"requires fixed solver runtime Python 3\.14\.4 / SciPy 1\.18\.0",
    ):
        load_s05_finalize_config(CONFIG)


def test_s05_finalize_config_locks_full_sequence_and_uses_runtime_counts() -> None:
    config, _, _ = load_s05_finalize_config(CONFIG)

    assert config.expected_sequence_id == "dairy_farm_1_gopro1_20250505"
    assert config.clip_order == EXPECTED_CLIP_ORDER
    assert config.frame_counts_by_clip == EXPECTED_FRAME_COUNTS
    assert config.expected_valid_detections_by_clip == (
        745052, 103319, 248019, 332754, 413161, 264397,
        154326, 196054, 215158, 386669, 259204,
    )
    assert config.expected_invalid_detections_by_clip == (
        18, 26, 753, 59, 305, 1160, 135, 134, 1417, 57, 732,
    )
    assert config.expected_frame_count == 826492
    assert config.expected_stable_track_count is None
    assert config.expected_microtrack_count is None
    assert config.expected_detection_count == 3318113
    assert config.expected_invalid_detections == 4796
    assert config.expected_candidate_count is None
    assert config.expected_proposal_count is None


def test_s05_finalize_schemas_are_unique_and_auditable() -> None:
    for schema in (
        GLOBAL_CANDIDATE_EDGES_SCHEMA,
        STABLE_TO_GLOBAL_SCHEMA,
        GLOBAL_TRACKS_SCHEMA,
        DET_TO_GLOBAL_SCHEMA,
    ):
        assert len(schema.names) == len(set(schema.names))

    candidate_required = {
        "candidate_id",
        "proposal_id",
        "source_stable_id",
        "target_stable_id",
        "source_end_clip_id",
        "target_start_clip_id",
        "temporal_gap_sec",
        "model_probability",
        "candidate_margin",
        "proposal_evidence_status",
        "proposal_review_status",
        "proposal_selected_by_solver",
        "proposal_confirmed",
        "proposal_merge_applied",
        "operator_approved",
        "approval_strategy",
        "authorization_basis",
        "eligible_for_solver",
        "selected_by_solver",
        "solver_cost_int",
        "global_link_id",
        "final_decision",
    }
    assert candidate_required <= set(GLOBAL_CANDIDATE_EDGES_SCHEMA.names)
    assert GLOBAL_CANDIDATE_EDGES_SCHEMA.field("proposal_id").nullable
    assert GLOBAL_CANDIDATE_EDGES_SCHEMA.field("model_probability").nullable
    assert not GLOBAL_CANDIDATE_EDGES_SCHEMA.field("proposal_confirmed").nullable
    assert not GLOBAL_CANDIDATE_EDGES_SCHEMA.field("operator_approved").nullable

    assert STABLE_TO_GLOBAL_SCHEMA.field("predecessor_stable_id").nullable
    assert STABLE_TO_GLOBAL_SCHEMA.field("predecessor_candidate_id").nullable
    assert not STABLE_TO_GLOBAL_SCHEMA.field("global_track_uuid").nullable
    assert GLOBAL_TRACKS_SCHEMA.field("clip_ids").type == pa.list_(
        pa.field("element", pa.string())
    )


def test_det_to_global_carries_full_video_time_and_mapping_contract() -> None:
    expected = {
        "det_id": pa.int64(),
        "sequence_id": pa.string(),
        "clip_id": pa.string(),
        "clip_order": pa.int16(),
        "local_frame": pa.int32(),
        "global_frame": pa.int64(),
        "global_time_sec": pa.float64(),
        "valid": pa.bool_(),
        "micro_id": pa.int64(),
        "stable_id": pa.int64(),
        "global_track_id": pa.int64(),
        "global_track_uuid": pa.string(),
        "display_global_id": pa.string(),
        "order_in_micro": pa.int32(),
        "order_in_stable": pa.int32(),
        "order_in_stable_detection": pa.int64(),
        "order_in_global_stable": pa.int32(),
        "order_in_global_detection": pa.int64(),
        "identity_basis": pa.string(),
        "id_status": pa.string(),
    }
    assert DET_TO_GLOBAL_SCHEMA.names == list(expected)
    for name, data_type in expected.items():
        field = DET_TO_GLOBAL_SCHEMA.field(name)
        assert field.type == data_type
        assert field.nullable is False


@pytest.mark.parametrize(
    ("section", "name", "value"),
    [
        ("pipeline", "operator_approved", False),
        ("pipeline", "certification_claim_allowed", True),
        ("approval", "strategy", "confirmed_only"),
        ("approval", "review_output_consumed", True),
        ("approval", "require_proposal_confirmed_false", False),
        ("approval", "preserve_proposal_evidence_status", False),
        ("solver", "objective", "minimum_cost_only"),
        ("solver", "candidate_policy", "selected_gate_only"),
        ("population", "policy", "force_target"),
        ("population", "affects_solver", True),
        ("population", "threshold_adaptation_allowed", True),
    ],
)
def test_s05_finalize_config_rejects_strategy_or_evidence_substitution(
    tmp_path: Path, section: str, name: str, value: object
) -> None:
    payload = _payload()
    payload[section][name] = value
    with pytest.raises(ContractError):
        load_s05_finalize_config(_write(tmp_path, payload))


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("expected_detection_count", 3318112),
        ("clip_order", list(reversed(EXPECTED_CLIP_ORDER))),
    ],
)
def test_s05_finalize_config_rejects_snapshot_substitution(
    tmp_path: Path, name: str, value: object
) -> None:
    payload = _payload()
    payload["inputs"][name] = value
    with pytest.raises(ContractError):
        load_s05_finalize_config(_write(tmp_path, payload))


def test_s05_finalize_config_accepts_optional_runtime_counts_and_validates_them(
    tmp_path: Path,
) -> None:
    payload = _payload()
    payload["inputs"]["expected_stable_track_count"] = 100
    payload["inputs"]["expected_microtrack_count"] = 120
    payload["inputs"]["expected_candidate_count"] = 200
    payload["inputs"]["expected_proposal_count"] = 150
    config, _, _ = load_s05_finalize_config(_write(tmp_path, payload))
    assert config.expected_stable_track_count == 100
    assert config.expected_microtrack_count == 120
    assert config.expected_candidate_count == 200
    assert config.expected_proposal_count == 150

    for name, value in (
        ("expected_stable_track_count", True),
        ("expected_microtrack_count", -1),
        ("expected_candidate_count", -1),
        ("expected_proposal_count", -1),
    ):
        invalid = _payload()
        invalid["inputs"][name] = value
        with pytest.raises(ContractError):
            load_s05_finalize_config(_write(tmp_path, invalid))


def test_s05_finalize_config_rejects_clip_artifact_unknown_and_integer_bool(
    tmp_path: Path,
) -> None:
    payload = _payload()
    payload["inputs"]["expected_valid_detections_by_clip"]["GX050006"] = 413160
    with pytest.raises(ContractError):
        load_s05_finalize_config(_write(tmp_path, payload))

    payload = _payload()
    payload["artifacts"]["global_tracks"] = "unsafe.parquet"
    with pytest.raises(ContractError):
        load_s05_finalize_config(_write(tmp_path, payload))

    payload = _payload()
    payload["unexpected"] = {}
    with pytest.raises(ContractError):
        load_s05_finalize_config(_write(tmp_path, payload))

    payload = _payload()
    payload["pipeline"]["operator_approved"] = 1
    with pytest.raises(ContractError):
        load_s05_finalize_config(_write(tmp_path, payload))
