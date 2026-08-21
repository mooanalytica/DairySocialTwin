from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cowtrack.config import ContractError
from cowtrack.linking.features import LONG_FEATURE_SCHEMA
from cowtrack.linking.long_calibration_config import load_long_calibration_config
from cowtrack.linking.runtime import fingerprint_file
from cowtrack.linking.s05_runtime import (
    S05SelectedGateEvidence,
    load_s05_long_calibration,
)
from cowtrack.schemas.s05 import LONG_CALIBRATION_PAIRS_SCHEMA
import cowtrack.stages.s05_calibrate_long as stage


ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "configs" / "s05_long_calibration.yaml"
EXPECTED_S03_CONFIG_HASH = load_long_calibration_config(CONFIG)[0].expected_s03_config_hash
PARTITION_SIZES = {
    "train": 100,
    "threshold_selection": 25,
    "certification": 25,
    # Production S05A requires the nominal per-clip evidence budget in the
    # aggregate: 10 groups * 11 configured clips.
    "audit": 110,
}
PARTITION_OFFSETS = {
    "train": 0,
    "threshold_selection": 200,
    "certification": 400,
    "audit": 600,
}


def _gallery(row: dict[str, Any], prefix: str, token: int) -> None:
    row.update(
        {
            f"{prefix}_sample_ids": [token],
            f"{prefix}_gallery_det_ids": [-token - 1],
            f"{prefix}_embedding_rows": [token],
            f"{prefix}_medoid_sample_id": token,
            f"{prefix}_appearance_quality": 0.9,
            f"{prefix}_internal_cosine_p10": 0.8,
            f"{prefix}_internal_cosine_p50": 0.9,
            f"{prefix}_internal_cosine_min": 0.75,
            f"{prefix}_gallery_num_input_samples": 3,
            f"{prefix}_gallery_num_overlap_rejected": 0,
            f"{prefix}_gallery_num_review_excluded": 0,
            f"{prefix}_gallery_num_local_outliers": 0,
            f"{prefix}_gallery_max_other_bbox_iou": 0.1,
            f"{prefix}_gallery_max_clean_other_bbox_iou": 0.1,
        }
    )


def _segment(
    row: dict[str, Any],
    prefix: str,
    *,
    stable_id: int,
    det_id: int,
    frame: int,
    time_sec: float,
    clip_id: str,
) -> None:
    row.update(
        {
            f"{prefix}_stable_id": stable_id,
            f"{prefix}_constituent_micro_ids": [stable_id],
            f"{prefix}_start_det_id": det_id,
            f"{prefix}_end_det_id": det_id,
            f"{prefix}_start_global_frame": frame,
            f"{prefix}_end_global_frame": frame,
            f"{prefix}_start_time_sec": time_sec,
            f"{prefix}_end_time_sec": time_sec,
            f"{prefix}_start_clip_id": clip_id,
            f"{prefix}_end_clip_id": clip_id,
            f"{prefix}_num_detections": 1,
        }
    )


def _pair_row(
    partition: str,
    parent: int,
    local_index: int,
    *,
    label: bool,
) -> dict[str, Any]:
    target = parent if label else parent + 1_000
    clip_id = "GX040006" if local_index % 2 == 0 else "GX050006"
    score = 0.9 if label else 0.1
    margin = 0.8 if label else -0.8
    source_time = float(parent * 10)
    target_time = source_time + 6.0
    source_frame = parent * 300
    target_frame = source_frame + 180
    source_det = -(parent * 100 + (1 if label else 11))
    target_det = source_det - 1
    features = {name: 0.1 for name in LONG_FEATURE_SCHEMA}
    features.update(
        {
            "prototype_cosine_max": score,
            "prototype_cosine_top3_mean": score,
            "medoid_cosine": score,
            "mutual_prototype_score": score,
            "appearance_quality_min": 0.9,
            "appearance_quality_mean": 0.9,
            "gap_sec": 6.0,
            "log1p_gap_sec": float(np.log1p(6.0)),
            "log_width_ratio": 0.0,
            "log_height_ratio": 0.0,
            "log_area_ratio": 0.0,
            "src_end_max_other_iou": 0.1,
            "dst_start_max_other_iou": 0.1,
            "src_end_boundary_distance": 0.5,
            "dst_start_boundary_distance": 0.5,
            "is_clip_boundary": 0.0,
        }
    )
    candidate_group = (
        f"partition-{partition}:parent-stable-{parent}"
        if label
        else f"partition-{partition}:stable-pair-{parent}-{target}"
    )
    row: dict[str, Any] = {
        "pair_id": f"{candidate_group}:{'positive' if label else 'negative'}",
        "candidate_group_id": candidate_group,
        "parent_group_id": f"stable-{parent}",
        "partition": partition,
        "split": (
            "train"
            if partition == "train"
            else "audit"
            if partition == "audit"
            else "calibration"
        ),
        "calibration_role": (
            partition
            if partition in {"threshold_selection", "certification"}
            else "not_applicable"
        ),
        "mode": "long",
        "label": label,
        "pair_kind": (
            "stable_path_pseudo_positive"
            if label
            else "simultaneous_stable_hard_negative"
        ),
        "parent_stable_id": parent,
        "source_clip_id": clip_id,
        "target_clip_id": clip_id,
        "hard_negative_rank": 0 if label else 1,
        "appearance_present": True,
        "high_overlap": False,
        "cooccurrence_clip_id": None if label else clip_id,
        "cooccurrence_global_frame": None if label else target_frame,
        "cooccurrence_parent_det_id": None if label else target_det - 1,
        "cooccurrence_other_det_id": None if label else target_det,
        "features": features,
        "candidate_margin": margin,
    }
    _segment(
        row,
        "source",
        stable_id=parent,
        det_id=source_det,
        frame=source_frame,
        time_sec=source_time,
        clip_id=clip_id,
    )
    _segment(
        row,
        "target",
        stable_id=target,
        det_id=target_det,
        frame=target_frame,
        time_sec=target_time,
        clip_id=clip_id,
    )
    _gallery(row, "source", parent * 10 + (1 if label else 3))
    _gallery(row, "target", parent * 10 + (2 if label else 4))
    row.update(features)
    return row


def _rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for partition, count in PARTITION_SIZES.items():
        offset = PARTITION_OFFSETS[partition]
        for local_index in range(count):
            parent = offset + local_index
            rows.extend(
                (
                    _pair_row(partition, parent, local_index, label=True),
                    _pair_row(partition, parent, local_index, label=False),
                )
            )
    return rows


def _disabled_rows() -> list[dict[str, Any]]:
    return [
        _pair_row(partition, offset, 0, label=label)
        for partition, offset in PARTITION_OFFSETS.items()
        for label in (True, False)
    ]


def _output_fingerprint(path: Path, directory: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    return {
        "path": str(path.relative_to(directory)),
        "size_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _refresh_output_fingerprint(directory: Path, name: str) -> None:
    marker_path = directory / "_SUCCESS.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    replacement = _output_fingerprint(directory / name, directory)
    marker["output_fingerprints"] = [
        replacement if item["path"] == name else item
        for item in marker["output_fingerprints"]
    ]
    marker_path.write_text(json.dumps(marker), encoding="utf-8")


def _runtime_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    synthetic_rows: list[dict[str, Any]] | None = None,
) -> tuple[Path, Path]:
    directories = {
        name: tmp_path / name
        for name in (
            "00_ingest",
            "01_microtrack",
            "02_appearance",
            "03_calibration",
            "04_short_stable",
        )
    }
    for directory in directories.values():
        directory.mkdir()
    immutable = directories["00_ingest"] / "immutable.bin"
    immutable.write_bytes(b"immutable-upstream")
    s03_artifact = directories["03_calibration"] / "runtime.bin"
    s03_artifact.write_bytes(b"s03-runtime")
    s04_artifact = directories["04_short_stable"] / "stable.bin"
    s04_artifact.write_bytes(b"s04-stable")

    production = SimpleNamespace(
        input_fingerprints=(fingerprint_file(immutable),),
    )
    stable = SimpleNamespace(
        micro_to_stable={0: 0},
        micro_order_in_stable={0: 0},
        micro_ids=np.arange(4_659, dtype=np.int64),
        stable_ids=np.arange(3_769, dtype=np.int64),
        stable_appearance={
            stable_id: SimpleNamespace(appearance_usable=stable_id < 1_416)
            for stable_id in range(3_769)
        },
        success_marker={
            "input_fingerprints": [fingerprint_file(immutable).as_dict()]
        },
        input_fingerprints=(fingerprint_file(s04_artifact),),
    )
    legacy = {
        "config_hash": EXPECTED_S03_CONFIG_HASH,
        "model_enabled": False,
        "confirmed_enabled": False,
        "disabled_reason": "legacy_long_model_disabled",
        "used_for_scoring_or_thresholds": False,
    }
    monkeypatch.setattr(
        stage, "load_production_inputs", lambda *args, **kwargs: production
    )
    monkeypatch.setattr(
        stage,
        "_load_s03_provenance",
        lambda *args, **kwargs: ([fingerprint_file(s03_artifact)], legacy),
    )
    monkeypatch.setattr(stage, "load_s04_finalized", lambda *args, **kwargs: stable)
    monkeypatch.setattr(
        stage, "_validate_s04_against_runtime", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        stage, "generate_stable_long_pairs", lambda *args, **kwargs: (object(),)
    )
    rows = _rows() if synthetic_rows is None else synthetic_rows
    monkeypatch.setattr(
        stage, "stable_long_pairs_as_rows", lambda pairs: rows
    )

    output = tmp_path / "05_long_calibration"
    stage.run_s05_calibrate_long(
        directories["00_ingest"],
        directories["01_microtrack"],
        directories["02_appearance"],
        directories["03_calibration"],
        directories["04_short_stable"],
        CONFIG,
        output,
        logger=lambda message: None,
    )
    return output, immutable


def test_loader_exposes_only_published_runtime_gates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output, _ = _runtime_fixture(tmp_path, monkeypatch)

    bundle = load_s05_long_calibration(output)

    assert bundle.directory == output.resolve()
    assert len(bundle.config_hash) == 64
    assert bundle.model_enabled is True
    assert bundle.confirmed_enabled is False
    assert bundle.thresholds is bundle.runtime_thresholds
    assert bundle.thresholds.provisional_threshold is not None
    assert bundle.thresholds.confirmed_threshold is None
    assert bundle.thresholds.appearance_margin_threshold is None
    assert not hasattr(bundle.thresholds, "selected_confirmed_threshold")
    assert bundle.selected_gate_evidence.probability_threshold is not None
    assert bundle.selected_gate_evidence.margin_threshold is not None
    assert bundle.selected_gate_evidence.certified is False
    assert bundle.selected_gate_evidence.executable is False
    adequacy = bundle.report["audit_adequacy"]
    assert adequacy["evidence_scope"] == "aggregate_observed_rows"
    assert adequacy["expected_clip_count"] == 11
    assert adequacy["required_aggregate_groups"] == {
        "positive": 110,
        "hard_negative": 110,
    }
    assert adequacy["sufficient_for_selected_scope"] is True
    assert adequacy["strict_per_clip_sufficient"] is False
    assert len(adequacy["warnings"]) == 9
    assert len(bundle.output_fingerprints) == 9
    assert set(bundle.pair_paths) == set(PARTITION_SIZES)

    selection = pq.read_table(
        bundle.pair_paths["threshold_selection"]
    ).to_pylist()
    positive = next(row for row in selection if row["label"])
    features = {name: positive[name] for name in LONG_FEATURE_SCHEMA}
    result = bundle.scorer.score_features(
        features,
        appearance_present=True,
        high_overlap=False,
        candidate_margin=100.0,
    )
    assert result.decision == "provisional"


def test_selected_gate_evidence_cannot_be_made_executable() -> None:
    with pytest.raises(ContractError, match="report-only"):
        S05SelectedGateEvidence(
            probability_threshold=0.5,
            margin_threshold=0.1,
            certified=False,
            false_accepts=0,
            false_accept_upper=0.1,
            certification_true_accepts=1,
            certification_hard_negative_count=10,
            executable=True,
        )


def test_loader_preserves_valid_disabled_model_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output, _ = _runtime_fixture(
        tmp_path, monkeypatch, synthetic_rows=_disabled_rows()
    )

    bundle = load_s05_long_calibration(output)

    assert bundle.model_enabled is False
    assert bundle.confirmed_enabled is False
    assert bundle.thresholds.provisional_threshold is None
    assert bundle.selected_gate_evidence.probability_threshold is None
    assert bundle.selected_gate_evidence.margin_threshold is None
    assert bundle.selected_gate_evidence.executable is False
    result = bundle.scorer.score_features(
        None,
        appearance_present=True,
        high_overlap=False,
        candidate_margin=100.0,
    )
    assert result.decision == "reject"
    assert result.reason and "below_minimum" in result.reason


def test_loader_rejects_changed_output_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output, _ = _runtime_fixture(tmp_path, monkeypatch)
    thresholds = output / "thresholds.json"
    thresholds.write_bytes(thresholds.read_bytes() + b"\n")

    with pytest.raises(ContractError, match="artifact changed"):
        load_s05_long_calibration(output)


def test_loader_rejects_internal_gate_promoted_with_refreshed_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output, _ = _runtime_fixture(tmp_path, monkeypatch)
    path = output / "thresholds.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    selected = payload["long"]["selected_confirmed_threshold"]
    payload["long"]["confirmed_threshold"] = selected
    payload["long_confirmed_threshold"] = selected
    path.write_text(json.dumps(payload), encoding="utf-8")
    _refresh_output_fingerprint(output, "thresholds.json")

    with pytest.raises(ContractError, match="disabled confirmation"):
        load_s05_long_calibration(output)


def test_loader_rejects_pair_schema_change_with_refreshed_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output, _ = _runtime_fixture(tmp_path, monkeypatch)
    name = "long_pairs_audit.parquet"
    path = output / name
    table = pq.read_table(path).drop(["decision"])
    assert not table.schema.equals(LONG_CALIBRATION_PAIRS_SCHEMA)
    pq.write_table(table, path)
    _refresh_output_fingerprint(output, name)

    with pytest.raises(ContractError, match="pair schema differs"):
        load_s05_long_calibration(output)


def test_loader_rejects_report_policy_change_with_refreshed_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output, _ = _runtime_fixture(tmp_path, monkeypatch)
    name = "long_calibration_report.json"
    path = output / name
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["global_merge_allowed"] = True
    path.write_text(json.dumps(payload), encoding="utf-8")
    _refresh_output_fingerprint(output, name)

    with pytest.raises(ContractError, match="report policy"):
        load_s05_long_calibration(output)


def test_loader_rejects_changed_recorded_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output, immutable = _runtime_fixture(tmp_path, monkeypatch)
    immutable.write_bytes(b"changed-upstream")

    with pytest.raises(ContractError, match="recorded S05A input changed"):
        load_s05_long_calibration(output)


def test_loader_requires_all_nine_ordered_output_fingerprints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output, _ = _runtime_fixture(tmp_path, monkeypatch)
    marker_path = output / "_SUCCESS.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["output_fingerprints"][0], marker["output_fingerprints"][1] = (
        marker["output_fingerprints"][1],
        marker["output_fingerprints"][0],
    )
    marker_path.write_text(json.dumps(marker), encoding="utf-8")

    with pytest.raises(ContractError, match="set/order"):
        load_s05_long_calibration(output)
