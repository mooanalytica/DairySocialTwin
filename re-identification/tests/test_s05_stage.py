from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cowtrack.config import ContractError
from cowtrack.linking.calibration_core import CalibrationMinimums
from cowtrack.linking.features import LONG_FEATURE_SCHEMA
from cowtrack.linking.model import load_link_model
from cowtrack.linking.runtime import fingerprint_file
import cowtrack.stages.s05_calibrate_long as stage


ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "configs" / "s05_long_calibration.yaml"
PARTITIONS = ("train", "threshold_selection", "certification", "audit")
PARQUETS = {
    "train": "long_pairs_train.parquet",
    "threshold_selection": "long_pairs_calibration_selection.parquet",
    "certification": "long_pairs_calibration_certification.parquet",
    "audit": "long_pairs_audit.parquet",
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
            f"{prefix}_start_clip_id": "GX040006",
            f"{prefix}_end_clip_id": "GX040006",
            f"{prefix}_num_detections": 1,
        }
    )


def _pair_row(partition: str, partition_index: int, *, label: bool) -> dict[str, Any]:
    parent = partition_index
    target = parent if label else parent + 100
    kind = "positive" if label else "negative"
    source_time = float(partition_index * 100)
    target_time = source_time + 6.0
    score = 0.9 if label else 0.1
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
            "is_clip_boundary": 0.0,
        }
    )
    det_base = -(partition_index * 10 + (1 if label else 5) + 100)
    row: dict[str, Any] = {
        "pair_id": f"{partition}-{kind}",
        "candidate_group_id": f"{partition}-{kind}-group",
        "parent_group_id": f"stable-{parent}",
        "partition": partition,
        "split": "train"
        if partition == "train"
        else "audit"
        if partition == "audit"
        else "calibration",
        "calibration_role": partition
        if partition in {"threshold_selection", "certification"}
        else "not_applicable",
        "mode": "long",
        "label": label,
        "pair_kind": "stable_path_pseudo_positive"
        if label
        else "simultaneous_stable_hard_negative",
        "parent_stable_id": parent,
        "source_clip_id": "GX040006",
        "target_clip_id": "GX040006",
        "hard_negative_rank": 0 if label else 1,
        "appearance_present": True,
        "high_overlap": False,
        "cooccurrence_clip_id": None if label else "GX040006",
        "cooccurrence_global_frame": (
            None if label else partition_index * 100 + 6
        ),
        "cooccurrence_parent_det_id": None if label else det_base - 2,
        "cooccurrence_other_det_id": None if label else det_base - 1,
        "features": features,
        "candidate_margin": score - (0.1 if label else 0.9),
    }
    _segment(
        row,
        "source",
        stable_id=parent,
        det_id=det_base,
        frame=partition_index * 100,
        time_sec=source_time,
    )
    _segment(
        row,
        "target",
        stable_id=target,
        det_id=det_base - 1,
        frame=partition_index * 100 + 6,
        time_sec=target_time,
    )
    _gallery(row, "source", partition_index * 20 + (1 if label else 3))
    _gallery(row, "target", partition_index * 20 + (2 if label else 4))
    row.update(features)
    return row


def _rows() -> list[dict[str, Any]]:
    return [
        _pair_row(partition, partition_index, label=label)
        for partition_index, partition in enumerate(PARTITIONS)
        for label in (True, False)
    ]


def _update_output_fingerprint(output: Path, name: str) -> None:
    marker_path = output / "_SUCCESS.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    payload = (output / name).read_bytes()
    replacement = {
        "path": name,
        "size_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    marker["output_fingerprints"] = [
        replacement if item["path"] == name else item
        for item in marker["output_fingerprints"]
    ]
    marker_path.write_text(json.dumps(marker), encoding="utf-8")


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    directories = {
        name: tmp_path / name
        for name in ("00_ingest", "01_microtrack", "02_appearance", "03_calibration", "04_short_stable")
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
        micro_ids=np.arange(8, dtype=np.int64),
        stable_ids=np.arange(8, dtype=np.int64),
        stable_appearance={
            stable_id: SimpleNamespace(appearance_usable=False)
            for stable_id in range(8)
        },
        success_marker={
            "input_fingerprints": [fingerprint_file(immutable).as_dict()]
        },
        input_fingerprints=(fingerprint_file(s04_artifact),),
    )
    legacy = {
        "config_hash": "e" * 64,
        "model_enabled": False,
        "confirmed_enabled": False,
        "disabled_reason": "legacy_disabled",
        "used_for_scoring_or_thresholds": False,
    }
    monkeypatch.setattr(stage, "load_production_inputs", lambda *args, **kwargs: production)
    monkeypatch.setattr(
        stage,
        "_load_s03_provenance",
        lambda *args, **kwargs: ([fingerprint_file(s03_artifact)], legacy),
    )
    monkeypatch.setattr(stage, "load_s04_finalized", lambda *args, **kwargs: stable)
    monkeypatch.setattr(stage, "_validate_s04_against_runtime", lambda *args, **kwargs: None)
    monkeypatch.setattr(stage, "generate_stable_long_pairs", lambda *args, **kwargs: (object(),))
    monkeypatch.setattr(stage, "stable_long_pairs_as_rows", lambda pairs: _rows())

    output = tmp_path / "05_long_calibration"

    def run(target: Path = output) -> dict[str, Any]:
        return stage.run_s05_calibrate_long(
            directories["00_ingest"],
            directories["01_microtrack"],
            directories["02_appearance"],
            directories["03_calibration"],
            directories["04_short_stable"],
            CONFIG,
            target,
            logger=lambda message: None,
        )

    return output, run


def test_disabled_stage_atomically_commits_four_partition_parquets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output, run = _fixture(tmp_path, monkeypatch)

    marker = run()

    assert output.is_dir()
    assert not list(tmp_path.glob(".05_long_calibration.staging-*"))
    assert marker["stats"]["num_pairs"] == 8
    assert marker["stats"]["num_global_merges"] == 0
    for partition, name in PARQUETS.items():
        table = pq.read_table(output / name)
        assert table.num_rows == 2
        assert set(table["partition"].to_pylist()) == {partition}
    thresholds = json.loads((output / "thresholds.json").read_text(encoding="utf-8"))
    assert thresholds["long_model_enabled"] is False
    assert thresholds["long_confirmed_enabled"] is False
    assert thresholds["long_confirmed_threshold"] is None
    assert thresholds["global_merge_allowed"] is False
    assert load_link_model(output / "link_model_long.joblib").pipeline is None
    assert not (output / "stable_to_global.parquet").exists()


def test_partial_write_failure_leaves_no_committed_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output, run = _fixture(tmp_path, monkeypatch)
    real_write = stage._write_parquet
    calls = 0

    def fail_after_one(*args: Any, **kwargs: Any) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ContractError("synthetic write failure")
        real_write(*args, **kwargs)

    monkeypatch.setattr(stage, "_write_parquet", fail_after_one)
    with pytest.raises(ContractError, match="synthetic write failure"):
        run()
    assert not output.exists()
    assert not list(tmp_path.glob(".05_long_calibration.staging-*"))


def test_completed_rerun_revalidates_without_remining_and_rejects_semantic_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output, run = _fixture(tmp_path, monkeypatch)
    first = run()
    monkeypatch.setattr(
        stage,
        "generate_stable_long_pairs",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must not remine")),
    )
    assert run() == first

    thresholds_path = output / "thresholds.json"
    thresholds = json.loads(thresholds_path.read_text(encoding="utf-8"))
    thresholds["global_merge_allowed"] = True
    thresholds_path.write_text(json.dumps(thresholds), encoding="utf-8")
    _update_output_fingerprint(output, "thresholds.json")
    with pytest.raises(ContractError, match="threshold policy differs"):
        run()


def test_completed_rerun_rejects_report_count_tamper_even_with_updated_fingerprint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output, run = _fixture(tmp_path, monkeypatch)
    run()
    report_path = output / "long_calibration_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["pair_counts"]["total_rows"] += 1
    report_path.write_text(json.dumps(report), encoding="utf-8")
    _update_output_fingerprint(output, "long_calibration_report.json")
    with pytest.raises(ContractError, match="report policy/content differs"):
        run()


def test_completed_rerun_recomputes_enabled_model_scores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output, run = _fixture(tmp_path, monkeypatch)
    real_settings = stage._settings

    def tiny_settings(config):
        settings = real_settings(config)
        return replace(
            settings,
            expected_clip_ids=("GX040006",),
            minimums=CalibrationMinimums(
                train_positive_groups=1,
                train_hard_negative_groups=1,
                selection_positive_groups=1,
                selection_hard_negative_groups=1,
                certification_positive_groups=1,
                certification_hard_negative_groups=1,
                audit_positive_groups=1,
                audit_hard_negative_groups=1,
                parent_groups_per_partition=1,
                audit_groups_per_class_per_clip=1,
            ),
        )

    monkeypatch.setattr(stage, "_settings", tiny_settings)
    marker = run()
    assert marker["stats"]["long_model_enabled"] is True

    parquet_path = output / PARQUETS["train"]
    rows = pq.read_table(parquet_path).to_pylist()
    rows[0]["model_probability"] += 0.01
    pq.write_table(
        pa.Table.from_pylist(rows, schema=stage.LONG_CALIBRATION_PAIRS_SCHEMA),
        parquet_path,
    )
    _update_output_fingerprint(output, PARQUETS["train"])
    with pytest.raises(ContractError, match="model scores differ"):
        run()


def test_nonempty_or_file_output_without_success_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, run = _fixture(tmp_path, monkeypatch)
    nonempty = tmp_path / "nonempty"
    nonempty.mkdir()
    (nonempty / "partial.tmp").write_bytes(b"partial")
    with pytest.raises(ContractError, match="non-empty without _SUCCESS"):
        run(nonempty)

    file_output = tmp_path / "ordinary-file"
    file_output.write_bytes(b"not a directory")
    with pytest.raises(ContractError, match="output path is not a directory"):
        run(file_output)
