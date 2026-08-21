from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from cowtrack.config import ContractError
from cowtrack.linking.config import load_link_calibration_config
from cowtrack.linking.dataset_contract import EXPECTED_CLIP_ORDER
from cowtrack.linking.features import feature_schema
from cowtrack.schemas.calibration import PSEUDO_PAIRS_SCHEMA
from cowtrack.stages.s03_calibrate import (
    _audit_metrics,
    _build_calibration_input,
    _canonical_pair_rows,
    _effective_sample_requirements,
    _feature_schema_payload,
    _fingerprint,
    _fit_and_score,
    _pair_table,
    _verify_upstream_fingerprints,
    run_s03,
)


CONFIG = Path(__file__).resolve().parents[1] / "configs" / "s03_calibration.yaml"


def _config():
    config, _, _ = load_link_calibration_config(CONFIG)
    return replace(
        config,
        min_train_positive_per_model=1,
        min_train_hard_negative_per_model=1,
        min_calibration_positive_per_model=1,
        min_calibration_hard_negative_per_model=1,
        min_audit_positive_per_model=1,
        min_audit_hard_negative_per_model=1,
        min_audit_pairs_per_class_per_clip=1,
        min_parent_groups_per_split=1,
    )


def _features(mode: str, *, positive: bool) -> dict[str, float]:
    high = 0.95 if positive else 0.10
    values = {
        "prototype_cosine_max": high,
        "prototype_cosine_top3_mean": high,
        "medoid_cosine": high,
        "mutual_prototype_score": high,
        "appearance_quality_min": 0.9 if positive else 0.6,
        "appearance_quality_mean": 0.9 if positive else 0.6,
        "gap_sec": 2.0 if mode == "short" else 10.0,
        "log1p_gap_sec": 1.0986122887 if mode == "short" else 2.3978952728,
        "log_width_ratio": 0.0,
        "log_height_ratio": 0.0,
        "log_area_ratio": 0.0,
        "src_end_max_other_iou": 0.0,
        "dst_start_max_other_iou": 0.0,
        "src_end_boundary_distance": 0.5,
        "dst_start_boundary_distance": 0.5,
        "is_clip_boundary": 0.0,
    }
    if mode == "short":
        values["predicted_center_residual"] = 0.05 if positive else 1.0
        values["predicted_iou"] = 0.8 if positive else 0.0
    return {name: values[name] for name in feature_schema(mode)}


def _row(
    *,
    mode: str,
    split: str,
    clip: str,
    parent: int,
    label: bool,
    calibration_role: str = "not_applicable",
) -> dict[str, object]:
    group = f"{mode}-{split}-{clip}-{parent}"
    source_base = parent * 1000
    target_base = source_base + (100 if label else 200)
    row: dict[str, object] = {
        "pair_id": f"{group}-{'p' if label else 'n'}",
        "candidate_group_id": group,
        "parent_group_id": f"parent-{parent}",
        "split": split,
        "calibration_role": calibration_role,
        "mode": mode,
        "label": label,
        "pair_kind": "pseudo_positive" if label else "hard_negative_simultaneous",
        "stratum": "positive_clean" if label else "hard_negative",
        "parent_micro_id": parent,
        "source_micro_id": parent,
        "target_micro_id": parent if label else parent + 100,
        "source_start_det_id": source_base,
        "source_end_det_id": source_base + 10,
        "target_start_det_id": target_base,
        "target_end_det_id": target_base + 10,
        "source_end_global_frame": source_base + 10,
        "target_start_global_frame": source_base + 12,
        "source_start_global_frame": source_base,
        "target_end_global_frame": source_base + 20,
        "source_start_time_sec": float(source_base),
        "source_end_time_sec": float(source_base + 10),
        "target_start_time_sec": float(source_base + 12),
        "target_end_time_sec": float(source_base + 20),
        "source_clip_id": clip,
        "target_clip_id": clip,
        "source_segment_start_clip_id": clip,
        "target_segment_end_clip_id": clip,
        "source_num_detections": 11,
        "target_num_detections": 9,
        "source_sample_ids": [source_base + 1, source_base + 2, source_base + 3],
        "target_sample_ids": [target_base + 1, target_base + 2, target_base + 3],
        "source_gallery_det_ids": [source_base + 1, source_base + 2, source_base + 3],
        "target_gallery_det_ids": [target_base + 1, target_base + 2, target_base + 3],
        "source_embedding_rows": [source_base + 1, source_base + 2, source_base + 3],
        "target_embedding_rows": [target_base + 1, target_base + 2, target_base + 3],
        "source_medoid_sample_id": source_base + 2,
        "target_medoid_sample_id": target_base + 2,
        "source_appearance_quality": 0.9,
        "target_appearance_quality": 0.9 if label else 0.6,
        "source_internal_cosine_p10": 0.9,
        "target_internal_cosine_p10": 0.9,
        "source_internal_cosine_p50": 0.95,
        "target_internal_cosine_p50": 0.95,
        "source_internal_cosine_min": 0.85,
        "target_internal_cosine_min": 0.85,
        "source_gallery_num_input_samples": 3,
        "target_gallery_num_input_samples": 3,
        "source_gallery_num_overlap_rejected": 0,
        "target_gallery_num_overlap_rejected": 0,
        "source_gallery_num_review_excluded": 0,
        "target_gallery_num_review_excluded": 0,
        "source_gallery_num_local_outliers": 0,
        "target_gallery_num_local_outliers": 0,
        "source_gallery_max_other_bbox_iou": 0.0,
        "target_gallery_max_other_bbox_iou": 0.0,
        "source_gallery_max_clean_other_bbox_iou": 0.0,
        "target_gallery_max_clean_other_bbox_iou": 0.0,
        "gap_target_sec": 2.0 if mode == "short" else 10.0,
        "gap_error_sec": 0.0,
        "hard_negative_rank": 0 if label else 1,
        "appearance_present": True,
        "high_overlap": False,
    }
    row.update(_features(mode, positive=label))
    return row


def _rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    parent = 1
    for mode in ("short", "long"):
        for split in ("train", "calibration"):
            for clip in ("GX040006", "GX050006"):
                groups = 2 if split == "calibration" else 1
                for group_index in range(groups):
                    calibration_role = (
                        "threshold_selection"
                        if split == "calibration" and group_index == 0
                        else "certification"
                        if split == "calibration"
                        else "not_applicable"
                    )
                    for label in (True, False):
                        rows.append(
                            _row(
                                mode=mode,
                                split=split,
                                clip=clip,
                                parent=parent,
                                label=label,
                                calibration_role=calibration_role,
                            )
                        )
                    parent += 1
        for clip in ("GX040006", "GX050006"):
            for label in (True, False):
                rows.append(
                    _row(
                        mode=mode,
                        split="audit",
                        clip=clip,
                        parent=parent,
                        label=label,
                    )
                )
            parent += 1
    return rows


def test_per_clip_shortfall_is_warning_when_aggregate_evidence_is_sufficient() -> None:
    rows = _rows()
    for row in rows:
        if row["mode"] == "short" and row["split"] == "calibration":
            row["source_clip_id"] = "GX040006"
            row["target_clip_id"] = "GX040006"
    models, thresholds = _fit_and_score(
        rows, _config(), ["GX040006", "GX050006"]
    )
    assert models["short"].pipeline is not None
    assert thresholds["short"]["model_enabled"] is True

    metrics = _audit_metrics(
        rows, thresholds, _config(), ["GX040006", "GX050006"]
    )
    warning = next(
        item
        for item in metrics["short"]["per_clip_evidence_warnings"]
        if item["split"] == "calibration"
        and item["calibration_role"] == "not_applicable"
        and item["clip_id"] == "GX050006"
    )
    assert warning["positive"] == warning["hard_negative"] == 0
    assert warning["model_enablement_effect"] == (
        "none_when_aggregate_minimum_is_met"
    )


def _aggregate_only_rows(
    clip_ids: tuple[str, ...], *, evidence_clip: str
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    parent = 1
    groups_per_class = len(clip_ids)
    for mode in ("short", "long"):
        for split, roles in (
            ("train", ("not_applicable",)),
            (
                "calibration",
                ("threshold_selection", "certification"),
            ),
            ("audit", ("not_applicable",)),
        ):
            for role in roles:
                for _ in range(groups_per_class):
                    for label in (True, False):
                        rows.append(
                            _row(
                                mode=mode,
                                split=split,
                                clip=evidence_clip,
                                parent=parent,
                                label=label,
                                calibration_role=role,
                            )
                        )
                    parent += 1
    return rows


def test_arbitrary_clip_ids_use_scaled_aggregate_gate_without_dataset_contract() -> None:
    clip_ids = (
        "farm2_cam3_a",
        "farm2_cam3_b",
        "farm2_cam3_c",
    )
    rows = _aggregate_only_rows(clip_ids, evidence_clip=clip_ids[0])
    config = _config()
    requirements = _effective_sample_requirements(config, clip_ids)
    assert requirements["evidence_scope"] == "aggregate_observed_rows"
    assert requirements["clip_count"] == 3
    assert requirements["splits"]["audit"]["positive"] == 3
    assert requirements["calibration_roles"]["certification"]["positive"] == 3

    models, thresholds = _fit_and_score(rows, config, clip_ids)
    assert models["short"].pipeline is not None
    assert models["long"].pipeline is not None
    metrics = _audit_metrics(rows, thresholds, config, clip_ids)
    assert metrics["short"]["aggregate_evidence"]["audit"] == {
        "positive": 3,
        "hard_negative": 3,
        "required_positive": 3,
        "required_hard_negative": 3,
        "meets_minimum": True,
    }
    assert {
        warning["clip_id"]
        for warning in metrics["short"]["per_clip_evidence_warnings"]
    } >= {"farm2_cam3_b", "farm2_cam3_c"}
    assert metrics["short"]["confirmed_gate"]["fail_closed"] is True
    assert thresholds["short"]["confirmed_enabled"] is False


def test_clip_count_scaling_disables_model_when_total_evidence_is_too_small() -> None:
    clip_ids = ("farm2_cam3_a", "farm2_cam3_b", "farm2_cam3_c")
    rows = _aggregate_only_rows(clip_ids, evidence_clip=clip_ids[0])
    audit_parents = {
        row["parent_micro_id"]
        for row in rows
        if row["mode"] == "short" and row["split"] == "audit"
    }
    removed_parent = max(audit_parents)
    rows = [
        row
        for row in rows
        if not (
            row["mode"] == "short"
            and row["split"] == "audit"
            and row["parent_micro_id"] == removed_parent
        )
    ]
    models, thresholds = _fit_and_score(rows, _config(), clip_ids)
    assert models["short"].pipeline is None
    assert thresholds["short"]["disabled_reason"].startswith(
        "audit_samples_below_minimum:positive=2/3,hard_negative=2/3"
    )


def test_consumed_upstream_artifact_must_match_success_fingerprint(tmp_path) -> None:
    stage_dir = tmp_path / "upstream"
    stage_dir.mkdir()
    artifact = stage_dir / "artifact.bin"
    artifact.write_bytes(b"immutable")
    original = _fingerprint(artifact)
    (stage_dir / "_SUCCESS.json").write_text(
        json.dumps(
            {
                "stage": "S00",
                "output_fingerprints": [
                    {
                        "path": "artifact.bin",
                        "size_bytes": original["size_bytes"],
                        "sha256": original["sha256"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    _verify_upstream_fingerprints(
        stage_dir,
        "S00",
        ("artifact.bin",),
        {str(artifact.resolve()): original},
    )

    artifact.write_bytes(b"tampered")
    modified = _fingerprint(artifact)
    with pytest.raises(ContractError, match="completed S00 artifact changed"):
        _verify_upstream_fingerprints(
            stage_dir,
            "S00",
            ("artifact.bin",),
            {str(artifact.resolve()): modified},
        )


def test_fit_score_and_arrow_schema_fail_closed_without_far_evidence() -> None:
    rows = _rows()
    models, thresholds = _fit_and_score(
        rows, _config(), ["GX040006", "GX050006"]
    )
    assert models["short"].pipeline is not None
    assert models["long"].pipeline is not None
    assert thresholds["short"]["confirmed_enabled"] is False
    assert thresholds["long"]["confirmed_enabled"] is False
    assert {row["decision"] for row in rows}.issubset({"provisional", "reject"})

    for row in rows:
        if row["mode"] == "long":
            row["predicted_center_residual"] = None
            row["predicted_iou"] = None
    table = _pair_table(rows)
    assert table.schema == PSEUDO_PAIRS_SCHEMA
    assert table.num_rows == len(rows)


def test_feature_schema_contains_only_allowed_model_features() -> None:
    payload = _feature_schema_payload()
    serialized = repr(payload).lower()
    assert "legacy" not in serialized
    assert "keypoint" not in serialized
    assert payload["short"]["ordered_features"] == list(feature_schema("short"))
    assert payload["long"]["ordered_features"] == list(feature_schema("long"))


def test_zero_pairs_produce_explicit_disabled_models_and_empty_schema() -> None:
    models, thresholds = _fit_and_score(
        [], _config(), ["GX040006", "GX050006"]
    )
    assert models["short"].pipeline is None
    assert models["long"].pipeline is None
    assert thresholds["short"]["disabled_reason"].startswith(
        "train_samples_below_minimum"
    )
    assert _pair_table([]).schema == PSEUDO_PAIRS_SCHEMA
    assert _pair_table([]).num_rows == 0


def test_group_without_negative_and_gap_leakage_are_rejected() -> None:
    class Pair:
        def __init__(self, row):
            self.row = row

        def as_row(self):
            return self.row

    positive = _row(
        mode="short", split="train", clip="GX040006", parent=1, label=True
    )
    with pytest.raises(ContractError, match="1-positive/N-negative"):
        _canonical_pair_rows([Pair(positive)])

    negative = _row(
        mode="short", split="train", clip="GX040006", parent=1, label=False
    )
    negative["candidate_group_id"] = positive["candidate_group_id"]
    negative["gap_sec"] = -1.0
    with pytest.raises(ContractError, match="non-positive gap"):
        _canonical_pair_rows([Pair(positive), Pair(negative)])


def test_mocked_s03_stage_commits_and_revalidates_without_real_data(
    tmp_path, monkeypatch
) -> None:
    import cowtrack.linking.pseudo_pairs as pseudo_module
    import cowtrack.stages.s03_calibrate as stage

    class Pair:
        def __init__(self, row):
            self.row = row

        def as_row(self):
            return self.row

    pairs = [Pair(row) for row in _rows()]
    fake_data = {
        "micro_appearance": {
            "micro_id": [1, 2],
            "appearance_usable": [True, False],
        },
        "micro_id_by_detection": [1, 2],
        "clip_id": ["GX040006", "GX050006"],
        "appearance_report": {
            "warnings": ["hard_negative_separation_below_configured_margin"],
            "stats": {"hard_negative_separation_warning": True},
        },
        "encoder_choice": {"selection_metrics": {"fpr_at_tpr95": 0.1}},
    }

    monkeypatch.setattr(stage, "_stage_input_paths", lambda *args: [])
    monkeypatch.setattr(stage, "_verify_upstream_fingerprints", lambda *args: None)
    fake_calibration_input = type(
        "FakeCalibrationInput",
        (),
        {
            "parent_micro_ids": [1, 2],
            "timeline_clip_ids": list(EXPECTED_CLIP_ORDER),
        },
    )()
    fake_bundle = type(
        "FakeProductionInputBundle",
        (),
        {
            "calibration_input": fake_calibration_input,
            "consumed_paths": (),
            "micro_appearance": fake_data["micro_appearance"],
            "detections": type(
                "FakeDetections",
                (),
                {
                    "micro_ids": fake_data["micro_id_by_detection"],
                    "clip_ids": fake_data["clip_id"],
                },
            )(),
            "appearance_report": fake_data["appearance_report"],
            "encoder_choice": fake_data["encoder_choice"],
        },
    )()
    monkeypatch.setattr(
        stage, "load_production_inputs", lambda *args, **kwargs: fake_bundle
    )

    def fake_generate(data, config, **kwargs):
        kwargs["clean_gallery_status"].update({1: True, 2: False})
        return tuple(pairs)

    monkeypatch.setattr(
        pseudo_module,
        "generate_pseudo_pairs",
        fake_generate,
    )
    output = tmp_path / "03_calibration"
    success = run_s03(
        tmp_path / "00_ingest",
        tmp_path / "01_microtrack",
        tmp_path / "02_appearance",
        CONFIG,
        output,
        logger=lambda message: None,
    )
    assert success["stage"] == "S03"
    assert success["stats"]["short_model_enabled"] is False
    assert (output / "_SUCCESS.json").is_file()
    assert (output / "pseudo_pairs_audit.parquet").is_file()
    report = json.loads((output / "calibration_report.json").read_text(encoding="utf-8"))
    assert report["s02_appearance_population"] == {
        "present": 1,
        "missing": 1,
        "per_clip": {
            clip: {
                "present": 1 if clip == "GX040006" else 0,
                "missing": 1 if clip == "GX050006" else 0,
            }
            for clip in EXPECTED_CLIP_ORDER
        },
    }
    assert report["s03_clean_gallery_population"]["present"] == 1
    assert report["s03_clean_gallery_population"]["missing"] == 1
    for mode in ("short", "long"):
        assert set(report["metrics"][mode]["per_split_per_clip"]) == {
            "train",
            "calibration",
            "audit",
        }
        assert set(
            report["metrics"][mode]["per_split_per_clip"]["audit"]
        ) == set(EXPECTED_CLIP_ORDER)

    repeated = run_s03(
        tmp_path / "00_ingest",
        tmp_path / "01_microtrack",
        tmp_path / "02_appearance",
        CONFIG,
        output,
        logger=lambda message: None,
    )
    assert repeated == success


def test_stage_bridge_matches_columnar_pseudo_pair_contract() -> None:
    import numpy as np

    data = {
        "micro_summary": {
            "micro_id": np.asarray([7]),
            "status": np.asarray(["valid"], dtype=object),
            "num_detections": np.asarray([1]),
            "local_purity_score": np.asarray([1.0]),
            "bidirectional_agreement": np.asarray([1.0]),
        },
        "micro_appearance": {
            "micro_id": np.asarray([7]),
            "internal_cosine_p10": np.asarray([0.9]),
        },
        "frame_clip_id": np.asarray(["GX040006", "GX040006"], dtype=object),
        "frame_global_time_sec": np.asarray([0.0, 1.0]),
        "det_id": np.asarray([70]),
        "micro_id_by_detection": np.asarray([7]),
        "order_in_micro": np.asarray([0]),
        "clip_id": np.asarray(["GX040006"], dtype=object),
        "global_frame": np.asarray([0]),
        "global_time_sec": np.asarray([0.0]),
        "cx_norm": np.asarray([0.5]),
        "cy_norm": np.asarray([0.5]),
        "w_norm": np.asarray([0.2]),
        "h_norm": np.asarray([0.2]),
        "other_bbox_max_iou": np.asarray([0.0]),
        "boundary_distance": np.asarray([0.5]),
        "review_excluded": np.asarray([False]),
        "sample_id": np.asarray([0]),
        "sample_micro_id": np.asarray([7]),
        "sample_det_id": np.asarray([70]),
        "sample_crop_quality": np.asarray([1.0]),
        "sample_other_bbox_max_iou": np.asarray([0.0]),
        "sample_s02_inlier": np.asarray([True]),
        "sample_embedding_row": np.asarray([0]),
        "sample_embeddings": np.asarray([[1.0, 0.0]], dtype=np.float16),
    }
    bridged = _build_calibration_input(data)
    assert bridged.parent_micro_ids.tolist() == [7]
    assert bridged.parent_local_purity_score.tolist() == [1.0]
    assert bridged.det_other_bbox_max_iou.tolist() == [0.0]
