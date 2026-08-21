from __future__ import annotations

import copy

import numpy as np
import pytest

from cowtrack.config import ContractError
from cowtrack.linking.calibration_core import (
    AUDIT_EVIDENCE_AGGREGATE_OBSERVED_ROWS,
    CalibrationMinimums,
    LongCalibrationSettings,
    calibrate_long_rows,
    certify_long_gate,
)
from cowtrack.linking.model import LinkScorer


FEATURES = ("prototype_cosine_max", "quality")


def _minimums(**changes: int) -> CalibrationMinimums:
    values = {
        "train_positive_groups": 2,
        "train_hard_negative_groups": 2,
        "selection_positive_groups": 2,
        "selection_hard_negative_groups": 2,
        "certification_positive_groups": 2,
        "certification_hard_negative_groups": 2,
        "audit_positive_groups": 2,
        "audit_hard_negative_groups": 2,
        "parent_groups_per_partition": 2,
        "audit_groups_per_class_per_clip": 1,
    }
    values.update(changes)
    return CalibrationMinimums(**values)


def _settings(**changes: object) -> LongCalibrationSettings:
    values: dict[str, object] = {
        "random_seed": 19,
        "logistic_c": 1.0,
        "max_iter": 2_000,
        "tolerance": 1e-6,
        "confirmed_far_target": 0.4,
        "confidence_level": 0.5,
        "provisional_tpr_target": 2.0 / 3.0,
        "minimums": _minimums(),
    }
    values.update(changes)
    return LongCalibrationSettings(**values)


def _rows(*, explicit_margin: bool = True) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for partition_index, partition in enumerate(
        ("train", "threshold_selection", "certification", "audit")
    ):
        for group_index in range(3):
            parent = partition_index * 10 + group_index
            group_id = f"{partition}-{group_index}"
            clip = "clip-a" if group_index != 2 else "clip-b"
            for label in (True, False):
                high = 0.95 - 0.02 * group_index if label else 0.10 + 0.01 * group_index
                row: dict[str, object] = {
                    "pair_id": f"{group_id}-{'positive' if label else 'negative'}",
                    "partition": partition,
                    "label": label,
                    "candidate_group_id": group_id,
                    "parent_stable_id": parent,
                    "source_clip_id": clip,
                    "target_clip_id": clip,
                    "stratum": "positive_clean" if label else "hard_negative",
                    "features": {
                        "prototype_cosine_max": high,
                        "quality": 0.9 if label else 0.2,
                    },
                }
                if explicit_margin:
                    row["candidate_margin"] = 0.4 if label else -0.4
                rows.append(row)
    return rows


def test_calibration_is_shuffle_invariant_and_link_scorer_compatible() -> None:
    rows = _rows()
    first = calibrate_long_rows(rows, FEATURES, settings=_settings())
    shuffled = list(reversed(rows))
    second = calibrate_long_rows(shuffled, FEATURES, settings=_settings())

    assert first.model_enabled is True
    assert first.thresholds == second.thresholds
    assert first.scored_rows == second.scored_rows
    assert first.thresholds["confirmed_enabled"] is True
    assert first.thresholds["provisional_threshold_source"] == (
        "threshold_selection_positive_tpr"
    )
    assert first.thresholds["counting_unit"].startswith("unique_candidate_group_id")
    LinkScorer(first.model, first.thresholds)


def test_certification_positives_and_audit_cannot_select_or_move_gates() -> None:
    baseline_rows = _rows()
    changed_rows = copy.deepcopy(baseline_rows)
    for row in changed_rows:
        if row["partition"] == "certification" and row["label"] is True:
            row["features"]["prototype_cosine_max"] = 0.01
            row["features"]["quality"] = 0.01
            row["candidate_margin"] = -10.0
        if row["partition"] == "audit":
            row["features"]["prototype_cosine_max"] = 0.5
            row["features"]["quality"] = 0.5

    baseline = calibrate_long_rows(baseline_rows, FEATURES, settings=_settings())
    changed = calibrate_long_rows(changed_rows, FEATURES, settings=_settings())
    for name in (
        "provisional_threshold",
        "selected_confirmed_threshold",
        "selected_appearance_margin_threshold",
        "confirmed_enabled",
        "confirmed_threshold",
        "appearance_margin_threshold",
        "confirmed_false_accepts",
        "confirmed_false_accept_upper",
    ):
        assert changed.thresholds[name] == baseline.thresholds[name]
    assert changed.thresholds["certification_true_accepts"] == 0
    assert baseline.thresholds["certification_true_accepts"] > 0
    assert changed.audit_metrics != baseline.audit_metrics


def test_certification_counts_unique_negative_groups_not_duplicate_rows() -> None:
    result = certify_long_gate(
        np.asarray([0.1, 0.2, 0.2, 0.9]),
        np.asarray([False, False, False, True]),
        np.asarray([-0.2, -0.1, -0.1, 0.4]),
        ["negative-a", "negative-b", "negative-b", "positive-a"],
        probability_threshold=0.8,
        margin_threshold=0.0,
        confirmed_far_target=0.9,
        confidence_level=0.5,
    )
    assert result.hard_negative_groups == 2
    assert result.false_accepts == 0
    assert result.positive_groups == 1
    assert result.positive_accepts == 1


def test_certification_can_only_disable_the_frozen_selection_gate() -> None:
    rows = _rows()
    for row in rows:
        if row["partition"] == "certification" and row["label"] is False:
            row["features"]["prototype_cosine_max"] = 0.99
            row["features"]["quality"] = 0.99
            row["candidate_margin"] = 0.5
    result = calibrate_long_rows(rows, FEATURES, settings=_settings())
    assert result.thresholds["selected_confirmed_threshold"] is not None
    assert result.thresholds["selected_appearance_margin_threshold"] is not None
    assert result.thresholds["confirmed_enabled"] is False
    assert result.thresholds["confirmed_threshold"] is None
    assert result.thresholds["appearance_margin_threshold"] is None
    assert result.thresholds["confirmed_disabled_reason"] == (
        "certification_false_accept_upper_exceeds_target"
    )


def test_insufficient_groups_produce_disabled_artifact_without_lowering_minimum() -> None:
    result = calibrate_long_rows(
        _rows(),
        FEATURES,
        settings=_settings(
            minimums=_minimums(train_positive_groups=4),
        ),
    )
    assert result.model_enabled is False
    assert result.model.disabled_reason == (
        "train_groups_below_minimum:positive=3/4,hard_negative=3/2"
    )
    assert result.thresholds["model_enabled"] is False
    assert result.thresholds["provisional_threshold"] is None
    assert {row["decision"] for row in result.scored_rows} == {"reject"}
    assert all(row["model_probability"] is None for row in result.scored_rows)


def test_aggregate_audit_uses_dynamic_clip_count_and_keeps_sparse_clip_warning() -> None:
    rows = _rows()
    replacements = {
        "clip-a": "farm2-camera3-part-a",
        "clip-b": "farm2-camera3-part-b",
    }
    for row in rows:
        row["source_clip_id"] = replacements[str(row["source_clip_id"])]
        row["target_clip_id"] = replacements[str(row["target_clip_id"])]
    clips = (
        "farm2-camera3-part-a",
        "farm2-camera3-part-b",
        "farm2-camera3-unobserved-part",
    )

    result = calibrate_long_rows(
        rows,
        FEATURES,
        settings=_settings(
            audit_evidence_scope=AUDIT_EVIDENCE_AGGREGATE_OBSERVED_ROWS,
            expected_clip_ids=clips,
        ),
    )

    assert result.model_enabled is True
    adequacy = result.audit_adequacy
    assert adequacy["evidence_scope"] == "aggregate_observed_rows"
    assert adequacy["expected_clip_count"] == 3
    assert adequacy["clip_count_scaled_minimum_per_class"] == 3
    assert adequacy["required_aggregate_groups"] == {
        "positive": 3,
        "hard_negative": 3,
    }
    assert adequacy["sufficient_for_selected_scope"] is True
    sparse = adequacy["per_clip"]["farm2-camera3-unobserved-part"]
    assert sparse["positive_groups"] == 0
    assert sparse["hard_negative_groups"] == 0
    assert sparse["meets_nominal_minimum"] is False
    assert len(adequacy["warnings"]) == 1


def test_aggregate_audit_scaled_minimum_disables_without_special_casing_clip() -> None:
    result = calibrate_long_rows(
        _rows(),
        FEATURES,
        settings=_settings(
            audit_evidence_scope=AUDIT_EVIDENCE_AGGREGATE_OBSERVED_ROWS,
            expected_clip_ids=("clip-a", "clip-b", "clip-c", "clip-d"),
        ),
    )

    assert result.model_enabled is False
    assert result.model.disabled_reason == (
        "audit_aggregate_groups_below_scaled_minimum:"
        "positive=3/4,hard_negative=3/4,clip_count=4"
    )


def test_aggregate_audit_retains_partition_global_minimum() -> None:
    result = calibrate_long_rows(
        _rows(),
        FEATURES,
        settings=_settings(
            minimums=_minimums(audit_positive_groups=4),
            audit_evidence_scope=AUDIT_EVIDENCE_AGGREGATE_OBSERVED_ROWS,
            expected_clip_ids=("clip-a", "clip-b"),
        ),
    )

    assert result.model_enabled is False
    assert result.model.disabled_reason == (
        "audit_groups_below_minimum:positive=3/4,hard_negative=3/2"
    )


def test_malformed_rows_fail_closed_before_fitting() -> None:
    nonfinite = _rows()
    nonfinite[0]["features"]["quality"] = np.nan
    with pytest.raises(ContractError, match="non-finite"):
        calibrate_long_rows(nonfinite, FEATURES, settings=_settings())

    non_boolean = _rows()
    non_boolean[0]["label"] = 1
    with pytest.raises(ContractError, match="label must be boolean"):
        calibrate_long_rows(non_boolean, FEATURES, settings=_settings())

    leakage = _rows()
    leakage[-1]["parent_stable_id"] = leakage[0]["parent_stable_id"]
    with pytest.raises(ContractError, match="crosses partitions"):
        calibrate_long_rows(leakage, FEATURES, settings=_settings())


def test_margin_can_be_derived_from_fixed_feature_within_candidate_group() -> None:
    result = calibrate_long_rows(
        _rows(explicit_margin=False), FEATURES, settings=_settings()
    )
    by_group: dict[str, list[dict[str, object]]] = {}
    for row in result.scored_rows:
        by_group.setdefault(str(row["candidate_group_id"]), []).append(row)
    for group in by_group.values():
        positive = next(row for row in group if row["label"] is True)
        negative = next(row for row in group if row["label"] is False)
        assert positive["candidate_margin"] > 0.0
        assert negative["candidate_margin"] < 0.0
        assert positive["candidate_margin"] == pytest.approx(
            -float(negative["candidate_margin"])
        )


def test_explicit_margin_must_be_all_or_none() -> None:
    rows = _rows()
    del rows[0]["candidate_margin"]
    with pytest.raises(ContractError, match="every row or none"):
        calibrate_long_rows(rows, FEATURES, settings=_settings())


def test_high_overlap_is_part_of_the_frozen_confirmation_predicate() -> None:
    rows = _rows()
    blocked_pair_id = "certification-0-positive"
    for row in rows:
        row["high_overlap"] = row["pair_id"] == blocked_pair_id
    result = calibrate_long_rows(rows, FEATURES, settings=_settings())
    blocked = next(
        row for row in result.scored_rows if row["pair_id"] == blocked_pair_id
    )
    assert blocked["model_probability"] >= result.thresholds["confirmed_threshold"]
    assert blocked["candidate_margin"] >= result.thresholds["appearance_margin_threshold"]
    assert blocked["decision"] != "confirmed"

    certificate = certify_long_gate(
        np.asarray([0.99, 0.99]),
        np.asarray([False, True]),
        np.asarray([0.5, 0.5]),
        ["negative", "positive"],
        probability_threshold=0.9,
        margin_threshold=0.1,
        confirmed_far_target=0.9,
        confidence_level=0.5,
        reliability_eligible=np.asarray([False, True]),
    )
    assert certificate.false_accepts == 0
