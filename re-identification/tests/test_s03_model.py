from __future__ import annotations

import numpy as np
import pytest

from cowtrack.config import ContractError
from cowtrack.linking.model import (
    LinkScorer,
    calibrate_mode_thresholds,
    clopper_pearson_upper,
    disabled_link_model,
    fit_link_model,
    load_link_model,
    save_link_model,
)


def _fitted_model():
    negative = np.column_stack((np.linspace(-2.0, -0.2, 40), np.zeros(40)))
    positive = np.column_stack((np.linspace(0.2, 2.0, 40), np.ones(40)))
    return fit_link_model(
        np.vstack((negative, positive)),
        np.r_[np.zeros(40, dtype=np.int8), np.ones(40, dtype=np.int8)],
        ("similarity", "quality"),
        mode="short",
        random_seed=20260710,
        logistic_c=1.0,
        max_iter=1000,
        minimum_per_class=20,
    )


def _enabled_thresholds(mode: str, *, margin: float | None = None) -> dict[str, object]:
    certification_negatives = 30_000
    return {
        "mode": mode,
        "model_enabled": True,
        "disabled_reason": None,
        "confirmed_enabled": True,
        "confirmed_threshold": 0.5,
        "provisional_threshold": 0.4,
        "appearance_margin_threshold": margin,
        "confirmed_far_target": 1e-4,
        "confidence_level": 0.95,
        "confirmed_false_accepts": 0,
        "certification_hard_negative_count": certification_negatives,
        "confirmed_false_accept_upper": clopper_pearson_upper(
            0, certification_negatives, 0.95
        ),
        "confirmed_certified_independently": True,
    }


def test_model_round_trip_is_exact(tmp_path) -> None:
    artifact = _fitted_model()
    probe = np.asarray([[-0.5, 0.0], [0.5, 1.0]], dtype=np.float64)
    before_probability = artifact.probabilities(probe)
    before_raw = artifact.raw_scores(probe)
    path = tmp_path / "link_model_short.joblib"
    save_link_model(path, artifact)
    loaded = load_link_model(path)
    np.testing.assert_array_equal(loaded.probabilities(probe), before_probability)
    np.testing.assert_array_equal(loaded.raw_scores(probe), before_raw)
    assert loaded.feature_names == ("similarity", "quality")


def test_model_rejects_nonfinite_and_insufficient_training_data() -> None:
    with pytest.raises(ContractError, match="non-finite"):
        fit_link_model(
            np.asarray([[0.0], [np.nan]]),
            np.asarray([0, 1]),
            ("x",),
            mode="short",
            random_seed=1,
            logistic_c=1.0,
            max_iter=10,
            minimum_per_class=1,
        )
    with pytest.raises(ContractError, match="insufficient"):
        fit_link_model(
            np.asarray([[0.0], [1.0]]),
            np.asarray([0, 1]),
            ("x",),
            mode="short",
            random_seed=1,
            logistic_c=1.0,
            max_iter=10,
            minimum_per_class=2,
        )


def test_model_fails_closed_when_logistic_regression_does_not_converge() -> None:
    features = np.r_[np.linspace(-5.0, -0.1, 100), np.linspace(0.1, 5.0, 100)][
        :, None
    ]
    labels = np.r_[np.zeros(100, dtype=np.int8), np.ones(100, dtype=np.int8)]
    with pytest.raises(ContractError, match="did not converge"):
        fit_link_model(
            features,
            labels,
            ("x",),
            mode="short",
            random_seed=1,
            logistic_c=1.0,
            max_iter=1,
            minimum_per_class=20,
            tolerance=1e-12,
        )


def test_clopper_pearson_requires_about_thirty_thousand_zero_error_trials() -> None:
    assert clopper_pearson_upper(0, 100, 0.95) > 1e-4
    assert clopper_pearson_upper(0, 30_000, 0.95) < 1e-4


def test_threshold_disables_confirmed_when_sample_cannot_prove_far() -> None:
    probabilities = np.asarray([0.95, 0.9, 0.3, 0.2], dtype=np.float64)
    labels = np.asarray([1, 1, 0, 0], dtype=np.int8)
    thresholds = calibrate_mode_thresholds(
        probabilities,
        labels,
        labels == 0,
        mode="short",
        confirmed_far_target=1e-4,
        confidence_level=0.95,
        provisional_tpr_target=0.95,
        certification_probabilities=np.asarray([0.96, 0.1]),
        certification_labels=np.asarray([1, 0], dtype=np.int8),
        certification_hard_negative_mask=np.asarray([False, True]),
    )
    assert thresholds["confirmed_enabled"] is False
    assert thresholds["confirmed_threshold"] is None
    assert thresholds["provisional_threshold"] == pytest.approx(0.96)
    assert thresholds["confirmed_false_accept_upper"] > 1e-4


def test_threshold_enables_only_a_proven_safe_region() -> None:
    positive = np.asarray([0.99, 0.98, 0.97], dtype=np.float64)
    negative = np.full(100, 0.1, dtype=np.float64)
    scores = np.r_[positive, negative]
    labels = np.r_[np.ones(len(positive), dtype=np.int8), np.zeros(len(negative), dtype=np.int8)]
    certification_scores = np.r_[[0.98, 0.96], np.full(30_000, 0.1)]
    certification_labels = np.r_[
        np.ones(2, dtype=np.int8), np.zeros(30_000, dtype=np.int8)
    ]
    thresholds = calibrate_mode_thresholds(
        scores,
        labels,
        labels == 0,
        mode="short",
        confirmed_far_target=1e-4,
        confidence_level=0.95,
        provisional_tpr_target=0.95,
        certification_probabilities=certification_scores,
        certification_labels=certification_labels,
        certification_hard_negative_mask=certification_labels == 0,
    )
    assert thresholds["confirmed_enabled"] is True
    assert thresholds["confirmed_false_accepts"] == 0
    assert thresholds["confirmed_false_accept_upper"] < 1e-4
    assert thresholds["confirmed_threshold"] >= thresholds["provisional_threshold"]


def test_train_selected_gate_is_disabled_when_calibration_certificate_fails() -> None:
    selection_scores = np.asarray([0.99, 0.98, 0.1, 0.1])
    selection_labels = np.asarray([1, 1, 0, 0], dtype=np.int8)
    certification_scores = np.r_[[0.99, 0.97], np.full(30_000, 0.995)]
    certification_labels = np.r_[
        np.ones(2, dtype=np.int8), np.zeros(30_000, dtype=np.int8)
    ]
    thresholds = calibrate_mode_thresholds(
        selection_scores,
        selection_labels,
        selection_labels == 0,
        mode="short",
        confirmed_far_target=1e-4,
        confidence_level=0.95,
        provisional_tpr_target=0.95,
        certification_probabilities=certification_scores,
        certification_labels=certification_labels,
        certification_hard_negative_mask=certification_labels == 0,
    )
    assert thresholds["selected_confirmed_threshold"] is not None
    assert thresholds["confirmed_enabled"] is False
    assert thresholds["confirmed_threshold"] is None
    assert thresholds["confirmed_disabled_reason"] == (
        "certification_false_accept_upper_exceeds_target"
    )


def test_scorer_fails_closed_for_missing_and_caps_high_overlap() -> None:
    artifact = _fitted_model()
    thresholds = _enabled_thresholds("short")
    scorer = LinkScorer(artifact, thresholds)
    missing = scorer.score_features(
        None, appearance_present=False, high_overlap=False
    )
    assert missing.decision == "reject"
    assert missing.probability is None
    overlap = scorer.score_features(
        {"similarity": 2.0, "quality": 1.0},
        appearance_present=True,
        high_overlap=True,
    )
    assert overlap.decision == "provisional"
    assert overlap.reason == "high_overlap_confirmation_blocked"


def test_long_threshold_jointly_requires_a_safe_margin() -> None:
    positive = np.asarray([0.99, 0.98], dtype=np.float64)
    negative = np.full(100, 0.1, dtype=np.float64)
    scores = np.r_[positive, negative]
    labels = np.r_[np.ones(2, dtype=np.int8), np.zeros(len(negative), dtype=np.int8)]
    margins = np.r_[np.asarray([0.5, 0.4]), np.full(len(negative), -0.2)]
    certification_scores = np.r_[[0.99, 0.97], np.full(30_000, 0.1)]
    certification_labels = np.r_[
        np.ones(2, dtype=np.int8), np.zeros(30_000, dtype=np.int8)
    ]
    certification_margins = np.r_[[0.45, 0.45], np.full(30_000, -0.2)]
    thresholds = calibrate_mode_thresholds(
        scores,
        labels,
        labels == 0,
        mode="long",
        confirmed_far_target=1e-4,
        confidence_level=0.95,
        provisional_tpr_target=0.95,
        candidate_margins=margins,
        certification_probabilities=certification_scores,
        certification_labels=certification_labels,
        certification_hard_negative_mask=certification_labels == 0,
        certification_candidate_margins=certification_margins,
    )
    assert thresholds["confirmed_enabled"] is True
    assert thresholds["appearance_margin_threshold"] is not None
    assert thresholds["confirmed_false_accepts"] == 0


def test_disabled_model_round_trip_rejects_without_fake_probability(tmp_path) -> None:
    artifact = disabled_link_model(
        ("similarity",), mode="long", random_seed=7, reason="insufficient_samples"
    )
    path = tmp_path / "link_model_long.joblib"
    save_link_model(path, artifact)
    loaded = load_link_model(path)
    scorer = LinkScorer(
        loaded,
        {
            "mode": "long",
            "model_enabled": False,
            "disabled_reason": "insufficient_samples",
            "confirmed_enabled": False,
            "confirmed_threshold": None,
            "provisional_threshold": None,
            "appearance_margin_threshold": None,
        },
    )
    result = scorer.score_features(
        {"similarity": 1.0},
        appearance_present=True,
        high_overlap=False,
        candidate_margin=1.0,
    )
    assert result.decision == "reject"
    assert result.probability is None
    assert result.reason == "insufficient_samples"


def test_scorer_rejects_uncertified_or_corrupted_confirmed_gate() -> None:
    artifact = _fitted_model()
    corrupted = _enabled_thresholds("short")
    corrupted["confirmed_threshold"] = -1.0
    with pytest.raises(ContractError, match="confirmed threshold is invalid"):
        LinkScorer(artifact, corrupted)

    uncertified = _enabled_thresholds("short")
    uncertified["certification_hard_negative_count"] = 10
    with pytest.raises(ContractError, match="Clopper-Pearson bound is invalid"):
        LinkScorer(artifact, uncertified)
