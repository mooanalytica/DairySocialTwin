"""Deterministic S03 link models and fail-closed decision thresholds."""

from __future__ import annotations

import math
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import joblib
import numpy as np
from scipy.stats import beta
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from cowtrack.config import ContractError

if TYPE_CHECKING:
    from cowtrack.linking.pseudo_pairs import GalleryAssessment


MODEL_ARTIFACT_VERSION = "1.0"


def _finite_matrix(values: np.ndarray, *, columns: int | None = None) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] == 0:
        raise ContractError("S03 feature matrix must have shape [N, D] with D > 0")
    if columns is not None and matrix.shape[1] != columns:
        raise ContractError("S03 feature matrix width differs from feature schema")
    if not np.all(np.isfinite(matrix)):
        raise ContractError("S03 feature matrix contains missing or non-finite values")
    return matrix


def _binary_labels(values: np.ndarray, *, rows: int) -> np.ndarray:
    labels = np.asarray(values)
    if labels.ndim != 1 or len(labels) != rows:
        raise ContractError("S03 labels must be one-dimensional and match features")
    if labels.dtype.kind == "b":
        result = labels.astype(np.int8, copy=False)
    elif labels.dtype.kind in "iu" and set(map(int, np.unique(labels))).issubset({0, 1}):
        result = labels.astype(np.int8, copy=False)
    else:
        raise ContractError("S03 labels must contain only 0 and 1")
    return result


@dataclass(frozen=True)
class LinkModelArtifact:
    """Serializable fitted model together with its exact ordered feature schema."""

    mode: str
    feature_names: tuple[str, ...]
    random_seed: int
    pipeline: Pipeline | None
    disabled_reason: str | None = None

    def feature_matrix(self, rows: Sequence[Mapping[str, float]]) -> np.ndarray:
        matrix = np.asarray(
            [[float(row[name]) for name in self.feature_names] for row in rows],
            dtype=np.float64,
        )
        return _finite_matrix(matrix, columns=len(self.feature_names))

    def probabilities(self, matrix: np.ndarray) -> np.ndarray:
        if self.pipeline is None:
            raise ContractError(f"S03 {self.mode} model is disabled: {self.disabled_reason}")
        checked = _finite_matrix(matrix, columns=len(self.feature_names))
        values = np.asarray(self.pipeline.predict_proba(checked)[:, 1], dtype=np.float64)
        if values.shape != (len(checked),) or not np.all(np.isfinite(values)):
            raise ContractError("S03 model returned invalid probabilities")
        return values

    def raw_scores(self, matrix: np.ndarray) -> np.ndarray:
        if self.pipeline is None:
            raise ContractError(f"S03 {self.mode} model is disabled: {self.disabled_reason}")
        checked = _finite_matrix(matrix, columns=len(self.feature_names))
        values = np.asarray(self.pipeline.decision_function(checked), dtype=np.float64)
        if values.shape != (len(checked),) or not np.all(np.isfinite(values)):
            raise ContractError("S03 model returned invalid raw scores")
        return values


def fit_link_model(
    features: np.ndarray,
    labels: np.ndarray,
    feature_names: Sequence[str],
    *,
    mode: str,
    random_seed: int,
    logistic_c: float,
    max_iter: int,
    minimum_per_class: int,
    tolerance: float = 1e-4,
    fit_intercept: bool = True,
) -> LinkModelArtifact:
    """Fit the fixed StandardScaler + LogisticRegression train-only pipeline."""

    names = tuple(map(str, feature_names))
    if mode not in {"short", "long"}:
        raise ContractError("S03 model mode must be short or long")
    if not names or len(set(names)) != len(names) or any(not name for name in names):
        raise ContractError("S03 feature names must be unique and non-empty")
    matrix = _finite_matrix(features, columns=len(names))
    target = _binary_labels(labels, rows=len(matrix))
    if isinstance(minimum_per_class, bool) or int(minimum_per_class) < 1:
        raise ContractError("S03 minimum_per_class must be positive")
    counts = np.bincount(target, minlength=2)
    if np.any(counts < int(minimum_per_class)):
        raise ContractError(
            f"S03 {mode} training samples are insufficient per class: {counts.tolist()}"
        )
    if not math.isfinite(float(logistic_c)) or float(logistic_c) <= 0.0:
        raise ContractError("S03 logistic C must be positive and finite")
    if isinstance(max_iter, bool) or int(max_iter) < 1:
        raise ContractError("S03 logistic max_iter must be positive")
    if not math.isfinite(float(tolerance)) or float(tolerance) <= 0.0:
        raise ContractError("S03 logistic tolerance must be positive and finite")
    if not isinstance(fit_intercept, bool):
        raise ContractError("S03 logistic fit_intercept must be boolean")

    pipeline = Pipeline(
        steps=[
            ("scale", StandardScaler()),
            (
                "logistic",
                LogisticRegression(
                    C=float(logistic_c),
                    class_weight="balanced",
                    fit_intercept=fit_intercept,
                    max_iter=int(max_iter),
                    random_state=int(random_seed),
                    solver="lbfgs",
                    tol=float(tolerance),
                ),
            ),
        ]
    )
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            pipeline.fit(matrix, target)
    except ConvergenceWarning as exc:
        raise ContractError(f"S03 {mode} LogisticRegression did not converge") from exc
    except ValueError as exc:
        raise ContractError(f"S03 {mode} LogisticRegression fitting failed: {exc}") from exc
    iterations = np.asarray(
        pipeline.named_steps["logistic"].n_iter_, dtype=np.int64
    )
    if iterations.shape != (1,) or np.any(iterations >= int(max_iter)):
        raise ContractError(f"S03 {mode} LogisticRegression did not converge")
    return LinkModelArtifact(
        mode=mode,
        feature_names=names,
        random_seed=int(random_seed),
        pipeline=pipeline,
    )


def disabled_link_model(
    feature_names: Sequence[str], *, mode: str, random_seed: int, reason: str
) -> LinkModelArtifact:
    names = tuple(map(str, feature_names))
    if mode not in {"short", "long"} or not names or len(set(names)) != len(names):
        raise ContractError("S03 disabled model contract is invalid")
    if not isinstance(reason, str) or not reason.strip():
        raise ContractError("S03 disabled model requires a reason")
    return LinkModelArtifact(
        mode=mode,
        feature_names=names,
        random_seed=int(random_seed),
        pipeline=None,
        disabled_reason=reason.strip(),
    )


def save_link_model(path: Path, artifact: LinkModelArtifact) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    payload = {
        "artifact_version": MODEL_ARTIFACT_VERSION,
        "mode": artifact.mode,
        "feature_names": artifact.feature_names,
        "random_seed": artifact.random_seed,
        "pipeline": artifact.pipeline,
        "disabled_reason": artifact.disabled_reason,
    }
    try:
        joblib.dump(payload, temporary, compress=3)
        os.replace(temporary, path)
    except (OSError, ValueError, TypeError) as exc:
        temporary.unlink(missing_ok=True)
        raise ContractError(f"cannot write S03 model artifact {path}: {exc}") from exc


def load_link_model(path: Path) -> LinkModelArtifact:
    path = path.resolve()
    if not path.is_file():
        raise ContractError(f"S03 model artifact does not exist: {path}")
    try:
        payload = joblib.load(path)
    except (OSError, ValueError, TypeError, EOFError) as exc:
        raise ContractError(f"cannot load S03 model artifact {path}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("artifact_version") != MODEL_ARTIFACT_VERSION:
        raise ContractError("S03 model artifact version mismatch")
    mode = payload.get("mode")
    names = payload.get("feature_names")
    pipeline = payload.get("pipeline")
    disabled_reason = payload.get("disabled_reason")
    seed = payload.get("random_seed")
    if mode not in {"short", "long"}:
        raise ContractError("S03 model artifact has invalid mode")
    if not isinstance(names, (tuple, list)) or not names:
        raise ContractError("S03 model artifact lacks feature names")
    if pipeline is None:
        if not isinstance(disabled_reason, str) or not disabled_reason.strip():
            raise ContractError("S03 disabled model artifact lacks a reason")
    elif not isinstance(pipeline, Pipeline) or disabled_reason is not None:
        raise ContractError("S03 model artifact has an invalid pipeline state")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ContractError("S03 model artifact has invalid random seed")
    return LinkModelArtifact(
        mode=str(mode),
        feature_names=tuple(map(str, names)),
        random_seed=int(seed),
        pipeline=pipeline,
        disabled_reason=disabled_reason,
    )


def clopper_pearson_upper(
    false_accepts: int, total_negatives: int, confidence_level: float
) -> float:
    """One-sided exact binomial upper confidence bound."""

    if (
        isinstance(false_accepts, bool)
        or isinstance(total_negatives, bool)
        or int(false_accepts) < 0
        or int(total_negatives) < 1
        or int(false_accepts) > int(total_negatives)
    ):
        raise ContractError("S03 false-accept counts are invalid")
    confidence = float(confidence_level)
    if not math.isfinite(confidence) or not 0.0 < confidence < 1.0:
        raise ContractError("S03 confidence_level must be in (0, 1)")
    failures = int(false_accepts)
    total = int(total_negatives)
    if failures == total:
        return 1.0
    upper = float(beta.ppf(confidence, failures + 1, total - failures))
    if not math.isfinite(upper) or not 0.0 <= upper <= 1.0:
        raise ContractError("S03 Clopper-Pearson bound is non-finite")
    return upper


def _recall_threshold(positive_scores: np.ndarray, target_tpr: float) -> float:
    positives = np.sort(np.asarray(positive_scores, dtype=np.float64))[::-1]
    if not len(positives) or not np.all(np.isfinite(positives)):
        raise ContractError("S03 calibration requires finite positive scores")
    target = float(target_tpr)
    if not 0.0 < target <= 1.0 or not math.isfinite(target):
        raise ContractError("S03 provisional TPR target must be in (0, 1]")
    retained = max(1, int(math.ceil(target * len(positives))))
    return float(positives[retained - 1])


def _safe_probability_threshold(
    positive_scores: np.ndarray,
    hard_negative_scores: np.ndarray,
    *,
    max_false_accepts: int,
    floor: float,
) -> float | None:
    negatives = np.sort(np.asarray(hard_negative_scores, dtype=np.float64))[::-1]
    if max_false_accepts < 0:
        return None
    if max_false_accepts >= len(negatives):
        candidate = float(floor)
    else:
        candidate = float(np.nextafter(negatives[max_false_accepts], math.inf))
    candidate = max(candidate, float(floor))
    if candidate > 1.0:
        return None
    if not np.any(np.asarray(positive_scores, dtype=np.float64) >= candidate):
        return None
    return candidate


def _joint_long_gate(
    scores: np.ndarray,
    labels: np.ndarray,
    hard_negative: np.ndarray,
    margins: np.ndarray,
    *,
    max_false_accepts: int,
    probability_floor: float,
) -> tuple[float, float] | None:
    """Choose a deterministic probability/margin gate with best positive recall.

    The exact safety statement is always evaluated over every held-out hard
    negative.  A bounded deterministic margin grid controls runtime without
    using audit data.
    """

    if max_false_accepts < 0:
        return None
    hard_margins = margins[hard_negative]
    positive_margins = margins[labels == 1]
    candidates = np.unique(np.concatenate((hard_margins, positive_margins, [0.0])))
    if len(candidates) > 512:
        indices = np.linspace(0, len(candidates) - 1, 512).round().astype(np.int64)
        candidates = candidates[np.unique(indices)]
    candidates = np.unique(np.maximum(candidates, 0.0))
    best: tuple[int, float, float] | None = None
    positive_scores = scores[labels == 1]
    for margin_threshold in candidates:
        eligible_hard = hard_negative & (margins >= margin_threshold)
        safe_probability = _safe_probability_threshold(
            positive_scores[margins[labels == 1] >= margin_threshold],
            scores[eligible_hard],
            max_false_accepts=max_false_accepts,
            floor=probability_floor,
        )
        if safe_probability is None:
            continue
        accepted_positive = int(
            np.count_nonzero(
                (labels == 1)
                & (scores >= safe_probability)
                & (margins >= margin_threshold)
            )
        )
        option = (accepted_positive, -float(safe_probability), -float(margin_threshold))
        if best is None or option > best:
            best = option
    if best is None or best[0] <= 0:
        return None
    return -best[1], -best[2]


def calibrate_mode_thresholds(
    probabilities: np.ndarray,
    labels: np.ndarray,
    hard_negative_mask: np.ndarray,
    *,
    mode: str,
    confirmed_far_target: float,
    confidence_level: float,
    provisional_tpr_target: float,
    candidate_margins: np.ndarray | None = None,
    certification_probabilities: np.ndarray,
    certification_labels: np.ndarray,
    certification_hard_negative_mask: np.ndarray,
    certification_candidate_margins: np.ndarray | None = None,
) -> dict[str, Any]:
    """Select and certify a gate on pre-separated calibration time blocks."""

    if mode not in {"short", "long"}:
        raise ContractError("S03 threshold mode must be short or long")
    scores = np.asarray(probabilities, dtype=np.float64)
    if scores.ndim != 1 or not len(scores) or not np.all(np.isfinite(scores)):
        raise ContractError("S03 calibration probabilities must be finite and non-empty")
    if np.any((scores < 0.0) | (scores > 1.0)):
        raise ContractError("S03 calibration probabilities must be in [0, 1]")
    target = _binary_labels(labels, rows=len(scores))
    hard = np.asarray(hard_negative_mask, dtype=np.bool_)
    if hard.shape != scores.shape or np.any(hard & (target != 0)):
        raise ContractError("S03 hard-negative mask must select only negative rows")
    positives = scores[target == 1]
    hard_scores = scores[hard]
    if not len(positives) or not len(hard_scores):
        raise ContractError("S03 threshold selection requires positives and hard negatives")
    certification_scores = np.asarray(
        certification_probabilities, dtype=np.float64
    )
    if (
        certification_scores.ndim != 1
        or not len(certification_scores)
        or not np.all(np.isfinite(certification_scores))
        or np.any((certification_scores < 0.0) | (certification_scores > 1.0))
    ):
        raise ContractError(
            "S03 certification probabilities must be finite, non-empty, and in [0, 1]"
        )
    certification_target = _binary_labels(
        certification_labels, rows=len(certification_scores)
    )
    certification_hard = np.asarray(
        certification_hard_negative_mask, dtype=np.bool_
    )
    if certification_hard.shape != certification_scores.shape or np.any(
        certification_hard & (certification_target != 0)
    ):
        raise ContractError(
            "S03 certification hard-negative mask must select only negative rows"
        )
    certification_positives = certification_scores[certification_target == 1]
    certification_hard_scores = certification_scores[certification_hard]
    if not len(certification_positives) or not len(certification_hard_scores):
        raise ContractError("S03 certification requires positives and hard negatives")
    far = float(confirmed_far_target)
    if not math.isfinite(far) or not 0.0 < far < 1.0:
        raise ContractError("S03 confirmed FAR target must be in (0, 1)")
    selection_floor = _recall_threshold(positives, provisional_tpr_target)
    confirmed: float | None = None
    margin_threshold: float | None = None
    certification_margins: np.ndarray | None = None
    if mode == "long":
        if candidate_margins is None or certification_candidate_margins is None:
            raise ContractError(
                "S03 long threshold selection/certification requires candidate margins"
            )
        margins = np.asarray(candidate_margins, dtype=np.float64)
        if margins.shape != scores.shape or not np.all(np.isfinite(margins)):
            raise ContractError(
                "S03 long selection candidate margins must be finite and aligned"
            )
        certification_margins = np.asarray(
            certification_candidate_margins, dtype=np.float64
        )
        if certification_margins.shape != certification_scores.shape or not np.all(
            np.isfinite(certification_margins)
        ):
            raise ContractError(
                "S03 long certification candidate margins must be finite and aligned"
            )
        joint = _joint_long_gate(
            scores,
            target,
            hard,
            margins,
            max_false_accepts=0,
            probability_floor=selection_floor,
        )
        if joint is not None:
            confirmed, margin_threshold = joint
    else:
        confirmed = _safe_probability_threshold(
            positives,
            hard_scores,
            max_false_accepts=0,
            floor=selection_floor,
        )

    selected_confirmed = confirmed
    selected_margin = margin_threshold
    provisional = _recall_threshold(
        certification_positives, provisional_tpr_target
    )
    disabled_reason: str | None = None
    if confirmed is not None and confirmed < provisional:
        confirmed = None
        margin_threshold = None
        disabled_reason = "train_selected_gate_below_calibration_provisional"
    if confirmed is None:
        false_accepts = 0
        true_accepts = 0
        upper = clopper_pearson_upper(
            0, len(certification_hard_scores), confidence_level
        )
        if disabled_reason is None:
            disabled_reason = "no_threshold_selection_candidate"
    else:
        accepted = certification_hard_scores >= confirmed
        if margin_threshold is not None:
            assert certification_margins is not None
            accepted &= certification_margins[certification_hard] >= margin_threshold
        false_accepts = int(np.count_nonzero(accepted))
        upper = clopper_pearson_upper(
            false_accepts, len(certification_hard_scores), confidence_level
        )
        accepted_positive = certification_positives >= confirmed
        if margin_threshold is not None:
            assert certification_margins is not None
            accepted_positive &= (
                certification_margins[certification_target == 1] >= margin_threshold
            )
        true_accepts = int(np.count_nonzero(accepted_positive))
        if upper > far:
            confirmed = None
            margin_threshold = None
            disabled_reason = "certification_false_accept_upper_exceeds_target"
        elif true_accepts == 0:
            confirmed = None
            margin_threshold = None
            disabled_reason = "certification_accepts_no_positive"

    return {
        "mode": mode,
        "confirmed_enabled": confirmed is not None,
        "confirmed_threshold": confirmed,
        "provisional_threshold": provisional,
        "appearance_margin_threshold": margin_threshold,
        "confirmed_far_target": far,
        "confidence_level": float(confidence_level),
        "provisional_tpr_target": float(provisional_tpr_target),
        "calibration_positive_count": int(len(certification_positives)),
        "calibration_hard_negative_count": int(len(certification_hard_scores)),
        "threshold_selection_positive_count": int(len(positives)),
        "threshold_selection_hard_negative_count": int(len(hard_scores)),
        "certification_positive_count": int(len(certification_positives)),
        "certification_hard_negative_count": int(len(certification_hard_scores)),
        "certification_true_accepts": true_accepts,
        "confirmed_false_accepts": false_accepts,
        "confirmed_false_accept_upper": upper,
        "confirmed_certified_independently": True,
        "confirmed_gate_selection_split": "calibration_threshold_selection",
        "confirmed_gate_certification_split": "calibration_certification",
        "provisional_threshold_source": "calibration_positive_tpr",
        "confirmed_disabled_reason": disabled_reason,
        "selected_confirmed_threshold": selected_confirmed,
        "selected_appearance_margin_threshold": selected_margin,
    }


@dataclass(frozen=True)
class LinkResult:
    probability: float | None
    raw_score: float | None
    decision: str
    feature_values: dict[str, float] | None
    reason: str | None = None
    appearance_present: bool = True
    high_overlap: bool = False
    source_gallery: "GalleryAssessment | None" = None
    target_gallery: "GalleryAssessment | None" = None


class LinkScorer:
    """Apply a fitted model and serialized thresholds with reliability gates."""

    def __init__(self, model: LinkModelArtifact, thresholds: Mapping[str, Any]) -> None:
        if not isinstance(thresholds, Mapping):
            raise ContractError("S03 thresholds must be a mapping")
        if thresholds.get("mode") != model.mode:
            raise ContractError("S03 model/threshold mode mismatch")
        payload = dict(thresholds)
        model_enabled = payload.get("model_enabled")
        if model.pipeline is None:
            if (
                model_enabled is not False
                or payload.get("disabled_reason") != model.disabled_reason
                or payload.get("confirmed_enabled") is not False
                or payload.get("confirmed_threshold") is not None
                or payload.get("provisional_threshold") is not None
                or payload.get("appearance_margin_threshold") is not None
            ):
                raise ContractError("S03 disabled model/threshold state is inconsistent")
            self.model = model
            self.thresholds = payload
            return
        if model_enabled is not True or payload.get("disabled_reason") is not None:
            raise ContractError("S03 enabled model/threshold state is inconsistent")
        provisional = payload.get("provisional_threshold")
        if (
            isinstance(provisional, bool)
            or not isinstance(provisional, (int, float))
            or not math.isfinite(float(provisional))
            or not 0.0 <= float(provisional) <= 1.0
        ):
            raise ContractError("S03 thresholds lack a valid provisional threshold")
        confirmed_enabled = payload.get("confirmed_enabled")
        if not isinstance(confirmed_enabled, bool):
            raise ContractError("S03 confirmed_enabled must be boolean")
        confirmed = payload.get("confirmed_threshold")
        margin = payload.get("appearance_margin_threshold")
        if confirmed_enabled:
            if (
                isinstance(confirmed, bool)
                or not isinstance(confirmed, (int, float))
                or not math.isfinite(float(confirmed))
                or not float(provisional) <= float(confirmed) <= 1.0
            ):
                raise ContractError("S03 confirmed threshold is invalid")
            if model.mode == "long":
                if (
                    isinstance(margin, bool)
                    or not isinstance(margin, (int, float))
                    or not math.isfinite(float(margin))
                    or float(margin) < 0.0
                ):
                    raise ContractError("S03 long confirmed margin is invalid")
            elif margin is not None:
                raise ContractError("S03 short thresholds cannot contain a margin gate")
        elif confirmed is not None or margin is not None:
            raise ContractError("S03 disabled confirmation must not expose a gate")

        far = payload.get("confirmed_far_target")
        confidence = payload.get("confidence_level")
        if (
            isinstance(far, bool)
            or not isinstance(far, (int, float))
            or not math.isfinite(float(far))
            or not 0.0 < float(far) < 1.0
            or isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not math.isfinite(float(confidence))
            or not 0.0 < float(confidence) < 1.0
        ):
            raise ContractError("S03 confirmed FAR/confidence contract is invalid")
        false_accepts = payload.get("confirmed_false_accepts")
        certification_negatives = payload.get("certification_hard_negative_count")
        if (
            isinstance(false_accepts, bool)
            or not isinstance(false_accepts, (int, np.integer))
            or isinstance(certification_negatives, bool)
            or not isinstance(certification_negatives, (int, np.integer))
            or int(certification_negatives) < 1
            or int(false_accepts) < 0
            or int(false_accepts) > int(certification_negatives)
            or payload.get("confirmed_certified_independently") is not True
        ):
            raise ContractError("S03 independent certification counts are invalid")
        recorded_upper = payload.get("confirmed_false_accept_upper")
        expected_upper = clopper_pearson_upper(
            int(false_accepts), int(certification_negatives), float(confidence)
        )
        if (
            isinstance(recorded_upper, bool)
            or not isinstance(recorded_upper, (int, float))
            or not math.isfinite(float(recorded_upper))
            or not math.isclose(
                float(recorded_upper), expected_upper, rel_tol=0.0, abs_tol=1e-15
            )
        ):
            raise ContractError("S03 recorded Clopper-Pearson bound is invalid")
        if confirmed_enabled and expected_upper > float(far):
            raise ContractError("S03 confirmed gate lacks the required FAR certificate")
        self.model = model
        self.thresholds = payload

    def score_features(
        self,
        feature_values: Mapping[str, float] | None,
        *,
        appearance_present: bool,
        high_overlap: bool,
        candidate_margin: float | None = None,
    ) -> LinkResult:
        if not appearance_present:
            return LinkResult(
                None,
                None,
                "reject",
                None,
                "appearance_missing",
                appearance_present=False,
                high_overlap=high_overlap,
            )
        if self.model.pipeline is None:
            return LinkResult(
                None,
                None,
                "reject",
                None,
                self.model.disabled_reason or "model_disabled",
                appearance_present=True,
                high_overlap=high_overlap,
            )
        if feature_values is None:
            raise ContractError("S03 present appearance requires feature values")
        if tuple(feature_values) != self.model.feature_names:
            raise ContractError("S03 score feature schema differs from runtime model")
        try:
            row = {name: float(feature_values[name]) for name in self.model.feature_names}
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ContractError("S03 score features are not finite numbers") from exc
        if not all(math.isfinite(value) for value in row.values()):
            raise ContractError("S03 score features contain non-finite values")
        matrix = self.model.feature_matrix([row])
        probability = float(self.model.probabilities(matrix)[0])
        raw_score = float(self.model.raw_scores(matrix)[0])
        provisional = float(self.thresholds["provisional_threshold"])
        confirmed_enabled = bool(self.thresholds.get("confirmed_enabled", False))
        confirmed_value = self.thresholds.get("confirmed_threshold")
        confirmed = float(confirmed_value) if confirmed_value is not None else None

        decision = "reject"
        reason: str | None = None
        if probability >= provisional:
            decision = "provisional"
        if confirmed_enabled and confirmed is not None and probability >= confirmed:
            margin_ok = True
            required_margin = self.thresholds.get("appearance_margin_threshold")
            if self.model.mode == "long":
                margin_ok = (
                    required_margin is not None
                    and candidate_margin is not None
                    and math.isfinite(float(candidate_margin))
                    and float(candidate_margin) >= float(required_margin)
                )
            if margin_ok and not high_overlap:
                decision = "confirmed"
            elif high_overlap:
                reason = "high_overlap_confirmation_blocked"
            elif not margin_ok:
                reason = "appearance_margin_not_met"
        return LinkResult(
            probability,
            raw_score,
            decision,
            row,
            reason,
            appearance_present=True,
            high_overlap=high_overlap,
        )


__all__ = [
    "LinkModelArtifact",
    "LinkResult",
    "LinkScorer",
    "calibrate_mode_thresholds",
    "clopper_pearson_upper",
    "disabled_link_model",
    "fit_link_model",
    "load_link_model",
    "save_link_model",
]
