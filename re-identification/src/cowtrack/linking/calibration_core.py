"""Public, split-safe calibration core shared by S03 and S05.

The caller owns pair mining and partition assignment.  This module accepts
canonical rows that have already been assigned to ``train``,
``threshold_selection``, ``certification``, or ``audit``.  It deliberately
keeps the four roles separate:

* only ``train`` rows fit the deterministic logistic model;
* only ``threshold_selection`` rows select provisional and candidate-confirmed
  probability/margin gates;
* only ``certification`` hard-negative groups certify the frozen predicate;
* ``audit`` is report-only.

Counts used for minimum-sample checks and Clopper--Pearson certification are
unique ``candidate_group_id`` values within a label and partition.  Repeated,
correlated rows therefore cannot inflate the effective certification sample.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from cowtrack.config import ContractError
from cowtrack.linking.features import LONG_FEATURE_SCHEMA
from cowtrack.linking.model import (
    LinkModelArtifact,
    clopper_pearson_upper,
    disabled_link_model,
    fit_link_model,
)


CALIBRATION_PARTITIONS = (
    "train",
    "threshold_selection",
    "certification",
    "audit",
)

AUDIT_EVIDENCE_PER_CLIP_STRICT = "per_clip_strict"
AUDIT_EVIDENCE_AGGREGATE_OBSERVED_ROWS = "aggregate_observed_rows"
_AUDIT_EVIDENCE_SCOPES = {
    AUDIT_EVIDENCE_PER_CLIP_STRICT,
    AUDIT_EVIDENCE_AGGREGATE_OBSERVED_ROWS,
}


@dataclass(frozen=True)
class CalibrationMinimums:
    """Independent-group minimums; these are never silently relaxed."""

    train_positive_groups: int = 100
    train_hard_negative_groups: int = 100
    selection_positive_groups: int = 25
    selection_hard_negative_groups: int = 25
    certification_positive_groups: int = 25
    certification_hard_negative_groups: int = 25
    audit_positive_groups: int = 50
    audit_hard_negative_groups: int = 50
    parent_groups_per_partition: int = 10
    audit_groups_per_class_per_clip: int = 10

    def __post_init__(self) -> None:
        for name, value in vars(self).items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ContractError(f"calibration minimum {name} must be positive")

    def class_groups(self, partition: str) -> tuple[int, int]:
        if partition == "train":
            return self.train_positive_groups, self.train_hard_negative_groups
        if partition == "threshold_selection":
            return self.selection_positive_groups, self.selection_hard_negative_groups
        if partition == "certification":
            return (
                self.certification_positive_groups,
                self.certification_hard_negative_groups,
            )
        if partition == "audit":
            return self.audit_positive_groups, self.audit_hard_negative_groups
        raise ContractError(f"unknown calibration partition: {partition}")


@dataclass(frozen=True)
class LongCalibrationSettings:
    """Fixed estimator, threshold, and evidence settings for long calibration."""

    random_seed: int = 20260710
    logistic_c: float = 1.0
    max_iter: int = 2_000
    tolerance: float = 1e-6
    fit_intercept: bool = True
    confirmed_far_target: float = 1e-4
    confidence_level: float = 0.95
    provisional_tpr_target: float = 0.95
    margin_feature_name: str = "prototype_cosine_max"
    minimums: CalibrationMinimums = field(default_factory=CalibrationMinimums)
    audit_evidence_scope: str = AUDIT_EVIDENCE_PER_CLIP_STRICT
    expected_clip_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.random_seed, bool) or not isinstance(self.random_seed, int):
            raise ContractError("long calibration random_seed must be an integer")
        for name in ("logistic_c", "tolerance"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ContractError(f"long calibration {name} must be positive and finite")
        if isinstance(self.max_iter, bool) or not isinstance(self.max_iter, int):
            raise ContractError("long calibration max_iter must be an integer")
        if self.max_iter < 1:
            raise ContractError("long calibration max_iter must be positive")
        if not isinstance(self.fit_intercept, bool):
            raise ContractError("long calibration fit_intercept must be boolean")
        for name, upper in (
            ("confirmed_far_target", 1.0),
            ("confidence_level", 1.0),
            ("provisional_tpr_target", 1.0),
        ):
            raw = getattr(self, name)
            if isinstance(raw, bool) or not isinstance(
                raw, (int, float, np.integer, np.floating)
            ):
                raise ContractError(f"long calibration {name} must be numeric")
            value = float(raw)
            if not math.isfinite(value) or not (0.0 < value <= upper):
                interval = "(0, 1]" if name == "provisional_tpr_target" else "(0, 1)"
                raise ContractError(f"long calibration {name} must be in {interval}")
            if name != "provisional_tpr_target" and value == upper:
                raise ContractError(f"long calibration {name} must be in (0, 1)")
        if not isinstance(self.margin_feature_name, str) or not self.margin_feature_name:
            raise ContractError("long calibration margin_feature_name must be non-empty")
        if not isinstance(self.minimums, CalibrationMinimums):
            raise ContractError("long calibration minimums have the wrong type")
        if self.audit_evidence_scope not in _AUDIT_EVIDENCE_SCOPES:
            raise ContractError(
                "long calibration audit_evidence_scope is unsupported: "
                f"{self.audit_evidence_scope!r}"
            )
        clips = self.expected_clip_ids
        if (
            not isinstance(clips, tuple)
            or any(not isinstance(clip_id, str) or not clip_id for clip_id in clips)
            or len(set(clips)) != len(clips)
        ):
            raise ContractError(
                "long calibration expected_clip_ids must be unique non-empty strings"
            )
        if (
            self.audit_evidence_scope == AUDIT_EVIDENCE_AGGREGATE_OBSERVED_ROWS
            and not clips
        ):
            raise ContractError(
                "aggregate observed-row audit evidence requires expected_clip_ids"
            )


@dataclass(frozen=True)
class SelectedLongGates:
    """Gates frozen using threshold-selection rows only."""

    provisional_threshold: float
    confirmed_probability_threshold: float | None
    appearance_margin_threshold: float | None
    positive_rows: int
    hard_negative_rows: int


@dataclass(frozen=True)
class LongGateCertification:
    """Independent certification of one already-frozen predicate."""

    certified: bool
    false_accepts: int
    hard_negative_groups: int
    false_accept_upper: float
    positive_accepts: int
    positive_groups: int


@dataclass(frozen=True)
class LongCalibrationResult:
    """Fitted/disabled artifact plus JSON- and Parquet-friendly results."""

    model: LinkModelArtifact
    thresholds: dict[str, Any]
    scored_rows: tuple[dict[str, Any], ...]
    stratified_counts: dict[str, Any]
    audit_metrics: dict[str, Any]
    audit_adequacy: dict[str, Any]

    @property
    def model_enabled(self) -> bool:
        return self.model.pipeline is not None


def _feature_schema(feature_names: Sequence[str]) -> tuple[str, ...]:
    names = tuple(map(str, feature_names))
    if not names or len(set(names)) != len(names) or any(not name for name in names):
        raise ContractError("long calibration feature names must be unique and non-empty")
    return names


def _sort_token(value: Any) -> tuple[Any, ...]:
    """Return a total, deterministic token for ordinary row metadata."""

    if isinstance(value, np.generic):
        value = value.item()
    if value is None:
        return ("none",)
    if isinstance(value, bool):
        return ("bool", int(value))
    if isinstance(value, int):
        return ("int", value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return ("float", repr(value))
        return ("float", value.hex())
    if isinstance(value, str):
        return ("str", value)
    if isinstance(value, Mapping):
        return (
            "mapping",
            tuple(sorted((str(key), _sort_token(item)) for key, item in value.items())),
        )
    if isinstance(value, (tuple, list)):
        return ("sequence", tuple(_sort_token(item) for item in value))
    return ("other", type(value).__module__, type(value).__qualname__, repr(value))


def _canonical_rows(
    rows: Sequence[Mapping[str, Any]], names: tuple[str, ...]
) -> list[dict[str, Any]]:
    if isinstance(rows, (str, bytes)):
        raise ContractError("long calibration rows must be a sequence of mappings")
    normalized: list[dict[str, Any]] = []
    for index, source in enumerate(rows):
        if not isinstance(source, Mapping):
            raise ContractError(f"long calibration row {index} must be a mapping")
        missing = {
            "partition",
            "label",
            "candidate_group_id",
            "parent_stable_id",
            "source_clip_id",
            "target_clip_id",
            "features",
        } - set(source)
        if missing:
            raise ContractError(
                f"long calibration row {index} lacks fields: {sorted(missing)}"
            )
        item = dict(source)
        partition = item["partition"]
        if partition not in CALIBRATION_PARTITIONS:
            raise ContractError(
                f"long calibration row {index} has invalid partition: {partition!r}"
            )
        label = item["label"]
        if not isinstance(label, (bool, np.bool_)):
            raise ContractError(f"long calibration row {index} label must be boolean")
        item["label"] = bool(label)
        group_id = item["candidate_group_id"]
        if not isinstance(group_id, str) or not group_id.strip():
            raise ContractError(
                f"long calibration row {index} candidate_group_id must be non-empty"
            )
        item["candidate_group_id"] = group_id.strip()
        parent = item["parent_stable_id"]
        if (
            isinstance(parent, bool)
            or not isinstance(parent, (int, np.integer))
            or int(parent) < 0
        ):
            raise ContractError(
                f"long calibration row {index} parent_stable_id must be non-negative"
            )
        item["parent_stable_id"] = int(parent)
        for field_name in ("source_clip_id", "target_clip_id"):
            value = item[field_name]
            if not isinstance(value, str) or not value.strip():
                raise ContractError(
                    f"long calibration row {index} {field_name} must be non-empty"
                )
            item[field_name] = value.strip()
        features = item["features"]
        if not isinstance(features, Mapping) or tuple(features) != names:
            actual = tuple(features) if isinstance(features, Mapping) else None
            raise ContractError(
                "long calibration row feature schema differs from fixed ordered schema: "
                f"expected={names!r}, actual={actual!r}"
            )
        checked_features: dict[str, float] = {}
        for name in names:
            value = features[name]
            if isinstance(value, bool) or not isinstance(
                value, (int, float, np.integer, np.floating)
            ):
                raise ContractError(
                    f"long calibration feature {name!r} must be numeric"
                )
            number = float(value)
            if not math.isfinite(number):
                raise ContractError(
                    f"long calibration feature {name!r} is non-finite"
                )
            checked_features[name] = number
        item["features"] = checked_features
        if "appearance_present" in item and item["appearance_present"] is not True:
            raise ContractError(
                "long calibration rows with missing appearance cannot be calibrated"
            )
        if "high_overlap" in item and not isinstance(
            item["high_overlap"], (bool, np.bool_)
        ):
            raise ContractError("long calibration high_overlap must be boolean")
        item["high_overlap"] = bool(item.get("high_overlap", False))
        if "candidate_margin" in item and item["candidate_margin"] is not None:
            margin = item["candidate_margin"]
            if isinstance(margin, bool) or not isinstance(
                margin, (int, float, np.integer, np.floating)
            ):
                raise ContractError("long calibration candidate_margin must be numeric")
            item["candidate_margin"] = float(margin)
            if not math.isfinite(item["candidate_margin"]):
                raise ContractError("long calibration candidate_margin is non-finite")
        normalized.append(item)

    normalized.sort(
        key=lambda row: (
            CALIBRATION_PARTITIONS.index(str(row["partition"])),
            str(row["candidate_group_id"]),
            int(row["parent_stable_id"]),
            str(row["source_clip_id"]),
            str(row["target_clip_id"]),
            not bool(row["label"]),
            tuple(float(row["features"][name]) for name in names),
            _sort_token(row),
        )
    )
    parent_partitions: dict[int, set[str]] = defaultdict(set)
    candidate_partitions: dict[str, set[str]] = defaultdict(set)
    for row in normalized:
        partition = str(row["partition"])
        parent_partitions[int(row["parent_stable_id"])].add(partition)
        candidate_partitions[str(row["candidate_group_id"])].add(partition)
    crossing_parent = sorted(
        parent for parent, partitions in parent_partitions.items() if len(partitions) != 1
    )
    if crossing_parent:
        raise ContractError(
            "long calibration parent_stable_id crosses partitions: "
            f"{crossing_parent[:10]}"
        )
    crossing_group = sorted(
        group for group, partitions in candidate_partitions.items() if len(partitions) != 1
    )
    if crossing_group:
        raise ContractError(
            "long calibration candidate_group_id crosses partitions: "
            f"{crossing_group[:10]}"
        )
    return normalized


def _selected(
    rows: Sequence[Mapping[str, Any]], partition: str
) -> list[Mapping[str, Any]]:
    return [row for row in rows if row["partition"] == partition]


def _group_count(rows: Sequence[Mapping[str, Any]], label: bool) -> int:
    return len(
        {
            str(row["candidate_group_id"])
            for row in rows
            if bool(row["label"]) is label
        }
    )


def stratified_group_counts(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Return row, independent-group, parent, clip, and stratum counts."""

    result: dict[str, Any] = {
        "counting_unit": "unique_candidate_group_id_within_partition_and_label",
        "total_rows": int(len(rows)),
        "partitions": {},
        "dimensions": {},
    }
    all_clips = sorted(
        {str(row["source_clip_id"]) for row in rows}
        | {str(row["target_clip_id"]) for row in rows}
    )
    result["clip_ids"] = all_clips
    for partition in CALIBRATION_PARTITIONS:
        part = _selected(rows, partition)
        per_clip: dict[str, Any] = {}
        for clip_id in all_clips:
            clip_rows = [
                row
                for row in part
                if row["source_clip_id"] == clip_id or row["target_clip_id"] == clip_id
            ]
            per_clip[clip_id] = {
                "rows": len(clip_rows),
                "positive_groups": _group_count(clip_rows, True),
                "hard_negative_groups": _group_count(clip_rows, False),
            }
        result["partitions"][partition] = {
            "rows": len(part),
            "positive_rows": sum(bool(row["label"]) for row in part),
            "hard_negative_rows": sum(not bool(row["label"]) for row in part),
            "positive_groups": _group_count(part, True),
            "hard_negative_groups": _group_count(part, False),
            "parent_groups": len({int(row["parent_stable_id"]) for row in part}),
            "per_clip": per_clip,
        }
    for dimension in (
        "partition",
        "source_clip_id",
        "target_clip_id",
    ):
        result["dimensions"][dimension] = dict(
            sorted(Counter(str(row[dimension]) for row in rows).items())
        )
    result["dimensions"]["label"] = {
        "positive": sum(bool(row["label"]) for row in rows),
        "hard_negative": sum(not bool(row["label"]) for row in rows),
    }
    if all("stratum" in row for row in rows):
        result["dimensions"]["stratum"] = dict(
            sorted(Counter(str(row["stratum"]) for row in rows).items())
        )
    return result


def audit_adequacy_report(
    counts: Mapping[str, Any],
    minimums: CalibrationMinimums,
    *,
    evidence_scope: str = AUDIT_EVIDENCE_PER_CLIP_STRICT,
    expected_clip_ids: Sequence[str] = (),
) -> dict[str, Any]:
    """Describe audit evidence without treating an unobserved frame as absence.

    ``aggregate_observed_rows`` keeps the configured per-clip number as a
    clip-count-scaled *aggregate* evidence budget.  Individual clips remain
    visible as warnings, but a clip with no eligible observed rows does not by
    itself disable a globally calibrated model.
    """

    if not isinstance(minimums, CalibrationMinimums):
        raise ContractError("audit adequacy minimums have the wrong type")
    if evidence_scope not in _AUDIT_EVIDENCE_SCOPES:
        raise ContractError(f"unsupported audit evidence scope: {evidence_scope!r}")
    clips = tuple(expected_clip_ids)
    if not clips:
        raw_clips = counts.get("clip_ids")
        if not isinstance(raw_clips, (tuple, list)):
            raise ContractError("audit adequacy counts lack clip_ids")
        clips = tuple(map(str, raw_clips))
    if (
        any(not isinstance(clip_id, str) or not clip_id for clip_id in clips)
        or len(set(clips)) != len(clips)
    ):
        raise ContractError("audit adequacy clip IDs must be unique non-empty strings")
    if evidence_scope == AUDIT_EVIDENCE_AGGREGATE_OBSERVED_ROWS and not clips:
        raise ContractError("aggregate observed-row audit evidence requires clip IDs")
    observed_clips = counts.get("clip_ids")
    if not isinstance(observed_clips, (tuple, list)):
        raise ContractError("audit adequacy counts lack clip_ids")
    unknown = sorted(set(map(str, observed_clips)) - set(clips))
    if unknown:
        raise ContractError(
            f"audit evidence references clips outside expected_clip_ids: {unknown}"
        )
    try:
        audit = counts["partitions"]["audit"]
        per_clip_counts = audit["per_clip"]
        observed_positive = int(audit["positive_groups"])
        observed_negative = int(audit["hard_negative_groups"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ContractError("audit adequacy counts are malformed") from exc
    if not isinstance(per_clip_counts, Mapping):
        raise ContractError("audit adequacy per-clip counts are malformed")

    nominal = minimums.audit_groups_per_class_per_clip
    scaled = nominal * len(clips)
    required_positive = max(minimums.audit_positive_groups, scaled)
    required_negative = max(minimums.audit_hard_negative_groups, scaled)
    per_clip: dict[str, Any] = {}
    warnings: list[str] = []
    strict_per_clip_sufficient = True
    for clip_id in clips:
        values = per_clip_counts.get(clip_id, {})
        try:
            positive = int(values.get("positive_groups", 0))
            negative = int(values.get("hard_negative_groups", 0))
        except (AttributeError, TypeError, ValueError) as exc:
            raise ContractError(
                f"audit adequacy counts for clip {clip_id!r} are malformed"
            ) from exc
        sufficient = positive >= nominal and negative >= nominal
        warning = None
        if not sufficient:
            strict_per_clip_sufficient = False
            warning = (
                f"audit_clip_{clip_id}_groups_below_nominal:"
                f"positive={positive}/{nominal},"
                f"hard_negative={negative}/{nominal}"
            )
            warnings.append(warning)
        per_clip[clip_id] = {
            "positive_groups": positive,
            "hard_negative_groups": negative,
            "nominal_minimum_per_class": nominal,
            "meets_nominal_minimum": sufficient,
            "warning": warning,
        }

    aggregate_sufficient = (
        observed_positive >= required_positive
        and observed_negative >= required_negative
    )
    sufficient = (
        strict_per_clip_sufficient
        if evidence_scope == AUDIT_EVIDENCE_PER_CLIP_STRICT
        else aggregate_sufficient
    )
    return {
        "evidence_scope": evidence_scope,
        "counting_unit": "unique_candidate_group_id_within_partition_and_label",
        "expected_clip_count": len(clips),
        "expected_clip_ids": list(clips),
        "configured_global_minimum_groups": {
            "positive": minimums.audit_positive_groups,
            "hard_negative": minimums.audit_hard_negative_groups,
        },
        "nominal_groups_per_class_per_clip": nominal,
        "clip_count_scaled_minimum_per_class": scaled,
        "required_aggregate_groups": {
            "positive": required_positive,
            "hard_negative": required_negative,
        },
        "observed_aggregate_groups": {
            "positive": observed_positive,
            "hard_negative": observed_negative,
        },
        "aggregate_sufficient": aggregate_sufficient,
        "strict_per_clip_sufficient": strict_per_clip_sufficient,
        "sufficient_for_selected_scope": sufficient,
        "per_clip": per_clip,
        "warnings": warnings,
    }


def calibration_minimum_failure_reason(
    counts: Mapping[str, Any],
    minimums: CalibrationMinimums,
    *,
    evidence_scope: str = AUDIT_EVIDENCE_PER_CLIP_STRICT,
    expected_clip_ids: Sequence[str] = (),
) -> str | None:
    """Return the first deterministic model-adequacy failure, if any."""

    for partition in CALIBRATION_PARTITIONS:
        values = counts["partitions"][partition]
        required_positive, required_negative = minimums.class_groups(partition)
        if (
            values["positive_groups"] < required_positive
            or values["hard_negative_groups"] < required_negative
        ):
            return (
                f"{partition}_groups_below_minimum:"
                f"positive={values['positive_groups']}/{required_positive},"
                f"hard_negative={values['hard_negative_groups']}/{required_negative}"
            )
        if values["parent_groups"] < minimums.parent_groups_per_partition:
            return (
                f"{partition}_parent_groups_below_minimum:"
                f"{values['parent_groups']}/{minimums.parent_groups_per_partition}"
            )
    adequacy = audit_adequacy_report(
        counts,
        minimums,
        evidence_scope=evidence_scope,
        expected_clip_ids=expected_clip_ids,
    )
    if evidence_scope == AUDIT_EVIDENCE_PER_CLIP_STRICT:
        if adequacy["warnings"]:
            return str(adequacy["warnings"][0]).replace(
                "_below_nominal:", "_below_minimum:"
            )
    elif not adequacy["aggregate_sufficient"]:
        observed = adequacy["observed_aggregate_groups"]
        required = adequacy["required_aggregate_groups"]
        return (
            "audit_aggregate_groups_below_scaled_minimum:"
            f"positive={observed['positive']}/{required['positive']},"
            f"hard_negative={observed['hard_negative']}/"
            f"{required['hard_negative']},"
            f"clip_count={adequacy['expected_clip_count']}"
        )
    return None


def _recall_threshold(scores: np.ndarray, target_tpr: float) -> float:
    ordered = np.sort(np.asarray(scores, dtype=np.float64))[::-1]
    if not len(ordered) or not np.all(np.isfinite(ordered)):
        raise ContractError("threshold selection requires finite positive scores")
    retained = max(1, int(math.ceil(float(target_tpr) * len(ordered))))
    return float(ordered[retained - 1])


def select_long_gates(
    probabilities: np.ndarray,
    labels: np.ndarray,
    candidate_margins: np.ndarray,
    *,
    provisional_tpr_target: float,
    reliability_eligible: np.ndarray | None = None,
) -> SelectedLongGates:
    """Select all gates from threshold-selection rows, without certification data."""

    scores = np.asarray(probabilities, dtype=np.float64)
    target = np.asarray(labels)
    margins = np.asarray(candidate_margins, dtype=np.float64)
    if (
        scores.ndim != 1
        or not len(scores)
        or margins.shape != scores.shape
        or not np.all(np.isfinite(scores))
        or not np.all(np.isfinite(margins))
        or np.any((scores < 0.0) | (scores > 1.0))
    ):
        raise ContractError("long threshold-selection scores/margins are invalid")
    if target.ndim != 1 or len(target) != len(scores):
        raise ContractError("long threshold-selection labels are misaligned")
    if target.dtype.kind == "b":
        target = target.astype(np.int8, copy=False)
    elif target.dtype.kind in "iu" and set(map(int, np.unique(target))) <= {0, 1}:
        target = target.astype(np.int8, copy=False)
    else:
        raise ContractError("long threshold-selection labels must be binary")
    positives = target == 1
    negatives = ~positives
    eligible = (
        np.ones(len(scores), dtype=np.bool_)
        if reliability_eligible is None
        else np.asarray(reliability_eligible)
    )
    if eligible.shape != scores.shape or eligible.dtype.kind != "b":
        raise ContractError("long threshold-selection reliability mask is invalid")
    if not np.any(positives) or not np.any(negatives):
        raise ContractError("long threshold selection requires both classes")
    if not math.isfinite(float(provisional_tpr_target)) or not (
        0.0 < float(provisional_tpr_target) <= 1.0
    ):
        raise ContractError("long provisional_tpr_target must be in (0, 1]")
    provisional = _recall_threshold(scores[positives], provisional_tpr_target)

    candidates = np.unique(np.r_[margins, 0.0])
    if len(candidates) > 512:
        indices = np.linspace(0, len(candidates) - 1, 512).round().astype(np.int64)
        candidates = candidates[np.unique(indices)]
    candidates = np.unique(np.maximum(candidates, 0.0))
    best: tuple[int, float, float] | None = None
    for margin_threshold in candidates:
        eligible_negative = negatives & eligible & (margins >= margin_threshold)
        if np.any(eligible_negative):
            probability_threshold = float(
                np.nextafter(np.max(scores[eligible_negative]), math.inf)
            )
        else:
            probability_threshold = provisional
        probability_threshold = max(probability_threshold, provisional)
        if probability_threshold > 1.0:
            continue
        accepted_positive = int(
            np.count_nonzero(
                positives
                & eligible
                & (scores >= probability_threshold)
                & (margins >= margin_threshold)
            )
        )
        if accepted_positive < 1:
            continue
        if np.any(
            negatives
            & eligible
            & (scores >= probability_threshold)
            & (margins >= margin_threshold)
        ):
            raise ContractError("selected long gate does not reject selection negatives")
        option = (accepted_positive, -probability_threshold, -float(margin_threshold))
        if best is None or option > best:
            best = option
    return SelectedLongGates(
        provisional_threshold=provisional,
        confirmed_probability_threshold=None if best is None else -best[1],
        appearance_margin_threshold=None if best is None else -best[2],
        positive_rows=int(np.count_nonzero(positives)),
        hard_negative_rows=int(np.count_nonzero(negatives)),
    )


def certify_long_gate(
    probabilities: np.ndarray,
    labels: np.ndarray,
    candidate_margins: np.ndarray,
    candidate_group_ids: Sequence[str],
    *,
    probability_threshold: float | None,
    margin_threshold: float | None,
    confirmed_far_target: float,
    confidence_level: float,
    reliability_eligible: np.ndarray | None = None,
) -> LongGateCertification:
    """Apply a frozen gate and certify unique hard-negative groups only."""

    scores = np.asarray(probabilities, dtype=np.float64)
    raw_target = np.asarray(labels)
    if raw_target.ndim != 1 or len(raw_target) != len(scores):
        raise ContractError("long certification labels are misaligned")
    if raw_target.dtype.kind == "b":
        target = raw_target.astype(np.bool_, copy=False)
    elif raw_target.dtype.kind in "iu" and set(map(int, np.unique(raw_target))) <= {0, 1}:
        target = raw_target.astype(np.bool_, copy=False)
    else:
        raise ContractError("long certification labels must be binary")
    margins = np.asarray(candidate_margins, dtype=np.float64)
    groups = np.asarray(list(candidate_group_ids), dtype=object)
    if (
        scores.ndim != 1
        or not len(scores)
        or target.shape != scores.shape
        or margins.shape != scores.shape
        or groups.shape != scores.shape
        or not np.all(np.isfinite(scores))
        or not np.all(np.isfinite(margins))
        or np.any((scores < 0.0) | (scores > 1.0))
        or any(not isinstance(value, str) or not value for value in groups.tolist())
    ):
        raise ContractError("long certification arrays are invalid")
    far = float(confirmed_far_target)
    confidence = float(confidence_level)
    if not math.isfinite(far) or not 0.0 < far < 1.0:
        raise ContractError("long certification FAR target must be in (0, 1)")
    if not math.isfinite(confidence) or not 0.0 < confidence < 1.0:
        raise ContractError("long certification confidence level must be in (0, 1)")
    negative_groups = sorted(set(map(str, groups[~target])))
    positive_groups = sorted(set(map(str, groups[target])))
    eligible = (
        np.ones(len(scores), dtype=np.bool_)
        if reliability_eligible is None
        else np.asarray(reliability_eligible)
    )
    if eligible.shape != scores.shape or eligible.dtype.kind != "b":
        raise ContractError("long certification reliability mask is invalid")
    if not negative_groups:
        raise ContractError("long certification requires hard-negative groups")
    gate_exists = probability_threshold is not None and margin_threshold is not None
    if gate_exists:
        probability_gate = float(probability_threshold)
        margin_gate = float(margin_threshold)
        if (
            not math.isfinite(probability_gate)
            or not 0.0 <= probability_gate <= 1.0
            or not math.isfinite(margin_gate)
            or margin_gate < 0.0
        ):
            raise ContractError("frozen long certification gate is invalid")
        accepted = eligible & (scores >= probability_gate) & (margins >= margin_gate)
    else:
        if probability_threshold is not None or margin_threshold is not None:
            raise ContractError("frozen long certification gate is incomplete")
        accepted = np.zeros(len(scores), dtype=np.bool_)
    accepted_negative_groups = {
        str(group)
        for group, label, decision in zip(groups, target, accepted, strict=True)
        if not bool(label) and bool(decision)
    }
    accepted_positive_groups = {
        str(group)
        for group, label, decision in zip(groups, target, accepted, strict=True)
        if bool(label) and bool(decision)
    }
    false_accepts = len(accepted_negative_groups)
    upper = clopper_pearson_upper(
        false_accepts, len(negative_groups), confidence
    )
    return LongGateCertification(
        certified=bool(gate_exists and upper <= far),
        false_accepts=false_accepts,
        hard_negative_groups=len(negative_groups),
        false_accept_upper=upper,
        positive_accepts=len(accepted_positive_groups),
        positive_groups=len(positive_groups),
    )


def _feature_matrix(
    rows: Sequence[Mapping[str, Any]], names: tuple[str, ...]
) -> np.ndarray:
    matrix = np.asarray(
        [[float(row["features"][name]) for name in names] for row in rows],
        dtype=np.float64,
    )
    if matrix.shape != (len(rows), len(names)) or not np.all(np.isfinite(matrix)):
        raise ContractError("long calibration feature matrix is invalid")
    return matrix


def _candidate_margins(
    rows: Sequence[Mapping[str, Any]], *, margin_feature_name: str
) -> np.ndarray:
    explicit = [row.get("candidate_margin") for row in rows]
    present = [value is not None for value in explicit]
    if any(present):
        if not all(present):
            raise ContractError(
                "long calibration candidate_margin must be explicit for every row or none"
            )
        result = np.asarray(explicit, dtype=np.float64)
        if not np.all(np.isfinite(result)):
            raise ContractError("long calibration candidate_margin is non-finite")
        return result
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[str(row["candidate_group_id"])].append(index)
    result = np.empty(len(rows), dtype=np.float64)
    for group_id, indices in groups.items():
        if len(indices) < 2:
            raise ContractError(
                "cannot derive candidate margin from singleton group " f"{group_id!r}"
            )
        values = np.asarray(
            [float(rows[index]["features"][margin_feature_name]) for index in indices],
            dtype=np.float64,
        )
        for local_index, row_index in enumerate(indices):
            result[row_index] = float(values[local_index] - np.max(np.delete(values, local_index)))
    return result


def _group_decision_metrics(
    rows: Sequence[Mapping[str, Any]],
    *,
    probability_threshold: float,
    selected_probability_threshold: float | None,
    selected_margin_threshold: float | None,
    confirmed_enabled: bool,
) -> dict[str, Any]:
    labels = np.asarray([bool(row["label"]) for row in rows], dtype=np.bool_)
    groups = np.asarray([str(row["candidate_group_id"]) for row in rows], dtype=object)
    scores = np.asarray([float(row["model_probability"]) for row in rows])
    margins = np.asarray([float(row["candidate_margin"]) for row in rows])
    provisional = scores >= float(probability_threshold)
    selected = np.zeros(len(rows), dtype=np.bool_)
    eligible = np.asarray(
        [not bool(row.get("high_overlap", False)) for row in rows], dtype=np.bool_
    )
    if selected_probability_threshold is not None and selected_margin_threshold is not None:
        selected = eligible & (scores >= selected_probability_threshold) & (
            margins >= selected_margin_threshold
        )
    confirmed = selected if confirmed_enabled else np.zeros(len(rows), dtype=np.bool_)

    def accepted_groups(mask: np.ndarray, label: bool) -> int:
        return len(
            {
                str(group)
                for group, row_label, accepted in zip(groups, labels, mask, strict=True)
                if bool(row_label) is label and bool(accepted)
            }
        )

    return {
        "rows": len(rows),
        "positive_groups": len(set(map(str, groups[labels]))),
        "hard_negative_groups": len(set(map(str, groups[~labels]))),
        "provisional_positive_accepts": accepted_groups(provisional, True),
        "provisional_false_accepts": accepted_groups(provisional, False),
        "selected_gate_positive_accepts": accepted_groups(selected, True),
        "selected_gate_false_accepts": accepted_groups(selected, False),
        "confirmed_positive_accepts": accepted_groups(confirmed, True),
        "confirmed_false_accepts": accepted_groups(confirmed, False),
    }


def _audit_report(
    rows: Sequence[Mapping[str, Any]], thresholds: Mapping[str, Any]
) -> dict[str, Any]:
    audit = _selected(rows, "audit")
    if not audit or thresholds["provisional_threshold"] is None:
        return {
            "report_only": True,
            "model_enabled": thresholds["model_enabled"],
            "rows": len(audit),
        }
    kwargs = {
        "probability_threshold": float(thresholds["provisional_threshold"]),
        "selected_probability_threshold": thresholds["selected_confirmed_threshold"],
        "selected_margin_threshold": thresholds[
            "selected_appearance_margin_threshold"
        ],
        "confirmed_enabled": bool(thresholds["confirmed_enabled"]),
    }
    report: dict[str, Any] = {
        "report_only": True,
        "overall": _group_decision_metrics(audit, **kwargs),
        "per_clip": {},
        "by_stratum": {},
    }
    clips = sorted(
        {str(row["source_clip_id"]) for row in audit}
        | {str(row["target_clip_id"]) for row in audit}
    )
    for clip_id in clips:
        selected_rows = [
            row
            for row in audit
            if row["source_clip_id"] == clip_id or row["target_clip_id"] == clip_id
        ]
        report["per_clip"][clip_id] = _group_decision_metrics(
            selected_rows, **kwargs
        )
    if all("stratum" in row for row in audit):
        for stratum in sorted({str(row["stratum"]) for row in audit}):
            report["by_stratum"][stratum] = _group_decision_metrics(
                [row for row in audit if str(row["stratum"]) == stratum], **kwargs
            )
    return report


def _base_thresholds(
    settings: LongCalibrationSettings,
    counts: Mapping[str, Any],
) -> dict[str, Any]:
    selection = counts["partitions"]["threshold_selection"]
    certification = counts["partitions"]["certification"]
    audit = counts["partitions"]["audit"]
    return {
        "mode": "long",
        "confirmed_far_target": float(settings.confirmed_far_target),
        "confidence_level": float(settings.confidence_level),
        "provisional_tpr_target": float(settings.provisional_tpr_target),
        "counting_unit": "unique_candidate_group_id_within_partition_and_label",
        "threshold_selection_positive_count": selection["positive_groups"],
        "threshold_selection_hard_negative_count": selection[
            "hard_negative_groups"
        ],
        "certification_positive_count": certification["positive_groups"],
        "certification_hard_negative_count": certification[
            "hard_negative_groups"
        ],
        "audit_positive_count": audit["positive_groups"],
        "audit_hard_negative_count": audit["hard_negative_groups"],
        "calibration_positive_count": (
            selection["positive_groups"] + certification["positive_groups"]
        ),
        "calibration_hard_negative_count": (
            selection["hard_negative_groups"]
            + certification["hard_negative_groups"]
        ),
        "confirmed_gate_selection_split": "threshold_selection",
        "confirmed_gate_certification_split": "certification",
        "provisional_threshold_source": "threshold_selection_positive_tpr",
        "certification_positive_policy": "report_only_not_threshold_selection",
    }


def calibrate_long_rows(
    rows: Sequence[Mapping[str, Any]],
    feature_names: Sequence[str] = LONG_FEATURE_SCHEMA,
    *,
    settings: LongCalibrationSettings | None = None,
) -> LongCalibrationResult:
    """Fit, select, independently certify, and audit canonical long rows.

    Malformed data raises :class:`ContractError`.  Valid data that does not
    satisfy the configured independent-group minima returns a serializable
    disabled model instead of lowering a minimum.
    """

    policy = settings if settings is not None else LongCalibrationSettings()
    if not isinstance(policy, LongCalibrationSettings):
        raise ContractError("long calibration settings have the wrong type")
    names = _feature_schema(feature_names)
    if policy.margin_feature_name not in names:
        raise ContractError("long calibration margin feature is absent from schema")
    canonical = _canonical_rows(rows, names)
    counts = stratified_group_counts(canonical)
    base = _base_thresholds(policy, counts)
    adequacy = audit_adequacy_report(
        counts,
        policy.minimums,
        evidence_scope=policy.audit_evidence_scope,
        expected_clip_ids=policy.expected_clip_ids,
    )
    failure = calibration_minimum_failure_reason(
        counts,
        policy.minimums,
        evidence_scope=policy.audit_evidence_scope,
        expected_clip_ids=policy.expected_clip_ids,
    )
    if failure is not None:
        model = disabled_link_model(
            names, mode="long", random_seed=policy.random_seed, reason=failure
        )
        certification_negatives = int(base["certification_hard_negative_count"])
        thresholds = {
            **base,
            "model_enabled": False,
            "disabled_reason": failure,
            "confirmed_enabled": False,
            "confirmed_threshold": None,
            "provisional_threshold": None,
            "appearance_margin_threshold": None,
            "selected_confirmed_threshold": None,
            "selected_appearance_margin_threshold": None,
            "confirmed_disabled_reason": "model_disabled",
            "confirmed_false_accepts": 0,
            "confirmed_false_accept_upper": (
                clopper_pearson_upper(
                    0, certification_negatives, policy.confidence_level
                )
                if certification_negatives
                else None
            ),
            "certification_true_accepts": 0,
            "confirmed_certified_independently": False,
        }
        scored = []
        for row in canonical:
            item = dict(row)
            item.update(
                model_probability=None,
                model_raw_score=None,
                candidate_margin=None,
                decision="reject",
            )
            scored.append(item)
        return LongCalibrationResult(
            model=model,
            thresholds=thresholds,
            scored_rows=tuple(scored),
            stratified_counts=counts,
            audit_metrics=_audit_report(scored, thresholds),
            audit_adequacy=adequacy,
        )

    train = _selected(canonical, "train")
    train_labels = np.asarray([bool(row["label"]) for row in train], dtype=np.int8)
    # Exact, possibly asymmetric group minima were already enforced above.
    # The public model primitive accepts only one symmetric row minimum.
    model = fit_link_model(
        _feature_matrix(train, names),
        train_labels,
        names,
        mode="long",
        random_seed=policy.random_seed,
        logistic_c=policy.logistic_c,
        max_iter=policy.max_iter,
        minimum_per_class=min(
            policy.minimums.train_positive_groups,
            policy.minimums.train_hard_negative_groups,
        ),
        tolerance=policy.tolerance,
        fit_intercept=policy.fit_intercept,
    )
    matrix = _feature_matrix(canonical, names)
    probabilities = model.probabilities(matrix)
    raw_scores = model.raw_scores(matrix)
    margins = _candidate_margins(
        canonical, margin_feature_name=policy.margin_feature_name
    )
    scored: list[dict[str, Any]] = []
    for row, probability, raw_score, margin in zip(
        canonical, probabilities, raw_scores, margins, strict=True
    ):
        item = dict(row)
        item["model_probability"] = float(probability)
        item["model_raw_score"] = float(raw_score)
        item["candidate_margin"] = float(margin)
        scored.append(item)

    selection = _selected(scored, "threshold_selection")
    selected = select_long_gates(
        np.asarray([row["model_probability"] for row in selection]),
        np.asarray([row["label"] for row in selection], dtype=np.int8),
        np.asarray([row["candidate_margin"] for row in selection]),
        provisional_tpr_target=policy.provisional_tpr_target,
        reliability_eligible=np.asarray(
            [not bool(row.get("high_overlap", False)) for row in selection],
            dtype=np.bool_,
        ),
    )
    certification_rows = _selected(scored, "certification")
    certificate = certify_long_gate(
        np.asarray([row["model_probability"] for row in certification_rows]),
        np.asarray([row["label"] for row in certification_rows], dtype=np.bool_),
        np.asarray([row["candidate_margin"] for row in certification_rows]),
        [str(row["candidate_group_id"]) for row in certification_rows],
        probability_threshold=selected.confirmed_probability_threshold,
        margin_threshold=selected.appearance_margin_threshold,
        confirmed_far_target=policy.confirmed_far_target,
        confidence_level=policy.confidence_level,
        reliability_eligible=np.asarray(
            [not bool(row.get("high_overlap", False)) for row in certification_rows],
            dtype=np.bool_,
        ),
    )
    disabled_reason: str | None = None
    if selected.confirmed_probability_threshold is None:
        disabled_reason = "no_threshold_selection_candidate"
    elif not certificate.certified:
        disabled_reason = "certification_false_accept_upper_exceeds_target"
    thresholds = {
        **base,
        "model_enabled": True,
        "disabled_reason": None,
        "confirmed_enabled": certificate.certified,
        "confirmed_threshold": (
            selected.confirmed_probability_threshold if certificate.certified else None
        ),
        "provisional_threshold": selected.provisional_threshold,
        "appearance_margin_threshold": (
            selected.appearance_margin_threshold if certificate.certified else None
        ),
        "selected_confirmed_threshold": selected.confirmed_probability_threshold,
        "selected_appearance_margin_threshold": selected.appearance_margin_threshold,
        "confirmed_disabled_reason": disabled_reason,
        "confirmed_false_accepts": certificate.false_accepts,
        "confirmed_false_accept_upper": certificate.false_accept_upper,
        "certification_true_accepts": certificate.positive_accepts,
        "confirmed_certified_independently": True,
    }
    for row in scored:
        probability = float(row["model_probability"])
        margin = float(row["candidate_margin"])
        decision = "reject"
        if probability >= selected.provisional_threshold:
            decision = "provisional"
        if (
            certificate.certified
            and probability >= float(selected.confirmed_probability_threshold)
            and margin >= float(selected.appearance_margin_threshold)
            and not bool(row.get("high_overlap", False))
        ):
            decision = "confirmed"
        row["decision"] = decision
    return LongCalibrationResult(
        model=model,
        thresholds=thresholds,
        scored_rows=tuple(scored),
        stratified_counts=counts,
        audit_metrics=_audit_report(scored, thresholds),
        audit_adequacy=adequacy,
    )


__all__ = [
    "AUDIT_EVIDENCE_AGGREGATE_OBSERVED_ROWS",
    "AUDIT_EVIDENCE_PER_CLIP_STRICT",
    "CALIBRATION_PARTITIONS",
    "CalibrationMinimums",
    "LongCalibrationResult",
    "LongCalibrationSettings",
    "LongGateCertification",
    "SelectedLongGates",
    "audit_adequacy_report",
    "calibrate_long_rows",
    "calibration_minimum_failure_reason",
    "certify_long_gate",
    "select_long_gates",
    "stratified_group_counts",
]
