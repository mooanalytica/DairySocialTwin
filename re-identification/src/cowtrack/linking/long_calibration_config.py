"""Strict, fixed configuration contract for S05A long calibration."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from cowtrack.config import ContractError
from cowtrack.linking.dataset_contract import EXPECTED_CLIP_ORDER


@dataclass(frozen=True)
class LongCalibrationArtifacts:
    pairs_train: str
    pairs_threshold_selection: str
    pairs_certification: str
    pairs_audit: str
    link_model_long: str
    thresholds: str
    pair_feature_schema: str
    report: str
    effective_config: str
    success: str


@dataclass(frozen=True)
class LongCalibrationConfig:
    schema_version: str
    random_seed: int
    execution_mode: str
    global_merge_allowed: bool
    expected_s03_config_hash: str
    expected_s04_finalize_config_hash: str
    expected_microtrack_count: int | None
    expected_stable_track_count: int | None
    expected_detection_count: int
    clip_order: tuple[str, ...]
    train_fraction: float
    threshold_selection_fraction: float
    certification_fraction: float
    audit_fraction: float
    min_gap_sec_exclusive: float
    max_pairs_per_parent: int
    clean_max_other_bbox_iou: float
    min_clean_samples_per_side: int
    require_s02_prototype_inlier: bool
    exclude_review_detections: bool
    outlier_medoid_cosine: float
    outlier_support_cosine: float
    new_prototype_cosine: float
    min_side_internal_cosine_p10: float
    min_simultaneous_frames: int
    logistic_c: float
    max_iterations: int
    tolerance: float
    confidence_level: float
    confirmed_far_target: float
    provisional_tpr_target: float
    min_train_positive: int
    min_train_hard_negative: int
    min_selection_positive: int
    min_selection_hard_negative: int
    min_certification_positive: int
    min_certification_hard_negative: int
    min_audit_positive: int
    min_audit_hard_negative: int
    min_parent_groups_per_partition: int
    min_audit_pairs_per_class_per_clip: int
    worker_count: int
    progress_interval_sec: float
    parquet_compression: str
    deterministic_row_sort: bool
    log_flush: bool
    artifacts: LongCalibrationArtifacts


_FIXED_ARTIFACTS = {
    "pairs_train": "long_pairs_train.parquet",
    "pairs_threshold_selection": "long_pairs_calibration_selection.parquet",
    "pairs_certification": "long_pairs_calibration_certification.parquet",
    "pairs_audit": "long_pairs_audit.parquet",
    "link_model_long": "link_model_long.joblib",
    "thresholds": "thresholds.json",
    "pair_feature_schema": "pair_feature_schema.json",
    "report": "long_calibration_report.json",
    "effective_config": "effective_config.json",
    "success": "_SUCCESS.json",
}


def _exact(value: object, expected: set[str], context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ContractError(f"{context} must be a mapping")
    keys = set(value)
    if keys != expected:
        raise ContractError(
            f"{context} keys mismatch; missing={sorted(expected - keys)}, "
            f"extra={sorted(keys - expected)}"
        )
    return value


def _fixed(actual: object, expected: object, name: str) -> None:
    if actual != expected or (
        isinstance(expected, bool) and type(actual) is not bool
    ):
        raise ContractError(f"fixed S05A requires {name}={expected!r}")


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ContractError(f"{name} must be finite")
    return result


def _integer(value: object, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ContractError(f"{name} must be an integer >= {minimum}")
    return value


def _optional_integer(
    value: object, name: str, *, minimum: int = 0
) -> int | None:
    if value is None:
        return None
    return _integer(value, name, minimum=minimum)


def _digest(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ContractError(f"{name} must be a lowercase SHA-256 digest")
    return value


def load_long_calibration_config(
    path: Path,
) -> tuple[LongCalibrationConfig, dict[str, Any], str]:
    """Load the deliberately narrow S05A production configuration."""

    path = path.resolve()
    if not path.is_file():
        raise ContractError(f"S05A config does not exist: {path}")
    try:
        serialized = path.read_bytes()
        payload = (
            json.loads(serialized)
            if path.suffix.lower() == ".json"
            else yaml.safe_load(serialized)
        )
    except (OSError, UnicodeError, json.JSONDecodeError, yaml.YAMLError) as exc:
        raise ContractError(f"cannot read S05A config {path}: {exc}") from exc
    top = _exact(
        payload,
        {
            "pipeline",
            "inputs",
            "split",
            "pseudo_positive",
            "appearance",
            "hard_negative",
            "model",
            "thresholds",
            "minimum_samples",
            "runtime",
            "artifacts",
        },
        "S05A",
    )
    pipeline = _exact(
        top["pipeline"],
        {"schema_version", "random_seed", "execution_mode", "global_merge_allowed"},
        "S05A.pipeline",
    )
    inputs = _exact(
        top["inputs"],
        {
            "expected_s03_config_hash",
            "expected_s04_finalize_config_hash",
            "expected_microtrack_count",
            "expected_stable_track_count",
            "expected_detection_count",
            "clip_order",
        },
        "S05A.inputs",
    )
    split = _exact(
        top["split"],
        {
            "strategy",
            "train_fraction",
            "threshold_selection_fraction",
            "certification_fraction",
            "audit_fraction",
            "held_out_partition_wins",
        },
        "S05A.split",
    )
    positive = _exact(
        top["pseudo_positive"],
        {"min_gap_sec_exclusive", "max_pairs_per_parent"},
        "S05A.pseudo_positive",
    )
    appearance = _exact(
        top["appearance"],
        {
            "clean_max_other_bbox_iou_exclusive",
            "min_clean_samples_per_side",
            "require_s02_prototype_inlier",
            "exclude_review_detections",
            "outlier_medoid_cosine",
            "outlier_support_cosine",
            "new_prototype_cosine",
            "min_side_internal_cosine_p10",
        },
        "S05A.appearance",
    )
    negative = _exact(
        top["hard_negative"],
        {
            "strategy",
            "unique_unit",
            "min_simultaneous_frames",
            "deterministic_deduplication",
        },
        "S05A.hard_negative",
    )
    model = _exact(
        top["model"],
        {
            "estimator",
            "standardize_features",
            "penalty",
            "regularization_c",
            "solver",
            "class_weight",
            "fit_intercept",
            "max_iterations",
            "tolerance",
        },
        "S05A.model",
    )
    thresholds = _exact(
        top["thresholds"],
        {
            "confidence_level",
            "confirmed_false_accept_rate_target",
            "confirmed_method",
            "provisional_tpr_target",
            "selection_and_certification_separate",
            "disable_confirmed_when_unproven",
            "score_comparison",
        },
        "S05A.thresholds",
    )
    minimums = _exact(
        top["minimum_samples"],
        {
            "train_positive",
            "train_hard_negative",
            "threshold_selection_positive",
            "threshold_selection_hard_negative",
            "certification_positive",
            "certification_hard_negative",
            "audit_positive",
            "audit_hard_negative",
            "parent_groups_per_partition",
            "audit_pairs_per_class_per_clip",
            "insufficient_action",
        },
        "S05A.minimum_samples",
    )
    runtime = _exact(
        top["runtime"],
        {
            "worker_count",
            "progress_interval_sec",
            "parquet_compression",
            "deterministic_row_sort",
            "log_flush",
        },
        "S05A.runtime",
    )
    artifacts = _exact(top["artifacts"], set(_FIXED_ARTIFACTS), "S05A.artifacts")

    fixed = {
        "pipeline.schema_version": (pipeline["schema_version"], "1.0"),
        "pipeline.execution_mode": (pipeline["execution_mode"], "long_calibration_only"),
        "pipeline.global_merge_allowed": (pipeline["global_merge_allowed"], False),
        "inputs.clip_order": (inputs["clip_order"], list(EXPECTED_CLIP_ORDER)),
        "split.strategy": (split["strategy"], "stable_parent_time_block"),
        "split.held_out_partition_wins": (split["held_out_partition_wins"], True),
        "pseudo_positive.max_pairs_per_parent": (positive["max_pairs_per_parent"], 1),
        "appearance.min_clean_samples_per_side": (
            appearance["min_clean_samples_per_side"],
            3,
        ),
        "appearance.require_s02_prototype_inlier": (
            appearance["require_s02_prototype_inlier"],
            True,
        ),
        "appearance.exclude_review_detections": (
            appearance["exclude_review_detections"],
            True,
        ),
        "hard_negative.strategy": (
            negative["strategy"],
            "simultaneous_stable_at_positive_destination",
        ),
        "hard_negative.unique_unit": (
            negative["unique_unit"],
            "unordered_stable_pair_per_partition",
        ),
        "hard_negative.deterministic_deduplication": (
            negative["deterministic_deduplication"],
            True,
        ),
        "model.estimator": (model["estimator"], "logistic_regression"),
        "model.standardize_features": (model["standardize_features"], True),
        "model.penalty": (model["penalty"], "l2"),
        "model.solver": (model["solver"], "lbfgs"),
        "model.class_weight": (model["class_weight"], "balanced"),
        "model.fit_intercept": (model["fit_intercept"], True),
        "thresholds.confirmed_method": (
            thresholds["confirmed_method"],
            "one_sided_clopper_pearson_upper",
        ),
        "thresholds.selection_and_certification_separate": (
            thresholds["selection_and_certification_separate"],
            True,
        ),
        "thresholds.disable_confirmed_when_unproven": (
            thresholds["disable_confirmed_when_unproven"],
            True,
        ),
        "thresholds.score_comparison": (
            thresholds["score_comparison"],
            "greater_than_or_equal",
        ),
        "minimum_samples.insufficient_action": (
            minimums["insufficient_action"],
            "disable_model",
        ),
        "runtime.deterministic_row_sort": (runtime["deterministic_row_sort"], True),
        "runtime.log_flush": (runtime["log_flush"], True),
    }
    for name, (actual, expected) in fixed.items():
        _fixed(actual, expected, name)

    fractions = tuple(
        _finite(split[name], f"S05A.split.{name}")
        for name in (
            "train_fraction",
            "threshold_selection_fraction",
            "certification_fraction",
            "audit_fraction",
        )
    )
    if fractions != (0.6, 0.1, 0.1, 0.2) or not math.isclose(sum(fractions), 1.0):
        raise ContractError("fixed S05A split fractions must be 0.60/0.10/0.10/0.20")

    numeric_fixed = {
        "pseudo_positive.min_gap_sec_exclusive": (
            _finite(positive["min_gap_sec_exclusive"], "S05A min gap"),
            5.0,
        ),
        "appearance.clean_max_other_bbox_iou_exclusive": (
            _finite(appearance["clean_max_other_bbox_iou_exclusive"], "S05A clean IoU"),
            0.25,
        ),
        "appearance.outlier_medoid_cosine": (
            _finite(appearance["outlier_medoid_cosine"], "S05A medoid cosine"),
            0.65,
        ),
        "appearance.outlier_support_cosine": (
            _finite(appearance["outlier_support_cosine"], "S05A support cosine"),
            0.70,
        ),
        "appearance.new_prototype_cosine": (
            _finite(appearance["new_prototype_cosine"], "S05A prototype cosine"),
            0.92,
        ),
        "appearance.min_side_internal_cosine_p10": (
            _finite(appearance["min_side_internal_cosine_p10"], "S05A p10"),
            0.70,
        ),
        "thresholds.confidence_level": (
            _finite(thresholds["confidence_level"], "S05A confidence"),
            0.95,
        ),
        "thresholds.confirmed_false_accept_rate_target": (
            _finite(
                thresholds["confirmed_false_accept_rate_target"], "S05A FAR target"
            ),
            0.0001,
        ),
        "thresholds.provisional_tpr_target": (
            _finite(thresholds["provisional_tpr_target"], "S05A TPR target"),
            0.95,
        ),
    }
    for name, (actual, expected) in numeric_fixed.items():
        if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12):
            raise ContractError(f"fixed S05A requires {name}={expected!r}")

    minimum_values = {
        "train_positive": 100,
        "train_hard_negative": 100,
        "threshold_selection_positive": 25,
        "threshold_selection_hard_negative": 25,
        "certification_positive": 25,
        "certification_hard_negative": 25,
        "audit_positive": 50,
        "audit_hard_negative": 50,
        "parent_groups_per_partition": 10,
        "audit_pairs_per_class_per_clip": 10,
    }
    parsed_minimums: dict[str, int] = {}
    for name, expected in minimum_values.items():
        value = _integer(minimums[name], f"S05A.minimum_samples.{name}", minimum=1)
        _fixed(value, expected, f"minimum_samples.{name}")
        parsed_minimums[name] = value
    for name, expected in _FIXED_ARTIFACTS.items():
        _fixed(artifacts[name], expected, f"artifacts.{name}")

    seed = _integer(pipeline["random_seed"], "S05A.pipeline.random_seed")
    _fixed(seed, 20260710, "pipeline.random_seed")
    workers = _integer(runtime["worker_count"], "S05A.runtime.worker_count", minimum=1)
    interval = _finite(runtime["progress_interval_sec"], "S05A progress interval")
    if interval <= 0.0:
        raise ContractError("S05A progress interval must be positive")
    compression = runtime["parquet_compression"]
    if not isinstance(compression, str) or not compression:
        raise ContractError("S05A parquet compression cannot be blank")
    _fixed(compression, "zstd", "runtime.parquet_compression")
    s03_hash = _digest(inputs["expected_s03_config_hash"], "expected_s03_config_hash")
    s04_hash = _digest(
        inputs["expected_s04_finalize_config_hash"],
        "expected_s04_finalize_config_hash",
    )
    micro_count = _optional_integer(
        inputs["expected_microtrack_count"], "expected_microtrack_count", minimum=1
    )
    stable_count = _optional_integer(
        inputs["expected_stable_track_count"], "expected_stable_track_count", minimum=1
    )
    detection_count = _integer(inputs["expected_detection_count"], "expected_detection_count", minimum=1)
    if (
        stable_count is not None
        and micro_count is not None
        and stable_count > micro_count
    ) or (micro_count is not None and detection_count < micro_count):
        raise ContractError("S05A expected input counts are inconsistent")

    result = LongCalibrationConfig(
        schema_version="1.0",
        random_seed=seed,
        execution_mode="long_calibration_only",
        global_merge_allowed=False,
        expected_s03_config_hash=s03_hash,
        expected_s04_finalize_config_hash=s04_hash,
        expected_microtrack_count=micro_count,
        expected_stable_track_count=stable_count,
        expected_detection_count=detection_count,
        clip_order=EXPECTED_CLIP_ORDER,
        train_fraction=fractions[0],
        threshold_selection_fraction=fractions[1],
        certification_fraction=fractions[2],
        audit_fraction=fractions[3],
        min_gap_sec_exclusive=5.0,
        max_pairs_per_parent=1,
        clean_max_other_bbox_iou=0.25,
        min_clean_samples_per_side=3,
        require_s02_prototype_inlier=True,
        exclude_review_detections=True,
        outlier_medoid_cosine=0.65,
        outlier_support_cosine=0.70,
        new_prototype_cosine=0.92,
        min_side_internal_cosine_p10=0.70,
        min_simultaneous_frames=_integer(
            negative["min_simultaneous_frames"],
            "S05A.hard_negative.min_simultaneous_frames",
            minimum=1,
        ),
        logistic_c=_finite(model["regularization_c"], "S05A.model.regularization_c"),
        max_iterations=_integer(
            model["max_iterations"], "S05A.model.max_iterations", minimum=1
        ),
        tolerance=_finite(model["tolerance"], "S05A.model.tolerance"),
        confidence_level=0.95,
        confirmed_far_target=0.0001,
        provisional_tpr_target=0.95,
        min_train_positive=parsed_minimums["train_positive"],
        min_train_hard_negative=parsed_minimums["train_hard_negative"],
        min_selection_positive=parsed_minimums["threshold_selection_positive"],
        min_selection_hard_negative=parsed_minimums["threshold_selection_hard_negative"],
        min_certification_positive=parsed_minimums["certification_positive"],
        min_certification_hard_negative=parsed_minimums["certification_hard_negative"],
        min_audit_positive=parsed_minimums["audit_positive"],
        min_audit_hard_negative=parsed_minimums["audit_hard_negative"],
        min_parent_groups_per_partition=parsed_minimums[
            "parent_groups_per_partition"
        ],
        min_audit_pairs_per_class_per_clip=parsed_minimums[
            "audit_pairs_per_class_per_clip"
        ],
        worker_count=workers,
        progress_interval_sec=interval,
        parquet_compression=compression,
        deterministic_row_sort=True,
        log_flush=True,
        artifacts=LongCalibrationArtifacts(**artifacts),
    )
    if result.logistic_c <= 0.0 or result.tolerance <= 0.0:
        raise ContractError("S05A model C and tolerance must be positive")
    _fixed(result.min_simultaneous_frames, 1, "hard_negative.min_simultaneous_frames")
    _fixed(model["max_iterations"], 2000, "model.max_iterations")
    _fixed(model["regularization_c"], 1.0, "model.regularization_c")
    _fixed(model["tolerance"], 0.000001, "model.tolerance")
    _fixed(runtime["worker_count"], 16, "runtime.worker_count")

    canonical = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return result, payload, hashlib.sha256(canonical).hexdigest()


__all__ = [
    "LongCalibrationArtifacts",
    "LongCalibrationConfig",
    "load_long_calibration_config",
]
