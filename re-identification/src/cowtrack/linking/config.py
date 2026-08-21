"""Strict, dataset-fixed configuration contract for S03 link calibration."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from cowtrack.config import ContractError


@dataclass(frozen=True)
class LinkArtifacts:
    """Canonical S03 artifact basenames."""

    link_model_short: str
    link_model_long: str
    thresholds: str
    pair_feature_schema: str
    calibration_report: str
    pseudo_pairs_train: str
    pseudo_pairs_calibration: str
    pseudo_pairs_audit: str
    effective_config: str
    success: str


@dataclass(frozen=True)
class LinkCalibrationConfig:
    """Validated S03 settings exposed as a stage-friendly flat contract."""

    schema_version: str
    random_seed: int
    use_keypoints: bool
    use_legacy_tracking_id: bool

    split_strategy: str
    train_fraction: float
    calibration_fraction: float
    calibration_selection_fraction: float
    calibration_certification_fraction: float
    audit_fraction: float
    split_group_by_parent_micro: bool
    split_group_cross_clip_micro: bool
    split_time_ordered_within_clip: bool
    split_hard_negative_mining_per_split: bool
    split_require_all_clips_in_audit: bool

    required_parent_status: str
    short_gap_targets_sec: tuple[float, ...]
    long_gap_targets_sec: tuple[float, ...]
    gap_target_tolerance_sec: float
    min_parent_detections: int
    min_local_purity_score: float
    min_bidirectional_agreement: float
    min_internal_cosine_p10: float
    max_positive_pairs_per_parent_per_gap: int

    clean_max_other_bbox_iou: float
    clean_iou_comparison: str
    clean_min_samples_per_side: int
    require_s02_prototype_inlier: bool
    exclude_review_detections: bool
    prototypes_per_side: int
    outlier_medoid_cosine: float
    outlier_support_cosine: float
    new_prototype_cosine: float
    missing_appearance_decision: str
    high_overlap_max_decision: str

    short_gap_max_sec: float
    velocity_history_detections: int
    min_velocity_detections: int
    center_residual_normalization: str
    include_constant_velocity_short: bool
    include_constant_velocity_long: bool

    hard_negative_enabled: bool
    hard_negative_strategy: str
    hard_negative_max_candidates_per_positive: int
    hard_negative_appearance_top_k_per_positive: int
    hard_negative_min_simultaneous_frames: int
    hard_negative_gap_match_tolerance_sec: float
    hard_negative_require_positive_gap: bool
    hard_negative_retain_high_overlap_stress: bool
    hard_negative_high_overlap_iou_threshold: float
    hard_negative_per_split_independent: bool

    model_estimator: str
    model_separate_short_long: bool
    model_standardize_features: bool
    model_imputation: str
    model_feature_selection: str
    model_penalty: str
    model_regularization_c: float
    model_solver: str
    model_class_weight: str
    model_fit_intercept: bool
    model_max_iterations: int
    model_tolerance: float

    threshold_confidence_level: float
    confirmed_false_accept_rate_target: float
    confirmed_method: str
    confirmed_negative_stratum: str
    provisional_tpr_target: float
    provisional_method: str
    long_appearance_margin_method: str
    disable_confirmed_when_unproven: bool
    disable_long_confirmed_without_safe_margin: bool
    aggressive_global_merge_allowed: bool
    score_comparison: str

    min_train_positive_per_model: int
    min_train_hard_negative_per_model: int
    min_calibration_positive_per_model: int
    min_calibration_hard_negative_per_model: int
    min_audit_positive_per_model: int
    min_audit_hard_negative_per_model: int
    min_audit_pairs_per_class_per_clip: int
    min_parent_groups_per_split: int
    insufficient_sample_action: str

    worker_count: int
    progress_interval_sec: float
    parquet_compression: str
    log_flush: bool
    deterministic_row_sort: bool
    artifacts: LinkArtifacts

    @property
    def split_fractions(self) -> dict[str, float]:
        """Return the canonical train/calibration/audit proportions."""

        return {
            "train": self.train_fraction,
            "calibration": self.calibration_fraction,
            "audit": self.audit_fraction,
        }


_FIXED_ARTIFACTS = {
    "link_model_short": "link_model_short.joblib",
    "link_model_long": "link_model_long.joblib",
    "thresholds": "thresholds.json",
    "pair_feature_schema": "pair_feature_schema.json",
    "calibration_report": "calibration_report.json",
    "pseudo_pairs_train": "pseudo_pairs_train.parquet",
    "pseudo_pairs_calibration": "pseudo_pairs_calibration.parquet",
    "pseudo_pairs_audit": "pseudo_pairs_audit.parquet",
    "effective_config": "effective_config.json",
    "success": "_SUCCESS.json",
}


def _exact_keys(
    mapping: object, expected: set[str], context: str
) -> dict[str, Any]:
    if not isinstance(mapping, dict):
        raise ContractError(f"{context} must be a mapping")
    if any(not isinstance(key, str) for key in mapping):
        raise ContractError(f"{context} config keys must be strings")
    actual = set(mapping)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ContractError(
            f"{context} config keys mismatch; missing={missing}, extra={extra}"
        )
    return mapping


def _string(mapping: dict[str, Any], key: str, context: str) -> str:
    value = mapping[key]
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise ContractError(f"{context}.{key} must be a non-blank string")
    return value


def _integer(mapping: dict[str, Any], key: str, context: str) -> int:
    value = mapping[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ContractError(f"{context}.{key} must be an integer")
    return value


def _finite(mapping: dict[str, Any], key: str, context: str) -> float:
    value = mapping[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"{context}.{key} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ContractError(f"{context}.{key} must be a finite number")
    return result


def _boolean(mapping: dict[str, Any], key: str, context: str) -> bool:
    value = mapping[key]
    if not isinstance(value, bool):
        raise ContractError(f"{context}.{key} must be a boolean")
    return value


def _finite_list(
    mapping: dict[str, Any], key: str, context: str
) -> tuple[float, ...]:
    value = mapping[key]
    if not isinstance(value, list) or not value:
        raise ContractError(f"{context}.{key} must be a non-empty number list")
    parsed: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ContractError(
                f"{context}.{key} must be a non-empty finite number list"
            )
        number = float(item)
        if not math.isfinite(number):
            raise ContractError(
                f"{context}.{key} must be a non-empty finite number list"
            )
        parsed.append(number)
    return tuple(parsed)


def _require_fixed(
    actual: object, expected: object, field: str, *, tolerance: float = 0.0
) -> None:
    if isinstance(expected, float) and isinstance(actual, (int, float)):
        matches = math.isclose(
            float(actual), expected, rel_tol=0.0, abs_tol=tolerance
        )
    else:
        matches = actual == expected
    if not matches:
        raise ContractError(f"fixed S03 requires {field}={expected!r}")


def load_link_calibration_config(
    path: Path,
) -> tuple[LinkCalibrationConfig, dict[str, Any], str]:
    """Load and validate the fixed S03 config without running calibration."""

    path = path.resolve()
    if not path.is_file():
        raise ContractError(f"S03 calibration config does not exist: {path}")
    try:
        raw = path.read_bytes()
        # S03 persists this validated payload as JSON.  Parse JSON first so
        # finite exponent-form numbers (for example ``1e-06``) remain numeric;
        # PyYAML's YAML-1.1 resolver otherwise treats that valid JSON number as
        # a string.  Source configs continue to use the strict YAML path.
        try:
            payload = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError):
            payload = yaml.safe_load(raw)
    except (OSError, yaml.YAMLError) as exc:
        raise ContractError(f"cannot read S03 calibration config {path}: {exc}") from exc

    top = _exact_keys(
        payload,
        {
            "pipeline",
            "split",
            "pseudo_positive",
            "appearance",
            "motion",
            "hard_negative",
            "model",
            "thresholds",
            "minimum_samples",
            "runtime",
            "artifacts",
        },
        "S03 calibration config",
    )
    pipeline = _exact_keys(
        top["pipeline"],
        {
            "schema_version",
            "random_seed",
            "use_keypoints",
            "use_legacy_tracking_id",
        },
        "pipeline",
    )
    split = _exact_keys(
        top["split"],
        {
            "strategy",
            "train_fraction",
            "calibration_fraction",
            "calibration_selection_fraction",
            "calibration_certification_fraction",
            "audit_fraction",
            "group_by_parent_micro",
            "group_cross_clip_micro",
            "time_ordered_within_clip",
            "hard_negative_mining_per_split",
            "require_all_clips_in_audit",
        },
        "split",
    )
    positive = _exact_keys(
        top["pseudo_positive"],
        {
            "required_parent_status",
            "short_gap_targets_sec",
            "long_gap_targets_sec",
            "gap_target_tolerance_sec",
            "min_parent_detections",
            "min_local_purity_score",
            "min_bidirectional_agreement",
            "min_internal_cosine_p10",
            "max_pairs_per_parent_per_gap",
        },
        "pseudo_positive",
    )
    appearance = _exact_keys(
        top["appearance"],
        {
            "clean_max_other_bbox_iou",
            "clean_iou_comparison",
            "clean_min_samples_per_side",
            "require_s02_prototype_inlier",
            "exclude_review_detections",
            "prototypes_per_side",
            "outlier_medoid_cosine",
            "outlier_support_cosine",
            "new_prototype_cosine",
            "missing_appearance_decision",
            "high_overlap_max_decision",
        },
        "appearance",
    )
    motion = _exact_keys(
        top["motion"],
        {
            "short_gap_max_sec",
            "velocity_history_detections",
            "min_velocity_detections",
            "center_residual_normalization",
            "include_constant_velocity_short",
            "include_constant_velocity_long",
        },
        "motion",
    )
    hard_negative = _exact_keys(
        top["hard_negative"],
        {
            "enabled",
            "strategy",
            "max_candidates_per_positive",
            "appearance_top_k_per_positive",
            "min_simultaneous_frames",
            "gap_match_tolerance_sec",
            "require_positive_gap",
            "retain_high_overlap_stress",
            "high_overlap_iou_threshold",
            "per_split_independent",
        },
        "hard_negative",
    )
    model = _exact_keys(
        top["model"],
        {
            "estimator",
            "separate_short_long",
            "standardize_features",
            "imputation",
            "feature_selection",
            "penalty",
            "regularization_c",
            "solver",
            "class_weight",
            "fit_intercept",
            "max_iterations",
            "tolerance",
        },
        "model",
    )
    thresholds = _exact_keys(
        top["thresholds"],
        {
            "confidence_level",
            "confirmed_false_accept_rate_target",
            "confirmed_method",
            "confirmed_negative_stratum",
            "provisional_tpr_target",
            "provisional_method",
            "long_appearance_margin_method",
            "disable_confirmed_when_unproven",
            "disable_long_confirmed_without_safe_margin",
            "aggressive_global_merge_allowed",
            "score_comparison",
        },
        "thresholds",
    )
    minimum = _exact_keys(
        top["minimum_samples"],
        {
            "train_positive_per_model",
            "train_hard_negative_per_model",
            "calibration_positive_per_model",
            "calibration_hard_negative_per_model",
            "audit_positive_per_model",
            "audit_hard_negative_per_model",
            "audit_pairs_per_class_per_clip",
            "parent_groups_per_split",
            "insufficient_action",
        },
        "minimum_samples",
    )
    runtime = _exact_keys(
        top["runtime"],
        {
            "worker_count",
            "progress_interval_sec",
            "parquet_compression",
            "log_flush",
            "deterministic_row_sort",
        },
        "runtime",
    )
    artifact_payload = _exact_keys(
        top["artifacts"], set(_FIXED_ARTIFACTS), "artifacts"
    )

    artifacts = LinkArtifacts(
        **{
            key: _string(artifact_payload, key, "artifacts")
            for key in _FIXED_ARTIFACTS
        }
    )
    result = LinkCalibrationConfig(
        schema_version=_string(pipeline, "schema_version", "pipeline"),
        random_seed=_integer(pipeline, "random_seed", "pipeline"),
        use_keypoints=_boolean(pipeline, "use_keypoints", "pipeline"),
        use_legacy_tracking_id=_boolean(
            pipeline, "use_legacy_tracking_id", "pipeline"
        ),
        split_strategy=_string(split, "strategy", "split"),
        train_fraction=_finite(split, "train_fraction", "split"),
        calibration_fraction=_finite(split, "calibration_fraction", "split"),
        calibration_selection_fraction=_finite(
            split, "calibration_selection_fraction", "split"
        ),
        calibration_certification_fraction=_finite(
            split, "calibration_certification_fraction", "split"
        ),
        audit_fraction=_finite(split, "audit_fraction", "split"),
        split_group_by_parent_micro=_boolean(
            split, "group_by_parent_micro", "split"
        ),
        split_group_cross_clip_micro=_boolean(
            split, "group_cross_clip_micro", "split"
        ),
        split_time_ordered_within_clip=_boolean(
            split, "time_ordered_within_clip", "split"
        ),
        split_hard_negative_mining_per_split=_boolean(
            split, "hard_negative_mining_per_split", "split"
        ),
        split_require_all_clips_in_audit=_boolean(
            split, "require_all_clips_in_audit", "split"
        ),
        required_parent_status=_string(
            positive, "required_parent_status", "pseudo_positive"
        ),
        short_gap_targets_sec=_finite_list(
            positive, "short_gap_targets_sec", "pseudo_positive"
        ),
        long_gap_targets_sec=_finite_list(
            positive, "long_gap_targets_sec", "pseudo_positive"
        ),
        gap_target_tolerance_sec=_finite(
            positive, "gap_target_tolerance_sec", "pseudo_positive"
        ),
        min_parent_detections=_integer(
            positive, "min_parent_detections", "pseudo_positive"
        ),
        min_local_purity_score=_finite(
            positive, "min_local_purity_score", "pseudo_positive"
        ),
        min_bidirectional_agreement=_finite(
            positive, "min_bidirectional_agreement", "pseudo_positive"
        ),
        min_internal_cosine_p10=_finite(
            positive, "min_internal_cosine_p10", "pseudo_positive"
        ),
        max_positive_pairs_per_parent_per_gap=_integer(
            positive, "max_pairs_per_parent_per_gap", "pseudo_positive"
        ),
        clean_max_other_bbox_iou=_finite(
            appearance, "clean_max_other_bbox_iou", "appearance"
        ),
        clean_iou_comparison=_string(
            appearance, "clean_iou_comparison", "appearance"
        ),
        clean_min_samples_per_side=_integer(
            appearance, "clean_min_samples_per_side", "appearance"
        ),
        require_s02_prototype_inlier=_boolean(
            appearance, "require_s02_prototype_inlier", "appearance"
        ),
        exclude_review_detections=_boolean(
            appearance, "exclude_review_detections", "appearance"
        ),
        prototypes_per_side=_integer(
            appearance, "prototypes_per_side", "appearance"
        ),
        outlier_medoid_cosine=_finite(
            appearance, "outlier_medoid_cosine", "appearance"
        ),
        outlier_support_cosine=_finite(
            appearance, "outlier_support_cosine", "appearance"
        ),
        new_prototype_cosine=_finite(
            appearance, "new_prototype_cosine", "appearance"
        ),
        missing_appearance_decision=_string(
            appearance, "missing_appearance_decision", "appearance"
        ),
        high_overlap_max_decision=_string(
            appearance, "high_overlap_max_decision", "appearance"
        ),
        short_gap_max_sec=_finite(motion, "short_gap_max_sec", "motion"),
        velocity_history_detections=_integer(
            motion, "velocity_history_detections", "motion"
        ),
        min_velocity_detections=_integer(
            motion, "min_velocity_detections", "motion"
        ),
        center_residual_normalization=_string(
            motion, "center_residual_normalization", "motion"
        ),
        include_constant_velocity_short=_boolean(
            motion, "include_constant_velocity_short", "motion"
        ),
        include_constant_velocity_long=_boolean(
            motion, "include_constant_velocity_long", "motion"
        ),
        hard_negative_enabled=_boolean(
            hard_negative, "enabled", "hard_negative"
        ),
        hard_negative_strategy=_string(
            hard_negative, "strategy", "hard_negative"
        ),
        hard_negative_max_candidates_per_positive=_integer(
            hard_negative, "max_candidates_per_positive", "hard_negative"
        ),
        hard_negative_appearance_top_k_per_positive=_integer(
            hard_negative, "appearance_top_k_per_positive", "hard_negative"
        ),
        hard_negative_min_simultaneous_frames=_integer(
            hard_negative, "min_simultaneous_frames", "hard_negative"
        ),
        hard_negative_gap_match_tolerance_sec=_finite(
            hard_negative, "gap_match_tolerance_sec", "hard_negative"
        ),
        hard_negative_require_positive_gap=_boolean(
            hard_negative, "require_positive_gap", "hard_negative"
        ),
        hard_negative_retain_high_overlap_stress=_boolean(
            hard_negative, "retain_high_overlap_stress", "hard_negative"
        ),
        hard_negative_high_overlap_iou_threshold=_finite(
            hard_negative, "high_overlap_iou_threshold", "hard_negative"
        ),
        hard_negative_per_split_independent=_boolean(
            hard_negative, "per_split_independent", "hard_negative"
        ),
        model_estimator=_string(model, "estimator", "model"),
        model_separate_short_long=_boolean(
            model, "separate_short_long", "model"
        ),
        model_standardize_features=_boolean(
            model, "standardize_features", "model"
        ),
        model_imputation=_string(model, "imputation", "model"),
        model_feature_selection=_string(model, "feature_selection", "model"),
        model_penalty=_string(model, "penalty", "model"),
        model_regularization_c=_finite(model, "regularization_c", "model"),
        model_solver=_string(model, "solver", "model"),
        model_class_weight=_string(model, "class_weight", "model"),
        model_fit_intercept=_boolean(model, "fit_intercept", "model"),
        model_max_iterations=_integer(model, "max_iterations", "model"),
        model_tolerance=_finite(model, "tolerance", "model"),
        threshold_confidence_level=_finite(
            thresholds, "confidence_level", "thresholds"
        ),
        confirmed_false_accept_rate_target=_finite(
            thresholds, "confirmed_false_accept_rate_target", "thresholds"
        ),
        confirmed_method=_string(
            thresholds, "confirmed_method", "thresholds"
        ),
        confirmed_negative_stratum=_string(
            thresholds, "confirmed_negative_stratum", "thresholds"
        ),
        provisional_tpr_target=_finite(
            thresholds, "provisional_tpr_target", "thresholds"
        ),
        provisional_method=_string(
            thresholds, "provisional_method", "thresholds"
        ),
        long_appearance_margin_method=_string(
            thresholds, "long_appearance_margin_method", "thresholds"
        ),
        disable_confirmed_when_unproven=_boolean(
            thresholds, "disable_confirmed_when_unproven", "thresholds"
        ),
        disable_long_confirmed_without_safe_margin=_boolean(
            thresholds,
            "disable_long_confirmed_without_safe_margin",
            "thresholds",
        ),
        aggressive_global_merge_allowed=_boolean(
            thresholds, "aggressive_global_merge_allowed", "thresholds"
        ),
        score_comparison=_string(
            thresholds, "score_comparison", "thresholds"
        ),
        min_train_positive_per_model=_integer(
            minimum, "train_positive_per_model", "minimum_samples"
        ),
        min_train_hard_negative_per_model=_integer(
            minimum, "train_hard_negative_per_model", "minimum_samples"
        ),
        min_calibration_positive_per_model=_integer(
            minimum, "calibration_positive_per_model", "minimum_samples"
        ),
        min_calibration_hard_negative_per_model=_integer(
            minimum,
            "calibration_hard_negative_per_model",
            "minimum_samples",
        ),
        min_audit_positive_per_model=_integer(
            minimum, "audit_positive_per_model", "minimum_samples"
        ),
        min_audit_hard_negative_per_model=_integer(
            minimum, "audit_hard_negative_per_model", "minimum_samples"
        ),
        min_audit_pairs_per_class_per_clip=_integer(
            minimum, "audit_pairs_per_class_per_clip", "minimum_samples"
        ),
        min_parent_groups_per_split=_integer(
            minimum, "parent_groups_per_split", "minimum_samples"
        ),
        insufficient_sample_action=_string(
            minimum, "insufficient_action", "minimum_samples"
        ),
        worker_count=_integer(runtime, "worker_count", "runtime"),
        progress_interval_sec=_finite(
            runtime, "progress_interval_sec", "runtime"
        ),
        parquet_compression=_string(
            runtime, "parquet_compression", "runtime"
        ),
        log_flush=_boolean(runtime, "log_flush", "runtime"),
        deterministic_row_sort=_boolean(
            runtime, "deterministic_row_sort", "runtime"
        ),
        artifacts=artifacts,
    )

    fixed_values: tuple[tuple[object, object, str, float], ...] = (
        (result.schema_version, "1.0", "pipeline.schema_version", 0.0),
        (result.random_seed, 20260710, "pipeline.random_seed", 0.0),
        (result.use_keypoints, False, "pipeline.use_keypoints", 0.0),
        (
            result.use_legacy_tracking_id,
            False,
            "pipeline.use_legacy_tracking_id",
            0.0,
        ),
        (result.split_strategy, "parent_group_time_block", "split.strategy", 0.0),
        (result.train_fraction, 0.60, "split.train_fraction", 1e-12),
        (
            result.calibration_fraction,
            0.20,
            "split.calibration_fraction",
            1e-12,
        ),
        (
            result.calibration_selection_fraction,
            0.50,
            "split.calibration_selection_fraction",
            1e-12,
        ),
        (
            result.calibration_certification_fraction,
            0.50,
            "split.calibration_certification_fraction",
            1e-12,
        ),
        (result.audit_fraction, 0.20, "split.audit_fraction", 1e-12),
        (
            result.split_group_by_parent_micro,
            True,
            "split.group_by_parent_micro",
            0.0,
        ),
        (
            result.split_group_cross_clip_micro,
            True,
            "split.group_cross_clip_micro",
            0.0,
        ),
        (
            result.split_time_ordered_within_clip,
            True,
            "split.time_ordered_within_clip",
            0.0,
        ),
        (
            result.split_hard_negative_mining_per_split,
            True,
            "split.hard_negative_mining_per_split",
            0.0,
        ),
        (
            result.split_require_all_clips_in_audit,
            True,
            "split.require_all_clips_in_audit",
            0.0,
        ),
        (result.required_parent_status, "valid", "pseudo_positive.required_parent_status", 0.0),
        (
            result.short_gap_targets_sec,
            (0.1, 0.5, 1.0, 2.0, 5.0),
            "pseudo_positive.short_gap_targets_sec",
            0.0,
        ),
        (
            result.long_gap_targets_sec,
            (10.0, 30.0, 60.0),
            "pseudo_positive.long_gap_targets_sec",
            0.0,
        ),
        (
            result.clean_max_other_bbox_iou,
            0.25,
            "appearance.clean_max_other_bbox_iou",
            1e-12,
        ),
        (
            result.clean_iou_comparison,
            "less_than",
            "appearance.clean_iou_comparison",
            0.0,
        ),
        (
            result.clean_min_samples_per_side,
            3,
            "appearance.clean_min_samples_per_side",
            0.0,
        ),
        (
            result.require_s02_prototype_inlier,
            True,
            "appearance.require_s02_prototype_inlier",
            0.0,
        ),
        (
            result.exclude_review_detections,
            True,
            "appearance.exclude_review_detections",
            0.0,
        ),
        (result.prototypes_per_side, 3, "appearance.prototypes_per_side", 0.0),
        (
            result.outlier_medoid_cosine,
            0.65,
            "appearance.outlier_medoid_cosine",
            1e-12,
        ),
        (
            result.outlier_support_cosine,
            0.70,
            "appearance.outlier_support_cosine",
            1e-12,
        ),
        (
            result.new_prototype_cosine,
            0.92,
            "appearance.new_prototype_cosine",
            1e-12,
        ),
        (
            result.missing_appearance_decision,
            "reject",
            "appearance.missing_appearance_decision",
            0.0,
        ),
        (
            result.high_overlap_max_decision,
            "provisional",
            "appearance.high_overlap_max_decision",
            0.0,
        ),
        (result.short_gap_max_sec, 5.0, "motion.short_gap_max_sec", 1e-12),
        (
            result.velocity_history_detections,
            4,
            "motion.velocity_history_detections",
            0.0,
        ),
        (
            result.min_velocity_detections,
            2,
            "motion.min_velocity_detections",
            0.0,
        ),
        (
            result.center_residual_normalization,
            "mean_bbox_diagonal",
            "motion.center_residual_normalization",
            0.0,
        ),
        (
            result.include_constant_velocity_short,
            True,
            "motion.include_constant_velocity_short",
            0.0,
        ),
        (
            result.include_constant_velocity_long,
            False,
            "motion.include_constant_velocity_long",
            0.0,
        ),
        (result.hard_negative_enabled, True, "hard_negative.enabled", 0.0),
        (
            result.hard_negative_strategy,
            "simultaneous_at_positive_destination",
            "hard_negative.strategy",
            0.0,
        ),
        (
            result.hard_negative_require_positive_gap,
            True,
            "hard_negative.require_positive_gap",
            0.0,
        ),
        (
            result.hard_negative_retain_high_overlap_stress,
            True,
            "hard_negative.retain_high_overlap_stress",
            0.0,
        ),
        (
            result.hard_negative_high_overlap_iou_threshold,
            0.25,
            "hard_negative.high_overlap_iou_threshold",
            1e-12,
        ),
        (
            result.hard_negative_per_split_independent,
            True,
            "hard_negative.per_split_independent",
            0.0,
        ),
        (
            result.model_estimator,
            "logistic_regression",
            "model.estimator",
            0.0,
        ),
        (
            result.model_separate_short_long,
            True,
            "model.separate_short_long",
            0.0,
        ),
        (
            result.model_standardize_features,
            True,
            "model.standardize_features",
            0.0,
        ),
        (result.model_imputation, "none", "model.imputation", 0.0),
        (
            result.model_feature_selection,
            "fixed_schema",
            "model.feature_selection",
            0.0,
        ),
        (result.model_penalty, "l2", "model.penalty", 0.0),
        (result.model_solver, "lbfgs", "model.solver", 0.0),
        (result.model_class_weight, "balanced", "model.class_weight", 0.0),
        (result.model_fit_intercept, True, "model.fit_intercept", 0.0),
        (
            result.threshold_confidence_level,
            0.95,
            "thresholds.confidence_level",
            1e-12,
        ),
        (
            result.confirmed_false_accept_rate_target,
            0.0001,
            "thresholds.confirmed_false_accept_rate_target",
            1e-15,
        ),
        (
            result.confirmed_method,
            "one_sided_clopper_pearson_upper",
            "thresholds.confirmed_method",
            0.0,
        ),
        (
            result.confirmed_negative_stratum,
            "hard_negative",
            "thresholds.confirmed_negative_stratum",
            0.0,
        ),
        (
            result.provisional_tpr_target,
            0.95,
            "thresholds.provisional_tpr_target",
            1e-12,
        ),
        (
            result.provisional_method,
            "positive_tpr_quantile",
            "thresholds.provisional_method",
            0.0,
        ),
        (
            result.long_appearance_margin_method,
            "joint_with_confirmed_threshold",
            "thresholds.long_appearance_margin_method",
            0.0,
        ),
        (
            result.disable_confirmed_when_unproven,
            True,
            "thresholds.disable_confirmed_when_unproven",
            0.0,
        ),
        (
            result.disable_long_confirmed_without_safe_margin,
            True,
            "thresholds.disable_long_confirmed_without_safe_margin",
            0.0,
        ),
        (
            result.aggressive_global_merge_allowed,
            False,
            "thresholds.aggressive_global_merge_allowed",
            0.0,
        ),
        (
            result.score_comparison,
            "greater_than_or_equal",
            "thresholds.score_comparison",
            0.0,
        ),
        (
            result.insufficient_sample_action,
            "disable_affected_model",
            "minimum_samples.insufficient_action",
            0.0,
        ),
        (result.parquet_compression, "zstd", "runtime.parquet_compression", 0.0),
        (result.log_flush, True, "runtime.log_flush", 0.0),
        (
            result.deterministic_row_sort,
            True,
            "runtime.deterministic_row_sort",
            0.0,
        ),
    )
    for actual, expected, field, tolerance in fixed_values:
        _require_fixed(actual, expected, field, tolerance=tolerance)

    if not math.isclose(
        sum(result.split_fractions.values()), 1.0, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ContractError("S03 split fractions must sum to 1")
    if not math.isclose(
        result.calibration_selection_fraction
        + result.calibration_certification_fraction,
        1.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ContractError("S03 calibration selection/certification fractions must sum to 1")
    if result.gap_target_tolerance_sec <= 0.0:
        raise ContractError("pseudo_positive.gap_target_tolerance_sec must be positive")
    if result.min_parent_detections < 2 * result.clean_min_samples_per_side:
        raise ContractError(
            "pseudo-positive parents need at least two clean sample sides"
        )
    for name, value in (
        ("min_local_purity_score", result.min_local_purity_score),
        ("min_bidirectional_agreement", result.min_bidirectional_agreement),
    ):
        if not 0.0 < value <= 1.0:
            raise ContractError(f"pseudo_positive.{name} must be in (0, 1]")
    if not -1.0 <= result.min_internal_cosine_p10 <= 1.0:
        raise ContractError(
            "pseudo_positive.min_internal_cosine_p10 must be in [-1, 1]"
        )
    if result.max_positive_pairs_per_parent_per_gap < 1:
        raise ContractError(
            "pseudo_positive.max_pairs_per_parent_per_gap must be positive"
        )
    if any(
        not -1.0 <= value <= 1.0
        for value in (
            result.outlier_medoid_cosine,
            result.outlier_support_cosine,
            result.new_prototype_cosine,
        )
    ):
        raise ContractError("appearance cosine thresholds must be in [-1, 1]")
    if not 0.0 <= result.clean_max_other_bbox_iou <= 1.0:
        raise ContractError("appearance.clean_max_other_bbox_iou must be in [0, 1]")
    if result.min_velocity_detections > result.velocity_history_detections:
        raise ContractError(
            "motion.min_velocity_detections cannot exceed velocity history"
        )
    if result.hard_negative_max_candidates_per_positive < 1:
        raise ContractError(
            "hard_negative.max_candidates_per_positive must be positive"
        )
    if not (
        1
        <= result.hard_negative_appearance_top_k_per_positive
        <= result.hard_negative_max_candidates_per_positive
    ):
        raise ContractError(
            "hard-negative top-k must be within the candidate count"
        )
    if result.hard_negative_min_simultaneous_frames < 1:
        raise ContractError(
            "hard_negative.min_simultaneous_frames must be positive"
        )
    if result.hard_negative_gap_match_tolerance_sec < 0.0:
        raise ContractError(
            "hard_negative.gap_match_tolerance_sec cannot be negative"
        )
    if not 0.0 <= result.hard_negative_high_overlap_iou_threshold <= 1.0:
        raise ContractError(
            "hard_negative.high_overlap_iou_threshold must be in [0, 1]"
        )
    if result.model_regularization_c <= 0.0:
        raise ContractError("model.regularization_c must be positive")
    if result.model_max_iterations < 1 or result.model_tolerance <= 0.0:
        raise ContractError("model iterations and tolerance must be positive")
    minimum_counts = (
        result.min_train_positive_per_model,
        result.min_train_hard_negative_per_model,
        result.min_calibration_positive_per_model,
        result.min_calibration_hard_negative_per_model,
        result.min_audit_positive_per_model,
        result.min_audit_hard_negative_per_model,
        result.min_audit_pairs_per_class_per_clip,
        result.min_parent_groups_per_split,
    )
    if any(value < 1 for value in minimum_counts):
        raise ContractError("all minimum_samples counts must be positive")
    if result.worker_count < 1:
        raise ContractError("runtime.worker_count must be positive")
    if result.progress_interval_sec <= 0.0:
        raise ContractError("runtime.progress_interval_sec must be positive")

    actual_artifacts = {
        key: getattr(result.artifacts, key) for key in _FIXED_ARTIFACTS
    }
    if actual_artifacts != _FIXED_ARTIFACTS:
        raise ContractError("S03 artifact filenames are fixed and cannot be changed")
    if any(Path(name).name != name for name in actual_artifacts.values()):
        raise ContractError("S03 artifact values must be basenames")

    canonical = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return result, payload, hashlib.sha256(canonical).hexdigest()


# Concise aliases make the stage-facing API unsurprising while preserving the
# explicit public name above.
CalibrationConfig = LinkCalibrationConfig
load_calibration_config = load_link_calibration_config
load_s03_calibration_config = load_link_calibration_config


__all__ = [
    "CalibrationConfig",
    "LinkArtifacts",
    "LinkCalibrationConfig",
    "load_calibration_config",
    "load_link_calibration_config",
    "load_s03_calibration_config",
]
