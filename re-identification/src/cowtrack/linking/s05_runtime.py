"""Strict, read-only runtime loader for completed S05A long calibration.

Downstream S05 proposal code consumes this module instead of importing the
calibration stage's private resume validator.  A calibration directory is
accepted only as one immutable snapshot: the success-marker policy, every
recorded input and output byte fingerprint, the exact feature/Parquet schemas,
the fitted model, the published thresholds, and the report must all agree.

The public scorer deliberately applies only the published runtime gates.
Threshold-selection candidates recorded for calibration diagnostics are never
exposed as executable confirmation gates when ``confirmed_enabled`` is false.
"""

from __future__ import annotations

import hashlib
import json
import math
import stat
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from cowtrack.config import ContractError
from cowtrack.linking.calibration_core import (
    AUDIT_EVIDENCE_AGGREGATE_OBSERVED_ROWS,
    CALIBRATION_PARTITIONS,
    CalibrationMinimums,
    audit_adequacy_report,
    calibration_minimum_failure_reason,
    certify_long_gate,
    select_long_gates,
    stratified_group_counts,
)
from cowtrack.linking.features import LONG_FEATURE_SCHEMA
from cowtrack.linking.long_calibration_config import (
    LongCalibrationConfig,
    load_long_calibration_config,
)
from cowtrack.linking.model import (
    LinkModelArtifact,
    LinkResult,
    LinkScorer,
    clopper_pearson_upper,
    load_link_model,
)
from cowtrack.linking.runtime import FileFingerprint
from cowtrack.schemas.s05 import LONG_CALIBRATION_PAIRS_SCHEMA


_STAGE = "S05_CALIBRATE_LONG"
_PARTITIONS = tuple(CALIBRATION_PARTITIONS)
_PAIR_NAMES = MappingProxyType(
    {
        "train": "long_pairs_train.parquet",
        "threshold_selection": "long_pairs_calibration_selection.parquet",
        "certification": "long_pairs_calibration_certification.parquet",
        "audit": "long_pairs_audit.parquet",
    }
)
_OUTPUT_NAMES = (
    "effective_config.json",
    "link_model_long.joblib",
    "long_calibration_report.json",
    "long_pairs_audit.parquet",
    "long_pairs_calibration_certification.parquet",
    "long_pairs_calibration_selection.parquet",
    "long_pairs_train.parquet",
    "pair_feature_schema.json",
    "thresholds.json",
)
_CONSUMED_NAMES = ("_SUCCESS.json", *_OUTPUT_NAMES)
_MARKER_KEYS = {
    "schema_version",
    "stage",
    "config_hash",
    "execution_mode",
    "global_merge_allowed",
    "solver_used",
    "num_global_merges",
    "input_fingerprints",
    "output_fingerprints",
    "stats",
    "elapsed_sec",
}
_STATS_KEYS = {
    "num_pairs",
    "num_train_pairs",
    "num_threshold_selection_pairs",
    "num_certification_pairs",
    "num_audit_pairs",
    "num_positive_pairs",
    "num_hard_negative_pairs",
    "long_model_enabled",
    "long_confirmed_enabled",
    "num_global_merges",
    "global_identity_artifacts_emitted",
}
_THRESHOLD_KEYS = {
    "schema_version",
    "stage",
    "execution_mode",
    "global_merge_allowed",
    "aggressive_global_merge_allowed",
    "long_model_enabled",
    "long_confirmed_enabled",
    "long_confirmed_threshold",
    "long_provisional_threshold",
    "appearance_margin_threshold",
    "missing_appearance_decision",
    "high_overlap_max_decision",
    "long",
}
_LONG_THRESHOLD_KEYS = {
    "mode",
    "confirmed_far_target",
    "confidence_level",
    "provisional_tpr_target",
    "counting_unit",
    "threshold_selection_positive_count",
    "threshold_selection_hard_negative_count",
    "certification_positive_count",
    "certification_hard_negative_count",
    "audit_positive_count",
    "audit_hard_negative_count",
    "calibration_positive_count",
    "calibration_hard_negative_count",
    "confirmed_gate_selection_split",
    "confirmed_gate_certification_split",
    "provisional_threshold_source",
    "certification_positive_policy",
    "model_enabled",
    "disabled_reason",
    "confirmed_enabled",
    "confirmed_threshold",
    "provisional_threshold",
    "appearance_margin_threshold",
    "selected_confirmed_threshold",
    "selected_appearance_margin_threshold",
    "confirmed_disabled_reason",
    "confirmed_false_accepts",
    "confirmed_false_accept_upper",
    "certification_true_accepts",
    "confirmed_certified_independently",
}
_REPORT_KEYS = {
    "schema_version",
    "stage",
    "config_hash",
    "execution_mode",
    "global_merge_allowed",
    "solver_used",
    "path_cover_used",
    "global_identity_artifacts_emitted",
    "input_coordinate_system",
    "video_decoded",
    "encoder_run",
    "keypoints_used",
    "legacy_tracking_id_used_as_identity_label",
    "approved_existing_s02_artifacts_reused_without_rebuilding",
    "upstream_s02_exclusion_policy_changed",
    "legacy_s03_long_model",
    "split_policy",
    "pair_policy",
    "appearance_policy",
    "gallery_availability",
    "pair_counts",
    "audit_adequacy",
    "audit_metrics",
    "thresholds",
    "certification_capacity",
    "disabled_reason",
    "confirmed_disabled_reason",
    "runtime",
}


@dataclass(frozen=True)
class S05LongRuntimeThresholds:
    """Only gates that a downstream runtime is permitted to execute."""

    model_enabled: bool
    confirmed_enabled: bool
    provisional_threshold: float | None
    confirmed_threshold: float | None
    appearance_margin_threshold: float | None
    confirmed_far_target: float
    confidence_level: float
    disabled_reason: str | None
    confirmed_disabled_reason: str | None


@dataclass(frozen=True)
class S05SelectedGateEvidence:
    """Validated threshold-selection evidence, never an executable gate."""

    probability_threshold: float | None
    margin_threshold: float | None
    certified: bool
    false_accepts: int
    false_accept_upper: float | None
    certification_true_accepts: int
    certification_hard_negative_count: int
    executable: bool = False

    def __post_init__(self) -> None:
        if self.executable is not False:
            raise ContractError("S05A selected-gate evidence is report-only")


class S05LongScorer:
    """Fail-closed scoring facade around the validated published long policy."""

    __slots__ = ("_scorer", "_runtime_thresholds")

    def __init__(
        self,
        scorer: LinkScorer,
        runtime_thresholds: S05LongRuntimeThresholds,
    ) -> None:
        self._scorer = scorer
        self._runtime_thresholds = runtime_thresholds

    @property
    def thresholds(self) -> S05LongRuntimeThresholds:
        return self._runtime_thresholds

    @property
    def model_enabled(self) -> bool:
        return self._runtime_thresholds.model_enabled

    @property
    def confirmed_enabled(self) -> bool:
        return self._runtime_thresholds.confirmed_enabled

    def score_features(
        self,
        feature_values: Mapping[str, float] | None,
        *,
        appearance_present: bool,
        high_overlap: bool,
        candidate_margin: float | None = None,
    ) -> LinkResult:
        """Score one exact-schema long pair using only published gates."""

        result = self._scorer.score_features(
            feature_values,
            appearance_present=appearance_present,
            high_overlap=high_overlap,
            candidate_margin=candidate_margin,
        )
        if not self.confirmed_enabled and result.decision == "confirmed":
            raise ContractError("S05A disabled confirmation was promoted at runtime")
        return result


@dataclass(frozen=True)
class S05LongCalibrationBundle:
    """Immutable, semantically validated S05A runtime snapshot."""

    directory: Path
    config_hash: str
    model: LinkModelArtifact
    scorer: S05LongScorer
    runtime_thresholds: S05LongRuntimeThresholds
    selected_gate_evidence: S05SelectedGateEvidence
    effective_config: Mapping[str, Any]
    pair_feature_schema: Mapping[str, Any]
    report: Mapping[str, Any]
    success_marker: Mapping[str, Any]
    pair_paths: Mapping[str, Path]
    consumed_paths: tuple[Path, ...]
    input_fingerprints: tuple[FileFingerprint, ...]
    output_fingerprints: tuple[FileFingerprint, ...]

    @property
    def model_enabled(self) -> bool:
        return self.runtime_thresholds.model_enabled

    @property
    def confirmed_enabled(self) -> bool:
        return self.runtime_thresholds.confirmed_enabled

    @property
    def thresholds(self) -> S05LongRuntimeThresholds:
        """Safe runtime gates; selected review evidence is intentionally separate."""

        return self.runtime_thresholds


def _read_json(path: Path, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read {label} {path}: {exc}") from exc


def _freeze_json(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType(
            {str(key): _freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _fingerprint_file(path: Path) -> FileFingerprint:
    resolved = path.resolve()
    if not resolved.is_file():
        raise ContractError(f"required S05A artifact does not exist: {resolved}")
    digest = hashlib.sha256()
    try:
        with resolved.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
        size = int(resolved.stat().st_size)
    except OSError as exc:
        raise ContractError(f"cannot fingerprint S05A artifact {resolved}: {exc}") from exc
    return FileFingerprint(str(resolved), size, digest.hexdigest())


def _fingerprint_records(value: Any, label: str) -> tuple[tuple[str, int, str], ...]:
    if not isinstance(value, list):
        raise ContractError(f"{label} must be a list")
    records: list[tuple[str, int, str]] = []
    for item in value:
        if not isinstance(item, dict) or set(item) != {
            "path",
            "size_bytes",
            "sha256",
        }:
            raise ContractError(f"{label} contains an invalid fingerprint record")
        path, size, sha256 = item["path"], item["size_bytes"], item["sha256"]
        if (
            not isinstance(path, str)
            or not path
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
            or not isinstance(sha256, str)
            or len(sha256) != 64
            or any(character not in "0123456789abcdef" for character in sha256)
        ):
            raise ContractError(f"{label} contains invalid fingerprint fields")
        records.append((path, size, sha256))
    if len({path for path, _, _ in records}) != len(records):
        raise ContractError(f"{label} contains duplicate paths")
    return tuple(records)


def _recorded_input_snapshot(value: Any) -> tuple[FileFingerprint, ...]:
    records = _fingerprint_records(value, "S05A input_fingerprints")
    if not records:
        raise ContractError("S05A input_fingerprints cannot be empty")
    if tuple(path for path, _, _ in records) != tuple(
        sorted(path for path, _, _ in records)
    ):
        raise ContractError("S05A input_fingerprints order is not canonical")
    result: list[FileFingerprint] = []
    for raw_path, size, sha256 in records:
        path = Path(raw_path)
        if not path.is_absolute():
            raise ContractError(f"S05A input fingerprint path is not absolute: {path}")
        try:
            resolved = path.resolve(strict=True)
            mode = path.stat().st_mode
        except OSError as exc:
            raise ContractError(f"cannot resolve recorded S05A input {path}: {exc}") from exc
        if resolved != path or not stat.S_ISREG(mode):
            raise ContractError(
                f"S05A input fingerprint is not a canonical regular file: {path}"
            )
        current = _fingerprint_file(path)
        if current.size_bytes != size or current.sha256 != sha256:
            raise ContractError(f"recorded S05A input changed: {path}")
        result.append(current)
    return tuple(result)


def _validate_marker(marker: Any) -> Mapping[str, Any]:
    if not isinstance(marker, dict) or set(marker) != _MARKER_KEYS:
        raise ContractError("S05A success marker fields differ")
    config_hash = marker.get("config_hash")
    if (
        marker.get("schema_version") != "1.0"
        or marker.get("stage") != _STAGE
        or marker.get("execution_mode") != "long_calibration_only"
        or marker.get("global_merge_allowed") is not False
        or marker.get("solver_used") is not False
        or marker.get("num_global_merges") != 0
        or not isinstance(config_hash, str)
        or len(config_hash) != 64
        or any(character not in "0123456789abcdef" for character in config_hash)
    ):
        raise ContractError("S05A success marker policy/config differs")
    elapsed = marker.get("elapsed_sec")
    if (
        isinstance(elapsed, bool)
        or not isinstance(elapsed, (int, float))
        or not math.isfinite(float(elapsed))
        or float(elapsed) < 0.0
    ):
        raise ContractError("S05A success marker elapsed_sec is invalid")
    stats = marker.get("stats")
    if not isinstance(stats, dict) or set(stats) != _STATS_KEYS:
        raise ContractError("S05A success marker statistics differ")
    for name in (
        "num_pairs",
        "num_train_pairs",
        "num_threshold_selection_pairs",
        "num_certification_pairs",
        "num_audit_pairs",
        "num_positive_pairs",
        "num_hard_negative_pairs",
        "num_global_merges",
    ):
        value = stats.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ContractError(f"S05A marker statistic {name} is invalid")
    for name in (
        "long_model_enabled",
        "long_confirmed_enabled",
        "global_identity_artifacts_emitted",
    ):
        if not isinstance(stats.get(name), bool):
            raise ContractError(f"S05A marker statistic {name} is invalid")
    if (
        stats["num_pairs"]
        != sum(
            stats[name]
            for name in (
                "num_train_pairs",
                "num_threshold_selection_pairs",
                "num_certification_pairs",
                "num_audit_pairs",
            )
        )
        or stats["num_pairs"]
        != stats["num_positive_pairs"] + stats["num_hard_negative_pairs"]
        or stats["num_global_merges"] != 0
        or stats["global_identity_artifacts_emitted"] is not False
    ):
        raise ContractError("S05A marker statistics are internally inconsistent")
    return marker


def _verify_output_fingerprints(
    marker: Mapping[str, Any], current: Sequence[FileFingerprint]
) -> tuple[FileFingerprint, ...]:
    records = _fingerprint_records(
        marker.get("output_fingerprints"), "S05A output_fingerprints"
    )
    if tuple(name for name, _, _ in records) != _OUTPUT_NAMES:
        raise ContractError("S05A output fingerprint set/order differs")
    by_name = {Path(item.path).name: item for item in current}
    result: list[FileFingerprint] = []
    for name, size, sha256 in records:
        item = by_name.get(name)
        if item is None or item.size_bytes != size or item.sha256 != sha256:
            raise ContractError(f"completed S05A artifact changed: {name}")
        result.append(item)
    return tuple(result)


def _expected_feature_schema() -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "stage": _STAGE,
        "mode": "long",
        "ordered_features": list(LONG_FEATURE_SCHEMA),
        "feature_source": "side_local_constituent_s02_clean_galleries",
        "path_identity": "stable_id",
        "candidate_margin": (
            "own_prototype_cosine_max_minus_best_parent_competitor"
        ),
        "independent_negative_counting_unit": (
            "unique_unordered_stable_pair_within_partition"
        ),
        "parquet_fields": [
            {
                "name": field.name,
                "type": str(field.type),
                "nullable": field.nullable,
            }
            for field in LONG_CALIBRATION_PAIRS_SCHEMA
        ],
    }


def _read_pair_rows(
    path: Path, partition: str
) -> list[dict[str, Any]]:
    try:
        parquet = pq.ParquetFile(path)
        if not parquet.schema_arrow.equals(
            LONG_CALIBRATION_PAIRS_SCHEMA, check_metadata=False
        ):
            raise ContractError(f"S05A {partition} pair schema differs")
        rows = pq.read_table(path).to_pylist()
    except ContractError:
        raise
    except (OSError, pa.ArrowException, TypeError, ValueError) as exc:
        raise ContractError(f"cannot read S05A {partition} pairs {path}: {exc}") from exc
    if any(row["partition"] != partition for row in rows):
        raise ContractError(f"S05A {partition} file contains another partition")
    return rows


def _candidate_margins_by_parent(
    rows: Sequence[Mapping[str, Any]], model_enabled: bool
) -> None:
    grouped: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["partition"]), int(row["parent_stable_id"]))].append(row)
    for (partition, parent), candidates in grouped.items():
        positives = [row for row in candidates if bool(row["label"])]
        if len(positives) != 1:
            raise ContractError(
                "S05A candidate parent must have exactly one positive: "
                f"partition={partition}, parent_stable_id={parent}"
            )
        appearance = [float(row["prototype_cosine_max"]) for row in candidates]
        for index, row in enumerate(candidates):
            competitors = appearance[:index] + appearance[index + 1 :]
            expected = appearance[index] - max(competitors) if competitors else 0.0
            recorded = row["candidate_margin"]
            if not model_enabled:
                if recorded is not None:
                    raise ContractError("disabled S05A model published a candidate margin")
            elif (
                recorded is None
                or not math.isclose(
                    float(recorded), expected, rel_tol=0.0, abs_tol=1e-12
                )
            ):
                raise ContractError("S05A candidate margin differs from competitors")


def _validate_pair_rows(
    rows_by_partition: Mapping[str, Sequence[Mapping[str, Any]]],
    model: LinkModelArtifact,
    long_thresholds: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], tuple[dict[str, Any], ...]]:
    rows = tuple(
        dict(row)
        for partition in _PARTITIONS
        for row in rows_by_partition[partition]
    )
    matrix = (
        np.asarray(
            [[float(row[name]) for name in LONG_FEATURE_SCHEMA] for row in rows],
            dtype=np.float64,
        )
        if rows
        else np.empty((0, len(LONG_FEATURE_SCHEMA)), dtype=np.float64)
    )
    if matrix.shape != (len(rows), len(LONG_FEATURE_SCHEMA)) or not np.all(
        np.isfinite(matrix)
    ):
        raise ContractError("S05A pair feature matrix is invalid")
    model_enabled = model.pipeline is not None
    probabilities = model.probabilities(matrix) if model_enabled else None
    raw_scores = model.raw_scores(matrix) if model_enabled else None
    pair_ids: set[str] = set()
    parent_partitions: dict[int, set[str]] = defaultdict(set)
    candidate_partitions: dict[str, set[str]] = defaultdict(set)
    negative_groups: set[tuple[str, str]] = set()
    for index, row in enumerate(rows):
        partition = str(row["partition"])
        pair_id = str(row["pair_id"])
        group_id = str(row["candidate_group_id"])
        if (
            partition not in _PARTITIONS
            or row["mode"] != "long"
            or not pair_id
            or pair_id in pair_ids
            or not group_id
        ):
            raise ContractError("S05A pair identity/partition differs")
        pair_ids.add(pair_id)
        parent = int(row["parent_stable_id"])
        parent_partitions[parent].add(partition)
        candidate_partitions[group_id].add(partition)
        if row["appearance_present"] is not True:
            raise ContractError("S05A pair has missing appearance")
        if (
            not math.isfinite(float(row["gap_sec"]))
            or float(row["gap_sec"]) <= 5.0
            or not float(row["source_end_time_sec"])
            < float(row["target_start_time_sec"])
        ):
            raise ContractError("S05A pair temporal contract differs")
        source_id = int(row["source_stable_id"])
        target_id = int(row["target_stable_id"])
        evidence = (
            row["cooccurrence_clip_id"],
            row["cooccurrence_global_frame"],
            row["cooccurrence_parent_det_id"],
            row["cooccurrence_other_det_id"],
        )
        if bool(row["label"]):
            if source_id != parent or target_id != parent or any(
                value is not None for value in evidence
            ):
                raise ContractError("S05A positive path identity differs")
        else:
            if (
                source_id != parent
                or target_id == parent
                or row["cooccurrence_clip_id"] != row["target_start_clip_id"]
                or row["cooccurrence_global_frame"]
                != row["target_start_global_frame"]
                or row["cooccurrence_other_det_id"] != row["target_start_det_id"]
                or not isinstance(row["cooccurrence_parent_det_id"], int)
                or row["cooccurrence_parent_det_id"]
                == row["cooccurrence_other_det_id"]
            ):
                raise ContractError("S05A hard-negative evidence differs")
            token = (partition, group_id)
            if token in negative_groups:
                raise ContractError("S05A hard-negative group is duplicated")
            negative_groups.add(token)

        probability = row["model_probability"]
        raw_score = row["model_raw_score"]
        margin = row["candidate_margin"]
        if not model_enabled:
            if probability is not None or raw_score is not None or margin is not None:
                raise ContractError("disabled S05A model published scores")
            expected_decision = "reject"
        else:
            assert probabilities is not None and raw_scores is not None
            if any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in (probability, raw_score, margin)
            ):
                raise ContractError("enabled S05A pair lacks finite model outputs")
            if not math.isclose(
                float(probability),
                float(probabilities[index]),
                rel_tol=0.0,
                abs_tol=1e-12,
            ) or not math.isclose(
                float(raw_score),
                float(raw_scores[index]),
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ContractError("S05A pair scores differ from persisted model")
            expected_decision = "reject"
            if float(probability) >= float(long_thresholds["provisional_threshold"]):
                expected_decision = "provisional"
            if (
                long_thresholds["confirmed_enabled"] is True
                and float(probability)
                >= float(long_thresholds["confirmed_threshold"])
                and float(margin)
                >= float(long_thresholds["appearance_margin_threshold"])
                and row["high_overlap"] is False
            ):
                expected_decision = "confirmed"
        if row["decision"] != expected_decision:
            raise ContractError("S05A pair decision differs from published gates")
    if any(len(values) != 1 for values in parent_partitions.values()):
        raise ContractError("S05A parent stable ID crosses partitions")
    if any(len(values) != 1 for values in candidate_partitions.values()):
        raise ContractError("S05A candidate group crosses partitions")
    _candidate_margins_by_parent(rows, model_enabled)
    positive_parents = [int(row["parent_stable_id"]) for row in rows if row["label"]]
    if len(positive_parents) != len(set(positive_parents)):
        raise ContractError("S05A emits multiple positives for one parent")

    core_rows = tuple(
        {
            **row,
            "features": {name: float(row[name]) for name in LONG_FEATURE_SCHEMA},
        }
        for row in rows
    )
    counts = stratified_group_counts(core_rows)
    stats = {
        "num_pairs": len(rows),
        "num_train_pairs": len(rows_by_partition["train"]),
        "num_threshold_selection_pairs": len(
            rows_by_partition["threshold_selection"]
        ),
        "num_certification_pairs": len(rows_by_partition["certification"]),
        "num_audit_pairs": len(rows_by_partition["audit"]),
        "num_positive_pairs": sum(bool(row["label"]) for row in rows),
        "num_hard_negative_pairs": sum(not bool(row["label"]) for row in rows),
        "long_model_enabled": model_enabled,
        "long_confirmed_enabled": bool(long_thresholds["confirmed_enabled"]),
        "num_global_merges": 0,
        "global_identity_artifacts_emitted": False,
    }
    return stats, counts, core_rows


def _calibration_minimums(config: LongCalibrationConfig) -> CalibrationMinimums:
    return CalibrationMinimums(
        train_positive_groups=config.min_train_positive,
        train_hard_negative_groups=config.min_train_hard_negative,
        selection_positive_groups=config.min_selection_positive,
        selection_hard_negative_groups=config.min_selection_hard_negative,
        certification_positive_groups=config.min_certification_positive,
        certification_hard_negative_groups=config.min_certification_hard_negative,
        audit_positive_groups=config.min_audit_positive,
        audit_hard_negative_groups=config.min_audit_hard_negative,
        parent_groups_per_partition=config.min_parent_groups_per_partition,
        audit_groups_per_class_per_clip=config.min_audit_pairs_per_class_per_clip,
    )


def _minimum_failure_reason(
    counts: Mapping[str, Any],
    minimums: CalibrationMinimums,
    expected_clip_ids: Sequence[str],
) -> str | None:
    return calibration_minimum_failure_reason(
        counts,
        minimums,
        evidence_scope=AUDIT_EVIDENCE_AGGREGATE_OBSERVED_ROWS,
        expected_clip_ids=expected_clip_ids,
    )


def _validate_thresholds(
    payload: Any,
    model: LinkModelArtifact,
    config: LongCalibrationConfig,
    counts: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> tuple[Mapping[str, Any], LinkScorer, S05LongRuntimeThresholds]:
    if not isinstance(payload, dict) or set(payload) != _THRESHOLD_KEYS:
        raise ContractError("S05A threshold root fields differ")
    if (
        payload["schema_version"] != "1.0"
        or payload["stage"] != _STAGE
        or payload["execution_mode"] != "long_calibration_only"
        or payload["global_merge_allowed"] is not False
        or payload["aggressive_global_merge_allowed"] is not False
        or payload["missing_appearance_decision"] != "reject"
        or payload["high_overlap_max_decision"] != "provisional"
        or not isinstance(payload["long"], dict)
    ):
        raise ContractError("S05A threshold policy differs")
    long = payload["long"]
    if set(long) != _LONG_THRESHOLD_KEYS:
        raise ContractError("S05A long threshold fields differ")
    scorer = LinkScorer(model, long)
    if (
        payload["long_model_enabled"] is not (model.pipeline is not None)
        or payload["long_model_enabled"] is not long["model_enabled"]
        or payload["long_confirmed_enabled"] is not long["confirmed_enabled"]
        or payload["long_confirmed_threshold"] != long["confirmed_threshold"]
        or payload["long_provisional_threshold"] != long["provisional_threshold"]
        or payload["appearance_margin_threshold"]
        != long["appearance_margin_threshold"]
    ):
        raise ContractError("S05A threshold aliases differ")
    fixed = {
        "mode": "long",
        "confirmed_far_target": config.confirmed_far_target,
        "confidence_level": config.confidence_level,
        "provisional_tpr_target": config.provisional_tpr_target,
        "counting_unit": "unique_candidate_group_id_within_partition_and_label",
        "confirmed_gate_selection_split": "threshold_selection",
        "confirmed_gate_certification_split": "certification",
        "provisional_threshold_source": "threshold_selection_positive_tpr",
        "certification_positive_policy": "report_only_not_threshold_selection",
    }
    if any(long.get(name) != value for name, value in fixed.items()):
        raise ContractError("S05A threshold calibration policy differs")
    partitions = counts["partitions"]
    aliases = {
        "threshold_selection_positive_count": partitions["threshold_selection"][
            "positive_groups"
        ],
        "threshold_selection_hard_negative_count": partitions[
            "threshold_selection"
        ]["hard_negative_groups"],
        "certification_positive_count": partitions["certification"][
            "positive_groups"
        ],
        "certification_hard_negative_count": partitions["certification"][
            "hard_negative_groups"
        ],
        "audit_positive_count": partitions["audit"]["positive_groups"],
        "audit_hard_negative_count": partitions["audit"]["hard_negative_groups"],
    }
    aliases["calibration_positive_count"] = (
        aliases["threshold_selection_positive_count"]
        + aliases["certification_positive_count"]
    )
    aliases["calibration_hard_negative_count"] = (
        aliases["threshold_selection_hard_negative_count"]
        + aliases["certification_hard_negative_count"]
    )
    if any(long.get(name) != value for name, value in aliases.items()):
        raise ContractError("S05A threshold count aliases differ")

    minimums = _calibration_minimums(config)
    failure = _minimum_failure_reason(counts, minimums, config.clip_order)
    if model.pipeline is None:
        certification_n = aliases["certification_hard_negative_count"]
        expected_upper = (
            clopper_pearson_upper(0, certification_n, config.confidence_level)
            if certification_n
            else None
        )
        if (
            failure is None
            or model.disabled_reason != failure
            or long["disabled_reason"] != failure
            or long["confirmed_disabled_reason"] != "model_disabled"
            or long["selected_confirmed_threshold"] is not None
            or long["selected_appearance_margin_threshold"] is not None
            or long["confirmed_false_accepts"] != 0
            or long["certification_true_accepts"] != 0
            or long["confirmed_certified_independently"] is not False
            or long["confirmed_false_accept_upper"] != expected_upper
        ):
            raise ContractError("S05A disabled model/threshold state differs")
    else:
        if failure is not None or model.disabled_reason is not None:
            raise ContractError("S05A enabled model does not satisfy sample minima")
        selection = [row for row in rows if row["partition"] == "threshold_selection"]
        selected = select_long_gates(
            np.asarray([row["model_probability"] for row in selection]),
            np.asarray([row["label"] for row in selection], dtype=np.int8),
            np.asarray([row["candidate_margin"] for row in selection]),
            provisional_tpr_target=config.provisional_tpr_target,
            reliability_eligible=np.asarray(
                [not bool(row["high_overlap"]) for row in selection],
                dtype=np.bool_,
            ),
        )
        certification = [row for row in rows if row["partition"] == "certification"]
        certificate = certify_long_gate(
            np.asarray([row["model_probability"] for row in certification]),
            np.asarray([row["label"] for row in certification], dtype=np.bool_),
            np.asarray([row["candidate_margin"] for row in certification]),
            [str(row["candidate_group_id"]) for row in certification],
            probability_threshold=selected.confirmed_probability_threshold,
            margin_threshold=selected.appearance_margin_threshold,
            confirmed_far_target=config.confirmed_far_target,
            confidence_level=config.confidence_level,
            reliability_eligible=np.asarray(
                [not bool(row["high_overlap"]) for row in certification],
                dtype=np.bool_,
            ),
        )
        selected_probability = selected.confirmed_probability_threshold
        selected_margin = selected.appearance_margin_threshold
        expected_reason = (
            None
            if certificate.certified
            else (
                "no_threshold_selection_candidate"
                if selected_probability is None
                else "certification_false_accept_upper_exceeds_target"
            )
        )
        expected_published_probability = (
            selected_probability if certificate.certified else None
        )
        expected_published_margin = selected_margin if certificate.certified else None
        numeric_pairs = (
            (long["provisional_threshold"], selected.provisional_threshold),
            (long["selected_confirmed_threshold"], selected_probability),
            (long["selected_appearance_margin_threshold"], selected_margin),
            (long["confirmed_false_accept_upper"], certificate.false_accept_upper),
        )
        for actual, expected in numeric_pairs:
            if (actual is None) != (expected is None) or (
                actual is not None
                and not math.isclose(
                    float(actual), float(expected), rel_tol=0.0, abs_tol=1e-15
                )
            ):
                raise ContractError("S05A selected/certified threshold differs")
        if (
            long["disabled_reason"] is not None
            or long["confirmed_enabled"] is not certificate.certified
            or long["confirmed_threshold"] != expected_published_probability
            or long["appearance_margin_threshold"] != expected_published_margin
            or long["confirmed_disabled_reason"] != expected_reason
            or long["confirmed_false_accepts"] != certificate.false_accepts
            or long["certification_true_accepts"] != certificate.positive_accepts
            or long["confirmed_certified_independently"] is not True
        ):
            raise ContractError("S05A independent certification state differs")

    runtime = S05LongRuntimeThresholds(
        model_enabled=bool(long["model_enabled"]),
        confirmed_enabled=bool(long["confirmed_enabled"]),
        provisional_threshold=(
            None
            if long["provisional_threshold"] is None
            else float(long["provisional_threshold"])
        ),
        confirmed_threshold=(
            None
            if long["confirmed_threshold"] is None
            else float(long["confirmed_threshold"])
        ),
        appearance_margin_threshold=(
            None
            if long["appearance_margin_threshold"] is None
            else float(long["appearance_margin_threshold"])
        ),
        confirmed_far_target=float(long["confirmed_far_target"]),
        confidence_level=float(long["confidence_level"]),
        disabled_reason=long["disabled_reason"],
        confirmed_disabled_reason=long["confirmed_disabled_reason"],
    )
    if not runtime.confirmed_enabled and (
        runtime.confirmed_threshold is not None
        or runtime.appearance_margin_threshold is not None
    ):
        raise ContractError("S05A disabled confirmation exposed a runtime gate")
    return long, scorer, runtime


def _group_metrics(
    rows: Sequence[Mapping[str, Any]], long: Mapping[str, Any]
) -> dict[str, Any]:
    labels = np.asarray([bool(row["label"]) for row in rows], dtype=np.bool_)
    groups = np.asarray([str(row["candidate_group_id"]) for row in rows], dtype=object)
    scores = np.asarray([float(row["model_probability"]) for row in rows])
    margins = np.asarray([float(row["candidate_margin"]) for row in rows])
    provisional = scores >= float(long["provisional_threshold"])
    selected = np.zeros(len(rows), dtype=np.bool_)
    selected_probability = long["selected_confirmed_threshold"]
    selected_margin = long["selected_appearance_margin_threshold"]
    eligible = np.asarray(
        [not bool(row["high_overlap"]) for row in rows], dtype=np.bool_
    )
    if selected_probability is not None and selected_margin is not None:
        selected = (
            eligible
            & (scores >= float(selected_probability))
            & (margins >= float(selected_margin))
        )
    confirmed = selected if long["confirmed_enabled"] else np.zeros(len(rows), bool)

    def accepted(mask: np.ndarray, label: bool) -> int:
        return len(
            {
                str(group)
                for group, row_label, decision in zip(
                    groups, labels, mask, strict=True
                )
                if bool(row_label) is label and bool(decision)
            }
        )

    return {
        "rows": len(rows),
        "positive_groups": len(set(map(str, groups[labels]))),
        "hard_negative_groups": len(set(map(str, groups[~labels]))),
        "provisional_positive_accepts": accepted(provisional, True),
        "provisional_false_accepts": accepted(provisional, False),
        "selected_gate_positive_accepts": accepted(selected, True),
        "selected_gate_false_accepts": accepted(selected, False),
        "confirmed_positive_accepts": accepted(confirmed, True),
        "confirmed_false_accepts": accepted(confirmed, False),
    }


def _audit_metrics(
    rows: Sequence[Mapping[str, Any]], long: Mapping[str, Any]
) -> dict[str, Any]:
    audit = [row for row in rows if row["partition"] == "audit"]
    if not audit or long["provisional_threshold"] is None:
        return {
            "report_only": True,
            "model_enabled": bool(long["model_enabled"]),
            "rows": len(audit),
        }
    result: dict[str, Any] = {
        "report_only": True,
        "overall": _group_metrics(audit, long),
        "per_clip": {},
        "by_stratum": {},
    }
    clips = sorted(
        {str(row["source_clip_id"]) for row in audit}
        | {str(row["target_clip_id"]) for row in audit}
    )
    for clip_id in clips:
        result["per_clip"][clip_id] = _group_metrics(
            [
                row
                for row in audit
                if row["source_clip_id"] == clip_id
                or row["target_clip_id"] == clip_id
            ],
            long,
        )
    if all("stratum" in row for row in audit):
        for stratum in sorted({str(row["stratum"]) for row in audit}):
            result["by_stratum"][stratum] = _group_metrics(
                [row for row in audit if str(row["stratum"]) == stratum], long
            )
    return result


def _zero_error_required(far: float, confidence: float) -> int:
    return int(math.ceil(math.log1p(-confidence) / math.log1p(-far)))


def _validate_report(
    report: Any,
    marker: Mapping[str, Any],
    config: LongCalibrationConfig,
    config_hash: str,
    threshold_payload: Mapping[str, Any],
    long: Mapping[str, Any],
    counts: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    if not isinstance(report, dict) or set(report) != _REPORT_KEYS:
        raise ContractError("S05A report fields differ")
    fixed = {
        "schema_version": "1.0",
        "stage": _STAGE,
        "config_hash": config_hash,
        "execution_mode": "long_calibration_only",
        "global_merge_allowed": False,
        "solver_used": False,
        "path_cover_used": False,
        "global_identity_artifacts_emitted": False,
        "input_coordinate_system": "raw_encoded_landscape_no_autorotate",
        "video_decoded": False,
        "encoder_run": False,
        "keypoints_used": False,
        "legacy_tracking_id_used_as_identity_label": False,
        "approved_existing_s02_artifacts_reused_without_rebuilding": True,
        "upstream_s02_exclusion_policy_changed": False,
    }
    if any(
        report.get(name) != value
        or (isinstance(value, bool) and type(report.get(name)) is not bool)
        for name, value in fixed.items()
    ):
        raise ContractError("S05A report policy/config differs")
    legacy = report.get("legacy_s03_long_model")
    if (
        not isinstance(legacy, dict)
        or set(legacy)
        != {
            "config_hash",
            "model_enabled",
            "confirmed_enabled",
            "disabled_reason",
            "used_for_scoring_or_thresholds",
        }
        or legacy["config_hash"] != config.expected_s03_config_hash
        or legacy["model_enabled"] is not False
        or legacy["confirmed_enabled"] is not False
        or not isinstance(legacy["disabled_reason"], str)
        or not legacy["disabled_reason"]
        or legacy["used_for_scoring_or_thresholds"] is not False
    ):
        raise ContractError("S05A legacy S03 provenance differs")
    expected_split = {
        "strategy": "stable_parent_time_block",
        "fractions": {
            "train": config.train_fraction,
            "threshold_selection": config.threshold_selection_fraction,
            "certification": config.certification_fraction,
            "audit": config.audit_fraction,
        },
        "assigned_before_pair_or_hard_negative_mining": True,
        "held_out_partition_wins": True,
        "parent_leakage_allowed": False,
    }
    expected_pair = {
        "positive": "same_operator_approved_stable_path_disjoint_segments",
        "min_gap_sec_exclusive": config.min_gap_sec_exclusive,
        "max_positive_pairs_per_parent": config.max_pairs_per_parent,
        "hard_negative": "same_frame_distinct_stable_ids",
        "hard_negative_counting_unit": (
            "unique_unordered_stable_pair_within_partition"
        ),
        "hard_negative_same_frame_evidence_persisted": True,
    }
    expected_appearance = {
        "source": "constituent_s02_clean_sample_embeddings",
        "whole_track_prototypes_used": False,
        "clean_max_other_bbox_iou_exclusive": config.clean_max_other_bbox_iou,
        "minimum_clean_samples_per_side": config.min_clean_samples_per_side,
        "min_side_internal_cosine_p10": config.min_side_internal_cosine_p10,
        "missing_or_nonfinite_decision": "reject",
    }
    positive_parents = {
        int(row["parent_stable_id"]) for row in rows if bool(row["label"])
    }
    per_partition = {
        partition: len(
            {
                int(row["parent_stable_id"])
                for row in rows
                if bool(row["label"]) and row["partition"] == partition
            }
        )
        for partition in _PARTITIONS
    }
    gallery = report.get("gallery_availability")
    if not isinstance(gallery, dict) or set(gallery) != {
        "stable_paths_total",
        "s04_whole_stable_appearance",
        "side_local_gallery_positive_parent_coverage",
        "gallery_rebuilt_from_constituent_s02_samples",
        "whole_micro_or_stable_prototypes_used",
    }:
        raise ContractError("S05A gallery availability fields differ")
    stable_total = gallery["stable_paths_total"]
    s04_gallery = gallery["s04_whole_stable_appearance"]
    side_gallery = gallery["side_local_gallery_positive_parent_coverage"]
    if (
        isinstance(stable_total, bool)
        or not isinstance(stable_total, int)
        or stable_total < 1
        or config.expected_stable_track_count not in (None, stable_total)
        or not isinstance(s04_gallery, dict)
        or set(s04_gallery)
        != {"usable", "missing", "used_as_calibration_gallery"}
        or any(
            isinstance(s04_gallery.get(name), bool)
            or not isinstance(s04_gallery.get(name), int)
            or s04_gallery[name] < 0
            for name in ("usable", "missing")
        )
        or s04_gallery["usable"] + s04_gallery["missing"] != stable_total
        or s04_gallery["used_as_calibration_gallery"] is not False
        or not isinstance(side_gallery, dict)
        or set(side_gallery)
        != {"available", "unavailable_or_ineligible", "per_partition"}
        or side_gallery["available"] != len(positive_parents)
        or side_gallery["unavailable_or_ineligible"]
        != stable_total - len(positive_parents)
        or side_gallery["per_partition"] != per_partition
        or gallery["gallery_rebuilt_from_constituent_s02_samples"] is not True
        or gallery["whole_micro_or_stable_prototypes_used"] is not False
    ):
        raise ContractError("S05A gallery availability counts differ")
    certificate_n = int(long["certification_hard_negative_count"])
    expected_capacity = {
        "independent_hard_negative_groups": certificate_n,
        "zero_false_accept_groups_required_for_target": _zero_error_required(
            config.confirmed_far_target, config.confidence_level
        ),
        "far_target": config.confirmed_far_target,
        "confidence_level": config.confidence_level,
        "sufficient_even_at_zero_false_accepts": certificate_n
        >= _zero_error_required(config.confirmed_far_target, config.confidence_level),
    }
    expected_runtime = {
        "configured_worker_count": config.worker_count,
        "pair_mining_execution": "deterministic_single_process",
        "progress_interval_sec": config.progress_interval_sec,
    }
    expected_audit_adequacy = audit_adequacy_report(
        counts,
        _calibration_minimums(config),
        evidence_scope=AUDIT_EVIDENCE_AGGREGATE_OBSERVED_ROWS,
        expected_clip_ids=config.clip_order,
    )
    if (
        report["split_policy"] != expected_split
        or report["pair_policy"] != expected_pair
        or report["appearance_policy"] != expected_appearance
        or report["pair_counts"] != counts
        or report["audit_adequacy"] != expected_audit_adequacy
        or report["audit_metrics"] != _audit_metrics(rows, long)
        or report["thresholds"] != threshold_payload
        or report["certification_capacity"] != expected_capacity
        or report["disabled_reason"] != long["disabled_reason"]
        or report["confirmed_disabled_reason"]
        != long["confirmed_disabled_reason"]
        or report["runtime"] != expected_runtime
        or marker["config_hash"] != report["config_hash"]
    ):
        raise ContractError("S05A report content differs from runtime artifacts")
    return report


def load_s05_long_calibration(output_dir: Path) -> S05LongCalibrationBundle:
    """Load and fully validate one completed S05A calibration without writes."""

    directory = output_dir.resolve()
    if not directory.is_dir():
        raise ContractError(f"S05A calibration directory does not exist: {directory}")
    entries = tuple(directory.iterdir())
    actual_names = {path.name for path in entries if path.is_file()}
    if actual_names != set(_CONSUMED_NAMES) or any(path.is_dir() for path in entries):
        raise ContractError("S05A calibration artifact tree differs")
    consumed_paths = tuple(directory / name for name in _CONSUMED_NAMES)
    before = tuple(_fingerprint_file(path) for path in consumed_paths)

    marker = _validate_marker(
        _read_json(directory / "_SUCCESS.json", "S05A success marker")
    )
    output_fingerprints = _verify_output_fingerprints(marker, before)
    inputs_before = _recorded_input_snapshot(marker["input_fingerprints"])

    config, config_payload, config_hash = load_long_calibration_config(
        directory / "effective_config.json"
    )
    if marker["config_hash"] != config_hash or tuple(
        sorted(vars(config.artifacts).values())
    ) != tuple(sorted(_CONSUMED_NAMES)):
        raise ContractError("S05A effective config/hash differs")
    feature_schema = _read_json(
        directory / "pair_feature_schema.json", "S05A feature schema"
    )
    if feature_schema != _expected_feature_schema():
        raise ContractError("S05A persisted feature schema differs")
    model = load_link_model(directory / "link_model_long.joblib")
    if (
        model.mode != "long"
        or model.feature_names != LONG_FEATURE_SCHEMA
        or model.random_seed != config.random_seed
    ):
        raise ContractError("S05A runtime model schema differs")

    rows_by_partition = {
        partition: _read_pair_rows(directory / _PAIR_NAMES[partition], partition)
        for partition in _PARTITIONS
    }
    threshold_payload = _read_json(
        directory / "thresholds.json", "S05A thresholds"
    )
    # First validate the exact published shape and fail-closed model state;
    # row-derived selection/certification is checked immediately afterwards.
    if (
        not isinstance(threshold_payload, dict)
        or set(threshold_payload) != _THRESHOLD_KEYS
        or not isinstance(threshold_payload.get("long"), dict)
        or set(threshold_payload["long"]) != _LONG_THRESHOLD_KEYS
    ):
        raise ContractError("S05A thresholds root is invalid")
    provisional_long = threshold_payload["long"]
    LinkScorer(model, provisional_long)
    stats, counts, rows = _validate_pair_rows(
        rows_by_partition, model, provisional_long
    )
    long, link_scorer, runtime_thresholds = _validate_thresholds(
        threshold_payload, model, config, counts, rows
    )
    # Revalidate decisions now that the complete threshold contract has been
    # reconstructed from selection and independent certification rows.
    stats, counts, rows = _validate_pair_rows(rows_by_partition, model, long)
    if marker["stats"] != stats:
        raise ContractError("S05A marker statistics differ from pair artifacts")
    report = _validate_report(
        _read_json(directory / "long_calibration_report.json", "S05A report"),
        marker,
        config,
        config_hash,
        threshold_payload,
        long,
        counts,
        rows,
    )
    selected_gate_evidence = S05SelectedGateEvidence(
        probability_threshold=(
            None
            if long["selected_confirmed_threshold"] is None
            else float(long["selected_confirmed_threshold"])
        ),
        margin_threshold=(
            None
            if long["selected_appearance_margin_threshold"] is None
            else float(long["selected_appearance_margin_threshold"])
        ),
        certified=bool(long["confirmed_enabled"]),
        false_accepts=int(long["confirmed_false_accepts"]),
        false_accept_upper=(
            None
            if long["confirmed_false_accept_upper"] is None
            else float(long["confirmed_false_accept_upper"])
        ),
        certification_true_accepts=int(long["certification_true_accepts"]),
        certification_hard_negative_count=int(
            long["certification_hard_negative_count"]
        ),
        executable=False,
    )

    after = tuple(_fingerprint_file(path) for path in consumed_paths)
    if before != after:
        raise ContractError("S05A artifacts changed while being loaded")
    inputs_after = _recorded_input_snapshot(marker["input_fingerprints"])
    if inputs_before != inputs_after:
        raise ContractError("recorded S05A inputs changed while being loaded")

    return S05LongCalibrationBundle(
        directory=directory,
        config_hash=config_hash,
        model=model,
        scorer=S05LongScorer(link_scorer, runtime_thresholds),
        runtime_thresholds=runtime_thresholds,
        selected_gate_evidence=selected_gate_evidence,
        effective_config=_freeze_json(config_payload),
        pair_feature_schema=_freeze_json(feature_schema),
        report=_freeze_json(report),
        success_marker=_freeze_json(marker),
        pair_paths=MappingProxyType(
            {
                partition: directory / name
                for partition, name in _PAIR_NAMES.items()
            }
        ),
        consumed_paths=consumed_paths,
        input_fingerprints=inputs_before,
        output_fingerprints=output_fingerprints,
    )


__all__ = [
    "S05LongCalibrationBundle",
    "S05LongRuntimeThresholds",
    "S05LongScorer",
    "S05SelectedGateEvidence",
    "load_s05_long_calibration",
]
