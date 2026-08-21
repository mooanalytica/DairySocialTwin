"""S05A stable-path long calibration; this stage never performs identity merging."""

from __future__ import annotations

import json
import math
import os
import shutil
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from cowtrack.config import ContractError
from cowtrack.linking.calibration_core import (
    AUDIT_EVIDENCE_AGGREGATE_OBSERVED_ROWS,
    CalibrationMinimums,
    LongCalibrationResult,
    LongCalibrationSettings,
    audit_adequacy_report,
    calibrate_long_rows,
    stratified_group_counts,
)
from cowtrack.linking.features import LONG_FEATURE_SCHEMA
from cowtrack.linking.long_calibration_config import (
    LongCalibrationConfig,
    load_long_calibration_config,
)
from cowtrack.linking.model import LinkModelArtifact, LinkScorer, load_link_model, save_link_model
from cowtrack.linking.runtime import (
    FileFingerprint,
    ProductionInputBundle,
    fingerprint_file,
    load_production_inputs,
    load_s03_runtime_config,
    validate_s03_input_fingerprints,
)
from cowtrack.linking.s04_runtime import S04FinalizedBundle, load_s04_finalized
from cowtrack.linking.stable_long_pairs import (
    StableLongInput,
    StableLongPolicy,
    generate_stable_long_pairs,
    stable_long_pairs_as_rows,
)
from cowtrack.schemas.s05 import LONG_CALIBRATION_PAIRS_SCHEMA


LogFn = Callable[[str], None]
_STAGE = "S05_CALIBRATE_LONG"
_PARTITIONS = ("train", "threshold_selection", "certification", "audit")
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


def log(message: str) -> None:
    print(message, flush=True)


def _read_json(path: Path, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read {label} {path}: {exc}") from exc


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(
                _plain(payload), ensure_ascii=False, sort_keys=True, indent=2,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    except (OSError, TypeError, ValueError) as exc:
        temporary.unlink(missing_ok=True)
        raise ContractError(f"cannot atomically write S05A JSON {path}: {exc}") from exc


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _fingerprint(path: Path, *, relative_to: Path | None = None) -> dict[str, Any]:
    item = fingerprint_file(path)
    display = (
        str(Path(item.path).relative_to(relative_to.resolve()))
        if relative_to is not None
        else item.path
    )
    return {"path": display, "size_bytes": item.size_bytes, "sha256": item.sha256}


def _normalize_fingerprints(
    values: Sequence[FileFingerprint | Mapping[str, Any]],
) -> list[dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for value in values:
        raw = value.as_dict() if isinstance(value, FileFingerprint) else dict(value)
        if set(raw) != {"path", "size_bytes", "sha256"}:
            raise ContractError("S05A fingerprint record fields differ")
        path, size, sha256 = raw["path"], raw["size_bytes"], raw["sha256"]
        if (
            not isinstance(path, str)
            or not Path(path).is_absolute()
            or str(Path(path).resolve()) != path
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
            or not isinstance(sha256, str)
            or len(sha256) != 64
            or any(character not in "0123456789abcdef" for character in sha256)
        ):
            raise ContractError("S05A fingerprint record is invalid")
        row = {"path": path, "size_bytes": size, "sha256": sha256}
        if path in records and records[path] != row:
            raise ContractError(f"S05A fingerprint path is inconsistent: {path}")
        records[path] = row
    return [records[path] for path in sorted(records)]


def _verify_unchanged(records: Sequence[Mapping[str, Any]]) -> None:
    for expected in records:
        current = fingerprint_file(Path(str(expected["path"])))
        if (
            current.size_bytes != expected["size_bytes"]
            or current.sha256 != expected["sha256"]
        ):
            raise ContractError(f"S05A input changed: {expected['path']}")


def _reject_path_overlap(output_dir: Path, inputs: Sequence[Path]) -> None:
    output = output_dir.resolve()
    for raw in inputs:
        item = raw.resolve()
        if output == item or output in item.parents or item in output.parents:
            raise ContractError(f"S05A output/input paths overlap: {output}, {item}")


def _marker_outputs(
    marker: Mapping[str, Any], required: Sequence[str], directory: Path, label: str
) -> list[FileFingerprint]:
    records = marker.get("output_fingerprints")
    if not isinstance(records, (tuple, list)):
        raise ContractError(f"{label} marker lacks output fingerprints")
    by_name: dict[str, Mapping[str, Any]] = {}
    for record in records:
        if not isinstance(record, Mapping) or not isinstance(record.get("path"), str):
            raise ContractError(f"{label} marker output fingerprint is invalid")
        name = str(record["path"])
        if name in by_name:
            raise ContractError(f"{label} marker has duplicate output fingerprints")
        by_name[name] = record
    missing = sorted(set(required) - set(by_name))
    if missing:
        raise ContractError(f"{label} marker lacks runtime artifacts: {missing}")
    result: list[FileFingerprint] = []
    for name in required:
        current = fingerprint_file(directory / name)
        recorded = by_name[name]
        if (
            current.size_bytes != recorded.get("size_bytes")
            or current.sha256 != recorded.get("sha256")
        ):
            raise ContractError(f"completed {label} artifact changed: {name}")
        result.append(current)
    return result


def _load_s03_provenance(
    calibration_dir: Path,
    production: ProductionInputBundle,
    config: LongCalibrationConfig,
) -> tuple[list[FileFingerprint], dict[str, Any]]:
    runtime = load_s03_runtime_config(calibration_dir)
    if runtime.config_hash != config.expected_s03_config_hash:
        raise ContractError("S05A S03 config hash differs from the fixed input")
    validate_s03_input_fingerprints(calibration_dir, production)
    required = (
        "effective_config.json",
        "link_model_long.joblib",
        "thresholds.json",
        "pair_feature_schema.json",
    )
    fingerprints = [fingerprint_file(calibration_dir / "_SUCCESS.json")]
    fingerprints.extend(
        _marker_outputs(runtime.success_marker, required, calibration_dir, "S03")
    )
    feature_schema = _read_json(
        calibration_dir / "pair_feature_schema.json", "S03 pair feature schema"
    )
    long_schema = feature_schema.get("long") if isinstance(feature_schema, dict) else None
    if (
        not isinstance(long_schema, dict)
        or long_schema.get("ordered_features") != list(LONG_FEATURE_SCHEMA)
    ):
        raise ContractError("S05A S03 long feature schema differs")
    thresholds = _read_json(calibration_dir / "thresholds.json", "S03 thresholds")
    if not isinstance(thresholds, dict) or thresholds.get(
        "aggressive_global_merge_allowed"
    ) is not False:
        raise ContractError("S05A S03 threshold merge policy differs")
    long_thresholds = thresholds.get("long")
    if not isinstance(long_thresholds, dict):
        raise ContractError("S05A S03 thresholds lack long policy")
    model = load_link_model(calibration_dir / "link_model_long.joblib")
    if model.mode != "long" or model.feature_names != LONG_FEATURE_SCHEMA:
        raise ContractError("S05A S03 long model schema differs")
    LinkScorer(model, long_thresholds)
    if (
        model.pipeline is not None
        or long_thresholds.get("model_enabled") is not False
        or long_thresholds.get("confirmed_enabled") is not False
    ):
        raise ContractError("fixed S05A expects the legacy S03 long model to be disabled")
    return fingerprints, {
        "config_hash": runtime.config_hash,
        "model_enabled": False,
        "confirmed_enabled": False,
        "disabled_reason": model.disabled_reason,
        "used_for_scoring_or_thresholds": False,
    }


def _record_map(values: Any, label: str) -> dict[str, Mapping[str, Any]]:
    if not isinstance(values, (tuple, list)):
        raise ContractError(f"{label} must be a list")
    result: dict[str, Mapping[str, Any]] = {}
    for value in values:
        if not isinstance(value, Mapping) or not isinstance(value.get("path"), str):
            raise ContractError(f"{label} contains an invalid record")
        path = str(Path(str(value["path"])).resolve())
        if path != value["path"] or path in result:
            raise ContractError(f"{label} paths are non-canonical or duplicated")
        result[path] = value
    return result


def _validate_s04_against_runtime(
    stable: S04FinalizedBundle,
    production: ProductionInputBundle,
    s03_fingerprints: Sequence[FileFingerprint],
    config: LongCalibrationConfig,
) -> None:
    if (
        len(stable.micro_ids) != config.expected_microtrack_count
        or len(stable.stable_ids) != config.expected_stable_track_count
        or len(stable.det_ids) != config.expected_detection_count
    ):
        raise ContractError("S05A S04 fixed counts differ")
    micros = np.asarray(production.calibration_input.parent_micro_ids, dtype=np.int64)
    if not np.array_equal(stable.micro_ids, np.sort(micros)):
        raise ContractError("S05A S04/runtime micro ID sets differ")
    if set(stable.micro_to_stable) != set(production.micro_paths):
        raise ContractError("S05A S04/runtime micro path sets differ")

    left = np.argsort(stable.det_ids, kind="stable")
    right = np.argsort(production.detections.det_ids, kind="stable")
    if (
        not np.array_equal(stable.det_ids[left], production.detections.det_ids[right])
        or not np.array_equal(
            stable.det_micro_ids[left], production.detections.micro_ids[right]
        )
        or not np.array_equal(
            stable.det_order_in_micro[left], production.detections.order_in_micro[right]
        )
    ):
        raise ContractError("S05A S04/runtime detection bijection differs")
    clips = tuple(map(str, production.calibration_input.timeline_clip_ids))
    if clips != config.clip_order:
        raise ContractError(
            f"S05A requires clip order {config.clip_order!r}, got {clips!r}"
        )

    recorded = _record_map(
        stable.success_marker.get("input_fingerprints"),
        "S04 marker input_fingerprints",
    )
    current = [*production.input_fingerprints, *s03_fingerprints]
    for item in current:
        expected = recorded.get(item.path)
        if expected is None or (
            item.size_bytes != expected.get("size_bytes")
            or item.sha256 != expected.get("sha256")
        ):
            raise ContractError(
                f"S04 stable output was not built from current runtime bytes: {item.path}"
            )


def _policy(config: LongCalibrationConfig) -> StableLongPolicy:
    return StableLongPolicy(
        train_fraction=config.train_fraction,
        threshold_selection_fraction=config.threshold_selection_fraction,
        certification_fraction=config.certification_fraction,
        audit_fraction=config.audit_fraction,
        min_gap_sec_exclusive=config.min_gap_sec_exclusive,
        clean_max_other_bbox_iou_exclusive=config.clean_max_other_bbox_iou,
        min_clean_samples_per_side=config.min_clean_samples_per_side,
        outlier_medoid_cosine=config.outlier_medoid_cosine,
        outlier_support_cosine=config.outlier_support_cosine,
        new_prototype_cosine=config.new_prototype_cosine,
        min_side_internal_cosine_p10=config.min_side_internal_cosine_p10,
        high_overlap_iou_threshold=config.clean_max_other_bbox_iou,
    )


def _settings(config: LongCalibrationConfig) -> LongCalibrationSettings:
    return LongCalibrationSettings(
        random_seed=config.random_seed,
        logistic_c=config.logistic_c,
        max_iter=config.max_iterations,
        tolerance=config.tolerance,
        fit_intercept=True,
        confirmed_far_target=config.confirmed_far_target,
        confidence_level=config.confidence_level,
        provisional_tpr_target=config.provisional_tpr_target,
        audit_evidence_scope=AUDIT_EVIDENCE_AGGREGATE_OBSERVED_ROWS,
        expected_clip_ids=config.clip_order,
        minimums=CalibrationMinimums(
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
        ),
    )


def _feature_schema_payload() -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "stage": _STAGE,
        "mode": "long",
        "ordered_features": list(LONG_FEATURE_SCHEMA),
        "feature_source": "side_local_constituent_s02_clean_galleries",
        "path_identity": "stable_id",
        "candidate_margin": "own_prototype_cosine_max_minus_best_parent_competitor",
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


def _threshold_payload(result: LongCalibrationResult) -> dict[str, Any]:
    long = _plain(result.thresholds)
    return {
        "schema_version": "1.0",
        "stage": _STAGE,
        "execution_mode": "long_calibration_only",
        "global_merge_allowed": False,
        "aggressive_global_merge_allowed": False,
        "long_model_enabled": bool(long["model_enabled"]),
        "long_confirmed_enabled": bool(long["confirmed_enabled"]),
        "long_confirmed_threshold": long["confirmed_threshold"],
        "long_provisional_threshold": long["provisional_threshold"],
        "appearance_margin_threshold": long["appearance_margin_threshold"],
        "missing_appearance_decision": "reject",
        "high_overlap_max_decision": "provisional",
        "long": long,
    }


def _zero_error_required(far: float, confidence: float) -> int:
    return int(math.ceil(math.log1p(-confidence) / math.log1p(-far)))


def _gallery_report(
    stable: S04FinalizedBundle, rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    eligible = {int(row["parent_stable_id"]) for row in rows if bool(row["label"])}
    usable = sum(item.appearance_usable for item in stable.stable_appearance.values())
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
    return {
        "stable_paths_total": len(stable.stable_ids),
        "s04_whole_stable_appearance": {
            "usable": int(usable),
            "missing": int(len(stable.stable_ids) - usable),
            "used_as_calibration_gallery": False,
        },
        "side_local_gallery_positive_parent_coverage": {
            "available": len(eligible),
            "unavailable_or_ineligible": len(stable.stable_ids) - len(eligible),
            "per_partition": per_partition,
        },
        "gallery_rebuilt_from_constituent_s02_samples": True,
        "whole_micro_or_stable_prototypes_used": False,
    }


def _report(
    config: LongCalibrationConfig,
    config_hash: str,
    stable: S04FinalizedBundle,
    result: LongCalibrationResult,
    thresholds: Mapping[str, Any],
    legacy_s03: Mapping[str, Any],
) -> dict[str, Any]:
    certificate_n = int(
        result.thresholds.get("certification_hard_negative_count", 0)
    )
    required = _zero_error_required(config.confirmed_far_target, config.confidence_level)
    return {
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
        "legacy_s03_long_model": _plain(legacy_s03),
        "split_policy": {
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
        },
        "pair_policy": {
            "positive": "same_operator_approved_stable_path_disjoint_segments",
            "min_gap_sec_exclusive": config.min_gap_sec_exclusive,
            "max_positive_pairs_per_parent": config.max_pairs_per_parent,
            "hard_negative": "same_frame_distinct_stable_ids",
            "hard_negative_counting_unit": (
                "unique_unordered_stable_pair_within_partition"
            ),
            "hard_negative_same_frame_evidence_persisted": True,
        },
        "appearance_policy": {
            "source": "constituent_s02_clean_sample_embeddings",
            "whole_track_prototypes_used": False,
            "clean_max_other_bbox_iou_exclusive": config.clean_max_other_bbox_iou,
            "minimum_clean_samples_per_side": config.min_clean_samples_per_side,
            "min_side_internal_cosine_p10": config.min_side_internal_cosine_p10,
            "missing_or_nonfinite_decision": "reject",
        },
        "gallery_availability": _gallery_report(stable, result.scored_rows),
        "pair_counts": _plain(result.stratified_counts),
        "audit_adequacy": _plain(result.audit_adequacy),
        "audit_metrics": _plain(result.audit_metrics),
        "thresholds": _plain(thresholds),
        "certification_capacity": {
            "independent_hard_negative_groups": certificate_n,
            "zero_false_accept_groups_required_for_target": required,
            "far_target": config.confirmed_far_target,
            "confidence_level": config.confidence_level,
            "sufficient_even_at_zero_false_accepts": certificate_n >= required,
        },
        "disabled_reason": result.thresholds.get("disabled_reason"),
        "confirmed_disabled_reason": result.thresholds.get(
            "confirmed_disabled_reason"
        ),
        "runtime": {
            "configured_worker_count": config.worker_count,
            "pair_mining_execution": "deterministic_single_process",
            "progress_interval_sec": config.progress_interval_sec,
        },
    }


def _artifact_names(config: LongCalibrationConfig) -> dict[str, str]:
    artifacts = config.artifacts
    return {
        "train": artifacts.pairs_train,
        "threshold_selection": artifacts.pairs_threshold_selection,
        "certification": artifacts.pairs_certification,
        "audit": artifacts.pairs_audit,
        "model": artifacts.link_model_long,
        "thresholds": artifacts.thresholds,
        "feature_schema": artifacts.pair_feature_schema,
        "report": artifacts.report,
        "config": artifacts.effective_config,
    }


def _write_parquet(
    path: Path, rows: Sequence[Mapping[str, Any]], compression: str
) -> None:
    fields = set(LONG_CALIBRATION_PAIRS_SCHEMA.names)
    for index, row in enumerate(rows):
        missing = fields - set(row)
        if missing:
            raise ContractError(
                f"S05A pair row {index} lacks Parquet fields: {sorted(missing)}"
            )
    try:
        table = pa.Table.from_pylist(
            [dict(row) for row in rows], schema=LONG_CALIBRATION_PAIRS_SCHEMA
        )
        pq.write_table(table, path, compression=compression, version="2.6")
    except (OSError, pa.ArrowException, TypeError, ValueError) as exc:
        raise ContractError(f"cannot write S05A pair Parquet {path}: {exc}") from exc


def _validate_thresholds(
    payload: Any, model: LinkModelArtifact
) -> Mapping[str, Any]:
    expected = {
        "schema_version", "stage", "execution_mode", "global_merge_allowed",
        "aggressive_global_merge_allowed", "long_model_enabled",
        "long_confirmed_enabled", "long_confirmed_threshold",
        "long_provisional_threshold", "appearance_margin_threshold",
        "missing_appearance_decision", "high_overlap_max_decision", "long",
    }
    if not isinstance(payload, dict) or set(payload) != expected:
        raise ContractError("completed S05A threshold root differs")
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
        raise ContractError("completed S05A threshold policy differs")
    long = payload["long"]
    if set(long) != _LONG_THRESHOLD_KEYS:
        raise ContractError("completed S05A long threshold fields differ")
    LinkScorer(model, long)
    if (
        payload["long_model_enabled"] is not (model.pipeline is not None)
        or payload["long_model_enabled"] is not long.get("model_enabled")
        or payload["long_confirmed_enabled"] is not long.get("confirmed_enabled")
        or payload["long_confirmed_threshold"] != long.get("confirmed_threshold")
        or payload["long_provisional_threshold"] != long.get("provisional_threshold")
        or payload["appearance_margin_threshold"]
        != long.get("appearance_margin_threshold")
    ):
        raise ContractError("completed S05A threshold aliases differ")
    return long


def _validate_pair_rows(
    rows_by_partition: Mapping[str, Sequence[Mapping[str, Any]]],
    model: LinkModelArtifact,
    long_thresholds: Mapping[str, Any],
) -> dict[str, Any]:
    all_rows = [dict(row) for partition in _PARTITIONS for row in rows_by_partition[partition]]
    feature_matrix = (
        np.asarray(
            [[float(row[name]) for name in LONG_FEATURE_SCHEMA] for row in all_rows],
            dtype=np.float64,
        )
        if all_rows
        else np.empty((0, len(LONG_FEATURE_SCHEMA)), dtype=np.float64)
    )
    if feature_matrix.shape != (len(all_rows), len(LONG_FEATURE_SCHEMA)) or not np.all(
        np.isfinite(feature_matrix)
    ):
        raise ContractError("completed S05A pair feature matrix is invalid")
    expected_probabilities: np.ndarray | None = None
    expected_raw_scores: np.ndarray | None = None
    if model.pipeline is not None:
        expected_probabilities = model.probabilities(feature_matrix)
        expected_raw_scores = model.raw_scores(feature_matrix)

    pair_ids: set[str] = set()
    parent_partitions: dict[int, set[str]] = defaultdict(set)
    negative_groups: set[tuple[str, str]] = set()
    rows_by_parent: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row_index, row in enumerate(all_rows):
        partition = str(row["partition"])
        if partition not in _PARTITIONS or row["mode"] != "long":
            raise ContractError("completed S05A pair partition/mode differs")
        pair_id = str(row["pair_id"])
        if not pair_id or pair_id in pair_ids:
            raise ContractError("completed S05A pair IDs are blank or duplicated")
        pair_ids.add(pair_id)
        parent = int(row["parent_stable_id"])
        parent_partitions[parent].add(partition)
        rows_by_parent[(partition, parent)].append(row)
        if row["appearance_present"] is not True:
            raise ContractError("completed S05A pair has missing appearance")
        feature_values = np.asarray(
            [row[name] for name in LONG_FEATURE_SCHEMA], dtype=np.float64
        )
        if not np.all(np.isfinite(feature_values)) or float(row["gap_sec"]) <= 5.0:
            raise ContractError("completed S05A pair features are invalid")
        if not float(row["source_end_time_sec"]) < float(
            row["target_start_time_sec"]
        ):
            raise ContractError("completed S05A pair overlaps or runs backward")
        source_id, target_id = int(row["source_stable_id"]), int(row["target_stable_id"])
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
                raise ContractError("completed S05A positive path identity differs")
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
                raise ContractError("completed S05A hard-negative identity differs")
            token = (partition, str(row["candidate_group_id"]))
            if token in negative_groups:
                raise ContractError("completed S05A hard-negative group is duplicated")
            negative_groups.add(token)

        probability, raw_score, margin = (
            row["model_probability"], row["model_raw_score"], row["candidate_margin"]
        )
        if model.pipeline is None:
            if probability is not None or raw_score is not None or margin is not None:
                raise ContractError("disabled S05A model published scores or margins")
            expected_decision = "reject"
        else:
            values = (probability, raw_score, margin)
            if any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in values
            ):
                raise ContractError("enabled S05A model row lacks finite scores")
            assert expected_probabilities is not None
            assert expected_raw_scores is not None
            if not math.isclose(
                float(probability),
                float(expected_probabilities[row_index]),
                rel_tol=0.0,
                abs_tol=1e-12,
            ) or not math.isclose(
                float(raw_score),
                float(expected_raw_scores[row_index]),
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ContractError(
                    "completed S05A model scores differ from the persisted model"
                )
            expected_decision = "reject"
            if float(probability) >= float(long_thresholds["provisional_threshold"]):
                expected_decision = "provisional"
            if (
                long_thresholds["confirmed_enabled"] is True
                and float(probability) >= float(long_thresholds["confirmed_threshold"])
                and float(margin)
                >= float(long_thresholds["appearance_margin_threshold"])
                and row["high_overlap"] is False
            ):
                expected_decision = "confirmed"
        if row["decision"] != expected_decision:
            raise ContractError("completed S05A pair decision differs from frozen gates")
    if any(len(partitions) != 1 for partitions in parent_partitions.values()):
        raise ContractError("completed S05A parent stable ID crosses partitions")

    for (partition, parent), parent_rows in sorted(rows_by_parent.items()):
        positives = [row for row in parent_rows if bool(row["label"])]
        if len(positives) != 1:
            raise ContractError(
                "completed S05A candidate parent must have exactly one positive: "
                f"partition={partition}, parent_stable_id={parent}"
            )
        appearance_scores = [
            float(row["prototype_cosine_max"]) for row in parent_rows
        ]
        for index, row in enumerate(parent_rows):
            competitors = appearance_scores[:index] + appearance_scores[index + 1 :]
            expected_margin = (
                appearance_scores[index] - max(competitors) if competitors else 0.0
            )
            recorded_margin = row["candidate_margin"]
            if model.pipeline is None:
                if recorded_margin is not None:
                    raise ContractError(
                        "disabled S05A model published a candidate margin"
                    )
            elif not math.isclose(
                float(recorded_margin),
                expected_margin,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ContractError(
                    "completed S05A candidate margin differs from parent competitors"
                )

    positive_parents = [int(row["parent_stable_id"]) for row in all_rows if row["label"]]
    if len(positive_parents) != len(set(positive_parents)):
        raise ContractError("completed S05A emits multiple positives for one parent")
    return {
        "num_pairs": len(all_rows),
        "num_train_pairs": len(rows_by_partition["train"]),
        "num_threshold_selection_pairs": len(rows_by_partition["threshold_selection"]),
        "num_certification_pairs": len(rows_by_partition["certification"]),
        "num_audit_pairs": len(rows_by_partition["audit"]),
        "num_positive_pairs": sum(bool(row["label"]) for row in all_rows),
        "num_hard_negative_pairs": sum(not bool(row["label"]) for row in all_rows),
        "long_model_enabled": model.pipeline is not None,
        "long_confirmed_enabled": bool(long_thresholds["confirmed_enabled"]),
        "num_global_merges": 0,
        "global_identity_artifacts_emitted": False,
    }


def _validate_completed_output(
    output_dir: Path,
    success: Mapping[str, Any],
    input_fingerprints: Sequence[Mapping[str, Any]],
    config: LongCalibrationConfig,
    config_payload: Mapping[str, Any],
    config_hash: str,
) -> None:
    success_keys = {
        "schema_version", "stage", "config_hash", "execution_mode",
        "global_merge_allowed", "solver_used", "num_global_merges",
        "input_fingerprints", "output_fingerprints", "stats", "elapsed_sec",
    }
    if not isinstance(success, Mapping) or set(success) != success_keys:
        raise ContractError("completed S05A success marker fields differ")
    if (
        success["schema_version"] != "1.0"
        or success["stage"] != _STAGE
        or success["config_hash"] != config_hash
        or success["execution_mode"] != "long_calibration_only"
        or success["global_merge_allowed"] is not False
        or success["solver_used"] is not False
        or success["num_global_merges"] != 0
    ):
        raise ContractError("completed S05A marker policy/config differs")
    elapsed = success["elapsed_sec"]
    if isinstance(elapsed, bool) or not isinstance(elapsed, (int, float)) or not math.isfinite(float(elapsed)) or float(elapsed) < 0.0:
        raise ContractError("completed S05A elapsed time is invalid")
    if success.get("input_fingerprints") != list(input_fingerprints):
        raise ContractError("completed S05A input fingerprints differ")

    names = _artifact_names(config)
    expected = set(names.values())
    actual = {
        str(path.relative_to(output_dir))
        for path in output_dir.rglob("*")
        if path.is_file() and path.name != config.artifacts.success
    }
    if actual != expected or any(path.is_dir() for path in output_dir.iterdir()):
        raise ContractError("completed S05A artifact tree differs")
    records = success.get("output_fingerprints")
    if not isinstance(records, list):
        raise ContractError("completed S05A lacks output fingerprints")
    recorded = {str(item.get("path")): item for item in records if isinstance(item, dict)}
    if set(recorded) != expected or len(recorded) != len(records):
        raise ContractError("completed S05A output fingerprint set differs")
    for name, item in recorded.items():
        current = _fingerprint(output_dir / name, relative_to=output_dir)
        if current != item:
            raise ContractError(f"completed S05A artifact changed: {name}")

    persisted_config, persisted_payload, persisted_hash = load_long_calibration_config(
        output_dir / config.artifacts.effective_config
    )
    persisted_config = replace(
        persisted_config,
        expected_microtrack_count=config.expected_microtrack_count,
        expected_stable_track_count=config.expected_stable_track_count,
    )
    if persisted_hash != config_hash or persisted_payload != dict(config_payload) or persisted_config != config:
        raise ContractError("completed S05A effective config differs")
    if _read_json(output_dir / config.artifacts.pair_feature_schema, "S05A feature schema") != _feature_schema_payload():
        raise ContractError("completed S05A feature schema differs")
    model = load_link_model(output_dir / config.artifacts.link_model_long)
    if (
        model.mode != "long"
        or model.feature_names != LONG_FEATURE_SCHEMA
        or model.random_seed != config.random_seed
    ):
        raise ContractError("completed S05A model schema differs")
    threshold_payload = _read_json(output_dir / config.artifacts.thresholds, "S05A thresholds")
    long_thresholds = _validate_thresholds(threshold_payload, model)

    rows_by_partition: dict[str, list[dict[str, Any]]] = {}
    for partition in _PARTITIONS:
        path = output_dir / names[partition]
        try:
            parquet = pq.ParquetFile(path)
            if not parquet.schema_arrow.equals(LONG_CALIBRATION_PAIRS_SCHEMA, check_metadata=False):
                raise ContractError(f"completed S05A {partition} schema differs")
            rows = pq.read_table(path).to_pylist()
        except ContractError:
            raise
        except (OSError, pa.ArrowException, TypeError, ValueError) as exc:
            raise ContractError(f"cannot read completed S05A {partition}: {exc}") from exc
        if any(row["partition"] != partition for row in rows):
            raise ContractError(f"completed S05A {partition} file contains another partition")
        rows_by_partition[partition] = rows
    derived_stats = _validate_pair_rows(rows_by_partition, model, long_thresholds)
    if success.get("stats") != derived_stats:
        raise ContractError("completed S05A marker statistics differ")
    all_rows = [
        row
        for partition in _PARTITIONS
        for row in rows_by_partition[partition]
    ]
    derived_counts = stratified_group_counts(all_rows)
    partition_counts = derived_counts["partitions"]
    count_aliases = {
        "threshold_selection_positive_count": partition_counts[
            "threshold_selection"
        ]["positive_groups"],
        "threshold_selection_hard_negative_count": partition_counts[
            "threshold_selection"
        ]["hard_negative_groups"],
        "certification_positive_count": partition_counts["certification"][
            "positive_groups"
        ],
        "certification_hard_negative_count": partition_counts["certification"][
            "hard_negative_groups"
        ],
        "audit_positive_count": partition_counts["audit"]["positive_groups"],
        "audit_hard_negative_count": partition_counts["audit"][
            "hard_negative_groups"
        ],
    }
    count_aliases["calibration_positive_count"] = (
        count_aliases["threshold_selection_positive_count"]
        + count_aliases["certification_positive_count"]
    )
    count_aliases["calibration_hard_negative_count"] = (
        count_aliases["threshold_selection_hard_negative_count"]
        + count_aliases["certification_hard_negative_count"]
    )
    if any(long_thresholds.get(name) != value for name, value in count_aliases.items()):
        raise ContractError("completed S05A threshold count aliases differ")
    if (
        long_thresholds.get("confirmed_far_target") != config.confirmed_far_target
        or long_thresholds.get("confidence_level") != config.confidence_level
        or long_thresholds.get("provisional_tpr_target")
        != config.provisional_tpr_target
    ):
        raise ContractError("completed S05A threshold calibration policy differs")
    report = _read_json(output_dir / config.artifacts.report, "S05A report")
    settings = _settings(config)
    expected_audit_adequacy = audit_adequacy_report(
        derived_counts,
        settings.minimums,
        evidence_scope=settings.audit_evidence_scope,
        expected_clip_ids=settings.expected_clip_ids,
    )
    if (
        not isinstance(report, dict)
        or report.get("stage") != _STAGE
        or report.get("config_hash") != config_hash
        or report.get("global_merge_allowed") is not False
        or report.get("solver_used") is not False
        or report.get("path_cover_used") is not False
        or report.get("global_identity_artifacts_emitted") is not False
        or report.get("thresholds") != threshold_payload
        or report.get("pair_counts") != derived_counts
        or report.get("audit_adequacy") != expected_audit_adequacy
        or report.get("disabled_reason") != long_thresholds.get("disabled_reason")
        or report.get("confirmed_disabled_reason")
        != long_thresholds.get("confirmed_disabled_reason")
    ):
        raise ContractError("completed S05A report policy/content differs")
    _verify_unchanged(input_fingerprints)


def run_s05_calibrate_long(
    ingest_dir: Path,
    microtrack_dir: Path,
    appearance_dir: Path,
    calibration_dir: Path,
    stable_dir: Path,
    config_path: Path,
    output_dir: Path,
    *,
    logger: LogFn = log,
) -> dict[str, Any]:
    """Recalibrate only the long scorer on finalized stable paths."""

    if not callable(logger):
        raise ContractError("S05A logger must be callable")
    started = time.monotonic()
    ingest_dir, microtrack_dir, appearance_dir, calibration_dir, stable_dir, config_path, output_dir = (
        path.resolve()
        for path in (
            ingest_dir, microtrack_dir, appearance_dir, calibration_dir,
            stable_dir, config_path, output_dir,
        )
    )
    _reject_path_overlap(
        output_dir,
        (ingest_dir, microtrack_dir, appearance_dir, calibration_dir, stable_dir, config_path),
    )
    config, config_payload, config_hash = load_long_calibration_config(config_path)

    logger("[s05a] strict-loading immutable S00/S01/S02 inputs")
    production = load_production_inputs(
        ingest_dir, microtrack_dir, appearance_dir, logger=logger
    )
    logger("[s05a] validating disabled legacy S03 long runtime provenance")
    s03_fingerprints, legacy_s03 = _load_s03_provenance(
        calibration_dir, production, config
    )
    logger("[s05a] strict-loading finalized operator-approved S04 stable paths")
    stable = load_s04_finalized(stable_dir)
    observed_micro_count = len(stable.micro_ids)
    observed_stable_count = len(stable.stable_ids)
    if (
        config.expected_microtrack_count not in (None, observed_micro_count)
        or config.expected_stable_track_count not in (None, observed_stable_count)
    ):
        raise ContractError("S05A configured upstream counts differ")
    config = replace(
        config,
        expected_microtrack_count=observed_micro_count,
        expected_stable_track_count=observed_stable_count,
    )
    _validate_s04_against_runtime(stable, production, s03_fingerprints, config)

    s04_upstream = stable.success_marker.get("input_fingerprints")
    if not isinstance(s04_upstream, (tuple, list)) or any(
        not isinstance(item, Mapping) for item in s04_upstream
    ):
        raise ContractError("S04 marker input fingerprints are invalid")

    input_fingerprints = _normalize_fingerprints(
        [
            fingerprint_file(config_path),
            *production.input_fingerprints,
            *s03_fingerprints,
            *s04_upstream,
            *stable.input_fingerprints,
        ]
    )
    success_path = output_dir / config.artifacts.success
    if success_path.is_file():
        existing = _read_json(success_path, "S05A success marker")
        _validate_completed_output(
            output_dir, existing, input_fingerprints, config, config_payload, config_hash
        )
        logger(f"[s05a] already complete and fully revalidated: {success_path}")
        return dict(existing)
    if output_dir.exists():
        if not output_dir.is_dir():
            raise ContractError(f"S05A output path is not a directory: {output_dir}")
        if any(output_dir.iterdir()):
            raise ContractError(
                f"S05A output directory is non-empty without _SUCCESS.json: {output_dir}"
            )

    logger("[s05a] assigning stable partitions before mining any pairs")
    stable_input = StableLongInput(
        production=production,
        micro_to_stable=stable.micro_to_stable,
        micro_order_in_stable=stable.micro_order_in_stable,
    )
    pairs = generate_stable_long_pairs(
        stable_input,
        _policy(config),
        logger=logger,
        progress_interval_sec=config.progress_interval_sec,
    )
    rows = stable_long_pairs_as_rows(pairs)
    logger(f"[s05a] calibrating long-only model from {len(rows):,} pairs")
    settings = _settings(config)
    result = calibrate_long_rows(rows, LONG_FEATURE_SCHEMA, settings=settings)
    required_audit = result.audit_adequacy["required_aggregate_groups"]
    logger(
        "[s05a] calibration evidence policy: "
        f"scope={settings.audit_evidence_scope}, "
        f"clips={len(settings.expected_clip_ids)}, "
        f"audit_positive_groups>={required_audit['positive']}, "
        f"audit_hard_negative_groups>={required_audit['hard_negative']}; "
        "per-clip shortfalls are warnings"
    )
    thresholds = _threshold_payload(result)
    report = _report(config, config_hash, stable, result, thresholds, legacy_s03)

    final_output = output_dir
    final_output.parent.mkdir(parents=True, exist_ok=True)
    staging = final_output.parent / f".{final_output.name}.staging-{os.getpid()}"
    if staging.exists():
        raise ContractError(f"S05A staging directory already exists: {staging}")
    staging.mkdir(parents=False)
    try:
        names = _artifact_names(config)
        save_link_model(staging / names["model"], result.model)
        _write_json(staging / names["thresholds"], thresholds)
        _write_json(staging / names["feature_schema"], _feature_schema_payload())
        _write_json(staging / names["report"], report)
        _write_json(staging / names["config"], config_payload)
        for partition in _PARTITIONS:
            partition_rows = [
                row for row in result.scored_rows if row["partition"] == partition
            ]
            _write_parquet(
                staging / names[partition], partition_rows, config.parquet_compression
            )

        _verify_unchanged(input_fingerprints)
        output_fingerprints = [
            _fingerprint(staging / name, relative_to=staging)
            for name in sorted(set(names.values()))
        ]
        rows_by_partition = {
            partition: [
                dict(row)
                for row in result.scored_rows
                if row["partition"] == partition
            ]
            for partition in _PARTITIONS
        }
        stats = _validate_pair_rows(
            rows_by_partition, result.model, result.thresholds
        )
        success = {
            "schema_version": "1.0",
            "stage": _STAGE,
            "config_hash": config_hash,
            "execution_mode": "long_calibration_only",
            "global_merge_allowed": False,
            "solver_used": False,
            "num_global_merges": 0,
            "input_fingerprints": input_fingerprints,
            "output_fingerprints": output_fingerprints,
            "stats": stats,
            "elapsed_sec": float(time.monotonic() - started),
        }
        _write_json(staging / config.artifacts.success, success)
        _validate_completed_output(
            staging, success, input_fingerprints, config, config_payload, config_hash
        )
        if final_output.exists():
            try:
                final_output.rmdir()
            except OSError as exc:
                raise ContractError(
                    f"S05A output directory cannot be atomically committed: {final_output}"
                ) from exc
        os.replace(staging, final_output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    logger(f"[s05a] complete: {final_output}")
    return success


__all__ = ["run_s05_calibrate_long"]
