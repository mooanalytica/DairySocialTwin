"""Proposal-only S05 long-gap candidate retrieval and scoring stage."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import stat
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from cowtrack.config import ContractError
from cowtrack.linking.features import LONG_FEATURE_SCHEMA
from cowtrack.linking.long_proposal_config import (
    LongProposalConfig,
    load_long_proposal_config,
)
from cowtrack.linking.long_proposals import (
    build_long_proposals,
    enumerate_long_candidates,
    long_review_manifest,
    score_long_candidates,
)
from cowtrack.linking.runtime import (
    FileFingerprint,
    fingerprint_file,
    load_production_inputs,
)
from cowtrack.linking.s04_runtime import S04FinalizedBundle, load_s04_finalized
from cowtrack.linking.s05_runtime import (
    S05LongCalibrationBundle,
    load_s05_long_calibration,
)
from cowtrack.linking.stable_long_pairs import (
    StableLongInput,
    StableLongPolicy,
    StablePathFeatureStore,
)
from cowtrack.schemas.s05_proposals import (
    LONG_CANDIDATE_EDGES_SCHEMA,
    LONG_LINK_PROPOSALS_SCHEMA,
)


LogFn = Callable[[str], None]
_STAGE = "S05_PROPOSE"


def log(message: str) -> None:
    print(message, flush=True)


def _read_json(path: Path, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read {label} {path}: {exc}") from exc


def _write_json(path: Path, payload: Any) -> None:
    try:
        path.write_text(
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
    except (OSError, TypeError, ValueError) as exc:
        raise ContractError(f"cannot write S05 proposal JSON {path}: {exc}") from exc


def _write_parquet(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    schema: pa.Schema,
    compression: str,
) -> None:
    try:
        table = pa.Table.from_pylist([dict(row) for row in rows], schema=schema)
        pq.write_table(table, path, compression=compression, version="2.6")
    except (OSError, pa.ArrowException, TypeError, ValueError) as exc:
        raise ContractError(f"cannot write S05 proposal Parquet {path}: {exc}") from exc


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise ContractError(f"cannot fingerprint S05 proposal artifact {path}: {exc}") from exc
    return digest.hexdigest()


def _output_fingerprint(path: Path, directory: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise ContractError(f"required S05 proposal artifact does not exist: {resolved}")
    return {
        "path": str(resolved.relative_to(directory.resolve())),
        "size_bytes": int(resolved.stat().st_size),
        "sha256": _sha256(resolved),
    }


def _normalize_fingerprints(
    records: Sequence[FileFingerprint | Mapping[str, Any]],
) -> list[dict[str, Any]]:
    by_path: dict[str, dict[str, Any]] = {}
    for record in records:
        source = record.as_dict() if isinstance(record, FileFingerprint) else dict(record)
        raw_path = source.get("path")
        size = source.get("size_bytes")
        sha = source.get("sha256")
        if (
            not isinstance(raw_path, str)
            or not raw_path
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
            or not isinstance(sha, str)
            or len(sha) != 64
            or any(character not in "0123456789abcdef" for character in sha)
        ):
            raise ContractError("S05 proposal input fingerprint is invalid")
        path = str(Path(raw_path).resolve())
        normalized = {"path": path, "size_bytes": size, "sha256": sha}
        if path in by_path and by_path[path] != normalized:
            raise ContractError("S05 proposal input fingerprint conflicts by path")
        by_path[path] = normalized
    return [by_path[path] for path in sorted(by_path)]


def _verify_unchanged(records: Sequence[Mapping[str, Any]]) -> None:
    for expected in records:
        current = fingerprint_file(Path(str(expected["path"]))).as_dict()
        if current != dict(expected):
            raise ContractError(f"S05 proposal input changed: {expected['path']}")


def _artifact_names(config: LongProposalConfig) -> dict[str, str]:
    return {
        "candidates": config.artifacts.candidate_edges,
        "proposals": config.artifacts.proposals,
        "manifest": config.artifacts.review_manifest,
        "report": config.artifacts.report,
        "config": config.artifacts.effective_config,
    }


def _stable_policy_from_runtime(
    long_runtime: S05LongCalibrationBundle,
) -> StableLongPolicy:
    """Bind proposal galleries to the exact validated S05A cleaning policy."""

    payload = long_runtime.effective_config
    split = payload.get("split") if isinstance(payload, Mapping) else None
    pseudo = payload.get("pseudo_positive") if isinstance(payload, Mapping) else None
    appearance = payload.get("appearance") if isinstance(payload, Mapping) else None
    if not all(isinstance(value, Mapping) for value in (split, pseudo, appearance)):
        raise ContractError("S05A runtime lacks the persisted stable-gallery policy")
    try:
        return StableLongPolicy(
            train_fraction=float(split["train_fraction"]),
            threshold_selection_fraction=float(
                split["threshold_selection_fraction"]
            ),
            certification_fraction=float(split["certification_fraction"]),
            audit_fraction=float(split["audit_fraction"]),
            min_gap_sec_exclusive=float(pseudo["min_gap_sec_exclusive"]),
            clean_max_other_bbox_iou_exclusive=float(
                appearance["clean_max_other_bbox_iou_exclusive"]
            ),
            min_clean_samples_per_side=int(
                appearance["min_clean_samples_per_side"]
            ),
            outlier_medoid_cosine=float(appearance["outlier_medoid_cosine"]),
            outlier_support_cosine=float(appearance["outlier_support_cosine"]),
            new_prototype_cosine=float(appearance["new_prototype_cosine"]),
            min_side_internal_cosine_p10=float(
                appearance["min_side_internal_cosine_p10"]
            ),
            high_overlap_iou_threshold=float(
                appearance["clean_max_other_bbox_iou_exclusive"]
            ),
        )
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ContractError("S05A persisted stable-gallery policy is invalid") from exc


def _reject_path_overlap(output_dir: Path, inputs: Sequence[Path]) -> None:
    output = output_dir.resolve()
    for raw in inputs:
        item = raw.resolve()
        if output == item or output in item.parents or item in output.parents:
            raise ContractError(f"S05 proposal output/input paths overlap: {output}, {item}")


def _fingerprint_map(
    records: Sequence[FileFingerprint], label: str
) -> dict[str, FileFingerprint]:
    result: dict[str, FileFingerprint] = {}
    for record in records:
        path = str(Path(record.path).resolve())
        if path != record.path or path in result:
            raise ContractError(f"{label} has non-canonical or duplicate paths")
        result[path] = record
    return result


def _validate_input_alignment(
    production: Any,
    stable: S04FinalizedBundle,
    long_runtime: S05LongCalibrationBundle,
    config: LongProposalConfig,
) -> None:
    if stable.success_marker.get("config_hash") != config.expected_s04_finalize_config_hash:
        raise ContractError("S05 proposal S04 config hash differs")
    if long_runtime.config_hash != config.expected_long_calibration_config_hash:
        raise ContractError("S05 proposal long-calibration config hash differs")
    if (
        len(stable.stable_ids) != config.expected_stable_track_count
        or len(stable.det_ids) != config.expected_detection_count
    ):
        raise ContractError("S05 proposal fixed S04 counts differ")
    if not long_runtime.model_enabled or long_runtime.confirmed_enabled:
        raise ContractError(
            "S05 proposal requires enabled long model with confirmation disabled"
        )
    selected = long_runtime.selected_gate_evidence
    if (
        selected.executable
        or selected.certified
        or selected.probability_threshold is None
        or selected.margin_threshold is None
    ):
        raise ContractError("S05 proposal selected-gate evidence policy differs")
    clips = tuple(map(str, production.calibration_input.timeline_clip_ids))
    if clips != config.clip_order:
        raise ContractError(f"S05 proposal clip order differs: {clips!r}")
    micros = np.asarray(production.calibration_input.parent_micro_ids, dtype=np.int64)
    if not np.array_equal(stable.micro_ids, np.sort(micros)):
        raise ContractError("S05 proposal production/S04 micro IDs differ")
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
        raise ContractError("S05 proposal production/S04 detection bijection differs")
    recorded = _fingerprint_map(long_runtime.input_fingerprints, "S05A inputs")
    for current in (*production.input_fingerprints, *stable.input_fingerprints):
        expected = recorded.get(current.path)
        if expected != current:
            raise ContractError(
                f"S05A calibration was not built from current upstream bytes: {current.path}"
            )


def _read_parquet(path: Path, schema: pa.Schema, label: str) -> list[dict[str, Any]]:
    try:
        parquet = pq.ParquetFile(path)
        if not parquet.schema_arrow.equals(schema, check_metadata=False):
            raise ContractError(f"{label} schema mismatch: {path}")
        return pq.read_table(path).to_pylist()
    except ContractError:
        raise
    except (OSError, pa.ArrowException, TypeError, ValueError) as exc:
        raise ContractError(f"cannot read {label} {path}: {exc}") from exc


def _project_rows(
    rows: Sequence[Mapping[str, Any]], schema: pa.Schema, label: str
) -> list[dict[str, Any]]:
    try:
        return pa.Table.from_pylist([dict(row) for row in rows], schema=schema).to_pylist()
    except (pa.ArrowException, TypeError, ValueError) as exc:
        raise ContractError(f"cannot canonicalize {label}: {exc}") from exc


def _expected_candidate_id(source_id: int, target_id: int) -> str:
    return f"s05c-{source_id:06d}-{target_id:06d}"


_GALLERY_PROVENANCE_SUFFIXES = (
    "sample_ids",
    "gallery_det_ids",
    "embedding_rows",
    "medoid_sample_id",
    "appearance_quality",
    "internal_cosine_p10",
    "internal_cosine_p50",
    "internal_cosine_min",
    "gallery_num_input_samples",
    "gallery_num_overlap_rejected",
    "gallery_num_review_excluded",
    "gallery_num_local_outliers",
    "gallery_max_other_bbox_iou",
    "gallery_max_clean_other_bbox_iou",
)


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"S05 proposal {label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ContractError(f"S05 proposal {label} must be finite")
    return result


def _validate_gallery_provenance(
    row: Mapping[str, Any], prefix: str, *, present: bool
) -> None:
    values = {suffix: row[f"{prefix}_{suffix}"] for suffix in _GALLERY_PROVENANCE_SUFFIXES}
    if not present:
        if any(value is not None for value in values.values()):
            raise ContractError(f"S05 missing {prefix} gallery retained provenance")
        return
    if any(value is None for value in values.values()):
        raise ContractError(f"S05 present {prefix} gallery lacks provenance")
    sample_ids = list(values["sample_ids"])
    det_ids = list(values["gallery_det_ids"])
    embedding_rows = list(values["embedding_rows"])
    if (
        len(sample_ids) < 3
        or not (len(sample_ids) == len(det_ids) == len(embedding_rows))
        or len(sample_ids) != len(set(map(int, sample_ids)))
        or len(det_ids) != len(set(map(int, det_ids)))
        or len(embedding_rows) != len(set(map(int, embedding_rows)))
        or int(values["medoid_sample_id"]) not in set(map(int, sample_ids))
    ):
        raise ContractError(f"S05 {prefix} gallery identifier provenance differs")
    for suffix in (
        "appearance_quality",
        "internal_cosine_p10",
        "internal_cosine_p50",
        "internal_cosine_min",
        "gallery_max_other_bbox_iou",
        "gallery_max_clean_other_bbox_iou",
    ):
        number = _finite_number(values[suffix], f"{prefix}_{suffix}")
        if suffix == "appearance_quality" and not 0.0 <= number <= 1.0:
            raise ContractError(f"S05 {prefix} gallery quality is outside [0,1]")
        if "cosine" in suffix and not -1.0 <= number <= 1.0:
            raise ContractError(f"S05 {prefix} gallery cosine is outside [-1,1]")
        if "bbox_iou" in suffix and not 0.0 <= number <= 1.0:
            raise ContractError(f"S05 {prefix} gallery IoU is outside [0,1]")
    for suffix in (
        "gallery_num_input_samples",
        "gallery_num_overlap_rejected",
        "gallery_num_review_excluded",
        "gallery_num_local_outliers",
    ):
        value = values[suffix]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ContractError(f"S05 {prefix}_{suffix} must be non-negative")
    if int(values["gallery_num_input_samples"]) < len(sample_ids):
        raise ContractError(f"S05 {prefix} gallery input count is too small")


def _validate_rows(
    candidates: Sequence[Mapping[str, Any]],
    proposals: Sequence[Mapping[str, Any]],
    long_runtime: S05LongCalibrationBundle,
    *,
    stable_count: int,
) -> dict[str, Any]:
    provisional = long_runtime.thresholds.provisional_threshold
    selected = long_runtime.selected_gate_evidence
    if provisional is None or selected.probability_threshold is None or selected.margin_threshold is None:
        raise ContractError("S05 proposal runtime thresholds are unavailable")
    if isinstance(stable_count, bool) or not isinstance(stable_count, int) or stable_count < 1:
        raise ContractError("S05 proposal stable count is invalid")
    candidate_ids: set[str] = set()
    candidate_map: dict[str, Mapping[str, Any]] = {}
    previous = ""
    for row in candidates:
        candidate_id = str(row["candidate_id"])
        source_id, target_id = int(row["source_stable_id"]), int(row["target_stable_id"])
        if (
            candidate_id <= previous
            or candidate_id in candidate_ids
            or candidate_id != _expected_candidate_id(source_id, target_id)
            or source_id == target_id
            or not 0 <= source_id < stable_count
            or not 0 <= target_id < stable_count
        ):
            raise ContractError("S05 candidate IDs/order are invalid")
        previous = candidate_id
        candidate_ids.add(candidate_id)
        candidate_map[candidate_id] = row
        gap = float(row["target_start_time_sec"] - row["source_end_time_sec"])
        for name in (
            "source_start_time_sec",
            "source_end_time_sec",
            "target_start_time_sec",
            "target_end_time_sec",
            "temporal_gap_sec",
        ):
            _finite_number(row[name], name)
        if (
            gap <= 5.0
            or not math.isclose(
                float(row["temporal_gap_sec"]), gap, rel_tol=0.0, abs_tol=2e-6
            )
            or row["temporally_nonoverlapping"] is not True
            or not (
                bool(row["selected_by_appearance_topk"])
                or bool(row["selected_by_temporal_nearest"])
            )
        ):
            raise ContractError("S05 candidate temporal/retrieval policy differs")
        if (
            float(row["source_start_time_sec"]) > float(row["source_end_time_sec"])
            or float(row["target_start_time_sec"]) > float(row["target_end_time_sec"])
            or int(row["source_start_global_frame"]) > int(row["source_end_global_frame"])
            or int(row["target_start_global_frame"]) > int(row["target_end_global_frame"])
            or not row["source_constituent_micro_ids"]
            or not row["target_constituent_micro_ids"]
        ):
            raise ContractError("S05 candidate stable-path provenance is invalid")
        if (
            float(row["provisional_threshold"]) != float(provisional)
            or float(row["selected_probability_threshold"])
            != float(selected.probability_threshold)
            or float(row["selected_margin_threshold"])
            != float(selected.margin_threshold)
        ):
            raise ContractError("S05 candidate thresholds differ from S05A")
        if any(bool(row[name]) for name in ("selected_by_solver", "confirmed", "merge_applied")):
            raise ContractError("S05 proposal candidate crossed the no-merge boundary")
        if row["decision"] not in {"reject", "provisional"}:
            raise ContractError("S05 proposal candidate has an invalid decision")
        if not isinstance(row["decision_reason"], str) or not row["decision_reason"]:
            raise ContractError("S05 proposal candidate lacks a decision reason")
        probability = row["model_probability"]
        if probability is not None:
            probability = _finite_number(probability, "model_probability")
            if not 0.0 <= probability <= 1.0:
                raise ContractError("S05 model probability is outside [0,1]")
        probability_pass = probability is not None and float(probability) >= float(provisional)
        selected_probability_pass = (
            probability is not None
            and float(probability) >= float(selected.probability_threshold)
        )
        out_margin, in_margin = row["best_margin_out"], row["best_margin_in"]
        for name, value in (
            ("best_margin_out", out_margin),
            ("best_margin_in", in_margin),
        ):
            if value is not None:
                _finite_number(value, name)
        for name in ("appearance_rank_out", "appearance_rank_in"):
            value = row[name]
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 1
            ):
                raise ContractError(f"S05 {name} must be a positive integer or null")
        expected_margin = (
            min(float(out_margin), float(in_margin))
            if out_margin is not None and in_margin is not None
            else None
        )
        actual_margin = row["candidate_margin"]
        if (actual_margin is None) != (expected_margin is None) or (
            actual_margin is not None
            and not math.isclose(
                float(actual_margin), float(expected_margin), rel_tol=0.0, abs_tol=1e-12
            )
        ):
            raise ContractError("S05 candidate bidirectional margin differs")
        margin_pass = expected_margin is not None and expected_margin >= float(
            selected.margin_threshold
        )
        selected_pass = selected_probability_pass and margin_pass and not bool(
            row["high_overlap"]
        )
        if (
            bool(row["passes_provisional_threshold"]) != probability_pass
            or bool(row["passes_selected_probability_gate"])
            != selected_probability_pass
            or bool(row["passes_selected_margin_gate"]) != margin_pass
            or bool(row["passes_selected_gate"]) != selected_pass
            or bool(row["proposed_for_review"]) != probability_pass
            or (row["decision"] == "provisional") != probability_pass
        ):
            raise ContractError("S05 candidate gate evidence differs")
        source_gallery = bool(row["source_gallery_present"])
        target_gallery = bool(row["target_gallery_present"])
        _validate_gallery_provenance(row, "source", present=source_gallery)
        _validate_gallery_provenance(row, "target", present=target_gallery)
        if source_gallery and target_gallery:
            for left, right, label in (
                (row["source_sample_ids"], row["target_sample_ids"], "sample IDs"),
                (
                    row["source_gallery_det_ids"],
                    row["target_gallery_det_ids"],
                    "detection IDs",
                ),
                (
                    row["source_embedding_rows"],
                    row["target_embedding_rows"],
                    "embedding rows",
                ),
            ):
                if set(map(int, left)) & set(map(int, right)):
                    raise ContractError(f"S05 candidate galleries share {label}")
        expected_appearance = bool(
            source_gallery
            and target_gallery
            and not row["source_endpoint_review_excluded"]
            and not row["target_endpoint_review_excluded"]
        )
        if bool(row["appearance_present"]) != expected_appearance:
            raise ContractError("S05 candidate appearance/gating availability differs")
        gallery_scores = [
            row[name]
            for name in (
                "gallery_score_max",
                "gallery_score_top3",
                "gallery_score_src_to_dst",
                "gallery_score_dst_to_src",
                "gallery_score_mutual",
            )
        ]
        if source_gallery and target_gallery:
            if any(value is None for value in gallery_scores):
                raise ContractError("S05 available galleries lack retrieval scores")
            for value in gallery_scores:
                score = _finite_number(value, "gallery score")
                if not -1.0 <= score <= 1.0:
                    raise ContractError("S05 gallery score is outside [-1,1]")
        elif any(value is not None for value in gallery_scores):
            raise ContractError("S05 missing gallery unexpectedly has retrieval scores")

        feature_list = [row[name] for name in LONG_FEATURE_SCHEMA]
        if expected_appearance:
            if any(value is None for value in feature_list):
                raise ContractError("S05 present appearance lacks model features")
            if probability is None or row["model_raw_score"] is None:
                raise ContractError("S05 present appearance lacks model scores")
            features = {
                name: _finite_number(row[name], name) for name in LONG_FEATURE_SCHEMA
            }
            raw_score = _finite_number(row["model_raw_score"], "model_raw_score")
            rescored = long_runtime.scorer.score_features(
                features,
                appearance_present=True,
                high_overlap=bool(row["high_overlap"]),
                candidate_margin=expected_margin,
            )
            if (
                rescored.probability is None
                or rescored.raw_score is None
                or not math.isclose(
                    float(probability), rescored.probability, rel_tol=0.0, abs_tol=1e-12
                )
                or not math.isclose(
                    raw_score, rescored.raw_score, rel_tol=0.0, abs_tol=1e-12
                )
                or row["decision"] != rescored.decision
            ):
                raise ContractError("S05 persisted model score/decision differs on rescore")
            expected_reason = (
                "selected_gate_evidence_uncertified"
                if probability_pass and selected_pass
                else "provisional_high_overlap"
                if probability_pass and bool(row["high_overlap"])
                else "provisional_threshold_met"
                if probability_pass
                else rescored.reason or "below_provisional_threshold"
            )
            if (
                not math.isclose(float(row["gap_sec"]), gap, rel_tol=0.0, abs_tol=2e-6)
                or not math.isclose(
                    float(row["gallery_score_max"]),
                    float(row["prototype_cosine_max"]),
                    rel_tol=0.0,
                    abs_tol=2e-6,
                )
                or not math.isclose(
                    float(row["gallery_score_top3"]),
                    float(row["prototype_cosine_top3_mean"]),
                    rel_tol=0.0,
                    abs_tol=2e-6,
                )
                or not math.isclose(
                    float(row["gallery_score_mutual"]),
                    float(row["mutual_prototype_score"]),
                    rel_tol=0.0,
                    abs_tol=2e-6,
                )
                or not math.isclose(
                    float(row["gallery_score_mutual"]),
                    0.5
                    * (
                        float(row["gallery_score_src_to_dst"])
                        + float(row["gallery_score_dst_to_src"])
                    ),
                    rel_tol=0.0,
                    abs_tol=2e-6,
                )
            ):
                raise ContractError("S05 candidate gallery/model features differ")
        else:
            if any(value is not None for value in feature_list) or probability is not None or row["model_raw_score"] is not None:
                raise ContractError("S05 missing appearance unexpectedly has model values")
            expected_reason = (
                "review_excluded_endpoint"
                if row["source_endpoint_review_excluded"]
                or row["target_endpoint_review_excluded"]
                else "source_target_gallery_missing"
                if not source_gallery and not target_gallery
                else "source_gallery_missing"
                if not source_gallery
                else "target_gallery_missing"
            )
        if row["decision_reason"] != expected_reason:
            raise ContractError("S05 candidate decision reason differs")

    proposal_ids: set[str] = set()
    proposed_candidates: set[str] = set()
    previous = ""
    for row in proposals:
        proposal_id = str(row["proposal_id"])
        candidate_id = str(row["candidate_id"])
        if (
            proposal_id <= previous
            or proposal_id in proposal_ids
            or proposal_id != f"s05p-{candidate_id[5:]}"
            or candidate_id not in candidate_map
            or row["review_status"] != "pending"
            or row["decision"] != "provisional"
        ):
            raise ContractError("S05 review proposal identity/status differs")
        previous = proposal_id
        proposal_ids.add(proposal_id)
        proposed_candidates.add(candidate_id)
        source = candidate_map[candidate_id]
        for name in LONG_LINK_PROPOSALS_SCHEMA.names:
            if name in {"proposal_id", "review_status", "evidence_status"}:
                continue
            left, right = row[name], source[name]
            if isinstance(left, float) and isinstance(right, float):
                if not math.isclose(left, right, rel_tol=0.0, abs_tol=0.0):
                    raise ContractError("S05 proposal/candidate float differs")
            elif left != right:
                raise ContractError("S05 proposal/candidate row differs")
        expected_evidence = (
            "selected_gate_uncertified"
            if source["passes_selected_gate"] is True
            else "provisional"
        )
        if row["evidence_status"] != expected_evidence:
            raise ContractError("S05 proposal evidence status differs")
    expected_proposals = {
        str(row["candidate_id"])
        for row in candidates
        if row["decision"] == "provisional" and row["proposed_for_review"] is True
    }
    if proposed_candidates != expected_proposals:
        raise ContractError("S05 proposal table does not exactly cover provisional rows")
    return {
        "num_candidates": len(candidates),
        "num_proposals": len(proposals),
        "num_rejects": len(candidates) - len(proposals),
        "num_confirmed": 0,
        "num_solver_selected": 0,
        "num_merges": 0,
    }


def _clip_distribution(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return dict(
        sorted(
            Counter(
                f"{row['source_end_clip_id']}->{row['target_start_clip_id']}"
                for row in rows
            ).items()
        )
    )


def _report(
    candidates: Sequence[Mapping[str, Any]],
    proposals: Sequence[Mapping[str, Any]],
    config: LongProposalConfig,
    config_hash: str,
    long_runtime: S05LongCalibrationBundle,
    input_fingerprints: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    selected = long_runtime.selected_gate_evidence
    return {
        "schema_version": "1.0",
        "stage": _STAGE,
        "config_hash": config_hash,
        "execution_mode": "long_proposal_only",
        "candidate_identity": "finalized_s04_stable_path_id",
        "retrieval": {
            "method": "exact_cosine",
            "approximate_index_used": False,
            "appearance_topk": config.appearance_topk,
            "temporal_nearest_k": config.temporal_nearest_k,
            "candidate_union": config.candidate_union,
            "future_only": True,
            "min_gap_sec_exclusive": config.min_gap_sec_exclusive,
            "non_overlapping_required": True,
            "representative_prototypes_per_path_max": config.max_representative_prototypes,
        },
        "long_calibration": {
            "config_hash": long_runtime.config_hash,
            "model_enabled": long_runtime.model_enabled,
            "confirmed_enabled": long_runtime.confirmed_enabled,
            "provisional_threshold": long_runtime.thresholds.provisional_threshold,
            "confirmed_disabled_reason": long_runtime.thresholds.confirmed_disabled_reason,
            "selected_gate_evidence": {
                "probability_threshold": selected.probability_threshold,
                "margin_threshold": selected.margin_threshold,
                "certified": selected.certified,
                "executable": selected.executable,
                "false_accepts": selected.false_accepts,
                "certification_hard_negative_count": selected.certification_hard_negative_count,
                "false_accept_upper": selected.false_accept_upper,
                "certification_true_accepts": selected.certification_true_accepts,
            },
        },
        "safety_boundary": {
            "selected_gate_is_review_priority_only": True,
            "automatic_merge_allowed": False,
            "confirmed_links_allowed": False,
            "solver_used": False,
            "path_cover_used": False,
            "human_labels_applied": False,
            "global_identity_artifacts_emitted": False,
            "population_prior_used": False,
            "population_threshold_adapted": False,
            "soft_population_max": 57,
            "population_overflow_allowance": 5,
            "population_check_deferred_until_global_paths_exist": True,
        },
        "counts": {
            "stable_paths": config.expected_stable_track_count,
            "candidates": len(candidates),
            "selected_by_appearance_topk": sum(
                bool(row["selected_by_appearance_topk"]) for row in candidates
            ),
            "selected_by_temporal_nearest": sum(
                bool(row["selected_by_temporal_nearest"]) for row in candidates
            ),
            "selected_by_both": sum(
                bool(row["selected_by_appearance_topk"])
                and bool(row["selected_by_temporal_nearest"])
                for row in candidates
            ),
            "appearance_present": sum(bool(row["appearance_present"]) for row in candidates),
            "appearance_missing": sum(not bool(row["appearance_present"]) for row in candidates),
            "high_overlap": sum(bool(row["high_overlap"]) for row in candidates),
            "provisional_review_proposals": len(proposals),
            "selected_gate_review_priority": sum(
                bool(row["passes_selected_gate"]) for row in candidates
            ),
            "confirmed": 0,
            "solver_selected": 0,
            "merges": 0,
        },
        "candidate_decision_reasons": dict(
            sorted(Counter(str(row["decision_reason"]) for row in candidates).items())
        ),
        "candidate_clip_distribution": _clip_distribution(candidates),
        "proposal_clip_distribution": _clip_distribution(proposals),
        "input_fingerprints": list(input_fingerprints),
    }


def _validate_completed_output(
    directory: Path,
    success: Mapping[str, Any],
    config: LongProposalConfig,
    config_payload: Mapping[str, Any],
    config_hash: str,
    long_runtime: S05LongCalibrationBundle,
    input_fingerprints: Sequence[Mapping[str, Any]],
    expected_candidates: Sequence[Mapping[str, Any]],
    expected_proposals: Sequence[Mapping[str, Any]],
) -> None:
    expected_keys = {
        "schema_version",
        "stage",
        "config_hash",
        "execution_mode",
        "automatic_merge_allowed",
        "confirmed_links_allowed",
        "solver_used",
        "path_cover_used",
        "num_merges",
        "input_fingerprints",
        "output_fingerprints",
        "stats",
        "elapsed_sec",
    }
    if not isinstance(success, Mapping) or set(success) != expected_keys:
        raise ContractError("completed S05 proposal success marker fields differ")
    consumed_paths = (
        directory / config.artifacts.success,
        *(directory / name for name in sorted(_artifact_names(config).values())),
    )
    before = tuple(fingerprint_file(path).as_dict() for path in consumed_paths)
    persisted_success = _read_json(
        directory / config.artifacts.success, "S05 proposal success marker"
    )
    if persisted_success != dict(success):
        raise ContractError("completed S05 proposal marker changed before validation")
    if (
        success["schema_version"] != "1.0"
        or success["stage"] != _STAGE
        or success["config_hash"] != config_hash
        or success["execution_mode"] != "long_proposal_only"
        or success["automatic_merge_allowed"] is not False
        or success["confirmed_links_allowed"] is not False
        or success["solver_used"] is not False
        or success["path_cover_used"] is not False
        or type(success["num_merges"]) is not int
        or success["num_merges"] != 0
        or success["input_fingerprints"] != list(input_fingerprints)
    ):
        raise ContractError("completed S05 proposal marker policy/input differs")
    elapsed = success["elapsed_sec"]
    if isinstance(elapsed, bool) or not isinstance(elapsed, (int, float)) or not math.isfinite(float(elapsed)) or float(elapsed) < 0.0:
        raise ContractError("completed S05 proposal elapsed time is invalid")
    stats_payload = success["stats"]
    expected_stat_keys = {
        "num_candidates",
        "num_proposals",
        "num_rejects",
        "num_confirmed",
        "num_solver_selected",
        "num_merges",
    }
    if (
        not isinstance(stats_payload, dict)
        or set(stats_payload) != expected_stat_keys
        or any(type(value) is not int or value < 0 for value in stats_payload.values())
        or any(
            stats_payload[name] != 0
            for name in ("num_confirmed", "num_solver_selected", "num_merges")
        )
    ):
        raise ContractError("completed S05 proposal marker stats are invalid")
    names = _artifact_names(config)
    expected_files = set(names.values())
    actual_files: set[str] = set()
    for path in directory.iterdir():
        try:
            mode = path.lstat().st_mode
        except OSError as exc:
            raise ContractError(f"cannot inspect S05 proposal artifact {path}: {exc}") from exc
        if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
            raise ContractError(
                f"completed S05 proposal contains a non-regular artifact: {path}"
            )
        if path.name != config.artifacts.success:
            actual_files.add(path.name)
    if actual_files != expected_files:
        raise ContractError("completed S05 proposal artifact tree differs")
    records = success["output_fingerprints"]
    if not isinstance(records, list):
        raise ContractError("completed S05 proposal lacks output fingerprints")
    if [item.get("path") for item in records if isinstance(item, dict)] != sorted(
        expected_files
    ):
        raise ContractError("completed S05 proposal output fingerprints are not ordered")
    for item in records:
        if (
            not isinstance(item, dict)
            or set(item) != {"path", "size_bytes", "sha256"}
            or isinstance(item["size_bytes"], bool)
            or not isinstance(item["size_bytes"], int)
            or item["size_bytes"] < 0
            or not isinstance(item["sha256"], str)
            or len(item["sha256"]) != 64
            or any(character not in "0123456789abcdef" for character in item["sha256"])
        ):
            raise ContractError("completed S05 proposal output fingerprint is invalid")
    by_name = {str(item.get("path")): item for item in records if isinstance(item, dict)}
    if set(by_name) != expected_files or len(by_name) != len(records):
        raise ContractError("completed S05 proposal output fingerprints differ")
    for name, expected in by_name.items():
        if _output_fingerprint(directory / name, directory) != expected:
            raise ContractError(f"completed S05 proposal artifact changed: {name}")
    persisted_config, persisted_payload, persisted_hash = load_long_proposal_config(
        directory / names["config"]
    )
    persisted_config = replace(
        persisted_config,
        expected_stable_track_count=config.expected_stable_track_count,
    )
    if persisted_config != config or persisted_payload != dict(config_payload) or persisted_hash != config_hash:
        raise ContractError("completed S05 proposal effective config differs")
    candidates = _read_parquet(
        directory / names["candidates"], LONG_CANDIDATE_EDGES_SCHEMA, "S05 candidates"
    )
    proposals = _read_parquet(
        directory / names["proposals"], LONG_LINK_PROPOSALS_SCHEMA, "S05 proposals"
    )
    canonical_expected_candidates = _project_rows(
        expected_candidates, LONG_CANDIDATE_EDGES_SCHEMA, "expected S05 candidates"
    )
    canonical_expected_proposals = _project_rows(
        expected_proposals, LONG_LINK_PROPOSALS_SCHEMA, "expected S05 proposals"
    )
    if candidates != canonical_expected_candidates or proposals != canonical_expected_proposals:
        raise ContractError(
            "completed S05 proposal rows differ from exact retrieval/model recomputation"
        )
    stats = _validate_rows(
        candidates,
        proposals,
        long_runtime,
        stable_count=config.expected_stable_track_count,
    )
    if success["stats"] != stats:
        raise ContractError("completed S05 proposal stats differ")
    manifest = _read_json(directory / names["manifest"], "S05 review manifest")
    if manifest != long_review_manifest(proposals):
        raise ContractError("completed S05 review manifest differs")
    report = _read_json(directory / names["report"], "S05 proposal report")
    if report != _report(
        candidates, proposals, config, config_hash, long_runtime, input_fingerprints
    ):
        raise ContractError("completed S05 proposal report differs")
    _verify_unchanged(input_fingerprints)
    after = tuple(fingerprint_file(path).as_dict() for path in consumed_paths)
    if before != after:
        raise ContractError("S05 proposal artifacts changed while being validated")


def _generate_candidate_outputs(
    production: Any,
    stable: S04FinalizedBundle,
    long_runtime: S05LongCalibrationBundle,
    config: LongProposalConfig,
    *,
    logger: LogFn,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Rebuild exact candidates and scores for both production and resume audit."""

    stable_input = StableLongInput(
        production=production,
        micro_to_stable=stable.micro_to_stable,
        micro_order_in_stable=stable.micro_order_in_stable,
    )
    logger("[s05-propose] rebuilding clean whole-path galleries from S02 samples")
    store = StablePathFeatureStore(
        stable_input, _stable_policy_from_runtime(long_runtime)
    )
    descriptors = store.retrieval_descriptors(
        logger=logger, progress_interval_sec=config.progress_interval_sec
    )
    if any(
        descriptor.prototypes is not None
        and len(descriptor.prototypes) > config.max_representative_prototypes
        for descriptor in descriptors
    ):
        raise ContractError("S05 retrieval gallery exceeds calibrated prototype count")
    logger("[s05-propose] exact cosine retrieval over legal future paths")
    candidates = enumerate_long_candidates(
        descriptors,
        appearance_topk=config.appearance_topk,
        temporal_nearest_k=config.temporal_nearest_k,
        min_gap_sec_exclusive=config.min_gap_sec_exclusive,
        logger=logger,
        progress_interval_sec=config.progress_interval_sec,
    )
    selected = long_runtime.selected_gate_evidence
    provisional_threshold = long_runtime.thresholds.provisional_threshold
    if (
        selected.probability_threshold is None
        or selected.margin_threshold is None
        or provisional_threshold is None
    ):
        raise ContractError("S05 proposal scoring thresholds unexpectedly disappeared")
    logger(f"[s05-propose] scoring {len(candidates):,} proposal candidates")
    scored = score_long_candidates(
        candidates,
        store,
        long_runtime.scorer,
        provisional_threshold=provisional_threshold,
        selected_probability_threshold=selected.probability_threshold,
        selected_margin_threshold=selected.margin_threshold,
        worker_count=config.worker_count,
        logger=logger,
        progress_interval_sec=config.progress_interval_sec,
    )
    proposals = build_long_proposals(scored)
    return scored, proposals


def validate_s05_proposal_input_alignment(
    production: Any,
    stable: S04FinalizedBundle,
    long_runtime: S05LongCalibrationBundle,
    config: LongProposalConfig,
) -> None:
    """Public strict alignment check shared with the S05 finalizer."""

    _validate_input_alignment(production, stable, long_runtime, config)


def validate_s05_proposal_rows(
    candidates: Sequence[Mapping[str, Any]],
    proposals: Sequence[Mapping[str, Any]],
    long_runtime: S05LongCalibrationBundle,
    *,
    stable_count: int,
) -> dict[str, Any]:
    """Public row-level proposal contract shared with the S05 finalizer."""

    return _validate_rows(
        candidates,
        proposals,
        long_runtime,
        stable_count=stable_count,
    )


def generate_s05_candidate_outputs(
    production: Any,
    stable: S04FinalizedBundle,
    long_runtime: S05LongCalibrationBundle,
    config: LongProposalConfig,
    *,
    logger: LogFn,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Recompute the exact proposal graph for downstream byte/row audits."""

    return _generate_candidate_outputs(
        production,
        stable,
        long_runtime,
        config,
        logger=logger,
    )


def run_s05_propose(
    ingest_dir: Path,
    microtrack_dir: Path,
    appearance_dir: Path,
    stable_dir: Path,
    long_calibration_dir: Path,
    config_path: Path,
    output_dir: Path,
    *,
    logger: LogFn = log,
) -> dict[str, Any]:
    """Generate exact long candidates and pending review proposals, never links."""

    if not callable(logger):
        raise ContractError("S05 proposal logger must be callable")
    started = time.monotonic()
    ingest_dir, microtrack_dir, appearance_dir, stable_dir, long_calibration_dir, config_path, output_dir = (
        path.resolve()
        for path in (
            ingest_dir,
            microtrack_dir,
            appearance_dir,
            stable_dir,
            long_calibration_dir,
            config_path,
            output_dir,
        )
    )
    _reject_path_overlap(
        output_dir,
        (
            ingest_dir,
            microtrack_dir,
            appearance_dir,
            stable_dir,
            long_calibration_dir,
            config_path,
        ),
    )
    config_fingerprint = fingerprint_file(config_path)
    config, config_payload, config_hash = load_long_proposal_config(config_path)
    if fingerprint_file(config_path) != config_fingerprint:
        raise ContractError("S05 proposal config changed while being loaded")
    logger("[s05-propose] strict-loading immutable S00/S01/S02 inputs")
    production = load_production_inputs(
        ingest_dir, microtrack_dir, appearance_dir, logger=logger
    )
    logger("[s05-propose] strict-loading finalized S04 stable paths")
    stable = load_s04_finalized(stable_dir)
    observed_stable_count = len(stable.stable_ids)
    if config.expected_stable_track_count not in (None, observed_stable_count):
        raise ContractError("S05 proposal configured stable count differs")
    config = replace(
        config, expected_stable_track_count=observed_stable_count
    )
    logger("[s05-propose] strict-loading S05A long calibration")
    long_success_fingerprint = fingerprint_file(
        long_calibration_dir / "_SUCCESS.json"
    )
    long_runtime = load_s05_long_calibration(long_calibration_dir)
    if (
        fingerprint_file(long_calibration_dir / "_SUCCESS.json")
        != long_success_fingerprint
    ):
        raise ContractError("S05A success marker changed while being loaded")
    _validate_input_alignment(production, stable, long_runtime, config)

    input_fingerprints = _normalize_fingerprints(
        [
            config_fingerprint,
            *production.input_fingerprints,
            *stable.input_fingerprints,
            *long_runtime.input_fingerprints,
            *long_runtime.output_fingerprints,
            long_success_fingerprint,
        ]
    )
    success_path = output_dir / config.artifacts.success
    existing_success: Mapping[str, Any] | None = None
    if success_path.is_file():
        payload = _read_json(success_path, "S05 proposal success marker")
        if not isinstance(payload, Mapping):
            raise ContractError("S05 proposal success marker must be an object")
        existing_success = payload
    elif output_dir.exists():
        if not output_dir.is_dir():
            raise ContractError(f"S05 proposal output path is not a directory: {output_dir}")
        if any(output_dir.iterdir()):
            raise ContractError(
                f"S05 proposal output is non-empty without _SUCCESS.json: {output_dir}"
            )
    scored, proposals = _generate_candidate_outputs(
        production, stable, long_runtime, config, logger=logger
    )
    if existing_success is not None:
        _validate_completed_output(
            output_dir,
            existing_success,
            config,
            config_payload,
            config_hash,
            long_runtime,
            input_fingerprints,
            scored,
            proposals,
        )
        logger(f"[s05-propose] already complete and fully revalidated: {success_path}")
        return dict(existing_success)
    stats = _validate_rows(
        scored,
        proposals,
        long_runtime,
        stable_count=config.expected_stable_track_count,
    )
    report = _report(
        scored, proposals, config, config_hash, long_runtime, input_fingerprints
    )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = output_dir.parent / f".{output_dir.name}.staging-{os.getpid()}"
    if staging.exists():
        raise ContractError(f"S05 proposal staging directory already exists: {staging}")
    staging.mkdir(parents=False)
    try:
        names = _artifact_names(config)
        _write_parquet(
            staging / names["candidates"],
            scored,
            LONG_CANDIDATE_EDGES_SCHEMA,
            config.parquet_compression,
        )
        _write_parquet(
            staging / names["proposals"],
            proposals,
            LONG_LINK_PROPOSALS_SCHEMA,
            config.parquet_compression,
        )
        _write_json(staging / names["manifest"], long_review_manifest(proposals))
        _write_json(staging / names["report"], report)
        _write_json(staging / names["config"], config_payload)
        _verify_unchanged(input_fingerprints)
        output_fingerprints = [
            _output_fingerprint(staging / name, staging)
            for name in sorted(names.values())
        ]
        success = {
            "schema_version": "1.0",
            "stage": _STAGE,
            "config_hash": config_hash,
            "execution_mode": "long_proposal_only",
            "automatic_merge_allowed": False,
            "confirmed_links_allowed": False,
            "solver_used": False,
            "path_cover_used": False,
            "num_merges": 0,
            "input_fingerprints": input_fingerprints,
            "output_fingerprints": output_fingerprints,
            "stats": stats,
            "elapsed_sec": float(time.monotonic() - started),
        }
        _write_json(staging / config.artifacts.success, success)
        _validate_completed_output(
            staging,
            success,
            config,
            config_payload,
            config_hash,
            long_runtime,
            input_fingerprints,
            scored,
            proposals,
        )
        if output_dir.exists():
            try:
                output_dir.rmdir()
            except OSError as exc:
                raise ContractError(
                    f"S05 proposal output cannot be atomically committed: {output_dir}"
                ) from exc
        os.replace(staging, output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    logger(f"[s05-propose] complete: {output_dir}")
    return success


__all__ = [
    "generate_s05_candidate_outputs",
    "run_s05_propose",
    "validate_s05_proposal_input_alignment",
    "validate_s05_proposal_rows",
]
