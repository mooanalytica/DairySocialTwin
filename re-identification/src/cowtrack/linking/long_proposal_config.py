"""Strict, independent configuration contract for S05 proposal-only retrieval."""

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
from cowtrack.linking.features import LONG_FEATURE_SCHEMA


@dataclass(frozen=True)
class LongProposalArtifacts:
    candidate_edges: str
    proposals: str
    review_manifest: str
    report: str
    effective_config: str
    success: str


@dataclass(frozen=True)
class LongProposalConfig:
    schema_version: str
    random_seed: int
    execution_mode: str
    global_merge_allowed: bool
    solver_allowed: bool
    path_cover_allowed: bool
    confirmed_links_allowed: bool
    human_labels_applied: bool
    expected_s04_finalize_config_hash: str
    expected_long_calibration_config_hash: str
    expected_stable_track_count: int | None
    expected_detection_count: int
    clip_order: tuple[str, ...]
    require_long_model_enabled: bool
    require_long_confirmed_enabled: bool
    retrieval_method: str
    approximate_index_allowed: bool
    retrieval_direction: str
    candidate_union: str
    appearance_topk: int
    temporal_nearest_k: int
    min_gap_sec_exclusive: float
    require_non_overlapping: bool
    max_representative_prototypes: int
    deterministic_tie_break: str
    model_mode: str
    ordered_features: tuple[str, ...]
    gallery_score_method: str
    gallery_scores: tuple[str, ...]
    record_bidirectional_ranks: bool
    record_bidirectional_second_best_margins: bool
    accepted_decision_for_review: str
    selected_gate_report_only: bool
    missing_appearance_decision: str
    confirmed_decision_allowed: bool
    short_model_fallback_allowed: bool
    raw_cosine_decision_fallback_allowed: bool
    motion_only_fallback_allowed: bool
    worker_count: int
    progress_interval_sec: float
    parquet_compression: str
    deterministic_row_sort: bool
    log_flush: bool
    artifacts: LongProposalArtifacts


_FIXED_ARTIFACTS = {
    "candidate_edges": "long_candidate_edges.parquet",
    "proposals": "long_link_proposals.parquet",
    "review_manifest": "s05_review_manifest.json",
    "report": "s05_proposal_report.json",
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
        raise ContractError(f"fixed S05 proposal requires {name}={expected!r}")


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


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ContractError(f"{name} must be finite")
    return result


def load_long_proposal_config(
    path: Path,
) -> tuple[LongProposalConfig, dict[str, Any], str]:
    """Load the deliberately narrow, proposal-only S05 configuration."""

    path = path.resolve()
    if not path.is_file():
        raise ContractError(f"S05 proposal config does not exist: {path}")
    try:
        payload = yaml.safe_load(path.read_bytes())
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ContractError(f"cannot read S05 proposal config {path}: {exc}") from exc

    top = _exact(
        payload,
        {"pipeline", "inputs", "retrieval", "scoring", "runtime", "artifacts"},
        "S05 proposal",
    )
    pipeline = _exact(
        top["pipeline"],
        {
            "schema_version",
            "random_seed",
            "execution_mode",
            "global_merge_allowed",
            "solver_allowed",
            "path_cover_allowed",
            "confirmed_links_allowed",
            "human_labels_applied",
        },
        "S05 proposal.pipeline",
    )
    inputs = _exact(
        top["inputs"],
        {
            "expected_s04_finalize_config_hash",
            "expected_long_calibration_config_hash",
            "expected_stable_track_count",
            "expected_detection_count",
            "clip_order",
            "require_long_model_enabled",
            "require_long_confirmed_enabled",
        },
        "S05 proposal.inputs",
    )
    retrieval = _exact(
        top["retrieval"],
        {
            "method",
            "approximate_index_allowed",
            "direction",
            "candidate_union",
            "appearance_topk",
            "temporal_nearest_k",
            "min_gap_sec_exclusive",
            "require_non_overlapping",
            "max_representative_prototypes",
            "deterministic_tie_break",
        },
        "S05 proposal.retrieval",
    )
    scoring = _exact(
        top["scoring"],
        {
            "model_mode",
            "ordered_features",
            "gallery_score_method",
            "gallery_scores",
            "record_bidirectional_ranks",
            "record_bidirectional_second_best_margins",
            "accepted_decision_for_review",
            "selected_gate_report_only",
            "missing_appearance_decision",
            "confirmed_decision_allowed",
            "short_model_fallback_allowed",
            "raw_cosine_decision_fallback_allowed",
            "motion_only_fallback_allowed",
        },
        "S05 proposal.scoring",
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
        "S05 proposal.runtime",
    )
    artifacts = _exact(
        top["artifacts"], set(_FIXED_ARTIFACTS), "S05 proposal.artifacts"
    )

    fixed = {
        "pipeline.schema_version": (pipeline["schema_version"], "1.0"),
        "pipeline.execution_mode": (
            pipeline["execution_mode"],
            "long_proposal_only",
        ),
        "pipeline.global_merge_allowed": (
            pipeline["global_merge_allowed"],
            False,
        ),
        "pipeline.solver_allowed": (pipeline["solver_allowed"], False),
        "pipeline.path_cover_allowed": (pipeline["path_cover_allowed"], False),
        "pipeline.confirmed_links_allowed": (
            pipeline["confirmed_links_allowed"],
            False,
        ),
        "pipeline.human_labels_applied": (
            pipeline["human_labels_applied"],
            False,
        ),
        "inputs.clip_order": (
            inputs["clip_order"],
            list(EXPECTED_CLIP_ORDER),
        ),
        "inputs.require_long_model_enabled": (
            inputs["require_long_model_enabled"],
            True,
        ),
        # Current calibration is intentionally uncertified.  Proposal review
        # may proceed, but no row may be promoted to a confirmed link.
        "inputs.require_long_confirmed_enabled": (
            inputs["require_long_confirmed_enabled"],
            False,
        ),
        "retrieval.method": (retrieval["method"], "exact_cosine"),
        "retrieval.approximate_index_allowed": (
            retrieval["approximate_index_allowed"],
            False,
        ),
        "retrieval.direction": (retrieval["direction"], "future_only"),
        "retrieval.candidate_union": (
            retrieval["candidate_union"],
            "appearance_topk_or_temporal_nearest",
        ),
        "retrieval.require_non_overlapping": (
            retrieval["require_non_overlapping"],
            True,
        ),
        "retrieval.deterministic_tie_break": (
            retrieval["deterministic_tie_break"],
            "score_desc_gap_asc_stable_id_asc",
        ),
        "scoring.model_mode": (scoring["model_mode"], "long"),
        "scoring.ordered_features": (
            scoring["ordered_features"],
            list(LONG_FEATURE_SCHEMA),
        ),
        "scoring.gallery_score_method": (
            scoring["gallery_score_method"],
            "exact_all_pair_cosine",
        ),
        "scoring.gallery_scores": (
            scoring["gallery_scores"],
            ["max", "top3", "src_to_dst", "dst_to_src", "mutual"],
        ),
        "scoring.record_bidirectional_ranks": (
            scoring["record_bidirectional_ranks"],
            True,
        ),
        "scoring.record_bidirectional_second_best_margins": (
            scoring["record_bidirectional_second_best_margins"],
            True,
        ),
        "scoring.accepted_decision_for_review": (
            scoring["accepted_decision_for_review"],
            "provisional",
        ),
        "scoring.selected_gate_report_only": (
            scoring["selected_gate_report_only"],
            True,
        ),
        "scoring.missing_appearance_decision": (
            scoring["missing_appearance_decision"],
            "reject",
        ),
        "scoring.confirmed_decision_allowed": (
            scoring["confirmed_decision_allowed"],
            False,
        ),
        "scoring.short_model_fallback_allowed": (
            scoring["short_model_fallback_allowed"],
            False,
        ),
        "scoring.raw_cosine_decision_fallback_allowed": (
            scoring["raw_cosine_decision_fallback_allowed"],
            False,
        ),
        "scoring.motion_only_fallback_allowed": (
            scoring["motion_only_fallback_allowed"],
            False,
        ),
        "runtime.parquet_compression": (
            runtime["parquet_compression"],
            "zstd",
        ),
        "runtime.deterministic_row_sort": (
            runtime["deterministic_row_sort"],
            True,
        ),
        "runtime.log_flush": (runtime["log_flush"], True),
    }
    for name, (actual, expected) in fixed.items():
        _fixed(actual, expected, name)

    seed = _integer(pipeline["random_seed"], "S05 proposal.pipeline.random_seed")
    _fixed(seed, 20260710, "pipeline.random_seed")
    stable_count = _optional_integer(
        inputs["expected_stable_track_count"],
        "S05 proposal.inputs.expected_stable_track_count",
        minimum=1,
    )
    detection_count = _integer(
        inputs["expected_detection_count"],
        "S05 proposal.inputs.expected_detection_count",
        minimum=1,
    )
    if stable_count is not None and detection_count < stable_count:
        raise ContractError("S05 proposal expected counts are inconsistent")
    s04_hash = _digest(
        inputs["expected_s04_finalize_config_hash"],
        "expected_s04_finalize_config_hash",
    )
    calibration_hash = _digest(
        inputs["expected_long_calibration_config_hash"],
        "expected_long_calibration_config_hash",
    )
    appearance_topk = _integer(
        retrieval["appearance_topk"],
        "S05 proposal.retrieval.appearance_topk",
        minimum=1,
    )
    temporal_nearest_k = _integer(
        retrieval["temporal_nearest_k"],
        "S05 proposal.retrieval.temporal_nearest_k",
        minimum=1,
    )
    max_representative_prototypes = _integer(
        retrieval["max_representative_prototypes"],
        "S05 proposal.retrieval.max_representative_prototypes",
        minimum=1,
    )
    _fixed(appearance_topk, 20, "retrieval.appearance_topk")
    _fixed(temporal_nearest_k, 5, "retrieval.temporal_nearest_k")
    _fixed(
        max_representative_prototypes,
        3,
        "retrieval.max_representative_prototypes",
    )
    gap = _finite(
        retrieval["min_gap_sec_exclusive"],
        "S05 proposal.retrieval.min_gap_sec_exclusive",
    )
    if not math.isclose(gap, 5.0, rel_tol=0.0, abs_tol=1e-12):
        raise ContractError(
            "fixed S05 proposal requires retrieval.min_gap_sec_exclusive=5.0"
        )
    workers = _integer(
        runtime["worker_count"],
        "S05 proposal.runtime.worker_count",
        minimum=1,
    )
    _fixed(workers, 16, "runtime.worker_count")
    interval = _finite(
        runtime["progress_interval_sec"],
        "S05 proposal.runtime.progress_interval_sec",
    )
    if interval <= 0.0:
        raise ContractError("S05 proposal progress interval must be positive")
    for name, expected in _FIXED_ARTIFACTS.items():
        _fixed(artifacts[name], expected, f"artifacts.{name}")

    result = LongProposalConfig(
        schema_version="1.0",
        random_seed=seed,
        execution_mode="long_proposal_only",
        global_merge_allowed=False,
        solver_allowed=False,
        path_cover_allowed=False,
        confirmed_links_allowed=False,
        human_labels_applied=False,
        expected_s04_finalize_config_hash=s04_hash,
        expected_long_calibration_config_hash=calibration_hash,
        expected_stable_track_count=stable_count,
        expected_detection_count=detection_count,
        clip_order=EXPECTED_CLIP_ORDER,
        require_long_model_enabled=True,
        require_long_confirmed_enabled=False,
        retrieval_method="exact_cosine",
        approximate_index_allowed=False,
        retrieval_direction="future_only",
        candidate_union="appearance_topk_or_temporal_nearest",
        appearance_topk=appearance_topk,
        temporal_nearest_k=temporal_nearest_k,
        min_gap_sec_exclusive=gap,
        require_non_overlapping=True,
        max_representative_prototypes=max_representative_prototypes,
        deterministic_tie_break="score_desc_gap_asc_stable_id_asc",
        model_mode="long",
        ordered_features=tuple(LONG_FEATURE_SCHEMA),
        gallery_score_method="exact_all_pair_cosine",
        gallery_scores=("max", "top3", "src_to_dst", "dst_to_src", "mutual"),
        record_bidirectional_ranks=True,
        record_bidirectional_second_best_margins=True,
        accepted_decision_for_review="provisional",
        selected_gate_report_only=True,
        missing_appearance_decision="reject",
        confirmed_decision_allowed=False,
        short_model_fallback_allowed=False,
        raw_cosine_decision_fallback_allowed=False,
        motion_only_fallback_allowed=False,
        worker_count=workers,
        progress_interval_sec=interval,
        parquet_compression="zstd",
        deterministic_row_sort=True,
        log_flush=True,
        artifacts=LongProposalArtifacts(**artifacts),
    )
    canonical = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return result, payload, hashlib.sha256(canonical).hexdigest()


__all__ = [
    "LongProposalArtifacts",
    "LongProposalConfig",
    "load_long_proposal_config",
]
