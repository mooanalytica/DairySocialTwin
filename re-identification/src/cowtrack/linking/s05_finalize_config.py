"""Strict configuration contract for the fixed operator-approved S05 finalize.

This contract is deliberately dataset-specific.  It authorizes every current
``decision == 'provisional'`` proposal to enter a deterministic path-cover
solver, while requiring the original uncertified/confirmed-false evidence to
remain unchanged.  The independent 50-video review output is not an input.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
import scipy

from cowtrack.config import ContractError
from cowtrack.linking.dataset_contract import (
    EXPECTED_CLIP_ORDER,
    EXPECTED_SEQUENCE_ID,
)


@dataclass(frozen=True)
class FinalizeArtifacts:
    candidate_edges: str
    stable_to_global: str
    global_tracks: str
    det_to_global: str
    report: str
    effective_config: str
    success: str


@dataclass(frozen=True)
class S05FinalizeConfig:
    schema_version: str
    random_seed: int
    execution_mode: str
    operator_approved: bool
    global_merge_allowed: bool
    solver_allowed: bool
    path_cover_allowed: bool
    certification_claim_allowed: bool

    expected_s04_finalize_config_hash: str
    expected_long_calibration_config_hash: str
    expected_long_proposal_config_hash: str
    expected_sequence_id: str
    expected_stable_track_count: int | None
    expected_microtrack_count: int | None
    expected_detection_count: int
    expected_invalid_detections: int
    expected_frame_count: int
    expected_candidate_count: int | None
    expected_proposal_count: int | None
    clip_order: tuple[str, ...]
    frame_counts_by_clip: tuple[int, ...]
    expected_valid_detections_by_clip: tuple[int, ...]
    expected_invalid_detections_by_clip: tuple[int, ...]

    approval_strategy: str
    authorization_basis: str
    required_proposal_decision: str
    required_proposal_review_status: str
    review_output_consumed: bool
    review_labels_consumed: bool
    require_proposal_selected_by_solver_false: bool
    require_proposal_confirmed_false: bool
    require_proposal_merge_applied_false: bool
    preserve_proposal_evidence_status: bool

    solver: str
    objective: str
    candidate_policy: str
    graph_node_identity: str
    direction: str
    require_strict_non_overlap: bool
    max_incoming_links: int
    max_outgoing_links: int
    maximum_iterations: int
    evidence_cost: str
    deterministic_tie_break: str

    soft_max_global_ids: int
    overflow_allowance: int
    population_policy: str
    population_affects_solver: bool
    force_merge_allowed: bool
    threshold_adaptation_allowed: bool

    worker_count: int
    expected_python_version: str
    expected_scipy_version: str
    progress_interval_sec: float
    parquet_compression: str
    deterministic_row_sort: bool
    atomic_commit: bool
    resume_revalidate: bool
    log_flush: bool
    artifacts: FinalizeArtifacts

    @property
    def population_warning_threshold(self) -> int:
        return self.soft_max_global_ids + self.overflow_allowance


_ARTIFACTS = {
    "candidate_edges": "global_candidate_edges.parquet",
    "stable_to_global": "stable_to_global.parquet",
    "global_tracks": "global_tracks.parquet",
    "det_to_global": "det_to_global.parquet",
    "report": "s05_finalize_report.json",
    "effective_config": "effective_config.json",
    "success": "_SUCCESS.json",
}

def _exact(value: object, keys: set[str], context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ContractError(f"{context} must be a mapping")
    actual = set(value)
    if actual != keys:
        raise ContractError(
            f"{context} keys mismatch; missing={sorted(keys - actual)}, "
            f"extra={sorted(actual - keys)}"
        )
    return value


def _fixed(actual: object, expected: object, name: str) -> None:
    if actual != expected or (
        isinstance(expected, bool) and type(actual) is not bool
    ):
        raise ContractError(f"fixed S05 finalize requires {name}={expected!r}")


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


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ContractError(f"{name} must be finite")
    return result


def _clip_counts(value: object, context: str) -> tuple[int, ...]:
    mapping = _exact(value, set(EXPECTED_CLIP_ORDER), context)
    result: list[int] = []
    for clip_id in EXPECTED_CLIP_ORDER:
        count = _integer(mapping[clip_id], f"{context}.{clip_id}")
        result.append(count)
    return tuple(result)


def _digest(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ContractError(f"{name} must be a lowercase SHA-256 digest")
    return value


def load_s05_finalize_config(
    path: Path,
) -> tuple[S05FinalizeConfig, dict[str, Any], str]:
    """Load and validate the single approved S05 finalization strategy."""

    path = path.resolve()
    if not path.is_file():
        raise ContractError(f"S05 finalize config does not exist: {path}")
    try:
        payload = yaml.safe_load(path.read_bytes())
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ContractError(f"cannot read S05 finalize config {path}: {exc}") from exc

    top = _exact(
        payload,
        {
            "pipeline",
            "inputs",
            "approval",
            "solver",
            "population",
            "runtime",
            "artifacts",
        },
        "S05 finalize",
    )
    pipeline = _exact(
        top["pipeline"],
        {
            "schema_version",
            "random_seed",
            "execution_mode",
            "operator_approved",
            "global_merge_allowed",
            "solver_allowed",
            "path_cover_allowed",
            "certification_claim_allowed",
        },
        "S05 finalize.pipeline",
    )
    inputs = _exact(
        top["inputs"],
        {
            "expected_s04_finalize_config_hash",
            "expected_long_calibration_config_hash",
            "expected_long_proposal_config_hash",
            "expected_sequence_id",
            "expected_stable_track_count",
            "expected_microtrack_count",
            "expected_detection_count",
            "expected_invalid_detections",
            "expected_frame_count",
            "expected_candidate_count",
            "expected_proposal_count",
            "clip_order",
            "frame_counts_by_clip",
            "expected_valid_detections_by_clip",
            "expected_invalid_detections_by_clip",
        },
        "S05 finalize.inputs",
    )
    approval = _exact(
        top["approval"],
        {
            "strategy",
            "authorization_basis",
            "required_proposal_decision",
            "required_proposal_review_status",
            "review_output_consumed",
            "review_labels_consumed",
            "require_proposal_selected_by_solver_false",
            "require_proposal_confirmed_false",
            "require_proposal_merge_applied_false",
            "preserve_proposal_evidence_status",
        },
        "S05 finalize.approval",
    )
    solver = _exact(
        top["solver"],
        {
            "algorithm",
            "objective",
            "candidate_policy",
            "graph_node_identity",
            "direction",
            "require_strict_non_overlap",
            "max_incoming_links",
            "max_outgoing_links",
            "maximum_iterations",
            "evidence_cost",
            "deterministic_tie_break",
        },
        "S05 finalize.solver",
    )
    population = _exact(
        top["population"],
        {
            "soft_max_global_ids",
            "overflow_allowance",
            "policy",
            "affects_solver",
            "force_merge_allowed",
            "threshold_adaptation_allowed",
        },
        "S05 finalize.population",
    )
    runtime = _exact(
        top["runtime"],
        {
            "expected_python_version",
            "expected_scipy_version",
            "worker_count",
            "progress_interval_sec",
            "parquet_compression",
            "deterministic_row_sort",
            "atomic_commit",
            "resume_revalidate",
            "log_flush",
        },
        "S05 finalize.runtime",
    )
    artifacts = _exact(
        top["artifacts"], set(_ARTIFACTS), "S05 finalize.artifacts"
    )

    fixed = {
        "pipeline.schema_version": (pipeline["schema_version"], "1.0"),
        "pipeline.execution_mode": (
            pipeline["execution_mode"],
            "operator_approved_global_path_cover",
        ),
        "pipeline.operator_approved": (pipeline["operator_approved"], True),
        "pipeline.global_merge_allowed": (
            pipeline["global_merge_allowed"],
            True,
        ),
        "pipeline.solver_allowed": (pipeline["solver_allowed"], True),
        "pipeline.path_cover_allowed": (pipeline["path_cover_allowed"], True),
        "pipeline.certification_claim_allowed": (
            pipeline["certification_claim_allowed"],
            False,
        ),
        "inputs.expected_sequence_id": (
            inputs["expected_sequence_id"],
            EXPECTED_SEQUENCE_ID,
        ),
        "inputs.clip_order": (inputs["clip_order"], list(EXPECTED_CLIP_ORDER)),
        "approval.strategy": (
            approval["strategy"],
            "all_provisional_proposals",
        ),
        "approval.authorization_basis": (
            approval["authorization_basis"],
            "operator_blanket_strategy",
        ),
        "approval.required_proposal_decision": (
            approval["required_proposal_decision"],
            "provisional",
        ),
        "approval.required_proposal_review_status": (
            approval["required_proposal_review_status"],
            "pending",
        ),
        "approval.review_output_consumed": (
            approval["review_output_consumed"],
            False,
        ),
        "approval.review_labels_consumed": (
            approval["review_labels_consumed"],
            False,
        ),
        "approval.require_proposal_selected_by_solver_false": (
            approval["require_proposal_selected_by_solver_false"],
            True,
        ),
        "approval.require_proposal_confirmed_false": (
            approval["require_proposal_confirmed_false"],
            True,
        ),
        "approval.require_proposal_merge_applied_false": (
            approval["require_proposal_merge_applied_false"],
            True,
        ),
        "approval.preserve_proposal_evidence_status": (
            approval["preserve_proposal_evidence_status"],
            True,
        ),
        "solver.algorithm": (
            solver["algorithm"],
            "deterministic_bipartite_matching",
        ),
        "solver.objective": (
            solver["objective"],
            "maximum_cardinality_then_evidence_cost",
        ),
        "solver.candidate_policy": (
            solver["candidate_policy"],
            "all_provisional_proposals",
        ),
        "solver.graph_node_identity": (
            solver["graph_node_identity"],
            "stable_id",
        ),
        "solver.direction": (solver["direction"], "future_only"),
        "solver.require_strict_non_overlap": (
            solver["require_strict_non_overlap"],
            True,
        ),
        "solver.evidence_cost": (
            solver["evidence_cost"],
            "deterministic_evidence_ordinal",
        ),
        "solver.deterministic_tie_break": (
            solver["deterministic_tie_break"],
            "canonical_csr_cost_asc_fixed_scipy_1_18_0",
        ),
        "population.policy": (population["policy"], "warning_only"),
        "population.affects_solver": (population["affects_solver"], False),
        "population.force_merge_allowed": (
            population["force_merge_allowed"],
            False,
        ),
        "population.threshold_adaptation_allowed": (
            population["threshold_adaptation_allowed"],
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
        "runtime.atomic_commit": (runtime["atomic_commit"], True),
        "runtime.resume_revalidate": (runtime["resume_revalidate"], True),
        "runtime.log_flush": (runtime["log_flush"], True),
        "runtime.expected_python_version": (
            runtime["expected_python_version"],
            "3.14.4",
        ),
        "runtime.expected_scipy_version": (
            runtime["expected_scipy_version"],
            "1.18.0",
        ),
    }
    for name, (actual, expected) in fixed.items():
        _fixed(actual, expected, name)
    if platform.python_version() != "3.14.4" or scipy.__version__ != "1.18.0":
        raise ContractError(
            "S05 finalize requires fixed solver runtime Python 3.14.4 / SciPy 1.18.0"
        )

    seed = _integer(pipeline["random_seed"], "S05 finalize.pipeline.random_seed")
    _fixed(seed, 20260710, "pipeline.random_seed")

    optional_integer_inputs = {
        "expected_stable_track_count": 1,
        "expected_microtrack_count": 1,
        "expected_candidate_count": 0,
        "expected_proposal_count": 0,
    }
    integer_inputs = {
        "expected_detection_count": 1,
        "expected_invalid_detections": 0,
        "expected_frame_count": 1,
    }
    parsed_inputs: dict[str, int | None] = {}
    for name, minimum in optional_integer_inputs.items():
        parsed_inputs[name] = _optional_integer(
            inputs[name], f"S05 finalize.inputs.{name}", minimum=minimum
        )
    for name, minimum in integer_inputs.items():
        value = _integer(inputs[name], f"S05 finalize.inputs.{name}", minimum=minimum)
        parsed_inputs[name] = value

    frame_counts = _clip_counts(
        inputs["frame_counts_by_clip"],
        "S05 finalize.inputs.frame_counts_by_clip",
    )
    valid_counts = _clip_counts(
        inputs["expected_valid_detections_by_clip"],
        "S05 finalize.inputs.expected_valid_detections_by_clip",
    )
    invalid_counts = _clip_counts(
        inputs["expected_invalid_detections_by_clip"],
        "S05 finalize.inputs.expected_invalid_detections_by_clip",
    )
    if (
        sum(frame_counts) != parsed_inputs["expected_frame_count"]
        or sum(valid_counts) != parsed_inputs["expected_detection_count"]
        or sum(invalid_counts) != parsed_inputs["expected_invalid_detections"]
        or (
            parsed_inputs["expected_stable_track_count"] is not None
            and parsed_inputs["expected_microtrack_count"] is not None
            and parsed_inputs["expected_stable_track_count"]
            > parsed_inputs["expected_microtrack_count"]
        )
        or (
            parsed_inputs["expected_microtrack_count"] is not None
            and parsed_inputs["expected_microtrack_count"]
            > parsed_inputs["expected_detection_count"]
        )
        or (
            parsed_inputs["expected_proposal_count"] is not None
            and parsed_inputs["expected_candidate_count"] is not None
            and parsed_inputs["expected_proposal_count"]
            > parsed_inputs["expected_candidate_count"]
        )
    ):
        raise ContractError("S05 finalize expected counts are inconsistent")
    s04_hash = _digest(inputs["expected_s04_finalize_config_hash"], "expected_s04_finalize_config_hash")
    calibration_hash = _digest(inputs["expected_long_calibration_config_hash"], "expected_long_calibration_config_hash")
    proposal_hash = _digest(inputs["expected_long_proposal_config_hash"], "expected_long_proposal_config_hash")

    max_incoming = _integer(
        solver["max_incoming_links"],
        "S05 finalize.solver.max_incoming_links",
        minimum=1,
    )
    max_outgoing = _integer(
        solver["max_outgoing_links"],
        "S05 finalize.solver.max_outgoing_links",
        minimum=1,
    )
    maximum_iterations = _integer(
        solver["maximum_iterations"],
        "S05 finalize.solver.maximum_iterations",
        minimum=1,
    )
    _fixed(max_incoming, 1, "solver.max_incoming_links")
    _fixed(max_outgoing, 1, "solver.max_outgoing_links")
    _fixed(maximum_iterations, 1, "solver.maximum_iterations")

    soft_max = _integer(
        population["soft_max_global_ids"],
        "S05 finalize.population.soft_max_global_ids",
        minimum=1,
    )
    overflow = _integer(
        population["overflow_allowance"],
        "S05 finalize.population.overflow_allowance",
        minimum=0,
    )
    _fixed(soft_max, 57, "population.soft_max_global_ids")
    _fixed(overflow, 5, "population.overflow_allowance")

    worker_count = _integer(
        runtime["worker_count"], "S05 finalize.runtime.worker_count", minimum=1
    )
    _fixed(worker_count, 16, "runtime.worker_count")
    interval = _finite(
        runtime["progress_interval_sec"],
        "S05 finalize.runtime.progress_interval_sec",
    )
    if not math.isclose(interval, 10.0, rel_tol=0.0, abs_tol=1e-12):
        raise ContractError(
            "fixed S05 finalize requires runtime.progress_interval_sec=10.0"
        )
    for name, expected in _ARTIFACTS.items():
        _fixed(artifacts[name], expected, f"artifacts.{name}")

    result = S05FinalizeConfig(
        schema_version="1.0",
        random_seed=seed,
        execution_mode="operator_approved_global_path_cover",
        operator_approved=True,
        global_merge_allowed=True,
        solver_allowed=True,
        path_cover_allowed=True,
        certification_claim_allowed=False,
        expected_s04_finalize_config_hash=s04_hash,
        expected_long_calibration_config_hash=calibration_hash,
        expected_long_proposal_config_hash=proposal_hash,
        expected_sequence_id=EXPECTED_SEQUENCE_ID,
        expected_stable_track_count=parsed_inputs["expected_stable_track_count"],
        expected_microtrack_count=parsed_inputs["expected_microtrack_count"],
        expected_detection_count=parsed_inputs["expected_detection_count"],
        expected_invalid_detections=parsed_inputs["expected_invalid_detections"],
        expected_frame_count=parsed_inputs["expected_frame_count"],
        expected_candidate_count=parsed_inputs["expected_candidate_count"],
        expected_proposal_count=parsed_inputs["expected_proposal_count"],
        clip_order=EXPECTED_CLIP_ORDER,
        frame_counts_by_clip=frame_counts,
        expected_valid_detections_by_clip=valid_counts,
        expected_invalid_detections_by_clip=invalid_counts,
        approval_strategy="all_provisional_proposals",
        authorization_basis="operator_blanket_strategy",
        required_proposal_decision="provisional",
        required_proposal_review_status="pending",
        review_output_consumed=False,
        review_labels_consumed=False,
        require_proposal_selected_by_solver_false=True,
        require_proposal_confirmed_false=True,
        require_proposal_merge_applied_false=True,
        preserve_proposal_evidence_status=True,
        solver="deterministic_bipartite_matching",
        objective="maximum_cardinality_then_evidence_cost",
        candidate_policy="all_provisional_proposals",
        graph_node_identity="stable_id",
        direction="future_only",
        require_strict_non_overlap=True,
        max_incoming_links=max_incoming,
        max_outgoing_links=max_outgoing,
        maximum_iterations=maximum_iterations,
        evidence_cost="deterministic_evidence_ordinal",
        deterministic_tie_break=(
            "canonical_csr_cost_asc_fixed_scipy_1_18_0"
        ),
        soft_max_global_ids=soft_max,
        overflow_allowance=overflow,
        population_policy="warning_only",
        population_affects_solver=False,
        force_merge_allowed=False,
        threshold_adaptation_allowed=False,
        worker_count=worker_count,
        expected_python_version="3.14.4",
        expected_scipy_version="1.18.0",
        progress_interval_sec=interval,
        parquet_compression="zstd",
        deterministic_row_sort=True,
        atomic_commit=True,
        resume_revalidate=True,
        log_flush=True,
        artifacts=FinalizeArtifacts(**artifacts),
    )
    canonical = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return result, payload, hashlib.sha256(canonical).hexdigest()


__all__ = [
    "EXPECTED_CLIP_ORDER",
    "EXPECTED_SEQUENCE_ID",
    "FinalizeArtifacts",
    "S05FinalizeConfig",
    "load_s05_finalize_config",
]
