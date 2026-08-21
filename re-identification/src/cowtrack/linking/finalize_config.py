"""Strict fixed-data configuration for operator-approved S04 finalization."""

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
class FinalizeArtifacts:
    micro_to_stable: str
    stable_tracklets: str
    det_to_stable: str
    stable_prototypes: str
    stable_prototype_mask: str
    stable_appearance: str
    report: str
    effective_config: str
    success: str


@dataclass(frozen=True)
class S04FinalizeConfig:
    schema_version: str
    random_seed: int
    execution_mode: str
    operator_approved: bool
    merge_policy: str
    graph_node_identity: str
    require_strict_non_overlap: bool
    stable_id_policy: str
    read_review_labels: bool
    solver: str
    parquet_compression: str
    progress_interval_sec: float
    artifacts: FinalizeArtifacts


_ARTIFACTS = {
    "micro_to_stable": "micro_to_stable.parquet",
    "stable_tracklets": "stable_tracklets.parquet",
    "det_to_stable": "det_to_stable.parquet",
    "stable_prototypes": "stable_prototypes.f16.npy",
    "stable_prototype_mask": "stable_prototype_mask.npy",
    "stable_appearance": "stable_appearance.parquet",
    "report": "finalize_report.json",
    "effective_config": "effective_config.json",
    "success": "_SUCCESS.json",
}


def _exact(value: object, keys: set[str], context: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        actual = set(value) if isinstance(value, dict) else set()
        raise ContractError(
            f"{context} config keys mismatch; missing={sorted(keys-actual)}, "
            f"extra={sorted(actual-keys)}"
        )
    return value


def load_s04_finalize_config(
    path: Path,
) -> tuple[S04FinalizeConfig, dict[str, Any], str]:
    path = path.resolve()
    if not path.is_file():
        raise ContractError(f"S04 finalize config does not exist: {path}")
    try:
        payload = yaml.safe_load(path.read_bytes())
    except (OSError, yaml.YAMLError) as exc:
        raise ContractError(f"cannot read S04 finalize config {path}: {exc}") from exc
    top = _exact(payload, {"pipeline", "finalize", "artifacts"}, "S04 finalize")
    pipeline = _exact(
        top["pipeline"],
        {"schema_version", "random_seed", "execution_mode", "operator_approved"},
        "S04 finalize.pipeline",
    )
    finalize = _exact(
        top["finalize"],
        {
            "merge_policy",
            "graph_node_identity",
            "require_strict_non_overlap",
            "stable_id_policy",
            "read_review_labels",
            "solver",
            "parquet_compression",
            "progress_interval_sec",
        },
        "S04 finalize.finalize",
    )
    artifacts = _exact(top["artifacts"], set(_ARTIFACTS), "S04 finalize.artifacts")
    boolean_fields = {
        "pipeline.operator_approved": pipeline["operator_approved"],
        "finalize.require_strict_non_overlap": finalize[
            "require_strict_non_overlap"
        ],
        "finalize.read_review_labels": finalize["read_review_labels"],
    }
    for field, value in boolean_fields.items():
        if type(value) is not bool:
            raise ContractError(f"S04 finalize {field} must be a boolean")
    fixed = {
        "pipeline.schema_version": (pipeline["schema_version"], "1.0"),
        "pipeline.random_seed": (pipeline["random_seed"], 20260710),
        "pipeline.execution_mode": (
            pipeline["execution_mode"],
            "operator_approved_component_union",
        ),
        "pipeline.operator_approved": (pipeline["operator_approved"], True),
        "finalize.merge_policy": (
            finalize["merge_policy"],
            "all_provisional_undirected_components",
        ),
        "finalize.graph_node_identity": (
            finalize["graph_node_identity"],
            "actual_micro_id",
        ),
        "finalize.require_strict_non_overlap": (
            finalize["require_strict_non_overlap"],
            True,
        ),
        "finalize.stable_id_policy": (
            finalize["stable_id_policy"],
            "canonical_component_order_zero_based",
        ),
        "finalize.read_review_labels": (finalize["read_review_labels"], False),
        "finalize.solver": (finalize["solver"], "none"),
    }
    for field, (actual, expected) in fixed.items():
        if actual != expected:
            raise ContractError(f"fixed S04 finalize requires {field}={expected!r}")
    interval = finalize["progress_interval_sec"]
    if (
        isinstance(interval, bool)
        or not isinstance(interval, (int, float))
        or not math.isfinite(float(interval))
        or float(interval) <= 0.0
    ):
        raise ContractError("S04 finalize progress interval must be positive")
    compression = finalize["parquet_compression"]
    if not isinstance(compression, str) or not compression:
        raise ContractError("S04 finalize parquet compression cannot be blank")
    for key, expected in _ARTIFACTS.items():
        if artifacts[key] != expected:
            raise ContractError(
                f"fixed S04 finalize requires artifacts.{key}={expected!r}"
            )
    result = S04FinalizeConfig(
        schema_version="1.0",
        random_seed=20260710,
        execution_mode="operator_approved_component_union",
        operator_approved=True,
        merge_policy="all_provisional_undirected_components",
        graph_node_identity="actual_micro_id",
        require_strict_non_overlap=True,
        stable_id_policy="canonical_component_order_zero_based",
        read_review_labels=False,
        solver="none",
        parquet_compression=compression,
        progress_interval_sec=float(interval),
        artifacts=FinalizeArtifacts(**artifacts),
    )
    canonical = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return result, payload, hashlib.sha256(canonical).hexdigest()


__all__ = ["FinalizeArtifacts", "S04FinalizeConfig", "load_s04_finalize_config"]
