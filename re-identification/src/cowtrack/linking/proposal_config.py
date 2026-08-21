"""Strict configuration contract for the S04 proposal-only stage."""

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
class ProposalArtifacts:
    candidate_edges: str
    proposals: str
    review_manifest: str
    review_labels: str
    report: str
    effective_config: str
    success: str


@dataclass(frozen=True)
class ShortProposalConfig:
    schema_version: str
    random_seed: int
    execution_mode: str
    accepted_decision_for_review: str
    max_gap_sec: float
    clip_order: tuple[str, ...]
    parquet_compression: str
    progress_interval_sec: float
    deterministic_row_sort: bool
    automatic_merge_allowed: bool
    human_labels_applied: bool
    artifacts: ProposalArtifacts


_FIXED_ARTIFACTS = {
    "candidate_edges": "short_candidate_edges.parquet",
    "proposals": "short_link_proposals.parquet",
    "review_manifest": "review_manifest.json",
    "review_labels": "review_labels.csv",
    "report": "s04_proposal_report.json",
    "effective_config": "effective_config.json",
    "success": "_SUCCESS.json",
}


def _exact(mapping: object, expected: set[str], context: str) -> dict[str, Any]:
    if not isinstance(mapping, dict):
        raise ContractError(f"{context} must be a mapping")
    if set(mapping) != expected:
        raise ContractError(
            f"{context} config keys mismatch; "
            f"missing={sorted(expected - set(mapping))}, "
            f"extra={sorted(set(mapping) - expected)}"
        )
    return mapping


def load_short_proposal_config(
    path: Path,
) -> tuple[ShortProposalConfig, dict[str, Any], str]:
    """Load the fixed, deliberately narrow S04 proposal configuration."""

    path = path.resolve()
    if not path.is_file():
        raise ContractError(f"S04 proposal config does not exist: {path}")
    try:
        payload = yaml.safe_load(path.read_bytes())
    except (OSError, yaml.YAMLError) as exc:
        raise ContractError(f"cannot read S04 proposal config {path}: {exc}") from exc
    top = _exact(payload, {"pipeline", "proposals", "artifacts"}, "S04")
    pipeline = _exact(
        top["pipeline"],
        {
            "schema_version",
            "random_seed",
            "execution_mode",
            "automatic_merge_allowed",
            "human_labels_applied",
        },
        "S04.pipeline",
    )
    proposals = _exact(
        top["proposals"],
        {
            "accepted_decision_for_review",
            "max_gap_sec",
            "clip_order",
            "parquet_compression",
            "progress_interval_sec",
            "deterministic_row_sort",
        },
        "S04.proposals",
    )
    artifacts = _exact(top["artifacts"], set(_FIXED_ARTIFACTS), "S04.artifacts")

    seed = pipeline["random_seed"]
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ContractError("S04.pipeline.random_seed must be a non-negative integer")
    gap = proposals["max_gap_sec"]
    interval = proposals["progress_interval_sec"]
    if (
        isinstance(gap, bool)
        or not isinstance(gap, (int, float))
        or not math.isfinite(float(gap))
        or float(gap) != 5.0
    ):
        raise ContractError("fixed S04 requires proposals.max_gap_sec=5.0")
    if (
        isinstance(interval, bool)
        or not isinstance(interval, (int, float))
        or not math.isfinite(float(interval))
        or float(interval) <= 0.0
    ):
        raise ContractError("S04.proposals.progress_interval_sec must be positive")
    clips = proposals["clip_order"]
    if clips != list(EXPECTED_CLIP_ORDER):
        raise ContractError(
            "fixed S04 requires the canonical 11-clip order"
        )
    fixed_values = {
        "schema_version": (pipeline["schema_version"], "1.0"),
        "execution_mode": (pipeline["execution_mode"], "proposal_only"),
        "automatic_merge_allowed": (pipeline["automatic_merge_allowed"], False),
        "human_labels_applied": (pipeline["human_labels_applied"], False),
        "accepted_decision_for_review": (
            proposals["accepted_decision_for_review"],
            "provisional",
        ),
        "deterministic_row_sort": (proposals["deterministic_row_sort"], True),
    }
    for name, (actual, expected) in fixed_values.items():
        if actual != expected:
            raise ContractError(f"fixed S04 requires {name}={expected!r}")
    compression = proposals["parquet_compression"]
    if not isinstance(compression, str) or not compression:
        raise ContractError("S04.proposals.parquet_compression cannot be blank")
    for name, expected in _FIXED_ARTIFACTS.items():
        if artifacts[name] != expected:
            raise ContractError(
                f"fixed S04 requires artifacts.{name}={expected!r}"
            )

    result = ShortProposalConfig(
        schema_version="1.0",
        random_seed=seed,
        execution_mode="proposal_only",
        accepted_decision_for_review="provisional",
        max_gap_sec=5.0,
        clip_order=tuple(clips),
        parquet_compression=compression,
        progress_interval_sec=float(interval),
        deterministic_row_sort=True,
        automatic_merge_allowed=False,
        human_labels_applied=False,
        artifacts=ProposalArtifacts(**artifacts),
    )
    canonical = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return result, payload, hashlib.sha256(canonical).hexdigest()


__all__ = [
    "ProposalArtifacts",
    "ShortProposalConfig",
    "load_short_proposal_config",
]
