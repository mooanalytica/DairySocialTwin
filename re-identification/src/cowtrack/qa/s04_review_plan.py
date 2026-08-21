"""Deterministic selection plan for S04 provisional-link review videos.

This module is deliberately pure: it reads no artifacts and opens no videos.
Only provisional rows from ``short_link_proposals.parquet`` are accepted.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import math
import random
import re
from typing import Any

import numpy as np

from cowtrack.config import ContractError


PLAN_SCHEMA_VERSION = "cowtrack.s04-review-plan.v1"

PROPOSAL_COLUMNS = (
    "proposal_id",
    "edge_id",
    "source_micro_id",
    "target_micro_id",
    "probability",
    "high_overlap",
    "source_rank",
    "target_rank",
    "conflict_degree",
    "conflict_group_id",
    "conflict_group_edge_count",
    "conflict_group_node_count",
    "review_status",
)

PURPLE_BOX_POLICY = (
    "For a selected proposal, collect every source/target micro_id used by another "
    "provisional proposal in the same immutable conflict_group_id; remove the "
    "selected source and target. Draw an actual S01-assigned detection from that set "
    "purple only on frames strictly after the selected source_end_global_frame and "
    "strictly before the selected target_start_global_frame. No interpolation, "
    "prediction, synthetic box, geometry change, or legacy track ID is used."
)


@dataclass(frozen=True)
class ProposalRecord:
    proposal_id: str
    edge_id: str
    source_micro_id: int
    target_micro_id: int
    probability: float
    high_overlap: bool
    source_rank: int
    target_rank: int
    conflict_degree: int
    conflict_group_id: str
    conflict_group_edge_count: int
    conflict_group_node_count: int
    review_status: str


@dataclass(frozen=True)
class S04ReviewCase:
    proposal: ProposalRecord
    selection_group: str
    selection_order: int

    @property
    def suggested_filename(self) -> str:
        safe_id = safe_filename_token(self.proposal.proposal_id)
        return (
            f"{self.selection_order:03d}_{self.selection_group}_{safe_id}_"
            f"p{self.proposal.probability:.4f}.mp4"
        )

    def to_manifest_record(self) -> dict[str, Any]:
        proposal = self.proposal
        return {
            "proposal_id": proposal.proposal_id,
            "edge_id": proposal.edge_id,
            "source_micro_id": proposal.source_micro_id,
            "target_micro_id": proposal.target_micro_id,
            "probability": proposal.probability,
            "probability_filename_precision": f"{proposal.probability:.4f}",
            "high_overlap": proposal.high_overlap,
            "source_rank": proposal.source_rank,
            "target_rank": proposal.target_rank,
            "conflict_degree": proposal.conflict_degree,
            "conflict_group_id": proposal.conflict_group_id,
            "conflict_group_edge_count": proposal.conflict_group_edge_count,
            "conflict_group_node_count": proposal.conflict_group_node_count,
            "review_status": proposal.review_status,
            "selection_group": self.selection_group,
            "selection_order": self.selection_order,
            "suggested_filename": self.suggested_filename,
        }


@dataclass(frozen=True)
class S04ReviewPlan:
    proposals: tuple[ProposalRecord, ...]
    cases: tuple[S04ReviewCase, ...]
    random_seed: int
    top_count: int
    bottom_count: int
    random_count: int

    def intermediate_micro_ids(self, case: S04ReviewCase) -> tuple[int, ...]:
        """Return immutable conflict endpoints eligible for purple rendering."""

        focal = case.proposal
        result: set[int] = set()
        for proposal in self.proposals:
            if proposal.conflict_group_id == focal.conflict_group_id:
                result.add(proposal.source_micro_id)
                result.add(proposal.target_micro_id)
        result.discard(focal.source_micro_id)
        result.discard(focal.target_micro_id)
        return tuple(sorted(result))

    def to_manifest_payload(self) -> dict[str, Any]:
        return {
            "schema_version": PLAN_SCHEMA_VERSION,
            "input_decision": "provisional_only",
            "selection_is_read_only": True,
            "selection_policy": {
                "if_fewer_than_50": "select_all_probability_desc_then_proposal_id",
                "otherwise": [
                    {
                        "group": "top",
                        "count": self.top_count,
                        "order": "probability_desc_then_proposal_id",
                    },
                    {
                        "group": "bottom",
                        "count": self.bottom_count,
                        "order": "probability_asc_then_proposal_id",
                    },
                    {
                        "group": "random",
                        "count": self.random_count,
                        "population": "remaining_proposal_id_ascending",
                        "algorithm": "python_random.Random(seed).sample_without_replacement",
                        "seed": self.random_seed,
                    },
                ],
                "duplicates_allowed": False,
            },
            "render_window": {
                "source_tail_sec": 2.0,
                "gap": "all real S00 frames between source and target",
                "target_head_sec": 2.0,
                "cross_clip": "continuous chronological encoded frames",
            },
            "visual_contract": {
                "original_frame_and_bboxes_only": True,
                "text": False,
                "ids": False,
                "score_overlay": False,
                "legend": False,
                "panels": False,
                "inset": False,
                "trajectory": False,
                "keypoints": False,
                "box_geometry_modified": False,
                "synthetic_boxes": False,
                "purple_box_policy": PURPLE_BOX_POLICY,
            },
            "summary": {
                "num_available_proposals": len(self.proposals),
                "num_selected_cases": len(self.cases),
                "num_top": sum(c.selection_group == "top" for c in self.cases),
                "num_bottom": sum(
                    c.selection_group == "bottom" for c in self.cases
                ),
                "num_random": sum(
                    c.selection_group == "random" for c in self.cases
                ),
                "num_all_available": sum(
                    c.selection_group == "all" for c in self.cases
                ),
            },
            "cases": [case.to_manifest_record() for case in self.cases],
        }


def safe_filename_token(value: str) -> str:
    """Return a bounded filename token, hashing unsafe proposal identifiers."""

    if re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", value):
        return value
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]
    return f"proposal-{digest}"


def _one_dimensional(
    columns: Mapping[str, Sequence[Any] | np.ndarray], name: str, count: int
) -> np.ndarray:
    if name not in columns:
        raise ContractError(f"S04 review proposals missing column: {name}")
    result = np.asarray(columns[name])
    if result.ndim != 1 or len(result) != count:
        raise ContractError(
            f"S04 review proposals.{name} must be one-dimensional with {count} rows"
        )
    return result


def parse_proposal_records(
    columns: Mapping[str, Sequence[Any] | np.ndarray],
) -> tuple[ProposalRecord, ...]:
    """Parse and validate untouched provisional proposal rows."""

    if "proposal_id" not in columns:
        raise ContractError("S04 review proposals missing column: proposal_id")
    proposal_ids = np.asarray(columns["proposal_id"])
    if proposal_ids.ndim != 1:
        raise ContractError("S04 review proposals.proposal_id must be one-dimensional")
    count = len(proposal_ids)
    arrays = {
        name: _one_dimensional(columns, name, count) for name in PROPOSAL_COLUMNS
    }
    records: list[ProposalRecord] = []
    for row in range(count):
        probability = float(arrays["probability"][row])
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ContractError(
                f"S04 proposal probability must be finite in [0, 1] at row {row}"
            )
        proposal_id = str(arrays["proposal_id"][row])
        edge_id = str(arrays["edge_id"][row])
        group_id = str(arrays["conflict_group_id"][row])
        status = str(arrays["review_status"][row])
        if not proposal_id or not edge_id or not group_id:
            raise ContractError(f"S04 proposal identifiers cannot be blank at row {row}")
        if status != "pending":
            raise ContractError(
                "S04 review accepts only untouched provisional proposals with "
                f"review_status='pending', got {status!r} at row {row}"
            )
        high_overlap_value = arrays["high_overlap"][row]
        if not isinstance(high_overlap_value, (bool, np.bool_)):
            raise ContractError(f"S04 proposal high_overlap must be boolean at row {row}")
        integer_fields = {
            name: int(arrays[name][row])
            for name in (
                "source_micro_id",
                "target_micro_id",
                "source_rank",
                "target_rank",
                "conflict_degree",
                "conflict_group_edge_count",
                "conflict_group_node_count",
            )
        }
        if integer_fields["source_micro_id"] == integer_fields["target_micro_id"]:
            raise ContractError(f"S04 proposal self-link at row {row}")
        if min(integer_fields["source_rank"], integer_fields["target_rank"]) < 1:
            raise ContractError(f"S04 proposal ranks must be positive at row {row}")
        if min(
            integer_fields["conflict_degree"],
            integer_fields["conflict_group_edge_count"],
            integer_fields["conflict_group_node_count"],
        ) < 0:
            raise ContractError(f"S04 proposal conflict counts cannot be negative")
        records.append(
            ProposalRecord(
                proposal_id=proposal_id,
                edge_id=edge_id,
                probability=probability,
                high_overlap=bool(high_overlap_value),
                conflict_group_id=group_id,
                review_status=status,
                **integer_fields,
            )
        )
    if len({record.proposal_id for record in records}) != count:
        raise ContractError("S04 proposal_id values must be unique")
    if len({record.edge_id for record in records}) != count:
        raise ContractError("S04 proposal edge_id values must be unique")
    return tuple(records)


def build_s04_review_plan(
    proposals: Mapping[str, Sequence[Any] | np.ndarray],
    *,
    random_seed: int,
    top_count: int = 15,
    bottom_count: int = 15,
    random_count: int = 20,
) -> S04ReviewPlan:
    """Select top 15, bottom 15, then 20 seeded random remainder rows."""

    if isinstance(random_seed, bool) or not isinstance(random_seed, int):
        raise ContractError("S04 review random_seed must be an integer")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in (top_count, bottom_count, random_count)
    ):
        raise ContractError("S04 review selection counts must be non-negative integers")
    records = parse_proposal_records(proposals)
    requested = top_count + bottom_count + random_count
    cases: list[S04ReviewCase] = []
    if len(records) < requested:
        selected = sorted(
            records, key=lambda item: (-item.probability, item.proposal_id)
        )
        groups = [(item, "all") for item in selected]
    else:
        top = sorted(
            records, key=lambda item: (-item.probability, item.proposal_id)
        )[:top_count]
        chosen = {item.proposal_id for item in top}
        bottom = [
            item
            for item in sorted(
                records, key=lambda item: (item.probability, item.proposal_id)
            )
            if item.proposal_id not in chosen
        ][:bottom_count]
        chosen.update(item.proposal_id for item in bottom)
        remainder = sorted(
            (item for item in records if item.proposal_id not in chosen),
            key=lambda item: item.proposal_id,
        )
        if len(remainder) < random_count:
            raise ContractError("S04 review selection has an undersized random remainder")
        random_rows = random.Random(random_seed).sample(remainder, random_count)
        groups = (
            [(item, "top") for item in top]
            + [(item, "bottom") for item in bottom]
            + [(item, "random") for item in random_rows]
        )
    for order, (proposal, group) in enumerate(groups, start=1):
        cases.append(S04ReviewCase(proposal, group, order))
    if len({case.proposal.proposal_id for case in cases}) != len(cases):
        raise ContractError("internal error: duplicate S04 review case selected")
    if len(records) < requested and len(cases) != len(records):
        raise ContractError("internal error: fewer-than-50 policy did not select all")
    if len(records) >= requested and len(cases) != requested:
        raise ContractError("internal error: fixed S04 review selection is not complete")
    return S04ReviewPlan(
        proposals=records,
        cases=tuple(cases),
        random_seed=random_seed,
        top_count=top_count,
        bottom_count=bottom_count,
        random_count=random_count,
    )


__all__ = [
    "PLAN_SCHEMA_VERSION",
    "PROPOSAL_COLUMNS",
    "PURPLE_BOX_POLICY",
    "ProposalRecord",
    "S04ReviewCase",
    "S04ReviewPlan",
    "build_s04_review_plan",
    "parse_proposal_records",
    "safe_filename_token",
]
