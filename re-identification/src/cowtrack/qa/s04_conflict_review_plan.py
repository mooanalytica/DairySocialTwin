"""Pure deterministic selection plan for S04 conflict-group review.

This module reads no artifacts and renders no video. It selects only untouched
provisional groups containing more than one edge, then caps each selected group
to its three highest-probability proposals.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import random
from typing import Any

import numpy as np

from cowtrack.config import ContractError
from cowtrack.qa.s04_review_plan import (
    ProposalRecord,
    parse_proposal_records,
    safe_filename_token,
)


PLAN_SCHEMA_VERSION = "cowtrack.s04-conflict-review-plan.v1"


@dataclass(frozen=True)
class S04ConflictReviewCase:
    proposal: ProposalRecord
    group_selection: str
    group_selection_order: int
    edge_selection_order: int
    conflict_group_edge_count: int
    conflict_group_node_count: int
    rendered_subset_is_partial: bool

    @property
    def suggested_filename(self) -> str:
        safe_id = safe_filename_token(self.proposal.proposal_id)
        return (
            f"{self.group_selection_order:03d}_{self.group_selection}_"
            f"edge{self.edge_selection_order:02d}_{safe_id}_"
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
            "high_overlap": proposal.high_overlap,
            "source_rank": proposal.source_rank,
            "target_rank": proposal.target_rank,
            "conflict_degree": proposal.conflict_degree,
            "conflict_group_id": proposal.conflict_group_id,
            "conflict_group_edge_count": self.conflict_group_edge_count,
            "conflict_group_node_count": self.conflict_group_node_count,
            "review_status": proposal.review_status,
            "group_selection": self.group_selection,
            "group_selection_order": self.group_selection_order,
            "edge_selection_order": self.edge_selection_order,
            "rendered_subset_is_partial": self.rendered_subset_is_partial,
            "suggested_filename": self.suggested_filename,
        }


@dataclass(frozen=True)
class S04ConflictReviewGroup:
    conflict_group_id: str
    conflict_group_edge_count: int
    conflict_group_node_count: int
    selection_group: str
    selection_order: int
    rendered_proposals: tuple[ProposalRecord, ...]

    @property
    def rendered_subset_is_partial(self) -> bool:
        return len(self.rendered_proposals) < self.conflict_group_edge_count

    @property
    def cases(self) -> tuple[S04ConflictReviewCase, ...]:
        return tuple(
            S04ConflictReviewCase(
                proposal=proposal,
                group_selection=self.selection_group,
                group_selection_order=self.selection_order,
                edge_selection_order=edge_order,
                conflict_group_edge_count=self.conflict_group_edge_count,
                conflict_group_node_count=self.conflict_group_node_count,
                rendered_subset_is_partial=self.rendered_subset_is_partial,
            )
            for edge_order, proposal in enumerate(self.rendered_proposals, start=1)
        )

    def to_manifest_record(self) -> dict[str, Any]:
        return {
            "conflict_group_id": self.conflict_group_id,
            "conflict_group_edge_count": self.conflict_group_edge_count,
            "conflict_group_node_count": self.conflict_group_node_count,
            "selection_group": self.selection_group,
            "selection_order": self.selection_order,
            "num_rendered_edges": len(self.rendered_proposals),
            "rendered_subset_is_partial": self.rendered_subset_is_partial,
            "rendered_edges": [case.to_manifest_record() for case in self.cases],
        }


@dataclass(frozen=True)
class S04ConflictReviewPlan:
    proposals: tuple[ProposalRecord, ...]
    eligible_proposals: tuple[ProposalRecord, ...]
    groups: tuple[S04ConflictReviewGroup, ...]
    random_seed: int
    largest_group_count: int
    random_group_count: int
    max_edges_per_group: int

    @property
    def cases(self) -> tuple[S04ConflictReviewCase, ...]:
        return tuple(case for group in self.groups for case in group.cases)

    def intermediate_micro_ids(
        self, case: S04ConflictReviewCase
    ) -> tuple[int, ...]:
        """Return every other endpoint in the focal full conflict group."""

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
                "eligible_group": "conflict_group_edge_count > 1",
                "largest_groups": {
                    "count": self.largest_group_count,
                    "order": (
                        "conflict_group_edge_count_desc_then_"
                        "conflict_group_node_count_desc_then_group_id"
                    ),
                },
                "random_groups": {
                    "count": self.random_group_count,
                    "population": "remaining_group_id_ascending",
                    "algorithm": (
                        "python_random.Random(seed).sample_without_replacement"
                    ),
                    "seed": self.random_seed,
                },
                "edges_per_group": {
                    "maximum": self.max_edges_per_group,
                    "order": "probability_desc_then_proposal_id",
                },
                "duplicates_allowed": False,
                "if_fewer_groups_are_available": "select_all_available",
            },
            "summary": {
                "num_eligible_proposals": len(self.eligible_proposals),
                "num_eligible_groups": len(
                    {p.conflict_group_id for p in self.eligible_proposals}
                ),
                "num_selected_groups": len(self.groups),
                "num_selected_edges": len(self.cases),
                "num_partial_groups": sum(
                    group.rendered_subset_is_partial for group in self.groups
                ),
            },
            "groups": [group.to_manifest_record() for group in self.groups],
        }


@dataclass(frozen=True)
class _CompleteGroup:
    conflict_group_id: str
    conflict_group_edge_count: int
    conflict_group_node_count: int
    proposals: tuple[ProposalRecord, ...]


def _validated_groups(
    records: tuple[ProposalRecord, ...],
) -> tuple[_CompleteGroup, ...]:
    by_group: dict[str, list[ProposalRecord]] = {}
    for proposal in records:
        by_group.setdefault(proposal.conflict_group_id, []).append(proposal)

    result: list[_CompleteGroup] = []
    for group_id, proposals in by_group.items():
        edge_counts = {proposal.conflict_group_edge_count for proposal in proposals}
        node_counts = {proposal.conflict_group_node_count for proposal in proposals}
        if len(edge_counts) != 1 or len(node_counts) != 1:
            raise ContractError(
                f"S04 conflict group metadata is inconsistent: {group_id}"
            )
        edge_count = next(iter(edge_counts))
        node_count = next(iter(node_counts))
        if edge_count < 1 or node_count < 2:
            raise ContractError(
                f"S04 conflict group counts are invalid: {group_id}"
            )
        if edge_count != len(proposals):
            raise ContractError(
                "S04 conflict group edge count does not match proposal rows: "
                f"{group_id}"
            )
        observed_nodes = {
            (role, micro_id)
            for proposal in proposals
            for role, micro_id in (
                ("source", proposal.source_micro_id),
                ("target", proposal.target_micro_id),
            )
        }
        if node_count != len(observed_nodes):
            raise ContractError(
                "S04 conflict group node count does not match proposal rows: "
                f"{group_id}"
            )
        result.append(
            _CompleteGroup(
                conflict_group_id=group_id,
                conflict_group_edge_count=edge_count,
                conflict_group_node_count=node_count,
                proposals=tuple(proposals),
            )
        )
    return tuple(result)


def build_s04_conflict_review_plan(
    proposals: Mapping[str, Sequence[Any] | np.ndarray],
    *,
    random_seed: int,
    largest_group_count: int = 3,
    random_group_count: int = 3,
    max_edges_per_group: int = 3,
) -> S04ConflictReviewPlan:
    """Select three largest and three seeded-random conflict groups."""

    if isinstance(random_seed, bool) or not isinstance(random_seed, int):
        raise ContractError("S04 conflict review random_seed must be an integer")
    counts = (largest_group_count, random_group_count, max_edges_per_group)
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in counts
    ):
        raise ContractError(
            "S04 conflict review selection counts must be non-negative integers"
        )
    if max_edges_per_group == 0:
        raise ContractError("S04 conflict review max_edges_per_group must be positive")

    records = parse_proposal_records(proposals)
    complete_groups = _validated_groups(records)
    eligible_groups = tuple(
        group
        for group in complete_groups
        if group.conflict_group_edge_count > 1
    )
    eligible_ids = {group.conflict_group_id for group in eligible_groups}
    eligible_proposals = tuple(
        proposal
        for proposal in records
        if proposal.conflict_group_id in eligible_ids
    )

    largest_order = sorted(
        eligible_groups,
        key=lambda group: (
            -group.conflict_group_edge_count,
            -group.conflict_group_node_count,
            group.conflict_group_id,
        ),
    )
    largest = largest_order[:largest_group_count]
    largest_ids = {group.conflict_group_id for group in largest}
    remainder = sorted(
        (
            group
            for group in eligible_groups
            if group.conflict_group_id not in largest_ids
        ),
        key=lambda group: group.conflict_group_id,
    )
    random_groups = random.Random(random_seed).sample(
        remainder, min(random_group_count, len(remainder))
    )
    selections = (
        [(group, "largest") for group in largest]
        + [(group, "random") for group in random_groups]
    )

    selected_groups: list[S04ConflictReviewGroup] = []
    for selection_order, (group, selection_group) in enumerate(selections, start=1):
        rendered = tuple(
            sorted(
                group.proposals,
                key=lambda proposal: (-proposal.probability, proposal.proposal_id),
            )[:max_edges_per_group]
        )
        selected_groups.append(
            S04ConflictReviewGroup(
                conflict_group_id=group.conflict_group_id,
                conflict_group_edge_count=group.conflict_group_edge_count,
                conflict_group_node_count=group.conflict_group_node_count,
                selection_group=selection_group,
                selection_order=selection_order,
                rendered_proposals=rendered,
            )
        )

    plan = S04ConflictReviewPlan(
        proposals=records,
        eligible_proposals=eligible_proposals,
        groups=tuple(selected_groups),
        random_seed=random_seed,
        largest_group_count=largest_group_count,
        random_group_count=random_group_count,
        max_edges_per_group=max_edges_per_group,
    )
    group_ids = [group.conflict_group_id for group in plan.groups]
    proposal_ids = [case.proposal.proposal_id for case in plan.cases]
    if len(group_ids) != len(set(group_ids)):
        raise ContractError("internal error: duplicate S04 conflict group selected")
    if len(proposal_ids) != len(set(proposal_ids)):
        raise ContractError("internal error: duplicate S04 conflict edge selected")
    expected_group_count = min(
        len(eligible_groups), largest_group_count + random_group_count
    )
    if len(plan.groups) != expected_group_count:
        raise ContractError(
            "internal error: S04 conflict group selection is incomplete"
        )
    return plan


__all__ = [
    "PLAN_SCHEMA_VERSION",
    "S04ConflictReviewCase",
    "S04ConflictReviewGroup",
    "S04ConflictReviewPlan",
    "build_s04_conflict_review_plan",
]
