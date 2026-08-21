from __future__ import annotations

import itertools
import random

import numpy as np
import pytest

from cowtrack.config import ContractError
from cowtrack.qa.s04_conflict_review_plan import (
    build_s04_conflict_review_plan,
)


def _proposals(
    groups: list[tuple[str, int, int]],
) -> dict[str, np.ndarray]:
    rows: list[dict[str, object]] = []
    next_micro_id = 1_000
    for group_id, edge_count, node_count in groups:
        source_count = next(
            count
            for count in range(1, node_count)
            if count * (node_count - count) >= edge_count
            and max(count, node_count - count) <= edge_count
        )
        target_count = node_count - source_count
        sources = list(range(next_micro_id, next_micro_id + source_count))
        targets = list(
            range(
                next_micro_id + source_count,
                next_micro_id + source_count + target_count,
            )
        )
        next_micro_id += source_count + target_count + 10
        initial = [
            (sources[index % source_count], targets[index % target_count])
            for index in range(max(source_count, target_count))
        ]
        candidates = initial + list(itertools.product(sources, targets))
        pairs = list(dict.fromkeys(candidates))[:edge_count]
        assert len(pairs) == edge_count
        assert (
            len({source for source, _ in pairs})
            + len({target for _, target in pairs})
            == node_count
        )
        for edge_index, (source, target) in enumerate(pairs):
            token = f"{group_id}-{edge_index}"
            rows.append(
                {
                    "proposal_id": f"proposal-{token}",
                    "edge_id": f"edge-{token}",
                    "source_micro_id": source,
                    "target_micro_id": target,
                    "probability": 0.5 + edge_index / 100.0,
                    "high_overlap": False,
                    "source_rank": edge_index + 1,
                    "target_rank": edge_index + 1,
                    "conflict_degree": max(edge_count - 1, 0),
                    "conflict_group_id": group_id,
                    "conflict_group_edge_count": edge_count,
                    "conflict_group_node_count": node_count,
                    "review_status": "pending",
                }
            )
    return {
        name: np.asarray([row[name] for row in rows])
        for name in rows[0]
    }


def test_selects_largest_then_seeded_random_groups_without_duplicates() -> None:
    columns = _proposals(
        [
            ("largest", 5, 6),
            ("same-edges-more-nodes", 4, 5),
            ("same-edges-fewer-nodes", 4, 4),
            ("random-a", 2, 3),
            ("random-b", 2, 3),
            ("random-c", 2, 3),
            ("random-d", 2, 3),
            ("random-e", 2, 3),
            ("isolated", 1, 2),
        ]
    )

    first = build_s04_conflict_review_plan(columns, random_seed=20260710)
    second = build_s04_conflict_review_plan(columns, random_seed=20260710)

    selected_ids = [group.conflict_group_id for group in first.groups]
    assert selected_ids[:3] == [
        "largest",
        "same-edges-more-nodes",
        "same-edges-fewer-nodes",
    ]
    remainder = [
        "random-a",
        "random-b",
        "random-c",
        "random-d",
        "random-e",
    ]
    assert selected_ids[3:] == random.Random(20260710).sample(remainder, 3)
    assert selected_ids == [group.conflict_group_id for group in second.groups]
    assert len(selected_ids) == len(set(selected_ids)) == 6
    assert "isolated" not in selected_ids
    proposal_ids = [case.proposal.proposal_id for case in first.cases]
    assert len(proposal_ids) == len(set(proposal_ids))


def test_selects_top_three_edges_and_records_full_group_sizes() -> None:
    plan = build_s04_conflict_review_plan(
        _proposals([("large", 5, 6), ("small", 2, 3)]),
        random_seed=20260710,
    )

    large = plan.groups[0]
    assert large.conflict_group_id == "large"
    assert (large.conflict_group_edge_count, large.conflict_group_node_count) == (5, 6)
    assert [proposal.probability for proposal in large.rendered_proposals] == [
        0.54,
        0.53,
        0.52,
    ]
    assert large.rendered_subset_is_partial is True
    small = plan.groups[1]
    assert len(small.rendered_proposals) == 2
    assert small.rendered_subset_is_partial is False

    payload = plan.to_manifest_payload()
    assert payload["summary"] == {
        "num_eligible_proposals": 7,
        "num_eligible_groups": 2,
        "num_selected_groups": 2,
        "num_selected_edges": 5,
        "num_partial_groups": 1,
    }
    assert payload["groups"][0]["conflict_group_edge_count"] == 5
    assert payload["groups"][0]["conflict_group_node_count"] == 6
    assert payload["groups"][0]["rendered_subset_is_partial"] is True


def test_shared_renderer_interfaces_use_full_group_not_rendered_subset() -> None:
    columns = _proposals([("group", 5, 6), ("isolated", 1, 2)])
    plan = build_s04_conflict_review_plan(columns, random_seed=20260710)
    focal = plan.cases[0]

    assert len(plan.proposals) == 6
    assert len(plan.eligible_proposals) == 5
    assert len(plan.groups[0].rendered_proposals) == 3
    assert plan.intermediate_micro_ids(focal) == tuple(
        sorted(
            {
                int(value)
                for name in ("source_micro_id", "target_micro_id")
                for value in columns[name][:5]
            }
            - {
                focal.proposal.source_micro_id,
                focal.proposal.target_micro_id,
            }
        )
    )
    assert focal.suggested_filename.startswith("001_largest_edge01_")
    assert focal.suggested_filename.endswith(
        f"p{focal.proposal.probability:.4f}.mp4"
    )
    assert focal.to_manifest_record()["suggested_filename"] == focal.suggested_filename


def test_group_node_validation_is_bipartite_role_aware() -> None:
    columns = _proposals([("group", 3, 4)])
    columns["source_micro_id"][:] = [1, 1, 2]
    columns["target_micro_id"][:] = [2, 3, 3]

    plan = build_s04_conflict_review_plan(columns, random_seed=20260710)

    assert plan.groups[0].conflict_group_node_count == 4


def test_filename_hashes_unsafe_proposal_token() -> None:
    columns = _proposals([("group", 2, 3)])
    columns["proposal_id"][1] = "bad/name"

    case = build_s04_conflict_review_plan(
        columns, random_seed=20260710
    ).cases[0]

    assert case.proposal.proposal_id == "bad/name"
    assert "bad/name" not in case.suggested_filename
    assert "proposal-" in case.suggested_filename


def test_probability_ties_are_broken_by_proposal_id() -> None:
    columns = _proposals([("group", 4, 5)])
    columns["probability"][:] = 0.8

    plan = build_s04_conflict_review_plan(columns, random_seed=20260710)

    assert [proposal.proposal_id for proposal in plan.groups[0].rendered_proposals] == [
        "proposal-group-0",
        "proposal-group-1",
        "proposal-group-2",
    ]


def test_selects_all_available_groups_when_fewer_than_six() -> None:
    plan = build_s04_conflict_review_plan(
        _proposals([("a", 2, 3), ("b", 3, 4), ("c", 2, 3), ("d", 2, 3)]),
        random_seed=20260710,
    )

    assert len(plan.groups) == 4
    assert {group.conflict_group_id for group in plan.groups} == {"a", "b", "c", "d"}


def test_rejects_inconsistent_group_metadata() -> None:
    columns = _proposals([("group", 2, 3)])
    columns["conflict_group_edge_count"][1] = 3

    with pytest.raises(ContractError, match="metadata is inconsistent"):
        build_s04_conflict_review_plan(columns, random_seed=20260710)
