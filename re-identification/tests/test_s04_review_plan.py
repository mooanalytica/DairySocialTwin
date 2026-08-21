from __future__ import annotations

import numpy as np
import pytest

from cowtrack.config import ContractError
from cowtrack.qa.s04_review_plan import build_s04_review_plan


def _proposals(count: int) -> dict[str, np.ndarray]:
    return {
        "proposal_id": np.asarray([f"p{index:03d}" for index in range(count)]),
        "edge_id": np.asarray([f"e{index:03d}" for index in range(count)]),
        "source_micro_id": np.arange(count, dtype=np.int64),
        "target_micro_id": np.arange(count, dtype=np.int64) + 1000,
        "probability": np.linspace(0.01, 0.99, count, dtype=np.float64),
        "high_overlap": np.zeros(count, dtype=np.bool_),
        "source_rank": np.ones(count, dtype=np.int32),
        "target_rank": np.ones(count, dtype=np.int32),
        "conflict_degree": np.zeros(count, dtype=np.int32),
        "conflict_group_id": np.asarray(
            [f"g{index:03d}" for index in range(count)]
        ),
        "conflict_group_edge_count": np.ones(count, dtype=np.int32),
        "conflict_group_node_count": np.full(count, 2, dtype=np.int32),
        "review_status": np.asarray(["pending"] * count),
    }


def test_selects_top_bottom_then_seeded_random_without_duplicates() -> None:
    columns = _proposals(75)
    first = build_s04_review_plan(columns, random_seed=20260710)
    second = build_s04_review_plan(columns, random_seed=20260710)

    ids = [case.proposal.proposal_id for case in first.cases]
    assert len(ids) == 50
    assert len(set(ids)) == 50
    assert ids[:15] == [f"p{index:03d}" for index in range(74, 59, -1)]
    assert ids[15:30] == [f"p{index:03d}" for index in range(15)]
    assert ids[30:] == [case.proposal.proposal_id for case in second.cases[30:]]
    assert [case.selection_group for case in first.cases] == (
        ["top"] * 15 + ["bottom"] * 15 + ["random"] * 20
    )


def test_fewer_than_50_selects_every_proposal_once() -> None:
    plan = build_s04_review_plan(_proposals(17), random_seed=20260710)

    assert len(plan.cases) == 17
    assert [case.selection_group for case in plan.cases] == ["all"] * 17
    assert [case.proposal.proposal_id for case in plan.cases] == [
        f"p{index:03d}" for index in range(16, -1, -1)
    ]


def test_filename_contains_probability_to_four_decimal_places() -> None:
    columns = _proposals(1)
    columns["probability"][0] = 0.98765
    case = build_s04_review_plan(columns, random_seed=20260710).cases[0]

    assert "p0.9877" in case.suggested_filename
    assert case.suggested_filename.endswith(".mp4")


def test_intermediate_ids_are_other_conflict_component_endpoints() -> None:
    columns = _proposals(3)
    columns["source_micro_id"][:] = [1, 1, 4]
    columns["target_micro_id"][:] = [2, 3, 2]
    columns["conflict_group_id"][:] = "same"
    columns["conflict_group_edge_count"][:] = 3
    columns["conflict_group_node_count"][:] = 4
    plan = build_s04_review_plan(columns, random_seed=20260710)
    focal = next(case for case in plan.cases if case.proposal.proposal_id == "p000")

    assert plan.intermediate_micro_ids(focal) == (3, 4)
    payload = plan.to_manifest_payload()
    assert "No interpolation" in payload["visual_contract"]["purple_box_policy"]
    assert payload["visual_contract"]["text"] is False


def test_rejects_non_pending_or_duplicate_proposals() -> None:
    columns = _proposals(2)
    columns["review_status"][1] = "accepted"
    with pytest.raises(ContractError, match="pending"):
        build_s04_review_plan(columns, random_seed=20260710)

    columns = _proposals(2)
    columns["proposal_id"][1] = columns["proposal_id"][0]
    with pytest.raises(ContractError, match="proposal_id values must be unique"):
        build_s04_review_plan(columns, random_seed=20260710)
