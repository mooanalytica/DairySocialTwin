from __future__ import annotations

import numpy as np
import pytest

from cowtrack.config import ContractError
from cowtrack.qa.s05_review_plan import CANDIDATE_COLUMNS, build_s05_review_plan
from cowtrack.schemas.s05_proposals import LONG_CANDIDATE_EDGES_SCHEMA


def _candidates(count: int = 65) -> dict[str, np.ndarray]:
    if count < 1:
        raise ValueError("test fixture count must be positive")
    probabilities = np.full(count, 0.65, dtype=np.float64)
    margins = np.full(count, 0.20, dtype=np.float64)
    rank_out = np.ones(count, dtype=np.int32)
    rank_in = np.ones(count, dtype=np.int32)
    for index in range(min(15, count)):
        probabilities[index] = 0.99 - index * 0.004
    for index in range(15, min(30, count)):
        probabilities[index] = 0.5005 + (index - 15) * 0.001
    for index in range(30, min(40, count)):
        probabilities[index] = 0.75 - (index - 30) * 0.001
        margins[index] = 0.01
        rank_out[index] = 2
        rank_in[index] = 2
    margin_gate = margins >= 0.10
    probability_gate = probabilities >= 0.70
    high_overlap = np.zeros(count, dtype=np.bool_)
    return {
        "candidate_id": np.asarray([f"c{index:03d}" for index in range(count)]),
        "source_stable_id": np.arange(count, dtype=np.int64),
        "target_stable_id": np.arange(count, dtype=np.int64) + 1000,
        "temporally_nonoverlapping": np.ones(count, dtype=np.bool_),
        "appearance_present": np.ones(count, dtype=np.bool_),
        "high_overlap": high_overlap,
        "appearance_rank_out": rank_out,
        "appearance_rank_in": rank_in,
        "best_margin_out": margins + 0.03,
        "best_margin_in": margins,
        "model_probability": probabilities,
        "candidate_margin": margins.copy(),
        "provisional_threshold": np.full(count, 0.50, dtype=np.float64),
        "selected_probability_threshold": np.full(
            count, 0.70, dtype=np.float64
        ),
        "selected_margin_threshold": np.full(count, 0.10, dtype=np.float64),
        "passes_provisional_threshold": np.ones(count, dtype=np.bool_),
        "passes_selected_probability_gate": probability_gate,
        "passes_selected_margin_gate": margin_gate,
        "passes_selected_gate": probability_gate & margin_gate & ~high_overlap,
        "decision": np.asarray(["provisional"] * count),
        "selected_by_solver": np.zeros(count, dtype=np.bool_),
        "confirmed": np.zeros(count, dtype=np.bool_),
        "merge_applied": np.zeros(count, dtype=np.bool_),
        "proposed_for_review": np.ones(count, dtype=np.bool_),
    }


def _build(columns: dict[str, np.ndarray]):
    return build_s05_review_plan(
        columns,
        provisional_threshold=0.50,
        random_seed=20260710,
    )


def test_review_projection_is_promised_by_candidate_edge_schema() -> None:
    assert set(CANDIDATE_COLUMNS) <= set(LONG_CANDIDATE_EDGES_SCHEMA.names)


def test_stratified_selection_is_bounded_deterministic_and_shuffle_invariant() -> None:
    columns = _candidates()
    first = _build(columns)
    second = _build({name: values[::-1] for name, values in columns.items()})

    first_ids = [case.candidate.candidate_id for case in first.cases]
    second_ids = [case.candidate.candidate_id for case in second.cases]
    assert first_ids == second_ids
    assert len(first_ids) == 50
    assert len(set(first_ids)) == 50
    assert first_ids[:15] == [f"c{index:03d}" for index in range(15)]
    assert first_ids[15:30] == [f"c{index:03d}" for index in range(15, 30)]
    assert set(first_ids[30:40]) == {f"c{index:03d}" for index in range(30, 40)}
    assert [case.selection_group for case in first.cases] == (
        ["high_score"] * 15
        + ["threshold_near"] * 15
        + ["ambiguous"] * 10
        + ["random"] * 10
    )
    payload = first.to_manifest_payload()
    assert payload["candidate_semantics"] == "provisional_review_evidence_only"
    assert payload["automatic_merge_allowed"] is False
    assert payload["solver_used"] is False
    assert payload["visual_contract"]["text"] is False
    assert payload["visual_contract"]["colors"] == {
        "unrelated": "gray",
        "source": "yellow",
        "target": "green",
        "intermediate_or_competing_stable_state": "purple",
    }


def test_small_input_selects_every_provisional_candidate_once() -> None:
    plan = _build(_candidates(9))
    assert len(plan.cases) == 9
    assert {case.selection_group for case in plan.cases} == {"all_available"}
    assert [case.candidate.candidate_id for case in plan.cases] == [
        f"c{index:03d}" for index in range(9)
    ]


def test_candidate_margin_is_directional_min_and_missing_is_ambiguous() -> None:
    columns = _candidates()
    columns["best_margin_out"] = columns["best_margin_out"].astype(object)
    columns["best_margin_in"] = columns["best_margin_in"].astype(object)
    columns["appearance_rank_out"] = columns["appearance_rank_out"].astype(object)
    columns["best_margin_out"][30] = None
    columns["candidate_margin"] = columns["candidate_margin"].astype(object)
    columns["candidate_margin"][30] = None
    columns["appearance_rank_out"][30] = None
    columns["passes_selected_margin_gate"][30] = False
    columns["passes_selected_gate"][30] = False
    plan = _build(columns)
    record = next(item for item in plan.candidates if item.candidate_id == "c030")
    assert record.candidate_margin is None
    assert record.appearance_rank_out is None
    assert any(
        case.candidate.candidate_id == "c030" and case.selection_group == "ambiguous"
        for case in plan.cases
    )
    ordinary = next(item for item in plan.candidates if item.candidate_id == "c031")
    assert ordinary.candidate_margin == pytest.approx(0.01)


def test_competing_ids_come_from_rows_sharing_focal_endpoint() -> None:
    columns = _candidates(3)
    columns["source_stable_id"][:] = [1, 1, 4]
    columns["target_stable_id"][:] = [2, 3, 2]
    plan = _build(columns)
    focal = next(case for case in plan.cases if case.candidate.candidate_id == "c000")

    assert plan.competing_stable_ids(focal) == (3, 4)
    assert "Never interpolate" in plan.to_manifest_payload()["visual_contract"][
        "purple_stable_policy"
    ]


def test_rejects_nonprovisional_or_identity_changing_rows() -> None:
    columns = _candidates(2)
    columns["decision"][0] = "confirmed"
    with pytest.raises(ContractError, match="decision='provisional'"):
        _build(columns)

    columns = _candidates(2)
    columns["proposed_for_review"][0] = False
    with pytest.raises(ContractError, match="proposed_for_review=true"):
        _build(columns)

    columns = _candidates(2)
    columns["merge_applied"][0] = True
    with pytest.raises(ContractError, match="merge_applied=false"):
        _build(columns)


def test_selected_gate_evidence_requires_probability_margin_and_no_overlap() -> None:
    columns = _candidates(2)
    columns["high_overlap"][0] = True
    # Deliberately leave the selected gate true to prove overlap is part of it.
    with pytest.raises(ContractError, match="selected-gate evidence is inconsistent"):
        _build(columns)

    columns["passes_selected_gate"][0] = False
    plan = _build(columns)
    assert plan.candidates[0].high_overlap is True
    assert plan.candidates[0].passes_selected_gate is False


def test_rejects_probability_below_supplied_frozen_threshold() -> None:
    columns = _candidates(2)
    columns["provisional_threshold"][:] = 0.51
    with pytest.raises(ContractError, match="differs from the supplied frozen threshold"):
        _build(columns)
