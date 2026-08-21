from __future__ import annotations

import random
from dataclasses import replace

import pytest

from cowtrack.config import ContractError
from cowtrack.linking.path_cover import (
    GlobalProposalEdge,
    GlobalStableNode,
    global_proposal_edge_from_row,
    global_stable_node_from_row,
    solve_operator_approved_path_cover,
)


def _node(
    stable_id: int,
    start: float,
    end: float,
    *,
    clip: str = "GX040006",
) -> GlobalStableNode:
    return GlobalStableNode(
        stable_id=stable_id,
        start_clip_id=clip,
        end_clip_id=clip,
        start_global_frame=round(start * 10),
        end_global_frame=round(end * 10),
        start_time_sec=start,
        end_time_sec=end,
        num_microtracklets=1,
        num_detections=10,
    )


def _edge(
    source: GlobalStableNode,
    target: GlobalStableNode,
    token: str,
    *,
    probability: float = 0.9,
    margin: float | None = 0.1,
    rank_out: int = 1,
    rank_in: int = 1,
    high_overlap: bool = False,
    mutual: float = 0.8,
    gap: float | None = None,
) -> GlobalProposalEdge:
    return GlobalProposalEdge(
        proposal_id=f"proposal-{token}",
        candidate_id=f"candidate-{token}",
        source_stable_id=source.stable_id,
        target_stable_id=target.stable_id,
        probability=probability,
        candidate_margin=margin,
        rank_out=rank_out,
        rank_in=rank_in,
        high_overlap=high_overlap,
        gallery_score_mutual=mutual,
        temporal_gap_sec=(
            target.start_time_sec - source.end_time_sec if gap is None else gap
        ),
    )


def _selected_ids(result) -> tuple[str, ...]:
    return tuple(edge.candidate_id for edge in result.selected_edges)


def test_empty_edges_produce_deterministic_total_singleton_cover() -> None:
    nodes = [_node(9, 30.0, 31.0), _node(2, 0.0, 1.0), _node(5, 10.0, 11.0)]

    result = solve_operator_approved_path_cover(nodes, [])

    assert result.selected_edges == ()
    assert result.max_cardinality == 0
    assert result.paths == ((2,), (5,), (9,))
    assert result.num_nodes == 3
    assert result.num_selected_links == 0
    assert result.num_paths == 3
    assert result.predecessor_by_stable == {2: None, 5: None, 9: None}
    assert result.successor_by_stable == {2: None, 5: None, 9: None}
    assert result.solver_cost_by_candidate == {}


def test_primary_objective_maximizes_cardinality_before_evidence() -> None:
    n0 = _node(0, 0.0, 1.0)
    n1 = _node(1, 0.2, 1.2)
    n2 = _node(2, 10.0, 11.0)
    n3 = _node(3, 20.0, 21.0)
    excellent_but_blocking = _edge(n0, n2, "02", probability=0.9999)
    weak_03 = _edge(
        n0,
        n3,
        "03",
        probability=0.01,
        margin=-0.8,
        rank_out=20,
        rank_in=20,
        high_overlap=True,
        mutual=-0.5,
    )
    weak_12 = _edge(
        n1,
        n2,
        "12",
        probability=0.02,
        margin=None,
        rank_out=19,
        rank_in=18,
        high_overlap=True,
        mutual=-0.4,
    )

    result = solve_operator_approved_path_cover(
        [n3, n1, n0, n2], [excellent_but_blocking, weak_03, weak_12]
    )

    assert set(_selected_ids(result)) == {"candidate-03", "candidate-12"}
    assert result.max_cardinality == 2
    assert result.num_paths == len(result.predecessor_by_stable) - 2
    assert result.successor_by_stable[0] == 3
    assert result.successor_by_stable[1] == 2
    assert result.predecessor_by_stable[2] == 1
    assert result.predecessor_by_stable[3] == 0


def test_secondary_cost_prefers_clean_ranks_probability_margin_and_mutual() -> None:
    source = _node(0, 0.0, 1.0)
    targets = [_node(index, 10.0 * index, 10.0 * index + 1.0) for index in range(1, 7)]
    clean = _edge(source, targets[0], "clean", probability=0.1)
    crowded = _edge(
        source, targets[1], "crowded", probability=1.0, high_overlap=True
    )
    rank_good = _edge(source, targets[2], "rank-good", probability=0.1)
    rank_bad = _edge(
        source,
        targets[3],
        "rank-bad",
        probability=1.0,
        rank_out=2,
        rank_in=1,
    )
    probability_high = _edge(
        source, targets[4], "p-high", probability=0.9, margin=0.01
    )
    probability_low = _edge(
        source, targets[5], "p-low", probability=0.8, margin=0.9
    )
    nodes = [source, *targets]

    first = solve_operator_approved_path_cover(nodes, [crowded, clean])
    ranks = solve_operator_approved_path_cover(nodes, [rank_bad, rank_good])
    probabilities = solve_operator_approved_path_cover(
        nodes, [probability_low, probability_high]
    )
    same_probability = solve_operator_approved_path_cover(
        nodes,
        [
            replace(probability_high, candidate_id="candidate-margin-low"),
            replace(
                probability_low,
                probability=0.9,
                candidate_margin=0.2,
                candidate_id="candidate-margin-high",
            ),
        ],
    )
    same_margin = solve_operator_approved_path_cover(
        nodes,
        [
            replace(clean, candidate_id="candidate-mutual-low", gallery_score_mutual=0.2),
            replace(
                rank_good,
                candidate_id="candidate-mutual-high",
                probability=clean.probability,
                gallery_score_mutual=0.9,
            ),
        ],
    )

    assert _selected_ids(first) == ("candidate-clean",)
    assert _selected_ids(ranks) == ("candidate-rank-good",)
    assert _selected_ids(probabilities) == ("candidate-p-high",)
    assert _selected_ids(same_probability) == ("candidate-margin-high",)
    assert _selected_ids(same_margin) == ("candidate-mutual-high",)


def test_incoming_and_outgoing_conflicts_obey_one_port_capacity() -> None:
    n0 = _node(0, 0.0, 1.0)
    n1 = _node(1, 0.1, 1.1)
    n2 = _node(2, 10.0, 11.0)
    n3 = _node(3, 20.0, 21.0)
    edges = [
        _edge(n0, n2, "02", probability=0.8),
        _edge(n0, n3, "03", probability=0.9),
        _edge(n1, n3, "13", probability=0.95),
    ]

    result = solve_operator_approved_path_cover([n0, n1, n2, n3], edges)

    assert result.max_cardinality == 2
    assert set(_selected_ids(result)) == {"candidate-02", "candidate-13"}
    assert all(
        sum(edge.source_stable_id == stable_id for edge in result.selected_edges) <= 1
        for stable_id in result.successor_by_stable
    )
    assert all(
        sum(edge.target_stable_id == stable_id for edge in result.selected_edges) <= 1
        for stable_id in result.predecessor_by_stable
    )


def test_chains_are_allowed_and_p_equals_n_minus_l_with_total_coverage() -> None:
    nodes = [_node(0, 0.0, 1.0), _node(1, 10.0, 11.0), _node(2, 20.0, 21.0)]
    edges = [
        _edge(nodes[0], nodes[1], "01"),
        _edge(nodes[1], nodes[2], "12"),
        _edge(nodes[0], nodes[2], "02", probability=1.0),
    ]

    result = solve_operator_approved_path_cover(nodes, edges)

    assert set(_selected_ids(result)) == {"candidate-01", "candidate-12"}
    assert result.paths == ((0, 1, 2),)
    assert result.predecessor_by_stable == {0: None, 1: 0, 2: 1}
    assert result.successor_by_stable == {0: 1, 1: 2, 2: None}
    assert result.num_paths == result.num_nodes - result.num_selected_links == 1
    assert sorted(stable_id for path in result.paths for stable_id in path) == [0, 1, 2]


def test_node_and_edge_row_shuffle_is_deterministic_even_with_cost_tie() -> None:
    nodes = [
        _node(0, 0.0, 1.0),
        _node(1, 0.1, 1.1),
        _node(2, 10.0, 11.0),
        _node(3, 10.1, 11.1),
    ]
    edges = [
        _edge(nodes[0], nodes[2], "a"),
        _edge(nodes[0], nodes[3], "b"),
        _edge(nodes[1], nodes[2], "c"),
        _edge(nodes[1], nodes[3], "d"),
    ]
    expected = solve_operator_approved_path_cover(nodes, edges)
    randomizer = random.Random(20260710)

    for _ in range(20):
        shuffled_nodes = nodes.copy()
        shuffled_edges = edges.copy()
        randomizer.shuffle(shuffled_nodes)
        randomizer.shuffle(shuffled_edges)
        actual = solve_operator_approved_path_cover(shuffled_nodes, shuffled_edges)
        assert actual.selected_edges == expected.selected_edges
        assert actual.paths == expected.paths
        assert actual.predecessor_by_stable == expected.predecessor_by_stable
        assert actual.successor_by_stable == expected.successor_by_stable
        assert actual.solver_cost_by_candidate == expected.solver_cost_by_candidate


def test_cross_clip_forward_edge_uses_global_chronology() -> None:
    source = _node(7, 0.0, 1.0, clip="GX040006")
    target = _node(8, 10.0, 11.0, clip="GX050006")

    result = solve_operator_approved_path_cover(
        [target, source], [_edge(source, target, "cross")]
    )

    assert result.paths == ((7, 8),)


def test_cycle_is_rejected_before_assignment() -> None:
    first = _node(0, 0.0, 1.0)
    second = _node(1, 10.0, 11.0)
    cycle = [
        _edge(first, second, "01"),
        _edge(second, first, "10", gap=6.0),
    ]

    with pytest.raises(ContractError, match="cycle"):
        solve_operator_approved_path_cover([first, second], cycle)


def test_reverse_overlap_and_non_long_gap_are_rejected() -> None:
    early = _node(0, 0.0, 10.0)
    overlapping = _node(1, 5.0, 15.0)
    late = _node(2, 20.0, 21.0)

    with pytest.raises(ContractError, match="reverse, overlapping, or has a false gap"):
        solve_operator_approved_path_cover(
            [early, late], [_edge(late, early, "reverse", gap=6.0)]
        )
    with pytest.raises(ContractError, match="reverse, overlapping, or has a false gap"):
        solve_operator_approved_path_cover(
            [early, overlapping], [_edge(early, overlapping, "overlap", gap=6.0)]
        )
    with pytest.raises(ContractError, match=">5s"):
        solve_operator_approved_path_cover(
            [early, late], [_edge(early, late, "five", gap=5.0)]
        )

    exact_five = _node(3, 15.0, 16.0)
    with pytest.raises(ContractError, match="reverse, overlapping, or has a false gap"):
        solve_operator_approved_path_cover(
            [early, exact_five],
            [_edge(early, exact_five, "rounded-over-five", gap=5.000001)],
        )


def test_malformed_public_inputs_fail_with_contract_error() -> None:
    first = _node(0, 0.0, 1.0)

    with pytest.raises(ContractError, match="node has the wrong type"):
        solve_operator_approved_path_cover([first, object()], [])  # type: ignore[list-item]
    with pytest.raises(ContractError, match="edge has the wrong type"):
        solve_operator_approved_path_cover([first], [object()])  # type: ignore[list-item]


def test_duplicate_directed_pair_and_unknown_node_fail_closed() -> None:
    first = _node(0, 0.0, 1.0)
    second = _node(1, 10.0, 11.0)
    duplicate = replace(
        _edge(first, second, "a"),
        proposal_id="proposal-b",
        candidate_id="candidate-b",
    )

    with pytest.raises(ContractError, match="directed proposal pair"):
        solve_operator_approved_path_cover(
            [first, second], [_edge(first, second, "a"), duplicate]
        )
    with pytest.raises(ContractError, match="unknown stable ID"):
        solve_operator_approved_path_cover(
            [first, second],
            [replace(_edge(first, second, "unknown"), target_stable_id=99)],
        )


def test_row_builders_preserve_provisional_semantics() -> None:
    node = global_stable_node_from_row(
        {
            "stable_id": 3,
            "start_clip_id": "GX040006",
            "end_clip_id": "GX050006",
            "start_global_frame": 4,
            "end_global_frame": 40,
            "start_time_sec": 0.4,
            "end_time_sec": 4.0,
            "num_microtracklets": 2,
            "num_detections": 20,
        }
    )
    proposal = {
        "proposal_id": "s05p-000003-000004",
        "candidate_id": "s05c-000003-000004",
        "source_stable_id": 3,
        "target_stable_id": 4,
        "model_probability": 0.8,
        "candidate_margin": -0.2,
        "appearance_rank_out": 2,
        "appearance_rank_in": 4,
        "high_overlap": True,
        "gallery_score_mutual": 0.7,
        "temporal_gap_sec": 8.0,
        "appearance_present": True,
        "temporally_nonoverlapping": True,
        "decision": "provisional",
        "review_status": "pending",
        "selected_by_solver": False,
        "confirmed": False,
        "merge_applied": False,
    }

    edge = global_proposal_edge_from_row(proposal)

    assert node.stable_id == 3
    assert node.start_clip_id == "GX040006"
    assert node.end_clip_id == "GX050006"
    assert edge.probability == 0.8
    assert edge.high_overlap is True
    assert edge.candidate_margin == -0.2

    with pytest.raises(ContractError, match="pending provisional"):
        global_proposal_edge_from_row({**proposal, "decision": "confirmed"})
    with pytest.raises(ContractError, match="identity-changing state"):
        global_proposal_edge_from_row({**proposal, "selected_by_solver": True})
    with pytest.raises(ContractError, match="lacks appearance"):
        global_proposal_edge_from_row({**proposal, "appearance_present": False})
    with pytest.raises(ContractError, match="lacks candidate_id"):
        global_proposal_edge_from_row(
            {key: value for key, value in proposal.items() if key != "candidate_id"}
        )
