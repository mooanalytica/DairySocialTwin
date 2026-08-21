from __future__ import annotations

import random
from dataclasses import replace
from functools import lru_cache

import pytest

from cowtrack.config import ContractError
from cowtrack.linking.forced_path_cover import (
    ForcedAppearanceEdge,
    solve_forced_fixed_path_cover,
)
from cowtrack.linking.path_cover import GlobalStableNode


def _node(
    stable_id: int,
    *,
    start_frame: int,
    end_frame: int,
    start_time: float | None = None,
    end_time: float | None = None,
) -> GlobalStableNode:
    return GlobalStableNode(
        stable_id=stable_id,
        start_clip_id="clip",
        end_clip_id="clip",
        start_global_frame=start_frame,
        end_global_frame=end_frame,
        start_time_sec=float(start_frame if start_time is None else start_time),
        end_time_sec=float(end_frame if end_time is None else end_time),
        num_microtracklets=1,
        num_detections=2,
    )


def _edge(
    source: int,
    target: int,
    cost: int,
    *,
    token: str | None = None,
) -> ForcedAppearanceEdge:
    return ForcedAppearanceEdge(
        edge_id=token or f"edge-{source}-{target}",
        source_stable_id=source,
        target_stable_id=target,
        appearance_cost_int=cost,
    )


def _two_layer_nodes() -> list[GlobalStableNode]:
    return [
        _node(0, start_frame=0, end_frame=4),
        _node(1, start_frame=0, end_frame=4),
        _node(2, start_frame=10, end_frame=14),
        _node(3, start_frame=10, end_frame=14),
    ]


def test_exact_k_cover_minimizes_global_cost_not_greedy_cost() -> None:
    nodes = _two_layer_nodes()
    edges = [
        _edge(0, 2, 0),
        _edge(0, 3, 1),
        _edge(1, 2, 1),
        _edge(1, 3, 100),
    ]

    result = solve_forced_fixed_path_cover(nodes, edges, target_num_paths=2)

    assert [(edge.source_stable_id, edge.target_stable_id) for edge in result.selected_edges] == [
        (0, 3),
        (1, 2),
    ]
    assert result.total_appearance_cost_int == 2
    assert result.required_links == 2
    assert result.maximum_feasible_links == 2
    assert result.num_selected_links == 2
    assert result.num_paths == result.target_num_paths == 2
    assert sorted(stable_id for path in result.paths for stable_id in path) == [0, 1, 2, 3]


def test_target_k_selects_exactly_n_minus_k_even_when_more_links_are_possible() -> None:
    nodes = _two_layer_nodes()
    edges = [
        _edge(0, 2, 0),
        _edge(0, 3, 1),
        _edge(1, 2, 1),
        _edge(1, 3, 0),
    ]

    result = solve_forced_fixed_path_cover(nodes, edges, target_num_paths=3)

    assert result.maximum_feasible_links == 2
    assert result.required_links == 1
    assert result.num_selected_links == 1
    assert result.num_paths == 3


def test_fixed_solver_reports_maximum_and_min_cost_flow_phases() -> None:
    nodes = _two_layer_nodes()
    edges = [
        _edge(0, 2, 0),
        _edge(0, 3, 3),
        _edge(1, 2, 2),
        _edge(1, 3, 0),
    ]
    messages: list[str] = []

    result = solve_forced_fixed_path_cover(
        nodes,
        edges,
        target_num_paths=3,
        progress_interval_sec=0.01,
        progress_logger=messages.append,
        progress_label="fixed-unit",
    )

    assert result.num_selected_links == 1
    assert any(
        "fixed-unit:" in message and "entered maximum-flow" in message
        for message in messages
    )
    assert any(
        "fixed-unit:" in message and "entered min-cost-flow" in message
        for message in messages
    )


def test_selected_paths_allow_one_in_and_one_out_at_an_internal_node() -> None:
    nodes = [
        _node(0, start_frame=0, end_frame=1),
        _node(1, start_frame=5, end_frame=6),
        _node(2, start_frame=10, end_frame=11),
        _node(3, start_frame=15, end_frame=16),
    ]
    edges = [
        _edge(0, 1, 0),
        _edge(1, 2, 0),
        _edge(2, 3, 0),
        _edge(0, 2, 50),
        _edge(1, 3, 50),
    ]

    result = solve_forced_fixed_path_cover(nodes, edges, target_num_paths=1)

    assert result.paths == ((0, 1, 2, 3),)
    assert result.predecessor_by_stable == {0: None, 1: 0, 2: 1, 3: 2}
    assert result.successor_by_stable == {0: 1, 1: 2, 2: 3, 3: None}


def test_infeasible_target_fails_closed_with_minimum_path_count() -> None:
    nodes = _two_layer_nodes()
    edges = [_edge(0, 2, 0), _edge(1, 3, 0)]

    with pytest.raises(ContractError, match=r"minimum feasible paths=2"):
        solve_forced_fixed_path_cover(nodes, edges, target_num_paths=1)


def test_all_singletons_are_a_valid_exact_n_path_cover() -> None:
    nodes = _two_layer_nodes()

    result = solve_forced_fixed_path_cover(nodes, [], target_num_paths=4)

    assert result.selected_edges == ()
    assert result.paths == ((0,), (1,), (2,), (3,))
    assert result.required_links == 0
    assert result.maximum_feasible_links == 0
    assert result.total_appearance_cost_int == 0


def test_node_and_edge_input_order_do_not_change_tied_solution() -> None:
    nodes = _two_layer_nodes()
    edges = [
        _edge(0, 2, 0),
        _edge(0, 3, 0),
        _edge(1, 2, 0),
        _edge(1, 3, 0),
    ]
    baseline = solve_forced_fixed_path_cover(nodes, edges, target_num_paths=2)
    expected = tuple(edge.edge_id for edge in baseline.selected_edges)

    for seed in range(12):
        shuffled_nodes = list(nodes)
        shuffled_edges = list(edges)
        random.Random(seed).shuffle(shuffled_nodes)
        random.Random(seed + 100).shuffle(shuffled_edges)
        observed = solve_forced_fixed_path_cover(
            shuffled_nodes, shuffled_edges, target_num_paths=2
        )
        assert tuple(edge.edge_id for edge in observed.selected_edges) == expected
        assert observed.paths == baseline.paths
        assert dict(observed.solver_cost_by_edge) == dict(baseline.solver_cost_by_edge)


@pytest.mark.parametrize(
    "bad_edge",
    [
        _edge(2, 0, 0),
        _edge(0, 1, 0),
    ],
)
def test_reverse_or_temporally_overlapping_edges_are_rejected(
    bad_edge: ForcedAppearanceEdge,
) -> None:
    with pytest.raises(ContractError, match=r"reverse, overlapping"):
        solve_forced_fixed_path_cover(
            _two_layer_nodes(), [bad_edge], target_num_paths=4
        )


def test_exact_touching_time_or_frame_is_not_strictly_future_only() -> None:
    nodes = [
        _node(0, start_frame=0, end_frame=5, start_time=0.0, end_time=1.0),
        _node(1, start_frame=6, end_frame=10, start_time=1.0, end_time=2.0),
    ]

    with pytest.raises(ContractError, match=r"not strictly future-only"):
        solve_forced_fixed_path_cover(
            nodes, [_edge(0, 1, 0)], target_num_paths=2
        )


def test_duplicate_pair_and_duplicate_edge_id_fail_closed() -> None:
    nodes = _two_layer_nodes()
    with pytest.raises(ContractError, match="directed edge pair"):
        solve_forced_fixed_path_cover(
            nodes,
            [_edge(0, 2, 1, token="a"), _edge(0, 2, 2, token="b")],
            target_num_paths=4,
        )
    with pytest.raises(ContractError, match="edge IDs are duplicated"):
        solve_forced_fixed_path_cover(
            nodes,
            [_edge(0, 2, 1, token="same"), _edge(1, 3, 2, token="same")],
            target_num_paths=4,
        )


def test_bad_cost_unknown_node_and_bad_target_fail_closed() -> None:
    nodes = _two_layer_nodes()
    with pytest.raises(ContractError, match="appearance_cost_int"):
        solve_forced_fixed_path_cover(
            nodes, [_edge(0, 2, -1)], target_num_paths=4
        )
    with pytest.raises(ContractError, match="unknown stable ID"):
        solve_forced_fixed_path_cover(
            nodes, [_edge(0, 99, 0)], target_num_paths=4
        )
    with pytest.raises(ContractError, match="target_num_paths"):
        solve_forced_fixed_path_cover(nodes, [], target_num_paths=0)
    with pytest.raises(ContractError, match="exceeds node count"):
        solve_forced_fixed_path_cover(nodes, [], target_num_paths=5)


def test_malformed_nodes_and_edges_raise_contract_error() -> None:
    nodes = _two_layer_nodes()
    with pytest.raises(ContractError, match="node has the wrong type"):
        solve_forced_fixed_path_cover([object()], [], target_num_paths=1)  # type: ignore[list-item]
    with pytest.raises(ContractError, match="edge has the wrong type"):
        solve_forced_fixed_path_cover(
            nodes,
            [object()],  # type: ignore[list-item]
            target_num_paths=4,
        )
    with pytest.raises(ContractError, match="stable IDs are duplicated"):
        solve_forced_fixed_path_cover(
            [nodes[0], replace(nodes[1], stable_id=0)], [], target_num_paths=2
        )


def test_cost_composition_fails_closed_before_losing_integer_exactness() -> None:
    nodes = _two_layer_nodes()
    edges = [
        _edge(0, 2, 2**53),
        _edge(1, 3, 2**53),
    ]

    with pytest.raises(ContractError, match="exact float64 integer range"):
        solve_forced_fixed_path_cover(nodes, edges, target_num_paths=2)


@pytest.mark.parametrize(
    ("certificate", "message"),
    [
        (((0,), (1,), (2,)), "do not cover every node once"),
        (((0, 2), (1, 3)), "certified path edge is absent from candidates"),
        (
            ((0,), (1,), (2,), (3,)),
            "certified path count differs from interval width",
        ),
    ],
)
def test_fixed_solver_rejects_invalid_path_cover_certificate(
    certificate: tuple[tuple[int, ...], ...],
    message: str,
) -> None:
    nodes = _two_layer_nodes()
    edges = [_edge(0, 2, 0)]

    with pytest.raises(ContractError, match=message):
        solve_forced_fixed_path_cover(
            nodes,
            edges,
            target_num_paths=2,
            certified_path_cover=certificate,
        )


def test_large_candidate_graph_runs_highs_and_reports_phases() -> None:
    """A constructive maximum certificate skips Dinic on a large graph."""

    node_count = 4_776
    target_paths = 62
    required_links = node_count - target_paths
    expected_real_edges = 228_806
    nodes = [
        _node(index, start_frame=2 * index, end_frame=2 * index + 1)
        for index in range(node_count)
    ]
    # Forty-eight strict-future bands give 228,072 unique edges.  A partial
    # 49th band brings the graph to the requested stress-test edge count.
    edges = [
        _edge(source, source + delta, 0)
        for delta in range(1, 49)
        for source in range(node_count - delta)
    ]
    edges.extend(
        _edge(source, source + 49, 0)
        for source in range(expected_real_edges - len(edges))
    )
    assert len(edges) == expected_real_edges
    messages: list[str] = []

    result = solve_forced_fixed_path_cover(
        nodes,
        edges,
        target_num_paths=target_paths,
        certified_path_cover=(tuple(range(node_count)),),
        progress_interval_sec=0.01,
        progress_logger=messages.append,
        progress_label="production-shape",
    )

    assert result.required_links == result.num_selected_links == required_links
    assert result.maximum_feasible_links == node_count - 1 == 4_775
    assert result.num_paths == target_paths
    assert result.total_appearance_cost_int == 0
    assert sorted(stable_id for path in result.paths for stable_id in path) == list(
        range(node_count)
    )
    maximum_phase = next(
        message
        for message in messages
        if "entered certified-cardinality" in message
    )
    flow_phase = next(
        message for message in messages if "entered min-cost-flow" in message
    )
    assert "production-shape:" in maximum_phase
    assert "edge_count=228,806" in maximum_phase
    assert "maximum=4,775" in maximum_phase
    assert "certificate_paths=1" in maximum_phase
    assert "production-shape:" in flow_phase
    assert "flow_variables=238,358" in flow_phase
    assert "constraint_nonzeros=471,940" in flow_phase
    assert any(
        "phase=min-cost-flow" in message
        and "elapsed=" in message
        and "cpu=" in message
        and "rss=" in message
        for message in messages
    )
    assert any("completed; selected=4,714" in message for message in messages)
    assert all("maximum-flow" not in message for message in messages)


def _brute_force_fixed_matching_cost(
    node_count: int,
    edges: list[ForcedAppearanceEdge],
    required_links: int,
) -> int:
    """Independent exact-cardinality reference for a small bipartite graph."""

    outgoing: dict[int, tuple[tuple[int, int], ...]] = {}
    for source in range(node_count):
        outgoing[source] = tuple(
            sorted(
                (
                    (edge.target_stable_id, edge.appearance_cost_int)
                    for edge in edges
                    if edge.source_stable_id == source
                ),
            )
        )

    @lru_cache(maxsize=None)
    def search(source: int, used_targets: int, remaining: int) -> int | None:
        if remaining == 0:
            return 0
        if source == node_count or node_count - source < remaining:
            return None
        best = search(source + 1, used_targets, remaining)
        for target, cost in outgoing[source]:
            target_bit = 1 << target
            if used_targets & target_bit:
                continue
            suffix = search(source + 1, used_targets | target_bit, remaining - 1)
            if suffix is not None and (best is None or cost + suffix < best):
                best = cost + suffix
        return best

    optimum = search(0, 0, required_links)
    if optimum is None:
        raise ValueError("reference exact-cardinality matching is infeasible")
    return optimum


def test_min_cost_flow_matches_brute_force_business_objective() -> None:
    """The network solve minimizes appearance at every feasible path count."""

    nodes = [
        _node(index, start_frame=4 * index, end_frame=4 * index + 1)
        for index in range(8)
    ]
    rng = random.Random(7291)
    edges = [
        _edge(source, target, rng.randrange(0, 30))
        for source in range(8)
        for target in range(source + 1, 8)
        if rng.random() < 0.72
    ]
    # The adjacent chain keeps every requested K feasible.
    existing = {(edge.source_stable_id, edge.target_stable_id) for edge in edges}
    for source in range(7):
        if (source, source + 1) not in existing:
            edges.append(_edge(source, source + 1, rng.randrange(0, 30)))

    for target_paths in range(1, len(nodes) + 1):
        result = solve_forced_fixed_path_cover(
            nodes, edges, target_num_paths=target_paths
        )
        reference_cost = _brute_force_fixed_matching_cost(
            len(nodes), edges, len(nodes) - target_paths
        )
        assert result.required_links == len(nodes) - target_paths
        assert result.total_appearance_cost_int == reference_cost
