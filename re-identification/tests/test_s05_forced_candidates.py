from __future__ import annotations

import random

import numpy as np
import pytest

import cowtrack.linking.forced_candidates as forced_candidates
from cowtrack.config import ContractError
from cowtrack.linking.forced_candidates import (
    GRADE_A_CLEAN,
    GRADE_B_EXISTING_DEGRADED,
    GRADE_C_REENCODED_DEGRADED,
    build_forced_candidate_graph,
)
from cowtrack.linking.forced_path_cover import solve_forced_fixed_path_cover
from cowtrack.linking.path_cover import GlobalStableNode


def _node(
    stable_id: int,
    start_frame: int,
    end_frame: int,
    *,
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
        num_detections=1,
    )


def _records_by_pair(graph: object) -> dict[tuple[int, int], object]:
    return {
        (record.edge.source_stable_id, record.edge.target_stable_id): record
        for record in graph.candidates  # type: ignore[attr-defined]
    }


def test_minimum_width_backbone_guarantees_every_k_at_or_above_width() -> None:
    nodes = [
        _node(0, 0, 4),
        _node(1, 0, 4),
        _node(2, 5, 9),
        _node(3, 5, 9),
        _node(4, 10, 14),
    ]
    embeddings = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
        ],
        dtype=np.float32,
    )
    grades = [GRADE_A_CLEAN] * len(nodes)

    graph = build_forced_candidate_graph(
        nodes, embeddings, grades, set(), top_k=1
    )

    assert graph.max_concurrent == graph.chain_count == 2
    assert graph.backbone_chains == ((0, 2, 4), (1, 3))
    backbone_pairs = {
        (record.edge.source_stable_id, record.edge.target_stable_id)
        for record in graph.candidates
        if record.selected_by_backbone
    }
    assert backbone_pairs == {(0, 2), (1, 3), (2, 4)}
    assert len(backbone_pairs) == len(nodes) - graph.chain_count

    for target_k in range(graph.chain_count, len(nodes) + 1):
        cover = solve_forced_fixed_path_cover(
            nodes, graph.edges, target_num_paths=target_k
        )
        assert cover.num_paths == target_k
        assert cover.num_selected_links == len(nodes) - target_k


def test_source_topk_cosine_tie_breaks_by_target_stable_id() -> None:
    nodes = [_node(10, 0, 1), _node(30, 4, 5), _node(20, 4, 5)]
    embeddings = np.asarray([[1.0, 0.0]] * 3, dtype=np.float32)
    graph = build_forced_candidate_graph(
        nodes,
        embeddings,
        [GRADE_A_CLEAN] * 3,
        set(),
        top_k=1,
    )
    records = _records_by_pair(graph)

    assert records[(10, 20)].selected_by_source_topk is True
    if (10, 30) in records:
        assert records[(10, 30)].selected_by_source_topk is False


def test_target_topk_cosine_tie_breaks_by_source_stable_id() -> None:
    nodes = [_node(20, 0, 1), _node(10, 0, 1), _node(30, 4, 5)]
    embeddings = np.asarray([[1.0, 0.0]] * 3, dtype=np.float32)
    graph = build_forced_candidate_graph(
        nodes,
        embeddings,
        [GRADE_A_CLEAN] * 3,
        set(),
        top_k=1,
    )
    records = _records_by_pair(graph)

    assert records[(10, 30)].selected_by_target_topk is True
    if (20, 30) in records:
        assert records[(20, 30)].selected_by_target_topk is False


def test_prior_adds_legal_non_topk_edge_and_applies_grade_cost_and_bonus() -> None:
    nodes = [_node(0, 0, 1), _node(1, 4, 5), _node(2, 8, 9)]
    embeddings = np.asarray(
        [[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]], dtype=np.float32
    )
    grades = [
        GRADE_B_EXISTING_DEGRADED,
        GRADE_A_CLEAN,
        GRADE_C_REENCODED_DEGRADED,
    ]
    graph = build_forced_candidate_graph(
        nodes, embeddings, grades, {(0, 2)}, top_k=1
    )
    record = _records_by_pair(graph)[(0, 2)]

    assert record.selected_by_prior is True
    assert record.selected_by_source_topk is False
    assert record.selected_by_target_topk is False
    assert record.cosine_similarity == pytest.approx(0.0)
    assert record.source_grade == GRADE_B_EXISTING_DEGRADED
    assert record.target_grade == GRADE_C_REENCODED_DEGRADED
    assert record.base_cost_int == 400 + 5 + 15 - 3
    assert record.edge.appearance_cost_int == record.base_cost_int


def test_prior_bonus_cost_is_clamped_at_zero() -> None:
    nodes = [_node(0, 0, 1), _node(1, 4, 5)]
    embeddings = np.asarray([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    graph = build_forced_candidate_graph(
        nodes,
        embeddings,
        [GRADE_A_CLEAN, GRADE_A_CLEAN],
        {(0, 1)},
        top_k=1,
    )

    assert _records_by_pair(graph)[(0, 1)].base_cost_int == 0


def test_custom_cost_parameters_are_validated_and_applied() -> None:
    nodes = [_node(0, 0, 1), _node(1, 4, 5)]
    root_half = np.float32(1.0 / np.sqrt(2.0))
    embeddings = np.asarray(
        [[1.0, 0.0], [root_half, root_half]], dtype=np.float32
    )
    graph = build_forced_candidate_graph(
        nodes,
        embeddings,
        [GRADE_B_EXISTING_DEGRADED, GRADE_C_REENCODED_DEGRADED],
        set(),
        top_k=1,
        cost_scale=100,
        grade_penalties={
            GRADE_A_CLEAN: 1,
            GRADE_B_EXISTING_DEGRADED: 2,
            GRADE_C_REENCODED_DEGRADED: 3,
        },
        prior_bonus=7,
    )
    record = _records_by_pair(graph)[(0, 1)]

    expected = round((1.0 - record.cosine_similarity) * 100) + 2 + 3
    assert record.base_cost_int == expected
    assert graph.cost_scale == 100
    assert graph.prior_bonus == 7


def test_aligned_input_shuffle_does_not_change_graph_or_backbone() -> None:
    nodes = [
        _node(0, 0, 2),
        _node(1, 0, 2),
        _node(2, 5, 7),
        _node(3, 9, 11),
    ]
    embeddings = np.asarray(
        [[1.0, 0.0], [0.0, 1.0], [0.8, 0.6], [0.6, 0.8]],
        dtype=np.float32,
    )
    grades = [
        GRADE_A_CLEAN,
        GRADE_B_EXISTING_DEGRADED,
        GRADE_C_REENCODED_DEGRADED,
        GRADE_A_CLEAN,
    ]
    baseline = build_forced_candidate_graph(
        nodes, embeddings, grades, {(0, 3)}, top_k=2
    )

    for seed in range(10):
        order = list(range(len(nodes)))
        random.Random(seed).shuffle(order)
        observed = build_forced_candidate_graph(
            [nodes[index] for index in order],
            embeddings[order],
            [grades[index] for index in order],
            {(0, 3)},
            top_k=2,
        )
        assert observed == baseline


def test_blockwise_bidirectional_topk_matches_dense_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Multiple cosine slabs retain exhaustive dense top-k semantics."""

    rng = np.random.default_rng(2187)
    node_count = 43
    nodes = [
        _node(index, 3 * index, 3 * index + 1)
        for index in range(node_count)
    ]
    embeddings = rng.normal(size=(node_count, 12)).astype(np.float32)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    top_k = 5
    monkeypatch.setattr(forced_candidates, "_COSINE_BLOCK_ROWS", 7)
    graph = build_forced_candidate_graph(
        nodes,
        embeddings,
        [GRADE_A_CLEAN] * node_count,
        set(),
        top_k=top_k,
    )

    similarities = np.clip(
        np.asarray(embeddings @ embeddings.T, dtype=np.float32), -1.0, 1.0
    )
    stable_ids = np.arange(node_count, dtype=np.int64)
    compatible = stable_ids[:, None] < stable_ids[None, :]
    expected_source: set[tuple[int, int]] = set()
    expected_target: set[tuple[int, int]] = set()
    for source in range(node_count):
        targets = np.flatnonzero(compatible[source])
        order = np.lexsort((stable_ids[targets], -similarities[source, targets]))
        expected_source.update(
            (source, int(target)) for target in targets[order[:top_k]]
        )
    for target in range(node_count):
        sources = np.flatnonzero(compatible[:, target])
        order = np.lexsort((stable_ids[sources], -similarities[sources, target]))
        expected_target.update(
            (int(source), target) for source in sources[order[:top_k]]
        )

    records = _records_by_pair(graph)
    assert {
        pair for pair, record in records.items() if record.selected_by_source_topk
    } == expected_source
    assert {
        pair for pair, record in records.items() if record.selected_by_target_topk
    } == expected_target
    for pair, record in records.items():
        assert record.cosine_similarity == pytest.approx(
            float(similarities[pair]), abs=2e-6
        )


@pytest.mark.parametrize(
    "embeddings",
    [
        np.asarray([1.0, 0.0], dtype=np.float32),
        np.asarray([[2.0, 0.0]], dtype=np.float32),
        np.asarray([[np.nan, 0.0]], dtype=np.float32),
        np.asarray([[1, 0]], dtype=np.int64),
    ],
)
def test_bad_embedding_contract_fails_closed(embeddings: np.ndarray) -> None:
    with pytest.raises(ContractError, match="center embeddings"):
        build_forced_candidate_graph(
            [_node(0, 0, 1)],
            embeddings,
            [GRADE_A_CLEAN],
            set(),
            top_k=1,
        )


def test_bad_grade_and_grade_penalty_contract_fail_closed() -> None:
    nodes = [_node(0, 0, 1)]
    embeddings = np.asarray([[1.0, 0.0]], dtype=np.float32)
    with pytest.raises(ContractError, match="grades must"):
        build_forced_candidate_graph(
            nodes, embeddings, ["A"], set(), top_k=1
        )
    with pytest.raises(ContractError, match="invalid keys"):
        build_forced_candidate_graph(
            nodes,
            embeddings,
            [GRADE_A_CLEAN],
            set(),
            top_k=1,
            grade_penalties={GRADE_A_CLEAN: 0},
        )


def test_prior_pairs_must_be_known_strictly_future_set_pairs() -> None:
    nodes = [_node(0, 0, 4), _node(1, 4, 8), _node(2, 10, 12)]
    embeddings = np.asarray([[1.0, 0.0]] * 3, dtype=np.float32)
    grades = [GRADE_A_CLEAN] * 3
    with pytest.raises(ContractError, match="must be a set"):
        build_forced_candidate_graph(
            nodes, embeddings, grades, [(0, 2)], top_k=1  # type: ignore[arg-type]
        )
    with pytest.raises(ContractError, match="unknown stable ID"):
        build_forced_candidate_graph(
            nodes, embeddings, grades, {(0, 99)}, top_k=1
        )
    with pytest.raises(ContractError, match="reverse or temporally overlapping"):
        build_forced_candidate_graph(
            nodes, embeddings, grades, {(1, 0)}, top_k=1
        )
    with pytest.raises(ContractError, match="reverse or temporally overlapping"):
        build_forced_candidate_graph(
            nodes, embeddings, grades, {(0, 1)}, top_k=1
        )


def test_touching_closed_intervals_are_concurrent_and_never_linked() -> None:
    nodes = [_node(0, 0, 5), _node(1, 5, 10)]
    embeddings = np.asarray([[1.0, 0.0]] * 2, dtype=np.float32)
    graph = build_forced_candidate_graph(
        nodes,
        embeddings,
        [GRADE_A_CLEAN] * 2,
        set(),
        top_k=1,
    )

    assert graph.max_concurrent == graph.chain_count == 2
    assert (0, 1) not in _records_by_pair(graph)


def test_global_frame_time_order_must_be_consistent() -> None:
    nodes = [
        _node(0, 0, 1, start_time=0.0, end_time=2.0),
        _node(1, 3, 4, start_time=1.0, end_time=4.0),
    ]
    embeddings = np.asarray([[1.0, 0.0]] * 2, dtype=np.float32)

    with pytest.raises(ContractError, match="frame/time ordering"):
        build_forced_candidate_graph(
            nodes,
            embeddings,
            [GRADE_A_CLEAN] * 2,
            set(),
            top_k=1,
        )


def test_scalar_parameters_and_aligned_lengths_are_strict() -> None:
    node = _node(0, 0, 1)
    embedding = np.asarray([[1.0, 0.0]], dtype=np.float32)
    with pytest.raises(ContractError, match="top_k"):
        build_forced_candidate_graph(
            [node], embedding, [GRADE_A_CLEAN], set(), top_k=0
        )
    with pytest.raises(ContractError, match="cost_scale"):
        build_forced_candidate_graph(
            [node], embedding, [GRADE_A_CLEAN], set(), top_k=1, cost_scale=True
        )
    with pytest.raises(ContractError, match="grade count"):
        build_forced_candidate_graph(
            [node], embedding, [], set(), top_k=1
        )
