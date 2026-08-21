"""Deterministic operator-approved path cover over S05 proposals.

This module deliberately does not reinterpret provisional model evidence as
certified evidence.  The caller supplies the complete set of proposal edges
covered by the current blanket operator approval.  Every structurally legal
edge remains eligible; model evidence is used only as the secondary objective
after maximum selected-link cardinality.
"""

from __future__ import annotations

import heapq
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import (
    maximum_bipartite_matching,
    min_weight_full_bipartite_matching,
)

from cowtrack.config import ContractError


_MIN_LONG_GAP_SEC = 5.0
_GAP_ABS_TOL = 2e-6
_MAX_EXACT_INTEGER = 2**53


@dataclass(frozen=True, slots=True)
class GlobalStableNode:
    """One finalized S04 stable path presented to the global path cover."""

    stable_id: int
    start_clip_id: str
    end_clip_id: str
    start_global_frame: int
    end_global_frame: int
    start_time_sec: float
    end_time_sec: float
    num_microtracklets: int
    num_detections: int


@dataclass(frozen=True, slots=True)
class GlobalProposalEdge:
    """One operator-authorized provisional S05 proposal edge.

    The fields retain the original uncertified model evidence.  Selection by
    this module means only that the blanket operator policy chose the edge for
    the structural path cover.
    """

    proposal_id: str
    candidate_id: str
    source_stable_id: int
    target_stable_id: int
    probability: float
    candidate_margin: float | None
    rank_out: int
    rank_in: int
    high_overlap: bool
    gallery_score_mutual: float
    temporal_gap_sec: float


@dataclass(frozen=True, slots=True)
class PathCoverResult:
    """Immutable total path cover and the exact secondary solver costs."""

    selected_edges: tuple[GlobalProposalEdge, ...]
    paths: tuple[tuple[int, ...], ...]
    predecessor_by_stable: Mapping[int, int | None]
    successor_by_stable: Mapping[int, int | None]
    solver_cost_by_candidate: Mapping[str, int]
    max_cardinality: int

    @property
    def num_nodes(self) -> int:
        return len(self.predecessor_by_stable)

    @property
    def num_selected_links(self) -> int:
        return len(self.selected_edges)

    @property
    def num_paths(self) -> int:
        return len(self.paths)


def _strict_int(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ContractError(f"S05 path-cover {label} must be an integer")
    result = int(value)
    if result < minimum:
        raise ContractError(
            f"S05 path-cover {label} must be >= {minimum}"
        )
    return result


def _finite(value: Any, label: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ContractError(f"S05 path-cover {label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ContractError(f"S05 path-cover {label} must be finite")
    return result


def _validated_nodes(
    nodes: Sequence[GlobalStableNode],
) -> tuple[tuple[GlobalStableNode, ...], dict[int, GlobalStableNode]]:
    if isinstance(nodes, (str, bytes)):
        raise ContractError("S05 path-cover nodes must be a sequence")
    try:
        supplied = tuple(nodes)
    except TypeError as exc:
        raise ContractError("S05 path-cover nodes must be a sequence") from exc
    if not supplied:
        raise ContractError("S05 path-cover requires at least one stable node")
    if any(not isinstance(node, GlobalStableNode) for node in supplied):
        raise ContractError("S05 path-cover node has the wrong type")
    ordered = tuple(
        sorted(
            supplied,
            key=lambda item: _strict_int(item.stable_id, "stable_id"),
        )
    )
    by_id: dict[int, GlobalStableNode] = {}
    for node in ordered:
        stable_id = _strict_int(node.stable_id, "stable_id")
        if stable_id in by_id:
            raise ContractError("S05 path-cover stable IDs are duplicated")
        if (
            not isinstance(node.start_clip_id, str)
            or not node.start_clip_id
            or not isinstance(node.end_clip_id, str)
            or not node.end_clip_id
        ):
            raise ContractError("S05 path-cover node clip IDs are invalid")
        start_frame = _strict_int(node.start_global_frame, "start_global_frame")
        end_frame = _strict_int(node.end_global_frame, "end_global_frame")
        start_time = _finite(node.start_time_sec, "start_time_sec")
        end_time = _finite(node.end_time_sec, "end_time_sec")
        if start_frame > end_frame or start_time > end_time:
            raise ContractError("S05 path-cover node temporal bounds are invalid")
        _strict_int(node.num_microtracklets, "num_microtracklets", minimum=1)
        _strict_int(node.num_detections, "num_detections", minimum=1)
        by_id[stable_id] = node
    return ordered, by_id


def _validated_edges(
    edges: Sequence[GlobalProposalEdge],
    nodes_by_id: Mapping[int, GlobalStableNode],
) -> tuple[GlobalProposalEdge, ...]:
    if isinstance(edges, (str, bytes)):
        raise ContractError("S05 path-cover edges must be a sequence")
    try:
        supplied = tuple(edges)
    except TypeError as exc:
        raise ContractError("S05 path-cover edges must be a sequence") from exc
    if any(not isinstance(edge, GlobalProposalEdge) for edge in supplied):
        raise ContractError("S05 path-cover edge has the wrong type")
    ordered = tuple(
        sorted(
            supplied,
            key=lambda item: (
                str(item.proposal_id),
                str(item.candidate_id),
                _strict_int(item.source_stable_id, "source_stable_id"),
                _strict_int(item.target_stable_id, "target_stable_id"),
            ),
        )
    )
    proposal_ids: set[str] = set()
    candidate_ids: set[str] = set()
    pairs: set[tuple[int, int]] = set()
    for edge in ordered:
        if (
            not isinstance(edge.proposal_id, str)
            or not edge.proposal_id
            or not isinstance(edge.candidate_id, str)
            or not edge.candidate_id
        ):
            raise ContractError("S05 path-cover proposal/candidate ID is invalid")
        if edge.proposal_id in proposal_ids or edge.candidate_id in candidate_ids:
            raise ContractError("S05 path-cover proposal/candidate IDs are duplicated")
        proposal_ids.add(edge.proposal_id)
        candidate_ids.add(edge.candidate_id)
        source_id = _strict_int(edge.source_stable_id, "source_stable_id")
        target_id = _strict_int(edge.target_stable_id, "target_stable_id")
        pair = (source_id, target_id)
        if source_id == target_id or pair in pairs:
            raise ContractError("S05 path-cover directed proposal pair is invalid")
        pairs.add(pair)
        if source_id not in nodes_by_id or target_id not in nodes_by_id:
            raise ContractError("S05 path-cover edge references an unknown stable ID")
        probability = _finite(edge.probability, "probability")
        if not 0.0 <= probability <= 1.0:
            raise ContractError("S05 path-cover probability is outside [0,1]")
        if edge.candidate_margin is not None:
            _finite(edge.candidate_margin, "candidate_margin")
        _strict_int(edge.rank_out, "rank_out", minimum=1)
        _strict_int(edge.rank_in, "rank_in", minimum=1)
        if type(edge.high_overlap) is not bool:
            raise ContractError("S05 path-cover high_overlap must be boolean")
        mutual = _finite(edge.gallery_score_mutual, "gallery_score_mutual")
        if not -1.0 <= mutual <= 1.0:
            raise ContractError(
                "S05 path-cover gallery_score_mutual is outside [-1,1]"
            )
        gap = _finite(edge.temporal_gap_sec, "temporal_gap_sec")
        if gap <= _MIN_LONG_GAP_SEC:
            raise ContractError("S05 path-cover proposal gap must be >5s")

    _assert_acyclic(nodes_by_id, ordered, label="proposal graph")
    for edge in ordered:
        source = nodes_by_id[int(edge.source_stable_id)]
        target = nodes_by_id[int(edge.target_stable_id)]
        expected_gap = float(target.start_time_sec) - float(source.end_time_sec)
        if (
            expected_gap <= _MIN_LONG_GAP_SEC
            or float(source.end_time_sec) >= float(target.start_time_sec)
            or int(source.end_global_frame) >= int(target.start_global_frame)
            or not math.isclose(
                float(edge.temporal_gap_sec),
                expected_gap,
                rel_tol=0.0,
                abs_tol=_GAP_ABS_TOL,
            )
        ):
            raise ContractError(
                "S05 path-cover proposal is reverse, overlapping, or has a false gap"
            )
    return ordered


def _assert_acyclic(
    nodes_by_id: Mapping[int, GlobalStableNode],
    edges: Sequence[GlobalProposalEdge],
    *,
    label: str,
) -> None:
    outgoing: dict[int, list[int]] = {stable_id: [] for stable_id in nodes_by_id}
    indegree = {stable_id: 0 for stable_id in nodes_by_id}
    for edge in edges:
        source_id = int(edge.source_stable_id)
        target_id = int(edge.target_stable_id)
        if source_id not in outgoing or target_id not in indegree:
            continue
        outgoing[source_id].append(target_id)
        indegree[target_id] += 1
    queue = [stable_id for stable_id, degree in indegree.items() if degree == 0]
    heapq.heapify(queue)
    visited = 0
    while queue:
        source_id = heapq.heappop(queue)
        visited += 1
        for target_id in sorted(outgoing[source_id]):
            indegree[target_id] -= 1
            if indegree[target_id] == 0:
                heapq.heappush(queue, target_id)
    if visited != len(nodes_by_id):
        raise ContractError(f"S05 path-cover {label} contains a cycle")


def _evidence_key(edge: GlobalProposalEdge) -> tuple[Any, ...]:
    margin_missing = edge.candidate_margin is None
    margin = 0.0 if margin_missing else float(edge.candidate_margin)
    return (
        bool(edge.high_overlap),
        max(int(edge.rank_out), int(edge.rank_in)),
        int(edge.rank_out) + int(edge.rank_in),
        -float(edge.probability),
        margin_missing,
        -margin,
        -float(edge.gallery_score_mutual),
        str(edge.proposal_id),
        str(edge.candidate_id),
        int(edge.source_stable_id),
        int(edge.target_stable_id),
    )


def _solver_costs(edges: Sequence[GlobalProposalEdge]) -> dict[str, int]:
    by_evidence = sorted(edges, key=_evidence_key)
    return {
        edge.candidate_id: ordinal
        for ordinal, edge in enumerate(by_evidence, start=1)
    }


def _solve_assignment(
    nodes: Sequence[GlobalStableNode],
    edges: Sequence[GlobalProposalEdge],
    costs: Mapping[str, int],
) -> tuple[tuple[GlobalProposalEdge, ...], int]:
    node_index = {node.stable_id: index for index, node in enumerate(nodes)}
    edge_by_index_pair = {
        (node_index[edge.source_stable_id], node_index[edge.target_stable_id]): edge
        for edge in edges
    }
    node_count = len(nodes)
    edge_count = len(edges)
    max_real_cost = max(costs.values(), default=1)
    dummy_cost = node_count * max_real_cost + 1
    max_total = node_count * dummy_cost
    if max_total >= _MAX_EXACT_INTEGER:
        raise ContractError(
            "S05 path-cover problem is too large for exact integer assignment"
        )

    real_rows = [pair[0] for pair in edge_by_index_pair]
    real_cols = [pair[1] for pair in edge_by_index_pair]
    real_data = [
        costs[edge_by_index_pair[pair].candidate_id]
        for pair in edge_by_index_pair
    ]
    dummy_rows = list(range(node_count))
    dummy_cols = [node_count + index for index in range(node_count)]
    matrix = coo_matrix(
        (
            np.asarray([*real_data, *([dummy_cost] * node_count)], dtype=np.int64),
            (
                np.asarray([*real_rows, *dummy_rows], dtype=np.int64),
                np.asarray([*real_cols, *dummy_cols], dtype=np.int64),
            ),
        ),
        shape=(node_count, 2 * node_count),
        dtype=np.int64,
    ).tocsr()
    matrix.sort_indices()
    try:
        row_ind, col_ind = min_weight_full_bipartite_matching(matrix)
    except ValueError as exc:
        raise ContractError(f"S05 path-cover assignment failed: {exc}") from exc
    if len(row_ind) != node_count or not np.array_equal(
        row_ind, np.arange(node_count, dtype=row_ind.dtype)
    ):
        raise ContractError("S05 path-cover assignment did not cover every source port")
    selected = tuple(
        edge_by_index_pair[(int(row), int(column))]
        for row, column in zip(row_ind, col_ind, strict=True)
        if int(column) < node_count
    )

    if edge_count:
        real_graph = csr_matrix(
            (
                np.ones(edge_count, dtype=np.int8),
                (
                    np.asarray(real_rows, dtype=np.int64),
                    np.asarray(real_cols, dtype=np.int64),
                ),
            ),
            shape=(node_count, node_count),
        )
        maximum = maximum_bipartite_matching(
            real_graph, perm_type="column"
        )
        max_cardinality = int(np.count_nonzero(maximum >= 0))
    else:
        max_cardinality = 0
    if len(selected) != max_cardinality:
        raise ContractError(
            "S05 path-cover assignment did not achieve maximum link cardinality"
        )
    return selected, max_cardinality


def _build_total_paths(
    nodes: Sequence[GlobalStableNode],
    nodes_by_id: Mapping[int, GlobalStableNode],
    selected: Sequence[GlobalProposalEdge],
) -> tuple[
    tuple[tuple[int, ...], ...],
    Mapping[int, int | None],
    Mapping[int, int | None],
]:
    predecessor: dict[int, int | None] = {node.stable_id: None for node in nodes}
    successor: dict[int, int | None] = {node.stable_id: None for node in nodes}
    for edge in selected:
        source_id = int(edge.source_stable_id)
        target_id = int(edge.target_stable_id)
        if successor[source_id] is not None or predecessor[target_id] is not None:
            raise ContractError("S05 selected links violate one-in/one-out capacity")
        successor[source_id] = target_id
        predecessor[target_id] = source_id

    _assert_acyclic(nodes_by_id, selected, label="selected graph")
    starts = sorted(
        (stable_id for stable_id, value in predecessor.items() if value is None),
        key=lambda stable_id: (
            float(nodes_by_id[stable_id].start_time_sec),
            int(nodes_by_id[stable_id].start_global_frame),
            stable_id,
        ),
    )
    paths: list[tuple[int, ...]] = []
    visited: set[int] = set()
    for start in starts:
        path: list[int] = []
        current: int | None = start
        while current is not None:
            if current in visited:
                raise ContractError("S05 selected path cover revisits a stable node")
            visited.add(current)
            path.append(current)
            following = successor[current]
            if following is not None:
                source = nodes_by_id[current]
                target = nodes_by_id[following]
                if (
                    float(source.end_time_sec) >= float(target.start_time_sec)
                    or int(source.end_global_frame) >= int(target.start_global_frame)
                ):
                    raise ContractError("S05 selected path is not strictly future-only")
            current = following
        paths.append(tuple(path))
    if visited != set(nodes_by_id):
        raise ContractError("S05 path cover did not include every stable node exactly once")
    if len(paths) != len(nodes) - len(selected):
        raise ContractError("S05 path count differs from P=N-L")
    return (
        tuple(paths),
        MappingProxyType(dict(predecessor)),
        MappingProxyType(dict(successor)),
    )


def solve_operator_approved_path_cover(
    nodes: Sequence[GlobalStableNode],
    edges: Sequence[GlobalProposalEdge],
) -> PathCoverResult:
    """Return the exact maximum-cardinality operator-approved path cover.

    The primary objective maximizes the number of selected proposal links.
    Among maximum-cardinality matchings, lower evidence rank is preferred:
    non-high-overlap, better mutual ranks, higher probability, better margin,
    and higher mutual gallery score.  No evidence value excludes an otherwise
    structurally legal proposal.
    """

    ordered_nodes, nodes_by_id = _validated_nodes(nodes)
    ordered_edges = _validated_edges(edges, nodes_by_id)
    costs = _solver_costs(ordered_edges)
    selected, max_cardinality = _solve_assignment(
        ordered_nodes, ordered_edges, costs
    )
    canonical_selected = tuple(
        sorted(
            selected,
            key=lambda edge: (
                int(edge.source_stable_id),
                int(edge.target_stable_id),
                str(edge.proposal_id),
                str(edge.candidate_id),
            ),
        )
    )
    paths, predecessor, successor = _build_total_paths(
        ordered_nodes, nodes_by_id, canonical_selected
    )
    return PathCoverResult(
        selected_edges=canonical_selected,
        paths=paths,
        predecessor_by_stable=predecessor,
        successor_by_stable=successor,
        solver_cost_by_candidate=MappingProxyType(dict(sorted(costs.items()))),
        max_cardinality=max_cardinality,
    )


def global_stable_node_from_row(row: Mapping[str, Any]) -> GlobalStableNode:
    """Build one solver node from an exact finalized-S04 stable row."""
    if not isinstance(row, Mapping):
        raise ContractError("S05 path-cover stable row must be a mapping")
    try:
        return GlobalStableNode(
            stable_id=row["stable_id"],
            start_clip_id=row["start_clip_id"],
            end_clip_id=row["end_clip_id"],
            start_global_frame=row["start_global_frame"],
            end_global_frame=row["end_global_frame"],
            start_time_sec=row["start_time_sec"],
            end_time_sec=row["end_time_sec"],
            num_microtracklets=row["num_microtracklets"],
            num_detections=row["num_detections"],
        )
    except KeyError as exc:
        raise ContractError(f"S05 path-cover stable row lacks {exc.args[0]}") from exc


def global_proposal_edge_from_row(row: Mapping[str, Any]) -> GlobalProposalEdge:
    """Build one edge while preserving its provisional model semantics."""
    if not isinstance(row, Mapping):
        raise ContractError("S05 path-cover proposal row must be a mapping")
    if row.get("decision") != "provisional" or row.get("review_status") != "pending":
        raise ContractError(
            "S05 operator path-cover input must be a pending provisional proposal"
        )
    if any(
        row.get(name) is not False
        for name in ("selected_by_solver", "confirmed", "merge_applied")
    ):
        raise ContractError(
            "S05 operator path-cover proposal contains identity-changing state"
        )
    if row.get("appearance_present") is not True or row.get(
        "temporally_nonoverlapping"
    ) is not True:
        raise ContractError(
            "S05 operator path-cover proposal lacks appearance or non-overlap evidence"
        )
    try:
        return GlobalProposalEdge(
            proposal_id=row["proposal_id"],
            candidate_id=row["candidate_id"],
            source_stable_id=row["source_stable_id"],
            target_stable_id=row["target_stable_id"],
            probability=row["model_probability"],
            candidate_margin=row["candidate_margin"],
            rank_out=row["appearance_rank_out"],
            rank_in=row["appearance_rank_in"],
            high_overlap=row["high_overlap"],
            gallery_score_mutual=row["gallery_score_mutual"],
            temporal_gap_sec=row["temporal_gap_sec"],
        )
    except KeyError as exc:
        raise ContractError(f"S05 path-cover proposal row lacks {exc.args[0]}") from exc


__all__ = [
    "GlobalProposalEdge",
    "GlobalStableNode",
    "PathCoverResult",
    "global_proposal_edge_from_row",
    "global_stable_node_from_row",
    "solve_operator_approved_path_cover",
]
