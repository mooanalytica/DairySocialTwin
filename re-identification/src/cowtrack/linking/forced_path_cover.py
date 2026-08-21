"""Exact fixed-cardinality path cover for operator-forced appearance linking.

The ordinary S05 path cover maximizes link cardinality over an already approved
proposal graph.  Forced appearance consolidation has a different contract: the
caller supplies a requested number of output paths and integer appearance
costs, while this module preserves all temporal hard constraints and minimizes
the total appearance cost among covers with exactly that many paths.

The exact fixed-cardinality problem is solved as a fixed-flow min-cost network:
source to left nodes, real candidate arcs to right nodes, then right nodes to
sink.  Unit capacities and a fixed integer flow of ``N-K`` make the
network-incidence LP integral, so no dummy vertices or dummy edges are required.
Production callers provide the candidate builder's minimum-width path cover;
validating that constructive certificate gives the exact maximum cardinality
without another graph search.  Generic callers without a certificate use Dinic
maximum flow.  Native work runs in a spawned child process, whose parent remains
responsive, emits truthful heartbeats, and can terminate only that owned child
on ``KeyboardInterrupt``.  Canonical node and edge ordering plus pinned
SciPy/HiGHS make equal-cost solutions input-order invariant.
"""

from __future__ import annotations

import heapq
import math
import multiprocessing as multiprocessing_module
import os
import time
import traceback
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, csr_array
from scipy.sparse.csgraph import maximum_flow

from cowtrack.config import ContractError
from cowtrack.linking.path_cover import GlobalStableNode


_MAX_EXACT_INTEGER = 2**53
_FLOW_INTEGRAL_ATOL = 1e-7
_CHILD_JOIN_GRACE_SEC = 5.0


@dataclass(frozen=True, slots=True)
class ForcedAppearanceEdge:
    """One temporally legal appearance edge with an exact integer base cost."""

    edge_id: str
    source_stable_id: int
    target_stable_id: int
    appearance_cost_int: int


@dataclass(frozen=True, slots=True)
class ForcedPathCoverResult:
    """A total fixed-K path cover and its deterministic solver provenance."""

    selected_edges: tuple[ForcedAppearanceEdge, ...]
    paths: tuple[tuple[int, ...], ...]
    predecessor_by_stable: Mapping[int, int | None]
    successor_by_stable: Mapping[int, int | None]
    solver_cost_by_edge: Mapping[str, int]
    target_num_paths: int
    required_links: int
    maximum_feasible_links: int
    total_appearance_cost_int: int

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
        raise ContractError(f"forced path-cover {label} must be an integer")
    result = int(value)
    if result < minimum:
        raise ContractError(
            f"forced path-cover {label} must be >= {minimum}"
        )
    return result


def _finite(value: Any, label: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ContractError(f"forced path-cover {label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ContractError(f"forced path-cover {label} must be finite")
    return result


def _validated_progress(
    progress_interval_sec: Any,
    progress_logger: Callable[[str], None] | None,
    progress_label: Any,
) -> tuple[float, Callable[[str], None] | None, str]:
    interval = _finite(progress_interval_sec, "progress_interval_sec")
    if interval <= 0.0:
        raise ContractError(
            "forced path-cover progress_interval_sec must be positive"
        )
    if progress_logger is not None and not callable(progress_logger):
        raise ContractError("forced path-cover progress_logger must be callable")
    if not isinstance(progress_label, str) or not progress_label.strip():
        raise ContractError("forced path-cover progress_label must be non-empty")
    return interval, progress_logger, progress_label.strip()


def _validated_nodes(
    nodes: Sequence[GlobalStableNode],
) -> tuple[tuple[GlobalStableNode, ...], dict[int, GlobalStableNode]]:
    if isinstance(nodes, (str, bytes)):
        raise ContractError("forced path-cover nodes must be a sequence")
    try:
        supplied = tuple(nodes)
    except TypeError as exc:
        raise ContractError("forced path-cover nodes must be a sequence") from exc
    if not supplied:
        raise ContractError("forced path-cover requires at least one stable node")
    if any(not isinstance(node, GlobalStableNode) for node in supplied):
        raise ContractError("forced path-cover node has the wrong type")
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
            raise ContractError("forced path-cover stable IDs are duplicated")
        if (
            not isinstance(node.start_clip_id, str)
            or not node.start_clip_id
            or not isinstance(node.end_clip_id, str)
            or not node.end_clip_id
        ):
            raise ContractError("forced path-cover node clip IDs are invalid")
        start_frame = _strict_int(node.start_global_frame, "start_global_frame")
        end_frame = _strict_int(node.end_global_frame, "end_global_frame")
        start_time = _finite(node.start_time_sec, "start_time_sec")
        end_time = _finite(node.end_time_sec, "end_time_sec")
        if start_frame > end_frame or start_time > end_time:
            raise ContractError("forced path-cover node temporal bounds are invalid")
        _strict_int(node.num_microtracklets, "num_microtracklets", minimum=1)
        _strict_int(node.num_detections, "num_detections", minimum=1)
        by_id[stable_id] = node
    return ordered, by_id


def _assert_acyclic(
    nodes_by_id: Mapping[int, GlobalStableNode],
    edges: Sequence[ForcedAppearanceEdge],
    *,
    label: str,
) -> None:
    outgoing: dict[int, list[int]] = {stable_id: [] for stable_id in nodes_by_id}
    indegree = {stable_id: 0 for stable_id in nodes_by_id}
    for edge in edges:
        source_id = int(edge.source_stable_id)
        target_id = int(edge.target_stable_id)
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
        raise ContractError(f"forced path-cover {label} contains a cycle")


def _validated_edges(
    edges: Sequence[ForcedAppearanceEdge],
    nodes_by_id: Mapping[int, GlobalStableNode],
) -> tuple[ForcedAppearanceEdge, ...]:
    if isinstance(edges, (str, bytes)):
        raise ContractError("forced path-cover edges must be a sequence")
    try:
        supplied = tuple(edges)
    except TypeError as exc:
        raise ContractError("forced path-cover edges must be a sequence") from exc
    if any(not isinstance(edge, ForcedAppearanceEdge) for edge in supplied):
        raise ContractError("forced path-cover edge has the wrong type")
    ordered = tuple(
        sorted(
            supplied,
            key=lambda edge: (
                _strict_int(edge.source_stable_id, "source_stable_id"),
                _strict_int(edge.target_stable_id, "target_stable_id"),
                str(edge.edge_id),
            ),
        )
    )
    edge_ids: set[str] = set()
    pairs: set[tuple[int, int]] = set()
    for edge in ordered:
        if not isinstance(edge.edge_id, str) or not edge.edge_id:
            raise ContractError("forced path-cover edge ID is invalid")
        if edge.edge_id in edge_ids:
            raise ContractError("forced path-cover edge IDs are duplicated")
        edge_ids.add(edge.edge_id)
        source_id = _strict_int(edge.source_stable_id, "source_stable_id")
        target_id = _strict_int(edge.target_stable_id, "target_stable_id")
        pair = (source_id, target_id)
        if source_id == target_id or pair in pairs:
            raise ContractError("forced path-cover directed edge pair is invalid")
        pairs.add(pair)
        if source_id not in nodes_by_id or target_id not in nodes_by_id:
            raise ContractError("forced path-cover edge references an unknown stable ID")
        _strict_int(edge.appearance_cost_int, "appearance_cost_int")
        source = nodes_by_id[source_id]
        target = nodes_by_id[target_id]
        if (
            float(source.end_time_sec) >= float(target.start_time_sec)
            or int(source.end_global_frame) >= int(target.start_global_frame)
        ):
            raise ContractError(
                "forced path-cover edge is reverse, overlapping, or not strictly future-only"
            )
    _assert_acyclic(nodes_by_id, ordered, label="candidate graph")
    return ordered


def _maximum_frame_concurrency(nodes: Sequence[GlobalStableNode]) -> int:
    events = [
        event
        for node in nodes
        for event in (
            (int(node.start_global_frame), 0),
            (int(node.end_global_frame), 1),
        )
    ]
    concurrent = 0
    maximum = 0
    for _, kind in sorted(events):
        if kind == 0:
            concurrent += 1
            maximum = max(maximum, concurrent)
        else:
            concurrent -= 1
    if concurrent != 0:
        raise ContractError(
            "forced path-cover interval concurrency is invalid"
        )
    return maximum


def _certified_maximum_from_path_cover(
    nodes: Sequence[GlobalStableNode],
    edges: Sequence[ForcedAppearanceEdge],
    certified_path_cover: Sequence[Sequence[int]] | None,
) -> int | None:
    """Validate a constructive exact-maximum certificate when supplied."""

    if certified_path_cover is None:
        return None
    if isinstance(certified_path_cover, (str, bytes)):
        raise ContractError(
            "forced path-cover certified_path_cover must be a sequence"
        )
    try:
        raw_paths = tuple(certified_path_cover)
    except TypeError as exc:
        raise ContractError(
            "forced path-cover certified_path_cover must be a sequence"
        ) from exc
    if not raw_paths:
        raise ContractError(
            "forced path-cover certified_path_cover cannot be empty"
        )
    paths: list[tuple[int, ...]] = []
    for raw_path in raw_paths:
        if isinstance(raw_path, (str, bytes)):
            raise ContractError(
                "forced path-cover certified path must be a sequence"
            )
        try:
            path = tuple(
                _strict_int(stable_id, "certified stable_id")
                for stable_id in raw_path
            )
        except TypeError as exc:
            raise ContractError(
                "forced path-cover certified path must be a sequence"
            ) from exc
        if not path:
            raise ContractError(
                "forced path-cover certified path cannot be empty"
            )
        paths.append(path)

    expected_ids = {int(node.stable_id) for node in nodes}
    flattened = [stable_id for path in paths for stable_id in path]
    if (
        len(flattened) != len(expected_ids)
        or len(flattened) != len(set(flattened))
        or set(flattened) != expected_ids
    ):
        raise ContractError(
            "forced path-cover certified paths do not cover every node once"
        )
    candidate_pairs = {
        (int(edge.source_stable_id), int(edge.target_stable_id))
        for edge in edges
    }
    certified_pairs = {
        pair
        for path in paths
        for pair in zip(path, path[1:], strict=False)
    }
    if not certified_pairs <= candidate_pairs:
        raise ContractError(
            "forced path-cover certified path edge is absent from candidates"
        )

    width = _maximum_frame_concurrency(nodes)
    if len(paths) != width:
        raise ContractError(
            "forced path-cover certified path count differs from interval width"
        )
    return len(nodes) - len(paths)


def _solver_costs(
    edges: Sequence[ForcedAppearanceEdge],
    required_links: int,
) -> dict[str, int]:
    if not edges:
        return {}
    # Preserve the existing strictly-positive solver-cost provenance.  Every
    # feasible solution contains exactly ``required_links`` real links, so the
    # constant offset cannot change the declared minimum-appearance objective.
    # Do not encode a global ordinal sum here: at 11-clip scale its multiplier
    # exceeds float64's exact integer range without improving that objective.
    costs = {
        edge.edge_id: int(edge.appearance_cost_int) + 2
        for edge in edges
    }
    maximum = max(costs.values())
    maximum_total = max(maximum, required_links * maximum)
    if maximum_total >= _MAX_EXACT_INTEGER:
        raise ContractError(
            "forced path-cover costs exceed exact float64 integer range; "
            "reduce the appearance cost range or candidate graph"
        )
    return costs


def _maximum_cardinality_via_flow(
    left_count: int,
    right_count: int,
    source_indices: np.ndarray,
    target_indices: np.ndarray,
) -> tuple[int, int, int]:
    """Return an exact bipartite maximum with a verified Dinic certificate."""

    edge_count = len(source_indices)
    vertex_count = left_count + right_count + 2
    source_vertex = 0
    sink_vertex = vertex_count - 1
    left_vertices = 1 + np.arange(left_count, dtype=np.int64)
    right_vertices = 1 + left_count + np.arange(
        right_count, dtype=np.int64
    )
    rows = np.concatenate(
        (
            np.full(left_count, source_vertex, dtype=np.int64),
            1 + source_indices,
            right_vertices,
        )
    )
    columns = np.concatenate(
        (
            left_vertices,
            1 + left_count + target_indices,
            np.full(right_count, sink_vertex, dtype=np.int64),
        )
    )
    capacity = csr_array(
        (
            np.ones(left_count + edge_count + right_count, dtype=np.int64),
            (rows, columns),
        ),
        shape=(vertex_count, vertex_count),
    )
    capacity.sum_duplicates()
    expected_arcs = left_count + edge_count + right_count
    if capacity.nnz != expected_arcs or np.any(capacity.data != 1):
        raise RuntimeError("Dinic capacity network differs")

    optimized = maximum_flow(
        capacity,
        source_vertex,
        sink_vertex,
        method="dinic",
    )
    maximum = int(optimized.flow_value)
    flow = optimized.flow.tocsr()
    if (
        maximum < 0
        or maximum > min(left_count, right_count)
        or flow.shape != capacity.shape
        or not np.issubdtype(flow.dtype, np.integer)
    ):
        raise RuntimeError("Dinic maximum-flow result differs")
    balances = np.asarray(flow.sum(axis=1)).reshape(-1)
    if (
        int(balances[source_vertex]) != maximum
        or int(balances[sink_vertex]) != -maximum
        or np.any(balances[1:sink_vertex] != 0)
    ):
        raise RuntimeError("Dinic maximum-flow conservation differs")

    residual = (capacity - flow).tocsr()
    residual.eliminate_zeros()
    if np.any(residual.data <= 0):
        raise RuntimeError("Dinic residual capacity differs")
    reachable = np.zeros(vertex_count, dtype=np.bool_)
    reachable[source_vertex] = True
    stack = [source_vertex]
    while stack:
        vertex = stack.pop()
        for target in residual.indices[
            residual.indptr[vertex] : residual.indptr[vertex + 1]
        ]:
            target_int = int(target)
            if not reachable[target_int]:
                reachable[target_int] = True
                stack.append(target_int)
    if reachable[sink_vertex]:
        raise RuntimeError("Dinic residual still reaches the sink")
    capacity_coo = capacity.tocoo()
    crosses_cut = reachable[capacity_coo.row] & ~reachable[capacity_coo.col]
    cut_capacity = int(np.sum(capacity_coo.data[crosses_cut]))
    if cut_capacity != maximum:
        raise RuntimeError("Dinic min-cut certificate differs")
    return maximum, vertex_count, expected_arcs


def _min_cost_flow_worker(
    send_connection: Any,
    left_count: int,
    right_count: int,
    source_indices: np.ndarray,
    target_indices: np.ndarray,
    edge_costs: np.ndarray,
    required_flow: int,
    certified_maximum: int | None,
) -> None:
    """Child-only exact fixed-flow solve; all failures cross the pipe."""

    try:
        edge_count = len(edge_costs)
        if certified_maximum is None:
            send_connection.send(
                {
                    "kind": "phase",
                    "phase": "maximum-flow",
                    "edge_count": edge_count,
                    "flow_vertices": left_count + right_count + 2,
                    "flow_arcs": left_count + edge_count + right_count,
                }
            )
            maximum, flow_vertices, flow_arcs = (
                _maximum_cardinality_via_flow(
                    left_count,
                    right_count,
                    source_indices,
                    target_indices,
                )
            )
            if (
                flow_vertices != left_count + right_count + 2
                or flow_arcs != left_count + edge_count + right_count
            ):
                raise RuntimeError("Dinic maximum-flow metrics differ")
        else:
            maximum = int(certified_maximum)
            send_connection.send(
                {
                    "kind": "phase",
                    "phase": "certified-cardinality",
                    "edge_count": edge_count,
                    "maximum": maximum,
                    "certificate_paths": left_count - maximum,
                }
            )
        if maximum < required_flow:
            send_connection.send(
                {
                    "kind": "infeasible",
                    "maximum": maximum,
                    "required_flow": required_flow,
                }
            )
            return
        if required_flow == 0:
            send_connection.send(
                {
                    "kind": "result",
                    "maximum": maximum,
                    "selected_positions": np.empty(0, dtype=np.int64),
                    "simplex_iterations": 0,
                    "flow_variables": edge_count + left_count + right_count,
                    "constraint_nonzeros": 2 * edge_count + 2 * left_count + right_count,
                }
            )
            return

        # Variables are source->left, real candidate, and right->sink arcs.
        # The sink conservation row is redundant and deliberately omitted.
        variable_count = left_count + edge_count + right_count
        source_variables = np.arange(left_count, dtype=np.int64)
        edge_variables = left_count + np.arange(edge_count, dtype=np.int64)
        sink_variables = left_count + edge_count + np.arange(
            right_count, dtype=np.int64
        )
        constraint_rows = np.concatenate(
            (
                np.zeros(left_count, dtype=np.int64),
                1 + np.arange(left_count, dtype=np.int64),
                1 + source_indices,
                1 + left_count + target_indices,
                1 + left_count + np.arange(right_count, dtype=np.int64),
            )
        )
        constraint_columns = np.concatenate(
            (
                source_variables,
                source_variables,
                edge_variables,
                edge_variables,
                sink_variables,
            )
        )
        constraint_data = np.concatenate(
            (
                np.ones(left_count, dtype=np.float64),
                np.ones(left_count, dtype=np.float64),
                -np.ones(edge_count, dtype=np.float64),
                np.ones(edge_count, dtype=np.float64),
                -np.ones(right_count, dtype=np.float64),
            )
        )
        constraints = coo_matrix(
            (constraint_data, (constraint_rows, constraint_columns)),
            shape=(1 + left_count + right_count, variable_count),
        ).tocsr()
        right_hand_side = np.zeros(
            1 + left_count + right_count, dtype=np.float64
        )
        right_hand_side[0] = required_flow
        objective = np.zeros(variable_count, dtype=np.float64)
        objective[left_count : left_count + edge_count] = edge_costs
        send_connection.send(
            {
                "kind": "phase",
                "phase": "min-cost-flow",
                "maximum": maximum,
                "flow_variables": variable_count,
                "constraint_nonzeros": int(constraints.nnz),
            }
        )
        optimized = linprog(
            objective,
            A_eq=constraints,
            b_eq=right_hand_side,
            bounds=(0.0, 1.0),
            method="highs-ds",
            options={"presolve": True},
        )
        if not optimized.success or optimized.x is None:
            raise RuntimeError(
                "HiGHS fixed-flow optimization failed: "
                f"status={optimized.status}, message={optimized.message}"
            )
        flow = np.asarray(optimized.x, dtype=np.float64)
        rounded = np.rint(flow)
        if (
            flow.shape != (variable_count,)
            or not np.all(np.isfinite(flow))
            or np.max(np.abs(flow - rounded), initial=0.0) > _FLOW_INTEGRAL_ATOL
            or np.any((rounded < 0.0) | (rounded > 1.0))
        ):
            raise RuntimeError("HiGHS fixed-flow solution is not integral")
        selected_positions = np.flatnonzero(
            rounded[left_count : left_count + edge_count] == 1.0
        ).astype(np.int64, copy=False)
        if len(selected_positions) != required_flow:
            raise RuntimeError("HiGHS fixed-flow selected cardinality differs")
        if (
            np.max(
                np.bincount(
                    source_indices[selected_positions], minlength=left_count
                ),
                initial=0,
            )
            > 1
            or np.max(
                np.bincount(
                    target_indices[selected_positions], minlength=right_count
                ),
                initial=0,
            )
            > 1
        ):
            raise RuntimeError("HiGHS fixed-flow violates matching degrees")
        exact_objective = sum(
            int(edge_costs[index]) for index in selected_positions
        )
        if not math.isclose(
            float(optimized.fun),
            float(exact_objective),
            rel_tol=0.0,
            abs_tol=1e-5,
        ):
            raise RuntimeError("HiGHS fixed-flow objective differs")
        send_connection.send(
            {
                "kind": "result",
                "maximum": maximum,
                "selected_positions": selected_positions,
                "simplex_iterations": int(optimized.nit),
                "flow_variables": variable_count,
                "constraint_nonzeros": int(constraints.nnz),
            }
        )
    except BaseException as exc:
        try:
            send_connection.send(
                {
                    "kind": "error",
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
        except BaseException:
            pass
    finally:
        send_connection.close()


def _child_resource_summary(pid: int) -> str:
    """Best-effort Linux child CPU/RSS telemetry for a heartbeat."""

    try:
        stat_text = (Path(f"/proc/{pid}/stat")).read_text(encoding="utf-8")
        after_name = stat_text.rsplit(")", 1)[1].split()
        ticks = int(after_name[11]) + int(after_name[12])
        cpu_sec = ticks / float(os.sysconf("SC_CLK_TCK"))
        status = Path(f"/proc/{pid}/status").read_text(encoding="utf-8")
        rss_line = next(line for line in status.splitlines() if line.startswith("VmRSS:"))
        rss_mib = int(rss_line.split()[1]) / 1024.0
        return f"cpu={cpu_sec:.1f}s, rss={rss_mib:.1f} MiB"
    except (OSError, ValueError, IndexError, StopIteration):
        return "cpu/rss=unavailable"


def _terminate_owned_child(process: Any) -> None:
    if not process.is_alive():
        process.join(timeout=_CHILD_JOIN_GRACE_SEC)
        return
    process.terminate()
    process.join(timeout=_CHILD_JOIN_GRACE_SEC)
    if process.is_alive():
        process.kill()
        process.join(timeout=_CHILD_JOIN_GRACE_SEC)


def _solve_min_cost_flow_subprocess(
    *,
    left_count: int,
    right_count: int,
    source_indices: np.ndarray,
    target_indices: np.ndarray,
    edge_costs: np.ndarray,
    required_flow: int,
    certified_maximum: int | None,
    progress_interval_sec: float,
    progress_logger: Callable[[str], None] | None,
    progress_label: str,
) -> tuple[np.ndarray, int, Mapping[str, int]]:
    arrays = tuple(
        np.asarray(values)
        for values in (source_indices, target_indices, edge_costs)
    )
    sources, targets, costs = arrays
    certificate_is_invalid = (
        certified_maximum is not None
        and (
            isinstance(certified_maximum, (bool, np.bool_))
            or not isinstance(certified_maximum, (int, np.integer))
            or int(certified_maximum) < 0
            or int(certified_maximum) > min(left_count, right_count)
        )
    )
    if (
        any(values.ndim != 1 or values.dtype != np.int64 for values in arrays)
        or not (len(sources) == len(targets) == len(costs))
        or left_count < 1
        or right_count < 1
        or required_flow < 0
        or required_flow > min(left_count, right_count)
        or (len(sources) and (np.any(sources < 0) or np.any(sources >= left_count)))
        or (len(targets) and (np.any(targets < 0) or np.any(targets >= right_count)))
        or np.any(costs < 1)
        or certificate_is_invalid
    ):
        raise ContractError("forced path-cover fixed-flow input differs")
    interval = float(progress_interval_sec)
    if not math.isfinite(interval) or interval <= 0.0:
        raise ContractError("forced path-cover progress interval must be positive")

    context = multiprocessing_module.get_context("spawn")
    receive_connection, send_connection = context.Pipe(duplex=False)
    process = context.Process(
        target=_min_cost_flow_worker,
        args=(
            send_connection,
            int(left_count),
            int(right_count),
            np.array(sources, copy=True),
            np.array(targets, copy=True),
            np.array(costs, copy=True),
            int(required_flow),
            (
                None
                if certified_maximum is None
                else int(certified_maximum)
            ),
        ),
        name="forced-min-cost-flow",
    )
    try:
        process.start()
    except BaseException as exc:
        receive_connection.close()
        send_connection.close()
        raise ContractError(f"cannot start forced path-cover child: {exc}") from exc
    send_connection.close()
    started = time.monotonic()
    phase = "starting"
    payload: Mapping[str, Any] | None = None
    try:
        while payload is None:
            if receive_connection.poll(interval):
                try:
                    message = receive_connection.recv()
                except EOFError:
                    message = None
                if not isinstance(message, Mapping):
                    if process.is_alive():
                        continue
                    raise ContractError(
                        "forced path-cover child exited without a result"
                    )
                if message.get("kind") == "phase":
                    phase = str(message.get("phase"))
                    if progress_logger is not None:
                        extras = ", ".join(
                            f"{key}={int(message[key]):,}"
                            for key in (
                                "edge_count",
                                "maximum",
                                "flow_vertices",
                                "flow_arcs",
                                "certificate_paths",
                                "flow_variables",
                                "constraint_nonzeros",
                            )
                            if key in message
                        )
                        progress_logger(
                            f"{progress_label}: child pid={process.pid} entered "
                            f"{phase}" + (f"; {extras}" if extras else "")
                        )
                    continue
                payload = message
                break
            if not process.is_alive():
                if receive_connection.poll(0.0):
                    continue
                raise ContractError(
                    "forced path-cover child exited before publishing a result"
                )
            if progress_logger is not None:
                elapsed = time.monotonic() - started
                progress_logger(
                    f"{progress_label}: child pid={process.pid} phase={phase}; "
                    f"elapsed={elapsed:.0f}s, {_child_resource_summary(process.pid)}"
                )
    except KeyboardInterrupt:
        if progress_logger is not None:
            progress_logger(
                f"{progress_label}: interrupt received; terminating owned child "
                f"pid={process.pid}"
            )
        _terminate_owned_child(process)
        raise
    except BaseException:
        _terminate_owned_child(process)
        raise
    finally:
        receive_connection.close()

    process.join(timeout=_CHILD_JOIN_GRACE_SEC)
    if process.is_alive():
        _terminate_owned_child(process)
        raise ContractError("forced path-cover child did not exit after publishing")
    if process.exitcode != 0:
        raise ContractError(
            f"forced path-cover child exited with code {process.exitcode}"
        )
    if payload is None:
        raise ContractError("forced path-cover child result is missing")
    kind = str(payload.get("kind"))
    if kind == "error":
        raise ContractError(
            "forced path-cover child failed: "
            f"{payload.get('error_type')}: {payload.get('message')}"
        )
    maximum = int(payload.get("maximum", -1))
    if kind == "infeasible":
        if progress_logger is not None:
            elapsed = time.monotonic() - started
            progress_logger(
                f"{progress_label}: child pid={process.pid} proved infeasible; "
                f"maximum={maximum:,}, required={required_flow:,}, "
                f"elapsed={elapsed:.1f}s"
            )
        return np.empty(0, dtype=np.int64), maximum, MappingProxyType({})
    if kind != "result":
        raise ContractError("forced path-cover child result kind differs")
    selected = np.asarray(payload.get("selected_positions"))
    if (
        selected.dtype != np.int64
        or selected.ndim != 1
        or len(selected) != required_flow
        or len(np.unique(selected)) != len(selected)
        or (len(selected) and (np.any(selected < 0) or np.any(selected >= len(costs))))
    ):
        raise ContractError("forced path-cover child selection differs")
    metrics = MappingProxyType(
        {
            "simplex_iterations": int(payload.get("simplex_iterations", -1)),
            "flow_variables": int(payload.get("flow_variables", -1)),
            "constraint_nonzeros": int(payload.get("constraint_nonzeros", -1)),
        }
    )
    if any(value < 0 for value in metrics.values()) or maximum < required_flow:
        raise ContractError("forced path-cover child metrics differ")
    if progress_logger is not None:
        elapsed = time.monotonic() - started
        progress_logger(
            f"{progress_label}: child pid={process.pid} completed; "
            f"selected={len(selected):,}, maximum={maximum:,}, "
            f"simplex_iterations={metrics['simplex_iterations']:,}, "
            f"flow_variables={metrics['flow_variables']:,}, "
            f"constraint_nonzeros={metrics['constraint_nonzeros']:,}, "
            f"elapsed={elapsed:.1f}s"
        )
    return selected, maximum, metrics


def _solve_fixed_assignment(
    nodes: Sequence[GlobalStableNode],
    edges: Sequence[ForcedAppearanceEdge],
    costs: Mapping[str, int],
    *,
    target_num_paths: int,
    certified_maximum: int | None,
    progress_interval_sec: float,
    progress_logger: Callable[[str], None] | None,
    progress_label: str,
) -> tuple[tuple[ForcedAppearanceEdge, ...], int]:
    node_count = len(nodes)
    required_links = node_count - target_num_paths
    node_index = {int(node.stable_id): index for index, node in enumerate(nodes)}
    sources = np.fromiter(
        (node_index[int(edge.source_stable_id)] for edge in edges),
        dtype=np.int64,
        count=len(edges),
    )
    targets = np.fromiter(
        (node_index[int(edge.target_stable_id)] for edge in edges),
        dtype=np.int64,
        count=len(edges),
    )
    edge_costs = np.fromiter(
        (costs[edge.edge_id] for edge in edges),
        dtype=np.int64,
        count=len(edges),
    )
    selected_positions, maximum, _metrics = _solve_min_cost_flow_subprocess(
        left_count=node_count,
        right_count=node_count,
        source_indices=sources,
        target_indices=targets,
        edge_costs=edge_costs,
        required_flow=required_links,
        certified_maximum=certified_maximum,
        progress_interval_sec=progress_interval_sec,
        progress_logger=progress_logger,
        progress_label=progress_label,
    )
    if maximum < required_links:
        minimum_paths = node_count - maximum
        raise ContractError(
            "forced path-cover target is infeasible: "
            f"requires {required_links} links but at most {maximum} are possible; "
            f"minimum feasible paths={minimum_paths}"
        )
    selected = tuple(edges[int(position)] for position in selected_positions)
    if len(selected) != required_links:
        raise ContractError(
            "forced path-cover fixed-flow did not select exactly N-K real links"
        )
    return selected, maximum


def _build_paths(
    nodes: Sequence[GlobalStableNode],
    nodes_by_id: Mapping[int, GlobalStableNode],
    selected: Sequence[ForcedAppearanceEdge],
    *,
    target_num_paths: int,
) -> tuple[
    tuple[tuple[int, ...], ...],
    Mapping[int, int | None],
    Mapping[int, int | None],
]:
    predecessor: dict[int, int | None] = {int(node.stable_id): None for node in nodes}
    successor: dict[int, int | None] = {int(node.stable_id): None for node in nodes}
    for edge in selected:
        source_id = int(edge.source_stable_id)
        target_id = int(edge.target_stable_id)
        if successor[source_id] is not None or predecessor[target_id] is not None:
            raise ContractError("forced path-cover selected links violate one-in/one-out")
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
                raise ContractError("forced path-cover selected path revisits a stable node")
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
                    raise ContractError(
                        "forced path-cover selected path is not strictly future-only"
                    )
            current = following
        paths.append(tuple(path))
    if visited != set(nodes_by_id):
        raise ContractError("forced path-cover did not cover every stable node once")
    if len(paths) != target_num_paths or len(paths) != len(nodes) - len(selected):
        raise ContractError("forced path-cover result does not contain exactly K paths")
    return (
        tuple(paths),
        MappingProxyType(dict(predecessor)),
        MappingProxyType(dict(successor)),
    )


def solve_forced_fixed_path_cover(
    nodes: Sequence[GlobalStableNode],
    edges: Sequence[ForcedAppearanceEdge],
    *,
    target_num_paths: int,
    certified_path_cover: Sequence[Sequence[int]] | None = None,
    progress_interval_sec: float = 10.0,
    progress_logger: Callable[[str], None] | None = None,
    progress_label: str = "forced fixed path-cover",
) -> ForcedPathCoverResult:
    """Minimize integer appearance cost subject to an exact ``K`` path count.

    Every candidate edge must be strictly future-only and temporally disjoint.
    The selected graph has at most one incoming and one outgoing edge per stable
    node.  If a matching of exactly ``N-K`` edges does not exist, the operation
    fails closed and reports the minimum path count allowed by the supplied
    graph.  A supplied ``certified_path_cover`` must be a minimum-width cover
    made entirely from candidate edges; it proves the exact maximum matching
    cardinality and skips the generic Dinic precheck.
    """

    ordered_nodes, nodes_by_id = _validated_nodes(nodes)
    interval, logger, label = _validated_progress(
        progress_interval_sec,
        progress_logger,
        progress_label,
    )
    target = _strict_int(target_num_paths, "target_num_paths", minimum=1)
    if target > len(ordered_nodes):
        raise ContractError("forced path-cover target_num_paths exceeds node count")
    ordered_edges = _validated_edges(edges, nodes_by_id)
    certified_maximum = _certified_maximum_from_path_cover(
        ordered_nodes,
        ordered_edges,
        certified_path_cover,
    )
    required_links = len(ordered_nodes) - target
    costs = _solver_costs(
        ordered_edges,
        required_links,
    )
    selected, maximum = _solve_fixed_assignment(
        ordered_nodes,
        ordered_edges,
        costs,
        target_num_paths=target,
        certified_maximum=certified_maximum,
        progress_interval_sec=interval,
        progress_logger=logger,
        progress_label=label,
    )
    canonical_selected = tuple(
        sorted(
            selected,
            key=lambda edge: (
                int(edge.source_stable_id),
                int(edge.target_stable_id),
                str(edge.edge_id),
            ),
        )
    )
    paths, predecessor, successor = _build_paths(
        ordered_nodes,
        nodes_by_id,
        canonical_selected,
        target_num_paths=target,
    )
    return ForcedPathCoverResult(
        selected_edges=canonical_selected,
        paths=paths,
        predecessor_by_stable=predecessor,
        successor_by_stable=successor,
        solver_cost_by_edge=MappingProxyType(dict(sorted(costs.items()))),
        target_num_paths=target,
        required_links=required_links,
        maximum_feasible_links=maximum,
        total_appearance_cost_int=sum(
            int(edge.appearance_cost_int) for edge in canonical_selected
        ),
    )


__all__ = [
    "ForcedAppearanceEdge",
    "ForcedPathCoverResult",
    "solve_forced_fixed_path_cover",
]
