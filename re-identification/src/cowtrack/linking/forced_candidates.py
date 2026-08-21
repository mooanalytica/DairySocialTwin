"""Deterministic appearance candidates for fixed-K forced path cover.

The graph is the union of bidirectional temporal top-k retrieval, explicitly
supplied prior links, and a minimum-width interval-chain backbone.  The
backbone contains ``N-width`` one-in/one-out edges, so every requested path
count ``K >= width`` is structurally feasible without ever adding an overlap or
same-frame link.  The complete sequence always uses this single minimum-width
construction.
"""

from __future__ import annotations

import heapq
import math
from collections.abc import Mapping, Sequence, Set
from dataclasses import dataclass
from typing import Any

import numpy as np

from cowtrack.config import ContractError
from cowtrack.linking.forced_path_cover import ForcedAppearanceEdge
from cowtrack.linking.path_cover import GlobalStableNode


GRADE_A_CLEAN = "A_CLEAN"
GRADE_B_EXISTING_DEGRADED = "B_EXISTING_DEGRADED"
GRADE_C_REENCODED_DEGRADED = "C_REENCODED_DEGRADED"

DEFAULT_GRADE_PENALTIES: Mapping[str, int] = {
    GRADE_A_CLEAN: 0,
    GRADE_B_EXISTING_DEGRADED: 5,
    GRADE_C_REENCODED_DEGRADED: 15,
}

# Bound the largest temporary cosine slab.  Retrieval remains exhaustive:
# every source/target pair is scored exactly once, but no N-by-N matrix is
# materialized for the full 11-clip stable-track population.
_COSINE_BLOCK_ROWS = 128


@dataclass(frozen=True, slots=True)
class ForcedCandidateRecord:
    """One union-graph candidate and all cost/provenance fields."""

    edge: ForcedAppearanceEdge
    cosine_similarity: float
    source_grade: str
    target_grade: str
    selected_by_source_topk: bool
    selected_by_target_topk: bool
    selected_by_backbone: bool
    selected_by_prior: bool
    base_cost_int: int


@dataclass(frozen=True, slots=True)
class ForcedCandidateGraph:
    """Canonical candidate records plus its deterministic backbone."""

    candidates: tuple[ForcedCandidateRecord, ...]
    backbone_chains: tuple[tuple[int, ...], ...]
    max_concurrent: int
    chain_count: int
    top_k: int
    cost_scale: int
    prior_bonus: int
    grade_penalties: tuple[tuple[str, int], ...]

    @property
    def edges(self) -> tuple[ForcedAppearanceEdge, ...]:
        return tuple(record.edge for record in self.candidates)


def _strict_int(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ContractError(f"forced candidates {label} must be an integer")
    result = int(value)
    if result < minimum:
        raise ContractError(f"forced candidates {label} must be >= {minimum}")
    return result


def _finite(value: Any, label: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ContractError(f"forced candidates {label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ContractError(f"forced candidates {label} must be finite")
    return result


def _validated_aligned_inputs(
    nodes: Sequence[GlobalStableNode],
    center_embeddings: np.ndarray,
    appearance_grades: Sequence[str],
) -> tuple[tuple[GlobalStableNode, ...], np.ndarray, tuple[str, ...]]:
    if isinstance(nodes, (str, bytes)) or isinstance(appearance_grades, (str, bytes)):
        raise ContractError("forced candidates nodes/grades must be sequences")
    try:
        supplied_nodes = tuple(nodes)
        supplied_grades = tuple(appearance_grades)
    except TypeError as exc:
        raise ContractError("forced candidates nodes/grades must be sequences") from exc
    if not supplied_nodes:
        raise ContractError("forced candidates require at least one stable node")
    if any(not isinstance(node, GlobalStableNode) for node in supplied_nodes):
        raise ContractError("forced candidates node has the wrong type")
    if len(supplied_grades) != len(supplied_nodes):
        raise ContractError("forced candidates grade count differs from node count")

    embeddings = np.asarray(center_embeddings)
    if (
        embeddings.ndim != 2
        or embeddings.shape[0] != len(supplied_nodes)
        or embeddings.shape[1] <= 0
        or embeddings.dtype.kind != "f"
    ):
        raise ContractError(
            "forced candidates center embeddings must be floating [N,D]"
        )
    embeddings = embeddings.astype(np.float32, copy=False)
    if not np.all(np.isfinite(embeddings)):
        raise ContractError("forced candidates center embeddings must be finite")
    norms = np.linalg.norm(embeddings, axis=1)
    if not np.allclose(norms, 1.0, rtol=0.0, atol=2e-3):
        raise ContractError("forced candidates center embeddings must be L2-normalized")

    valid_grades = set(DEFAULT_GRADE_PENALTIES)
    for grade in supplied_grades:
        if not isinstance(grade, str) or grade not in valid_grades:
            raise ContractError(
                "forced candidates grades must be A_CLEAN, "
                "B_EXISTING_DEGRADED, or C_REENCODED_DEGRADED"
            )

    stable_ids: set[int] = set()
    indexed: list[tuple[int, GlobalStableNode, np.ndarray, str]] = []
    endpoint_times: dict[int, float] = {}
    for input_index, (node, grade) in enumerate(
        zip(supplied_nodes, supplied_grades, strict=True)
    ):
        stable_id = _strict_int(node.stable_id, "stable_id")
        if stable_id in stable_ids:
            raise ContractError("forced candidates stable IDs are duplicated")
        stable_ids.add(stable_id)
        if (
            not isinstance(node.start_clip_id, str)
            or not node.start_clip_id
            or not isinstance(node.end_clip_id, str)
            or not node.end_clip_id
        ):
            raise ContractError("forced candidates node clip IDs are invalid")
        start_frame = _strict_int(node.start_global_frame, "start_global_frame")
        end_frame = _strict_int(node.end_global_frame, "end_global_frame")
        start_time = _finite(node.start_time_sec, "start_time_sec")
        end_time = _finite(node.end_time_sec, "end_time_sec")
        if start_frame > end_frame or start_time > end_time:
            raise ContractError("forced candidates node temporal bounds are invalid")
        _strict_int(node.num_microtracklets, "num_microtracklets", minimum=1)
        _strict_int(node.num_detections, "num_detections", minimum=1)
        for frame, timestamp in (
            (start_frame, start_time),
            (end_frame, end_time),
        ):
            previous = endpoint_times.get(frame)
            if previous is not None and not math.isclose(
                previous, timestamp, rel_tol=0.0, abs_tol=2e-6
            ):
                raise ContractError(
                    "forced candidates global frame maps to inconsistent times"
                )
            endpoint_times[frame] = timestamp
        indexed.append((stable_id, node, embeddings[input_index], grade))

    ordered_endpoints = sorted(endpoint_times.items())
    for (left_frame, left_time), (right_frame, right_time) in zip(
        ordered_endpoints, ordered_endpoints[1:]
    ):
        if left_frame >= right_frame or left_time >= right_time:
            raise ContractError(
                "forced candidates global frame/time ordering is inconsistent"
            )

    indexed.sort(key=lambda item: item[0])
    ordered_nodes = tuple(item[1] for item in indexed)
    ordered_embeddings = np.stack([item[2] for item in indexed]).astype(
        np.float32, copy=False
    )
    ordered_grades = tuple(item[3] for item in indexed)
    return ordered_nodes, ordered_embeddings, ordered_grades


def _validated_penalties(
    value: Mapping[str, int] | None,
) -> dict[str, int]:
    supplied = DEFAULT_GRADE_PENALTIES if value is None else value
    if not isinstance(supplied, Mapping) or set(supplied) != set(
        DEFAULT_GRADE_PENALTIES
    ):
        raise ContractError("forced candidates grade penalties have invalid keys")
    return {
        grade: _strict_int(supplied[grade], f"grade penalty {grade}")
        for grade in DEFAULT_GRADE_PENALTIES
    }


def _strictly_future(source: GlobalStableNode, target: GlobalStableNode) -> bool:
    return bool(
        int(source.end_global_frame) < int(target.start_global_frame)
        and float(source.end_time_sec) < float(target.start_time_sec)
    )


def _validated_prior_pairs(
    prior_pairs: Set[tuple[int, int]],
    nodes_by_id: Mapping[int, GlobalStableNode],
) -> set[tuple[int, int]]:
    if not isinstance(prior_pairs, Set) or isinstance(prior_pairs, (str, bytes)):
        raise ContractError("forced candidates prior_pairs must be a set")
    result: set[tuple[int, int]] = set()
    for raw_pair in prior_pairs:
        if not isinstance(raw_pair, tuple) or len(raw_pair) != 2:
            raise ContractError("forced candidates prior pair must be an integer tuple")
        source_id = _strict_int(raw_pair[0], "prior source_stable_id")
        target_id = _strict_int(raw_pair[1], "prior target_stable_id")
        if source_id == target_id:
            raise ContractError("forced candidates prior pair cannot be self-directed")
        if source_id not in nodes_by_id or target_id not in nodes_by_id:
            raise ContractError("forced candidates prior pair references an unknown stable ID")
        if not _strictly_future(nodes_by_id[source_id], nodes_by_id[target_id]):
            raise ContractError(
                "forced candidates prior pair is reverse or temporally overlapping"
            )
        result.add((source_id, target_id))
    return result


def _maximum_concurrent(nodes: Sequence[GlobalStableNode]) -> int:
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
        raise ContractError("forced candidates interval concurrency is invalid")
    return maximum


def _interval_backbone(
    nodes: Sequence[GlobalStableNode],
) -> tuple[tuple[tuple[int, ...], ...], set[tuple[int, int]], int]:
    ordered = sorted(
        nodes,
        key=lambda node: (
            int(node.start_global_frame),
            float(node.start_time_sec),
            int(node.stable_id),
        ),
    )
    # (last end frame, last end time, chain id, last stable id)
    available: list[tuple[int, float, int, int]] = []
    chains: list[list[int]] = []
    backbone: set[tuple[int, int]] = set()
    for node in ordered:
        start_frame = int(node.start_global_frame)
        start_time = float(node.start_time_sec)
        if (
            available
            and available[0][0] < start_frame
            and available[0][1] < start_time
        ):
            _, _, chain_id, previous_id = heapq.heappop(available)
            backbone.add((previous_id, int(node.stable_id)))
            chains[chain_id].append(int(node.stable_id))
        else:
            chain_id = len(chains)
            chains.append([int(node.stable_id)])
        heapq.heappush(
            available,
            (
                int(node.end_global_frame),
                float(node.end_time_sec),
                chain_id,
                int(node.stable_id),
            ),
        )

    maximum = _maximum_concurrent(nodes)
    if maximum != len(chains):
        raise ContractError(
            "forced candidates interval backbone is not minimum-width"
        )
    if len(backbone) != len(nodes) - len(chains):
        raise ContractError("forced candidates backbone edge count differs from N-width")
    return tuple(tuple(chain) for chain in chains), backbone, maximum


def _topk_pairs(
    stable_ids: np.ndarray,
    embeddings: np.ndarray,
    end_frames: np.ndarray,
    start_frames: np.ndarray,
    end_times: np.ndarray,
    start_times: np.ndarray,
    *,
    top_k: int,
) -> tuple[
    dict[tuple[int, int], float],
    dict[tuple[int, int], float],
]:
    """Return exhaustive bidirectional top-k without dense N-by-N storage."""

    node_count = len(stable_ids)
    sentinel_id = np.iinfo(np.int64).max
    incoming_scores = np.full(
        (top_k, node_count), -np.inf, dtype=np.float32
    )
    incoming_source_ids = np.full(
        (top_k, node_count), sentinel_id, dtype=np.int64
    )
    source_scores: dict[tuple[int, int], float] = {}

    for block_start in range(0, node_count, _COSINE_BLOCK_ROWS):
        block_stop = min(block_start + _COSINE_BLOCK_ROWS, node_count)
        similarities = np.asarray(
            embeddings[block_start:block_stop] @ embeddings.T,
            dtype=np.float32,
        )
        if not np.all(np.isfinite(similarities)) or np.any(
            (similarities < -1.002) | (similarities > 1.002)
        ):
            raise ContractError("forced candidates cosine slab is invalid")
        np.clip(similarities, -1.0, 1.0, out=similarities)
        compatible = (
            end_frames[block_start:block_stop, None] < start_frames[None, :]
        ) & (
            end_times[block_start:block_stop, None] < start_times[None, :]
        )

        # Exact outgoing top-k for every source in this slab.
        for local_index, source_index in enumerate(
            range(block_start, block_stop)
        ):
            targets = np.flatnonzero(compatible[local_index])
            if not len(targets):
                continue
            order = np.lexsort(
                (stable_ids[targets], -similarities[local_index, targets])
            )
            source_id = int(stable_ids[source_index])
            for target_index in targets[order[:top_k]]:
                pair = (source_id, int(stable_ids[target_index]))
                source_scores[pair] = float(
                    similarities[local_index, target_index]
                )

        # Merge this source slab into every target's exact incoming top-k.
        # Keeping the prior top-k is sufficient: a previously discarded item
        # cannot enter the global top-k after more candidates are added.
        slab_scores = np.where(compatible, similarities, -np.inf)
        candidate_scores = np.concatenate(
            (incoming_scores, slab_scores), axis=0
        )
        slab_source_ids = np.broadcast_to(
            stable_ids[block_start:block_stop, None],
            slab_scores.shape,
        )
        candidate_source_ids = np.concatenate(
            (incoming_source_ids, slab_source_ids), axis=0
        )
        order = np.lexsort(
            (candidate_source_ids, -candidate_scores), axis=0
        )[:top_k]
        incoming_scores = np.take_along_axis(candidate_scores, order, axis=0)
        incoming_source_ids = np.take_along_axis(
            candidate_source_ids, order, axis=0
        )

    target_scores: dict[tuple[int, int], float] = {}
    for target_index, target_id in enumerate(stable_ids):
        for rank in range(top_k):
            source_id = int(incoming_source_ids[rank, target_index])
            score = float(incoming_scores[rank, target_index])
            if source_id == sentinel_id or not math.isfinite(score):
                break
            target_scores[(source_id, int(target_id))] = score
    return source_scores, target_scores


def build_forced_candidate_graph(
    nodes: Sequence[GlobalStableNode],
    center_embeddings: np.ndarray,
    appearance_grades: Sequence[str],
    prior_pairs: Set[tuple[int, int]],
    *,
    top_k: int,
    cost_scale: int = 400,
    grade_penalties: Mapping[str, int] | None = None,
    prior_bonus: int = 3,
) -> ForcedCandidateGraph:
    """Build the full-sequence union graph and minimum-width backbone."""

    requested_top_k = _strict_int(top_k, "top_k", minimum=1)
    scale = _strict_int(cost_scale, "cost_scale", minimum=1)
    bonus = _strict_int(prior_bonus, "prior_bonus")
    penalties = _validated_penalties(grade_penalties)
    ordered_nodes, embeddings, grades = _validated_aligned_inputs(
        nodes, center_embeddings, appearance_grades
    )
    stable_ids = np.asarray(
        [int(node.stable_id) for node in ordered_nodes], dtype=np.int64
    )
    nodes_by_id = {
        int(node.stable_id): node for node in ordered_nodes
    }
    grade_by_id = {
        int(stable_id): grade
        for stable_id, grade in zip(stable_ids, grades, strict=True)
    }
    index_by_id = {
        int(stable_id): index for index, stable_id in enumerate(stable_ids)
    }
    validated_priors = _validated_prior_pairs(prior_pairs, nodes_by_id)

    end_frames = np.asarray(
        [int(node.end_global_frame) for node in ordered_nodes], dtype=np.int64
    )
    start_frames = np.asarray(
        [int(node.start_global_frame) for node in ordered_nodes], dtype=np.int64
    )
    end_times = np.asarray(
        [float(node.end_time_sec) for node in ordered_nodes], dtype=np.float64
    )
    start_times = np.asarray(
        [float(node.start_time_sec) for node in ordered_nodes], dtype=np.float64
    )
    source_topk_scores, target_topk_scores = _topk_pairs(
        stable_ids,
        embeddings,
        end_frames,
        start_frames,
        end_times,
        start_times,
        top_k=requested_top_k,
    )
    source_topk = set(source_topk_scores)
    target_topk = set(target_topk_scores)
    chains, backbone, max_concurrent = _interval_backbone(ordered_nodes)
    candidate_pairs = source_topk | target_topk | validated_priors | backbone
    cosine_by_pair = dict(target_topk_scores)
    cosine_by_pair.update(source_topk_scores)
    for source_id, target_id in sorted(candidate_pairs - set(cosine_by_pair)):
        cosine = float(
            np.dot(
                embeddings[index_by_id[source_id]],
                embeddings[index_by_id[target_id]],
            )
        )
        if not math.isfinite(cosine) or cosine < -1.002 or cosine > 1.002:
            raise ContractError("forced candidates pair cosine is invalid")
        cosine_by_pair[(source_id, target_id)] = min(1.0, max(-1.0, cosine))
    records: list[ForcedCandidateRecord] = []
    for source_id, target_id in sorted(candidate_pairs):
        source = nodes_by_id[source_id]
        target = nodes_by_id[target_id]
        if not _strictly_future(source, target):
            raise ContractError(
                "forced candidates union contains a reverse or overlapping edge"
            )
        cosine = cosine_by_pair[(source_id, target_id)]
        source_grade = grade_by_id[source_id]
        target_grade = grade_by_id[target_id]
        selected_by_prior = (source_id, target_id) in validated_priors
        base_cost = (
            int(round((1.0 - cosine) * scale))
            + penalties[source_grade]
            + penalties[target_grade]
            - (bonus if selected_by_prior else 0)
        )
        base_cost = max(0, base_cost)
        edge = ForcedAppearanceEdge(
            edge_id=f"forced-{source_id:06d}-{target_id:06d}",
            source_stable_id=source_id,
            target_stable_id=target_id,
            appearance_cost_int=base_cost,
        )
        records.append(
            ForcedCandidateRecord(
                edge=edge,
                cosine_similarity=cosine,
                source_grade=source_grade,
                target_grade=target_grade,
                selected_by_source_topk=(source_id, target_id) in source_topk,
                selected_by_target_topk=(source_id, target_id) in target_topk,
                selected_by_backbone=(source_id, target_id) in backbone,
                selected_by_prior=selected_by_prior,
                base_cost_int=base_cost,
            )
        )

    observed_pairs = {
        (record.edge.source_stable_id, record.edge.target_stable_id)
        for record in records
    }
    flattened_chains = [stable_id for chain in chains for stable_id in chain]
    observed_backbone = {
        pair
        for chain in chains
        for pair in zip(chain, chain[1:], strict=False)
    }
    source_topk_counts: dict[int, int] = {}
    target_topk_counts: dict[int, int] = {}
    for source_id, target_id in source_topk:
        source_topk_counts[source_id] = source_topk_counts.get(source_id, 0) + 1
    for _, target_id in target_topk:
        target_topk_counts[target_id] = target_topk_counts.get(target_id, 0) + 1
    if (
        observed_pairs != candidate_pairs
        or not backbone <= observed_pairs
        or not validated_priors <= observed_pairs
        or observed_backbone != backbone
        or sorted(flattened_chains) != list(map(int, stable_ids))
        or len(flattened_chains) != len(set(flattened_chains))
        or any(count > requested_top_k for count in source_topk_counts.values())
        or any(count > requested_top_k for count in target_topk_counts.values())
        or len({record.edge.edge_id for record in records}) != len(records)
        or any(record.edge.appearance_cost_int != record.base_cost_int for record in records)
    ):
        raise ContractError("forced candidates final graph validation failed")
    return ForcedCandidateGraph(
        candidates=tuple(records),
        backbone_chains=chains,
        max_concurrent=max_concurrent,
        chain_count=len(chains),
        top_k=requested_top_k,
        cost_scale=scale,
        prior_bonus=bonus,
        grade_penalties=tuple((grade, penalties[grade]) for grade in penalties),
    )


__all__ = [
    "DEFAULT_GRADE_PENALTIES",
    "GRADE_A_CLEAN",
    "GRADE_B_EXISTING_DEGRADED",
    "GRADE_C_REENCODED_DEGRADED",
    "ForcedCandidateGraph",
    "ForcedCandidateRecord",
    "build_forced_candidate_graph",
]
