"""Deterministic operator-approved component union for S04 finalization."""

from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Sequence

from cowtrack.config import ContractError


@dataclass(frozen=True)
class FinalizeMicro:
    micro_id: int
    start_det_id: int
    end_det_id: int
    start_clip_id: str
    end_clip_id: str
    start_global_frame: int
    end_global_frame: int
    start_time_sec: float
    end_time_sec: float
    num_detections: int


@dataclass(frozen=True)
class FinalizeProposal:
    proposal_id: str
    edge_id: str
    source_micro_id: int
    target_micro_id: int
    probability: float


@dataclass(frozen=True)
class ComponentUnionResult:
    micro_to_stable_rows: tuple[dict[str, object], ...]
    stable_tracklet_rows: tuple[dict[str, object], ...]
    micro_to_stable: dict[int, int]
    components: tuple[tuple[int, ...], ...]


def _validate_micros(micros: Sequence[FinalizeMicro]) -> dict[int, FinalizeMicro]:
    by_id: dict[int, FinalizeMicro] = {}
    for micro in micros:
        if not isinstance(micro, FinalizeMicro):
            raise ContractError("S04 finalize micros must be FinalizeMicro values")
        if micro.micro_id in by_id or micro.micro_id < 0:
            raise ContractError("S04 finalize micro IDs must be unique non-negative values")
        if (
            not math.isfinite(micro.start_time_sec)
            or not math.isfinite(micro.end_time_sec)
            or micro.end_time_sec < micro.start_time_sec
            or micro.start_global_frame < 0
            or micro.end_global_frame < micro.start_global_frame
            or micro.num_detections < 1
            or not micro.start_clip_id
            or not micro.end_clip_id
        ):
            raise ContractError(f"S04 finalize micro {micro.micro_id} is inconsistent")
        by_id[micro.micro_id] = micro
    return by_id


def _validate_proposals(
    proposals: Sequence[FinalizeProposal], by_micro: dict[int, FinalizeMicro]
) -> tuple[dict[int, set[int]], dict[frozenset[int], FinalizeProposal]]:
    adjacency = {micro_id: set() for micro_id in by_micro}
    by_pair: dict[frozenset[int], FinalizeProposal] = {}
    proposal_ids: set[str] = set()
    edge_ids: set[str] = set()
    for proposal in proposals:
        if not isinstance(proposal, FinalizeProposal):
            raise ContractError("S04 finalize proposals must be FinalizeProposal values")
        if (
            not proposal.proposal_id
            or proposal.proposal_id in proposal_ids
            or not proposal.edge_id
            or proposal.edge_id in edge_ids
        ):
            raise ContractError("S04 finalize proposal/edge IDs must be unique")
        proposal_ids.add(proposal.proposal_id)
        edge_ids.add(proposal.edge_id)
        source = proposal.source_micro_id
        target = proposal.target_micro_id
        if source == target:
            raise ContractError("S04 finalize proposal contains a self edge")
        if source not in by_micro or target not in by_micro:
            raise ContractError("S04 finalize proposal references an unknown micro")
        if (
            isinstance(proposal.probability, bool)
            or not math.isfinite(proposal.probability)
            or not 0.0 <= proposal.probability <= 1.0
        ):
            raise ContractError("S04 finalize proposal probability is invalid")
        source_micro = by_micro[source]
        target_micro = by_micro[target]
        if (
            source_micro.end_time_sec >= target_micro.start_time_sec
            or source_micro.end_global_frame >= target_micro.start_global_frame
        ):
            raise ContractError(
                "S04 finalize proposal endpoints are not strictly chronological"
            )
        key = frozenset((source, target))
        if key in by_pair:
            raise ContractError("S04 finalize has duplicate evidence for a micro pair")
        by_pair[key] = proposal
        adjacency[source].add(target)
        adjacency[target].add(source)
    return adjacency, by_pair


def _components(adjacency: dict[int, set[int]]) -> list[set[int]]:
    unseen = set(adjacency)
    result: list[set[int]] = []
    while unseen:
        first = min(unseen)
        queue = deque([first])
        component: set[int] = set()
        while queue:
            current = queue.popleft()
            if current in component:
                continue
            component.add(current)
            queue.extend(sorted(adjacency[current] - component))
        unseen -= component
        result.append(component)
    return result


def build_component_union(
    micros: Sequence[FinalizeMicro], proposals: Sequence[FinalizeProposal]
) -> ComponentUnionResult:
    """Merge every undirected proposal component after strict overlap checks."""

    by_micro = _validate_micros(micros)
    adjacency, by_pair = _validate_proposals(proposals, by_micro)
    chronological_components: list[tuple[FinalizeMicro, ...]] = []
    for component in _components(adjacency):
        ordered = tuple(
            sorted(
                (by_micro[micro_id] for micro_id in component),
                key=lambda micro: (
                    micro.start_time_sec,
                    micro.start_global_frame,
                    micro.micro_id,
                ),
            )
        )
        for previous, current in zip(ordered, ordered[1:]):
            if (
                previous.end_time_sec >= current.start_time_sec
                or previous.end_global_frame >= current.start_global_frame
            ):
                raise ContractError(
                    "S04 finalize component contains overlapping microtracklets"
                )
        chronological_components.append(ordered)
    chronological_components.sort(
        key=lambda component: (
            component[0].start_time_sec,
            component[0].start_global_frame,
            component[0].micro_id,
            tuple(micro.micro_id for micro in component),
        )
    )

    mapping_rows: list[dict[str, object]] = []
    stable_rows: list[dict[str, object]] = []
    mapping: dict[int, int] = {}
    component_ids: list[tuple[int, ...]] = []
    for stable_id, component in enumerate(chronological_components):
        component_set = {micro.micro_id for micro in component}
        evidence = sorted(
            (
                proposal
                for pair, proposal in by_pair.items()
                if pair.issubset(component_set)
            ),
            key=lambda proposal: proposal.edge_id,
        )
        probabilities = [proposal.probability for proposal in evidence]
        minimum = min(probabilities) if probabilities else None
        mean = sum(probabilities) / len(probabilities) if probabilities else None
        maximum = max(probabilities) if probabilities else None
        for order, micro in enumerate(component):
            predecessor = component[order - 1] if order else None
            direct = (
                by_pair.get(frozenset((predecessor.micro_id, micro.micro_id)))
                if predecessor is not None
                else None
            )
            mapping[micro.micro_id] = stable_id
            mapping_rows.append(
                {
                    "micro_id": micro.micro_id,
                    "stable_id": stable_id,
                    "order_in_stable": order,
                    "predecessor_micro_id": (
                        predecessor.micro_id if predecessor is not None else None
                    ),
                    "predecessor_edge_id": direct.edge_id if direct else None,
                    "predecessor_link_probability": (
                        direct.probability if direct else None
                    ),
                    "component_num_microtracklets": len(component),
                    "component_num_proposal_edges": len(evidence),
                    "component_min_proposal_probability": minimum,
                    "component_mean_proposal_probability": mean,
                    "component_max_proposal_probability": maximum,
                }
            )
        first, last = component[0], component[-1]
        stable_rows.append(
            {
                "stable_id": stable_id,
                "first_micro_id": first.micro_id,
                "last_micro_id": last.micro_id,
                "start_det_id": first.start_det_id,
                "end_det_id": last.end_det_id,
                "start_clip_id": first.start_clip_id,
                "end_clip_id": last.end_clip_id,
                "start_global_frame": first.start_global_frame,
                "end_global_frame": last.end_global_frame,
                "start_time_sec": first.start_time_sec,
                "end_time_sec": last.end_time_sec,
                "num_microtracklets": len(component),
                "num_detections": sum(micro.num_detections for micro in component),
                "num_proposal_edges": len(evidence),
                "min_proposal_probability": minimum,
                "mean_proposal_probability": mean,
                "max_proposal_probability": maximum,
                "is_singleton": len(component) == 1,
            }
        )
        component_ids.append(tuple(micro.micro_id for micro in component))
    mapping_rows.sort(key=lambda row: int(row["micro_id"]))
    if set(mapping) != set(by_micro) or set(mapping.values()) != set(
        range(len(chronological_components))
    ):
        raise ContractError("S04 finalize component mapping is not total/dense")
    return ComponentUnionResult(
        tuple(mapping_rows),
        tuple(stable_rows),
        mapping,
        tuple(component_ids),
    )


def build_det_to_stable_rows(
    *,
    det_ids: Sequence[int],
    micro_ids: Sequence[int],
    order_in_micro: Sequence[int],
    micro_to_stable_rows: Sequence[dict[str, object]],
) -> tuple[dict[str, object], ...]:
    """Create a total detection mapping in canonical stable chronology."""

    if not (len(det_ids) == len(micro_ids) == len(order_in_micro)):
        raise ContractError("S04 finalize detection columns have inconsistent lengths")
    if len(set(map(int, det_ids))) != len(det_ids):
        raise ContractError("S04 finalize detection IDs must be unique")
    mapping = {int(row["micro_id"]): row for row in micro_to_stable_rows}
    by_micro: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for det_id, micro_id, order in zip(
        det_ids, micro_ids, order_in_micro, strict=True
    ):
        micro = int(micro_id)
        if micro not in mapping:
            raise ContractError("S04 finalize detection references an unknown micro")
        by_micro[micro].append((int(order), int(det_id)))
    if set(by_micro) != set(mapping):
        raise ContractError("S04 finalize micro lacks mapped detections")
    rows: list[dict[str, object]] = []
    for micro_id in by_micro:
        values = sorted(by_micro[micro_id])
        if [order for order, _ in values] != list(range(len(values))):
            raise ContractError("S04 finalize order_in_micro is not contiguous")
    stable_detection_order: dict[int, int] = defaultdict(int)
    ordered_micros = sorted(
        mapping.values(), key=lambda row: (int(row["stable_id"]), int(row["order_in_stable"]))
    )
    for mapping_row in ordered_micros:
        micro_id = int(mapping_row["micro_id"])
        stable_id = int(mapping_row["stable_id"])
        for order, det_id in sorted(by_micro[micro_id]):
            rows.append(
                {
                    "det_id": det_id,
                    "micro_id": micro_id,
                    "stable_id": stable_id,
                    "order_in_stable": int(mapping_row["order_in_stable"]),
                    "order_in_micro": order,
                    "order_in_stable_detection": stable_detection_order[stable_id],
                }
            )
            stable_detection_order[stable_id] += 1
    rows.sort(key=lambda row: int(row["det_id"]))
    return tuple(rows)


__all__ = [
    "ComponentUnionResult",
    "FinalizeMicro",
    "FinalizeProposal",
    "build_component_union",
    "build_det_to_stable_rows",
]
