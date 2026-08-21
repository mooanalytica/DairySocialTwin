"""Deterministic structural enumeration and scoring for S04 short proposals.

This module has no solver and no stable-track representation.  A provisional
score is only a review proposal; it is never interpreted as an accepted edge.
"""

from __future__ import annotations

import bisect
import hashlib
import json
import math
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Protocol, Sequence

from cowtrack.config import ContractError
from cowtrack.linking.dataset_contract import EXPECTED_CLIP_ORDER
from cowtrack.linking.features import SHORT_FEATURE_SCHEMA


@dataclass(frozen=True)
class MicroEndpoint:
    micro_id: int
    start_det_id: int
    end_det_id: int
    start_clip_id: str
    end_clip_id: str
    start_global_frame: int
    end_global_frame: int
    start_time_sec: float
    end_time_sec: float


class ShortScorer(Protocol):
    def score_pair(
        self, source_tracklet_id: int, target_tracklet_id: int, mode: str
    ) -> Any: ...


def endpoints_from_calibration_input(data: Any) -> list[MicroEndpoint]:
    """Extract whole-micro endpoint provenance from the public runtime input."""

    required = (
        "det_ids",
        "det_micro_ids",
        "det_order_in_micro",
        "det_clip_ids",
        "det_global_frames",
        "det_global_time_sec",
    )
    if any(not hasattr(data, name) for name in required):
        raise ContractError("S04 production input lacks endpoint columns")
    columns = {name: list(getattr(data, name)) for name in required}
    lengths = {len(values) for values in columns.values()}
    if len(lengths) != 1:
        raise ContractError("S04 production endpoint columns have inconsistent lengths")
    by_micro: dict[int, list[int]] = defaultdict(list)
    for position, micro_id in enumerate(columns["det_micro_ids"]):
        by_micro[int(micro_id)].append(position)
    endpoints: list[MicroEndpoint] = []
    for micro_id in sorted(by_micro):
        positions = sorted(
            by_micro[micro_id],
            key=lambda position: (
                int(columns["det_order_in_micro"][position]),
                int(columns["det_ids"][position]),
            ),
        )
        orders = [int(columns["det_order_in_micro"][position]) for position in positions]
        if orders != list(range(len(orders))):
            raise ContractError(f"S04 micro {micro_id} order_in_micro is not contiguous")
        first, last = positions[0], positions[-1]
        endpoints.append(
            MicroEndpoint(
                micro_id=micro_id,
                start_det_id=int(columns["det_ids"][first]),
                end_det_id=int(columns["det_ids"][last]),
                start_clip_id=str(columns["det_clip_ids"][first]),
                end_clip_id=str(columns["det_clip_ids"][last]),
                start_global_frame=int(columns["det_global_frames"][first]),
                end_global_frame=int(columns["det_global_frames"][last]),
                start_time_sec=float(columns["det_global_time_sec"][first]),
                end_time_sec=float(columns["det_global_time_sec"][last]),
            )
        )
    parent_ids = getattr(data, "parent_micro_ids", None)
    if parent_ids is not None and set(map(int, parent_ids)) != set(by_micro):
        raise ContractError("S04 parent summary and detection micro ID sets differ")
    return endpoints


_GALLERY_SUFFIXES = (
    "sample_ids",
    "gallery_det_ids",
    "embedding_rows",
    "medoid_sample_id",
    "appearance_quality",
    "internal_cosine_p10",
    "internal_cosine_p50",
    "internal_cosine_min",
    "gallery_num_input_samples",
    "gallery_num_overlap_rejected",
    "gallery_num_review_excluded",
    "gallery_num_local_outliers",
    "gallery_max_other_bbox_iou",
    "gallery_max_clean_other_bbox_iou",
)


def _stable_id(prefix: str, payload: Mapping[str, Any]) -> str:
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return f"{prefix}_{hashlib.sha256(canonical).hexdigest()[:24]}"


def _validate_endpoints(
    endpoints: Sequence[MicroEndpoint], clip_order: Sequence[str]
) -> list[MicroEndpoint]:
    clips = tuple(clip_order)
    if len(clips) != len(set(clips)) or not clips:
        raise ContractError("S04 clip order must contain unique clip IDs")
    clip_rank = {clip: rank for rank, clip in enumerate(clips)}
    observed_ids: set[int] = set()
    validated: list[MicroEndpoint] = []
    for endpoint in endpoints:
        if not isinstance(endpoint, MicroEndpoint):
            raise ContractError("S04 endpoints must be MicroEndpoint values")
        if endpoint.micro_id in observed_ids:
            raise ContractError("S04 micro endpoint IDs must be unique")
        observed_ids.add(endpoint.micro_id)
        if endpoint.start_clip_id not in clip_rank or endpoint.end_clip_id not in clip_rank:
            raise ContractError("S04 endpoint references a clip outside fixed clip order")
        if clip_rank[endpoint.start_clip_id] > clip_rank[endpoint.end_clip_id]:
            raise ContractError("S04 endpoint clip order runs backwards")
        numeric = (endpoint.start_time_sec, endpoint.end_time_sec)
        if not all(math.isfinite(value) for value in numeric):
            raise ContractError("S04 endpoint times must be finite")
        # S00 detection IDs are signed int64 hashes. Either sign is valid;
        # unlike a frame index, a negative det_id is not a sentinel.
        if (
            endpoint.start_global_frame < 0
            or endpoint.end_global_frame < endpoint.start_global_frame
            or endpoint.end_time_sec < endpoint.start_time_sec
        ):
            raise ContractError("S04 endpoint provenance is inconsistent")
        validated.append(endpoint)
    return validated


def _candidate_row(source: MicroEndpoint, target: MicroEndpoint) -> dict[str, Any]:
    gap = float(target.start_time_sec - source.end_time_sec)
    identity = {
        "source_micro_id": source.micro_id,
        "source_end_det_id": source.end_det_id,
        "target_micro_id": target.micro_id,
        "target_start_det_id": target.start_det_id,
    }
    return {
        "edge_id": _stable_id("s04e", identity),
        "source_micro_id": source.micro_id,
        "target_micro_id": target.micro_id,
        "source_start_det_id": source.start_det_id,
        "source_end_det_id": source.end_det_id,
        "target_start_det_id": target.start_det_id,
        "target_end_det_id": target.end_det_id,
        "source_start_clip_id": source.start_clip_id,
        "source_end_clip_id": source.end_clip_id,
        "target_start_clip_id": target.start_clip_id,
        "target_end_clip_id": target.end_clip_id,
        "source_start_global_frame": source.start_global_frame,
        "source_end_global_frame": source.end_global_frame,
        "target_start_global_frame": target.start_global_frame,
        "target_end_global_frame": target.end_global_frame,
        "source_start_time_sec": source.start_time_sec,
        "source_end_time_sec": source.end_time_sec,
        "target_start_time_sec": target.start_time_sec,
        "target_end_time_sec": target.end_time_sec,
        "gap_sec": gap,
    }


def enumerate_short_candidates(
    endpoints: Sequence[MicroEndpoint],
    *,
    max_gap_sec: float = 5.0,
    clip_order: Sequence[str] = EXPECTED_CLIP_ORDER,
) -> list[dict[str, Any]]:
    """Enumerate ``0 < gap <= max_gap`` candidates with a time window.

    The only pair filters are source/target inequality, strict positive global
    time gap, maximum global time gap, and non-reversing fixed clip order.
    """

    if not math.isfinite(max_gap_sec) or max_gap_sec <= 0.0:
        raise ContractError("S04 maximum gap must be positive and finite")
    values = _validate_endpoints(endpoints, clip_order)
    clip_rank = {clip: rank for rank, clip in enumerate(clip_order)}
    sources = sorted(values, key=lambda item: (item.end_time_sec, item.micro_id))
    end_times = [item.end_time_sec for item in sources]
    rows: list[dict[str, Any]] = []
    for target in sorted(values, key=lambda item: (item.start_time_sec, item.micro_id)):
        lower = bisect.bisect_left(end_times, target.start_time_sec - max_gap_sec)
        upper = bisect.bisect_left(end_times, target.start_time_sec)
        for source in sources[lower:upper]:
            if source.micro_id == target.micro_id:
                continue
            if clip_rank[source.end_clip_id] > clip_rank[target.start_clip_id]:
                continue
            row = _candidate_row(source, target)
            if not 0.0 < row["gap_sec"] <= max_gap_sec:
                raise ContractError("S04 time-window candidate escaped structural bounds")
            rows.append(row)
    rows.sort(key=lambda row: str(row["edge_id"]))
    if len({row["edge_id"] for row in rows}) != len(rows):
        raise ContractError("S04 deterministic edge IDs are not unique")
    return rows


def brute_force_short_candidates(
    endpoints: Sequence[MicroEndpoint],
    *,
    max_gap_sec: float = 5.0,
    clip_order: Sequence[str] = EXPECTED_CLIP_ORDER,
) -> list[dict[str, Any]]:
    """Reference implementation used to prove the time-window equivalence."""

    values = _validate_endpoints(endpoints, clip_order)
    clip_rank = {clip: rank for rank, clip in enumerate(clip_order)}
    rows = []
    for source in values:
        for target in values:
            gap = target.start_time_sec - source.end_time_sec
            if (
                source.micro_id != target.micro_id
                and 0.0 < gap <= max_gap_sec
                and clip_rank[source.end_clip_id] <= clip_rank[target.start_clip_id]
            ):
                rows.append(_candidate_row(source, target))
    return sorted(rows, key=lambda row: str(row["edge_id"]))


def _empty_gallery_row(prefix: str) -> dict[str, Any]:
    return {f"{prefix}_{suffix}": None for suffix in _GALLERY_SUFFIXES}


def _gallery_row(value: Any, prefix: str) -> dict[str, Any]:
    if value is None:
        return _empty_gallery_row(prefix)
    if hasattr(value, "prefixed_row"):
        raw = value.prefixed_row(prefix)
    elif isinstance(value, Mapping):
        raw = {
            (key if str(key).startswith(f"{prefix}_") else f"{prefix}_{key}"): item
            for key, item in value.items()
        }
    else:
        raw = {
            f"{prefix}_{suffix}": getattr(value, suffix)
            for suffix in _GALLERY_SUFFIXES
            if hasattr(value, suffix)
        }
    expected = {f"{prefix}_{suffix}" for suffix in _GALLERY_SUFFIXES}
    unknown = set(raw) - expected
    if unknown:
        raise ContractError(f"S04 gallery provenance has unknown fields: {sorted(unknown)}")
    return {name: raw.get(name) for name in sorted(expected)}


def _gallery_assessment(result: Any, prefix: str) -> tuple[bool, str | None, Any]:
    assessment = getattr(result, f"{prefix}_gallery", None)
    if assessment is None:
        # A scorer result must expose the public side-local audit assessment,
        # including for a missing gallery.
        raise ContractError(f"S04 scorer result lacks {prefix} gallery assessment")
    present = getattr(assessment, "present", None)
    reason = getattr(assessment, "reason", None)
    provenance = getattr(assessment, "provenance", None)
    if not isinstance(present, bool):
        raise ContractError(f"S04 {prefix} gallery presence must be boolean")
    if reason is not None and (not isinstance(reason, str) or not reason):
        raise ContractError(f"S04 {prefix} gallery reason must be null or non-blank")
    if present != (provenance is not None):
        raise ContractError(f"S04 {prefix} gallery provenance/presence differs")
    if present and reason is not None:
        raise ContractError(f"S04 present {prefix} gallery cannot have a missing reason")
    if not present and reason is None:
        raise ContractError(f"S04 missing {prefix} gallery requires a reason")
    return present, reason, provenance


def _finite_optional(value: Any, *, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"S04 scorer {name} must be numeric or null")
    result = float(value)
    if not math.isfinite(result):
        raise ContractError(f"S04 scorer {name} must be finite")
    return result


def score_short_candidates(
    candidates: Sequence[Mapping[str, Any]],
    scorer: ShortScorer,
    *,
    progress: Any | None = None,
    progress_interval_sec: float = 10.0,
) -> list[dict[str, Any]]:
    """Score every structural candidate through only the calibrated short API."""

    if not math.isfinite(progress_interval_sec) or progress_interval_sec <= 0.0:
        raise ContractError("S04 scoring progress interval must be positive")
    rows: list[dict[str, Any]] = []
    ordered = sorted(candidates, key=lambda row: str(row["edge_id"]))
    last_progress = time.monotonic()
    for completed, candidate in enumerate(ordered, start=1):
        result = scorer.score_pair(
            int(candidate["source_micro_id"]),
            int(candidate["target_micro_id"]),
            "short",
        )
        decision = getattr(result, "decision", None)
        if decision not in {"reject", "provisional", "confirmed"}:
            raise ContractError("S04 scorer returned an unsupported decision")
        probability = _finite_optional(
            getattr(result, "probability", None), name="probability"
        )
        raw_score = _finite_optional(getattr(result, "raw_score", None), name="raw_score")
        if probability is not None and not 0.0 <= probability <= 1.0:
            raise ContractError("S04 scorer probability must be in [0, 1]")
        appearance_present = getattr(result, "appearance_present", None)
        if not isinstance(appearance_present, bool):
            raise ContractError("S04 scorer appearance_present must be boolean")
        high_overlap = getattr(result, "high_overlap", None)
        if not isinstance(high_overlap, bool):
            raise ContractError("S04 scorer result must expose boolean high_overlap")
        feature_values = getattr(result, "feature_values", None)
        if feature_values is None:
            feature_values = {}
        if not isinstance(feature_values, Mapping):
            raise ContractError("S04 scorer feature_values must be a mapping")
        if feature_values:
            if set(feature_values) != set(SHORT_FEATURE_SCHEMA):
                raise ContractError("S04 short scorer feature schema differs")
            parsed_features = {
                name: _finite_optional(feature_values[name], name=f"feature {name}")
                for name in SHORT_FEATURE_SCHEMA
            }
            if any(value is None for value in parsed_features.values()):
                raise ContractError("S04 scored short features cannot be null")
            if not math.isclose(
                float(parsed_features["gap_sec"]),
                float(candidate["gap_sec"]),
                rel_tol=0.0,
                abs_tol=1e-6,
            ):
                raise ContractError("S04 scorer and structural candidate gap differ")
        else:
            parsed_features = {name: None for name in SHORT_FEATURE_SCHEMA}
        if feature_values and (
            not appearance_present or probability is None or raw_score is None
        ):
            raise ContractError("S04 scored candidate has incomplete model output")
        if not feature_values and (probability is not None or raw_score is not None):
            raise ContractError("S04 unscored candidate contains model output")
        if decision != "reject" and (probability is None or not appearance_present):
            raise ContractError("S04 non-reject candidate lacks a valid score")
        if high_overlap and decision == "confirmed":
            raise ContractError("S04 high-overlap candidate cannot be confirmed")

        reason = getattr(result, "reason", None)
        if reason is not None and (not isinstance(reason, str) or not reason):
            raise ContractError("S04 scorer reason must be null or non-blank")
        source_present, source_reason, source_provenance = _gallery_assessment(
            result, "source"
        )
        target_present, target_reason, target_provenance = _gallery_assessment(
            result, "target"
        )
        motion_history_present = (
            False
            if reason == "source_motion_history_missing"
            else True
            if feature_values
            else None
        )
        row = dict(candidate)
        row.update(_empty_gallery_row("source"))
        row.update(_empty_gallery_row("target"))
        row.update(
            _gallery_row(source_provenance, "source")
        )
        row.update(
            _gallery_row(target_provenance, "target")
        )
        row.update(parsed_features)
        # One exact non-null value serves both the structural gap and the
        # fixed short-feature position. The physical Arrow field is nullable
        # to match the immutable production artifact; row validation keeps
        # every structural candidate value mandatory.
        row["gap_sec"] = float(candidate["gap_sec"])
        row.update(
            {
                "appearance_present": appearance_present,
                "high_overlap": high_overlap,
                "motion_history_present": motion_history_present,
                "source_gallery_present": source_present,
                "source_gallery_reason": source_reason,
                "target_gallery_present": target_present,
                "target_gallery_reason": target_reason,
                "probability": probability,
                "raw_score": raw_score,
                "decision": decision,
                "decision_reason": reason
                or (
                    "below_provisional_threshold"
                    if decision == "reject"
                    else "meets_provisional_threshold"
                    if decision == "provisional"
                    else "meets_confirmed_threshold"
                ),
                "proposed_for_review": decision == "provisional",
            }
        )
        rows.append(row)
        now = time.monotonic()
        if progress is not None and (
            now - last_progress >= progress_interval_sec or completed == len(ordered)
        ):
            progress(completed, len(ordered))
            last_progress = now
    return rows


def _component_metadata(
    proposals: Sequence[Mapping[str, Any]],
) -> dict[str, tuple[str, int, int]]:
    node_edges: dict[tuple[str, int], set[str]] = defaultdict(set)
    by_edge: dict[str, Mapping[str, Any]] = {}
    for row in proposals:
        edge = str(row["edge_id"])
        by_edge[edge] = row
        node_edges[("source", int(row["source_micro_id"]))].add(edge)
        node_edges[("target", int(row["target_micro_id"]))].add(edge)
    unseen = set(by_edge)
    result: dict[str, tuple[str, int, int]] = {}
    while unseen:
        first = min(unseen)
        queue = deque([first])
        component_edges: set[str] = set()
        component_nodes: set[tuple[str, int]] = set()
        while queue:
            edge = queue.popleft()
            if edge in component_edges:
                continue
            component_edges.add(edge)
            row = by_edge[edge]
            nodes = {
                ("source", int(row["source_micro_id"])),
                ("target", int(row["target_micro_id"])),
            }
            component_nodes.update(nodes)
            for node in nodes:
                queue.extend(sorted(node_edges[node] - component_edges))
        unseen -= component_edges
        group_id = _stable_id("s04g", {"edge_ids": sorted(component_edges)})
        metadata = (group_id, len(component_edges), len(component_nodes))
        result.update({edge: metadata for edge in component_edges})
    return result


def build_proposals(scored_candidates: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Build rank/conflict metadata for provisional review proposals only."""

    provisional = [
        row
        for row in scored_candidates
        if row.get("decision") == "provisional"
        and row.get("proposed_for_review") is True
    ]
    if any(row.get("probability") is None for row in provisional):
        raise ContractError("S04 provisional candidate lacks probability")
    source_groups: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    target_groups: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in provisional:
        source_groups[int(row["source_micro_id"])].append(row)
        target_groups[int(row["target_micro_id"])].append(row)
    rank_key = lambda row: (-float(row["probability"]), str(row["edge_id"]))
    source_rank = {
        str(row["edge_id"]): rank
        for group in source_groups.values()
        for rank, row in enumerate(sorted(group, key=rank_key), start=1)
    }
    target_rank = {
        str(row["edge_id"]): rank
        for group in target_groups.values()
        for rank, row in enumerate(sorted(group, key=rank_key), start=1)
    }
    components = _component_metadata(provisional)
    proposals: list[dict[str, Any]] = []
    for row in provisional:
        edge = str(row["edge_id"])
        group_id, edge_count, node_count = components[edge]
        same_source = {str(item["edge_id"]) for item in source_groups[int(row["source_micro_id"])]}
        same_target = {str(item["edge_id"]) for item in target_groups[int(row["target_micro_id"])]}
        conflict_degree = len((same_source | same_target) - {edge})
        proposals.append(
            {
                "proposal_id": _stable_id("s04p", {"edge_id": edge}),
                "edge_id": edge,
                "source_micro_id": int(row["source_micro_id"]),
                "target_micro_id": int(row["target_micro_id"]),
                "probability": float(row["probability"]),
                "high_overlap": bool(row["high_overlap"]),
                "source_rank": source_rank[edge],
                "target_rank": target_rank[edge],
                "conflict_degree": conflict_degree,
                "conflict_group_id": group_id,
                "conflict_group_edge_count": edge_count,
                "conflict_group_node_count": node_count,
                "review_status": "pending",
            }
        )
    proposals.sort(key=lambda row: str(row["proposal_id"]))
    if len({row["proposal_id"] for row in proposals}) != len(proposals):
        raise ContractError("S04 deterministic proposal IDs are not unique")
    return proposals


def review_manifest(proposals: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = [dict(row) for row in proposals]
    rows.sort(key=lambda row: str(row["proposal_id"]))
    return {
        "schema_version": "1.0",
        "stage": "S04_PROPOSE",
        "execution_mode": "proposal_only",
        "automatic_merge_allowed": False,
        "num_proposals": len(rows),
        "proposals": rows,
    }


__all__ = [
    "MicroEndpoint",
    "ShortScorer",
    "brute_force_short_candidates",
    "build_proposals",
    "enumerate_short_candidates",
    "endpoints_from_calibration_input",
    "review_manifest",
    "score_short_candidates",
]
