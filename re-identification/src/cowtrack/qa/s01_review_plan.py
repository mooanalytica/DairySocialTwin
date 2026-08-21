"""Deterministic, read-only case selection for S01 video review.

The public :func:`build_review_plan` API accepts already-loaded one-dimensional
NumPy columns.  It deliberately has no Parquet, video, or filesystem I/O.  In
particular, ``legacy_track_id`` is read only to select QA cases after S01 has
finished; it is never used to form, split, or otherwise change a micro-track.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, TypeAlias

import numpy as np

from cowtrack.config import ContractError


JsonScalar: TypeAlias = str | int | float | bool | None
EventFields: TypeAlias = tuple[tuple[str, JsonScalar], ...]

PLAN_SCHEMA_VERSION = "cowtrack.s01-review-plan.v1"

LONG_TRACK_MIN_DETECTIONS = 10_000
LOW_PURITY_THRESHOLD = 0.8
LARGE_JUMP_THRESHOLD = 0.1
QUALITY_MIN_DETECTIONS = 300
QUALITY_MAX_DETECTIONS = 1_800
QUALITY_MIN_PURITY = 0.99
QUALITY_MAX_JUMP = 0.03
DEFAULT_LONG_TRACK_CHUNK_SEC = 300.0
DEFAULT_LONG_TRACK_OVERLAP_SEC = 1.0

DETECTION_COLUMNS = (
    "det_id",
    "sequence_id",
    "clip_id",
    "local_frame",
    "global_frame",
    "global_time_sec",
    "cx_norm",
    "cy_norm",
    "w_norm",
    "h_norm",
    "legacy_track_id",
    "valid",
)
DET_TO_MICRO_COLUMNS = ("det_id", "micro_id", "order_in_micro")
MICROTRACKLET_COLUMNS = (
    "micro_id",
    "num_detections",
    "start_time_sec",
    "end_time_sec",
    "local_purity_score",
    "max_internal_center_jump",
    "status",
)
DET_EDGE_COLUMNS = (
    "src_det_id",
    "dst_det_id",
    "forward_cost",
    "backward_cost",
)

_REASON_ORDER = {
    "legacy_id_transition": 0,
    "large_center_jump": 1,
    "low_purity": 2,
    "long_full_review": 3,
    "quality_reference": 4,
}


@dataclass(frozen=True)
class ReviewCase:
    """One bounded video-review case.

    ``events`` preserves every triggering event when nearby triggers on the
    same micro-track are merged into one case.  Each event is represented as a
    tuple of key/value pairs so that this frozen dataclass contains no mutable
    dictionaries.
    """

    case_id: str
    case_kind: str
    micro_id: int
    anchor_det_id: int
    anchor_sequence_id: str
    anchor_clip_id: str
    anchor_local_frame: int
    anchor_global_frame: int
    anchor_global_time_sec: float
    window_start_time_sec: float
    window_end_time_sec: float
    reasons: tuple[str, ...]
    events: tuple[EventFields, ...]
    num_detections: int
    status: str
    local_purity_score: float
    max_internal_center_jump: float

    @property
    def suggested_filename(self) -> str:
        return f"{self.case_id}.mp4"

    def to_manifest_record(self) -> dict[str, Any]:
        """Return a JSON-serializable case record for a renderer manifest."""

        return {
            "case_id": self.case_id,
            "case_kind": self.case_kind,
            "suggested_filename": self.suggested_filename,
            "micro_id": self.micro_id,
            "anchor": {
                "det_id": self.anchor_det_id,
                "sequence_id": self.anchor_sequence_id,
                "clip_id": self.anchor_clip_id,
                "local_frame": self.anchor_local_frame,
                "global_frame": self.anchor_global_frame,
                "global_time_sec": self.anchor_global_time_sec,
            },
            "window": {
                "start_global_time_sec": self.window_start_time_sec,
                "end_global_time_sec": self.window_end_time_sec,
                "duration_sec": self.window_end_time_sec
                - self.window_start_time_sec,
            },
            "reasons": list(self.reasons),
            "events": [dict(fields) for fields in self.events],
            "microtrack": {
                "num_detections": self.num_detections,
                "status": self.status,
                "local_purity_score": self.local_purity_score,
                "max_internal_center_jump": self.max_internal_center_jump,
            },
        }


@dataclass(frozen=True)
class ReviewPlan:
    """Complete deterministic S01 review selection."""

    cases: tuple[ReviewCase, ...]
    window_before_sec: float
    window_after_sec: float
    merge_distance_sec: float
    long_track_min_detections: int
    long_track_chunk_sec: float
    long_track_chunk_overlap_sec: float
    low_purity_threshold: float
    high_jump_threshold: float
    quality_min_detections: int
    quality_max_detections: int
    quality_min_purity: float
    quality_max_jump: float
    requested_quality_samples: int
    eligible_quality_microtracks: int
    risk_microtracks: tuple[int, ...]

    @property
    def risk_cases(self) -> tuple[ReviewCase, ...]:
        return tuple(case for case in self.cases if case.case_kind == "risk")

    @property
    def quality_cases(self) -> tuple[ReviewCase, ...]:
        return tuple(
            case for case in self.cases if case.case_kind == "quality_reference"
        )

    def to_manifest_payload(self) -> dict[str, Any]:
        """Return the stable payload which the video renderer can extend."""

        risk_cases = self.risk_cases
        quality_cases = self.quality_cases
        return {
            "schema_version": PLAN_SCHEMA_VERSION,
            "selection_is_read_only": True,
            "legacy_track_id_usage": "qa_case_selection_only",
            "legacy_track_id_used_for_tracking": False,
            "selection_policy": {
                "window_before_sec": self.window_before_sec,
                "window_after_sec": self.window_after_sec,
                "same_micro_merge_distance_sec": self.merge_distance_sec,
                "risk": {
                    "long_track_num_detections_gte": (
                        self.long_track_min_detections
                    ),
                    "long_track_review": "complete start-to-end coverage",
                    "long_track_chunk_sec_max": self.long_track_chunk_sec,
                    "long_track_adjacent_overlap_sec": (
                        self.long_track_chunk_overlap_sec
                    ),
                    "long_track_event_handling": (
                        "merge into one containing full-review chunk"
                    ),
                    "local_purity_score_lt": self.low_purity_threshold,
                    "low_purity_status": "valid",
                    "max_internal_center_jump_gt": self.high_jump_threshold,
                    "same_clip_legacy_id_transition": "every accepted edge",
                },
                "quality_reference": {
                    "status": "valid",
                    "num_detections_inclusive": [
                        self.quality_min_detections,
                        self.quality_max_detections,
                    ],
                    "local_purity_score_gte": self.quality_min_purity,
                    "max_internal_center_jump_lte": self.quality_max_jump,
                    "exclude_entire_microtrack_if_any_risk": True,
                    "selection": "deterministic temporal-rank strata",
                    "requested_samples": self.requested_quality_samples,
                    "anchor": "detection nearest microtrack temporal midpoint",
                },
            },
            "summary": {
                "num_cases": len(self.cases),
                "num_risk_cases": len(risk_cases),
                "num_quality_reference_cases": len(quality_cases),
                "num_risk_microtracks": len(self.risk_microtracks),
                "num_quality_eligible_microtracks": (
                    self.eligible_quality_microtracks
                ),
            },
            "risk_micro_ids": list(self.risk_microtracks),
            "cases": [case.to_manifest_record() for case in self.cases],
        }


@dataclass(frozen=True)
class _DetectionView:
    det_id: np.ndarray
    sequence_id: np.ndarray
    clip_id: np.ndarray
    local_frame: np.ndarray
    global_frame: np.ndarray
    global_time_sec: np.ndarray
    cx_norm: np.ndarray
    cy_norm: np.ndarray
    w_norm: np.ndarray
    h_norm: np.ndarray
    legacy_track_id: np.ndarray
    valid: np.ndarray
    sorted_det_id: np.ndarray
    sorted_to_position: np.ndarray

    def lookup(self, det_ids: np.ndarray, *, label: str) -> np.ndarray:
        values = np.asarray(det_ids, dtype=np.int64)
        locations = np.searchsorted(self.sorted_det_id, values)
        clipped = np.minimum(locations, len(self.sorted_det_id) - 1)
        found = (locations < len(self.sorted_det_id)) & (
            self.sorted_det_id[clipped] == values
        )
        if not np.all(found):
            missing = values[np.flatnonzero(~found)[:5]].tolist()
            raise ContractError(f"{label} contains unknown det_id values: {missing}")
        return self.sorted_to_position[locations]


@dataclass(frozen=True)
class _Summary:
    micro_id: int
    num_detections: int
    start_time_sec: float
    end_time_sec: float
    local_purity_score: float
    max_internal_center_jump: float
    status: str


@dataclass(frozen=True)
class _Candidate:
    micro_id: int
    det_position: int
    reason: str
    event: EventFields


def _columns(
    source: Mapping[str, np.ndarray], required: tuple[str, ...], *, label: str
) -> dict[str, np.ndarray]:
    missing = [name for name in required if name not in source]
    if missing:
        raise ContractError(f"{label} is missing columns: {missing}")
    result = {name: np.asarray(source[name]) for name in required}
    for name, array in result.items():
        if array.ndim != 1:
            raise ContractError(f"{label}.{name} must be one-dimensional")
    lengths = {len(array) for array in result.values()}
    if len(lengths) != 1:
        raise ContractError(f"{label} columns have inconsistent lengths")
    return result


def _load_detections(source: Mapping[str, np.ndarray]) -> _DetectionView:
    columns = _columns(source, DETECTION_COLUMNS, label="detections")
    det_id = columns["det_id"].astype(np.int64, copy=False)
    if det_id.size == 0:
        raise ContractError("detections is empty")
    sorted_to_position = np.argsort(det_id, kind="stable")
    sorted_det_id = det_id[sorted_to_position]
    if np.any(np.diff(sorted_det_id) == 0):
        raise ContractError("detections.det_id must be unique")

    valid = columns["valid"].astype(np.bool_, copy=False)
    if not np.any(valid):
        raise ContractError("detections contains no valid S00 bbox")
    global_time_sec = columns["global_time_sec"].astype(np.float64, copy=False)
    if not np.all(np.isfinite(global_time_sec)):
        raise ContractError("detections.global_time_sec must be finite")
    for name in ("cx_norm", "cy_norm", "w_norm", "h_norm"):
        values = columns[name].astype(np.float64, copy=False)
        if not np.all(np.isfinite(values[valid])):
            raise ContractError(f"valid detections.{name} must be finite")
    w_norm = columns["w_norm"].astype(np.float64, copy=False)
    h_norm = columns["h_norm"].astype(np.float64, copy=False)
    if np.any(w_norm[valid] <= 0.0) or np.any(h_norm[valid] <= 0.0):
        raise ContractError("valid detections normalized bbox sizes must be positive")

    return _DetectionView(
        det_id=det_id,
        sequence_id=columns["sequence_id"],
        clip_id=columns["clip_id"],
        local_frame=columns["local_frame"].astype(np.int64, copy=False),
        global_frame=columns["global_frame"].astype(np.int64, copy=False),
        global_time_sec=global_time_sec,
        cx_norm=columns["cx_norm"].astype(np.float64, copy=False),
        cy_norm=columns["cy_norm"].astype(np.float64, copy=False),
        w_norm=w_norm,
        h_norm=h_norm,
        legacy_track_id=columns["legacy_track_id"],
        valid=valid,
        sorted_det_id=sorted_det_id,
        sorted_to_position=sorted_to_position,
    )


def _load_summaries(
    source: Mapping[str, np.ndarray],
) -> dict[int, _Summary]:
    columns = _columns(source, MICROTRACKLET_COLUMNS, label="microtracklets")
    micro_ids = columns["micro_id"].astype(np.int64, copy=False)
    if np.unique(micro_ids).size != len(micro_ids):
        raise ContractError("microtracklets.micro_id must be unique")
    numeric = (
        "start_time_sec",
        "end_time_sec",
        "local_purity_score",
        "max_internal_center_jump",
    )
    for name in numeric:
        if not np.all(np.isfinite(columns[name].astype(np.float64, copy=False))):
            raise ContractError(f"microtracklets.{name} must be finite")

    result: dict[int, _Summary] = {}
    for row, micro_value in enumerate(micro_ids):
        micro_id = int(micro_value)
        summary = _Summary(
            micro_id=micro_id,
            num_detections=int(columns["num_detections"][row]),
            start_time_sec=float(columns["start_time_sec"][row]),
            end_time_sec=float(columns["end_time_sec"][row]),
            local_purity_score=float(columns["local_purity_score"][row]),
            max_internal_center_jump=float(
                columns["max_internal_center_jump"][row]
            ),
            status=str(columns["status"][row]),
        )
        if summary.num_detections <= 0:
            raise ContractError(
                f"microtrack {micro_id} has non-positive num_detections"
            )
        if summary.end_time_sec < summary.start_time_sec:
            raise ContractError(f"microtrack {micro_id} has reversed time range")
        result[micro_id] = summary
    return result


def _load_paths(
    source: Mapping[str, np.ndarray],
    detections: _DetectionView,
    summaries: Mapping[int, _Summary],
) -> tuple[dict[int, np.ndarray], np.ndarray, np.ndarray]:
    columns = _columns(source, DET_TO_MICRO_COLUMNS, label="det_to_micro")
    det_ids = columns["det_id"].astype(np.int64, copy=False)
    if np.unique(det_ids).size != len(det_ids):
        raise ContractError("det_to_micro.det_id must be unique")
    positions = detections.lookup(det_ids, label="det_to_micro")
    if not np.all(detections.valid[positions]):
        raise ContractError("det_to_micro contains invalid S00 detections")

    valid_det_ids = np.sort(detections.det_id[detections.valid])
    if len(valid_det_ids) != len(det_ids) or not np.array_equal(
        valid_det_ids, np.sort(det_ids)
    ):
        raise ContractError(
            "det_to_micro must contain every valid detection exactly once"
        )

    micro_ids = columns["micro_id"].astype(np.int64, copy=False)
    order_in_micro = columns["order_in_micro"].astype(np.int64, copy=False)
    unknown = sorted(set(map(int, np.unique(micro_ids))) - set(summaries))
    if unknown:
        raise ContractError(f"det_to_micro contains unknown micro_ids: {unknown[:5]}")

    order = np.lexsort((det_ids, order_in_micro, micro_ids))
    sorted_micro = micro_ids[order]
    starts = np.flatnonzero(
        np.r_[True, sorted_micro[1:] != sorted_micro[:-1]]
    )
    stops = np.r_[starts[1:], len(order)]
    paths: dict[int, np.ndarray] = {}
    for start, stop in zip(starts, stops, strict=True):
        rows = order[int(start) : int(stop)]
        micro_id = int(micro_ids[rows[0]])
        actual_order = order_in_micro[rows]
        if not np.array_equal(actual_order, np.arange(len(rows), dtype=np.int64)):
            raise ContractError(
                f"microtrack {micro_id} order_in_micro is not contiguous from zero"
            )
        path = positions[rows]
        if np.any(np.diff(detections.global_frame[path]) <= 0) or np.any(
            np.diff(detections.global_time_sec[path]) <= 0.0
        ):
            raise ContractError(
                f"microtrack {micro_id} is not strictly increasing in time"
            )
        if summaries[micro_id].num_detections != len(path):
            raise ContractError(
                f"microtrack {micro_id} summary length does not match mapping"
            )
        paths[micro_id] = path

    if set(paths) != set(summaries):
        missing = sorted(set(summaries) - set(paths))
        raise ContractError(f"microtracklets missing from mapping: {missing[:5]}")
    return paths, micro_ids, order_in_micro


def _load_edges(
    source: Mapping[str, np.ndarray],
    detections: _DetectionView,
    mapping_micro_ids: np.ndarray,
    mapping_order: np.ndarray,
    det_to_micro_source: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    columns = _columns(source, DET_EDGE_COLUMNS, label="det_edges")
    src_det_id = columns["src_det_id"].astype(np.int64, copy=False)
    dst_det_id = columns["dst_det_id"].astype(np.int64, copy=False)
    accepted = np.ones(len(src_det_id), dtype=np.bool_)
    if "accepted" in source:
        raw_accepted = np.asarray(source["accepted"])
        if raw_accepted.ndim != 1 or len(raw_accepted) != len(src_det_id):
            raise ContractError("det_edges.accepted has invalid shape")
        accepted = raw_accepted.astype(np.bool_, copy=False)
    src_det_id = src_det_id[accepted]
    dst_det_id = dst_det_id[accepted]
    forward_cost = columns["forward_cost"].astype(np.float64, copy=False)[accepted]
    backward_cost = columns["backward_cost"].astype(np.float64, copy=False)[accepted]
    if not np.all(np.isfinite(forward_cost)) or not np.all(
        np.isfinite(backward_cost)
    ):
        raise ContractError("accepted det_edges costs must be finite")
    if np.any(forward_cost < 0.0) or np.any(backward_cost < 0.0):
        raise ContractError("accepted det_edges costs must be non-negative")

    expected_edges = len(mapping_micro_ids) - len(np.unique(mapping_micro_ids))
    if len(src_det_id) != expected_edges:
        raise ContractError(
            "accepted det_edges count does not form every mapped microtrack path"
        )
    if np.unique(src_det_id).size != len(src_det_id) or np.unique(
        dst_det_id
    ).size != len(dst_det_id):
        raise ContractError("accepted det_edges must be one-to-one")

    src_positions = detections.lookup(src_det_id, label="det_edges.src_det_id")
    dst_positions = detections.lookup(dst_det_id, label="det_edges.dst_det_id")

    mapping_det_ids = np.asarray(det_to_micro_source["det_id"], dtype=np.int64)
    map_sort = np.argsort(mapping_det_ids, kind="stable")
    sorted_mapping_det_ids = mapping_det_ids[map_sort]

    def mapping_rows(values: np.ndarray, label: str) -> np.ndarray:
        locations = np.searchsorted(sorted_mapping_det_ids, values)
        clipped = np.minimum(locations, len(sorted_mapping_det_ids) - 1)
        found = (locations < len(sorted_mapping_det_ids)) & (
            sorted_mapping_det_ids[clipped] == values
        )
        if not np.all(found):
            missing = values[np.flatnonzero(~found)[:5]].tolist()
            raise ContractError(f"{label} is absent from det_to_micro: {missing}")
        return map_sort[locations]

    src_rows = mapping_rows(src_det_id, "det_edges.src_det_id")
    dst_rows = mapping_rows(dst_det_id, "det_edges.dst_det_id")
    src_micro = mapping_micro_ids[src_rows]
    dst_micro = mapping_micro_ids[dst_rows]
    if not np.array_equal(src_micro, dst_micro):
        raise ContractError("accepted edge crosses microtrack boundaries")
    if not np.all(mapping_order[dst_rows] == mapping_order[src_rows] + 1):
        raise ContractError("accepted edge is not consecutive within its microtrack")

    return (
        src_positions,
        dst_positions,
        src_micro.astype(np.int64, copy=False),
        forward_cost,
        backward_cost,
    )


def _actual_center_jumps(
    detections: _DetectionView,
    src_positions: np.ndarray,
    dst_positions: np.ndarray,
) -> np.ndarray:
    distance = np.hypot(
        detections.cx_norm[dst_positions] - detections.cx_norm[src_positions],
        detections.cy_norm[dst_positions] - detections.cy_norm[src_positions],
    )
    src_diagonal = np.hypot(
        detections.w_norm[src_positions], detections.h_norm[src_positions]
    )
    dst_diagonal = np.hypot(
        detections.w_norm[dst_positions], detections.h_norm[dst_positions]
    )
    return distance / (
        0.5 * (src_diagonal + dst_diagonal) + np.finfo(np.float64).eps
    )


def _edge_rows_by_micro(edge_micro_ids: np.ndarray) -> dict[int, np.ndarray]:
    if len(edge_micro_ids) == 0:
        return {}
    order = np.argsort(edge_micro_ids, kind="stable")
    values = edge_micro_ids[order]
    starts = np.flatnonzero(np.r_[True, values[1:] != values[:-1]])
    stops = np.r_[starts[1:], len(order)]
    return {
        int(values[start]): order[int(start) : int(stop)]
        for start, stop in zip(starts, stops, strict=True)
    }


def _nearest_path_position(
    path: np.ndarray, detections: _DetectionView, target_time: float
) -> int:
    path_times = detections.global_time_sec[path]
    distance = np.abs(path_times - target_time)
    minimum = float(np.min(distance))
    tied = path[np.flatnonzero(distance == minimum)]
    tie_order = np.lexsort(
        (detections.det_id[tied], detections.global_frame[tied])
    )
    return int(tied[tie_order[0]])


def _best_edge_row(
    rows: np.ndarray,
    values: np.ndarray,
    dst_positions: np.ndarray,
    detections: _DetectionView,
) -> int:
    maximum = float(np.max(values[rows]))
    tied = rows[np.flatnonzero(values[rows] == maximum)]
    tie_order = np.lexsort(
        (
            detections.det_id[dst_positions[tied]],
            detections.global_frame[dst_positions[tied]],
        )
    )
    return int(tied[tie_order[0]])


def _event(**values: JsonScalar) -> EventFields:
    return tuple(values.items())


def _legacy_value(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, (float, np.floating)) and np.isnan(value):
        return None
    text = str(value).strip()
    return text if text else None


def _risk_candidates(
    detections: _DetectionView,
    summaries: Mapping[int, _Summary],
    paths: Mapping[int, np.ndarray],
    src_positions: np.ndarray,
    dst_positions: np.ndarray,
    edge_micro_ids: np.ndarray,
    mean_costs: np.ndarray,
    center_jumps: np.ndarray,
    *,
    low_purity_threshold: float,
    high_jump_threshold: float,
) -> list[_Candidate]:
    edge_rows = _edge_rows_by_micro(edge_micro_ids)
    candidates: list[_Candidate] = []

    for micro_id in sorted(summaries):
        summary = summaries[micro_id]
        path = paths[micro_id]
        rows = edge_rows.get(micro_id, np.empty(0, dtype=np.int64))

        if (
            summary.status == "valid"
            and summary.local_purity_score < low_purity_threshold
        ):
            if len(rows):
                row = _best_edge_row(
                    rows, mean_costs, dst_positions, detections
                )
                det_position = int(dst_positions[row])
                event = _event(
                    reason="low_purity",
                    anchor_strategy="maximum_mean_directional_edge_cost_dst",
                    src_det_id=int(detections.det_id[src_positions[row]]),
                    dst_det_id=int(detections.det_id[dst_positions[row]]),
                    mean_directional_edge_cost=float(mean_costs[row]),
                    local_purity_score=summary.local_purity_score,
                    global_frame=int(detections.global_frame[det_position]),
                    global_time_sec=float(
                        detections.global_time_sec[det_position]
                    ),
                )
            else:
                # S01 defines singleton purity as zero.  It has no edge whose
                # destination can be used, so the sole detection is the only
                # truthful review anchor.
                det_position = _nearest_path_position(
                    path,
                    detections,
                    0.5 * (summary.start_time_sec + summary.end_time_sec),
                )
                event = _event(
                    reason="low_purity",
                    anchor_strategy="only_detection_no_internal_edge",
                    local_purity_score=summary.local_purity_score,
                    global_frame=int(detections.global_frame[det_position]),
                    global_time_sec=float(
                        detections.global_time_sec[det_position]
                    ),
                )
            candidates.append(
                _Candidate(micro_id, det_position, "low_purity", event)
            )

        if summary.max_internal_center_jump > high_jump_threshold:
            if not len(rows):
                raise ContractError(
                    f"microtrack {micro_id} reports a large jump but has no edge"
                )
            row = _best_edge_row(rows, center_jumps, dst_positions, detections)
            det_position = int(dst_positions[row])
            candidates.append(
                _Candidate(
                    micro_id=micro_id,
                    det_position=det_position,
                    reason="large_center_jump",
                    event=_event(
                        reason="large_center_jump",
                        anchor_strategy="maximum_actual_center_jump_dst",
                        src_det_id=int(detections.det_id[src_positions[row]]),
                        dst_det_id=int(detections.det_id[dst_positions[row]]),
                        actual_center_jump=float(center_jumps[row]),
                        reported_max_internal_center_jump=(
                            summary.max_internal_center_jump
                        ),
                        global_frame=int(detections.global_frame[det_position]),
                        global_time_sec=float(
                            detections.global_time_sec[det_position]
                        ),
                    ),
                )
            )

    for row in range(len(src_positions)):
        source = int(src_positions[row])
        destination = int(dst_positions[row])
        same_clip = (
            str(detections.sequence_id[source])
            == str(detections.sequence_id[destination])
            and str(detections.clip_id[source])
            == str(detections.clip_id[destination])
        )
        if not same_clip:
            continue
        legacy_from = _legacy_value(detections.legacy_track_id[source])
        legacy_to = _legacy_value(detections.legacy_track_id[destination])
        if legacy_from is None or legacy_to is None or legacy_from == legacy_to:
            continue
        micro_id = int(edge_micro_ids[row])
        candidates.append(
            _Candidate(
                micro_id=micro_id,
                det_position=destination,
                reason="legacy_id_transition",
                event=_event(
                    reason="legacy_id_transition",
                    anchor_strategy="same_clip_transition_edge_dst",
                    src_det_id=int(detections.det_id[source]),
                    dst_det_id=int(detections.det_id[destination]),
                    legacy_track_id_from=legacy_from,
                    legacy_track_id_to=legacy_to,
                    clip_id=str(detections.clip_id[destination]),
                    global_frame=int(detections.global_frame[destination]),
                    global_time_sec=float(
                        detections.global_time_sec[destination]
                    ),
                ),
            )
        )
    return candidates


def _candidate_sort_key(
    candidate: _Candidate, detections: _DetectionView
) -> tuple[float, int, int, int]:
    position = candidate.det_position
    return (
        float(detections.global_time_sec[position]),
        int(detections.global_frame[position]),
        _REASON_ORDER[candidate.reason],
        int(detections.det_id[position]),
    )


def _reason_sort_key(reason: str) -> tuple[int, str]:
    return (_REASON_ORDER.get(reason, len(_REASON_ORDER) + 1), reason)


def _make_case(
    *,
    case_kind: str,
    micro_id: int,
    grouped: list[_Candidate],
    detections: _DetectionView,
    summary: _Summary,
    window_before_sec: float,
    window_after_sec: float,
    timeline_start_time_sec: float,
    timeline_end_time_sec: float,
    window_override: tuple[float, float] | None = None,
    extra_reasons: tuple[str, ...] = (),
    case_id_suffix: str = "",
    preferred_anchor_reason: str | None = None,
) -> ReviewCase:
    event_times = np.asarray(
        [detections.global_time_sec[item.det_position] for item in grouped],
        dtype=np.float64,
    )
    target = 0.5 * (float(np.min(event_times)) + float(np.max(event_times)))
    representative_pool = grouped
    if preferred_anchor_reason is not None:
        preferred = [
            item for item in grouped if item.reason == preferred_anchor_reason
        ]
        if preferred:
            representative_pool = preferred
    representative = min(
        representative_pool,
        key=lambda item: (
            abs(float(detections.global_time_sec[item.det_position]) - target),
            _REASON_ORDER[item.reason],
            int(detections.global_frame[item.det_position]),
            int(detections.det_id[item.det_position]),
        ),
    )
    position = representative.det_position
    anchor_time = float(detections.global_time_sec[position])
    prefix = "risk" if case_kind == "risk" else "quality"
    case_id = (
        f"s01_{prefix}_m{micro_id:06d}_g"
        f"{int(detections.global_frame[position]):09d}{case_id_suffix}"
    )
    reasons = tuple(
        sorted(
            {item.reason for item in grouped} | set(extra_reasons),
            key=_reason_sort_key,
        )
    )
    ordered_events = sorted(
        grouped, key=lambda item: _candidate_sort_key(item, detections)
    )
    if window_override is None:
        window_start = max(
            timeline_start_time_sec,
            float(np.min(event_times)) - window_before_sec,
        )
        window_end = min(
            timeline_end_time_sec,
            float(np.max(event_times)) + window_after_sec,
        )
    else:
        window_start = max(timeline_start_time_sec, window_override[0])
        window_end = min(timeline_end_time_sec, window_override[1])
        if window_end < window_start:
            raise ContractError("review-case window is reversed")

    return ReviewCase(
        case_id=case_id,
        case_kind=case_kind,
        micro_id=micro_id,
        anchor_det_id=int(detections.det_id[position]),
        anchor_sequence_id=str(detections.sequence_id[position]),
        anchor_clip_id=str(detections.clip_id[position]),
        anchor_local_frame=int(detections.local_frame[position]),
        anchor_global_frame=int(detections.global_frame[position]),
        anchor_global_time_sec=anchor_time,
        window_start_time_sec=window_start,
        window_end_time_sec=window_end,
        reasons=reasons,
        events=tuple(item.event for item in ordered_events),
        num_detections=summary.num_detections,
        status=summary.status,
        local_purity_score=summary.local_purity_score,
        max_internal_center_jump=summary.max_internal_center_jump,
    )


def _long_track_cases(
    detections: _DetectionView,
    summaries: Mapping[int, _Summary],
    paths: Mapping[int, np.ndarray],
    event_candidates: list[_Candidate],
    *,
    min_detections: int,
    chunk_sec: float,
    overlap_sec: float,
    window_before_sec: float,
    window_after_sec: float,
    timeline_start_time_sec: float,
    timeline_end_time_sec: float,
) -> list[ReviewCase]:
    """Partition every long micro-track into complete, overlapping coverage."""

    cases: list[ReviewCase] = []
    advance_sec = chunk_sec - overlap_sec
    events_by_micro: dict[int, list[_Candidate]] = {}
    for candidate in event_candidates:
        events_by_micro.setdefault(candidate.micro_id, []).append(candidate)
    for micro_id in sorted(summaries):
        summary = summaries[micro_id]
        if summary.num_detections < min_detections:
            continue
        start_time = summary.start_time_sec
        end_time = summary.end_time_sec
        chunk_bounds: list[tuple[float, float]] = []
        chunk_start = start_time
        while True:
            chunk_end = min(end_time, chunk_start + chunk_sec)
            chunk_bounds.append((chunk_start, chunk_end))
            if chunk_end >= end_time:
                break
            chunk_start += advance_sec

        total_chunks = len(chunk_bounds)
        attached: list[list[_Candidate]] = [[] for _ in chunk_bounds]
        for event_candidate in events_by_micro.get(micro_id, []):
            event_time = float(
                detections.global_time_sec[event_candidate.det_position]
            )
            containing = [
                index
                for index, (chunk_start, chunk_end) in enumerate(chunk_bounds)
                if chunk_start <= event_time <= chunk_end
            ]
            if not containing:
                raise ContractError(
                    f"long microtrack {micro_id} event lies outside its time range"
                )
            selected_chunk = min(
                containing,
                key=lambda index: (
                    abs(
                        event_time
                        - 0.5
                        * (chunk_bounds[index][0] + chunk_bounds[index][1])
                    ),
                    index,
                ),
            )
            attached[selected_chunk].append(event_candidate)
        path = paths[micro_id]
        for chunk_index, (chunk_start, chunk_end) in enumerate(
            chunk_bounds, start=1
        ):
            position = _nearest_path_position(
                path, detections, 0.5 * (chunk_start + chunk_end)
            )
            chunk_reason = f"chunk_{chunk_index}_of_{total_chunks}"
            candidate = _Candidate(
                micro_id=micro_id,
                det_position=position,
                reason="long_full_review",
                event=_event(
                    reason="long_full_review",
                    chunk=chunk_reason,
                    anchor_strategy="nearest_detection_to_chunk_midpoint",
                    chunk_index=chunk_index,
                    total_chunks=total_chunks,
                    chunk_start_global_time_sec=chunk_start,
                    chunk_end_global_time_sec=chunk_end,
                    num_detections=summary.num_detections,
                    global_frame=int(detections.global_frame[position]),
                    global_time_sec=float(
                        detections.global_time_sec[position]
                    ),
                ),
            )
            grouped = [candidate, *attached[chunk_index - 1]]
            cases.append(
                _make_case(
                    case_kind="risk",
                    micro_id=micro_id,
                    grouped=grouped,
                    detections=detections,
                    summary=summary,
                    window_before_sec=window_before_sec,
                    window_after_sec=window_after_sec,
                    timeline_start_time_sec=timeline_start_time_sec,
                    timeline_end_time_sec=timeline_end_time_sec,
                    window_override=(chunk_start, chunk_end),
                    extra_reasons=(chunk_reason,),
                    case_id_suffix=(
                        f"_chunk{chunk_index:03d}of{total_chunks:03d}"
                    ),
                    preferred_anchor_reason="long_full_review",
                )
            )
    return cases


def _merge_risk_candidates(
    candidates: list[_Candidate],
    detections: _DetectionView,
    summaries: Mapping[int, _Summary],
    *,
    merge_distance_sec: float,
    window_before_sec: float,
    window_after_sec: float,
    timeline_start_time_sec: float,
    timeline_end_time_sec: float,
) -> list[ReviewCase]:
    by_micro: dict[int, list[_Candidate]] = {}
    for candidate in candidates:
        by_micro.setdefault(candidate.micro_id, []).append(candidate)

    cases: list[ReviewCase] = []
    for micro_id in sorted(by_micro):
        ordered = sorted(
            by_micro[micro_id],
            key=lambda item: _candidate_sort_key(item, detections),
        )
        group: list[_Candidate] = []
        previous_time = 0.0
        for candidate in ordered:
            candidate_time = float(
                detections.global_time_sec[candidate.det_position]
            )
            if group and candidate_time - previous_time > merge_distance_sec:
                cases.append(
                    _make_case(
                        case_kind="risk",
                        micro_id=micro_id,
                        grouped=group,
                        detections=detections,
                        summary=summaries[micro_id],
                        window_before_sec=window_before_sec,
                        window_after_sec=window_after_sec,
                        timeline_start_time_sec=timeline_start_time_sec,
                        timeline_end_time_sec=timeline_end_time_sec,
                    )
                )
                group = []
            if not group:
                previous_time = candidate_time
            group.append(candidate)
            previous_time = candidate_time
        if group:
            cases.append(
                _make_case(
                    case_kind="risk",
                    micro_id=micro_id,
                    grouped=group,
                    detections=detections,
                    summary=summaries[micro_id],
                    window_before_sec=window_before_sec,
                    window_after_sec=window_after_sec,
                    timeline_start_time_sec=timeline_start_time_sec,
                    timeline_end_time_sec=timeline_end_time_sec,
                )
            )
    return cases


def _quality_micro_ids(
    summaries: Mapping[int, _Summary],
    risk_micro_ids: set[int],
    *,
    min_detections: int,
    max_detections: int,
    min_purity: float,
    max_jump: float,
) -> list[int]:
    eligible = [
        micro_id
        for micro_id, summary in summaries.items()
        if micro_id not in risk_micro_ids
        and summary.status == "valid"
        and min_detections <= summary.num_detections <= max_detections
        and summary.local_purity_score >= min_purity
        and summary.max_internal_center_jump <= max_jump
    ]
    return sorted(
        eligible,
        key=lambda micro_id: (
            0.5
            * (
                summaries[micro_id].start_time_sec
                + summaries[micro_id].end_time_sec
            ),
            micro_id,
        ),
    )


def _stratified_quality_selection(
    eligible: list[int],
    summaries: Mapping[int, _Summary],
    requested_samples: int,
) -> list[tuple[int, int, int]]:
    sample_count = min(requested_samples, len(eligible))
    if sample_count == 0:
        return []
    result: list[tuple[int, int, int]] = []
    for stratum_index, indices in enumerate(
        np.array_split(np.arange(len(eligible), dtype=np.int64), sample_count)
    ):
        micro_ids = [eligible[int(index)] for index in indices]
        midpoint_times = np.asarray(
            [
                0.5
                * (
                    summaries[micro_id].start_time_sec
                    + summaries[micro_id].end_time_sec
                )
                for micro_id in micro_ids
            ],
            dtype=np.float64,
        )
        target_time = float(np.median(midpoint_times))
        selected = min(
            micro_ids,
            key=lambda micro_id: (
                abs(
                    0.5
                    * (
                        summaries[micro_id].start_time_sec
                        + summaries[micro_id].end_time_sec
                    )
                    - target_time
                ),
                micro_id,
            ),
        )
        result.append((selected, stratum_index + 1, sample_count))
    return result


def build_review_plan(
    *,
    detections: Mapping[str, np.ndarray],
    det_to_micro: Mapping[str, np.ndarray],
    microtracklets: Mapping[str, np.ndarray],
    det_edges: Mapping[str, np.ndarray],
    window_before_sec: float = 3.0,
    window_after_sec: float = 3.0,
    merge_distance_sec: float | None = None,
    long_track_min_detections: int = LONG_TRACK_MIN_DETECTIONS,
    long_track_chunk_sec: float = DEFAULT_LONG_TRACK_CHUNK_SEC,
    long_track_chunk_overlap_sec: float = DEFAULT_LONG_TRACK_OVERLAP_SEC,
    low_purity_threshold: float = LOW_PURITY_THRESHOLD,
    high_jump_threshold: float = LARGE_JUMP_THRESHOLD,
    requested_quality_samples: int = 20,
    quality_min_detections: int = QUALITY_MIN_DETECTIONS,
    quality_max_detections: int = QUALITY_MAX_DETECTIONS,
    quality_min_purity: float = QUALITY_MIN_PURITY,
    quality_max_jump: float = QUALITY_MAX_JUMP,
    timeline_start_time_sec: float | None = None,
    timeline_end_time_sec: float | None = None,
) -> ReviewPlan:
    """Build all risk cases and deterministic high-quality comparisons.

    Parameters are column mappings, normally created by converting selected
    Arrow columns to NumPy once in the calling layer.  The four mappings must
    correspond to S00 ``detections.parquet`` and S01 ``det_to_micro.parquet``,
    ``microtracklets.parquet``, and ``det_edges.parquet`` respectively.

    Nearby triggers are merged only within the same micro-track.  By default,
    successive triggers no farther apart than the shorter side of the review
    window form one chain.  The resulting case window is the union of every
    trigger's before/after window, so no merged trigger is hidden.
    """

    for label, value in (
        ("window_before_sec", window_before_sec),
        ("window_after_sec", window_after_sec),
    ):
        if not np.isfinite(value) or value <= 0.0:
            raise ContractError(f"{label} must be finite and positive")
    if requested_quality_samples < 0:
        raise ContractError("requested_quality_samples must be non-negative")
    if long_track_min_detections < 3:
        raise ContractError("long_track_min_detections must be at least three")
    if merge_distance_sec is None:
        merge_distance_sec = min(window_before_sec, window_after_sec)
    if not np.isfinite(merge_distance_sec) or merge_distance_sec < 0.0:
        raise ContractError("merge_distance_sec must be finite and non-negative")
    if not np.isfinite(long_track_chunk_sec) or long_track_chunk_sec <= 0.0:
        raise ContractError("long_track_chunk_sec must be finite and positive")
    if (
        not np.isfinite(long_track_chunk_overlap_sec)
        or long_track_chunk_overlap_sec < 0.0
        or long_track_chunk_overlap_sec >= long_track_chunk_sec
    ):
        raise ContractError(
            "long_track_chunk_overlap_sec must be finite and in "
            "[0, long_track_chunk_sec)"
        )
    if not np.isfinite(low_purity_threshold) or not (
        0.0 < low_purity_threshold <= 1.0
    ):
        raise ContractError("low_purity_threshold must be in (0, 1]")
    if not np.isfinite(high_jump_threshold) or high_jump_threshold <= 0.0:
        raise ContractError("high_jump_threshold must be finite and positive")
    if not 3 <= quality_min_detections <= quality_max_detections:
        raise ContractError("quality detection length range is invalid")
    if not np.isfinite(quality_min_purity) or not (
        0.0 < quality_min_purity <= 1.0
    ):
        raise ContractError("quality_min_purity must be in (0, 1]")
    if not np.isfinite(quality_max_jump) or quality_max_jump <= 0.0:
        raise ContractError("quality_max_jump must be finite and positive")

    detection_view = _load_detections(detections)
    summaries = _load_summaries(microtracklets)
    paths, mapping_micro_ids, mapping_order = _load_paths(
        det_to_micro, detection_view, summaries
    )
    (
        src_positions,
        dst_positions,
        edge_micro_ids,
        forward_cost,
        backward_cost,
    ) = _load_edges(
        det_edges,
        detection_view,
        mapping_micro_ids,
        mapping_order,
        det_to_micro,
    )
    mean_costs = 0.5 * (forward_cost + backward_cost)
    center_jumps = _actual_center_jumps(
        detection_view, src_positions, dst_positions
    )

    if timeline_start_time_sec is None:
        timeline_start_time_sec = float(np.min(detection_view.global_time_sec))
    if timeline_end_time_sec is None:
        timeline_end_time_sec = float(np.max(detection_view.global_time_sec))
    if (
        not np.isfinite(timeline_start_time_sec)
        or not np.isfinite(timeline_end_time_sec)
        or timeline_end_time_sec < timeline_start_time_sec
    ):
        raise ContractError("timeline bounds must be finite and ordered")

    candidates = _risk_candidates(
        detection_view,
        summaries,
        paths,
        src_positions,
        dst_positions,
        edge_micro_ids,
        mean_costs,
        center_jumps,
        low_purity_threshold=float(low_purity_threshold),
        high_jump_threshold=float(high_jump_threshold),
    )
    long_micro_ids = {
        micro_id
        for micro_id, summary in summaries.items()
        if summary.num_detections >= long_track_min_detections
    }
    risk_micro_ids = {
        candidate.micro_id for candidate in candidates
    } | long_micro_ids
    non_long_candidates = [
        candidate
        for candidate in candidates
        if candidate.micro_id not in long_micro_ids
    ]
    long_event_candidates = [
        candidate
        for candidate in candidates
        if candidate.micro_id in long_micro_ids
    ]
    event_risk_cases = _merge_risk_candidates(
        non_long_candidates,
        detection_view,
        summaries,
        merge_distance_sec=merge_distance_sec,
        window_before_sec=window_before_sec,
        window_after_sec=window_after_sec,
        timeline_start_time_sec=timeline_start_time_sec,
        timeline_end_time_sec=timeline_end_time_sec,
    )
    long_risk_cases = _long_track_cases(
        detection_view,
        summaries,
        paths,
        long_event_candidates,
        min_detections=int(long_track_min_detections),
        chunk_sec=float(long_track_chunk_sec),
        overlap_sec=float(long_track_chunk_overlap_sec),
        window_before_sec=window_before_sec,
        window_after_sec=window_after_sec,
        timeline_start_time_sec=timeline_start_time_sec,
        timeline_end_time_sec=timeline_end_time_sec,
    )

    eligible = _quality_micro_ids(
        summaries,
        risk_micro_ids,
        min_detections=int(quality_min_detections),
        max_detections=int(quality_max_detections),
        min_purity=float(quality_min_purity),
        max_jump=float(quality_max_jump),
    )
    quality_cases: list[ReviewCase] = []
    for micro_id, stratum, total_strata in _stratified_quality_selection(
        eligible, summaries, requested_quality_samples
    ):
        summary = summaries[micro_id]
        position = _nearest_path_position(
            paths[micro_id],
            detection_view,
            0.5 * (summary.start_time_sec + summary.end_time_sec),
        )
        candidate = _Candidate(
            micro_id=micro_id,
            det_position=position,
            reason="quality_reference",
            event=_event(
                reason="quality_reference",
                anchor_strategy="nearest_detection_to_temporal_midpoint",
                temporal_stratum=stratum,
                total_temporal_strata=total_strata,
                global_frame=int(detection_view.global_frame[position]),
                global_time_sec=float(
                    detection_view.global_time_sec[position]
                ),
            ),
        )
        quality_cases.append(
            _make_case(
                case_kind="quality_reference",
                micro_id=micro_id,
                grouped=[candidate],
                detections=detection_view,
                summary=summary,
                window_before_sec=window_before_sec,
                window_after_sec=window_after_sec,
                timeline_start_time_sec=timeline_start_time_sec,
                timeline_end_time_sec=timeline_end_time_sec,
            )
        )

    all_cases = sorted(
        [*long_risk_cases, *event_risk_cases, *quality_cases],
        key=lambda case: (
            case.anchor_global_time_sec,
            0 if case.case_kind == "risk" else 1,
            case.micro_id,
            case.case_id,
        ),
    )
    return ReviewPlan(
        cases=tuple(all_cases),
        window_before_sec=float(window_before_sec),
        window_after_sec=float(window_after_sec),
        merge_distance_sec=float(merge_distance_sec),
        long_track_min_detections=int(long_track_min_detections),
        long_track_chunk_sec=float(long_track_chunk_sec),
        long_track_chunk_overlap_sec=float(long_track_chunk_overlap_sec),
        low_purity_threshold=float(low_purity_threshold),
        high_jump_threshold=float(high_jump_threshold),
        quality_min_detections=int(quality_min_detections),
        quality_max_detections=int(quality_max_detections),
        quality_min_purity=float(quality_min_purity),
        quality_max_jump=float(quality_max_jump),
        requested_quality_samples=int(requested_quality_samples),
        eligible_quality_microtracks=len(eligible),
        risk_microtracks=tuple(sorted(risk_micro_ids)),
    )


__all__ = [
    "DETECTION_COLUMNS",
    "DET_EDGE_COLUMNS",
    "DET_TO_MICRO_COLUMNS",
    "MICROTRACKLET_COLUMNS",
    "ReviewCase",
    "ReviewPlan",
    "build_review_plan",
]
