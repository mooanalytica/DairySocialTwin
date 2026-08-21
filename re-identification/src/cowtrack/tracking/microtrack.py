from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np

from cowtrack.config import ContractError


ProgressFn = Callable[[str, int, int], None]


@dataclass(frozen=True)
class DetectionBatch:
    """Columnar valid detections sorted by ``(global_frame, det_id)``."""

    det_id: np.ndarray
    global_frame: np.ndarray
    global_time_sec: np.ndarray
    x1: np.ndarray
    y1: np.ndarray
    x2: np.ndarray
    y2: np.ndarray
    cx_norm: np.ndarray
    cy_norm: np.ndarray
    w_norm: np.ndarray
    h_norm: np.ndarray

    def __post_init__(self) -> None:
        arrays = (
            self.det_id,
            self.global_frame,
            self.global_time_sec,
            self.x1,
            self.y1,
            self.x2,
            self.y2,
            self.cx_norm,
            self.cy_norm,
            self.w_norm,
            self.h_norm,
        )
        lengths = {len(array) for array in arrays}
        if len(lengths) != 1:
            raise ContractError("S01 detection columns have inconsistent lengths")
        if not arrays or len(self.det_id) == 0:
            raise ContractError("S01 has no valid detections")
        if np.unique(self.det_id).size != len(self.det_id):
            raise ContractError("S01 valid det_id values are not unique")
        if not np.all(np.isfinite(self.global_time_sec)):
            raise ContractError("S01 detection times must be finite")
        geometry = np.column_stack(
            (self.x1, self.y1, self.x2, self.y2, self.cx_norm, self.cy_norm, self.w_norm, self.h_norm)
        )
        if not np.all(np.isfinite(geometry)):
            raise ContractError("S01 valid detection geometry must be finite")
        if not np.all((self.x2 > self.x1) & (self.y2 > self.y1)):
            raise ContractError("S01 valid detections must have positive pixel size")
        if not np.all((self.w_norm > 0.0) & (self.h_norm > 0.0)):
            raise ContractError("S01 valid detections must have positive normalized size")
        expected_order = np.lexsort((self.det_id, self.global_frame))
        if not np.array_equal(expected_order, np.arange(len(self.det_id))):
            raise ContractError("S01 detections must be sorted by global_frame and det_id")


@dataclass(frozen=True)
class MicrotrackSettings:
    center_distance_weight: float
    iou_weight: float
    size_weight: float
    max_time_gap_sec: float
    center_distance_gate: float
    max_area_ratio: float
    min_iou: float
    alternate_center_gate: float
    ambiguity_margin: float
    velocity_history_detections: int
    grid_width: int
    grid_height: int
    motion_prior_gate_floor: float
    motion_prior_min_edges_per_cell: int


@dataclass(frozen=True)
class MotionPrior:
    count: np.ndarray
    residual_p50: np.ndarray
    residual_p95: np.ndarray
    residual_p99: np.ndarray
    scale_p50: np.ndarray
    scale_p95: np.ndarray
    scale_p99: np.ndarray


@dataclass(frozen=True)
class LinkResult:
    """Only bidirectionally accepted, high-purity edges are retained."""

    src_index: np.ndarray
    dst_index: np.ndarray
    forward_cost: np.ndarray
    backward_cost: np.ndarray
    forward_rank: np.ndarray
    backward_rank: np.ndarray
    center_residual: np.ndarray
    scale_ratio: np.ndarray

    @property
    def num_edges(self) -> int:
        return len(self.src_index)

    def edge_pairs(self) -> set[tuple[int, int]]:
        return {
            (int(src), int(dst))
            for src, dst in zip(self.src_index, self.dst_index, strict=True)
        }


@dataclass(frozen=True)
class AssignmentResult:
    micro_id: np.ndarray
    order_in_micro: np.ndarray
    incoming_edge_score: np.ndarray
    paths: tuple[np.ndarray, ...]


@dataclass(frozen=True)
class _DirectionalEdge:
    cost: float
    rank: int
    center_residual: float
    scale_ratio: float


def infer_auto_max_time_gap(frame_times: np.ndarray, multiplier: float) -> float:
    frame_times = np.asarray(frame_times, dtype=np.float64)
    if frame_times.ndim != 1 or frame_times.size < 2:
        raise ContractError("at least two frame timestamps are required for auto max gap")
    differences = np.diff(frame_times)
    if not np.all(np.isfinite(differences)) or not np.all(differences > 0.0):
        raise ContractError("frame timestamps must be finite and strictly increasing")
    if not np.isfinite(multiplier) or multiplier <= 0.0:
        raise ContractError("max time gap multiplier must be positive")
    return float(np.median(differences) * multiplier)


def _frame_groups(batch: DetectionBatch) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    frames, starts = np.unique(batch.global_frame, return_index=True)
    stops = np.append(starts[1:], len(batch.det_id))
    groups = {
        int(frame): np.arange(int(start), int(stop), dtype=np.int64)
        for frame, start, stop in zip(frames, starts, stops, strict=True)
    }
    return frames.astype(np.int64, copy=False), groups


def _state(batch: DetectionBatch, indices: np.ndarray) -> np.ndarray:
    return np.column_stack(
        (
            batch.cx_norm[indices],
            batch.cy_norm[indices],
            np.log(batch.w_norm[indices]),
            np.log(batch.h_norm[indices]),
        )
    ).astype(np.float64, copy=False)


def _predict_state(
    batch: DetectionBatch,
    history: tuple[int, ...],
    target_time: float,
    history_limit: int,
) -> np.ndarray:
    selected = np.asarray(history[-history_limit:], dtype=np.int64)
    states = _state(batch, selected)
    if len(selected) < 2:
        return states[-1].copy()
    times = batch.global_time_sec[selected].astype(np.float64, copy=False)
    centered = times - float(np.mean(times))
    denominator = float(np.dot(centered, centered))
    if denominator <= np.finfo(np.float64).eps:
        return states[-1].copy()
    slopes = centered @ states / denominator
    intercept = np.mean(states, axis=0)
    prediction = intercept + slopes * (target_time - float(np.mean(times)))
    prediction[2:] = np.clip(prediction[2:], -20.0, 2.0)
    return prediction


def _iou_from_state(
    prediction: np.ndarray,
    candidate_cx: float,
    candidate_cy: float,
    candidate_width: float,
    candidate_height: float,
) -> float:
    pred_w, pred_h = np.exp(prediction[2:])
    pred_x1 = prediction[0] - pred_w * 0.5
    pred_y1 = prediction[1] - pred_h * 0.5
    pred_x2 = prediction[0] + pred_w * 0.5
    pred_y2 = prediction[1] + pred_h * 0.5
    cand_x1 = candidate_cx - candidate_width * 0.5
    cand_y1 = candidate_cy - candidate_height * 0.5
    cand_x2 = candidate_cx + candidate_width * 0.5
    cand_y2 = candidate_cy + candidate_height * 0.5
    intersection = max(0.0, min(pred_x2, cand_x2) - max(pred_x1, cand_x1)) * max(
        0.0, min(pred_y2, cand_y2) - max(pred_y1, cand_y1)
    )
    union = pred_w * pred_h + candidate_width * candidate_height - intersection
    return intersection / union if union > 0.0 else 0.0


def _grid_cell(
    cx: float, cy: float, *, width: int, height: int
) -> tuple[int, int]:
    column = min(width - 1, max(0, int(np.floor(cx * width))))
    row = min(height - 1, max(0, int(np.floor(cy * height))))
    return row, column


def _adaptive_center_gate(
    batch: DetectionBatch,
    source_index: int,
    settings: MicrotrackSettings,
    prior: MotionPrior | None,
) -> float | None:
    if prior is None:
        return settings.center_distance_gate
    row, column = _grid_cell(
        float(batch.cx_norm[source_index]),
        float(batch.cy_norm[source_index]),
        width=settings.grid_width,
        height=settings.grid_height,
    )
    count = int(prior.count[row, column])
    if count < settings.motion_prior_min_edges_per_cell:
        return None
    p99 = float(prior.residual_p99[row, column])
    if not np.isfinite(p99):
        raise ContractError("reliable motion-prior cell has non-finite p99")
    return min(
        settings.center_distance_gate,
        max(settings.motion_prior_gate_floor, p99),
    )


def _candidate_metrics(
    batch: DetectionBatch,
    source_index: int,
    destination_index: int,
    prediction: np.ndarray,
    settings: MicrotrackSettings,
    center_gate: float,
) -> tuple[float, float, float] | None:
    pred_w, pred_h = np.exp(prediction[2:])
    candidate_cx = float(batch.cx_norm[destination_index])
    candidate_cy = float(batch.cy_norm[destination_index])
    cand_w = float(batch.w_norm[destination_index])
    cand_h = float(batch.h_norm[destination_index])
    diagonal_pred = float(np.hypot(pred_w, pred_h))
    diagonal_candidate = float(np.hypot(cand_w, cand_h))
    center_residual = float(
        np.hypot(candidate_cx - prediction[0], candidate_cy - prediction[1])
        / (0.5 * (diagonal_pred + diagonal_candidate) + np.finfo(np.float64).eps)
    )
    source_area = float(batch.w_norm[source_index] * batch.h_norm[source_index])
    candidate_area = float(batch.w_norm[destination_index] * batch.h_norm[destination_index])
    scale_ratio = max(source_area, candidate_area) / min(source_area, candidate_area)
    iou = _iou_from_state(
        prediction, candidate_cx, candidate_cy, cand_w, cand_h
    )
    if center_residual > center_gate:
        return None
    if scale_ratio > settings.max_area_ratio:
        return None
    if iou < settings.min_iou and center_residual > settings.alternate_center_gate:
        return None
    size_distance = abs(float(np.log(cand_w) - prediction[2])) + abs(
        float(np.log(cand_h) - prediction[3])
    )
    cost = (
        settings.center_distance_weight * center_residual
        + settings.iou_weight * (1.0 - iou)
        + settings.size_weight * size_distance
    )
    return float(cost), center_residual, float(scale_ratio)


def _ambiguous(sorted_costs: np.ndarray, margin: float) -> bool:
    if len(sorted_costs) < 2:
        return False
    return float(sorted_costs[1] - sorted_costs[0]) < margin


def _directional_pass(
    batch: DetectionBatch,
    settings: MicrotrackSettings,
    *,
    reverse: bool,
    prior: MotionPrior | None,
    allowed_edge_pairs: set[tuple[int, int]] | None,
    progress: ProgressFn | None,
) -> dict[tuple[int, int], _DirectionalEdge]:
    frames, groups = _frame_groups(batch)
    ordered_frames: Iterable[int]
    ordered_frames = reversed(frames.tolist()) if reverse else frames.tolist()
    frame_list = list(ordered_frames)
    histories: dict[int, tuple[int, ...]] = {}
    selected_edges: dict[tuple[int, int], _DirectionalEdge] = {}
    total_pairs = max(0, len(frame_list) - 1)

    for step, (current_frame, next_frame) in enumerate(
        zip(frame_list, frame_list[1:]), start=1
    ):
        current_indices = groups[int(current_frame)]
        next_indices = groups[int(next_frame)]
        for index in current_indices:
            histories.setdefault(int(index), (int(index),))
        if abs(int(next_frame) - int(current_frame)) != 1:
            for index in next_indices:
                histories[int(index)] = (int(index),)
            for index in current_indices:
                histories.pop(int(index), None)
            if progress is not None:
                progress("backward" if reverse else "forward", step, total_pairs)
            continue
        delta_time = abs(
            float(batch.global_time_sec[int(next_indices[0])])
            - float(batch.global_time_sec[int(current_indices[0])])
        )
        if delta_time <= 0.0 or delta_time > settings.max_time_gap_sec:
            for index in next_indices:
                histories[int(index)] = (int(index),)
            for index in current_indices:
                histories.pop(int(index), None)
            if progress is not None:
                progress("backward" if reverse else "forward", step, total_pairs)
            continue

        costs = np.full((len(current_indices), len(next_indices)), np.inf, dtype=np.float64)
        residuals = np.full_like(costs, np.nan)
        scale_ratios = np.full_like(costs, np.nan)
        for row, source in enumerate(current_indices):
            source_index = int(source)
            target_time = float(batch.global_time_sec[int(next_indices[0])])
            prediction = _predict_state(
                batch,
                histories[source_index],
                target_time,
                settings.velocity_history_detections,
            )
            for column, destination in enumerate(next_indices):
                destination_index = int(destination)
                chronological_source = destination_index if reverse else source_index
                center_gate = _adaptive_center_gate(
                    batch, chronological_source, settings, prior
                )
                if center_gate is None:
                    continue
                chronological_pair = (
                    (destination_index, source_index)
                    if reverse
                    else (source_index, destination_index)
                )
                metrics = _candidate_metrics(
                    batch,
                    source_index,
                    destination_index,
                    prediction,
                    settings,
                    center_gate,
                )
                if metrics is None:
                    continue
                costs[row, column], residuals[row, column], scale_ratios[row, column] = metrics

        row_best: dict[int, int] = {}
        row_ambiguous: set[int] = set()
        for row in range(len(current_indices)):
            finite_columns = np.flatnonzero(np.isfinite(costs[row]))
            if finite_columns.size == 0:
                continue
            ordered = sorted(
                finite_columns.tolist(),
                key=lambda column: (
                    float(costs[row, column]),
                    int(batch.det_id[int(next_indices[column])]),
                ),
            )
            row_best[row] = int(ordered[0])
            ordered_costs = np.asarray([costs[row, column] for column in ordered])
            if _ambiguous(ordered_costs, settings.ambiguity_margin):
                row_ambiguous.add(row)

        column_best: dict[int, int] = {}
        column_ambiguous: set[int] = set()
        for column in range(len(next_indices)):
            finite_rows = np.flatnonzero(np.isfinite(costs[:, column]))
            if finite_rows.size == 0:
                continue
            ordered = sorted(
                finite_rows.tolist(),
                key=lambda row: (
                    float(costs[row, column]),
                    int(batch.det_id[int(current_indices[row])]),
                ),
            )
            column_best[column] = int(ordered[0])
            ordered_costs = np.asarray([costs[row, column] for row in ordered])
            if _ambiguous(ordered_costs, settings.ambiguity_margin):
                column_ambiguous.add(column)

        linked_destinations: set[int] = set()
        for row, column in sorted(row_best.items()):
            if column_best.get(column) != row:
                continue
            if row in row_ambiguous or column in column_ambiguous:
                continue
            source_index = int(current_indices[row])
            destination_index = int(next_indices[column])
            chronological_pair = (
                (destination_index, source_index)
                if reverse
                else (source_index, destination_index)
            )
            # Non-allowed candidates still participate in best/second-best and
            # ambiguity decisions. The mask controls only which winning edges
            # may extend history, so a pruning pass cannot hide a competitor.
            if (
                allowed_edge_pairs is not None
                and chronological_pair not in allowed_edge_pairs
            ):
                continue
            history = histories[source_index]
            histories[destination_index] = (
                history + (destination_index,)
            )[-settings.velocity_history_detections :]
            linked_destinations.add(destination_index)
            selected_edges[chronological_pair] = _DirectionalEdge(
                cost=float(costs[row, column]),
                rank=1,
                center_residual=float(residuals[row, column]),
                scale_ratio=float(scale_ratios[row, column]),
            )
        for destination in next_indices:
            destination_index = int(destination)
            if destination_index not in linked_destinations:
                histories[destination_index] = (destination_index,)
        for source in current_indices:
            histories.pop(int(source), None)
        if progress is not None:
            progress("backward" if reverse else "forward", step, total_pairs)
    return selected_edges


def link_microtracks(
    batch: DetectionBatch,
    settings: MicrotrackSettings,
    *,
    prior: MotionPrior | None = None,
    allowed_edge_pairs: set[tuple[int, int]] | None = None,
    progress: ProgressFn | None = None,
) -> LinkResult:
    """Retain a fixed point of independent forward/backward associations.

    Directional histories initially contain edges that the opposite direction
    may reject. Re-running with the intersection as an acceptance mask removes
    that contamination. The mask can only shrink and never creates a new edge.
    """

    acceptance_mask = None if allowed_edge_pairs is None else set(allowed_edge_pairs)
    forward: dict[tuple[int, int], _DirectionalEdge] = {}
    backward: dict[tuple[int, int], _DirectionalEdge] = {}
    accepted_set: set[tuple[int, int]] = set()
    while True:
        forward = _directional_pass(
            batch,
            settings,
            reverse=False,
            prior=prior,
            allowed_edge_pairs=acceptance_mask,
            progress=progress,
        )
        backward = _directional_pass(
            batch,
            settings,
            reverse=True,
            prior=prior,
            allowed_edge_pairs=acceptance_mask,
            progress=progress,
        )
        accepted_set = set(forward) & set(backward)
        if acceptance_mask is not None and not accepted_set.issubset(acceptance_mask):
            raise ContractError("microtrack pruning unexpectedly created a new edge")
        if acceptance_mask == accepted_set:
            break
        if acceptance_mask is not None and progress is not None:
            progress("fixed_point", len(accepted_set), len(acceptance_mask))
        acceptance_mask = accepted_set
        # Avoid retaining two previous large directional dictionaries while
        # allocating the next fixed-point iteration.
        forward.clear()
        backward.clear()

    accepted_pairs = sorted(
        accepted_set,
        key=lambda pair: (
            int(batch.global_frame[pair[0]]),
            int(batch.det_id[pair[0]]),
            int(batch.det_id[pair[1]]),
        ),
    )
    src = np.asarray([pair[0] for pair in accepted_pairs], dtype=np.int64)
    dst = np.asarray([pair[1] for pair in accepted_pairs], dtype=np.int64)
    return LinkResult(
        src_index=src,
        dst_index=dst,
        forward_cost=np.asarray([forward[pair].cost for pair in accepted_pairs], dtype=np.float32),
        backward_cost=np.asarray([backward[pair].cost for pair in accepted_pairs], dtype=np.float32),
        forward_rank=np.asarray([forward[pair].rank for pair in accepted_pairs], dtype=np.int16),
        backward_rank=np.asarray([backward[pair].rank for pair in accepted_pairs], dtype=np.int16),
        center_residual=np.asarray(
            [
                max(
                    forward[pair].center_residual,
                    backward[pair].center_residual,
                )
                for pair in accepted_pairs
            ],
            dtype=np.float32,
        ),
        scale_ratio=np.asarray(
            [forward[pair].scale_ratio for pair in accepted_pairs], dtype=np.float32
        ),
    )


def build_motion_prior(
    batch: DetectionBatch,
    links: LinkResult,
    settings: MicrotrackSettings,
) -> MotionPrior:
    shape = (settings.grid_height, settings.grid_width)
    count = np.zeros(shape, dtype=np.int64)
    residual_cells: list[list[list[float]]] = [
        [[] for _ in range(settings.grid_width)] for _ in range(settings.grid_height)
    ]
    scale_cells: list[list[list[float]]] = [
        [[] for _ in range(settings.grid_width)] for _ in range(settings.grid_height)
    ]
    for edge_index, source in enumerate(links.src_index):
        source_index = int(source)
        row, column = _grid_cell(
            float(batch.cx_norm[source_index]),
            float(batch.cy_norm[source_index]),
            width=settings.grid_width,
            height=settings.grid_height,
        )
        residual_cells[row][column].append(float(links.center_residual[edge_index]))
        scale_cells[row][column].append(
            float(np.sqrt(batch.w_norm[source_index] * batch.h_norm[source_index]))
        )
        count[row, column] += 1

    def percentile_grid(cells: list[list[list[float]]], percentile: float) -> np.ndarray:
        result = np.full(shape, np.nan, dtype=np.float32)
        for row in range(settings.grid_height):
            for column in range(settings.grid_width):
                if cells[row][column]:
                    result[row, column] = np.float32(
                        np.percentile(cells[row][column], percentile, method="linear")
                    )
        return result

    return MotionPrior(
        count=count,
        residual_p50=percentile_grid(residual_cells, 50.0),
        residual_p95=percentile_grid(residual_cells, 95.0),
        residual_p99=percentile_grid(residual_cells, 99.0),
        scale_p50=percentile_grid(scale_cells, 50.0),
        scale_p95=percentile_grid(scale_cells, 95.0),
        scale_p99=percentile_grid(scale_cells, 99.0),
    )


def assign_microtracks(batch: DetectionBatch, links: LinkResult) -> AssignmentResult:
    num_detections = len(batch.det_id)
    predecessor = np.full(num_detections, -1, dtype=np.int64)
    successor = np.full(num_detections, -1, dtype=np.int64)
    edge_scores: dict[tuple[int, int], float] = {}
    for edge_index, (source, destination) in enumerate(
        zip(links.src_index, links.dst_index, strict=True)
    ):
        source_index = int(source)
        destination_index = int(destination)
        if successor[source_index] != -1 or predecessor[destination_index] != -1:
            raise ContractError("accepted S01 edges violate one-in/one-out invariant")
        if batch.global_frame[destination_index] != batch.global_frame[source_index] + 1:
            raise ContractError("accepted S01 edge does not join consecutive global frames")
        successor[source_index] = destination_index
        predecessor[destination_index] = source_index
        mean_cost = 0.5 * (
            float(links.forward_cost[edge_index]) + float(links.backward_cost[edge_index])
        )
        edge_scores[(source_index, destination_index)] = float(np.exp(-mean_cost))

    starts = np.flatnonzero(predecessor == -1)
    starts = np.asarray(
        sorted(
            starts.tolist(),
            key=lambda index: (
                int(batch.global_frame[index]),
                int(batch.det_id[index]),
            ),
        ),
        dtype=np.int64,
    )
    micro_id = np.full(num_detections, -1, dtype=np.int64)
    order_in_micro = np.full(num_detections, -1, dtype=np.int32)
    incoming_score = np.full(num_detections, np.nan, dtype=np.float32)
    paths: list[np.ndarray] = []
    for next_micro_id, start in enumerate(starts):
        path: list[int] = []
        current = int(start)
        while current != -1:
            if micro_id[current] != -1:
                raise ContractError("accepted S01 edge graph contains a cycle or merge")
            order = len(path)
            micro_id[current] = next_micro_id
            order_in_micro[current] = order
            path.append(current)
            next_index = int(successor[current])
            if next_index != -1:
                incoming_score[next_index] = np.float32(
                    edge_scores[(current, next_index)]
                )
            current = next_index
        paths.append(np.asarray(path, dtype=np.int64))
    if np.any(micro_id < 0) or np.any(order_in_micro < 0):
        raise ContractError("not every valid detection was assigned to a micro-tracklet")
    return AssignmentResult(
        micro_id=micro_id,
        order_in_micro=order_in_micro,
        incoming_edge_score=incoming_score,
        paths=tuple(paths),
    )
