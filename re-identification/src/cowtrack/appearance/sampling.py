"""Representative-detection selection for S02.

The selector only consumes static detection metadata.  Red-window detections
are supplied through ``appearance_excluded`` and can never be selected.
"""

from __future__ import annotations

import math

import numpy as np

from cowtrack.config import ContractError


def _one_dimensional(name: str, values: np.ndarray, length: int | None = None) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ContractError(f"S02 {name} must be one-dimensional")
    if length is not None and len(array) != length:
        raise ContractError(f"S02 {name} has an inconsistent length")
    return array


def _validate_inputs(
    micro_ids: np.ndarray,
    global_time_sec: np.ndarray,
    global_frames: np.ndarray,
    other_bbox_max_iou: np.ndarray,
    appearance_excluded: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    micro_ids = _one_dimensional("micro_ids", micro_ids)
    length = len(micro_ids)
    global_time_sec = _one_dimensional("global_time_sec", global_time_sec, length)
    global_frames = _one_dimensional("global_frames", global_frames, length)
    other_bbox_max_iou = _one_dimensional(
        "other_bbox_max_iou", other_bbox_max_iou, length
    )
    appearance_excluded = _one_dimensional(
        "appearance_excluded", appearance_excluded, length
    )
    if length == 0:
        return (
            micro_ids.astype(np.int64, copy=False),
            global_time_sec.astype(np.float64, copy=False),
            global_frames.astype(np.int64, copy=False),
            other_bbox_max_iou.astype(np.float64, copy=False),
            appearance_excluded.astype(np.bool_, copy=False),
        )
    if micro_ids.dtype.kind not in "iu" or global_frames.dtype.kind not in "iu":
        raise ContractError("S02 micro_ids and global_frames must be integer arrays")
    if appearance_excluded.dtype.kind != "b":
        raise ContractError("S02 appearance_excluded must be a boolean array")
    times = global_time_sec.astype(np.float64, copy=False)
    overlaps = other_bbox_max_iou.astype(np.float64, copy=False)
    if not np.all(np.isfinite(times)):
        raise ContractError("S02 representative-sample times must be finite")
    if not np.all(np.isfinite(overlaps)) or not np.all(
        (overlaps >= 0.0) & (overlaps <= 1.0)
    ):
        raise ContractError("S02 other_bbox_max_iou must be finite and in [0, 1]")
    return (
        micro_ids.astype(np.int64, copy=False),
        times,
        global_frames.astype(np.int64, copy=False),
        overlaps,
        appearance_excluded.astype(np.bool_, copy=False),
    )


def _evenly_spaced_subset(indices: list[int], count: int) -> list[int]:
    """Retain temporal coverage without relying on floating-point rounding."""

    if count >= len(indices):
        return indices
    if count <= 0:
        return []
    if count == 1:
        return [indices[(len(indices) - 1) // 2]]
    last = len(indices) - 1
    denominator = count - 1
    positions = [
        (position * last * 2 + denominator) // (2 * denominator)
        for position in range(count)
    ]
    if len(set(positions)) != count:
        raise ContractError("S02 internal representative-sampling collision")
    return [indices[position] for position in positions]


def _select_one_microtrack(
    indices: np.ndarray,
    *,
    times: np.ndarray,
    frames: np.ndarray,
    overlaps: np.ndarray,
    period_sec: float,
    max_samples: int,
    endpoint_samples: int,
    preferred_max_iou: float,
) -> list[int]:
    order = np.lexsort((indices, frames[indices], times[indices]))
    ordered = indices[order]
    ordered_times = times[ordered]
    ordered_frames = frames[ordered]
    if len(ordered) > 1:
        if np.any(np.diff(ordered_times) < 0.0):
            raise ContractError("S02 internal microtrack time ordering failed")
        if np.unique(ordered_frames).size != len(ordered_frames):
            raise ContractError("S02 microtrack contains duplicate global_frame values")

    head = ordered[:endpoint_samples].tolist()
    tail = ordered[max(0, len(ordered) - endpoint_samples) :].tolist()
    endpoints = list(dict.fromkeys([*head, *tail]))
    if len(endpoints) > max_samples:
        raise ContractError("S02 max_samples cannot hold the required endpoint samples")
    if len(ordered) <= len(endpoints):
        return endpoints

    selected = set(endpoints)
    middle: list[int] = []
    start = float(ordered_times[0])
    stop = float(ordered_times[-1])
    anchor = start + period_sec
    half_period = period_sec * 0.5
    # Integer-like loop bounds avoid a malformed input causing an unbounded loop.
    anchor_count = max(0, int(math.ceil((stop - start) / period_sec)) - 1)
    for _ in range(anchor_count):
        if anchor >= stop:
            break
        left = anchor - half_period
        right = anchor + half_period
        left_index = int(np.searchsorted(ordered_times, left, side="left"))
        right_index = int(np.searchsorted(ordered_times, right, side="left"))
        slot = ordered[left_index:right_index]
        candidates = np.asarray(
            [int(index) for index in slot if int(index) not in selected],
            dtype=np.int64,
        )
        if len(candidates):
            # A crop under the preferred overlap threshold wins first.  Within
            # that class, lower overlap wins, followed by temporal proximity and
            # stable frame/input-row tie breaks.
            best = min(
                (int(index) for index in candidates),
                key=lambda index: (
                    float(overlaps[index]) > preferred_max_iou,
                    float(overlaps[index]),
                    abs(float(times[index]) - anchor),
                    int(frames[index]),
                    index,
                ),
            )
            selected.add(best)
            middle.append(best)
        anchor += period_sec

    capacity = max_samples - len(endpoints)
    middle.sort(key=lambda index: (float(times[index]), int(frames[index]), index))
    middle = _evenly_spaced_subset(middle, capacity)
    return [*endpoints, *middle]


def select_representative_indices(
    micro_ids: np.ndarray,
    global_time_sec: np.ndarray,
    global_frames: np.ndarray,
    other_bbox_max_iou: np.ndarray,
    appearance_excluded: np.ndarray,
    *,
    period_sec: float = 0.75,
    max_samples: int = 24,
    endpoint_samples: int = 3,
    preferred_max_iou: float = 0.25,
) -> np.ndarray:
    """Return deterministic global row indices for representative crops.

    Every non-empty microtrack retains its first and last ``endpoint_samples``
    eligible rows.  Interior time slots are centered every ``period_sec`` and
    prefer low-overlap detections.  If the limit is reached, interior slots are
    downsampled uniformly while endpoints remain intact.
    """

    (
        micro_ids,
        times,
        frames,
        overlaps,
        excluded,
    ) = _validate_inputs(
        micro_ids,
        global_time_sec,
        global_frames,
        other_bbox_max_iou,
        appearance_excluded,
    )
    if not math.isfinite(period_sec) or period_sec <= 0.0:
        raise ContractError("S02 representative period_sec must be positive")
    if isinstance(max_samples, bool) or not isinstance(max_samples, (int, np.integer)):
        raise ContractError("S02 max_samples must be an integer")
    if isinstance(endpoint_samples, bool) or not isinstance(
        endpoint_samples, (int, np.integer)
    ):
        raise ContractError("S02 endpoint_samples must be an integer")
    if endpoint_samples != 3:
        raise ContractError("S02 endpoint_samples is fixed at 3")
    if max_samples != 24:
        raise ContractError("S02 max_samples is fixed at 24")
    if not math.isfinite(preferred_max_iou) or not 0.0 <= preferred_max_iou <= 1.0:
        raise ContractError("S02 preferred_max_iou must be in [0, 1]")
    if len(micro_ids) == 0:
        return np.empty(0, dtype=np.int64)

    eligible = ~excluded
    selected: list[int] = []
    eligible_indices = np.flatnonzero(eligible).astype(np.int64, copy=False)
    if not len(eligible_indices):
        return np.empty(0, dtype=np.int64)
    micro_order = np.argsort(micro_ids[eligible_indices], kind="stable")
    grouped_indices = eligible_indices[micro_order]
    grouped_micro_ids = micro_ids[grouped_indices]
    starts = np.flatnonzero(
        np.r_[True, grouped_micro_ids[1:] != grouped_micro_ids[:-1]]
    )
    stops = np.r_[starts[1:], len(grouped_indices)]
    for start, stop in zip(starts, stops, strict=True):
        indices = grouped_indices[int(start) : int(stop)]
        selected.extend(
            _select_one_microtrack(
                indices,
                times=times,
                frames=frames,
                overlaps=overlaps,
                period_sec=float(period_sec),
                max_samples=int(max_samples),
                endpoint_samples=int(endpoint_samples),
                preferred_max_iou=float(preferred_max_iou),
            )
        )

    if not selected:
        return np.empty(0, dtype=np.int64)
    result = np.asarray(sorted(set(selected)), dtype=np.int64)
    order = np.lexsort(
        (result, micro_ids[result], times[result], frames[result])
    )
    result = result[order]
    if np.any(excluded[result]):
        raise ContractError("S02 excluded detection escaped representative sampling")
    return result


def select_candidate_pool_indices(
    micro_ids: np.ndarray,
    global_time_sec: np.ndarray,
    global_frames: np.ndarray,
    other_bbox_max_iou: np.ndarray,
    appearance_excluded: np.ndarray,
    *,
    period_sec: float = 0.375,
    max_samples: int = 48,
    endpoint_samples: int = 6,
    preferred_max_iou: float = 0.25,
) -> np.ndarray:
    """Select a fixed 2x pool so post-decode quality rejection has replacements.

    This is not an output sample contract. It intentionally over-samples the
    temporal slots once, before the single sequential video pass. The final
    representative selector is applied again after blur and crop quality are
    known, restoring the public 3/3, 0.75-second, 24-sample contract.
    """

    (
        micro_ids,
        times,
        frames,
        overlaps,
        excluded,
    ) = _validate_inputs(
        micro_ids,
        global_time_sec,
        global_frames,
        other_bbox_max_iou,
        appearance_excluded,
    )
    if not math.isclose(float(period_sec), 0.375, rel_tol=0.0, abs_tol=1e-12):
        raise ContractError("S02 candidate-pool period_sec is fixed at 0.375")
    if int(max_samples) != 48 or int(endpoint_samples) != 6:
        raise ContractError("S02 candidate pool is fixed at 6 endpoints and 48 samples")
    if not math.isfinite(preferred_max_iou) or not 0.0 <= preferred_max_iou <= 1.0:
        raise ContractError("S02 candidate-pool preferred_max_iou must be in [0, 1]")
    if len(micro_ids) == 0:
        return np.empty(0, dtype=np.int64)

    eligible = ~excluded
    selected: list[int] = []
    grouped_order = np.argsort(micro_ids, kind="stable")
    grouped_micro = micro_ids[grouped_order]
    starts = np.flatnonzero(np.r_[True, grouped_micro[1:] != grouped_micro[:-1]])
    stops = np.r_[starts[1:], len(grouped_order)]
    for start, stop in zip(starts, stops, strict=True):
        group = grouped_order[int(start) : int(stop)]
        indices = group[eligible[group]].astype(np.int64, copy=False)
        if not len(indices):
            continue
        selected.extend(
            _select_one_microtrack(
                indices,
                times=times,
                frames=frames,
                overlaps=overlaps,
                period_sec=float(period_sec),
                max_samples=int(max_samples),
                endpoint_samples=int(endpoint_samples),
                preferred_max_iou=float(preferred_max_iou),
            )
        )
    if not selected:
        return np.empty(0, dtype=np.int64)
    result = np.asarray(sorted(set(selected)), dtype=np.int64)
    order = np.lexsort((result, micro_ids[result], times[result], frames[result]))
    result = result[order]
    if np.any(excluded[result]):
        raise ContractError("S02 excluded detection escaped candidate-pool sampling")
    return result


__all__ = ["select_candidate_pool_indices", "select_representative_indices"]
