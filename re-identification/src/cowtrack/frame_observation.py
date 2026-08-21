"""Source-frame observation semantics shared by ingest and final QA.

A video frame is *observed* only when its source bbox CSV contains at least
one row for that frame.  A frame without such a row is not evidence of an
empty scene: it is unobserved and outside the re-identification scope.  The
source video frame itself is still preserved by the full-video renderer.
"""

from __future__ import annotations

from collections.abc import Iterable
from numbers import Integral
from typing import Any

from cowtrack.config import ContractError


OBSERVED_FRAME_DEFINITION = "at_least_one_source_bbox_csv_row"
UNOBSERVED_FRAME_INTERPRETATION = (
    "outside_reidentification_scope_not_evidence_of_empty_scene"
)
UNOBSERVED_FRAME_RENDER_POLICY = "preserve_source_frame_without_identity_overlay"


def summarize_frame_observation(
    num_frames: int,
    observed_frames: Iterable[int],
) -> dict[str, Any]:
    """Return a deterministic, JSON-ready summary of observed source frames.

    Intervals are inclusive and use the source video's zero-based local frame
    index.  Duplicate observed indices are harmless because multiple source
    detection rows can belong to the same frame.
    """

    if isinstance(num_frames, bool) or not isinstance(num_frames, Integral):
        raise ContractError("frame observation num_frames must be an integer")
    frame_count = int(num_frames)
    if frame_count < 1:
        raise ContractError("frame observation num_frames must be positive")

    normalized: set[int] = set()
    for value in observed_frames:
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise ContractError("observed frame indices must be integers")
        frame = int(value)
        if not 0 <= frame < frame_count:
            raise ContractError(
                f"observed frame index {frame} is outside [0, {frame_count})"
            )
        normalized.add(frame)

    intervals: list[dict[str, int]] = []
    next_possible_start = 0
    for frame in sorted(normalized):
        if frame > next_possible_start:
            start = next_possible_start
            end = frame - 1
            intervals.append(
                {
                    "start_frame": start,
                    "end_frame": end,
                    "num_frames": end - start + 1,
                }
            )
        next_possible_start = frame + 1
    if next_possible_start < frame_count:
        start = next_possible_start
        end = frame_count - 1
        intervals.append(
            {
                "start_frame": start,
                "end_frame": end,
                "num_frames": end - start + 1,
            }
        )

    observed_count = len(normalized)
    unobserved_count = frame_count - observed_count
    longest = max((item["num_frames"] for item in intervals), default=0)
    return {
        "num_frames": frame_count,
        "num_observed_frames": observed_count,
        "num_unobserved_frames": unobserved_count,
        "observed_frame_fraction": observed_count / frame_count,
        "num_unobserved_intervals": len(intervals),
        "longest_unobserved_interval_frames": longest,
        "unobserved_frame_intervals": intervals,
    }


def frame_observation_policy() -> dict[str, Any]:
    """Return the stable machine-readable interpretation used by all datasets."""

    return {
        "observed_frame_definition": OBSERVED_FRAME_DEFINITION,
        "unobserved_frame_interpretation": UNOBSERVED_FRAME_INTERPRETATION,
        "unobserved_frames_are_empty_scene_evidence": False,
        "reidentification_scope": "source_bbox_csv_rows_only",
        "full_video_render_policy": UNOBSERVED_FRAME_RENDER_POLICY,
        "local_frame_index_base": 0,
        "interval_endpoints": "inclusive",
    }


__all__ = [
    "OBSERVED_FRAME_DEFINITION",
    "UNOBSERVED_FRAME_INTERPRETATION",
    "UNOBSERVED_FRAME_RENDER_POLICY",
    "frame_observation_policy",
    "summarize_frame_observation",
]
