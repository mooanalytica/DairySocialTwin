"""Pure bbox-only annotation for S01 human-review videos.

The review frame contains no text, inset, crop, trajectory, marker, or panel.
All boxes are drawn in the raw 3840x2160 coordinate system and the complete
frame is then resized to 1920x1080.
"""

from __future__ import annotations

from collections.abc import Sequence
import math
from typing import TypeAlias

import cv2
import numpy as np
from numpy.typing import NDArray


RawFrame: TypeAlias = NDArray[np.uint8]
NumericSequence: TypeAlias = Sequence[float] | NDArray[np.number]

RAW_WIDTH = 3840
RAW_HEIGHT = 2160
REVIEW_WIDTH = 1920
REVIEW_HEIGHT = 1080

# OpenCV BGR colors. These are the complete visual vocabulary of the review
# video: gray=context, orange=risk target, green=quality target, red=event.
CONTEXT_BGR = (150, 150, 150)
RISK_BGR = (0, 165, 255)
QUALITY_BGR = (45, 205, 70)
EVENT_BGR = (40, 40, 235)

_ALLOWED_CASE_KINDS = frozenset(("risk", "quality_reference"))
_RAW_CONTEXT_THICKNESS = 4
_RAW_TARGET_THICKNESS = 10


def render_review_frame(
    raw_frame: RawFrame,
    *,
    output_width: int,
    output_height: int,
    context_boxes: NumericSequence | Sequence[NumericSequence],
    target_bbox: NumericSequence | None,
    case_kind: str,
    event_frame: bool,
) -> RawFrame:
    """Return a full-frame bbox-only review image without mutating the input."""

    _validate_frame(raw_frame)
    if (output_width, output_height) != (REVIEW_WIDTH, REVIEW_HEIGHT):
        raise ValueError(
            "S01 review output must be exactly "
            f"{REVIEW_WIDTH}x{REVIEW_HEIGHT}, got {output_width}x{output_height}"
        )
    if case_kind not in _ALLOWED_CASE_KINDS:
        raise ValueError(
            "case_kind must be 'risk' or 'quality_reference', "
            f"got {case_kind!r}"
        )
    if not isinstance(event_frame, bool):
        raise TypeError("event_frame must be a boolean")

    contexts = _as_rows(context_boxes, width=4, label="context_boxes")
    for index, box in enumerate(contexts):
        _validate_box(box, label=f"context_boxes[{index}]")

    target = None
    if target_bbox is not None:
        target = _as_vector(target_bbox, width=4, label="target_bbox")
        _validate_box(target, label="target_bbox")

    target_color = (
        EVENT_BGR
        if event_frame
        else RISK_BGR if case_kind == "risk" else QUALITY_BGR
    )
    annotated_raw = raw_frame.copy()
    for box in contexts:
        _draw_box(
            annotated_raw,
            box,
            color=CONTEXT_BGR,
            thickness=_RAW_CONTEXT_THICKNESS,
        )
    if target is not None:
        _draw_box(
            annotated_raw,
            target,
            color=target_color,
            thickness=_RAW_TARGET_THICKNESS,
        )
    return cv2.resize(
        annotated_raw,
        (output_width, output_height),
        interpolation=cv2.INTER_AREA,
    )


def _validate_frame(frame: RawFrame) -> None:
    if not isinstance(frame, np.ndarray):
        raise TypeError(f"raw_frame must be a NumPy array, got {type(frame)!r}")
    if frame.dtype != np.uint8:
        raise TypeError(f"raw_frame dtype must be uint8, got {frame.dtype}")
    expected = (RAW_HEIGHT, RAW_WIDTH, 3)
    if frame.shape != expected:
        raise ValueError(f"raw_frame shape must be {expected}, got {frame.shape}")


def _as_rows(
    values: NumericSequence | Sequence[NumericSequence],
    *,
    width: int,
    label: str,
) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return np.empty((0, width), dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != width:
        raise ValueError(f"{label} must have shape (N, {width}), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{label} must contain only finite coordinates")
    return array


def _as_vector(
    values: NumericSequence, *, width: int, label: str
) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (width,):
        raise ValueError(f"{label} must have shape ({width},), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{label} must contain only finite coordinates")
    return array


def _validate_box(box: NDArray[np.float64], *, label: str) -> None:
    x1, y1, x2, y2 = (float(value) for value in box)
    if not (0.0 <= x1 < x2 <= RAW_WIDTH):
        raise ValueError(
            f"{label} x coordinates must satisfy 0 <= x1 < x2 <= "
            f"{RAW_WIDTH}, got {(x1, x2)}"
        )
    if not (0.0 <= y1 < y2 <= RAW_HEIGHT):
        raise ValueError(
            f"{label} y coordinates must satisfy 0 <= y1 < y2 <= "
            f"{RAW_HEIGHT}, got {(y1, y2)}"
        )


def _pixel_box(box: NDArray[np.float64]) -> tuple[tuple[int, int], tuple[int, int]]:
    x1, y1, x2, y2 = box
    return (
        (int(math.floor(x1)), int(math.floor(y1))),
        (
            min(RAW_WIDTH - 1, int(math.ceil(x2)) - 1),
            min(RAW_HEIGHT - 1, int(math.ceil(y2)) - 1),
        ),
    )


def _draw_box(
    canvas: RawFrame,
    box: NDArray[np.float64],
    *,
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    top_left, bottom_right = _pixel_box(box)
    cv2.rectangle(
        canvas,
        top_left,
        bottom_right,
        color,
        thickness=thickness,
        lineType=cv2.LINE_8,
    )


__all__ = [
    "CONTEXT_BGR",
    "EVENT_BGR",
    "QUALITY_BGR",
    "REVIEW_HEIGHT",
    "REVIEW_WIDTH",
    "RISK_BGR",
    "render_review_frame",
]
