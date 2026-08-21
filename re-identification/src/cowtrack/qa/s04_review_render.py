"""BBox-only raster annotation for S04 proposal review.

There is intentionally no text-rendering API in this module.  It draws only
the unmodified S00 boxes on a copy of the raw, no-autorotate landscape frame,
then resizes that whole frame to the configured landscape output size.
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

# OpenCV BGR values.  These four box colors are the entire visual vocabulary.
UNRELATED_BGR = (150, 150, 150)
SOURCE_BGR = (0, 255, 255)  # yellow
TARGET_BGR = (0, 205, 0)  # green
INTERMEDIATE_BGR = (205, 0, 205)  # purple

_CONTEXT_THICKNESS = 4
_FOCAL_THICKNESS = 10
_INTERMEDIATE_THICKNESS = 8


def render_s04_review_frame(
    raw_frame: RawFrame,
    *,
    output_width: int,
    output_height: int,
    unrelated_boxes: NumericSequence | Sequence[NumericSequence],
    source_boxes: NumericSequence | Sequence[NumericSequence],
    target_boxes: NumericSequence | Sequence[NumericSequence],
    intermediate_boxes: NumericSequence | Sequence[NumericSequence],
) -> RawFrame:
    """Return one original-frame-plus-bboxes image without mutating the input."""

    raw_width, raw_height = _validate_frame(raw_frame)
    if (
        isinstance(output_width, bool)
        or isinstance(output_height, bool)
        or not isinstance(output_width, int)
        or not isinstance(output_height, int)
        or output_width <= 0
        or output_height <= 0
    ):
        raise ValueError("output dimensions must be positive integers")
    if output_width * raw_height != output_height * raw_width:
        raise ValueError("output dimensions must preserve the raw landscape aspect ratio")

    roles = (
        (
            "unrelated_boxes",
            unrelated_boxes,
            UNRELATED_BGR,
            _CONTEXT_THICKNESS,
        ),
        (
            "intermediate_boxes",
            intermediate_boxes,
            INTERMEDIATE_BGR,
            _INTERMEDIATE_THICKNESS,
        ),
        ("source_boxes", source_boxes, SOURCE_BGR, _FOCAL_THICKNESS),
        ("target_boxes", target_boxes, TARGET_BGR, _FOCAL_THICKNESS),
    )
    annotated = raw_frame.copy()
    for label, values, color, thickness in roles:
        boxes = _as_boxes(values, label=label)
        for index, box in enumerate(boxes):
            _validate_box(
                box,
                width=raw_width,
                height=raw_height,
                label=f"{label}[{index}]",
            )
            _draw_box(
                annotated,
                box,
                width=raw_width,
                height=raw_height,
                color=color,
                thickness=thickness,
            )
    if (output_width, output_height) == (raw_width, raw_height):
        return annotated
    return cv2.resize(
        annotated,
        (output_width, output_height),
        interpolation=cv2.INTER_AREA,
    )


def _validate_frame(frame: RawFrame) -> tuple[int, int]:
    if not isinstance(frame, np.ndarray):
        raise TypeError(f"raw_frame must be a NumPy array, got {type(frame)!r}")
    if frame.dtype != np.uint8:
        raise TypeError(f"raw_frame dtype must be uint8, got {frame.dtype}")
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"raw_frame must have shape (height, width, 3), got {frame.shape}")
    height, width = frame.shape[:2]
    if width <= height or width * 9 != height * 16:
        raise ValueError("raw_frame must use the unrotated 16:9 landscape geometry")
    return width, height


def _as_boxes(
    values: NumericSequence | Sequence[NumericSequence], *, label: str
) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return np.empty((0, 4), dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 4:
        raise ValueError(f"{label} must have shape (N, 4), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{label} must contain only finite coordinates")
    return array


def _validate_box(
    box: NDArray[np.float64], *, width: int, height: int, label: str
) -> None:
    x1, y1, x2, y2 = map(float, box)
    if not (0.0 <= x1 < x2 <= width):
        raise ValueError(
            f"{label} x coordinates must satisfy 0 <= x1 < x2 <= {width}, "
            f"got {(x1, x2)}"
        )
    if not (0.0 <= y1 < y2 <= height):
        raise ValueError(
            f"{label} y coordinates must satisfy 0 <= y1 < y2 <= {height}, "
            f"got {(y1, y2)}"
        )


def _draw_box(
    canvas: RawFrame,
    box: NDArray[np.float64],
    *,
    width: int,
    height: int,
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    x1, y1, x2, y2 = box
    top_left = (int(math.floor(x1)), int(math.floor(y1)))
    bottom_right = (
        min(width - 1, int(math.ceil(x2)) - 1),
        min(height - 1, int(math.ceil(y2)) - 1),
    )
    cv2.rectangle(
        canvas,
        top_left,
        bottom_right,
        color,
        thickness=thickness,
        lineType=cv2.LINE_8,
    )


__all__ = [
    "INTERMEDIATE_BGR",
    "SOURCE_BGR",
    "TARGET_BGR",
    "UNRELATED_BGR",
    "render_s04_review_frame",
]
