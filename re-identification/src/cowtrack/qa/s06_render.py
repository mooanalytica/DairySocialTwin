"""Deterministic raster helpers for forced-appearance S06 QA exports.

The full-video overlay is deliberately simple: real valid S00 boxes and the
corresponding ``Gxxxx`` label only.  Coordinates are drawn on the raw,
unrotated 3840x2160 frame before the whole frame is resized to 1920x1080.
No link score, rank, status, legend, trajectory, or synthetic box is drawn.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import colorsys
import math
from typing import Any, TypeAlias

import cv2
import numpy as np
from numpy.typing import NDArray

from cowtrack.linking.dataset_contract import MAX_GLOBAL_TRACK_COUNT
from cowtrack.qa.s06_plan import (
    CONTACT_SHEET_ROLES,
    ContactSheetPlan,
    CropSource,
)


RawFrame: TypeAlias = NDArray[np.uint8]

RAW_WIDTH = 3840
RAW_HEIGHT = 2160
OVERLAY_WIDTH = 1920
OVERLAY_HEIGHT = 1080
NUM_GLOBAL_IDS = MAX_GLOBAL_TRACK_COUNT

_RAW_BOX_THICKNESS = 8
_RAW_FONT_SCALE = 1.8
_RAW_TEXT_THICKNESS = 4
_RAW_TEXT_OUTLINE_THICKNESS = 8
_FONT = cv2.FONT_HERSHEY_SIMPLEX


@dataclass(frozen=True)
class OverlayDetection:
    """Minimal joined S00/S05 row needed by the full-video renderer."""

    det_id: int
    valid: bool
    x1: float | None
    y1: float | None
    x2: float | None
    y2: float | None
    global_track_id: int | None
    display_global_id: str | None


def _positive_integer(value: Any, *, label: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise TypeError(f"{label} must be an integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{label} must be positive")
    return result


def _global_id(value: Any) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise TypeError("global_track_id must be an integer")
    result = int(value)
    if not 0 <= result < NUM_GLOBAL_IDS:
        raise ValueError(
            f"global_track_id must be in [0, {NUM_GLOBAL_IDS - 1}], got {result}"
        )
    return result


def global_id_color(global_track_id: int) -> tuple[int, int, int]:
    """Return a bright deterministic BGR color for one of the exact 62 IDs."""

    value = _global_id(global_track_id)
    # Golden-ratio hue stepping avoids adjacent IDs receiving adjacent colors.
    hue = (0.071 + value * 0.6180339887498949) % 1.0
    red, green, blue = colorsys.hsv_to_rgb(hue, 0.82, 0.96)
    return (
        int(round(blue * 255.0)),
        int(round(green * 255.0)),
        int(round(red * 255.0)),
    )


def _validate_raw_frame(raw_frame: RawFrame) -> None:
    if not isinstance(raw_frame, np.ndarray):
        raise TypeError(f"raw_frame must be a NumPy array, got {type(raw_frame)!r}")
    if raw_frame.dtype != np.uint8:
        raise TypeError(f"raw_frame dtype must be uint8, got {raw_frame.dtype}")
    expected = (RAW_HEIGHT, RAW_WIDTH, 3)
    if raw_frame.shape != expected:
        raise ValueError(
            "raw_frame must be raw unrotated 3840x2160 BGR with shape "
            f"{expected}, got {raw_frame.shape}"
        )


def _mapping_overlay_detection(
    row: Mapping[str, Any], *, index: int
) -> OverlayDetection:
    required = {
        "det_id",
        "valid",
        "x1",
        "y1",
        "x2",
        "y2",
        "global_track_id",
        "display_global_id",
    }
    missing = sorted(required - set(row))
    if missing:
        raise ValueError(
            f"detections[{index}] missing fields: {', '.join(missing)}"
        )
    return OverlayDetection(**{name: row[name] for name in required})


def _validate_overlay_detection(
    item: OverlayDetection | Mapping[str, Any], *, index: int
) -> OverlayDetection | None:
    if isinstance(item, Mapping):
        item = _mapping_overlay_detection(item, index=index)
    if not isinstance(item, OverlayDetection):
        raise TypeError(f"detections[{index}] has invalid type")
    if isinstance(item.det_id, (bool, np.bool_)) or not isinstance(
        item.det_id, (int, np.integer)
    ):
        raise TypeError(f"detections[{index}].det_id must be an integer")
    if not -(1 << 63) <= int(item.det_id) < (1 << 63):
        raise ValueError(f"detections[{index}].det_id must fit signed int64")
    if not isinstance(item.valid, (bool, np.bool_)):
        raise TypeError(f"detections[{index}].valid must be boolean")
    if not bool(item.valid):
        if item.global_track_id is not None or item.display_global_id is not None:
            raise ValueError(
                f"detections[{index}] invalid row must have null global identity"
            )
        return None
    global_id = _global_id(item.global_track_id)
    expected_display = f"G{global_id + 1:04d}"
    if item.display_global_id != expected_display:
        raise ValueError(
            f"detections[{index}].display_global_id must be {expected_display!r}"
        )
    coordinates: list[float] = []
    for name, value in zip(
        ("x1", "y1", "x2", "y2"),
        (item.x1, item.y1, item.x2, item.y2),
        strict=True,
    ):
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, float, np.integer, np.floating)
        ):
            raise TypeError(f"detections[{index}].{name} must be numeric")
        result = float(value)
        if not math.isfinite(result):
            raise ValueError(f"detections[{index}].{name} must be finite")
        coordinates.append(result)
    x1, y1, x2, y2 = coordinates
    if not (
        0.0 <= x1 < x2 <= RAW_WIDTH and 0.0 <= y1 < y2 <= RAW_HEIGHT
    ):
        raise ValueError(
            f"detections[{index}] bbox must satisfy raw 3840x2160 bounds"
        )
    return OverlayDetection(
        det_id=int(item.det_id),
        valid=True,
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
        global_track_id=global_id,
        display_global_id=expected_display,
    )


def _pixel_box(
    detection: OverlayDetection,
) -> tuple[tuple[int, int], tuple[int, int]]:
    assert detection.x1 is not None
    assert detection.y1 is not None
    assert detection.x2 is not None
    assert detection.y2 is not None
    return (
        (int(math.floor(detection.x1)), int(math.floor(detection.y1))),
        (
            min(RAW_WIDTH - 1, int(math.ceil(detection.x2)) - 1),
            min(RAW_HEIGHT - 1, int(math.ceil(detection.y2)) - 1),
        ),
    )


def _draw_label(
    canvas: RawFrame,
    *,
    label: str,
    top_left: tuple[int, int],
    color: tuple[int, int, int],
) -> None:
    (text_width, text_height), baseline = cv2.getTextSize(
        label,
        _FONT,
        _RAW_FONT_SCALE,
        _RAW_TEXT_THICKNESS,
    )
    x = min(max(top_left[0], 0), RAW_WIDTH - text_width - 1)
    above_y = top_left[1] - 12
    y = (
        above_y
        if above_y - text_height - baseline >= 0
        else min(RAW_HEIGHT - baseline - 1, top_left[1] + text_height + 12)
    )
    origin = (x, y)
    cv2.putText(
        canvas,
        label,
        origin,
        _FONT,
        _RAW_FONT_SCALE,
        (0, 0, 0),
        thickness=_RAW_TEXT_OUTLINE_THICKNESS,
        lineType=cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        label,
        origin,
        _FONT,
        _RAW_FONT_SCALE,
        color,
        thickness=_RAW_TEXT_THICKNESS,
        lineType=cv2.LINE_AA,
    )


def render_s06_overlay_frame(
    raw_frame: RawFrame,
    detections: Sequence[OverlayDetection | Mapping[str, Any]],
    *,
    output_width: int = OVERLAY_WIDTH,
    output_height: int = OVERLAY_HEIGHT,
) -> RawFrame:
    """Draw valid real boxes plus only their ``Gxxxx`` text, then resize 0.5x."""

    _validate_raw_frame(raw_frame)
    width = _positive_integer(output_width, label="output_width")
    height = _positive_integer(output_height, label="output_height")
    if (width, height) != (OVERLAY_WIDTH, OVERLAY_HEIGHT):
        raise ValueError(
            f"S06 overlay output must be {OVERLAY_WIDTH}x{OVERLAY_HEIGHT}"
        )
    if isinstance(detections, (str, bytes)) or not isinstance(
        detections, Sequence
    ):
        raise TypeError("detections must be a sequence")
    valid: list[OverlayDetection] = []
    seen_det_ids: set[int] = set()
    for index, source in enumerate(detections):
        item = _validate_overlay_detection(source, index=index)
        det_id = (
            int(source.det_id)
            if isinstance(source, OverlayDetection)
            else int(source["det_id"])
        )
        if det_id in seen_det_ids:
            raise ValueError("detections contain duplicate det_id")
        seen_det_ids.add(det_id)
        if item is not None:
            valid.append(item)
    valid.sort(
        key=lambda item: (
            int(item.global_track_id),
            item.det_id,
            float(item.x1),
            float(item.y1),
        )
    )
    annotated = raw_frame.copy()
    for item in valid:
        assert item.global_track_id is not None
        assert item.display_global_id is not None
        color = global_id_color(item.global_track_id)
        top_left, bottom_right = _pixel_box(item)
        cv2.rectangle(
            annotated,
            top_left,
            bottom_right,
            color,
            thickness=_RAW_BOX_THICKNESS,
            lineType=cv2.LINE_8,
        )
        _draw_label(
            annotated,
            label=item.display_global_id,
            top_left=top_left,
            color=color,
        )
    return cv2.resize(
        annotated,
        (OVERLAY_WIDTH, OVERLAY_HEIGHT),
        interpolation=cv2.INTER_AREA,
    )


def extract_detection_crop(
    raw_frame: RawFrame,
    source: CropSource,
    *,
    padding_fraction: float = 0.05,
) -> RawFrame:
    """Extract one real bbox crop; padding never changes upstream coordinates."""

    _validate_raw_frame(raw_frame)
    if not isinstance(source, CropSource):
        raise TypeError("source must be a CropSource")
    if isinstance(padding_fraction, bool) or not isinstance(
        padding_fraction, (int, float)
    ):
        raise TypeError("padding_fraction must be numeric")
    padding = float(padding_fraction)
    if not math.isfinite(padding) or not 0.0 <= padding <= 0.5:
        raise ValueError("padding_fraction must be finite in [0, 0.5]")
    values = (source.x1, source.y1, source.x2, source.y2)
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("crop source coordinates must be finite")
    if not (
        0.0 <= source.x1 < source.x2 <= RAW_WIDTH
        and 0.0 <= source.y1 < source.y2 <= RAW_HEIGHT
    ):
        raise ValueError("crop source bbox must satisfy raw 3840x2160 bounds")
    pad_x = (source.x2 - source.x1) * padding
    pad_y = (source.y2 - source.y1) * padding
    left = max(0, int(math.floor(source.x1 - pad_x)))
    top = max(0, int(math.floor(source.y1 - pad_y)))
    right = min(RAW_WIDTH, int(math.ceil(source.x2 + pad_x)))
    bottom = min(RAW_HEIGHT, int(math.ceil(source.y2 + pad_y)))
    if left >= right or top >= bottom:
        raise ValueError("crop source produces an empty crop")
    return np.ascontiguousarray(raw_frame[top:bottom, left:right].copy())


def _validate_crop(crop: RawFrame, *, det_id: int) -> RawFrame:
    if not isinstance(crop, np.ndarray):
        raise TypeError(f"crop for det_id={det_id} must be a NumPy array")
    if crop.dtype != np.uint8:
        raise TypeError(f"crop for det_id={det_id} must have uint8 dtype")
    if crop.ndim != 3 or crop.shape[2] != 3 or min(crop.shape[:2]) <= 0:
        raise ValueError(
            f"crop for det_id={det_id} must have nonempty shape (H, W, 3)"
        )
    return crop


def _fit_crop(crop: RawFrame, *, width: int, height: int) -> RawFrame:
    scale = min(width / crop.shape[1], height / crop.shape[0])
    resized_width = max(1, min(width, int(round(crop.shape[1] * scale))))
    resized_height = max(1, min(height, int(round(crop.shape[0] * scale))))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(
        crop,
        (resized_width, resized_height),
        interpolation=interpolation,
    )
    canvas = np.full((height, width, 3), 24, dtype=np.uint8)
    left = (width - resized_width) // 2
    top = (height - resized_height) // 2
    canvas[top : top + resized_height, left : left + resized_width] = resized
    return canvas


def _draw_lines(
    canvas: RawFrame,
    lines: Sequence[str],
    *,
    x: int,
    first_baseline_y: int,
    color: tuple[int, int, int],
    scale: float,
    thickness: int,
    line_height: int,
) -> None:
    for offset, line in enumerate(lines):
        cv2.putText(
            canvas,
            line,
            (x, first_baseline_y + offset * line_height),
            _FONT,
            scale,
            color,
            thickness=thickness,
            lineType=cv2.LINE_AA,
        )


def _bounded_annotation(value: str, *, maximum: int = 66) -> str:
    compact = " ".join(value.split())
    return compact if len(compact) <= maximum else compact[: maximum - 3] + "..."


def render_contact_sheet(
    plan: ContactSheetPlan,
    crops_by_det_id: Mapping[int, RawFrame],
    *,
    tile_width: int = 480,
    tile_height: int = 340,
    columns: int = 4,
    header_height: int = 92,
) -> RawFrame:
    """Render fixed deterministic slots; unavailable/deduped slots stay blank."""

    if not isinstance(plan, ContactSheetPlan):
        raise TypeError("plan must be a ContactSheetPlan")
    if tuple(slot.role for slot in plan.slots) != CONTACT_SHEET_ROLES:
        raise ValueError("contact sheet plan has noncanonical roles")
    if len(plan.slots) != len(CONTACT_SHEET_ROLES):
        raise ValueError("contact sheet plan must contain seven fixed slots")
    if not isinstance(crops_by_det_id, Mapping):
        raise TypeError("crops_by_det_id must be a mapping")
    tile_w = _positive_integer(tile_width, label="tile_width")
    tile_h = _positive_integer(tile_height, label="tile_height")
    num_columns = _positive_integer(columns, label="columns")
    header = _positive_integer(header_height, label="header_height")
    if num_columns > len(CONTACT_SHEET_ROLES):
        raise ValueError("columns cannot exceed the seven contact-sheet slots")
    rows = math.ceil(len(plan.slots) / num_columns)
    canvas = np.full(
        (header + rows * tile_h, num_columns * tile_w, 3),
        18,
        dtype=np.uint8,
    )
    color = global_id_color(plan.global_track_id)
    title = f"{plan.display_global_id} | {plan.id_status}"
    _draw_lines(
        canvas,
        (title, "real S00 crops; blank slots are never duplicated"),
        x=18,
        first_baseline_y=34,
        color=color,
        scale=0.82,
        thickness=2,
        line_height=34,
    )
    used_det_ids: set[int] = set()
    caption_height = 76
    for slot_index, slot in enumerate(plan.slots):
        row, column = divmod(slot_index, num_columns)
        left = column * tile_w
        top = header + row * tile_h
        image_height = tile_h - caption_height
        tile = np.full((tile_h, tile_w, 3), 24, dtype=np.uint8)
        if slot.source is None:
            center_text = (
                "DEDUPLICATED - BLANK"
                if slot.deduplicated_to_role is not None
                else "NO DISTINCT CROP"
            )
            (text_width, _), _ = cv2.getTextSize(center_text, _FONT, 0.62, 1)
            cv2.putText(
                tile,
                center_text,
                (max(8, (tile_w - text_width) // 2), max(28, image_height // 2)),
                _FONT,
                0.62,
                (125, 125, 125),
                thickness=1,
                lineType=cv2.LINE_AA,
            )
        else:
            det_id = slot.source.det_id
            if det_id in used_det_ids:
                raise ValueError("contact sheet plan duplicates a real crop")
            used_det_ids.add(det_id)
            crop = crops_by_det_id.get(det_id)
            if crop is not None:
                tile[:image_height] = _fit_crop(
                    _validate_crop(crop, det_id=det_id),
                    width=tile_w,
                    height=image_height,
                )
            else:
                missing = "MISSING DECODED CROP"
                (text_width, _), _ = cv2.getTextSize(missing, _FONT, 0.62, 1)
                cv2.putText(
                    tile,
                    missing,
                    (max(8, (tile_w - text_width) // 2), max(28, image_height // 2)),
                    _FONT,
                    0.62,
                    (70, 70, 220),
                    thickness=1,
                    lineType=cv2.LINE_AA,
                )
        cv2.rectangle(
            tile,
            (0, 0),
            (tile_w - 1, tile_h - 1),
            color,
            thickness=2,
            lineType=cv2.LINE_8,
        )
        time_text = (
            "time=n/a"
            if slot.global_time_sec is None
            else f"time={slot.global_time_sec:.3f}s"
        )
        _draw_lines(
            tile,
            (
                slot.role,
                f"{time_text} | {_bounded_annotation(slot.annotation)}",
            ),
            x=10,
            first_baseline_y=image_height + 27,
            color=(235, 235, 235),
            scale=0.54,
            thickness=1,
            line_height=27,
        )
        canvas[top : top + tile_h, left : left + tile_w] = tile
    return canvas


__all__ = [
    "NUM_GLOBAL_IDS",
    "OVERLAY_HEIGHT",
    "OVERLAY_WIDTH",
    "RAW_HEIGHT",
    "RAW_WIDTH",
    "OverlayDetection",
    "extract_detection_crop",
    "global_id_color",
    "render_contact_sheet",
    "render_s06_overlay_frame",
]
