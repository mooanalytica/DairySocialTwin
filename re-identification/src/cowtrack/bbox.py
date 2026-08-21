from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import xxhash

from cowtrack.schemas.detections import BBoxQAFlag, INVALID_BBOX_MASK


@dataclass(frozen=True)
class BoxGeometry:
    x1: float
    y1: float
    x2: float
    y2: float
    cx_norm: float
    cy_norm: float
    w_norm: float
    h_norm: float
    area_norm: float
    qa_flags: int
    valid: bool


def stable_det_id(
    sequence_id: str, clip_id: str, local_frame: int, csv_row_index: int
) -> int:
    canonical = f"{sequence_id}|{clip_id}|{local_frame}|{csv_row_index}"
    unsigned = xxhash.xxh64(canonical.encode("utf-8"), seed=0).intdigest()
    return unsigned if unsigned < (1 << 63) else unsigned - (1 << 64)


def normalize_xywh(
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    frame_width: int,
    frame_height: int,
    minimum_area: float,
    minimum_retained_fraction: float,
) -> BoxGeometry:
    flags = BBoxQAFlag(0)
    values = (x, y, width, height)
    if not all(math.isfinite(value) for value in values):
        flags |= BBoxQAFlag.NONFINITE_NUMERIC
        nan = float("nan")
        return BoxGeometry(
            nan,
            nan,
            nan,
            nan,
            nan,
            nan,
            nan,
            nan,
            nan,
            int(flags),
            False,
        )

    raw_x1 = float(x)
    raw_y1 = float(y)
    raw_x2 = float(x + width)
    raw_y2 = float(y + height)
    if width <= 0.0 or height <= 0.0:
        flags |= BBoxQAFlag.NONPOSITIVE_SOURCE_SIZE

    x1 = min(max(raw_x1, 0.0), float(frame_width))
    y1 = min(max(raw_y1, 0.0), float(frame_height))
    x2 = min(max(raw_x2, 0.0), float(frame_width))
    y2 = min(max(raw_y2, 0.0), float(frame_height))
    if any(
        before != after
        for before, after in zip((raw_x1, raw_y1, raw_x2, raw_y2), (x1, y1, x2, y2))
    ):
        flags |= BBoxQAFlag.CLAMPED_TO_IMAGE

    clamped_width = max(0.0, x2 - x1)
    clamped_height = max(0.0, y2 - y1)
    clamped_area = clamped_width * clamped_height
    if x2 <= x1 or y2 <= y1:
        flags |= BBoxQAFlag.EMPTY_AFTER_CLAMP
    if clamped_area < minimum_area:
        flags |= BBoxQAFlag.AREA_BELOW_MINIMUM

    raw_area = width * height if width > 0.0 and height > 0.0 else 0.0
    retained_fraction = clamped_area / raw_area if raw_area > 0.0 else 0.0
    if raw_area > 0.0 and retained_fraction < minimum_retained_fraction:
        flags |= BBoxQAFlag.MOSTLY_OUTSIDE

    cx_norm = ((x1 + x2) * 0.5) / frame_width
    cy_norm = ((y1 + y2) * 0.5) / frame_height
    w_norm = clamped_width / frame_width
    h_norm = clamped_height / frame_height
    area_norm = clamped_area / (frame_width * frame_height)
    valid = (int(flags) & INVALID_BBOX_MASK) == 0
    return BoxGeometry(
        x1,
        y1,
        x2,
        y2,
        cx_norm,
        cy_norm,
        w_norm,
        h_norm,
        area_norm,
        int(flags),
        valid,
    )


def intersection_over_union(a: dict[str, Any], b: dict[str, Any]) -> float:
    ix1 = max(float(a["x1"]), float(b["x1"]))
    iy1 = max(float(a["y1"]), float(b["y1"]))
    ix2 = min(float(a["x2"]), float(b["x2"]))
    iy2 = min(float(a["y2"]), float(b["y2"]))
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, float(a["x2"]) - float(a["x1"])) * max(
        0.0, float(a["y2"]) - float(a["y1"])
    )
    area_b = max(0.0, float(b["x2"]) - float(b["x1"])) * max(
        0.0, float(b["y2"]) - float(b["y1"])
    )
    union = area_a + area_b - intersection
    return intersection / union if union > 0.0 else 0.0


def suppress_duplicate_boxes(
    frame_rows: list[dict[str, Any]],
    *,
    iou_threshold: float,
    minimum_area_similarity: float,
) -> int:
    """Greedy deterministic same-frame suppression while retaining every row."""

    candidates = [index for index, row in enumerate(frame_rows) if bool(row["valid"])]
    candidates.sort(
        key=lambda index: (
            -float(frame_rows[index]["bbox_confidence"])
            if frame_rows[index]["bbox_confidence"] is not None
            else math.inf,
            int(frame_rows[index]["csv_row_index"]),
        )
    )
    kept: list[int] = []
    suppressed = 0
    for candidate_index in candidates:
        candidate = frame_rows[candidate_index]
        candidate_area = (float(candidate["x2"]) - float(candidate["x1"])) * (
            float(candidate["y2"]) - float(candidate["y1"])
        )
        duplicate = False
        for kept_index in kept:
            existing = frame_rows[kept_index]
            existing_area = (float(existing["x2"]) - float(existing["x1"])) * (
                float(existing["y2"]) - float(existing["y1"])
            )
            similarity = min(candidate_area, existing_area) / max(
                candidate_area, existing_area
            )
            if similarity < minimum_area_similarity:
                continue
            if intersection_over_union(candidate, existing) >= iou_threshold:
                duplicate = True
                break
        if duplicate:
            candidate["qa_flags"] = int(candidate["qa_flags"]) | int(
                BBoxQAFlag.HIGH_IOU_DUPLICATE
            )
            candidate["valid"] = False
            suppressed += 1
        else:
            kept.append(candidate_index)
    return suppressed
