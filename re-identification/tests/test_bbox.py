from __future__ import annotations

import math

from cowtrack.bbox import normalize_xywh, stable_det_id, suppress_duplicate_boxes
from cowtrack.schemas.detections import BBoxQAFlag


def test_normalize_flags_and_half_open_clamp() -> None:
    normal = normalize_xywh(
        10.0,
        20.0,
        30.0,
        40.0,
        frame_width=100,
        frame_height=100,
        minimum_area=256.0,
        minimum_retained_fraction=0.10,
    )
    assert normal.valid
    assert (normal.x1, normal.y1, normal.x2, normal.y2) == (10.0, 20.0, 40.0, 60.0)
    assert normal.qa_flags == 0

    clamped = normalize_xywh(
        -5.0,
        10.0,
        20.0,
        20.0,
        frame_width=100,
        frame_height=100,
        minimum_area=1.0,
        minimum_retained_fraction=0.10,
    )
    assert clamped.valid
    assert clamped.x1 == 0.0
    assert clamped.qa_flags == int(BBoxQAFlag.CLAMPED_TO_IMAGE)

    mostly_outside = normalize_xywh(
        -95.0,
        0.0,
        100.0,
        100.0,
        frame_width=100,
        frame_height=100,
        minimum_area=1.0,
        minimum_retained_fraction=0.10,
    )
    assert not mostly_outside.valid
    assert mostly_outside.qa_flags & int(BBoxQAFlag.MOSTLY_OUTSIDE)

    nonfinite = normalize_xywh(
        math.nan,
        0.0,
        10.0,
        10.0,
        frame_width=100,
        frame_height=100,
        minimum_area=1.0,
        minimum_retained_fraction=0.10,
    )
    assert not nonfinite.valid
    assert nonfinite.qa_flags & int(BBoxQAFlag.NONFINITE_NUMERIC)
    assert math.isnan(nonfinite.x1)


def test_duplicate_suppression_is_score_then_row_deterministic() -> None:
    rows = [
        {
            "x1": 0.0,
            "y1": 0.0,
            "x2": 10.0,
            "y2": 10.0,
            "bbox_confidence": 0.8,
            "csv_row_index": 0,
            "qa_flags": 0,
            "valid": True,
        },
        {
            "x1": 0.0,
            "y1": 0.0,
            "x2": 10.0,
            "y2": 10.0,
            "bbox_confidence": 0.9,
            "csv_row_index": 1,
            "qa_flags": 0,
            "valid": True,
        },
    ]
    assert suppress_duplicate_boxes(
        rows, iou_threshold=0.90, minimum_area_similarity=0.70
    ) == 1
    assert not rows[0]["valid"]
    assert rows[0]["qa_flags"] & int(BBoxQAFlag.HIGH_IOU_DUPLICATE)
    assert rows[1]["valid"]


def test_det_id_is_reproducible_signed_int64() -> None:
    first = stable_det_id("sequence", "clip", 123, 456)
    assert first == stable_det_id("sequence", "clip", 123, 456)
    assert -(1 << 63) <= first < (1 << 63)
    assert first != stable_det_id("sequence", "clip", 123, 457)

