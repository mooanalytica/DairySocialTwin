from __future__ import annotations

import cv2
import numpy as np
import pytest

from cowtrack.qa.s01_review_render import (
    CONTEXT_BGR,
    EVENT_BGR,
    QUALITY_BGR,
    REVIEW_HEIGHT,
    REVIEW_WIDTH,
    RISK_BGR,
    render_review_frame,
)


CONTEXTS = np.asarray(
    [[200.0, 700.0, 600.0, 1200.0], [1000.0, 700.0, 1400.0, 1200.0]],
    dtype=np.float64,
)
TARGET = np.asarray([1000.0, 700.0, 1400.0, 1200.0], dtype=np.float64)


def _render(
    raw: np.ndarray,
    *,
    case_kind: str = "risk",
    target_bbox: np.ndarray | None = TARGET,
    event_frame: bool = False,
) -> np.ndarray:
    return render_review_frame(
        raw,
        output_width=REVIEW_WIDTH,
        output_height=REVIEW_HEIGHT,
        context_boxes=CONTEXTS,
        target_bbox=target_bbox,
        case_kind=case_kind,
        event_frame=event_frame,
    )


def _expected(
    raw: np.ndarray,
    *,
    target_bbox: np.ndarray | None,
    target_color: tuple[int, int, int],
) -> np.ndarray:
    canvas = raw.copy()
    for x1, y1, x2, y2 in CONTEXTS:
        cv2.rectangle(
            canvas,
            (int(x1), int(y1)),
            (int(x2) - 1, int(y2) - 1),
            CONTEXT_BGR,
            4,
            cv2.LINE_8,
        )
    if target_bbox is not None:
        x1, y1, x2, y2 = target_bbox
        cv2.rectangle(
            canvas,
            (int(x1), int(y1)),
            (int(x2) - 1, int(y2) - 1),
            target_color,
            10,
            cv2.LINE_8,
        )
    return cv2.resize(canvas, (1920, 1080), interpolation=cv2.INTER_AREA)


def test_bbox_only_renderer_matches_exact_rectangle_only_reference() -> None:
    raw = np.zeros((2160, 3840, 3), dtype=np.uint8)
    before = raw.copy()

    review = _render(raw)

    assert review.shape == (1080, 1920, 3)
    assert review.dtype == np.uint8
    assert np.array_equal(raw, before)
    assert np.array_equal(
        review,
        _expected(raw, target_bbox=TARGET, target_color=RISK_BGR),
    )


@pytest.mark.parametrize(
    ("case_kind", "event_frame", "expected_color"),
    [
        ("risk", False, RISK_BGR),
        ("quality_reference", False, QUALITY_BGR),
        ("risk", True, EVENT_BGR),
        ("quality_reference", True, EVENT_BGR),
    ],
)
def test_target_bbox_color_is_the_only_case_or_event_signal(
    case_kind: str,
    event_frame: bool,
    expected_color: tuple[int, int, int],
) -> None:
    raw = np.zeros((2160, 3840, 3), dtype=np.uint8)

    review = _render(raw, case_kind=case_kind, event_frame=event_frame)

    assert np.array_equal(
        review,
        _expected(raw, target_bbox=TARGET, target_color=expected_color),
    )


def test_missing_target_draws_only_gray_context_boxes() -> None:
    raw = np.zeros((2160, 3840, 3), dtype=np.uint8)

    review = _render(raw, target_bbox=None, event_frame=True)

    assert np.array_equal(
        review,
        _expected(raw, target_bbox=None, target_color=EVENT_BGR),
    )


def test_invalid_coordinate_contract_fails_instead_of_clamping() -> None:
    raw = np.zeros((2160, 3840, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="target_bbox x coordinates"):
        _render(
            raw,
            target_bbox=np.asarray([-1.0, 700.0, 1400.0, 1200.0]),
        )
