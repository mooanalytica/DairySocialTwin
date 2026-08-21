from __future__ import annotations

import inspect

import numpy as np
import pytest

from cowtrack.qa import s05_review_render
from cowtrack.qa.s05_review_render import (
    INTERMEDIATE_BGR,
    SOURCE_BGR,
    TARGET_BGR,
    UNRELATED_BGR,
    render_s05_review_frame,
)


def test_draws_only_four_bbox_roles_without_mutating_raw_frame() -> None:
    raw = np.full((90, 160, 3), (17, 18, 19), dtype=np.uint8)
    before = raw.copy()

    rendered = render_s05_review_frame(
        raw,
        output_width=160,
        output_height=90,
        unrelated_boxes=[[10, 10, 30, 30]],
        source_boxes=[[40, 10, 60, 30]],
        target_boxes=[[70, 10, 90, 30]],
        intermediate_boxes=[[100, 10, 120, 30]],
    )

    np.testing.assert_array_equal(raw, before)
    assert tuple(rendered[10, 10]) == UNRELATED_BGR
    assert tuple(rendered[10, 40]) == SOURCE_BGR
    assert tuple(rendered[10, 70]) == TARGET_BGR
    assert tuple(rendered[10, 100]) == INTERMEDIATE_BGR
    assert tuple(rendered[45, 80]) == (17, 18, 19)


def test_renderer_exposes_no_text_or_non_bbox_overlay_primitive() -> None:
    source = inspect.getsource(s05_review_render)
    assert "putText" not in source
    assert "polylines" not in source
    assert "circle(" not in source
    assert "drawMarker" not in source


def test_rejects_changed_coordinates_or_rotation() -> None:
    raw = np.zeros((90, 160, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="x coordinates"):
        render_s05_review_frame(
            raw,
            output_width=160,
            output_height=90,
            unrelated_boxes=[[-1, 1, 20, 20]],
            source_boxes=[],
            target_boxes=[],
            intermediate_boxes=[],
        )

    portrait = np.zeros((160, 90, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="landscape"):
        render_s05_review_frame(
            portrait,
            output_width=90,
            output_height=160,
            unrelated_boxes=[],
            source_boxes=[],
            target_boxes=[],
            intermediate_boxes=[],
        )


def test_focal_roles_overpaint_context_without_changing_geometry() -> None:
    raw = np.zeros((90, 160, 3), dtype=np.uint8)
    rendered = render_s05_review_frame(
        raw,
        output_width=160,
        output_height=90,
        unrelated_boxes=[[10, 10, 30, 30]],
        intermediate_boxes=[[10, 10, 30, 30]],
        source_boxes=[[10, 10, 30, 30]],
        target_boxes=[],
    )
    assert tuple(rendered[10, 10]) == SOURCE_BGR
