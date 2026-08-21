from __future__ import annotations

import inspect

import numpy as np
import pytest

from cowtrack.qa import s04_review_render
from cowtrack.qa.s04_review_render import (
    INTERMEDIATE_BGR,
    SOURCE_BGR,
    TARGET_BGR,
    UNRELATED_BGR,
    render_s04_review_frame,
)


def test_draws_only_the_four_bbox_roles_without_mutating_raw_frame() -> None:
    raw = np.full((90, 160, 3), (17, 18, 19), dtype=np.uint8)
    before = raw.copy()

    rendered = render_s04_review_frame(
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


def test_renderer_source_contains_no_text_or_overlay_primitive() -> None:
    source = inspect.getsource(s04_review_render)
    assert "putText" not in source
    assert "polylines" not in source
    assert "circle(" not in source
    assert "drawMarker" not in source


def test_rejects_box_geometry_outside_the_original_frame() -> None:
    raw = np.zeros((90, 160, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="x coordinates"):
        render_s04_review_frame(
            raw,
            output_width=160,
            output_height=90,
            unrelated_boxes=[[-1, 1, 20, 20]],
            source_boxes=[],
            target_boxes=[],
            intermediate_boxes=[],
        )


def test_rejects_rotation_or_aspect_change() -> None:
    portrait = np.zeros((160, 90, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="landscape"):
        render_s04_review_frame(
            portrait,
            output_width=90,
            output_height=160,
            unrelated_boxes=[],
            source_boxes=[],
            target_boxes=[],
            intermediate_boxes=[],
        )
