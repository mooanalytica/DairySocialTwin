from __future__ import annotations

import numpy as np
import pytest

from cowtrack.appearance.quality import (
    bbox_area_percentiles,
    combine_crop_quality,
    crop_with_padding,
    laplacian_blur_score,
    maximum_other_bbox_iou,
    soft_border_mask,
)
from cowtrack.config import ContractError


def test_maximum_other_iou_is_limited_to_same_frame() -> None:
    boxes = np.asarray(
        [
            [0.0, 0.0, 10.0, 10.0],
            [5.0, 0.0, 15.0, 10.0],
            [20.0, 0.0, 30.0, 10.0],
            [0.0, 0.0, 10.0, 10.0],
        ]
    )
    frames = np.asarray([0, 0, 0, 1], dtype=np.int64)

    result = maximum_other_bbox_iou(boxes, frames)

    np.testing.assert_allclose(result, [1.0 / 3.0, 1.0 / 3.0, 0.0, 0.0])


def test_area_percentiles_use_midrank_for_ties() -> None:
    boxes = np.asarray(
        [
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 2.0, 1.0],
            [0.0, 0.0, 2.0, 1.0],
            [0.0, 0.0, 4.0, 1.0],
        ]
    )

    np.testing.assert_allclose(
        bbox_area_percentiles(boxes), [0.0, 0.5, 0.5, 1.0]
    )
    np.testing.assert_array_equal(
        bbox_area_percentiles(np.asarray([[0.0, 0.0, 1.0, 1.0]])),
        [1.0],
    )


def test_crop_padding_reports_clipping_and_boundary_distance() -> None:
    frame = np.arange(10 * 10 * 3, dtype=np.uint8).reshape(10, 10, 3)

    crop, clipped, boundary = crop_with_padding(
        frame, [0.0, 2.0, 4.0, 6.0], 0.25
    )
    centered, centered_clipped, centered_boundary = crop_with_padding(
        frame, [3.0, 3.0, 7.0, 7.0], 0.25
    )

    assert crop.shape == (6, 5, 3)
    assert clipped == pytest.approx(1.0 / 6.0)
    assert boundary == 0.0
    assert centered.shape == (6, 6, 3)
    assert centered_clipped == 0.0
    assert centered_boundary == 1.0


def test_soft_border_mask_preserves_center_and_attenuates_corner() -> None:
    crop = np.full((20, 20, 3), 200, dtype=np.uint8)

    masked = soft_border_mask(crop)

    assert masked.dtype == np.uint8
    assert masked.shape == crop.shape
    assert masked[10, 10].tolist() == [200, 200, 200]
    assert np.all(masked[0, 0] < 100)
    np.testing.assert_array_equal(masked, soft_border_mask(crop))


def test_laplacian_blur_score_separates_flat_and_checkerboard() -> None:
    flat = np.full((12, 12, 3), 127, dtype=np.uint8)
    checker = ((np.indices((12, 12)).sum(axis=0) % 2) * 255).astype(np.uint8)
    checker_rgb = np.repeat(checker[:, :, None], 3, axis=2)

    assert laplacian_blur_score(flat) == 0.0
    assert laplacian_blur_score(checker_rgb) > 1000.0


def test_fixed_quality_formula_has_bounded_endpoints() -> None:
    result = combine_crop_quality(
        np.asarray([0.0, 0.5]),
        np.asarray([0.0, 0.25]),
        np.asarray([0.5, 0.0]),
        np.asarray([400.0, 20.0]),
        np.asarray([1.0, 0.0]),
    )

    np.testing.assert_allclose(result, [1.0, 0.0], atol=1e-7)
    assert result.dtype == np.float32


def test_crop_rejects_bbox_outside_frame() -> None:
    with pytest.raises(ContractError, match="clamped"):
        crop_with_padding(
            np.zeros((10, 10, 3), dtype=np.uint8),
            [-1.0, 0.0, 5.0, 5.0],
            0.05,
        )
