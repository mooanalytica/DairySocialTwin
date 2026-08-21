from __future__ import annotations

import numpy as np
import pytest

from cowtrack.appearance.prototypes import (
    build_micro_prototypes,
    build_tracklet_prototypes,
)
from cowtrack.config import ContractError


def _unit_angles(degrees: list[float]) -> np.ndarray:
    radians = np.deg2rad(np.asarray(degrees, dtype=np.float32))
    return np.column_stack((np.cos(radians), np.sin(radians))).astype(np.float32)


def test_short_and_missing_microtracks_keep_zero_prototype_slots() -> None:
    result = build_tracklet_prototypes(
        _unit_angles([0.0, 10.0]),
        np.asarray([10, 10], dtype=np.int64),
        np.asarray([0.9, 0.8], dtype=np.float32),
        np.asarray([10, 20], dtype=np.int64),
    )

    assert result.prototypes.shape == (2, 3, 2)
    assert not np.any(result.prototype_mask)
    assert not np.any(result.prototypes)
    assert result.num_samples.tolist() == [2, 0]
    assert result.appearance_quality.tolist() == pytest.approx([0.85, 0.0])
    assert np.all(np.isfinite(result.internal_cosine_p10))
    assert result.internal_cosine_p10[1] == 0.0


def test_single_mode_prototype_is_quality_weighted_and_normalized() -> None:
    embeddings = _unit_angles([0.0, 5.0, 10.0])
    quality = np.asarray([1.0, 0.5, 0.25], dtype=np.float32)

    result = build_tracklet_prototypes(
        embeddings,
        np.zeros(3, dtype=np.int64),
        quality,
        np.asarray([0], dtype=np.int64),
    )

    expected = np.sum(embeddings * quality[:, None], axis=0)
    expected /= np.linalg.norm(expected)
    assert result.prototype_mask.tolist() == [[True, False, False]]
    np.testing.assert_allclose(result.prototypes[0, 0], expected, atol=1e-6)
    assert np.linalg.norm(result.prototypes[0, 0]) == pytest.approx(1.0)
    assert not np.any(result.prototypes[0, 1:])


def test_isolated_medoid_outlier_is_removed() -> None:
    embeddings = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )

    result = build_tracklet_prototypes(
        embeddings,
        np.zeros(4, dtype=np.int64),
        np.ones(4, dtype=np.float32),
        np.asarray([0], dtype=np.int64),
    )

    assert result.sample_outlier_mask.tolist() == [False, False, False, True]
    assert result.sample_inlier_mask.tolist() == [True, True, True, False]
    assert result.outlier_count.tolist() == [1]
    assert result.prototype_mask.tolist() == [[True, False, False]]
    np.testing.assert_array_equal(result.prototypes[0, 0], [1.0, 0.0, 0.0])


def test_three_supported_views_produce_at_most_three_deterministic_prototypes() -> None:
    embeddings = _unit_angles([-30.0, 0.0, 30.0])
    arguments = (
        embeddings,
        np.zeros(3, dtype=np.int64),
        np.ones(3, dtype=np.float32),
        np.asarray([0], dtype=np.int64),
    )

    result = build_tracklet_prototypes(*arguments)
    repeated = build_micro_prototypes(*arguments)

    assert result.prototype_mask.tolist() == [[True, True, True]]
    assert result.outlier_count.tolist() == [0]
    np.testing.assert_allclose(
        np.linalg.norm(result.prototypes[0], axis=1), np.ones(3), atol=1e-6
    )
    np.testing.assert_array_equal(result.prototypes, repeated.prototypes)
    np.testing.assert_array_equal(result.medoid_sample_indices, [
        [1, 0, 2]
    ])


def test_zero_quality_sample_is_unusable_but_not_embedding_outlier() -> None:
    result = build_tracklet_prototypes(
        _unit_angles([0.0, 0.0, 0.0, 90.0]),
        np.zeros(4, dtype=np.int64),
        np.asarray([1.0, 1.0, 1.0, 0.0], dtype=np.float32),
        np.asarray([0], dtype=np.int64),
    )

    assert result.num_samples.tolist() == [3]
    assert result.sample_inlier_mask.tolist() == [True, True, True, False]
    assert result.sample_outlier_mask.tolist() == [False, False, False, False]
    assert result.prototype_mask.tolist() == [[True, False, False]]


def test_non_normalized_embedding_is_rejected() -> None:
    with pytest.raises(ContractError, match="L2-normalized"):
        build_tracklet_prototypes(
            np.asarray([[2.0, 0.0]], dtype=np.float32),
            np.asarray([0], dtype=np.int64),
            np.asarray([1.0], dtype=np.float32),
            np.asarray([0], dtype=np.int64),
        )
