from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from cowtrack.config import ContractError
from cowtrack.linking.features import (
    APPEARANCE_FEATURE_NAMES,
    LONG_FEATURE_SCHEMA,
    SHORT_FEATURE_SCHEMA,
    CleanGallery,
    EndpointContext,
    EndpointGeometry,
    MotionHistory,
    appearance_pair_features,
    build_clean_gallery,
    context_pair_features,
    feature_schema,
    has_high_endpoint_overlap,
    motion_pair_features,
    pair_features,
    validate_disjoint_sides,
)


def _unit_angles(degrees: list[float]) -> np.ndarray:
    radians = np.deg2rad(np.asarray(degrees, dtype=np.float32))
    return np.column_stack((np.cos(radians), np.sin(radians))).astype(np.float32)


def _build_gallery(
    degrees: list[float],
    *,
    offset: int,
    quality: np.ndarray | None = None,
) -> CleanGallery:
    count = len(degrees)
    result = build_clean_gallery(
        _unit_angles(degrees),
        sample_ids=np.arange(offset, offset + count, dtype=np.int64),
        det_ids=np.arange(offset + 1_000, offset + 1_000 + count, dtype=np.int64),
        embedding_rows=np.arange(
            offset + 2_000, offset + 2_000 + count, dtype=np.int64
        ),
        crop_quality=(
            np.ones(count, dtype=np.float32) if quality is None else quality
        ),
        other_bbox_max_iou=np.zeros(count, dtype=np.float32),
        s02_inlier_mask=np.ones(count, dtype=np.bool_),
    )
    assert result is not None
    return result


def test_clean_gallery_applies_all_masks_then_recomputes_local_outliers() -> None:
    embeddings = _unit_angles([0.0, 0.0, 0.0, 90.0, 0.0, 0.0, 0.0])
    result = build_clean_gallery(
        embeddings,
        sample_ids=np.arange(7, dtype=np.int64),
        det_ids=np.arange(100, 107, dtype=np.int64),
        embedding_rows=np.arange(200, 207, dtype=np.int64),
        crop_quality=np.ones(7, dtype=np.float32),
        other_bbox_max_iou=np.asarray(
            [0.0, 0.0, 0.0, 0.0, 0.25, 0.0, 0.0], dtype=np.float32
        ),
        s02_inlier_mask=np.asarray(
            [True, True, True, True, True, False, True], dtype=np.bool_
        ),
        review_excluded_mask=np.asarray(
            [False, False, False, False, False, False, True], dtype=np.bool_
        ),
    )

    assert result is not None
    assert result.clean_candidate_mask.tolist() == [
        True,
        True,
        True,
        True,
        False,
        False,
        False,
    ]
    assert result.clean_inlier_mask.tolist() == [
        True,
        True,
        True,
        False,
        False,
        False,
        False,
    ]
    assert result.local_outlier_mask.tolist() == [
        False,
        False,
        False,
        True,
        False,
        False,
        False,
    ]
    assert result.sample_ids.tolist() == [0, 1, 2]
    assert result.num_overlap_rejected == 1
    assert result.num_review_excluded == 1
    assert result.num_local_outliers == 1
    assert result.max_other_bbox_iou == pytest.approx(0.25)
    assert result.max_clean_other_bbox_iou == 0.0
    np.testing.assert_allclose(result.prototypes, [[1.0, 0.0]], atol=1e-7)


def test_zero_embedding_is_rejected_and_insufficient_local_inliers_fail_closed() -> None:
    with pytest.raises(ContractError, match="L2-normalized"):
        build_clean_gallery(
            np.asarray([[1.0, 0.0], [1.0, 0.0], [0.0, 0.0]], dtype=np.float32),
            sample_ids=np.arange(3, dtype=np.int64),
            det_ids=np.arange(10, 13, dtype=np.int64),
            embedding_rows=np.arange(20, 23, dtype=np.int64),
            crop_quality=np.ones(3, dtype=np.float32),
            other_bbox_max_iou=np.zeros(3, dtype=np.float32),
            s02_inlier_mask=np.ones(3, dtype=np.bool_),
        )

    assert (
        build_clean_gallery(
            _unit_angles([0.0, 0.0, 90.0]),
            sample_ids=np.arange(3, dtype=np.int64),
            det_ids=np.arange(10, 13, dtype=np.int64),
            embedding_rows=np.arange(20, 23, dtype=np.int64),
            crop_quality=np.ones(3, dtype=np.float32),
            other_bbox_max_iou=np.zeros(3, dtype=np.float32),
            s02_inlier_mask=np.ones(3, dtype=np.bool_),
        )
        is None
    )

    source = _build_gallery([0.0, 0.0, 0.0], offset=100)
    target = _build_gallery([5.0, 5.0, 5.0], offset=10_000)
    zero_prototype = replace(source, prototypes=np.zeros_like(source.prototypes))
    with pytest.raises(ContractError, match="L2-normalized"):
        appearance_pair_features(zero_prototype, target)


def test_clean_gallery_and_appearance_features_are_input_order_invariant() -> None:
    embeddings = _unit_angles([-30.0, 0.0, 30.0, 5.0])
    sample_ids = np.asarray([40, 10, 30, 20], dtype=np.int64)
    det_ids = np.asarray([140, 110, 130, 120], dtype=np.int64)
    embedding_rows = np.asarray([240, 210, 230, 220], dtype=np.int64)
    quality = np.asarray([0.6, 1.0, 0.7, 0.8], dtype=np.float32)

    def build(order: np.ndarray) -> CleanGallery:
        gallery = build_clean_gallery(
            embeddings[order],
            sample_ids=sample_ids[order],
            det_ids=det_ids[order],
            embedding_rows=embedding_rows[order],
            crop_quality=quality[order],
            other_bbox_max_iou=np.zeros(4, dtype=np.float32),
            s02_inlier_mask=np.ones(4, dtype=np.bool_),
        )
        assert gallery is not None
        return gallery

    first = build(np.arange(4))
    shuffled = build(np.asarray([2, 0, 3, 1]))
    np.testing.assert_array_equal(first.sample_ids, shuffled.sample_ids)
    np.testing.assert_allclose(first.embeddings, shuffled.embeddings, atol=0.0)
    np.testing.assert_allclose(first.prototypes, shuffled.prototypes, atol=0.0)
    np.testing.assert_allclose(first.medoid_embedding, shuffled.medoid_embedding, atol=0.0)
    assert first.appearance_quality == shuffled.appearance_quality

    target = _build_gallery([10.0, 15.0, 20.0], offset=1_000)
    assert appearance_pair_features(first, target) == appearance_pair_features(
        shuffled, target
    )


@pytest.mark.parametrize("field", ["sample_ids", "det_ids", "embedding_rows"])
def test_disjoint_validator_rejects_every_kind_of_side_leakage(field: str) -> None:
    source = _build_gallery([0.0, 0.0, 0.0], offset=0)
    target = _build_gallery([5.0, 5.0, 5.0], offset=10_000)
    leaked = getattr(target, field).copy()
    leaked[0] = getattr(source, field)[0]
    target = replace(target, **{field: leaked})

    with pytest.raises(ContractError, match=field.removesuffix("s")):
        validate_disjoint_sides(source, target)


def test_appearance_features_have_fixed_all_pair_and_mutual_semantics() -> None:
    source = _build_gallery([-30.0, 0.0, 30.0], offset=0)
    target = _build_gallery([0.0, 10.0, 20.0], offset=10_000)

    result = appearance_pair_features(source, target)

    assert tuple(result) == APPEARANCE_FEATURE_NAMES
    assert all(np.isfinite(list(result.values())))
    assert -1.0 <= result["prototype_cosine_top3_mean"] <= 1.0
    assert -1.0 <= result["mutual_prototype_score"] <= 1.0
    assert result["appearance_quality_min"] == 1.0
    assert result["appearance_quality_mean"] == 1.0


def _history() -> MotionHistory:
    return MotionHistory(
        time_sec=np.asarray([0.0, 1.0], dtype=np.float64),
        cx_norm=np.asarray([0.1, 0.2], dtype=np.float32),
        cy_norm=np.asarray([0.2, 0.2], dtype=np.float32),
        w_norm=np.asarray([0.1, 0.1], dtype=np.float32),
        h_norm=np.asarray([0.2, 0.2], dtype=np.float32),
    )


def test_short_motion_predicts_from_history_and_long_removes_velocity_features() -> None:
    source = EndpointGeometry(1.0, 0.2, 0.2, 0.1, 0.2)
    short_target = EndpointGeometry(2.0, 0.3, 0.2, 0.1, 0.2)
    short = motion_pair_features(
        source, short_target, mode="short", source_history=_history()
    )

    assert short["gap_sec"] == 1.0
    assert short["predicted_center_residual"] == pytest.approx(0.0, abs=1e-6)
    assert short["predicted_iou"] == pytest.approx(1.0, abs=1e-6)
    assert short["log_area_ratio"] == pytest.approx(0.0)

    long_target = EndpointGeometry(7.0, 0.8, 0.2, 0.1, 0.2)
    long = motion_pair_features(source, long_target, mode="long")
    assert "predicted_center_residual" not in long
    assert "predicted_iou" not in long
    assert long["gap_sec"] == 6.0


def test_history_row_order_does_not_change_short_motion() -> None:
    source = EndpointGeometry(1.0, 0.2, 0.2, 0.1, 0.2)
    target = EndpointGeometry(2.0, 0.3, 0.2, 0.1, 0.2)
    history = _history()
    reversed_history = MotionHistory(
        time_sec=history.time_sec[::-1],
        cx_norm=history.cx_norm[::-1],
        cy_norm=history.cy_norm[::-1],
        w_norm=history.w_norm[::-1],
        h_norm=history.h_norm[::-1],
    )
    assert motion_pair_features(
        source, target, mode="short", source_history=history
    ) == motion_pair_features(
        source, target, mode="short", source_history=reversed_history
    )


def test_context_preserves_overlap_metadata_for_explicit_gating() -> None:
    source = EndpointContext(0.4, 0.1, "clip-a")
    target = EndpointContext(0.2, 0.3, "clip-b")

    result = context_pair_features(source, target)

    assert result == {
        "src_end_max_other_iou": 0.4,
        "dst_start_max_other_iou": 0.2,
        "src_end_boundary_distance": 0.1,
        "dst_start_boundary_distance": 0.3,
        "is_clip_boundary": 1.0,
    }
    assert has_high_endpoint_overlap(source, target, threshold=0.25)


def test_pair_schema_excludes_forbidden_and_long_velocity_features() -> None:
    forbidden = ("keypoint", "legacy", "raw_x", "raw_y", "cx_norm", "cy_norm")
    assert feature_schema("short") == SHORT_FEATURE_SCHEMA
    assert feature_schema("long") == LONG_FEATURE_SCHEMA
    assert all(not any(token in name for token in forbidden) for name in SHORT_FEATURE_SCHEMA)
    assert "predicted_center_residual" not in LONG_FEATURE_SCHEMA
    assert "predicted_iou" not in LONG_FEATURE_SCHEMA

    source_gallery = _build_gallery([0.0, 0.0, 0.0], offset=0)
    target_gallery = _build_gallery([5.0, 5.0, 5.0], offset=10_000)
    values = pair_features(
        source_gallery,
        target_gallery,
        EndpointGeometry(1.0, 0.2, 0.2, 0.1, 0.2),
        EndpointGeometry(2.0, 0.3, 0.2, 0.1, 0.2),
        EndpointContext(0.0, 0.2, "clip-a"),
        EndpointContext(0.0, 0.2, "clip-a"),
        mode="short",
        source_history=_history(),
    )
    assert tuple(values) == SHORT_FEATURE_SCHEMA
