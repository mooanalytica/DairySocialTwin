from __future__ import annotations

import numpy as np
import pytest

from cowtrack.config import ContractError
from cowtrack.linking.forced_appearance import (
    GRADE_A_CLEAN,
    GRADE_B_EXISTING_DEGRADED,
    GRADE_C_SUPPLEMENTAL,
    GRADE_MISSING,
    SupplementalStableEmbeddings,
    build_forced_appearance_base,
    complete_forced_appearance_with_supplemental,
    restore_completed_forced_appearance,
)


def _unit(*values: float) -> np.ndarray:
    row = np.asarray(values, dtype=np.float32)
    return row / np.linalg.norm(row)


def _empty_existing(count: int, dimension: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.zeros((count, 3, dimension), dtype=np.float16),
        np.zeros((count, 3), dtype=np.bool_),
        np.zeros(count, dtype=np.bool_),
    )


def _base(
    *,
    count: int,
    dimension: int,
    sample_ids: list[int],
    sample_stable_ids: list[int],
    embeddings: list[np.ndarray],
    quality: list[float],
    iou: list[float],
    inlier: list[bool],
    existing: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
):
    prototypes, mask, usable = existing or _empty_existing(count, dimension)
    matrix = (
        np.stack(embeddings).astype(np.float32)
        if embeddings
        else np.empty((0, dimension), dtype=np.float32)
    )
    return build_forced_appearance_base(
        stable_ids=np.arange(count, dtype=np.int64),
        existing_prototypes=prototypes,
        existing_prototype_mask=mask,
        existing_usable=usable,
        sample_ids=np.asarray(sample_ids, dtype=np.int64),
        sample_embeddings=matrix,
        sample_stable_ids=np.asarray(sample_stable_ids, dtype=np.int64),
        sample_quality=np.asarray(quality, dtype=np.float32),
        sample_other_bbox_max_iou=np.asarray(iou, dtype=np.float32),
        sample_s02_inlier=np.asarray(inlier, dtype=np.bool_),
    )


def test_a_is_copied_exactly_and_zero_sample_gap_is_explicit() -> None:
    prototypes, mask, usable = _empty_existing(3, 4)
    prototypes[0, 0] = _unit(1, 1, 0, 0).astype(np.float16)
    prototypes[0, 1] = _unit(1, 0, 1, 0).astype(np.float16)
    mask[0, :2] = True
    usable[0] = True
    original = prototypes.copy()

    result = _base(
        count=3,
        dimension=4,
        sample_ids=[11],
        sample_stable_ids=[1],
        embeddings=[_unit(0, 1, 0, 0)],
        quality=[0.8],
        iou=[0.1],
        inlier=[True],
        existing=(prototypes, mask, usable),
    )

    assert np.array_equal(result.prototypes[0], original[0])
    assert np.array_equal(result.prototype_mask[0], mask[0])
    assert result.rows[0]["evidence_grade"] == GRADE_A_CLEAN
    assert result.rows[1]["evidence_grade"] == GRADE_B_EXISTING_DEGRADED
    assert result.rows[2]["evidence_grade"] == GRADE_MISSING
    assert result.zero_sample_stable_ids == (2,)
    assert result.prototypes.dtype == np.float16
    assert result.descriptor_centers.dtype == np.float32
    assert not result.prototypes.flags.writeable


def test_b_uses_the_fixed_progressive_relaxation_order() -> None:
    result = _base(
        count=4,
        dimension=4,
        sample_ids=[10, 11, 20, 21, 22, 30, 31],
        sample_stable_ids=[0, 0, 1, 1, 1, 2, 2],
        embeddings=[
            _unit(1, 0, 0, 0),
            _unit(0, 1, 0, 0),
            _unit(0, 0, 1, 0),
            _unit(0, 0, 0, 1),
            _unit(1, 1, 0, 0),
            _unit(1, 0, 1, 0),
            _unit(0, 1, 0, 1),
        ],
        quality=[0.9] * 7,
        iou=[0.1, 0.7, 0.4, 0.6, 0.1, 0.1, 0.2],
        inlier=[True, True, True, True, False, False, False],
    )

    assert result.rows[0]["selection_policy"] == "inlier_iou_lt_0_25"
    assert result.rows[0]["selected_sample_ids"] == (10,)
    assert result.rows[1]["selection_policy"] == "all_inlier"
    assert result.rows[1]["selected_sample_ids"] == (20, 21)
    assert result.rows[2]["selection_policy"] == "all_samples"
    assert result.rows[2]["selected_sample_ids"] == (30, 31)
    assert result.rows[2]["used_s02_outlier"] is True
    assert result.zero_sample_stable_ids == (3,)


def test_one_two_and_many_sample_b_descriptors_are_supported_and_normalized() -> None:
    result = _base(
        count=3,
        dimension=4,
        sample_ids=[1, 2, 3, 4, 5, 6],
        sample_stable_ids=[0, 1, 1, 2, 2, 2],
        embeddings=[
            _unit(1, 0, 0, 0),
            _unit(0, 1, 0, 0),
            _unit(0, 0.98, 0.2, 0),
            _unit(0, 0, 1, 0),
            _unit(0, 0, 0.98, 0.2),
            _unit(0.2, 0, 0.98, 0),
        ],
        quality=[0.7, 0.8, 0.9, 0.7, 0.8, 0.9],
        iou=[0.1] * 6,
        inlier=[True] * 6,
    )

    assert np.count_nonzero(result.prototype_mask[0]) == 1
    assert np.count_nonzero(result.prototype_mask[1]) == 2
    assert 1 <= np.count_nonzero(result.prototype_mask[2]) <= 3
    valid = result.prototypes[result.prototype_mask].astype(np.float32)
    assert np.allclose(np.linalg.norm(valid, axis=1), 1.0, atol=2e-3, rtol=0)
    assert np.allclose(
        np.linalg.norm(result.descriptor_centers, axis=1), 1.0, atol=2e-6, rtol=0
    )


def test_one_sample_float16_quantization_is_renormalized() -> None:
    quantized = _unit(1.0, 0.37, 0.11).astype(np.float16)
    assert not np.isclose(np.linalg.norm(quantized.astype(np.float32)), 1.0, atol=2e-6)

    result = build_forced_appearance_base(
        stable_ids=np.asarray([0], dtype=np.int64),
        existing_prototypes=np.zeros((1, 3, 3), dtype=np.float16),
        existing_prototype_mask=np.zeros((1, 3), dtype=np.bool_),
        existing_usable=np.zeros(1, dtype=np.bool_),
        sample_ids=np.asarray([1], dtype=np.int64),
        sample_embeddings=quantized[None, :],
        sample_stable_ids=np.asarray([0], dtype=np.int64),
        sample_quality=np.asarray([0.8], dtype=np.float32),
        sample_other_bbox_max_iou=np.asarray([0.1], dtype=np.float32),
        sample_s02_inlier=np.asarray([True], dtype=np.bool_),
    )

    prototype = result.prototypes[0, 0].astype(np.float32)
    assert np.isclose(np.linalg.norm(prototype), 1.0, atol=2e-3)
    assert np.isclose(np.linalg.norm(result.descriptor_centers[0]), 1.0, atol=2e-6)


def test_relaxed_many_sample_result_is_invariant_to_input_row_order() -> None:
    ids = np.asarray([30, 10, 40, 20], dtype=np.int64)
    embeddings = np.stack(
        [
            _unit(1.0, 0.0, 0.0),
            _unit(0.99, 0.1, 0.0),
            _unit(0.0, 1.0, 0.0),
            _unit(0.1, 0.99, 0.0),
        ]
    )
    order = np.asarray([2, 0, 3, 1])

    first = _base(
        count=1,
        dimension=3,
        sample_ids=ids.tolist(),
        sample_stable_ids=[0] * 4,
        embeddings=list(embeddings),
        quality=[0.7, 0.8, 0.9, 1.0],
        iou=[0.1] * 4,
        inlier=[True] * 4,
    )
    second = _base(
        count=1,
        dimension=3,
        sample_ids=ids[order].tolist(),
        sample_stable_ids=[0] * 4,
        embeddings=list(embeddings[order]),
        quality=np.asarray([0.7, 0.8, 0.9, 1.0])[order].tolist(),
        iou=[0.1] * 4,
        inlier=[True] * 4,
    )

    assert np.array_equal(first.prototypes, second.prototypes)
    assert np.array_equal(first.prototype_mask, second.prototype_mask)
    assert np.array_equal(first.descriptor_centers, second.descriptor_centers)
    assert first.rows[0]["selected_sample_ids"] == second.rows[0]["selected_sample_ids"]


def test_supplemental_completion_fills_every_c_gap_and_preserves_a() -> None:
    prototypes, mask, usable = _empty_existing(3, 4)
    prototypes[0, 0] = _unit(1, 1, 0, 0).astype(np.float16)
    mask[0, 0] = True
    usable[0] = True
    base = _base(
        count=3,
        dimension=4,
        sample_ids=[],
        sample_stable_ids=[],
        embeddings=[],
        quality=[],
        iou=[],
        inlier=[],
        existing=(prototypes, mask, usable),
    )
    before_a = base.prototypes[0].copy()

    result = complete_forced_appearance_with_supplemental(
        base,
        (
            SupplementalStableEmbeddings(
                stable_id=1,
                sample_ids=np.asarray([100], dtype=np.int64),
                embeddings=np.stack([_unit(0, 1, 0, 0)]),
                quality=np.asarray([0.8], dtype=np.float32),
            ),
            SupplementalStableEmbeddings(
                stable_id=2,
                sample_ids=np.asarray([200, 201, 202], dtype=np.int64),
                embeddings=np.stack(
                    [
                        _unit(0, 0, 1, 0),
                        _unit(0, 0, 0.98, 0.2),
                        _unit(0.1, 0, 0.99, 0),
                    ]
                ),
                quality=np.asarray([0.7, 0.8, 0.9], dtype=np.float32),
            ),
        ),
    )

    assert result.zero_sample_stable_ids == ()
    assert np.all(np.any(result.prototype_mask, axis=1))
    assert np.array_equal(result.prototypes[0], before_a)
    assert result.rows[0]["evidence_grade"] == GRADE_A_CLEAN
    assert result.rows[1]["evidence_grade"] == GRADE_C_SUPPLEMENTAL
    assert result.rows[2]["evidence_grade"] == GRADE_C_SUPPLEMENTAL


def test_supplemental_completion_fails_closed_on_missing_or_extra_stable() -> None:
    base = _base(
        count=2,
        dimension=3,
        sample_ids=[],
        sample_stable_ids=[],
        embeddings=[],
        quality=[],
        iou=[],
        inlier=[],
    )
    one = SupplementalStableEmbeddings(
        stable_id=0,
        sample_ids=np.asarray([1], dtype=np.int64),
        embeddings=np.stack([_unit(1, 0, 0)]),
        quality=np.asarray([0.9], dtype=np.float32),
    )
    with pytest.raises(ContractError, match="exactly cover C gaps"):
        complete_forced_appearance_with_supplemental(base, (one,))

    extra = SupplementalStableEmbeddings(
        stable_id=2,
        sample_ids=np.asarray([2], dtype=np.int64),
        embeddings=np.stack([_unit(0, 1, 0)]),
        quality=np.asarray([0.9], dtype=np.float32),
    )
    with pytest.raises(ContractError, match="exactly cover C gaps"):
        complete_forced_appearance_with_supplemental(base, (one, extra))


def test_completed_bundle_round_trips_persisted_float32_centers_exactly() -> None:
    base = _base(
        count=1,
        dimension=3,
        sample_ids=[7],
        sample_stable_ids=[0],
        embeddings=[_unit(1.0, 0.37, 0.11)],
        quality=[0.8],
        iou=[0.1],
        inlier=[True],
    )

    restored = restore_completed_forced_appearance(
        stable_ids=base.stable_ids,
        prototypes=base.prototypes,
        prototype_mask=base.prototype_mask,
        descriptor_centers=base.descriptor_centers,
        rows=base.rows,
    )

    assert np.array_equal(restored.prototypes, base.prototypes)
    assert np.array_equal(restored.prototype_mask, base.prototype_mask)
    assert np.array_equal(restored.descriptor_centers, base.descriptor_centers)
    assert restored.descriptor_centers.dtype == np.float32
    assert not restored.descriptor_centers.flags.writeable


def test_completed_bundle_restore_rejects_reconstructed_or_wrong_centers() -> None:
    base = _base(
        count=1,
        dimension=3,
        sample_ids=[7],
        sample_stable_ids=[0],
        embeddings=[_unit(1.0, 0.37, 0.11)],
        quality=[0.8],
        iou=[0.1],
        inlier=[True],
    )
    wrong_dtype = base.descriptor_centers.astype(np.float16)

    with pytest.raises(ContractError, match="restored prototype tensors"):
        restore_completed_forced_appearance(
            stable_ids=base.stable_ids,
            prototypes=base.prototypes,
            prototype_mask=base.prototype_mask,
            descriptor_centers=wrong_dtype,
            rows=base.rows,
        )


def test_malformed_inputs_fail_closed() -> None:
    prototypes, mask, usable = _empty_existing(1, 3)
    prototypes[0, 0] = _unit(1, 0, 0).astype(np.float16)
    mask[0, 0] = True
    with pytest.raises(ContractError, match="usable/mask"):
        _base(
            count=1,
            dimension=3,
            sample_ids=[],
            sample_stable_ids=[],
            embeddings=[],
            quality=[],
            iou=[],
            inlier=[],
            existing=(prototypes, mask, usable),
        )

    with pytest.raises(ContractError, match="L2-normalized"):
        _base(
            count=1,
            dimension=3,
            sample_ids=[1],
            sample_stable_ids=[0],
            embeddings=[np.asarray([2.0, 0.0, 0.0], dtype=np.float32)],
            quality=[0.8],
            iou=[0.1],
            inlier=[True],
        )

    with pytest.raises(ContractError, match="sample IDs must be unique"):
        _base(
            count=1,
            dimension=3,
            sample_ids=[1, 1],
            sample_stable_ids=[0, 0],
            embeddings=[_unit(1, 0, 0), _unit(0, 1, 0)],
            quality=[0.8, 0.8],
            iou=[0.1, 0.1],
            inlier=[True, True],
        )
