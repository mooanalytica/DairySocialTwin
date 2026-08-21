from __future__ import annotations

import numpy as np
import pytest

from cowtrack.appearance.sampling import (
    select_candidate_pool_indices,
    select_representative_indices,
)
from cowtrack.config import ContractError


def test_exclusions_are_never_selected_and_endpoints_use_eligible_rows() -> None:
    micro_ids = np.asarray([1, 0, 1, 0, 1, 0, 1, 0], dtype=np.int64)
    times = np.asarray([0.0, 0.0, 0.5, 0.5, 1.0, 1.0, 1.5, 1.5])
    frames = np.asarray([10, 0, 15, 5, 20, 10, 25, 15], dtype=np.int64)
    overlap = np.zeros(8, dtype=np.float32)
    excluded = np.asarray([True, False, False, False, False, False, False, False])

    selected = select_representative_indices(
        micro_ids, times, frames, overlap, excluded
    )

    assert 0 not in selected
    assert selected.tolist() == [1, 3, 5, 2, 7, 4, 6]
    assert np.all(np.diff(frames[selected]) >= 0)


def test_periodic_slot_prioritizes_low_overlap_crop() -> None:
    times = np.arange(21, dtype=np.float64) * 0.1
    frames = np.arange(21, dtype=np.int64)
    overlap = np.full(21, 0.40, dtype=np.float32)
    overlap[10] = 0.10
    overlap[14] = 0.05

    selected = select_representative_indices(
        np.zeros(21, dtype=np.int64),
        times,
        frames,
        overlap,
        np.zeros(21, dtype=np.bool_),
    )

    assert selected.tolist() == [0, 1, 2, 10, 14, 18, 19, 20]


def test_long_microtrack_preserves_endpoints_and_caps_at_24() -> None:
    count = 241
    frames = np.arange(count, dtype=np.int64)
    selected = select_representative_indices(
        np.zeros(count, dtype=np.int64),
        frames.astype(np.float64) * 0.25,
        frames,
        np.zeros(count, dtype=np.float32),
        np.zeros(count, dtype=np.bool_),
    )

    assert len(selected) == 24
    assert set([0, 1, 2, 238, 239, 240]).issubset(set(selected.tolist()))
    assert np.all(np.diff(frames[selected]) > 0)
    repeated = select_representative_indices(
        np.zeros(count, dtype=np.int64),
        frames.astype(np.float64) * 0.25,
        frames,
        np.zeros(count, dtype=np.float32),
        np.zeros(count, dtype=np.bool_),
    )
    np.testing.assert_array_equal(selected, repeated)


def test_double_candidate_pool_supplies_post_quality_replacements() -> None:
    count = 241
    frames = np.arange(count, dtype=np.int64)
    times = frames.astype(np.float64) * 0.25
    micro_ids = np.zeros(count, dtype=np.int64)
    overlaps = np.zeros(count, dtype=np.float32)
    excluded = np.zeros(count, dtype=np.bool_)

    candidate = select_candidate_pool_indices(
        micro_ids, times, frames, overlaps, excluded
    )
    assert len(candidate) == 48
    # Treat the first three candidate crops as failing decoded blur/quality.
    retained = candidate[3:]
    final_local = select_representative_indices(
        micro_ids[retained],
        times[retained],
        frames[retained],
        overlaps[retained],
        np.zeros(len(retained), dtype=np.bool_),
    )
    final = retained[final_local]

    assert len(final) == 24
    assert final[:3].tolist() == retained[:3].tolist()
    assert set(candidate[:3]).isdisjoint(set(final.tolist()))


def test_duplicate_frame_in_one_microtrack_is_rejected() -> None:
    with pytest.raises(ContractError, match="duplicate global_frame"):
        select_representative_indices(
            np.asarray([0, 0], dtype=np.int64),
            np.asarray([0.0, 0.1]),
            np.asarray([2, 2], dtype=np.int64),
            np.zeros(2, dtype=np.float32),
            np.zeros(2, dtype=np.bool_),
        )


def test_non_boolean_exclusion_mask_is_rejected() -> None:
    with pytest.raises(ContractError, match="must be a boolean"):
        select_representative_indices(
            np.asarray([0], dtype=np.int64),
            np.asarray([0.0]),
            np.asarray([0], dtype=np.int64),
            np.asarray([0.0]),
            np.asarray([0], dtype=np.int8),
        )
