from __future__ import annotations

import pytest

from cowtrack.config import ContractError
from cowtrack.frame_observation import (
    frame_observation_policy,
    summarize_frame_observation,
)


def test_frame_observation_reports_inclusive_unobserved_intervals() -> None:
    summary = summarize_frame_observation(10, [0, 4, 4, 5, 9])

    assert summary == {
        "num_frames": 10,
        "num_observed_frames": 4,
        "num_unobserved_frames": 6,
        "observed_frame_fraction": 0.4,
        "num_unobserved_intervals": 2,
        "longest_unobserved_interval_frames": 3,
        "unobserved_frame_intervals": [
            {"start_frame": 1, "end_frame": 3, "num_frames": 3},
            {"start_frame": 6, "end_frame": 8, "num_frames": 3},
        ],
    }


def test_frame_observation_handles_fully_observed_and_fully_unobserved() -> None:
    full = summarize_frame_observation(3, [2, 0, 1])
    assert full["num_unobserved_frames"] == 0
    assert full["unobserved_frame_intervals"] == []
    assert full["longest_unobserved_interval_frames"] == 0

    absent = summarize_frame_observation(3, [])
    assert absent["num_observed_frames"] == 0
    assert absent["unobserved_frame_intervals"] == [
        {"start_frame": 0, "end_frame": 2, "num_frames": 3}
    ]


@pytest.mark.parametrize("observed", [[-1], [3], [1.5], [True]])
def test_frame_observation_rejects_invalid_indices(observed: list[object]) -> None:
    with pytest.raises(ContractError, match="observed frame"):
        summarize_frame_observation(3, observed)  # type: ignore[arg-type]


def test_frame_observation_policy_marks_missing_rows_as_unobserved() -> None:
    policy = frame_observation_policy()
    assert policy["unobserved_frames_are_empty_scene_evidence"] is False
    assert policy["reidentification_scope"] == "source_bbox_csv_rows_only"
    assert (
        policy["full_video_render_policy"]
        == "preserve_source_frame_without_identity_overlay"
    )
