from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dairy_social.communities import _time_indexed_frame, compute_community_stability


EDGE_COLUMNS = ["cow_i", "cow_j", "layer", "expected_seconds"]


def _config(window_s: float, step_s: float) -> dict[str, Any]:
    return {
        "time": {"fps": 1.0},
        "community": {
            "enabled": True,
            "window_s": window_s,
            "step_s": step_s,
            "min_visible_time_s": 0.0,
            "min_edges_per_window": 0,
        },
    }


def _trajectories(times: list[float], row_ids: list[str] | None = None) -> pd.DataFrame:
    if row_ids is None:
        row_ids = [f"t{index}" for index in range(len(times))]
    return pd.DataFrame(
        {
            "time_s": times,
            "cow_id": ["1"] * len(times),
            "row_id": row_ids,
        }
    )


def _interactions(times: list[float], row_ids: list[str] | None = None) -> pd.DataFrame:
    if row_ids is None:
        row_ids = [f"i{index}" for index in range(len(times))]
    return pd.DataFrame(
        {
            "time_s": times,
            "cow_i": ["1"] * len(times),
            "cow_j": ["1"] * len(times),
            "row_id": row_ids,
        }
    )


class SliceRecorder:
    def __init__(self) -> None:
        self.calls: list[tuple[pd.DataFrame, pd.DataFrame]] = []

    def __call__(
        self,
        trajectories: pd.DataFrame,
        interactions: pd.DataFrame,
        config: dict[str, Any],
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        del config
        self.calls.append((trajectories.copy(), interactions.copy()))
        return pd.DataFrame(columns=EDGE_COLUMNS), pd.DataFrame()


def _recorded_times(recorder: SliceRecorder, frame_index: int) -> list[list[float]]:
    return [call[frame_index]["time_s"].astype(float).tolist() for call in recorder.calls]


class CommunityTimeIndexTest(unittest.TestCase):
    def test_monotonic_time_index_reuses_the_input_frame(self) -> None:
        trajectories = _trajectories([0.0, 0.0, 1.0, 2.0])

        indexed, times = _time_indexed_frame(trajectories, "trajectory")

        self.assertIs(indexed, trajectories)
        self.assertEqual(times.tolist(), [0.0, 0.0, 1.0, 2.0])

    def test_non_finite_window_settings_are_rejected(self) -> None:
        trajectories = _trajectories([0.0])
        interactions = _interactions([])
        for window_s, step_s in ((np.nan, 5.0), (np.inf, 5.0), (5.0, np.nan), (5.0, np.inf)):
            with self.subTest(window_s=window_s, step_s=step_s):
                with self.assertRaisesRegex(ValueError, "finite and positive"):
                    compute_community_stability(
                        trajectories,
                        interactions,
                        _config(window_s=window_s, step_s=step_s),
                        SliceRecorder(),
                    )

    def test_non_finite_times_are_rejected(self) -> None:
        cases = (
            (_trajectories([0.0, np.nan]), _interactions([]), "trajectory"),
            (_trajectories([0.0, np.inf]), _interactions([]), "trajectory"),
            (_trajectories([0.0]), _interactions([np.nan]), "interaction"),
            (_trajectories([0.0]), _interactions([-np.inf]), "interaction"),
        )
        for trajectories, interactions, label in cases:
            with self.subTest(label=label):
                with self.assertRaisesRegex(ValueError, rf"{label}\.time_s.*finite"):
                    compute_community_stability(
                        trajectories,
                        interactions,
                        _config(window_s=5.0, step_s=5.0),
                        SliceRecorder(),
                    )

    def test_windows_are_half_open_at_exact_boundaries(self) -> None:
        trajectories = _trajectories([0.0, 4.999, 5.0, 9.999, 10.0])
        interactions = _interactions([0.0, 4.999, 5.0, 9.999, 10.0])
        recorder = SliceRecorder()

        windows, _, _ = compute_community_stability(
            trajectories,
            interactions,
            _config(window_s=5.0, step_s=5.0),
            recorder,
        )

        expected = [[0.0, 4.999], [5.0, 9.999], [10.0]]
        self.assertEqual(_recorded_times(recorder, 0), expected)
        self.assertEqual(_recorded_times(recorder, 1), expected)
        self.assertEqual(windows["window_index"].astype(int).tolist(), [0, 1, 2])
        self.assertEqual(windows["visible_time_s_in_window"].tolist(), [2.0, 2.0, 1.0])

    def test_empty_windows_remain_in_the_adjacent_window_sequence(self) -> None:
        trajectories = _trajectories([0.0, 20.0])
        interactions = _interactions([0.0, 20.0])
        recorder = SliceRecorder()

        windows, summary, _ = compute_community_stability(
            trajectories,
            interactions,
            _config(window_s=5.0, step_s=5.0),
            recorder,
        )

        self.assertEqual(_recorded_times(recorder, 0), [[0.0], [20.0]])
        self.assertEqual(windows["window_index"].astype(int).tolist(), [0, 4])
        self.assertEqual(
            list(zip(summary["window_a"].astype(int), summary["window_b"].astype(int))),
            [(0, 1), (1, 2), (2, 3), (3, 4)],
        )
        self.assertEqual(summary["n_common_cows"].astype(int).tolist(), [0, 0, 0, 0])

    def test_unsorted_inputs_are_stably_sorted_without_mutating_callers(self) -> None:
        trajectories = _trajectories(
            [2.0, 1.0, 2.0, 0.0],
            ["first-at-2", "at-1", "second-at-2", "at-0"],
        )
        interactions = _interactions(
            [1.0, 2.0, 1.0, 0.0],
            ["first-at-1", "at-2", "second-at-1", "at-0"],
        )
        original_trajectory_order = trajectories["row_id"].tolist()
        original_interaction_order = interactions["row_id"].tolist()
        recorder = SliceRecorder()

        compute_community_stability(
            trajectories,
            interactions,
            _config(window_s=10.0, step_s=10.0),
            recorder,
        )

        self.assertEqual(
            recorder.calls[0][0]["row_id"].tolist(),
            ["at-0", "at-1", "first-at-2", "second-at-2"],
        )
        self.assertEqual(
            recorder.calls[0][1]["row_id"].tolist(),
            ["at-0", "first-at-1", "second-at-1", "at-2"],
        )
        self.assertEqual(trajectories["row_id"].tolist(), original_trajectory_order)
        self.assertEqual(interactions["row_id"].tolist(), original_interaction_order)

    def test_overlapping_windows_include_rows_in_each_matching_window(self) -> None:
        trajectories = _trajectories([0.0, 5.0, 9.0, 10.0, 14.0])
        interactions = _interactions([0.0, 5.0, 9.0, 10.0, 14.0])
        recorder = SliceRecorder()

        compute_community_stability(
            trajectories,
            interactions,
            _config(window_s=10.0, step_s=5.0),
            recorder,
        )

        expected = [
            [0.0, 5.0, 9.0],
            [5.0, 9.0, 10.0, 14.0],
            [10.0, 14.0],
        ]
        self.assertEqual(_recorded_times(recorder, 0), expected)
        self.assertEqual(_recorded_times(recorder, 1), expected)

    def test_time_index_slices_match_stable_boolean_mask_reference(self) -> None:
        trajectories = _trajectories(
            [12.0, 0.0, 7.0, 3.0, 15.0, 7.0, 21.0],
            ["t12", "t0", "t7-first", "t3", "t15", "t7-second", "t21"],
        )
        interactions = _interactions(
            [21.0, 1.0, 8.0, 12.0, 7.0, 16.0],
            ["i21", "i1", "i8", "i12", "i7", "i16"],
        )
        recorder = SliceRecorder()
        window_s = 9.0
        step_s = 4.0

        windows, _, _ = compute_community_stability(
            trajectories,
            interactions,
            _config(window_s=window_s, step_s=step_s),
            recorder,
        )

        stable_trajectories = trajectories.sort_values("time_s", kind="stable")
        stable_interactions = interactions.sort_values("time_s", kind="stable")
        starts = [0.0, 4.0, 8.0, 12.0, 16.0, 20.0]
        expected_trajectory_ids: list[list[str]] = []
        expected_interaction_ids: list[list[str]] = []
        for start in starts:
            end = start + window_s
            trajectory_mask = stable_trajectories["time_s"].ge(start) & stable_trajectories[
                "time_s"
            ].lt(end)
            interaction_mask = stable_interactions["time_s"].ge(start) & stable_interactions[
                "time_s"
            ].lt(end)
            expected_trajectory_ids.append(
                stable_trajectories.loc[trajectory_mask, "row_id"].tolist()
            )
            expected_interaction_ids.append(
                stable_interactions.loc[interaction_mask, "row_id"].tolist()
            )

        self.assertEqual(
            [call[0]["row_id"].tolist() for call in recorder.calls],
            expected_trajectory_ids,
        )
        self.assertEqual(
            [call[1]["row_id"].tolist() for call in recorder.calls],
            expected_interaction_ids,
        )
        self.assertEqual(
            windows["visible_time_s_in_window"].tolist(),
            [float(len(items)) for items in expected_trajectory_ids],
        )


if __name__ == "__main__":
    unittest.main()
