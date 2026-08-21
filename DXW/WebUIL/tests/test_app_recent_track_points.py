from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import _RecentTrackPoints


def point(track_id: int, global_track_uuid: str) -> dict[str, Any]:
    return {"trackId": track_id, "globalTrackUuid": global_track_uuid}


class RecentTrackPointsTest(unittest.TestCase):
    def _advance(
        self,
        recent: _RecentTrackPoints,
        frame: int,
        points: list[dict[str, Any]],
    ) -> dict[str, tuple[int, int, dict[str, Any]]]:
        recent.begin_frame(frame)
        current_points = {int(item["trackId"]): item for item in points}
        current_global_tracks = {
            str(item["globalTrackUuid"]): int(item["trackId"]) for item in points
        }
        for track_id, item in sorted(current_points.items()):
            recent.remember(track_id, frame, item)
        return recent.frozen_candidates(frame, current_points, current_global_tracks)

    def _legacy_advance(
        self,
        last_seen: dict[int, tuple[int, dict[str, Any]]],
        freeze_frames: int,
        frame: int,
        points: list[dict[str, Any]],
    ) -> dict[str, tuple[int, int, dict[str, Any]]]:
        current_points = {int(item["trackId"]): item for item in points}
        current_global_tracks = {
            str(item["globalTrackUuid"]): int(item["trackId"]) for item in points
        }
        for track_id, item in sorted(current_points.items()):
            last_seen[track_id] = (frame, item)

        candidates: dict[str, tuple[int, int, dict[str, Any]]] = {}
        for track_id, (last_frame, last_point) in sorted(last_seen.items()):
            if track_id in current_points:
                continue
            age_frames = frame - last_frame
            if 0 < age_frames <= freeze_frames:
                global_track_uuid = str(last_point["globalTrackUuid"])
                if global_track_uuid in current_global_tracks:
                    continue
                previous = candidates.get(global_track_uuid)
                if previous is not None:
                    previous_track, previous_frame, _previous_point = previous
                    if previous_frame == last_frame and previous_track != track_id:
                        raise RuntimeError("ambiguous legacy fixture")
                    if previous_frame > last_frame:
                        continue
                candidates[global_track_uuid] = (track_id, last_frame, last_point)
        return candidates

    def test_freeze_window_is_inclusive_and_then_expires(self) -> None:
        recent = _RecentTrackPoints(freeze_frames=2)

        self.assertEqual(self._advance(recent, 10, [point(7, "global-a")]), {})
        frame_11 = self._advance(recent, 11, [])
        frame_12 = self._advance(recent, 12, [])
        self.assertEqual(frame_11["global-a"][:2], (7, 10))
        self.assertEqual(frame_12["global-a"][:2], (7, 10))
        self.assertEqual(self._advance(recent, 13, []), {})
        self.assertEqual(recent.active_track_count, 0)

    def test_newest_local_track_wins_and_current_global_suppresses_freeze(self) -> None:
        recent = _RecentTrackPoints(freeze_frames=3)

        self._advance(recent, 0, [point(1, "global-a")])
        self.assertEqual(self._advance(recent, 1, [point(2, "global-a")]), {})
        candidates = self._advance(recent, 2, [])

        self.assertEqual(candidates["global-a"][:2], (2, 1))

    def test_candidates_match_the_unbounded_legacy_algorithm(self) -> None:
        freeze_frames = 3
        recent = _RecentTrackPoints(freeze_frames=freeze_frames)
        legacy_last_seen: dict[int, tuple[int, dict[str, Any]]] = {}
        timeline = [
            [point(1, "global-a"), point(10, "global-b")],
            [point(1, "global-a")],
            [point(2, "global-a"), point(10, "global-b")],
            [],
            [point(3, "global-a"), point(11, "global-b")],
            [point(11, "global-b"), point(20, "global-c")],
            [],
            [],
            [point(20, "global-c")],
            [],
        ]

        for frame, points in enumerate(timeline):
            actual = self._advance(recent, frame, points)
            expected = self._legacy_advance(
                legacy_last_seen,
                freeze_frames,
                frame,
                points,
            )
            actual_keys = {
                global_id: (track_id, last_frame)
                for global_id, (track_id, last_frame, _point) in actual.items()
            }
            expected_keys = {
                global_id: (track_id, last_frame)
                for global_id, (track_id, last_frame, _point) in expected.items()
            }
            self.assertEqual(actual_keys, expected_keys, f"frame={frame}")

        self.assertLess(recent.active_track_count, len(legacy_last_seen))

    def test_equal_last_seen_frames_remain_an_error(self) -> None:
        recent = _RecentTrackPoints(freeze_frames=2)
        recent.begin_frame(0)
        recent.remember(3, 0, point(3, "global-a"))
        recent.remember(9, 0, point(9, "global-a"))

        recent.begin_frame(1)
        with self.assertRaisesRegex(
            RuntimeError,
            r"frame=1, last_seen=0, global=global-a, local=3/9",
        ):
            recent.frozen_candidates(1, {}, {})

    def test_fragmented_track_history_stays_bounded_by_freeze_window(self) -> None:
        freeze_frames = 4
        recent = _RecentTrackPoints(freeze_frames=freeze_frames)

        for frame in range(1_000):
            self._advance(recent, frame, [point(frame, f"global-{frame}")])
            self.assertLessEqual(recent.active_track_count, freeze_frames + 1)


if __name__ == "__main__":
    unittest.main()
