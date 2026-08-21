from __future__ import annotations

import sys
import unittest
from collections import defaultdict, deque
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stage2_runtime import Stage2Runtime


class _ItemsForbiddenDefaultDict(defaultdict):
    def items(self):
        raise AssertionError("the full historical pair dictionary was scanned")


def _event(
    *,
    active: bool = False,
    label: str = "friendly",
    start_frame: int | None = None,
    last_positive_frame: int = -1,
) -> dict:
    return {
        "active": active,
        "cur_label": label,
        "start_f": start_frame,
        "last_pos_frame": last_positive_frame,
        "cur_stage2_label": label,
        "conf_stage1_max": 0.81,
        "conf_stage2_max": 0.72,
        "conf_valence_max": 0.63,
        "conf_friendly_max": 0.54,
        "conf_unfriendly_max": 0.09,
    }


def _runtime_for_events() -> Stage2Runtime:
    runtime = Stage2Runtime.__new__(Stage2Runtime)
    runtime.gap_tol_fr = 2
    runtime.video_name = "sample.mp4"
    runtime.recent_event_keys = set()
    runtime.event_expiry_frame = {}
    runtime.event_expiry_keys = defaultdict(set)

    def reset_stage2_event(evt: dict) -> None:
        evt["active"] = False
        evt["cur_label"] = None
        evt["start_f"] = None
        evt["last_pos_frame"] = -1

    runtime.s2 = SimpleNamespace(reset_stage2_event=reset_stage2_event)
    runtime.pair_evt = _ItemsForbiddenDefaultDict(lambda: _event())
    return runtime


class Stage2RuntimePairStateTest(unittest.TestCase):
    def test_raw_output_reads_only_recent_event_index(self) -> None:
        runtime = _runtime_for_events()
        for index in range(10_000):
            runtime.pair_evt[(index, index + 20_000)] = _event()

        friendly_key = (2, 7)
        unfriendly_key = (3, 8)
        stale_key = (4, 9)
        runtime.pair_evt[friendly_key] = _event(
            active=True,
            label="friendly",
            start_frame=6,
            last_positive_frame=10,
        )
        runtime.pair_evt[unfriendly_key] = _event(
            active=True,
            label="unfriendly",
            start_frame=7,
            last_positive_frame=11,
        )
        runtime.pair_evt[stale_key] = _event(
            active=True,
            label="friendly",
            start_frame=1,
            last_positive_frame=8,
        )
        runtime.recent_event_keys.update({friendly_key, unfriendly_key, stale_key})

        actual = runtime._raw_active_interactions_unlocked(11)

        self.assertEqual([(row["class"], row["tidA"], row["tidB"]) for row in actual], [
            ("friendly", 2, 7),
            ("unfriendly", 3, 8),
        ])
        self.assertEqual(actual[0]["startFrame"], 6)
        self.assertEqual(actual[0]["lastPositiveFrame"], 10)

    def test_expiry_hides_absent_event_but_preserves_reappearance_state(self) -> None:
        runtime = _runtime_for_events()
        key = (11, 12)
        evt = _event(active=True, label="friendly", start_frame=4, last_positive_frame=10)
        runtime.pair_evt[key] = evt

        runtime._index_recent_event_unlocked(key, evt)
        self.assertEqual(runtime.event_expiry_frame[key], 13)
        self.assertEqual(len(runtime._raw_active_interactions_unlocked(12)), 1)

        runtime._expire_recent_events_unlocked(13)
        self.assertNotIn(key, runtime.recent_event_keys)
        self.assertTrue(evt["active"])
        self.assertEqual(evt["start_f"], 4)

        evt["last_pos_frame"] = 20
        runtime._index_recent_event_unlocked(key, evt)
        reappeared = runtime._raw_active_interactions_unlocked(20)
        self.assertEqual(len(reappeared), 1)
        self.assertEqual(reappeared[0]["startFrame"], 4)
        self.assertEqual(reappeared[0]["lastPositiveFrame"], 20)

        runtime._finish_event_unlocked(key, evt)
        self.assertNotIn(key, runtime.recent_event_keys)
        self.assertNotIn(key, runtime.event_expiry_frame)
        self.assertFalse(evt["active"])

    def test_old_event_expiry_bucket_does_not_override_refreshed_deadline(self) -> None:
        runtime = _runtime_for_events()
        key = (13, 14)
        evt = _event(active=True, label="friendly", start_frame=4, last_positive_frame=10)
        runtime.pair_evt[key] = evt

        runtime._index_recent_event_unlocked(key, evt)
        self.assertEqual(runtime.event_expiry_frame[key], 13)

        evt["last_pos_frame"] = 11
        runtime._index_recent_event_unlocked(key, evt)
        self.assertEqual(runtime.event_expiry_frame[key], 14)

        runtime._expire_recent_events_unlocked(13)
        self.assertIn(key, runtime.recent_event_keys)
        self.assertEqual(runtime.event_expiry_frame[key], 14)
        self.assertEqual(len(runtime._raw_active_interactions_unlocked(13)), 1)

        runtime._expire_recent_events_unlocked(14)
        self.assertNotIn(key, runtime.recent_event_keys)
        self.assertNotIn(key, runtime.event_expiry_frame)

    def test_geometry_aging_reads_only_live_pair_index(self) -> None:
        runtime = Stage2Runtime.__new__(Stage2Runtime)
        runtime.gap_tol_fr = 2
        runtime.pair_geom = _ItemsForbiddenDefaultDict(runtime._make_pair_geom)
        runtime.pair_prox_hist = {}
        runtime.pair_buf = {}
        runtime.expired_pair_geom_keys = set()

        historical_keys = [(index, index + 30_000) for index in range(10_000)]
        for key in historical_keys:
            runtime.pair_geom[key] = {
                "suspect_frames": 0,
                "miss": 99,
                "prox": None,
                "last_ok_frame": 0,
            }

        suspect_key = (1, 2)
        aging_key = (3, 4)
        capped_key = (5, 6)
        runtime.pair_geom[suspect_key] = {
            "suspect_frames": 8,
            "miss": 0,
            "prox": 0.2,
            "last_ok_frame": 12,
        }
        runtime.pair_geom[aging_key] = {
            "suspect_frames": 7,
            "miss": 1,
            "prox": 0.3,
            "last_ok_frame": 11,
        }
        runtime.pair_geom[capped_key] = {
            "suspect_frames": 1,
            "miss": 0,
            "prox": 0.4,
            "last_ok_frame": 12,
        }
        runtime.pair_prox_hist.update({aging_key: object(), capped_key: object()})
        runtime.pair_buf.update({aging_key: object(), capped_key: object()})
        runtime.live_pair_geom_keys = {suspect_key, aging_key, capped_key}

        runtime._age_pair_geometry_unlocked({suspect_key})
        self.assertEqual(runtime.pair_geom[suspect_key]["miss"], 0)
        self.assertEqual(runtime.pair_geom[aging_key]["miss"], 2)
        self.assertEqual(runtime.pair_geom[capped_key]["miss"], 1)
        self.assertEqual(runtime.pair_geom[historical_keys[0]]["miss"], 99)

        runtime._age_pair_geometry_unlocked({suspect_key})
        self.assertNotIn(aging_key, runtime.live_pair_geom_keys)
        self.assertNotIn(aging_key, runtime.pair_geom)
        self.assertNotIn(aging_key, runtime.pair_prox_hist)
        self.assertNotIn(aging_key, runtime.pair_buf)
        self.assertIn(aging_key, runtime.expired_pair_geom_keys)
        self.assertIn(capped_key, runtime.live_pair_geom_keys)

        runtime.expired_pair_geom_keys.discard(aging_key)
        reappeared = runtime.pair_geom[aging_key]
        reappeared["suspect_frames"] += 1
        reappeared["miss"] = 0
        reappeared["last_ok_frame"] = 20
        reappeared["prox"] = 0.1
        self.assertEqual(reappeared, {
            "suspect_frames": 1,
            "miss": 0,
            "prox": 0.1,
            "last_ok_frame": 20,
        })

    def test_expired_geometry_tombstone_blocks_proximity_history_until_stable(self) -> None:
        runtime = Stage2Runtime.__new__(Stage2Runtime)
        key = (7, 8)
        never_stable_key = (9, 10)
        runtime.expired_pair_geom_keys = {key}
        runtime.pair_prox_hist = {never_stable_key: {1.0: [0.95]}}

        runtime._resolve_expired_geometry_reappearance_unlocked(
            never_stable_key,
            stable_gate=False,
        )
        self.assertEqual(runtime.pair_prox_hist[never_stable_key][1.0], [0.95])

        for value in (0.9, 0.8, 0.7):
            runtime.pair_prox_hist[key] = {1.0: [value]}
            runtime._resolve_expired_geometry_reappearance_unlocked(key, stable_gate=False)
            self.assertNotIn(key, runtime.pair_prox_hist)

        runtime.pair_prox_hist[key] = {1.0: [0.6]}
        runtime._resolve_expired_geometry_reappearance_unlocked(key, stable_gate=True)
        self.assertEqual(runtime.pair_prox_hist[key][1.0], [0.6])
        self.assertNotIn(key, runtime.expired_pair_geom_keys)

    def test_step_applies_expired_geometry_tombstone_on_reappearance(self) -> None:
        runtime = Stage2Runtime.__new__(Stage2Runtime)
        key = (7, 8)

        class Box:
            def __init__(self, tid: int) -> None:
                self.tid = tid
                self.center = (float(tid), 0.0)
                self.area_scale = 1.0

        s2 = SimpleNamespace(
            np=SimpleNamespace(),
            INTERACT_DIAG_SIM_RATIO=0.5,
            PROX_HARD_CAP=0.1,
            Q_WINDOWS_SEC=(1.0,),
            Q_PROX=0.5,
            GATE_STABLE_K=2,
            MAX_SUSPECT_PAIRS=12,
            boxes_overlap=lambda _a, _b: (True, 0.0),
            normalized_center_distance=lambda _a, _b: (0.5, 1.0, 1.0),
        )
        runtime.s2 = s2
        runtime.fps = 10.0
        runtime.gap_tol_fr = 2
        runtime.classify_every = 100
        runtime.cooldown_frames = 5
        runtime.video_name = "sample.mp4"
        runtime.boxes_by_frame = {
            0: {7: Box(7), 8: Box(8)},
            1: {7: Box(7), 8: Box(8)},
        }
        runtime.track_last_frame = {7: 10, 8: 10}
        runtime.kpts_by_key = {}
        runtime.pair_geom = defaultdict(runtime._make_pair_geom)
        runtime.pair_prox_hist = defaultdict(
            lambda: {1.0: deque(maxlen=10)}
        )
        runtime.pair_buf = defaultdict(
            lambda: {
                name: deque(maxlen=10)
                for name in ("frames", "kptsA", "kptsB", "centerA", "centerB", "scaleA", "scaleB")
            }
        )
        runtime.pair_evt = defaultdict(
            lambda: {
                "gate_hist": deque(maxlen=8),
                "active": False,
                "last_pos_frame": -1,
            }
        )
        runtime.live_pair_geom_keys = set()
        runtime.expired_pair_geom_keys = {key}
        runtime.recent_event_keys = set()
        runtime.event_expiry_frame = {}
        runtime.event_expiry_keys = defaultdict(set)
        runtime.pair_terminal_expiry_frame = {}
        runtime.pair_terminal_expiry_keys = defaultdict(set)
        runtime.pair_cooldown_until = {}
        runtime.line_window_frames = 10
        runtime.line_threshold_frames = 5
        runtime.line_history = {}
        runtime.line_visible = set()
        runtime.line_last_item = {}
        runtime.raw_frame_interactions = {}
        runtime.raw_frame_geometry_candidates = {}
        runtime._classify_unlocked = lambda _frame, _pairs: None

        runtime._step_unlocked(0)
        self.assertIn(key, runtime.expired_pair_geom_keys)
        self.assertNotIn(key, runtime.pair_prox_hist)
        self.assertEqual(runtime.raw_frame_geometry_candidates[0], set())

        s2.PROX_HARD_CAP = 1.0
        s2.GATE_STABLE_K = 1
        runtime._step_unlocked(1)
        self.assertNotIn(key, runtime.expired_pair_geom_keys)
        self.assertEqual(list(runtime.pair_prox_hist[key][1.0]), [0.5])
        self.assertEqual(runtime.raw_frame_geometry_candidates[1], {key})
        self.assertIn(key, runtime.live_pair_geom_keys)

    def test_terminal_expiry_removes_state_only_after_reappearance_is_impossible(self) -> None:
        runtime = _runtime_for_events()
        key = (21, 22)
        runtime.track_last_frame = {21: 10, 22: 30}
        runtime.pair_terminal_expiry_frame = {}
        runtime.pair_terminal_expiry_keys = defaultdict(set)
        runtime.live_pair_geom_keys = {key}
        runtime.expired_pair_geom_keys = {key}
        runtime.pair_geom = {key: object()}
        runtime.pair_prox_hist = {key: object()}
        runtime.pair_buf = {key: object()}
        runtime.pair_cooldown_until = {key: 99}
        evt = _event(active=True, label="friendly", start_frame=4, last_positive_frame=10)
        runtime.pair_evt[key] = evt
        runtime._index_recent_event_unlocked(key, evt)

        runtime._index_terminal_pair_expiry_unlocked(key)
        self.assertEqual(runtime.pair_terminal_expiry_frame[key], 13)

        runtime._expire_terminal_pair_states_unlocked(12)
        self.assertIn(key, runtime.pair_evt)
        self.assertIn(key, runtime.recent_event_keys)

        runtime._expire_terminal_pair_states_unlocked(13)
        self.assertNotIn(key, runtime.pair_terminal_expiry_frame)
        self.assertNotIn(key, runtime.live_pair_geom_keys)
        self.assertNotIn(key, runtime.expired_pair_geom_keys)
        self.assertNotIn(key, runtime.recent_event_keys)
        self.assertNotIn(key, runtime.event_expiry_frame)
        self.assertNotIn(key, runtime.pair_geom)
        self.assertNotIn(key, runtime.pair_prox_hist)
        self.assertNotIn(key, runtime.pair_buf)
        self.assertNotIn(key, runtime.pair_evt)
        self.assertNotIn(key, runtime.pair_cooldown_until)


if __name__ == "__main__":
    unittest.main()
