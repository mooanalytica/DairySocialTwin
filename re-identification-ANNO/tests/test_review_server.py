from __future__ import annotations

import csv
import hashlib
import io
import json
import unittest
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import review_server as server


def occurrence_row() -> dict[str, str]:
    return {
        "occurrence_id": "O000001", "clip_order": "0", "clip_id": "GX040006",
        "display_global_id": "G0002", "legacy_track_id": "2",
        "start_frame": "10", "end_frame": "12", "start_time_sec": "0.333666667",
        "end_time_sec_inclusive": "0.400400000", "end_time_sec_exclusive": "0.433766667",
        "span_duration_sec": "0.100100000", "num_valid_detections": "3",
        "num_missing_frames": "0", "max_internal_gap_missing_frames": "0",
        "start_det_id": "11", "end_det_id": "13",
    }


class Fixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.occurrences = root / "occurrence_segments.csv"
        self.cache = root / "cached_clips"
        self.cache.mkdir()
        self.clip = self.cache / "O000001.mp4"
        self.clip.write_bytes(b"0123456789")
        self.manifest = self.cache / "clip_manifest.json"
        self.reviews = root / "occurrence_reviews.csv"
        self.web = root / "web"
        self.web.mkdir()
        (self.web / "index.html").write_text("INDEX", encoding="utf-8")
        (self.web / "app.js").write_text("APP", encoding="utf-8")
        with self.occurrences.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=server.SOURCE_FIELDS, lineterminator="\n")
            writer.writeheader()
            writer.writerow(occurrence_row())
        item = {
            "occurrence_id": "O000001", "relative_path": "O000001.mp4",
            "status": "complete", "clip_id": "GX040006", "anchor_frame": 10,
            "display_global_id": "G0002", "legacy_track_id": "2",
            "window_start_frame": 0, "window_end_frame": 899, "frame_count": 900,
            "anchor_offset_frame": 10,
            "red_box": {"x": 100, "y": 100, "width": 200, "height": 200},
            "size_bytes": 10, "sha256": hashlib.sha256(b"0123456789").hexdigest(),
        }
        self.manifest.write_text(json.dumps({"schema_version": "1.0", "clips": [item]}), encoding="utf-8")

    def app(self) -> server.ReviewApplication:
        return server.create_application(
            occurrences_path=self.occurrences, manifest_path=self.manifest,
            reviews_path=self.reviews, web_root=self.web,
            clock=lambda: datetime(2026, 7, 21, 12, 0, tzinfo=timezone.utc),
            expected_occurrence_count=1,
        )


class ReviewServerTests(unittest.TestCase):
    def test_playback_starts_three_seconds_before_occurrence(self) -> None:
        self.assertEqual(server.playback_start_seconds(0), 0.0)
        self.assertEqual(server.playback_start_seconds(89), 0.0)
        self.assertAlmostEqual(server.playback_start_seconds(450), 12.015)
        self.assertAlmostEqual(
            server.playback_start_seconds(695),
            695 * 1_001 / 30_000 - 3,
        )

    def test_actions_normalize_and_edit_atomically(self) -> None:
        with TemporaryDirectory() as directory:
            fixture = Fixture(Path(directory))
            app = fixture.app()
            first = app.submit("O000001", {"action": "update_id", "new_id": "10"})
            self.assertEqual(first["review"]["reviewed_global_id"], "G0010")
            edited = app.submit("O000001", {"action": "invalid_multiple_cows"})
            self.assertTrue(edited["review"]["invalid_multiple_cows"])
            with fixture.reviews.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["reviewed_global_id"], "G0002")
            self.assertEqual(rows[0]["invalid_multiple_cows"], "true")

    def test_rejects_invalid_id_and_manifest_identity(self) -> None:
        with TemporaryDirectory() as directory:
            fixture = Fixture(Path(directory))
            app = fixture.app()
            for value in ("0", "63", "G0001", 1.5, True):
                with self.subTest(value=value), self.assertRaises(server.ContractError):
                    app.submit("O000001", {"action": "update_id", "new_id": value})
            payload = json.loads(fixture.manifest.read_text(encoding="utf-8"))
            payload["clips"][0]["occurrence_id"] = "O000002"
            fixture.manifest.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(server.ContractError):
                server.create_application(
                    occurrences_path=fixture.occurrences, manifest_path=fixture.manifest,
                    reviews_path=fixture.reviews, web_root=fixture.web,
                    expected_occurrence_count=1,
                )

    def test_manifest_anchor_may_move_inside_occurrence_only(self) -> None:
        with TemporaryDirectory() as directory:
            fixture = Fixture(Path(directory))
            payload = json.loads(fixture.manifest.read_text(encoding="utf-8"))
            item = payload["clips"][0]
            item["anchor_frame"] = 12
            item["anchor_offset_frame"] = 12
            fixture.manifest.write_text(json.dumps(payload), encoding="utf-8")
            fixture.app()

            item["anchor_frame"] = 13
            item["anchor_offset_frame"] = 13
            fixture.manifest.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(server.ContractError, "outside occurrence"):
                fixture.app()

    def test_failed_persistence_rolls_back_memory_state(self) -> None:
        with TemporaryDirectory() as directory:
            fixture = Fixture(Path(directory))
            app = fixture.app()
            with patch.object(
                app.store, "_write_locked", side_effect=OSError("disk failure")
            ):
                with self.assertRaises(OSError):
                    app.submit("O000001", {"action": "accept"})
            self.assertEqual(app.store.snapshot(), {})
            self.assertIsNone(app.state()["occurrences"][0]["review"])

    def test_export_lock_rejects_review_before_memory_changes(self) -> None:
        with TemporaryDirectory() as directory:
            fixture = Fixture(Path(directory))
            app = fixture.app()
            with server.ReviewFileLock(app.store.file_lock_path, blocking=False):
                with self.assertRaises(server.ReviewLockedError):
                    app.submit("O000001", {"action": "accept"})
            self.assertEqual(app.store.snapshot(), {})
            self.assertFalse(fixture.reviews.exists())

    def test_pending_clip_cannot_be_reviewed(self) -> None:
        with TemporaryDirectory() as directory:
            fixture = Fixture(Path(directory))
            payload = json.loads(fixture.manifest.read_text(encoding="utf-8"))
            item = payload["clips"][0]
            item["status"] = "pending"
            item.pop("size_bytes")
            item.pop("sha256")
            fixture.manifest.write_text(json.dumps(payload), encoding="utf-8")
            app = fixture.app()
            self.assertFalse(app.state()["occurrences"][0]["clip_ready"])
            with self.assertRaisesRegex(server.ContractError, "not complete"):
                app.submit("O000001", {"action": "accept"})

    def test_state_submit_and_mocked_single_range(self) -> None:
        with TemporaryDirectory() as directory:
            fixture = Fixture(Path(directory))
            app = fixture.app()
            state = app.state()
            self.assertIsNone(state["occurrences"][0]["review"])
            self.assertEqual(state["occurrences"][0]["playback_start_sec"], 0.0)
            result = app.submit("O000001", {"action": "accept"})
            self.assertEqual(result["review"]["reviewed_global_id"], "G0002")

            handler = object.__new__(server.ReviewRequestHandler)
            handler.headers = {"Range": "bytes=2-5"}
            handler.wfile = io.BytesIO()
            status: list[int] = []
            headers: dict[str, str] = {}
            handler.send_response = lambda value: status.append(int(value))
            handler.send_header = lambda key, value: headers.__setitem__(key, value)
            handler.end_headers = lambda: None
            handler._file(fixture.clip, "video/mp4", send_body=True, allow_range=True)
            self.assertEqual((status, handler.wfile.getvalue()), ([206], b"2345"))
            self.assertEqual(headers["Content-Range"], "bytes 2-5/10")

    def test_range_parser(self) -> None:
        self.assertEqual(server.parse_single_range("bytes=3-", 10), (3, 9))
        self.assertEqual(server.parse_single_range("bytes=-4", 10), (6, 9))
        with self.assertRaises(server.RangeNotSatisfiable):
            server.parse_single_range("bytes=1-2,4-5", 10)


if __name__ == "__main__":
    unittest.main()
