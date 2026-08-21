from __future__ import annotations

import csv
import unittest
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

import export_corrected_results as exporter


def detection_row(
    *,
    det_id: int,
    csv_row_index: int,
    legacy_track_id: int,
    gid: int | None,
    local_frame: int,
    stable_id: int | None,
    valid: bool = True,
) -> dict[str, str]:
    display = "" if gid is None else f"G{gid + 1:04d}"
    uuid = "" if gid is None else f"uuid-{gid + 1}"
    return {
        "sequence_id": "dairy_farm_1_gopro1_20250505",
        "clip_id": "GX040006",
        "clip_order": "0",
        "det_id": str(det_id),
        "csv_row_index": str(csv_row_index),
        "legacy_track_id": str(legacy_track_id),
        "global_track_id": "" if gid is None else str(gid),
        "global_track_uuid": uuid,
        "display_global_id": display,
        "id_status": "" if gid is None else "forced_provisional",
        "identity_basis": "" if gid is None else "operator_forced",
        "local_frame": str(local_frame),
        "global_frame": str(local_frame),
        "global_time_sec": str(local_frame / 30),
        "x1": "100",
        "y1": "200",
        "x2": "500",
        "y2": "700",
        "bbox_confidence": "0.9",
        "valid": "true" if valid else "false",
        "qa_flags": "0" if valid else "8",
        "invalid_reason": "" if valid else "source_bbox_invalid",
        "micro_id": "" if stable_id is None else str(stable_id),
        "stable_id": "" if stable_id is None else str(stable_id),
        "order_in_micro": "" if stable_id is None else "0",
        "order_in_stable": "" if stable_id is None else "0",
        "order_in_stable_detection": "" if stable_id is None else "0",
        "order_in_global_stable": "" if gid is None else "99",
        "order_in_global_detection": "" if gid is None else "99",
        "local_purity_score": "0.8" if valid else "",
        "assignment_confidence": "",
        "incoming_link_probability": "",
        "outgoing_link_probability": "",
    }


def occurrence_row(
    *, occurrence_id: str, legacy: int, gid: int, frame: int, det_id: int
) -> dict[str, str]:
    seconds = f"{frame / 30:.9f}"
    return {
        "occurrence_id": occurrence_id,
        "clip_order": "0",
        "clip_id": "GX040006",
        "display_global_id": f"G{gid + 1:04d}",
        "legacy_track_id": str(legacy),
        "start_frame": str(frame),
        "end_frame": str(frame),
        "start_time_sec": seconds,
        "end_time_sec_inclusive": seconds,
        "end_time_sec_exclusive": seconds,
        "span_duration_sec": "0.033366667",
        "num_valid_detections": "1",
        "num_missing_frames": "0",
        "max_internal_gap_missing_frames": "0",
        "start_det_id": str(det_id),
        "end_det_id": str(det_id),
    }


class MappingFixture:
    def __init__(self, root: Path) -> None:
        self.occurrences_path = root / "occurrences.csv"
        self.reviews_path = root / "reviews.csv"
        self.source_path = root / "source.csv"
        self.output_path = root / "output.csv"
        occurrences = [
            occurrence_row(
                occurrence_id="O000001", legacy=1, gid=0, frame=0, det_id=20
            ),
            occurrence_row(
                occurrence_id="O000002", legacy=2, gid=1, frame=0, det_id=10
            ),
            occurrence_row(
                occurrence_id="O000003", legacy=3, gid=2, frame=1, det_id=30
            ),
        ]
        actions = [
            ("accept", "G0001", "false"),
            ("update_id", "G0003", "false"),
            ("invalid_multiple_cows", "G0003", "true"),
        ]
        with self.occurrences_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=exporter.SOURCE_FIELDS)
            writer.writeheader()
            writer.writerows(occurrences)
        with self.reviews_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=exporter.SOURCE_FIELDS + exporter.REVIEW_FIELDS,
            )
            writer.writeheader()
            for occurrence, (action, reviewed_gid, invalid) in zip(
                occurrences, actions, strict=True
            ):
                writer.writerow(
                    occurrence
                    | {
                        "review_action": action,
                        "reviewed_global_id": reviewed_gid,
                        "invalid_multiple_cows": invalid,
                        "reviewed_at_utc": "2026-07-21T12:00:00Z",
                    }
                )
        self.source_rows = [
            detection_row(
                det_id=20,
                csv_row_index=0,
                legacy_track_id=1,
                gid=0,
                local_frame=0,
                stable_id=0,
            ),
            detection_row(
                det_id=10,
                csv_row_index=1,
                legacy_track_id=2,
                gid=1,
                local_frame=0,
                stable_id=1,
            ),
            detection_row(
                det_id=30,
                csv_row_index=2,
                legacy_track_id=3,
                gid=2,
                local_frame=1,
                stable_id=2,
            ),
            detection_row(
                det_id=40,
                csv_row_index=3,
                legacy_track_id=4,
                gid=None,
                local_frame=1,
                stable_id=None,
                valid=False,
            ),
        ]
        with self.source_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=exporter.DETECTION_FIELDS)
            writer.writeheader()
            writer.writerows(self.source_rows)


class ExportCorrectedResultsTests(unittest.TestCase):
    @staticmethod
    def _loaded_fixture(
        fixture: MappingFixture,
    ) -> tuple[
        tuple[exporter.OccurrenceReview, ...],
        dict[tuple[str, str], exporter.TrackIntervals],
    ]:
        return exporter.load_reviews(
            fixture.occurrences_path,
            fixture.reviews_path,
            expected_count=3,
            expected_actions={
                "accept": 1,
                "update_id": 1,
                "invalid_multiple_cows": 1,
            },
        )

    @staticmethod
    def _gid_mapping() -> dict[str, tuple[int, str]]:
        return {
            "G0001": (0, "uuid-1"),
            "G0002": (1, "uuid-2"),
            "G0003": (2, "uuid-3"),
        }

    def test_all_actions_map_all_identity_fields_and_recompute_orders(self) -> None:
        with TemporaryDirectory() as directory:
            fixture = MappingFixture(Path(directory))
            reviews, intervals = self._loaded_fixture(fixture)
            stats, render_index = exporter.transform_detections(
                fixture.source_path,
                intervals,
                reviews,
                self._gid_mapping(),
                output_path=fixture.output_path,
                build_render_index=True,
                expected_rows=4,
                expected_valid_rows=3,
                expected_original_invalid_rows=1,
                expected_action_detections={
                    "accept": 1,
                    "update_id": 1,
                    "invalid_multiple_cows": 1,
                },
            )
            with fixture.output_path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                self.assertEqual(tuple(reader.fieldnames or ()), exporter.DETECTION_FIELDS)
                rows = list(reader)

            accepted, updated, reviewed_invalid, source_invalid = rows
            self.assertEqual(
                (
                    accepted["global_track_id"],
                    accepted["display_global_id"],
                    accepted["order_in_global_stable"],
                    accepted["order_in_global_detection"],
                ),
                ("0", "G0001", "0", "0"),
            )
            self.assertEqual(
                (
                    updated["global_track_id"],
                    updated["global_track_uuid"],
                    updated["display_global_id"],
                    updated["order_in_global_stable"],
                    updated["order_in_global_detection"],
                ),
                ("2", "uuid-3", "G0003", "0", "0"),
            )
            self.assertEqual(updated["id_status"], "forced_provisional")
            self.assertEqual(updated["identity_basis"], "operator_forced")
            self.assertEqual(
                (
                    reviewed_invalid["global_track_id"],
                    reviewed_invalid["global_track_uuid"],
                    reviewed_invalid["display_global_id"],
                    reviewed_invalid["id_status"],
                    reviewed_invalid["identity_basis"],
                    reviewed_invalid["valid"],
                    reviewed_invalid["qa_flags"],
                    reviewed_invalid["invalid_reason"],
                    reviewed_invalid["stable_id"],
                    reviewed_invalid["order_in_global_stable"],
                    reviewed_invalid["order_in_global_detection"],
                ),
                ("-1", "", "-1", "", "", "true", "0", "multiple_cows", "2", "", ""),
            )
            self.assertEqual(source_invalid, fixture.source_rows[3])
            self.assertEqual(stats.total_rows, 4)
            self.assertEqual(stats.valid_rows, 3)
            self.assertEqual(stats.original_invalid_rows, 1)
            assert render_index is not None
            self.assertEqual(len(render_index["GX040006"][0]), 2)
            self.assertTrue(render_index["GX040006"][1][0].reviewed_invalid)

    def test_rejects_corrected_same_frame_gid_collision_without_partial_csv(self) -> None:
        with TemporaryDirectory() as directory:
            fixture = MappingFixture(Path(directory))
            with fixture.reviews_path.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            rows[1]["reviewed_global_id"] = "G0001"
            with fixture.reviews_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=exporter.SOURCE_FIELDS + exporter.REVIEW_FIELDS,
                )
                writer.writeheader()
                writer.writerows(rows)
            reviews, intervals = self._loaded_fixture(fixture)
            with self.assertRaisesRegex(
                exporter.ContractError, "corrected same-frame G-ID collision"
            ):
                exporter.transform_detections(
                    fixture.source_path,
                    intervals,
                    reviews,
                    self._gid_mapping(),
                    output_path=fixture.output_path,
                    build_render_index=False,
                    expected_rows=None,
                    expected_valid_rows=None,
                    expected_original_invalid_rows=None,
                    expected_action_detections=None,
                )
            self.assertFalse(fixture.output_path.exists())

    def test_rejects_source_gid_uuid_inconsistency(self) -> None:
        with TemporaryDirectory() as directory:
            fixture = MappingFixture(Path(directory))
            with fixture.source_path.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            rows[0]["global_track_uuid"] = "wrong"
            with fixture.source_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=exporter.DETECTION_FIELDS)
                writer.writeheader()
                writer.writerows(rows)
            reviews, intervals = self._loaded_fixture(fixture)
            with self.assertRaisesRegex(exporter.ContractError, "ID/UUID"):
                exporter.transform_detections(
                    fixture.source_path,
                    intervals,
                    reviews,
                    self._gid_mapping(),
                    output_path=None,
                    build_render_index=False,
                    expected_rows=None,
                    expected_valid_rows=None,
                    expected_original_invalid_rows=None,
                    expected_action_detections=None,
                )

    def test_rejects_duplicate_det_id_across_frames(self) -> None:
        with TemporaryDirectory() as directory:
            fixture = MappingFixture(Path(directory))
            with fixture.source_path.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            rows[2]["det_id"] = rows[0]["det_id"]
            with fixture.source_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=exporter.DETECTION_FIELDS)
                writer.writeheader()
                writer.writerows(rows)
            reviews, intervals = self._loaded_fixture(fixture)
            with self.assertRaisesRegex(exporter.ContractError, "duplicate source det_id"):
                exporter.transform_detections(
                    fixture.source_path,
                    intervals,
                    reviews,
                    self._gid_mapping(),
                    output_path=None,
                    build_render_index=False,
                    expected_rows=None,
                    expected_valid_rows=None,
                    expected_original_invalid_rows=None,
                    expected_action_detections=None,
                )

    def test_failed_post_validation_removes_partial_csv(self) -> None:
        with TemporaryDirectory() as directory:
            fixture = MappingFixture(Path(directory))
            reviews, intervals = self._loaded_fixture(fixture)
            with self.assertRaisesRegex(exporter.ContractError, "source row count"):
                exporter.transform_detections(
                    fixture.source_path,
                    intervals,
                    reviews,
                    self._gid_mapping(),
                    output_path=fixture.output_path,
                    build_render_index=False,
                    expected_rows=999,
                    expected_valid_rows=3,
                    expected_original_invalid_rows=1,
                    expected_action_detections={
                        "accept": 1,
                        "update_id": 1,
                        "invalid_multiple_cows": 1,
                    },
                )
            self.assertFalse(fixture.output_path.exists())

    def test_renderer_draws_corrected_label_and_unlabelled_gray_dashes(self) -> None:
        frame = np.zeros((exporter.RAW_HEIGHT, exporter.RAW_WIDTH, 3), dtype=np.uint8)
        detections = [
            exporter.RenderDetection(
                1, 100, 100, 300, 300, 0, "G0001", False, "accept"
            ),
            exporter.RenderDetection(
                2,
                400,
                400,
                800,
                800,
                None,
                None,
                True,
                "invalid_multiple_cows",
            ),
        ]
        with patch.object(
            exporter, "_draw_label", wraps=exporter._draw_label
        ) as draw_label:
            output = exporter.render_corrected_frame(frame, detections)
        self.assertEqual(output.shape, (1080, 1920, 3))
        self.assertEqual(draw_label.call_count, 1)
        self.assertEqual(draw_label.call_args.kwargs["label"], "G0001")
        dash_pixel = output[200, 205]
        self.assertGreater(int(dash_pixel[0]), 50)
        self.assertEqual(int(dash_pixel[0]), int(dash_pixel[1]))
        self.assertEqual(int(dash_pixel[1]), int(dash_pixel[2]))
        gap_pixel = output[200, 220]
        self.assertEqual(tuple(int(value) for value in gap_pixel), (0, 0, 0))

    def test_short_dashed_edge_keeps_a_gap(self) -> None:
        canvas = np.zeros((80, 80, 3), dtype=np.uint8)
        exporter._draw_dashed_line(canvas, (10, 40), (35, 40), thickness=2)
        self.assertGreater(int(canvas[40, 15, 0]), 0)
        self.assertEqual(int(canvas[40, 32, 0]), 0)

    def test_renderer_rejects_invalid_row_that_retains_identity(self) -> None:
        frame = np.zeros((exporter.RAW_HEIGHT, exporter.RAW_WIDTH, 3), dtype=np.uint8)
        detection = exporter.RenderDetection(
            1, 100, 100, 300, 300, 0, "G0001", True, "invalid_multiple_cows"
        )
        with self.assertRaisesRegex(exporter.ContractError, "must not retain"):
            exporter.render_corrected_frame(frame, [detection])

    def test_sample_duration_is_hard_capped_at_three_minutes(self) -> None:
        self.assertEqual(exporter.sample_frame_count(Decimal("180")), 5394)
        with self.assertRaises(exporter.ContractError):
            exporter.sample_frame_count(Decimal("180.0001"))


if __name__ == "__main__":
    unittest.main()
