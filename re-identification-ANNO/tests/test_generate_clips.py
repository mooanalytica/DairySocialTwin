from __future__ import annotations

import csv
import json
import subprocess
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path

import generate_clips as clips


def write_csv(path: Path, columns: tuple[str, ...], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def occurrence_csv_row(
    ordinal: int,
    *,
    start_frame: int,
    end_frame: int | None = None,
    start_det_id: str | None = None,
    gid: str = "G0001",
    legacy_track_id: str = "7",
) -> dict[str, str]:
    if end_frame is None:
        end_frame = start_frame
    if start_det_id is None:
        start_det_id = f"det-{ordinal}"
    values = {column: "" for column in clips.OCCURRENCE_COLUMNS}
    values.update(
        {
            "occurrence_id": f"O{ordinal:06d}",
            "clip_order": "0",
            "clip_id": "GX040006",
            "display_global_id": gid,
            "legacy_track_id": legacy_track_id,
            "start_frame": str(start_frame),
            "end_frame": str(end_frame),
            "start_time_sec": "0.000000000",
            "end_time_sec_inclusive": "0.000000000",
            "end_time_sec_exclusive": "0.033366667",
            "span_duration_sec": "0.033366667",
            "num_valid_detections": "1",
            "num_missing_frames": "0",
            "max_internal_gap_missing_frames": "0",
            "start_det_id": start_det_id,
            "end_det_id": start_det_id,
        }
    )
    return values


def detection_csv_row(
    *,
    det_id: str,
    local_frame: int,
    gid: str = "G0001",
    legacy_track_id: str = "7",
    x1: str = "100",
    y1: str = "200",
    x2: str = "300",
    y2: str = "600",
) -> dict[str, str]:
    values = {column: "" for column in clips.DETECTION_COLUMNS}
    values.update(
        {
            "sequence_id": "dairy_farm_1_gopro1_20250505",
            "clip_id": "GX040006",
            "clip_order": "0",
            "det_id": det_id,
            "csv_row_index": str(local_frame),
            "legacy_track_id": legacy_track_id,
            "display_global_id": gid,
            "local_frame": str(local_frame),
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "valid": "true",
        }
    )
    return values


def sample_occurrence(
    occurrence_id: str = "O000001",
    *,
    start_frame: int = 1_000,
    end_frame: int | None = None,
) -> clips.Occurrence:
    if end_frame is None:
        end_frame = start_frame
    return clips.Occurrence(
        occurrence_id=occurrence_id,
        clip_order=0,
        clip_id="GX040006",
        display_global_id="G0001",
        legacy_track_id="7",
        start_frame=start_frame,
        end_frame=end_frame,
        start_det_id="det-1",
    )


def probe_result(
    command: list[str],
    *,
    frame_count: int = clips.WINDOW_FRAME_COUNT,
    **overrides: object,
) -> subprocess.CompletedProcess[str]:
    stream: dict[str, object] = {
        "codec_type": "video",
        "codec_name": "h264",
        "width": 1_920,
        "height": 1_080,
        "pix_fmt": "yuv420p",
        "r_frame_rate": "30000/1001",
        "avg_frame_rate": "30000/1001",
        "nb_read_frames": str(frame_count),
    }
    stream.update(overrides)
    return subprocess.CompletedProcess(
        command,
        0,
        stdout=json.dumps({"streams": [stream]}),
        stderr="",
    )


class GenerateClipsTests(unittest.TestCase):
    def test_window_slides_at_beginning_middle_and_end(self) -> None:
        self.assertEqual(clips.compute_window(0, 2_000), (0, 899, 0))
        self.assertEqual(clips.compute_window(1_000, 2_000), (550, 1_449, 450))
        self.assertEqual(clips.compute_window(1_999, 2_000), (1_100, 1_999, 899))

    def test_bbox_scales_expands_about_center_and_clamps(self) -> None:
        box = clips.make_red_box(
            Decimal("100"),
            Decimal("200"),
            Decimal("300"),
            Decimal("600"),
        )
        self.assertEqual(box, clips.RedBox(x=37, y=75, width=126, height=250))

        clamped = clips.make_red_box(
            Decimal("0"),
            Decimal("0"),
            Decimal("100"),
            Decimal("100"),
        )
        self.assertEqual(clamped, clips.RedBox(x=0, y=0, width=57, height=57))
        with self.assertRaisesRegex(clips.ContractError, "0 <= x1"):
            clips.make_red_box(
                Decimal("-1"),
                Decimal("0"),
                Decimal("100"),
                Decimal("100"),
            )

    def test_load_occurrences_enforces_order_count_and_fixed_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "occurrences.csv"
            write_csv(
                path,
                clips.OCCURRENCE_COLUMNS,
                [
                    occurrence_csv_row(1, start_frame=10),
                    occurrence_csv_row(2, start_frame=20, gid="G0062"),
                ],
            )
            specs = {
                "GX040006": clips.VideoSpec(Path(directory) / "video.mp4", 2_000)
            }
            loaded = clips.load_occurrences(
                path, expected_count=2, video_specs=specs
            )
            self.assertEqual([item.occurrence_id for item in loaded], ["O000001", "O000002"])
            self.assertEqual(loaded[1].display_global_id, "G0062")

            with self.assertRaisesRegex(clips.ContractError, "expected 3"):
                clips.load_occurrences(path, expected_count=3, video_specs=specs)

    def test_one_pass_lookup_finds_exact_start_row_and_rejects_duplicate(self) -> None:
        occurrence = sample_occurrence(start_frame=100)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "detections.csv"
            write_csv(
                path,
                clips.DETECTION_COLUMNS,
                [
                    detection_csv_row(det_id="unrelated", local_frame=99),
                    detection_csv_row(det_id="det-1", local_frame=100),
                ],
            )
            found = clips.lookup_start_bboxes(
                [occurrence], path, expected_rows=2, logger=lambda _: None
            )
            self.assertEqual(
                found["O000001"],
                clips.RedBox(x=37, y=75, width=126, height=250),
            )

            write_csv(
                path,
                clips.DETECTION_COLUMNS,
                [
                    detection_csv_row(det_id="det-1", local_frame=100),
                    detection_csv_row(det_id="det-1", local_frame=100),
                ],
            )
            with self.assertRaisesRegex(clips.ContractError, "more than once"):
                clips.lookup_start_bboxes(
                    [occurrence], path, expected_rows=2, logger=lambda _: None
                )

    def test_anchor_overrides_drive_bbox_and_manifest_window(self) -> None:
        occurrence = sample_occurrence(start_frame=100, end_frame=200)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            overrides_path = root / "anchors.json"
            overrides_path.write_text(
                json.dumps(
                    {
                        "schema_version": "1.0",
                        "anchors": [
                            {"occurrence_id": "O000001", "anchor_frame": 180}
                        ],
                    }
                ),
                encoding="utf-8",
            )
            overrides = clips.load_anchor_overrides(overrides_path, [occurrence])
            self.assertEqual(overrides, {"O000001": 180})

            detections_path = root / "detections.csv"
            write_csv(
                detections_path,
                clips.DETECTION_COLUMNS,
                [
                    detection_csv_row(det_id="det-1", local_frame=100),
                    detection_csv_row(
                        det_id="det-anchor",
                        local_frame=180,
                        x1="200",
                        y1="300",
                        x2="400",
                        y2="700",
                    ),
                ],
            )
            boxes = clips.lookup_anchor_bboxes(
                [occurrence],
                detections_path,
                anchor_frames=overrides,
                anchor_det_ids={"O000001": "det-anchor"},
                expected_rows=2,
                logger=lambda _: None,
            )
            self.assertEqual(
                boxes["O000001"],
                clips.RedBox(x=87, y=125, width=126, height=250),
            )
            records = clips.build_pending_records(
                [occurrence],
                boxes,
                anchor_frames=overrides,
                video_specs={"GX040006": clips.VideoSpec(root / "video.mp4", 2_000)},
            )
            self.assertEqual(records[0]["anchor_frame"], 180)
            self.assertEqual(records[0]["window_start_frame"], 0)
            self.assertEqual(records[0]["anchor_offset_frame"], 180)

            payload = json.loads(overrides_path.read_text(encoding="utf-8"))
            payload["anchors"][0]["anchor_frame"] = 201
            overrides_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(clips.ContractError, "inside"):
                clips.load_anchor_overrides(overrides_path, [occurrence])

            with self.assertRaisesRegex(clips.ContractError, "det_id"):
                clips.lookup_anchor_bboxes(
                    [occurrence],
                    detections_path,
                    anchor_frames={"O000001": 180},
                    anchor_det_ids={"O000001": "wrong-det"},
                    expected_rows=2,
                    logger=lambda _: None,
                )

    def test_lookup_rejects_wrong_frame_track_or_gid(self) -> None:
        occurrence = sample_occurrence(start_frame=100)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "detections.csv"
            write_csv(
                path,
                clips.DETECTION_COLUMNS,
                [detection_csv_row(det_id="det-1", local_frame=101)],
            )
            with self.assertRaisesRegex(clips.ContractError, "local_frame"):
                clips.lookup_start_bboxes(
                    [occurrence], path, expected_rows=1, logger=lambda _: None
                )

    def test_manifest_contains_all_pending_records_in_order_and_is_strict(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            specs = {"GX040006": clips.VideoSpec(root / "video.mp4", 2_000)}
            occurrences = [
                sample_occurrence("O000001", start_frame=0),
                clips.Occurrence(
                    occurrence_id="O000002",
                    clip_order=0,
                    clip_id="GX040006",
                    display_global_id="G0002",
                    legacy_track_id="8",
                    start_frame=1_999,
                    end_frame=1_999,
                    start_det_id="det-2",
                ),
            ]
            boxes = {
                "O000001": clips.RedBox(1, 2, 3, 4),
                "O000002": clips.RedBox(5, 6, 7, 8),
            }
            pending = clips.build_pending_records(
                occurrences, boxes, video_specs=specs
            )
            manifest_path = root / "cached_clips" / "manifest.json"
            payload = clips.load_or_create_manifest(manifest_path, pending)

            self.assertEqual(set(payload), {"schema_version", "clips"})
            self.assertEqual(payload["schema_version"], "1.0")
            self.assertEqual(
                [entry["occurrence_id"] for entry in payload["clips"]],
                ["O000001", "O000002"],
            )
            self.assertEqual(payload["clips"][0]["window_start_frame"], 0)
            self.assertEqual(payload["clips"][1]["window_end_frame"], 1_999)
            self.assertTrue(all(entry["status"] == "pending" for entry in payload["clips"]))

            changed = json.loads(json.dumps(payload))
            changed["clips"].reverse()
            with self.assertRaisesRegex(clips.ContractError, "immutable field"):
                clips.validate_manifest(changed, pending)

    def test_ffmpeg_command_is_nvenc_only_and_draws_one_fixed_box(self) -> None:
        record = clips.build_pending_records(
            [sample_occurrence()],
            {"O000001": clips.RedBox(10, 20, 30, 40)},
            video_specs={"GX040006": clips.VideoSpec(Path("input.mp4"), 2_000)},
        )[0]
        command = clips.build_ffmpeg_command(
            record,
            Path("input.mp4"),
            Path("O000001.mp4.part"),
            ffmpeg_binary="/mock/ffmpeg",
        )
        self.assertEqual(command[0], "/mock/ffmpeg")
        self.assertIn("h264_nvenc", command)
        self.assertNotIn("libx264", command)
        self.assertEqual(command[command.index("-frames:v") + 1], "900")
        self.assertEqual(command[command.index("-gpu") + 1], "0")
        self.assertEqual(
            command[command.index("-vf") + 1],
            "drawbox=x=10:y=20:w=30:h=40:color=red@1.0:t=8",
        )
        self.assertIn("-noautorotate", command[: command.index("-i")])
        self.assertEqual(command[command.index("-fps_mode") + 1], "passthrough")
        self.assertEqual(command[-3:], ["-f", "mp4", "O000001.mp4.part"])

    def test_nvenc_environment_requires_physical_gpu_one(self) -> None:
        with self.assertRaisesRegex(clips.ContractError, "exactly '1'"):
            clips.require_nvenc_environment(
                {"CUDA_VISIBLE_DEVICES": "0"}, which=lambda _: "/mock/ffmpeg"
            )
        encoder_tools = clips.require_nvenc_environment(
            {"CUDA_VISIBLE_DEVICES": "1"},
            which=lambda name: f"/mock/{name}",
        )
        self.assertEqual(encoder_tools.ffmpeg_binary, "/mock/ffmpeg")
        self.assertEqual(encoder_tools.ffprobe_binary, "/mock/ffprobe")

    def test_ffprobe_contract_rejects_extra_streams_and_rotation(self) -> None:
        path = Path("fixture.mp4")
        clips.probe_video(
            path,
            ffprobe_binary="/mock/ffprobe",
            expected_frame_count=900,
            runner=probe_result,
        )

        def extra_stream_runner(
            command: list[str], **_: object
        ) -> subprocess.CompletedProcess[str]:
            payload = json.loads(probe_result(command).stdout)
            payload["streams"].append({"codec_type": "audio"})
            return subprocess.CompletedProcess(
                command, 0, stdout=json.dumps(payload), stderr=""
            )

        with self.assertRaisesRegex(clips.ContractError, "exactly one"):
            clips.probe_video(
                path,
                ffprobe_binary="/mock/ffprobe",
                expected_frame_count=900,
                runner=extra_stream_runner,
            )

        def rotation_runner(
            command: list[str], **_: object
        ) -> subprocess.CompletedProcess[str]:
            return probe_result(command, tags={"rotate": "0"})

        with self.assertRaisesRegex(clips.ContractError, "rotation"):
            clips.probe_video(
                path,
                ffprobe_binary="/mock/ffprobe",
                expected_frame_count=900,
                runner=rotation_runner,
            )

    def test_mocked_generation_publishes_atomically_and_resumes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache_dir = root / "cached_clips"
            cache_dir.mkdir()
            source = root / "tracked.mp4"
            source.write_bytes(b"fixture source, never decoded")
            specs = {"GX040006": clips.VideoSpec(source, 2_000)}
            pending = clips.build_pending_records(
                [sample_occurrence()],
                {"O000001": clips.RedBox(10, 20, 30, 40)},
                video_specs=specs,
            )
            manifest_path = cache_dir / "manifest.json"
            manifest = clips.load_or_create_manifest(manifest_path, pending)
            calls: list[list[str]] = []

            def fake_runner(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
                calls.append(command)
                self.assertEqual(
                    kwargs["env"], {"CUDA_VISIBLE_DEVICES": "1"}
                )
                Path(command[-1]).write_bytes(b"mock encoded mp4")
                return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

            clips.generate_selected(
                manifest,
                manifest["clips"],
                manifest_path=manifest_path,
                cache_dir=cache_dir,
                video_specs=specs,
                ffmpeg_binary="/mock/ffmpeg",
                ffprobe_binary="/mock/ffprobe",
                workers=1,
                runner=fake_runner,
                probe_runner=probe_result,
                env={"CUDA_VISIBLE_DEVICES": "1"},
                logger=lambda _: None,
            )
            final_path = cache_dir / "O000001.mp4"
            self.assertTrue(final_path.is_file())
            self.assertFalse((cache_dir / "O000001.mp4.part").exists())
            self.assertEqual(len(calls), 1)

            persisted = json.loads(manifest_path.read_text(encoding="utf-8"))
            complete = persisted["clips"][0]
            self.assertEqual(complete["status"], "complete")
            self.assertEqual(complete["size_bytes"], len(b"mock encoded mp4"))
            self.assertEqual(complete["sha256"], clips.file_sha256(final_path))

            def forbidden_runner(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
                raise AssertionError("resume must not invoke ffmpeg")

            clips.generate_selected(
                persisted,
                persisted["clips"],
                manifest_path=manifest_path,
                cache_dir=cache_dir,
                video_specs=specs,
                ffmpeg_binary="/mock/ffmpeg",
                ffprobe_binary="/mock/ffprobe",
                workers=1,
                runner=forbidden_runner,
                probe_runner=probe_result,
                env={"CUDA_VISIBLE_DEVICES": "1"},
                logger=lambda _: None,
            )

    def test_mocked_ffmpeg_failure_removes_partial_file_without_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cache_dir = Path(directory)
            record = clips.build_pending_records(
                [sample_occurrence()],
                {"O000001": clips.RedBox(10, 20, 30, 40)},
                video_specs={
                    "GX040006": clips.VideoSpec(Path(directory) / "input.mp4", 2_000)
                },
            )[0]

            def failing_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
                Path(command[-1]).write_bytes(b"partial")
                return subprocess.CompletedProcess(
                    command, 1, stdout="", stderr="nvenc failed"
                )

            with self.assertRaisesRegex(clips.ContractError, "h264_nvenc failed"):
                clips.generate_part(
                    record,
                    source_video=Path(directory) / "input.mp4",
                    cache_dir=cache_dir,
                    ffmpeg_binary="/mock/ffmpeg",
                    ffprobe_binary="/mock/ffprobe",
                    runner=failing_runner,
                    probe_runner=probe_result,
                    env={"CUDA_VISIBLE_DEVICES": "1"},
                )
            self.assertFalse((cache_dir / "O000001.mp4.part").exists())
            self.assertFalse((cache_dir / "O000001.mp4").exists())

    def test_pending_final_from_interrupted_manifest_commit_is_adopted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache_dir = root / "cached_clips"
            cache_dir.mkdir()
            source = root / "tracked.mp4"
            source.write_bytes(b"fixture source")
            specs = {"GX040006": clips.VideoSpec(source, 2_000)}
            pending = clips.build_pending_records(
                [sample_occurrence()],
                {"O000001": clips.RedBox(10, 20, 30, 40)},
                video_specs=specs,
            )
            manifest_path = cache_dir / "manifest.json"
            manifest = clips.load_or_create_manifest(manifest_path, pending)
            final_path = cache_dir / "O000001.mp4"
            final_path.write_bytes(b"complete but not yet committed")

            def forbidden_runner(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
                raise AssertionError("adoption must not invoke ffmpeg")

            clips.generate_selected(
                manifest,
                manifest["clips"],
                manifest_path=manifest_path,
                cache_dir=cache_dir,
                video_specs=specs,
                ffmpeg_binary="/mock/ffmpeg",
                ffprobe_binary="/mock/ffprobe",
                workers=1,
                runner=forbidden_runner,
                probe_runner=probe_result,
                env={"CUDA_VISIBLE_DEVICES": "1"},
                logger=lambda _: None,
            )
            persisted = json.loads(manifest_path.read_text(encoding="utf-8"))
            record = persisted["clips"][0]
            self.assertEqual(record["status"], "complete")
            self.assertEqual(record["size_bytes"], final_path.stat().st_size)
            self.assertEqual(record["sha256"], clips.file_sha256(final_path))

    def test_selection_supports_inclusive_ranges_and_explicit_ids(self) -> None:
        records = [
            {"occurrence_id": f"O{index:06d}"} for index in range(1, 6)
        ]
        selected = clips.select_records(records, start_index=2, end_index=4)
        self.assertEqual(
            [item["occurrence_id"] for item in selected],
            ["O000002", "O000003", "O000004"],
        )
        explicit = clips.select_records(
            records, occurrence_ids=["O000005", "O000001"]
        )
        self.assertEqual(
            [item["occurrence_id"] for item in explicit],
            ["O000005", "O000001"],
        )

        parsed = clips.build_parser().parse_args(
            [
                "--anchor-overrides",
                "anchors.json",
                "--overrides-only",
                "--expected-override-count",
                "2",
            ]
        )
        self.assertEqual(parsed.anchor_overrides, Path("anchors.json"))
        self.assertTrue(parsed.overrides_only)
        clips.validate_cli_selection(parsed, {"O000001": 10, "O000002": 20})

        missing = clips.build_parser().parse_args(
            ["--anchor-overrides", "anchors.json"]
        )
        with self.assertRaisesRegex(clips.ContractError, "unsafe"):
            clips.validate_cli_selection(missing, {"O000001": 10})

        wrong_count = clips.build_parser().parse_args(
            [
                "--anchor-overrides",
                "anchors.json",
                "--overrides-only",
                "--expected-override-count",
                "2",
            ]
        )
        with self.assertRaisesRegex(clips.ContractError, "expected 2"):
            clips.validate_cli_selection(wrong_count, {"O000001": 10})


if __name__ == "__main__":
    unittest.main()
