from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from cowtrack.config import ClipManifest, IngestConfig, InputCSVConfig
from cowtrack.schemas.frames import FRAMES_SCHEMA
from cowtrack.stages.s00_ingest import (
    ClipRuntime,
    _validate_outputs,
    _write_detections,
)
from cowtrack.video import PacketTimeline, VideoStreamMetadata


def _config() -> IngestConfig:
    return IngestConfig(
        schema_version="1.0",
        random_seed=1,
        input_csv=InputCSVConfig("video", "frame", "x", "y", "w", "h", "score", "track_id"),
        ffprobe_binary="ffprobe",
        opencv_auto_rotate=False,
        clamp_boxes=True,
        retain_invalid_rows=True,
        minimum_bbox_area_pixels=1.0,
        minimum_retained_area_fraction=0.10,
        duplicate_iou_threshold=0.90,
        duplicate_area_similarity_min=0.70,
        parquet_compression="zstd",
        parquet_batch_rows=2,
        progress_interval_sec=10.0,
        create_overlay_samples=False,
        overlay_jpeg_quality=90,
        acceptance={},
    )


def _run_csv(tmp_path: Path, legacy_values: tuple[str, str, str]):
    csv_path = tmp_path / f"boxes_{legacy_values[0]}.csv"
    csv_path.write_text(
        "video,frame,track_id,x,y,w,h,score,identity,id_conf\n"
        f"clip.MP4,0,{legacy_values[0]},10,10,20,20,0.9,ignored,1\n"
        f"clip.MP4,0,{legacy_values[1]},10,10,20,20,0.8,ignored,1\n"
        f"clip.MP4,1,{legacy_values[2]},-1,5,20,20,0.7,ignored,1\n",
        encoding="utf-8",
    )
    video_path = tmp_path / "clip.MP4"
    video_path.write_bytes(b"placeholder")
    manifest = ClipManifest("seq", 0, "clip", video_path, csv_path, 0, "xywh")
    metadata = VideoStreamMetadata(
        "hevc", 100, 100, Fraction(1, 30), Fraction(30, 1), 0, 2, 2, 0, None, None
    )
    runtime = ClipRuntime(
        manifest,
        metadata,
        PacketTimeline((0, 1), (1, 1)),
        0,
        Fraction(0, 1),
        Fraction(1, 30),
        np.array([0.0, 1.0 / 30.0]),
        np.array([0.0, 1.0 / 30.0]),
    )
    output = tmp_path / f"detections_{legacy_values[0]}.parquet"
    stats = _write_detections([runtime], output, _config(), lambda _: None)
    return pq.read_table(output), stats


def test_csv_rows_are_retained_and_legacy_does_not_affect_results(tmp_path: Path) -> None:
    first, stats = _run_csv(tmp_path, ("1", "2", "3"))
    second, _ = _run_csv(tmp_path, ("91", "92", "93"))
    assert first.num_rows == 3
    assert stats["num_valid_boxes"] == 2
    assert stats["num_invalid_boxes"] == 1
    assert stats["num_duplicate_boxes_removed"] == 1
    assert stats["num_clamped_boxes"] == 1
    assert stats["per_clip"][0]["frame_observation"] == {
        "num_frames": 2,
        "num_observed_frames": 2,
        "num_unobserved_frames": 0,
        "observed_frame_fraction": 1.0,
        "num_unobserved_intervals": 0,
        "longest_unobserved_interval_frames": 0,
        "unobserved_frame_intervals": [],
    }
    compare_columns = [name for name in first.column_names if name != "legacy_track_id"]
    assert first.select(compare_columns).equals(second.select(compare_columns))


def test_output_validation_accepts_legacy_report_without_observation_policy(
    tmp_path: Path,
) -> None:
    detections, stats = _run_csv(tmp_path, ("11", "12", "13"))
    detections_path = tmp_path / "legacy_detections.parquet"
    pq.write_table(detections, detections_path)
    frames_path = tmp_path / "legacy_frames.parquet"
    frames = pa.Table.from_pylist(
        [
            {
                "sequence_id": "seq",
                "clip_id": "clip",
                "clip_order": 0,
                "local_frame": 0,
                "global_frame": 0,
                "pts_sec": 0.0,
                "global_time_sec": 0.0,
                "width": 100,
                "height": 100,
            },
            {
                "sequence_id": "seq",
                "clip_id": "clip",
                "clip_order": 0,
                "local_frame": 1,
                "global_frame": 1,
                "pts_sec": 1.0 / 30.0,
                "global_time_sec": 1.0 / 30.0,
                "width": 100,
                "height": 100,
            },
        ],
        schema=FRAMES_SCHEMA,
    )
    pq.write_table(frames, frames_path)
    report = {
        "num_clips": 1,
        "num_frames": 2,
        "num_input_boxes": 3,
        "num_valid_boxes": 2,
        "num_invalid_boxes": 1,
        "num_duplicate_boxes_removed": 1,
        "num_clamped_boxes": 1,
        "per_clip": stats["per_clip"],
    }
    # Simulate a completed pre-policy S00 report: old artifacts contain only
    # the aggregate with/without-box counts.
    report["per_clip"][0].pop("frame_observation")
    config = replace(
        _config(),
        acceptance={
            "expected_num_clips": 1,
            "expected_num_frames": 2,
            "expected_num_input_boxes": 3,
            "expected_num_valid_boxes": 2,
            "expected_num_invalid_boxes": 1,
            "expected_num_duplicate_boxes_removed": 1,
            "expected_num_clamped_boxes": 1,
            "expected_width": 100,
            "expected_height": 100,
        },
    )

    _validate_outputs(frames_path, detections_path, report, config)
