from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import cowtrack.stages.s01_microtrack as stage
from cowtrack.config import ContractError
from cowtrack.schemas.detections import DETECTIONS_SCHEMA
from cowtrack.schemas.frames import FRAMES_SCHEMA


TINY_CLIP_IDS = tuple(f"C{index:02d}" for index in range(11))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_tiny_s00(input_dir: Path, *, legacy_prefix: str = "legacy") -> None:
    input_dir.mkdir()
    frames = []
    detections = []
    num_frames = 2 * len(TINY_CLIP_IDS)
    for frame in range(num_frames):
        clip_order = frame // 2
        clip_id = TINY_CLIP_IDS[clip_order]
        local_frame = frame % 2
        time_sec = frame * 0.1
        frames.append(
            {
                "sequence_id": "tiny_sequence",
                "clip_id": clip_id,
                "clip_order": clip_order,
                "local_frame": local_frame,
                "global_frame": frame,
                "pts_sec": local_frame * 0.1,
                "global_time_sec": time_sec,
                "width": 3840,
                "height": 2160,
            }
        )
        cx = 0.20 + frame * 0.01
        width = 0.20
        height = 0.20
        detections.append(
            {
                "det_id": frame + 10,
                "sequence_id": "tiny_sequence",
                "clip_id": clip_id,
                "local_frame": local_frame,
                "global_frame": frame,
                "global_time_sec": time_sec,
                "x1": (cx - width / 2) * 3840,
                "y1": (0.50 - height / 2) * 2160,
                "x2": (cx + width / 2) * 3840,
                "y2": (0.50 + height / 2) * 2160,
                "cx_norm": cx,
                "cy_norm": 0.50,
                "w_norm": width,
                "h_norm": height,
                "area_norm": width * height,
                "bbox_confidence": 0.9,
                "legacy_track_id": f"{legacy_prefix}-{999 - frame}",
                "csv_row_index": frame,
                "valid": True,
                "qa_flags": 0,
            }
        )
    frames_path = input_dir / "frames.parquet"
    detections_path = input_dir / "detections.parquet"
    pq.write_table(pa.Table.from_pylist(frames, schema=FRAMES_SCHEMA), frames_path)
    pq.write_table(
        pa.Table.from_pylist(detections, schema=DETECTIONS_SCHEMA), detections_path
    )

    resolved_manifest = [
        {
            "sequence_id": "tiny_sequence",
            "clip_order": clip_order,
            "clip_id": clip_id,
            "video_path": str(input_dir / f"{clip_id}.MP4"),
            "bbox_csv_path": str(input_dir / f"{clip_id}.csv"),
            "frame_index_base": 0,
            "bbox_format": "xywh",
        }
        for clip_order, clip_id in enumerate(TINY_CLIP_IDS)
    ]
    (input_dir / "resolved_manifest.json").write_text(
        json.dumps(resolved_manifest), encoding="utf-8"
    )
    report = {
        "schema_version": "1.0",
        "sequence_id": "tiny_sequence",
        "num_clips": len(TINY_CLIP_IDS),
        "num_frames": num_frames,
        "num_input_boxes": num_frames,
        "num_valid_boxes": num_frames,
        "num_invalid_boxes": 0,
        "num_duplicate_boxes_removed": 0,
        "num_clamped_boxes": 0,
        "sequence_end_time_sec_exclusive": num_frames * 0.1,
        "source_rows_retained": True,
        "keypoints_used": False,
        "legacy_track_id_used": False,
        "coordinate_system": "raw_encoded_landscape_no_autorotate",
        "clip_boundaries": [
            {
                "clip_id": clip_id,
                "clip_order": clip_order,
                "num_frames": 2,
                "start_global_frame": 2 * clip_order,
                "end_global_frame_inclusive": 2 * clip_order + 1,
                "start_global_time_sec": 0.2 * clip_order,
                "end_global_time_sec_inclusive": 0.2 * clip_order + 0.1,
                "end_global_time_sec_exclusive": 0.2 * clip_order + 0.2,
                "width": 3840,
                "height": 2160,
            }
            for clip_order, clip_id in enumerate(TINY_CLIP_IDS)
        ],
        "per_clip": [
            {
                "clip_id": clip_id,
                "num_input_boxes": 2,
                "num_valid_boxes": 2,
                "num_invalid_boxes": 0,
                "num_duplicate_boxes_removed": 0,
                "num_clamped_boxes": 0,
                "num_frames_with_boxes": 2,
                "num_frames_without_boxes": 0,
            }
            for clip_id in TINY_CLIP_IDS
        ],
    }
    (input_dir / "ingest_report.json").write_text(
        json.dumps(report), encoding="utf-8"
    )
    artifacts = {
        path.name: {"size_bytes": path.stat().st_size, "sha256": _sha256(path)}
        for path in (
            frames_path,
            detections_path,
            input_dir / "resolved_manifest.json",
            input_dir / "ingest_report.json",
        )
    }
    success = {
        "stage": "S00",
        "stats": {
            "num_clips": len(TINY_CLIP_IDS),
            "num_frames": num_frames,
            "num_input_boxes": num_frames,
            "num_valid_boxes": num_frames,
            "num_invalid_boxes": 0,
            "num_duplicate_boxes_removed": 0,
            "num_clamped_boxes": 0,
        },
        "output_fingerprints": [
            {"path": name, **fingerprint}
            for name, fingerprint in artifacts.items()
        ],
    }
    (input_dir / "_SUCCESS.json").write_text(
        json.dumps(success), encoding="utf-8"
    )


def _write_config(path: Path) -> None:
    path.write_text(
        """
pipeline:
  schema_version: "1.0"
  random_seed: 1
  use_keypoints: false
  use_legacy_tracking_id: false
microtrack:
  cost_weights: {center_distance: 0.55, iou: 0.30, size: 0.15}
  max_time_gap_sec: auto
  max_time_gap_multiplier: 1.5
  center_distance_gate: 0.75
  max_area_ratio: 1.8
  min_iou: 0.01
  alternate_center_gate: 0.35
  ambiguity_margin: 0.08
  bidirectional_edges_only: true
  bridge_missing_detections: false
  min_length_detections: 3
  velocity_history_detections: 4
  fisheye_motion_grid: [2, 2]
  motion_prior_percentile: 99.0
  motion_prior_gate_floor: 0.10
  motion_prior_min_edges_per_cell: 1
  parquet_compression: zstd
  progress_interval_sec: 10.0
""".lstrip(),
        encoding="utf-8",
    )


def test_tiny_s01_stage_writes_resumes_and_detects_tampering(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "00_ingest"
    output_dir = tmp_path / "01_microtrack"
    config_path = tmp_path / "s01.yaml"
    _write_tiny_s00(input_dir)
    _write_config(config_path)

    first = stage.run_s01(input_dir, config_path, output_dir)
    assert first["stage"] == "S01"
    assert first["program_commit_hash"] is None
    assert first["stats"]["num_clips"] == 11
    assert first["stats"]["num_valid_detections"] == 22
    assert first["stats"]["num_microtracklets"] == 1
    assert (output_dir / "_SUCCESS.json").is_file()
    assert (output_dir / "motion_prior.npz").is_file()
    mapping = pq.read_table(output_dir / "det_to_micro.parquet")
    assert mapping["det_id"].to_pylist() == list(range(10, 32))
    # The C00->C01 edge proves that local_frame/clip_id never reset S01 tracking.
    edges = pq.read_table(output_dir / "det_edges.parquet")
    assert (11, 12) in set(
        zip(edges["src_det_id"].to_pylist(), edges["dst_det_id"].to_pylist())
    )

    resumed = stage.run_s01(input_dir, config_path, output_dir)
    assert resumed == first

    shuffled_input = tmp_path / "00_ingest_legacy_shuffled"
    shuffled_output = tmp_path / "01_microtrack_legacy_shuffled"
    _write_tiny_s00(shuffled_input, legacy_prefix="unrelated-shuffled-value")
    stage.run_s01(shuffled_input, config_path, shuffled_output)
    for filename in (
        "det_edges.parquet",
        "det_to_micro.parquet",
        "microtracklets.parquet",
    ):
        assert pq.read_table(output_dir / filename).equals(
            pq.read_table(shuffled_output / filename)
        )

    report_path = output_dir / "microtrack_report.json"
    report_path.write_text(report_path.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ContractError, match="artifact changed"):
        stage.run_s01(input_dir, config_path, output_dir)


def test_s01_rejects_tampered_or_internally_inconsistent_s00_contract(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "s01.yaml"
    _write_config(config_path)

    tampered_input = tmp_path / "00_ingest_tampered"
    _write_tiny_s00(tampered_input)
    resolved_path = tampered_input / "resolved_manifest.json"
    resolved_path.write_text(
        resolved_path.read_text(encoding="utf-8") + "\n", encoding="utf-8"
    )
    with pytest.raises(ContractError, match="differs from completed S00"):
        stage.run_s01(
            tampered_input,
            config_path,
            tmp_path / "01_microtrack_tampered",
        )

    inconsistent_input = tmp_path / "00_ingest_inconsistent"
    _write_tiny_s00(inconsistent_input)
    success_path = inconsistent_input / "_SUCCESS.json"
    success = json.loads(success_path.read_text(encoding="utf-8"))
    success["stats"]["num_clips"] = len(TINY_CLIP_IDS) + 1
    success_path.write_text(json.dumps(success), encoding="utf-8")
    with pytest.raises(ContractError, match="clip count differs"):
        stage.run_s01(
            inconsistent_input,
            config_path,
            tmp_path / "01_microtrack_inconsistent",
        )
