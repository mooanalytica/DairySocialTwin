from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cowtrack.config import ContractError
from cowtrack.qa.s01_review_plan import ReviewCase, ReviewPlan
import cowtrack.qa.s01_review_plan_only as stage
from cowtrack.schemas.detections import DETECTIONS_SCHEMA
from cowtrack.schemas.edges import DET_EDGES_SCHEMA
from cowtrack.schemas.frames import FRAMES_SCHEMA
from cowtrack.schemas.tracklets import DET_TO_MICRO_SCHEMA, MICROTRACKLETS_SCHEMA
import cowtrack.stages.s02_appearance as appearance


ROOT = Path(__file__).resolve().parents[1]


def _case() -> ReviewCase:
    return ReviewCase(
        case_id="risk-m000000-e000",
        case_kind="risk",
        micro_id=0,
        anchor_det_id=101,
        anchor_sequence_id="tiny_sequence",
        anchor_clip_id="clip_00",
        anchor_local_frame=10,
        anchor_global_frame=10,
        anchor_global_time_sec=10.0 / 30.0,
        window_start_time_sec=0.0,
        window_end_time_sec=2.0,
        reasons=("large_center_jump",),
        events=(
            (
                ("reason", "large_center_jump"),
                ("src_det_id", 100),
                ("dst_det_id", 101),
                ("clip_id", "clip_00"),
                ("global_frame", 10),
                ("global_time_sec", 10.0 / 30.0),
            ),
        ),
        num_detections=4,
        status="valid",
        local_purity_score=0.99,
        max_internal_center_jump=0.2,
    )


def _plan(case: ReviewCase) -> ReviewPlan:
    return ReviewPlan(
        cases=(case,),
        window_before_sec=3.0,
        window_after_sec=3.0,
        merge_distance_sec=6.0,
        long_track_min_detections=10_000,
        long_track_chunk_sec=300.0,
        long_track_chunk_overlap_sec=1.0,
        low_purity_threshold=0.8,
        high_jump_threshold=0.1,
        quality_min_detections=300,
        quality_max_detections=1_800,
        quality_min_purity=0.99,
        quality_max_jump=0.03,
        requested_quality_samples=20,
        eligible_quality_microtracks=0,
        risk_microtracks=(0,),
    )


def _appearance_data() -> appearance.LoadedAppearanceData:
    frames = np.asarray([0, 10, 20, 40], dtype=np.int64)
    return appearance.LoadedAppearanceData(
        sequence_id="tiny_sequence",
        frames={},
        detections={
            "det_id": np.asarray([100, 101, 102, 103], dtype=np.int64),
            "global_frame": frames,
            "global_time_sec": frames.astype(np.float64) / 30.0,
            "clip_id": np.asarray(["clip_00"] * 4, dtype=object),
        },
        microtracklets={
            "micro_id": np.asarray([0], dtype=np.int64),
            "num_detections": np.asarray([4], dtype=np.int32),
            "status": np.asarray(["valid"], dtype=object),
        },
        micro_ids_by_detection=np.zeros(4, dtype=np.int64),
        order_in_micro_by_detection=np.arange(4, dtype=np.int32),
        paths={0: np.arange(4, dtype=np.int64)},
        positions_by_frame=np.arange(4, dtype=np.int64),
        frame_offsets=np.zeros(42, dtype=np.int64),
        video_paths={},
        input_fingerprints=(),
    )


def _patch_tiny_plan(
    monkeypatch: pytest.MonkeyPatch, input_path: Path
) -> tuple[ReviewPlan, stage.review.RenderCase]:
    case = _case()
    plan = _plan(case)
    render_case = stage.review.RenderCase(
        case=case,
        start_global_frame=0,
        end_global_frame=40,
        output_relative_path="risk/risk-m000000-e000.mp4",
    )
    fingerprint = stage.review._fingerprint(input_path)
    data = SimpleNamespace(input_fingerprints=(fingerprint,))
    monkeypatch.setattr(stage, "_load_plan_data", lambda *args, **kwargs: data)
    monkeypatch.setattr(stage.review, "_plan", lambda *args, **kwargs: plan)
    monkeypatch.setattr(
        stage.review,
        "_render_cases",
        lambda *args, **kwargs: (render_case,),
    )
    monkeypatch.setattr(stage.review, "_validate_plan", lambda *args: None)
    monkeypatch.setattr(
        stage.review,
        "_preflight_nvenc",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("plan-only must not run NVENC preflight")
        ),
    )
    return plan, render_case


def _write_parquet(path: Path, schema: pa.Schema, rows: list[dict]) -> None:
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path)


def _artifact_record(path: Path) -> dict[str, object]:
    fingerprint = stage.review._fingerprint(path)
    return {
        "path": path.name,
        "size_bytes": fingerprint["size_bytes"],
        "sha256": fingerprint["sha256"],
    }


def _write_tiny_completed_inputs(
    ingest_dir: Path, microtrack_dir: Path
) -> tuple[Path, ...]:
    ingest_dir.mkdir()
    microtrack_dir.mkdir()
    period = stage.review.EXPECTED_FRAME_PERIOD_SEC
    clip_ids = tuple(f"clip_{index:02d}" for index in range(11))
    frames: list[dict] = []
    detections: list[dict] = []
    for frame, clip_id in enumerate(clip_ids):
        local_frame = 0
        global_time = frame * period
        frames.append(
            {
                "sequence_id": "tiny_sequence",
                "clip_id": clip_id,
                "clip_order": frame,
                "local_frame": local_frame,
                "global_frame": frame,
                "pts_sec": local_frame * period,
                "global_time_sec": global_time,
                "width": stage.review.RAW_WIDTH,
                "height": stage.review.RAW_HEIGHT,
            }
        )
        detections.append(
            {
                "det_id": 100 + frame,
                "sequence_id": "tiny_sequence",
                "clip_id": clip_id,
                "local_frame": local_frame,
                "global_frame": frame,
                "global_time_sec": global_time,
                "x1": 100.0 + frame,
                "y1": 200.0,
                "x2": 500.0 + frame,
                "y2": 600.0,
                "cx_norm": (300.0 + frame) / stage.review.RAW_WIDTH,
                "cy_norm": 400.0 / stage.review.RAW_HEIGHT,
                "w_norm": 400.0 / stage.review.RAW_WIDTH,
                "h_norm": 400.0 / stage.review.RAW_HEIGHT,
                "area_norm": (
                    160_000.0
                    / (stage.review.RAW_WIDTH * stage.review.RAW_HEIGHT)
                ),
                "bbox_confidence": 0.9,
                "legacy_track_id": "legacy",
                "csv_row_index": frame,
                "valid": True,
                "qa_flags": 0,
            }
        )

    frames_path = ingest_dir / "frames.parquet"
    detections_path = ingest_dir / "detections.parquet"
    resolved_path = ingest_dir / "resolved_manifest.json"
    ingest_report_path = ingest_dir / "ingest_report.json"
    _write_parquet(frames_path, FRAMES_SCHEMA, frames)
    _write_parquet(detections_path, DETECTIONS_SCHEMA, detections)
    missing_videos = tuple(
        ingest_dir / f"raw-video-does-not-exist-{clip_id}.mp4"
        for clip_id in clip_ids
    )
    resolved_path.write_text(
        json.dumps(
            [
                {
                    "sequence_id": "tiny_sequence",
                    "clip_order": index,
                    "clip_id": clip_id,
                    "video_path": str(missing_videos[index]),
                }
                for index, clip_id in enumerate(clip_ids)
            ]
        ),
        encoding="utf-8",
    )
    ingest_report_path.write_text(
        json.dumps(
            {
                "sequence_id": "tiny_sequence",
                "coordinate_system": "raw_encoded_landscape_no_autorotate",
                "num_clips": len(clip_ids),
                "num_frames": len(clip_ids),
                "num_input_boxes": len(clip_ids),
                "num_valid_boxes": len(clip_ids),
                "clip_boundaries": [
                    {
                        "clip_id": clip_id,
                        "clip_order": index,
                        "num_frames": 1,
                        "start_global_frame": index,
                        "end_global_frame_inclusive": index,
                        "opencv_auto_rotate": False,
                    }
                    for index, clip_id in enumerate(clip_ids)
                ],
            }
        ),
        encoding="utf-8",
    )
    s00 = {
        "stage": "S00",
        "stats": {
            "num_clips": len(clip_ids),
            "num_frames": len(clip_ids),
            "num_input_boxes": len(clip_ids),
            "num_valid_boxes": len(clip_ids),
        },
        "output_fingerprints": [
            _artifact_record(path)
            for path in (
                frames_path,
                detections_path,
                resolved_path,
                ingest_report_path,
            )
        ],
    }
    (ingest_dir / stage.review.SUCCESS_NAME).write_text(
        json.dumps(s00), encoding="utf-8"
    )

    mapping_path = microtrack_dir / "det_to_micro.parquet"
    microtracklets_path = microtrack_dir / "microtracklets.parquet"
    edges_path = microtrack_dir / "det_edges.parquet"
    _write_parquet(
        mapping_path,
        DET_TO_MICRO_SCHEMA,
        [
            {
                "det_id": 100 + row,
                "micro_id": 0,
                "order_in_micro": row,
                "incoming_edge_score": None if row == 0 else 0.99,
            }
            for row in range(len(clip_ids))
        ],
    )
    _write_parquet(
        microtracklets_path,
        MICROTRACKLETS_SCHEMA,
        [
            {
                "micro_id": 0,
                "start_global_frame": 0,
                "end_global_frame": len(clip_ids) - 1,
                "start_time_sec": 0.0,
                "end_time_sec": (len(clip_ids) - 1) * period,
                "num_detections": len(clip_ids),
                "duration_sec": len(clip_ids) * period,
                "start_x1": 100.0,
                "start_y1": 200.0,
                "start_x2": 500.0,
                "start_y2": 600.0,
                "end_x1": 100.0 + len(clip_ids) - 1,
                "end_y1": 200.0,
                "end_x2": 500.0 + len(clip_ids) - 1,
                "end_y2": 600.0,
                "end_vx_norm_per_sec": 0.0,
                "end_vy_norm_per_sec": 0.0,
                "start_vx_norm_per_sec": 0.0,
                "start_vy_norm_per_sec": 0.0,
                "max_internal_center_jump": 0.01,
                "median_internal_cost": 0.01,
                "bidirectional_agreement": 1.0,
                "local_purity_score": 0.99,
                "status": "valid",
            }
        ],
    )
    _write_parquet(
        edges_path,
        DET_EDGES_SCHEMA,
        [
            {
                "src_det_id": 100 + row,
                "dst_det_id": 101 + row,
                "delta_time_sec": period,
                "forward_cost": 0.01,
                "backward_cost": 0.01,
                "forward_rank": 1,
                "backward_rank": 1,
                "bidirectional_agree": True,
                "accepted": True,
                "reject_reason": "",
            }
            for row in range(len(clip_ids) - 1)
        ],
    )
    microtrack_report_path = microtrack_dir / "microtrack_report.json"
    microtrack_report_path.write_text(
        json.dumps(
            {
                "sequence_id": "tiny_sequence",
                "input_coordinate_system": (
                    "raw_encoded_landscape_no_autorotate"
                ),
                "input_rows_modified": False,
                "stats": {
                    "num_frames": len(clip_ids),
                    "num_valid_detections": len(clip_ids),
                    "num_microtracklets": 1,
                    "num_accepted_edges": len(clip_ids) - 1,
                },
            }
        ),
        encoding="utf-8",
    )
    s01 = {
        "stage": "S01",
        "stats": {
            "num_frames": len(clip_ids),
            "num_valid_detections": len(clip_ids),
            "num_microtracklets": 1,
            "num_accepted_edges": len(clip_ids) - 1,
        },
        "output_fingerprints": [
            _artifact_record(path)
            for path in (
                mapping_path,
                microtracklets_path,
                edges_path,
                microtrack_report_path,
            )
        ],
    }
    (microtrack_dir / stage.review.SUCCESS_NAME).write_text(
        json.dumps(s01), encoding="utf-8"
    )
    return missing_videos


def test_plan_loader_uses_only_s00_s01_tables_and_config(tmp_path: Path) -> None:
    ingest_dir = tmp_path / "00_ingest"
    microtrack_dir = tmp_path / "01_microtrack"
    missing_videos = _write_tiny_completed_inputs(ingest_dir, microtrack_dir)

    loaded = stage._load_plan_data(
        ingest_dir,
        microtrack_dir,
        ROOT / "configs" / "s01_review.yaml",
        progress_interval_sec=10.0,
        logger=lambda _: None,
    )

    assert all(not path.exists() for path in missing_videos)
    assert tuple(loaded.video_paths) == tuple(
        f"clip_{index:02d}" for index in range(11)
    )
    assert tuple(loaded.video_paths.values()) == missing_videos
    fingerprint_paths = {
        Path(str(record["path"])) for record in loaded.input_fingerprints
    }
    assert not fingerprint_paths.intersection(missing_videos)
    assert loaded.micro_paths[0].tolist() == list(range(11))


def test_plan_only_runs_tiny_eleven_clip_inputs_without_video(tmp_path: Path) -> None:
    ingest_dir = tmp_path / "00_ingest"
    microtrack_dir = tmp_path / "01_microtrack"
    missing_videos = _write_tiny_completed_inputs(ingest_dir, microtrack_dir)
    output_dir = tmp_path / "01_review_plan"

    success = stage.run_s01_review_plan_only(
        ingest_dir,
        microtrack_dir,
        ROOT / "configs" / "s01_review.yaml",
        output_dir,
        logger=lambda _: None,
    )

    assert success["stats"] == {
        "num_cases": 0,
        "num_risk_cases": 0,
        "num_quality_reference_cases": 0,
        "num_frames_planned": 0,
        "num_frames_rendered": 0,
        "num_videos_rendered": 0,
    }
    manifest = json.loads(
        (output_dir / stage.review.MANIFEST_NAME).read_text(encoding="utf-8")
    )
    assert manifest["selection_plan"]["cases"] == []
    assert manifest["cases"] == []
    assert all(not path.exists() for path in missing_videos)


def test_plan_only_commits_revalidates_and_is_s02_compatible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ingest_dir = tmp_path / "00_ingest"
    microtrack_dir = tmp_path / "01_microtrack"
    ingest_dir.mkdir()
    microtrack_dir.mkdir()
    immutable_input = tmp_path / "immutable-input.bin"
    immutable_input.write_bytes(b"immutable")
    output_dir = tmp_path / "01_review_plan"
    _patch_tiny_plan(monkeypatch, immutable_input)

    validation_paths: list[Path] = []
    original_validate = stage._validate_completed_output

    def recording_validate(output: Path, *args, **kwargs) -> None:
        validation_paths.append(output)
        original_validate(output, *args, **kwargs)

    monkeypatch.setattr(stage, "_validate_completed_output", recording_validate)
    first = stage.run_s01_review_plan_only(
        ingest_dir,
        microtrack_dir,
        ROOT / "configs" / "s01_review.yaml",
        output_dir,
        logger=lambda _: None,
    )

    assert first["stage"] == stage.STAGE_NAME
    assert first["program_commit_hash"] is None
    assert first["stats"]["num_videos_rendered"] == 0
    assert [path.name for path in validation_paths] == [
        f".{output_dir.name}.staging-{stage.os.getpid()}",
        output_dir.name,
    ]
    assert {path.name for path in output_dir.iterdir()} == {
        stage.review.MANIFEST_NAME,
        stage.review.SUCCESS_NAME,
    }
    assert not (output_dir / stage.review.LABELS_NAME).exists()
    assert not list(output_dir.rglob("*.mp4"))

    manifest_path = output_dir / stage.review.MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == appearance.REVIEW_SCHEMA_VERSION
    assert manifest["execution_mode"] == stage.EXECUTION_MODE
    assert manifest["videos_rendered"] is False
    assert manifest["human_labels_required"] is False
    assert manifest["cases"][0]["render"] == {
        "status": "not_rendered",
        "reason": "plan_only",
    }

    exclusions = appearance._load_review_exclusions(
        manifest_path, _appearance_data()
    )
    assert exclusions.num_true_events == 1
    assert np.flatnonzero(exclusions.excluded).tolist() == [0, 1, 2]

    validation_paths.clear()
    resumed = stage.run_s01_review_plan_only(
        ingest_dir,
        microtrack_dir,
        ROOT / "configs" / "s01_review.yaml",
        output_dir,
        logger=lambda _: None,
    )
    assert resumed == first
    assert validation_paths == [output_dir]


def test_plan_only_rejects_nonempty_output_without_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ingest_dir = tmp_path / "00_ingest"
    microtrack_dir = tmp_path / "01_microtrack"
    ingest_dir.mkdir()
    microtrack_dir.mkdir()
    output_dir = tmp_path / "01_review_plan"
    output_dir.mkdir()
    (output_dir / "orphan.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ContractError, match="non-empty without _SUCCESS.json"):
        stage.run_s01_review_plan_only(
            ingest_dir,
            microtrack_dir,
            ROOT / "configs" / "s01_review.yaml",
            output_dir,
            logger=lambda _: None,
        )


def test_plan_only_detects_committed_manifest_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ingest_dir = tmp_path / "00_ingest"
    microtrack_dir = tmp_path / "01_microtrack"
    ingest_dir.mkdir()
    microtrack_dir.mkdir()
    immutable_input = tmp_path / "immutable-input.bin"
    immutable_input.write_bytes(b"immutable")
    output_dir = tmp_path / "01_review_plan"
    _patch_tiny_plan(monkeypatch, immutable_input)
    stage.run_s01_review_plan_only(
        ingest_dir,
        microtrack_dir,
        ROOT / "configs" / "s01_review.yaml",
        output_dir,
        logger=lambda _: None,
    )
    manifest_path = output_dir / stage.review.MANIFEST_NAME
    manifest_path.write_text(
        manifest_path.read_text(encoding="utf-8") + " ", encoding="utf-8"
    )

    with pytest.raises(ContractError, match="artifact changed"):
        stage.run_s01_review_plan_only(
            ingest_dir,
            microtrack_dir,
            ROOT / "configs" / "s01_review.yaml",
            output_dir,
            logger=lambda _: None,
        )


def test_default_plan_only_log_flushes() -> None:
    with patch("builtins.print") as mocked_print:
        stage.log("visible")

    mocked_print.assert_called_once_with("visible", flush=True)
