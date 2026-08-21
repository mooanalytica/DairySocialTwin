from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import cowtrack.stages.s02_appearance as stage
from cowtrack.appearance.config import load_appearance_config
from cowtrack.appearance.encoders import EncoderProfile, production_encoder_profiles
from cowtrack.config import ClipManifest, ContractError


ROOT = Path(__file__).resolve().parents[1]


def _locked_profiles() -> tuple[EncoderProfile, ...]:
    return tuple(
        profile
        for profile in production_encoder_profiles()
        if profile.name == "megadescriptor_l_384_imagenet"
    )


def _manifest(tmp_path: Path, count: int = 11) -> list[ClipManifest]:
    return [
        ClipManifest(
            sequence_id="tiny_sequence",
            clip_order=index,
            clip_id=f"clip_{index:02d}",
            video_path=tmp_path / f"clip_{index:02d}.MP4",
            bbox_csv_path=tmp_path / f"clip_{index:02d}.csv",
            frame_index_base=0,
            bbox_format="xywh",
        )
        for index in range(count)
    ]


def test_manifest_and_boundaries_accept_ordered_eleven_clip_contract(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path)
    resolved = tmp_path / "resolved_manifest.json"
    resolved.write_text(json.dumps(stage._manifest_rows(manifest)), encoding="utf-8")

    video_paths = stage._validate_manifest_bijection(manifest, resolved)
    assert tuple(video_paths) == tuple(row.clip_id for row in manifest)

    boundaries = []
    for index, clip in enumerate(manifest):
        boundaries.append(
            {
                "clip_id": clip.clip_id,
                "clip_order": clip.clip_order,
                "num_frames": 2,
                "start_global_frame": index * 2,
                "end_global_frame_inclusive": index * 2 + 1,
                "width": stage.RAW_WIDTH,
                "height": stage.RAW_HEIGHT,
                "opencv_auto_rotate": False,
            }
        )
    validated = stage._validate_clip_boundaries(
        manifest, boundaries, num_frames=22
    )
    assert len(validated) == 11

    boundaries[5] = dict(boundaries[5], clip_id="wrong")
    with pytest.raises(ContractError, match="boundary order differs"):
        stage._validate_clip_boundaries(manifest, boundaries, num_frames=22)


def test_upstream_counts_are_cross_validated_without_fixed_totals() -> None:
    s00 = {
        "stats": {
            "num_clips": 11,
            "num_frames": 123,
            "num_input_boxes": 456,
            "num_valid_boxes": 450,
        }
    }
    s01 = {
        "stats": {
            "num_frames": 123,
            "num_valid_detections": 450,
            "num_microtracklets": 37,
        }
    }

    assert stage._validate_upstream_stats(
        s00, s01, manifest_num_clips=11
    ) == {
        "num_clips": 11,
        "num_frames": 123,
        "num_input_detections": 456,
        "num_valid_detections": 450,
        "num_microtracklets": 37,
    }

    s01["stats"]["num_valid_detections"] = 449
    with pytest.raises(ContractError, match="valid detection counts differ"):
        stage._validate_upstream_stats(s00, s01, manifest_num_clips=11)


def _data() -> stage.LoadedAppearanceData:
    det_ids = np.arange(8, dtype=np.int64) + 100
    frames = np.asarray([0, 10, 20, 40, 0, 10, 20, 40], dtype=np.int64)
    micro = np.asarray([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64)
    paths = {0: np.arange(4, dtype=np.int64), 1: np.arange(4, 8, dtype=np.int64)}
    return stage.LoadedAppearanceData(
        sequence_id="tiny_sequence",
        frames={},
        detections={
            "det_id": det_ids,
            "global_frame": frames,
            "global_time_sec": frames.astype(np.float64) / 30.0,
            "clip_id": np.asarray(["clip_00"] * 8, dtype=object),
        },
        microtracklets={
            "micro_id": np.asarray([0, 1], dtype=np.int64),
            "num_detections": np.asarray([4, 4], dtype=np.int32),
            "status": np.asarray(["valid", "valid"], dtype=object),
        },
        micro_ids_by_detection=micro,
        order_in_micro_by_detection=np.asarray([0, 1, 2, 3, 0, 1, 2, 3], dtype=np.int32),
        paths=paths,
        positions_by_frame=np.arange(8, dtype=np.int64),
        frame_offsets=np.zeros(42, dtype=np.int64),
        video_paths={},
        input_fingerprints=(),
    )


def test_review_exclusion_targets_only_case_micro_and_true_window(
    tmp_path: Path,
) -> None:
    case = {
        "case_id": "risk-1",
        "case_kind": "risk",
        "micro_id": 0,
        "reasons": ["large_center_jump", "long_full_review"],
        "events": [
            {
                "reason": "large_center_jump",
                "global_frame": 10,
                "global_time_sec": 10.0 / 30.0,
                "src_det_id": 100,
                "dst_det_id": 101,
            },
            {
                "reason": "long_full_review",
                "global_frame": 40,
            },
        ],
        "anchor": {"global_frame": 10},
        "microtrack": {"status": "valid", "num_detections": 4},
    }
    payload = {
        "schema_version": stage.REVIEW_SCHEMA_VERSION,
        "coordinate_system": stage.EXPECTED_COORDINATE_SYSTEM,
        "output_video_contract": {
            "true_event_highlight_radius_frames": 15,
            "synthetic_long_and_quality_anchors_are_not_events": True,
        },
        "cases": [dict(case, render={"status": "completed"})],
        "selection_plan": {"cases": [case]},
    }
    path = tmp_path / "review_manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    result = stage._load_review_exclusions(path, _data())

    assert result.num_true_events == 1
    assert np.flatnonzero(result.excluded).tolist() == [0, 1, 2]
    assert not np.any(result.excluded[4:])
    assert result.reasons[1] == "large_center_jump"
    assert result.trigger_frames[1] == "10"


def test_selection_reason_uses_eligible_not_original_endpoint() -> None:
    data = _data()
    excluded = np.zeros(8, dtype=np.bool_)
    excluded[0] = True
    selected = np.asarray([1, 2, 3], dtype=np.int64)

    reasons = stage._selection_reasons(data, selected, excluded)

    assert reasons[1] == "head;tail"
    assert reasons[2] == "head;tail"
    assert reasons[3] == "head;tail"


class _FakeEncoder:
    def __init__(self, profile: EncoderProfile) -> None:
        self.profile = profile

    def embed(self, crops: list[np.ndarray], *, batch_size: int) -> np.ndarray:
        result = np.zeros((len(crops), self.profile.embedding_dim), dtype=np.float32)
        for row, crop in enumerate(crops):
            result[row, 0 if int(crop[0, 0, 0]) < 100 else 1] = 1.0
        return result


def test_embedding_session_builds_and_runs_only_locked_existing_winner() -> None:
    config, _, _ = load_appearance_config(ROOT / "configs" / "s02_appearance.yaml")
    profiles = _locked_profiles()
    built: list[str] = []

    def factory(profile: EncoderProfile, *, device: str) -> _FakeEncoder:
        built.append(profile.name)
        return _FakeEncoder(profile)

    session = stage._EmbeddingSession(
        profiles,
        config,
        device="cuda:0",
        encoder_factory=factory,
        logger=lambda message: None,
    )
    identity_a = np.full((4, 5, 3), 20, dtype=np.uint8)
    identity_b = np.full((4, 5, 3), 200, dtype=np.uint8)
    frames = (0, 0, 20, 20, 40, 40, 60, 60)
    micros = (0, 1, 0, 1, 0, 1, 0, 1)
    crops = [identity_a, identity_b] * 4
    records = [
        stage.CropRecord(
            detection_position=row,
            micro_id=micros[row],
            det_id=100 + row,
            clip_id="clip_00",
            global_frame=frames[row],
            global_time_sec=frames[row] / 30.0,
            crop_quality=1.0,
            other_bbox_max_iou=0.0,
            clipped_fraction=0.0,
            bbox_area_percentile=1.0,
            blur_score=100.0,
            distance_to_image_boundary=1.0,
            selection_reason="head",
        )
        for row in range(8)
    ]

    session.consume(crops, crops, records)
    embeddings, winner, report, warning = session.finalize()

    assert built == ["megadescriptor_l_384_imagenet"]
    assert winner == "megadescriptor_l_384_imagenet"
    assert embeddings.shape == (8, 1536)
    assert np.allclose(np.linalg.norm(embeddings, axis=1), 1.0)
    assert report["selection_mode"] == "locked_existing_winner"
    assert report["evaluation_mode"] == "locked_existing_winner"
    assert report["num_embedding_rows"] == 0
    assert report["num_production_embedding_rows"] == 8
    assert report["pair_counts"] == {"positive": 0, "negative": 0}
    assert not warning


def test_run_s02_mocked_end_to_end_commits_and_revalidates(
    tmp_path: Path, monkeypatch
) -> None:
    config, payload, _ = load_appearance_config(
        ROOT / "configs" / "s02_appearance.yaml"
    )
    clip_ids = tuple(f"clip_{index:02d}" for index in range(11))
    frames = np.repeat(np.arange(len(clip_ids), dtype=np.int64), 2)
    local_frames = np.zeros(len(frames), dtype=np.int64)
    micros = np.tile(np.asarray([0, 1], dtype=np.int64), len(clip_ids))
    det_ids = np.arange(len(frames), dtype=np.int64) + 1_000
    clips = np.repeat(np.asarray(clip_ids, dtype=object), 2)
    boxes = np.asarray(
        [[50.0, 50.0, 150.0, 150.0], [350.0, 50.0, 450.0, 150.0]]
        * len(clip_ids),
        dtype=np.float32,
    )
    paths = {
        0: np.arange(0, len(frames), 2, dtype=np.int64),
        1: np.arange(1, len(frames), 2, dtype=np.int64),
    }
    data = stage.LoadedAppearanceData(
        sequence_id="tiny_sequence",
        frames={},
        detections={
            "det_id": det_ids,
            "clip_id": clips,
            "local_frame": local_frames,
            "global_frame": frames,
            "global_time_sec": frames.astype(np.float64),
            "x1": boxes[:, 0],
            "y1": boxes[:, 1],
            "x2": boxes[:, 2],
            "y2": boxes[:, 3],
        },
        microtracklets={"micro_id": np.asarray([0, 1], dtype=np.int64)},
        micro_ids_by_detection=micros,
        order_in_micro_by_detection=np.repeat(
            np.arange(len(clip_ids), dtype=np.int32), 2
        ),
        paths=paths,
        positions_by_frame=np.arange(len(frames), dtype=np.int64),
        frame_offsets=np.arange(0, len(frames) + 2, 2, dtype=np.int64),
        video_paths={clip_id: tmp_path / f"{clip_id}.MP4" for clip_id in clip_ids},
        input_fingerprints=(),
    )
    exclusions = stage.ExclusionResult(
        excluded=np.zeros(len(frames), dtype=np.bool_),
        reasons=("",) * len(frames),
        case_ids=("",) * len(frames),
        trigger_frames=("",) * len(frames),
        num_true_events=0,
    )

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    monkeypatch.setattr(
        stage,
        "load_appearance_config",
        lambda path: (config, payload, "a" * 64),
    )
    monkeypatch.setattr(stage, "_load_data", lambda *args, **kwargs: data)
    monkeypatch.setattr(
        stage,
        "_resolve_encoder_profiles",
        lambda configured, fingerprints: _locked_profiles(),
    )
    monkeypatch.setattr(
        stage, "_load_review_exclusions", lambda path, loaded: exclusions
    )
    # Keep the expensive-encoder row count distinct from the final representative
    # count. Production has the same distinction (historically 59,423 vs 32,635).
    monkeypatch.setattr(
        stage,
        "_finalize_representative_samples",
        lambda records, embeddings, loaded_config: (
            [replace(record, selection_reason="head") for record in records[:8]],
            embeddings[:8],
        ),
    )

    base_frame = np.empty((stage.RAW_HEIGHT, stage.RAW_WIDTH, 3), dtype=np.uint8)
    base_frame[:, :250] = 20
    base_frame[:, 250:] = 200
    captures: list[object] = []

    class FakeCapture:
        def __init__(self) -> None:
            self.reads = 0
            self.released = False

        def read(self):  # type: ignore[no-untyped-def]
            self.reads += 1
            return True, base_frame

        def release(self) -> None:
            self.released = True

    def open_capture(path: Path):
        capture = FakeCapture()
        captures.append(capture)
        return capture

    monkeypatch.setattr(stage, "open_raw_video_capture", open_capture)
    output = tmp_path / "02_appearance"
    args = (
        tmp_path / "manifest.csv",
        tmp_path / "00_ingest",
        tmp_path / "01_microtrack",
        tmp_path / "review_manifest.json",
        tmp_path / "s02.yaml",
        "cuda:0",
        output,
    )
    first = stage.run_s02(
        *args,
        logger=lambda message: None,
        encoder_factory=lambda profile, *, device: _FakeEncoder(profile),
    )

    assert first["stats"]["num_frames"] == 11
    assert first["stats"]["num_valid_detections"] == 22
    assert first["stats"]["num_microtracklets"] == 2
    assert first["stats"]["num_high_quality_candidates"] == 22
    assert first["stats"]["num_appearance_samples"] == 8
    assert first["stats"]["num_bakeoff_crops"] == 0
    assert first["stats"]["selected_encoder"] == "megadescriptor_l_384_imagenet"
    choice = json.loads((output / "encoder_choice.json").read_text(encoding="utf-8"))
    benchmark = json.loads(
        (output / "encoder_benchmark.json").read_text(encoding="utf-8")
    )
    assert choice["selection_mode"] == "locked_existing_winner"
    assert choice["selected_profile"] == "megadescriptor_l_384_imagenet"
    assert choice["embedding_dim"] == 1536
    assert choice["no_runtime_fallback"] is True
    assert benchmark["selection_mode"] == "locked_existing_winner"
    assert benchmark["num_embedding_rows"] == 0
    assert benchmark["num_production_embedding_rows"] == first["stats"][
        "num_high_quality_candidates"
    ]
    assert len(captures) == 11
    assert [capture.reads for capture in captures] == [1] * 11  # type: ignore[attr-defined]
    assert all(capture.released for capture in captures)  # type: ignore[attr-defined]
    assert (output / "_SUCCESS.json").is_file()
    assert not list(tmp_path.glob(".02_appearance.staging-*"))

    second = stage.run_s02(
        *args,
        logger=lambda message: None,
        encoder_factory=lambda profile, *, device: _FakeEncoder(profile),
    )
    assert second == first
    assert len(captures) == 11
