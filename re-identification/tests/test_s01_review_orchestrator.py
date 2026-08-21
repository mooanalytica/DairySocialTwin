from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from cowtrack.config import ContractError
from cowtrack.qa.s01_review_plan import ReviewCase
import cowtrack.qa.s01_review as review
from cowtrack.qa.review_config import load_s01_review_config


ROOT = Path(__file__).resolve().parents[1]


def _case(*events: tuple[tuple[str, object], ...]) -> ReviewCase:
    return ReviewCase(
        case_id="case",
        case_kind="risk",
        micro_id=1,
        anchor_det_id=10,
        anchor_sequence_id="sequence",
        anchor_clip_id="clip",
        anchor_local_frame=10,
        anchor_global_frame=10,
        anchor_global_time_sec=1.0,
        window_start_time_sec=0.0,
        window_end_time_sec=2.0,
        reasons=("legacy_id_transition",),
        events=events,
        num_detections=20,
        status="valid",
        local_purity_score=0.99,
        max_internal_center_jump=0.01,
    )


def test_only_real_risk_triggers_are_event_frames() -> None:
    case = _case(
        (("reason", "long_full_review"), ("global_frame", 10)),
        (("reason", "quality_reference"), ("global_frame", 11)),
        (("reason", "legacy_id_transition"), ("global_frame", 12)),
        (("reason", "large_center_jump"), ("global_frame", 13)),
        (("reason", "low_purity"), ("global_frame", 14)),
    )

    assert review._true_trigger_frames(case) == {12, 13, 14}


def test_bbox_only_case_record_contains_no_crop_metadata() -> None:
    render_case = review.RenderCase(
        case=_case(),
        start_global_frame=0,
        end_global_frame=9,
        output_relative_path="risk/case.mp4",
    )

    record = review._case_record(render_case)

    assert record["expected_frame_count"] == 10
    assert record["output_path"] == "risk/case.mp4"
    assert not any("crop" in key for key in record)


def test_detection_frame_foreign_key_rejects_wrong_clip() -> None:
    frames = {
        "sequence_id": np.asarray(["sequence", "sequence"], dtype=object),
        "clip_id": np.asarray(["clip", "clip"], dtype=object),
        "local_frame": np.asarray([0, 1], dtype=np.int64),
        "global_frame": np.asarray([0, 1], dtype=np.int64),
        "global_time_sec": np.asarray([0.0, 0.1], dtype=np.float64),
    }
    detections = {
        "sequence_id": np.asarray(["sequence", "sequence"], dtype=object),
        "clip_id": np.asarray(["clip", "wrong"], dtype=object),
        "local_frame": np.asarray([0, 1], dtype=np.int64),
        "global_frame": np.asarray([0, 1], dtype=np.int64),
        "global_time_sec": np.asarray([0.0, 0.1], dtype=np.float64),
        "x1": np.asarray([1.0, 2.0]),
        "y1": np.asarray([1.0, 2.0]),
        "x2": np.asarray([10.0, 20.0]),
        "y2": np.asarray([10.0, 20.0]),
        "valid": np.asarray([True, True]),
    }

    with pytest.raises(ContractError, match="foreign key mismatch for clip_id"):
        review._validate_detection_frame_foreign_keys(frames, detections)


class _FakeCapture:
    def __init__(self) -> None:
        self.position = 0
        self.decoded = -1
        self.released = False
        self.seek_values: list[int] = []
        self.frame = np.zeros((review.RAW_HEIGHT, review.RAW_WIDTH, 3), dtype=np.uint8)

    def set(self, prop: int, value: float) -> bool:
        assert prop == cv2.CAP_PROP_POS_FRAMES
        self.position = int(value)
        self.seek_values.append(self.position)
        return True

    def read(self) -> tuple[bool, np.ndarray]:
        self.decoded = self.position
        self.position += 1
        return True, self.frame

    def get(self, prop: int) -> float:
        if prop == cv2.CAP_PROP_POS_FRAMES:
            return float(self.position)
        if prop == cv2.CAP_PROP_PTS:
            return float(self.decoded)
        if prop == cv2.CAP_PROP_POS_MSEC:
            return self.decoded * review.EXPECTED_FRAME_PERIOD_SEC * 1000.0
        raise AssertionError(prop)

    def release(self) -> None:
        self.released = True


def test_raw_reader_checks_identity_and_reads_sequentially(tmp_path: Path) -> None:
    capture = _FakeCapture()
    with review._RawFrameReader(
        {"clip": tmp_path / "source.mp4"},
        capture_factory=lambda _: capture,  # type: ignore[arg-type]
    ) as reader:
        first = reader.read("clip", 0, 0.0)
        second = reader.read("clip", 1, review.EXPECTED_FRAME_PERIOD_SEC)

    assert first.shape == (2160, 3840, 3)
    assert second.shape == (2160, 3840, 3)
    assert capture.seek_values == [0]
    assert capture.released is True


def test_labels_lock_immutable_columns_but_allow_human_fields(tmp_path: Path) -> None:
    path = tmp_path / review.LABELS_NAME
    records = [
        {
            "case_id": "case",
            "case_kind": "risk",
            "reasons": ["low_purity"],
            "output_path": "risk/case.mp4",
        }
    ]
    review._write_or_validate_labels(path, records)
    with path.open("r", encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle))
    row["verdict"] = "PASS"
    row["notes"] = "same cow"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=review.LABEL_COLUMNS)
        writer.writeheader()
        writer.writerow(row)
    review._write_or_validate_labels(path, records)

    row["video_path"] = "risk/wrong.mp4"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=review.LABEL_COLUMNS)
        writer.writeheader()
        writer.writerow(row)
    with pytest.raises(ContractError, match="immutable field changed"):
        review._write_or_validate_labels(path, records)


def test_output_lock_uses_kernel_exclusion_and_reuses_persistent_file(
    tmp_path: Path,
) -> None:
    lock_path = tmp_path / review.LOCK_NAME
    lock_path.write_text(json.dumps({"pid": 999_999_999}), encoding="utf-8")
    first = review._acquire_output_lock(tmp_path)
    assert first.path == lock_path
    with pytest.raises(ContractError, match="another S01 review process is active"):
        review._acquire_output_lock(tmp_path)
    review._release_output_lock(first)

    second = review._acquire_output_lock(tmp_path)
    review._release_output_lock(second)
    assert lock_path.is_file()


def test_output_tree_must_not_overlap_inputs(tmp_path: Path) -> None:
    ingest = tmp_path / "00_ingest"
    with pytest.raises(ContractError, match="must not overlap"):
        review.run_s01_review(
            ingest,
            tmp_path / "01_microtrack",
            tmp_path / "missing.yaml",
            ingest / "qa",
        )


def test_final_input_check_detects_same_size_same_mtime_content_change(
    tmp_path: Path,
) -> None:
    path = tmp_path / "input.bin"
    path.write_bytes(b"original")
    fingerprint = review._fingerprint(path)
    stat = path.stat()
    path.write_bytes(b"modified")
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))

    with pytest.raises(ContractError, match=r"changed during rendering.+sha256"):
        review._verify_inputs_unchanged(
            (fingerprint,),
            progress_interval_sec=10.0,
            logger=lambda _: None,
        )


def test_nvenc_preflight_uses_only_masked_gpu_and_fixed_codec(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_s01_review_config(ROOT / "configs" / "s01_review.yaml")[0]
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    monkeypatch.setattr(review.shutil, "which", lambda value: f"/usr/bin/{value}")
    calls: list[tuple[list[str], dict[str, object]]] = []

    def fake_run(command: list[str], **kwargs: object):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, stdout=b"", stderr=b"")

    monkeypatch.setattr(review.subprocess, "run", fake_run)
    review._preflight_nvenc(config)

    command, kwargs = calls[0]
    assert "h264_nvenc" in command
    assert command[command.index("-video_size") + 1] == "1920x1080"
    assert command[command.index("-framerate") + 1] == "30000/1001"
    assert command[command.index("-gpu") + 1] == "0"
    assert command[command.index("-preset") + 1] == "p4"
    assert command[command.index("-tune") + 1] == "hq"
    assert command[command.index("-rc:v") + 1] == "vbr"
    assert command[command.index("-cq:v") + 1] == "21"
    assert command[command.index("-b:v") + 1] == "0"
    assert command[command.index("-pix_fmt") + 1] == "yuv420p"
    assert "libx264" not in command
    assert kwargs["shell"] is False
    assert len(kwargs["input"]) == 1920 * 1080 * 3  # type: ignore[arg-type]


def test_nvenc_preflight_surfaces_hardware_error_without_retry_or_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_s01_review_config(ROOT / "configs" / "s01_review.yaml")[0]
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    monkeypatch.setattr(review.shutil, "which", lambda value: f"/usr/bin/{value}")
    calls: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: object):
        calls.append(command)
        return subprocess.CompletedProcess(
            command,
            234,
            stdout=b"",
            stderr=b"InitializeEncoder failed",
        )

    monkeypatch.setattr(review.subprocess, "run", fake_run)
    with pytest.raises(
        ContractError,
        match=r"GPU 1 NVENC preflight failed \(234\).+InitializeEncoder failed",
    ):
        review._preflight_nvenc(config)

    assert len(calls) == 1
    assert "h264_nvenc" in calls[0]
    assert "libx264" not in calls[0]


def test_nvenc_preflight_wrong_gpu_mask_never_starts_ffmpeg(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_s01_review_config(ROOT / "configs" / "s01_review.yaml")[0]
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    calls: list[list[str]] = []
    monkeypatch.setattr(
        review.subprocess,
        "run",
        lambda command, **kwargs: calls.append(command),
    )

    with pytest.raises(ContractError, match="exactly equal to '1'"):
        review._preflight_nvenc(config)
    assert calls == []


def test_new_output_rejects_orphan_video(tmp_path: Path) -> None:
    config = load_s01_review_config(ROOT / "configs" / "s01_review.yaml")[0]
    (tmp_path / "risk").mkdir()
    (tmp_path / "quality_reference").mkdir()
    (tmp_path / "risk" / "orphan.mp4").write_bytes(b"orphan")

    with pytest.raises(ContractError, match="orphan content"):
        review._initial_or_resumed_manifest(
            tmp_path,
            config_payload={},
            identity={"identity": "test"},
            plan_payload={},
            case_records=[],
            config=config,
        )


def test_resume_adopts_strictly_valid_uncommitted_video(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_s01_review_config(ROOT / "configs" / "s01_review.yaml")[0]
    (tmp_path / "risk").mkdir()
    (tmp_path / "quality_reference").mkdir()
    video_path = tmp_path / "risk" / "case.mp4"
    video_path.write_bytes(b"valid after injected ffprobe")
    record = {
        "case_id": "case",
        "case_kind": "risk",
        "output_path": "risk/case.mp4",
        "expected_frame_count": 10,
        "render": {"status": "pending"},
    }
    identity = {"identity": "test"}
    payload = {
        "schema_version": review.MANIFEST_SCHEMA_VERSION,
        "coordinate_system": "raw_encoded_landscape_no_autorotate",
        "output_video_contract": {
            "codec": "h264_nvenc",
            "physical_gpu": 1,
            "logical_gpu": config.logical_gpu,
            "width": config.output_width,
            "height": config.output_height,
            "average_frame_rate": str(review.EXPECTED_FRAME_RATE),
            "rotation": None,
            "cpu_encoder_fallback": False,
            "render_style": "bbox_colors_only_v1",
            "text": False,
            "inset": False,
            "trajectory": False,
            "markers": False,
            "bbox_colors_bgr": {
                "context": [150, 150, 150],
                "risk_target": [0, 165, 255],
                "quality_target": [45, 205, 70],
                "event_target": [40, 40, 235],
            },
            "true_event_highlight_radius_frames": (
                review.EVENT_HIGHLIGHT_RADIUS_FRAMES
            ),
            "synthetic_long_and_quality_anchors_are_not_events": True,
        },
        "selection_warning": (
            "Risk reasons select material for human review; they are not automatic FAIL labels. "
            "legacy_track_id is QA-only and is not ground truth."
        ),
        "identity": identity,
        "effective_config": {},
        "selection_plan": {},
        "cases": [record],
    }
    (tmp_path / review.MANIFEST_NAME).write_text(
        json.dumps(payload), encoding="utf-8"
    )
    fingerprint = review._fingerprint(video_path)
    render = {
        "status": "completed",
        "completed_at_unix_sec": 1.0,
        "output_fingerprint": {
            "path": "risk/case.mp4",
            "size_bytes": fingerprint["size_bytes"],
            "mtime_ns": fingerprint["mtime_ns"],
            "sha256": fingerprint["sha256"],
        },
        "video_validation": {
            "codec_name": "h264",
            "width": 1920,
            "height": 1080,
            "average_frame_rate": str(review.EXPECTED_FRAME_RATE),
            "num_frames": 10,
        },
    }
    journal = {
        "schema_version": review.COMPLETION_SCHEMA_VERSION,
        "case_id": "case",
        "case_commitment": review._case_commitment(record),
        "expected_frame_count": 10,
        "output_path": "risk/case.mp4",
        "render": render,
    }
    review._completion_journal_path(video_path).write_text(
        json.dumps(journal), encoding="utf-8"
    )
    monkeypatch.setattr(
        review,
        "validate_qa_mp4",
        lambda *args, **kwargs: SimpleNamespace(
            codec_name="h264",
            width=1920,
            height=1080,
            average_frame_rate=review.EXPECTED_FRAME_RATE,
            frame_count=10,
        ),
    )

    resumed = review._initial_or_resumed_manifest(
        tmp_path,
        config_payload={},
        identity=identity,
        plan_payload={},
        case_records=[record],
        config=config,
    )

    assert resumed["cases"][0]["render"]["status"] == "completed"
    assert resumed["cases"][0]["render"]["recovered_after_interruption"] is True
    assert not review._completion_journal_path(video_path).exists()
