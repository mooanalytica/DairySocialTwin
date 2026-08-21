from __future__ import annotations

from fractions import Fraction
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from cowtrack.config import ContractError
from cowtrack.qa import s05_review
from cowtrack.qa.ffprobe import QaMp4Metadata
from cowtrack.qa.s05_review import (
    EXPECTED_FRAME_PERIOD_SEC,
    EXPECTED_FRAME_RATE,
    CandidateEndpoint,
    LoadedS05ReviewData,
    RenderCase,
)
from cowtrack.qa.s05_review_config import load_s05_review_config
from cowtrack.qa.s05_review_plan import (
    S05EvidenceRecord,
    S05ReviewCase,
    S05ReviewPlan,
)


CONFIG = Path(__file__).parents[1] / "configs" / "s05_review.yaml"


def _evidence(index: int) -> S05EvidenceRecord:
    return S05EvidenceRecord(
        candidate_id=f"candidate-{index}",
        source_stable_id=1 + index * 2,
        target_stable_id=2 + index * 2,
        probability=0.9 - index * 0.01,
        candidate_margin=0.2,
        provisional_threshold=0.5,
        selected_probability_threshold=0.7,
        selected_margin_threshold=0.1,
        appearance_rank_out=1,
        appearance_rank_in=1,
        high_overlap=False,
        passes_selected_probability_gate=True,
        passes_selected_margin_gate=True,
        passes_selected_gate=True,
    )


def _plan(count: int) -> S05ReviewPlan:
    candidates = tuple(_evidence(index) for index in range(count))
    cases = tuple(
        S05ReviewCase(item, "all_available", index + 1)
        for index, item in enumerate(candidates)
    )
    return S05ReviewPlan(
        candidates=candidates,
        cases=cases,
        provisional_threshold=0.5,
        random_seed=20260710,
        maximum_cases=50,
        high_score_count=15,
        threshold_near_count=15,
        ambiguous_count=10,
        random_count=10,
        threshold_probability_band=0.03,
        ambiguity_margin_upper=0.08,
        ambiguous_mutual_rank_above=1,
    )


def _data(count: int = 1) -> LoadedS05ReviewData:
    frame_count = 400
    times = np.arange(frame_count, dtype=np.float64) * EXPECTED_FRAME_PERIOD_SEC
    plan = _plan(count)
    endpoints = {
        candidate.candidate_id: CandidateEndpoint(
            candidate_id=candidate.candidate_id,
            source_stable_id=candidate.source_stable_id,
            target_stable_id=candidate.target_stable_id,
            source_start_global_frame=99,
            source_end_global_frame=100,
            target_start_global_frame=280,
            target_end_global_frame=281,
            source_start_time_sec=float(times[99]),
            source_end_time_sec=float(times[100]),
            target_start_time_sec=float(times[280]),
            target_end_time_sec=float(times[281]),
            source_end_clip_id="clip",
            target_start_clip_id="clip",
        )
        for candidate in plan.candidates
    }
    return LoadedS05ReviewData(
        frames={
            "global_frame": np.arange(frame_count),
            "global_time_sec": times,
            "clip_id": np.asarray(["clip"] * frame_count, dtype=object),
            "local_frame": np.arange(frame_count),
            "pts_sec": times.copy(),
            "width": np.full(frame_count, 3840),
            "height": np.full(frame_count, 2160),
        },
        detections={
            "x1": np.empty(0),
            "y1": np.empty(0),
            "x2": np.empty(0),
            "y2": np.empty(0),
        },
        stable_id_by_detection_position=np.empty(0, dtype=np.int64),
        valid_detection_positions_by_frame=np.empty(0, dtype=np.int64),
        valid_detection_frame_offsets=np.zeros(frame_count + 1, dtype=np.int64),
        candidates_by_id=endpoints,
        video_paths={"clip": Path("clip.mp4")},
        plan=plan,
        input_fingerprints=(),
    )


def test_review_windows_are_two_disjoint_ranges_and_omit_the_long_gap() -> None:
    config, _, _ = load_s05_review_config(CONFIG)
    data = _data()
    render_case = s05_review._render_cases(data, config)[0]

    assert (
        render_case.source_window_start_global_frame,
        render_case.source_window_end_global_frame,
    ) == (41, 100)
    assert (
        render_case.target_window_start_global_frame,
        render_case.target_window_end_global_frame,
    ) == (280, 339)
    assert render_case.expected_frame_count == 120
    assert render_case.frame_sequence == tuple(range(41, 101)) + tuple(range(280, 340))
    assert not set(range(101, 280)).intersection(render_case.frame_sequence)

    record = s05_review._case_record(data, render_case)
    assert record["source_window"]["frame_count"] == 60
    assert record["target_window"]["frame_count"] == 60
    assert record["expected_frame_count"] == 120
    assert record["concatenation_order"] == ["source_window", "target_window"]
    assert record["long_gap_frames_rendered"] == 0


def test_frame_roles_use_only_real_competing_stable_detections() -> None:
    plan = _plan(1)
    evidence = plan.candidates[0]
    case = plan.cases[0]
    boxes = np.asarray(
        [[1, 1, 5, 5], [11, 1, 15, 5], [21, 1, 25, 5], [31, 1, 35, 5]],
        dtype=np.float64,
    )
    endpoint = CandidateEndpoint(
        candidate_id=evidence.candidate_id,
        source_stable_id=evidence.source_stable_id,
        target_stable_id=evidence.target_stable_id,
        source_start_global_frame=0,
        source_end_global_frame=0,
        target_start_global_frame=8,
        target_end_global_frame=8,
        source_start_time_sec=0.0,
        source_end_time_sec=0.0,
        target_start_time_sec=8.0,
        target_end_time_sec=8.0,
        source_end_clip_id="clip",
        target_start_clip_id="clip",
    )
    data = LoadedS05ReviewData(
        frames={"global_frame": np.asarray([0])},
        detections={
            "x1": boxes[:, 0],
            "y1": boxes[:, 1],
            "x2": boxes[:, 2],
            "y2": boxes[:, 3],
        },
        stable_id_by_detection_position=np.asarray(
            [evidence.source_stable_id, evidence.target_stable_id, 3, 9]
        ),
        valid_detection_positions_by_frame=np.arange(4),
        valid_detection_frame_offsets=np.asarray([0, 4]),
        candidates_by_id={evidence.candidate_id: endpoint},
        video_paths={},
        plan=plan,
        input_fingerprints=(),
    )
    render_case = RenderCase(
        case=case,
        endpoint=endpoint,
        competing_stable_ids=(3,),
        source_window_start_global_frame=0,
        source_window_end_global_frame=0,
        target_window_start_global_frame=8,
        target_window_end_global_frame=8,
        output_relative_path="videos/case.mp4",
    )

    unrelated, source, target, competing = s05_review._frame_role_boxes(
        data, render_case, 0
    )
    assert [len(item) for item in (unrelated, source, target, competing)] == [1, 1, 1, 1]
    np.testing.assert_array_equal(unrelated[0], [31, 1, 35, 5])
    np.testing.assert_array_equal(competing[0], [21, 1, 25, 5])


def test_render_reads_exact_concatenated_sequence_and_uses_nvenc_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = _data()
    base = s05_review._render_cases(
        data, load_s05_review_config(CONFIG)[0]
    )[0]
    render_case = RenderCase(
        case=base.case,
        endpoint=base.endpoint,
        competing_stable_ids=(),
        source_window_start_global_frame=1,
        source_window_end_global_frame=2,
        target_window_start_global_frame=8,
        target_window_end_global_frame=9,
        output_relative_path="videos/case.mp4",
    )
    reads: list[int] = []
    writer_contract: dict[str, Any] = {}

    class FakeReader:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def __enter__(self) -> FakeReader:
            return self

        def __exit__(self, *args: Any) -> bool:
            return False

        def read(self, clip_id: str, local_frame: int, pts: float) -> np.ndarray:
            reads.append(local_frame)
            return np.zeros((2, 4, 3), dtype=np.uint8)

    class FakeWriter:
        def __init__(self, output_path: Path, **kwargs: Any) -> None:
            self.output_path = output_path
            writer_contract.update(kwargs)

        def __enter__(self) -> FakeWriter:
            return self

        def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
            if exc_type is None:
                self.output_path.write_bytes(b"synthetic-mp4")
            return False

        def write(self, frame: np.ndarray) -> None:
            assert frame.shape == (1080, 1920, 3)

    output_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    monkeypatch.setattr(s05_review, "RawFrameReader", FakeReader)
    monkeypatch.setattr(s05_review, "NvencVideoWriter", FakeWriter)
    monkeypatch.setattr(s05_review, "render_s05_review_frame", lambda *a, **k: output_frame)
    monkeypatch.setattr(
        s05_review,
        "validate_qa_mp4",
        lambda path, **kwargs: QaMp4Metadata(
            path=Path(path),
            stream_index=0,
            codec_name="h264",
            width=1920,
            height=1080,
            average_frame_rate=EXPECTED_FRAME_RATE,
            frame_count=4,
        ),
    )
    config, _, _ = load_s05_review_config(CONFIG)

    result = s05_review._render_one_case(
        data, render_case, config, tmp_path, logger=lambda _: None
    )

    assert reads == [1, 2, 8, 9]
    assert writer_contract["expected_frame_count"] == 4
    assert writer_contract["logical_gpu"] == 0
    assert writer_contract["pixel_format"] == "yuv420p"
    assert result["status"] == "completed"
    assert result["output_fingerprint"]["path"] == "videos/case.mp4"


def test_interrupted_run_resumes_completed_case_and_detects_video_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    proposals = tmp_path / "proposals"
    ingest = tmp_path / "ingest"
    stable = tmp_path / "stable"
    output = tmp_path / "review"
    for path in (proposals, ingest, stable):
        path.mkdir()
    data = _data(2)
    rendered: list[str] = []
    validated: list[str] = []
    fail_second = {"value": True}

    monkeypatch.setattr(s05_review, "_preflight_nvenc", lambda config: None)
    monkeypatch.setattr(s05_review, "_load_data", lambda *args, **kwargs: data)
    monkeypatch.setattr(
        s05_review,
        "_verify_fingerprints_unchanged",
        lambda *args, **kwargs: None,
    )

    def fake_render(
        loaded: LoadedS05ReviewData,
        render_case: RenderCase,
        config: Any,
        output_dir: Path,
        *,
        logger: Any,
    ) -> dict[str, Any]:
        candidate_id = render_case.case.candidate.candidate_id
        if candidate_id == "candidate-1" and fail_second["value"]:
            raise ContractError("synthetic interruption")
        rendered.append(candidate_id)
        path = output_dir / render_case.output_relative_path
        path.write_bytes(f"video:{candidate_id}".encode())
        fingerprint = s05_review._relative_fingerprint(path, output_dir)
        return {
            "status": "completed",
            "completed_at_unix_sec": 1.0,
            "output_fingerprint": fingerprint,
            "video_validation": {
                "codec_name": "h264",
                "width": 1920,
                "height": 1080,
                "average_frame_rate": str(EXPECTED_FRAME_RATE),
                "num_frames": render_case.expected_frame_count,
            },
        }

    def fake_validate(record: dict[str, Any], output_dir: Path, config: Any) -> None:
        validated.append(str(record["candidate_id"]))
        current = s05_review._relative_fingerprint(
            output_dir / record["output_path"], output_dir
        )
        if current != record["render"]["output_fingerprint"]:
            raise ContractError("completed S05 review video changed")

    monkeypatch.setattr(s05_review, "_render_one_case", fake_render)
    monkeypatch.setattr(s05_review, "_validate_completed_video", fake_validate)

    with pytest.raises(ContractError, match="synthetic interruption"):
        s05_review.run_s05_review(proposals, ingest, stable, CONFIG, output)
    interrupted = json.loads((output / s05_review.MANIFEST_NAME).read_text())
    assert [item["render"]["status"] for item in interrupted["cases"]] == [
        "completed",
        "pending",
    ]
    assert rendered == ["candidate-0"]

    fail_second["value"] = False
    resumed = s05_review.run_s05_review(proposals, ingest, stable, CONFIG, output)
    assert [item["render"]["status"] for item in resumed["cases"]] == [
        "completed",
        "completed",
    ]
    assert rendered == ["candidate-0", "candidate-1"]
    assert "candidate-0" in validated
    assert (output / s05_review.SUCCESS_NAME).is_file()
    assert resumed["video_contract"]["frame_selection"]["full_long_gap_rendered"] is False
    assert resumed["video_contract"]["encoder"] == {
        "codec": "h264_nvenc",
        "physical_gpu": 1,
        "cuda_visible_devices": "1",
        "logical_gpu": 0,
        "cpu_encoder_fallback": False,
        "opencv_encoder_fallback": False,
        "pixel_format": "yuv420p",
    }

    first_video = output / resumed["cases"][0]["output_path"]
    first_video.write_bytes(b"tampered-video")
    with pytest.raises(ContractError, match="video changed"):
        s05_review.run_s05_review(proposals, ingest, stable, CONFIG, output)


def test_nvenc_preflight_is_gpu1_only_and_has_no_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _, _ = load_s05_review_config(CONFIG)
    calls: list[list[str]] = []
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(s05_review.subprocess, "run", lambda *a, **k: None)
    with pytest.raises(ContractError, match="exactly equal to '1'"):
        s05_review._preflight_nvenc(config)

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    monkeypatch.setattr(s05_review.shutil, "which", lambda name: f"/usr/bin/{name}")

    def fail(command: list[str], **kwargs: Any) -> SimpleNamespace:
        calls.append(command)
        return SimpleNamespace(returncode=234, stderr=b"synthetic nvenc failure")

    monkeypatch.setattr(s05_review.subprocess, "run", fail)
    with pytest.raises(ContractError, match="preflight failed"):
        s05_review._preflight_nvenc(config)
    assert len(calls) == 1
    assert "h264_nvenc" in calls[0]
    assert calls[0][calls[0].index("-gpu") + 1] == "0"
    assert "libx264" not in calls[0]


def test_s05_proposal_marker_policy_is_exact_and_identity_safe() -> None:
    fingerprint = {"size_bytes": 1, "sha256": "b" * 64}
    marker = {
        "schema_version": "1.0",
        "stage": "S05_PROPOSE",
        "config_hash": "a" * 64,
        "execution_mode": "long_proposal_only",
        "automatic_merge_allowed": False,
        "confirmed_links_allowed": False,
        "solver_used": False,
        "path_cover_used": False,
        "num_merges": 0,
        "input_fingerprints": [
            {"path": "/tmp/s05-review-input.bin", **fingerprint}
        ],
        "output_fingerprints": [
            {"path": name, **fingerprint}
            for name in sorted(s05_review._PROPOSAL_OUTPUTS)
        ],
        "stats": {
            "num_candidates": 3,
            "num_proposals": 2,
            "num_rejects": 1,
            "num_confirmed": 0,
            "num_solver_selected": 0,
            "num_merges": 0,
        },
        "elapsed_sec": 1.0,
    }
    assert s05_review._validate_proposal_marker(marker) is marker

    changed = dict(marker, automatic_merge_allowed=True)
    with pytest.raises(ContractError, match="proposal-only policy"):
        s05_review._validate_proposal_marker(changed)
    changed = {**marker, "extra": False}
    with pytest.raises(ContractError, match="fields differ"):
        s05_review._validate_proposal_marker(changed)
    changed = {**marker, "stats": {**marker["stats"], "num_rejects": 2}}
    with pytest.raises(ContractError, match="counts disagree"):
        s05_review._validate_proposal_marker(changed)
    changed = {**marker, "stats": {**marker["stats"], "num_confirmed": True}}
    with pytest.raises(ContractError, match="stat num_confirmed"):
        s05_review._validate_proposal_marker(changed)
    changed = {**marker, "num_merges": False}
    with pytest.raises(ContractError, match="proposal-only policy"):
        s05_review._validate_proposal_marker(changed)


def test_s00_valid_detection_ids_join_s04_by_key_not_row_order() -> None:
    frames = {"global_frame": np.arange(2)}
    detections = {
        "det_id": np.asarray([5, 999, -2], dtype=np.int64),
        "valid": np.asarray([True, False, True]),
        "global_frame": np.asarray([1, 0, 0], dtype=np.int64),
    }
    stable = SimpleNamespace(
        det_ids=np.asarray([-2, 5], dtype=np.int64),
        det_stable_ids=np.asarray([7, 8], dtype=np.int64),
    )

    by_position, canonical, offsets = s05_review._build_detection_indices(
        frames, detections, stable
    )
    np.testing.assert_array_equal(by_position, [8, -1, 7])
    np.testing.assert_array_equal(canonical, [2, 0])
    np.testing.assert_array_equal(offsets, [0, 1, 2])

    stable.det_ids = np.asarray([-3, 5], dtype=np.int64)
    with pytest.raises(ContractError, match="det_id set differs"):
        s05_review._build_detection_indices(frames, detections, stable)


def test_run_fails_before_creating_output_when_gpu_contract_is_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = [tmp_path / name for name in ("proposals", "ingest", "stable")]
    for path in inputs:
        path.mkdir()
    output = tmp_path / "review"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")

    with pytest.raises(ContractError, match="exactly equal to '1'"):
        s05_review.run_s05_review(*inputs, CONFIG, output)
    assert not output.exists()


def test_run_binds_identity_to_one_config_byte_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = [tmp_path / name for name in ("proposals", "ingest", "stable")]
    for path in inputs:
        path.mkdir()
    config_path = tmp_path / "s05_review.yaml"
    config_path.write_bytes(CONFIG.read_bytes())
    output = tmp_path / "review"
    real_loader = load_s05_review_config

    def mutate_after_load(path: Path):
        result = real_loader(path)
        path.write_bytes(path.read_bytes() + b"\n")
        return result

    monkeypatch.setattr(s05_review, "load_s05_review_config", mutate_after_load)
    with pytest.raises(ContractError, match="config changed while being loaded"):
        s05_review.run_s05_review(*inputs, config_path, output)
    assert not output.exists()
