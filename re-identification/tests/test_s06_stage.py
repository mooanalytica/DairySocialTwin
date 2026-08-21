from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pyarrow as pa
import pytest

from cowtrack.config import ContractError
from cowtrack.qa.s06_config import load_s06_export_config
from cowtrack.qa.s06_plan import CropRequest, CropSource, CropUse, build_crop_request_index
from cowtrack.qa.s06_report import StructuralMetrics
from cowtrack.schemas.frames import FRAMES_SCHEMA
from cowtrack.schemas.s06 import DETECTIONS_WITH_GLOBAL_ID_SCHEMA
from cowtrack.schemas.s05_forced import FORCED_CANDIDATE_EDGES_SCHEMA
import cowtrack.stages.s06_export as stage


def _config():
    fixed, _, _ = load_s06_export_config(
        Path(__file__).parents[1] / "configs/s06_export.yaml"
    )
    return fixed


def _csv_row(*, valid: bool) -> dict[str, object]:
    identity: dict[str, object] = {
        "global_track_id": 0,
        "global_track_uuid": "forced-u0",
        "display_global_id": "G0001",
        "id_status": "forced_provisional",
        "identity_basis": "operator_forced_appearance_exact_62",
        "micro_id": 0,
        "stable_id": 0,
        "order_in_micro": 0,
        "order_in_stable": 0,
        "order_in_stable_detection": 0,
        "order_in_global_stable": 0,
        "order_in_global_detection": 0,
        "local_purity_score": 0.9,
    }
    if not valid:
        identity = {name: None for name in identity}
    return {
        "sequence_id": "seq",
        "clip_id": "A",
        "clip_order": 0,
        "det_id": 10 if valid else 11,
        "csv_row_index": 0 if valid else 1,
        "legacy_track_id": "old-7",
        **identity,
        "local_frame": 0,
        "global_frame": 0,
        "global_time_sec": 0.0,
        "x1": 1.0,
        "y1": 2.0,
        "x2": 10.0,
        "y2": 20.0,
        "bbox_confidence": 0.8,
        "valid": valid,
        "qa_flags": 0 if valid else 16,
        "invalid_reason": None if valid else "AREA_BELOW_MINIMUM",
        "assignment_confidence": None,
        "incoming_link_probability": None,
        "outgoing_link_probability": None,
    }


def test_frame_observation_report_is_clip_generic_and_uses_all_source_rows() -> None:
    frames = pa.table(
        {
            "clip_id": ["farm2_cam3"] * 5 + ["future_clip"] * 3,
            "local_frame": [0, 1, 2, 3, 4, 0, 1, 2],
        }
    )
    detections = pa.table(
        {
            "clip_id": ["farm2_cam3", "farm2_cam3", "farm2_cam3", "future_clip"],
            "local_frame": [0, 0, 4, 1],
        }
    )

    report = stage._build_frame_observation_report(
        frames,
        detections,
        ("farm2_cam3", "future_clip"),
    )

    assert report["num_frames"] == 8
    assert report["num_observed_frames"] == 3
    assert report["num_unobserved_frames"] == 5
    assert report["unobserved_frames_are_empty_scene_evidence"] is False
    assert report["per_clip"]["farm2_cam3"]["unobserved_frame_intervals"] == [
        {"start_frame": 1, "end_frame": 3, "num_frames": 3}
    ]
    assert report["per_clip"]["future_clip"]["unobserved_frame_intervals"] == [
        {"start_frame": 0, "end_frame": 0, "num_frames": 1},
        {"start_frame": 2, "end_frame": 2, "num_frames": 1},
    ]


def test_qa_metrics_disclose_unobserved_passthrough_policy() -> None:
    config = replace(_config(), clip_order=("farm2_cam3",))
    frames = pa.table(
        {"clip_id": ["farm2_cam3"] * 3, "local_frame": [0, 1, 2]}
    )
    export = pa.table(
        {
            "clip_id": ["farm2_cam3"],
            "local_frame": [1],
            "valid": [True],
        }
    )
    rescue = pa.table(
        {
            "selected_for_descriptor": [False],
            "other_bbox_max_iou": [0.0],
            "stable_id": [0],
        }
    )
    bundle = SimpleNamespace(
        frames=frames,
        microtracklets=pa.table({"micro_id": [0]}),
        warnings=(),
        forced=SimpleNamespace(
            stable_to_global=pa.table({"stable_id": [0]}),
            rescue_samples=rescue,
            global_tracks=pa.table(
                {"duration_visible_sec": [1.0], "population_warning": [False]}
            ),
        ),
    )
    summary = pa.table({"duration_visible_sec": [1.0]})

    metrics = stage._build_qa_metrics(
        bundle,
        export,
        summary,
        (),
        StructuralMetrics(0, 0, 0, 0),
        config,
        config_hash="synthetic",
    )

    assert metrics["counts"]["observed_frames"] == 1
    assert metrics["counts"]["unobserved_frames"] == 2
    assert metrics["frame_observation"]["unobserved_frames_are_empty_scene_evidence"] is False
    assert metrics["render_contract"]["all_source_video_frames_preserved"] is True
    assert any("treated as unobserved" in warning for warning in metrics["warnings"])


def test_detection_csv_resume_validation_preserves_invalid_nulls(tmp_path: Path) -> None:
    config = replace(
        _config(),
        expected_total_detection_count=2,
        expected_valid_detection_count=1,
        expected_invalid_detection_count=1,
        expected_global_track_count=1,
    )
    table = pa.Table.from_pylist(
        [_csv_row(valid=True), _csv_row(valid=False)],
        schema=DETECTIONS_WITH_GLOBAL_ID_SCHEMA,
    )
    path = tmp_path / "detections.csv"
    stage._write_csv(path, table, DETECTIONS_WITH_GLOBAL_ID_SCHEMA)
    stage._validate_detection_csv(path, config)

    bad_rows = [_csv_row(valid=True), _csv_row(valid=False)]
    bad_rows[1]["global_track_id"] = 0
    bad = pa.Table.from_pylist(bad_rows, schema=DETECTIONS_WITH_GLOBAL_ID_SCHEMA)
    stage._write_csv(path, bad, DETECTIONS_WITH_GLOBAL_ID_SCHEMA)
    with pytest.raises(ContractError, match="invalid row has identity"):
        stage._validate_detection_csv(path, config)


def test_candidate_adapter_carries_authorization_provenance() -> None:
    row = {
        "candidate_id": "candidate-1",
        "source_stable_id": 0,
        "target_stable_id": 1,
        "source_end_clip_id": "A",
        "target_start_clip_id": "A",
        "source_end_global_frame": 0,
        "target_start_global_frame": 1,
        "source_end_time_sec": 0.0,
        "target_start_time_sec": 1.0,
        "temporal_gap_sec": 1.0,
        "strictly_nonoverlapping": True,
        "appearance_cosine": 0.75,
        "source_evidence_grade": "A_CLEAN",
        "target_evidence_grade": "B_EXISTING_DEGRADED",
        "selected_by_source_topk": True,
        "selected_by_target_topk": True,
        "temporal_backbone": False,
        "prior_global_link": False,
        "appearance_cost_int": 1,
        "solver_cost_int": 1,
        "selected_by_solver": True,
        "global_link_id": "link-1",
        "authorization_basis": "operator_forced_appearance_exact_62",
        "id_status": "forced_provisional",
    }
    table = pa.Table.from_pylist([row], schema=FORCED_CANDIDATE_EDGES_SCHEMA)
    metrics = stage._candidate_metrics(table)
    assert len(metrics) == 1
    assert metrics[0].authorization_basis == row["authorization_basis"]


def test_observed_qa_accepts_scores_without_historical_reference() -> None:
    config = _config()
    metrics = (
        SimpleNamespace(selected_by_solver=True, appearance_cosine=0.2),
        SimpleNamespace(selected_by_solver=True, appearance_cosine=0.8),
        SimpleNamespace(selected_by_solver=False, appearance_cosine=-0.1),
    )

    observed = stage._compute_observed_qa(
        metrics, StructuralMetrics(0, 0, 0, 0), config
    )

    assert observed["reference_mode"] == "observed_only"
    assert observed["reference_available"] is False
    assert observed["reference_comparison_performed"] is False
    assert observed["selected_link_count"] == 2
    assert observed["appearance_cosine"]["mean"] == pytest.approx(0.5)
    assert observed["counts_below_threshold"] == {
        "0.3": 1,
        "0.4": 1,
        "0.5": 1,
        "0.6": 1,
    }


def test_observed_qa_allows_zero_selected_links() -> None:
    observed = stage._compute_observed_qa(
        (SimpleNamespace(selected_by_solver=False, appearance_cosine=0.4),),
        StructuralMetrics(0, 0, 0, 0),
        _config(),
    )
    assert observed["selected_link_count"] == 0
    assert observed["appearance_cosine"] == {
        "min": None,
        "p10": None,
        "mean": None,
        "max": None,
    }
    assert set(observed["counts_below_threshold"].values()) == {0}


@pytest.mark.parametrize("structural", [
    StructuralMetrics(1, 0, 0, 0),
    StructuralMetrics(0, 1, 0, 0),
    StructuralMetrics(0, 0, 1, 0),
    StructuralMetrics(0, 0, 0, 1),
])
def test_observed_qa_still_rejects_structural_violations(
    structural: StructuralMetrics,
) -> None:
    with pytest.raises(ContractError, match="structure is invalid"):
        stage._compute_observed_qa(
            (SimpleNamespace(selected_by_solver=True, appearance_cosine=0.7),),
            structural,
            _config(),
        )


def test_provenance_audit_allows_no_reference_warning() -> None:
    rescue = pa.table(
        {
            "selected_for_descriptor": pa.array([True]),
            "other_bbox_max_iou": pa.array([0.9], type=pa.float32()),
            "stable_id": pa.array([999], type=pa.int64()),
        }
    )
    upstream_globals = pa.table(
        {
            "duration_visible_sec": pa.array([1.0], type=pa.float64()),
            "population_warning": pa.array([False]),
        }
    )
    summary = pa.table(
        {"duration_visible_sec": pa.array([1.0], type=pa.float64())}
    )
    bundle = SimpleNamespace(
        forced=SimpleNamespace(
            rescue_samples=rescue,
            global_tracks=upstream_globals,
        ),
        warnings=(),
    )

    warnings, audit = stage._provenance_audit(bundle, summary, _config())

    assert warnings == []
    assert audit["c_grade_selected_high_overlap_crop_count"] == 1
    assert audit["c_grade_high_overlap_stable_ids"] == [999]
    assert audit["s05_visible_duration_mismatch_global_track_count"] == 0
    assert audit["s05_population_warning_disagreement_count"] == 0


def test_nvenc_preflight_has_no_software_fallback(monkeypatch) -> None:
    config = replace(_config(), output_width=2, output_height=2)
    calls: list[tuple[list[str], int]] = []

    def fake_run(command: list[str], **kwargs: object) -> SimpleNamespace:
        calls.append((command, len(kwargs["input"])))  # type: ignore[arg-type]
        return SimpleNamespace(returncode=0, stderr=b"")

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    monkeypatch.setattr(stage.shutil, "which", lambda _binary: "/usr/bin/tool")
    monkeypatch.setattr(stage.subprocess, "run", fake_run)

    stage._preflight_nvenc(config)
    assert len(calls) == 1
    command, payload_size = calls[0]
    assert command[command.index("-c:v") + 1] == "h264_nvenc"
    assert command[command.index("-gpu") + 1] == "0"
    assert "libx264" not in command
    assert payload_size == 12


def test_nvenc_preflight_fails_without_retry_or_fallback(monkeypatch) -> None:
    config = replace(_config(), output_width=2, output_height=2)
    calls = 0

    def fail_once(_command: list[str], **_kwargs: object) -> SimpleNamespace:
        nonlocal calls
        calls += 1
        return SimpleNamespace(returncode=1, stderr=b"synthetic NVENC failure")

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    monkeypatch.setattr(stage.shutil, "which", lambda _binary: "/usr/bin/tool")
    monkeypatch.setattr(stage.subprocess, "run", fail_once)

    with pytest.raises(ContractError, match="NVENC preflight failed"):
        stage._preflight_nvenc(config)
    assert calls == 1


def test_input_stat_snapshot_detects_midrun_change(tmp_path: Path) -> None:
    path = (tmp_path / "input.bin").resolve()
    path.write_bytes(b"before")
    record = {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": stage._sha256(path),
    }
    snapshot = stage._input_stat_snapshot([record])
    stage._verify_input_stat_snapshot(snapshot)
    path.write_bytes(b"changed-content")
    with pytest.raises(ContractError, match="changed while exporting"):
        stage._verify_input_stat_snapshot(snapshot)


def test_input_stat_handoff_rejects_same_size_token_change(tmp_path: Path) -> None:
    path = (tmp_path / "input.bin").resolve()
    path.write_bytes(b"same-size")
    stat = path.stat()
    record = {
        "path": str(path),
        "size_bytes": stat.st_size,
        "sha256": stage._sha256(path),
    }
    stale = {
        str(path): (int(stat.st_size), int(stat.st_mtime_ns) - 1, int(stat.st_ino))
    }
    with pytest.raises(ContractError, match="changed after fingerprinting"):
        stage._input_stat_snapshot([record], expected_tokens=stale)


def test_output_enumerator_rejects_symlinks(tmp_path: Path) -> None:
    root = tmp_path / "output"
    root.mkdir()
    target = root / "real.txt"
    target.write_text("real", encoding="utf-8")
    (root / "alias.txt").symlink_to(target)
    with pytest.raises(ContractError, match="must not contain symlinks"):
        stage._official_output_files(root, "_SUCCESS.json")


def test_output_path_rejects_lexical_symlink_before_resolve(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(target, target_is_directory=True)
    with pytest.raises(ContractError, match="must not be a symlink"):
        stage._resolve_output_path(alias)


def test_decode_opens_each_clip_once_and_deduplicates_crop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixed = _config()
    artifacts = replace(
        fixed.artifacts,
        videos_by_clip=("qa/videos/A.mp4", "qa/videos/B.mp4"),
    )
    config = replace(
        fixed,
        clip_order=("A", "B"),
        frame_counts_by_clip=(2, 1),
        expected_valid_detections_by_clip=(1, 1),
        artifacts=artifacts,
        progress_interval_sec=999.0,
    )
    frames = pa.Table.from_pylist(
        [
            {
                "sequence_id": "seq",
                "clip_id": "A",
                "clip_order": 0,
                "local_frame": 0,
                "global_frame": 0,
                "pts_sec": 0.0,
                "global_time_sec": 0.0,
                "width": 3840,
                "height": 2160,
            },
            {
                "sequence_id": "seq",
                "clip_id": "A",
                "clip_order": 0,
                "local_frame": 1,
                "global_frame": 1,
                "pts_sec": 1001 / 30000,
                "global_time_sec": 1001 / 30000,
                "width": 3840,
                "height": 2160,
            },
            {
                "sequence_id": "seq",
                "clip_id": "B",
                "clip_order": 1,
                "local_frame": 0,
                "global_frame": 2,
                "pts_sec": 0.0,
                "global_time_sec": 2 * 1001 / 30000,
                "width": 3840,
                "height": 2160,
            },
        ],
        schema=FRAMES_SCHEMA,
    )
    rows = [_csv_row(valid=True), _csv_row(valid=True)]
    rows[0].update({"det_id": -7, "clip_id": "A", "clip_order": 0})
    rows[1].update(
        {
            "det_id": 8,
            "clip_id": "B",
            "clip_order": 1,
            "csv_row_index": 0,
            "global_frame": 2,
        }
    )
    export = pa.Table.from_pylist(rows, schema=DETECTIONS_WITH_GLOBAL_ID_SCHEMA)
    source = CropSource(-7, "A", 0, 1.0, 2.0, 10.0, 20.0)
    low = (CropRequest(source, CropUse("low_confidence_link:c", "source", "s")),)
    contact = (CropRequest(source, CropUse("contact_sheet:G0001", "start", "s")),)
    index = build_crop_request_index((*low, *contact))

    raw = np.zeros((2160, 3840, 3), dtype=np.uint8)
    opened: list[str] = []
    captures: list[SimpleNamespace] = []

    class FakeCapture:
        def __init__(self, clip: str) -> None:
            self.clip = clip
            self.count = 0
            self.released = False

        def read(self):  # type: ignore[no-untyped-def]
            self.count += 1
            return True, raw

        def get(self, prop: int) -> float:
            if prop == cv2.CAP_PROP_POS_FRAMES:
                return float(self.count)
            if prop == cv2.CAP_PROP_PTS:
                return float(self.count - 1)
            if prop == cv2.CAP_PROP_POS_MSEC:
                return float((self.count - 1) * 1001 / 30)
            return 0.0

        def release(self) -> None:
            self.released = True

    def open_capture(path: Path):  # type: ignore[no-untyped-def]
        clip = path.stem
        opened.append(clip)
        capture = FakeCapture(clip)
        captures.append(capture)  # type: ignore[arg-type]
        return capture

    written: dict[str, int] = {}

    class FakeWriter:
        def __init__(self, output_path: Path, **_kwargs: object) -> None:
            self.name = output_path.stem
            written[self.name] = 0

        def __enter__(self):  # type: ignore[no-untyped-def]
            return self

        def __exit__(self, *_args: object) -> bool:
            return False

        def write(self, _frame: np.ndarray) -> None:
            written[self.name] += 1

    validated: list[str] = []
    monkeypatch.setattr(stage, "open_raw_video_capture", open_capture)
    monkeypatch.setattr(stage, "NvencVideoWriter", FakeWriter)
    monkeypatch.setattr(
        stage,
        "render_s06_overlay_frame",
        lambda *_args, **_kwargs: np.zeros((1080, 1920, 3), dtype=np.uint8),
    )
    monkeypatch.setattr(
        stage,
        "validate_qa_mp4",
        lambda path, **_kwargs: validated.append(Path(path).stem),
    )
    bundle = SimpleNamespace(
        frames=frames,
        video_paths={"A": tmp_path / "A", "B": tmp_path / "B"},
    )
    staging = tmp_path / "staging"
    staging.mkdir()

    crop_paths = stage._decode_render_and_extract(
        bundle, export, index, low, contact, staging, config, logger=lambda _m: None
    )

    assert opened == ["A", "B"]
    assert [capture.count for capture in captures] == [2, 1]
    assert all(capture.released for capture in captures)
    assert written == {"A": 2, "B": 1}
    assert validated == ["A", "B"]
    assert set(crop_paths) == {-7}
    assert crop_paths[-7].is_file()


def _mock_bundle() -> SimpleNamespace:
    forced = SimpleNamespace(
        det_to_global=object(),
        stable_to_global=object(),
        candidate_edges=object(),
        global_tracks=object(),
        graded_stable_appearance=object(),
    )
    return SimpleNamespace(
        input_fingerprints=(),
        input_stat_tokens={},
        video_paths={},
        detections=object(),
        microtracklets=object(),
        forced=forced,
    )


def test_resolve_config_binds_null_counts_to_validated_bundle() -> None:
    config = _config()
    candidates = pa.table(
        {
            "selected_by_solver": [True, False],
            "prior_global_link": [True, False],
        }
    )
    rescue = pa.table(
        {
            "selected_for_descriptor": [True, False],
            "review_excluded": [True, False],
            "selection_reason": ["best_degraded_fallback", "candidate_not_selected"],
        }
    )
    forced = SimpleNamespace(
        candidate_edges=candidates,
        rescue_samples=rescue,
        graded_stable_appearance=pa.table(
            {
                "evidence_grade": [
                    "A_CLEAN",
                    "B_EXISTING_DEGRADED",
                    "C_REENCODED_DEGRADED",
                ]
            }
        ),
        stable_to_global=pa.table({"stable_id": [0, 1, 2]}),
    )
    stable_tracklets = {
        index: SimpleNamespace(start_global_frame=index * 2, end_global_frame=index * 2)
        for index in range(3)
    }
    bundle = SimpleNamespace(
        microtracklets=pa.table({"micro_id": [0, 1, 2, 3]}),
        forced=forced,
        s04=SimpleNamespace(stable_tracklets=stable_tracklets),
    )

    resolved = stage._resolve_config_from_bundle(config, bundle)

    assert config.expected_microtrack_count is None
    assert resolved.expected_microtrack_count == 4
    assert resolved.expected_stable_track_count == 3
    assert resolved.expected_candidate_edge_count == 2
    assert resolved.expected_selected_link_count == 1
    assert resolved.expected_selected_prior_link_count == 1
    assert resolved.expected_max_concurrent_stable_track_count == 1
    assert resolved.expected_maximum_feasible_link_count == 2
    assert resolved.expected_backbone_chain_count == 1
    assert resolved.expected_rescue_candidate_count == 2
    assert resolved.expected_rescue_embedding_count == 1
    assert resolved.expected_grade_c_reencoded_degraded_count == 1


def test_run_s06_resume_revalidates_without_preflight_or_render(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "output"
    output.mkdir()
    (output / "_SUCCESS.json").write_text('{"resume": true}\n', encoding="utf-8")
    monkeypatch.setattr(stage, "load_s06_inputs", lambda *_a, **_k: _mock_bundle())
    monkeypatch.setattr(stage, "_resolve_config_from_bundle", lambda config, _bundle: config)
    validated: list[Path] = []
    monkeypatch.setattr(
        stage,
        "_validate_completed_output",
        lambda path, *_a, **_k: validated.append(Path(path)),
    )
    monkeypatch.setattr(
        stage,
        "_preflight_nvenc",
        lambda _config: pytest.fail("resume must not require NVENC"),
    )
    monkeypatch.setattr(
        stage,
        "build_detection_export_table",
        lambda *_a, **_k: pytest.fail("resume must not rebuild CSV"),
    )

    marker = stage.run_s06(
        tmp_path / "manifest.csv",
        tmp_path / "ingest",
        tmp_path / "micro",
        tmp_path / "stable",
        tmp_path / "forced",
        Path(__file__).parents[1] / "configs/s06_export.yaml",
        output,
        logger=lambda _message: None,
    )

    assert marker == {"resume": True}
    assert validated == [output.resolve()]


def test_run_s06_render_failure_cleans_staging_and_publishes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = Path(__file__).parents[1] / "configs/s06_export.yaml"
    output = tmp_path / "output"
    bundle = _mock_bundle()
    metrics = tuple(SimpleNamespace(selected_by_solver=False) for _ in range(133))
    monkeypatch.setattr(stage, "_preflight_nvenc", lambda _config: None)
    monkeypatch.setattr(stage, "load_s06_inputs", lambda *_a, **_k: bundle)
    monkeypatch.setattr(stage, "_resolve_config_from_bundle", lambda config, _bundle: config)
    monkeypatch.setattr(stage, "build_detection_export_table", lambda *_a, **_k: object())
    monkeypatch.setattr(
        stage, "recompute_structural_metrics", lambda *_a: StructuralMetrics(0, 0, 0, 0)
    )
    monkeypatch.setattr(stage, "_candidate_metrics", lambda _table: metrics)
    monkeypatch.setattr(stage, "select_low_confidence_links", lambda *_a, **_k: metrics)
    monkeypatch.setattr(stage, "build_global_track_summary", lambda *_a, **_k: object())
    monkeypatch.setattr(
        stage,
        "_build_crop_and_contact_plans",
        lambda *_a, **_k: ((), (), (), SimpleNamespace()),
    )
    monkeypatch.setattr(
        stage,
        "_build_qa_metrics",
        lambda *_a, **_k: {
            "counts": {
                "total_detection_rows": 746279,
                "valid_detections": 745915,
                "invalid_detections": 364,
                "microtracklets": 4659,
                "stable_tracklets": 3769,
                "candidate_edges": 133,
                "selected_links": 0,
                "global_ids": 62,
            }
        },
    )
    monkeypatch.setattr(stage, "_write_csv", lambda *_a, **_k: None)
    monkeypatch.setattr(
        stage,
        "_decode_render_and_extract",
        lambda *_a, **_k: (_ for _ in ()).throw(ContractError("synthetic render failure")),
    )

    with pytest.raises(ContractError, match="synthetic render failure"):
        stage.run_s06(
            tmp_path / "manifest.csv",
            tmp_path / "ingest",
            tmp_path / "micro",
            tmp_path / "stable",
            tmp_path / "forced",
            config_path,
            output,
            logger=lambda _message: None,
        )

    assert not output.exists()
    assert not list(tmp_path.glob(".output.staging-*"))


def test_run_s06_success_commits_staging_atomically(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = Path(__file__).parents[1] / "configs/s06_export.yaml"
    output = tmp_path / "output"
    bundle = _mock_bundle()
    metrics = tuple(SimpleNamespace(selected_by_solver=False) for _ in range(133))
    contacts = tuple(object() for _ in range(62))
    monkeypatch.setattr(stage, "_preflight_nvenc", lambda _config: None)
    monkeypatch.setattr(stage, "load_s06_inputs", lambda *_a, **_k: bundle)
    monkeypatch.setattr(stage, "_resolve_config_from_bundle", lambda config, _bundle: config)
    monkeypatch.setattr(stage, "build_detection_export_table", lambda *_a, **_k: object())
    monkeypatch.setattr(
        stage, "recompute_structural_metrics", lambda *_a: StructuralMetrics(0, 0, 0, 0)
    )
    monkeypatch.setattr(stage, "_candidate_metrics", lambda _table: metrics)
    monkeypatch.setattr(stage, "select_low_confidence_links", lambda *_a, **_k: metrics)
    monkeypatch.setattr(stage, "build_global_track_summary", lambda *_a, **_k: object())
    monkeypatch.setattr(
        stage,
        "_build_crop_and_contact_plans",
        lambda *_a, **_k: (contacts, (), (), SimpleNamespace()),
    )
    monkeypatch.setattr(
        stage,
        "_build_qa_metrics",
        lambda *_a, **_k: {
            "counts": {
                "total_detection_rows": 746279,
                "valid_detections": 745915,
                "invalid_detections": 364,
                "microtracklets": 4659,
                "stable_tracklets": 3769,
                "candidate_edges": 133,
                "selected_links": 0,
                "global_ids": 62,
            }
        },
    )
    monkeypatch.setattr(stage, "_write_csv", lambda *_a, **_k: None)

    def fake_decode(*args: object, **_kwargs: object) -> dict[int, Path]:
        staging = Path(args[5])
        (staging / ".contact_crop_cache").mkdir()
        return {}

    monkeypatch.setattr(stage, "_decode_render_and_extract", fake_decode)
    monkeypatch.setattr(stage, "_render_contact_sheets", lambda *_a, **_k: None)
    monkeypatch.setattr(stage, "_render_low_confidence_html", lambda *_a, **_k: None)
    monkeypatch.setattr(stage, "_verify_input_fingerprints", lambda *_a, **_k: None)
    monkeypatch.setattr(
        stage,
        "_output_fingerprints",
        lambda *_a, **_k: [
            {"path": "effective_config.json", "size_bytes": 1, "sha256": "0" * 64}
        ],
    )
    validated_staging: list[Path] = []

    def validate(path: Path, *_args: object, **_kwargs: object) -> None:
        path = Path(path)
        assert (path / "_SUCCESS.json").is_file()
        validated_staging.append(path)

    monkeypatch.setattr(stage, "_validate_completed_output", validate)

    marker = stage.run_s06(
        tmp_path / "manifest.csv",
        tmp_path / "ingest",
        tmp_path / "micro",
        tmp_path / "stable",
        tmp_path / "forced",
        config_path,
        output,
        logger=lambda _message: None,
    )

    assert marker["stage"] == "S06_EXPORT"
    assert len(validated_staging) == 1
    assert not validated_staging[0].exists()
    assert output.is_dir()
    assert (output / "_SUCCESS.json").is_file()
    assert not list(tmp_path.glob(".output.staging-*"))
