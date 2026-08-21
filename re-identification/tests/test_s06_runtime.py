from __future__ import annotations

import hashlib
import json
import os
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pytest

from cowtrack.config import ContractError
from cowtrack.linking import s06_runtime as runtime
from cowtrack.linking.forced_appearance_config import load_forced_appearance_config
from cowtrack.qa.s06_config import load_s06_export_config
from cowtrack.schemas.detections import BBoxQAFlag, DETECTIONS_SCHEMA
from cowtrack.schemas.frames import FRAMES_SCHEMA
from cowtrack.schemas.s05_forced import (
    GRADED_STABLE_APPEARANCE_SCHEMA,
    RESCUE_SAMPLES_SCHEMA,
)


def _grade_row(stable_id: int, *, stored_iou: float = 0.0) -> dict[str, object]:
    return {
        "stable_id": stable_id,
        "evidence_grade": "C_REENCODED_DEGRADED",
        "descriptor_usable": True,
        "selection_policy": "supplemental_embeddings",
        "num_valid_prototypes": 1,
        "num_input_samples": 1,
        "num_selected_samples": 1,
        "input_sample_ids": [1_000_000 + stable_id],
        "selected_sample_ids": [1_000_000 + stable_id],
        "selected_quality_min": 0.75,
        "selected_quality_mean": 0.75,
        "selected_max_other_bbox_iou": stored_iou,
        "used_high_overlap": stored_iou >= 0.25,
        "used_s02_outlier": False,
        "missing_reason": None,
    }


def _rescue_row(
    stable_id: int,
    *,
    iou: float,
    selected: bool = True,
    embedding_row: int | None = 0,
) -> dict[str, object]:
    return {
        "stable_id": stable_id,
        "micro_id": stable_id + 10,
        "det_id": stable_id + 100,
        "clip_id": "GX040006",
        "local_frame": 1,
        "global_frame": 1,
        "global_time_sec": 1.0,
        "x1": 1.0,
        "y1": 2.0,
        "x2": 11.0,
        "y2": 12.0,
        "crop_quality": 0.75,
        "other_bbox_max_iou": iou,
        "clipped_fraction": 0.0,
        "bbox_area_percentile": 0.5,
        "blur_score": 100.0,
        "distance_to_image_boundary": 1.0,
        "review_excluded": True,
        "quality_gate_passed": True,
        "selected_for_descriptor": selected,
        "selection_reason": "quality_ranked" if selected else "candidate_not_selected",
        "embedding_row": embedding_row,
    }


def _table(rows: list[dict[str, object]], schema: pa.Schema) -> pa.Table:
    return pa.Table.from_pylist(rows, schema=schema)


def test_fixed_s06_config_matches_runtime_contract() -> None:
    config, _, _ = load_s06_export_config(Path("configs/s06_export.yaml"))
    expected = runtime._expectations(config)
    assert expected.total_detections == 3_322_909
    assert expected.invalid_detections == 4_796
    assert len(expected.invalid_by_clip) == 11
    assert expected.microtracks is None
    assert expected.stable_tracks is None
    s00_outputs = runtime._s00_output_names(expected)
    assert len(s00_outputs) == 39
    assert "qa/overlays/GX010006_back_f088319.jpg" in s00_outputs
    assert "qa/overlays/GX110006_middle_f023485.jpg" in s00_outputs


def test_runtime_rejects_config_identity_source_change() -> None:
    config, _, _ = load_s06_export_config(Path("configs/s06_export.yaml"))
    values = dict(vars(config))
    values["identity_source"] = "05_global_link"
    with pytest.raises(ContractError, match="identity_source"):
        runtime._expectations(SimpleNamespace(**values))


def test_runtime_does_not_treat_s05_observations_as_historical_references() -> None:
    config, _, _ = load_s06_export_config(Path("configs/s06_export.yaml"))
    values = dict(vars(config))
    values.update(
        expected_selected_link_count=123,
        expected_candidate_edge_count=456,
        expected_rescue_candidate_count=7,
        expected_rescue_embedding_count=8,
    )

    expected = runtime._expectations(SimpleNamespace(**values))

    assert expected.global_tracks == 62


def test_s00_input_roles_accept_manifest_relocation_by_content_hash(
    tmp_path: Path,
) -> None:
    manifest_hash = "1" * 64
    video = (tmp_path / "video.mp4").resolve()
    boxes = (tmp_path / "boxes.csv").resolve()
    recorded_manifest = (tmp_path / "old_repo" / "data" / "manifest.csv").resolve()
    recorded_config = (
        tmp_path / "old_repo" / "configs" / "production.yaml"
    ).resolve()
    rows = [SimpleNamespace(video_path=video, bbox_csv_path=boxes)]
    records = (
        runtime.FileFingerprint(str(recorded_manifest), 10, manifest_hash),
        runtime.FileFingerprint(str(recorded_config), 20, "2" * 64),
        runtime.FileFingerprint(str(video), 30, "3" * 64),
        runtime.FileFingerprint(str(boxes), 40, "4" * 64),
    )

    observed_manifest, observed_config = (
        runtime._validate_s00_recorded_input_roles(
            records,
            rows,
            manifest_hash,
        )
    )

    assert observed_manifest == recorded_manifest
    assert observed_config == recorded_config


def test_s00_input_roles_keep_video_and_bbox_paths_exact(tmp_path: Path) -> None:
    manifest_hash = "1" * 64
    video = (tmp_path / "video.mp4").resolve()
    boxes = (tmp_path / "boxes.csv").resolve()
    rows = [SimpleNamespace(video_path=video, bbox_csv_path=boxes)]
    records = (
        runtime.FileFingerprint(
            str((tmp_path / "repo" / "data" / "manifest.csv").resolve()),
            10,
            manifest_hash,
        ),
        runtime.FileFingerprint(
            str((tmp_path / "repo" / "configs" / "production.yaml").resolve()),
            20,
            "2" * 64,
        ),
        runtime.FileFingerprint(str(video), 30, "3" * 64),
    )

    with pytest.raises(ContractError, match="video/bbox fingerprint paths"):
        runtime._validate_s00_recorded_input_roles(
            records,
            rows,
            manifest_hash,
        )


def test_s00_input_roles_reject_ambiguous_manifest_hash(tmp_path: Path) -> None:
    manifest_hash = "1" * 64
    video = (tmp_path / "video.mp4").resolve()
    boxes = (tmp_path / "boxes.csv").resolve()
    rows = [SimpleNamespace(video_path=video, bbox_csv_path=boxes)]
    records = (
        runtime.FileFingerprint(
            str((tmp_path / "repo" / "data" / "manifest.csv").resolve()),
            10,
            manifest_hash,
        ),
        runtime.FileFingerprint(
            str((tmp_path / "repo" / "configs" / "production.yaml").resolve()),
            20,
            manifest_hash,
        ),
        runtime.FileFingerprint(str(video), 30, "3" * 64),
        runtime.FileFingerprint(str(boxes), 40, "4" * 64),
    )

    with pytest.raises(ContractError, match="not uniquely identified"):
        runtime._validate_s00_recorded_input_roles(
            records,
            rows,
            manifest_hash,
        )


def test_s01_input_roles_accept_exact_snapshot_relocation(tmp_path: Path) -> None:
    old_snapshot = (tmp_path / "old" / "00_ingest").resolve()
    current_snapshot = (tmp_path / "current" / "00_ingest").resolve()
    current: dict[str, runtime.FileFingerprint] = {}
    recorded: list[runtime.FileFingerprint] = []
    for index, name in enumerate(runtime._S01_S00_INPUT_NAMES, start=1):
        digest = f"{index:064x}"
        size = index * 10
        current[name] = runtime.FileFingerprint(
            str(current_snapshot / name), size, digest
        )
        recorded.append(
            runtime.FileFingerprint(str(old_snapshot / name), size, digest)
        )
    old_config = (tmp_path / "old" / "configs" / "s01_microtrack.yaml").resolve()
    recorded.append(runtime.FileFingerprint(str(old_config), 70, "f" * 64))

    assert runtime._validate_s01_recorded_input_roles(recorded, current) == old_config


def test_s01_input_roles_reject_changed_live_upstream(tmp_path: Path) -> None:
    old_snapshot = (tmp_path / "old" / "00_ingest").resolve()
    current_snapshot = (tmp_path / "current" / "00_ingest").resolve()
    current: dict[str, runtime.FileFingerprint] = {}
    recorded: list[runtime.FileFingerprint] = []
    for index, name in enumerate(runtime._S01_S00_INPUT_NAMES, start=1):
        digest = f"{index:064x}"
        current[name] = runtime.FileFingerprint(
            str(current_snapshot / name), index, digest
        )
        recorded.append(
            runtime.FileFingerprint(str(old_snapshot / name), index, digest)
        )
    current["frames.parquet"] = runtime.FileFingerprint(
        str(current_snapshot / "frames.parquet"),
        current["frames.parquet"].size_bytes,
        "e" * 64,
    )
    recorded.append(
        runtime.FileFingerprint(
            str((tmp_path / "old" / "configs" / "s01.yaml").resolve()),
            1,
            "f" * 64,
        )
    )

    with pytest.raises(ContractError, match="frames.parquet byte provenance"):
        runtime._validate_s01_recorded_input_roles(recorded, current)


def test_s04_role_map_rebases_only_fixed_current_inputs(tmp_path: Path) -> None:
    repository = (tmp_path / "current_repo").resolve()
    sequence = repository / "work" / "sequence"
    stable = sequence / "04_short_stable"
    stable.mkdir(parents=True)
    records: list[dict[str, object]] = []
    for directory, names in runtime._S04_RECORDED_INPUT_ROLES.items():
        for name in names:
            if directory == "configs":
                old_path = Path("/historical/repo/configs") / name
            else:
                old_path = Path("/historical/repo/work/sequence") / directory / name
            records.append(
                {
                    "path": str(old_path),
                    "size_bytes": 1,
                    "sha256": "a" * 64,
                }
            )
    (stable / "_SUCCESS.json").write_text(
        json.dumps({"input_fingerprints": records}), encoding="utf-8"
    )

    replacements = runtime._s04_recorded_input_relocations(stable, repository)

    assert len(replacements) == 27
    assert replacements["/historical/repo/configs/s04_finalize.yaml"] == (
        repository / "configs" / "s04_finalize.yaml"
    )
    assert replacements[
        "/historical/repo/work/sequence/02_appearance/sample_embeddings.f16.npy"
    ] == sequence / "02_appearance" / "sample_embeddings.f16.npy"


def test_c_grade_iou_uses_rescue_rows_and_emits_warning() -> None:
    graded = _table([_grade_row(7)], GRADED_STABLE_APPEARANCE_SCHEMA)
    rescue = _table([_rescue_row(7, iou=0.4)], RESCUE_SAMPLES_SCHEMA)

    provenance, warnings = runtime._derive_c_grade_provenance(
        rescue,
        graded,
        overlap_threshold=0.25,
    )

    assert provenance[7].selected_max_other_bbox_iou == pytest.approx(0.4)
    assert provenance[7].used_high_overlap is True
    assert provenance[7].selected_det_ids == (107,)
    assert warnings and "rescue_samples.parquet is authoritative" in warnings[0]
    with pytest.raises(TypeError):
        provenance[8] = provenance[7]  # type: ignore[index]


def test_c_grade_matching_rescue_truth_has_no_warning() -> None:
    graded = _table([_grade_row(7, stored_iou=0.4)], GRADED_STABLE_APPEARANCE_SCHEMA)
    rescue = _table([_rescue_row(7, iou=0.4)], RESCUE_SAMPLES_SCHEMA)
    _, warnings = runtime._derive_c_grade_provenance(
        rescue, graded, overlap_threshold=0.25
    )
    assert warnings == ()


def test_rescue_row_for_non_c_grade_is_rejected() -> None:
    row = _grade_row(7)
    row["evidence_grade"] = "A_CLEAN"
    graded = _table([row], GRADED_STABLE_APPEARANCE_SCHEMA)
    rescue = _table([_rescue_row(7, iou=0.0)], RESCUE_SAMPLES_SCHEMA)
    with pytest.raises(ContractError, match="non-C-grade"):
        runtime._derive_c_grade_provenance(
            rescue, graded, overlap_threshold=0.25
        )


def test_probability_columns_must_remain_null() -> None:
    all_null = pa.table({"probability": pa.array([None, None], type=pa.float64())})
    runtime._all_null(all_null, ("probability",), "test probability")
    non_null = pa.table({"probability": pa.array([None, 0.9], type=pa.float64())})
    with pytest.raises(ContractError, match="entirely null"):
        runtime._all_null(non_null, ("probability",), "test probability")


def test_path_graph_recomputes_temporal_overlap() -> None:
    stable = SimpleNamespace(
        stable_tracklets={
            0: SimpleNamespace(
                end_global_frame=10,
                end_time_sec=1.0,
                start_global_frame=0,
                start_time_sec=0.0,
            ),
            1: SimpleNamespace(
                end_global_frame=20,
                end_time_sec=2.0,
                start_global_frame=10,
                start_time_sec=1.0,
            ),
        }
    )
    cycles, overlaps = runtime._validate_path_graph(
        np.array([0, 1]),
        np.array([0, 0]),
        np.array([0, 1]),
        [None, 0],
        stable,
        expected_global_count=1,
    )
    assert cycles == 0
    assert overlaps == 1


def test_path_graph_recomputes_cycle_independently_of_marker() -> None:
    stable = SimpleNamespace(
        stable_tracklets={
            0: SimpleNamespace(
                start_global_frame=0,
                start_time_sec=0.0,
                end_global_frame=0,
                end_time_sec=0.0,
            ),
            1: SimpleNamespace(
                start_global_frame=1,
                start_time_sec=1.0,
                end_global_frame=1,
                end_time_sec=1.0,
            ),
        }
    )
    cycles, _ = runtime._validate_path_graph(
        np.array([0, 1]),
        np.array([0, 0]),
        np.array([0, 1]),
        [1, 0],
        stable,
        expected_global_count=1,
    )
    assert cycles == 1


def _observed_report_fixture() -> tuple[
    dict[str, object],
    dict[str, object],
    runtime.S06ForcedTables,
    object,
    runtime._Expectations,
    object,
]:
    tables = runtime.S06ForcedTables(
        rescue_samples=pa.table(
            {
                "selected_for_descriptor": [True, True],
                "selection_reason": ["best_degraded_fallback", "quality_ranked"],
                "review_excluded": [True, False],
            }
        ),
        graded_stable_appearance=pa.table(
            {
                "evidence_grade": [
                    "A_CLEAN",
                    "B_EXISTING_DEGRADED",
                    "C_REENCODED_DEGRADED",
                    "C_REENCODED_DEGRADED",
                ]
            }
        ),
        candidate_edges=pa.table(
            {
                "selected_by_solver": [True, True, False],
                "appearance_cost_int": pa.array([1, 2, 3], type=pa.int64()),
                "prior_global_link": [True, False, False],
                "appearance_cosine": [0.4, 0.8, 0.2],
            }
        ),
        stable_to_global=pa.table({"stable_id": [0, 1, 2, 3]}),
        global_tracks=pa.table({"num_stable_tracklets": [2, 2]}),
        det_to_global=pa.table({}),
    )
    stable = SimpleNamespace(
        stable_tracklets={
            0: SimpleNamespace(start_global_frame=0, end_global_frame=0),
            1: SimpleNamespace(start_global_frame=1, end_global_frame=1),
            2: SimpleNamespace(start_global_frame=2, end_global_frame=2),
            3: SimpleNamespace(start_global_frame=3, end_global_frame=3),
        }
    )
    expected = runtime._Expectations(
        microtracks=4,
        stable_tracks=4,
        valid_detections=4,
        invalid_detections=0,
        global_tracks=2,
    )
    marker: dict[str, object] = {
        "stats": {
            "num_stable_tracks": 4,
            "num_selected_links": 2,
            "num_global_tracks": 2,
            "num_rescue_stable_tracks": 2,
            "num_rescue_embeddings": 2,
            "num_candidate_edges": 3,
            "max_concurrent_stable_tracks": 1,
        }
    }
    report: dict[str, object] = {
        "coverage": {
            "stable_tracks_mapped_once": 4,
            "microtracklets_mapped_once": 4,
            "valid_detections_mapped_once": 4,
            "invalid_detections_excluded": 0,
            "same_frame_same_global_id_violations": 0,
            "temporal_overlap_link_violations": 0,
            "cycles": 0,
            "exact_global_track_count": 2,
        },
        "graded_appearance": {
            "counts_by_grade": {
                "A_CLEAN": 1,
                "B_EXISTING_DEGRADED": 1,
                "C_REENCODED_DEGRADED": 2,
            },
            "all_stable_tracks_have_descriptor": True,
            "num_stable_tracks": 4,
            "prototype_slots": 3,
        },
        "rescue": {
            "zero_s02_sample_stable_tracks": 2,
            "candidate_crops_decoded": 2,
            "selected_crops_encoded": 2,
            "selected_best_degraded_crops": 1,
            "selected_review_excluded_crops": 1,
        },
        "solver": {
            "solve_mode": "full_sequence_single_solve",
            "algorithm": (
                "deterministic_full_sequence_fixed_cardinality_min_cost_flow"
            ),
            "objective": (
                "exact_62_global_paths_then_minimum_full_sequence_appearance_cost"
            ),
            "candidate_scope": "complete_sequence",
            "assignment": "exact_fixed_flow_min_cost_network_without_dummies",
            "cardinality_certificate": "minimum_width_interval_backbone",
            "solve_pass_count": 1,
            "candidate_count": 3,
            "source_top_k": 5,
            "target_top_k": 6,
            "interval_width": 1,
            "backbone_chain_count": 1,
            "maximum_feasible_links": 3,
            "required_links": 2,
            "selected_links": 2,
            "global_paths": 2,
            "total_appearance_cost_int": 3,
            "selected_prior_links": 1,
            "selected_appearance_cosine": {
                "min": 0.4,
                "p10": 0.44,
                "mean": 0.6,
                "max": 0.8,
            },
            "solve_passes": [
                {
                    "pass_index": 0,
                    "source_clip_ids": list(runtime.EXPECTED_CLIP_ORDER),
                    "solver_nodes": 4,
                    "candidate_count": 3,
                    "interval_width": 1,
                    "backbone_chain_count": 1,
                    "maximum_feasible_links": 3,
                    "required_links": 2,
                    "selected_links": 2,
                    "total_appearance_cost_int": 3,
                }
            ],
        },
        "global_path_size_distribution": {"2": 2},
    }
    forced_config = SimpleNamespace(
        source_top_k=5,
        target_top_k=6,
    )
    return marker, report, tables, stable, expected, forced_config


def test_forced_report_counts_are_observed_from_current_tables() -> None:
    marker, report, tables, stable, expected, forced_config = (
        _observed_report_fixture()
    )

    runtime._validate_forced_report_counts(
        marker,
        report,
        tables,
        stable,
        0,
        0,
        0,
        expected,
        forced_config,
        3,
    )


@pytest.mark.parametrize(
    "field",
    (
        "solve_mode",
        "algorithm",
        "objective",
        "candidate_scope",
        "assignment",
        "cardinality_certificate",
        "backbone_chain_count",
    ),
)
def test_forced_report_rejects_non_full_sequence_solver_contract(
    field: str,
) -> None:
    marker, report, tables, stable, expected, forced_config = (
        _observed_report_fixture()
    )
    solver = dict(report["solver"])  # type: ignore[arg-type]
    solver[field] = -1 if field == "backbone_chain_count" else "tampered"
    report["solver"] = solver

    with pytest.raises(ContractError, match="solver differs from current tables"):
        runtime._validate_forced_report_counts(
            marker,
            report,
            tables,
            stable,
            0,
            0,
            0,
            expected,
            forced_config,
            3,
        )


def test_forced_report_rejects_marker_count_not_backed_by_current_tables() -> None:
    marker, report, tables, stable, expected, forced_config = (
        _observed_report_fixture()
    )
    marker_stats = dict(marker["stats"])  # type: ignore[arg-type]
    marker_stats["num_candidate_edges"] = 177_853
    marker["stats"] = marker_stats

    with pytest.raises(ContractError, match="current tables"):
        runtime._validate_forced_report_counts(
            marker,
            report,
            tables,
            stable,
            0,
            0,
            0,
            expected,
            forced_config,
            3,
        )


def test_forced_report_rejects_non_single_solve_pass_count() -> None:
    marker, report, tables, stable, expected, forced_config = (
        _observed_report_fixture()
    )
    solver = dict(report["solver"])  # type: ignore[arg-type]
    solver["solve_pass_count"] = 2
    report["solver"] = solver

    with pytest.raises(ContractError, match="exactly one validated full-sequence"):
        runtime._validate_forced_report_counts(
            marker,
            report,
            tables,
            stable,
            0,
            0,
            0,
            expected,
            forced_config,
            3,
        )


def test_forced_report_rejects_duplicate_solve_pass() -> None:
    marker, report, tables, stable, expected, forced_config = (
        _observed_report_fixture()
    )
    solver = dict(report["solver"])  # type: ignore[arg-type]
    solve_passes = list(solver["solve_passes"])  # type: ignore[arg-type]
    solver["solve_passes"] = [*solve_passes, deepcopy(solve_passes[0])]
    report["solver"] = solver

    with pytest.raises(ContractError, match="exactly one validated full-sequence"):
        runtime._validate_forced_report_counts(
            marker,
            report,
            tables,
            stable,
            0,
            0,
            0,
            expected,
            forced_config,
            3,
        )


def test_forced_report_rejects_solve_pass_not_backed_by_tables() -> None:
    marker, report, tables, stable, expected, forced_config = (
        _observed_report_fixture()
    )
    solver = dict(report["solver"])  # type: ignore[arg-type]
    solve_pass = dict(solver["solve_passes"][0])  # type: ignore[index]
    solve_pass["candidate_count"] = 177_853
    solver["solve_passes"] = [solve_pass]
    report["solver"] = solver

    with pytest.raises(ContractError, match="exactly one validated full-sequence"):
        runtime._validate_forced_report_counts(
            marker,
            report,
            tables,
            stable,
            0,
            0,
            0,
            expected,
            forced_config,
            3,
        )


def _small_frames() -> pa.Table:
    return _table(
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
                "pts_sec": 0.1,
                "global_time_sec": 0.1,
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
                "global_time_sec": 0.2,
                "width": 3840,
                "height": 2160,
            },
        ],
        FRAMES_SCHEMA,
    )


def _detection(det_id: int, clip: str, frame: int, row: int, *, valid: bool) -> dict[str, object]:
    local = frame if clip == "A" else 0
    return {
        "det_id": det_id,
        "sequence_id": "seq",
        "clip_id": clip,
        "local_frame": local,
        "global_frame": frame,
        "global_time_sec": frame / 10.0,
        "x1": 1.0,
        "y1": 1.0,
        "x2": 11.0,
        "y2": 11.0,
        "cx_norm": 0.1,
        "cy_norm": 0.1,
        "w_norm": 0.1,
        "h_norm": 0.1,
        "area_norm": 0.01,
        "bbox_confidence": 0.9,
        "legacy_track_id": "1",
        "csv_row_index": row,
        "valid": valid,
        "qa_flags": 0 if valid else int(BBoxQAFlag.HIGH_IOU_DUPLICATE),
    }


def test_detection_validation_preserves_invalid_source_row() -> None:
    expected = runtime._Expectations(
        sequence_id="seq",
        clip_order=("A", "B"),
        frame_counts=(2, 1),
        valid_by_clip=(1, 1),
        invalid_by_clip=(1, 0),
        frames=3,
        total_detections=3,
        valid_detections=2,
        invalid_detections=1,
        microtracks=2,
        stable_tracks=2,
        global_tracks=1,
    )
    frames = _small_frames()
    detections = _table(
        [
            _detection(1, "A", 0, 0, valid=True),
            _detection(2, "A", 1, 1, valid=False),
            _detection(3, "B", 2, 0, valid=True),
        ],
        DETECTIONS_SCHEMA,
    )
    runtime._validate_frames(frames, expected)
    runtime._validate_detections(detections, frames, expected)


def test_detection_row_shuffle_is_rejected() -> None:
    expected = runtime._Expectations(
        sequence_id="seq",
        clip_order=("A", "B"),
        frame_counts=(2, 1),
        valid_by_clip=(1, 1),
        invalid_by_clip=(1, 0),
        frames=3,
        total_detections=3,
        valid_detections=2,
        invalid_detections=1,
        microtracks=2,
        stable_tracks=2,
        global_tracks=1,
    )
    detections = _table(
        [
            _detection(2, "A", 1, 1, valid=False),
            _detection(1, "A", 0, 0, valid=True),
            _detection(3, "B", 2, 0, valid=True),
        ],
        DETECTIONS_SCHEMA,
    )
    with pytest.raises(ContractError, match="canonical source-row order"):
        runtime._validate_detections(detections, _small_frames(), expected)


def _output_record(name: str, path: Path) -> dict[str, object]:
    raw = path.read_bytes()
    return {
        "path": name,
        "size_bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def test_output_tree_rejects_hash_change(tmp_path: Path) -> None:
    (tmp_path / "_SUCCESS.json").write_text("{}", encoding="utf-8")
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"original")
    marker = {"output_fingerprints": [_output_record("artifact.bin", artifact)]}
    runtime._snapshot_output_tree(
        tmp_path, marker, ("artifact.bin",), {}, label="synthetic"
    )

    artifact.write_bytes(b"tampered")
    with pytest.raises(ContractError, match="artifact changed"):
        runtime._snapshot_output_tree(
            tmp_path, marker, ("artifact.bin",), {}, label="synthetic"
        )


def test_output_tree_rejects_unfingerprinted_file(tmp_path: Path) -> None:
    (tmp_path / "_SUCCESS.json").write_text("{}", encoding="utf-8")
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"original")
    (tmp_path / "extra.bin").write_bytes(b"extra")
    marker = {"output_fingerprints": [_output_record("artifact.bin", artifact)]}
    with pytest.raises(ContractError, match="artifact tree differs"):
        runtime._snapshot_output_tree(
            tmp_path, marker, ("artifact.bin",), {}, label="synthetic"
        )


def test_recorded_input_rejects_bad_sha256(tmp_path: Path) -> None:
    path = (tmp_path / "input.bin").resolve()
    path.write_bytes(b"immutable")
    record = {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": "0" * 64,
    }
    with pytest.raises(ContractError, match="upstream input changed"):
        runtime._verify_recorded_inputs(
            [record],
            label="synthetic inputs",
            with_mtime=False,
            cache={},
            logger=lambda _message: None,
        )


def test_recorded_input_accepts_historical_mtime_drift(tmp_path: Path) -> None:
    path = (tmp_path / "input.bin").resolve()
    path.write_bytes(b"immutable")
    record = {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "mtime_ns": path.stat().st_mtime_ns - 1,
    }

    messages: list[str] = []
    fingerprints = runtime._verify_recorded_inputs(
        [record],
        label="synthetic inputs",
        with_mtime=True,
        cache={},
        logger=messages.append,
    )

    assert fingerprints == (
        runtime.FileFingerprint(
            path=str(path),
            size_bytes=path.stat().st_size,
            sha256=record["sha256"],
        ),
    )
    assert any("recorded mtime differs but content hash matches" in item for item in messages)


def test_recorded_input_relocation_never_opens_historical_path(
    tmp_path: Path,
) -> None:
    historical = (tmp_path / "missing_old_repo" / "input.bin").resolve()
    current = (tmp_path / "current" / "input.bin").resolve()
    current.parent.mkdir()
    current.write_bytes(b"immutable")
    record = {
        "path": str(historical),
        "size_bytes": current.stat().st_size,
        "sha256": hashlib.sha256(current.read_bytes()).hexdigest(),
        "mtime_ns": 1,
    }
    messages: list[str] = []

    fingerprints = runtime._verify_recorded_inputs(
        [record],
        label="synthetic inputs",
        with_mtime=True,
        cache={},
        logger=messages.append,
        path_replacements={str(historical): current},
    )

    assert fingerprints[0].path == str(current)
    assert not historical.exists()
    assert any("accepted byte-identical provenance relocation" in item for item in messages)
    assert not any("recorded mtime differs" in item for item in messages)


def test_recorded_input_rejects_same_size_change_with_restored_mtime(
    tmp_path: Path,
) -> None:
    path = (tmp_path / "input.bin").resolve()
    path.write_bytes(b"original")
    original_stat = path.stat()
    record = {
        "path": str(path),
        "size_bytes": original_stat.st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "mtime_ns": original_stat.st_mtime_ns,
    }
    path.write_bytes(b"tampered")
    os.utime(path, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))

    with pytest.raises(ContractError, match="upstream input changed"):
        runtime._verify_recorded_inputs(
            [record],
            label="synthetic inputs",
            with_mtime=True,
            cache={},
            logger=lambda _message: None,
        )


def _forced_source_config_fixture() -> tuple[object, dict[str, object], str, dict[str, object]]:
    path = (Path(__file__).parents[1] / "configs/s05_force_appearance.yaml").resolve()
    config, payload, digest = load_forced_appearance_config(path)
    record: dict[str, object] = {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": digest,
    }
    return config, payload, digest, record


def test_forced_config_provenance_accepts_source_hash_with_equal_effective_payload() -> None:
    config, payload, digest, record = _forced_source_config_fixture()
    marker = {"config_hash": digest, "input_fingerprints": [record]}

    assert runtime._validate_forced_config_provenance(marker, config, payload) == digest


@pytest.mark.parametrize("duplicate", [False, True])
def test_forced_config_provenance_requires_one_matching_source_hash(
    duplicate: bool,
) -> None:
    config, payload, digest, record = _forced_source_config_fixture()
    records = [record, dict(record)] if duplicate else [record]
    marker_hash = digest if duplicate else "0" * 64
    marker = {"config_hash": marker_hash, "input_fingerprints": records}

    with pytest.raises(ContractError, match="exactly one source config"):
        runtime._validate_forced_config_provenance(marker, config, payload)


def test_forced_config_provenance_rejects_semantic_mismatch() -> None:
    config, payload, digest, record = _forced_source_config_fixture()
    different_payload = deepcopy(payload)
    different_payload["appearance"]["batch_size"] = 9
    marker = {"config_hash": digest, "input_fingerprints": [record]}

    with pytest.raises(ContractError, match="source/effective config provenance differs"):
        runtime._validate_forced_config_provenance(
            marker, config, different_payload
        )


def test_forced_provenance_requires_the_complete_role_set(tmp_path: Path) -> None:
    sequence = (tmp_path / "repo" / "work" / "sequence").resolve()
    appearance = sequence / "02_appearance"
    appearance.mkdir(parents=True)
    model = (tmp_path / "models" / "pytorch_model.bin").resolve()
    (appearance / "encoder_choice.json").write_text(
        json.dumps({"checkpoint_path": str(model)}), encoding="utf-8"
    )
    manifest = (tmp_path / "repo" / "data" / "manifest.csv").resolve()
    video = (tmp_path / "video.mp4").resolve()
    config = (tmp_path / "repo" / "configs" / "forced.yaml").resolve()
    s00_paths = tuple(
        sequence / "00_ingest" / name
        for name in (
            "_SUCCESS.json",
            "frames.parquet",
            "detections.parquet",
            "resolved_manifest.json",
        )
    )
    s01_paths = tuple(
        sequence / "01_microtrack" / name
        for name in (
            "_SUCCESS.json",
            "det_to_micro.parquet",
            "microtracklets.parquet",
        )
    )
    stable_dir = sequence / "04_short_stable"
    stable_paths = tuple(
        stable_dir / name
        for name in (
            "_SUCCESS.json",
            "det_to_stable.parquet",
            "effective_config.json",
            "finalize_report.json",
            "micro_to_stable.parquet",
            "stable_appearance.parquet",
            "stable_prototype_mask.npy",
            "stable_prototypes.f16.npy",
            "stable_tracklets.parquet",
        )
    )
    required_paths = {
        config,
        manifest,
        video,
        *s00_paths,
        *s01_paths,
        *stable_paths,
        *(
            appearance / name
            for name in runtime._S04_RECORDED_INPUT_ROLES["02_appearance"]
        ),
        sequence / "05_global_link" / "_SUCCESS.json",
        sequence / "05_global_link" / "stable_to_global.parquet",
        model,
    }
    config_hash = "a" * 64
    records = tuple(
        runtime.FileFingerprint(
            str(path),
            1,
            config_hash if path == config else "b" * 64,
        )
        for path in sorted(required_paths)
    )
    stable = SimpleNamespace(directory=stable_dir, consumed_paths=stable_paths)

    runtime._verify_required_forced_provenance(
        records,
        config_hash=config_hash,
        manifest_path=manifest,
        video_paths={"clip": video},
        s00_paths=s00_paths,
        s01_paths=s01_paths,
        stable=stable,
    )

    extra = runtime.FileFingerprint(str(tmp_path / "extra.bin"), 1, "c" * 64)
    with pytest.raises(ContractError, match="provenance role set differs"):
        runtime._verify_required_forced_provenance(
            (*records, extra),
            config_hash=config_hash,
            manifest_path=manifest,
            video_paths={"clip": video},
            s00_paths=s00_paths,
            s01_paths=s01_paths,
            stable=stable,
        )
