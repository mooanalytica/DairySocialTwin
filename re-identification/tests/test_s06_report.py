from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

from cowtrack.config import ContractError
from cowtrack.qa.s06_config import load_s06_export_config
from cowtrack.qa.s06_report import (
    StructuralMetrics,
    build_detection_export_table,
    build_global_track_summary,
    recompute_structural_metrics,
)
from cowtrack.schemas.detections import BBoxQAFlag, DETECTIONS_SCHEMA
from cowtrack.schemas.s05_finalize import (
    DET_TO_GLOBAL_SCHEMA,
    GLOBAL_TRACKS_SCHEMA,
    STABLE_TO_GLOBAL_SCHEMA,
)
from cowtrack.schemas.s05_forced import (
    FORCED_CANDIDATE_EDGES_SCHEMA,
    GRADED_STABLE_APPEARANCE_SCHEMA,
)
from cowtrack.schemas.s06 import (
    DETECTIONS_WITH_GLOBAL_ID_SCHEMA,
    GLOBAL_TRACK_SUMMARY_SCHEMA,
)
from cowtrack.schemas.tracklets import MICROTRACKLETS_SCHEMA


_DT = 1001.0 / 30000.0
_AUTH = "operator_forced_appearance_exact_62"
_STATUS = "forced_provisional"


def _table(schema: pa.Schema, rows: list[dict[str, object]]) -> pa.Table:
    return pa.Table.from_pylist(rows, schema=schema)


def _replace_column(
    table: pa.Table, schema: pa.Schema, name: str, values: pa.Array
) -> pa.Table:
    arrays = [values if field.name == name else table[field.name] for field in schema]
    return pa.Table.from_arrays(arrays, schema=schema)


def _detection(
    det_id: int,
    clip: str,
    csv_row: int,
    frame: int,
    legacy: str,
    *,
    valid: bool = True,
) -> dict[str, object]:
    flags = 0 if valid else int(BBoxQAFlag.AREA_BELOW_MINIMUM)
    return {
        "det_id": det_id,
        "sequence_id": "seq",
        "clip_id": clip,
        "local_frame": csv_row,
        "global_frame": frame,
        "global_time_sec": frame * _DT,
        "x1": 10.0,
        "y1": 20.0,
        "x2": 50.0,
        "y2": 80.0,
        "cx_norm": 0.1,
        "cy_norm": 0.2,
        "w_norm": 0.1,
        "h_norm": 0.1,
        "area_norm": 0.01,
        "bbox_confidence": 0.75,
        "legacy_track_id": legacy,
        "csv_row_index": csv_row,
        "valid": valid,
        "qa_flags": flags,
    }


def _mapping(
    det_id: int,
    clip: str,
    csv_row: int,
    frame: int,
    micro: int,
    stable: int,
    global_id: int,
    order_micro: int,
    order_stable_detection: int,
    order_global_detection: int,
    order_global_stable: int,
) -> dict[str, object]:
    return {
        "det_id": det_id,
        "sequence_id": "seq",
        "clip_id": clip,
        "clip_order": 0 if clip == "A" else 1,
        "local_frame": csv_row,
        "global_frame": frame,
        "global_time_sec": frame * _DT,
        "valid": True,
        "micro_id": micro,
        "stable_id": stable,
        "global_track_id": global_id,
        "global_track_uuid": f"u{global_id}",
        "display_global_id": f"G{global_id + 1:04d}",
        "order_in_micro": order_micro,
        "order_in_stable": 0,
        "order_in_stable_detection": order_stable_detection,
        "order_in_global_stable": order_global_stable,
        "order_in_global_detection": order_global_detection,
        "identity_basis": _AUTH,
        "id_status": _STATUS,
    }


def _micro(micro_id: int, purity: float) -> dict[str, object]:
    return {
        "micro_id": micro_id,
        "start_global_frame": micro_id,
        "end_global_frame": micro_id,
        "start_time_sec": micro_id * _DT,
        "end_time_sec": micro_id * _DT,
        "num_detections": 1,
        "duration_sec": np.float32(_DT),
        "start_x1": 10.0,
        "start_y1": 20.0,
        "start_x2": 50.0,
        "start_y2": 80.0,
        "end_x1": 10.0,
        "end_y1": 20.0,
        "end_x2": 50.0,
        "end_y2": 80.0,
        "end_vx_norm_per_sec": 0.0,
        "end_vy_norm_per_sec": 0.0,
        "start_vx_norm_per_sec": 0.0,
        "start_vy_norm_per_sec": 0.0,
        "max_internal_center_jump": 0.0,
        "median_internal_cost": 0.0,
        "bidirectional_agreement": 1.0,
        "local_purity_score": purity,
        "status": "valid",
    }


def _stable_mapping(
    stable: int,
    global_id: int,
    order: int,
    *,
    predecessor: int | None,
    detections: int,
    stable_count: int,
) -> dict[str, object]:
    return {
        "stable_id": stable,
        "global_track_id": global_id,
        "global_track_uuid": f"u{global_id}",
        "display_global_id": f"G{global_id + 1:04d}",
        "order_in_global_path": order,
        "predecessor_stable_id": predecessor,
        "predecessor_candidate_id": None if predecessor is None else "c0",
        "predecessor_proposal_id": None,
        "predecessor_global_link_id": None if predecessor is None else "l0",
        "predecessor_link_probability": None,
        "predecessor_link_margin": None,
        "predecessor_authorization_basis": None if predecessor is None else _AUTH,
        "link_type": "PATH_START" if predecessor is None else "FORCED_APPEARANCE",
        "cross_clip_boundary": False,
        "component_num_stable_tracklets": stable_count,
        "component_num_detections": detections,
        "component_num_long_links": stable_count - 1,
        "component_min_link_probability": None,
        "component_mean_link_probability": None,
        "component_max_link_probability": None,
        "identity_basis": _AUTH,
        "id_status": _STATUS,
    }


def _candidate(
    candidate_id: str,
    source: int,
    target: int,
    source_frame: int,
    target_frame: int,
    source_grade: str,
    target_grade: str,
    *,
    selected: bool = True,
    strict: bool = True,
) -> dict[str, object]:
    source_clip = "A" if source < 2 else "B"
    target_clip = "A" if target < 2 else "B"
    return {
        "candidate_id": candidate_id,
        "source_stable_id": source,
        "target_stable_id": target,
        "source_end_clip_id": source_clip,
        "target_start_clip_id": target_clip,
        "source_end_global_frame": source_frame,
        "target_start_global_frame": target_frame,
        "source_end_time_sec": source_frame * _DT,
        "target_start_time_sec": target_frame * _DT,
        "temporal_gap_sec": (target_frame - source_frame) * _DT,
        "strictly_nonoverlapping": strict,
        "appearance_cosine": 0.4,
        "source_evidence_grade": source_grade,
        "target_evidence_grade": target_grade,
        "selected_by_source_topk": True,
        "selected_by_target_topk": False,
        "temporal_backbone": True,
        "prior_global_link": False,
        "appearance_cost_int": 1,
        "solver_cost_int": 2,
        "selected_by_solver": selected,
        "global_link_id": f"l{candidate_id}" if selected else None,
        "authorization_basis": _AUTH,
        "id_status": _STATUS if selected else "candidate_only",
    }


def _grade(stable: int, grade: str) -> dict[str, object]:
    return {
        "stable_id": stable,
        "evidence_grade": grade,
        "descriptor_usable": True,
        "selection_policy": "synthetic",
        "num_valid_prototypes": 1,
        "num_input_samples": 1,
        "num_selected_samples": 1,
        "input_sample_ids": [stable],
        "selected_sample_ids": [stable],
        "selected_quality_min": 0.5,
        "selected_quality_mean": 0.5,
        "selected_max_other_bbox_iou": 0.0,
        "used_high_overlap": False,
        "used_s02_outlier": False,
        "missing_reason": None,
    }


def _global(
    global_id: int,
    stable_ids: tuple[int, ...],
    det_ids: tuple[int, int],
    clip: str,
    frames: tuple[int, int],
) -> dict[str, object]:
    return {
        "global_track_id": global_id,
        "global_track_uuid": f"u{global_id}",
        "display_global_id": f"G{global_id + 1:04d}",
        "sequence_id": "seq",
        "first_stable_id": stable_ids[0],
        "last_stable_id": stable_ids[-1],
        "start_det_id": det_ids[0],
        "end_det_id": det_ids[1],
        "start_clip_id": clip,
        "end_clip_id": clip,
        "clip_ids": [clip],
        "start_global_frame": frames[0],
        "end_global_frame": frames[1],
        "start_time_sec": frames[0] * _DT,
        "end_time_sec": frames[1] * _DT,
        "num_stable_tracklets": len(stable_ids),
        "num_microtracklets": len(stable_ids),
        "num_detections": 2,
        "num_long_links": len(stable_ids) - 1,
        # Deliberately carries the known upstream 1/30 cadence.  S06 must not
        # copy this value into its exact-NTSC summary.
        "duration_visible_sec": 2.0 / 30.0,
        "min_link_probability": None,
        "p10_link_probability": None,
        "mean_link_probability": None,
        "max_link_probability": None,
        "min_link_margin": None,
        "appearance_consistency": 0.4 if len(stable_ids) > 1 else None,
        "identity_basis": _AUTH,
        "id_status": _STATUS,
        "spans_multiple_clips": False,
        # Deliberately carries the known upstream false field.  S06 recomputes
        # one dataset-level warning for every output row.
        "population_warning": False,
    }


@pytest.fixture()
def synthetic() -> dict[str, object]:
    detection_rows = [
        _detection(100, "A", 0, 0, "old-a"),
        _detection(101, "A", 1, 1, "old-a"),
        _detection(102, "A", 2, 2, "old-invalid", valid=False),
        _detection(200, "B", 0, 3, "old-b"),
        _detection(201, "B", 1, 4, "old-b"),
    ]
    mapping_rows = [
        _mapping(100, "A", 0, 0, 0, 0, 0, 0, 0, 0, 0),
        _mapping(101, "A", 1, 1, 1, 1, 0, 0, 0, 1, 1),
        _mapping(200, "B", 0, 3, 2, 2, 1, 0, 0, 0, 0),
        _mapping(201, "B", 1, 4, 2, 2, 1, 1, 1, 1, 0),
    ]
    detections = _table(DETECTIONS_SCHEMA, detection_rows).take(
        pa.array([4, 2, 0, 3, 1])
    )
    mapping = _table(DET_TO_GLOBAL_SCHEMA, mapping_rows).take(pa.array([2, 0, 3, 1]))
    microtracklets = _table(
        MICROTRACKLETS_SCHEMA,
        [_micro(2, 0.7), _micro(0, 0.9), _micro(1, 0.8)],
    )
    stable = _table(
        STABLE_TO_GLOBAL_SCHEMA,
        [
            _stable_mapping(2, 1, 0, predecessor=None, detections=2, stable_count=1),
            _stable_mapping(1, 0, 1, predecessor=0, detections=2, stable_count=2),
            _stable_mapping(0, 0, 0, predecessor=None, detections=2, stable_count=2),
        ],
    )
    candidates = _table(
        FORCED_CANDIDATE_EDGES_SCHEMA,
        [_candidate("c0", 0, 1, 0, 1, "A_CLEAN", "B_EXISTING_DEGRADED")],
    )
    globals_table = _table(
        GLOBAL_TRACKS_SCHEMA,
        [
            _global(1, (2,), (200, 201), "B", (3, 4)),
            _global(0, (0, 1), (100, 101), "A", (0, 1)),
        ],
    )
    graded = _table(
        GRADED_STABLE_APPEARANCE_SCHEMA,
        [
            _grade(1, "B_EXISTING_DEGRADED"),
            _grade(2, "C_REENCODED_DEGRADED"),
            _grade(0, "A_CLEAN"),
        ],
    )
    fixed, _, _ = load_s06_export_config(
        Path(__file__).parents[1] / "configs/s06_export.yaml"
    )
    config = replace(
        fixed,
        expected_sequence_id="seq",
        clip_order=("A", "B"),
        expected_frame_count=5,
        frame_counts_by_clip=(3, 2),
        expected_total_detection_count=5,
        expected_valid_detection_count=4,
        expected_invalid_detection_count=1,
        expected_total_detections_by_clip=(3, 2),
        expected_valid_detections_by_clip=(2, 2),
        expected_invalid_detections_by_clip=(1, 0),
        expected_microtrack_count=3,
        expected_stable_track_count=3,
        expected_global_track_count=2,
        expected_selected_link_count=1,
        expected_candidate_edge_count=1,
        expected_grade_a_clean_count=1,
        expected_grade_b_existing_degraded_count=1,
        expected_grade_c_reencoded_degraded_count=1,
        num_provisional_ids=2,
        population_soft_max=1,
        population_overflow=1,
    )
    return {
        "detections": detections,
        "mapping": mapping,
        "microtracklets": microtracklets,
        "stable": stable,
        "candidates": candidates,
        "globals": globals_table,
        "graded": graded,
        "config": config,
    }


def _export(data: dict[str, object], mapping: pa.Table | None = None) -> pa.Table:
    return build_detection_export_table(
        data["detections"],  # type: ignore[arg-type]
        data["mapping"] if mapping is None else mapping,  # type: ignore[arg-type]
        data["microtracklets"],  # type: ignore[arg-type]
        clip_order=("A", "B"),
    )


def test_detection_export_is_a_deterministic_left_join(synthetic: dict[str, object]) -> None:
    result = _export(synthetic)
    assert result.schema == DETECTIONS_WITH_GLOBAL_ID_SCHEMA
    assert result["det_id"].to_pylist() == [100, 101, 102, 200, 201]
    assert result["legacy_track_id"].to_pylist() == [
        "old-a",
        "old-a",
        "old-invalid",
        "old-b",
        "old-b",
    ]
    assert result["display_global_id"].to_pylist() == [
        "G0001",
        "G0001",
        None,
        "G0002",
        "G0002",
    ]
    assert result["local_purity_score"].to_pylist() == pytest.approx(
        [0.9, 0.8, None, 0.7, 0.7], nan_ok=True
    )
    assert result["invalid_reason"].to_pylist() == [
        None,
        None,
        "AREA_BELOW_MINIMUM",
        None,
        None,
    ]
    for name in (
        "assignment_confidence",
        "incoming_link_probability",
        "outgoing_link_probability",
    ):
        assert result[name].null_count == result.num_rows
    invalid = result.slice(2, 1)
    for name in (
        "global_track_id",
        "display_global_id",
        "id_status",
        "micro_id",
        "stable_id",
        "order_in_global_detection",
        "local_purity_score",
    ):
        assert invalid[name].null_count == 1
    # Source bbox evidence remains auditable even when identity is invalid.
    assert invalid["bbox_confidence"][0].as_py() == pytest.approx(0.75)

    shuffled_again = synthetic["detections"].take(pa.array([1, 4, 3, 0, 2]))
    second = build_detection_export_table(
        shuffled_again,
        synthetic["mapping"].take(pa.array([3, 1, 0, 2])),
        synthetic["microtracklets"].take(pa.array([2, 0, 1])),
        clip_order=("A", "B"),
    )
    assert result.equals(second)


def test_missing_valid_mapping_stays_null_and_is_recomputed(
    synthetic: dict[str, object],
) -> None:
    mapping = synthetic["mapping"].filter(
        pa.compute.not_equal(synthetic["mapping"]["det_id"], 201)
    )
    result = _export(synthetic, mapping)
    assert result["global_track_id"].to_pylist()[-1] is None
    metrics = recompute_structural_metrics(
        result, synthetic["stable"], synthetic["candidates"]
    )
    assert metrics == StructuralMetrics(0, 0, 0, 1)


def test_structural_metrics_recompute_duplicates_overlap_and_cycles(
    synthetic: dict[str, object],
) -> None:
    mapping = synthetic["mapping"].filter(
        pa.compute.not_equal(synthetic["mapping"]["det_id"], 201)
    )
    export = _export(synthetic, mapping)
    frames = export["global_frame"].to_pylist()
    frames[1] = frames[0]
    export = _replace_column(
        export,
        DETECTIONS_WITH_GLOBAL_ID_SCHEMA,
        "global_frame",
        pa.array(frames, type=pa.int64()),
    )
    candidates = _table(
        FORCED_CANDIDATE_EDGES_SCHEMA,
        [
            _candidate("forward", 0, 1, 0, 1, "A_CLEAN", "B_EXISTING_DEGRADED"),
            _candidate(
                "reverse",
                1,
                0,
                1,
                0,
                "B_EXISTING_DEGRADED",
                "A_CLEAN",
                strict=False,
            ),
        ],
    )
    metrics = recompute_structural_metrics(export, synthetic["stable"], candidates)
    assert metrics.same_frame_duplicate_count == 1
    assert metrics.selected_temporal_overlap_count == 1
    assert metrics.cycle_count == 1
    assert metrics.unassigned_valid_detection_count == 1


def test_global_summary_recomputes_exact_duration_cosine_and_population(
    synthetic: dict[str, object],
) -> None:
    export = _export(synthetic)
    summary = build_global_track_summary(
        export,
        synthetic["stable"],
        synthetic["globals"],
        synthetic["candidates"],
        synthetic["graded"],
        config=synthetic["config"],
    )
    assert summary.schema == GLOBAL_TRACK_SUMMARY_SCHEMA
    assert summary.num_rows == 2
    rows = summary.to_pylist()
    assert [row["display_global_id"] for row in rows] == ["G0001", "G0002"]
    assert rows[0]["duration_visible_sec"] == pytest.approx(2 * 1001 / 30000)
    assert rows[0]["selected_link_cosine_min"] == pytest.approx(0.4)
    assert rows[0]["selected_link_cosine_p10"] == pytest.approx(0.4)
    assert rows[0]["selected_link_cosine_mean"] == pytest.approx(0.4)
    assert rows[0]["selected_link_cosine_max"] == pytest.approx(0.4)
    assert rows[0]["longest_gap_sec"] == pytest.approx(_DT)
    assert rows[0]["num_links_cosine_below_0_5"] == 1
    assert rows[0]["num_link_endpoints_grade_a_clean"] == 1
    assert rows[0]["num_link_endpoints_grade_b_existing_degraded"] == 1
    assert rows[1]["selected_link_cosine_min"] is None
    for row in rows:
        assert row["population_soft_max"] == 1
        assert row["population_overflow"] == 1
        assert row["population_warning"] is True
        assert row["min_link_probability"] is None
        assert row["certification_claimed"] is False
        assert "not_a_calibrated_probability" in row[
            "probability_not_available_reason"
        ]


def test_global_summary_uses_observed_snapshot_counts_and_sequence(
    synthetic: dict[str, object],
) -> None:
    """Historical snapshot numbers must never gate a structurally valid export."""

    config = replace(
        synthetic["config"],
        expected_sequence_id="stale-reference-sequence",
        expected_frame_count=999,
        expected_total_detection_count=999,
        expected_valid_detection_count=998,
        expected_invalid_detection_count=1,
        expected_total_detections_by_clip=(500, 499),
        expected_valid_detections_by_clip=(499, 499),
        expected_invalid_detections_by_clip=(1, 0),
        expected_microtrack_count=999,
        expected_stable_track_count=999,
        expected_selected_link_count=998,
        expected_candidate_edge_count=999,
        expected_selected_prior_link_count=999,
        expected_rescue_candidate_count=999,
        expected_rescue_embedding_count=999,
        expected_grade_a_clean_count=997,
        expected_grade_b_existing_degraded_count=1,
        expected_grade_c_reencoded_degraded_count=1,
        expected_cycle_count=7,
        expected_same_frame_violation_count=7,
        expected_temporal_overlap_violation_count=7,
        expected_unassigned_valid_detection_count=7,
        population_overflow=999,
    )
    summary = build_global_track_summary(
        _export(synthetic),
        synthetic["stable"],
        synthetic["globals"],
        synthetic["candidates"],
        synthetic["graded"],
        config=config,
    )

    assert summary.num_rows == 2
    assert set(summary["sequence_id"].to_pylist()) == {"seq"}
    assert summary["population_overflow"].to_pylist() == [1, 1]


def test_global_summary_rejects_structure_even_if_config_allows_it(
    synthetic: dict[str, object],
) -> None:
    export = _export(synthetic)
    frames = export["global_frame"].to_pylist()
    det_ids = export["det_id"].to_pylist()
    frames[det_ids.index(201)] = frames[det_ids.index(200)]
    export = _replace_column(
        export,
        DETECTIONS_WITH_GLOBAL_ID_SCHEMA,
        "global_frame",
        pa.array(frames, type=pa.int64()),
    )
    config = replace(
        synthetic["config"], expected_same_frame_violation_count=1
    )

    with pytest.raises(ContractError, match="structural metrics differ"):
        build_global_track_summary(
            export,
            synthetic["stable"],
            synthetic["globals"],
            synthetic["candidates"],
            synthetic["graded"],
            config=config,
        )


def test_global_summary_keeps_target_global_count_fatal(
    synthetic: dict[str, object],
) -> None:
    config = replace(
        synthetic["config"],
        expected_global_track_count=3,
        num_provisional_ids=3,
    )

    with pytest.raises(ContractError, match="global IDs differ from config"):
        build_global_track_summary(
            _export(synthetic),
            synthetic["stable"],
            synthetic["globals"],
            synthetic["candidates"],
            synthetic["graded"],
            config=config,
        )


def test_detection_export_rejects_identity_upgrade(synthetic: dict[str, object]) -> None:
    mapping = synthetic["mapping"]
    statuses = ["confirmed"] * mapping.num_rows
    mapping = _replace_column(
        mapping,
        DET_TO_GLOBAL_SCHEMA,
        "id_status",
        pa.array(statuses, type=pa.string()),
    )
    with pytest.raises(ContractError, match="forced_provisional"):
        _export(synthetic, mapping)


def test_detection_export_requires_old_id_for_every_valid_row(
    synthetic: dict[str, object],
) -> None:
    detections = synthetic["detections"]
    legacy = detections["legacy_track_id"].to_pylist()
    det_ids = detections["det_id"].to_pylist()
    legacy[det_ids.index(100)] = None
    detections = _replace_column(
        detections,
        DETECTIONS_SCHEMA,
        "legacy_track_id",
        pa.array(legacy, type=pa.string()),
    )
    with pytest.raises(ContractError, match="old-to-new ID audit"):
        build_detection_export_table(
            detections,
            synthetic["mapping"],
            synthetic["microtracklets"],
            clip_order=("A", "B"),
        )
