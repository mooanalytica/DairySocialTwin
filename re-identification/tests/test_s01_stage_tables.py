from __future__ import annotations

import numpy as np

from cowtrack.schemas.edges import DET_EDGES_SCHEMA
from cowtrack.schemas.tracklets import DET_TO_MICRO_SCHEMA, MICROTRACKLETS_SCHEMA
from cowtrack.stages.s01_microtrack import (
    S01_DETECTION_COLUMNS,
    LoadedInput,
    _edge_table,
    _mapping_table,
    _tracklet_table,
    _validate_results,
)
from cowtrack.tracking.microtrack import (
    DetectionBatch,
    MicrotrackSettings,
    assign_microtracks,
    build_motion_prior,
    link_microtracks,
)


def test_s01_tables_form_a_complete_path_partition() -> None:
    batch = DetectionBatch(
        det_id=np.asarray([10, 11, 12], dtype=np.int64),
        global_frame=np.asarray([0, 1, 2], dtype=np.int64),
        global_time_sec=np.asarray([0.0, 0.1, 0.2]),
        x1=np.asarray([100.0, 110.0, 120.0]),
        y1=np.asarray([100.0, 100.0, 100.0]),
        x2=np.asarray([300.0, 310.0, 320.0]),
        y2=np.asarray([300.0, 300.0, 300.0]),
        cx_norm=np.asarray([0.20, 0.21, 0.22]),
        cy_norm=np.asarray([0.20, 0.20, 0.20]),
        w_norm=np.asarray([0.20, 0.20, 0.20]),
        h_norm=np.asarray([0.20, 0.20, 0.20]),
    )
    settings = MicrotrackSettings(
        center_distance_weight=0.55,
        iou_weight=0.30,
        size_weight=0.15,
        max_time_gap_sec=0.15,
        center_distance_gate=0.75,
        max_area_ratio=1.8,
        min_iou=0.01,
        alternate_center_gate=0.35,
        ambiguity_margin=0.08,
        velocity_history_detections=4,
        grid_width=2,
        grid_height=2,
        motion_prior_gate_floor=0.10,
        motion_prior_min_edges_per_cell=1,
    )
    first = link_microtracks(batch, settings)
    prior = build_motion_prior(batch, first, settings)
    second = link_microtracks(
        batch,
        settings,
        prior=prior,
        allowed_edge_pairs=first.edge_pairs(),
    )
    assignment = assign_microtracks(batch, second)
    edges = _edge_table(batch, second)
    mapping = _mapping_table(batch, assignment)
    tracklets = _tracklet_table(
        batch,
        second,
        assignment,
        frame_period_sec=0.1,
        min_length_detections=3,
        velocity_history_detections=4,
    )
    loaded = LoadedInput(
        sequence_id="seq",
        clip_ids=("clip",),
        frame_times=np.asarray([0.0, 0.1, 0.2]),
        frame_period_sec=0.1,
        batch=batch,
    )
    _validate_results(
        loaded,
        first,
        second,
        prior,
        assignment,
        edges,
        mapping,
        tracklets,
        settings,
    )

    assert edges.schema == DET_EDGES_SCHEMA
    assert mapping.schema == DET_TO_MICRO_SCHEMA
    assert tracklets.schema == MICROTRACKLETS_SCHEMA
    assert edges.num_rows == 2
    assert mapping.num_rows == 3
    assert tracklets.num_rows == 1
    assert tracklets["duration_sec"][0].as_py() == np.float32(0.3)
    assert tracklets["status"][0].as_py() == "valid"
    assert "legacy_track_id" not in S01_DETECTION_COLUMNS
