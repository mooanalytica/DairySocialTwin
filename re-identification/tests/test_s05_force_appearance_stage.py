from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow as pa

from cowtrack.linking.forced_appearance_config import load_forced_appearance_config
from cowtrack.linking.forced_candidates import build_forced_candidate_graph
from cowtrack.linking.forced_path_cover import solve_forced_fixed_path_cover
from cowtrack.linking.path_cover import GlobalStableNode
from cowtrack.linking.s04_runtime import StableTracklet
from cowtrack.schemas.s05_finalize import (
    DET_TO_GLOBAL_SCHEMA,
    GLOBAL_TRACKS_SCHEMA,
    STABLE_TO_GLOBAL_SCHEMA,
)
from cowtrack.schemas.s05_forced import FORCED_CANDIDATE_EDGES_SCHEMA
from cowtrack.stages.s05_finalize import build_detection_to_global_table
import cowtrack.stages.s05_force_appearance as stage


def _track(stable_id: int, start: int, end: int) -> StableTracklet:
    return StableTracklet(
        stable_id=stable_id,
        first_micro_id=stable_id,
        last_micro_id=stable_id,
        start_det_id=100 + stable_id,
        end_det_id=100 + stable_id,
        start_clip_id="A",
        end_clip_id="A",
        start_global_frame=start,
        end_global_frame=end,
        start_time_sec=float(start),
        end_time_sec=float(end),
        num_microtracklets=1,
        num_detections=1,
        num_proposal_edges=0,
        min_proposal_probability=None,
        mean_proposal_probability=None,
        max_proposal_probability=None,
        is_singleton=True,
    )


def _fixture():
    tracks = {
        0: _track(0, 0, 1),
        1: _track(1, 0, 1),
        2: _track(2, 5, 6),
        3: _track(3, 5, 6),
    }
    stable = SimpleNamespace(
        stable_ids=np.arange(4, dtype=np.int64),
        stable_tracklets=tracks,
        det_ids=np.asarray([100, 101, 102, 103], dtype=np.int64),
        det_micro_ids=np.arange(4, dtype=np.int64),
        det_stable_ids=np.arange(4, dtype=np.int64),
        det_order_in_micro=np.zeros(4, dtype=np.int32),
        det_order_in_stable=np.zeros(4, dtype=np.int32),
        det_order_in_stable_detection=np.zeros(4, dtype=np.int64),
    )
    nodes = [
        GlobalStableNode(
            row.stable_id,
            row.start_clip_id,
            row.end_clip_id,
            row.start_global_frame,
            row.end_global_frame,
            row.start_time_sec,
            row.end_time_sec,
            row.num_microtracklets,
            row.num_detections,
        )
        for row in tracks.values()
    ]
    centers = np.asarray(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]],
        dtype=np.float32,
    )
    graph = build_forced_candidate_graph(
        nodes, centers, ["A_CLEAN"] * 4, set(), top_k=1
    )
    result = solve_forced_fixed_path_cover(nodes, graph.edges, target_num_paths=2)
    config, _, _ = load_forced_appearance_config(
        Path("configs/s05_force_appearance.yaml")
    )
    config = replace(
        config,
        expected_stable_track_count=4,
        expected_microtrack_count=4,
        expected_valid_detection_count=4,
        expected_clean_appearance_count=4,
        expected_missing_clean_appearance_count=0,
    )
    config = replace(
        config,
        expected_sequence_id="seq",
        expected_stable_track_count=4,
        expected_microtrack_count=4,
        expected_valid_detection_count=4,
        expected_invalid_detection_count=0,
        expected_frame_count=7,
        target_global_track_count=2,
        clip_order=("A",),
        frame_counts_by_clip=(7,),
        expected_valid_detections_by_clip=(4,),
    )
    return stable, graph, result, config


def test_forced_input_contract_accepts_locked_s02_choice_metadata() -> None:
    config, _, _ = load_forced_appearance_config(
        Path("configs/s05_force_appearance.yaml")
    )
    config = replace(
        config,
        expected_stable_track_count=4,
        expected_microtrack_count=4,
        expected_valid_detection_count=4,
        expected_clean_appearance_count=4,
        expected_missing_clean_appearance_count=0,
    )
    production = SimpleNamespace(
        detections=SimpleNamespace(
            det_ids=np.empty(config.expected_valid_detection_count, dtype=np.int64)
        ),
        encoder_choice={
            "selection_mode": "locked_existing_winner",
            "selected_profile": "megadescriptor_l_384_imagenet",
            "embedding_dim": 1536,
            "selection_metrics": {},
            "no_runtime_fallback": True,
        },
    )
    stable = SimpleNamespace(
        stable_ids=np.empty(config.expected_stable_track_count, dtype=np.int64),
        micro_ids=np.empty(config.expected_microtrack_count, dtype=np.int64),
        det_ids=np.empty(config.expected_valid_detection_count, dtype=np.int64),
        stable_appearance={
            stable_id: SimpleNamespace(appearance_usable=True)
            for stable_id in range(config.expected_clean_appearance_count)
        },
    )

    stage._validate_fixed_inputs(production, stable, config)


def test_forced_global_rows_and_detection_mapping_keep_provisional_semantics() -> None:
    stable, graph, result, config = _fixture()
    mapping, globals_, by_stable = stage._build_global_rows(
        stable, graph, result, config
    )
    assert len(mapping) == 4
    assert len(globals_) == 2
    assert sum(row["component_num_long_links"] for row in mapping if row["link_type"] == "PATH_START") == 2
    assert all(row["id_status"] == "forced_provisional" for row in mapping)
    assert all(row["predecessor_link_probability"] is None for row in mapping)
    assert all(row["id_status"] == "forced_provisional" for row in globals_)
    assert all(row["min_link_probability"] is None for row in globals_)

    identity = {
        "det_id": np.asarray([100, 101, 102, 103], dtype=np.int64),
        "sequence_id": np.asarray(["seq"] * 4, dtype=object),
        "clip_id": np.asarray(["A"] * 4, dtype=object),
        "local_frame": np.asarray([0, 0, 5, 5], dtype=np.int32),
        "global_frame": np.asarray([0, 0, 5, 5], dtype=np.int64),
        "global_time_sec": np.asarray([0.0, 0.0, 5.0, 5.0], dtype=np.float64),
    }
    table = build_detection_to_global_table(
        identity, stable, by_stable, globals_, config  # type: ignore[arg-type]
    )
    assert table.schema.equals(DET_TO_GLOBAL_SCHEMA, check_metadata=False)
    assert table.num_rows == 4
    assert set(table["id_status"].to_pylist()) == {"forced_provisional"}
    assert len(set(table["global_track_id"].to_pylist())) == 2

    assert pa.Table.from_pylist(mapping, schema=STABLE_TO_GLOBAL_SCHEMA).num_rows == 4
    assert pa.Table.from_pylist(globals_, schema=GLOBAL_TRACKS_SCHEMA).num_rows == 2


def test_forced_candidate_audit_is_total_and_keeps_cosine_separate() -> None:
    stable, graph, result, _ = _fixture()
    rows = stage._candidate_rows(graph, result, stable)
    table = pa.Table.from_pylist(rows, schema=FORCED_CANDIDATE_EDGES_SCHEMA)
    assert table.num_rows == len(graph.candidates)
    assert sum(table["selected_by_solver"].to_pylist()) == 2
    assert all(row["strictly_nonoverlapping"] for row in rows)
    assert all("probability" not in row for row in rows)
    assert {
        row["id_status"] for row in rows if row["selected_by_solver"]
    } == {"forced_provisional"}
