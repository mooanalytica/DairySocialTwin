from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from cowtrack.config import ContractError
from cowtrack.linking.forced_appearance import ForcedAppearanceBundle
from cowtrack.linking.forced_appearance_config import load_forced_appearance_config
from cowtrack.linking.s04_runtime import StableTracklet
import cowtrack.stages.s05_force_appearance as stage


def _track(
    stable_id: int,
    *,
    clip_id: str,
    start_frame: int,
) -> StableTracklet:
    return StableTracklet(
        stable_id=stable_id,
        first_micro_id=stable_id,
        last_micro_id=stable_id,
        start_det_id=stable_id * 10,
        end_det_id=stable_id * 10 + 1,
        start_clip_id=clip_id,
        end_clip_id=clip_id,
        start_global_frame=start_frame,
        end_global_frame=start_frame + 1,
        start_time_sec=float(start_frame),
        end_time_sec=float(start_frame + 1),
        num_microtracklets=1,
        num_detections=2,
        num_proposal_edges=0,
        min_proposal_probability=None,
        mean_proposal_probability=None,
        max_proposal_probability=None,
        is_singleton=True,
    )


def _fixture(
    clip_order: Sequence[str],
    *,
    target_paths: int = 2,
) -> tuple[SimpleNamespace, ForcedAppearanceBundle, object]:
    identity_centers = np.asarray(
        ([1.0, 0.0], [0.0, 1.0]), dtype=np.float32
    )
    tracks: dict[int, StableTracklet] = {}
    centers: list[np.ndarray] = []
    stable_id = 0
    for clip_index, clip_id in enumerate(clip_order):
        start_frame = clip_index * 100
        for center in identity_centers:
            tracks[stable_id] = _track(
                stable_id,
                clip_id=clip_id,
                start_frame=start_frame,
            )
            centers.append(center)
            stable_id += 1

    center_array = np.asarray(centers, dtype=np.float32)
    stable_ids = np.arange(len(tracks), dtype=np.int64)
    stable = SimpleNamespace(
        stable_ids=stable_ids,
        stable_tracklets=tracks,
    )
    prototypes = np.zeros(
        (len(tracks), 3, center_array.shape[1]), dtype=np.float16
    )
    prototypes[:, 0] = center_array.astype(np.float16)
    prototype_mask = np.zeros((len(tracks), 3), dtype=np.bool_)
    prototype_mask[:, 0] = True
    appearance = ForcedAppearanceBundle(
        stable_ids=stable_ids,
        prototypes=prototypes,
        prototype_mask=prototype_mask,
        descriptor_centers=center_array,
        rows=tuple(
            {
                "stable_id": int(item),
                "evidence_grade": "A_CLEAN",
                "descriptor_usable": True,
                "num_valid_prototypes": 1,
            }
            for item in stable_ids
        ),
        zero_sample_stable_ids=(),
    )
    config, _, _ = load_forced_appearance_config(
        Path("configs/s05_force_appearance.yaml")
    )
    config = replace(
        config,
        expected_stable_track_count=len(tracks),
        expected_microtrack_count=len(tracks),
        expected_valid_detection_count=2 * len(tracks),
        expected_invalid_detection_count=0,
        expected_clean_appearance_count=len(tracks),
        expected_missing_clean_appearance_count=0,
        expected_no_s02_sample_count=0,
        expected_prior_link_count=0,
        clip_order=tuple(clip_order),
        frame_counts_by_clip=tuple(100 for _ in clip_order),
        expected_valid_detections_by_clip=tuple(4 for _ in clip_order),
        target_global_track_count=target_paths,
        source_top_k=4,
        target_top_k=4,
        progress_interval_sec=0.05,
    )
    return stable, appearance, config


def _selected_pairs(result: object) -> set[tuple[int, int]]:
    return {
        (int(edge.source_stable_id), int(edge.target_stable_id))
        for edge in result.selected_edges
    }


def test_full_sequence_builds_one_graph_and_exact_cover(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stable, appearance, config = _fixture(("A", "B", "C", "D"))
    original_builder = stage.build_forced_candidate_graph
    builder_calls: list[tuple[int, ...]] = []

    def recording_builder(nodes, *args, **kwargs):
        builder_calls.append(tuple(int(node.stable_id) for node in nodes))
        return original_builder(nodes, *args, **kwargs)

    monkeypatch.setattr(
        stage, "build_forced_candidate_graph", recording_builder
    )
    messages: list[str] = []

    graph, result, solve_passes = stage._build_graph_and_cover(
        stable,
        appearance,
        set(),
        config,
        logger=messages.append,
    )

    assert builder_calls == [tuple(range(8))]
    assert graph.max_concurrent == graph.chain_count == 2
    assert sorted(
        stable_id
        for chain in graph.backbone_chains
        for stable_id in chain
    ) == list(range(8))
    assert result.paths == ((0, 2, 4, 6), (1, 3, 5, 7))
    assert _selected_pairs(result) == {
        (0, 2),
        (1, 3),
        (2, 4),
        (3, 5),
        (4, 6),
        (5, 7),
    }
    assert result.num_paths == result.target_num_paths == 2
    assert result.required_links == result.num_selected_links == 6
    assert result.maximum_feasible_links == 6

    assert len(solve_passes) == 1
    solve_pass = solve_passes[0]
    assert solve_pass.pass_index == 0
    assert solve_pass.source_clip_ids == ("A", "B", "C", "D")
    assert solve_pass.solver_nodes == 8
    assert solve_pass.candidate_count == len(graph.candidates)
    assert solve_pass.interval_width == 2
    assert solve_pass.backbone_chain_count == 2
    assert solve_pass.maximum_feasible_links == 6
    assert solve_pass.required_links == solve_pass.selected_links == 6
    assert solve_pass.total_appearance_cost_int == (
        result.total_appearance_cost_int
    )
    assert any(
        "full-sequence ranks[0:8) clips=A,B,C,D" in message
        for message in messages
    )

    candidate_rows = stage._candidate_rows(graph, result, stable)
    _, global_rows, _ = stage._build_global_rows(
        stable, graph, result, config
    )
    report = stage._build_report(
        config=config,
        config_hash="synthetic-config",
        appearance=appearance,
        rescue_rows=(),
        graph=graph,
        result=result,
        solve_passes=solve_passes,
        candidate_rows=candidate_rows,
        global_rows=global_rows,
        input_fingerprints=(),
        elapsed_sec=1.0,
    )
    solver = report["solver"]
    assert solver["solve_mode"] == "full_sequence_single_solve"
    assert solver["candidate_scope"] == "complete_sequence"
    assert solver["solve_pass_count"] == 1
    assert solver["solve_passes"] == [solve_passes[0].as_dict()]
    expected_uuid = str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"cowtrack://{config.expected_sequence_id}/"
            "s05-force/full-sequence/0/0",
        )
    )
    assert global_rows[0]["global_track_uuid"] == expected_uuid


def test_full_sequence_passes_interval_backbone_to_fixed_solver_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stable, appearance, config = _fixture(("A", "B", "C"))
    observed: dict[str, object] = {}

    def fake_fixed_solver(
        supplied_nodes,
        supplied_edges,
        *,
        target_num_paths,
        certified_path_cover,
        progress_interval_sec,
        progress_logger,
        progress_label,
    ):
        observed["nodes"] = supplied_nodes
        observed["edges"] = supplied_edges
        observed["target"] = target_num_paths
        observed["certificate"] = certified_path_cover
        observed["interval"] = progress_interval_sec
        observed["logger"] = progress_logger
        observed["label"] = progress_label
        maximum = len(supplied_nodes) - len(certified_path_cover)
        required = len(supplied_nodes) - target_num_paths
        return SimpleNamespace(
            target_num_paths=target_num_paths,
            required_links=required,
            num_selected_links=required,
            maximum_feasible_links=maximum,
            total_appearance_cost_int=123,
            num_paths=target_num_paths,
        )

    monkeypatch.setattr(
        stage, "solve_forced_fixed_path_cover", fake_fixed_solver
    )
    messages: list[str] = []
    logger = messages.append

    graph, result, solve_passes = stage._build_graph_and_cover(
        stable,
        appearance,
        set(),
        config,
        logger=logger,
    )

    assert tuple(int(node.stable_id) for node in observed["nodes"]) == tuple(
        range(6)
    )
    assert observed["edges"] == graph.edges
    assert observed["target"] == config.target_global_track_count == 2
    assert observed["certificate"] is graph.backbone_chains
    assert len(graph.backbone_chains) == graph.max_concurrent == 2
    assert observed["interval"] == config.progress_interval_sec
    assert observed["logger"] is logger
    assert "full-sequence ranks[0:6)" in str(observed["label"])
    assert result.required_links == result.num_selected_links == 4
    assert result.maximum_feasible_links == 4
    assert len(solve_passes) == 1
    assert solve_passes[0].pass_index == 0
    assert solve_passes[0].solver_nodes == 6
    assert solve_passes[0].source_clip_ids == ("A", "B", "C")


def test_full_sequence_rejects_target_below_interval_width_before_build(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stable, appearance, config = _fixture(
        ("A", "B", "C"), target_paths=1
    )

    def forbidden_builder(*_args, **_kwargs):
        raise AssertionError("infeasible concurrency must fail before graph build")

    monkeypatch.setattr(
        stage, "build_forced_candidate_graph", forbidden_builder
    )

    with pytest.raises(
        ContractError,
        match=r"concurrency lower bound: 2 > 1",
    ):
        stage._build_graph_and_cover(
            stable,
            appearance,
            set(),
            config,
            logger=lambda _message: None,
        )
