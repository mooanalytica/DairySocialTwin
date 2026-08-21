from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

import cowtrack.tracking.microtrack as microtrack_module

from cowtrack.tracking.microtrack import (
    DetectionBatch,
    LinkResult,
    MicrotrackSettings,
    MotionPrior,
    assign_microtracks,
    build_motion_prior,
    link_microtracks,
)


def _batch(
    rows: list[tuple[int, int, float, float]],
    *,
    width_norm: float = 0.20,
    height_norm: float = 0.20,
) -> DetectionBatch:
    """Build sorted detections from (det_id, frame, cx_norm, cy_norm)."""

    det_id = np.asarray([row[0] for row in rows], dtype=np.int64)
    global_frame = np.asarray([row[1] for row in rows], dtype=np.int64)
    cx_norm = np.asarray([row[2] for row in rows], dtype=np.float64)
    cy_norm = np.asarray([row[3] for row in rows], dtype=np.float64)
    w_norm = np.full(len(rows), width_norm, dtype=np.float64)
    h_norm = np.full(len(rows), height_norm, dtype=np.float64)

    # The core consumes normalized geometry for matching but the input contract
    # also requires the original pixel-space box columns.
    image_width = 1_000.0
    image_height = 800.0
    x1 = (cx_norm - 0.5 * w_norm) * image_width
    y1 = (cy_norm - 0.5 * h_norm) * image_height
    x2 = (cx_norm + 0.5 * w_norm) * image_width
    y2 = (cy_norm + 0.5 * h_norm) * image_height

    return DetectionBatch(
        det_id=det_id,
        global_frame=global_frame,
        global_time_sec=global_frame.astype(np.float64) * 0.1,
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
        cx_norm=cx_norm,
        cy_norm=cy_norm,
        w_norm=w_norm,
        h_norm=h_norm,
    )


def _links(pairs: list[tuple[int, int]]) -> LinkResult:
    count = len(pairs)
    return LinkResult(
        src_index=np.asarray([pair[0] for pair in pairs], dtype=np.int64),
        dst_index=np.asarray([pair[1] for pair in pairs], dtype=np.int64),
        forward_cost=np.full(count, 0.1, dtype=np.float32),
        backward_cost=np.full(count, 0.1, dtype=np.float32),
        forward_rank=np.ones(count, dtype=np.int16),
        backward_rank=np.ones(count, dtype=np.int16),
        center_residual=np.full(count, 0.05, dtype=np.float32),
        scale_ratio=np.ones(count, dtype=np.float32),
    )


def _det_id_pairs(batch: DetectionBatch, links: LinkResult) -> set[tuple[int, int]]:
    return {
        (int(batch.det_id[source]), int(batch.det_id[destination]))
        for source, destination in links.edge_pairs()
    }


@pytest.fixture
def settings() -> MicrotrackSettings:
    return MicrotrackSettings(
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
        grid_width=4,
        grid_height=1,
        motion_prior_gate_floor=0.10,
        motion_prior_min_edges_per_cell=2,
    )


def test_clear_consecutive_track_links_and_assigns(
    settings: MicrotrackSettings,
) -> None:
    batch = _batch(
        [
            (10, 0, 0.20, 0.50),
            (11, 1, 0.30, 0.50),
            (12, 2, 0.40, 0.50),
        ]
    )

    links = link_microtracks(batch, settings)
    assignment = assign_microtracks(batch, links)

    assert _det_id_pairs(batch, links) == {(10, 11), (11, 12)}
    assert assignment.micro_id.tolist() == [0, 0, 0]
    assert assignment.order_in_micro.tolist() == [0, 1, 2]
    assert np.isnan(assignment.incoming_edge_score[0])
    assert np.all(np.isfinite(assignment.incoming_edge_score[1:]))
    assert [batch.det_id[path].tolist() for path in assignment.paths] == [[10, 11, 12]]


def test_missing_global_frame_is_never_bridged(
    settings: MicrotrackSettings,
) -> None:
    batch = _batch(
        [
            (10, 0, 0.20, 0.50),
            (12, 2, 0.21, 0.50),
        ]
    )

    links = link_microtracks(batch, replace(settings, max_time_gap_sec=1.0))
    assignment = assign_microtracks(batch, links)

    assert links.num_edges == 0
    assert assignment.micro_id.tolist() == [0, 1]
    assert assignment.order_in_micro.tolist() == [0, 0]


def test_exact_cost_tie_is_cut_by_ambiguity_margin(
    settings: MicrotrackSettings,
) -> None:
    batch = _batch(
        [
            (10, 0, 0.50, 0.50),
            (20, 1, 0.50, 0.50),
            (21, 1, 0.50, 0.50),
        ]
    )

    without_margin = link_microtracks(
        batch, replace(settings, ambiguity_margin=0.0)
    )
    with_margin = link_microtracks(batch, settings)

    assert _det_id_pairs(batch, without_margin) == {(10, 20)}
    assert with_margin.num_edges == 0
    assert len(assign_microtracks(batch, with_margin).paths) == 3


def test_crossing_cuts_at_ambiguity_without_mixing_truth(
    settings: MicrotrackSettings,
) -> None:
    batch = _batch(
        [
            (100, 0, 0.20, 0.50),
            (200, 0, 0.80, 0.50),
            (101, 1, 0.35, 0.50),
            (201, 1, 0.65, 0.50),
            (102, 2, 0.49, 0.50),
            (202, 2, 0.51, 0.50),
            (103, 3, 0.65, 0.50),
            (203, 3, 0.35, 0.50),
            (104, 4, 0.80, 0.50),
            (204, 4, 0.20, 0.50),
        ]
    )
    truth = {
        100: "a",
        101: "a",
        102: "a",
        103: "a",
        104: "a",
        200: "b",
        201: "b",
        202: "b",
        203: "b",
        204: "b",
    }

    links = link_microtracks(batch, settings)
    assignment = assign_microtracks(batch, links)

    frame_pairs = {
        (int(batch.global_frame[source]), int(batch.global_frame[destination]))
        for source, destination in links.edge_pairs()
    }
    assert frame_pairs == {(0, 1), (3, 4)}
    assert links.num_edges == 4
    assert all(
        truth[int(batch.det_id[source])] == truth[int(batch.det_id[destination])]
        for source, destination in links.edge_pairs()
    )
    for path in assignment.paths:
        assert len({truth[int(det_id)] for det_id in batch.det_id[path]}) == 1


def test_motion_prior_grid_rejects_sparse_and_empty_cells_on_second_pass(
    settings: MicrotrackSettings,
) -> None:
    batch = _batch(
        [
            (10, 0, 0.10, 0.50),
            (20, 0, 0.35, 0.50),
            (11, 1, 0.11, 0.50),
            (21, 1, 0.36, 0.50),
            (12, 2, 0.12, 0.50),
        ]
    )

    first_pass = link_microtracks(batch, settings)
    prior = build_motion_prior(batch, first_pass, settings)
    second_pass = link_microtracks(
        batch,
        settings,
        prior=prior,
        allowed_edge_pairs=first_pass.edge_pairs(),
    )

    np.testing.assert_array_equal(prior.count, np.asarray([[2, 1, 0, 0]]))
    assert np.all(np.isfinite(prior.residual_p99[0, :2]))
    assert np.all(np.isnan(prior.residual_p99[0, 2:]))
    assert _det_id_pairs(batch, first_pass) == {(10, 11), (11, 12), (20, 21)}
    assert _det_id_pairs(batch, second_pass) == {(10, 11), (11, 12)}

    empty_cell_probe = _batch(
        [
            (30, 0, 0.65, 0.50),
            (31, 1, 0.66, 0.50),
        ]
    )
    assert link_microtracks(empty_cell_probe, settings, prior=prior).num_edges == 0


def test_reliable_motion_prior_p99_prunes_residual_outlier(
    settings: MicrotrackSettings,
) -> None:
    count = np.asarray([[8, 0, 0, 0]], dtype=np.int64)
    values = np.asarray([[0.10, np.nan, np.nan, np.nan]], dtype=np.float32)
    scales = np.asarray([[0.20, np.nan, np.nan, np.nan]], dtype=np.float32)
    prior = MotionPrior(count, values, values, values, scales, scales, scales)
    outlier = _batch([(10, 0, 0.10, 0.50), (11, 1, 0.15, 0.50)])
    inlier = _batch([(20, 0, 0.10, 0.50), (21, 1, 0.12, 0.50)])

    assert link_microtracks(outlier, settings, prior=prior).num_edges == 0
    assert _det_id_pairs(inlier, link_microtracks(inlier, settings, prior=prior)) == {
        (20, 21)
    }


def test_area_ratio_hard_gate_rejects_scale_jump(
    settings: MicrotrackSettings,
) -> None:
    batch = _batch([(10, 0, 0.30, 0.50), (11, 1, 0.30, 0.50)])
    batch = replace(
        batch,
        w_norm=np.asarray([0.20, 0.40]),
        h_norm=np.asarray([0.20, 0.40]),
    )

    assert link_microtracks(batch, settings).num_edges == 0


def test_only_forward_backward_intersection_is_returned(
    settings: MicrotrackSettings, monkeypatch
) -> None:
    batch = _batch(
        [
            (10, 0, 0.20, 0.50),
            (20, 0, 0.80, 0.50),
            (11, 1, 0.21, 0.50),
            (21, 1, 0.79, 0.50),
        ]
    )
    common = (0, 2)
    forward_only = (1, 3)

    def fake_directional_pass(*_args, reverse: bool, allowed_edge_pairs, **_kwargs):
        pairs = {common} if reverse else {common, forward_only}
        if allowed_edge_pairs is not None:
            pairs &= allowed_edge_pairs
        cost = 0.2 if reverse else 0.1
        return {
            pair: SimpleNamespace(
                cost=cost,
                rank=1,
                center_residual=cost,
                scale_ratio=1.0,
            )
            for pair in pairs
        }

    monkeypatch.setattr(
        microtrack_module, "_directional_pass", fake_directional_pass
    )
    links = link_microtracks(batch, settings)

    assert links.edge_pairs() == {common}
    assert links.forward_cost.tolist() == pytest.approx([0.1])
    assert links.backward_cost.tolist() == pytest.approx([0.2])


def test_allowed_edge_pairs_make_relinking_a_strict_first_pass_subset(
    settings: MicrotrackSettings,
) -> None:
    batch = _batch(
        [
            (10, 0, 0.20, 0.50),
            (20, 0, 0.80, 0.50),
            (11, 1, 0.21, 0.50),
            (21, 1, 0.79, 0.50),
        ]
    )
    first_pass = link_microtracks(batch, settings)
    allowed = {min(first_pass.edge_pairs())}

    second_pass = link_microtracks(
        batch,
        settings,
        allowed_edge_pairs=allowed,
    )

    assert first_pass.num_edges == 2
    assert second_pass.edge_pairs() == allowed
    assert second_pass.edge_pairs() < first_pass.edge_pairs()


def test_non_allowed_competitor_still_triggers_ambiguity(
    settings: MicrotrackSettings,
) -> None:
    batch = _batch(
        [
            (10, 0, 0.50, 0.50),
            (20, 1, 0.49, 0.50),
            (21, 1, 0.51, 0.50),
        ]
    )
    allowed = {(0, 1)}

    links = link_microtracks(batch, settings, allowed_edge_pairs=allowed)

    assert links.num_edges == 0


def test_non_allowed_edges_never_extend_hidden_motion_history(
    settings: MicrotrackSettings,
) -> None:
    batch = _batch(
        [
            (10, 0, 0.10, 0.50),
            (11, 1, 0.30, 0.50),
            (12, 2, 0.50, 0.50),
            (13, 3, 0.70, 0.50),
        ]
    )

    links = link_microtracks(batch, settings, allowed_edge_pairs={(1, 2)})

    assert links.num_edges == 0


def test_assignment_covers_every_detection_once_with_deterministic_micro_ids() -> None:
    batch = _batch(
        [
            (10, 0, 0.20, 0.50),
            (20, 0, 0.80, 0.50),
            (5, 1, 0.50, 0.50),
            (11, 1, 0.21, 0.50),
            (21, 1, 0.79, 0.50),
        ]
    )

    assignment = assign_microtracks(batch, _links([(1, 4), (0, 3)]))
    reordered = assign_microtracks(batch, _links([(0, 3), (1, 4)]))

    visited = np.concatenate(assignment.paths)
    np.testing.assert_array_equal(np.sort(visited), np.arange(len(batch.det_id)))
    np.testing.assert_array_equal(
        np.bincount(visited, minlength=len(batch.det_id)),
        np.ones(len(batch.det_id), dtype=np.int64),
    )
    assert assignment.micro_id.tolist() == [0, 1, 2, 0, 1]
    assert assignment.order_in_micro.tolist() == [0, 0, 0, 1, 1]
    np.testing.assert_array_equal(assignment.micro_id, reordered.micro_id)
    np.testing.assert_array_equal(assignment.order_in_micro, reordered.order_in_micro)
    assert [batch.det_id[path].tolist() for path in assignment.paths] == [
        [10, 11],
        [20, 21],
        [5],
    ]
