from __future__ import annotations

from typing import Any

import numpy as np

from cowtrack.qa.s01_review_plan import build_review_plan


def _review_inputs(
    specs: list[dict[str, Any]],
) -> tuple[
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
]:
    det_ids: list[int] = []
    micro_ids: list[int] = []
    orders: list[int] = []
    times: list[float] = []
    cx_values: list[float] = []
    legacy_values: list[str] = []
    src_ids: list[int] = []
    dst_ids: list[int] = []
    forward_costs: list[float] = []
    backward_costs: list[float] = []
    summary_rows: list[dict[str, Any]] = []

    next_det_id = 0
    for micro_id, spec in enumerate(specs):
        count = int(spec["count"])
        start_time = float(spec.get("start_time", micro_id * 100.0))
        step_sec = float(spec.get("step_sec", 1.0))
        track_times = start_time + np.arange(count, dtype=np.float64) * step_sec
        track_cx = list(spec.get("cx", [0.5] * count))
        track_legacy = list(spec.get("legacy", ["stable"] * count))
        track_cost = list(spec.get("cost", [0.01] * max(0, count - 1)))
        assert len(track_cx) == count
        assert len(track_legacy) == count
        assert len(track_cost) == max(0, count - 1)

        track_det_ids = list(range(next_det_id, next_det_id + count))
        det_ids.extend(track_det_ids)
        micro_ids.extend([micro_id] * count)
        orders.extend(range(count))
        times.extend(track_times.tolist())
        cx_values.extend(track_cx)
        legacy_values.extend(track_legacy)
        src_ids.extend(track_det_ids[:-1])
        dst_ids.extend(track_det_ids[1:])
        forward_costs.extend(track_cost)
        backward_costs.extend(track_cost)
        summary_rows.append(
            {
                "micro_id": micro_id,
                "num_detections": count,
                "start_time_sec": float(track_times[0]),
                "end_time_sec": float(track_times[-1]),
                "local_purity_score": float(spec["purity"]),
                "max_internal_center_jump": float(spec.get("max_jump", 0.0)),
                "status": str(spec["status"]),
            }
        )
        next_det_id += count

    time_array = np.asarray(times, dtype=np.float64)
    global_frames = np.rint(time_array * 10.0).astype(np.int64)
    num_detections = len(det_ids)
    detections = {
        "det_id": np.asarray(det_ids, dtype=np.int64),
        "sequence_id": np.full(num_detections, "sequence"),
        "clip_id": np.full(num_detections, "clip"),
        "local_frame": global_frames.astype(np.int32),
        "global_frame": global_frames,
        "global_time_sec": time_array,
        "cx_norm": np.asarray(cx_values, dtype=np.float64),
        "cy_norm": np.full(num_detections, 0.5),
        "w_norm": np.full(num_detections, 0.2),
        "h_norm": np.full(num_detections, 0.2),
        "legacy_track_id": np.asarray(legacy_values, dtype=object),
        "valid": np.ones(num_detections, dtype=np.bool_),
    }
    mapping = {
        "det_id": np.asarray(det_ids, dtype=np.int64),
        "micro_id": np.asarray(micro_ids, dtype=np.int64),
        "order_in_micro": np.asarray(orders, dtype=np.int32),
    }
    tracklets = {
        name: np.asarray([row[name] for row in summary_rows])
        for name in summary_rows[0]
    }
    edges = {
        "src_det_id": np.asarray(src_ids, dtype=np.int64),
        "dst_det_id": np.asarray(dst_ids, dtype=np.int64),
        "forward_cost": np.asarray(forward_costs, dtype=np.float64),
        "backward_cost": np.asarray(backward_costs, dtype=np.float64),
        "accepted": np.ones(len(src_ids), dtype=np.bool_),
    }
    return detections, mapping, tracklets, edges


def test_long_track_chunks_cover_all_and_absorb_event_cases() -> None:
    inputs = _review_inputs(
        [
            {
                "count": 8,
                "status": "valid",
                "purity": 0.7,
                "max_jump": 0.71,
                "cx": [0.5, 0.5, 0.5, 0.5, 0.7, 0.7, 0.7, 0.7],
                "legacy": ["a", "a", "a", "a", "b", "b", "b", "b"],
                "cost": [0.01, 0.01, 0.01, 0.4, 0.01, 0.01, 0.01],
            }
        ]
    )
    plan = build_review_plan(
        detections=inputs[0],
        det_to_micro=inputs[1],
        microtracklets=inputs[2],
        det_edges=inputs[3],
        long_track_min_detections=6,
        long_track_chunk_sec=4.0,
        long_track_chunk_overlap_sec=1.0,
        requested_quality_samples=0,
        timeline_start_time_sec=0.0,
        timeline_end_time_sec=7.0,
    )

    assert len(plan.risk_cases) == 2
    assert [
        (case.window_start_time_sec, case.window_end_time_sec)
        for case in plan.risk_cases
    ] == [(0.0, 4.0), (3.0, 7.0)]
    assert all(
        case.window_end_time_sec - case.window_start_time_sec <= 4.0
        for case in plan.risk_cases
    )
    event_chunk = next(
        case for case in plan.risk_cases if "low_purity" in case.reasons
    )
    assert set(event_chunk.reasons) == {
        "long_full_review",
        "chunk_2_of_2",
        "low_purity",
        "large_center_jump",
        "legacy_id_transition",
    }
    assert len(event_chunk.events) == 4
    assert len({case.case_id for case in plan.risk_cases}) == 2


def test_low_purity_short_fragment_is_not_a_risk_and_good_sample_is_selected() -> None:
    inputs = _review_inputs(
        [
            {"count": 2, "status": "short_fragment", "purity": 0.0},
            {"count": 3, "status": "valid", "purity": 0.7},
            {"count": 4, "status": "valid", "purity": 0.999},
        ]
    )
    plan = build_review_plan(
        detections=inputs[0],
        det_to_micro=inputs[1],
        microtracklets=inputs[2],
        det_edges=inputs[3],
        long_track_min_detections=100,
        requested_quality_samples=1,
        quality_min_detections=3,
        quality_max_detections=5,
        timeline_start_time_sec=0.0,
        timeline_end_time_sec=204.0,
    )

    assert plan.risk_microtracks == (1,)
    assert len(plan.risk_cases) == 1
    assert plan.risk_cases[0].reasons == ("low_purity",)
    assert len(plan.quality_cases) == 1
    assert plan.quality_cases[0].micro_id == 2
    payload = plan.to_manifest_payload()
    assert payload["legacy_track_id_used_for_tracking"] is False
    assert payload["selection_policy"]["risk"]["low_purity_status"] == "valid"


def test_chain_of_nearby_events_is_one_case_with_union_window() -> None:
    inputs = _review_inputs(
        [
            {
                "count": 5,
                "status": "valid",
                "purity": 0.999,
                "legacy": ["a", "b", "c", "d", "d"],
                "step_sec": 2.0,
            }
        ]
    )
    plan = build_review_plan(
        detections=inputs[0],
        det_to_micro=inputs[1],
        microtracklets=inputs[2],
        det_edges=inputs[3],
        long_track_min_detections=100,
        requested_quality_samples=0,
        window_before_sec=3.0,
        window_after_sec=3.0,
        timeline_start_time_sec=0.0,
        timeline_end_time_sec=12.0,
    )

    assert len(plan.risk_cases) == 1
    case = plan.risk_cases[0]
    assert len(case.events) == 3
    assert case.window_start_time_sec == 0.0
    assert case.window_end_time_sec == 9.0
