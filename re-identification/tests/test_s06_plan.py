from __future__ import annotations

import random

import pytest

from cowtrack.config import ContractError
from cowtrack.qa.s06_plan import (
    CONTACT_SHEET_ROLES,
    CandidateMetric,
    CropRequest,
    CropSource,
    CropUse,
    DetectionCrop,
    build_contact_sheet_plan,
    build_crop_request_index,
    build_low_confidence_crop_requests,
    compute_directional_candidate_metrics,
    select_low_confidence_links,
)


def _candidate(
    candidate_id: str,
    source: int,
    target: int,
    cosine: float,
    *,
    selected: bool = False,
    source_frame: int | None = None,
    target_frame: int | None = None,
    gap: float | None = None,
) -> dict[str, object]:
    source_global_frame = source * 10 + 1 if source_frame is None else source_frame
    target_global_frame = target * 10 if target_frame is None else target_frame
    source_time = float(source_global_frame)
    target_time = float(target_global_frame)
    temporal_gap = target_time - source_time if gap is None else gap
    return {
        "candidate_id": candidate_id,
        "source_stable_id": source,
        "target_stable_id": target,
        "source_end_clip_id": "GX040006",
        "target_start_clip_id": "GX040006",
        "source_end_global_frame": source_global_frame,
        "target_start_global_frame": target_global_frame,
        "source_end_time_sec": source_time,
        "target_start_time_sec": target_time,
        "temporal_gap_sec": temporal_gap,
        "strictly_nonoverlapping": True,
        "appearance_cosine": cosine,
        "source_evidence_grade": "A",
        "target_evidence_grade": "B",
        "selected_by_source_topk": True,
        "selected_by_target_topk": False,
        "temporal_backbone": False,
        "prior_global_link": False,
        "selected_by_solver": selected,
        "global_link_id": f"link-{candidate_id}" if selected else None,
        "authorization_basis": "operator_forced_appearance_exact_62",
        "id_status": "forced_provisional" if selected else "candidate_only",
    }


def _metric(
    candidate_id: str,
    source: int,
    target: int,
    cosine: float,
    *,
    gap: float,
    source_frame: int,
    target_frame: int,
) -> CandidateMetric:
    return CandidateMetric(
        candidate_id=candidate_id,
        source_stable_id=source,
        target_stable_id=target,
        source_end_clip_id="GX040006",
        target_start_clip_id="GX040006",
        source_end_global_frame=source_frame,
        target_start_global_frame=target_frame,
        source_end_time_sec=float(source_frame),
        target_start_time_sec=float(target_frame),
        temporal_gap_sec=gap,
        appearance_cosine=cosine,
        source_evidence_grade="A",
        target_evidence_grade="B",
        selected_by_source_topk=True,
        selected_by_target_topk=True,
        temporal_backbone=False,
        prior_global_link=False,
        selected_by_solver=True,
        global_link_id=f"link-{candidate_id}",
        authorization_basis="operator_forced_appearance_exact_62",
        id_status="forced_provisional",
        outgoing_rank=1,
        incoming_rank=1,
        outgoing_margin=0.1,
        incoming_margin=0.1,
        conservative_margin=0.1,
    )


def _detection(
    det_id: int,
    stable_id: int,
    frame: int,
    *,
    global_id: int = 0,
) -> DetectionCrop:
    return DetectionCrop(
        det_id=det_id,
        clip_id="GX040006",
        local_frame=frame,
        global_frame=frame,
        global_time_sec=float(frame),
        stable_id=stable_id,
        global_track_id=global_id,
        display_global_id=f"G{global_id + 1:04d}",
        id_status="forced_provisional",
        x1=100.0 + det_id,
        y1=200.0,
        x2=300.0 + det_id,
        y2=500.0,
    )


def test_directional_rank_and_margin_are_persisted_graph_scoped_and_stable() -> None:
    rows = [
        _candidate("c-a", 1, 10, 0.8, selected=True),
        _candidate("c-b", 1, 11, 0.8),
        _candidate("c-c", 2, 10, 0.9),
        _candidate("c-d", 3, 12, 0.4),
    ]
    shuffled = rows.copy()
    random.Random(91).shuffle(shuffled)

    first = compute_directional_candidate_metrics(rows)
    second = compute_directional_candidate_metrics(shuffled)
    assert first == second
    by_id = {item.candidate_id: item for item in first}
    assert by_id["c-a"].outgoing_rank == 1  # target 10 wins exact-score tie.
    assert by_id["c-b"].outgoing_rank == 2
    assert by_id["c-a"].outgoing_margin == pytest.approx(0.0)
    assert by_id["c-b"].outgoing_margin == pytest.approx(0.0)
    assert by_id["c-c"].incoming_rank == 1
    assert by_id["c-c"].incoming_margin == pytest.approx(0.1)
    assert by_id["c-a"].incoming_rank == 2
    assert by_id["c-a"].incoming_margin == pytest.approx(-0.1)
    assert by_id["c-a"].conservative_margin == pytest.approx(-0.1)
    assert by_id["c-d"].outgoing_margin is None
    assert by_id["c-d"].incoming_margin is None
    record = by_id["c-a"].to_record()
    assert record["rank_scope"] == "persisted_candidate_graph"
    assert record["score_semantics"] == "cosine_not_probability"


def test_low_confidence_order_is_cosine_then_candidate_id_and_selected_only() -> None:
    rows = [
        _candidate("c-z", 1, 10, 0.30, selected=True),
        _candidate("c-a", 2, 11, 0.30, selected=True),
        _candidate("c-mid", 3, 12, 0.49, selected=True),
        _candidate("c-cutoff", 4, 13, 0.50, selected=True),
        _candidate("c-not-selected", 5, 14, 0.10),
    ]
    metrics = compute_directional_candidate_metrics(rows[::-1])
    selected = select_low_confidence_links(metrics, threshold=0.5)
    assert [item.candidate_id for item in selected] == ["c-a", "c-z", "c-mid"]
    assert all(item.selected_by_solver for item in selected)


def test_contact_sheet_uses_time_representatives_and_link_endpoints_once() -> None:
    detections = [
        _detection(1, 1, 0),
        _detection(2, 1, 1),
        _detection(3, 2, 10),
        _detection(4, 2, 11),
        _detection(5, 3, 30),
        _detection(6, 3, 31),
    ]
    weakest = _metric(
        "weak", 1, 2, 0.2, gap=9.0, source_frame=1, target_frame=10
    )
    longest = _metric(
        "long", 2, 3, 0.4, gap=19.0, source_frame=11, target_frame=30
    )

    plan = build_contact_sheet_plan("G0001", detections[::-1], [longest, weakest])

    assert tuple(slot.role for slot in plan.slots) == CONTACT_SHEET_ROLES
    assert plan.weakest_link == weakest
    assert plan.longest_gap_link == longest
    sources = {
        slot.role: None if slot.source is None else slot.source.det_id
        for slot in plan.slots
    }
    assert sources == {
        "start": 1,
        "middle": 4,
        "end": 6,
        "weakest_link_source": 2,
        "weakest_link_target": 3,
        "longest_gap_source": None,  # Same real crop as middle; never copied.
        "longest_gap_target": 5,
    }
    duplicate = plan.slots[5]
    assert duplicate.deduplicated_to_role == "middle"
    requested = [request.source.det_id for request in plan.crop_requests()]
    assert requested == [1, 4, 6, 2, 3, 5]
    assert len(requested) == len(set(requested))


def test_contact_sheet_with_one_detection_leaves_six_blank_slots() -> None:
    plan = build_contact_sheet_plan("G0001", [_detection(-9, 4, 5)], [])
    assert plan.slots[0].source is not None
    assert plan.slots[0].source.det_id == -9
    assert all(slot.source is None for slot in plan.slots[1:])
    assert len(plan.items) == 1


def test_low_confidence_crop_plan_uses_last_six_and_first_six() -> None:
    source = [_detection(index, 1, index) for index in range(1, 9)]
    target = [_detection(index + 20, 2, index + 20) for index in range(1, 9)]
    link = _metric(
        "low", 1, 2, 0.2, gap=13.0, source_frame=8, target_frame=21
    )
    requests = build_low_confidence_crop_requests(
        [link], target[::-1] + source[::-1], crops_per_endpoint=6
    )
    assert [item.source.det_id for item in requests[:6]] == [3, 4, 5, 6, 7, 8]
    assert [item.source.det_id for item in requests[6:]] == [21, 22, 23, 24, 25, 26]
    assert requests[0].use.role == "source_tail_01"
    assert requests[-1].use.role == "target_head_06"


def test_crop_request_index_deduplicates_decode_crop_and_use() -> None:
    first = CropSource(-7, "GX040006", 10, 1.0, 2.0, 11.0, 22.0)
    second = CropSource(8, "GX040006", 10, 20.0, 2.0, 30.0, 22.0)
    use_a = CropUse("contact:G0001", "start", "start")
    use_b = CropUse("html:c1", "source_tail_01", "source")
    requests = [
        CropRequest(second, use_b),
        CropRequest(first, use_b),
        CropRequest(first, use_a),
        CropRequest(first, use_a),
    ]

    index = build_crop_request_index(requests)

    assert index.num_decode_frames == 1
    assert index.num_unique_crops == 2
    assert index.num_uses == 3
    assert [crop.source.det_id for crop in index.frames[0].crops] == [-7, 8]
    assert index.uses_for_det_id(-7) == tuple(sorted((use_a, use_b)))


def test_crop_request_index_rejects_conflicting_provenance_for_det_id() -> None:
    use = CropUse("contact:G0001", "start", "start")
    with pytest.raises(ContractError, match="conflicting crop provenance"):
        build_crop_request_index(
            [
                CropRequest(
                    CropSource(7, "GX040006", 10, 1.0, 2.0, 11.0, 22.0),
                    use,
                ),
                CropRequest(
                    CropSource(7, "GX040006", 11, 1.0, 2.0, 11.0, 22.0),
                    use,
                ),
            ]
        )


def test_candidate_parser_rejects_probability_like_or_identity_upgrades() -> None:
    row = _candidate("bad", 1, 10, 0.2, selected=True)
    row["id_status"] = "confirmed"
    with pytest.raises(ContractError, match="must remain forced_provisional"):
        compute_directional_candidate_metrics([row])

    row = _candidate("bad-score", 1, 10, 1.1)
    with pytest.raises(ContractError, match=r"appearance_cosine must be in \[-1, 1\]"):
        compute_directional_candidate_metrics([row])
