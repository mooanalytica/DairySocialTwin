from __future__ import annotations

from pathlib import Path
import os

import cv2
import numpy as np
import pytest

from cowtrack.config import ContractError
from cowtrack.qa.s04_review import (
    EXPECTED_FRAME_PERIOD_SEC,
    CandidateEndpoint,
    LoadedS04ReviewData,
    RawFrameReader,
    RenderCase,
    _effective_config_output_record,
    _fingerprint,
    _frame_role_boxes,
    _render_cases,
    _validate_proposal_candidate_join,
    _verify_inputs_unchanged,
)
from cowtrack.qa.s04_review_config import load_s04_review_config
from cowtrack.qa.s04_review_plan import (
    ProposalRecord,
    S04ReviewCase,
    S04ReviewPlan,
)


class _FakeCapture:
    def __init__(self, color: int) -> None:
        self.color = color
        self.position = 0
        self.released = False

    def set(self, prop: int, value: float) -> bool:
        assert prop == cv2.CAP_PROP_POS_FRAMES
        self.position = int(value)
        return True

    def read(self) -> tuple[bool, np.ndarray]:
        frame = np.full((90, 160, 3), self.color + self.position, dtype=np.uint8)
        self.position += 1
        return True, frame

    def get(self, prop: int) -> float:
        if prop == cv2.CAP_PROP_POS_FRAMES:
            return float(self.position)
        if prop == cv2.CAP_PROP_PTS:
            return float(self.position - 1)
        if prop == cv2.CAP_PROP_POS_MSEC:
            return float(self.position - 1) * 1000.0 * 1001.0 / 30000.0
        raise AssertionError(prop)

    def release(self) -> None:
        self.released = True


def test_raw_reader_switches_clips_without_inserting_frames() -> None:
    captures: dict[str, _FakeCapture] = {}

    def factory(path: Path) -> _FakeCapture:
        capture = _FakeCapture(10 if path.name == "a.mp4" else 20)
        captures[path.name] = capture
        return capture

    with RawFrameReader(
        {"a": Path("a.mp4"), "b": Path("b.mp4")},
        raw_width=160,
        raw_height=90,
        capture_factory=factory,
    ) as reader:
        a0 = reader.read("a", 0, 0.0)
        a1 = reader.read("a", 1, 1001.0 / 30000.0)
        b0 = reader.read("b", 0, 0.0)

    assert int(a0[0, 0, 0]) == 10
    assert int(a1[0, 0, 0]) == 11
    assert int(b0[0, 0, 0]) == 20
    assert captures["a.mp4"].released is True
    assert captures["b.mp4"].released is True


def _proposal(
    proposal_id: str, edge_id: str, source: int, target: int
) -> ProposalRecord:
    return ProposalRecord(
        proposal_id=proposal_id,
        edge_id=edge_id,
        source_micro_id=source,
        target_micro_id=target,
        probability=0.9,
        high_overlap=False,
        source_rank=1,
        target_rank=1,
        conflict_degree=1,
        conflict_group_id="group",
        conflict_group_edge_count=2,
        conflict_group_node_count=3,
        review_status="pending",
    )


def test_purple_is_limited_to_real_competing_micro_detections_inside_gap() -> None:
    focal = _proposal("p0", "e0", 1, 2)
    competing = _proposal("p1", "e1", 1, 3)
    case = S04ReviewCase(focal, "all", 1)
    plan = S04ReviewPlan(
        proposals=(focal, competing),
        cases=(case,),
        random_seed=20260710,
        top_count=15,
        bottom_count=15,
        random_count=20,
    )
    endpoint = CandidateEndpoint(
        edge_id="e0",
        source_micro_id=1,
        target_micro_id=2,
        source_start_global_frame=0,
        source_end_global_frame=0,
        target_start_global_frame=2,
        target_end_global_frame=2,
        source_start_time_sec=0.0,
        source_end_time_sec=0.0,
        target_start_time_sec=2.0,
        target_end_time_sec=2.0,
        source_end_clip_id="clip",
        target_start_clip_id="clip",
        probability=0.9,
        high_overlap=False,
    )
    boxes = np.asarray(
        [[1, 1, 5, 5], [11, 1, 15, 5], [21, 1, 25, 5], [31, 1, 35, 5]]
        * 2,
        dtype=np.float64,
    )
    data = LoadedS04ReviewData(
        frames={"global_frame": np.arange(3)},
        detections={
            "x1": boxes[:, 0],
            "y1": boxes[:, 1],
            "x2": boxes[:, 2],
            "y2": boxes[:, 3],
        },
        micro_id_by_detection_position=np.asarray([1, 2, 3, 9] * 2),
        valid_detection_positions_by_frame=np.arange(8),
        valid_detection_frame_offsets=np.asarray([0, 4, 8, 8]),
        candidates_by_edge_id={"e0": endpoint},
        video_paths={},
        plan=plan,
        input_fingerprints=(),
    )
    render_case = RenderCase(
        case=case,
        endpoint=endpoint,
        intermediate_micro_ids=plan.intermediate_micro_ids(case),
        start_global_frame=0,
        end_global_frame=2,
        output_relative_path="videos/case.mp4",
    )

    outside = _frame_role_boxes(data, render_case, 0)
    inside = _frame_role_boxes(data, render_case, 1)

    assert [len(role) for role in outside] == [2, 1, 1, 0]
    assert [len(role) for role in inside] == [1, 1, 1, 1]
    np.testing.assert_array_equal(inside[3][0], [21, 1, 25, 5])


def test_review_window_is_two_raw_seconds_not_clamped_to_short_micros() -> None:
    proposal = _proposal("p0", "e0", 1, 2)
    case = S04ReviewCase(proposal, "all", 1)
    plan = S04ReviewPlan(
        proposals=(proposal,),
        cases=(case,),
        random_seed=20260710,
        top_count=15,
        bottom_count=15,
        random_count=20,
    )
    endpoint = CandidateEndpoint(
        edge_id="e0",
        source_micro_id=1,
        target_micro_id=2,
        source_start_global_frame=99,
        source_end_global_frame=100,
        target_start_global_frame=110,
        target_end_global_frame=111,
        source_start_time_sec=99 * EXPECTED_FRAME_PERIOD_SEC,
        source_end_time_sec=100 * EXPECTED_FRAME_PERIOD_SEC,
        target_start_time_sec=110 * EXPECTED_FRAME_PERIOD_SEC,
        target_end_time_sec=111 * EXPECTED_FRAME_PERIOD_SEC,
        source_end_clip_id="clip",
        target_start_clip_id="clip",
        probability=0.9,
        high_overlap=False,
    )
    data = LoadedS04ReviewData(
        frames={
            "global_frame": np.arange(220),
            "global_time_sec": np.arange(220) * EXPECTED_FRAME_PERIOD_SEC,
        },
        detections={},
        micro_id_by_detection_position=np.empty(0, dtype=np.int64),
        valid_detection_positions_by_frame=np.empty(0, dtype=np.int64),
        valid_detection_frame_offsets=np.zeros(221, dtype=np.int64),
        candidates_by_edge_id={"e0": endpoint},
        video_paths={},
        plan=plan,
        input_fingerprints=(),
    )
    config_path = Path(__file__).parents[1] / "configs" / "s04_review.yaml"
    config, _, _ = load_s04_review_config(config_path)

    rendered = _render_cases(data, config)[0]

    assert rendered.start_global_frame < endpoint.source_start_global_frame
    assert rendered.end_global_frame > endpoint.target_end_global_frame
    assert rendered.start_global_frame == 41
    assert rendered.end_global_frame == 169


def test_candidate_proposal_join_is_exact_and_bijective() -> None:
    proposal = _proposal("p0", "e0", 1, 2)
    case = S04ReviewCase(proposal, "all", 1)
    plan = S04ReviewPlan(
        proposals=(proposal,),
        cases=(case,),
        random_seed=20260710,
        top_count=15,
        bottom_count=15,
        random_count=20,
    )
    endpoint = CandidateEndpoint(
        edge_id="e0",
        source_micro_id=1,
        target_micro_id=2,
        source_start_global_frame=0,
        source_end_global_frame=1,
        target_start_global_frame=2,
        target_end_global_frame=3,
        source_start_time_sec=0.0,
        source_end_time_sec=1.0,
        target_start_time_sec=2.0,
        target_end_time_sec=3.0,
        source_end_clip_id="clip",
        target_start_clip_id="clip",
        probability=0.9,
        high_overlap=False,
    )
    _validate_proposal_candidate_join(plan, {"e0": endpoint})

    with pytest.raises(ContractError, match="not identical"):
        _validate_proposal_candidate_join(plan, {"e0": endpoint, "extra": endpoint})

    changed_probability = CandidateEndpoint(
        **{**endpoint.__dict__, "probability": np.nextafter(0.9, 1.0)}
    )
    with pytest.raises(ContractError, match="differs from candidate"):
        _validate_proposal_candidate_join(plan, {"e0": changed_probability})


def test_effective_config_output_is_required_and_immutable(tmp_path: Path) -> None:
    payload = {"pipeline": {"schema_version": "1.0"}}
    path = tmp_path / "effective_config.json"
    path.write_text('{"pipeline":{"schema_version":"1.0"}}\n', encoding="utf-8")

    record = _effective_config_output_record(tmp_path, payload)
    assert record["path"] == "effective_config.json"
    assert len(record["sha256"]) == 64

    path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ContractError, match="changed"):
        _effective_config_output_record(tmp_path, payload)


def test_end_verification_detects_same_size_same_mtime_content_change(
    tmp_path: Path,
) -> None:
    path = tmp_path / "input.bin"
    path.write_bytes(b"abc")
    recorded = _fingerprint(path)
    path.write_bytes(b"xyz")
    os.utime(path, ns=(recorded["mtime_ns"], recorded["mtime_ns"]))
    data = LoadedS04ReviewData(
        frames={},
        detections={},
        micro_id_by_detection_position=np.empty(0, dtype=np.int64),
        valid_detection_positions_by_frame=np.empty(0, dtype=np.int64),
        valid_detection_frame_offsets=np.empty(0, dtype=np.int64),
        candidates_by_edge_id={},
        video_paths={},
        plan=S04ReviewPlan((), (), 20260710, 15, 15, 20),
        input_fingerprints=(recorded,),
    )

    with pytest.raises(ContractError, match="sha256"):
        _verify_inputs_unchanged(
            data, progress_interval_sec=10.0, logger=lambda _: None
        )
