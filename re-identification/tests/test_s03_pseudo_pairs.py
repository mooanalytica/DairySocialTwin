from __future__ import annotations

from dataclasses import fields, replace
from pathlib import Path

import numpy as np
import pytest

from cowtrack.config import ContractError
from cowtrack.linking.config import load_link_calibration_config
from cowtrack.linking.features import feature_schema
from cowtrack.linking.pseudo_pairs import (
    CalibrationInput,
    assign_parent_splits,
    generate_pseudo_pairs,
    pseudo_pairs_as_rows,
)
from cowtrack.schemas.calibration import PSEUDO_PAIRS_SCHEMA
from cowtrack.stages.s03_calibrate import _pair_table


REPOSITORY_CONFIG = (
    Path(__file__).resolve().parents[1] / "configs" / "s03_calibration.yaml"
)


def _config():
    config, _, _ = load_link_calibration_config(REPOSITORY_CONFIG)
    return replace(
        config,
        short_gap_targets_sec=(4.0,),
        long_gap_targets_sec=(),
        gap_target_tolerance_sec=1e-6,
        hard_negative_appearance_top_k_per_positive=1,
        hard_negative_max_candidates_per_positive=2,
    )


def _unit_angle(degrees: float) -> np.ndarray:
    radians = np.deg2rad(degrees)
    return np.asarray([np.cos(radians), np.sin(radians)], dtype=np.float32)


def _input() -> CalibrationInput:
    parent_ids = np.arange(5, dtype=np.int64)
    parent_count = len(parent_ids)
    detections_per_parent = 10

    det_micro: list[int] = []
    det_order: list[int] = []
    det_ids: list[int] = []
    det_frames: list[int] = []
    det_times: list[float] = []
    det_iou: list[float] = []
    sample_ids: list[int] = []
    sample_micro: list[int] = []
    sample_det: list[int] = []
    sample_iou: list[float] = []
    embedding_rows: list[int] = []
    embeddings: list[np.ndarray] = []
    angles = (0.0, 5.0, 80.0, 150.0, 220.0)

    for micro_id in parent_ids:
        for order in range(detections_per_parent):
            det_id = int(micro_id) * 100 + order
            overlap = 0.40 if int(micro_id) == 2 and order == 8 else 0.0
            if int(micro_id) <= 2:
                frame = order
                time_sec = float(order)
            elif int(micro_id) == 3:
                frame = 30 + order
                time_sec = 30.0 + 0.5 * order
            else:
                frame = 40 + order
                time_sec = 40.0 + 0.5 * order
            det_micro.append(int(micro_id))
            det_order.append(order)
            det_ids.append(det_id)
            det_frames.append(frame)
            det_times.append(time_sec)
            det_iou.append(overlap)
            sample_id = len(sample_ids)
            sample_ids.append(sample_id)
            sample_micro.append(int(micro_id))
            sample_det.append(det_id)
            sample_iou.append(overlap)
            embedding_rows.append(sample_id)
            embeddings.append(_unit_angle(angles[int(micro_id)]))

    det_count = len(det_ids)
    sample_count = len(sample_ids)
    return CalibrationInput(
        timeline_clip_ids=np.asarray(["clip-a"], dtype=object),
        timeline_clip_start_time_sec=np.asarray([0.0], dtype=np.float64),
        timeline_clip_end_time_sec=np.asarray([50.0], dtype=np.float64),
        parent_micro_ids=parent_ids,
        parent_status=np.asarray(["valid"] * parent_count, dtype=object),
        parent_num_detections=np.full(
            parent_count, detections_per_parent, dtype=np.int32
        ),
        parent_local_purity_score=np.ones(parent_count, dtype=np.float32),
        parent_bidirectional_agreement=np.ones(parent_count, dtype=np.float32),
        parent_internal_cosine_p10=np.ones(parent_count, dtype=np.float32),
        det_ids=np.asarray(det_ids, dtype=np.int64),
        det_micro_ids=np.asarray(det_micro, dtype=np.int64),
        det_order_in_micro=np.asarray(det_order, dtype=np.int32),
        det_clip_ids=np.asarray(["clip-a"] * det_count, dtype=object),
        det_global_frames=np.asarray(det_frames, dtype=np.int64),
        det_global_time_sec=np.asarray(det_times, dtype=np.float64),
        det_cx_norm=np.full(det_count, 0.4, dtype=np.float32),
        det_cy_norm=np.full(det_count, 0.5, dtype=np.float32),
        det_w_norm=np.full(det_count, 0.2, dtype=np.float32),
        det_h_norm=np.full(det_count, 0.3, dtype=np.float32),
        det_other_bbox_max_iou=np.asarray(det_iou, dtype=np.float32),
        det_boundary_distance=np.full(det_count, 0.2, dtype=np.float32),
        det_review_excluded=np.zeros(det_count, dtype=np.bool_),
        sample_ids=np.asarray(sample_ids, dtype=np.int64),
        sample_micro_ids=np.asarray(sample_micro, dtype=np.int64),
        sample_det_ids=np.asarray(sample_det, dtype=np.int64),
        sample_crop_quality=np.ones(sample_count, dtype=np.float32),
        sample_other_bbox_max_iou=np.asarray(sample_iou, dtype=np.float32),
        sample_s02_inlier=np.ones(sample_count, dtype=np.bool_),
        sample_embedding_rows=np.asarray(embedding_rows, dtype=np.int64),
        embeddings=np.stack(embeddings),
    )


def _shuffle_input(data: CalibrationInput) -> CalibrationInput:
    parent_order = np.asarray([3, 0, 4, 1, 2])
    det_order = np.random.default_rng(123).permutation(len(data.det_ids))
    sample_order = np.random.default_rng(456).permutation(len(data.sample_ids))
    values: dict[str, object] = {}
    for field in fields(CalibrationInput):
        value = getattr(data, field.name)
        if field.name == "embeddings":
            values[field.name] = value
        elif field.name.startswith("timeline_"):
            values[field.name] = value
        elif field.name.startswith("parent_"):
            values[field.name] = value[parent_order]
        elif field.name.startswith("det_"):
            values[field.name] = value[det_order]
        elif field.name.startswith("sample_"):
            values[field.name] = value[sample_order]
        else:  # pragma: no cover - protects this helper when fields are added
            raise AssertionError(field.name)
    return CalibrationInput(**values)  # type: ignore[arg-type]


def test_parent_split_is_time_blocked_and_cross_clip_parent_stays_one_group() -> None:
    data = _input()
    assignments = assign_parent_splits(data, _config())

    assert assignments == {
        0: "train",
        1: "train",
        2: "train",
        3: "calibration",
        4: "audit",
    }

    # One parent appearing in two clips still has exactly one group assignment.
    clips = data.det_clip_ids.copy()
    parent_zero = data.det_micro_ids == 0
    clips[np.flatnonzero(parent_zero)[5:]] = "clip-b"
    cross_clip = replace(
        data,
        det_clip_ids=clips,
        timeline_clip_ids=np.asarray(["clip-a", "clip-b"], dtype=object),
        timeline_clip_start_time_sec=np.asarray([0.0, 5.0], dtype=np.float64),
        timeline_clip_end_time_sec=np.asarray([50.0, 10.0], dtype=np.float64),
    )
    cross_assignments = assign_parent_splits(cross_clip, _config())
    assert set(cross_assignments) == set(assignments)
    assert isinstance(cross_assignments[0], str)
    assert cross_assignments[0] == "audit"  # held-out precedence is conservative


def test_split_uses_contiguous_time_ranges_not_parent_count_rank() -> None:
    data = _input()
    times = data.det_global_time_sec.copy()
    frames = data.det_global_frames.copy()
    for micro_id, start in enumerate((0.0, 10.0, 20.0, 30.0, 85.0)):
        positions = np.flatnonzero(data.det_micro_ids == micro_id)
        times[positions] = start + np.arange(len(positions), dtype=np.float64)
        frames[positions] = int(start) + np.arange(len(positions), dtype=np.int64)
    uneven = replace(
        data,
        det_global_time_sec=times,
        det_global_frames=frames,
        timeline_clip_start_time_sec=np.asarray([0.0]),
        timeline_clip_end_time_sec=np.asarray([100.0]),
    )
    assignments = assign_parent_splits(uneven, _config())
    assert assignments[3] == "train"
    assert assignments[4] == "audit"

    positions = np.flatnonzero(data.det_micro_ids == 3)
    crossing_times = times.copy()
    crossing_times[positions] = np.linspace(59.0, 61.0, len(positions))
    crossing = replace(uneven, det_global_time_sec=crossing_times)
    assert assign_parent_splits(crossing, _config())[3] == "calibration"


def test_pairs_are_leakage_safe_positive_gap_and_fully_auditable() -> None:
    data = _input()
    config = _config()
    assignments = assign_parent_splits(data, config)
    pairs = generate_pseudo_pairs(data, config)

    assert pairs
    positives = {pair.candidate_group_id: pair for pair in pairs if pair.label}
    negatives = [pair for pair in pairs if not pair.label]
    assert positives
    assert negatives

    det_lookup = {
        int(det_id): row for row, det_id in enumerate(data.det_ids)
    }
    for pair in pairs:
        assert pair.features["gap_sec"] > 0.0
        assert tuple(pair.features) == feature_schema(pair.mode)
        assert pair.appearance_present is True
        assert pair.parent_group_id == f"parent-{pair.parent_micro_id}"
        assert pair.split == assignments[pair.parent_micro_id]
        assert set(pair.source_gallery.sample_ids).isdisjoint(
            pair.target_gallery.sample_ids
        )
        assert set(pair.source_gallery.det_ids).isdisjoint(pair.target_gallery.det_ids)
        assert set(pair.source_gallery.embedding_rows).isdisjoint(
            pair.target_gallery.embedding_rows
        )
        row = pair.as_row()
        assert row["parent_micro_id"] == pair.parent_micro_id
        assert row["candidate_group_id"] == pair.candidate_group_id
        assert row["source_gallery_det_ids"]
        assert row["target_gallery_det_ids"]
        assert row["stratum"] == pair.stratum
        assert row["high_overlap"] == pair.high_overlap

    for negative in negatives:
        positive = positives[negative.candidate_group_id]
        assert negative.source == positive.source
        assert negative.target.micro_id != positive.target.micro_id
        assert negative.split == assignments[negative.target.micro_id]
        positive_row = det_lookup[positive.target.start_det_id]
        negative_row = det_lookup[negative.target.start_det_id]
        assert data.det_clip_ids[positive_row] == data.det_clip_ids[negative_row]
        assert data.det_global_frames[positive_row] == data.det_global_frames[negative_row]
        assert negative.features["gap_sec"] == positive.features["gap_sec"]


def test_hardest_top_k_keeps_high_overlap_stress_outside_top_k() -> None:
    data = _input()
    pairs = generate_pseudo_pairs(data, _config())
    positive = next(
        pair for pair in pairs if pair.label and pair.parent_micro_id == 0
    )
    negatives = [
        pair
        for pair in pairs
        if not pair.label and pair.candidate_group_id == positive.candidate_group_id
    ]

    # Micro 1 is appearance-hardest and occupies top-1.  Micro 2 is much less
    # similar but has a non-endpoint contaminated sample in its target segment,
    # so it is retained as a separate stress stratum instead of disappearing.
    assert [pair.target.micro_id for pair in negatives] == [1, 2]
    assert negatives[0].hard_negative_rank == 1
    assert negatives[0].stratum == "hard_negative"
    assert negatives[1].hard_negative_rank == 2
    assert negatives[1].high_overlap is True
    assert negatives[1].stratum == "high_overlap_stress"
    target_start = {
        int(det_id): row for row, det_id in enumerate(data.det_ids)
    }[negatives[1].target.start_det_id]
    assert data.det_other_bbox_max_iou[target_start] == 0.0
    assert negatives[1].target_gallery.max_other_bbox_iou == pytest.approx(0.4)


def test_negative_target_needs_clean_gallery_not_positive_parent_eligibility() -> None:
    data = _input()
    status = data.parent_status.copy()
    status[1] = "short_fragment"
    pairs = generate_pseudo_pairs(replace(data, parent_status=status), _config())
    positive = next(
        pair for pair in pairs if pair.label and pair.parent_micro_id == 0
    )
    negative_targets = {
        pair.target.micro_id
        for pair in pairs
        if not pair.label and pair.candidate_group_id == positive.candidate_group_id
    }

    # Simultaneity proves D differs from B.  D need not qualify as a positive
    # parent, but it still has to produce a valid independent clean gallery.
    assert 1 in negative_targets


def test_positive_group_without_same_split_negative_is_not_emitted() -> None:
    pairs = generate_pseudo_pairs(_input(), _config())

    # Micros 3 and 4 are the sole calibration/audit groups in this tiny input,
    # so neither has an independently proven same-split negative candidate.
    assert all(pair.parent_micro_id not in {3, 4} for pair in pairs)


def test_calibration_selection_and_certification_mine_negatives_independently() -> None:
    data = _input()
    times = data.det_global_time_sec.copy()
    frames = data.det_global_frames.copy()
    audit_micro = np.flatnonzero(data.det_micro_ids == 2)
    times[audit_micro] = np.linspace(45.0, 49.0, len(audit_micro))
    frames[audit_micro] = 45 + np.arange(len(audit_micro), dtype=np.int64)
    certification_micro = np.flatnonzero(data.det_micro_ids == 4)
    times[certification_micro] = np.linspace(32.5, 38.8, len(certification_micro))
    frames[certification_micro] = 35 + np.arange(
        len(certification_micro), dtype=np.int64
    )
    separated = replace(data, det_global_time_sec=times, det_global_frames=frames)
    assignments = assign_parent_splits(separated, _config())
    assert assignments[3] == assignments[4] == "calibration"

    config = replace(
        _config(), short_gap_targets_sec=(1.5,), gap_target_tolerance_sec=1e-6
    )
    pairs = generate_pseudo_pairs(separated, config)
    # Parent 3 is threshold-selection; the simultaneous D is certification.
    # Public split equality is insufficient: pre-mining partitions must match.
    assert all(pair.parent_micro_id != 3 for pair in pairs)


def test_parent_eligibility_and_review_endpoints_fail_closed() -> None:
    data = _input()
    config = _config()
    bad_inputs = (
        replace(
            data,
            parent_status=np.asarray(
                ["short_fragment", "valid", "valid", "valid", "valid"],
                dtype=object,
            ),
        ),
        replace(
            data,
            parent_local_purity_score=np.asarray(
                [0.5, 1.0, 1.0, 1.0, 1.0], dtype=np.float32
            ),
        ),
        replace(
            data,
            parent_bidirectional_agreement=np.asarray(
                [0.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32
            ),
        ),
        replace(
            data,
            parent_internal_cosine_p10=np.asarray(
                [0.69, 1.0, 1.0, 1.0, 1.0], dtype=np.float32
            ),
        ),
    )
    for bad in bad_inputs:
        assert all(
            pair.parent_micro_id != 0 for pair in generate_pseudo_pairs(bad, config)
        )

    review = data.det_review_excluded.copy()
    review[data.det_micro_ids == 0] = True
    excluded = replace(data, det_review_excluded=review)
    assert all(
        pair.parent_micro_id != 0
        for pair in generate_pseudo_pairs(excluded, config)
    )


def test_row_shuffle_is_exactly_invariant_and_forbidden_identity_is_absent() -> None:
    data = _input()
    config = _config()
    expected = pseudo_pairs_as_rows(generate_pseudo_pairs(data, config))
    actual = pseudo_pairs_as_rows(generate_pseudo_pairs(_shuffle_input(data), config))

    assert actual == expected
    input_fields = {field.name for field in fields(CalibrationInput)}
    assert not any("legacy" in name or "keypoint" in name for name in input_fields)
    assert not any(
        "review_case" in key or "review_reason" in key
        for row in actual
        for key in row
    )


def test_insufficient_clean_samples_emit_no_pair_instead_of_missing_sentinel() -> None:
    data = _input()
    inlier = data.sample_s02_inlier.copy()
    # Leave only two usable samples for parent zero, below the fixed side-local
    # minimum.  No cosine=0/NaN/motion-only fallback may be emitted.
    rows = np.flatnonzero(data.sample_micro_ids == 0)
    inlier[rows[2:]] = False
    insufficient = replace(data, sample_s02_inlier=inlier)
    clean_gallery_status: dict[int, bool] = {}
    pairs = generate_pseudo_pairs(
        insufficient, _config(), clean_gallery_status=clean_gallery_status
    )

    assert all(pair.parent_micro_id != 0 for pair in pairs)
    assert all(np.isfinite(list(pair.features.values())).all() for pair in pairs)
    assert clean_gallery_status[0] is False


def test_real_pseudo_rows_fit_the_persisted_arrow_schema() -> None:
    rows = pseudo_pairs_as_rows(generate_pseudo_pairs(_input(), _config()))
    table = _pair_table(rows)
    assert table.schema == PSEUDO_PAIRS_SCHEMA
    assert table.num_rows == len(rows)


def test_optional_progress_logger_reports_both_phases_and_final_count() -> None:
    messages: list[str] = []
    pairs = generate_pseudo_pairs(
        _input(),
        _config(),
        logger=messages.append,
        progress_interval_sec=1e-9,
    )

    assert any("pseudo-positive mining complete" in message for message in messages)
    assert any("group mining complete" in message for message in messages)
    assert messages[-1] == f"[s03] pseudo-pair construction complete: {len(pairs):,} rows"

    with pytest.raises(ContractError, match="progress interval must be positive"):
        generate_pseudo_pairs(
            _input(), _config(), logger=messages.append, progress_interval_sec=0.0
        )
