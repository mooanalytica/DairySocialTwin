from __future__ import annotations

import json
from types import MappingProxyType

import numpy as np
import pytest

from cowtrack.config import ContractError
from cowtrack.linking.features import LONG_FEATURE_SCHEMA
from cowtrack.linking.pseudo_pairs import CalibrationInput
from cowtrack.linking.runtime import (
    CanonicalDetectionMetadata,
    ProductionInputBundle,
)
from cowtrack.linking.stable_long_pairs import (
    StableLongInput,
    StablePathFeatureStore,
    assign_stable_partitions,
    generate_stable_long_pairs,
    stable_long_pairs_as_rows,
)


def _stable_input(
    *, shuffle_rows: bool = False, positive_target_start: float = 20.0
) -> StableLongInput:
    # Stable 0 is an explicit two-micro S04 path.  Stable 1 is a second long
    # path at the same frames so both orientations prove the same unordered
    # negative group.  Stable 2 is a clean future production target.  The last
    # three paths exercise held-out-wins partition boundaries.
    specifications = (
        (10, (1.0, 2.0, 3.0), True, (1.0, 0.0, 0.0)),
        (
            16,
            tuple(positive_target_start + offset for offset in (0.0, 1.0, 2.0)),
            True,
            (1.0, 0.0, 0.0),
        ),
        (
            11,
            (
                1.0,
                2.0,
                3.0,
                positive_target_start,
                positive_target_start + 1.0,
                positive_target_start + 2.0,
            ),
            True,
            (0.8, 0.6, 0.0),
        ),
        (12, (40.0, 41.0, 42.0), True, (0.0, 0.0, 1.0)),
        (13, (55.0, 61.0), False, (0.0, 1.0, 0.0)),
        (14, (65.0, 75.0), False, (0.0, 1.0, 0.0)),
        (15, (79.0, 85.0), False, (0.0, 1.0, 0.0)),
    )
    detections: list[dict[str, object]] = []
    next_det_id = 1_000
    for micro_id, times, sampled, identity in specifications:
        for order, time_sec in enumerate(times):
            detections.append(
                {
                    "det_id": next_det_id,
                    "micro_id": micro_id,
                    "order": order,
                    "clip": "clip-a",
                    "frame": int(round(time_sec * 10.0)),
                    "time": time_sec,
                    "sampled": sampled,
                    "identity": identity,
                }
            )
            next_det_id += 1

    det_permutation = np.arange(len(detections))
    if shuffle_rows:
        det_permutation = np.random.default_rng(9481).permutation(det_permutation)
    ordered = [detections[int(index)] for index in det_permutation]

    def det_array(name: str, dtype: object) -> np.ndarray:
        return np.asarray([row[name] for row in ordered], dtype=dtype)

    det_ids = det_array("det_id", np.int64)
    det_micro = det_array("micro_id", np.int64)
    det_order = det_array("order", np.int64)
    det_clips = det_array("clip", object)
    det_frames = det_array("frame", np.int64)
    det_times = det_array("time", np.float64)
    count = len(ordered)
    micro_ids = np.asarray([10, 16, 11, 12, 13, 14, 15], dtype=np.int64)
    micro_paths: dict[int, np.ndarray] = {}
    for micro_id in micro_ids:
        positions = np.flatnonzero(det_micro == micro_id)
        positions = positions[np.argsort(det_order[positions], kind="stable")]
        micro_paths[int(micro_id)] = positions

    original_sample_rows = [row for row in detections if bool(row["sampled"])]
    embeddings = np.asarray(
        [row["identity"] for row in original_sample_rows], dtype=np.float32
    )
    sample_records = [
        {
            "sample_id": index,
            "embedding_row": index,
            "det_id": row["det_id"],
            "micro_id": row["micro_id"],
        }
        for index, row in enumerate(original_sample_rows)
    ]
    if shuffle_rows:
        order = np.random.default_rng(329).permutation(len(sample_records))
        sample_records = [sample_records[int(index)] for index in order]

    def sample_array(name: str) -> np.ndarray:
        return np.asarray([row[name] for row in sample_records], dtype=np.int64)

    sample_count = len(sample_records)
    parent_counts = np.asarray(
        [len(micro_paths[int(micro_id)]) for micro_id in micro_ids],
        dtype=np.int64,
    )
    calibration = CalibrationInput(
        timeline_clip_ids=np.asarray(["clip-a"], dtype=object),
        timeline_clip_start_time_sec=np.asarray([0.0], dtype=np.float64),
        timeline_clip_end_time_sec=np.asarray([100.0], dtype=np.float64),
        parent_micro_ids=micro_ids,
        parent_status=np.asarray(["valid"] * len(micro_ids), dtype=object),
        parent_num_detections=parent_counts,
        parent_local_purity_score=np.ones(len(micro_ids), dtype=np.float64),
        parent_bidirectional_agreement=np.ones(len(micro_ids), dtype=np.float64),
        parent_internal_cosine_p10=np.ones(len(micro_ids), dtype=np.float64),
        det_ids=det_ids,
        det_micro_ids=det_micro,
        det_order_in_micro=det_order,
        det_clip_ids=det_clips,
        det_global_frames=det_frames,
        det_global_time_sec=det_times,
        det_cx_norm=np.full(count, 0.5, dtype=np.float64),
        det_cy_norm=np.full(count, 0.5, dtype=np.float64),
        det_w_norm=np.full(count, 0.2, dtype=np.float64),
        det_h_norm=np.full(count, 0.2, dtype=np.float64),
        det_other_bbox_max_iou=np.zeros(count, dtype=np.float32),
        det_boundary_distance=np.ones(count, dtype=np.float32),
        det_review_excluded=np.zeros(count, dtype=np.bool_),
        sample_ids=sample_array("sample_id"),
        sample_micro_ids=sample_array("micro_id"),
        sample_det_ids=sample_array("det_id"),
        sample_crop_quality=np.ones(sample_count, dtype=np.float32),
        sample_other_bbox_max_iou=np.zeros(sample_count, dtype=np.float32),
        sample_s02_inlier=np.ones(sample_count, dtype=np.bool_),
        sample_embedding_rows=sample_array("embedding_row"),
        embeddings=embeddings,
    )
    metadata = CanonicalDetectionMetadata(
        det_ids=det_ids,
        micro_ids=det_micro,
        order_in_micro=det_order,
        clip_ids=det_clips,
        global_frames=det_frames,
        global_time_sec=det_times,
        x1=np.zeros(count, dtype=np.float64),
        y1=np.zeros(count, dtype=np.float64),
        x2=np.ones(count, dtype=np.float64),
        y2=np.ones(count, dtype=np.float64),
        other_bbox_max_iou=calibration.det_other_bbox_max_iou,
        boundary_distance=calibration.det_boundary_distance,
        review_excluded=calibration.det_review_excluded,
    )
    production = ProductionInputBundle(
        calibration_input=calibration,
        detections=metadata,
        micro_paths=MappingProxyType(micro_paths),
        endpoints=MappingProxyType({}),
        micro_appearance=MappingProxyType({}),
        appearance_report=MappingProxyType({}),
        encoder_choice=MappingProxyType({}),
        consumed_paths=(),
        input_fingerprints=(),
    )
    micro_to_stable = {
        10: 0,
        16: 0,
        11: 1,
        12: 2,
        13: 3,
        14: 4,
        15: 5,
    }
    micro_order = {micro_id: 0 for micro_id in micro_to_stable}
    micro_order[16] = 1
    return StableLongInput(
        production,
        MappingProxyType(micro_to_stable),
        MappingProxyType(micro_order),
    )


def test_partitions_are_stable_parent_held_out_wins() -> None:
    assert assign_stable_partitions(_stable_input()) == {
        0: "train",
        1: "train",
        2: "train",
        3: "threshold_selection",
        4: "certification",
        5: "audit",
    }


def test_pairs_rebuild_disjoint_galleries_and_deduplicate_unordered_negative() -> None:
    pairs = generate_stable_long_pairs(_stable_input())
    positives = [pair for pair in pairs if pair.label]
    negatives = [pair for pair in pairs if not pair.label]

    assert [pair.parent_stable_id for pair in positives] == [0, 1]
    assert len(negatives) == 1
    assert negatives[0].candidate_group_id == "partition-train:stable-pair-0-1"
    assert negatives[0].parent_stable_id == 0
    assert negatives[0].hard_negative_rank == 1
    assert negatives[0].cooccurrence_global_frame == (
        negatives[0].target.start_global_frame
    )
    assert negatives[0].cooccurrence_other_det_id == negatives[0].target.start_det_id
    assert negatives[0].cooccurrence_parent_det_id != (
        negatives[0].cooccurrence_other_det_id
    )
    assert all(
        positive.cooccurrence_global_frame is None for positive in positives
    )
    assert positives[0].source.constituent_micro_ids == (10,)
    assert positives[0].target.constituent_micro_ids == (16,)
    assert positives[0].features["gap_sec"] > 5.0
    assert set(positives[0].source_gallery.sample_ids).isdisjoint(
        positives[0].target_gallery.sample_ids
    )
    assert positives[0].source_gallery.internal_cosine_p10 >= 0.70
    assert positives[0].target_gallery.internal_cosine_p10 >= 0.70
    assert tuple(positives[0].features) == LONG_FEATURE_SCHEMA


def test_candidate_margins_are_complete_and_candidate_relative() -> None:
    pairs = generate_stable_long_pairs(_stable_input())
    assert all(
        pair.candidate_margin is not None
        and np.isfinite(pair.candidate_margin)
        for pair in pairs
    )
    positive_by_parent = {
        pair.parent_stable_id: pair for pair in pairs if pair.label
    }
    negative = next(pair for pair in pairs if not pair.label)
    positive = positive_by_parent[negative.parent_stable_id]
    assert positive.candidate_margin == pytest.approx(
        positive.features["prototype_cosine_max"]
        - negative.features["prototype_cosine_max"]
    )
    assert negative.candidate_margin == pytest.approx(
        negative.features["prototype_cosine_max"]
        - positive.features["prototype_cosine_max"]
    )
    # Stable 1 lost the unordered-pair tie-break and therefore has no proven
    # competitor in its retained parent group: explicit zero is fail-closed.
    assert positive_by_parent[1].candidate_margin == 0.0


def test_rows_are_serializable_and_include_core_and_provenance_fields() -> None:
    rows = stable_long_pairs_as_rows(generate_stable_long_pairs(_stable_input()))
    json.dumps(rows, allow_nan=False)
    row = rows[0]
    assert row["partition"] == "train"
    assert row["mode"] == "long"
    assert row["features"] == {
        name: row[name] for name in LONG_FEATURE_SCHEMA
    }
    assert row["source_sample_ids"]
    assert row["target_gallery_det_ids"]
    assert row["source_constituent_micro_ids"]


def test_detection_and_sample_row_shuffle_is_invariant() -> None:
    ordinary = stable_long_pairs_as_rows(
        generate_stable_long_pairs(_stable_input(shuffle_rows=False))
    )
    shuffled = stable_long_pairs_as_rows(
        generate_stable_long_pairs(_stable_input(shuffle_rows=True))
    )
    assert shuffled == ordinary


def test_gap_is_strictly_greater_than_five_seconds() -> None:
    pairs = generate_stable_long_pairs(
        _stable_input(positive_target_start=8.0)
    )
    assert not pairs


def test_stable_path_feature_store_is_long_only_and_fail_closed() -> None:
    store = StablePathFeatureStore(_stable_input())
    assert store.path_ids == (0, 1, 2, 3, 4, 5)

    available = store.build_pair(0, 2)
    assert available.appearance_present
    assert available.reason is None
    assert available.feature_values is not None
    assert tuple(available.feature_values) == LONG_FEATURE_SCHEMA
    assert available.feature_values["gap_sec"] > 5.0

    missing = store.build_pair(0, 3)
    assert not missing.appearance_present
    assert missing.feature_values is None
    assert missing.reason == "target_gallery_missing"

    with pytest.raises(ContractError, match="overlap, reverse, or have gap <=5s"):
        store.build_pair(0, 1)
    with pytest.raises(ContractError, match="unknown path ID"):
        store.build_pair(0, 999)
    with pytest.raises(ContractError, match="integer stable/path ID"):
        store.build_pair(True, 2)


def test_mapping_contracts_fail_closed() -> None:
    valid = _stable_input()
    missing_mapping = dict(valid.micro_to_stable)
    del missing_mapping[16]
    broken = StableLongInput(
        valid.production,
        missing_mapping,
        valid.micro_order_in_stable,
    )
    with pytest.raises(ContractError, match="exactly cover"):
        generate_stable_long_pairs(broken)

    wrong_order = dict(valid.micro_order_in_stable)
    wrong_order[16] = 2
    broken = StableLongInput(
        valid.production,
        valid.micro_to_stable,
        wrong_order,
    )
    with pytest.raises(ContractError, match="micro order is not contiguous"):
        generate_stable_long_pairs(broken)


def test_progress_logging_and_interval_contract() -> None:
    messages: list[str] = []
    generate_stable_long_pairs(
        _stable_input(), logger=messages.append, progress_interval_sec=1.0
    )
    assert any("positive mining: stable 6/6" in message for message in messages)
    assert any("negative mining: positive 2/2" in message for message in messages)
    assert messages[-1].startswith("[s05a] long pairs complete:")

    with pytest.raises(ContractError, match="progress interval"):
        generate_stable_long_pairs(_stable_input(), progress_interval_sec=0.0)
