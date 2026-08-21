from __future__ import annotations

import math
from collections.abc import Mapping
from types import MappingProxyType

import numpy as np
import pyarrow as pa
import pytest

from cowtrack.config import ContractError
from cowtrack.linking.features import LONG_FEATURE_SCHEMA
from cowtrack.linking.long_proposals import (
    build_long_proposals,
    enumerate_long_candidates,
    long_review_manifest,
    score_long_candidates,
)
from cowtrack.linking.model import LinkResult
from cowtrack.linking.pseudo_pairs import GalleryProvenance
from cowtrack.linking.stable_long_pairs import (
    StablePathPairFeatures,
    StablePathRetrievalDescriptor,
    StableSegmentProvenance,
)
from cowtrack.schemas.s05_proposals import (
    LONG_CANDIDATE_EDGES_SCHEMA,
    LONG_LINK_PROPOSALS_SCHEMA,
)


def _unit(x: float, y: float) -> tuple[float, float]:
    vector = np.asarray([x, y], dtype=np.float64)
    vector /= np.linalg.norm(vector)
    return float(vector[0]), float(vector[1])


def _gallery(stable_id: int) -> GalleryProvenance:
    base = stable_id * 10
    return GalleryProvenance(
        sample_ids=(base + 1, base + 2, base + 3),
        det_ids=(base + 101, base + 102, base + 103),
        embedding_rows=(base + 201, base + 202, base + 203),
        medoid_sample_id=base + 1,
        appearance_quality=0.9,
        internal_cosine_p10=0.9,
        internal_cosine_p50=0.95,
        internal_cosine_min=0.85,
        num_input_samples=3,
        num_overlap_rejected=0,
        num_review_excluded=0,
        num_local_outliers=0,
        max_other_bbox_iou=0.0,
        max_clean_other_bbox_iou=0.0,
    )


def _descriptor(
    stable_id: int,
    start: float,
    end: float,
    vectors: tuple[tuple[float, float], ...] | None,
) -> StablePathRetrievalDescriptor:
    segment = StableSegmentProvenance(
        stable_id=stable_id,
        constituent_micro_ids=(100 + stable_id,),
        start_det_id=1_000 + stable_id * 2,
        end_det_id=1_001 + stable_id * 2,
        start_global_frame=int(start * 10),
        end_global_frame=int(end * 10),
        start_time_sec=start,
        end_time_sec=end,
        start_clip_id="clip-a",
        end_clip_id="clip-a",
        num_detections=2,
    )
    if vectors is None:
        return StablePathRetrievalDescriptor(
            stable_id=stable_id,
            segment=segment,
            gallery=None,
            prototypes=None,
            medoid_embedding=None,
            source_endpoint_review_excluded=False,
            target_endpoint_review_excluded=False,
            source_high_overlap=False,
            target_high_overlap=False,
            missing_reason="gallery_missing",
        )
    prototypes = np.asarray(vectors, dtype=np.float32)
    return StablePathRetrievalDescriptor(
        stable_id=stable_id,
        segment=segment,
        gallery=_gallery(stable_id),
        prototypes=prototypes,
        medoid_embedding=np.array(prototypes[0], copy=True),
        source_endpoint_review_excluded=False,
        target_endpoint_review_excluded=False,
        source_high_overlap=False,
        target_high_overlap=False,
        missing_reason=None,
    )


def _descriptors() -> tuple[StablePathRetrievalDescriptor, ...]:
    # For source 0: path 1 is overlapping/too close, path 2 is the nearest
    # legal temporal candidate but has no gallery, path 3 is temporal-only,
    # and path 4 is appearance top-1 but not temporal top-2.
    return (
        _descriptor(0, 0.0, 1.0, (_unit(1.0, 0.0), _unit(0.8, 0.6))),
        _descriptor(1, 2.0, 6.0, (_unit(0.7, 0.714),)),
        _descriptor(2, 7.0, 8.0, None),
        _descriptor(3, 9.0, 10.0, (_unit(0.0, 1.0),)),
        _descriptor(4, 20.0, 21.0, (_unit(0.99, 0.1), _unit(0.9, 0.436))),
    )


def _max_score(
    source: StablePathRetrievalDescriptor,
    target: StablePathRetrievalDescriptor,
) -> float:
    assert source.prototypes is not None and target.prototypes is not None
    return float(np.max(source.prototypes @ target.prototypes.T))


class _FeatureProvider:
    def __init__(
        self,
        descriptors: tuple[StablePathRetrievalDescriptor, ...],
        *,
        high_overlap_pairs: set[tuple[int, int]] | None = None,
    ) -> None:
        self.by_id = {item.stable_id: item for item in descriptors}
        self.high_overlap_pairs = high_overlap_pairs or set()

    def build_pair(self, source_path_id: int, target_path_id: int) -> StablePathPairFeatures:
        source = self.by_id[source_path_id]
        target = self.by_id[target_path_id]
        high_overlap = (source_path_id, target_path_id) in self.high_overlap_pairs
        if source.prototypes is None or target.prototypes is None:
            return StablePathPairFeatures(
                feature_values=None,
                appearance_present=False,
                high_overlap=high_overlap,
                reason="gallery_missing",
                source_gallery=source.gallery,
                target_gallery=target.gallery,
            )

        similarity = np.asarray(source.prototypes) @ np.asarray(target.prototypes).T
        flattened = np.sort(similarity, axis=None)[::-1]
        src_to_dst = float(np.mean(np.max(similarity, axis=1)))
        dst_to_src = float(np.mean(np.max(similarity, axis=0)))
        gap = float(target.segment.start_time_sec - source.segment.end_time_sec)
        raw = {
            "prototype_cosine_max": float(flattened[0]),
            "prototype_cosine_top3_mean": float(
                np.mean(flattened[: min(3, len(flattened))])
            ),
            "medoid_cosine": float(
                np.dot(source.medoid_embedding, target.medoid_embedding)
            ),
            "mutual_prototype_score": 0.5 * (src_to_dst + dst_to_src),
            "appearance_quality_min": 0.9,
            "appearance_quality_mean": 0.9,
            "gap_sec": gap,
            "log1p_gap_sec": math.log1p(gap),
            "log_width_ratio": 0.0,
            "log_height_ratio": 0.0,
            "log_area_ratio": 0.0,
            "src_end_max_other_iou": 0.0,
            "dst_start_max_other_iou": 0.0,
            "src_end_boundary_distance": 1.0,
            "dst_start_boundary_distance": 1.0,
            "is_clip_boundary": 0.0,
        }
        values = MappingProxyType(
            {name: float(raw[name]) for name in LONG_FEATURE_SCHEMA}
        )
        return StablePathPairFeatures(
            feature_values=values,
            appearance_present=True,
            high_overlap=high_overlap,
            reason=None,
            source_gallery=source.gallery,
            target_gallery=target.gallery,
        )


class _ProvisionalScorer:
    def score_features(
        self,
        feature_values: Mapping[str, float] | None,
        *,
        appearance_present: bool,
        high_overlap: bool,
        candidate_margin: float | None = None,
    ) -> LinkResult:
        if not appearance_present:
            return LinkResult(
                None,
                None,
                "reject",
                None,
                "appearance_missing",
                appearance_present=False,
                high_overlap=high_overlap,
            )
        assert feature_values is not None
        probability = 0.95
        return LinkResult(
            probability,
            math.log(probability / (1.0 - probability)),
            "provisional",
            dict(feature_values),
            None,
            appearance_present=True,
            high_overlap=high_overlap,
        )


def _retrieved() -> list[dict[str, object]]:
    return enumerate_long_candidates(
        _descriptors(),
        appearance_topk=1,
        temporal_nearest_k=2,
        min_gap_sec_exclusive=5.0,
    )


def test_exact_cosine_topk_union_temporal_nearest_and_strict_gap() -> None:
    descriptors = _descriptors()
    rows = _retrieved()
    shuffled = enumerate_long_candidates(
        tuple(reversed(descriptors)),
        appearance_topk=1,
        temporal_nearest_k=2,
        min_gap_sec_exclusive=5.0,
    )
    assert rows == shuffled

    source_zero = {
        int(row["target_stable_id"]): row
        for row in rows
        if row["source_stable_id"] == 0
    }
    assert set(source_zero) == {2, 3, 4}
    assert source_zero[2]["selected_by_temporal_nearest"] is True
    assert source_zero[2]["selected_by_appearance_topk"] is False
    assert source_zero[3]["selected_by_temporal_nearest"] is True
    assert source_zero[3]["selected_by_appearance_topk"] is False
    assert source_zero[4]["selected_by_temporal_nearest"] is False
    assert source_zero[4]["selected_by_appearance_topk"] is True
    assert 1 not in source_zero  # overlap / gap <= 5 is never emitted
    assert all(
        float(row["target_start_time_sec"])
        - float(row["source_end_time_sec"])
        > 5.0
        and row["temporally_nonoverlapping"] is True
        for row in rows
    )

    expected = np.asarray(descriptors[0].prototypes) @ np.asarray(
        descriptors[4].prototypes
    ).T
    flattened = np.sort(expected, axis=None)[::-1]
    selected = source_zero[4]
    assert selected["gallery_score_max"] == pytest.approx(float(flattened[0]))
    assert selected["gallery_score_top3"] == pytest.approx(
        float(np.mean(flattened[:3]))
    )
    src_to_dst = float(np.mean(np.max(expected, axis=1)))
    dst_to_src = float(np.mean(np.max(expected, axis=0)))
    assert selected["gallery_score_src_to_dst"] == pytest.approx(src_to_dst)
    assert selected["gallery_score_dst_to_src"] == pytest.approx(dst_to_src)
    assert selected["gallery_score_mutual"] == pytest.approx(
        0.5 * (src_to_dst + dst_to_src)
    )


def test_directional_ranks_and_second_best_margins_use_all_legal_galleries() -> None:
    descriptors = _descriptors()
    row = next(
        item
        for item in _retrieved()
        if item["source_stable_id"] == 0 and item["target_stable_id"] == 4
    )
    assert row["appearance_rank_out"] == 1
    assert row["appearance_rank_in"] == 1

    score = _max_score(descriptors[0], descriptors[4])
    out_runner_up = _max_score(descriptors[0], descriptors[3])
    in_runner_up = max(
        _max_score(descriptors[1], descriptors[4]),
        _max_score(descriptors[3], descriptors[4]),
    )
    assert row["best_margin_out"] == pytest.approx(score - out_runner_up)
    assert row["best_margin_in"] == pytest.approx(score - in_runner_up)

    missing = next(
        item
        for item in _retrieved()
        if item["source_stable_id"] == 0 and item["target_stable_id"] == 2
    )
    assert missing["appearance_rank_out"] is None
    assert missing["appearance_rank_in"] is None
    assert missing["best_margin_out"] is None
    assert missing["best_margin_in"] is None
    assert missing["gallery_score_max"] is None


def test_missing_gallery_fails_closed_and_selected_gate_stays_uncertified() -> None:
    descriptors = _descriptors()
    scored = score_long_candidates(
        _retrieved(),
        _FeatureProvider(descriptors, high_overlap_pairs={(1, 4)}),
        _ProvisionalScorer(),
        provisional_threshold=0.5,
        selected_probability_threshold=0.9,
        selected_margin_threshold=0.01,
        worker_count=2,
    )
    missing = next(
        row
        for row in scored
        if row["source_stable_id"] == 0 and row["target_stable_id"] == 2
    )
    assert missing["appearance_present"] is False
    assert missing["model_probability"] is None
    assert all(missing[name] is None for name in LONG_FEATURE_SCHEMA)
    assert missing["decision"] == "reject"
    assert missing["decision_reason"] == "gallery_missing"
    assert missing["proposed_for_review"] is False
    assert missing["passes_provisional_threshold"] is False
    assert missing["passes_selected_gate"] is False

    selected = next(
        row
        for row in scored
        if row["source_stable_id"] == 0 and row["target_stable_id"] == 4
    )
    assert selected["decision"] == "provisional"
    assert selected["passes_provisional_threshold"] is True
    assert selected["passes_selected_probability_gate"] is True
    assert selected["passes_selected_margin_gate"] is True
    assert selected["passes_selected_gate"] is True
    assert selected["decision_reason"] == "selected_gate_evidence_uncertified"

    high_overlap = next(
        row
        for row in scored
        if row["source_stable_id"] == 1 and row["target_stable_id"] == 4
    )
    assert high_overlap["decision"] == "provisional"
    assert high_overlap["high_overlap"] is True
    assert high_overlap["passes_selected_gate"] is False
    assert high_overlap["decision_reason"] == "provisional_high_overlap"

    proposals = build_long_proposals(scored)
    assert all(row["confirmed"] is False for row in scored + proposals)
    assert all(row["selected_by_solver"] is False for row in scored + proposals)
    assert all(row["merge_applied"] is False for row in scored + proposals)
    assert missing["candidate_id"] not in {
        row["candidate_id"] for row in proposals
    }
    manifest = long_review_manifest(proposals)
    assert manifest["confirmed_enabled"] is False
    assert manifest["automatic_merge_allowed"] is False
    assert manifest["solver_used"] is False
    assert manifest["path_cover_used"] is False


def test_candidate_and_proposal_rows_serialize_with_exact_arrow_schemas() -> None:
    descriptors = _descriptors()
    candidates = score_long_candidates(
        _retrieved(),
        _FeatureProvider(descriptors),
        _ProvisionalScorer(),
        provisional_threshold=0.5,
        selected_probability_threshold=0.9,
        selected_margin_threshold=0.01,
    )
    proposals = build_long_proposals(candidates)

    candidate_table = pa.Table.from_pylist(
        candidates, schema=LONG_CANDIDATE_EDGES_SCHEMA
    )
    proposal_table = pa.Table.from_pylist(
        proposals, schema=LONG_LINK_PROPOSALS_SCHEMA
    )
    assert candidate_table.schema.equals(LONG_CANDIDATE_EDGES_SCHEMA)
    assert proposal_table.schema.equals(LONG_LINK_PROPOSALS_SCHEMA)
    assert candidate_table.num_rows == len(candidates)
    assert proposal_table.num_rows == len(proposals)
    assert candidate_table.column("model_probability").null_count >= 1
    assert proposal_table.column("model_probability").null_count == 0
    assert proposal_table.column("confirmed").to_pylist() == [False] * len(
        proposals
    )


def test_confirmed_scorer_decision_is_rejected_at_proposal_boundary() -> None:
    class ConfirmedScorer(_ProvisionalScorer):
        def score_features(
            self,
            feature_values: Mapping[str, float] | None,
            *,
            appearance_present: bool,
            high_overlap: bool,
            candidate_margin: float | None = None,
        ) -> LinkResult:
            result = super().score_features(
                feature_values,
                appearance_present=appearance_present,
                high_overlap=high_overlap,
                candidate_margin=candidate_margin,
            )
            if result.probability is None:
                return result
            return LinkResult(
                result.probability,
                result.raw_score,
                "confirmed",
                result.feature_values,
                appearance_present=True,
                high_overlap=high_overlap,
            )

    candidate = next(
        row
        for row in _retrieved()
        if row["source_stable_id"] == 0 and row["target_stable_id"] == 4
    )
    with pytest.raises(ContractError, match="cannot emit a confirmed"):
        score_long_candidates(
            [candidate],
            _FeatureProvider(_descriptors()),
            ConfirmedScorer(),
            provisional_threshold=0.5,
            selected_probability_threshold=0.9,
            selected_margin_threshold=0.01,
        )
