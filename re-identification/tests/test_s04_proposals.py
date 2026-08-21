from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from cowtrack.config import ContractError
from cowtrack.linking.features import SHORT_FEATURE_SCHEMA
from cowtrack.linking.proposals import (
    MicroEndpoint,
    brute_force_short_candidates,
    build_proposals,
    enumerate_short_candidates,
    score_short_candidates,
)


def _endpoint(
    micro_id: int,
    start: float,
    end: float,
    *,
    start_clip: str = "GX040006",
    end_clip: str | None = None,
) -> MicroEndpoint:
    return MicroEndpoint(
        micro_id=micro_id,
        start_det_id=micro_id * 10,
        end_det_id=micro_id * 10 + 1,
        start_clip_id=start_clip,
        end_clip_id=end_clip or start_clip,
        start_global_frame=micro_id * 10,
        end_global_frame=micro_id * 10 + 1,
        start_time_sec=start,
        end_time_sec=end,
    )


def test_time_window_equals_brute_force_and_is_shuffle_invariant() -> None:
    endpoints = [
        _endpoint(1, 0.0, 1.0),
        _endpoint(2, 1.0, 1.5),  # zero gap from 1 is excluded
        _endpoint(3, 6.0, 6.2),  # exact 5-second gap from 1 is included
        _endpoint(4, 6.5, 7.0, start_clip="GX050006"),
        _endpoint(5, 7.3, 7.4, start_clip="GX050006"),
    ]
    window = enumerate_short_candidates(endpoints)
    brute = brute_force_short_candidates(endpoints)
    assert window == brute
    assert enumerate_short_candidates(list(reversed(endpoints))) == window
    pairs = {(row["source_micro_id"], row["target_micro_id"]): row for row in window}
    assert (1, 2) not in pairs
    assert pairs[(1, 3)]["gap_sec"] == 5.0
    assert (3, 4) in pairs
    assert all(0.0 < row["gap_sec"] <= 5.0 for row in window)


def test_signed_s00_detection_ids_are_valid_endpoint_provenance() -> None:
    source = replace(
        _endpoint(1, 0.0, 1.0),
        start_det_id=-(1 << 63) + 7,
        end_det_id=-17,
    )
    target = replace(
        _endpoint(2, 2.0, 3.0),
        start_det_id=-9,
        end_det_id=(1 << 63) - 8,
    )

    rows = enumerate_short_candidates([source, target])

    assert len(rows) == 1
    assert rows[0]["source_end_det_id"] == -17
    assert rows[0]["target_start_det_id"] == -9


def _provenance(seed: int) -> dict[str, object]:
    return {
        "sample_ids": [seed],
        "gallery_det_ids": [seed + 1],
        "embedding_rows": [seed + 2],
        "medoid_sample_id": seed,
        "appearance_quality": 0.9,
        "internal_cosine_p10": 0.8,
        "internal_cosine_p50": 0.9,
        "internal_cosine_min": 0.7,
        "gallery_num_input_samples": 3,
        "gallery_num_overlap_rejected": 0,
        "gallery_num_review_excluded": 0,
        "gallery_num_local_outliers": 0,
        "gallery_max_other_bbox_iou": 0.1,
        "gallery_max_clean_other_bbox_iou": 0.1,
    }


def _gallery(present: bool, seed: int = 1) -> SimpleNamespace:
    return SimpleNamespace(
        present=present,
        reason=None if present else "clean_gallery_missing",
        provenance=_provenance(seed) if present else None,
    )


def _features(gap: float) -> dict[str, float]:
    return {name: gap if name == "gap_sec" else 0.25 for name in SHORT_FEATURE_SCHEMA}


@dataclass
class _FakeScorer:
    results: dict[tuple[int, int], SimpleNamespace]

    def __post_init__(self) -> None:
        self.calls: list[tuple[int, int, str]] = []

    def score_pair(self, source: int, target: int, mode: str) -> SimpleNamespace:
        self.calls.append((source, target, mode))
        return self.results[(source, target)]


def _result(
    *,
    gap: float,
    decision: str,
    probability: float | None,
    appearance: bool,
    high_overlap: bool,
    reason: str | None = None,
    source_present: bool = True,
    target_present: bool = True,
) -> SimpleNamespace:
    return SimpleNamespace(
        probability=probability,
        raw_score=None if probability is None else probability + 0.1,
        decision=decision,
        feature_values=None if probability is None else _features(gap),
        reason=reason,
        appearance_present=appearance,
        high_overlap=high_overlap,
        source_gallery=_gallery(source_present, 10),
        target_gallery=_gallery(target_present, 20),
    )


def test_short_scoring_retains_unscored_rejects_and_high_overlap() -> None:
    endpoints = [
        _endpoint(1, 0.0, 1.0),
        _endpoint(2, 2.0, 2.5),
        _endpoint(3, 3.0, 3.5),
    ]
    candidates = enumerate_short_candidates(endpoints)
    results = {}
    for row in candidates:
        key = (row["source_micro_id"], row["target_micro_id"])
        if key == (1, 2):
            results[key] = _result(
                gap=1.0,
                decision="reject",
                probability=None,
                appearance=False,
                high_overlap=True,
                reason="source_gallery_missing",
                source_present=False,
            )
        elif key == (1, 3):
            results[key] = _result(
                gap=2.0,
                decision="reject",
                probability=None,
                appearance=True,
                high_overlap=False,
                reason="source_motion_history_missing",
            )
        else:
            results[key] = _result(
                gap=float(row["gap_sec"]),
                decision="provisional",
                probability=0.999,
                appearance=True,
                high_overlap=True,
            )
    scorer = _FakeScorer(results)
    rows = score_short_candidates(candidates, scorer)
    assert len(rows) == len(candidates)
    assert {mode for _, _, mode in scorer.calls} == {"short"}
    missing = next(row for row in rows if row["source_micro_id"] == 1 and row["target_micro_id"] == 2)
    assert missing["probability"] is None
    assert missing["source_gallery_present"] is False
    assert missing["high_overlap"] is True
    assert all(
        missing[name] is None for name in SHORT_FEATURE_SCHEMA if name != "gap_sec"
    )
    assert any(row["high_overlap"] and row["proposed_for_review"] for row in rows)


def test_proposal_ranks_conflicts_and_ids_ignore_candidate_order() -> None:
    rows = [
        {
            "edge_id": "edge-a",
            "source_micro_id": 1,
            "target_micro_id": 3,
            "probability": 0.9,
            "high_overlap": False,
            "decision": "provisional",
            "proposed_for_review": True,
        },
        {
            "edge_id": "edge-b",
            "source_micro_id": 1,
            "target_micro_id": 4,
            "probability": 0.8,
            "high_overlap": False,
            "decision": "provisional",
            "proposed_for_review": True,
        },
        {
            "edge_id": "edge-c",
            "source_micro_id": 2,
            "target_micro_id": 4,
            "probability": 0.95,
            "high_overlap": True,
            "decision": "provisional",
            "proposed_for_review": True,
        },
        {
            "edge_id": "reject",
            "source_micro_id": 9,
            "target_micro_id": 10,
            "probability": 0.1,
            "high_overlap": False,
            "decision": "reject",
            "proposed_for_review": False,
        },
    ]
    proposals = build_proposals(rows)
    assert build_proposals(list(reversed(rows))) == proposals
    by_edge = {row["edge_id"]: row for row in proposals}
    assert by_edge["edge-a"]["source_rank"] == 1
    assert by_edge["edge-b"]["source_rank"] == 2
    assert by_edge["edge-c"]["target_rank"] == 1
    assert by_edge["edge-b"]["target_rank"] == 2
    assert by_edge["edge-b"]["conflict_degree"] == 2
    assert {row["conflict_group_id"] for row in proposals} == {
        by_edge["edge-a"]["conflict_group_id"]
    }
    assert all(row["review_status"] == "pending" for row in proposals)


def test_scorer_contract_failure_is_not_downgraded_to_reject() -> None:
    candidate = enumerate_short_candidates(
        [_endpoint(1, 0.0, 1.0), _endpoint(2, 2.0, 3.0)]
    )[0]
    bad = _result(
        gap=1.0,
        decision="provisional",
        probability=float("nan"),
        appearance=True,
        high_overlap=False,
    )
    with pytest.raises(ContractError, match="finite"):
        score_short_candidates([candidate], _FakeScorer({(1, 2): bad}))
