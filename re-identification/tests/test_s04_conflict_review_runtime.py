from __future__ import annotations

from pathlib import Path

import numpy as np

import cowtrack.qa.s04_conflict_review as conflict_review


ROOT = Path(__file__).parents[1]


def _two_edge_conflict() -> dict[str, np.ndarray]:
    rows = [
        {
            "proposal_id": "p0",
            "edge_id": "e0",
            "source_micro_id": 1,
            "target_micro_id": 2,
            "probability": 0.998,
            "high_overlap": False,
            "source_rank": 1,
            "target_rank": 1,
            "conflict_degree": 1,
            "conflict_group_id": "g0",
            "conflict_group_edge_count": 2,
            "conflict_group_node_count": 3,
            "review_status": "pending",
        },
        {
            "proposal_id": "p1",
            "edge_id": "e1",
            "source_micro_id": 1,
            "target_micro_id": 3,
            "probability": 0.997,
            "high_overlap": False,
            "source_rank": 2,
            "target_rank": 1,
            "conflict_degree": 1,
            "conflict_group_id": "g0",
            "conflict_group_edge_count": 2,
            "conflict_group_node_count": 3,
            "review_status": "pending",
        },
    ]
    return {
        name: np.asarray([row[name] for row in rows])
        for name in rows[0]
    }


def test_conflict_runtime_reuses_strict_video_runner(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_runner(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {"stage": "synthetic"}

    monkeypatch.setattr(conflict_review, "run_s04_video_review", fake_runner)
    result = conflict_review.run_s04_conflict_review(
        proposals_dir=Path("work/04_short_proposals"),
        ingest_dir=Path("work/00_ingest"),
        microtrack_dir=Path("work/01_microtrack"),
        config_path=ROOT / "configs" / "s04_conflict_review.yaml",
        output_dir=Path("work/04_conflict_review"),
        logger=lambda _: None,
    )

    assert result == {"stage": "synthetic"}
    assert captured["stage"] == "S04_CONFLICT_REVIEW"
    assert captured["review_purpose"] == "representative_conflict_group_quality"
    assert captured["success_policy"] == {
        "aggregate_quality_feedback_only": False,
        "conflict_group_review_only": True,
    }
    assert captured["config_path"] == ROOT / "configs" / "s04_conflict_review.yaml"
    plan = captured["plan_factory"](_two_edge_conflict())  # type: ignore[operator]
    assert len(plan.groups) == 1
    assert len(plan.cases) == 2
    assert plan.cases[0].suggested_filename.endswith("p0.9980.mp4")
