"""Independent video review for representative S04 conflict groups."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from cowtrack.qa.s04_conflict_review_config import (
    load_s04_conflict_review_config,
)
from cowtrack.qa.s04_conflict_review_plan import (
    build_s04_conflict_review_plan,
)
from cowtrack.qa.s04_review import LogFn, log, run_s04_video_review


MANIFEST_SCHEMA_VERSION = "cowtrack.s04-conflict-video-review.v1"
SUCCESS_SCHEMA_VERSION = "cowtrack.s04-conflict-video-review-success.v1"


def run_s04_conflict_review(
    *,
    proposals_dir: Path,
    ingest_dir: Path,
    microtrack_dir: Path,
    config_path: Path,
    output_dir: Path,
    logger: LogFn = log,
) -> dict[str, Any]:
    """Render three edges from three largest and three sampled conflict groups."""

    config, config_payload, config_hash = load_s04_conflict_review_config(
        config_path
    )

    def plan_factory(columns: dict[str, Any]) -> Any:
        return build_s04_conflict_review_plan(
            columns,
            random_seed=config.random_seed,
            largest_group_count=config.largest_group_count,
            random_group_count=config.random_group_count,
            max_edges_per_group=config.max_edges_per_group,
        )

    return run_s04_video_review(
        proposals_dir=proposals_dir,
        ingest_dir=ingest_dir,
        microtrack_dir=microtrack_dir,
        config_path=config_path,
        output_dir=output_dir,
        config=config,  # shared renderer consumes the identical fixed fields
        config_payload=config_payload,
        config_hash=config_hash,
        plan_factory=plan_factory,
        manifest_schema_version=MANIFEST_SCHEMA_VERSION,
        success_schema_version=SUCCESS_SCHEMA_VERSION,
        stage="S04_CONFLICT_REVIEW",
        review_purpose="representative_conflict_group_quality",
        log_prefix="s04-conflict-review",
        success_policy={
            "aggregate_quality_feedback_only": False,
            "conflict_group_review_only": True,
        },
        logger=logger,
    )


__all__ = ["run_s04_conflict_review"]
