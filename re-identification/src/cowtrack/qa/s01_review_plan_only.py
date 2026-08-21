"""Strict S01 review planning without video rendering or human labels.

The manifest written here deliberately keeps the event-facing portion of the
``cowtrack.s01-video-review.v2`` contract.  S02 can therefore consume it in
exactly the same way as a rendered review manifest, while this stage never
opens source video, probes NVENC, invokes ffmpeg, or creates a labels file.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from cowtrack.config import ContractError
from cowtrack.qa.review_config import S01ReviewConfig, load_s01_review_config
from cowtrack.qa import s01_review as review
from cowtrack.schemas.detections import DETECTIONS_SCHEMA
from cowtrack.schemas.edges import DET_EDGES_SCHEMA
from cowtrack.schemas.frames import FRAMES_SCHEMA
from cowtrack.schemas.tracklets import DET_TO_MICRO_SCHEMA, MICROTRACKLETS_SCHEMA


LogFn = Callable[[str], None]

STAGE_NAME = "S01_REVIEW_PLAN_ONLY"
SUCCESS_SCHEMA_VERSION = "cowtrack.s01-review-plan-only-success.v1"
EXECUTION_MODE = "plan_only"
EXPECTED_COORDINATE_SYSTEM = "raw_encoded_landscape_no_autorotate"
EXPECTED_ARTIFACTS = frozenset({review.MANIFEST_NAME})


def log(message: str) -> None:
    """Write one immediately visible progress message."""

    print(message, flush=True)


def _validated_stage_files(
    directory: Path,
    *,
    expected_stage: str,
    required_names: tuple[str, ...],
    progress_interval_sec: float,
    logger: LogFn,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    success_path = directory / review.SUCCESS_NAME
    payload = review._read_json(success_path, f"{expected_stage} success marker")
    if not isinstance(payload, dict) or payload.get("stage") != expected_stage:
        raise ContractError(
            f"{success_path} is not a completed {expected_stage} output"
        )
    records = payload.get("output_fingerprints")
    if not isinstance(records, list):
        raise ContractError(
            f"{expected_stage} success marker lacks output fingerprints"
        )
    by_name: dict[str, Mapping[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("path"), str):
            raise ContractError(
                f"{expected_stage} success marker has an invalid output fingerprint"
            )
        name = str(record["path"])
        if name in by_name:
            raise ContractError(
                f"{expected_stage} success marker repeats artifact: {name}"
            )
        by_name[name] = record
    missing = sorted(set(required_names) - set(by_name))
    if missing:
        raise ContractError(
            f"{expected_stage} success marker lacks artifacts: {missing}"
        )

    observed: list[dict[str, Any]] = []
    for name in required_names:
        path = (directory / name).resolve()
        current = review._fingerprint(
            path,
            progress_interval_sec=progress_interval_sec,
            logger=logger,
        )
        recorded = by_name[name]
        for key in ("size_bytes", "sha256"):
            if current[key] != recorded.get(key):
                raise ContractError(
                    f"completed {expected_stage} artifact changed: {name} ({key})"
                )
        observed.append(current)
    observed.append(
        review._fingerprint(
            success_path.resolve(),
            progress_interval_sec=progress_interval_sec,
            logger=logger,
        )
    )
    return payload, observed


def _resolved_video_paths(
    resolved_manifest_path: Path,
) -> tuple[str, dict[str, Path]]:
    """Validate clip identity/order without opening or fingerprinting raw video."""

    # Path.resolve(strict=False) is intentionally metadata-only here. Planning
    # consumes S00/S01 tables, not raw video bytes.
    return review._resolved_manifest_contract(resolved_manifest_path)


def _load_plan_data(
    ingest_dir: Path,
    microtrack_dir: Path,
    config_path: Path,
    *,
    progress_interval_sec: float,
    logger: LogFn,
) -> review.LoadedReviewData:
    """Load only immutable S00/S01 artifacts needed to select review events."""

    s00_success, s00_fingerprints = _validated_stage_files(
        ingest_dir,
        expected_stage="S00",
        required_names=(
            "frames.parquet",
            "detections.parquet",
            "resolved_manifest.json",
            "ingest_report.json",
        ),
        progress_interval_sec=progress_interval_sec,
        logger=logger,
    )
    s01_success, s01_fingerprints = _validated_stage_files(
        microtrack_dir,
        expected_stage="S01",
        required_names=(
            "det_to_micro.parquet",
            "microtracklets.parquet",
            "det_edges.parquet",
            "microtrack_report.json",
        ),
        progress_interval_sec=progress_interval_sec,
        logger=logger,
    )
    sequence_id, video_paths = _resolved_video_paths(
        ingest_dir / "resolved_manifest.json"
    )
    counts = review._validate_upstream_stats(
        s00_success,
        s01_success,
        manifest_num_clips=len(video_paths),
    )
    review._validate_upstream_reports(
        ingest_dir / "ingest_report.json",
        microtrack_dir / "microtrack_report.json",
        sequence_id=sequence_id,
        video_paths=video_paths,
        counts=counts,
    )

    logger("[s01-review-plan] reading frames.parquet")
    frames = review._parquet_columns(
        ingest_dir / "frames.parquet",
        expected_schema=FRAMES_SCHEMA,
        expected_rows=counts["num_frames"],
        columns=tuple(field.name for field in FRAMES_SCHEMA),
        label="S00 frames",
    )
    logger("[s01-review-plan] reading detections.parquet")
    detections = review._parquet_columns(
        ingest_dir / "detections.parquet",
        expected_schema=DETECTIONS_SCHEMA,
        expected_rows=counts["num_detections"],
        columns=list(
            dict.fromkeys((*review.DETECTION_COLUMNS, "x1", "y1", "x2", "y2"))
        ),
        label="S00 detections",
    )
    logger("[s01-review-plan] reading S01 microtrack tables")
    mapping = review._parquet_columns(
        microtrack_dir / "det_to_micro.parquet",
        expected_schema=DET_TO_MICRO_SCHEMA,
        expected_rows=counts["num_valid_detections"],
        columns=list(review.DET_TO_MICRO_COLUMNS),
        label="S01 det_to_micro",
    )
    microtracklets = review._parquet_columns(
        microtrack_dir / "microtracklets.parquet",
        expected_schema=MICROTRACKLETS_SCHEMA,
        expected_rows=counts["num_microtracklets"],
        columns=list(review.MICROTRACKLET_COLUMNS),
        label="S01 microtracklets",
    )
    edges = review._parquet_columns(
        microtrack_dir / "det_edges.parquet",
        expected_schema=DET_EDGES_SCHEMA,
        expected_rows=counts["num_accepted_edges"],
        columns=list(dict.fromkeys((*review.DET_EDGE_COLUMNS, "accepted"))),
        label="S01 det_edges",
    )
    review._validate_frames(
        frames,
        video_paths,
        expected_sequence_id=sequence_id,
        expected_num_frames=counts["num_frames"],
    )
    review._validate_detection_frame_foreign_keys(
        frames,
        detections,
        expected_valid_detections=counts["num_valid_detections"],
    )
    indices = review._build_indices(frames, detections, mapping, microtracklets)
    config_fingerprint = review._fingerprint(
        config_path,
        progress_interval_sec=progress_interval_sec,
        logger=logger,
    )
    input_fingerprints = sorted(
        [*s00_fingerprints, *s01_fingerprints, config_fingerprint],
        key=lambda record: str(record["path"]),
    )
    return review.LoadedReviewData(
        frames=frames,
        detections=detections,
        det_to_micro=mapping,
        microtracklets=microtracklets,
        det_edges=edges,
        video_paths=video_paths,
        input_fingerprints=tuple(input_fingerprints),
        valid_detection_positions_by_frame=indices[0],
        valid_detection_frame_offsets=indices[1],
        micro_paths=indices[2],
    )


def _output_video_contract(config: S01ReviewConfig) -> dict[str, Any]:
    """Retain the event contract S02 reads while recording that no video exists."""

    return {
        "execution_mode": EXECUTION_MODE,
        "videos_rendered": False,
        "codec_if_rendered": "h264_nvenc",
        "physical_gpu_if_rendered": 1,
        "logical_gpu_if_rendered": config.logical_gpu,
        "width_if_rendered": config.output_width,
        "height_if_rendered": config.output_height,
        "average_frame_rate_if_rendered": str(review.EXPECTED_FRAME_RATE),
        "cpu_encoder_fallback": False,
        "true_event_highlight_radius_frames": review.EVENT_HIGHLIGHT_RADIUS_FRAMES,
        "synthetic_long_and_quality_anchors_are_not_events": True,
    }


def _manifest_payload(
    *,
    config: S01ReviewConfig,
    config_payload: dict[str, Any],
    identity: dict[str, Any],
    plan_payload: dict[str, Any],
    case_records: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": review.MANIFEST_SCHEMA_VERSION,
        "coordinate_system": EXPECTED_COORDINATE_SYSTEM,
        "execution_mode": EXECUTION_MODE,
        "videos_rendered": False,
        "human_labels_required": False,
        "output_video_contract": _output_video_contract(config),
        "selection_warning": (
            "Risk reasons select exclusion windows for automatic appearance sampling; "
            "they are not identity labels. legacy_track_id is QA-only and is not "
            "ground truth."
        ),
        "identity": identity,
        "effective_config": config_payload,
        "selection_plan": plan_payload,
        "cases": case_records,
    }


def _output_fingerprint(path: Path, output_dir: Path) -> dict[str, Any]:
    current = review._fingerprint(path)
    return {
        "path": str(path.relative_to(output_dir)),
        "size_bytes": current["size_bytes"],
        "sha256": current["sha256"],
    }


def _stats(
    plan: review.ReviewPlan, render_cases: tuple[review.RenderCase, ...]
) -> dict[str, int]:
    return {
        "num_cases": len(plan.cases),
        "num_risk_cases": len(plan.risk_cases),
        "num_quality_reference_cases": len(plan.quality_cases),
        "num_frames_planned": sum(item.expected_frame_count for item in render_cases),
        "num_frames_rendered": 0,
        "num_videos_rendered": 0,
    }


def _success_payload(
    output_dir: Path,
    *,
    config_hash: str,
    identity: dict[str, Any],
    input_fingerprints: tuple[dict[str, Any], ...],
    stats: dict[str, int],
) -> dict[str, Any]:
    manifest_path = output_dir / review.MANIFEST_NAME
    return {
        "stage": STAGE_NAME,
        "schema_version": SUCCESS_SCHEMA_VERSION,
        "config_hash": config_hash,
        "program_commit_hash": None,
        "identity": identity,
        "input_fingerprints": list(input_fingerprints),
        "manifest_sha256": review._sha256(manifest_path),
        "output_fingerprints": [_output_fingerprint(manifest_path, output_dir)],
        "stats": stats,
    }


def _validate_completed_output(
    output_dir: Path,
    success: Mapping[str, Any],
    *,
    expected_manifest: dict[str, Any],
    config_hash: str,
    identity: dict[str, Any],
    input_fingerprints: tuple[dict[str, Any], ...],
    stats: dict[str, int],
) -> None:
    expected_success_keys = {
        "stage",
        "schema_version",
        "config_hash",
        "program_commit_hash",
        "identity",
        "input_fingerprints",
        "manifest_sha256",
        "output_fingerprints",
        "stats",
    }
    if set(success) != expected_success_keys:
        raise ContractError("completed S01 review-plan _SUCCESS keys mismatch")
    expected_fields: dict[str, Any] = {
        "stage": STAGE_NAME,
        "schema_version": SUCCESS_SCHEMA_VERSION,
        "config_hash": config_hash,
        "program_commit_hash": None,
        "identity": identity,
        "input_fingerprints": list(input_fingerprints),
        "stats": stats,
    }
    for key, expected in expected_fields.items():
        if success.get(key) != expected:
            raise ContractError(
                f"completed S01 review-plan _SUCCESS mismatch: {key}"
            )

    actual_artifacts = {
        str(path.relative_to(output_dir))
        for path in output_dir.rglob("*")
        if path.is_file() and path.name != review.SUCCESS_NAME
    }
    if actual_artifacts != EXPECTED_ARTIFACTS:
        raise ContractError(
            "completed S01 review-plan artifact set mismatch: "
            f"{sorted(actual_artifacts)}"
        )
    records = success.get("output_fingerprints")
    if not isinstance(records, list) or len(records) != 1:
        raise ContractError(
            "completed S01 review-plan output fingerprints must contain one artifact"
        )
    recorded = records[0]
    if not isinstance(recorded, dict) or recorded.get("path") != review.MANIFEST_NAME:
        raise ContractError(
            "completed S01 review-plan manifest fingerprint is invalid"
        )
    manifest_path = output_dir / review.MANIFEST_NAME
    current = _output_fingerprint(manifest_path, output_dir)
    for key in ("path", "size_bytes", "sha256"):
        if current[key] != recorded.get(key):
            raise ContractError(
                f"completed S01 review-plan artifact changed: "
                f"{review.MANIFEST_NAME} ({key})"
            )
    if success.get("manifest_sha256") != current["sha256"]:
        raise ContractError(
            "completed S01 review-plan manifest differs from _SUCCESS"
        )
    observed_manifest = review._read_json(
        manifest_path, "S01 plan-only review manifest"
    )
    if observed_manifest != expected_manifest:
        raise ContractError("completed S01 plan-only review manifest contract changed")


def _verify_inputs_unchanged(
    fingerprints: tuple[dict[str, Any], ...],
    *,
    progress_interval_sec: float,
    logger: LogFn,
) -> None:
    for fingerprint in fingerprints:
        path = Path(str(fingerprint["path"]))
        current = review._fingerprint(
            path,
            progress_interval_sec=progress_interval_sec,
            logger=logger,
        )
        for key in ("size_bytes", "mtime_ns", "sha256"):
            if current[key] != fingerprint[key]:
                raise ContractError(
                    f"S01 review-plan input changed during planning: {path} ({key})"
                )


def _check_output_overlap(
    output_dir: Path, ingest_dir: Path, microtrack_dir: Path
) -> None:
    for label, input_dir in (
        ("S00 ingest", ingest_dir),
        ("S01 microtrack", microtrack_dir),
    ):
        if (
            output_dir == input_dir
            or output_dir.is_relative_to(input_dir)
            or input_dir.is_relative_to(output_dir)
        ):
            raise ContractError(
                f"review-plan output must not overlap the immutable {label} tree: "
                f"output={output_dir}, input={input_dir}"
            )


def run_s01_review_plan_only(
    ingest_dir: Path,
    microtrack_dir: Path,
    config_path: Path,
    output_dir: Path,
    *,
    logger: LogFn = log,
) -> dict[str, Any]:
    """Create and atomically commit the S01 event plan consumed by S02."""

    ingest_dir = ingest_dir.resolve()
    microtrack_dir = microtrack_dir.resolve()
    config_path = config_path.resolve()
    output_dir = output_dir.resolve()
    _check_output_overlap(output_dir, ingest_dir, microtrack_dir)

    config, config_payload, config_hash = load_s01_review_config(config_path)
    success_path = output_dir / review.SUCCESS_NAME
    if output_dir.exists() and not output_dir.is_dir():
        raise ContractError(f"S01 review-plan output is not a directory: {output_dir}")
    if (
        not success_path.is_file()
        and output_dir.exists()
        and any(output_dir.iterdir())
    ):
        raise ContractError(
            f"output directory is non-empty without {review.SUCCESS_NAME}: {output_dir}"
        )

    logger(
        "[s01-review-plan] validating immutable S00/S01 inputs; "
        "raw video, NVENC, and human labels are not used"
    )
    data = _load_plan_data(
        ingest_dir,
        microtrack_dir,
        config_path,
        progress_interval_sec=config.progress_interval_sec,
        logger=logger,
    )
    logger("[s01-review-plan] building deterministic review-event plan")
    plan = review._plan(data, config)
    render_cases = review._render_cases(data, plan)
    review._validate_plan(plan, render_cases)
    plan_payload = plan.to_manifest_payload()
    case_records = [review._case_record(item) for item in render_cases]
    for record in case_records:
        record["render"] = {
            "status": "not_rendered",
            "reason": EXECUTION_MODE,
        }
    identity = review._manifest_identity(
        config_hash=config_hash,
        input_fingerprints=data.input_fingerprints,
        plan_payload=plan_payload,
        case_records=case_records,
    )
    manifest = _manifest_payload(
        config=config,
        config_payload=config_payload,
        identity=identity,
        plan_payload=plan_payload,
        case_records=case_records,
    )
    stats = _stats(plan, render_cases)

    if success_path.is_file():
        existing = review._read_json(success_path, "S01 review-plan success marker")
        if not isinstance(existing, dict):
            raise ContractError(
                "existing S01 review-plan success marker is not an object"
            )
        _validate_completed_output(
            output_dir,
            existing,
            expected_manifest=manifest,
            config_hash=config_hash,
            identity=identity,
            input_fingerprints=data.input_fingerprints,
            stats=stats,
        )
        logger(
            f"[s01-review-plan] already complete and fully revalidated: {success_path}"
        )
        return dict(existing)

    final_output_dir = output_dir
    final_output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = final_output_dir.parent / (
        f".{final_output_dir.name}.staging-{os.getpid()}"
    )
    if staging_dir.exists():
        raise ContractError(
            f"S01 review-plan staging directory already exists: {staging_dir}"
        )
    staging_dir.mkdir(parents=False)

    review._atomic_write_json(staging_dir / review.MANIFEST_NAME, manifest)
    logger("[s01-review-plan] re-fingerprinting immutable inputs before commit")
    _verify_inputs_unchanged(
        data.input_fingerprints,
        progress_interval_sec=config.progress_interval_sec,
        logger=logger,
    )
    success = _success_payload(
        staging_dir,
        config_hash=config_hash,
        identity=identity,
        input_fingerprints=data.input_fingerprints,
        stats=stats,
    )
    review._atomic_write_json(staging_dir / review.SUCCESS_NAME, success)
    _validate_completed_output(
        staging_dir,
        success,
        expected_manifest=manifest,
        config_hash=config_hash,
        identity=identity,
        input_fingerprints=data.input_fingerprints,
        stats=stats,
    )

    if final_output_dir.exists():
        if any(final_output_dir.iterdir()):
            raise ContractError(
                "final S01 review-plan output became non-empty during staging: "
                f"{final_output_dir}"
            )
        final_output_dir.rmdir()
    os.replace(staging_dir, final_output_dir)
    _validate_completed_output(
        final_output_dir,
        success,
        expected_manifest=manifest,
        config_hash=config_hash,
        identity=identity,
        input_fingerprints=data.input_fingerprints,
        stats=stats,
    )
    final_success_path = final_output_dir / review.SUCCESS_NAME
    logger(f"[s01-review-plan] complete and revalidated: {final_success_path}")
    return success


__all__ = [
    "EXECUTION_MODE",
    "STAGE_NAME",
    "SUCCESS_SCHEMA_VERSION",
    "run_s01_review_plan_only",
]
