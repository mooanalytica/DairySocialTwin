"""Closed, dataset-specific configuration contract for S06 QA/export.

S06 consumes only the operator-forced appearance result.  It exports evidence
and provisional identities; it does not calibrate appearance cosine, certify
an identity, or re-run either path solver.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

from cowtrack.config import ContractError
from cowtrack.linking.dataset_contract import (
    EXPECTED_CLIP_ORDER,
    EXPECTED_FRAME_COUNT,
    EXPECTED_FRAME_COUNTS,
    EXPECTED_SEQUENCE_ID,
    MAX_GLOBAL_TRACK_COUNT,
)


APPEARANCE_GRADES = (
    "A_CLEAN",
    "B_EXISTING_DEGRADED",
    "C_REENCODED_DEGRADED",
)

# S00 is already complete for this closed dataset, so its row counts remain a
# strict input contract.  Later-stage counts are intentionally not repeated
# here: S06 resolves those from the validated S01/S04/S05 artifacts.
EXPECTED_TOTAL_DETECTIONS_BY_CLIP = (
    745_070,
    103_345,
    248_772,
    332_813,
    413_466,
    265_557,
    154_461,
    196_188,
    216_575,
    386_726,
    259_936,
)
EXPECTED_VALID_DETECTIONS_BY_CLIP = (
    745_052,
    103_319,
    248_019,
    332_754,
    413_161,
    264_397,
    154_326,
    196_054,
    215_158,
    386_669,
    259_204,
)
EXPECTED_INVALID_DETECTIONS_BY_CLIP = (
    18,
    26,
    753,
    59,
    305,
    1_160,
    135,
    134,
    1_417,
    57,
    732,
)
EXPECTED_TOTAL_DETECTIONS = sum(EXPECTED_TOTAL_DETECTIONS_BY_CLIP)
EXPECTED_VALID_DETECTIONS = sum(EXPECTED_VALID_DETECTIONS_BY_CLIP)
EXPECTED_INVALID_DETECTIONS = sum(EXPECTED_INVALID_DETECTIONS_BY_CLIP)


@dataclass(frozen=True)
class S06Artifacts:
    detections_csv: str
    qa_metrics: str
    global_track_summary: str
    low_confidence_links: str
    low_confidence_assets_dir: str
    contact_sheets_dir: str
    videos_dir: str
    videos_by_clip: tuple[str, ...]
    effective_config: str
    success: str


@dataclass(frozen=True)
class S06ExportConfig:
    schema_version: str
    random_seed: int
    execution_mode: str
    required_upstream_stage: str
    identity_source: str
    authorization_basis: str
    id_status: str
    certification_claimed: bool
    probability_from_cosine_allowed: bool

    expected_sequence_id: str
    clip_order: tuple[str, ...]
    expected_frame_count: int
    frame_counts_by_clip: tuple[int, ...]
    expected_total_detection_count: int
    expected_valid_detection_count: int
    expected_invalid_detection_count: int
    expected_total_detections_by_clip: tuple[int, ...]
    expected_valid_detections_by_clip: tuple[int, ...]
    expected_invalid_detections_by_clip: tuple[int, ...]
    expected_microtrack_count: int | None
    expected_stable_track_count: int | None
    expected_global_track_count: int
    expected_selected_link_count: int | None
    expected_candidate_edge_count: int | None
    expected_maximum_feasible_link_count: int | None
    expected_max_concurrent_stable_track_count: int | None
    expected_backbone_chain_count: int | None
    expected_selected_prior_link_count: int | None
    expected_rescue_candidate_count: int | None
    expected_rescue_embedding_count: int | None
    expected_rescue_stable_track_count: int | None
    expected_selected_best_degraded_crop_count: int | None
    expected_selected_review_excluded_crop_count: int | None
    expected_grade_a_clean_count: int | None
    expected_grade_b_existing_degraded_count: int | None
    expected_grade_c_reencoded_degraded_count: int | None
    expected_cycle_count: int
    expected_same_frame_violation_count: int
    expected_temporal_overlap_violation_count: int
    expected_unassigned_valid_detection_count: int

    num_confirmed_ids: int
    num_provisional_ids: int
    population_soft_max: int
    population_overflow: int
    display_id_prefix: str
    display_id_width: int
    invalid_identity_fields_null: bool

    reference_mode: str
    low_confidence_thresholds: tuple[float, ...]
    html_priority_threshold: float
    html_maximum_threshold: float
    crops_per_link_endpoint: int
    contact_sheet_rows: int
    contact_sheet_columns: int
    candidate_rank_scope: str
    candidate_rank_tie_break: str
    margin_semantics: str
    recompute_c_grade_iou_from_rescue_samples: bool
    c_grade_high_overlap_threshold: float
    unavailable_metrics_are_null: bool

    output_width: int
    output_height: int
    fps_numerator: int
    fps_denominator: int
    bbox_scale: float
    autorotate: bool
    include_audio: bool
    label_field: str
    color_policy: str
    draw_invalid_identity: bool

    ffmpeg_binary: str
    ffprobe_binary: str
    codec: str
    required_cuda_visible_devices: str
    logical_gpu: int
    preset: str
    cq: int
    pixel_format: str

    progress_interval_sec: float
    deterministic_row_sort: bool
    row_sort_keys: tuple[str, ...]
    atomic_commit: bool
    resume_revalidate: bool
    log_flush: bool
    validate_upstream_hashes: bool
    fingerprint_all_outputs: bool
    artifacts: S06Artifacts

    @property
    def expected_detection_count(self) -> int:
        """Alias for callers that use the generic stage count name."""

        return self.expected_total_detection_count

    @property
    def output_fps(self) -> float:
        return self.fps_numerator / self.fps_denominator


_TOP_KEYS = {
    "pipeline",
    "inputs",
    "identity",
    "qa",
    "render",
    "encoder",
    "runtime",
    "artifacts",
}
_PIPELINE_KEYS = {
    "schema_version",
    "random_seed",
    "execution_mode",
    "required_upstream_stage",
    "identity_source",
    "authorization_basis",
    "id_status",
    "certification_claimed",
    "probability_from_cosine_allowed",
}
_INPUT_KEYS = {
    "expected_sequence_id",
    "clip_order",
    "expected_frame_count",
    "frame_counts_by_clip",
    "expected_total_detection_count",
    "expected_valid_detection_count",
    "expected_invalid_detection_count",
    "total_detections_by_clip",
    "valid_detections_by_clip",
    "invalid_detections_by_clip",
    "expected_microtrack_count",
    "expected_stable_track_count",
    "expected_global_track_count",
    "expected_selected_link_count",
    "expected_candidate_edge_count",
    "expected_maximum_feasible_link_count",
    "expected_max_concurrent_stable_track_count",
    "expected_backbone_chain_count",
    "expected_selected_prior_link_count",
    "expected_rescue_candidate_count",
    "expected_rescue_embedding_count",
    "expected_rescue_stable_track_count",
    "expected_selected_best_degraded_crop_count",
    "expected_selected_review_excluded_crop_count",
    "expected_appearance_grade_counts",
    "expected_cycle_count",
    "expected_same_frame_violation_count",
    "expected_temporal_overlap_violation_count",
    "expected_unassigned_valid_detection_count",
}
_IDENTITY_KEYS = {
    "num_confirmed_ids",
    "num_provisional_ids",
    "population_soft_max",
    "population_overflow",
    "display_id_prefix",
    "display_id_width",
    "invalid_identity_fields_null",
}
_QA_KEYS = {
    "reference_mode",
    "low_confidence_thresholds",
    "html_priority_threshold",
    "html_maximum_threshold",
    "crops_per_link_endpoint",
    "contact_sheet_rows",
    "contact_sheet_columns",
    "candidate_rank_scope",
    "candidate_rank_tie_break",
    "margin_semantics",
    "recompute_c_grade_iou_from_rescue_samples",
    "c_grade_high_overlap_threshold",
    "unavailable_metrics_are_null",
}
_RENDER_KEYS = {
    "output_width",
    "output_height",
    "fps_numerator",
    "fps_denominator",
    "bbox_scale",
    "autorotate",
    "include_audio",
    "label_field",
    "color_policy",
    "draw_invalid_identity",
}
_ENCODER_KEYS = {
    "ffmpeg_binary",
    "ffprobe_binary",
    "codec",
    "required_cuda_visible_devices",
    "logical_gpu",
    "preset",
    "cq",
    "pixel_format",
}
_RUNTIME_KEYS = {
    "progress_interval_sec",
    "deterministic_row_sort",
    "row_sort_keys",
    "atomic_commit",
    "resume_revalidate",
    "log_flush",
    "validate_upstream_hashes",
    "fingerprint_all_outputs",
}
_ARTIFACT_KEYS = {
    "detections_csv",
    "qa_metrics",
    "global_track_summary",
    "low_confidence_links",
    "low_confidence_assets_dir",
    "contact_sheets_dir",
    "videos_dir",
    "videos_by_clip",
    "effective_config",
    "success",
}
def _mapping(value: object, expected: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ContractError(f"S06 {label} must be a mapping")
    actual = set(value)
    if actual != expected:
        raise ContractError(
            f"S06 {label} keys mismatch; missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )
    return value


def _fixed(actual: object, expected: object, label: str) -> None:
    bad_type = (
        (isinstance(expected, bool) and type(actual) is not bool)
        or (
            isinstance(expected, int)
            and not isinstance(expected, bool)
            and (isinstance(actual, bool) or not isinstance(actual, int))
        )
        or (isinstance(expected, str) and not isinstance(actual, str))
    )
    if bad_type or actual != expected:
        raise ContractError(f"fixed S06 export requires {label}={expected!r}")


def _integer(value: object, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ContractError(f"S06 {label} must be an integer >= {minimum}")
    return value


def _optional_integer(
    value: object, label: str, *, minimum: int = 0
) -> int | None:
    """Validate one upstream-derived count while preserving a YAML null."""

    if value is None:
        return None
    return _integer(value, label, minimum=minimum)


def _finite(value: object, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"S06 {label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        qualifier = "positive and finite" if positive else "finite"
        raise ContractError(f"S06 {label} must be {qualifier}")
    return result


def _relative_artifact(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ContractError(f"S06 {label} must be a non-empty relative POSIX path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != value:
        raise ContractError(f"S06 {label} must be a normalized relative POSIX path")
    return value


def load_s06_export_config(
    path: Path,
) -> tuple[S06ExportConfig, dict[str, Any], str]:
    """Load and validate the fixed S06 forced-provisional export contract."""

    path = path.resolve()
    if not path.is_file():
        raise ContractError(f"S06 export config does not exist: {path}")
    try:
        serialized = path.read_bytes()
        payload = (
            json.loads(serialized)
            if path.suffix.lower() == ".json"
            else yaml.safe_load(serialized)
        )
    except (OSError, UnicodeError, json.JSONDecodeError, yaml.YAMLError) as exc:
        raise ContractError(f"cannot read S06 export config {path}: {exc}") from exc

    root = _mapping(payload, _TOP_KEYS, "config")
    pipeline = _mapping(root["pipeline"], _PIPELINE_KEYS, "pipeline")
    inputs = _mapping(root["inputs"], _INPUT_KEYS, "inputs")
    identity = _mapping(root["identity"], _IDENTITY_KEYS, "identity")
    qa = _mapping(root["qa"], _QA_KEYS, "qa")
    render = _mapping(root["render"], _RENDER_KEYS, "render")
    encoder = _mapping(root["encoder"], _ENCODER_KEYS, "encoder")
    runtime = _mapping(root["runtime"], _RUNTIME_KEYS, "runtime")
    artifact_data = _mapping(root["artifacts"], _ARTIFACT_KEYS, "artifacts")
    frame_counts = _mapping(
        inputs["frame_counts_by_clip"], set(EXPECTED_CLIP_ORDER), "frame counts"
    )
    total_by_clip = _mapping(
        inputs["total_detections_by_clip"],
        set(EXPECTED_CLIP_ORDER),
        "total detections by clip",
    )
    valid_by_clip = _mapping(
        inputs["valid_detections_by_clip"],
        set(EXPECTED_CLIP_ORDER),
        "valid detections by clip",
    )
    invalid_by_clip = _mapping(
        inputs["invalid_detections_by_clip"],
        set(EXPECTED_CLIP_ORDER),
        "invalid detections by clip",
    )
    grades = _mapping(
        inputs["expected_appearance_grade_counts"],
        set(APPEARANCE_GRADES),
        "appearance grade counts",
    )
    video_paths = _mapping(
        artifact_data["videos_by_clip"],
        set(EXPECTED_CLIP_ORDER),
        "artifact videos by clip",
    )

    expected_scalars: dict[str, tuple[object, object]] = {
        "pipeline.schema_version": (pipeline["schema_version"], "1.0"),
        "pipeline.random_seed": (pipeline["random_seed"], 20260710),
        "pipeline.execution_mode": (
            pipeline["execution_mode"],
            "forced_provisional_qa_export",
        ),
        "pipeline.required_upstream_stage": (
            pipeline["required_upstream_stage"],
            "S05_FORCE_APPEARANCE",
        ),
        "pipeline.identity_source": (
            pipeline["identity_source"],
            "05_forced_appearance_only",
        ),
        "pipeline.authorization_basis": (
            pipeline["authorization_basis"],
            "operator_forced_appearance_exact_62",
        ),
        "pipeline.id_status": (pipeline["id_status"], "forced_provisional"),
        "pipeline.certification_claimed": (
            pipeline["certification_claimed"],
            False,
        ),
        "pipeline.probability_from_cosine_allowed": (
            pipeline["probability_from_cosine_allowed"],
            False,
        ),
        "inputs.expected_sequence_id": (
            inputs["expected_sequence_id"],
            EXPECTED_SEQUENCE_ID,
        ),
        "inputs.clip_order": (inputs["clip_order"], list(EXPECTED_CLIP_ORDER)),
        "inputs.expected_frame_count": (
            inputs["expected_frame_count"],
            EXPECTED_FRAME_COUNT,
        ),
        "inputs.expected_total_detection_count": (
            inputs["expected_total_detection_count"],
            EXPECTED_TOTAL_DETECTIONS,
        ),
        "inputs.expected_valid_detection_count": (
            inputs["expected_valid_detection_count"],
            EXPECTED_VALID_DETECTIONS,
        ),
        "inputs.expected_invalid_detection_count": (
            inputs["expected_invalid_detection_count"],
            EXPECTED_INVALID_DETECTIONS,
        ),
        "inputs.expected_global_track_count": (
            inputs["expected_global_track_count"],
            MAX_GLOBAL_TRACK_COUNT,
        ),
        "inputs.expected_cycle_count": (inputs["expected_cycle_count"], 0),
        "inputs.expected_same_frame_violation_count": (
            inputs["expected_same_frame_violation_count"],
            0,
        ),
        "inputs.expected_temporal_overlap_violation_count": (
            inputs["expected_temporal_overlap_violation_count"],
            0,
        ),
        "inputs.expected_unassigned_valid_detection_count": (
            inputs["expected_unassigned_valid_detection_count"],
            0,
        ),
        "identity.num_confirmed_ids": (identity["num_confirmed_ids"], 0),
        "identity.num_provisional_ids": (
            identity["num_provisional_ids"],
            MAX_GLOBAL_TRACK_COUNT,
        ),
        "identity.population_soft_max": (identity["population_soft_max"], 57),
        "identity.population_overflow": (identity["population_overflow"], 5),
        "identity.display_id_prefix": (identity["display_id_prefix"], "G"),
        "identity.display_id_width": (identity["display_id_width"], 4),
        "identity.invalid_identity_fields_null": (
            identity["invalid_identity_fields_null"],
            True,
        ),
        "qa.reference_mode": (qa["reference_mode"], "observed_only"),
        "qa.html_priority_threshold": (qa["html_priority_threshold"], 0.4),
        "qa.html_maximum_threshold": (qa["html_maximum_threshold"], 0.5),
        "qa.crops_per_link_endpoint": (qa["crops_per_link_endpoint"], 6),
        "qa.contact_sheet_rows": (qa["contact_sheet_rows"], 3),
        "qa.contact_sheet_columns": (qa["contact_sheet_columns"], 3),
        "qa.candidate_rank_scope": (
            qa["candidate_rank_scope"],
            "persisted_candidate_graph",
        ),
        "qa.candidate_rank_tie_break": (
            qa["candidate_rank_tie_break"],
            "appearance_cosine_desc_stable_id_asc",
        ),
        "qa.margin_semantics": (qa["margin_semantics"], "cosine_not_probability"),
        "qa.recompute_c_grade_iou_from_rescue_samples": (
            qa["recompute_c_grade_iou_from_rescue_samples"],
            True,
        ),
        "qa.c_grade_high_overlap_threshold": (
            qa["c_grade_high_overlap_threshold"],
            0.25,
        ),
        "qa.unavailable_metrics_are_null": (
            qa["unavailable_metrics_are_null"],
            True,
        ),
        "render.output_width": (render["output_width"], 1920),
        "render.output_height": (render["output_height"], 1080),
        "render.fps_numerator": (render["fps_numerator"], 30000),
        "render.fps_denominator": (render["fps_denominator"], 1001),
        "render.bbox_scale": (render["bbox_scale"], 0.5),
        "render.autorotate": (render["autorotate"], False),
        "render.include_audio": (render["include_audio"], False),
        "render.label_field": (render["label_field"], "display_global_id"),
        "render.color_policy": (
            render["color_policy"],
            "deterministic_by_global_track_id",
        ),
        "render.draw_invalid_identity": (render["draw_invalid_identity"], False),
        "encoder.ffmpeg_binary": (encoder["ffmpeg_binary"], "ffmpeg"),
        "encoder.ffprobe_binary": (encoder["ffprobe_binary"], "ffprobe"),
        "encoder.codec": (encoder["codec"], "h264_nvenc"),
        "encoder.required_cuda_visible_devices": (
            encoder["required_cuda_visible_devices"],
            "1",
        ),
        "encoder.logical_gpu": (encoder["logical_gpu"], 0),
        "encoder.preset": (encoder["preset"], "p4"),
        "encoder.cq": (encoder["cq"], 21),
        "encoder.pixel_format": (encoder["pixel_format"], "yuv420p"),
        "runtime.progress_interval_sec": (runtime["progress_interval_sec"], 10.0),
        "runtime.deterministic_row_sort": (
            runtime["deterministic_row_sort"],
            True,
        ),
        "runtime.row_sort_keys": (
            runtime["row_sort_keys"],
            ["clip_order", "csv_row_index"],
        ),
        "runtime.atomic_commit": (runtime["atomic_commit"], True),
        "runtime.resume_revalidate": (runtime["resume_revalidate"], True),
        "runtime.log_flush": (runtime["log_flush"], True),
        "runtime.validate_upstream_hashes": (
            runtime["validate_upstream_hashes"],
            True,
        ),
        "runtime.fingerprint_all_outputs": (
            runtime["fingerprint_all_outputs"],
            True,
        ),
    }
    for label, (actual, expected) in expected_scalars.items():
        _fixed(actual, expected, label)

    for clip, expected in zip(
        EXPECTED_CLIP_ORDER, EXPECTED_FRAME_COUNTS, strict=True
    ):
        _fixed(frame_counts[clip], expected, f"inputs.frame_counts_by_clip.{clip}")
    for label, mapping, expected_values in (
        (
            "total_detections_by_clip",
            total_by_clip,
            EXPECTED_TOTAL_DETECTIONS_BY_CLIP,
        ),
        (
            "valid_detections_by_clip",
            valid_by_clip,
            EXPECTED_VALID_DETECTIONS_BY_CLIP,
        ),
        (
            "invalid_detections_by_clip",
            invalid_by_clip,
            EXPECTED_INVALID_DETECTIONS_BY_CLIP,
        ),
    ):
        for clip, expected in zip(EXPECTED_CLIP_ORDER, expected_values, strict=True):
            _fixed(mapping[clip], expected, f"inputs.{label}.{clip}")

    optional_counts = {
        "expected_microtrack_count": inputs["expected_microtrack_count"],
        "expected_stable_track_count": inputs["expected_stable_track_count"],
        "expected_selected_link_count": inputs["expected_selected_link_count"],
        "expected_candidate_edge_count": inputs["expected_candidate_edge_count"],
        "expected_maximum_feasible_link_count": inputs[
            "expected_maximum_feasible_link_count"
        ],
        "expected_max_concurrent_stable_track_count": inputs[
            "expected_max_concurrent_stable_track_count"
        ],
        "expected_backbone_chain_count": inputs["expected_backbone_chain_count"],
        "expected_selected_prior_link_count": inputs[
            "expected_selected_prior_link_count"
        ],
        "expected_rescue_candidate_count": inputs["expected_rescue_candidate_count"],
        "expected_rescue_embedding_count": inputs["expected_rescue_embedding_count"],
        "expected_rescue_stable_track_count": inputs[
            "expected_rescue_stable_track_count"
        ],
        "expected_selected_best_degraded_crop_count": inputs[
            "expected_selected_best_degraded_crop_count"
        ],
        "expected_selected_review_excluded_crop_count": inputs[
            "expected_selected_review_excluded_crop_count"
        ],
    }
    positive_optional_counts = {
        "expected_microtrack_count",
        "expected_stable_track_count",
        "expected_max_concurrent_stable_track_count",
        "expected_backbone_chain_count",
    }
    for name, value in optional_counts.items():
        _optional_integer(
            value, name, minimum=1 if name in positive_optional_counts else 0
        )
    for grade in APPEARANCE_GRADES:
        _optional_integer(grades[grade], f"appearance grade count {grade}")

    thresholds = qa["low_confidence_thresholds"]
    _fixed(thresholds, [0.3, 0.4, 0.5, 0.6], "qa.low_confidence_thresholds")

    if sum(total_by_clip.values()) != inputs["expected_total_detection_count"]:
        raise ContractError("S06 per-clip total detections do not sum to the total")
    if sum(valid_by_clip.values()) != inputs["expected_valid_detection_count"]:
        raise ContractError("S06 per-clip valid detections do not sum to the total")
    if sum(invalid_by_clip.values()) != inputs["expected_invalid_detection_count"]:
        raise ContractError("S06 per-clip invalid detections do not sum to the total")
    for clip in EXPECTED_CLIP_ORDER:
        if total_by_clip[clip] != valid_by_clip[clip] + invalid_by_clip[clip]:
            raise ContractError(f"S06 detection partition mismatch for {clip}")
    if inputs["expected_total_detection_count"] != (
        inputs["expected_valid_detection_count"]
        + inputs["expected_invalid_detection_count"]
    ):
        raise ContractError("S06 valid/invalid detections do not partition all rows")
    if (
        inputs["expected_stable_track_count"] is not None
        and inputs["expected_selected_link_count"] is not None
        and inputs["expected_stable_track_count"]
        - inputs["expected_selected_link_count"]
        != inputs["expected_global_track_count"]
    ):
        raise ContractError("S06 path-cover counts are inconsistent")
    if identity["population_overflow"] != inputs["expected_global_track_count"] - identity["population_soft_max"]:
        raise ContractError("S06 population overflow is inconsistent")

    artifact_expected = {
        "detections_csv": "detections_with_global_id.csv",
        "qa_metrics": "qa_metrics.json",
        "global_track_summary": "qa/global_track_summary.csv",
        "low_confidence_links": "qa/low_confidence_links.html",
        "low_confidence_assets_dir": "qa/low_confidence_assets",
        "contact_sheets_dir": "qa/contact_sheets",
        "videos_dir": "qa/videos",
        "effective_config": "effective_config.json",
        "success": "_SUCCESS.json",
    }
    artifact_paths: list[str] = []
    for name, expected in artifact_expected.items():
        actual = _relative_artifact(artifact_data[name], f"artifacts.{name}")
        _fixed(actual, expected, f"artifacts.{name}")
        artifact_paths.append(actual)
    expected_videos = tuple(
        f"qa/videos/{clip}_tracked.mp4" for clip in EXPECTED_CLIP_ORDER
    )
    videos: list[str] = []
    for clip, expected in zip(EXPECTED_CLIP_ORDER, expected_videos, strict=True):
        actual = _relative_artifact(video_paths[clip], f"artifacts.videos_by_clip.{clip}")
        _fixed(actual, expected, f"artifacts.videos_by_clip.{clip}")
        videos.append(actual)
    if len(set((*artifact_paths, *videos))) != len(artifact_paths) + len(videos):
        raise ContractError("S06 artifact paths must be unique")

    artifacts = S06Artifacts(
        detections_csv=artifact_data["detections_csv"],
        qa_metrics=artifact_data["qa_metrics"],
        global_track_summary=artifact_data["global_track_summary"],
        low_confidence_links=artifact_data["low_confidence_links"],
        low_confidence_assets_dir=artifact_data["low_confidence_assets_dir"],
        contact_sheets_dir=artifact_data["contact_sheets_dir"],
        videos_dir=artifact_data["videos_dir"],
        videos_by_clip=tuple(videos),
        effective_config=artifact_data["effective_config"],
        success=artifact_data["success"],
    )
    config = S06ExportConfig(
        schema_version="1.0",
        random_seed=20260710,
        execution_mode=str(pipeline["execution_mode"]),
        required_upstream_stage=str(pipeline["required_upstream_stage"]),
        identity_source=str(pipeline["identity_source"]),
        authorization_basis=str(pipeline["authorization_basis"]),
        id_status=str(pipeline["id_status"]),
        certification_claimed=False,
        probability_from_cosine_allowed=False,
        expected_sequence_id=str(inputs["expected_sequence_id"]),
        clip_order=EXPECTED_CLIP_ORDER,
        expected_frame_count=_integer(inputs["expected_frame_count"], "expected_frame_count"),
        frame_counts_by_clip=tuple(int(frame_counts[clip]) for clip in EXPECTED_CLIP_ORDER),
        expected_total_detection_count=_integer(inputs["expected_total_detection_count"], "expected_total_detection_count"),
        expected_valid_detection_count=_integer(inputs["expected_valid_detection_count"], "expected_valid_detection_count"),
        expected_invalid_detection_count=_integer(inputs["expected_invalid_detection_count"], "expected_invalid_detection_count"),
        expected_total_detections_by_clip=tuple(int(total_by_clip[clip]) for clip in EXPECTED_CLIP_ORDER),
        expected_valid_detections_by_clip=tuple(int(valid_by_clip[clip]) for clip in EXPECTED_CLIP_ORDER),
        expected_invalid_detections_by_clip=tuple(int(invalid_by_clip[clip]) for clip in EXPECTED_CLIP_ORDER),
        expected_microtrack_count=_optional_integer(inputs["expected_microtrack_count"], "expected_microtrack_count", minimum=1),
        expected_stable_track_count=_optional_integer(inputs["expected_stable_track_count"], "expected_stable_track_count", minimum=1),
        expected_global_track_count=_integer(inputs["expected_global_track_count"], "expected_global_track_count"),
        expected_selected_link_count=_optional_integer(inputs["expected_selected_link_count"], "expected_selected_link_count"),
        expected_candidate_edge_count=_optional_integer(inputs["expected_candidate_edge_count"], "expected_candidate_edge_count"),
        expected_maximum_feasible_link_count=_optional_integer(inputs["expected_maximum_feasible_link_count"], "expected_maximum_feasible_link_count"),
        expected_max_concurrent_stable_track_count=_optional_integer(inputs["expected_max_concurrent_stable_track_count"], "expected_max_concurrent_stable_track_count", minimum=1),
        expected_backbone_chain_count=_optional_integer(inputs["expected_backbone_chain_count"], "expected_backbone_chain_count", minimum=1),
        expected_selected_prior_link_count=_optional_integer(inputs["expected_selected_prior_link_count"], "expected_selected_prior_link_count"),
        expected_rescue_candidate_count=_optional_integer(inputs["expected_rescue_candidate_count"], "expected_rescue_candidate_count"),
        expected_rescue_embedding_count=_optional_integer(inputs["expected_rescue_embedding_count"], "expected_rescue_embedding_count"),
        expected_rescue_stable_track_count=_optional_integer(inputs["expected_rescue_stable_track_count"], "expected_rescue_stable_track_count"),
        expected_selected_best_degraded_crop_count=_optional_integer(inputs["expected_selected_best_degraded_crop_count"], "expected_selected_best_degraded_crop_count"),
        expected_selected_review_excluded_crop_count=_optional_integer(inputs["expected_selected_review_excluded_crop_count"], "expected_selected_review_excluded_crop_count"),
        expected_grade_a_clean_count=_optional_integer(grades["A_CLEAN"], "grade A count"),
        expected_grade_b_existing_degraded_count=_optional_integer(grades["B_EXISTING_DEGRADED"], "grade B count"),
        expected_grade_c_reencoded_degraded_count=_optional_integer(grades["C_REENCODED_DEGRADED"], "grade C count"),
        expected_cycle_count=_integer(inputs["expected_cycle_count"], "cycle count"),
        expected_same_frame_violation_count=_integer(inputs["expected_same_frame_violation_count"], "same-frame violation count"),
        expected_temporal_overlap_violation_count=_integer(inputs["expected_temporal_overlap_violation_count"], "temporal-overlap violation count"),
        expected_unassigned_valid_detection_count=_integer(inputs["expected_unassigned_valid_detection_count"], "unassigned valid detection count"),
        num_confirmed_ids=_integer(identity["num_confirmed_ids"], "confirmed IDs"),
        num_provisional_ids=_integer(identity["num_provisional_ids"], "provisional IDs"),
        population_soft_max=_integer(identity["population_soft_max"], "population soft max"),
        population_overflow=_integer(identity["population_overflow"], "population overflow"),
        display_id_prefix=str(identity["display_id_prefix"]),
        display_id_width=_integer(identity["display_id_width"], "display ID width", minimum=1),
        invalid_identity_fields_null=True,
        reference_mode=str(qa["reference_mode"]),
        low_confidence_thresholds=tuple(float(value) for value in thresholds),
        html_priority_threshold=float(qa["html_priority_threshold"]),
        html_maximum_threshold=float(qa["html_maximum_threshold"]),
        crops_per_link_endpoint=_integer(qa["crops_per_link_endpoint"], "crops per link endpoint", minimum=1),
        contact_sheet_rows=_integer(qa["contact_sheet_rows"], "contact sheet rows", minimum=1),
        contact_sheet_columns=_integer(qa["contact_sheet_columns"], "contact sheet columns", minimum=1),
        candidate_rank_scope=str(qa["candidate_rank_scope"]),
        candidate_rank_tie_break=str(qa["candidate_rank_tie_break"]),
        margin_semantics=str(qa["margin_semantics"]),
        recompute_c_grade_iou_from_rescue_samples=True,
        c_grade_high_overlap_threshold=float(qa["c_grade_high_overlap_threshold"]),
        unavailable_metrics_are_null=True,
        output_width=_integer(render["output_width"], "output width", minimum=1),
        output_height=_integer(render["output_height"], "output height", minimum=1),
        fps_numerator=_integer(render["fps_numerator"], "fps numerator", minimum=1),
        fps_denominator=_integer(render["fps_denominator"], "fps denominator", minimum=1),
        bbox_scale=float(render["bbox_scale"]),
        autorotate=False,
        include_audio=False,
        label_field=str(render["label_field"]),
        color_policy=str(render["color_policy"]),
        draw_invalid_identity=False,
        ffmpeg_binary=str(encoder["ffmpeg_binary"]),
        ffprobe_binary=str(encoder["ffprobe_binary"]),
        codec=str(encoder["codec"]),
        required_cuda_visible_devices=str(encoder["required_cuda_visible_devices"]),
        logical_gpu=_integer(encoder["logical_gpu"], "logical GPU"),
        preset=str(encoder["preset"]),
        cq=_integer(encoder["cq"], "CQ"),
        pixel_format=str(encoder["pixel_format"]),
        progress_interval_sec=_finite(runtime["progress_interval_sec"], "progress interval", positive=True),
        deterministic_row_sort=True,
        row_sort_keys=tuple(str(value) for value in runtime["row_sort_keys"]),
        atomic_commit=True,
        resume_revalidate=True,
        log_flush=True,
        validate_upstream_hashes=True,
        fingerprint_all_outputs=True,
        artifacts=artifacts,
    )
    canonical = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return config, payload, hashlib.sha256(canonical).hexdigest()


__all__ = [
    "APPEARANCE_GRADES",
    "EXPECTED_CLIP_ORDER",
    "EXPECTED_INVALID_DETECTIONS",
    "EXPECTED_INVALID_DETECTIONS_BY_CLIP",
    "EXPECTED_SEQUENCE_ID",
    "EXPECTED_TOTAL_DETECTIONS",
    "EXPECTED_TOTAL_DETECTIONS_BY_CLIP",
    "EXPECTED_VALID_DETECTIONS",
    "EXPECTED_VALID_DETECTIONS_BY_CLIP",
    "S06Artifacts",
    "S06ExportConfig",
    "load_s06_export_config",
]
