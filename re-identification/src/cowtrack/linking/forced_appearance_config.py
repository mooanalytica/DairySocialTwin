"""Strict dataset-specific contract for operator-forced appearance linking."""

from __future__ import annotations

import hashlib
import math
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import scipy
import yaml

from cowtrack.config import ContractError
from cowtrack.linking.dataset_contract import (
    EXPECTED_CLIP_ORDER,
    EXPECTED_FRAME_COUNT,
    EXPECTED_FRAME_COUNTS,
    EXPECTED_SEQUENCE_ID,
    MAX_GLOBAL_TRACK_COUNT,
)


@dataclass(frozen=True)
class ForcedAppearanceArtifacts:
    rescue_samples: str
    rescue_embeddings: str
    graded_stable_appearance: str
    graded_prototypes: str
    graded_prototype_mask: str
    candidate_edges: str
    stable_to_global: str
    global_tracks: str
    det_to_global: str
    report: str
    effective_config: str
    success: str


@dataclass(frozen=True)
class ForcedAppearanceConfig:
    schema_version: str
    random_seed: int
    execution_mode: str
    expected_sequence_id: str
    expected_stable_track_count: int | None
    expected_microtrack_count: int | None
    expected_valid_detection_count: int
    expected_invalid_detection_count: int
    expected_frame_count: int
    expected_clean_appearance_count: int | None
    expected_missing_clean_appearance_count: int | None
    expected_no_s02_sample_count: int | None
    expected_prior_link_count: int | None
    clip_order: tuple[str, ...]
    frame_counts_by_clip: tuple[int, ...]
    expected_valid_detections_by_clip: tuple[int, ...]
    selected_encoder: str
    embedding_dim: int
    device: str
    batch_size: int
    clean_max_other_bbox_iou: float
    max_prototypes: int
    new_prototype_cosine: float
    rescue_candidate_period_sec: float
    rescue_candidate_max_samples: int
    rescue_candidate_endpoint_samples: int
    rescue_samples_per_stable: int
    bbox_padding_ratio: float
    blur_measurement_size: int
    blur_scale: float
    min_clean_crop_quality: float
    quality_weights: tuple[float, ...]
    target_global_track_count: int
    source_top_k: int
    target_top_k: int
    appearance_cost_scale: int
    grade_penalty: Mapping[str, int]
    prior_link_bonus: int
    expected_python_version: str
    expected_scipy_version: str
    progress_interval_sec: float
    parquet_compression: str
    artifacts: ForcedAppearanceArtifacts

    @property
    def expected_detection_count(self) -> int:
        """Structural alias used by the shared global mapping builder."""

        return self.expected_valid_detection_count


_TOP_KEYS = {"pipeline", "inputs", "appearance", "solver", "runtime", "artifacts"}
_PIPELINE_KEYS = {
    "schema_version", "random_seed", "execution_mode", "operator_approved",
    "certification_claim_allowed",
}
_INPUT_KEYS = {
    "expected_sequence_id", "expected_stable_track_count", "expected_microtrack_count",
    "expected_valid_detection_count", "expected_invalid_detection_count",
    "expected_frame_count", "expected_clean_appearance_count",
    "expected_missing_clean_appearance_count", "expected_prior_link_count",
    "expected_no_s02_sample_count",
    "clip_order", "frame_counts_by_clip", "valid_detections_by_clip",
}
_APPEARANCE_KEYS = {
    "selected_encoder", "embedding_dim", "device", "batch_size",
    "clean_max_other_bbox_iou", "max_prototypes", "new_prototype_cosine",
    "rescue_candidate_period_sec", "rescue_candidate_max_samples",
    "rescue_candidate_endpoint_samples", "rescue_samples_per_stable",
    "bbox_padding_ratio", "blur_measurement_size", "blur_scale",
    "min_clean_crop_quality", "quality_weights",
    "allow_existing_degraded_embeddings", "allow_reencode_missing_embeddings",
    "allow_best_degraded_crop",
}
_SOLVER_KEYS = {
    "algorithm", "objective", "target_global_track_count", "direction",
    "require_strict_non_overlap", "source_top_k", "target_top_k",
    "include_temporal_feasibility_backbone", "prior_global_links_are_soft",
    "appearance_cost_scale", "grade_penalty", "prior_link_bonus",
}
_RUNTIME_KEYS = {
    "expected_python_version", "expected_scipy_version", "progress_interval_sec",
    "parquet_compression", "deterministic_row_sort", "atomic_commit",
    "resume_revalidate", "log_flush",
}
_ARTIFACT_KEYS = set(ForcedAppearanceArtifacts.__dataclass_fields__)
_QUALITY_KEYS = {
    "other_bbox_iou", "clipped_fraction", "bbox_area_percentile", "blur_score",
    "boundary_distance",
}
_GRADE_KEYS = {"A_CLEAN", "B_EXISTING_DEGRADED", "C_REENCODED_DEGRADED"}


def _exact(value: object, keys: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ContractError(f"{label} must be a mapping")
    if set(value) != keys:
        raise ContractError(
            f"{label} keys mismatch; missing={sorted(keys-set(value))}, "
            f"extra={sorted(set(value)-keys)}"
        )
    return value


def _fixed(actual: object, expected: object, label: str) -> None:
    if actual != expected or (isinstance(expected, bool) and type(actual) is not bool):
        raise ContractError(f"forced appearance requires {label}={expected!r}")


def _integer(value: object, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ContractError(f"{label} must be an integer >= {minimum}")
    return value


def _optional_integer(
    value: object, label: str, minimum: int = 0
) -> int | None:
    if value is None:
        return None
    return _integer(value, label, minimum)


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ContractError(f"{label} must be finite")
    return result


def load_forced_appearance_config(
    path: Path,
) -> tuple[ForcedAppearanceConfig, dict[str, Any], str]:
    path = path.resolve()
    if not path.is_file():
        raise ContractError(f"forced appearance config does not exist: {path}")
    raw = path.read_bytes()
    try:
        payload = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise ContractError(f"cannot parse forced appearance config: {exc}") from exc
    top = _exact(payload, _TOP_KEYS, "forced appearance config")
    pipeline = _exact(top["pipeline"], _PIPELINE_KEYS, "pipeline")
    inputs = _exact(top["inputs"], _INPUT_KEYS, "inputs")
    appearance = _exact(top["appearance"], _APPEARANCE_KEYS, "appearance")
    solver = _exact(top["solver"], _SOLVER_KEYS, "solver")
    runtime = _exact(top["runtime"], _RUNTIME_KEYS, "runtime")
    artifacts_raw = _exact(top["artifacts"], _ARTIFACT_KEYS, "artifacts")
    weights = _exact(appearance["quality_weights"], _QUALITY_KEYS, "quality_weights")
    penalties = _exact(solver["grade_penalty"], _GRADE_KEYS, "grade_penalty")

    fixed_values = {
        "pipeline.schema_version": (pipeline["schema_version"], "1.0"),
        "pipeline.random_seed": (pipeline["random_seed"], 20260710),
        "pipeline.execution_mode": (
            pipeline["execution_mode"], "operator_forced_appearance_exact_population"
        ),
        "pipeline.operator_approved": (pipeline["operator_approved"], True),
        "pipeline.certification_claim_allowed": (
            pipeline["certification_claim_allowed"], False
        ),
        "inputs.expected_sequence_id": (inputs["expected_sequence_id"], EXPECTED_SEQUENCE_ID),
        "inputs.expected_frame_count": (inputs["expected_frame_count"], EXPECTED_FRAME_COUNT),
        "inputs.clip_order": (inputs["clip_order"], list(EXPECTED_CLIP_ORDER)),
        "appearance.selected_encoder": (appearance["selected_encoder"], "megadescriptor_l_384_imagenet"),
        "appearance.embedding_dim": (appearance["embedding_dim"], 1536),
        "appearance.device": (appearance["device"], "cuda:0"),
        "appearance.batch_size": (appearance["batch_size"], 8),
        "appearance.clean_max_other_bbox_iou": (appearance["clean_max_other_bbox_iou"], 0.25),
        "appearance.max_prototypes": (appearance["max_prototypes"], 3),
        "appearance.new_prototype_cosine": (appearance["new_prototype_cosine"], 0.92),
        "appearance.rescue_candidate_period_sec": (appearance["rescue_candidate_period_sec"], 0.375),
        "appearance.rescue_candidate_max_samples": (appearance["rescue_candidate_max_samples"], 48),
        "appearance.rescue_candidate_endpoint_samples": (appearance["rescue_candidate_endpoint_samples"], 6),
        "appearance.rescue_samples_per_stable": (appearance["rescue_samples_per_stable"], 8),
        "appearance.bbox_padding_ratio": (appearance["bbox_padding_ratio"], 0.05),
        "appearance.blur_measurement_size": (appearance["blur_measurement_size"], 256),
        "appearance.blur_scale": (appearance["blur_scale"], 380.0),
        "appearance.min_clean_crop_quality": (appearance["min_clean_crop_quality"], 0.45),
        "appearance.allow_existing_degraded_embeddings": (appearance["allow_existing_degraded_embeddings"], True),
        "appearance.allow_reencode_missing_embeddings": (appearance["allow_reencode_missing_embeddings"], True),
        "appearance.allow_best_degraded_crop": (appearance["allow_best_degraded_crop"], True),
        "solver.algorithm": (
            solver["algorithm"],
            "deterministic_full_sequence_fixed_cardinality_min_cost_flow",
        ),
        "solver.objective": (
            solver["objective"],
            "exact_62_global_paths_then_minimum_full_sequence_appearance_cost",
        ),
        "solver.target_global_track_count": (solver["target_global_track_count"], MAX_GLOBAL_TRACK_COUNT),
        "solver.direction": (solver["direction"], "future_only"),
        "solver.require_strict_non_overlap": (solver["require_strict_non_overlap"], True),
        "solver.source_top_k": (solver["source_top_k"], 32),
        "solver.target_top_k": (solver["target_top_k"], 32),
        "solver.include_temporal_feasibility_backbone": (solver["include_temporal_feasibility_backbone"], True),
        "solver.prior_global_links_are_soft": (solver["prior_global_links_are_soft"], True),
        "solver.appearance_cost_scale": (solver["appearance_cost_scale"], 400),
        "solver.prior_link_bonus": (solver["prior_link_bonus"], 3),
        "runtime.expected_python_version": (runtime["expected_python_version"], "3.14.4"),
        "runtime.expected_scipy_version": (runtime["expected_scipy_version"], "1.18.0"),
        "runtime.parquet_compression": (runtime["parquet_compression"], "zstd"),
        "runtime.deterministic_row_sort": (runtime["deterministic_row_sort"], True),
        "runtime.atomic_commit": (runtime["atomic_commit"], True),
        "runtime.resume_revalidate": (runtime["resume_revalidate"], True),
        "runtime.log_flush": (runtime["log_flush"], True),
    }
    for label, (actual, expected) in fixed_values.items():
        _fixed(actual, expected, label)
    for clip, expected in zip(EXPECTED_CLIP_ORDER, EXPECTED_FRAME_COUNTS, strict=True):
        _fixed(inputs["frame_counts_by_clip"].get(clip), expected, f"frame_counts_by_clip.{clip}")
    if set(inputs["frame_counts_by_clip"]) != set(EXPECTED_CLIP_ORDER) or set(inputs["valid_detections_by_clip"]) != set(EXPECTED_CLIP_ORDER):
        raise ContractError("forced appearance per-clip count keys differ")
    stable_count = _optional_integer(inputs["expected_stable_track_count"], "expected_stable_track_count", 1)
    micro_count = _optional_integer(inputs["expected_microtrack_count"], "expected_microtrack_count", 1)
    valid_count = _integer(inputs["expected_valid_detection_count"], "expected_valid_detection_count", 1)
    invalid_count = _integer(inputs["expected_invalid_detection_count"], "expected_invalid_detection_count")
    clean_count = _optional_integer(inputs["expected_clean_appearance_count"], "expected_clean_appearance_count")
    missing_count = _optional_integer(inputs["expected_missing_clean_appearance_count"], "expected_missing_clean_appearance_count")
    no_sample_count = _optional_integer(inputs["expected_no_s02_sample_count"], "expected_no_s02_sample_count")
    prior_count = _optional_integer(inputs["expected_prior_link_count"], "expected_prior_link_count")
    valid_by_clip = tuple(
        _integer(inputs["valid_detections_by_clip"][clip], f"valid_detections_by_clip.{clip}")
        for clip in EXPECTED_CLIP_ORDER
    )
    if (
        (stable_count is not None and stable_count < MAX_GLOBAL_TRACK_COUNT)
        or (
            stable_count is not None
            and micro_count is not None
            and stable_count > micro_count
        )
        or (micro_count is not None and micro_count > valid_count)
        or (
            clean_count is not None
            and missing_count is not None
            and stable_count is not None
            and clean_count + missing_count != stable_count
        )
        or (
            no_sample_count is not None
            and missing_count is not None
            and no_sample_count > missing_count
        )
        or (
            prior_count is not None
            and stable_count is not None
            and prior_count > stable_count - 1
        )
        or sum(valid_by_clip) != valid_count
    ):
        raise ContractError("forced appearance expected counts are inconsistent")
    expected_weights = {
        "other_bbox_iou": 0.30, "clipped_fraction": 0.25,
        "bbox_area_percentile": 0.15, "blur_score": 0.20,
        "boundary_distance": 0.10,
    }
    expected_penalties = {"A_CLEAN": 0, "B_EXISTING_DEGRADED": 5, "C_REENCODED_DEGRADED": 15}
    for name, expected in expected_weights.items():
        _fixed(weights[name], expected, f"quality_weights.{name}")
    for name, expected in expected_penalties.items():
        _fixed(penalties[name], expected, f"grade_penalty.{name}")
    progress = _finite(runtime["progress_interval_sec"], "progress_interval_sec")
    if progress <= 0.0:
        raise ContractError("progress_interval_sec must be positive")
    if platform.python_version() != "3.14.4" or scipy.__version__ != "1.18.0":
        raise ContractError("forced appearance requires Python 3.14.4 / SciPy 1.18.0")
    if len(set(artifacts_raw.values())) != len(artifacts_raw) or any(
        not isinstance(value, str) or not value or Path(value).name != value
        for value in artifacts_raw.values()
    ):
        raise ContractError("forced appearance artifact names must be unique basenames")
    artifacts = ForcedAppearanceArtifacts(**artifacts_raw)
    config = ForcedAppearanceConfig(
        schema_version="1.0", random_seed=20260710,
        execution_mode=str(pipeline["execution_mode"]),
        expected_sequence_id=str(inputs["expected_sequence_id"]),
        expected_stable_track_count=stable_count,
        expected_microtrack_count=micro_count,
        expected_valid_detection_count=valid_count,
        expected_invalid_detection_count=invalid_count,
        expected_frame_count=EXPECTED_FRAME_COUNT,
        expected_clean_appearance_count=clean_count,
        expected_missing_clean_appearance_count=missing_count,
        expected_no_s02_sample_count=no_sample_count,
        expected_prior_link_count=prior_count,
        clip_order=EXPECTED_CLIP_ORDER,
        frame_counts_by_clip=tuple(int(inputs["frame_counts_by_clip"][clip]) for clip in EXPECTED_CLIP_ORDER),
        expected_valid_detections_by_clip=valid_by_clip,
        selected_encoder=str(appearance["selected_encoder"]),
        embedding_dim=_integer(appearance["embedding_dim"], "embedding_dim", 1),
        device=str(appearance["device"]), batch_size=_integer(appearance["batch_size"], "batch_size", 1),
        clean_max_other_bbox_iou=float(appearance["clean_max_other_bbox_iou"]),
        max_prototypes=int(appearance["max_prototypes"]),
        new_prototype_cosine=float(appearance["new_prototype_cosine"]),
        rescue_candidate_period_sec=float(appearance["rescue_candidate_period_sec"]),
        rescue_candidate_max_samples=int(appearance["rescue_candidate_max_samples"]),
        rescue_candidate_endpoint_samples=int(appearance["rescue_candidate_endpoint_samples"]),
        rescue_samples_per_stable=int(appearance["rescue_samples_per_stable"]),
        bbox_padding_ratio=float(appearance["bbox_padding_ratio"]),
        blur_measurement_size=int(appearance["blur_measurement_size"]),
        blur_scale=float(appearance["blur_scale"]),
        min_clean_crop_quality=float(appearance["min_clean_crop_quality"]),
        quality_weights=tuple(float(weights[name]) for name in (
            "other_bbox_iou", "clipped_fraction", "bbox_area_percentile",
            "blur_score", "boundary_distance"
        )),
        target_global_track_count=int(solver["target_global_track_count"]),
        source_top_k=int(solver["source_top_k"]), target_top_k=int(solver["target_top_k"]),
        appearance_cost_scale=int(solver["appearance_cost_scale"]),
        grade_penalty=dict(expected_penalties), prior_link_bonus=int(solver["prior_link_bonus"]),
        expected_python_version=str(runtime["expected_python_version"]),
        expected_scipy_version=str(runtime["expected_scipy_version"]),
        progress_interval_sec=progress, parquet_compression=str(runtime["parquet_compression"]),
        artifacts=artifacts,
    )
    return config, payload, hashlib.sha256(raw).hexdigest()


__all__ = ["ForcedAppearanceArtifacts", "ForcedAppearanceConfig", "load_forced_appearance_config"]
