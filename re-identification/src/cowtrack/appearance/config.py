"""Strict, dataset-fixed configuration contract for S02 appearance extraction."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from cowtrack.config import ContractError


_SHA1_RE = re.compile(r"[0-9a-f]{40}\Z")
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


@dataclass(frozen=True)
class AppearanceModelProfile:
    """The immutable encoder/preprocessing profile used by S02."""

    name: str
    kind: str
    architecture: str
    checkpoint_path: Path
    checkpoint_filter: str
    input_size: int
    resize_mode: str
    crop_pct: float
    interpolation: str
    normalization_profile: str
    expected_revision: str
    expected_weight_sha256: str


@dataclass(frozen=True)
class AppearanceArtifacts:
    """Canonical S02 artifact basenames."""

    appearance_samples: str
    sample_embeddings: str
    appearance_exclusions: str
    micro_prototypes: str
    micro_prototype_mask: str
    micro_appearance: str
    micro_context: str
    encoder_choice: str
    encoder_benchmark: str
    effective_config: str
    appearance_report: str
    success: str


@dataclass(frozen=True)
class AppearanceConfig:
    """Validated S02 settings exposed as a stage-friendly flat contract."""

    schema_version: str
    random_seed: int
    selection_mode: str
    model_profiles: tuple[AppearanceModelProfile, ...]

    head_samples: int
    tail_samples: int
    representative_period_sec: float
    max_samples_per_microtrack: int
    min_samples_per_microtrack: int
    candidate_pool_period_sec: float
    candidate_pool_max_samples_per_microtrack: int
    candidate_pool_endpoint_samples: int
    max_other_bbox_iou_preferred: float
    bbox_padding_ratio: float
    rotations: tuple[int, ...]
    horizontal_flip: bool
    soft_border_mask: bool

    min_crop_quality: float
    quality_other_bbox_iou_weight: float
    quality_clipped_fraction_weight: float
    quality_bbox_area_percentile_weight: float
    quality_blur_score_weight: float
    quality_boundary_distance_weight: float
    quality_blur_scale: float
    quality_blur_measurement_size: int

    prototypes_per_tracklet: int
    prototype_outlier_cosine_min: float
    prototype_outlier_support_cosine: float
    prototype_cluster_min_separation: float

    review_event_half_window_frames: int
    review_event_reasons: tuple[str, ...]

    device: str
    batch_size: int
    progress_interval_sec: float
    parquet_compression: str
    artifacts: AppearanceArtifacts

    @property
    def quality_weights(self) -> dict[str, float]:
        """Return metric weights under their persisted metric names."""

        return {
            "other_bbox_iou": self.quality_other_bbox_iou_weight,
            "clipped_fraction": self.quality_clipped_fraction_weight,
            "bbox_area_percentile": self.quality_bbox_area_percentile_weight,
            "blur_score": self.quality_blur_score_weight,
            "boundary_distance": self.quality_boundary_distance_weight,
        }


LOCKED_SELECTION_MODE = "locked_existing_winner"
LOCKED_ENCODER_PROFILE = "megadescriptor_l_384_imagenet"


_FIXED_PROFILES: tuple[dict[str, Any], ...] = (
    {
        "name": LOCKED_ENCODER_PROFILE,
        "kind": "timm",
        "architecture": "swin_large_patch4_window12_384",
        "checkpoint_path": Path(
            "/home/hyw/re-identification-models/"
            "MegaDescriptor-L-384/pytorch_model.bin"
        ),
        "checkpoint_filter": "timm_swin",
        "input_size": 384,
        "resize_mode": "shortest_edge_center_crop",
        "crop_pct": 0.9,
        "interpolation": "bicubic",
        "normalization_profile": "imagenet",
        "expected_revision": "33b3c6f4ee0c386a4126cc3dcd23843920613fa1",
        "expected_weight_sha256": (
            "ccfe757f50f7984a115ffe00921cd4c09e260e645215463b23814586040227a3"
        ),
    },
)


_FIXED_ARTIFACTS = {
    "appearance_samples": "appearance_samples.parquet",
    "sample_embeddings": "sample_embeddings.f16.npy",
    "appearance_exclusions": "appearance_exclusions.parquet",
    "micro_prototypes": "micro_prototypes.f16.npy",
    "micro_prototype_mask": "micro_prototype_mask.npy",
    "micro_appearance": "micro_appearance.parquet",
    "micro_context": "micro_context.parquet",
    "encoder_choice": "encoder_choice.json",
    "encoder_benchmark": "encoder_benchmark.json",
    "effective_config": "effective_config.json",
    "appearance_report": "appearance_report.json",
    "success": "_SUCCESS.json",
}


_FIXED_REVIEW_REASONS = (
    "low_purity",
    "large_center_jump",
    "legacy_id_transition",
)


def _exact_keys(
    mapping: object, expected: set[str], context: str
) -> dict[str, Any]:
    if not isinstance(mapping, dict):
        raise ContractError(f"{context} must be a mapping")
    actual = set(mapping)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ContractError(
            f"{context} config keys mismatch; missing={missing}, extra={extra}"
        )
    if any(not isinstance(key, str) for key in mapping):
        raise ContractError(f"{context} config keys must be strings")
    return mapping


def _string(mapping: dict[str, Any], key: str, context: str) -> str:
    value = mapping[key]
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise ContractError(f"{context}.{key} must be a non-blank string")
    return value


def _integer(mapping: dict[str, Any], key: str, context: str) -> int:
    value = mapping[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ContractError(f"{context}.{key} must be an integer")
    return value


def _finite(mapping: dict[str, Any], key: str, context: str) -> float:
    value = mapping[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"{context}.{key} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ContractError(f"{context}.{key} must be a finite number")
    return result


def _boolean(mapping: dict[str, Any], key: str, context: str) -> bool:
    value = mapping[key]
    if not isinstance(value, bool):
        raise ContractError(f"{context}.{key} must be a boolean")
    return value


def _string_list(mapping: dict[str, Any], key: str, context: str) -> tuple[str, ...]:
    value = mapping[key]
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, str) or not item for item in value)
    ):
        raise ContractError(f"{context}.{key} must be a non-empty string list")
    return tuple(value)


def _parse_profiles(value: object) -> tuple[AppearanceModelProfile, ...]:
    if not isinstance(value, list) or len(value) != len(_FIXED_PROFILES):
        raise ContractError(
            "models.profiles must contain only the locked MegaDescriptor profile"
        )
    expected_keys = {
        "name",
        "kind",
        "architecture",
        "checkpoint_path",
        "checkpoint_filter",
        "input_size",
        "resize_mode",
        "crop_pct",
        "interpolation",
        "normalization_profile",
        "expected_revision",
        "expected_weight_sha256",
    }
    profiles: list[AppearanceModelProfile] = []
    for index, (raw_profile, fixed) in enumerate(zip(value, _FIXED_PROFILES)):
        context = f"models.profiles[{index}]"
        profile = _exact_keys(raw_profile, expected_keys, context)
        checkpoint_text = _string(profile, "checkpoint_path", context)
        checkpoint_path = Path(checkpoint_text)
        if not checkpoint_path.is_absolute():
            raise ContractError(f"{context}.checkpoint_path must be absolute")
        parsed = AppearanceModelProfile(
            name=_string(profile, "name", context),
            kind=_string(profile, "kind", context),
            architecture=_string(profile, "architecture", context),
            checkpoint_path=checkpoint_path,
            checkpoint_filter=_string(profile, "checkpoint_filter", context),
            input_size=_integer(profile, "input_size", context),
            resize_mode=_string(profile, "resize_mode", context),
            crop_pct=_finite(profile, "crop_pct", context),
            interpolation=_string(profile, "interpolation", context),
            normalization_profile=_string(
                profile, "normalization_profile", context
            ),
            expected_revision=_string(profile, "expected_revision", context),
            expected_weight_sha256=_string(
                profile, "expected_weight_sha256", context
            ),
        )
        for field_name, actual in (
            ("name", parsed.name),
            ("kind", parsed.kind),
            ("architecture", parsed.architecture),
            ("checkpoint_path", parsed.checkpoint_path),
            ("checkpoint_filter", parsed.checkpoint_filter),
            ("input_size", parsed.input_size),
            ("resize_mode", parsed.resize_mode),
            ("crop_pct", parsed.crop_pct),
            ("interpolation", parsed.interpolation),
            ("normalization_profile", parsed.normalization_profile),
            ("expected_revision", parsed.expected_revision),
            ("expected_weight_sha256", parsed.expected_weight_sha256),
        ):
            if actual != fixed[field_name]:
                raise ContractError(
                    f"{context}.{field_name} must be fixed to {fixed[field_name]!s}"
                )
        if not _SHA1_RE.fullmatch(parsed.expected_revision):
            raise ContractError(f"{context}.expected_revision must be lowercase SHA-1")
        if not _SHA256_RE.fullmatch(parsed.expected_weight_sha256):
            raise ContractError(
                f"{context}.expected_weight_sha256 must be lowercase SHA-256"
            )
        if not parsed.checkpoint_path.is_file():
            raise ContractError(
                f"fixed S02 checkpoint does not exist: {parsed.checkpoint_path}"
            )
        profiles.append(parsed)
    return tuple(profiles)


def load_appearance_config(
    path: Path,
) -> tuple[AppearanceConfig, dict[str, Any], str]:
    """Load and validate the fixed S02 config without loading model weights."""

    path = path.resolve()
    if not path.is_file():
        raise ContractError(f"S02 appearance config does not exist: {path}")
    try:
        raw = path.read_bytes()
        payload = yaml.safe_load(raw)
    except (OSError, yaml.YAMLError) as exc:
        raise ContractError(f"cannot read S02 appearance config {path}: {exc}") from exc

    top = _exact_keys(
        payload,
        {
            "pipeline",
            "models",
            "sampling",
            "quality",
            "prototypes",
            "review_exclusion",
            "runtime",
            "artifacts",
        },
        "S02 appearance config",
    )
    pipeline = _exact_keys(
        top["pipeline"], {"schema_version", "random_seed"}, "pipeline"
    )
    models = _exact_keys(
        top["models"], {"selection_mode", "profiles"}, "models"
    )
    sampling = _exact_keys(
        top["sampling"],
        {
            "head_samples",
            "tail_samples",
            "representative_period_sec",
            "max_samples_per_microtrack",
            "min_samples_per_microtrack",
            "candidate_pool_period_sec",
            "candidate_pool_max_samples_per_microtrack",
            "candidate_pool_endpoint_samples",
            "max_other_bbox_iou_preferred",
            "bbox_padding_ratio",
            "rotations",
            "horizontal_flip",
            "soft_border_mask",
        },
        "sampling",
    )
    quality = _exact_keys(
        top["quality"],
        {"min_crop_quality", "metric_weights", "blur_scale", "blur_measurement_size"},
        "quality",
    )
    metric_weights = _exact_keys(
        quality["metric_weights"],
        {
            "other_bbox_iou",
            "clipped_fraction",
            "bbox_area_percentile",
            "blur_score",
            "boundary_distance",
        },
        "quality.metric_weights",
    )
    prototypes = _exact_keys(
        top["prototypes"],
        {
            "count",
            "outlier_cosine_min",
            "outlier_support_cosine",
            "cluster_min_separation",
        },
        "prototypes",
    )
    review = _exact_keys(
        top["review_exclusion"],
        {"event_half_window_frames", "event_reasons"},
        "review_exclusion",
    )
    runtime = _exact_keys(
        top["runtime"],
        {"device", "batch_size", "progress_interval_sec", "parquet_compression"},
        "runtime",
    )
    artifact_payload = _exact_keys(
        top["artifacts"], set(_FIXED_ARTIFACTS), "artifacts"
    )

    rotations_value = sampling["rotations"]
    if (
        not isinstance(rotations_value, list)
        or any(isinstance(value, bool) or not isinstance(value, int) for value in rotations_value)
    ):
        raise ContractError("sampling.rotations must be an integer list")

    artifacts = AppearanceArtifacts(
        **{
            key: _string(artifact_payload, key, "artifacts")
            for key in _FIXED_ARTIFACTS
        }
    )
    result = AppearanceConfig(
        schema_version=_string(pipeline, "schema_version", "pipeline"),
        random_seed=_integer(pipeline, "random_seed", "pipeline"),
        selection_mode=_string(models, "selection_mode", "models"),
        model_profiles=_parse_profiles(models["profiles"]),
        head_samples=_integer(sampling, "head_samples", "sampling"),
        tail_samples=_integer(sampling, "tail_samples", "sampling"),
        representative_period_sec=_finite(
            sampling, "representative_period_sec", "sampling"
        ),
        max_samples_per_microtrack=_integer(
            sampling, "max_samples_per_microtrack", "sampling"
        ),
        min_samples_per_microtrack=_integer(
            sampling, "min_samples_per_microtrack", "sampling"
        ),
        candidate_pool_period_sec=_finite(
            sampling, "candidate_pool_period_sec", "sampling"
        ),
        candidate_pool_max_samples_per_microtrack=_integer(
            sampling, "candidate_pool_max_samples_per_microtrack", "sampling"
        ),
        candidate_pool_endpoint_samples=_integer(
            sampling, "candidate_pool_endpoint_samples", "sampling"
        ),
        max_other_bbox_iou_preferred=_finite(
            sampling, "max_other_bbox_iou_preferred", "sampling"
        ),
        bbox_padding_ratio=_finite(sampling, "bbox_padding_ratio", "sampling"),
        rotations=tuple(rotations_value),
        horizontal_flip=_boolean(sampling, "horizontal_flip", "sampling"),
        soft_border_mask=_boolean(sampling, "soft_border_mask", "sampling"),
        min_crop_quality=_finite(quality, "min_crop_quality", "quality"),
        quality_other_bbox_iou_weight=_finite(
            metric_weights, "other_bbox_iou", "quality.metric_weights"
        ),
        quality_clipped_fraction_weight=_finite(
            metric_weights, "clipped_fraction", "quality.metric_weights"
        ),
        quality_bbox_area_percentile_weight=_finite(
            metric_weights, "bbox_area_percentile", "quality.metric_weights"
        ),
        quality_blur_score_weight=_finite(
            metric_weights, "blur_score", "quality.metric_weights"
        ),
        quality_boundary_distance_weight=_finite(
            metric_weights, "boundary_distance", "quality.metric_weights"
        ),
        quality_blur_scale=_finite(quality, "blur_scale", "quality"),
        quality_blur_measurement_size=_integer(
            quality, "blur_measurement_size", "quality"
        ),
        prototypes_per_tracklet=_integer(prototypes, "count", "prototypes"),
        prototype_outlier_cosine_min=_finite(
            prototypes, "outlier_cosine_min", "prototypes"
        ),
        prototype_outlier_support_cosine=_finite(
            prototypes, "outlier_support_cosine", "prototypes"
        ),
        prototype_cluster_min_separation=_finite(
            prototypes, "cluster_min_separation", "prototypes"
        ),
        review_event_half_window_frames=_integer(
            review, "event_half_window_frames", "review_exclusion"
        ),
        review_event_reasons=_string_list(
            review, "event_reasons", "review_exclusion"
        ),
        device=_string(runtime, "device", "runtime"),
        batch_size=_integer(runtime, "batch_size", "runtime"),
        progress_interval_sec=_finite(
            runtime, "progress_interval_sec", "runtime"
        ),
        parquet_compression=_string(
            runtime, "parquet_compression", "runtime"
        ),
        artifacts=artifacts,
    )

    if result.schema_version != "1.0":
        raise ContractError("fixed S02 requires pipeline.schema_version=1.0")
    if result.random_seed != 20260710:
        raise ContractError("fixed S02 requires pipeline.random_seed=20260710")
    if result.selection_mode != LOCKED_SELECTION_MODE:
        raise ContractError(
            f"fixed S02 requires models.selection_mode={LOCKED_SELECTION_MODE}"
        )
    if (result.head_samples, result.tail_samples) != (3, 3):
        raise ContractError("fixed S02 requires three head and three tail samples")
    if not math.isclose(result.representative_period_sec, 0.75):
        raise ContractError("fixed S02 requires representative_period_sec=0.75")
    if result.max_samples_per_microtrack != 24:
        raise ContractError("fixed S02 requires max_samples_per_microtrack=24")
    if result.min_samples_per_microtrack != 3:
        raise ContractError("fixed S02 requires min_samples_per_microtrack=3")
    if not math.isclose(
        result.candidate_pool_period_sec, 0.375, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ContractError("fixed S02 requires candidate_pool_period_sec=0.375")
    if result.candidate_pool_max_samples_per_microtrack != 48:
        raise ContractError(
            "fixed S02 requires candidate_pool_max_samples_per_microtrack=48"
        )
    if result.candidate_pool_endpoint_samples != 6:
        raise ContractError("fixed S02 requires candidate_pool_endpoint_samples=6")
    if not math.isclose(result.max_other_bbox_iou_preferred, 0.25):
        raise ContractError("fixed S02 requires max_other_bbox_iou_preferred=0.25")
    if not math.isclose(result.bbox_padding_ratio, 0.05):
        raise ContractError("fixed S02 requires bbox_padding_ratio=0.05")
    if result.rotations != (0, 180):
        raise ContractError("fixed S02 requires sampling.rotations=[0, 180]")
    if result.horizontal_flip:
        raise ContractError("fixed S02 requires sampling.horizontal_flip=false")
    if not result.soft_border_mask:
        raise ContractError("fixed S02 requires sampling.soft_border_mask=true")

    if not math.isclose(
        result.min_crop_quality, 0.45, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ContractError("fixed S02 requires quality.min_crop_quality=0.45")
    weights = tuple(result.quality_weights.values())
    if any(
        not math.isclose(observed, expected, rel_tol=0.0, abs_tol=1e-12)
        for observed, expected in zip(
            weights, (0.30, 0.25, 0.15, 0.20, 0.10), strict=True
        )
    ):
        raise ContractError(
            "fixed S02 quality metric weights must be 0.30/0.25/0.15/0.20/0.10"
        )
    if not math.isclose(
        result.quality_blur_scale, 380.0, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ContractError("fixed S02 requires quality.blur_scale=380.0")
    if result.quality_blur_measurement_size != 256:
        raise ContractError("fixed S02 requires quality.blur_measurement_size=256")

    if result.prototypes_per_tracklet != 3:
        raise ContractError("fixed S02 requires prototypes.count=3")
    if not math.isclose(
        result.prototype_outlier_cosine_min, 0.65, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ContractError("fixed S02 requires prototypes.outlier_cosine_min=0.65")
    if not math.isclose(
        result.prototype_outlier_support_cosine,
        0.70,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ContractError(
            "fixed S02 requires prototypes.outlier_support_cosine=0.70"
        )
    if not math.isclose(
        result.prototype_cluster_min_separation,
        0.08,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ContractError(
            "fixed S02 requires prototypes.cluster_min_separation=0.08"
        )

    if result.review_event_half_window_frames != 15:
        raise ContractError(
            "fixed S02 requires review_exclusion.event_half_window_frames=15"
        )
    if result.review_event_reasons != _FIXED_REVIEW_REASONS:
        raise ContractError(
            "fixed S02 review exclusions require the three canonical event reasons"
        )
    if result.device != "cuda:0":
        raise ContractError(
            "with CUDA_VISIBLE_DEVICES=1, fixed S02 requires runtime.device=cuda:0"
        )
    if result.batch_size != 8:
        raise ContractError("fixed S02 requires runtime.batch_size=8")
    if not math.isclose(
        result.progress_interval_sec, 10.0, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ContractError("fixed S02 requires runtime.progress_interval_sec=10.0")
    if result.parquet_compression != "zstd":
        raise ContractError("fixed S02 requires runtime.parquet_compression=zstd")

    actual_artifacts = {
        key: getattr(result.artifacts, key) for key in _FIXED_ARTIFACTS
    }
    if actual_artifacts != _FIXED_ARTIFACTS:
        raise ContractError("S02 artifact filenames are fixed and cannot be changed")
    if any(Path(name).name != name for name in actual_artifacts.values()):
        raise ContractError("S02 artifact values must be basenames")

    canonical = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return result, payload, hashlib.sha256(canonical).hexdigest()
