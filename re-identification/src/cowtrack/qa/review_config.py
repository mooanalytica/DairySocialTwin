from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from cowtrack.config import ContractError


@dataclass(frozen=True)
class S01ReviewConfig:
    schema_version: str
    output_width: int
    output_height: int
    window_before_sec: float
    window_after_sec: float
    progress_interval_sec: float
    draw_context_boxes: bool
    long_track_min_detections: int
    long_track_chunk_sec: float
    long_track_chunk_overlap_sec: float
    low_purity_threshold: float
    high_jump_threshold: float
    good_sample_count: int
    good_min_detections: int
    good_max_detections: int
    good_min_purity: float
    good_max_jump: float
    ffmpeg_binary: str
    ffprobe_binary: str
    logical_gpu: int
    preset: str
    cq: int
    pixel_format: str


def _required(mapping: dict[str, Any], key: str, context: str) -> Any:
    if key not in mapping:
        raise ContractError(f"missing required config key: {context}.{key}")
    return mapping[key]


def _integer(mapping: dict[str, Any], key: str, context: str) -> int:
    value = _required(mapping, key, context)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ContractError(f"{context}.{key} must be an integer")
    return value


def _finite(mapping: dict[str, Any], key: str, context: str) -> float:
    value = _required(mapping, key, context)
    if isinstance(value, bool):
        raise ContractError(f"{context}.{key} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ContractError(f"{context}.{key} must be a finite number") from exc
    if not math.isfinite(result):
        raise ContractError(f"{context}.{key} must be a finite number")
    return result


def _boolean(mapping: dict[str, Any], key: str, context: str) -> bool:
    value = _required(mapping, key, context)
    if not isinstance(value, bool):
        raise ContractError(f"{context}.{key} must be a boolean")
    return value


def load_s01_review_config(
    path: Path,
) -> tuple[S01ReviewConfig, dict[str, Any], str]:
    path = path.resolve()
    if not path.is_file():
        raise ContractError(f"S01 review config does not exist: {path}")
    try:
        raw = path.read_bytes()
        payload = yaml.safe_load(raw)
    except (OSError, yaml.YAMLError) as exc:
        raise ContractError(f"cannot read S01 review config {path}: {exc}") from exc
    if not isinstance(payload, dict) or set(payload) != {"pipeline", "review", "encoder"}:
        raise ContractError(
            "S01 review config must contain exactly pipeline, review, and encoder"
        )
    pipeline = payload["pipeline"]
    review = payload["review"]
    encoder = payload["encoder"]
    if not all(isinstance(value, dict) for value in (pipeline, review, encoder)):
        raise ContractError("S01 review config sections must be mappings")
    expected_pipeline = {"schema_version"}
    expected_review = {
        "output_width",
        "output_height",
        "window_before_sec",
        "window_after_sec",
        "progress_interval_sec",
        "draw_context_boxes",
        "long_track_min_detections",
        "long_track_chunk_sec",
        "long_track_chunk_overlap_sec",
        "low_purity_threshold",
        "high_jump_threshold",
        "good_sample_count",
        "good_min_detections",
        "good_max_detections",
        "good_min_purity",
        "good_max_jump",
    }
    expected_encoder = {
        "ffmpeg_binary",
        "ffprobe_binary",
        "codec",
        "logical_gpu",
        "preset",
        "cq",
        "pixel_format",
    }
    for section_name, section, expected_keys in (
        ("pipeline", pipeline, expected_pipeline),
        ("review", review, expected_review),
        ("encoder", encoder, expected_encoder),
    ):
        if set(section) != expected_keys:
            missing = sorted(expected_keys - set(section))
            extra = sorted(set(section) - expected_keys)
            raise ContractError(
                f"{section_name} config keys mismatch; missing={missing}, extra={extra}"
            )

    codec = str(_required(encoder, "codec", "encoder"))
    result = S01ReviewConfig(
        schema_version=str(_required(pipeline, "schema_version", "pipeline")),
        output_width=_integer(review, "output_width", "review"),
        output_height=_integer(review, "output_height", "review"),
        window_before_sec=_finite(review, "window_before_sec", "review"),
        window_after_sec=_finite(review, "window_after_sec", "review"),
        progress_interval_sec=_finite(review, "progress_interval_sec", "review"),
        draw_context_boxes=_boolean(review, "draw_context_boxes", "review"),
        long_track_min_detections=_integer(
            review, "long_track_min_detections", "review"
        ),
        long_track_chunk_sec=_finite(review, "long_track_chunk_sec", "review"),
        long_track_chunk_overlap_sec=_finite(
            review, "long_track_chunk_overlap_sec", "review"
        ),
        low_purity_threshold=_finite(review, "low_purity_threshold", "review"),
        high_jump_threshold=_finite(review, "high_jump_threshold", "review"),
        good_sample_count=_integer(review, "good_sample_count", "review"),
        good_min_detections=_integer(review, "good_min_detections", "review"),
        good_max_detections=_integer(review, "good_max_detections", "review"),
        good_min_purity=_finite(review, "good_min_purity", "review"),
        good_max_jump=_finite(review, "good_max_jump", "review"),
        ffmpeg_binary=str(_required(encoder, "ffmpeg_binary", "encoder")),
        ffprobe_binary=str(_required(encoder, "ffprobe_binary", "encoder")),
        logical_gpu=_integer(encoder, "logical_gpu", "encoder"),
        preset=str(_required(encoder, "preset", "encoder")),
        cq=_integer(encoder, "cq", "encoder"),
        pixel_format=str(_required(encoder, "pixel_format", "encoder")),
    )
    if result.schema_version != "1.0":
        raise ContractError("fixed S01 review requires schema_version=1.0")
    if (result.output_width, result.output_height) != (1920, 1080):
        raise ContractError("fixed S01 review output must be 1920x1080")
    if any(
        value <= 0.0
        for value in (
            result.window_before_sec,
            result.window_after_sec,
            result.progress_interval_sec,
        )
    ):
        raise ContractError("S01 review time intervals must be positive")
    if result.long_track_min_detections < 3 or result.long_track_chunk_sec <= 0.0:
        raise ContractError("invalid long-track review settings")
    if not 0.0 <= result.long_track_chunk_overlap_sec < result.long_track_chunk_sec:
        raise ContractError("invalid long-track chunk overlap")
    if not 0.0 < result.low_purity_threshold <= 1.0:
        raise ContractError("review.low_purity_threshold must be in (0, 1]")
    if result.high_jump_threshold <= 0.0:
        raise ContractError("review.high_jump_threshold must be positive")
    if result.good_sample_count < 1:
        raise ContractError("review.good_sample_count must be positive")
    if not 3 <= result.good_min_detections <= result.good_max_detections:
        raise ContractError("invalid good-sample detection length range")
    if not 0.0 < result.good_min_purity <= 1.0:
        raise ContractError("review.good_min_purity must be in (0, 1]")
    if result.good_max_jump <= 0.0:
        raise ContractError("review.good_max_jump must be positive")
    if not result.ffmpeg_binary.strip() or not result.ffprobe_binary.strip():
        raise ContractError("encoder ffmpeg/ffprobe binaries cannot be blank")
    if "\x00" in result.ffmpeg_binary or "\x00" in result.ffprobe_binary:
        raise ContractError("encoder ffmpeg/ffprobe binaries cannot contain NUL")
    if codec != "h264_nvenc":
        raise ContractError("S01 review requires encoder.codec=h264_nvenc")
    if result.logical_gpu != 0:
        raise ContractError(
            "with CUDA_VISIBLE_DEVICES=1, S01 review requires encoder.logical_gpu=0"
        )
    if result.preset not in {f"p{index}" for index in range(1, 8)}:
        raise ContractError("encoder.preset must be p1..p7")
    if not 0 <= result.cq <= 51:
        raise ContractError("encoder.cq must be in [0, 51]")
    if result.pixel_format != "yuv420p":
        raise ContractError("S01 review requires encoder.pixel_format=yuv420p")
    canonical = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return result, payload, hashlib.sha256(canonical).hexdigest()
