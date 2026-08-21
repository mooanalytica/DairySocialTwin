"""Strict configuration for the S04 proposal video review."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import yaml

from cowtrack.config import ContractError


@dataclass(frozen=True)
class S04ReviewConfig:
    schema_version: str
    random_seed: int
    top_count: int
    bottom_count: int
    random_count: int
    source_tail_sec: float
    target_head_sec: float
    output_width: int
    output_height: int
    progress_interval_sec: float
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


def load_s04_review_config(
    path: Path,
) -> tuple[S04ReviewConfig, dict[str, Any], str]:
    """Load the closed S04 review config and return its canonical hash."""

    path = path.resolve()
    if not path.is_file():
        raise ContractError(f"S04 review config does not exist: {path}")
    try:
        payload = yaml.safe_load(path.read_bytes())
    except (OSError, yaml.YAMLError) as exc:
        raise ContractError(f"cannot read S04 review config {path}: {exc}") from exc
    if not isinstance(payload, dict) or set(payload) != {
        "pipeline",
        "selection",
        "render",
        "encoder",
    }:
        raise ContractError(
            "S04 review config must contain exactly pipeline, selection, render, encoder"
        )
    pipeline = payload["pipeline"]
    selection = payload["selection"]
    render = payload["render"]
    encoder = payload["encoder"]
    if not all(
        isinstance(section, dict)
        for section in (pipeline, selection, render, encoder)
    ):
        raise ContractError("S04 review config sections must be mappings")
    expected = {
        "pipeline": {"schema_version", "random_seed"},
        "selection": {
            "top_probability_count",
            "bottom_probability_count",
            "random_remainder_count",
        },
        "render": {
            "source_tail_sec",
            "target_head_sec",
            "output_width",
            "output_height",
            "progress_interval_sec",
        },
        "encoder": {
            "ffmpeg_binary",
            "ffprobe_binary",
            "codec",
            "logical_gpu",
            "preset",
            "cq",
            "pixel_format",
        },
    }
    for name, section in (
        ("pipeline", pipeline),
        ("selection", selection),
        ("render", render),
        ("encoder", encoder),
    ):
        if set(section) != expected[name]:
            raise ContractError(
                f"{name} config keys mismatch; "
                f"missing={sorted(expected[name] - set(section))}, "
                f"extra={sorted(set(section) - expected[name])}"
            )

    result = S04ReviewConfig(
        schema_version=str(_required(pipeline, "schema_version", "pipeline")),
        random_seed=_integer(pipeline, "random_seed", "pipeline"),
        top_count=_integer(selection, "top_probability_count", "selection"),
        bottom_count=_integer(
            selection, "bottom_probability_count", "selection"
        ),
        random_count=_integer(selection, "random_remainder_count", "selection"),
        source_tail_sec=_finite(render, "source_tail_sec", "render"),
        target_head_sec=_finite(render, "target_head_sec", "render"),
        output_width=_integer(render, "output_width", "render"),
        output_height=_integer(render, "output_height", "render"),
        progress_interval_sec=_finite(
            render, "progress_interval_sec", "render"
        ),
        ffmpeg_binary=str(_required(encoder, "ffmpeg_binary", "encoder")),
        ffprobe_binary=str(_required(encoder, "ffprobe_binary", "encoder")),
        logical_gpu=_integer(encoder, "logical_gpu", "encoder"),
        preset=str(_required(encoder, "preset", "encoder")),
        cq=_integer(encoder, "cq", "encoder"),
        pixel_format=str(_required(encoder, "pixel_format", "encoder")),
    )
    if result.schema_version != "1.0":
        raise ContractError("fixed S04 review requires pipeline.schema_version=1.0")
    if result.random_seed != 20260710:
        raise ContractError("fixed S04 review requires pipeline.random_seed=20260710")
    if min(result.top_count, result.bottom_count, result.random_count) < 0:
        raise ContractError("S04 review selection counts cannot be negative")
    if (result.top_count, result.bottom_count, result.random_count) != (15, 15, 20):
        raise ContractError(
            "fixed S04 review selection must be exactly top=15, bottom=15, random=20"
        )
    if result.source_tail_sec != 2.0 or result.target_head_sec != 2.0:
        raise ContractError("fixed S04 review windows must be exactly 2 seconds per side")
    if (result.output_width, result.output_height) != (1920, 1080):
        raise ContractError("fixed S04 review output must be exactly 1920x1080")
    if result.progress_interval_sec <= 0.0:
        raise ContractError("render.progress_interval_sec must be positive")
    if not result.ffmpeg_binary.strip() or not result.ffprobe_binary.strip():
        raise ContractError("encoder ffmpeg/ffprobe binaries cannot be blank")
    if "\x00" in result.ffmpeg_binary or "\x00" in result.ffprobe_binary:
        raise ContractError("encoder ffmpeg/ffprobe binaries cannot contain NUL")
    if str(_required(encoder, "codec", "encoder")) != "h264_nvenc":
        raise ContractError("S04 review requires encoder.codec=h264_nvenc")
    if result.logical_gpu != 0:
        raise ContractError(
            "with CUDA_VISIBLE_DEVICES=1, S04 review requires encoder.logical_gpu=0"
        )
    if result.preset not in {f"p{index}" for index in range(1, 8)}:
        raise ContractError("encoder.preset must be p1..p7")
    if not 0 <= result.cq <= 51:
        raise ContractError("encoder.cq must be in [0, 51]")
    if result.pixel_format != "yuv420p":
        raise ContractError("S04 review requires encoder.pixel_format=yuv420p")
    canonical = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return result, payload, hashlib.sha256(canonical).hexdigest()


__all__ = ["S04ReviewConfig", "load_s04_review_config"]
