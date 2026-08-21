"""Strict configuration for independent S05 provisional-evidence review.

The review is deliberately read-only.  It does not certify an edge, run a
solver, or authorize a stable-ID merge.  ``logical_gpu=0`` is interpreted
inside the required ``CUDA_VISIBLE_DEVICES=1`` process environment.
"""

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
class S05ReviewConfig:
    schema_version: str
    random_seed: int
    maximum_cases: int
    high_score_count: int
    threshold_near_count: int
    ambiguous_count: int
    random_count: int
    threshold_probability_band: float
    ambiguity_margin_upper: float
    ambiguous_mutual_rank_above: int
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


def _mapping(value: Any, expected: set[str], context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ContractError(f"S05 review {context} must be a mapping")
    if set(value) != expected:
        raise ContractError(
            f"S05 review {context} keys mismatch; "
            f"missing={sorted(expected - set(value))}, "
            f"extra={sorted(set(value) - expected)}"
        )
    return value


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ContractError(f"S05 review {name} must be an integer >= {minimum}")
    return value


def _finite(value: Any, name: str, *, minimum_exclusive: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"S05 review {name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ContractError(f"S05 review {name} must be finite")
    if minimum_exclusive is not None and result <= minimum_exclusive:
        raise ContractError(
            f"S05 review {name} must be greater than {minimum_exclusive}"
        )
    return result


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise ContractError(f"S05 review {name} must be a non-empty string without NUL")
    return value


def load_s05_review_config(
    path: Path,
) -> tuple[S05ReviewConfig, dict[str, Any], str]:
    """Load the closed S05 review contract and its canonical SHA-256."""

    path = path.resolve()
    if not path.is_file():
        raise ContractError(f"S05 review config does not exist: {path}")
    try:
        serialized = path.read_bytes()
        payload = (
            json.loads(serialized)
            if path.suffix.lower() == ".json"
            else yaml.safe_load(serialized)
        )
    except (OSError, UnicodeError, json.JSONDecodeError, yaml.YAMLError) as exc:
        raise ContractError(f"cannot read S05 review config {path}: {exc}") from exc
    root = _mapping(
        payload,
        {"pipeline", "selection", "render", "encoder"},
        "root",
    )
    pipeline = _mapping(
        root["pipeline"],
        {"schema_version", "random_seed", "review_semantics"},
        "pipeline",
    )
    selection = _mapping(
        root["selection"],
        {
            "maximum_cases",
            "high_score_count",
            "threshold_near_count",
            "ambiguous_count",
            "random_remainder_count",
            "threshold_probability_band",
            "ambiguity_margin_upper",
            "ambiguous_mutual_rank_above",
        },
        "selection",
    )
    render = _mapping(
        root["render"],
        {
            "source_tail_sec",
            "target_head_sec",
            "output_width",
            "output_height",
            "progress_interval_sec",
        },
        "render",
    )
    encoder = _mapping(
        root["encoder"],
        {
            "ffmpeg_binary",
            "ffprobe_binary",
            "codec",
            "logical_gpu",
            "preset",
            "cq",
            "pixel_format",
        },
        "encoder",
    )

    config = S05ReviewConfig(
        schema_version=_text(pipeline["schema_version"], "pipeline.schema_version"),
        random_seed=_integer(pipeline["random_seed"], "pipeline.random_seed"),
        maximum_cases=_integer(selection["maximum_cases"], "selection.maximum_cases"),
        high_score_count=_integer(
            selection["high_score_count"], "selection.high_score_count"
        ),
        threshold_near_count=_integer(
            selection["threshold_near_count"], "selection.threshold_near_count"
        ),
        ambiguous_count=_integer(
            selection["ambiguous_count"], "selection.ambiguous_count"
        ),
        random_count=_integer(
            selection["random_remainder_count"],
            "selection.random_remainder_count",
        ),
        threshold_probability_band=_finite(
            selection["threshold_probability_band"],
            "selection.threshold_probability_band",
            minimum_exclusive=0.0,
        ),
        ambiguity_margin_upper=_finite(
            selection["ambiguity_margin_upper"],
            "selection.ambiguity_margin_upper",
        ),
        ambiguous_mutual_rank_above=_integer(
            selection["ambiguous_mutual_rank_above"],
            "selection.ambiguous_mutual_rank_above",
            minimum=1,
        ),
        source_tail_sec=_finite(
            render["source_tail_sec"],
            "render.source_tail_sec",
            minimum_exclusive=0.0,
        ),
        target_head_sec=_finite(
            render["target_head_sec"],
            "render.target_head_sec",
            minimum_exclusive=0.0,
        ),
        output_width=_integer(
            render["output_width"], "render.output_width", minimum=1
        ),
        output_height=_integer(
            render["output_height"], "render.output_height", minimum=1
        ),
        progress_interval_sec=_finite(
            render["progress_interval_sec"],
            "render.progress_interval_sec",
            minimum_exclusive=0.0,
        ),
        ffmpeg_binary=_text(encoder["ffmpeg_binary"], "encoder.ffmpeg_binary"),
        ffprobe_binary=_text(encoder["ffprobe_binary"], "encoder.ffprobe_binary"),
        logical_gpu=_integer(encoder["logical_gpu"], "encoder.logical_gpu"),
        preset=_text(encoder["preset"], "encoder.preset"),
        cq=_integer(encoder["cq"], "encoder.cq"),
        pixel_format=_text(encoder["pixel_format"], "encoder.pixel_format"),
    )

    fixed = (
        config.schema_version == "1.0"
        and config.random_seed == 20260710
        and pipeline["review_semantics"] == "provisional_evidence_only"
        and config.maximum_cases == 50
        and (
            config.high_score_count,
            config.threshold_near_count,
            config.ambiguous_count,
            config.random_count,
        )
        == (15, 15, 10, 10)
        and math.isclose(
            config.threshold_probability_band, 0.03, rel_tol=0.0, abs_tol=1e-12
        )
        and math.isclose(
            config.ambiguity_margin_upper, 0.08, rel_tol=0.0, abs_tol=1e-12
        )
        and config.ambiguous_mutual_rank_above == 1
    )
    if not fixed:
        raise ContractError(
            "fixed S05 review requires seed=20260710, provisional-evidence-only, "
            "maximum=50, groups=15/15/10/10, threshold band=0.03, "
            "ambiguity margin=0.08, and ambiguous rank above 1"
        )
    if sum(
        (
            config.high_score_count,
            config.threshold_near_count,
            config.ambiguous_count,
            config.random_count,
        )
    ) != config.maximum_cases:
        raise ContractError("S05 review selection quotas must sum to maximum_cases")
    if (config.source_tail_sec, config.target_head_sec) != (2.0, 2.0):
        raise ContractError("fixed S05 review windows must be exactly 2 seconds per side")
    if (config.output_width, config.output_height) != (1920, 1080):
        raise ContractError("fixed S05 review output must be exactly 1920x1080")
    if encoder["codec"] != "h264_nvenc":
        raise ContractError("S05 review requires encoder.codec=h264_nvenc")
    if config.logical_gpu != 0:
        raise ContractError(
            "with CUDA_VISIBLE_DEVICES=1, S05 review requires encoder.logical_gpu=0"
        )
    if config.preset not in {f"p{index}" for index in range(1, 8)}:
        raise ContractError("S05 review encoder.preset must be p1..p7")
    if not 0 <= config.cq <= 51:
        raise ContractError("S05 review encoder.cq must be in [0, 51]")
    if config.pixel_format != "yuv420p":
        raise ContractError("S05 review requires encoder.pixel_format=yuv420p")

    canonical = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return config, payload, hashlib.sha256(canonical).hexdigest()


__all__ = ["S05ReviewConfig", "load_s05_review_config"]
