"""Strict post-encode validation for S01 QA MP4 files."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

from cowtrack.config import ContractError


RunFn = Callable[..., subprocess.CompletedProcess[str]]


@dataclass(frozen=True)
class QaMp4Metadata:
    """Metadata accepted by the fixed S01 review-video contract."""

    path: Path
    stream_index: int
    codec_name: str
    width: int
    height: int
    average_frame_rate: Fraction
    frame_count: int


def _positive_integer(value: object, context: str) -> int:
    if isinstance(value, bool):
        raise ContractError(f"{context} must be a positive integer")
    if isinstance(value, int):
        result = value
    elif isinstance(value, str) and value.isdecimal():
        result = int(value)
    else:
        raise ContractError(f"{context} must be a positive integer")
    if result <= 0:
        raise ContractError(f"{context} must be a positive integer")
    return result


def _stream_has_rotation_metadata(value: object) -> bool:
    """Return true for any rotate/rotation/display-matrix representation."""

    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized_key = str(key).casefold().replace("_", "").replace(" ", "")
            if normalized_key in {"rotate", "rotation", "displaymatrix"}:
                return True
            if normalized_key == "sidedatatype" and isinstance(child, str):
                normalized_value = child.casefold().replace("_", "").replace(" ", "")
                if "displaymatrix" in normalized_value:
                    return True
            if _stream_has_rotation_metadata(child):
                return True
        return False
    if isinstance(value, list):
        return any(_stream_has_rotation_metadata(child) for child in value)
    return False


def _parse_stream_fraction(value: object, context: str) -> Fraction:
    if not isinstance(value, str) or not value:
        raise ContractError(f"{context} must be an exact fraction string")
    try:
        result = Fraction(value)
    except (ValueError, ZeroDivisionError) as exc:
        raise ContractError(f"{context} must be an exact fraction string") from exc
    if result <= 0:
        raise ContractError(f"{context} must be positive")
    return result


def validate_qa_mp4(
    video_path: str | Path,
    *,
    expected_frame_rate: Fraction,
    expected_frame_count: int,
    ffprobe_binary: str = "ffprobe",
    run: RunFn = subprocess.run,
) -> QaMp4Metadata:
    """Validate one encoded review MP4 against the fixed QA contract.

    The injected ``run`` callable is invoked with an argument vector and
    ``shell=False``.  There is deliberately no secondary parser or codec
    fallback: absent or ambiguous ffprobe fields fail the contract.
    """

    path = Path(video_path)
    if not path.is_file():
        raise ContractError(f"QA MP4 does not exist: {path}")
    if not isinstance(expected_frame_rate, Fraction) or expected_frame_rate <= 0:
        raise ContractError("expected_frame_rate must be a positive Fraction")
    expected_frames = _positive_integer(
        expected_frame_count, "expected_frame_count"
    )
    if not isinstance(ffprobe_binary, str) or not ffprobe_binary.strip():
        raise ContractError("ffprobe_binary must be a non-blank string")
    if "\x00" in ffprobe_binary:
        raise ContractError("ffprobe_binary cannot contain a NUL byte")
    if not callable(run):
        raise ContractError("run must be callable")

    # Keep all stream fields, tags, and side data in the JSON.  Restricting
    # ``-show_entries`` could accidentally hide a vendor-specific spelling of
    # rotation metadata that this validator is intended to reject.
    command = [
        ffprobe_binary,
        "-v",
        "error",
        "-show_streams",
        "-of",
        "json",
        str(path),
    ]
    try:
        completed = run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            shell=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ContractError(f"failed to start ffprobe for {path}: {exc}") from exc

    try:
        return_code = int(completed.returncode)
        stdout = completed.stdout
        stderr = completed.stderr
    except (AttributeError, TypeError, ValueError) as exc:
        raise ContractError("ffprobe runner returned an invalid result") from exc
    if return_code != 0:
        detail = stderr.strip() if isinstance(stderr, str) else ""
        suffix = f": {detail}" if detail else ""
        raise ContractError(f"ffprobe failed ({return_code}) for {path}{suffix}")
    if not isinstance(stdout, str):
        raise ContractError(f"ffprobe returned non-text JSON for {path}")
    try:
        payload: Any = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise ContractError(f"invalid ffprobe JSON for {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ContractError(f"ffprobe JSON root must be an object: {path}")

    streams = payload.get("streams")
    if not isinstance(streams, list):
        raise ContractError(f"ffprobe JSON has no streams list: {path}")
    forbidden_types = {"audio", "subtitle", "data"}
    forbidden = [
        str(stream.get("codec_type"))
        for stream in streams
        if isinstance(stream, dict) and stream.get("codec_type") in forbidden_types
    ]
    if forbidden:
        raise ContractError(
            f"QA MP4 contains forbidden stream types {forbidden}: {path}"
        )
    if len(streams) != 1 or not isinstance(streams[0], dict):
        raise ContractError(f"QA MP4 must contain exactly one stream: {path}")
    stream: dict[str, Any] = streams[0]
    if stream.get("codec_type") != "video":
        raise ContractError(f"QA MP4's only stream must be video: {path}")
    if stream.get("codec_name") != "h264":
        raise ContractError(f"QA MP4 video codec must be h264: {path}")
    if stream.get("pix_fmt") != "yuv420p":
        raise ContractError(f"QA MP4 pixel format must be yuv420p: {path}")

    width = _positive_integer(stream.get("width"), "QA MP4 width")
    height = _positive_integer(stream.get("height"), "QA MP4 height")
    if (width, height) != (1920, 1080):
        raise ContractError(
            f"QA MP4 geometry must be 1920x1080, got {width}x{height}: {path}"
        )
    actual_frame_rate = _parse_stream_fraction(
        stream.get("avg_frame_rate"), "QA MP4 avg_frame_rate"
    )
    if actual_frame_rate != expected_frame_rate:
        raise ContractError(
            "QA MP4 avg_frame_rate mismatch: "
            f"expected {expected_frame_rate}, got {actual_frame_rate}: {path}"
        )
    actual_frames = _positive_integer(stream.get("nb_frames"), "QA MP4 nb_frames")
    if actual_frames != expected_frames:
        raise ContractError(
            "QA MP4 nb_frames mismatch: "
            f"expected {expected_frames}, got {actual_frames}: {path}"
        )
    if _stream_has_rotation_metadata(stream):
        raise ContractError(f"QA MP4 must not contain rotation/display matrix: {path}")

    stream_index_value = stream.get("index")
    if isinstance(stream_index_value, bool) or not isinstance(stream_index_value, int):
        raise ContractError(f"QA MP4 stream index must be an integer: {path}")
    if stream_index_value != 0:
        raise ContractError(f"QA MP4 video stream index must be zero: {path}")
    return QaMp4Metadata(
        path=path,
        stream_index=stream_index_value,
        codec_name="h264",
        width=width,
        height=height,
        average_frame_rate=actual_frame_rate,
        frame_count=actual_frames,
    )
