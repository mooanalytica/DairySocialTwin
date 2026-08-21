from __future__ import annotations

import json
import math
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
from typing import Any

from reid_index import EXPECTED_CLIP_IDS, REID_SEQUENCE_ID, resolve_reid_clip_source


ROOT = Path(__file__).resolve().parent
TRACK_ID_EXPORT_ROOT = Path(
    "/mnt/drive_bf/BF/re-identification-11B-GOOD/work/"
    "dairy_farm_1_gopro1_20250505_all11/06_export_complete_sequence"
)
TRACK_ID_EFFECTIVE_CONFIG = TRACK_ID_EXPORT_ROOT / "effective_config.json"
TRACK_ID_QA_METRICS = TRACK_ID_EXPORT_ROOT / "qa_metrics.json"
TRACK_ID_SUCCESS = TRACK_ID_EXPORT_ROOT / "_SUCCESS.json"
TRACK_ID_QA_VIDEOS_DIR = TRACK_ID_EXPORT_ROOT / "qa" / "videos"
TRACK_ID_VIDEO_CACHE_DIR = ROOT / ".cache" / "track_id_videos"
TRACK_ID_CACHE_MAX_DIMENSION = 960
TRACK_ID_SOURCE_WIDTH = 1920
TRACK_ID_SOURCE_HEIGHT = 1080
TRACK_ID_SOURCE_FPS_TEXT = "30000/1001"
TRACK_ID_SOURCE_FPS = float(Fraction(TRACK_ID_SOURCE_FPS_TEXT))


@dataclass(frozen=True, slots=True)
class QaTrackVideo:
    sequence_id: str
    clip_id: str
    path: Path
    source_fingerprint_path: str
    source_size_bytes: int
    source_sha256: str
    qa_video_size_bytes: int
    qa_video_sha256: str
    frame_count: int
    width: int
    height: int
    fps: float


@dataclass(frozen=True, slots=True)
class VideoProbe:
    codec_name: str
    codec_tag: str
    pixel_format: str
    width: int
    height: int
    fps: float
    frame_count: int
    duration: float
    audio_stream_count: int


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required track-ID video metadata is missing: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return data


def _require_exact_keys(actual: set[str], expected: set[str], label: str) -> None:
    if actual != expected:
        raise ValueError(
            f"{label} is not an exact current clip mapping; "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )


def _path_under(path: Path, root: Path) -> Path:
    resolved = path.resolve()
    root_resolved = root.resolve()
    if resolved != root_resolved and root_resolved not in resolved.parents:
        raise ValueError(f"Track-ID video path is outside the authorized QA video root: {resolved}")
    return resolved


def _require_sha256(value: Any, label: str) -> str:
    digest = str(value).strip().lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"Malformed SHA256 for {label}: {value!r}")
    return digest


@lru_cache(maxsize=1)
def load_qa_track_video_catalog() -> dict[str, QaTrackVideo]:
    effective = _read_json(TRACK_ID_EFFECTIVE_CONFIG)
    qa_metrics = _read_json(TRACK_ID_QA_METRICS)
    success = _read_json(TRACK_ID_SUCCESS)
    expected_clips = set(EXPECTED_CLIP_IDS)
    qa_config_hash = _require_sha256(qa_metrics.get("config_hash"), "track-ID QA config")
    success_config_hash = _require_sha256(success.get("config_hash"), "track-ID export config")
    if qa_config_hash != success_config_hash:
        raise ValueError("Track-ID qa_metrics.json and _SUCCESS.json belong to different exports")

    inputs = effective.get("inputs")
    artifacts = effective.get("artifacts")
    if not isinstance(inputs, dict) or not isinstance(artifacts, dict):
        raise ValueError(f"Malformed current track-ID effective config: {TRACK_ID_EFFECTIVE_CONFIG}")
    if str(inputs.get("expected_sequence_id", "")) != REID_SEQUENCE_ID:
        raise ValueError("Track-ID video sequence_id disagrees with the strict re-ID index")
    clip_order = tuple(str(value) for value in inputs.get("clip_order", []))
    if clip_order != EXPECTED_CLIP_IDS:
        raise ValueError(f"Unexpected track-ID video clip order: {clip_order}")
    if int(inputs.get("expected_total_detection_count", -1)) != 3_322_909:
        raise ValueError("Track-ID video export total detection count is not current")
    if int(inputs.get("expected_valid_detection_count", -1)) != 3_318_113:
        raise ValueError("Track-ID video export valid detection count is not current")
    if int(inputs.get("expected_invalid_detection_count", -1)) != 4_796:
        raise ValueError("Track-ID video export invalid detection count is not current")

    frame_counts_raw = inputs.get("frame_counts_by_clip")
    videos_raw = artifacts.get("videos_by_clip")
    if not isinstance(frame_counts_raw, dict) or not isinstance(videos_raw, dict):
        raise ValueError("Track-ID effective config is missing frame_counts_by_clip/videos_by_clip")
    _require_exact_keys(set(map(str, frame_counts_raw)), expected_clips, "frame_counts_by_clip")
    _require_exact_keys(set(map(str, videos_raw)), expected_clips, "videos_by_clip")

    if str(qa_metrics.get("sequence_id", "")) != REID_SEQUENCE_ID:
        raise ValueError("Track-ID QA metrics sequence_id disagrees with the strict re-ID index")
    render_contract = qa_metrics.get("render_contract")
    if not isinstance(render_contract, dict):
        raise ValueError("Track-ID QA metrics are missing render_contract")
    if render_contract.get("output_size") != [TRACK_ID_SOURCE_WIDTH, TRACK_ID_SOURCE_HEIGHT]:
        raise ValueError(f"Unexpected track-ID QA video dimensions: {render_contract.get('output_size')}")
    if str(render_contract.get("fps", "")) != TRACK_ID_SOURCE_FPS_TEXT:
        raise ValueError(f"Unexpected track-ID QA video fps: {render_contract.get('fps')}")
    if str(render_contract.get("pixel_format", "")) != "yuv420p":
        raise ValueError(f"Unexpected track-ID QA pixel format: {render_contract.get('pixel_format')}")
    if render_contract.get("audio") is not False:
        raise ValueError("Current track-ID QA videos must not contain audio")
    if render_contract.get("all_source_video_frames_preserved") is not True:
        raise ValueError("Track-ID QA videos do not guarantee all source frames")
    if str(render_contract.get("full_video_content", "")) != "valid_bbox_and_display_global_id_only":
        raise ValueError("Track-ID QA video render content is not the accepted valid-only identity overlay")
    if render_contract.get("invalid_identity_drawn") is not False:
        raise ValueError("Track-ID QA videos unexpectedly draw invalid identities")

    if str(success.get("execution_mode", "")) != "forced_provisional_qa_export":
        raise ValueError("Track-ID QA export execution mode is not current")
    if str(success.get("id_status", "")) != "forced_provisional":
        raise ValueError("Track-ID QA export identity status is not current")
    fingerprints = success.get("input_fingerprints")
    if not isinstance(fingerprints, list):
        raise ValueError("Track-ID QA _SUCCESS metadata is missing input_fingerprints")
    video_fingerprints: dict[str, dict[str, Any]] = {}
    for item in fingerprints:
        if not isinstance(item, dict):
            continue
        raw_path = str(item.get("path", "")).strip()
        path = Path(raw_path)
        clip_id = path.stem
        if path.suffix.lower() != ".mp4" or clip_id not in expected_clips:
            continue
        if not path.is_absolute():
            raise ValueError(f"Track-ID source fingerprint is not a full path: {raw_path}")
        if clip_id in video_fingerprints:
            raise ValueError(f"Duplicate full source fingerprint for track-ID clip {clip_id}")
        video_fingerprints[clip_id] = item
    _require_exact_keys(set(video_fingerprints), expected_clips, "track-ID full source fingerprints")

    output_fingerprints = success.get("output_fingerprints")
    if not isinstance(output_fingerprints, list):
        raise ValueError("Track-ID QA _SUCCESS metadata is missing output_fingerprints")
    qa_video_fingerprints: dict[str, dict[str, Any]] = {}
    for item in output_fingerprints:
        if not isinstance(item, dict):
            continue
        relative_path = Path(str(item.get("path", "")).strip())
        if relative_path.parent != Path("qa/videos") or relative_path.suffix.lower() != ".mp4":
            continue
        stem = relative_path.stem
        suffix = "_tracked"
        if not stem.endswith(suffix):
            continue
        clip_id = stem[: -len(suffix)]
        if clip_id not in expected_clips:
            continue
        if clip_id in qa_video_fingerprints:
            raise ValueError(f"Duplicate QA video fingerprint for track-ID clip {clip_id}")
        qa_video_fingerprints[clip_id] = item
    _require_exact_keys(set(qa_video_fingerprints), expected_clips, "track-ID QA video fingerprints")

    catalog: dict[str, QaTrackVideo] = {}
    for clip_id in EXPECTED_CLIP_IDS:
        relative_video = Path(str(videos_raw[clip_id]).strip())
        if relative_video.is_absolute():
            raise ValueError(f"Track-ID videos_by_clip entry must be relative: {relative_video}")
        video_path = _path_under(TRACK_ID_EXPORT_ROOT / relative_video, TRACK_ID_QA_VIDEOS_DIR)
        expected_name = f"{clip_id}_tracked.mp4"
        if video_path.name != expected_name or video_path.parent != TRACK_ID_QA_VIDEOS_DIR.resolve():
            raise ValueError(f"Unexpected track-ID QA video path for {clip_id}: {video_path}")
        if not video_path.is_file():
            raise FileNotFoundError(f"Required track-ID QA video is missing: {video_path}")
        fingerprint = video_fingerprints[clip_id]
        source_fingerprint_path = str(fingerprint.get("path", "")).strip()
        source_size_bytes = int(fingerprint.get("size_bytes", 0) or 0)
        if Path(source_fingerprint_path).name != f"{clip_id}.MP4" or source_size_bytes <= 0:
            raise ValueError(f"Malformed full source fingerprint for track-ID clip {clip_id}")
        source_sha256 = _require_sha256(fingerprint.get("sha256"), f"source clip {clip_id}")
        qa_fingerprint = qa_video_fingerprints[clip_id]
        qa_video_size_bytes = int(qa_fingerprint.get("size_bytes", 0) or 0)
        if qa_video_size_bytes <= 0 or video_path.stat().st_size != qa_video_size_bytes:
            raise ValueError(
                f"QA video size disagrees with _SUCCESS for {clip_id}: "
                f"expected={qa_video_size_bytes}, actual={video_path.stat().st_size}"
            )
        qa_video_sha256 = _require_sha256(qa_fingerprint.get("sha256"), f"QA video {clip_id}")
        frame_count = int(frame_counts_raw[clip_id])
        if frame_count <= 0:
            raise ValueError(f"Invalid track-ID frame count for {clip_id}: {frame_count}")
        catalog[clip_id] = QaTrackVideo(
            sequence_id=REID_SEQUENCE_ID,
            clip_id=clip_id,
            path=video_path,
            source_fingerprint_path=source_fingerprint_path,
            source_size_bytes=source_size_bytes,
            source_sha256=source_sha256,
            qa_video_size_bytes=qa_video_size_bytes,
            qa_video_sha256=qa_video_sha256,
            frame_count=frame_count,
            width=TRACK_ID_SOURCE_WIDTH,
            height=TRACK_ID_SOURCE_HEIGHT,
            fps=TRACK_ID_SOURCE_FPS,
        )
    return catalog


def _parse_fraction(value: Any, label: str) -> float:
    text = str(value).strip()
    if not text:
        raise ValueError(f"Missing {label}")
    parsed = float(Fraction(text))
    if not math.isfinite(parsed) or parsed <= 0:
        raise ValueError(f"Invalid {label}: {text!r}")
    return parsed


@lru_cache(maxsize=64)
def _probe_video_cached(path_text: str, size_bytes: int, mtime_ns: int) -> VideoProbe:
    del size_bytes, mtime_ns
    path = Path(path_text)
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            (
                "stream=codec_type,codec_name,codec_tag_string,pix_fmt,width,height,"
                "r_frame_rate,avg_frame_rate,nb_frames,duration"
            ),
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    data = json.loads(result.stdout)
    streams = data.get("streams", [])
    if not isinstance(streams, list):
        raise ValueError(f"ffprobe returned malformed streams for {path}")
    video_streams = [stream for stream in streams if stream.get("codec_type") == "video"]
    audio_streams = [stream for stream in streams if stream.get("codec_type") == "audio"]
    if len(video_streams) != 1:
        raise ValueError(f"Expected exactly one video stream in {path}: {len(video_streams)}")
    stream = video_streams[0]
    fps_value = stream.get("avg_frame_rate") or stream.get("r_frame_rate")
    return VideoProbe(
        codec_name=str(stream.get("codec_name", "")),
        codec_tag=str(stream.get("codec_tag_string", "")),
        pixel_format=str(stream.get("pix_fmt", "")),
        width=int(stream.get("width", 0) or 0),
        height=int(stream.get("height", 0) or 0),
        fps=_parse_fraction(fps_value, f"{path}:fps"),
        frame_count=int(stream.get("nb_frames", 0) or 0),
        duration=float(stream.get("duration", 0.0) or 0.0),
        audio_stream_count=len(audio_streams),
    )


def probe_video(path: Path) -> VideoProbe:
    if not path.is_file():
        raise FileNotFoundError(f"Required video is missing: {path}")
    stat = path.stat()
    return _probe_video_cached(str(path), int(stat.st_size), int(stat.st_mtime_ns))


def validate_qa_track_video(video: QaTrackVideo) -> VideoProbe:
    probe = probe_video(video.path)
    if probe.codec_name != "h264" or probe.codec_tag not in {"avc1", "avc3", "h264"}:
        raise ValueError(f"Track-ID QA video is not browser-compatible H.264: {video.path}")
    if probe.pixel_format != "yuv420p":
        raise ValueError(f"Track-ID QA video has unexpected pixel format {probe.pixel_format}: {video.path}")
    if (probe.width, probe.height) != (video.width, video.height):
        raise ValueError(
            f"Track-ID QA video dimensions mismatch for {video.clip_id}: "
            f"expected={video.width}x{video.height}, actual={probe.width}x{probe.height}"
        )
    if abs(probe.fps - video.fps) > 1e-9:
        raise ValueError(f"Track-ID QA video fps mismatch for {video.clip_id}: {probe.fps}")
    if probe.frame_count != video.frame_count:
        raise ValueError(
            f"Track-ID QA video frame count mismatch for {video.clip_id}: "
            f"expected={video.frame_count}, actual={probe.frame_count}"
        )
    expected_duration = video.frame_count / video.fps
    if abs(probe.duration - expected_duration) > (1.0 / video.fps) + 1e-3:
        raise ValueError(
            f"Track-ID QA video duration mismatch for {video.clip_id}: "
            f"expected={expected_duration}, actual={probe.duration}"
        )
    if probe.audio_stream_count != 0:
        raise ValueError(f"Track-ID QA video unexpectedly contains audio: {video.path}")
    return probe


def validate_track_id_cache(path: Path, expected_frame_count: int, fps: float) -> VideoProbe:
    expected_frames = int(expected_frame_count)
    expected_fps = float(fps)
    if expected_frames <= 0 or not math.isfinite(expected_fps) or expected_fps <= 0:
        raise ValueError(
            f"Invalid expected track-ID cache timing: frames={expected_frame_count}, fps={fps}"
        )
    probe = probe_video(path)
    if probe.codec_name != "h264" or probe.codec_tag not in {"avc1", "avc3", "h264"}:
        raise ValueError(f"Track-ID cache is not browser-compatible H.264: {path}")
    if probe.pixel_format != "yuv420p":
        raise ValueError(f"Track-ID cache has unexpected pixel format {probe.pixel_format}: {path}")
    if (probe.width, probe.height) != (TRACK_ID_CACHE_MAX_DIMENSION, 540):
        raise ValueError(
            f"Track-ID cache dimensions must be {TRACK_ID_CACHE_MAX_DIMENSION}x540: "
            f"actual={probe.width}x{probe.height}, path={path}"
        )
    if abs(probe.fps - expected_fps) > 1e-9:
        raise ValueError(f"Track-ID cache fps mismatch: expected={expected_fps}, actual={probe.fps}, path={path}")
    if probe.frame_count != expected_frames:
        raise ValueError(
            f"Track-ID cache frame count mismatch: expected={expected_frames}, "
            f"actual={probe.frame_count}, path={path}"
        )
    expected_duration = expected_frames / expected_fps
    if abs(probe.duration - expected_duration) > (1.0 / expected_fps) + 1e-3:
        raise ValueError(
            f"Track-ID cache duration mismatch: expected={expected_duration}, "
            f"actual={probe.duration}, path={path}"
        )
    if probe.audio_stream_count != 0:
        raise ValueError(f"Track-ID cache unexpectedly contains audio: {path}")
    return probe


def track_id_cache_is_ready(path: Path, expected_frame_count: int, fps: float) -> bool:
    try:
        validate_track_id_cache(path, expected_frame_count, fps)
    except (FileNotFoundError, OSError, subprocess.SubprocessError, ValueError, json.JSONDecodeError):
        return False
    return True


def qa_track_video_for_source(
    authoritative_source_path: str,
    manifest_source_size_bytes: int,
) -> QaTrackVideo:
    source = resolve_reid_clip_source(authoritative_source_path)
    if source.sequence_id != REID_SEQUENCE_ID:
        raise ValueError(f"Unexpected re-ID sequence for track-ID video: {source.sequence_id}")
    catalog = load_qa_track_video_catalog()
    video = catalog.get(source.clip_id)
    if video is None:
        raise ValueError(f"No current QA track-ID video for full source path: {authoritative_source_path}")
    if video.sequence_id != source.sequence_id:
        raise ValueError(f"QA track-ID video sequence mismatch for {authoritative_source_path}")
    if Path(authoritative_source_path).name != Path(video.source_fingerprint_path).name:
        raise ValueError(
            "Authoritative manifest source and QA export full-source fingerprint disagree: "
            f"manifest={authoritative_source_path}, export={video.source_fingerprint_path}"
        )
    source_size = int(manifest_source_size_bytes)
    if source_size <= 0 or source_size != video.source_size_bytes:
        raise ValueError(
            "Authoritative manifest source size and QA export full-source fingerprint disagree: "
            f"manifest={source_size}, export={video.source_size_bytes}, source={authoritative_source_path}"
        )
    validate_qa_track_video(video)
    return video


def track_id_cache_path(sample_id: str, source_video: QaTrackVideo) -> Path:
    sample = str(sample_id).strip()
    if not sample:
        raise ValueError("sample_id is required for track-ID cache path")
    return TRACK_ID_VIDEO_CACHE_DIR / (
        f"{sample}_{source_video.path.stem}_web{TRACK_ID_CACHE_MAX_DIMENSION}_h264.mp4"
    )
