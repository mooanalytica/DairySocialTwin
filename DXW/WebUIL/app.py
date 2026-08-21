from __future__ import annotations

import argparse
import csv
import json
import math
import mimetypes
import threading
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import parse_qs, urlparse

from reid_index import (
    REID_BBOX_TOLERANCE_PX,
    REID_CONFIDENCE_TOLERANCE,
    REID_INDEX_PATH,
    REID_SOURCE_CSV,
    REID_VALIDITY_POLICY,
    ReIdClip,
    ReIdRecord,
    load_reid_clip_for_source,
    validate_reid_index,
)
from trackid_video import (
    TRACK_ID_VIDEO_CACHE_DIR,
    qa_track_video_for_source,
    track_id_cache_is_ready,
    track_id_cache_path,
)


ROOT = Path(__file__).resolve().parent
STATIC_DIR = ROOT / "static"

FLOORPLAN_ROOT = Path("/home/hyw/FloorPlanAnnoEN/output15")
SNA_FLOORPLAN_ROOT = Path("/home/hyw/FloorPlanAnnoSR/output15")
STAGE2_CODE_DIR = Path("/home/hyw/DCSNA-ALL")
ROTATION_SETTINGS_FILE = ROOT / "rotation_settings.json"
WEBUIL_CACHE_DIR = ROOT / ".cache"
STAGE2_PRECOMPUTED_DIR = WEBUIL_CACHE_DIR / "stage2_precomputed"
STAGE2_PRECOMPUTED_SCHEMA_VERSION = 1
DEEP_LINK_GENERATION_MANIFEST = ROOT / "sna_inputs" / "F1_Gopro1_20250505" / "generation_manifest.json"
DEEP_LINK_PARAMETER_NAMES = ("farmID", "cameraID", "clipID", "segmentID", "frameID")
REQUIRED_CUDA_VISIBLE_DEVICES = "1"
Y_DRIVE_ROOT = Path("/mnt/data4t/hyw")
STAGE1_SEGMENTED_ROOT = Y_DRIVE_ROOT / "Stage1_segmented"
SEGMENT_DATA_ROOT = STAGE1_SEGMENTED_ROOT
# Active sample scope for the current Farm 1 / Gopro1 dataset.
ACTIVE_SAMPLE_FARM_ID: str | None = "1"
ACTIVE_SAMPLE_CAMERA_ID: str | None = "Gopro1"

RECTANGLE_FIT_METHOD = "minimum_area_rotated_bounding_rectangle_from_polygon_points"
CANVAS_WIDTH = 3840
CANVAS_HEIGHT = 2160
PORTRAIT_TRACKING_WIDTH = 2160
PORTRAIT_TRACKING_HEIGHT = 3840
LANDSCAPE_TRACKING_WIDTH = 3840
LANDSCAPE_TRACKING_HEIGHT = 2160
SQUEEZE_MARGIN_PX = 8.0
POINT_FREEZE_SECONDS = 0.5
SEMANTIC_INFLUENCE_MULTIPLIER = 2.0


Point = tuple[float, float]
ProgressCallback = Callable[[str, int | None, int | None], None]


class _RecentTrackPoints:
    """Keep only local tracks that can still contribute a frozen point."""

    def __init__(self, freeze_frames: int) -> None:
        if freeze_frames < 1:
            raise ValueError(f"freeze_frames must be positive, got {freeze_frames}")
        self.freeze_frames = freeze_frames
        self._last_seen: dict[int, tuple[int, dict[str, Any]]] = {}
        self._expiring: dict[int, list[tuple[int, int]]] = {}

    @property
    def active_track_count(self) -> int:
        return len(self._last_seen)

    def begin_frame(self, frame: int) -> None:
        """Expire entries before processing one frame in a contiguous timeline."""
        for track_id, last_frame in self._expiring.pop(frame, []):
            current = self._last_seen.get(track_id)
            if current is not None and current[0] == last_frame:
                del self._last_seen[track_id]

    def remember(self, track_id: int, frame: int, point: dict[str, Any]) -> None:
        self._last_seen[track_id] = (frame, point)
        expiration_frame = frame + self.freeze_frames + 1
        self._expiring.setdefault(expiration_frame, []).append((track_id, frame))

    def frozen_candidates(
        self,
        frame: int,
        current_points: dict[int, dict[str, Any]],
        current_global_tracks: dict[str, int],
    ) -> dict[str, tuple[int, int, dict[str, Any]]]:
        candidates: dict[str, tuple[int, int, dict[str, Any]]] = {}
        for track_id, (last_frame, last_point) in sorted(self._last_seen.items()):
            if track_id in current_points:
                continue
            age_frames = frame - last_frame
            if 0 < age_frames <= self.freeze_frames:
                global_track_uuid = str(last_point["globalTrackUuid"])
                if global_track_uuid in current_global_tracks:
                    continue
                previous = candidates.get(global_track_uuid)
                if previous is not None:
                    previous_track, previous_frame, _previous_point = previous
                    if previous_frame == last_frame and previous_track != track_id:
                        raise RuntimeError(
                            "Ambiguous frozen local tracks for one global identity: "
                            f"frame={frame}, last_seen={last_frame}, global={global_track_uuid}, "
                            f"local={previous_track}/{track_id}"
                        )
                    if previous_frame > last_frame:
                        continue
                candidates[global_track_uuid] = (track_id, last_frame, last_point)
        return candidates


def require_file(path: Path) -> None:
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Required file is missing: {path}")


def read_json(path: Path) -> Any:
    require_file(path)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def camera_number(camera_id: str) -> str:
    value = str(camera_id).strip()
    lower = value.lower()
    if lower.startswith("gopro"):
        value = value[5:]
    if not value or not value.isdigit():
        raise ValueError(f"Unsupported camera id: {camera_id}")
    return str(int(value))


def floorplan_group_id(farm_id: str, camera_id: str) -> str:
    farm = str(farm_id).strip()
    if not farm or not farm.isdigit():
        raise ValueError(f"Unsupported farm id: {farm_id}")
    return f"farm_ID_{int(farm)}_camera_ID_{camera_number(camera_id)}"


def floorplan_annotation_file(farm_id: str, camera_id: str) -> Path:
    return FLOORPLAN_ROOT / floorplan_group_id(farm_id, camera_id) / "floorplan_annotation.json"


def sna_floorplan_file(farm_id: str, camera_id: str) -> Path:
    return SNA_FLOORPLAN_ROOT / floorplan_group_id(farm_id, camera_id) / "sna_zones.json"


def active_sample_scope_matches(farm_id: str, camera_id: str) -> bool:
    if ACTIVE_SAMPLE_FARM_ID is not None:
        if str(int(str(farm_id).strip())) != str(int(ACTIVE_SAMPLE_FARM_ID)):
            return False
    if ACTIVE_SAMPLE_CAMERA_ID is not None:
        if camera_number(camera_id) != camera_number(ACTIVE_SAMPLE_CAMERA_ID):
            return False
    return True


def segment_dir_in_active_scope(path: Path) -> bool:
    rel = path.relative_to(STAGE1_SEGMENTED_ROOT)
    if len(rel.parts) != 3:
        return False
    return active_sample_scope_matches(rel.parts[0], rel.parts[1])


def stage2_precomputed_index_file() -> Path:
    return STAGE2_PRECOMPUTED_DIR / "index.json"


def require_under_directory(path: Path, directory: Path) -> Path:
    resolved = path.resolve()
    directory_resolved = directory.resolve()
    if resolved != directory_resolved and directory_resolved not in resolved.parents:
        raise ValueError(f"Path is outside {directory_resolved}: {resolved}")
    return resolved


def stage2_precomputed_file_from_entry(entry: dict[str, Any]) -> Path:
    relative_path = str(entry.get("path", "")).strip()
    if not relative_path:
        raise ValueError("Stage2 precomputed index entry is missing path")
    path = STAGE2_PRECOMPUTED_DIR / relative_path
    resolved = require_under_directory(path, STAGE2_PRECOMPUTED_DIR)
    require_file(resolved)
    return resolved


STAGE2_PRECOMPUTED_INDEX_CACHE: dict[str, dict[str, Any]] | None = None


def load_stage2_precomputed_index() -> dict[str, dict[str, Any]]:
    global STAGE2_PRECOMPUTED_INDEX_CACHE
    if STAGE2_PRECOMPUTED_INDEX_CACHE is not None:
        return STAGE2_PRECOMPUTED_INDEX_CACHE
    index_path = stage2_precomputed_index_file()
    if not index_path.is_file():
        STAGE2_PRECOMPUTED_INDEX_CACHE = {}
        return STAGE2_PRECOMPUTED_INDEX_CACHE
    data = read_json(index_path)
    if int(data.get("schemaVersion", -1)) != STAGE2_PRECOMPUTED_SCHEMA_VERSION:
        raise ValueError(f"Unsupported Stage2 precomputed index schema: {index_path}")
    samples = data.get("samples", {})
    if not isinstance(samples, dict):
        raise ValueError(f"Stage2 precomputed index samples must be an object: {index_path}")
    out: dict[str, dict[str, Any]] = {}
    for sample_id, entry in samples.items():
        if not isinstance(entry, dict) or not entry.get("complete", False):
            continue
        copied = dict(entry)
        copied["path"] = str(copied.get("path", "")).strip()
        stage2_precomputed_file_from_entry(copied)
        out[str(sample_id)] = copied
    STAGE2_PRECOMPUTED_INDEX_CACHE = out
    return out


def parse_segment_dir_name(name: str) -> tuple[str, int]:
    stem = str(name).strip()
    gx_name, sep, shard_raw = stem.rpartition("_")
    if not sep or not gx_name.startswith("GX") or not gx_name[2:].isdigit() or not shard_raw.isdigit():
        raise ValueError(f"Unsupported segment directory name: {name}")
    return gx_name, int(shard_raw)


def segment_dir_sort_key(path: Path) -> tuple[int, int, str, int]:
    rel = path.relative_to(STAGE1_SEGMENTED_ROOT)
    if len(rel.parts) != 3:
        raise ValueError(f"Unexpected Stage1 segmented layout: {path}")
    farm_dir, camera_dir, segment_dir = rel.parts
    gx_name, shard_index = parse_segment_dir_name(segment_dir)
    return int(farm_dir), int(camera_number(camera_dir)), gx_name, shard_index


def discover_segment_dirs() -> list[Path]:
    if not STAGE1_SEGMENTED_ROOT.is_dir():
        raise FileNotFoundError(f"Stage1 segmented directory is missing: {STAGE1_SEGMENTED_ROOT}")
    paths = [path.parent for path in STAGE1_SEGMENTED_ROOT.glob("*/*/*/playback_segment.json")]
    paths = [path for path in paths if segment_dir_in_active_scope(path)]
    if not paths:
        raise RuntimeError(
            f"No Stage1 segmented samples found under active scope "
            f"farm={ACTIVE_SAMPLE_FARM_ID or '*'} camera={ACTIVE_SAMPLE_CAMERA_ID or '*'} in {STAGE1_SEGMENTED_ROOT}"
        )
    paths.sort(key=segment_dir_sort_key)
    return paths


def normalize_playback_segment(source_dir: Path) -> dict[str, Any]:
    playback_path = source_dir / "playback_segment.json"
    playback = read_json(playback_path)
    rel = source_dir.relative_to(STAGE1_SEGMENTED_ROOT)
    if len(rel.parts) != 3:
        raise ValueError(f"Unexpected Stage1 segmented layout: {source_dir}")

    farm_dir, camera_dir, segment_dir = rel.parts
    dir_gx_name, dir_shard_index = parse_segment_dir_name(segment_dir)
    farm_id = str(int(str(playback.get("farm_id", "")).strip()))
    camera_id = camera_dir
    playback_camera = str(playback.get("camera", "")).strip()
    gx_name = str(playback.get("gx_id", "")).strip()
    shard_index = int(playback.get("shard_index"))
    shard_count = int(playback.get("shard_count"))
    if farm_id != str(int(farm_dir)):
        raise ValueError(f"Playback farm mismatch for {source_dir}: {farm_id} != {farm_dir}")
    if camera_number(playback_camera) != camera_number(camera_dir):
        raise ValueError(f"Playback camera mismatch for {source_dir}: {playback_camera} != {camera_dir}")
    if gx_name != dir_gx_name or shard_index != dir_shard_index:
        raise ValueError(f"Playback shard mismatch for {source_dir}: {gx_name}_{shard_index} != {segment_dir}")
    if shard_index < 1 or shard_count < shard_index:
        raise ValueError(f"Invalid shard index/count for {source_dir}: {shard_index}/{shard_count}")

    segment_start_frame = int(playback.get("segment_start_frame"))
    segment_end_frame = int(playback.get("segment_end_frame"))
    base_start_frame = int(playback.get("base_start_frame"))
    base_end_frame = int(playback.get("base_end_frame"))
    if segment_end_frame < segment_start_frame or base_end_frame < base_start_frame:
        raise ValueError(f"Invalid playback frame range for {source_dir}")
    segment_start_seconds = float(playback.get("segment_start_seconds"))
    segment_end_seconds = float(playback.get("segment_end_seconds_exclusive"))
    if segment_end_seconds <= segment_start_seconds:
        raise ValueError(f"Invalid playback second range for {source_dir}")

    return {
        "farmId": farm_id,
        "cameraId": camera_id,
        "gxName": gx_name,
        "shardIndex": shard_index,
        "shardCount": shard_count,
        "fps": float(playback.get("fps", 0.0) or 0.0),
        "baseStartFrame": base_start_frame,
        "baseEndFrame": base_end_frame,
        "segmentStartFrame": segment_start_frame,
        "segmentEndFrame": segment_end_frame,
        "segmentStartSeconds": segment_start_seconds,
        "segmentEndSecondsExclusive": segment_end_seconds,
        "sourceDirOriginal": str(playback.get("source_dir", "")),
    }


def track_id_video_info(
    sample_id: str,
    manifest_data: dict[str, Any],
    playback: dict[str, Any],
) -> dict[str, Any]:
    authoritative_source_path = source_video_path_from_manifest(manifest_data)
    manifest_source_size_bytes = source_video_size_from_manifest(manifest_data)
    source_video = qa_track_video_for_source(authoritative_source_path, manifest_source_size_bytes)
    cache_path = track_id_cache_path(sample_id, source_video)
    segment_start_frame = int(playback["segmentStartFrame"])
    segment_end_frame = int(playback["segmentEndFrame"])
    fps = float(playback.get("fps", 0.0) or 0.0)
    if segment_start_frame < 0 or segment_end_frame >= source_video.frame_count:
        raise ValueError(
            f"Playback frame range is outside the current QA track-ID video for {sample_id}: "
            f"[{segment_start_frame}, {segment_end_frame}] vs {source_video.frame_count} frames"
        )
    if not math.isclose(fps, source_video.fps, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(
            f"Playback FPS disagrees with the current QA track-ID video for {sample_id}: "
            f"playback={fps}, video={source_video.fps}"
        )
    expected_start_seconds = segment_start_frame / fps
    expected_end_seconds = (segment_end_frame + 1) / fps
    if not math.isclose(
        float(playback["segmentStartSeconds"]), expected_start_seconds, rel_tol=0.0, abs_tol=1e-6
    ) or not math.isclose(
        float(playback["segmentEndSecondsExclusive"]), expected_end_seconds, rel_tol=0.0, abs_tol=1e-6
    ):
        raise ValueError(f"Playback frame/second boundaries disagree for {sample_id}")
    expected_frame_count = segment_end_frame - segment_start_frame + 1
    segment_duration = expected_frame_count / fps
    cache_playable = track_id_cache_is_ready(cache_path, expected_frame_count, fps)
    selected_path = cache_path if cache_playable else None
    return {
        "sampleId": sample_id,
        "available": selected_path is not None,
        "selectedKind": "track_id_cache" if selected_path is not None else "",
        "selectedPath": selected_path,
        "trackIdSourcePath": source_video.path,
        "trackIdSourceClipId": source_video.clip_id,
        "trackIdSourceSha256": source_video.qa_video_sha256,
        "trackIdCachePath": cache_path if cache_playable else None,
        "trackIdCacheTargetPath": cache_path,
        "trackIdCachePlayable": cache_playable,
        "trackIdName": source_video.path.name,
        "expectedFrameCount": expected_frame_count,
        "loopStartFrame": segment_start_frame,
        "loopEndFrame": segment_end_frame,
        "loopStartSeconds": 0.0,
        "loopEndSeconds": segment_duration,
        "sourceLoopStartSeconds": float(playback["segmentStartSeconds"]),
        "sourceLoopEndSeconds": float(playback["segmentEndSecondsExclusive"]),
        "fps": fps,
        "baseStartFrame": int(playback["baseStartFrame"]),
        "baseEndFrame": int(playback["baseEndFrame"]),
        "shardIndex": int(playback["shardIndex"]),
        "shardCount": int(playback["shardCount"]),
    }


def public_video_info(sample: dict[str, Any]) -> dict[str, Any]:
    video = sample["video"]
    if not video["available"]:
        return {
            "available": False,
            "selectedKind": "",
            "selectedName": "",
            "selectedUrl": "",
            "trackIdCachePlayable": False,
            "trackIdName": "",
            "loopStartFrame": 0,
            "loopEndFrame": 0,
            "loopStartSeconds": 0.0,
            "loopEndSeconds": 0.0,
            "sourceLoopStartSeconds": 0.0,
            "sourceLoopEndSeconds": 0.0,
            "fps": 0.0,
            "baseStartFrame": 0,
            "baseEndFrame": 0,
            "shardIndex": 0,
            "shardCount": 0,
        }
    selected_path = video["selectedPath"]
    return {
        "available": True,
        "selectedKind": video["selectedKind"],
        "selectedName": selected_path.name,
        "selectedUrl": f"/api/video?sample={sample['id']}&kind=selected",
        "trackIdCachePlayable": video["trackIdCachePlayable"],
        "trackIdName": video["trackIdName"],
        "loopStartFrame": video["loopStartFrame"],
        "loopEndFrame": video["loopEndFrame"],
        "loopStartSeconds": video["loopStartSeconds"],
        "loopEndSeconds": video["loopEndSeconds"],
        "sourceLoopStartSeconds": video["sourceLoopStartSeconds"],
        "sourceLoopEndSeconds": video["sourceLoopEndSeconds"],
        "fps": video["fps"],
        "baseStartFrame": video["baseStartFrame"],
        "baseEndFrame": video["baseEndFrame"],
        "shardIndex": video["shardIndex"],
        "shardCount": video["shardCount"],
    }


def public_sample(sample: dict[str, Any]) -> dict[str, Any]:
    stage2_path = sample.get("stage2PrecomputedPath")
    sna_path = sna_floorplan_file(sample["farmId"], sample["cameraId"])
    return {
        "id": sample["id"],
        "label": sample["label"],
        "predLabel": sample["predLabel"],
        "displayClass": sample["displayClass"],
        "trackingCsv": str(sample["trackingCsv"]),
        "manifest": str(sample["manifest"]),
        "keypointsCsv": str(sample["keypointsCsv"]),
        "sourceDir": str(sample["sourceDir"]),
        "sourcePath": sample["sourcePath"],
        "sourceVideoId": sample["sourceVideoId"],
        "sourceVideoName": sample["sourceVideoName"],
        "farmId": sample["farmId"],
        "cameraId": sample["cameraId"],
        "goproId": sample["goproId"],
        "floorplanAnnotation": str(sample["floorplanAnnotation"]),
        "snaFloorplan": str(sna_path),
        "snaFloorplanAvailable": sna_path.is_file(),
        "clipName": sample["clipName"],
        "split": sample["split"],
        "shardIndex": sample["shardIndex"],
        "shardCount": sample["shardCount"],
        "playbackSegment": sample["playbackSegment"],
        "video": public_video_info(sample),
        "stage2Ready": stage2_path is not None,
        "stage2PrecomputedPath": str(stage2_path) if stage2_path is not None else "",
    }


SAMPLE_CACHE: dict[bool, list[dict[str, Any]]] = {}


def source_video_name_from_manifest(manifest_data: dict[str, Any], gx_name: str) -> str:
    source_path = source_video_path_from_manifest(manifest_data)
    source_name = Path(source_path.replace("\\", "/")).name
    if Path(source_name).stem != str(gx_name).strip():
        raise ValueError(
            f"Authoritative source path basename does not match current clip: gx={gx_name}, source={source_path}"
        )
    return source_name


def source_video_path_from_manifest(manifest_data: dict[str, Any]) -> str:
    video_manifest = manifest_data.get("video_manifest")
    if not isinstance(video_manifest, dict):
        raise ValueError("manifest.json is missing video_manifest")
    source_path = str(video_manifest.get("source_path", "")).strip()
    if not source_path:
        raise ValueError("manifest.json is missing authoritative video_manifest.source_path")
    if not Path(source_path).is_absolute():
        raise ValueError(f"video_manifest.source_path must be an absolute full path: {source_path}")
    return source_path


def source_video_size_from_manifest(manifest_data: dict[str, Any]) -> int:
    video_manifest = manifest_data.get("video_manifest")
    if not isinstance(video_manifest, dict):
        raise ValueError("manifest.json is missing video_manifest")
    source_size_bytes = int(video_manifest.get("file_size_bytes", 0) or 0)
    if source_size_bytes <= 0:
        raise ValueError("manifest.json is missing a positive video_manifest.file_size_bytes")
    return source_size_bytes


def load_samples(require_ready: bool = True) -> list[dict[str, Any]]:
    if require_ready in SAMPLE_CACHE:
        return SAMPLE_CACHE[require_ready]

    segment_dirs = discover_segment_dirs()

    samples: list[dict[str, Any]] = []
    seen: set[str] = set()
    precomputed_index = load_stage2_precomputed_index()
    for source_dir in segment_dirs:
        if not source_dir.is_dir():
            raise FileNotFoundError(f"Selected segment directory is missing: {source_dir}")
        playback = normalize_playback_segment(source_dir)
        farm_id = playback["farmId"]
        camera_id = playback["cameraId"]
        gx_name = playback["gxName"]
        tracking_csv = source_dir / "tracking_boxes.csv"
        keypoints_csv = source_dir / "keypoints.csv"
        manifest = source_dir / "manifest.json"
        floorplan = floorplan_annotation_file(farm_id, camera_id)
        for path in (tracking_csv, keypoints_csv, manifest, floorplan):
            require_file(path)

        sample_id = f"F{farm_id}_{camera_id}_{gx_name}_{playback['shardIndex']}"
        if sample_id in seen:
            raise ValueError(f"Duplicate selected sample id: {sample_id}")
        seen.add(sample_id)

        manifest_data = read_json(manifest)
        video = track_id_video_info(sample_id, manifest_data, playback)
        precomputed_entry = precomputed_index.get(sample_id)
        precomputed_path = (
            stage2_precomputed_file_from_entry(precomputed_entry)
            if precomputed_entry is not None
            else None
        )
        if require_ready and (not video["available"] or precomputed_path is None):
            continue
        samples.append(
            {
                "id": sample_id,
                "label": sample_id,
                "predLabel": "precomputed",
                "displayClass": "Precomputed inference",
                "trackingCsv": tracking_csv,
                "manifest": manifest,
                "keypointsCsv": keypoints_csv,
                "sourceDir": source_dir,
                "sourcePath": source_video_path_from_manifest(manifest_data),
                "sourceVideoId": gx_name,
                "sourceVideoName": source_video_name_from_manifest(manifest_data, gx_name),
                "farmId": farm_id,
                "cameraId": camera_id,
                "goproId": camera_number(camera_id),
                "floorplanAnnotation": floorplan,
                "clipName": f"{gx_name}_{playback['shardIndex']}",
                "split": "Stage1_segmented",
                "shardIndex": playback["shardIndex"],
                "shardCount": playback["shardCount"],
                "playbackSegment": {
                    "baseStartFrame": playback["baseStartFrame"],
                    "baseEndFrame": playback["baseEndFrame"],
                    "segmentStartFrame": playback["segmentStartFrame"],
                    "segmentEndFrame": playback["segmentEndFrame"],
                    "segmentStartSeconds": playback["segmentStartSeconds"],
                    "segmentEndSecondsExclusive": playback["segmentEndSecondsExclusive"],
                    "sourceDirOriginal": playback["sourceDirOriginal"],
                },
                "video": video,
                "stage2Precomputed": precomputed_entry,
                "stage2PrecomputedPath": precomputed_path,
            }
        )

    if not samples:
        if require_ready:
            raise RuntimeError(
                "No selectable Stage1 segmented sample is ready. Generate Stage2 results under "
                f"{STAGE2_PRECOMPUTED_DIR} and the track-ID video cache under {TRACK_ID_VIDEO_CACHE_DIR} first: "
                "CUDA_VISIBLE_DEVICES=1 python3 generate_stage2_precomputed.py --all; "
                "CUDA_VISIBLE_DEVICES=1 python3 -u generate_video_cache.py --all "
                "--encoder h264_nvenc --workers 2"
            )
        raise RuntimeError(f"No Stage1 segmented samples found in {STAGE1_SEGMENTED_ROOT}")

    samples.sort(
        key=lambda item: (
            int(item["farmId"]),
            int(camera_number(item["cameraId"])),
            item["sourceVideoId"],
            int(item["shardIndex"]),
        )
    )
    SAMPLE_CACHE[require_ready] = samples
    return samples


def default_sample(samples: list[dict[str, Any]]) -> dict[str, Any]:
    return samples[0]


def get_sample(sample_id: str | None = None) -> dict[str, Any]:
    samples = load_samples()
    if sample_id is None or not sample_id.strip():
        return default_sample(samples)
    wanted = sample_id.strip()
    for sample in samples:
        if sample["id"] == wanted:
            return sample
    raise ValueError(f"Unknown sample id: {sample_id}")


def samples_payload() -> dict[str, Any]:
    samples = load_samples()
    default = default_sample(samples)
    return {
        "ok": True,
        "defaultSampleId": default["id"],
        "samples": [public_sample(sample) for sample in samples],
    }


def required_manifest_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Deep-link generation manifest is missing {label}")
    return value.strip()


def required_manifest_integer(value: Any, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"Deep-link generation manifest has invalid {label}: {value!r}")
    return int(value)


def load_deep_link_manifest_index() -> dict[str, Any]:
    manifest = read_json(DEEP_LINK_GENERATION_MANIFEST)
    if not isinstance(manifest, dict):
        raise ValueError(f"Deep-link generation manifest must be an object: {DEEP_LINK_GENERATION_MANIFEST}")
    if required_manifest_integer(manifest.get("schema_version"), "schema_version", 1) != 1:
        raise ValueError(f"Unsupported deep-link generation manifest schema: {DEEP_LINK_GENERATION_MANIFEST}")

    farm_id = required_manifest_string(manifest.get("farm"), "farm")
    camera_id = required_manifest_string(manifest.get("camera"), "camera")
    combined_sample_id = required_manifest_string(
        manifest.get("combined_sample_id"), "combined_sample_id"
    )
    if not farm_id.isdigit():
        raise ValueError(f"Deep-link manifest farm must be numeric: {farm_id}")
    camera_number(camera_id)

    raw_samples = manifest.get("samples")
    if not isinstance(raw_samples, list) or not raw_samples:
        raise ValueError(f"Deep-link generation manifest samples must be a non-empty list: {DEEP_LINK_GENERATION_MANIFEST}")

    by_sample_id: dict[str, dict[str, Any]] = {}
    by_source_clip_id: dict[str, list[dict[str, Any]]] = {}
    source_path_by_clip: dict[str, str] = {}
    clip_by_source_path: dict[str, str] = {}
    shard_counts: dict[str, int] = {}
    shard_indexes: dict[str, set[int]] = {}

    for row_index, raw in enumerate(raw_samples):
        if not isinstance(raw, dict):
            raise ValueError(f"Deep-link manifest sample {row_index} must be an object")
        sample_id = required_manifest_string(raw.get("sample_id"), f"samples[{row_index}].sample_id")
        source_clip_id = required_manifest_string(
            raw.get("source_clip_id"), f"samples[{row_index}].source_clip_id"
        )
        source_path = required_manifest_string(raw.get("source_path"), f"samples[{row_index}].source_path")
        if not Path(source_path).is_absolute():
            raise ValueError(f"Deep-link manifest source_path must be an absolute full path: {source_path}")
        shard_index = required_manifest_integer(
            raw.get("shard_index"), f"samples[{row_index}].shard_index", 1
        )
        shard_count = required_manifest_integer(
            raw.get("shard_count"), f"samples[{row_index}].shard_count", 1
        )
        segment_start = required_manifest_integer(
            raw.get("segment_start_frame"), f"samples[{row_index}].segment_start_frame"
        )
        segment_end = required_manifest_integer(
            raw.get("segment_end_frame"), f"samples[{row_index}].segment_end_frame"
        )
        canonical_start = required_manifest_integer(
            raw.get("canonical_start_frame"), f"samples[{row_index}].canonical_start_frame"
        )
        canonical_end = required_manifest_integer(
            raw.get("canonical_end_frame"), f"samples[{row_index}].canonical_end_frame"
        )
        canonical_count = required_manifest_integer(
            raw.get("canonical_frame_count"), f"samples[{row_index}].canonical_frame_count", 1
        )
        if not segment_start <= canonical_start <= canonical_end <= segment_end:
            raise ValueError(f"Deep-link manifest sample has invalid physical/canonical bounds: {sample_id}")
        if canonical_count != canonical_end - canonical_start + 1:
            raise ValueError(f"Deep-link manifest sample has inconsistent canonical frame count: {sample_id}")
        if shard_index > shard_count:
            raise ValueError(f"Deep-link manifest sample has invalid shard index/count: {sample_id}")

        expected_sample_id = f"F{int(farm_id)}_{camera_id}_{source_clip_id}_{shard_index}"
        if sample_id != expected_sample_id:
            raise ValueError(
                f"Deep-link manifest sample id is not canonical: expected={expected_sample_id}, actual={sample_id}"
            )
        if sample_id in by_sample_id:
            raise ValueError(f"Duplicate deep-link manifest sample id: {sample_id}")

        old_path = source_path_by_clip.setdefault(source_clip_id, source_path)
        old_clip = clip_by_source_path.setdefault(source_path, source_clip_id)
        if old_path != source_path or old_clip != source_clip_id:
            raise ValueError(
                "Deep-link source clip/full-path mapping is not bijective: "
                f"clip={source_clip_id}, path={source_path}"
            )
        old_shard_count = shard_counts.setdefault(source_clip_id, shard_count)
        if old_shard_count != shard_count:
            raise ValueError(f"Deep-link manifest shard_count disagrees for {source_clip_id}")
        indexes = shard_indexes.setdefault(source_clip_id, set())
        if shard_index in indexes:
            raise ValueError(f"Duplicate deep-link shard index for {source_clip_id}: {shard_index}")
        indexes.add(shard_index)

        entry = {
            "sampleId": sample_id,
            "sourceClipId": source_clip_id,
            "sourcePath": source_path,
            "shardIndex": shard_index,
            "shardCount": shard_count,
            "segmentStartFrame": segment_start,
            "segmentEndFrame": segment_end,
            "canonicalStartFrame": canonical_start,
            "canonicalEndFrame": canonical_end,
        }
        by_sample_id[sample_id] = entry
        by_source_clip_id.setdefault(source_clip_id, []).append(entry)

    for source_clip_id, entries in by_source_clip_id.items():
        expected_indexes = set(range(1, shard_counts[source_clip_id] + 1))
        if shard_indexes[source_clip_id] != expected_indexes:
            raise ValueError(
                f"Deep-link manifest shard indexes are incomplete for {source_clip_id}: "
                f"expected={sorted(expected_indexes)}, actual={sorted(shard_indexes[source_clip_id])}"
            )
        entries.sort(key=lambda entry: (int(entry["canonicalStartFrame"]), int(entry["shardIndex"])))
        previous: dict[str, Any] | None = None
        for entry in entries:
            if previous is not None and int(entry["canonicalStartFrame"]) <= int(previous["canonicalEndFrame"]):
                raise ValueError(
                    "Deep-link canonical ranges overlap: "
                    f"{previous['sampleId']} and {entry['sampleId']}"
                )
            previous = entry

    return {
        "farmId": str(int(farm_id)),
        "cameraId": camera_id,
        "combinedSampleId": combined_sample_id,
        "bySampleId": by_sample_id,
        "bySourceClipId": by_source_clip_id,
    }


def invalid_deep_link_payload() -> dict[str, Any]:
    return {"ok": True, "valid": False}


def parse_deep_link_segment_id(value: str) -> tuple[str, int, int] | None:
    parts = value.rsplit("-", 2)
    if len(parts) != 3:
        return None
    source_clip_id, shard_raw, frame_raw = (part.strip() for part in parts)
    if not source_clip_id or not shard_raw.isdigit() or not frame_raw.isdigit():
        return None
    try:
        shard_index = int(shard_raw)
        frame_id = int(frame_raw)
    except ValueError:
        return None
    if shard_index < 1:
        return None
    return source_clip_id, shard_index, frame_id


def resolve_deep_link(params: dict[str, list[str]]) -> dict[str, Any]:
    values: dict[str, str] = {}
    for name in DEEP_LINK_PARAMETER_NAMES:
        raw_values = params.get(name, [])
        if len(raw_values) != 1 or not str(raw_values[0]).strip():
            return invalid_deep_link_payload()
        values[name] = str(raw_values[0]).strip()

    if not values["farmID"].isdigit() or not values["frameID"].isdigit():
        return invalid_deep_link_payload()
    try:
        farm_id = str(int(values["farmID"]))
        frame_id = int(values["frameID"])
    except ValueError:
        return invalid_deep_link_payload()
    try:
        camera_id = camera_number(values["cameraID"])
    except (TypeError, ValueError):
        return invalid_deep_link_payload()
    parsed_segment = parse_deep_link_segment_id(values["segmentID"])
    if parsed_segment is None:
        return invalid_deep_link_payload()
    source_clip_id, segment_index, segment_frame = parsed_segment
    if segment_frame != frame_id:
        return invalid_deep_link_payload()

    manifest_index = load_deep_link_manifest_index()
    samples = load_samples()
    selected_samples = [sample for sample in samples if str(sample["id"]) == values["clipID"]]
    if not selected_samples:
        return invalid_deep_link_payload()
    if len(selected_samples) != 1:
        raise RuntimeError(f"Duplicate selectable sample id for deep link: {values['clipID']}")
    sample = selected_samples[0]

    if str(int(str(sample["farmId"]))) != farm_id:
        return invalid_deep_link_payload()
    if camera_number(str(sample["cameraId"])) != camera_id:
        return invalid_deep_link_payload()
    if not bool(sample["video"].get("available")):
        return invalid_deep_link_payload()
    if str(sample["sourceVideoId"]) != source_clip_id or int(sample["shardIndex"]) != segment_index:
        return invalid_deep_link_payload()

    entry = manifest_index["bySampleId"].get(str(sample["id"]))
    if entry is None:
        raise RuntimeError(f"Selectable deep-link sample is absent from the generation manifest: {sample['id']}")
    playback = sample["playbackSegment"]
    data_checks = {
        "farm": str(int(str(sample["farmId"]))) == str(manifest_index["farmId"]),
        "camera": camera_number(str(sample["cameraId"])) == camera_number(str(manifest_index["cameraId"])),
        "source clip": str(sample["sourceVideoId"]) == str(entry["sourceClipId"]),
        "source full path": str(sample["sourcePath"]) == str(entry["sourcePath"]),
        "shard index": int(sample["shardIndex"]) == int(entry["shardIndex"]),
        "shard count": int(sample["shardCount"]) == int(entry["shardCount"]),
        "canonical start": int(playback["baseStartFrame"]) == int(entry["canonicalStartFrame"]),
        "canonical end": int(playback["baseEndFrame"]) == int(entry["canonicalEndFrame"]),
        "physical start": int(playback["segmentStartFrame"]) == int(entry["segmentStartFrame"]),
        "physical end": int(playback["segmentEndFrame"]) == int(entry["segmentEndFrame"]),
    }
    failed_checks = [label for label, passed in data_checks.items() if not passed]
    if failed_checks:
        raise RuntimeError(
            f"Deep-link manifest/selectable sample mismatch for {sample['id']}: {', '.join(failed_checks)}"
        )
    if not int(entry["canonicalStartFrame"]) <= frame_id <= int(entry["canonicalEndFrame"]):
        return invalid_deep_link_payload()

    canonical_matches = [
        candidate
        for candidate in manifest_index["bySourceClipId"].get(source_clip_id, [])
        if int(candidate["canonicalStartFrame"]) <= frame_id <= int(candidate["canonicalEndFrame"])
    ]
    if (
        len(canonical_matches) != 1
        or str(canonical_matches[0]["sampleId"]) != str(entry["sampleId"])
    ):
        matched_samples = [str(candidate["sampleId"]) for candidate in canonical_matches]
        raise RuntimeError(
            "Deep-link canonical mapping is not bijective: "
            f"clip={source_clip_id}, frame={frame_id}, matches={matched_samples}"
        )

    return {
        "ok": True,
        "valid": True,
        "sampleId": str(sample["id"]),
        "frameId": frame_id,
    }


def image_size_from_png(path: Path) -> tuple[int, int]:
    require_file(path)
    with path.open("rb") as handle:
        header = handle.read(24)
    if len(header) < 24 or header[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"Not a PNG file: {path}")
    return int.from_bytes(header[16:20], "big"), int.from_bytes(header[20:24], "big")


def polygon_area(points: list[Point]) -> float:
    total = 0.0
    for index, point in enumerate(points):
        next_point = points[(index + 1) % len(points)]
        total += point[0] * next_point[1] - next_point[0] * point[1]
    return total / 2.0


def polygon_centroid(points: list[Point]) -> Point:
    area = polygon_area(points)
    if abs(area) < 1e-9:
        x = sum(point[0] for point in points) / len(points)
        y = sum(point[1] for point in points) / len(points)
        return x, y
    cx = 0.0
    cy = 0.0
    for index, point in enumerate(points):
        next_point = points[(index + 1) % len(points)]
        cross = point[0] * next_point[1] - next_point[0] * point[1]
        cx += (point[0] + next_point[0]) * cross
        cy += (point[1] + next_point[1]) * cross
    factor = 1.0 / (6.0 * area)
    return cx * factor, cy * factor


def convex_hull(points: list[Point]) -> list[Point]:
    unique = sorted(set(points))
    if len(unique) <= 1:
        return unique

    def cross(o: Point, a: Point, b: Point) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[Point] = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)

    upper: list[Point] = []
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)

    return lower[:-1] + upper[:-1]


def rotate_point(point: Point, angle: float) -> Point:
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    return point[0] * cos_a - point[1] * sin_a, point[0] * sin_a + point[1] * cos_a


def min_area_rectangle(points: list[Point]) -> list[Point]:
    hull = convex_hull(points)
    if len(hull) < 3:
        return points

    best_area = math.inf
    best_rectangle: list[Point] | None = None
    for index, point in enumerate(hull):
        next_point = hull[(index + 1) % len(hull)]
        angle = -math.atan2(next_point[1] - point[1], next_point[0] - point[0])
        rotated = [rotate_point(item, angle) for item in hull]
        min_x = min(item[0] for item in rotated)
        max_x = max(item[0] for item in rotated)
        min_y = min(item[1] for item in rotated)
        max_y = max(item[1] for item in rotated)
        area = (max_x - min_x) * (max_y - min_y)
        if area < best_area:
            best_area = area
            corners = [(min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)]
            best_rectangle = [rotate_point(item, -angle) for item in corners]

    if best_rectangle is None:
        raise RuntimeError("Unable to derive minimum-area rectangle")
    return best_rectangle


def point_on_segment(point: Point, a: Point, b: Point, tolerance: float = 1e-7) -> bool:
    px, py = point
    ax, ay = a
    bx, by = b
    cross = (px - ax) * (by - ay) - (py - ay) * (bx - ax)
    if abs(cross) > tolerance:
        return False
    dot = (px - ax) * (px - bx) + (py - ay) * (py - by)
    return dot <= tolerance


def point_in_polygon(point: Point, polygon: list[Point]) -> bool:
    inside = False
    x, y = point
    count = len(polygon)
    for index in range(count):
        a = polygon[index]
        b = polygon[(index + 1) % count]
        if point_on_segment(point, a, b):
            return True
        xi, yi = a
        xj, yj = b
        intersects = (yi > y) != (yj > y)
        if intersects:
            x_at_y = (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi
            if x < x_at_y:
                inside = not inside
    return inside


def orientation(a: Point, b: Point, c: Point) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def segments_intersect(a: Point, b: Point, c: Point, d: Point, tolerance: float = 1e-7) -> bool:
    o1 = orientation(a, b, c)
    o2 = orientation(a, b, d)
    o3 = orientation(c, d, a)
    o4 = orientation(c, d, b)
    if abs(o1) <= tolerance and point_on_segment(c, a, b, tolerance):
        return True
    if abs(o2) <= tolerance and point_on_segment(d, a, b, tolerance):
        return True
    if abs(o3) <= tolerance and point_on_segment(a, c, d, tolerance):
        return True
    if abs(o4) <= tolerance and point_on_segment(b, c, d, tolerance):
        return True
    return (o1 > 0) != (o2 > 0) and (o3 > 0) != (o4 > 0)


def segment_intersects_polygon(a: Point, b: Point, polygon: list[Point]) -> bool:
    if point_in_polygon(a, polygon) or point_in_polygon(b, polygon):
        return True
    for index, point in enumerate(polygon):
        next_point = polygon[(index + 1) % len(polygon)]
        if segments_intersect(a, b, point, next_point):
            return True
    return False


def nearest_point_on_segment(point: Point, a: Point, b: Point) -> Point:
    px, py = point
    ax, ay = a
    bx, by = b
    dx = bx - ax
    dy = by - ay
    length_sq = dx * dx + dy * dy
    if length_sq <= 1e-12:
        return a
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / length_sq))
    return ax + t * dx, ay + t * dy


def point_to_segment_distance_sq(point: Point, a: Point, b: Point) -> float:
    return distance_sq(point, nearest_point_on_segment(point, a, b))


def point_to_polygon_distance_sq(point: Point, polygon: list[Point]) -> float:
    if point_in_polygon(point, polygon):
        return 0.0
    return min(
        point_to_segment_distance_sq(point, polygon[index], polygon[(index + 1) % len(polygon)])
        for index in range(len(polygon))
    )


def distance_sq(a: Point, b: Point) -> float:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return dx * dx + dy * dy


def clamp_point(point: Point, width: int, height: int) -> Point:
    return max(0.0, min(width - 1.0, point[0])), max(0.0, min(height - 1.0, point[1]))


def round_point(point: Point) -> list[float]:
    return [round(point[0], 2), round(point[1], 2)]


def bilinear_point(corners: list[Point], u: float, v: float) -> Point:
    p00, p10, p11, p01 = corners
    x = (
        (1.0 - u) * (1.0 - v) * p00[0]
        + u * (1.0 - v) * p10[0]
        + u * v * p11[0]
        + (1.0 - u) * v * p01[0]
    )
    y = (
        (1.0 - u) * (1.0 - v) * p00[1]
        + u * (1.0 - v) * p10[1]
        + u * v * p11[1]
        + (1.0 - u) * v * p01[1]
    )
    return x, y


def invert_bilinear_point(point: Point, corners: list[Point]) -> tuple[float, float] | None:
    u = 0.5
    v = 0.5
    for _ in range(12):
        x, y = bilinear_point(corners, u, v)
        fx = x - point[0]
        fy = y - point[1]
        if fx * fx + fy * fy < 1e-8:
            return u, v

        p00, p10, p11, p01 = corners
        du_x = (1.0 - v) * (p10[0] - p00[0]) + v * (p11[0] - p01[0])
        du_y = (1.0 - v) * (p10[1] - p00[1]) + v * (p11[1] - p01[1])
        dv_x = (1.0 - u) * (p01[0] - p00[0]) + u * (p11[0] - p10[0])
        dv_y = (1.0 - u) * (p01[1] - p00[1]) + u * (p11[1] - p10[1])
        det = du_x * dv_y - dv_x * du_y
        if abs(det) < 1e-9:
            return None

        step_u = (fx * dv_y - dv_x * fy) / det
        step_v = (du_x * fy - fx * du_y) / det
        u -= step_u
        v -= step_v
        if step_u * step_u + step_v * step_v < 1e-12:
            return u, v
    return u, v


def nearest_unique_points(targets: list[Point], candidates: list[Point]) -> list[Point] | None:
    remaining = list(candidates)
    if len(remaining) < len(targets):
        return None
    out: list[Point] = []
    for target in targets:
        best_index = min(range(len(remaining)), key=lambda index: distance_sq(target, remaining[index]))
        out.append(remaining.pop(best_index))
    return out


def plan_canvas_size(quarter_turns: int) -> tuple[int, int]:
    turns = normalize_quarter_turns(quarter_turns)
    if turns % 2:
        return CANVAS_HEIGHT, CANVAS_WIDTH
    return CANVAS_WIDTH, CANVAS_HEIGHT


def rotate_plan_point(point: Point, quarter_turns: int) -> Point:
    x, y = point
    turns = normalize_quarter_turns(quarter_turns)
    if turns == 0:
        return x, y
    if turns == 1:
        return CANVAS_HEIGHT - y, x
    if turns == 2:
        return CANVAS_WIDTH - x, CANVAS_HEIGHT - y
    return y, CANVAS_WIDTH - x


def polygon_bounds(points: list[Point]) -> dict[str, float]:
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return {
        "minX": round(min(xs), 2),
        "minY": round(min(ys), 2),
        "maxX": round(max(xs), 2),
        "maxY": round(max(ys), 2),
    }


def normalize_color(value: Any) -> str:
    color = str(value or "").strip()
    if not color:
        return "#0b7285"
    return color


def load_sna_resource_floorplan(sample: dict[str, Any], plan_quarter_turns: int) -> dict[str, Any]:
    path = sna_floorplan_file(sample["farmId"], sample["cameraId"])
    data = read_json(path)
    group_id = floorplan_group_id(sample["farmId"], sample["cameraId"])
    if str(data.get("schemaVersion", "")) != "sna_zones.v1":
        raise ValueError(f"Unsupported SNA floorplan schema: {path}")
    if str(data.get("groupId", "")) != group_id:
        raise ValueError(f"SNA floorplan group mismatch for {group_id}: {path}")
    image = data.get("image", {})
    if int(image.get("width", 0) or 0) != CANVAS_WIDTH or int(image.get("height", 0) or 0) != CANVAS_HEIGHT:
        raise ValueError(f"Unexpected SNA floorplan size for {group_id}: {path}")
    coordinate_name = str(data.get("coordinate_system") or data.get("coordinateSystem", {}).get("name") or "")
    if coordinate_name != "reference_frame_pixel":
        raise ValueError(f"Unexpected SNA floorplan coordinate system for {group_id}: {coordinate_name}")

    zones: list[dict[str, Any]] = []
    seen_zone_ids: set[str] = set()
    duplicate_zone_ids = 0
    raw_zones = data.get("zones", [])
    if not isinstance(raw_zones, list):
        raise ValueError(f"SNA floorplan zones must be a list: {path}")
    for index, zone in enumerate(raw_zones, start=1):
        if not isinstance(zone, dict):
            raise ValueError(f"SNA floorplan zone must be an object: {path}")
        zone_id = str(zone.get("zone_id") or f"zone_{index}").strip()
        if zone_id in seen_zone_ids:
            duplicate_zone_ids += 1
            continue
        seen_zone_ids.add(zone_id)
        zone_type = str(zone.get("zone_type") or "unknown").strip()
        label = str(zone.get("label") or zone_type or zone_id).strip()
        raw_points = zone.get("polygon", [])
        if not isinstance(raw_points, list) or len(raw_points) < 3:
            raise ValueError(f"SNA floorplan zone has fewer than 3 points: {zone_id}")
        points = []
        for point in raw_points:
            if not isinstance(point, (list, tuple)) or len(point) < 2:
                raise ValueError(f"SNA floorplan zone point is invalid: {zone_id}")
            points.append((float(point[0]), float(point[1])))
        rotated_points = [rotate_plan_point(point, plan_quarter_turns) for point in points]
        zones.append(
            {
                "zoneId": zone_id,
                "zoneType": zone_type,
                "label": label,
                "color": normalize_color(zone.get("color")),
                "points": [round_point(point) for point in rotated_points],
                "centroid": round_point(polygon_centroid(rotated_points)),
                "bounds": polygon_bounds(rotated_points),
            }
        )

    return {
        "available": True,
        "path": str(path),
        "schemaVersion": str(data.get("schemaVersion", "")),
        "groupId": group_id,
        "coordinateSystem": coordinate_name,
        "sourceWidth": CANVAS_WIDTH,
        "sourceHeight": CANVAS_HEIGHT,
        "planRotationQuarterTurns": normalize_quarter_turns(plan_quarter_turns),
        "unannotatedZoneType": "path",
        "unannotatedDisplayRule": "not drawn in WebUI; downstream SNA treats unannotated space as path",
        "zoneCount": len(zones),
        "duplicateZoneIdCount": duplicate_zone_ids,
        "zones": zones,
    }


def tracking_bbox_center(row: dict[str, str]) -> Point:
    x = float(row["x"])
    y = float(row["y"])
    w = float(row["w"])
    h = float(row["h"])
    return x + w / 2.0, y + h / 2.0


def mapping_profile_for_dimensions(width: int, height: int) -> dict[str, Any]:
    if (width, height) == (PORTRAIT_TRACKING_WIDTH, PORTRAIT_TRACKING_HEIGHT):
        return {
            "name": "portrait_ccw_to_floorplan",
            "typeId": "S",
            "trackingWidth": width,
            "trackingHeight": height,
            "description": "bbox_center_then_counterclockwise_90deg_to_2d_canvas: [x,y] -> [y,2160-x]",
        }
    if (width, height) == (LANDSCAPE_TRACKING_WIDTH, LANDSCAPE_TRACKING_HEIGHT):
        return {
            "name": "landscape_identity_to_floorplan",
            "typeId": "H",
            "trackingWidth": width,
            "trackingHeight": height,
            "description": "bbox_center_already_in_landscape_2d_canvas: [x,y] -> [x,y]",
        }
    raise ValueError(f"Unexpected tracking size: {width} x {height}")


def map_tracking_to_floorplan(point: Point, mapping_profile: dict[str, Any]) -> Point:
    name = mapping_profile["name"]
    if name == "portrait_ccw_to_floorplan":
        # Visual check selected this portrait-to-landscape rotation for short samples.
        return point[1], float(mapping_profile["trackingWidth"]) - point[0]
    if name == "landscape_identity_to_floorplan":
        # Current tracking CSVs are already in the floorplan's landscape orientation.
        return point
    raise ValueError(f"Unsupported mapping profile: {name}")


class FloorGeometry:
    def __init__(self, annotation: dict[str, Any], plan_quarter_turns: int = 0):
        source_width = int(annotation["image"]["width"])
        source_height = int(annotation["image"]["height"])
        if (source_width, source_height) != (CANVAS_WIDTH, CANVAS_HEIGHT):
            raise ValueError(f"Unexpected floorplan size: {source_width} x {source_height}")

        self.plan_quarter_turns = normalize_quarter_turns(plan_quarter_turns)
        self.width, self.height = plan_canvas_size(self.plan_quarter_turns)

        self.polygons: list[dict[str, Any]] = []
        self.base_shapes: list[dict[str, Any]] = []
        self.reference_shapes: list[dict[str, Any]] = []
        self.invalid_shapes: list[dict[str, Any]] = []
        self.red_shapes: list[dict[str, Any]] = []
        self.green_shapes: list[dict[str, Any]] = []
        self.semantic_regions: list[dict[str, Any]] = []
        for polygon in annotation.get("polygons", []):
            raw_points = [(float(point[0]), float(point[1])) for point in polygon["points"]]
            force_rectangle = bool(polygon.get("force2DRectangle"))
            shape_points = min_area_rectangle(raw_points) if force_rectangle else raw_points
            rotated_raw_points = [rotate_plan_point(point, self.plan_quarter_turns) for point in raw_points]
            rotated_shape_points = [rotate_plan_point(point, self.plan_quarter_turns) for point in shape_points]
            cleaned = {
                "id": polygon["id"],
                "classId": polygon["classId"],
                "label": polygon.get("label", polygon["classId"]),
                "color": polygon.get("color"),
                "force2DRectangle": force_rectangle,
                "rectangleFitMethod": RECTANGLE_FIT_METHOD if force_rectangle else None,
                "rawPoints": rotated_raw_points,
                "shapePoints": rotated_shape_points,
                "centroid": polygon_centroid(rotated_shape_points),
            }
            self.polygons.append(cleaned)

            is_base = force_rectangle and polygon["classId"] in {"obstacle", "resource"}
            is_reference = polygon["classId"] == "walkable_ground" or (
                polygon["classId"] == "obstacle" and not force_rectangle
            )
            if polygon["classId"] == "obstacle":
                self.red_shapes.append(cleaned)
                self.invalid_shapes.append(cleaned)
            elif polygon["classId"] == "resource":
                self.green_shapes.append(cleaned)
                self.invalid_shapes.append(cleaned)

            if is_base:
                cleaned["role"] = "base"
                self.base_shapes.append(cleaned)
            elif is_reference:
                cleaned["role"] = "reference"
                self.reference_shapes.append(cleaned)
            else:
                cleaned["role"] = "unused"

            semantic_region = self.make_semantic_region(cleaned)
            if semantic_region is not None:
                self.semantic_regions.append(semantic_region)

    def make_semantic_region(self, polygon: dict[str, Any]) -> dict[str, Any] | None:
        if not polygon["force2DRectangle"] or len(polygon["shapePoints"]) != 4 or len(polygon["rawPoints"]) < 4:
            return None
        source_quad = nearest_unique_points(polygon["shapePoints"], polygon["rawPoints"])
        if source_quad is None:
            return None
        target_quad = polygon["shapePoints"]
        area = abs(polygon_area(target_quad))
        if area < 1.0:
            return None
        class_weight = 1.0 if polygon["classId"] == "walkable_ground" else 0.42
        return {
            "id": polygon["id"],
            "classId": polygon["classId"],
            "sourcePolygon": polygon["rawPoints"],
            "sourceQuad": source_quad,
            "targetQuad": target_quad,
            "radius": max(96.0, math.sqrt(area) * 0.85),
            "weight": class_weight,
        }

    def final_class(self, point: Point) -> str:
        if not (0 <= point[0] < self.width and 0 <= point[1] < self.height):
            return "outside"
        for class_id in ("resource", "obstacle"):
            for polygon in self.invalid_shapes:
                if polygon["classId"] == class_id and point_in_polygon(point, polygon["shapePoints"]):
                    return class_id
        return "open_2d_space"

    def is_allowed(self, point: Point) -> bool:
        return self.final_class(point) == "open_2d_space"

    def nearest_allowed_point(self, point: Point) -> tuple[Point, str]:
        clamped = clamp_point(point, self.width, self.height)
        if self.is_allowed(clamped):
            reason = "clamped" if distance_sq(point, clamped) > 1e-6 else "unchanged"
            return clamped, reason

        candidates: list[tuple[float, Point]] = []
        for polygon in self.invalid_shapes:
            if not point_in_polygon(clamped, polygon["shapePoints"]):
                continue
            best_boundary: Point | None = None
            best_distance = math.inf
            shape_points = polygon["shapePoints"]
            for index, a in enumerate(shape_points):
                b = shape_points[(index + 1) % len(shape_points)]
                boundary = nearest_point_on_segment(clamped, a, b)
                current_distance = distance_sq(clamped, boundary)
                if current_distance < best_distance:
                    best_distance = current_distance
                    best_boundary = boundary
            if best_boundary is None:
                continue
            cx, cy = polygon["centroid"]
            vx = best_boundary[0] - cx
            vy = best_boundary[1] - cy
            length = math.hypot(vx, vy) or 1.0
            for margin in (SQUEEZE_MARGIN_PX, 16.0, 32.0, 64.0):
                candidate = (
                    best_boundary[0] + vx / length * margin,
                    best_boundary[1] + vy / length * margin,
                )
                candidate = clamp_point(candidate, self.width, self.height)
                if self.is_allowed(candidate):
                    candidates.append((distance_sq(clamped, candidate), candidate))

        if candidates:
            candidates.sort(key=lambda item: item[0])
            return candidates[0][1], "squeezed_to_boundary"

        for radius in range(8, int(math.hypot(self.width, self.height)) + 16, 8):
            for step in range(96):
                angle = math.tau * step / 96.0
                candidate = (
                    clamped[0] + math.cos(angle) * radius,
                    clamped[1] + math.sin(angle) * radius,
                )
                candidate = clamp_point(candidate, self.width, self.height)
                if self.is_allowed(candidate):
                    return candidate, "squeezed_by_radial_search"

        raise RuntimeError(f"No allowed walkable point found near {point}")

    def segment_crosses_red(self, a: Point, b: Point) -> bool:
        return any(segment_intersects_polygon(a, b, polygon["shapePoints"]) for polygon in self.red_shapes)

    def semantic_corresponded_point(self, point: Point) -> tuple[Point, str]:
        if not self.semantic_regions:
            return point, "semantic_unavailable"

        total_weight = 0.0
        dx_total = 0.0
        dy_total = 0.0
        inside_count = 0
        for region in self.semantic_regions:
            uv = invert_bilinear_point(point, region["sourceQuad"])
            if uv is None:
                continue
            u, v = uv
            if u < -0.85 or u > 1.85 or v < -0.85 or v > 1.85:
                continue
            mapped = bilinear_point(region["targetQuad"], u, v)
            move_dx = mapped[0] - point[0]
            move_dy = mapped[1] - point[1]
            max_move = max(128.0, region["radius"] * 1.25)
            move_len = math.hypot(move_dx, move_dy)
            if move_len > max_move:
                scale = max_move / move_len
                move_dx *= scale
                move_dy *= scale

            distance = math.sqrt(point_to_polygon_distance_sq(point, region["sourcePolygon"]))
            if distance <= 1e-6:
                proximity_weight = 1.0
                inside_count += 1
            else:
                influence = region["radius"] * SEMANTIC_INFLUENCE_MULTIPLIER
                if distance > influence:
                    continue
                proximity_weight = math.exp(-(distance * distance) / (2.0 * region["radius"] * region["radius"]))
            weight = proximity_weight * float(region["weight"])
            dx_total += move_dx * weight
            dy_total += move_dy * weight
            total_weight += weight

        if total_weight <= 1e-9:
            return point, "semantic_no_anchor"
        candidate = clamp_point((point[0] + dx_total / total_weight, point[1] + dy_total / total_weight), self.width, self.height)
        reason = "semantic_rectified_inside_region" if inside_count else "semantic_rectified_near_region"
        return candidate, reason


def report_progress(progress: ProgressCallback | None, label: str, current: int | None = None, total: int | None = None) -> None:
    if progress is not None:
        progress(label, current, total)


def count_csv_data_rows(path: Path, progress: ProgressCallback | None, label: str) -> int:
    total = 0
    report_progress(progress, label, 0, None)
    with path.open("r", encoding="utf-8", newline="") as handle:
        header = handle.readline()
        if not header:
            return 0
        for total, _line in enumerate(handle, start=1):
            if total % 10000 == 0:
                report_progress(progress, label, total, None)
    report_progress(progress, label, total, total)
    return total


def load_tracking_rows(
    sample: dict[str, Any],
    progress: ProgressCallback | None = None,
) -> tuple[list[dict[str, str]], dict[str, Any], dict[str, Any]]:
    tracking_csv = sample["trackingCsv"]
    require_file(tracking_csv)
    row_total = count_csv_data_rows(tracking_csv, progress, "Counting tracking_boxes.csv rows")
    rows: list[dict[str, str]] = []
    with tracking_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"video", "frame", "track_id", "x", "y", "w", "h", "score", "identity", "id_conf"}
        if set(reader.fieldnames or []) != required:
            missing = sorted(required - set(reader.fieldnames or []))
            extra = sorted(set(reader.fieldnames or []) - required)
            raise ValueError(f"Unexpected tracking CSV columns; missing={missing}, extra={extra}")
        report_progress(progress, "Reading tracking_boxes.csv rows", 0, row_total)
        for index, row in enumerate(reader, start=1):
            rows.append(row)
            if index % 10000 == 0:
                report_progress(progress, "Reading tracking_boxes.csv rows", index, row_total)
        report_progress(progress, "Reading tracking_boxes.csv rows", len(rows), row_total)

    report_progress(progress, "Reading manifest.json", 0, 1)
    manifest = read_json(sample["manifest"])
    report_progress(progress, "Reading manifest.json", 1, 1)
    video_manifest = manifest.get("video_manifest", {})
    mapping_profile = mapping_profile_for_dimensions(
        int(video_manifest.get("width")),
        int(video_manifest.get("height")),
    )
    return rows, manifest, mapping_profile


def mapping_profile_for_sample(sample: dict[str, Any]) -> dict[str, Any]:
    manifest = read_json(sample["manifest"])
    video_manifest = manifest.get("video_manifest", {})
    return mapping_profile_for_dimensions(
        int(video_manifest.get("width")),
        int(video_manifest.get("height")),
    )


def plan_rotation_for_sample(sample: dict[str, Any], mapping_profile: dict[str, Any] | None = None) -> int:
    profile = mapping_profile if mapping_profile is not None else mapping_profile_for_sample(sample)
    return get_saved_rotation("plan", sample["farmId"], sample["cameraId"], str(profile["typeId"]))


def verified_reid_record_for_tracking_row(
    row: dict[str, str],
    reid_clip: ReIdClip,
) -> tuple[int, int, ReIdRecord]:
    frame_value = float(row["frame"])
    track_value = float(row["track_id"])
    if not frame_value.is_integer() or not track_value.is_integer():
        raise ValueError(
            f"Tracking frame/track_id must be integers: frame={row['frame']!r}, track_id={row['track_id']!r}"
        )
    frame = int(frame_value)
    track_id = int(track_value)
    if str(row["video"]).strip() != f"{reid_clip.clip_id}.MP4":
        raise ValueError(
            "Tracking video column disagrees with the authoritative full-path re-ID mapping: "
            f"expected={reid_clip.clip_id}.MP4, actual={row['video']!r}, source={reid_clip.source_path}"
        )
    record = reid_clip.records.get((frame, track_id))
    if record is None:
        raise ValueError(
            "Tracking row is missing from the strict re-ID index: "
            f"sequence={reid_clip.sequence_id}, clip={reid_clip.clip_id}, frame={frame}, track={track_id}"
        )

    x = float(row["x"])
    y = float(row["y"])
    w = float(row["w"])
    h = float(row["h"])
    score = float(row["score"])
    if not all(math.isfinite(value) for value in (x, y, w, h, score)):
        raise ValueError(f"Non-finite tracking bbox for clip={reid_clip.clip_id}, frame={frame}, track={track_id}")
    if w < 0.0 or h < 0.0:
        raise ValueError(f"Negative tracking bbox size for clip={reid_clip.clip_id}, frame={frame}, track={track_id}")
    clamped_bbox = (
        max(0.0, min(float(reid_clip.width), x)),
        max(0.0, min(float(reid_clip.height), y)),
        max(0.0, min(float(reid_clip.width), x + w)),
        max(0.0, min(float(reid_clip.height), y + h)),
    )
    reid_bbox = (record.x1, record.y1, record.x2, record.y2)
    max_delta = max(abs(clamped_bbox[index] - reid_bbox[index]) for index in range(4))
    if max_delta > REID_BBOX_TOLERANCE_PX:
        raise ValueError(
            "Tracking/re-ID bbox mismatch after clamping to manifest bounds: "
            f"clip={reid_clip.clip_id}, frame={frame}, track={track_id}, "
            f"tracking={clamped_bbox}, reid={reid_bbox}, max_delta={max_delta}"
        )
    confidence_delta = abs(score - record.bbox_confidence)
    if confidence_delta > REID_CONFIDENCE_TOLERANCE:
        raise ValueError(
            "Tracking/re-ID bbox confidence mismatch: "
            f"clip={reid_clip.clip_id}, frame={frame}, track={track_id}, "
            f"tracking={score}, reid={record.bbox_confidence}, delta={confidence_delta}"
        )
    return frame, track_id, record


def build_payload(sample: dict[str, Any], progress: ProgressCallback | None = None) -> dict[str, Any]:
    rows, manifest, mapping_profile = load_tracking_rows(sample, progress)
    authoritative_source_path = source_video_path_from_manifest(manifest)
    if authoritative_source_path != str(sample["sourcePath"]):
        raise ValueError(
            "Sample sourcePath disagrees with authoritative manifest.video_manifest.source_path: "
            f"sample={sample['sourcePath']!r}, manifest={authoritative_source_path!r}"
        )
    report_progress(progress, "Loading strict re-identification index", 0, 1)
    reid_clip = load_reid_clip_for_source(authoritative_source_path)
    report_progress(progress, "Loading strict re-identification index", 1, 1)
    plan_turns = plan_rotation_for_sample(sample, mapping_profile)
    report_progress(progress, "Building floorplan geometry", 0, 1)
    geometry = get_geometry(sample, plan_turns)
    report_progress(progress, "Building floorplan geometry", 1, 1)
    video_manifest = manifest.get("video_manifest", {})
    fps = float(video_manifest.get("fps", 29.97002997002997))
    if (
        int(video_manifest.get("width")) != reid_clip.width
        or int(video_manifest.get("height")) != reid_clip.height
        or abs(fps - reid_clip.fps) > 1e-9
    ):
        raise ValueError(
            "Manifest geometry/fps disagrees with strict full-path re-ID index: "
            f"source={authoritative_source_path}"
        )
    freeze_frames = max(1, int(math.ceil(fps * POINT_FREEZE_SECONDS)))

    frames: dict[int, list[dict[str, Any]]] = {}
    track_ids: set[int] = set()
    seen_detection_keys: set[tuple[int, int]] = set()
    global_identities: dict[str, dict[str, Any]] = {}
    reid_valid_row_count = 0
    reid_invalid_row_count = 0
    reid_invalid_reason_counts: dict[str, int] = {}
    tracking_frame_min: int | None = None
    tracking_frame_max: int | None = None
    report_progress(progress, "Joining re-identification and mapping tracking boxes", 0, len(rows))
    for row_index, row in enumerate(rows, start=1):
        frame, track_id, reid_record = verified_reid_record_for_tracking_row(row, reid_clip)
        detection_key = (frame, track_id)
        if detection_key in seen_detection_keys:
            raise ValueError(
                f"Duplicate tracking key inside sample {sample['id']}: frame={frame}, track={track_id}"
            )
        seen_detection_keys.add(detection_key)
        tracking_frame_min = frame if tracking_frame_min is None else min(tracking_frame_min, frame)
        tracking_frame_max = frame if tracking_frame_max is None else max(tracking_frame_max, frame)
        if row_index % 10000 == 0:
            report_progress(
                progress,
                "Joining re-identification and mapping tracking boxes",
                row_index,
                len(rows),
            )

        if not reid_record.valid:
            reid_invalid_row_count += 1
            reason = reid_record.invalid_reason
            reid_invalid_reason_counts[reason] = reid_invalid_reason_counts.get(reason, 0) + 1
            continue
        if (
            reid_record.global_track_id is None
            or reid_record.global_track_uuid is None
            or reid_record.display_global_id is None
            or reid_record.id_status is None
        ):
            raise RuntimeError(
                f"valid re-ID record has incomplete global identity: clip={reid_clip.clip_id}, frame={frame}, track={track_id}"
            )
        reid_valid_row_count += 1
        global_track_id = int(reid_record.global_track_id)
        global_track_uuid = str(reid_record.global_track_uuid)
        display_global_id = str(reid_record.display_global_id)
        reid_id_status = str(reid_record.id_status)
        identity = global_identities.setdefault(
            global_track_uuid,
            {
                "globalTrackId": global_track_id,
                "globalTrackUuid": global_track_uuid,
                "displayGlobalId": display_global_id,
                "localTrackIds": set(),
                "idStatuses": set(),
            },
        )
        if (
            int(identity["globalTrackId"]) != global_track_id
            or str(identity["displayGlobalId"]) != display_global_id
        ):
            raise RuntimeError(f"Non-bijective global identity in strict re-ID index: {global_track_uuid}")
        identity["localTrackIds"].add(track_id)
        identity["idStatuses"].add(reid_id_status)

        raw_floor = clamp_point(
            map_tracking_to_floorplan(tracking_bbox_center(row), mapping_profile),
            geometry.width,
            geometry.height,
        )
        semantic_floor, semantic_reason = geometry.semantic_corresponded_point(raw_floor)
        raw_point = round_point(raw_floor)
        semantic_point = round_point(semantic_floor)
        track_ids.add(track_id)
        frames.setdefault(frame, []).append(
            {
                "trackId": track_id,
                "globalTrackId": global_track_id,
                "globalTrackUuid": global_track_uuid,
                "displayGlobalId": display_global_id,
                "reidIdStatus": reid_id_status,
                "sourceVideo": row["video"],
                "raw": raw_point,
                "point": raw_point,
                "adjustmentPx": 0.0,
                "adjustmentReason": "realtime_red_green_exclusion",
                "semanticRaw": raw_point,
                "semanticWarped": semantic_point,
                "semanticPoint": semantic_point,
                "semanticAdjustmentPx": 0.0,
                "semanticAdjustmentReason": f"{semantic_reason};realtime_red_green_exclusion",
                "classBeforeAdjustment": "realtime",
                "score": round(float(row["score"]), 4),
                "identity": row["identity"],
                "idConfidence": round(float(row["id_conf"]), 4),
            }
        )
    report_progress(
        progress,
        "Joining re-identification and mapping tracking boxes",
        len(rows),
        len(rows),
    )
    if len(seen_detection_keys) != len(rows):
        raise RuntimeError(
            f"Tracking/re-ID sample join was not one-to-one for {sample['id']}: keys={len(seen_detection_keys)}, rows={len(rows)}"
        )
    if reid_valid_row_count + reid_invalid_row_count != len(rows):
        raise RuntimeError(f"Tracking/re-ID validity accounting failed for {sample['id']}")

    serialized_global_identities = [
        {
            "globalTrackId": int(identity["globalTrackId"]),
            "globalTrackUuid": str(identity["globalTrackUuid"]),
            "displayGlobalId": str(identity["displayGlobalId"]),
            "localTrackIds": sorted(int(track_id) for track_id in identity["localTrackIds"]),
            "idStatuses": sorted(str(status) for status in identity["idStatuses"]),
        }
        for identity in sorted(global_identities.values(), key=lambda item: int(item["globalTrackId"]))
    ]
    reid_sequence_id = reid_clip.sequence_id
    reid_clip_id = reid_clip.clip_id
    del reid_clip, global_identities, seen_detection_keys

    raw_frame_min = tracking_frame_min if tracking_frame_min is not None else 0
    raw_frame_max = tracking_frame_max if tracking_frame_max is not None else 0
    manifest_frame_count = int(video_manifest.get("frame_count", 0) or 0)
    frame_min = raw_frame_min
    frame_max = max(raw_frame_max, manifest_frame_count - 1) if manifest_frame_count > 0 else raw_frame_max
    recent_track_points = _RecentTrackPoints(freeze_frames)
    frame_items: list[dict[str, Any]] = []
    frozen_point_count = 0
    max_frozen_points_in_frame = 0
    frames_with_frozen_points = 0
    frame_total = max(0, frame_max - frame_min + 1)

    report_progress(progress, "Building frame timeline", 0, frame_total)
    for frame_offset, frame in enumerate(range(frame_min, frame_max + 1), start=1):
        recent_track_points.begin_frame(frame)
        frame_points: list[dict[str, Any]] = []
        current_points = {int(point["trackId"]): point for point in frames.get(frame, [])}
        current_global_tracks: dict[str, int] = {}
        for track_id, point in sorted(current_points.items()):
            global_track_uuid = str(point["globalTrackUuid"])
            previous_track = current_global_tracks.get(global_track_uuid)
            if previous_track is not None and previous_track != track_id:
                raise RuntimeError(
                    "Multiple current local detections map to one global identity in the same frame: "
                    f"frame={frame}, global={global_track_uuid}, local={previous_track}/{track_id}"
                )
            current_global_tracks[global_track_uuid] = track_id
            copied = dict(point)
            copied["frozen"] = False
            copied["lastSeenFrame"] = frame
            copied["freezeAgeFrames"] = 0
            copied["freezeAgeSeconds"] = 0.0
            frame_points.append(copied)
            recent_track_points.remember(track_id, frame, copied)

        frozen_candidates = recent_track_points.frozen_candidates(
            frame,
            current_points,
            current_global_tracks,
        )

        for track_id, last_frame, last_point in sorted(
            frozen_candidates.values(), key=lambda item: item[0]
        ):
            age_frames = frame - last_frame
            copied = dict(last_point)
            copied["frozen"] = True
            copied["lastSeenFrame"] = last_frame
            copied["freezeAgeFrames"] = age_frames
            copied["freezeAgeSeconds"] = round(age_frames / fps, 4) if fps > 0 else 0.0
            frame_points.append(copied)

        frozen_this_frame = len(frozen_candidates)
        frozen_point_count += frozen_this_frame
        max_frozen_points_in_frame = max(max_frozen_points_in_frame, frozen_this_frame)
        if frozen_this_frame:
            frames_with_frozen_points += 1
        frame_points.sort(key=lambda point: int(point["trackId"]))
        frame_global_uuids = [str(point["globalTrackUuid"]) for point in frame_points]
        if len(frame_global_uuids) != len(set(frame_global_uuids)):
            raise RuntimeError(f"Duplicate global identity remained in display frame {frame}")
        frame_items.append({"frame": frame, "points": frame_points})
        if frame_offset % 1000 == 0:
            report_progress(progress, "Building frame timeline", frame_offset, frame_total)
    report_progress(progress, "Building frame timeline", frame_total, frame_total)

    def serialize_polygon(polygon: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": polygon["id"],
            "classId": polygon["classId"],
            "label": polygon["label"],
            "color": polygon["color"],
            "role": polygon["role"],
            "force2DRectangle": polygon["force2DRectangle"],
            "rectangleFitMethod": polygon["rectangleFitMethod"],
            "rawPoints": [round_point(point) for point in polygon["rawPoints"]],
            "shapePoints": [round_point(point) for point in polygon["shapePoints"]],
        }

    polygons = [
        {
            "id": polygon["id"],
            "classId": polygon["classId"],
            "label": polygon["label"],
            "color": polygon["color"],
            "role": polygon["role"],
            "force2DRectangle": polygon["force2DRectangle"],
            "rectangleFitMethod": polygon["rectangleFitMethod"],
            "rawPoints": [round_point(point) for point in polygon["rawPoints"]],
            "shapePoints": [round_point(point) for point in polygon["shapePoints"]],
        }
        for polygon in geometry.polygons
    ]
    resource_floorplan = load_sna_resource_floorplan(sample, plan_turns)

    return {
        "meta": {
            "title": "Dairy Cattle Floor Plan",
            "floorplanAnnotation": str(sample["floorplanAnnotation"]),
            "snaFloorplan": resource_floorplan["path"],
            "floorplanGroup": floorplan_group_id(sample["farmId"], sample["cameraId"]),
            "trackingCsv": str(sample["trackingCsv"]),
            "manifest": str(sample["manifest"]),
            "keypointsCsv": str(sample["keypointsCsv"]),
            "stage2SourceDir": str(sample["sourceDir"]),
            "stage2CodeDir": str(STAGE2_CODE_DIR),
            "sample": public_sample(sample),
            "samples": [public_sample(item) for item in load_samples()],
            "width": geometry.width,
            "height": geometry.height,
            "sourceFloorplanWidth": CANVAS_WIDTH,
            "sourceFloorplanHeight": CANVAS_HEIGHT,
            "planRotationQuarterTurns": plan_turns,
            "planRotationDegrees": plan_turns * 90,
            "trackingWidth": mapping_profile["trackingWidth"],
            "trackingHeight": mapping_profile["trackingHeight"],
            "mappingProfile": mapping_profile["name"],
            "mappingTypeId": mapping_profile["typeId"],
            "semanticMappingAvailable": bool(geometry.semantic_regions),
            "semanticMappingRegionCount": len(geometry.semantic_regions),
            "semanticMapping": (
                "display-only local polygon-to-rectangle correspondence field; "
                "Stage2 neural inference still uses the original CSV coordinates"
            ),
            "fps": fps,
            "frameMin": frame_min,
            "frameMax": frame_max,
            "frameCount": len(frame_items),
            "rawFrameMin": raw_frame_min,
            "rawFrameMax": raw_frame_max,
            "rawFrameCount": len(frames),
            "rowCount": len(rows),
            "displayDetectionRowCount": reid_valid_row_count,
            "reidValidRowCount": reid_valid_row_count,
            "reidInvalidRowCount": reid_invalid_row_count,
            "reidInvalidReasonCounts": dict(sorted(reid_invalid_reason_counts.items())),
            "reidSourceCsv": str(REID_SOURCE_CSV),
            "reidIndex": str(REID_INDEX_PATH),
            "reidSequenceId": reid_sequence_id,
            "reidClipId": reid_clip_id,
            "reidBboxRule": "clamp Stage1 xyxy to manifest width/height, then compare",
            "reidBboxTolerancePx": REID_BBOX_TOLERANCE_PX,
            "reidConfidenceTolerance": REID_CONFIDENCE_TOLERANCE,
            "reidValidityPolicy": REID_VALIDITY_POLICY,
            "trackIds": sorted(track_ids),
            "globalIdentities": serialized_global_identities,
            "pointFreezeSeconds": POINT_FREEZE_SECONDS,
            "pointFreezeFrames": freeze_frames,
            "frozenPointCount": frozen_point_count,
            "framesWithFrozenPoints": frames_with_frozen_points,
            "maxFrozenPointsInFrame": max_frozen_points_in_frame,
            "mapping": mapping_profile["description"],
            "structureRule": "forced 2D rectangles in red/green are the base 2D structure; all other canvas space is white/transparent",
            "referenceAreaRule": "manual blue polygons and non-rectangle red polygons are optional reference areas",
            "allowedClass": "open_2d_space outside every red/green polygon",
            "pointExclusion": "realtime red/green exclusion is applied when a frame is displayed or requested by Stage2",
            "baseShapeCount": len(geometry.base_shapes),
            "referenceShapeCount": len(geometry.reference_shapes),
            "redShapeCount": len(geometry.red_shapes),
            "greenShapeCount": len(geometry.green_shapes),
            "resourceFloorplanZoneCount": resource_floorplan["zoneCount"],
            "classCountsBeforeAdjustment": {},
            "movedPointCount": 0,
            "meanMovedDistancePx": 0.0,
            "maxMovedDistancePx": 0.0,
        },
        "polygons": polygons,
        "basePolygons": [serialize_polygon(polygon) for polygon in geometry.base_shapes],
        "referencePolygons": [serialize_polygon(polygon) for polygon in geometry.reference_shapes],
        "redPolygons": [serialize_polygon(polygon) for polygon in geometry.red_shapes],
        "greenPolygons": [serialize_polygon(polygon) for polygon in geometry.green_shapes],
        "invalidPolygons": [serialize_polygon(polygon) for polygon in geometry.invalid_shapes],
        "resourceFloorplan": resource_floorplan,
        "frames": frame_items,
    }


GEOMETRY_CACHE: dict[tuple[Path, int], FloorGeometry] = {}
DATA_CACHE: dict[str, dict[str, Any]] = {}
LOAD_JOBS: dict[str, dict[str, Any]] = {}
LOAD_JOBS_LOCK = threading.Lock()
ROTATION_SETTINGS_LOCK = threading.Lock()
ROTATION_SCOPES = ("map", "plan", "ui")


def rotation_key(farm_id: str, camera_id: str, type_id: str) -> str:
    farm = str(farm_id).strip()
    camera = camera_number(str(camera_id).strip())
    kind = str(type_id).strip().upper()
    if not farm:
        raise ValueError("missing farmId")
    if kind not in {"H", "S"}:
        raise ValueError(f"unsupported rotation typeId: {type_id}")
    return f"{farm}|{camera}|{kind}"


def normalize_quarter_turns(value: Any) -> int:
    return int(value) % 4


def normalize_rotation_settings(data: Any) -> dict[str, dict[str, int]]:
    if not isinstance(data, dict):
        raise ValueError(f"Rotation settings must be a JSON object: {ROTATION_SETTINGS_FILE}")
    settings: dict[str, dict[str, int]] = {scope: {} for scope in ROTATION_SCOPES}
    for scope in ROTATION_SCOPES:
        scoped = data.get(scope)
        if scoped is None:
            continue
        if not isinstance(scoped, dict):
            raise ValueError(f"Rotation settings scope must be a JSON object: {scope}")
        settings[scope] = {str(key): normalize_quarter_turns(value) for key, value in scoped.items()}

    for key, value in data.items():
        if key in ROTATION_SCOPES:
            continue
        if isinstance(value, dict):
            raise ValueError(f"Unsupported rotation settings section: {key}")
        settings["map"][str(key)] = normalize_quarter_turns(value)
    return settings


def read_rotation_settings() -> dict[str, dict[str, int]]:
    if not ROTATION_SETTINGS_FILE.exists():
        return {scope: {} for scope in ROTATION_SCOPES}
    data = read_json(ROTATION_SETTINGS_FILE)
    return normalize_rotation_settings(data)


def write_rotation_settings(settings: dict[str, dict[str, int]]) -> None:
    payload = json.dumps(settings, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")
    ROTATION_SETTINGS_FILE.write_bytes(payload + b"\n")


def rotation_setting_scope(payload: dict[str, Any]) -> str:
    scope = str(payload.get("scope", "map")).strip().lower()
    if scope not in ROTATION_SCOPES:
        raise ValueError(f"unsupported rotation scope: {scope}")
    return scope


def rotation_settings_payload() -> dict[str, Any]:
    with ROTATION_SETTINGS_LOCK:
        settings = read_rotation_settings()
    return {"ok": True, "settings": settings, "path": str(ROTATION_SETTINGS_FILE)}


def save_rotation_setting(payload: dict[str, Any]) -> dict[str, Any]:
    scope = rotation_setting_scope(payload)
    key = rotation_key(
        str(payload.get("farmId", "")),
        str(payload.get("cameraId", "")),
        str(payload.get("typeId", "")),
    )
    turns = normalize_quarter_turns(payload.get("quarterTurns", 0))
    with ROTATION_SETTINGS_LOCK:
        settings = read_rotation_settings()
        settings.setdefault(scope, {})[key] = turns
        write_rotation_settings(settings)
    if scope == "plan":
        DATA_CACHE.clear()
    return {"ok": True, "scope": scope, "key": key, "quarterTurns": turns, "settings": settings}


def get_saved_rotation(scope: str, farm_id: str, camera_id: str, type_id: str) -> int:
    key = rotation_key(farm_id, camera_id, type_id)
    with ROTATION_SETTINGS_LOCK:
        settings = read_rotation_settings()
    return normalize_quarter_turns(settings.get(scope, {}).get(key, 0))


def get_geometry(sample: dict[str, Any], plan_quarter_turns: int = 0) -> FloorGeometry:
    path = Path(sample["floorplanAnnotation"])
    turns = normalize_quarter_turns(plan_quarter_turns)
    cache_key = (path, turns)
    if cache_key not in GEOMETRY_CACHE:
        GEOMETRY_CACHE[cache_key] = FloorGeometry(read_json(path), turns)
    return GEOMETRY_CACHE[cache_key]


def get_points_for_frame(sample_id: str, frame: int) -> dict[int, list[float]]:
    sample = get_sample(sample_id)
    payload = get_payload(sample["id"])
    geometry = get_geometry(sample, int(payload["meta"].get("planRotationQuarterTurns", 0)))
    for item in payload["frames"]:
        if int(item["frame"]) == int(frame):
            return {
                int(point["trackId"]): round_point(realtime_adjusted_payload_point(point, geometry, "simple"))
                for point in item["points"]
            }
    return {}


STAGE2_PRECOMPUTED_CACHE: dict[str, dict[str, Any]] = {}


def load_stage2_precomputed(sample: dict[str, Any]) -> dict[str, Any]:
    sample_id = sample["id"]
    if sample_id in STAGE2_PRECOMPUTED_CACHE:
        return STAGE2_PRECOMPUTED_CACHE[sample_id]
    path = sample.get("stage2PrecomputedPath")
    if path is None:
        raise FileNotFoundError(f"Stage2 precomputed result is missing for sample {sample_id}")
    resolved = require_under_directory(Path(path), STAGE2_PRECOMPUTED_DIR)
    data = read_json(resolved)
    if int(data.get("schemaVersion", -1)) != STAGE2_PRECOMPUTED_SCHEMA_VERSION:
        raise ValueError(f"Unsupported Stage2 precomputed schema: {resolved}")
    if str(data.get("sampleId", "")) != sample_id:
        raise ValueError(f"Stage2 precomputed sample mismatch: {resolved}")
    if str(data.get("sourceDir", "")) != str(sample["sourceDir"]):
        raise ValueError(f"Stage2 precomputed source mismatch for sample {sample_id}: {resolved}")
    if not data.get("complete", False):
        raise ValueError(f"Stage2 precomputed result is incomplete: {resolved}")
    frames = data.get("frames", {})
    if not isinstance(frames, dict):
        raise ValueError(f"Stage2 precomputed frames must be an object: {resolved}")
    STAGE2_PRECOMPUTED_CACHE[sample_id] = data
    return data


def precomputed_interactions_for_frame(data: dict[str, Any], frame: int) -> list[dict[str, Any]]:
    frames = data.get("frames", {})
    items = frames.get(str(int(frame)), [])
    if not isinstance(items, list):
        raise ValueError(f"Stage2 precomputed frame entry must be a list: frame={frame}")
    return [dict(item) for item in items]


def display_point_field(display_mode: str | None) -> tuple[str, str]:
    mode = str(display_mode or "simple").strip().lower()
    if mode in {"semantic", "correspondence", "rectified"}:
        return "semantic", "semanticPoint"
    if mode in {"", "simple", "base"}:
        return "simple", "point"
    raise ValueError(f"Unsupported display mode: {display_mode}")


def payload_source_point(point: dict[str, Any], mode: str) -> Point:
    source = point.get("semanticPoint") if mode == "semantic" else point.get("point")
    if source is None:
        source = point.get("point") or point.get("raw")
    if not isinstance(source, (list, tuple)) or len(source) < 2:
        raise ValueError(f"Missing display point for track {point.get('trackId')}")
    return float(source[0]), float(source[1])


def realtime_adjusted_payload_point(point: dict[str, Any], geometry: FloorGeometry, mode: str) -> Point:
    source = payload_source_point(point, mode)
    adjusted, _reason = geometry.nearest_allowed_point(source)
    return adjusted


def get_stage2_payload(sample_id: str | None, frame: int, display_mode: str | None = None) -> dict[str, Any]:
    sample = get_sample(sample_id)
    mode, _point_field = display_point_field(display_mode)
    precomputed = load_stage2_precomputed(sample)
    raw_interactions = precomputed_interactions_for_frame(precomputed, frame)
    payload = get_payload(sample["id"])
    geometry = get_geometry(sample, int(payload["meta"].get("planRotationQuarterTurns", 0)))
    point_lookup: dict[int, list[float]] = {}
    identity_lookup: dict[int, dict[str, Any]] = {}
    for frame_item in payload["frames"]:
        if int(frame_item["frame"]) == int(frame):
            for point in frame_item["points"]:
                track_id = int(point["trackId"])
                if track_id in point_lookup:
                    raise RuntimeError(f"Duplicate local track point in display frame {frame}: {track_id}")
                point_lookup[track_id] = round_point(realtime_adjusted_payload_point(point, geometry, mode))
                identity_lookup[track_id] = point
            break

    interactions: list[dict[str, Any]] = []
    filtered_red = 0
    missing_endpoint = 0
    for item in raw_interactions:
        tid_a = int(item["tidA"])
        tid_b = int(item["tidB"])
        point_a = point_lookup.get(tid_a)
        point_b = point_lookup.get(tid_b)
        if point_a is None or point_b is None:
            missing_endpoint += 1
            continue
        a = (float(point_a[0]), float(point_a[1]))
        b = (float(point_b[0]), float(point_b[1]))
        if geometry.segment_crosses_red(a, b):
            filtered_red += 1
            continue
        copied = dict(item)
        copied["from"] = round_point(a)
        copied["to"] = round_point(b)
        identity_a = identity_lookup[tid_a]
        identity_b = identity_lookup[tid_b]
        copied["globalTrackIdA"] = int(identity_a["globalTrackId"])
        copied["globalTrackUuidA"] = str(identity_a["globalTrackUuid"])
        copied["displayGlobalIdA"] = str(identity_a["displayGlobalId"])
        copied["reidIdStatusA"] = str(identity_a["reidIdStatus"])
        copied["globalTrackIdB"] = int(identity_b["globalTrackId"])
        copied["globalTrackUuidB"] = str(identity_b["globalTrackUuid"])
        copied["displayGlobalIdB"] = str(identity_b["displayGlobalId"])
        copied["reidIdStatusB"] = str(identity_b["reidIdStatus"])
        interactions.append(copied)

    friendly_count = sum(1 for item in interactions if str(item.get("class")) == "friendly")
    unfriendly_count = sum(1 for item in interactions if str(item.get("class")) == "unfriendly")
    return {
        "ok": True,
        "sample": public_sample(sample),
        "frame": int(frame),
        "stage2CurrentFrame": int(frame),
        "displayMode": mode,
        "precomputed": True,
        "interactions": interactions,
        "stats": {
            "drawn": len(interactions),
            "friendly": friendly_count,
            "unfriendly": unfriendly_count,
            "filteredByRed": filtered_red,
            "missingEndpoint": missing_endpoint,
        },
        "meta": precomputed.get("meta", {}),
    }


def reset_stage2_runtime(sample_id: str | None) -> dict[str, Any]:
    sample = get_sample(sample_id)
    return {"ok": True, "sample": public_sample(sample), "stage2CurrentFrame": -1, "precomputed": True}


def set_load_job_progress(job_id: str, label: str, current: int | None = None, total: int | None = None) -> None:
    with LOAD_JOBS_LOCK:
        job = LOAD_JOBS.get(job_id)
        if job is None:
            return
        job["progress"] = {
            "label": label,
            "current": current,
            "total": total,
        }


def run_load_job(job_id: str, sample_id: str) -> None:
    try:
        set_load_job_progress(job_id, "Starting load", 0, 1)
        payload = get_payload(
            sample_id,
            progress=lambda label, current=None, total=None: set_load_job_progress(job_id, label, current, total),
        )
        with LOAD_JOBS_LOCK:
            job = LOAD_JOBS[job_id]
            job["done"] = True
            job["error"] = ""
            job["result"] = payload
            job["progress"] = {"label": "Load complete", "current": 1, "total": 1}
    except Exception as exc:
        with LOAD_JOBS_LOCK:
            job = LOAD_JOBS.get(job_id)
            if job is not None:
                job["done"] = True
                job["error"] = str(exc)
                job["result"] = None
                job["progress"] = {"label": "Load failed", "current": 1, "total": 1}


def load_job_public_payload(job: dict[str, Any], include_result: bool) -> dict[str, Any]:
    out = {
        "ok": not bool(job.get("error")),
        "jobId": job["id"],
        "sampleId": job["sampleId"],
        "done": bool(job.get("done")),
        "error": str(job.get("error") or ""),
        "progress": job.get("progress") or {"label": "Queued", "current": 0, "total": 1},
    }
    if include_result and job.get("done") and not job.get("error"):
        out["data"] = job.get("result")
    return out


def start_load_job(sample_id: str | None) -> dict[str, Any]:
    sample = get_sample(sample_id)
    job_id = uuid.uuid4().hex
    job = {
        "id": job_id,
        "sampleId": sample["id"],
        "done": False,
        "error": "",
        "result": None,
        "progress": {"label": "Queued", "current": 0, "total": 1},
    }
    with LOAD_JOBS_LOCK:
        LOAD_JOBS[job_id] = job
    thread = threading.Thread(target=run_load_job, args=(job_id, sample["id"]), daemon=True)
    thread.start()
    return load_job_public_payload(job, include_result=False)


def load_job_status(job_id: str | None) -> dict[str, Any]:
    wanted = str(job_id or "").strip()
    if not wanted:
        raise ValueError("missing required query parameter: job")
    with LOAD_JOBS_LOCK:
        job = LOAD_JOBS.get(wanted)
        if job is None:
            raise ValueError(f"Unknown load job: {wanted}")
        return load_job_public_payload(job, include_result=True)


def get_video_path(sample_id: str | None, kind: str | None) -> Path:
    sample = get_sample(sample_id)
    video = sample["video"]
    if not video["available"]:
        raise FileNotFoundError(f"No video is available for sample {sample['id']}")

    wanted = (kind or "selected").strip().lower()
    if wanted in {"selected", "trackid"}:
        path = video["selectedPath"]
    else:
        raise ValueError(f"Unsupported video kind: {kind}")

    if path is None or not path.is_file():
        raise FileNotFoundError(f"Video is missing for sample {sample['id']} kind {wanted}")
    return path


def parse_range_header(range_header: str, total_size: int) -> tuple[int, int]:
    if not range_header.startswith("bytes="):
        raise ValueError(f"Unsupported Range header: {range_header}")
    range_spec = range_header.removeprefix("bytes=").strip()
    if "," in range_spec:
        raise ValueError(f"Multiple ranges are not supported: {range_header}")
    start_raw, sep, end_raw = range_spec.partition("-")
    if not sep:
        raise ValueError(f"Invalid Range header: {range_header}")

    if start_raw == "":
        suffix_size = int(end_raw)
        if suffix_size <= 0:
            raise ValueError(f"Invalid suffix Range header: {range_header}")
        start = max(0, total_size - suffix_size)
        end = total_size - 1
    else:
        start = int(start_raw)
        end = int(end_raw) if end_raw else total_size - 1

    if start < 0 or end < start or start >= total_size:
        raise ValueError(f"Unsatisfiable Range header: {range_header}")
    return start, min(end, total_size - 1)


class WebUiHandler(BaseHTTPRequestHandler):
    server_version = "DairyFloorPlanWeb/1.0"

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/":
                self.send_file(STATIC_DIR / "index.html", "text/html; charset=utf-8")
            elif parsed.path == "/api/data":
                params = parse_qs(parsed.query)
                self.send_json(get_payload(first_query_value(params, "sample")))
            elif parsed.path == "/api/samples":
                self.send_json(samples_payload())
            elif parsed.path == "/api/deep-link/resolve":
                self.send_json(resolve_deep_link(parse_qs(parsed.query)))
            elif parsed.path == "/api/rotation":
                self.send_json(rotation_settings_payload())
            elif parsed.path == "/api/load/status":
                params = parse_qs(parsed.query)
                self.send_json(load_job_status(first_query_value(params, "job")))
            elif parsed.path == "/api/stage2":
                params = parse_qs(parsed.query)
                frame_values = params.get("frame", [])
                if not frame_values:
                    raise ValueError("missing required query parameter: frame")
                self.send_json(
                    get_stage2_payload(
                        first_query_value(params, "sample"),
                        int(float(frame_values[0])),
                        first_query_value(params, "displayMode"),
                    )
                )
            elif parsed.path == "/api/stage2/reset":
                params = parse_qs(parsed.query)
                self.send_json(reset_stage2_runtime(first_query_value(params, "sample")))
            elif parsed.path == "/api/video":
                params = parse_qs(parsed.query)
                self.send_video(get_video_path(first_query_value(params, "sample"), first_query_value(params, "kind")))
            elif parsed.path == "/api/health":
                self.send_json({"ok": True})
            elif parsed.path.startswith("/static/"):
                static_path = (STATIC_DIR / parsed.path.removeprefix("/static/")).resolve()
                if STATIC_DIR.resolve() not in static_path.parents and static_path != STATIC_DIR.resolve():
                    self.send_error(HTTPStatus.FORBIDDEN)
                    return
                self.send_file(static_path)
            else:
                self.send_error(HTTPStatus.NOT_FOUND)
        except Exception as exc:
            self.send_json({"ok": False, "error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/api/rotation":
                self.send_json(save_rotation_setting(self.read_json_body()))
            elif parsed.path == "/api/load/start":
                body = self.read_json_body()
                self.send_json(start_load_job(str(body.get("sample", ""))))
            else:
                self.send_error(HTTPStatus.NOT_FOUND)
        except Exception as exc:
            self.send_json({"ok": False, "error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def read_json_body(self) -> dict[str, Any]:
        length_raw = self.headers.get("Content-Length")
        if not length_raw:
            return {}
        length = int(length_raw)
        if length > 64 * 1024:
            raise ValueError("request body is too large")
        raw = self.rfile.read(length)
        data = json.loads(raw.decode("utf-8"))
        if not isinstance(data, dict):
            raise ValueError("request body must be a JSON object")
        return data

    def send_file(self, path: Path, content_type: str | None = None) -> None:
        if not path.exists() or not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        if content_type is None:
            content_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        content = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(content)

    def send_video(self, path: Path) -> None:
        total_size = path.stat().st_size
        content_type = mimetypes.guess_type(str(path))[0] or "video/mp4"
        range_header = self.headers.get("Range")
        start = 0
        end = total_size - 1
        partial = False

        if range_header:
            try:
                start, end = parse_range_header(range_header, total_size)
                partial = True
            except ValueError:
                self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                self.send_header("Content-Range", f"bytes */{total_size}")
                self.send_header("Accept-Ranges", "bytes")
                self.end_headers()
                return

        content_length = end - start + 1
        self.send_response(HTTPStatus.PARTIAL_CONTENT if partial else HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(content_length))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", "no-store")
        if partial:
            self.send_header("Content-Range", f"bytes {start}-{end}/{total_size}")
        self.end_headers()

        with path.open("rb") as handle:
            handle.seek(start)
            remaining = content_length
            while remaining > 0:
                chunk = handle.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)

    def send_json(self, data: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        content = json.dumps(data, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(content)

    def log_message(self, format: str, *args: Any) -> None:
        print(f"{self.log_date_time_string()} {self.address_string()} {format % args}", flush=True)


def first_query_value(params: dict[str, list[str]], key: str) -> str | None:
    values = params.get(key, [])
    return values[0] if values else None


def get_payload(sample_id: str | None = None, progress: ProgressCallback | None = None) -> dict[str, Any]:
    sample = get_sample(sample_id)
    mapping_profile = mapping_profile_for_sample(sample)
    plan_turns = plan_rotation_for_sample(sample, mapping_profile)
    cache_key = f"{sample['id']}|plan:{plan_turns}"
    if cache_key not in DATA_CACHE:
        DATA_CACHE[cache_key] = build_payload(sample, progress)
    else:
        report_progress(progress, "Using cached payload", 1, 1)
    return DATA_CACHE[cache_key]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the dairy cattle floor-plan web UI.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9922)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_reid_index()
    samples = load_samples()
    address = (args.host, args.port)
    httpd = ThreadingHTTPServer(address, WebUiHandler)
    print(f"Web UI: http://{args.host}:{args.port}/", flush=True)
    print(f"Ready with {len(samples)} segment samples. Select a sample and press Load.", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping web UI.", flush=True)


if __name__ == "__main__":
    main()
