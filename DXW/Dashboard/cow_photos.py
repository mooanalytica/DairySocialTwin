from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import sqlite3
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from decimal import Decimal, localcontext
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from PIL import Image

from appearance_data import DataContractError


PHOTO_SCHEMA_VERSION = 3
PHOTO_CONTRACT = (
    "first-ranked-appearance-start-bbox-pad5-landscape-raworientation-jpeg640x360-v3"
)
PHOTO_WIDTH = 640
PHOTO_HEIGHT = 360
PHOTO_MIME_TYPE = "image/jpeg"
PHOTO_PADDING_FRACTION = 0.05
EXPECTED_COWS = tuple(f"G{index:04d}" for index in range(1, 63))
EXPECTED_SEQUENCE_ID = "dairy_farm_1_gopro1_20250505"
EXPECTED_CLIP_FRAME_COUNTS = {
    "GX010006": 88_320,
    "GX020006": 84_480,
    "GX030006": 78_720,
    "GX040006": 84_480,
    "GX050006": 78_720,
    "GX060006": 78_720,
    "GX070006": 76_800,
    "GX080006": 71_040,
    "GX090006": 72_960,
    "GX100006": 65_280,
    "GX110006": 46_972,
}
EXPECTED_CLIP_OFFSETS: dict[str, int] = {}
_offset = 0
for _clip, _frame_count in EXPECTED_CLIP_FRAME_COUNTS.items():
    EXPECTED_CLIP_OFFSETS[_clip] = _offset
    _offset += _frame_count
EXPECTED_SEQUENCE_FRAME_COUNT = _offset
EXPECTED_REID_METADATA = {
    "schema_version": "3",
    "sequence_id": EXPECTED_SEQUENCE_ID,
    "source_csv": "/mnt/data4t/hyw/re-identification-results/1-1.csv",
    "stage1_root": "/mnt/data4t/hyw/Stage1_segmented/1/Gopro1",
    "row_count": "3322909",
    "valid_row_count": "3318113",
    "invalid_row_count": "4796",
    "physical_tracking_row_count": "3329043",
    "canonical_tracking_row_count": "3322909",
    "overlap_tracking_row_count": "6134",
    "global_identity_count": "62",
    "bbox_rule": "clamp_stage1_xyxy_to_manifest_bounds_then_compare",
    "bbox_tolerance_px": "0.001",
    "confidence_tolerance": "1e-06",
    "validity_policy": "valid_false_as_missed_detection_after_local_id_computation",
    "timeline_fps": str(30_000.0 / 1_001.0),
    "timeline_frame_count": str(EXPECTED_SEQUENCE_FRAME_COUNT),
    "timeline_policy": "reid_global_frame_and_global_time_sec_strict_contiguous_sequence",
    "validation_complete": "1",
}
EXPECTED_REID_SOURCE_FILE_COUNT = 37


@dataclass(frozen=True)
class CowPhoto:
    cow_id: str
    path: Path
    clip: str
    local_frame: int
    local_track_id: int
    width: int
    height: int
    mime_type: str
    sha256: str


@dataclass(frozen=True)
class _TrajectoryStart:
    cow_id: str
    cow_uuid: str
    clip: str
    local_frame: int
    local_track_id: int
    track_confidence: float


@dataclass(frozen=True)
class _PhotoSource:
    start: _TrajectoryStart
    bbox: tuple[float, float, float, float]
    bbox_confidence: float
    source_width: int
    source_height: int
    video_path: Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json_object(path: Path, context: str) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise DataContractError(f"Could not read {context}: {path}") from exc
    if not isinstance(value, dict):
        raise DataContractError(f"{context} must be a JSON object: {path}")
    return value


def _selection_document(
    appearances: Mapping[str, list[dict[str, Any]]],
) -> dict[str, dict[str, Any]]:
    if set(appearances) != set(EXPECTED_COWS):
        raise DataContractError("Cow photo source must contain exactly G0001..G0062")
    selected: dict[str, dict[str, Any]] = {}
    for cow in EXPECTED_COWS:
        rows = appearances[cow]
        if not rows or not isinstance(rows[0], dict):
            raise DataContractError(f"Cow photo source has no first appearance for {cow}")
        first = rows[0]
        clip = first.get("clip")
        start = first.get("startFrame")
        end = first.get("endFrameExclusive")
        count = first.get("frameCount")
        if (
            not isinstance(clip, str)
            or clip not in EXPECTED_CLIP_FRAME_COUNTS
            or isinstance(start, bool)
            or not isinstance(start, int)
            or isinstance(end, bool)
            or not isinstance(end, int)
            or isinstance(count, bool)
            or not isinstance(count, int)
            or start < 0
            or end <= start
            or count != end - start
            or end > EXPECTED_CLIP_FRAME_COUNTS[clip]
        ):
            raise DataContractError(f"Cow photo source appearance is invalid for {cow}")
        selected[cow] = {
            "clip": clip,
            "startFrame": start,
            "endFrameExclusive": end,
            "frameCount": count,
        }
    return selected


def _selection_sha256(selection: Mapping[str, Mapping[str, Any]]) -> str:
    encoded = json.dumps(
        selection,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _integer_array(series: pd.Series, context: str) -> np.ndarray:
    numeric = pd.to_numeric(series, errors="coerce")
    values = numeric.to_numpy(dtype=float)
    if not np.isfinite(values).all() or not np.array_equal(values, np.rint(values)):
        raise DataContractError(f"Cow photo trajectory {context} is not integral")
    return np.rint(values).astype(np.int64)


def _load_trajectory_starts(
    trajectories_path: Path,
    selection: Mapping[str, Mapping[str, Any]],
    identity_map: Mapping[str, str],
) -> dict[str, _TrajectoryStart]:
    if not trajectories_path.is_file():
        raise FileNotFoundError(f"WebUIL trajectories are missing: {trajectories_path}")
    display_to_uuid = {display: raw for raw, display in identity_map.items()}
    if set(display_to_uuid) != set(EXPECTED_COWS) or len(display_to_uuid) != 62:
        raise DataContractError("Cow photo identity map is not a 62-cow bijection")
    target_clips = {cow: str(value["clip"]) for cow, value in selection.items()}
    target_frames = {cow: int(value["startFrame"]) for cow, value in selection.items()}
    found: dict[str, _TrajectoryStart] = {}
    try:
        chunks = pd.read_csv(
            trajectories_path,
            usecols=[
                "cow_id",
                "track_conf",
                "frozen",
                "display_global_id",
                "source_clip_id",
                "local_frame",
                "local_track_id",
            ],
            dtype={
                "cow_id": "string",
                "frozen": "string",
                "display_global_id": "string",
                "source_clip_id": "string",
            },
            keep_default_na=False,
            chunksize=250_000,
        )
        for chunk in chunks:
            displays = chunk["display_global_id"].astype(str)
            source_clips = chunk["source_clip_id"].astype(str)
            local_frames = _integer_array(chunk["local_frame"], "local_frame")
            expected_clips = displays.map(target_clips)
            expected_frames = displays.map(target_frames)
            mask = (
                expected_clips.notna()
                & expected_frames.notna()
                & source_clips.eq(expected_clips)
                & (local_frames == expected_frames.fillna(-1).to_numpy(dtype=np.int64))
            )
            if not mask.any():
                continue
            selected = chunk.loc[mask].copy()
            selected["_local_frame"] = local_frames[mask.to_numpy()]
            selected["_local_track_id"] = _integer_array(
                selected["local_track_id"], "local_track_id"
            )
            confidences = pd.to_numeric(selected["track_conf"], errors="coerce").to_numpy(
                dtype=float
            )
            if not np.isfinite(confidences).all() or (
                (confidences < 0.0) | (confidences > 1.0)
            ).any():
                raise DataContractError("Cow photo trajectory has an invalid track_conf")
            for row, confidence in zip(selected.to_dict("records"), confidences):
                cow = str(row["display_global_id"])
                if cow in found:
                    raise DataContractError(f"Cow photo start trajectory is duplicated for {cow}")
                raw_uuid = str(row["cow_id"])
                if display_to_uuid.get(cow) != raw_uuid:
                    raise DataContractError(f"Cow photo UUID/display mapping disagrees for {cow}")
                frozen = str(row["frozen"]).strip().lower()
                if frozen != "false":
                    raise DataContractError(
                        f"Cow photo first-appearance start must be a live point, found frozen={frozen!r} for {cow}"
                    )
                found[cow] = _TrajectoryStart(
                    cow_id=cow,
                    cow_uuid=raw_uuid,
                    clip=str(row["source_clip_id"]),
                    local_frame=int(row["_local_frame"]),
                    local_track_id=int(row["_local_track_id"]),
                    track_confidence=float(confidence),
                )
    except (TypeError, ValueError) as exc:
        raise DataContractError(
            f"Could not parse cow photo start rows from {trajectories_path}"
        ) from exc
    if set(found) != set(EXPECTED_COWS):
        raise DataContractError(
            "Cow photo start trajectories are incomplete: "
            f"missing={sorted(set(EXPECTED_COWS) - set(found))}"
        )
    return found


def _stat_tuple(path: Path) -> tuple[int, int, int, int, int]:
    stat = path.stat()
    return (
        int(stat.st_dev),
        int(stat.st_ino),
        int(stat.st_size),
        int(stat.st_mtime_ns),
        int(stat.st_ctime_ns),
    )


def _open_validated_reid_index(path: Path) -> sqlite3.Connection:
    if not path.is_file():
        raise FileNotFoundError(f"Strict WebUIL re-ID index is missing: {path}")
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    connection.execute("PRAGMA query_only=ON")
    try:
        metadata = {
            str(row[0]): str(row[1])
            for row in connection.execute("SELECT key, value FROM metadata")
        }
        if metadata != EXPECTED_REID_METADATA:
            raise DataContractError("Strict WebUIL re-ID index metadata is stale")
        source_count = 0
        for row in connection.execute(
            "SELECT role, path, device, inode, size_bytes, mtime_ns, ctime_ns FROM source_files"
        ):
            source_count += 1
            source_path = Path(str(row[1]))
            if not source_path.is_file() or _stat_tuple(source_path) != tuple(
                int(value) for value in row[2:]
            ):
                raise DataContractError(
                    f"Strict WebUIL re-ID index source changed: role={row[0]}, path={source_path}"
                )
        if source_count != EXPECTED_REID_SOURCE_FILE_COUNT:
            raise DataContractError(
                f"Strict WebUIL re-ID index source inventory is incomplete: {source_count}"
            )
        clips = connection.execute(
            "SELECT clip_id, clip_order, width, height, fps FROM clips ORDER BY clip_order"
        ).fetchall()
        if [str(row[0]) for row in clips] != list(EXPECTED_CLIP_FRAME_COUNTS):
            raise DataContractError("Strict WebUIL re-ID clip order is unexpected")
        if any(
            int(row[1]) != index
            or int(row[2]) != 3840
            or int(row[3]) != 2160
            or not math.isclose(float(row[4]), 30_000 / 1_001, rel_tol=0.0, abs_tol=1e-12)
            for index, row in enumerate(clips)
        ):
            raise DataContractError("Strict WebUIL re-ID clip dimensions/fps are unexpected")
        identity_count = int(
            connection.execute("SELECT COUNT(*) FROM global_identities").fetchone()[0]
        )
        if identity_count != 62:
            raise DataContractError("Strict WebUIL re-ID index does not contain 62 identities")
        return connection
    except Exception:
        connection.close()
        raise


def _load_photo_sources(
    starts: Mapping[str, _TrajectoryStart],
    reid_index_path: Path,
    video_root: Path,
) -> dict[str, _PhotoSource]:
    connection = _open_validated_reid_index(reid_index_path)
    sources: dict[str, _PhotoSource] = {}
    query = """
        SELECT c.sequence_id, c.clip_id, c.width, c.height,
               d.local_frame, d.legacy_track_id, d.global_frame,
               d.global_track_uuid, g.display_global_id, d.id_status,
               d.x1, d.y1, d.x2, d.y2, d.bbox_confidence, d.valid
        FROM detections AS d
        JOIN clips AS c ON c.clip_key = d.clip_key
        LEFT JOIN global_identities AS g ON g.global_track_uuid = d.global_track_uuid
        WHERE c.clip_id = ? AND d.local_frame = ? AND d.legacy_track_id = ?
    """
    try:
        for cow in EXPECTED_COWS:
            start = starts[cow]
            rows = connection.execute(
                query,
                (start.clip, start.local_frame, start.local_track_id),
            ).fetchall()
            if len(rows) != 1:
                raise DataContractError(
                    f"Cow photo re-ID key must resolve once for {cow}, found {len(rows)}"
                )
            row = rows[0]
            if (
                str(row[0]) != EXPECTED_SEQUENCE_ID
                or str(row[1]) != start.clip
                or int(row[4]) != start.local_frame
                or int(row[5]) != start.local_track_id
                or int(row[6]) != EXPECTED_CLIP_OFFSETS[start.clip] + start.local_frame
                or str(row[7]) != start.cow_uuid
                or str(row[8]) != cow
                or str(row[9]) != "forced_provisional"
                or int(row[15]) != 1
            ):
                raise DataContractError(f"Cow photo re-ID identity/timeline mapping disagrees for {cow}")
            width = int(row[2])
            height = int(row[3])
            bbox = tuple(float(value) for value in row[10:14])
            bbox_confidence = float(row[14])
            x1, y1, x2, y2 = bbox
            if (
                not all(math.isfinite(value) for value in (*bbox, bbox_confidence))
                or not (0.0 <= x1 < x2 <= width and 0.0 <= y1 < y2 <= height)
                or not 0.0 <= bbox_confidence <= 1.0
                or not math.isclose(
                    start.track_confidence,
                    round(bbox_confidence, 4),
                    rel_tol=0.0,
                    abs_tol=1e-9,
                )
            ):
                raise DataContractError(f"Cow photo re-ID bbox/confidence is invalid for {cow}")
            video_path = video_root / f"{start.clip}.MP4"
            if not video_path.is_file():
                raise FileNotFoundError(f"Cow photo source video is missing: {video_path}")
            sources[cow] = _PhotoSource(
                start=start,
                bbox=bbox,
                bbox_confidence=bbox_confidence,
                source_width=width,
                source_height=height,
                video_path=video_path,
            )
    finally:
        connection.close()
    return sources


def _probe_video(path: Path, expected_frame_count: int, ffprobe_path: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [
            str(ffprobe_path),
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,width,height,pix_fmt,avg_frame_rate,nb_frames,start_time,time_base:"
            "stream_side_data=rotation",
            "-of",
            "json",
            "--",
            str(path),
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
        check=False,
    )
    if completed.returncode != 0:
        raise DataContractError(f"ffprobe failed for {path}: {completed.stderr.strip()}")
    try:
        document = json.loads(completed.stdout)
        streams = document["streams"]
        stream = streams[0]
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        raise DataContractError(f"ffprobe returned malformed data for {path}") from exc
    rotations = {
        int(item["rotation"])
        for item in stream.get("side_data_list", [])
        if isinstance(item, dict) and "rotation" in item
    }
    if (
        not isinstance(streams, list)
        or len(streams) != 1
        or stream.get("codec_name") != "hevc"
        or int(stream.get("width", -1)) != 3840
        or int(stream.get("height", -1)) != 2160
        or stream.get("avg_frame_rate") != "30000/1001"
        or stream.get("time_base") != "1/30000"
        or int(stream.get("nb_frames", -1)) != expected_frame_count
        or not math.isclose(float(stream.get("start_time", "nan")), 0.0, rel_tol=0.0, abs_tol=1e-12)
        or rotations != {-90}
    ):
        raise DataContractError(f"Cow photo video contract is unexpected: {path}")
    stat = path.stat()
    return {
        "path": str(path),
        "size": int(stat.st_size),
        "mtimeNs": int(stat.st_mtime_ns),
        "frameCount": expected_frame_count,
        "displayRotation": -90,
    }


def _padded_crop(source: _PhotoSource) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = source.bbox
    padding_x = (x2 - x1) * PHOTO_PADDING_FRACTION
    padding_y = (y2 - y1) * PHOTO_PADDING_FRACTION
    left = max(0, math.floor(x1 - padding_x))
    top = max(0, math.floor(y1 - padding_y))
    right = min(source.source_width, math.ceil(x2 + padding_x))
    bottom = min(source.source_height, math.ceil(y2 + padding_y))
    if right - left < 2 or bottom - top < 2:
        raise DataContractError(f"Cow photo crop is empty for {source.start.cow_id}")
    return left, top, right - left, bottom - top


def _frame_timestamp(local_frame: int) -> str:
    with localcontext() as context:
        context.prec = 30
        value = Decimal(local_frame * 1_001) / Decimal(30_000)
    return f"{value:.12f}"


def _extract_photo(
    source: _PhotoSource,
    output_path: Path,
    ffmpeg_path: Path,
) -> None:
    left, top, crop_width, crop_height = _padded_crop(source)
    video_filter = (
        f"crop={crop_width}:{crop_height}:{left}:{top},"
        f"scale={PHOTO_WIDTH}:{PHOTO_HEIGHT}:force_original_aspect_ratio=decrease,"
        f"pad={PHOTO_WIDTH}:{PHOTO_HEIGHT}:(ow-iw)/2:(oh-ih)/2:color=0x182426"
    )
    completed = subprocess.run(
        [
            str(ffmpeg_path),
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-noautorotate",
            "-ss",
            _frame_timestamp(source.start.local_frame),
            "-i",
            str(source.video_path),
            "-map",
            "0:v:0",
            "-frames:v",
            "1",
            "-vf",
            video_filter,
            "-q:v",
            "2",
            "-map_metadata",
            "-1",
            "-y",
            str(output_path),
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
        check=False,
    )
    if completed.returncode != 0 or not output_path.is_file():
        raise DataContractError(
            f"ffmpeg cow photo extraction failed for {source.start.cow_id}: "
            f"{completed.stderr.strip()}"
        )


def _validate_jpeg(path: Path, cow: str) -> None:
    try:
        with Image.open(path) as image:
            image.load()
            if image.format != "JPEG" or image.size != (PHOTO_WIDTH, PHOTO_HEIGHT):
                raise DataContractError(
                    f"Cow photo must be a {PHOTO_WIDTH}x{PHOTO_HEIGHT} JPEG for {cow}"
                )
    except (OSError, ValueError) as exc:
        raise DataContractError(f"Cow photo is not a valid JPEG for {cow}: {path}") from exc


def _record_for_source(source: _PhotoSource, path: Path) -> dict[str, Any]:
    _validate_jpeg(path, source.start.cow_id)
    left, top, crop_width, crop_height = _padded_crop(source)
    stat = source.video_path.stat()
    return {
        "fileName": f"{source.start.cow_id}.jpg",
        "clip": source.start.clip,
        "localFrame": source.start.local_frame,
        "sourcePts": source.start.local_frame * 1_001,
        "localTrackId": source.start.local_track_id,
        "globalTrackUuid": source.start.cow_uuid,
        "bbox": list(source.bbox),
        "bboxConfidence": source.bbox_confidence,
        "crop": [left, top, crop_width, crop_height],
        "sourceWidth": source.source_width,
        "sourceHeight": source.source_height,
        "sourceVideoPath": str(source.video_path),
        "sourceVideoSize": int(stat.st_size),
        "sourceVideoMtimeNs": int(stat.st_mtime_ns),
        "width": PHOTO_WIDTH,
        "height": PHOTO_HEIGHT,
        "mimeType": PHOTO_MIME_TYPE,
        "jpegBytes": int(path.stat().st_size),
        "sha256": _sha256(path),
    }


def _validate_manifest_photos(
    document: Mapping[str, Any],
    photo_dir: Path,
    selection: Mapping[str, Mapping[str, Any]],
    identity_map: Mapping[str, str],
) -> dict[str, CowPhoto]:
    photos = document.get("photos")
    if not isinstance(photos, dict) or set(photos) != set(EXPECTED_COWS):
        raise DataContractError("Cow photo manifest must contain exactly G0001..G0062")
    display_to_uuid = {display: raw for raw, display in identity_map.items()}
    expected_keys = {
        "fileName",
        "clip",
        "localFrame",
        "sourcePts",
        "localTrackId",
        "globalTrackUuid",
        "bbox",
        "bboxConfidence",
        "crop",
        "sourceWidth",
        "sourceHeight",
        "sourceVideoPath",
        "sourceVideoSize",
        "sourceVideoMtimeNs",
        "width",
        "height",
        "mimeType",
        "jpegBytes",
        "sha256",
    }
    actual_jpegs = {path.name for path in photo_dir.glob("*.jpg")} if photo_dir.is_dir() else set()
    expected_jpegs = {f"{cow}.jpg" for cow in EXPECTED_COWS}
    if actual_jpegs != expected_jpegs:
        raise DataContractError(
            "Cow photo cache files are incomplete or contain extras: "
            f"missing={sorted(expected_jpegs - actual_jpegs)}, extra={sorted(actual_jpegs - expected_jpegs)}"
        )
    validated: dict[str, CowPhoto] = {}
    for cow in EXPECTED_COWS:
        record = photos[cow]
        if not isinstance(record, dict) or set(record) != expected_keys:
            raise DataContractError(f"Cow photo manifest record is malformed for {cow}")
        expected_file = f"{cow}.jpg"
        selected = selection[cow]
        if (
            record["fileName"] != expected_file
            or record["clip"] != selected["clip"]
            or record["localFrame"] != selected["startFrame"]
            or record["sourcePts"] != int(selected["startFrame"]) * 1_001
            or record["globalTrackUuid"] != display_to_uuid.get(cow)
            or record["width"] != PHOTO_WIDTH
            or record["height"] != PHOTO_HEIGHT
            or record["mimeType"] != PHOTO_MIME_TYPE
            or isinstance(record["localTrackId"], bool)
            or not isinstance(record["localTrackId"], int)
            or record["localTrackId"] < 0
        ):
            raise DataContractError(f"Cow photo manifest source mapping disagrees for {cow}")
        checksum = record["sha256"]
        if not isinstance(checksum, str) or re.fullmatch(r"[0-9a-f]{64}", checksum) is None:
            raise DataContractError(f"Cow photo SHA-256 is malformed for {cow}")
        path = photo_dir / expected_file
        if path.stat().st_size != record["jpegBytes"] or _sha256(path) != checksum:
            raise DataContractError(f"Cow photo checksum/size disagrees for {cow}")
        _validate_jpeg(path, cow)
        validated[cow] = CowPhoto(
            cow_id=cow,
            path=path,
            clip=str(record["clip"]),
            local_frame=int(record["localFrame"]),
            local_track_id=int(record["localTrackId"]),
            width=PHOTO_WIDTH,
            height=PHOTO_HEIGHT,
            mime_type=PHOTO_MIME_TYPE,
            sha256=checksum,
        )
    return validated


class CowPhotoIndex:
    def __init__(self, photos: Mapping[str, CowPhoto]) -> None:
        self._photos = dict(photos)

    @classmethod
    def load_or_build(
        cls,
        *,
        appearances: Mapping[str, list[dict[str, Any]]],
        generation_id: str,
        identity_map: Mapping[str, str],
        trajectories_path: Path,
        reid_index_path: Path,
        video_root: Path,
        photo_dir: Path,
        manifest_path: Path,
        ffmpeg_path: Path = Path("/usr/bin/ffmpeg"),
        ffprobe_path: Path = Path("/usr/bin/ffprobe"),
    ) -> "CowPhotoIndex":
        selection = _selection_document(appearances)
        selection_digest = _selection_sha256(selection)
        manifest: dict[str, Any] | None = None
        if manifest_path.is_file():
            manifest = _read_json_object(manifest_path, "cow photo manifest")
        # A published photo is an immutable snapshot. On a cache hit, verify
        # the JPEG and its generation/first-ranked-appearance binding without
        # rebinding it to later re-ID or video-file changes. The source data is
        # consulted only when the snapshot must be rebuilt.
        if (
            manifest is not None
            and manifest.get("schemaVersion") == PHOTO_SCHEMA_VERSION
            and manifest.get("contract") == PHOTO_CONTRACT
            and manifest.get("generationId") == generation_id
            and manifest.get("selectionSha256") == selection_digest
        ):
            return cls(
                _validate_manifest_photos(
                    manifest,
                    photo_dir,
                    selection,
                    identity_map,
                )
            )

        if not ffmpeg_path.is_file() or not ffprobe_path.is_file():
            raise FileNotFoundError(
                f"Cow photo extraction requires ffmpeg={ffmpeg_path} and ffprobe={ffprobe_path}"
            )
        print("Building 62 fixed cow photos from first-ranked appearances ...", flush=True)
        starts = _load_trajectory_starts(trajectories_path, selection, identity_map)
        sources = _load_photo_sources(starts, reid_index_path, video_root)
        video_probes: dict[str, dict[str, Any]] = {}
        for clip in sorted({source.start.clip for source in sources.values()}):
            video_probes[clip] = _probe_video(
                video_root / f"{clip}.MP4",
                EXPECTED_CLIP_FRAME_COUNTS[clip],
                ffprobe_path,
            )

        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        photo_dir.mkdir(parents=True, exist_ok=True)
        temporary_root = Path(
            tempfile.mkdtemp(prefix=".cow_photos_build_", dir=manifest_path.parent)
        )
        try:
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = {
                    executor.submit(
                        _extract_photo,
                        sources[cow],
                        temporary_root / f"{cow}.jpg",
                        ffmpeg_path,
                    ): cow
                    for cow in EXPECTED_COWS
                }
                for future in as_completed(futures):
                    future.result()
            records = {
                cow: _record_for_source(
                    sources[cow],
                    temporary_root / f"{cow}.jpg",
                )
                for cow in EXPECTED_COWS
            }
            for cow in EXPECTED_COWS:
                os.replace(temporary_root / f"{cow}.jpg", photo_dir / f"{cow}.jpg")
            document = {
                "schemaVersion": PHOTO_SCHEMA_VERSION,
                "contract": PHOTO_CONTRACT,
                "generationId": generation_id,
                "selectionSha256": selection_digest,
                "videoProbes": video_probes,
                "photos": records,
            }
            manifest_tmp = temporary_root / "cow_photos.meta.json"
            with manifest_tmp.open("w", encoding="utf-8") as handle:
                json.dump(document, handle, ensure_ascii=False, indent=2, sort_keys=True)
                handle.write("\n")
            os.replace(manifest_tmp, manifest_path)
        finally:
            shutil.rmtree(temporary_root, ignore_errors=True)

        validated = _validate_manifest_photos(
            document,
            photo_dir,
            selection,
            identity_map,
        )
        print(f"Cow photo cache ready: count={len(validated)}, path={photo_dir}", flush=True)
        return cls(validated)

    def for_cow(self, cow_id: str) -> CowPhoto:
        try:
            return self._photos[cow_id]
        except KeyError as exc:
            raise ValueError(f"Unknown cattle identity: {cow_id}") from exc

    def all_photos(self) -> list[CowPhoto]:
        return [self._photos[cow] for cow in EXPECTED_COWS]
