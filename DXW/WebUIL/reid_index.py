from __future__ import annotations

import csv
import json
import math
import os
import sqlite3
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
REID_SOURCE_CSV = Path("/mnt/data4t/hyw/re-identification-results/1-1.csv")
REID_INDEX_PATH = ROOT / ".cache" / "reidentification_1-1.sqlite3"
STAGE1_CURRENT_ROOT = Path("/mnt/data4t/hyw/Stage1_segmented/1/Gopro1")
REID_SEQUENCE_ID = "dairy_farm_1_gopro1_20250505"
REID_SCHEMA_VERSION = 3
REID_BBOX_TOLERANCE_PX = 1e-3
REID_CONFIDENCE_TOLERANCE = 1e-6
REID_GLOBAL_TIME_TOLERANCE_SEC = 1e-9
REID_VALID_STATUS = "forced_provisional"
REID_VALID_IDENTITY_BASIS = "operator_forced_appearance_exact_62"
REID_VALIDITY_POLICY = "valid_false_as_missed_detection_after_local_id_computation"

EXPECTED_CLIP_IDS = tuple(f"GX{index:02d}0006" for index in range(1, 12))
EXPECTED_SEQUENCE_FPS = 30_000.0 / 1_001.0
EXPECTED_CLIP_TIMELINE = (
    ("GX010006", 88_320, 0),
    ("GX020006", 84_480, 88_320),
    ("GX030006", 78_720, 172_800),
    ("GX040006", 84_480, 251_520),
    ("GX050006", 78_720, 336_000),
    ("GX060006", 78_720, 414_720),
    ("GX070006", 76_800, 493_440),
    ("GX080006", 71_040, 570_240),
    ("GX090006", 72_960, 641_280),
    ("GX100006", 65_280, 714_240),
    ("GX110006", 46_972, 779_520),
)
EXPECTED_SEQUENCE_FRAME_COUNT = 826_492
EXPECTED_REID_ROW_COUNT = 3_322_909
EXPECTED_REID_VALID_ROW_COUNT = 3_318_113
EXPECTED_REID_INVALID_ROW_COUNT = 4_796
EXPECTED_PHYSICAL_TRACKING_ROW_COUNT = 3_329_043
EXPECTED_OVERLAP_TRACKING_ROW_COUNT = 6_134
EXPECTED_GLOBAL_IDENTITY_COUNT = 62
EXPECTED_SOURCE_FILE_COUNT = 1 + (12 * 3)
EXPECTED_INVALID_REASON_COUNTS = {
    "HIGH_IOU_DUPLICATE": 4_710,
    "CLAMPED_TO_IMAGE|HIGH_IOU_DUPLICATE": 86,
}


def _validated_timeline_maps() -> tuple[dict[str, int], dict[str, int]]:
    clip_ids = tuple(item[0] for item in EXPECTED_CLIP_TIMELINE)
    if clip_ids != EXPECTED_CLIP_IDS:
        raise RuntimeError(f"Expected re-ID timeline clip order is invalid: {clip_ids}")
    frame_counts: dict[str, int] = {}
    global_starts: dict[str, int] = {}
    expected_start = 0
    for clip_id, frame_count, global_start in EXPECTED_CLIP_TIMELINE:
        if frame_count <= 0 or global_start != expected_start:
            raise RuntimeError(
                "Expected re-ID timeline is not positive and contiguous: "
                f"clip={clip_id}, frames={frame_count}, start={global_start}, expected_start={expected_start}"
            )
        frame_counts[clip_id] = frame_count
        global_starts[clip_id] = global_start
        expected_start += frame_count
    if expected_start != EXPECTED_SEQUENCE_FRAME_COUNT:
        raise RuntimeError(
            "Expected re-ID timeline total is invalid: "
            f"expected={EXPECTED_SEQUENCE_FRAME_COUNT}, actual={expected_start}"
        )
    return frame_counts, global_starts


EXPECTED_CLIP_FRAME_COUNTS, EXPECTED_CLIP_GLOBAL_START_FRAMES = _validated_timeline_maps()


def reid_global_timeline(clip_id: str, local_frame: int) -> tuple[int, float]:
    """Return the strict re-ID global frame/time for one source-video frame."""
    wanted_clip = str(clip_id).strip()
    if wanted_clip not in EXPECTED_CLIP_FRAME_COUNTS:
        raise ValueError(f"Unexpected re-ID timeline clip: {clip_id!r}")
    frame = int(local_frame)
    frame_count = EXPECTED_CLIP_FRAME_COUNTS[wanted_clip]
    if frame < 0 or frame >= frame_count:
        raise ValueError(
            f"Local frame is outside the strict re-ID timeline: clip={wanted_clip}, "
            f"frame={frame}, valid_range=[0,{frame_count - 1}]"
        )
    global_frame = EXPECTED_CLIP_GLOBAL_START_FRAMES[wanted_clip] + frame
    return global_frame, global_frame / EXPECTED_SEQUENCE_FPS

REID_COLUMNS = (
    "sequence_id",
    "clip_id",
    "clip_order",
    "det_id",
    "csv_row_index",
    "legacy_track_id",
    "global_track_id",
    "global_track_uuid",
    "display_global_id",
    "id_status",
    "identity_basis",
    "local_frame",
    "global_frame",
    "global_time_sec",
    "x1",
    "y1",
    "x2",
    "y2",
    "bbox_confidence",
    "valid",
    "qa_flags",
    "invalid_reason",
    "micro_id",
    "stable_id",
    "order_in_micro",
    "order_in_stable",
    "order_in_stable_detection",
    "order_in_global_stable",
    "order_in_global_detection",
    "local_purity_score",
    "assignment_confidence",
    "incoming_link_probability",
    "outgoing_link_probability",
)

TRACKING_COLUMNS = (
    "video",
    "frame",
    "track_id",
    "x",
    "y",
    "w",
    "h",
    "score",
    "identity",
    "id_conf",
)


@dataclass(frozen=True, slots=True)
class SegmentSource:
    source_dir: Path
    manifest_path: Path
    playback_path: Path
    tracking_path: Path
    clip_id: str
    source_path: str
    width: int
    height: int
    fps: float
    shard_index: int
    shard_count: int
    base_start_frame: int
    base_end_frame: int
    segment_start_frame: int
    segment_end_frame: int


@dataclass(frozen=True, slots=True)
class ReIdRecord:
    valid: bool
    global_track_id: int | None
    global_track_uuid: str | None
    display_global_id: str | None
    id_status: str | None
    x1: float
    y1: float
    x2: float
    y2: float
    bbox_confidence: float
    qa_flags: int
    invalid_reason: str
    global_frame: int
    global_time_sec: float


@dataclass(frozen=True, slots=True)
class ReIdClip:
    sequence_id: str
    clip_id: str
    source_path: str
    width: int
    height: int
    fps: float
    detection_count: int
    valid_detection_count: int
    invalid_detection_count: int
    records: dict[tuple[int, int], ReIdRecord]


@dataclass(frozen=True, slots=True)
class ReIdClipSource:
    sequence_id: str
    clip_id: str
    source_path: str
    width: int
    height: int
    fps: float
    detection_count: int
    valid_detection_count: int
    invalid_detection_count: int


class PeriodicProgress:
    def __init__(self, label: str, total: int | None = None) -> None:
        self.label = label
        self.total = total
        self.started = time.monotonic()
        self.last_print = 0.0
        self.update(0, force=True)

    def update(self, current: int, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self.last_print < 10.0:
            return
        elapsed = max(now - self.started, 1e-9)
        rate = current / elapsed
        if self.total is None:
            suffix = f"{current:,} rows"
        else:
            percent = 100.0 * current / self.total if self.total else 100.0
            suffix = f"{current:,}/{self.total:,} rows ({percent:.2f}%)"
        print(f"[{self.label}] {suffix}; {rate:,.0f} rows/s", flush=True)
        self.last_print = now


def _require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Required file is missing: {path}")


def _read_json(path: Path) -> dict[str, Any]:
    _require_file(path)
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return data


def _required_int(value: Any, label: str) -> int:
    text = str(value).strip()
    if not text:
        raise ValueError(f"Missing integer {label}")
    try:
        parsed = int(text)
    except ValueError as exc:
        raise ValueError(f"Invalid integer {label}: {text!r}") from exc
    return parsed


def _required_float(value: Any, label: str) -> float:
    text = str(value).strip()
    if not text:
        raise ValueError(f"Missing float {label}")
    try:
        parsed = float(text)
    except ValueError as exc:
        raise ValueError(f"Invalid float {label}: {text!r}") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"Non-finite float {label}: {text!r}")
    return parsed


def _required_bool(value: Any, label: str) -> bool:
    text = str(value).strip().lower()
    if text == "true":
        return True
    if text == "false":
        return False
    raise ValueError(f"Invalid boolean {label}: {value!r}")


def _file_stat(path: Path) -> tuple[int, int, int, int, int]:
    _require_file(path)
    stat = path.stat()
    return (
        int(stat.st_dev),
        int(stat.st_ino),
        int(stat.st_size),
        int(stat.st_mtime_ns),
        int(stat.st_ctime_ns),
    )


def _authoritative_source_path(manifest: dict[str, Any], manifest_path: Path) -> str:
    video_manifest = manifest.get("video_manifest")
    if not isinstance(video_manifest, dict):
        raise ValueError(f"manifest.json is missing video_manifest: {manifest_path}")
    source_path = str(video_manifest.get("source_path", "")).strip()
    if not source_path:
        raise ValueError(f"manifest.json is missing authoritative video_manifest.source_path: {manifest_path}")
    if not Path(source_path).is_absolute():
        raise ValueError(f"video_manifest.source_path must be an absolute full path: {source_path}")
    return source_path


def discover_current_segments() -> list[SegmentSource]:
    if not STAGE1_CURRENT_ROOT.is_dir():
        raise FileNotFoundError(f"Current Stage1 root is missing: {STAGE1_CURRENT_ROOT}")
    playback_paths = sorted(STAGE1_CURRENT_ROOT.glob("*/playback_segment.json"))
    if not playback_paths:
        raise RuntimeError(f"No playback segments found under {STAGE1_CURRENT_ROOT}")

    segments: list[SegmentSource] = []
    source_by_clip: dict[str, str] = {}
    clip_by_source: dict[str, str] = {}
    for playback_path in playback_paths:
        source_dir = playback_path.parent
        manifest_path = source_dir / "manifest.json"
        tracking_path = source_dir / "tracking_boxes.csv"
        manifest = _read_json(manifest_path)
        playback = _read_json(playback_path)
        _require_file(tracking_path)

        farm_id = str(playback.get("farm_id", "")).strip()
        camera_id = str(playback.get("camera", "")).strip()
        clip_id = str(playback.get("gx_id", "")).strip()
        if farm_id != "1" or camera_id != "Gopro1":
            raise ValueError(f"Unexpected active Stage1 scope in {playback_path}: farm={farm_id}, camera={camera_id}")
        if clip_id not in EXPECTED_CLIP_IDS:
            raise ValueError(f"Unexpected current clip id in {playback_path}: {clip_id}")

        shard_index = _required_int(playback.get("shard_index"), f"{playback_path}:shard_index")
        shard_count = _required_int(playback.get("shard_count"), f"{playback_path}:shard_count")
        if source_dir.name != f"{clip_id}_{shard_index}":
            raise ValueError(f"Segment directory does not match playback identity: {source_dir}")

        source_path = _authoritative_source_path(manifest, manifest_path)
        if Path(source_path).stem != clip_id:
            raise ValueError(
                f"Full source path basename disagrees with the sequence/clip mapping: clip={clip_id}, source={source_path}"
            )
        old_source = source_by_clip.setdefault(clip_id, source_path)
        if old_source != source_path:
            raise ValueError(f"Clip maps to multiple full source paths: {clip_id}: {old_source!r}, {source_path!r}")
        old_clip = clip_by_source.setdefault(source_path, clip_id)
        if old_clip != clip_id:
            raise ValueError(f"Full source path maps to multiple clips: {source_path}: {old_clip}, {clip_id}")

        video_manifest = manifest["video_manifest"]
        width = _required_int(video_manifest.get("width"), f"{manifest_path}:width")
        height = _required_int(video_manifest.get("height"), f"{manifest_path}:height")
        fps = _required_float(video_manifest.get("fps"), f"{manifest_path}:fps")
        if (width, height) != (3840, 2160):
            raise ValueError(f"Unexpected source dimensions for current re-ID data: {source_path}: {width}x{height}")

        segment = SegmentSource(
            source_dir=source_dir,
            manifest_path=manifest_path,
            playback_path=playback_path,
            tracking_path=tracking_path,
            clip_id=clip_id,
            source_path=source_path,
            width=width,
            height=height,
            fps=fps,
            shard_index=shard_index,
            shard_count=shard_count,
            base_start_frame=_required_int(playback.get("base_start_frame"), f"{playback_path}:base_start_frame"),
            base_end_frame=_required_int(playback.get("base_end_frame"), f"{playback_path}:base_end_frame"),
            segment_start_frame=_required_int(
                playback.get("segment_start_frame"), f"{playback_path}:segment_start_frame"
            ),
            segment_end_frame=_required_int(playback.get("segment_end_frame"), f"{playback_path}:segment_end_frame"),
        )
        if segment.base_start_frame > segment.base_end_frame:
            raise ValueError(f"Invalid base frame range: {playback_path}")
        if not (
            segment.segment_start_frame <= segment.base_start_frame
            and segment.base_end_frame <= segment.segment_end_frame
        ):
            raise ValueError(f"Base frame range is outside segment range: {playback_path}")
        segments.append(segment)

    if set(source_by_clip) != set(EXPECTED_CLIP_IDS):
        missing = sorted(set(EXPECTED_CLIP_IDS) - set(source_by_clip))
        extra = sorted(set(source_by_clip) - set(EXPECTED_CLIP_IDS))
        raise ValueError(f"Current full-path clip mapping is not exact; missing={missing}, extra={extra}")

    for clip_id in EXPECTED_CLIP_IDS:
        clip_segments = sorted((item for item in segments if item.clip_id == clip_id), key=lambda item: item.shard_index)
        shard_counts = {item.shard_count for item in clip_segments}
        if len(shard_counts) != 1:
            raise ValueError(f"Inconsistent shard_count for {clip_id}: {sorted(shard_counts)}")
        expected_count = shard_counts.pop()
        if [item.shard_index for item in clip_segments] != list(range(1, expected_count + 1)):
            raise ValueError(f"Non-contiguous shard indexes for {clip_id}")
        for previous, current in zip(clip_segments, clip_segments[1:]):
            if current.base_start_frame != previous.base_end_frame + 1:
                raise ValueError(f"Non-contiguous canonical base ranges for {clip_id}")

    return sorted(segments, key=lambda item: (EXPECTED_CLIP_IDS.index(item.clip_id), item.shard_index))


def _create_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        PRAGMA foreign_keys=ON;

        CREATE TABLE metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        ) STRICT, WITHOUT ROWID;

        CREATE TABLE source_files (
            role TEXT NOT NULL,
            path TEXT NOT NULL,
            device INTEGER NOT NULL,
            inode INTEGER NOT NULL,
            size_bytes INTEGER NOT NULL,
            mtime_ns INTEGER NOT NULL,
            ctime_ns INTEGER NOT NULL,
            PRIMARY KEY (role, path)
        ) STRICT, WITHOUT ROWID;

        CREATE TABLE clips (
            clip_key INTEGER PRIMARY KEY,
            sequence_id TEXT NOT NULL,
            clip_id TEXT NOT NULL,
            clip_order INTEGER NOT NULL,
            source_path TEXT NOT NULL,
            width INTEGER NOT NULL,
            height INTEGER NOT NULL,
            fps REAL NOT NULL,
            detection_count INTEGER NOT NULL DEFAULT 0,
            valid_detection_count INTEGER NOT NULL DEFAULT 0,
            invalid_detection_count INTEGER NOT NULL DEFAULT 0,
            UNIQUE (sequence_id, clip_id),
            UNIQUE (sequence_id, clip_order),
            UNIQUE (source_path)
        ) STRICT;

        CREATE TABLE global_identities (
            global_track_uuid TEXT PRIMARY KEY,
            global_track_id INTEGER NOT NULL UNIQUE,
            display_global_id TEXT NOT NULL UNIQUE
        ) STRICT, WITHOUT ROWID;

        CREATE TABLE detections (
            clip_key INTEGER NOT NULL REFERENCES clips(clip_key),
            local_frame INTEGER NOT NULL,
            legacy_track_id INTEGER NOT NULL,
            global_frame INTEGER NOT NULL CHECK (global_frame >= 0),
            global_time_sec REAL NOT NULL CHECK (global_time_sec >= 0.0),
            global_track_uuid TEXT REFERENCES global_identities(global_track_uuid),
            id_status TEXT,
            x1 REAL NOT NULL,
            y1 REAL NOT NULL,
            x2 REAL NOT NULL,
            y2 REAL NOT NULL,
            bbox_confidence REAL NOT NULL,
            valid INTEGER NOT NULL CHECK (valid IN (0, 1)),
            qa_flags INTEGER NOT NULL,
            invalid_reason TEXT NOT NULL,
            PRIMARY KEY (clip_key, local_frame, legacy_track_id),
            CHECK (
                (valid = 1 AND global_track_uuid IS NOT NULL AND id_status IS NOT NULL)
                OR
                (valid = 0 AND global_track_uuid IS NULL AND id_status IS NULL)
            )
        ) STRICT, WITHOUT ROWID;
        """
    )


def _insert_clips(connection: sqlite3.Connection, segments: list[SegmentSource]) -> dict[str, int]:
    by_clip: dict[str, SegmentSource] = {}
    for segment in segments:
        by_clip.setdefault(segment.clip_id, segment)
    clip_keys: dict[str, int] = {}
    for clip_order, clip_id in enumerate(EXPECTED_CLIP_IDS):
        clip_key = clip_order + 1
        segment = by_clip[clip_id]
        connection.execute(
            """
            INSERT INTO clips (
                clip_key, sequence_id, clip_id, clip_order, source_path, width, height, fps
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                clip_key,
                REID_SEQUENCE_ID,
                clip_id,
                clip_order,
                segment.source_path,
                segment.width,
                segment.height,
                segment.fps,
            ),
        )
        clip_keys[clip_id] = clip_key
    return clip_keys


def _import_reid_csv(connection: sqlite3.Connection, clip_keys: dict[str, int]) -> dict[str, Any]:
    _require_file(REID_SOURCE_CSV)
    row_count = 0
    valid_count = 0
    invalid_count = 0
    invalid_reason_counts: dict[str, int] = {}
    per_clip_total = {clip_id: 0 for clip_id in EXPECTED_CLIP_IDS}
    per_clip_valid = {clip_id: 0 for clip_id in EXPECTED_CLIP_IDS}
    expected_csv_index = {clip_id: 0 for clip_id in EXPECTED_CLIP_IDS}
    uuid_by_global_id: dict[int, str] = {}
    display_by_global_id: dict[int, str] = {}
    global_id_by_uuid: dict[str, int] = {}
    global_id_by_display: dict[str, int] = {}
    batch: list[tuple[Any, ...]] = []
    progress = PeriodicProgress("reid-import", EXPECTED_REID_ROW_COUNT)

    with REID_SOURCE_CSV.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != REID_COLUMNS:
            raise ValueError(
                f"Unexpected re-ID CSV columns: expected={list(REID_COLUMNS)}, actual={reader.fieldnames}"
            )
        for row in reader:
            row_count += 1
            context = f"{REID_SOURCE_CSV}:line {reader.line_num}"
            sequence_id = str(row["sequence_id"]).strip()
            clip_id = str(row["clip_id"]).strip()
            if sequence_id != REID_SEQUENCE_ID:
                raise ValueError(f"Unexpected sequence_id at {context}: {sequence_id}")
            if clip_id not in clip_keys:
                raise ValueError(f"Unexpected clip_id at {context}: {clip_id}")
            clip_order = _required_int(row["clip_order"], f"{context}:clip_order")
            if clip_order != EXPECTED_CLIP_IDS.index(clip_id):
                raise ValueError(f"clip_order mismatch at {context}: {clip_id} -> {clip_order}")
            csv_row_index = _required_int(row["csv_row_index"], f"{context}:csv_row_index")
            if csv_row_index != expected_csv_index[clip_id]:
                raise ValueError(
                    f"Non-contiguous csv_row_index at {context}: expected={expected_csv_index[clip_id]}, actual={csv_row_index}"
                )
            expected_csv_index[clip_id] += 1

            _required_int(row["det_id"], f"{context}:det_id")
            local_frame = _required_int(row["local_frame"], f"{context}:local_frame")
            legacy_track_id = _required_int(row["legacy_track_id"], f"{context}:legacy_track_id")
            global_frame = _required_int(row["global_frame"], f"{context}:global_frame")
            global_time_sec = _required_float(row["global_time_sec"], f"{context}:global_time_sec")
            if legacy_track_id < 0:
                raise ValueError(f"Negative local track id at {context}")
            expected_global_frame, expected_global_time_sec = reid_global_timeline(clip_id, local_frame)
            if global_frame != expected_global_frame:
                raise ValueError(
                    f"global_frame disagrees with the strict re-ID timeline at {context}: "
                    f"expected={expected_global_frame}, actual={global_frame}"
                )
            if not math.isclose(
                global_time_sec,
                expected_global_time_sec,
                rel_tol=0.0,
                abs_tol=REID_GLOBAL_TIME_TOLERANCE_SEC,
            ):
                raise ValueError(
                    f"global_time_sec disagrees with the strict re-ID timeline at {context}: "
                    f"expected={expected_global_time_sec}, actual={global_time_sec}"
                )

            x1 = _required_float(row["x1"], f"{context}:x1")
            y1 = _required_float(row["y1"], f"{context}:y1")
            x2 = _required_float(row["x2"], f"{context}:x2")
            y2 = _required_float(row["y2"], f"{context}:y2")
            bbox_confidence = _required_float(row["bbox_confidence"], f"{context}:bbox_confidence")
            if not (0.0 <= x1 <= x2 <= 3840.0 and 0.0 <= y1 <= y2 <= 2160.0):
                raise ValueError(f"re-ID bbox is outside clamped 3840x2160 bounds at {context}")
            if not 0.0 <= bbox_confidence <= 1.0:
                raise ValueError(f"re-ID bbox confidence is outside [0,1] at {context}")

            valid = _required_bool(row["valid"], f"{context}:valid")
            qa_flags = _required_int(row["qa_flags"], f"{context}:qa_flags")
            invalid_reason = str(row["invalid_reason"]).strip()
            global_track_id: int | None
            id_status: str | None
            if valid:
                valid_count += 1
                per_clip_valid[clip_id] += 1
                if qa_flags not in {0, 1}:
                    raise ValueError(f"Unexpected valid-row qa_flags at {context}: {qa_flags}")
                global_track_id = _required_int(row["global_track_id"], f"{context}:global_track_id")
                global_track_uuid = str(row["global_track_uuid"]).strip()
                display_global_id = str(row["display_global_id"]).strip()
                id_status = str(row["id_status"]).strip()
                identity_basis = str(row["identity_basis"]).strip()
                if id_status != REID_VALID_STATUS:
                    raise ValueError(f"Unexpected id_status at {context}: {id_status!r}")
                if identity_basis != REID_VALID_IDENTITY_BASIS:
                    raise ValueError(f"Unexpected identity_basis at {context}: {identity_basis!r}")
                if invalid_reason:
                    raise ValueError(f"valid=true row has invalid_reason at {context}: {invalid_reason!r}")
                try:
                    canonical_uuid = str(uuid.UUID(global_track_uuid))
                except ValueError as exc:
                    raise ValueError(f"Invalid global_track_uuid at {context}: {global_track_uuid!r}") from exc
                if canonical_uuid != global_track_uuid:
                    raise ValueError(f"global_track_uuid is not canonical lowercase UUID at {context}")
                expected_display_id = f"G{global_track_id + 1:04d}"
                if display_global_id != expected_display_id:
                    raise ValueError(
                        f"display_global_id disagrees with global_track_id at {context}: "
                        f"expected={expected_display_id}, actual={display_global_id}"
                    )

                is_new_global_id = global_track_id not in uuid_by_global_id
                old_uuid = uuid_by_global_id.setdefault(global_track_id, global_track_uuid)
                old_display = display_by_global_id.setdefault(global_track_id, display_global_id)
                old_id_for_uuid = global_id_by_uuid.setdefault(global_track_uuid, global_track_id)
                old_id_for_display = global_id_by_display.setdefault(display_global_id, global_track_id)
                if (
                    old_uuid != global_track_uuid
                    or old_display != display_global_id
                    or old_id_for_uuid != global_track_id
                    or old_id_for_display != global_track_id
                ):
                    raise ValueError(f"Global identity mapping is not bijective at {context}")
                if is_new_global_id:
                    connection.execute(
                        "INSERT INTO global_identities VALUES (?, ?, ?)",
                        (global_track_uuid, global_track_id, display_global_id),
                    )
            else:
                invalid_count += 1
                global_track_id = None
                global_track_uuid = None
                id_status = None
                identity_fields = (
                    "global_track_id",
                    "global_track_uuid",
                    "display_global_id",
                    "id_status",
                    "identity_basis",
                )
                nonempty = {field: row[field] for field in identity_fields if str(row[field]).strip()}
                if nonempty:
                    raise ValueError(f"valid=false row unexpectedly has global identity at {context}: {nonempty}")
                if invalid_reason not in EXPECTED_INVALID_REASON_COUNTS:
                    raise ValueError(f"Unexpected invalid_reason at {context}: {invalid_reason!r}")
                expected_qa_flags = 64 if invalid_reason == "HIGH_IOU_DUPLICATE" else 65
                if qa_flags != expected_qa_flags:
                    raise ValueError(
                        f"invalid_reason/qa_flags mismatch at {context}: reason={invalid_reason}, qa_flags={qa_flags}"
                    )
                invalid_reason_counts[invalid_reason] = invalid_reason_counts.get(invalid_reason, 0) + 1

            batch.append(
                (
                    clip_keys[clip_id],
                    local_frame,
                    legacy_track_id,
                    global_frame,
                    global_time_sec,
                    global_track_uuid,
                    id_status,
                    x1,
                    y1,
                    x2,
                    y2,
                    bbox_confidence,
                    int(valid),
                    qa_flags,
                    invalid_reason,
                )
            )
            per_clip_total[clip_id] += 1
            if len(batch) >= 20_000:
                try:
                    connection.executemany(
                        """
                        INSERT INTO detections (
                            clip_key, local_frame, legacy_track_id, global_frame, global_time_sec,
                            global_track_uuid, id_status,
                            x1, y1, x2, y2, bbox_confidence, valid, qa_flags, invalid_reason
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        batch,
                    )
                except sqlite3.IntegrityError as exc:
                    raise ValueError(f"Duplicate/invalid re-ID mapping near {context}: {exc}") from exc
                batch.clear()
            progress.update(row_count)

    if batch:
        connection.executemany(
            """
            INSERT INTO detections (
                clip_key, local_frame, legacy_track_id, global_frame, global_time_sec,
                global_track_uuid, id_status,
                x1, y1, x2, y2, bbox_confidence, valid, qa_flags, invalid_reason
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            batch,
        )
    progress.update(row_count, force=True)

    if row_count != EXPECTED_REID_ROW_COUNT:
        raise ValueError(f"Unexpected re-ID row count: expected={EXPECTED_REID_ROW_COUNT}, actual={row_count}")
    if valid_count != EXPECTED_REID_VALID_ROW_COUNT or invalid_count != EXPECTED_REID_INVALID_ROW_COUNT:
        raise ValueError(
            "Unexpected re-ID validity counts: "
            f"valid={valid_count}, invalid={invalid_count}, "
            f"expected_valid={EXPECTED_REID_VALID_ROW_COUNT}, expected_invalid={EXPECTED_REID_INVALID_ROW_COUNT}"
        )
    if invalid_reason_counts != EXPECTED_INVALID_REASON_COUNTS:
        raise ValueError(
            f"Unexpected invalid reason counts: expected={EXPECTED_INVALID_REASON_COUNTS}, actual={invalid_reason_counts}"
        )
    if len(uuid_by_global_id) != EXPECTED_GLOBAL_IDENTITY_COUNT:
        raise ValueError(
            f"Unexpected global identity count: expected={EXPECTED_GLOBAL_IDENTITY_COUNT}, actual={len(uuid_by_global_id)}"
        )
    if set(uuid_by_global_id) != set(range(EXPECTED_GLOBAL_IDENTITY_COUNT)):
        raise ValueError("Current global_track_id values are not exactly 0..61")

    for clip_id in EXPECTED_CLIP_IDS:
        connection.execute(
            """
            UPDATE clips
            SET detection_count = ?, valid_detection_count = ?, invalid_detection_count = ?
            WHERE clip_key = ?
            """,
            (
                per_clip_total[clip_id],
                per_clip_valid[clip_id],
                per_clip_total[clip_id] - per_clip_valid[clip_id],
                clip_keys[clip_id],
            ),
        )
    return {
        "row_count": row_count,
        "valid_count": valid_count,
        "invalid_count": invalid_count,
        "invalid_reason_counts": invalid_reason_counts,
        "per_clip_total": per_clip_total,
    }


def _clamp(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)


def _verify_tracking_sources(
    connection: sqlite3.Connection,
    segments: list[SegmentSource],
    clip_keys: dict[str, int],
) -> dict[str, int | float]:
    physical_row_count = 0
    canonical_row_count = 0
    overlap_row_count = 0
    max_bbox_delta = 0.0
    max_confidence_delta = 0.0

    for clip_id in EXPECTED_CLIP_IDS:
        segment_items = [item for item in segments if item.clip_id == clip_id]
        first_segment = segment_items[0]
        reid_rows = connection.execute(
            """
            SELECT local_frame, legacy_track_id, x1, y1, x2, y2, bbox_confidence
            FROM detections
            WHERE clip_key = ?
            """,
            (clip_keys[clip_id],),
        ).fetchall()
        reid_map = {
            (int(row[0]), int(row[1])): (
                float(row[2]),
                float(row[3]),
                float(row[4]),
                float(row[5]),
                float(row[6]),
            )
            for row in reid_rows
        }
        expected_count = connection.execute(
            "SELECT detection_count FROM clips WHERE clip_key = ?", (clip_keys[clip_id],)
        ).fetchone()[0]
        if len(reid_map) != int(expected_count):
            raise ValueError(f"SQLite re-ID key count mismatch for {clip_id}")

        canonical_seen: set[tuple[int, int]] = set()
        all_seen: dict[tuple[int, int], tuple[float, float, float, float, float]] = {}
        for segment in segment_items:
            file_rows = 0
            seen_in_file: set[tuple[int, int]] = set()
            progress = PeriodicProgress(f"tracking-verify {segment.source_dir.name}")
            with segment.tracking_path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                if tuple(reader.fieldnames or ()) != TRACKING_COLUMNS:
                    raise ValueError(
                        f"Unexpected tracking CSV columns in {segment.tracking_path}: {reader.fieldnames}"
                    )
                for row in reader:
                    file_rows += 1
                    physical_row_count += 1
                    context = f"{segment.tracking_path}:line {reader.line_num}"
                    if str(row["video"]).strip() != f"{clip_id}.MP4":
                        raise ValueError(f"Tracking video column disagrees with full-path clip mapping at {context}")
                    frame = _required_int(row["frame"], f"{context}:frame")
                    track_id = _required_int(row["track_id"], f"{context}:track_id")
                    if not segment.segment_start_frame <= frame <= segment.segment_end_frame:
                        raise ValueError(f"Tracking frame is outside playback segment range at {context}: {frame}")
                    key = (frame, track_id)
                    if key in seen_in_file:
                        raise ValueError(
                            f"Duplicate tracking key inside one CSV: clip={clip_id}, frame={frame}, track={track_id}"
                        )
                    seen_in_file.add(key)
                    reid_bbox = reid_map.get(key)
                    if reid_bbox is None:
                        raise ValueError(
                            f"Tracking row has no unique re-ID key: clip={clip_id}, frame={frame}, track={track_id}"
                        )

                    x = _required_float(row["x"], f"{context}:x")
                    y = _required_float(row["y"], f"{context}:y")
                    w = _required_float(row["w"], f"{context}:w")
                    h = _required_float(row["h"], f"{context}:h")
                    score = _required_float(row["score"], f"{context}:score")
                    if w < 0.0 or h < 0.0:
                        raise ValueError(f"Negative tracking bbox size at {context}")
                    raw = (x, y, x + w, y + h, score)
                    clamped = (
                        _clamp(raw[0], 0.0, float(segment.width)),
                        _clamp(raw[1], 0.0, float(segment.height)),
                        _clamp(raw[2], 0.0, float(segment.width)),
                        _clamp(raw[3], 0.0, float(segment.height)),
                    )
                    bbox_delta = max(abs(clamped[index] - reid_bbox[index]) for index in range(4))
                    confidence_delta = abs(score - reid_bbox[4])
                    max_bbox_delta = max(max_bbox_delta, bbox_delta)
                    max_confidence_delta = max(max_confidence_delta, confidence_delta)
                    if bbox_delta > REID_BBOX_TOLERANCE_PX:
                        raise ValueError(
                            f"Clamped bbox mismatch at {context}: tracking={clamped}, reid={reid_bbox[:4]}, "
                            f"max_delta={bbox_delta}"
                        )
                    if confidence_delta > REID_CONFIDENCE_TOLERANCE:
                        raise ValueError(
                            f"bbox confidence mismatch at {context}: tracking={score}, reid={reid_bbox[4]}, "
                            f"delta={confidence_delta}"
                        )

                    previous = all_seen.get(key)
                    if previous is None:
                        all_seen[key] = raw
                    else:
                        overlap_row_count += 1
                        if any(abs(previous[index] - raw[index]) > 1e-9 for index in range(5)):
                            raise ValueError(
                                f"Overlapping Stage1 shard copies disagree for {clip_id}, frame={frame}, track={track_id}"
                            )

                    if segment.base_start_frame <= frame <= segment.base_end_frame:
                        if key in canonical_seen:
                            raise ValueError(
                                f"Canonical Stage1 base ranges contain duplicate key: {clip_id}, frame={frame}, track={track_id}"
                            )
                        canonical_seen.add(key)
                        canonical_row_count += 1
                    progress.update(file_rows)
            progress.update(file_rows, force=True)

        if len(canonical_seen) != len(reid_map):
            missing = next((key for key in reid_map if key not in canonical_seen), None)
            raise ValueError(
                f"Canonical Stage1/re-ID key sets differ for {clip_id}: "
                f"tracking={len(canonical_seen)}, reid={len(reid_map)}, first_missing={missing}"
            )
        del reid_map, reid_rows, canonical_seen, all_seen
        print(f"[tracking-verify] {clip_id} complete", flush=True)

    if physical_row_count != EXPECTED_PHYSICAL_TRACKING_ROW_COUNT:
        raise ValueError(
            f"Unexpected physical tracking row count: expected={EXPECTED_PHYSICAL_TRACKING_ROW_COUNT}, "
            f"actual={physical_row_count}"
        )
    if canonical_row_count != EXPECTED_REID_ROW_COUNT:
        raise ValueError(
            f"Unexpected canonical tracking row count: expected={EXPECTED_REID_ROW_COUNT}, actual={canonical_row_count}"
        )
    if overlap_row_count != EXPECTED_OVERLAP_TRACKING_ROW_COUNT:
        raise ValueError(
            f"Unexpected overlap tracking row count: expected={EXPECTED_OVERLAP_TRACKING_ROW_COUNT}, "
            f"actual={overlap_row_count}"
        )
    return {
        "physical_row_count": physical_row_count,
        "canonical_row_count": canonical_row_count,
        "overlap_row_count": overlap_row_count,
        "max_bbox_delta": max_bbox_delta,
        "max_confidence_delta": max_confidence_delta,
    }


def _record_source_files(
    connection: sqlite3.Connection,
    initial_stats: dict[tuple[str, Path], tuple[int, int, int, int, int]],
) -> None:
    for (role, path), before in sorted(initial_stats.items(), key=lambda item: (item[0][0], str(item[0][1]))):
        after = _file_stat(path)
        if after != before:
            raise RuntimeError(f"Source file changed while building re-ID index: {path}")
        connection.execute(
            "INSERT INTO source_files VALUES (?, ?, ?, ?, ?, ?, ?)",
            (role, str(path), *after),
        )


def build_reid_index() -> Path:
    segments = discover_current_segments()
    source_paths: dict[tuple[str, Path], tuple[int, int, int, int, int]] = {
        ("reid_csv", REID_SOURCE_CSV): _file_stat(REID_SOURCE_CSV)
    }
    for segment in segments:
        for role, path in (
            ("manifest", segment.manifest_path),
            ("playback", segment.playback_path),
            ("tracking", segment.tracking_path),
        ):
            source_paths[(role, path)] = _file_stat(path)

    REID_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = REID_INDEX_PATH.with_name(f".{REID_INDEX_PATH.name}.{os.getpid()}.tmp")
    if temporary_path.exists():
        temporary_path.unlink()
    connection: sqlite3.Connection | None = None
    try:
        connection = sqlite3.connect(temporary_path)
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("PRAGMA synchronous=OFF")
        connection.execute("PRAGMA temp_store=MEMORY")
        connection.execute("PRAGMA locking_mode=EXCLUSIVE")
        _create_schema(connection)
        clip_keys = _insert_clips(connection, segments)
        import_stats = _import_reid_csv(connection, clip_keys)
        tracking_stats = _verify_tracking_sources(connection, segments, clip_keys)
        _record_source_files(connection, source_paths)

        metadata = {
            "schema_version": str(REID_SCHEMA_VERSION),
            "sequence_id": REID_SEQUENCE_ID,
            "source_csv": str(REID_SOURCE_CSV),
            "stage1_root": str(STAGE1_CURRENT_ROOT),
            "row_count": str(import_stats["row_count"]),
            "valid_row_count": str(import_stats["valid_count"]),
            "invalid_row_count": str(import_stats["invalid_count"]),
            "physical_tracking_row_count": str(tracking_stats["physical_row_count"]),
            "canonical_tracking_row_count": str(tracking_stats["canonical_row_count"]),
            "overlap_tracking_row_count": str(tracking_stats["overlap_row_count"]),
            "global_identity_count": str(EXPECTED_GLOBAL_IDENTITY_COUNT),
            "bbox_rule": "clamp_stage1_xyxy_to_manifest_bounds_then_compare",
            "bbox_tolerance_px": str(REID_BBOX_TOLERANCE_PX),
            "confidence_tolerance": str(REID_CONFIDENCE_TOLERANCE),
            "validity_policy": REID_VALIDITY_POLICY,
            "timeline_fps": str(EXPECTED_SEQUENCE_FPS),
            "timeline_frame_count": str(EXPECTED_SEQUENCE_FRAME_COUNT),
            "timeline_policy": "reid_global_frame_and_global_time_sec_strict_contiguous_sequence",
            "validation_complete": "1",
        }
        connection.executemany("INSERT INTO metadata VALUES (?, ?)", sorted(metadata.items()))
        connection.commit()
        foreign_key_errors = connection.execute("PRAGMA foreign_key_check").fetchall()
        if foreign_key_errors:
            raise RuntimeError(f"Foreign key check failed: {foreign_key_errors[:3]}")
        quick_check = connection.execute("PRAGMA quick_check").fetchone()
        if quick_check is None or quick_check[0] != "ok":
            raise RuntimeError(f"SQLite quick_check failed: {quick_check}")
        connection.close()
        connection = None
        os.replace(temporary_path, REID_INDEX_PATH)
        print(
            f"[complete] strict re-ID index published: {REID_INDEX_PATH} "
            f"({EXPECTED_REID_VALID_ROW_COUNT:,} valid; {EXPECTED_REID_INVALID_ROW_COUNT:,} missed)",
            flush=True,
        )
        return REID_INDEX_PATH
    except Exception:
        if connection is not None:
            connection.close()
        if temporary_path.exists():
            temporary_path.unlink()
        raise


def _open_read_only_index() -> sqlite3.Connection:
    _require_file(REID_INDEX_PATH)
    connection = sqlite3.connect(f"file:{REID_INDEX_PATH}?mode=ro", uri=True)
    connection.execute("PRAGMA query_only=ON")
    return connection


def validate_reid_index() -> dict[str, str]:
    try:
        connection = _open_read_only_index()
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Strict re-ID index is missing: {REID_INDEX_PATH}. Build it with: python3 -u reid_index.py"
        ) from exc
    try:
        metadata = {str(row[0]): str(row[1]) for row in connection.execute("SELECT key, value FROM metadata")}
        expected = {
            "schema_version": str(REID_SCHEMA_VERSION),
            "sequence_id": REID_SEQUENCE_ID,
            "source_csv": str(REID_SOURCE_CSV),
            "stage1_root": str(STAGE1_CURRENT_ROOT),
            "row_count": str(EXPECTED_REID_ROW_COUNT),
            "valid_row_count": str(EXPECTED_REID_VALID_ROW_COUNT),
            "invalid_row_count": str(EXPECTED_REID_INVALID_ROW_COUNT),
            "physical_tracking_row_count": str(EXPECTED_PHYSICAL_TRACKING_ROW_COUNT),
            "canonical_tracking_row_count": str(EXPECTED_REID_ROW_COUNT),
            "overlap_tracking_row_count": str(EXPECTED_OVERLAP_TRACKING_ROW_COUNT),
            "global_identity_count": str(EXPECTED_GLOBAL_IDENTITY_COUNT),
            "bbox_rule": "clamp_stage1_xyxy_to_manifest_bounds_then_compare",
            "bbox_tolerance_px": str(REID_BBOX_TOLERANCE_PX),
            "confidence_tolerance": str(REID_CONFIDENCE_TOLERANCE),
            "validity_policy": REID_VALIDITY_POLICY,
            "timeline_fps": str(EXPECTED_SEQUENCE_FPS),
            "timeline_frame_count": str(EXPECTED_SEQUENCE_FRAME_COUNT),
            "timeline_policy": "reid_global_frame_and_global_time_sec_strict_contiguous_sequence",
            "validation_complete": "1",
        }
        if metadata != expected:
            raise RuntimeError(f"Strict re-ID index metadata is stale or incompatible: {metadata}")
        source_file_count = 0
        for row in connection.execute(
            "SELECT role, path, device, inode, size_bytes, mtime_ns, ctime_ns FROM source_files"
        ):
            source_file_count += 1
            role = str(row[0])
            path = Path(str(row[1]))
            expected_stat = tuple(int(value) for value in row[2:])
            actual_stat = _file_stat(path)
            if actual_stat != expected_stat:
                raise RuntimeError(f"Strict re-ID index is stale because {role} source changed: {path}")
        if source_file_count != EXPECTED_SOURCE_FILE_COUNT:
            raise RuntimeError(
                f"Strict re-ID index source-file inventory is incomplete: "
                f"expected={EXPECTED_SOURCE_FILE_COUNT}, actual={source_file_count}"
            )
        clip_count = int(connection.execute("SELECT COUNT(*) FROM clips").fetchone()[0])
        identity_count = int(connection.execute("SELECT COUNT(*) FROM global_identities").fetchone()[0])
        if clip_count != len(EXPECTED_CLIP_IDS) or identity_count != EXPECTED_GLOBAL_IDENTITY_COUNT:
            raise RuntimeError(
                f"Strict re-ID index dimension mismatch: clips={clip_count}, identities={identity_count}"
            )
        return metadata
    finally:
        connection.close()


def load_reid_clip_for_source(source_path: str) -> ReIdClip:
    validate_reid_index()
    authoritative_path = str(source_path).strip()
    if not authoritative_path or not Path(authoritative_path).is_absolute():
        raise ValueError(f"A non-empty absolute manifest video_manifest.source_path is required: {source_path!r}")
    connection = _open_read_only_index()
    try:
        clip_rows = connection.execute(
            """
            SELECT clip_key, sequence_id, clip_id, source_path, width, height, fps,
                   detection_count, valid_detection_count, invalid_detection_count
            FROM clips
            WHERE source_path = ?
            """,
            (authoritative_path,),
        ).fetchall()
        if len(clip_rows) != 1:
            raise ValueError(
                f"Full source path must map to exactly one current re-ID clip: {authoritative_path}; matches={len(clip_rows)}"
            )
        clip_row = clip_rows[0]
        clip_key = int(clip_row[0])
        records: dict[tuple[int, int], ReIdRecord] = {}
        for row in connection.execute(
            """
            SELECT d.local_frame, d.legacy_track_id, d.valid,
                   g.global_track_id, d.global_track_uuid, g.display_global_id, d.id_status,
                   d.x1, d.y1, d.x2, d.y2, d.bbox_confidence, d.qa_flags, d.invalid_reason,
                   d.global_frame, d.global_time_sec
            FROM detections AS d
            LEFT JOIN global_identities AS g ON g.global_track_uuid = d.global_track_uuid
            WHERE d.clip_key = ?
            ORDER BY d.local_frame, d.legacy_track_id
            """,
            (clip_key,),
        ):
            key = (int(row[0]), int(row[1]))
            valid = bool(int(row[2]))
            global_track_id = int(row[3]) if row[3] is not None else None
            global_track_uuid = str(row[4]) if row[4] is not None else None
            display_global_id = str(row[5]) if row[5] is not None else None
            id_status = str(row[6]) if row[6] is not None else None
            expected_global_frame, expected_global_time_sec = reid_global_timeline(str(clip_row[2]), key[0])
            if int(row[14]) != expected_global_frame or not math.isclose(
                float(row[15]),
                expected_global_time_sec,
                rel_tol=0.0,
                abs_tol=REID_GLOBAL_TIME_TOLERANCE_SEC,
            ):
                raise RuntimeError(f"Runtime re-ID timeline mismatch: clip={clip_row[2]}, key={key}")
            if valid and (
                global_track_id is None
                or not global_track_uuid
                or not display_global_id
                or not id_status
            ):
                raise RuntimeError(f"valid=true index row is missing global identity: {key}")
            if not valid and any(
                value is not None for value in (global_track_id, global_track_uuid, display_global_id, id_status)
            ):
                raise RuntimeError(f"valid=false index row unexpectedly has global identity: {key}")
            if key in records:
                raise RuntimeError(f"Duplicate re-ID index key at runtime: {key}")
            records[key] = ReIdRecord(
                valid=valid,
                global_track_id=global_track_id,
                global_track_uuid=global_track_uuid,
                display_global_id=display_global_id,
                id_status=id_status,
                x1=float(row[7]),
                y1=float(row[8]),
                x2=float(row[9]),
                y2=float(row[10]),
                bbox_confidence=float(row[11]),
                qa_flags=int(row[12]),
                invalid_reason=str(row[13]),
                global_frame=int(row[14]),
                global_time_sec=float(row[15]),
            )
        expected_count = int(clip_row[7])
        if len(records) != expected_count:
            raise RuntimeError(
                f"Runtime re-ID clip row count mismatch for {authoritative_path}: "
                f"expected={expected_count}, actual={len(records)}"
            )
        return ReIdClip(
            sequence_id=str(clip_row[1]),
            clip_id=str(clip_row[2]),
            source_path=str(clip_row[3]),
            width=int(clip_row[4]),
            height=int(clip_row[5]),
            fps=float(clip_row[6]),
            detection_count=expected_count,
            valid_detection_count=int(clip_row[8]),
            invalid_detection_count=int(clip_row[9]),
            records=records,
        )
    finally:
        connection.close()


def resolve_reid_clip_source(source_path: str) -> ReIdClipSource:
    validate_reid_index()
    authoritative_path = str(source_path).strip()
    if not authoritative_path or not Path(authoritative_path).is_absolute():
        raise ValueError(f"A non-empty absolute manifest video_manifest.source_path is required: {source_path!r}")
    connection = _open_read_only_index()
    try:
        rows = connection.execute(
            """
            SELECT sequence_id, clip_id, source_path, width, height, fps,
                   detection_count, valid_detection_count, invalid_detection_count
            FROM clips
            WHERE source_path = ?
            """,
            (authoritative_path,),
        ).fetchall()
        if len(rows) != 1:
            raise ValueError(
                f"Full source path must map to exactly one current re-ID clip: {authoritative_path}; matches={len(rows)}"
            )
        row = rows[0]
        return ReIdClipSource(
            sequence_id=str(row[0]),
            clip_id=str(row[1]),
            source_path=str(row[2]),
            width=int(row[3]),
            height=int(row[4]),
            fps=float(row[5]),
            detection_count=int(row[6]),
            valid_detection_count=int(row[7]),
            invalid_detection_count=int(row[8]),
        )
    finally:
        connection.close()


def main() -> None:
    build_reid_index()


if __name__ == "__main__":
    main()
