#!/usr/bin/env python3
"""Apply occurrence reviews to S6 detections and render corrected full videos."""

from __future__ import annotations

import argparse
import bisect
import colorsys
import csv
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_FLOOR
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping, Sequence, TextIO

from review_lock import ReviewFileLock, ReviewLockUnavailable


ROOT = Path(__file__).resolve().parent
S6_ROOT = Path("/home/hyw/re-identification-S6")
S6_SOURCE_ROOT = S6_ROOT / "src"
if str(S6_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(S6_SOURCE_ROOT))

try:
    import cv2
    import numpy as np
    from cowtrack.config import ContractError
    from cowtrack.qa.ffprobe import validate_qa_mp4
    from cowtrack.qa.nvenc import NvencVideoWriter
    from cowtrack.video import open_raw_video_capture, probe_video_stream
except ImportError as exc:  # pragma: no cover - environment preflight
    raise SystemExit(
        "error: this script requires the dcsna environment plus the fixed S6 source tree: "
        f"{exc}"
    ) from exc


OCCURRENCES_PATH = ROOT / "occurrence_segments.csv"
REVIEWS_PATH = ROOT / "occurrence_reviews.csv"
REVIEW_LOCK_PATH = ROOT / ".occurrence_reviews.csv.lock"
SUMMARY_PATH = ROOT / "bbox_occurrence_summary.json"
SOURCE_DETECTIONS_PATH = (
    S6_ROOT
    / "work/dairy_farm_1_gopro1_20250505/06_export/detections_with_global_id.csv"
)
GLOBAL_SUMMARY_PATH = (
    S6_ROOT
    / "work/dairy_farm_1_gopro1_20250505/06_export/qa/global_track_summary.csv"
)

RAW_VIDEO_PATHS = {
    "GX040006": Path(
        "/mnt/dairycow_sna/FULLDATA/Dairy Farm Videos/"
        "May 5 2025 Dairy Farm 1 Videos/Gopro1/100GOPRO/GX040006.MP4"
    ),
    "GX050006": Path(
        "/mnt/dairycow_sna/FULLDATA/Dairy Farm Videos/"
        "May 5 2025 Dairy Farm 1 Videos/Gopro1/100GOPRO/GX050006.MP4"
    ),
}
CLIP_FRAME_COUNTS = {"GX040006": 84_480, "GX050006": 78_720}
CLIP_ORDERS = {"GX040006": 0, "GX050006": 1}
CLIP_GLOBAL_FRAME_STARTS = {"GX040006": 0, "GX050006": 84_480}
RAW_VIDEO_SIZES = {"GX040006": 11_767_870_277, "GX050006": 11_634_906_256}
RAW_VIDEO_SHA256 = {
    "GX040006": "c6042d1db0ba91d985c0d3f84334007cd9cd8d17a61dafd664fe7296c6995383",
    "GX050006": "b0597a730d2a8260001e76ba4f76236927c5bcec52b130dbcbc68ecb99568957",
}

EXPECTED_SOURCE_SHA256 = "dcf6d6742acc85e44f1f4633b41f3c6df76b72069edbdbfa651c967082121e32"
EXPECTED_SOURCE_ROWS = 746_279
EXPECTED_VALID_ROWS = 745_915
EXPECTED_ORIGINAL_INVALID_ROWS = 364
EXPECTED_OCCURRENCES = 2_414
EXPECTED_REVIEWS_SHA256 = (
    "eb03c15d4c631a942c8f6173675c804c3c1cfc6849420f1aaec7a077ad427d11"
)
EXPECTED_ACTION_OCCURRENCES = {
    "accept": 402,
    "invalid_multiple_cows": 1_068,
    "update_id": 944,
}
EXPECTED_ACTION_DETECTIONS = {
    "accept": 303_191,
    "invalid_multiple_cows": 169_873,
    "update_id": 272_851,
}

FPS = Fraction(30_000, 1_001)
OUTPUT_WIDTH = 1_920
OUTPUT_HEIGHT = 1_080
RAW_WIDTH = 3_840
RAW_HEIGHT = 2_160
# Both fixed GoPro inputs carry -90 metadata.  The S6 coordinate contract uses
# the encoded 3840x2160 raster, so open_raw_video_capture disables autorotation.
RAW_ROTATION_METADATA_DEGREES = -90
PROGRESS_INTERVAL_SEC = 10.0
MAX_SAMPLE_SECONDS = Decimal("180")

SOURCE_FIELDS = (
    "occurrence_id",
    "clip_order",
    "clip_id",
    "display_global_id",
    "legacy_track_id",
    "start_frame",
    "end_frame",
    "start_time_sec",
    "end_time_sec_inclusive",
    "end_time_sec_exclusive",
    "span_duration_sec",
    "num_valid_detections",
    "num_missing_frames",
    "max_internal_gap_missing_frames",
    "start_det_id",
    "end_det_id",
)
REVIEW_FIELDS = (
    "review_action",
    "reviewed_global_id",
    "invalid_multiple_cows",
    "reviewed_at_utc",
)
DETECTION_FIELDS = (
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
STRING_DETECTION_FIELDS = frozenset(
    {
        "sequence_id",
        "clip_id",
        "legacy_track_id",
        "global_track_uuid",
        "display_global_id",
        "id_status",
        "identity_basis",
        "invalid_reason",
    }
)
IDENTITY_FIELDS = (
    "global_track_id",
    "global_track_uuid",
    "display_global_id",
    "id_status",
    "identity_basis",
)
ORIGINAL_INVALID_BLANK_FIELDS = IDENTITY_FIELDS + (
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
GID_PATTERN = re.compile(r"G([0-9]{4})\Z")
OCCURRENCE_PATTERN = re.compile(r"O[0-9]{6}\Z")


@dataclass(frozen=True, slots=True)
class OccurrenceReview:
    occurrence_id: str
    clip_id: str
    clip_order: int
    original_gid: str
    legacy_track_id: str
    start_frame: int
    end_frame: int
    expected_detections: int
    expected_missing_frames: int
    expected_max_gap: int
    start_det_id: str
    end_det_id: str
    action: str
    reviewed_gid: str


@dataclass(frozen=True, slots=True)
class TrackIntervals:
    starts: tuple[int, ...]
    items: tuple[OccurrenceReview, ...]


@dataclass(frozen=True, slots=True)
class RenderDetection:
    det_id: int
    x1: float
    y1: float
    x2: float
    y2: float
    global_track_id: int | None
    display_global_id: str | None
    reviewed_invalid: bool
    review_action: str


@dataclass(frozen=True)
class TransformStats:
    total_rows: int
    valid_rows: int
    original_invalid_rows: int
    detections_by_action: Mapping[str, int]
    output_rows_by_gid: Mapping[str, int]


def log(message: str) -> None:
    print(message, flush=True)


def _canonical_int(value: str, label: str, *, minimum: int | None = None) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"{label} must be an integer, got {value!r}") from exc
    if str(parsed) != value:
        raise ContractError(f"{label} must be a canonical integer, got {value!r}")
    if minimum is not None and parsed < minimum:
        raise ContractError(f"{label} must be >= {minimum}, got {parsed}")
    return parsed


def _gid_number(value: str, label: str) -> int:
    match = GID_PATTERN.fullmatch(value)
    if match is None:
        raise ContractError(f"{label} must be G0001-G0062, got {value!r}")
    number = int(match.group(1))
    if not 1 <= number <= 62:
        raise ContractError(f"{label} must be G0001-G0062, got {value!r}")
    return number


def file_sha256(path: Path, *, progress_label: str | None = None) -> str:
    digest = hashlib.sha256()
    total = path.stat().st_size
    processed = 0
    last_report = time.monotonic()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
            processed += len(chunk)
            now = time.monotonic()
            if progress_label and now - last_report >= PROGRESS_INTERVAL_SEC:
                log(f"[hash] {progress_label}: {processed:,}/{total:,} bytes")
                last_report = now
    return digest.hexdigest()


def file_stat_token(path: Path) -> tuple[int, int, int, int, int]:
    metadata = path.stat()
    return (
        int(metadata.st_dev),
        int(metadata.st_ino),
        int(metadata.st_size),
        int(metadata.st_mtime_ns),
        int(metadata.st_ctime_ns),
    )


def require_unchanged_files(
    snapshot: Mapping[Path, tuple[int, int, int, int, int]],
) -> None:
    for path, expected in snapshot.items():
        if file_stat_token(path) != expected:
            raise ContractError(f"fixed input changed while exporting: {path}")


def _require_regular_file(path: Path, label: str, *, size: int | None = None) -> None:
    if path.is_symlink() or not path.is_file():
        raise ContractError(f"missing fixed {label}: {path}")
    if size is not None and path.stat().st_size != size:
        raise ContractError(
            f"fixed {label} size differs: {path.stat().st_size} != {size}: {path}"
        )


def _strict_json(path: Path) -> Any:
    def pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ContractError(f"duplicate JSON key {key!r} in {path}")
            result[key] = value
        return result

    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle, object_pairs_hook=pairs_hook)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read strict JSON {path}: {exc}") from exc


def validate_fixed_inputs(*, hash_raw_clips: Sequence[str] = ()) -> None:
    for path, label in (
        (OCCURRENCES_PATH, "occurrence CSV"),
        (REVIEWS_PATH, "review CSV"),
        (SUMMARY_PATH, "occurrence summary"),
        (SOURCE_DETECTIONS_PATH, "S6 final detection CSV"),
        (GLOBAL_SUMMARY_PATH, "S6 global-track summary"),
    ):
        _require_regular_file(path, label)
    for clip_id, path in RAW_VIDEO_PATHS.items():
        _require_regular_file(path, f"raw video {clip_id}", size=RAW_VIDEO_SIZES[clip_id])

    summary = _strict_json(SUMMARY_PATH)
    try:
        recorded_source = summary["source"]["sha256"]
        recorded_segments = summary["outputs"]["segments_csv_sha256"]
    except (KeyError, TypeError) as exc:
        raise ContractError("occurrence summary lacks fixed source/output hashes") from exc
    if recorded_source != EXPECTED_SOURCE_SHA256:
        raise ContractError("occurrence summary source hash differs from fixed S6 source")
    actual_segments = file_sha256(OCCURRENCES_PATH)
    if actual_segments != recorded_segments:
        raise ContractError("occurrence CSV hash differs from occurrence summary")
    actual_source = file_sha256(SOURCE_DETECTIONS_PATH, progress_label="S6 detections")
    if actual_source != EXPECTED_SOURCE_SHA256:
        raise ContractError("S6 final detection CSV SHA-256 mismatch")
    for clip_id in hash_raw_clips:
        if clip_id not in RAW_VIDEO_PATHS:
            raise ContractError(f"unknown raw-video hash request: {clip_id}")
        actual_raw = file_sha256(
            RAW_VIDEO_PATHS[clip_id], progress_label=f"raw video {clip_id}"
        )
        if actual_raw != RAW_VIDEO_SHA256[clip_id]:
            raise ContractError(f"raw video SHA-256 mismatch for {clip_id}")


def load_reviews(
    occurrences_path: Path,
    reviews_path: Path,
    *,
    expected_count: int | None = EXPECTED_OCCURRENCES,
    expected_actions: Mapping[str, int] | None = EXPECTED_ACTION_OCCURRENCES,
) -> tuple[tuple[OccurrenceReview, ...], dict[tuple[str, str], TrackIntervals]]:
    try:
        occurrences_handle = occurrences_path.open("r", encoding="utf-8", newline="")
        reviews_handle = reviews_path.open("r", encoding="utf-8", newline="")
    except OSError as exc:
        raise ContractError(f"cannot open occurrence/review CSV: {exc}") from exc

    reviews: list[OccurrenceReview] = []
    action_counts: Counter[str] = Counter()
    with occurrences_handle, reviews_handle:
        occurrence_reader = csv.DictReader(occurrences_handle)
        review_reader = csv.DictReader(reviews_handle)
        if tuple(occurrence_reader.fieldnames or ()) != SOURCE_FIELDS:
            raise ContractError("occurrence CSV header differs from fixed schema")
        if tuple(review_reader.fieldnames or ()) != SOURCE_FIELDS + REVIEW_FIELDS:
            raise ContractError("review CSV header differs from fixed schema")

        sentinel = object()
        occurrence_iterator = iter(occurrence_reader)
        review_iterator = iter(review_reader)
        index = 0
        while True:
            occurrence = next(occurrence_iterator, sentinel)
            review = next(review_iterator, sentinel)
            if occurrence is sentinel and review is sentinel:
                break
            index += 1
            if occurrence is sentinel or review is sentinel:
                raise ContractError("occurrence/review CSV row counts differ")
            assert isinstance(occurrence, dict) and isinstance(review, dict)
            if any(value is None for value in occurrence.values()) or any(
                value is None for value in review.values()
            ):
                raise ContractError(f"malformed occurrence/review row {index}")
            if any(review[field] != occurrence[field] for field in SOURCE_FIELDS):
                raise ContractError(f"review source fields differ at occurrence row {index}")

            occurrence_id = occurrence["occurrence_id"]
            if (
                OCCURRENCE_PATTERN.fullmatch(occurrence_id) is None
                or occurrence_id != f"O{index:06d}"
            ):
                raise ContractError(f"non-sequential occurrence_id at row {index}")
            clip_id = occurrence["clip_id"]
            if clip_id not in CLIP_FRAME_COUNTS:
                raise ContractError(f"unknown clip_id at {occurrence_id}")
            clip_order = _canonical_int(
                occurrence["clip_order"], f"{occurrence_id}.clip_order", minimum=0
            )
            if clip_order != CLIP_ORDERS[clip_id]:
                raise ContractError(f"clip_order mismatch at {occurrence_id}")
            original_gid = occurrence["display_global_id"]
            _gid_number(original_gid, f"{occurrence_id}.display_global_id")
            start = _canonical_int(
                occurrence["start_frame"], f"{occurrence_id}.start_frame", minimum=0
            )
            end = _canonical_int(
                occurrence["end_frame"], f"{occurrence_id}.end_frame", minimum=start
            )
            if end >= CLIP_FRAME_COUNTS[clip_id]:
                raise ContractError(f"occurrence outside video at {occurrence_id}")
            expected_detections = _canonical_int(
                occurrence["num_valid_detections"],
                f"{occurrence_id}.num_valid_detections",
                minimum=1,
            )
            expected_missing_frames = _canonical_int(
                occurrence["num_missing_frames"],
                f"{occurrence_id}.num_missing_frames",
                minimum=0,
            )
            expected_max_gap = _canonical_int(
                occurrence["max_internal_gap_missing_frames"],
                f"{occurrence_id}.max_internal_gap_missing_frames",
                minimum=0,
            )
            if end - start + 1 - expected_detections != expected_missing_frames:
                raise ContractError(f"occurrence span/count mismatch at {occurrence_id}")
            if expected_max_gap > 30 or expected_max_gap > expected_missing_frames:
                raise ContractError(f"occurrence gap metadata mismatch at {occurrence_id}")
            action = review["review_action"]
            if action not in EXPECTED_ACTION_OCCURRENCES:
                raise ContractError(f"unknown review action at {occurrence_id}: {action!r}")
            reviewed_gid = review["reviewed_global_id"]
            _gid_number(reviewed_gid, f"{occurrence_id}.reviewed_global_id")
            expected_invalid = "true" if action == "invalid_multiple_cows" else "false"
            if review["invalid_multiple_cows"] != expected_invalid:
                raise ContractError(f"invalid_multiple_cows flag mismatch at {occurrence_id}")
            if action != "update_id" and reviewed_gid != original_gid:
                raise ContractError(f"retained review ID differs at {occurrence_id}")
            if not review["reviewed_at_utc"].endswith("Z"):
                raise ContractError(f"review timestamp is not UTC at {occurrence_id}")

            reviews.append(
                OccurrenceReview(
                    occurrence_id=occurrence_id,
                    clip_id=clip_id,
                    clip_order=clip_order,
                    original_gid=original_gid,
                    legacy_track_id=occurrence["legacy_track_id"],
                    start_frame=start,
                    end_frame=end,
                    expected_detections=expected_detections,
                    expected_missing_frames=expected_missing_frames,
                    expected_max_gap=expected_max_gap,
                    start_det_id=occurrence["start_det_id"],
                    end_det_id=occurrence["end_det_id"],
                    action=action,
                    reviewed_gid=reviewed_gid,
                )
            )
            action_counts[action] += 1

    if expected_count is not None and len(reviews) != expected_count:
        raise ContractError(f"expected {expected_count:,} reviews, found {len(reviews):,}")
    if expected_actions is not None and dict(action_counts) != dict(expected_actions):
        raise ContractError(
            f"review action counts differ: {dict(action_counts)} != {dict(expected_actions)}"
        )

    grouped: dict[tuple[str, str], list[OccurrenceReview]] = {}
    for review in reviews:
        key = (review.clip_id, review.legacy_track_id)
        grouped.setdefault(key, []).append(review)
    intervals: dict[tuple[str, str], TrackIntervals] = {}
    for key, items in grouped.items():
        items.sort(key=lambda item: (item.start_frame, item.end_frame, item.occurrence_id))
        previous: OccurrenceReview | None = None
        for item in items:
            if previous is not None and item.start_frame <= previous.end_frame:
                raise ContractError(
                    f"overlapping occurrences for {key}: {previous.occurrence_id}, "
                    f"{item.occurrence_id}"
                )
            if (
                previous is not None
                and item.original_gid == previous.original_gid
                and item.start_frame - previous.end_frame - 1 <= 30
            ):
                raise ContractError(
                    "adjacent same-GID occurrences violate the fixed 30-frame gap "
                    f"rule for {key}: {previous.occurrence_id}, {item.occurrence_id}"
                )
            previous = item
        intervals[key] = TrackIntervals(
            tuple(item.start_frame for item in items), tuple(items)
        )
    return tuple(reviews), intervals


def load_gid_mapping(
    path: Path, *, expected_count: int | None = 62
) -> dict[str, tuple[int, str]]:
    mapping: dict[str, tuple[int, str]] = {}
    used_ids: set[int] = set()
    used_uuids: set[str] = set()
    try:
        handle = path.open("r", encoding="utf-8", newline="")
    except OSError as exc:
        raise ContractError(f"cannot open global-track summary {path}: {exc}") from exc
    with handle:
        reader = csv.DictReader(handle)
        required = {"global_track_id", "global_track_uuid", "display_global_id"}
        if not required.issubset(reader.fieldnames or ()):
            raise ContractError("global-track summary lacks ID/UUID fields")
        for row_number, row in enumerate(reader, 2):
            gid = row["display_global_id"]
            number = _gid_number(gid, f"global summary row {row_number}")
            global_id = _canonical_int(
                row["global_track_id"], f"global summary row {row_number}.global_track_id"
            )
            uuid = row["global_track_uuid"]
            if global_id != number - 1 or not uuid:
                raise ContractError(f"global ID/UUID mapping mismatch at row {row_number}")
            if gid in mapping or global_id in used_ids or uuid in used_uuids:
                raise ContractError("global ID/UUID mapping is not bijective")
            mapping[gid] = (global_id, uuid)
            used_ids.add(global_id)
            used_uuids.add(uuid)
    if expected_count is not None and len(mapping) != expected_count:
        raise ContractError(
            f"expected {expected_count} global ID mappings, found {len(mapping)}"
        )
    if expected_count == 62 and set(mapping) != {
        f"G{number:04d}" for number in range(1, 63)
    }:
        raise ContractError("global summary does not cover exactly G0001-G0062")
    return mapping


def lookup_occurrence(
    intervals: Mapping[tuple[str, str], TrackIntervals],
    *,
    clip_id: str,
    legacy_track_id: str,
    local_frame: int,
) -> OccurrenceReview:
    key = (clip_id, legacy_track_id)
    try:
        track = intervals[key]
    except KeyError as exc:
        raise ContractError(f"valid source row has no occurrence stream {key}") from exc
    position = bisect.bisect_right(track.starts, local_frame) - 1
    if position < 0:
        raise ContractError(f"valid source row precedes first occurrence for {key}")
    occurrence = track.items[position]
    if local_frame > occurrence.end_frame:
        raise ContractError(
            f"valid source row is outside occurrence intervals: {key}, frame={local_frame}"
        )
    return occurrence


def _csv_quote(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def write_s6_csv_header(handle: TextIO) -> None:
    handle.write(",".join(_csv_quote(field) for field in DETECTION_FIELDS) + "\n")


def write_s6_csv_row(handle: TextIO, row: Mapping[str, str]) -> None:
    if set(row) != set(DETECTION_FIELDS):
        raise ContractError("output detection row fields differ from fixed 33-column schema")
    cells: list[str] = []
    for field in DETECTION_FIELDS:
        value = row[field]
        if not isinstance(value, str) or "\r" in value or "\n" in value:
            raise ContractError(f"unsafe output CSV value for {field}: {value!r}")
        cells.append(_csv_quote(value) if field in STRING_DETECTION_FIELDS and value else value)
    handle.write(",".join(cells) + "\n")


def transform_detections(
    source_path: Path,
    intervals: Mapping[tuple[str, str], TrackIntervals],
    reviews: Sequence[OccurrenceReview],
    gid_mapping: Mapping[str, tuple[int, str]],
    *,
    output_path: Path | None,
    build_render_index: bool,
    expected_rows: int | None = EXPECTED_SOURCE_ROWS,
    expected_valid_rows: int | None = EXPECTED_VALID_ROWS,
    expected_original_invalid_rows: int | None = EXPECTED_ORIGINAL_INVALID_ROWS,
    expected_action_detections: Mapping[str, int] | None = EXPECTED_ACTION_DETECTIONS,
) -> tuple[TransformStats, dict[str, list[list[RenderDetection]]] | None]:
    render_index = (
        {
            clip_id: [[] for _ in range(frame_count)]
            for clip_id, frame_count in CLIP_FRAME_COUNTS.items()
        }
        if build_render_index
        else None
    )
    observed_counts = {review.occurrence_id: 0 for review in reviews}
    observed_first_det: dict[str, str] = {}
    observed_last_det: dict[str, str] = {}
    observed_first_frame: dict[str, int] = {}
    observed_last_frame: dict[str, int] = {}
    observed_missing: Counter[str] = Counter()
    observed_max_gap: Counter[str] = Counter()
    action_detections: Counter[str] = Counter()
    output_by_gid: Counter[str] = Counter()
    detection_order: Counter[str] = Counter()
    stable_order: dict[str, dict[str, int]] = {}
    total_rows = 0
    valid_rows = 0
    original_invalid_rows = 0
    previous_sort_key: tuple[int, int] | None = None
    seen_det_ids: set[int] = set()
    last_report = time.monotonic()

    # The source is chronological, but rows inside one frame are in CSV order rather
    # than necessarily det_id order.  Buffer exactly one frame so corrected global
    # ranks follow (global_frame, det_id) while final CSV rows stay in source order.
    pending_global_frame: int | None = None
    pending_rows: list[dict[str, str]] = []
    pending_normal: list[tuple[dict[str, str], str, str, int]] = []

    output_handle: TextIO | None = None
    created_output = False
    completed = False

    def flush_pending_frame() -> None:
        nonlocal pending_global_frame
        if not pending_rows:
            return
        by_gid: dict[str, list[tuple[dict[str, str], str, int]]] = {}
        for corrected, target_gid, stable_id, det_id in pending_normal:
            by_gid.setdefault(target_gid, []).append((corrected, stable_id, det_id))
        for target_gid, items in by_gid.items():
            if len(items) > 1:
                det_ids = ",".join(
                    str(det_id)
                    for _, _, det_id in sorted(items, key=lambda item: item[2])
                )
                raise ContractError(
                    "corrected same-frame G-ID collision: "
                    f"global_frame={pending_global_frame}, "
                    f"display_global_id={target_gid}, det_ids={det_ids}"
                )
            by_stable = stable_order.setdefault(target_gid, {})
            first_det_by_stable: dict[str, int] = {}
            for _, stable_id, det_id in items:
                current = first_det_by_stable.get(stable_id)
                if current is None or det_id < current:
                    first_det_by_stable[stable_id] = det_id
            new_stables = [
                (first_det, _canonical_int(stable_id, "stable_id", minimum=0), stable_id)
                for stable_id, first_det in first_det_by_stable.items()
                if stable_id not in by_stable
            ]
            for _, _, stable_id in sorted(new_stables):
                by_stable[stable_id] = len(by_stable)
            seen_frame_det_ids: set[int] = set()
            for corrected, stable_id, det_id in sorted(items, key=lambda item: item[2]):
                if det_id in seen_frame_det_ids:
                    raise ContractError(
                        f"duplicate det_id {det_id} in global frame {pending_global_frame}"
                    )
                seen_frame_det_ids.add(det_id)
                corrected["order_in_global_stable"] = str(by_stable[stable_id])
                corrected["order_in_global_detection"] = str(
                    detection_order[target_gid]
                )
                detection_order[target_gid] += 1
        if output_handle is not None:
            for corrected in pending_rows:
                write_s6_csv_row(output_handle, corrected)
        pending_rows.clear()
        pending_normal.clear()

    try:
        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_handle = output_path.open("x", encoding="utf-8", newline="\n")
            created_output = True
            write_s6_csv_header(output_handle)

        with source_path.open("r", encoding="utf-8", newline="") as source_handle:
            reader = csv.DictReader(source_handle)
            if tuple(reader.fieldnames or ()) != DETECTION_FIELDS:
                raise ContractError("S6 detection CSV header differs from fixed schema")
            for row in reader:
                total_rows += 1
                if set(row) != set(DETECTION_FIELDS) or any(
                    value is None for value in row.values()
                ):
                    raise ContractError(f"malformed S6 detection row {total_rows}")
                clip_id = row["clip_id"]
                if clip_id not in CLIP_FRAME_COUNTS:
                    raise ContractError(f"unknown source clip at row {total_rows}: {clip_id}")
                clip_order = _canonical_int(
                    row["clip_order"], f"source row {total_rows}.clip_order", minimum=0
                )
                csv_row_index = _canonical_int(
                    row["csv_row_index"],
                    f"source row {total_rows}.csv_row_index",
                    minimum=0,
                )
                sort_key = (clip_order, csv_row_index)
                if clip_order != CLIP_ORDERS[clip_id] or (
                    previous_sort_key is not None and sort_key <= previous_sort_key
                ):
                    raise ContractError(f"source row order differs at row {total_rows}")
                previous_sort_key = sort_key

                local_frame = _canonical_int(
                    row["local_frame"], f"source row {total_rows}.local_frame", minimum=0
                )
                if local_frame >= CLIP_FRAME_COUNTS[clip_id]:
                    raise ContractError(f"source frame outside video at row {total_rows}")
                global_frame = _canonical_int(
                    row["global_frame"],
                    f"source row {total_rows}.global_frame",
                    minimum=0,
                )
                if global_frame != CLIP_GLOBAL_FRAME_STARTS[clip_id] + local_frame:
                    raise ContractError(f"source local/global frame mismatch at row {total_rows}")
                if pending_global_frame is None:
                    pending_global_frame = global_frame
                elif global_frame < pending_global_frame:
                    raise ContractError(f"source global frame order differs at row {total_rows}")
                elif global_frame != pending_global_frame:
                    flush_pending_frame()
                    pending_global_frame = global_frame

                det_id = _canonical_int(row["det_id"], f"source row {total_rows}.det_id")
                if det_id in seen_det_ids:
                    raise ContractError(f"duplicate source det_id at row {total_rows}: {det_id}")
                seen_det_ids.add(det_id)
                valid_text = row["valid"]
                if valid_text == "false":
                    original_invalid_rows += 1
                    if any(row[field] for field in ORIGINAL_INVALID_BLANK_FIELDS):
                        raise ContractError(
                            f"original invalid source row has identity/rank data at row {total_rows}"
                        )
                    corrected = dict(row)
                elif valid_text == "true":
                    valid_rows += 1
                    original_gid = row["display_global_id"]
                    try:
                        original_global_id, original_uuid = gid_mapping[original_gid]
                    except KeyError as exc:
                        raise ContractError(
                            f"source row has unknown global ID at row {total_rows}"
                        ) from exc
                    if (
                        row["global_track_id"] != str(original_global_id)
                        or row["global_track_uuid"] != original_uuid
                    ):
                        raise ContractError(
                            f"source global ID/UUID fields disagree at row {total_rows}"
                        )
                    occurrence = lookup_occurrence(
                        intervals,
                        clip_id=clip_id,
                        legacy_track_id=row["legacy_track_id"],
                        local_frame=local_frame,
                    )
                    if original_gid != occurrence.original_gid:
                        raise ContractError(
                            f"source GID differs from occurrence at {occurrence.occurrence_id}"
                        )
                    occurrence_id = occurrence.occurrence_id
                    previous_frame = observed_last_frame.get(occurrence_id)
                    if previous_frame is None:
                        observed_first_frame[occurrence_id] = local_frame
                        observed_first_det[occurrence_id] = row["det_id"]
                    else:
                        gap = local_frame - previous_frame - 1
                        if gap < 0:
                            raise ContractError(
                                f"non-increasing occurrence frames at {occurrence_id}"
                            )
                        observed_missing[occurrence_id] += gap
                        observed_max_gap[occurrence_id] = max(
                            observed_max_gap[occurrence_id], gap
                        )
                    observed_last_frame[occurrence_id] = local_frame
                    observed_last_det[occurrence_id] = row["det_id"]
                    observed_counts[occurrence_id] += 1
                    action_detections[occurrence.action] += 1
                    corrected = dict(row)

                    if occurrence.action == "invalid_multiple_cows":
                        corrected["global_track_id"] = "-1"
                        corrected["global_track_uuid"] = ""
                        corrected["display_global_id"] = "-1"
                        corrected["id_status"] = ""
                        corrected["identity_basis"] = ""
                        corrected["invalid_reason"] = "multiple_cows"
                        corrected["order_in_global_stable"] = ""
                        corrected["order_in_global_detection"] = ""
                        global_track_id: int | None = None
                        display_global_id: str | None = None
                        reviewed_invalid = True
                        output_by_gid["-1"] += 1
                    else:
                        target_gid = occurrence.reviewed_gid
                        try:
                            global_track_id, global_uuid = gid_mapping[target_gid]
                        except KeyError as exc:
                            raise ContractError(
                                f"review target lacks global mapping: {target_gid}"
                            ) from exc
                        corrected["global_track_id"] = str(global_track_id)
                        corrected["global_track_uuid"] = global_uuid
                        corrected["display_global_id"] = target_gid
                        stable_id = corrected["stable_id"]
                        _canonical_int(
                            stable_id, f"source row {total_rows}.stable_id", minimum=0
                        )
                        pending_normal.append(
                            (corrected, target_gid, stable_id, det_id)
                        )
                        display_global_id = target_gid
                        reviewed_invalid = False
                        output_by_gid[target_gid] += 1

                    if render_index is not None:
                        coordinates: list[float] = []
                        for field in ("x1", "y1", "x2", "y2"):
                            try:
                                value = float(row[field])
                            except ValueError as exc:
                                raise ContractError(
                                    f"invalid bbox coordinate at source row {total_rows}"
                                ) from exc
                            if not math.isfinite(value):
                                raise ContractError(
                                    f"non-finite bbox coordinate at source row {total_rows}"
                                )
                            coordinates.append(value)
                        x1, y1, x2, y2 = coordinates
                        if not (
                            0.0 <= x1 < x2 <= RAW_WIDTH
                            and 0.0 <= y1 < y2 <= RAW_HEIGHT
                        ):
                            raise ContractError(
                                f"bbox outside raw frame at source row {total_rows}"
                            )
                        render_index[clip_id][local_frame].append(
                            RenderDetection(
                                det_id=det_id,
                                x1=x1,
                                y1=y1,
                                x2=x2,
                                y2=y2,
                                global_track_id=global_track_id,
                                display_global_id=display_global_id,
                                reviewed_invalid=reviewed_invalid,
                                review_action=occurrence.action,
                            )
                        )
                else:
                    raise ContractError(
                        f"source valid must be true/false at row {total_rows}"
                    )

                pending_rows.append(corrected)
                now = time.monotonic()
                if now - last_report >= PROGRESS_INTERVAL_SEC:
                    log(
                        f"[csv] rows={total_rows:,}, valid={valid_rows:,}, "
                        f"mapped={sum(action_detections.values()):,}"
                    )
                    last_report = now

        flush_pending_frame()
        for review in reviews:
            occurrence_id = review.occurrence_id
            count = observed_counts[occurrence_id]
            if count != review.expected_detections:
                raise ContractError(
                    f"occurrence detection count differs for {occurrence_id}: "
                    f"{count} != {review.expected_detections}"
                )
            if observed_first_det.get(occurrence_id) != review.start_det_id:
                raise ContractError(f"start_det_id differs for {occurrence_id}")
            if observed_last_det.get(occurrence_id) != review.end_det_id:
                raise ContractError(f"end_det_id differs for {occurrence_id}")
            if observed_first_frame.get(occurrence_id) != review.start_frame:
                raise ContractError(f"start_frame differs for {occurrence_id}")
            if observed_last_frame.get(occurrence_id) != review.end_frame:
                raise ContractError(f"end_frame differs for {occurrence_id}")
            if observed_missing[occurrence_id] != review.expected_missing_frames:
                raise ContractError(f"missing-frame count differs for {occurrence_id}")
            if observed_max_gap[occurrence_id] != review.expected_max_gap:
                raise ContractError(f"maximum internal gap differs for {occurrence_id}")

        if expected_rows is not None and total_rows != expected_rows:
            raise ContractError(f"source row count differs: {total_rows} != {expected_rows}")
        if expected_valid_rows is not None and valid_rows != expected_valid_rows:
            raise ContractError(
                f"source valid-row count differs: {valid_rows} != {expected_valid_rows}"
            )
        if (
            expected_original_invalid_rows is not None
            and original_invalid_rows != expected_original_invalid_rows
        ):
            raise ContractError(
                "source original-invalid count differs: "
                f"{original_invalid_rows} != {expected_original_invalid_rows}"
            )
        if expected_action_detections is not None and dict(action_detections) != dict(
            expected_action_detections
        ):
            raise ContractError(
                "review detection coverage differs: "
                f"{dict(action_detections)} != {dict(expected_action_detections)}"
            )
        if valid_rows != sum(action_detections.values()):
            raise ContractError("valid source rows do not map bijectively to reviews")

        if output_handle is not None:
            output_handle.flush()
            os.fsync(output_handle.fileno())
        completed = True
    finally:
        if output_handle is not None:
            output_handle.close()
        if created_output and not completed and output_path is not None:
            output_path.unlink(missing_ok=True)

    return (
        TransformStats(
            total_rows=total_rows,
            valid_rows=valid_rows,
            original_invalid_rows=original_invalid_rows,
            detections_by_action=dict(sorted(action_detections.items())),
            output_rows_by_gid=dict(sorted(output_by_gid.items())),
        ),
        render_index,
    )


def global_id_color(global_track_id: int) -> tuple[int, int, int]:
    if isinstance(global_track_id, bool) or not 0 <= global_track_id < 62:
        raise ContractError("global_track_id for rendering must be in [0, 61]")
    hue = (0.071 + global_track_id * 0.6180339887498949) % 1.0
    red, green, blue = colorsys.hsv_to_rgb(hue, 0.82, 0.96)
    return (
        int(round(blue * 255.0)),
        int(round(green * 255.0)),
        int(round(red * 255.0)),
    )


def _pixel_box(detection: RenderDetection) -> tuple[tuple[int, int], tuple[int, int]]:
    return (
        (int(math.floor(detection.x1)), int(math.floor(detection.y1))),
        (
            min(RAW_WIDTH - 1, int(math.ceil(detection.x2)) - 1),
            min(RAW_HEIGHT - 1, int(math.ceil(detection.y2)) - 1),
        ),
    )


def _draw_label(
    canvas: np.ndarray,
    *,
    label: str,
    top_left: tuple[int, int],
    color: tuple[int, int, int],
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.8
    text_thickness = 4
    outline_thickness = 8
    (text_width, text_height), baseline = cv2.getTextSize(
        label, font, font_scale, text_thickness
    )
    x = min(max(top_left[0], 0), RAW_WIDTH - text_width - 1)
    above_y = top_left[1] - 12
    y = (
        above_y
        if above_y - text_height - baseline >= 0
        else min(RAW_HEIGHT - baseline - 1, top_left[1] + text_height + 12)
    )
    origin = (x, y)
    cv2.putText(
        canvas,
        label,
        origin,
        font,
        font_scale,
        (0, 0, 0),
        thickness=outline_thickness,
        lineType=cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        label,
        origin,
        font,
        font_scale,
        color,
        thickness=text_thickness,
        lineType=cv2.LINE_AA,
    )


def _draw_dashed_line(
    canvas: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
    *,
    color: tuple[int, int, int] = (160, 160, 160),
    thickness: int = 8,
    dash_length: int = 32,
    gap_length: int = 20,
) -> None:
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length = math.hypot(dx, dy)
    if length == 0:
        return
    effective_dash = float(dash_length)
    effective_gap = float(gap_length)
    if length < dash_length + gap_length:
        # Preserve a visible gap even on short bbox edges instead of silently
        # degenerating into a solid line.
        effective_dash = max(0.5, length * 0.58)
        effective_gap = max(0.5, length - effective_dash)
    ux, uy = dx / length, dy / length
    offset = 0.0
    while offset <= length:
        segment_end = min(length, offset + effective_dash)
        first = (round(start[0] + ux * offset), round(start[1] + uy * offset))
        last = (
            round(start[0] + ux * segment_end),
            round(start[1] + uy * segment_end),
        )
        cv2.line(canvas, first, last, color, thickness=thickness, lineType=cv2.LINE_8)
        offset += effective_dash + effective_gap


def _draw_dashed_box(
    canvas: np.ndarray,
    top_left: tuple[int, int],
    bottom_right: tuple[int, int],
) -> None:
    left, top = top_left
    right, bottom = bottom_right
    _draw_dashed_line(canvas, (left, top), (right, top))
    _draw_dashed_line(canvas, (right, top), (right, bottom))
    _draw_dashed_line(canvas, (right, bottom), (left, bottom))
    _draw_dashed_line(canvas, (left, bottom), (left, top))


def render_corrected_frame(
    raw_frame: np.ndarray, detections: Sequence[RenderDetection]
) -> np.ndarray:
    if not isinstance(raw_frame, np.ndarray) or raw_frame.dtype != np.uint8:
        raise ContractError("raw frame must be a uint8 NumPy array")
    if raw_frame.shape != (RAW_HEIGHT, RAW_WIDTH, 3):
        raise ContractError(
            f"raw frame must have shape {(RAW_HEIGHT, RAW_WIDTH, 3)}, "
            f"got {raw_frame.shape}"
        )
    seen: set[int] = set()
    ordered = sorted(
        detections,
        key=lambda item: (
            item.reviewed_invalid,
            -1 if item.global_track_id is None else item.global_track_id,
            item.det_id,
        ),
    )
    annotated = raw_frame.copy()
    for item in ordered:
        if item.det_id in seen:
            raise ContractError(f"duplicate render det_id {item.det_id}")
        seen.add(item.det_id)
        if not (
            0.0 <= item.x1 < item.x2 <= RAW_WIDTH
            and 0.0 <= item.y1 < item.y2 <= RAW_HEIGHT
        ):
            raise ContractError(f"render bbox outside raw frame for det_id={item.det_id}")
        top_left, bottom_right = _pixel_box(item)
        if item.reviewed_invalid:
            if item.review_action != "invalid_multiple_cows":
                raise ContractError("reviewed-invalid render action is inconsistent")
            if item.global_track_id is not None or item.display_global_id is not None:
                raise ContractError("reviewed-invalid render row must not retain an ID")
            _draw_dashed_box(annotated, top_left, bottom_right)
            continue
        if item.review_action not in {"accept", "update_id"}:
            raise ContractError("normal render action is inconsistent")
        if item.global_track_id is None or item.display_global_id is None:
            raise ContractError("normal render row lacks corrected global identity")
        expected_display = f"G{item.global_track_id + 1:04d}"
        if item.display_global_id != expected_display:
            raise ContractError("corrected render ID fields are inconsistent")
        color = global_id_color(item.global_track_id)
        cv2.rectangle(
            annotated,
            top_left,
            bottom_right,
            color,
            thickness=8,
            lineType=cv2.LINE_8,
        )
        _draw_label(
            annotated,
            label=item.display_global_id,
            top_left=top_left,
            color=color,
        )
    return cv2.resize(
        annotated,
        (OUTPUT_WIDTH, OUTPUT_HEIGHT),
        interpolation=cv2.INTER_AREA,
    )


def require_encoder_environment() -> tuple[str, str]:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
        raise ContractError(
            "CUDA_VISIBLE_DEVICES must be exactly '1'; CPU/OpenCV encoding is forbidden"
        )
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        raise ContractError("ffmpeg and ffprobe are required")
    return ffmpeg, ffprobe


def preflight_nvenc(ffmpeg_binary: str) -> None:
    command = [
        ffmpeg_binary,
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-f",
        "rawvideo",
        "-pixel_format",
        "bgr24",
        "-video_size",
        f"{OUTPUT_WIDTH}x{OUTPUT_HEIGHT}",
        "-framerate",
        f"{FPS.numerator}/{FPS.denominator}",
        "-i",
        "pipe:0",
        "-frames:v",
        "1",
        "-c:v",
        "h264_nvenc",
        "-gpu",
        "0",
        "-preset",
        "p4",
        "-tune",
        "hq",
        "-rc:v",
        "vbr",
        "-cq:v",
        "21",
        "-b:v",
        "0",
        "-pix_fmt",
        "yuv420p",
        "-f",
        "null",
        "-",
    ]
    try:
        completed = subprocess.run(
            command,
            input=bytes(OUTPUT_WIDTH * OUTPUT_HEIGHT * 3),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            check=False,
            shell=False,
        )
    except OSError as exc:
        raise ContractError(f"cannot start GPU 1 NVENC preflight: {exc}") from exc
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise ContractError(
            f"GPU 1 NVENC preflight failed ({completed.returncode}): "
            f"{detail or 'no FFmpeg diagnostics'}"
        )
    log("[preflight] GPU 1 h264_nvenc passed")


def render_video(
    *,
    clip_id: str,
    raw_video_path: Path,
    detections_by_frame: Sequence[Sequence[RenderDetection]],
    output_path: Path,
    start_frame: int,
    frame_count: int,
    ffmpeg_binary: str,
    ffprobe_binary: str,
    required_actions: frozenset[str] = frozenset(),
) -> None:
    total_frames = CLIP_FRAME_COUNTS[clip_id]
    if not 0 <= start_frame < total_frames:
        raise ContractError(f"sample start frame is outside {clip_id}")
    if frame_count <= 0 or start_frame + frame_count > total_frames:
        raise ContractError(f"render frame range is outside {clip_id}")
    if len(detections_by_frame) != total_frames:
        raise ContractError(f"render index frame count differs for {clip_id}")
    if output_path.exists() or output_path.is_symlink():
        raise ContractError(f"refusing to overwrite render output: {output_path}")

    action_coverage: Counter[str] = Counter()
    for frame_detections in detections_by_frame[start_frame : start_frame + frame_count]:
        action_coverage.update(item.review_action for item in frame_detections)
    log(
        f"[render] {clip_id} source range={start_frame}:"
        f"{start_frame + frame_count - 1}, action detections={dict(sorted(action_coverage.items()))}"
    )
    missing_actions = required_actions - action_coverage.keys()
    if missing_actions:
        raise ContractError(
            f"bounded render sample lacks required review actions: {sorted(missing_actions)}"
        )

    metadata = probe_video_stream(raw_video_path, ffprobe_binary)
    if (metadata.width, metadata.height) != (RAW_WIDTH, RAW_HEIGHT):
        raise ContractError(f"raw video geometry differs for {clip_id}")
    if metadata.average_frame_rate != FPS:
        raise ContractError(f"raw video frame rate differs for {clip_id}")
    if metadata.num_frames != total_frames:
        raise ContractError(
            f"raw video frame count differs for {clip_id}: "
            f"{metadata.num_frames} != {total_frames}"
        )
    if (
        metadata.rotation_degrees is None
        or metadata.rotation_degrees % 360
        != RAW_ROTATION_METADATA_DEGREES % 360
    ):
        raise ContractError(
            f"raw video rotation metadata differs for {clip_id}: "
            f"{metadata.rotation_degrees} != {RAW_ROTATION_METADATA_DEGREES}"
        )

    capture = open_raw_video_capture(raw_video_path)
    last_report = time.monotonic()
    try:
        width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
        height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        reported_frames = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        reported_fps = float(capture.get(cv2.CAP_PROP_FPS))
        if (width, height) != (RAW_WIDTH, RAW_HEIGHT):
            raise ContractError(f"raw video geometry differs for {clip_id}")
        if reported_frames != total_frames:
            raise ContractError(
                f"raw video frame count differs for {clip_id}: "
                f"{reported_frames} != {total_frames}"
            )
        if not math.isclose(reported_fps, float(FPS), rel_tol=0.0, abs_tol=1e-6):
            raise ContractError(
                f"raw video FPS differs for {clip_id}: {reported_fps} != {float(FPS)}"
            )
        if start_frame and not capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame):
            raise ContractError(f"cannot seek raw video {clip_id} to frame {start_frame}")
        if int(round(capture.get(cv2.CAP_PROP_POS_FRAMES))) != start_frame:
            raise ContractError(f"raw video seek position differs for {clip_id}")

        with NvencVideoWriter(
            output_path,
            width=OUTPUT_WIDTH,
            height=OUTPUT_HEIGHT,
            fps=FPS,
            expected_frame_count=frame_count,
            ffmpeg_binary=ffmpeg_binary,
            preset="p4",
            cq=21,
            logical_gpu=0,
            pixel_format="yuv420p",
        ) as writer:
            for offset in range(frame_count):
                local_frame = start_frame + offset
                ok, raw = capture.read()
                if not ok or raw is None:
                    raise ContractError(
                        f"cannot decode raw frame {clip_id}:{local_frame}"
                    )
                if raw.shape != (RAW_HEIGHT, RAW_WIDTH, 3) or raw.dtype != np.uint8:
                    raise ContractError(
                        f"raw frame geometry/type differs at {clip_id}:{local_frame}"
                    )
                position_after = int(round(capture.get(cv2.CAP_PROP_POS_FRAMES)))
                if position_after != local_frame + 1:
                    raise ContractError(
                        f"raw frame position differs at {clip_id}:{local_frame}"
                    )
                if not hasattr(cv2, "CAP_PROP_PTS"):
                    raise ContractError("OpenCV lacks CAP_PROP_PTS for exact raw decode")
                pts_frame = int(round(capture.get(cv2.CAP_PROP_PTS)))
                if pts_frame != local_frame:
                    raise ContractError(
                        f"decoded PTS frame differs at {clip_id}:{local_frame}"
                    )
                annotated = render_corrected_frame(
                    raw, detections_by_frame[local_frame]
                )
                writer.write(np.ascontiguousarray(annotated))
                now = time.monotonic()
                if now - last_report >= PROGRESS_INTERVAL_SEC:
                    log(
                        f"[render] {clip_id}: {offset + 1:,}/{frame_count:,} frames "
                        f"(source frame {local_frame:,})"
                    )
                    last_report = now
    finally:
        capture.release()

    try:
        validate_qa_mp4(
            output_path,
            expected_frame_rate=FPS,
            expected_frame_count=frame_count,
            ffprobe_binary=ffprobe_binary,
        )
    except Exception:
        output_path.unlink(missing_ok=True)
        raise
    log(f"[render] validated {output_path} ({frame_count:,} frames)")


def sample_frame_count(seconds: Decimal) -> int:
    if not seconds.is_finite() or seconds <= 0 or seconds > MAX_SAMPLE_SECONDS:
        raise ContractError(f"sample duration must be in (0, {MAX_SAMPLE_SECONDS}] seconds")
    frames = int(
        (seconds * FPS.numerator / FPS.denominator).to_integral_value(
            rounding=ROUND_FLOOR
        )
    )
    if frames <= 0:
        raise ContractError("sample duration is shorter than one output frame")
    return frames


def _decimal_argument(value: str) -> Decimal:
    try:
        parsed = Decimal(value)
    except InvalidOperation as exc:
        raise argparse.ArgumentTypeError("must be a decimal number") from exc
    if not parsed.is_finite() or parsed <= 0 or parsed > MAX_SAMPLE_SECONDS:
        raise argparse.ArgumentTypeError(
            f"must be greater than 0 and at most {MAX_SAMPLE_SECONDS}"
        )
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "corrected_export",
        help="Full-run output root (default: %(default)s).",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate the complete review-to-detection bijection without writing outputs.",
    )
    parser.add_argument(
        "--sample-output",
        type=Path,
        help="Render one bounded real-video sample instead of full outputs.",
    )
    parser.add_argument(
        "--sample-clip-id",
        choices=tuple(CLIP_FRAME_COUNTS),
        default="GX040006",
    )
    parser.add_argument("--sample-start-frame", type=int, default=0)
    parser.add_argument(
        "--sample-duration-seconds", type=_decimal_argument, default=Decimal("30")
    )
    return parser


def _prepare_inputs(
    *, build_render_index: bool, output_csv: Path | None
) -> tuple[
    str,
    TransformStats,
    dict[str, list[list[RenderDetection]]] | None,
]:
    input_tokens = {
        path: file_stat_token(path)
        for path in (
            OCCURRENCES_PATH,
            REVIEWS_PATH,
            SOURCE_DETECTIONS_PATH,
            GLOBAL_SUMMARY_PATH,
        )
    }
    review_sha = file_sha256(REVIEWS_PATH)
    if review_sha != EXPECTED_REVIEWS_SHA256:
        raise ContractError(
            "occurrence_reviews.csv differs from the checked final review snapshot: "
            f"{review_sha} != {EXPECTED_REVIEWS_SHA256}"
        )
    reviews, intervals = load_reviews(OCCURRENCES_PATH, REVIEWS_PATH)
    gid_mapping = load_gid_mapping(GLOBAL_SUMMARY_PATH)
    log(
        f"[reviews] loaded={len(reviews):,}, sha256={review_sha}, "
        f"streams={len(intervals):,}"
    )
    stats, render_index = transform_detections(
        SOURCE_DETECTIONS_PATH,
        intervals,
        reviews,
        gid_mapping,
        output_path=output_csv,
        build_render_index=build_render_index,
    )
    if file_sha256(REVIEWS_PATH) != review_sha:
        raise ContractError("occurrence_reviews.csv changed while building the corrected index")
    require_unchanged_files(input_tokens)
    log(
        f"[validated] rows={stats.total_rows:,}, valid={stats.valid_rows:,}, "
        f"original_invalid={stats.original_invalid_rows:,}, "
        f"actions={dict(stats.detections_by_action)}"
    )
    return review_sha, stats, render_index


def run_validate_only() -> None:
    _prepare_inputs(build_render_index=False, output_csv=None)
    log("[done] complete review-to-detection validation passed; no outputs written")


def run_sample(args: argparse.Namespace) -> None:
    output = args.sample_output.resolve()
    if output.suffix.lower() != ".mp4":
        raise ContractError("--sample-output must end with .mp4")
    if output.exists() or output.is_symlink():
        raise ContractError(f"refusing to overwrite sample output: {output}")
    raw_token = file_stat_token(RAW_VIDEO_PATHS[args.sample_clip_id])
    review_sha, _, render_index = _prepare_inputs(
        build_render_index=True, output_csv=None
    )
    assert render_index is not None
    frame_count = sample_frame_count(args.sample_duration_seconds)
    clip_id = args.sample_clip_id
    if args.sample_start_frame < 0:
        raise ContractError("--sample-start-frame must be non-negative")
    if args.sample_start_frame + frame_count > CLIP_FRAME_COUNTS[clip_id]:
        raise ContractError("sample range exceeds the selected raw video")
    ffmpeg, ffprobe = require_encoder_environment()
    output.parent.mkdir(parents=True, exist_ok=True)
    render_video(
        clip_id=clip_id,
        raw_video_path=RAW_VIDEO_PATHS[clip_id],
        detections_by_frame=render_index[clip_id],
        output_path=output,
        start_frame=args.sample_start_frame,
        frame_count=frame_count,
        ffmpeg_binary=ffmpeg,
        ffprobe_binary=ffprobe,
        required_actions=frozenset(EXPECTED_ACTION_OCCURRENCES),
    )
    if file_sha256(REVIEWS_PATH) != review_sha:
        output.unlink(missing_ok=True)
        raise ContractError("occurrence_reviews.csv changed during sample rendering")
    if file_stat_token(RAW_VIDEO_PATHS[clip_id]) != raw_token:
        output.unlink(missing_ok=True)
        raise ContractError(f"raw video changed during sample rendering: {clip_id}")
    log(f"[done] bounded render sample={output}")


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def run_full(output_dir: Path) -> None:
    output_dir = output_dir.resolve()
    if output_dir == ROOT or not output_dir.is_relative_to(ROOT):
        raise ContractError("--output-dir must be a new directory inside the QC workspace")
    if output_dir.exists() or output_dir.is_symlink():
        raise ContractError(f"refusing to overwrite output directory: {output_dir}")
    final_csv = output_dir / "detections_with_global_id.csv"
    final_videos = {
        clip_id: output_dir / "qa/videos" / f"{clip_id}_tracked.mp4"
        for clip_id in CLIP_FRAME_COUNTS
    }
    fixed_input_tokens = {
        path: file_stat_token(path)
        for path in (
            OCCURRENCES_PATH,
            REVIEWS_PATH,
            SOURCE_DETECTIONS_PATH,
            GLOBAL_SUMMARY_PATH,
            *RAW_VIDEO_PATHS.values(),
        )
    }
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.staging.", dir=output_dir.parent)
    )
    committed = False
    try:
        staging_csv = staging / "detections_with_global_id.csv"
        review_sha, _, render_index = _prepare_inputs(
            build_render_index=True, output_csv=staging_csv
        )
        assert render_index is not None
        ffmpeg, ffprobe = require_encoder_environment()
        staging_videos = {
            clip_id: staging / "qa/videos" / f"{clip_id}_tracked.mp4"
            for clip_id in CLIP_FRAME_COUNTS
        }
        for clip_id in CLIP_FRAME_COUNTS:
            render_video(
                clip_id=clip_id,
                raw_video_path=RAW_VIDEO_PATHS[clip_id],
                detections_by_frame=render_index[clip_id],
                output_path=staging_videos[clip_id],
                start_frame=0,
                frame_count=CLIP_FRAME_COUNTS[clip_id],
                ffmpeg_binary=ffmpeg,
                ffprobe_binary=ffprobe,
            )
        if file_sha256(REVIEWS_PATH) != review_sha:
            raise ContractError("occurrence_reviews.csv changed during full rendering")

        expected_stage_files = {
            Path("detections_with_global_id.csv"),
            Path("qa/videos/GX040006_tracked.mp4"),
            Path("qa/videos/GX050006_tracked.mp4"),
        }
        actual_stage_files = {
            path.relative_to(staging) for path in staging.rglob("*") if path.is_file()
        }
        if actual_stage_files != expected_stage_files:
            raise ContractError(
                "staging artifact set differs from the required CSV plus two MP4s"
            )
        for path in (staging_csv, *staging_videos.values()):
            _fsync_file(path)
        _fsync_directory(staging / "qa/videos")
        _fsync_directory(staging / "qa")
        _fsync_directory(staging)
        if file_sha256(REVIEWS_PATH) != review_sha:
            raise ContractError("occurrence_reviews.csv changed before final commit")
        require_unchanged_files(fixed_input_tokens)
        if output_dir.exists() or output_dir.is_symlink():
            raise ContractError(f"output directory appeared during rendering: {output_dir}")
        os.replace(staging, output_dir)
        committed = True
        log(f"[commit] atomically published {output_dir}")
        try:
            _fsync_directory(output_dir.parent)
        except OSError as exc:
            raise ContractError(
                f"outputs were published, but parent-directory fsync failed: {exc}"
            ) from exc
        log(f"[done] corrected CSV={final_csv}")
        for path in final_videos.values():
            log(f"[done] corrected video={path}")
    finally:
        if staging.exists():
            shutil.rmtree(staging)
        if not committed:
            log("[abort] no final corrected outputs were published")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.validate_only and args.sample_output is not None:
        parser.error("--validate-only cannot be combined with --sample-output")
    try:
        if args.sample_output is not None:
            early_sample = args.sample_output.resolve()
            if early_sample.suffix.lower() != ".mp4":
                raise ContractError("--sample-output must end with .mp4")
            if early_sample.exists() or early_sample.is_symlink():
                raise ContractError(f"refusing to overwrite sample output: {early_sample}")
        elif not args.validate_only:
            early_output_dir = args.output_dir.resolve()
            if early_output_dir == ROOT or not early_output_dir.is_relative_to(ROOT):
                raise ContractError(
                    "--output-dir must be a new directory inside the QC workspace"
                )
            if early_output_dir.exists() or early_output_dir.is_symlink():
                raise ContractError(
                    f"refusing to overwrite output directory: {early_output_dir}"
                )

        try:
            export_lock = ReviewFileLock(REVIEW_LOCK_PATH, blocking=False)
            with export_lock:
                if args.validate_only:
                    raw_hash_clips: tuple[str, ...] = ()
                elif args.sample_output is not None:
                    raw_hash_clips = (args.sample_clip_id,)
                    ffmpeg, _ = require_encoder_environment()
                    preflight_nvenc(ffmpeg)
                else:
                    raw_hash_clips = tuple(CLIP_FRAME_COUNTS)
                    ffmpeg, _ = require_encoder_environment()
                    preflight_nvenc(ffmpeg)
                validate_fixed_inputs(hash_raw_clips=raw_hash_clips)
                if args.validate_only:
                    run_validate_only()
                elif args.sample_output is not None:
                    run_sample(args)
                else:
                    run_full(args.output_dir)
        except ReviewLockUnavailable as exc:
            raise ContractError(
                "cannot freeze occurrence_reviews.csv because another review/export "
                f"operation holds the lock: {exc}"
            ) from exc
        return 0
    except (ContractError, OSError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
