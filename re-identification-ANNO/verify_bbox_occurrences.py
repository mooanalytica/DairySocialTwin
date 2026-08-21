from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from decimal import Decimal
from pathlib import Path


SOURCE = Path(
    "/home/hyw/re-identification-S6/work/"
    "dairy_farm_1_gopro1_20250505/06_export/"
    "detections_with_global_id.csv"
)
SEGMENTS = Path("/home/hyw/re-identification-QC/occurrence_segments.csv")
SUMMARY = Path("/home/hyw/re-identification-QC/bbox_occurrence_summary.json")

MAX_MISSING_FRAMES = 30
FPS_NUMERATOR = 30_000
FPS_DENOMINATOR = 1_001
EXPECTED_SOURCE_ROWS = 746_279
EXPECTED_VALID_ROWS = 745_915
EXPECTED_INVALID_ROWS = 364

SEGMENT_COLUMNS = (
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


def fail(message: str) -> None:
    raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def frame_time(frame: int) -> str:
    value = Decimal(frame * FPS_DENOMINATOR) / Decimal(FPS_NUMERATOR)
    return f"{value:.9f}"


def legacy_sort_key(value: str) -> tuple[int, int | str]:
    return (0, int(value)) if value.isdecimal() else (1, value)


def validate_segments(summary: dict) -> dict[str, object]:
    expected_result = summary["result"]
    by_clip: Counter[str] = Counter()
    by_gid: Counter[str] = Counter()
    one_frame_count = 0
    total_missing = 0
    row_count = 0
    previous_sort_key: tuple | None = None

    with SEGMENTS.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != SEGMENT_COLUMNS:
            fail("segment header mismatch")
        for row in reader:
            row_count += 1
            expected_id = f"O{row_count:06d}"
            if row["occurrence_id"] != expected_id:
                fail(f"non-sequential occurrence ID at row {row_count}")

            clip_order = int(row["clip_order"])
            start = int(row["start_frame"])
            end = int(row["end_frame"])
            detections = int(row["num_valid_detections"])
            missing = int(row["num_missing_frames"])
            max_gap = int(row["max_internal_gap_missing_frames"])
            if start < 0 or end < start:
                fail(f"invalid frame interval at {expected_id}")
            span_frames = end - start + 1
            if detections <= 0 or missing < 0 or detections + missing != span_frames:
                fail(f"span accounting mismatch at {expected_id}")
            if not 0 <= max_gap <= MAX_MISSING_FRAMES:
                fail(f"internal gap exceeds policy at {expected_id}")
            if row["start_time_sec"] != frame_time(start):
                fail(f"start time mismatch at {expected_id}")
            if row["end_time_sec_inclusive"] != frame_time(end):
                fail(f"inclusive end time mismatch at {expected_id}")
            if row["end_time_sec_exclusive"] != frame_time(end + 1):
                fail(f"exclusive end time mismatch at {expected_id}")
            if row["span_duration_sec"] != frame_time(span_frames):
                fail(f"duration mismatch at {expected_id}")

            sort_key = (
                clip_order,
                start,
                end,
                row["display_global_id"],
                legacy_sort_key(row["legacy_track_id"]),
            )
            if previous_sort_key is not None and sort_key < previous_sort_key:
                fail(f"segment sort order mismatch at {expected_id}")
            previous_sort_key = sort_key

            by_clip[row["clip_id"]] += 1
            by_gid[row["display_global_id"]] += 1
            one_frame_count += int(start == end)
            total_missing += missing

    observed = {
        "total_occurrence_count": row_count,
        "occurrences_by_clip": dict(sorted(by_clip.items())),
        "occurrences_by_global_id": dict(sorted(by_gid.items())),
        "one_frame_occurrences": one_frame_count,
        "total_internal_missing_frames": total_missing,
    }
    if observed != expected_result:
        fail("segment aggregates differ from summary JSON")
    return observed


def independently_recount_source() -> dict[str, object]:
    last_by_stream: dict[tuple[str, str], tuple[str, int]] = {}
    starts_by_clip: Counter[str] = Counter()
    starts_by_gid: Counter[str] = Counter()
    total_rows = 0
    valid_rows = 0
    invalid_rows = 0

    with SOURCE.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            total_rows += 1
            if row["valid"] == "false":
                invalid_rows += 1
                continue
            if row["valid"] != "true":
                fail(f"unexpected valid value on source line {reader.line_num}")
            valid_rows += 1
            clip_id = row["clip_id"]
            legacy_track_id = row["legacy_track_id"]
            gid = row["display_global_id"]
            frame = int(row["local_frame"])
            key = (clip_id, legacy_track_id)
            previous = last_by_stream.get(key)
            starts_new = previous is None
            if previous is not None:
                previous_gid, previous_frame = previous
                if frame <= previous_frame:
                    fail(
                        "source has duplicate/backward frame for "
                        f"{clip_id}, legacy_track_id={legacy_track_id}"
                    )
                missing = frame - previous_frame - 1
                starts_new = gid != previous_gid or missing > MAX_MISSING_FRAMES
            if starts_new:
                starts_by_clip[clip_id] += 1
                starts_by_gid[gid] += 1
            last_by_stream[key] = (gid, frame)

    if (total_rows, valid_rows, invalid_rows) != (
        EXPECTED_SOURCE_ROWS,
        EXPECTED_VALID_ROWS,
        EXPECTED_INVALID_ROWS,
    ):
        fail("independent source row counts differ from contract")
    return {
        "total_occurrence_count": sum(starts_by_clip.values()),
        "occurrences_by_clip": dict(sorted(starts_by_clip.items())),
        "occurrences_by_global_id": dict(sorted(starts_by_gid.items())),
    }


def main() -> None:
    with SUMMARY.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)

    observed_segments = validate_segments(summary)
    independently_counted = independently_recount_source()
    for field in (
        "total_occurrence_count",
        "occurrences_by_clip",
        "occurrences_by_global_id",
    ):
        if independently_counted[field] != observed_segments[field]:
            fail(f"independent recount differs for {field}")

    if sha256(SOURCE) != summary["source"]["sha256"]:
        fail("source SHA-256 mismatch")
    if sha256(SEGMENTS) != summary["outputs"]["segments_csv_sha256"]:
        fail("segments SHA-256 mismatch")

    print(
        "[verified] "
        f"source_rows={EXPECTED_SOURCE_ROWS:,}, "
        f"valid={EXPECTED_VALID_ROWS:,}, "
        f"invalid={EXPECTED_INVALID_ROWS:,}",
        flush=True,
    )
    print(
        f"[verified] total_occurrence_count="
        f"{observed_segments['total_occurrence_count']:,}",
        flush=True,
    )
    print("[verified] segment intervals, aggregates, ordering, times, and hashes", flush=True)


if __name__ == "__main__":
    main()
