#!/usr/bin/env python3
"""Count same-frame G-ID collisions directly from the current review decisions."""

from __future__ import annotations

import csv
import hashlib
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import count_corrected_frame_gid_duplicates as duplicate_counter
import export_corrected_results as exporter
from review_lock import ReviewFileLock, ReviewLockUnavailable


ROOT = Path(__file__).resolve().parent
PROGRESS_INTERVAL_SEC = 10.0


class ContractError(RuntimeError):
    """The current reviews or fixed source violate the counting contract."""


def log(message: str) -> None:
    print(message, flush=True)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_int(value: str, label: str, row_number: int) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ContractError(
            f"row {row_number}: {label} must be an integer, got {value!r}"
        ) from exc
    if str(parsed) != value:
        raise ContractError(
            f"row {row_number}: {label} must be canonical, got {value!r}"
        )
    return parsed


def scan_reviewed_assignments(
    source_path: Path,
    intervals: Mapping[tuple[str, str], exporter.TrackIntervals],
    *,
    expected_rows: int,
    expected_assigned_rows: int,
    expected_reviewed_invalid_rows: int,
    expected_original_invalid_rows: int,
) -> duplicate_counter.ScanStats:
    total_rows = 0
    assigned_rows = 0
    reviewed_invalid_rows = 0
    original_invalid_rows = 0
    seen_det_ids: set[int] = set()
    previous_sort_key: tuple[int, int] | None = None
    previous_global_frame: int | None = None
    current_frame: tuple[str, int, str, int, int] | None = None
    current_by_gid: dict[
        str, list[duplicate_counter.DetectionReference]
    ] = {}
    duplicate_groups: list[duplicate_counter.DuplicateGroup] = []
    last_report = time.monotonic()

    def flush_frame() -> None:
        if current_frame is None:
            return
        sequence_id, clip_order, clip_id, local_frame, global_frame = current_frame
        for gid, detections in sorted(current_by_gid.items()):
            if len(detections) > 1:
                duplicate_groups.append(
                    duplicate_counter.DuplicateGroup(
                        sequence_id=sequence_id,
                        clip_order=clip_order,
                        clip_id=clip_id,
                        local_frame=local_frame,
                        global_frame=global_frame,
                        display_global_id=gid,
                        detections=tuple(
                            sorted(detections, key=lambda item: item.det_id)
                        ),
                    )
                )
        current_by_gid.clear()

    with source_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != exporter.DETECTION_FIELDS:
            raise ContractError("source detection CSV header differs from fixed schema")
        for row in reader:
            total_rows += 1
            if set(row) != set(exporter.DETECTION_FIELDS) or any(
                value is None for value in row.values()
            ):
                raise ContractError(f"malformed source detection row {total_rows}")
            clip_id = row["clip_id"]
            if clip_id not in exporter.CLIP_ORDERS:
                raise ContractError(f"row {total_rows}: unknown clip {clip_id!r}")
            clip_order = canonical_int(row["clip_order"], "clip_order", total_rows)
            csv_row_index = canonical_int(
                row["csv_row_index"], "csv_row_index", total_rows
            )
            sort_key = (clip_order, csv_row_index)
            if clip_order != exporter.CLIP_ORDERS[clip_id] or (
                previous_sort_key is not None and sort_key <= previous_sort_key
            ):
                raise ContractError(f"row {total_rows}: source ordering changed")
            previous_sort_key = sort_key

            local_frame = canonical_int(row["local_frame"], "local_frame", total_rows)
            global_frame = canonical_int(
                row["global_frame"], "global_frame", total_rows
            )
            if global_frame != exporter.CLIP_GLOBAL_FRAME_STARTS[clip_id] + local_frame:
                raise ContractError(f"row {total_rows}: local/global frame mismatch")
            if previous_global_frame is not None and global_frame < previous_global_frame:
                raise ContractError(f"row {total_rows}: global frame moved backwards")
            previous_global_frame = global_frame

            frame = (
                row["sequence_id"],
                clip_order,
                clip_id,
                local_frame,
                global_frame,
            )
            if current_frame is None:
                current_frame = frame
            elif frame != current_frame:
                flush_frame()
                current_frame = frame

            det_id = canonical_int(row["det_id"], "det_id", total_rows)
            if det_id in seen_det_ids:
                raise ContractError(f"row {total_rows}: duplicate det_id {det_id}")
            seen_det_ids.add(det_id)
            if row["valid"] == "false":
                original_invalid_rows += 1
            elif row["valid"] == "true":
                occurrence = exporter.lookup_occurrence(
                    intervals,
                    clip_id=clip_id,
                    legacy_track_id=row["legacy_track_id"],
                    local_frame=local_frame,
                )
                if row["display_global_id"] != occurrence.original_gid:
                    raise ContractError(
                        f"row {total_rows}: source G-ID differs from "
                        f"{occurrence.occurrence_id}"
                    )
                if occurrence.action == "invalid_multiple_cows":
                    reviewed_invalid_rows += 1
                else:
                    stable_id = canonical_int(
                        row["stable_id"], "stable_id", total_rows
                    )
                    current_by_gid.setdefault(occurrence.reviewed_gid, []).append(
                        duplicate_counter.DetectionReference(
                            det_id=det_id,
                            legacy_track_id=row["legacy_track_id"],
                            stable_id=stable_id,
                        )
                    )
                    assigned_rows += 1
            else:
                raise ContractError(
                    f"row {total_rows}: valid must be true or false"
                )

            now = time.monotonic()
            if now - last_report >= PROGRESS_INTERVAL_SEC:
                log(
                    f"[count] rows={total_rows:,}, assigned={assigned_rows:,}, "
                    f"duplicate_groups_closed={len(duplicate_groups):,}"
                )
                last_report = now

    flush_frame()
    observed = (
        total_rows,
        assigned_rows,
        reviewed_invalid_rows,
        original_invalid_rows,
    )
    expected = (
        expected_rows,
        expected_assigned_rows,
        expected_reviewed_invalid_rows,
        expected_original_invalid_rows,
    )
    if observed != expected:
        raise ContractError(
            f"reviewed scan totals changed: observed={observed}, expected={expected}"
        )
    return duplicate_counter.ScanStats(
        total_rows=total_rows,
        assigned_gid_rows=assigned_rows,
        reviewed_invalid_rows=reviewed_invalid_rows,
        original_invalid_rows=original_invalid_rows,
        clips=tuple(
            sorted(
                (clip_order, clip_id)
                for clip_id, clip_order in exporter.CLIP_ORDERS.items()
            )
        ),
        duplicate_groups=tuple(duplicate_groups),
    )


def print_occurrence_sets(
    stats: duplicate_counter.ScanStats,
    intervals: Mapping[tuple[str, str], exporter.TrackIntervals],
) -> None:
    groups: Counter[tuple[str, tuple[str, ...]]] = Counter()
    first_frame: dict[tuple[str, tuple[str, ...]], int] = {}
    last_frame: dict[tuple[str, tuple[str, ...]], int] = {}
    for group in stats.duplicate_groups:
        occurrence_ids = tuple(
            sorted(
                exporter.lookup_occurrence(
                    intervals,
                    clip_id=group.clip_id,
                    legacy_track_id=detection.legacy_track_id,
                    local_frame=group.local_frame,
                ).occurrence_id
                for detection in group.detections
            )
        )
        key = (group.display_global_id, occurrence_ids)
        groups[key] += 1
        first_frame[key] = min(first_frame.get(key, group.local_frame), group.local_frame)
        last_frame[key] = max(last_frame.get(key, group.local_frame), group.local_frame)
    for (gid, occurrence_ids), count in groups.most_common():
        log(
            f"[occurrence-set] gid={gid}, occurrences={','.join(occurrence_ids)}, "
            f"duplicate_groups={count:,}, first_frame={first_frame[(gid, occurrence_ids)]}, "
            f"last_frame={last_frame[(gid, occurrence_ids)]}"
        )


def main(argv: Sequence[str] | None = None) -> int:
    if argv:
        raise ContractError("this dataset-specific checker accepts no arguments")
    try:
        lock = ReviewFileLock(exporter.REVIEW_LOCK_PATH, blocking=False)
        with lock:
            exporter.validate_fixed_inputs(hash_raw_clips=())
            review_sha256 = file_sha256(exporter.REVIEWS_PATH)
            source_token = exporter.file_stat_token(exporter.SOURCE_DETECTIONS_PATH)
            reviews, intervals = exporter.load_reviews(
                exporter.OCCURRENCES_PATH,
                exporter.REVIEWS_PATH,
                expected_actions=None,
            )
            gid_mapping = exporter.load_gid_mapping(exporter.GLOBAL_SUMMARY_PATH)
            transform_stats, _ = exporter.transform_detections(
                exporter.SOURCE_DETECTIONS_PATH,
                intervals,
                reviews,
                gid_mapping,
                output_path=None,
                build_render_index=False,
                expected_action_detections=None,
            )
            reviewed_invalid = transform_stats.output_rows_by_gid.get("-1", 0)
            assigned = sum(
                count
                for gid, count in transform_stats.output_rows_by_gid.items()
                if gid != "-1"
            )
            stats = scan_reviewed_assignments(
                exporter.SOURCE_DETECTIONS_PATH,
                intervals,
                expected_rows=transform_stats.total_rows,
                expected_assigned_rows=assigned,
                expected_reviewed_invalid_rows=reviewed_invalid,
                expected_original_invalid_rows=transform_stats.original_invalid_rows,
            )
            if file_sha256(exporter.REVIEWS_PATH) != review_sha256:
                raise ContractError("occurrence_reviews.csv changed during counting")
            if exporter.file_stat_token(exporter.SOURCE_DETECTIONS_PATH) != source_token:
                raise ContractError("source detection CSV changed during counting")
    except ReviewLockUnavailable as exc:
        raise ContractError(
            "review file is busy; finish the active review submission and retry"
        ) from exc

    action_counts = Counter(review.action for review in reviews)
    log(
        f"[reviews] loaded={len(reviews):,}, sha256={review_sha256}, "
        f"actions={dict(sorted(action_counts.items()))}"
    )
    duplicate_counter.print_summary(stats, show_groups=False)
    print_occurrence_sets(stats, intervals)
    log("[done] current reviews checked; no corrected CSV or video was written")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ContractError, exporter.ContractError, OSError) as exc:
        raise SystemExit(f"error: {exc}") from None
