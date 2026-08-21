from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
import time
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable


INPUT_CSV = Path(
    "/home/hyw/re-identification-S6/work/"
    "dairy_farm_1_gopro1_20250505/06_export/"
    "detections_with_global_id.csv"
)
OUTPUT_CSV = Path("/home/hyw/re-identification-QC/occurrence_segments.csv")
SUMMARY_JSON = Path("/home/hyw/re-identification-QC/bbox_occurrence_summary.json")

SEQUENCE_ID = "dairy_farm_1_gopro1_20250505"
FPS_NUMERATOR = 30_000
FPS_DENOMINATOR = 1_001
MAX_MISSING_FRAMES = 30
PROGRESS_INTERVAL_SEC = 10.0

EXPECTED_COLUMNS = (
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

EXPECTED_GLOBAL_IDS = tuple(f"G{index:04d}" for index in range(1, 63))
EXPECTED_GLOBAL_ID_SET = frozenset(EXPECTED_GLOBAL_IDS)

CLIPS = {
    "GX040006": {
        "clip_order": 0,
        "num_frames": 84_480,
        "total_rows": 332_813,
        "valid_rows": 332_754,
        "invalid_rows": 59,
    },
    "GX050006": {
        "clip_order": 1,
        "num_frames": 78_720,
        "total_rows": 413_466,
        "valid_rows": 413_161,
        "invalid_rows": 305,
    },
}

EXPECTED_TOTAL_ROWS = 746_279
EXPECTED_VALID_ROWS = 745_915
EXPECTED_INVALID_ROWS = 364

OUTPUT_COLUMNS = (
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


class ContractError(RuntimeError):
    pass


@dataclass
class ActiveOccurrence:
    clip_order: int
    clip_id: str
    display_global_id: str
    legacy_track_id: str
    start_frame: int
    end_frame: int
    num_valid_detections: int
    num_missing_frames: int
    max_internal_gap_missing_frames: int
    start_det_id: str
    end_det_id: str


@dataclass(frozen=True)
class OccurrenceSegment:
    clip_order: int
    clip_id: str
    display_global_id: str
    legacy_track_id: str
    start_frame: int
    end_frame: int
    num_valid_detections: int
    num_missing_frames: int
    max_internal_gap_missing_frames: int
    start_det_id: str
    end_det_id: str


def log(message: str) -> None:
    print(message, flush=True)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    total_bytes = path.stat().st_size
    processed_bytes = 0
    last_report = time.monotonic()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
            processed_bytes += len(chunk)
            now = time.monotonic()
            if now - last_report >= PROGRESS_INTERVAL_SEC:
                percentage = 100.0 * processed_bytes / total_bytes
                log(f"[hash] {processed_bytes:,}/{total_bytes:,} bytes ({percentage:.1f}%)")
                last_report = now
    return digest.hexdigest()


def parse_int(value: str, field: str, source_line: int) -> int:
    try:
        return int(value)
    except ValueError as exc:
        raise ContractError(
            f"line {source_line}: {field} must be an integer, got {value!r}"
        ) from exc


def seconds_for_frame(frame: int) -> str:
    seconds = Decimal(frame * FPS_DENOMINATOR) / Decimal(FPS_NUMERATOR)
    return f"{seconds:.9f}"


def segment_from_active(active: ActiveOccurrence) -> OccurrenceSegment:
    span_frames = active.end_frame - active.start_frame + 1
    expected_missing = span_frames - active.num_valid_detections
    if active.num_missing_frames != expected_missing:
        raise ContractError(
            "internal segment invariant failed: missing-frame total differs from span"
        )
    return OccurrenceSegment(
        clip_order=active.clip_order,
        clip_id=active.clip_id,
        display_global_id=active.display_global_id,
        legacy_track_id=active.legacy_track_id,
        start_frame=active.start_frame,
        end_frame=active.end_frame,
        num_valid_detections=active.num_valid_detections,
        num_missing_frames=active.num_missing_frames,
        max_internal_gap_missing_frames=active.max_internal_gap_missing_frames,
        start_det_id=active.start_det_id,
        end_det_id=active.end_det_id,
    )


def start_occurrence(
    *,
    clip_order: int,
    clip_id: str,
    display_global_id: str,
    legacy_track_id: str,
    local_frame: int,
    det_id: str,
) -> ActiveOccurrence:
    return ActiveOccurrence(
        clip_order=clip_order,
        clip_id=clip_id,
        display_global_id=display_global_id,
        legacy_track_id=legacy_track_id,
        start_frame=local_frame,
        end_frame=local_frame,
        num_valid_detections=1,
        num_missing_frames=0,
        max_internal_gap_missing_frames=0,
        start_det_id=det_id,
        end_det_id=det_id,
    )


def finalize_active(
    active_by_track: dict[tuple[str, str], ActiveOccurrence],
    segments: list[OccurrenceSegment],
) -> None:
    segments.extend(segment_from_active(active) for active in active_by_track.values())
    active_by_track.clear()


def read_segments() -> tuple[list[OccurrenceSegment], dict[str, Any]]:
    total_rows = 0
    valid_rows = 0
    invalid_rows = 0
    current_clip_order = -1
    last_csv_row_index: dict[str, int] = {}
    active_by_track: dict[tuple[str, str], ActiveOccurrence] = {}
    segments: list[OccurrenceSegment] = []
    seen_global_ids: set[str] = set()
    valid_rows_by_global_id = {global_id: 0 for global_id in EXPECTED_GLOBAL_IDS}
    counts_by_clip = {
        clip_id: {"total_rows": 0, "valid_rows": 0, "invalid_rows": 0}
        for clip_id in CLIPS
    }
    last_report = time.monotonic()

    with INPUT_CSV.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != EXPECTED_COLUMNS:
            raise ContractError("input CSV header differs from the fixed S06 contract")

        for row in reader:
            source_line = reader.line_num
            total_rows += 1
            if row["sequence_id"] != SEQUENCE_ID:
                raise ContractError(
                    f"line {source_line}: unexpected sequence_id {row['sequence_id']!r}"
                )

            clip_id = row["clip_id"]
            if clip_id not in CLIPS:
                raise ContractError(f"line {source_line}: unexpected clip_id {clip_id!r}")
            clip_spec = CLIPS[clip_id]
            clip_order = parse_int(row["clip_order"], "clip_order", source_line)
            if clip_order != clip_spec["clip_order"]:
                raise ContractError(
                    f"line {source_line}: clip_order does not match {clip_id}"
                )
            if clip_order < current_clip_order:
                raise ContractError(f"line {source_line}: clip order moved backwards")
            if clip_order > current_clip_order:
                if current_clip_order >= 0:
                    if clip_order != current_clip_order + 1:
                        raise ContractError(f"line {source_line}: clip order is not contiguous")
                    finalize_active(active_by_track, segments)
                current_clip_order = clip_order

            csv_row_index = parse_int(
                row["csv_row_index"], "csv_row_index", source_line
            )
            previous_csv_index = last_csv_row_index.get(clip_id)
            if previous_csv_index is not None and csv_row_index <= previous_csv_index:
                raise ContractError(
                    f"line {source_line}: csv_row_index is not strictly increasing in {clip_id}"
                )
            last_csv_row_index[clip_id] = csv_row_index

            local_frame = parse_int(row["local_frame"], "local_frame", source_line)
            if not 0 <= local_frame < clip_spec["num_frames"]:
                raise ContractError(
                    f"line {source_line}: local_frame is outside {clip_id}"
                )

            counts_by_clip[clip_id]["total_rows"] += 1
            valid_text = row["valid"]
            if valid_text == "false":
                invalid_rows += 1
                counts_by_clip[clip_id]["invalid_rows"] += 1
                if row["display_global_id"]:
                    raise ContractError(
                        f"line {source_line}: invalid row has a display_global_id"
                    )
                continue
            if valid_text != "true":
                raise ContractError(
                    f"line {source_line}: valid must be true or false, got {valid_text!r}"
                )

            valid_rows += 1
            counts_by_clip[clip_id]["valid_rows"] += 1
            legacy_track_id = row["legacy_track_id"]
            if not legacy_track_id:
                raise ContractError(
                    f"line {source_line}: valid row has no legacy_track_id"
                )
            display_global_id = row["display_global_id"]
            if display_global_id not in EXPECTED_GLOBAL_ID_SET:
                raise ContractError(
                    f"line {source_line}: invalid final G-ID {display_global_id!r}"
                )
            if row["id_status"] != "forced_provisional":
                raise ContractError(
                    f"line {source_line}: valid row is not forced_provisional"
                )
            if row["identity_basis"] != "operator_forced_appearance_exact_62":
                raise ContractError(
                    f"line {source_line}: unexpected identity_basis"
                )
            det_id = row["det_id"]
            if not det_id:
                raise ContractError(f"line {source_line}: valid row has no det_id")

            seen_global_ids.add(display_global_id)
            valid_rows_by_global_id[display_global_id] += 1
            stream_key = (clip_id, legacy_track_id)
            active = active_by_track.get(stream_key)
            if active is None:
                active_by_track[stream_key] = start_occurrence(
                    clip_order=clip_order,
                    clip_id=clip_id,
                    display_global_id=display_global_id,
                    legacy_track_id=legacy_track_id,
                    local_frame=local_frame,
                    det_id=det_id,
                )
            else:
                if local_frame <= active.end_frame:
                    relation = "duplicate" if local_frame == active.end_frame else "backward"
                    raise ContractError(
                        f"line {source_line}: {relation} frame for "
                        f"({clip_id}, legacy_track_id={legacy_track_id!r}, "
                        f"local_frame={local_frame})"
                    )
                missing_frames = local_frame - active.end_frame - 1
                if (
                    display_global_id != active.display_global_id
                    or missing_frames > MAX_MISSING_FRAMES
                ):
                    segments.append(segment_from_active(active))
                    active_by_track[stream_key] = start_occurrence(
                        clip_order=clip_order,
                        clip_id=clip_id,
                        display_global_id=display_global_id,
                        legacy_track_id=legacy_track_id,
                        local_frame=local_frame,
                        det_id=det_id,
                    )
                else:
                    active.end_frame = local_frame
                    active.num_valid_detections += 1
                    active.num_missing_frames += missing_frames
                    active.max_internal_gap_missing_frames = max(
                        active.max_internal_gap_missing_frames, missing_frames
                    )
                    active.end_det_id = det_id

            now = time.monotonic()
            if now - last_report >= PROGRESS_INTERVAL_SEC:
                log(
                    f"[count] rows={total_rows:,}, valid={valid_rows:,}, "
                    f"closed_segments={len(segments):,}"
                )
                last_report = now

    finalize_active(active_by_track, segments)

    if total_rows != EXPECTED_TOTAL_ROWS:
        raise ContractError(
            f"expected {EXPECTED_TOTAL_ROWS:,} rows, found {total_rows:,}"
        )
    if valid_rows != EXPECTED_VALID_ROWS:
        raise ContractError(
            f"expected {EXPECTED_VALID_ROWS:,} valid rows, found {valid_rows:,}"
        )
    if invalid_rows != EXPECTED_INVALID_ROWS:
        raise ContractError(
            f"expected {EXPECTED_INVALID_ROWS:,} invalid rows, found {invalid_rows:,}"
        )
    if seen_global_ids != EXPECTED_GLOBAL_ID_SET:
        missing = sorted(EXPECTED_GLOBAL_ID_SET - seen_global_ids)
        extra = sorted(seen_global_ids - EXPECTED_GLOBAL_ID_SET)
        raise ContractError(f"final G-ID set mismatch: missing={missing}, extra={extra}")

    for clip_id, expected in CLIPS.items():
        observed = counts_by_clip[clip_id]
        for count_name in ("total_rows", "valid_rows", "invalid_rows"):
            if observed[count_name] != expected[count_name]:
                raise ContractError(
                    f"{clip_id} {count_name}: expected {expected[count_name]:,}, "
                    f"found {observed[count_name]:,}"
                )

    segments.sort(
        key=lambda item: (
            item.clip_order,
            item.start_frame,
            item.end_frame,
            item.display_global_id,
            (0, int(item.legacy_track_id))
            if item.legacy_track_id.isdecimal()
            else (1, item.legacy_track_id),
        )
    )
    stats = {
        "total_rows": total_rows,
        "valid_rows": valid_rows,
        "invalid_rows": invalid_rows,
        "counts_by_clip": counts_by_clip,
        "valid_rows_by_global_id": valid_rows_by_global_id,
    }
    return segments, stats


def atomic_csv_write(path: Path, segments: Iterable[OccurrenceSegment]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            newline="",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            writer = csv.DictWriter(
                handle,
                fieldnames=OUTPUT_COLUMNS,
                lineterminator="\n",
                quoting=csv.QUOTE_MINIMAL,
            )
            writer.writeheader()
            for occurrence_number, segment in enumerate(segments, start=1):
                span_frames = segment.end_frame - segment.start_frame + 1
                writer.writerow(
                    {
                        "occurrence_id": f"O{occurrence_number:06d}",
                        "clip_order": segment.clip_order,
                        "clip_id": segment.clip_id,
                        "display_global_id": segment.display_global_id,
                        "legacy_track_id": segment.legacy_track_id,
                        "start_frame": segment.start_frame,
                        "end_frame": segment.end_frame,
                        "start_time_sec": seconds_for_frame(segment.start_frame),
                        "end_time_sec_inclusive": seconds_for_frame(segment.end_frame),
                        "end_time_sec_exclusive": seconds_for_frame(
                            segment.end_frame + 1
                        ),
                        "span_duration_sec": seconds_for_frame(span_frames),
                        "num_valid_detections": segment.num_valid_detections,
                        "num_missing_frames": segment.num_missing_frames,
                        "max_internal_gap_missing_frames": (
                            segment.max_internal_gap_missing_frames
                        ),
                        "start_det_id": segment.start_det_id,
                        "end_det_id": segment.end_det_id,
                    }
                )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
        temporary_name = None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def atomic_json_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
        temporary_name = None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def main() -> None:
    if not INPUT_CSV.is_file():
        raise ContractError(f"missing fixed input CSV: {INPUT_CSV}")
    if OUTPUT_CSV.exists() or SUMMARY_JSON.exists():
        raise ContractError(
            "refusing to overwrite an existing result; remove it explicitly first"
        )

    log(f"[input] {INPUT_CSV}")
    log(f"[policy] max_missing_frames={MAX_MISSING_FRAMES}, valid_only=true")
    input_stat = INPUT_CSV.stat()
    input_sha256 = file_sha256(INPUT_CSV)
    log(f"[input] sha256={input_sha256}")

    segments, stats = read_segments()
    log(
        f"[validated] rows={stats['total_rows']:,}, valid={stats['valid_rows']:,}, "
        f"invalid={stats['invalid_rows']:,}"
    )

    atomic_csv_write(OUTPUT_CSV, segments)
    output_sha256 = file_sha256(OUTPUT_CSV)

    occurrences_by_clip = {clip_id: 0 for clip_id in CLIPS}
    occurrences_by_global_id = {global_id: 0 for global_id in EXPECTED_GLOBAL_IDS}
    one_frame_occurrences = 0
    total_missing_frames = 0
    for segment in segments:
        occurrences_by_clip[segment.clip_id] += 1
        occurrences_by_global_id[segment.display_global_id] += 1
        one_frame_occurrences += int(segment.start_frame == segment.end_frame)
        total_missing_frames += segment.num_missing_frames

    summary = {
        "schema_version": "1.0",
        "source": {
            "path": str(INPUT_CSV),
            "size_bytes": input_stat.st_size,
            "sha256": input_sha256,
            "total_rows": stats["total_rows"],
            "valid_rows": stats["valid_rows"],
            "invalid_rows": stats["invalid_rows"],
            "valid_rows_by_global_id": stats["valid_rows_by_global_id"],
        },
        "policy": {
            "sequence_id": SEQUENCE_ID,
            "clips_are_independent": True,
            "clip_order": list(CLIPS),
            "stream_key": ["clip_id", "legacy_track_id"],
            "aggregate_identity": "display_global_id G0001-G0062",
            "valid_only": True,
            "max_missing_frames": MAX_MISSING_FRAMES,
            "missing_frames_definition": "current_frame - previous_frame - 1",
            "gid_change_always_splits": True,
            "legacy_track_change_always_splits": True,
            "cross_legacy_identity_inference": False,
            "frame_index_base": 0,
            "segment_interval": "closed",
            "fps_numerator": FPS_NUMERATOR,
            "fps_denominator": FPS_DENOMINATOR,
            "times_are_clip_relative": True,
        },
        "result": {
            "total_occurrence_count": len(segments),
            "occurrences_by_clip": occurrences_by_clip,
            "occurrences_by_global_id": occurrences_by_global_id,
            "one_frame_occurrences": one_frame_occurrences,
            "total_internal_missing_frames": total_missing_frames,
        },
        "outputs": {
            "segments_csv": str(OUTPUT_CSV),
            "segments_csv_sha256": output_sha256,
        },
    }
    atomic_json_write(SUMMARY_JSON, summary)

    log(f"[done] total_occurrence_count={len(segments):,}")
    log(f"[done] segments={OUTPUT_CSV}")
    log(f"[done] summary={SUMMARY_JSON}")


if __name__ == "__main__":
    main()
