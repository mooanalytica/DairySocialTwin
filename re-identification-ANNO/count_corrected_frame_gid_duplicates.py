"""Count corrected frames that contain the same assigned global ID more than once."""

from __future__ import annotations

import argparse
import csv
import re
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT_CSV = ROOT / "corrected_export" / "detections_with_global_id.csv"
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
GID_PATTERN = re.compile(r"G([0-9]{4})\Z")
EXPECTED_SEQUENCE_ID = "dairy_farm_1_gopro1_20250505"
EXPECTED_TOTAL_ROWS = 746_279
EXPECTED_ASSIGNED_GID_ROWS = 576_042
EXPECTED_REVIEWED_INVALID_ROWS = 169_873
EXPECTED_ORIGINAL_INVALID_ROWS = 364


class ContractError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class DetectionReference:
    det_id: int
    legacy_track_id: str
    stable_id: int


@dataclass(frozen=True, slots=True)
class DuplicateGroup:
    sequence_id: str
    clip_order: int
    clip_id: str
    local_frame: int
    global_frame: int
    display_global_id: str
    detections: tuple[DetectionReference, ...]

    @property
    def detection_count(self) -> int:
        return len(self.detections)

    @property
    def excess_detection_count(self) -> int:
        return self.detection_count - 1


@dataclass(frozen=True, slots=True)
class ScanStats:
    total_rows: int
    assigned_gid_rows: int
    reviewed_invalid_rows: int
    original_invalid_rows: int
    clips: tuple[tuple[int, str], ...]
    duplicate_groups: tuple[DuplicateGroup, ...]

    @property
    def duplicate_frames(self) -> frozenset[tuple[str, str, int]]:
        return frozenset(
            (group.sequence_id, group.clip_id, group.local_frame)
            for group in self.duplicate_groups
        )


@dataclass(frozen=True, slots=True)
class ClipSpec:
    clip_order: int
    frame_count: int
    global_frame_start: int
    total_rows: int
    assigned_gid_rows: int
    reviewed_invalid_rows: int
    original_invalid_rows: int


CLIP_SPECS = {
    "GX040006": ClipSpec(
        clip_order=0,
        frame_count=84_480,
        global_frame_start=0,
        total_rows=332_813,
        assigned_gid_rows=242_660,
        reviewed_invalid_rows=90_094,
        original_invalid_rows=59,
    ),
    "GX050006": ClipSpec(
        clip_order=1,
        frame_count=78_720,
        global_frame_start=84_480,
        total_rows=413_466,
        assigned_gid_rows=333_382,
        reviewed_invalid_rows=79_779,
        original_invalid_rows=305,
    ),
}


def log(message: str) -> None:
    print(message, flush=True)


def canonical_int(
    value: str, field: str, source_line: int, *, minimum: int | None = None
) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ContractError(
            f"line {source_line}: {field} must be an integer, got {value!r}"
        ) from exc
    if str(parsed) != value:
        raise ContractError(
            f"line {source_line}: {field} must be a canonical integer, got {value!r}"
        )
    if minimum is not None and parsed < minimum:
        raise ContractError(
            f"line {source_line}: {field} must be >= {minimum}, got {parsed}"
        )
    return parsed


def gid_number(value: str, source_line: int) -> int:
    match = GID_PATTERN.fullmatch(value)
    if match is None:
        raise ContractError(
            f"line {source_line}: display_global_id must be G0001-G0062, "
            f"-1, or blank; got {value!r}"
        )
    number = int(match.group(1))
    if not 1 <= number <= 62:
        raise ContractError(
            f"line {source_line}: display_global_id must be G0001-G0062, "
            f"got {value!r}"
        )
    return number


def scan_corrected_csv(path: Path) -> ScanStats:
    total_rows = 0
    assigned_gid_rows = 0
    reviewed_invalid_rows = 0
    original_invalid_rows = 0

    seen_det_ids: set[int] = set()
    gid_mapping: dict[str, tuple[int, str]] = {}
    gid_by_numeric_id: dict[int, str] = {}
    gid_by_uuid: dict[str, str] = {}

    rows_by_clip: Counter[str] = Counter()
    assigned_rows_by_clip: Counter[str] = Counter()
    reviewed_invalid_by_clip: Counter[str] = Counter()
    original_invalid_by_clip: Counter[str] = Counter()

    previous_source_sort_key: tuple[int, int] | None = None
    previous_global_frame: int | None = None

    # The corrected exporter guarantees chronological rows. Keeping only the
    # current frame bounds memory while still detecting every G-ID collision.
    current_frame: tuple[str, int, str, int, int] | None = None
    current_by_gid: dict[str, list[DetectionReference]] = {}
    duplicate_groups: list[DuplicateGroup] = []

    def flush_current_frame() -> None:
        if current_frame is None:
            return
        sequence_id, clip_order, clip_id, local_frame, global_frame = current_frame
        for display_global_id, detections in sorted(current_by_gid.items()):
            if len(detections) <= 1:
                continue
            duplicate_groups.append(
                DuplicateGroup(
                    sequence_id=sequence_id,
                    clip_order=clip_order,
                    clip_id=clip_id,
                    local_frame=local_frame,
                    global_frame=global_frame,
                    display_global_id=display_global_id,
                    detections=tuple(
                        sorted(detections, key=lambda detection: detection.det_id)
                    ),
                )
            )
        current_by_gid.clear()

    last_report = time.monotonic()
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = tuple(reader.fieldnames or ())
        if not fieldnames:
            raise ContractError("input CSV has no header")
        if fieldnames != EXPECTED_COLUMNS:
            raise ContractError("input CSV header differs from the corrected export schema")

        for row in reader:
            source_line = reader.line_num
            total_rows += 1
            if None in row or any(value is None for value in row.values()):
                raise ContractError(f"line {source_line}: malformed CSV row")

            sequence_id = row["sequence_id"]
            if sequence_id != EXPECTED_SEQUENCE_ID:
                raise ContractError(
                    f"line {source_line}: unexpected sequence_id {sequence_id!r}"
                )

            clip_id = row["clip_id"]
            try:
                clip_spec = CLIP_SPECS[clip_id]
            except KeyError as exc:
                raise ContractError(
                    f"line {source_line}: unexpected clip_id {clip_id!r}"
                ) from exc
            clip_order = canonical_int(
                row["clip_order"], "clip_order", source_line, minimum=0
            )
            if clip_order != clip_spec.clip_order:
                raise ContractError(
                    f"line {source_line}: clip_order disagrees with {clip_id}"
                )

            csv_row_index = canonical_int(
                row["csv_row_index"], "csv_row_index", source_line, minimum=0
            )
            if csv_row_index != rows_by_clip[clip_id]:
                raise ContractError(
                    f"line {source_line}: csv_row_index for {clip_id} must be "
                    f"{rows_by_clip[clip_id]}, got {csv_row_index}"
                )
            rows_by_clip[clip_id] += 1
            source_sort_key = (clip_order, csv_row_index)
            if (
                previous_source_sort_key is not None
                and source_sort_key <= previous_source_sort_key
            ):
                raise ContractError(
                    f"line {source_line}: (clip_order, csv_row_index) is not increasing"
                )
            previous_source_sort_key = source_sort_key

            local_frame = canonical_int(
                row["local_frame"], "local_frame", source_line, minimum=0
            )
            if local_frame >= clip_spec.frame_count:
                raise ContractError(
                    f"line {source_line}: local_frame is outside {clip_id}"
                )
            global_frame = canonical_int(
                row["global_frame"], "global_frame", source_line, minimum=0
            )
            if global_frame != clip_spec.global_frame_start + local_frame:
                raise ContractError(
                    f"line {source_line}: local/global frame mapping disagrees for "
                    f"{clip_id}"
                )
            if previous_global_frame is not None and global_frame < previous_global_frame:
                raise ContractError(f"line {source_line}: global_frame moved backwards")
            previous_global_frame = global_frame

            frame = (sequence_id, clip_order, clip_id, local_frame, global_frame)
            if current_frame is None:
                current_frame = frame
            elif frame != current_frame:
                if global_frame == current_frame[4]:
                    raise ContractError(
                        f"line {source_line}: one global_frame maps to multiple local frames"
                    )
                flush_current_frame()
                current_frame = frame

            det_id = canonical_int(row["det_id"], "det_id", source_line)
            if det_id in seen_det_ids:
                raise ContractError(
                    f"line {source_line}: duplicate det_id {det_id} in input CSV"
                )
            seen_det_ids.add(det_id)

            valid = row["valid"]
            display_global_id = row["display_global_id"]
            global_track_id_text = row["global_track_id"]
            global_track_uuid = row["global_track_uuid"]

            if valid == "false":
                original_invalid_rows += 1
                original_invalid_by_clip[clip_id] += 1
                if display_global_id or global_track_id_text or global_track_uuid:
                    raise ContractError(
                        f"line {source_line}: original invalid row retains global identity"
                    )
            elif valid == "true" and display_global_id == "-1":
                reviewed_invalid_rows += 1
                reviewed_invalid_by_clip[clip_id] += 1
                if (
                    global_track_id_text != "-1"
                    or global_track_uuid
                    or row["invalid_reason"] != "multiple_cows"
                ):
                    raise ContractError(
                        f"line {source_line}: reviewed-invalid identity fields disagree"
                    )
            elif valid == "true":
                number = gid_number(display_global_id, source_line)
                global_track_id = canonical_int(
                    global_track_id_text, "global_track_id", source_line, minimum=0
                )
                if global_track_id != number - 1:
                    raise ContractError(
                        f"line {source_line}: global_track_id disagrees with "
                        f"{display_global_id}"
                    )
                if not global_track_uuid:
                    raise ContractError(
                        f"line {source_line}: assigned global ID has a blank UUID"
                    )
                if row["invalid_reason"] == "multiple_cows":
                    raise ContractError(
                        f"line {source_line}: assigned global ID is marked multiple_cows"
                    )

                mapping = (global_track_id, global_track_uuid)
                previous_mapping = gid_mapping.setdefault(display_global_id, mapping)
                if previous_mapping != mapping:
                    raise ContractError(
                        f"line {source_line}: mapping changed for {display_global_id}"
                    )
                previous_gid = gid_by_numeric_id.setdefault(
                    global_track_id, display_global_id
                )
                if previous_gid != display_global_id:
                    raise ContractError(
                        f"line {source_line}: numeric global ID maps to multiple display IDs"
                    )
                previous_gid = gid_by_uuid.setdefault(
                    global_track_uuid, display_global_id
                )
                if previous_gid != display_global_id:
                    raise ContractError(
                        f"line {source_line}: global UUID maps to multiple display IDs"
                    )

                legacy_track_id = row["legacy_track_id"]
                if not legacy_track_id:
                    raise ContractError(
                        f"line {source_line}: assigned row has no legacy_track_id"
                    )
                stable_id = canonical_int(
                    row["stable_id"], "stable_id", source_line, minimum=0
                )
                current_by_gid.setdefault(display_global_id, []).append(
                    DetectionReference(
                        det_id=det_id,
                        legacy_track_id=legacy_track_id,
                        stable_id=stable_id,
                    )
                )
                assigned_gid_rows += 1
                assigned_rows_by_clip[clip_id] += 1
            else:
                raise ContractError(
                    f"line {source_line}: valid must be true or false, got {valid!r}"
                )

            now = time.monotonic()
            if now - last_report >= PROGRESS_INTERVAL_SEC:
                log(
                    f"[count] rows={total_rows:,}, assigned_gid={assigned_gid_rows:,}, "
                    f"duplicate_groups_closed={len(duplicate_groups):,}"
                )
                last_report = now

    flush_current_frame()
    if total_rows != EXPECTED_TOTAL_ROWS:
        raise ContractError(
            f"expected {EXPECTED_TOTAL_ROWS:,} rows, found {total_rows:,}"
        )
    if assigned_gid_rows != EXPECTED_ASSIGNED_GID_ROWS:
        raise ContractError(
            f"expected {EXPECTED_ASSIGNED_GID_ROWS:,} assigned G-ID rows, "
            f"found {assigned_gid_rows:,}"
        )
    if reviewed_invalid_rows != EXPECTED_REVIEWED_INVALID_ROWS:
        raise ContractError(
            f"expected {EXPECTED_REVIEWED_INVALID_ROWS:,} reviewed-invalid rows, "
            f"found {reviewed_invalid_rows:,}"
        )
    if original_invalid_rows != EXPECTED_ORIGINAL_INVALID_ROWS:
        raise ContractError(
            f"expected {EXPECTED_ORIGINAL_INVALID_ROWS:,} original-invalid rows, "
            f"found {original_invalid_rows:,}"
        )
    for clip_id, clip_spec in CLIP_SPECS.items():
        observed = (
            rows_by_clip[clip_id],
            assigned_rows_by_clip[clip_id],
            reviewed_invalid_by_clip[clip_id],
            original_invalid_by_clip[clip_id],
        )
        expected = (
            clip_spec.total_rows,
            clip_spec.assigned_gid_rows,
            clip_spec.reviewed_invalid_rows,
            clip_spec.original_invalid_rows,
        )
        if observed != expected:
            raise ContractError(
                f"{clip_id} row counts differ: observed={observed}, expected={expected}"
            )

    clips = tuple(
        sorted((clip_spec.clip_order, clip_id) for clip_id, clip_spec in CLIP_SPECS.items())
    )
    return ScanStats(
        total_rows=total_rows,
        assigned_gid_rows=assigned_gid_rows,
        reviewed_invalid_rows=reviewed_invalid_rows,
        original_invalid_rows=original_invalid_rows,
        clips=clips,
        duplicate_groups=tuple(duplicate_groups),
    )


def print_summary(stats: ScanStats, *, show_groups: bool) -> None:
    duplicate_frames = stats.duplicate_frames
    duplicate_ids = sorted(
        {group.display_global_id for group in stats.duplicate_groups}
    )
    duplicate_detection_rows = sum(
        group.detection_count for group in stats.duplicate_groups
    )
    excess_detection_rows = sum(
        group.excess_detection_count for group in stats.duplicate_groups
    )

    log(
        f"[validated] rows={stats.total_rows:,}, "
        f"assigned_gid={stats.assigned_gid_rows:,}, "
        f"reviewed_invalid={stats.reviewed_invalid_rows:,}, "
        f"original_invalid={stats.original_invalid_rows:,}"
    )
    log(f"[result] duplicate_frame_count={len(duplicate_frames):,}")
    log(
        f"[result] duplicate_gid_frame_group_count="
        f"{len(stats.duplicate_groups):,}"
    )
    log(f"[result] duplicate_detection_rows={duplicate_detection_rows:,}")
    log(f"[result] excess_detection_rows={excess_detection_rows:,}")
    log(f"[result] duplicate_global_id_count={len(duplicate_ids):,}")
    log(
        "[result] duplicate_global_ids="
        + (",".join(duplicate_ids) if duplicate_ids else "none")
    )

    groups_by_clip: Counter[str] = Counter()
    frames_by_clip: dict[str, set[tuple[str, int]]] = {
        clip_id: set() for _, clip_id in stats.clips
    }
    for group in stats.duplicate_groups:
        groups_by_clip[group.clip_id] += 1
        frames_by_clip.setdefault(group.clip_id, set()).add(
            (group.sequence_id, group.local_frame)
        )
    for _, clip_id in stats.clips:
        log(
            f"[clip] {clip_id}: duplicate_frames="
            f"{len(frames_by_clip[clip_id]):,}, "
            f"duplicate_gid_frame_groups={groups_by_clip[clip_id]:,}"
        )

    frames_by_gid: Counter[str] = Counter()
    rows_by_gid: Counter[str] = Counter()
    excess_by_gid: Counter[str] = Counter()
    maximum_by_gid: Counter[str] = Counter()
    for group in stats.duplicate_groups:
        gid = group.display_global_id
        frames_by_gid[gid] += 1
        rows_by_gid[gid] += group.detection_count
        excess_by_gid[gid] += group.excess_detection_count
        maximum_by_gid[gid] = max(maximum_by_gid[gid], group.detection_count)
    for gid in duplicate_ids:
        log(
            f"[gid] {gid}: duplicate_frames={frames_by_gid[gid]:,}, "
            f"detection_rows={rows_by_gid[gid]:,}, "
            f"excess_rows={excess_by_gid[gid]:,}, "
            f"max_per_frame={maximum_by_gid[gid]:,}"
        )

    if show_groups:
        for group in stats.duplicate_groups:
            det_ids = ";".join(str(item.det_id) for item in group.detections)
            legacy_ids = ";".join(
                item.legacy_track_id for item in group.detections
            )
            stable_ids = ";".join(
                str(item.stable_id) for item in group.detections
            )
            log(
                f"[duplicate] clip={group.clip_id}, "
                f"local_frame={group.local_frame}, "
                f"global_frame={group.global_frame}, "
                f"gid={group.display_global_id}, "
                f"detections={group.detection_count}, "
                f"det_ids={det_ids}, legacy_track_ids={legacy_ids}, "
                f"stable_ids={stable_ids}"
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Count frames in the corrected detection CSV where one G0001-G0062 "
            "identity is assigned to multiple detections."
        )
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=DEFAULT_INPUT_CSV,
        help=f"corrected detection CSV (default: {DEFAULT_INPUT_CSV})",
    )
    parser.add_argument(
        "--show-groups",
        action="store_true",
        help="print every duplicate clip/frame/G-ID group and its source IDs",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    input_csv = args.input_csv.resolve()
    if not input_csv.is_file():
        raise ContractError(f"input CSV does not exist or is not a file: {input_csv}")

    log(f"[input] {input_csv}")
    log(
        "[policy] group_by=(clip_id,local_frame,display_global_id), "
        "assigned_ids=G0001-G0062, exclude=-1/blank"
    )
    stats = scan_corrected_csv(input_csv)
    print_summary(stats, show_groups=args.show_groups)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ContractError, OSError) as exc:
        raise SystemExit(f"error: {exc}") from None
