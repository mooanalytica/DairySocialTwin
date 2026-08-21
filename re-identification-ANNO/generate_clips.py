from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from concurrent.futures import Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_CEILING, ROUND_FLOOR
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


ROOT = Path(__file__).resolve().parent
OCCURRENCES_CSV = ROOT / "occurrence_segments.csv"
DETECTIONS_CSV = Path(
    "/home/hyw/re-identification-S6/work/"
    "dairy_farm_1_gopro1_20250505/06_export/"
    "detections_with_global_id.csv"
)
CACHE_DIR = ROOT / "cached_clips"
MANIFEST_PATH = CACHE_DIR / "manifest.json"

FPS_NUMERATOR = 30_000
FPS_DENOMINATOR = 1_001
WINDOW_FRAME_COUNT = 900
OUTPUT_WIDTH = 1_920
OUTPUT_HEIGHT = 1_080
RAW_WIDTH = 3_840
RAW_HEIGHT = 2_160
RAW_TO_OUTPUT_SCALE = Decimal("0.5")
BBOX_EXPANSION = Decimal("1.25")
RED_BOX_THICKNESS = 8
EXPECTED_OCCURRENCE_COUNT = 2_414
EXPECTED_DETECTION_COUNT = 746_279
PROGRESS_INTERVAL_SEC = 10.0
MANIFEST_SCHEMA_VERSION = "1.0"


@dataclass(frozen=True)
class VideoSpec:
    path: Path
    frame_count: int


VIDEO_SPECS: dict[str, VideoSpec] = {
    "GX040006": VideoSpec(
        path=Path(
            "/home/hyw/re-identification-S6/work/"
            "dairy_farm_1_gopro1_20250505/06_export/qa/videos/"
            "GX040006_tracked.mp4"
        ),
        frame_count=84_480,
    ),
    "GX050006": VideoSpec(
        path=Path(
            "/home/hyw/re-identification-S6/work/"
            "dairy_farm_1_gopro1_20250505/06_export/qa/videos/"
            "GX050006_tracked.mp4"
        ),
        frame_count=78_720,
    ),
}

OCCURRENCE_COLUMNS = (
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

DETECTION_COLUMNS = (
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

EXPECTED_GIDS = frozenset(f"G{index:04d}" for index in range(1, 63))
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


class ContractError(RuntimeError):
    """Raised when a fixed input, cache, or NVENC contract is violated."""


@dataclass(frozen=True)
class Occurrence:
    occurrence_id: str
    clip_order: int
    clip_id: str
    display_global_id: str
    legacy_track_id: str
    start_frame: int
    end_frame: int
    start_det_id: str


@dataclass(frozen=True)
class RedBox:
    x: int
    y: int
    width: int
    height: int

    def as_dict(self) -> dict[str, int]:
        return {
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
        }


@dataclass(frozen=True)
class GeneratedPart:
    occurrence_id: str
    part_path: Path
    final_path: Path
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class EncoderTools:
    ffmpeg_binary: str
    ffprobe_binary: str


def log(message: str) -> None:
    print(message, flush=True)


def _parse_int(value: str, field: str, row_number: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ContractError(
            f"row {row_number}: {field} must be an integer, got {value!r}"
        ) from exc


def _parse_decimal(value: str, field: str, row_number: int) -> Decimal:
    try:
        parsed = Decimal(value)
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ContractError(
            f"row {row_number}: {field} must be a finite decimal, got {value!r}"
        ) from exc
    if not parsed.is_finite():
        raise ContractError(
            f"row {row_number}: {field} must be finite, got {value!r}"
        )
    return parsed


def load_occurrences(
    path: Path = OCCURRENCES_CSV,
    *,
    expected_count: int = EXPECTED_OCCURRENCE_COUNT,
    video_specs: Mapping[str, VideoSpec] = VIDEO_SPECS,
) -> list[Occurrence]:
    occurrences: list[Occurrence] = []
    seen_start_det_ids: set[str] = set()

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != OCCURRENCE_COLUMNS:
            raise ContractError("occurrence CSV header differs from the fixed contract")

        for row in reader:
            row_number = reader.line_num
            ordinal = len(occurrences) + 1
            occurrence_id = row["occurrence_id"]
            expected_id = f"O{ordinal:06d}"
            if occurrence_id != expected_id:
                raise ContractError(
                    f"row {row_number}: expected occurrence_id {expected_id}, "
                    f"got {occurrence_id!r}"
                )

            clip_id = row["clip_id"]
            if clip_id not in video_specs:
                raise ContractError(f"row {row_number}: unexpected clip_id {clip_id!r}")
            expected_clip_order = tuple(video_specs).index(clip_id)
            clip_order = _parse_int(row["clip_order"], "clip_order", row_number)
            if clip_order != expected_clip_order:
                raise ContractError(
                    f"row {row_number}: clip_order does not match {clip_id}"
                )

            start_frame = _parse_int(row["start_frame"], "start_frame", row_number)
            end_frame = _parse_int(row["end_frame"], "end_frame", row_number)
            if not 0 <= start_frame <= end_frame < video_specs[clip_id].frame_count:
                raise ContractError(
                    f"row {row_number}: occurrence interval is outside {clip_id}"
                )

            gid = row["display_global_id"]
            if gid not in EXPECTED_GIDS:
                raise ContractError(
                    f"row {row_number}: unexpected display_global_id {gid!r}"
                )
            legacy_track_id = row["legacy_track_id"]
            if not legacy_track_id:
                raise ContractError(f"row {row_number}: empty legacy_track_id")
            start_det_id = row["start_det_id"]
            if not start_det_id:
                raise ContractError(f"row {row_number}: empty start_det_id")
            if start_det_id in seen_start_det_ids:
                raise ContractError(
                    f"row {row_number}: duplicate start_det_id {start_det_id!r}"
                )
            seen_start_det_ids.add(start_det_id)

            occurrences.append(
                Occurrence(
                    occurrence_id=occurrence_id,
                    clip_order=clip_order,
                    clip_id=clip_id,
                    display_global_id=gid,
                    legacy_track_id=legacy_track_id,
                    start_frame=start_frame,
                    end_frame=end_frame,
                    start_det_id=start_det_id,
                )
            )

    if len(occurrences) != expected_count:
        raise ContractError(
            f"expected {expected_count:,} occurrences, found {len(occurrences):,}"
        )
    return occurrences


def make_red_box(
    x1: Decimal,
    y1: Decimal,
    x2: Decimal,
    y2: Decimal,
    *,
    scale: Decimal = RAW_TO_OUTPUT_SCALE,
    expansion: Decimal = BBOX_EXPANSION,
    output_width: int = OUTPUT_WIDTH,
    output_height: int = OUTPUT_HEIGHT,
) -> RedBox:
    if not (
        Decimal(0) <= x1 < x2 <= Decimal(RAW_WIDTH)
        and Decimal(0) <= y1 < y2 <= Decimal(RAW_HEIGHT)
    ):
        raise ContractError(
            "anchor-frame bbox must satisfy 0 <= x1 < x2 <= 3840 and "
            "0 <= y1 < y2 <= 2160"
        )
    if scale <= 0 or expansion <= 0:
        raise ContractError("bbox scale and expansion must be positive")

    scaled_x1 = x1 * scale
    scaled_y1 = y1 * scale
    scaled_x2 = x2 * scale
    scaled_y2 = y2 * scale
    center_x = (scaled_x1 + scaled_x2) / 2
    center_y = (scaled_y1 + scaled_y2) / 2
    half_width = (scaled_x2 - scaled_x1) * expansion / 2
    half_height = (scaled_y2 - scaled_y1) * expansion / 2

    left = int((center_x - half_width).to_integral_value(rounding=ROUND_FLOOR))
    top = int((center_y - half_height).to_integral_value(rounding=ROUND_FLOOR))
    right = int((center_x + half_width).to_integral_value(rounding=ROUND_CEILING))
    bottom = int((center_y + half_height).to_integral_value(rounding=ROUND_CEILING))

    left = max(0, min(output_width, left))
    top = max(0, min(output_height, top))
    right = max(0, min(output_width, right))
    bottom = max(0, min(output_height, bottom))
    if right <= left or bottom <= top:
        raise ContractError("expanded anchor-frame bbox is outside the output frame")
    return RedBox(x=left, y=top, width=right - left, height=bottom - top)


def load_anchor_overrides(
    path: Path | None,
    occurrences: Sequence[Occurrence],
) -> dict[str, int]:
    if path is None:
        return {}
    if path.is_symlink() or not path.is_file():
        raise ContractError(f"anchor override JSON is not a regular file: {path}")

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ContractError(f"duplicate JSON key {key!r} in {path}")
            result[key] = value
        return result

    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle, object_pairs_hook=reject_duplicate_keys)
    except json.JSONDecodeError as exc:
        raise ContractError(f"invalid anchor override JSON: {path}") from exc
    if not isinstance(payload, dict) or set(payload) != {"schema_version", "anchors"}:
        raise ContractError("anchor override JSON must contain schema_version and anchors")
    if payload["schema_version"] != "1.0" or not isinstance(payload["anchors"], list):
        raise ContractError("anchor override schema_version/anchors mismatch")

    by_id = {occurrence.occurrence_id: occurrence for occurrence in occurrences}
    overrides: dict[str, int] = {}
    previous_ordinal = 0
    for index, item in enumerate(payload["anchors"]):
        if not isinstance(item, dict) or set(item) != {"occurrence_id", "anchor_frame"}:
            raise ContractError(f"anchor override item {index} has invalid fields")
        occurrence_id = item["occurrence_id"]
        if not isinstance(occurrence_id, str) or occurrence_id not in by_id:
            raise ContractError(f"anchor override item {index} has unknown occurrence_id")
        ordinal = int(occurrence_id[1:])
        if ordinal <= previous_ordinal:
            raise ContractError("anchor overrides must be unique and ordered by occurrence_id")
        previous_ordinal = ordinal
        anchor = item["anchor_frame"]
        if isinstance(anchor, bool) or not isinstance(anchor, int):
            raise ContractError(f"anchor for {occurrence_id} must be an integer")
        occurrence = by_id[occurrence_id]
        if not occurrence.start_frame <= anchor <= occurrence.end_frame:
            raise ContractError(
                f"anchor for {occurrence_id} must be inside its occurrence interval"
            )
        overrides[occurrence_id] = anchor
    return overrides


def lookup_anchor_bboxes(
    occurrences: Sequence[Occurrence],
    detections_path: Path = DETECTIONS_CSV,
    *,
    anchor_frames: Mapping[str, int] | None = None,
    anchor_det_ids: Mapping[str, str] | None = None,
    expected_rows: int | None = EXPECTED_DETECTION_COUNT,
    progress_interval_sec: float = PROGRESS_INTERVAL_SEC,
    logger: Callable[[str], None] = log,
) -> dict[str, RedBox]:
    overrides = {} if anchor_frames is None else dict(anchor_frames)
    expected_det_ids = {} if anchor_det_ids is None else dict(anchor_det_ids)
    occurrence_ids = {item.occurrence_id for item in occurrences}
    extra_overrides = sorted(set(overrides) - occurrence_ids)
    if extra_overrides:
        raise ContractError(f"anchor overrides contain unknown occurrences: {extra_overrides}")
    extra_det_ids = sorted(set(expected_det_ids) - occurrence_ids)
    if extra_det_ids:
        raise ContractError(
            f"anchor det IDs contain unknown occurrences: {extra_det_ids}"
        )
    if any(not isinstance(value, str) or not value for value in expected_det_ids.values()):
        raise ContractError("anchor det IDs must be non-empty strings")

    targets_by_det_id: dict[str, Occurrence] = {}
    targets_by_stream_frame: dict[tuple[str, str, int], Occurrence] = {}
    resolved_anchors: dict[str, int] = {}
    for occurrence in occurrences:
        anchor = overrides.get(occurrence.occurrence_id, occurrence.start_frame)
        if not occurrence.start_frame <= anchor <= occurrence.end_frame:
            raise ContractError(
                f"anchor for {occurrence.occurrence_id} is outside its occurrence interval"
            )
        resolved_anchors[occurrence.occurrence_id] = anchor
        if anchor == occurrence.start_frame:
            if occurrence.start_det_id in targets_by_det_id:
                raise ContractError("start_det_id must be a bijection over occurrences")
            targets_by_det_id[occurrence.start_det_id] = occurrence
        else:
            key = (occurrence.clip_id, occurrence.legacy_track_id, anchor)
            if key in targets_by_stream_frame:
                raise ContractError(f"duplicate anchor stream/frame target: {key}")
            targets_by_stream_frame[key] = occurrence
    if len(targets_by_det_id) + len(targets_by_stream_frame) != len(occurrences):
        raise ContractError("start_det_id must be a bijection over occurrences")

    found: dict[str, RedBox] = {}
    row_count = 0
    last_report = time.monotonic()
    with detections_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != DETECTION_COLUMNS:
            raise ContractError("detection CSV header differs from the fixed S6 contract")

        for row in reader:
            row_count += 1
            det_id = row["det_id"]
            occurrence = targets_by_det_id.get(det_id)
            if row["valid"] == "true":
                frame_target = targets_by_stream_frame.get(
                    (
                        row["clip_id"],
                        row["legacy_track_id"],
                        _parse_int(row["local_frame"], "local_frame", reader.line_num),
                    )
                )
                if occurrence is not None and frame_target is not None:
                    raise ContractError(
                        f"detection row {reader.line_num} matches multiple anchor targets"
                    )
                if frame_target is not None:
                    occurrence = frame_target
            if occurrence is not None:
                if occurrence.occurrence_id in found:
                    raise ContractError(
                        f"anchor row for {occurrence.occurrence_id} occurs more than once"
                    )
                anchor = resolved_anchors[occurrence.occurrence_id]
                expected_fields = {
                    "clip_id": occurrence.clip_id,
                    "clip_order": str(occurrence.clip_order),
                    "legacy_track_id": occurrence.legacy_track_id,
                    "display_global_id": occurrence.display_global_id,
                    "local_frame": str(anchor),
                    "valid": "true",
                }
                expected_det_id = expected_det_ids.get(occurrence.occurrence_id)
                if expected_det_id is not None:
                    expected_fields["det_id"] = expected_det_id
                for field, expected_value in expected_fields.items():
                    if row[field] != expected_value:
                        raise ContractError(
                            f"detection row {reader.line_num}: {field} for "
                            f"{occurrence.occurrence_id} is {row[field]!r}, "
                            f"expected {expected_value!r}"
                        )
                found[occurrence.occurrence_id] = make_red_box(
                    _parse_decimal(row["x1"], "x1", reader.line_num),
                    _parse_decimal(row["y1"], "y1", reader.line_num),
                    _parse_decimal(row["x2"], "x2", reader.line_num),
                    _parse_decimal(row["y2"], "y2", reader.line_num),
                )

            now = time.monotonic()
            if now - last_report >= progress_interval_sec:
                logger(
                    f"[index] rows={row_count:,}, "
                    f"matched={len(found):,}/{len(occurrences):,}"
                )
                last_report = now

    if expected_rows is not None and row_count != expected_rows:
        raise ContractError(
            f"expected {expected_rows:,} detection rows, found {row_count:,}"
        )
    missing = [item.occurrence_id for item in occurrences if item.occurrence_id not in found]
    if missing:
        preview = ", ".join(missing[:10])
        raise ContractError(
            f"missing exact anchor bbox for {len(missing):,} occurrences: {preview}"
        )
    return found


def lookup_start_bboxes(
    occurrences: Sequence[Occurrence],
    detections_path: Path = DETECTIONS_CSV,
    *,
    expected_rows: int | None = EXPECTED_DETECTION_COUNT,
    progress_interval_sec: float = PROGRESS_INTERVAL_SEC,
    logger: Callable[[str], None] = log,
) -> dict[str, RedBox]:
    return lookup_anchor_bboxes(
        occurrences,
        detections_path,
        expected_rows=expected_rows,
        progress_interval_sec=progress_interval_sec,
        logger=logger,
    )


def compute_window(anchor_frame: int, total_frames: int) -> tuple[int, int, int]:
    if total_frames < WINDOW_FRAME_COUNT:
        raise ContractError(
            f"video has {total_frames} frames, fewer than required {WINDOW_FRAME_COUNT}"
        )
    if not 0 <= anchor_frame < total_frames:
        raise ContractError(f"anchor frame {anchor_frame} is outside the video")
    start = anchor_frame - WINDOW_FRAME_COUNT // 2
    start = max(0, min(start, total_frames - WINDOW_FRAME_COUNT))
    end = start + WINDOW_FRAME_COUNT - 1
    return start, end, anchor_frame - start


def build_pending_records(
    occurrences: Sequence[Occurrence],
    boxes: Mapping[str, RedBox],
    *,
    anchor_frames: Mapping[str, int] | None = None,
    video_specs: Mapping[str, VideoSpec] = VIDEO_SPECS,
) -> list[dict[str, Any]]:
    overrides = {} if anchor_frames is None else dict(anchor_frames)
    extra_overrides = sorted(
        set(overrides) - {occurrence.occurrence_id for occurrence in occurrences}
    )
    if extra_overrides:
        raise ContractError(f"anchor overrides contain unknown occurrences: {extra_overrides}")
    records: list[dict[str, Any]] = []
    for occurrence in occurrences:
        try:
            red_box = boxes[occurrence.occurrence_id]
        except KeyError as exc:
            raise ContractError(
                f"no anchor-frame bbox for {occurrence.occurrence_id}"
            ) from exc
        anchor = overrides.get(occurrence.occurrence_id, occurrence.start_frame)
        if not occurrence.start_frame <= anchor <= occurrence.end_frame:
            raise ContractError(
                f"anchor for {occurrence.occurrence_id} is outside its occurrence interval"
            )
        start, end, anchor_offset = compute_window(
            anchor, video_specs[occurrence.clip_id].frame_count
        )
        records.append(
            {
                "occurrence_id": occurrence.occurrence_id,
                "relative_path": f"{occurrence.occurrence_id}.mp4",
                "status": "pending",
                "clip_id": occurrence.clip_id,
                "display_global_id": occurrence.display_global_id,
                "legacy_track_id": occurrence.legacy_track_id,
                "anchor_frame": anchor,
                "window_start_frame": start,
                "window_end_frame": end,
                "frame_count": WINDOW_FRAME_COUNT,
                "anchor_offset_frame": anchor_offset,
                "red_box": red_box.as_dict(),
            }
        )
    return records


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
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
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
        temporary_name = None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def _validate_red_box(value: object, occurrence_id: str) -> None:
    if not isinstance(value, dict) or set(value) != {"x", "y", "width", "height"}:
        raise ContractError(f"manifest {occurrence_id}: invalid red_box schema")
    if any(isinstance(value[key], bool) or not isinstance(value[key], int) for key in value):
        raise ContractError(f"manifest {occurrence_id}: red_box values must be integers")
    x, y = value["x"], value["y"]
    width, height = value["width"], value["height"]
    if x < 0 or y < 0 or width <= 0 or height <= 0:
        raise ContractError(f"manifest {occurrence_id}: invalid red_box geometry")
    if x + width > OUTPUT_WIDTH or y + height > OUTPUT_HEIGHT:
        raise ContractError(f"manifest {occurrence_id}: red_box exceeds output frame")


def validate_manifest(
    payload: object,
    expected_pending_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not isinstance(payload, dict) or set(payload) != {"schema_version", "clips"}:
        raise ContractError("cache manifest top-level schema mismatch")
    if payload["schema_version"] != MANIFEST_SCHEMA_VERSION:
        raise ContractError("cache manifest schema_version mismatch")
    clips = payload["clips"]
    if not isinstance(clips, list) or len(clips) != len(expected_pending_records):
        raise ContractError("cache manifest must contain every occurrence in order")

    for actual, expected in zip(clips, expected_pending_records, strict=True):
        if not isinstance(actual, dict):
            raise ContractError("cache manifest clip entry must be an object")
        occurrence_id = expected["occurrence_id"]
        status = actual.get("status")
        if status not in {"pending", "complete"}:
            raise ContractError(
                f"manifest {occurrence_id}: status must be pending or complete"
            )
        expected_base_keys = set(expected)
        required_keys = expected_base_keys
        if status == "complete":
            required_keys = required_keys | {"size_bytes", "sha256"}
        if set(actual) != required_keys:
            raise ContractError(f"manifest {occurrence_id}: entry schema mismatch")

        for key, expected_value in expected.items():
            if key == "status":
                continue
            if actual[key] != expected_value:
                raise ContractError(
                    f"manifest {occurrence_id}: immutable field {key} changed"
                )
        _validate_red_box(actual["red_box"], str(occurrence_id))

        if status == "complete":
            size = actual["size_bytes"]
            digest = actual["sha256"]
            if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
                raise ContractError(
                    f"manifest {occurrence_id}: invalid complete size_bytes"
                )
            if not isinstance(digest, str) or SHA256_PATTERN.fullmatch(digest) is None:
                raise ContractError(f"manifest {occurrence_id}: invalid complete sha256")
    return payload


def load_or_create_manifest(
    path: Path,
    pending_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if path.is_symlink():
        raise ContractError(f"cache manifest must not be a symlink: {path}")
    if path.exists():
        if not path.is_file():
            raise ContractError(f"cache manifest is not a regular file: {path}")
        with path.open("r", encoding="utf-8") as handle:
            try:
                payload = json.load(handle)
            except json.JSONDecodeError as exc:
                raise ContractError(f"invalid cache manifest JSON: {path}") from exc
        return validate_manifest(payload, pending_records)

    payload: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "clips": [dict(record) for record in pending_records],
    }
    _atomic_write_json(path, payload)
    return payload


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _contains_rotation_metadata(value: object) -> bool:
    if isinstance(value, dict):
        for key, child in value.items():
            normalized_key = str(key).casefold()
            if normalized_key in {"rotate", "rotation"}:
                return True
            if normalized_key == "side_data_type" and "display matrix" in str(
                child
            ).casefold():
                return True
            if _contains_rotation_metadata(child):
                return True
    elif isinstance(value, list):
        return any(_contains_rotation_metadata(child) for child in value)
    return False


def probe_video(
    path: Path,
    *,
    ffprobe_binary: str,
    expected_frame_count: int,
    runner: Callable[..., subprocess.CompletedProcess[str]] | None = None,
) -> None:
    if runner is None:
        runner = subprocess.run
    command = [
        ffprobe_binary,
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_streams",
        "-count_frames",
        str(path),
    ]
    result = runner(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        raise ContractError(
            f"ffprobe failed for {path} with exit {result.returncode}: {stderr}"
        )
    try:
        payload = json.loads(result.stdout or "")
    except json.JSONDecodeError as exc:
        raise ContractError(f"ffprobe returned invalid JSON for {path}") from exc
    if not isinstance(payload, dict) or set(payload) != {"streams"}:
        raise ContractError(f"ffprobe stream payload schema mismatch for {path}")
    streams = payload["streams"]
    if not isinstance(streams, list) or len(streams) != 1:
        raise ContractError(f"{path} must contain exactly one video stream")
    stream = streams[0]
    if not isinstance(stream, dict) or stream.get("codec_type") != "video":
        raise ContractError(f"{path} must contain exactly one video stream")

    expected_fields: dict[str, object] = {
        "codec_name": "h264",
        "width": OUTPUT_WIDTH,
        "height": OUTPUT_HEIGHT,
        "pix_fmt": "yuv420p",
        "r_frame_rate": f"{FPS_NUMERATOR}/{FPS_DENOMINATOR}",
        "avg_frame_rate": f"{FPS_NUMERATOR}/{FPS_DENOMINATOR}",
        "nb_read_frames": str(expected_frame_count),
    }
    for field, expected in expected_fields.items():
        if stream.get(field) != expected:
            raise ContractError(
                f"{path}: ffprobe {field} is {stream.get(field)!r}, "
                f"expected {expected!r}"
            )
    if _contains_rotation_metadata(stream):
        raise ContractError(f"{path}: rotation/display-matrix metadata is forbidden")


def inspect_cached_record(
    record: Mapping[str, Any],
    cache_dir: Path,
    *,
    ffprobe_binary: str,
    probe_runner: Callable[..., subprocess.CompletedProcess[str]] | None = None,
) -> dict[str, object] | None:
    """Return adoption metadata for pending+final, an empty dict for complete, or None."""
    final_path = cache_dir / str(record["relative_path"])
    part_path = final_path.with_suffix(final_path.suffix + ".part")
    if record["status"] == "pending":
        if not final_path.exists():
            return None
        if final_path.is_symlink() or not final_path.is_file():
            raise ContractError(
                f"uncommitted cache output is not regular: {final_path}"
            )
        if part_path.exists():
            raise ContractError(
                f"pending cache output has both final and .part files: {final_path}"
            )
        probe_video(
            final_path,
            ffprobe_binary=ffprobe_binary,
            expected_frame_count=WINDOW_FRAME_COUNT,
            runner=probe_runner,
        )
        size_bytes = final_path.stat().st_size
        if size_bytes <= 0:
            raise ContractError(f"uncommitted cache output is empty: {final_path}")
        return {"size_bytes": size_bytes, "sha256": file_sha256(final_path)}

    if final_path.is_symlink() or not final_path.is_file():
        raise ContractError(
            f"complete cache output is missing or not regular: {final_path}"
        )
    if part_path.exists():
        raise ContractError(
            f"complete cache output has an unexpected .part sibling: {part_path}"
        )
    if final_path.stat().st_size != record["size_bytes"]:
        raise ContractError(f"cached size mismatch for {record['occurrence_id']}")
    if file_sha256(final_path) != record["sha256"]:
        raise ContractError(f"cached SHA-256 mismatch for {record['occurrence_id']}")
    probe_video(
        final_path,
        ffprobe_binary=ffprobe_binary,
        expected_frame_count=WINDOW_FRAME_COUNT,
        runner=probe_runner,
    )
    return {}


def _format_seek_time(frame: int) -> str:
    seconds = Decimal(frame * FPS_DENOMINATOR) / Decimal(FPS_NUMERATOR)
    return f"{seconds:.9f}"


def build_ffmpeg_command(
    record: Mapping[str, Any],
    source_video: Path,
    part_path: Path,
    *,
    ffmpeg_binary: str = "ffmpeg",
) -> list[str]:
    if part_path.suffix != ".part" or not part_path.name.endswith(".mp4.part"):
        raise ContractError("ffmpeg output must use an .mp4.part path")
    red_box = record["red_box"]
    _validate_red_box(red_box, str(record["occurrence_id"]))
    if record["frame_count"] != WINDOW_FRAME_COUNT:
        raise ContractError("manifest frame_count differs from fixed window")
    drawbox = (
        f"drawbox=x={red_box['x']}:y={red_box['y']}:"
        f"w={red_box['width']}:h={red_box['height']}:"
        f"color=red@1.0:t={RED_BOX_THICKNESS}"
    )
    return [
        ffmpeg_binary,
        "-hide_banner",
        "-nostdin",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        _format_seek_time(int(record["window_start_frame"])),
        "-noautorotate",
        "-i",
        str(source_video),
        "-map",
        "0:v:0",
        "-an",
        "-sn",
        "-dn",
        "-vf",
        drawbox,
        "-frames:v",
        str(WINDOW_FRAME_COUNT),
        "-c:v",
        "h264_nvenc",
        "-gpu",
        "0",
        "-preset",
        "p4",
        "-cq",
        "21",
        "-pix_fmt",
        "yuv420p",
        "-fps_mode",
        "passthrough",
        "-movflags",
        "+faststart",
        "-f",
        "mp4",
        str(part_path),
    ]


def require_nvenc_environment(
    env: Mapping[str, str] = os.environ,
    *,
    which: Callable[[str], str | None] = shutil.which,
) -> EncoderTools:
    visible_devices = env.get("CUDA_VISIBLE_DEVICES")
    if visible_devices != "1":
        raise ContractError(
            "CUDA_VISIBLE_DEVICES must be exactly '1'; CPU encoding and fallback are forbidden"
        )
    ffmpeg_binary = which("ffmpeg")
    if not ffmpeg_binary:
        raise ContractError("ffmpeg is required for h264_nvenc clip generation")
    ffprobe_binary = which("ffprobe")
    if not ffprobe_binary:
        raise ContractError("ffprobe is required for strict video validation")
    if Path(ffprobe_binary).parent != Path(ffmpeg_binary).parent:
        raise ContractError("ffprobe must be located alongside ffmpeg")
    return EncoderTools(
        ffmpeg_binary=ffmpeg_binary,
        ffprobe_binary=ffprobe_binary,
    )


def generate_part(
    record: Mapping[str, Any],
    *,
    source_video: Path,
    cache_dir: Path,
    ffmpeg_binary: str,
    ffprobe_binary: str,
    runner: Callable[..., subprocess.CompletedProcess[str]] | None = None,
    probe_runner: Callable[..., subprocess.CompletedProcess[str]] | None = None,
    env: Mapping[str, str] = os.environ,
) -> GeneratedPart:
    if runner is None:
        runner = subprocess.run
    if env.get("CUDA_VISIBLE_DEVICES") != "1":
        raise ContractError(
            "CUDA_VISIBLE_DEVICES must be exactly '1' for every ffmpeg invocation"
        )
    final_path = cache_dir / str(record["relative_path"])
    part_path = final_path.with_suffix(final_path.suffix + ".part")
    if final_path.exists():
        raise ContractError(f"refusing to overwrite cache output: {final_path}")
    if part_path.is_symlink():
        raise ContractError(f"refusing symlink .part output: {part_path}")
    part_path.unlink(missing_ok=True)

    command = build_ffmpeg_command(
        record, source_video, part_path, ffmpeg_binary=ffmpeg_binary
    )
    result = runner(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
        env=dict(env),
    )
    if result.returncode != 0:
        part_path.unlink(missing_ok=True)
        stderr = (result.stderr or "").strip()
        if len(stderr) > 4_000:
            stderr = stderr[-4_000:]
        raise ContractError(
            f"h264_nvenc failed for {record['occurrence_id']} "
            f"with exit {result.returncode}: {stderr}"
        )
    if part_path.is_symlink() or not part_path.is_file():
        raise ContractError(
            f"ffmpeg did not create a regular .part output for {record['occurrence_id']}"
        )
    size_bytes = part_path.stat().st_size
    if size_bytes <= 0:
        part_path.unlink(missing_ok=True)
        raise ContractError(f"ffmpeg created an empty output for {record['occurrence_id']}")
    try:
        probe_video(
            part_path,
            ffprobe_binary=ffprobe_binary,
            expected_frame_count=WINDOW_FRAME_COUNT,
            runner=probe_runner,
        )
    except Exception:
        part_path.unlink(missing_ok=True)
        raise
    return GeneratedPart(
        occurrence_id=str(record["occurrence_id"]),
        part_path=part_path,
        final_path=final_path,
        size_bytes=size_bytes,
        sha256=file_sha256(part_path),
    )


def select_records(
    records: Sequence[dict[str, Any]],
    *,
    occurrence_ids: Sequence[str] | None = None,
    start_index: int = 1,
    end_index: int | None = None,
) -> list[dict[str, Any]]:
    if occurrence_ids:
        if start_index != 1 or end_index is not None:
            raise ContractError(
                "--occurrence-id cannot be combined with --start-index/--end-index"
            )
        by_id = {str(record["occurrence_id"]): record for record in records}
        selected: list[dict[str, Any]] = []
        seen: set[str] = set()
        for occurrence_id in occurrence_ids:
            if occurrence_id in seen:
                raise ContractError(f"duplicate --occurrence-id {occurrence_id}")
            seen.add(occurrence_id)
            try:
                selected.append(by_id[occurrence_id])
            except KeyError as exc:
                raise ContractError(
                    f"unknown occurrence_id {occurrence_id!r}"
                ) from exc
        return selected

    if end_index is None:
        end_index = len(records)
    if not 1 <= start_index <= end_index <= len(records):
        raise ContractError(
            f"selection range must satisfy 1 <= start <= end <= {len(records)}"
        )
    return list(records[start_index - 1 : end_index])


def generate_selected(
    manifest: dict[str, Any],
    selected: Sequence[dict[str, Any]],
    *,
    manifest_path: Path,
    cache_dir: Path,
    video_specs: Mapping[str, VideoSpec],
    ffmpeg_binary: str,
    ffprobe_binary: str,
    workers: int,
    runner: Callable[..., subprocess.CompletedProcess[str]] | None = None,
    probe_runner: Callable[..., subprocess.CompletedProcess[str]] | None = None,
    env: Mapping[str, str] = os.environ,
    progress_interval_sec: float = PROGRESS_INTERVAL_SEC,
    logger: Callable[[str], None] = log,
) -> None:
    if workers <= 0:
        raise ContractError("workers must be a positive integer")
    cache_dir.mkdir(parents=True, exist_ok=True)

    pending: list[dict[str, Any]] = []
    already_complete = 0
    adopted = 0
    for record in selected:
        cache_state = inspect_cached_record(
            record,
            cache_dir,
            ffprobe_binary=ffprobe_binary,
            probe_runner=probe_runner,
        )
        if cache_state is None:
            pending.append(record)
            continue
        if cache_state:
            record["status"] = "complete"
            record.update(cache_state)
            _atomic_write_json(manifest_path, manifest)
            adopted += 1
        already_complete += 1
    logger(
        f"[cache] selected={len(selected):,}, complete={already_complete:,}, "
        f"adopted={adopted:,}, pending={len(pending):,}"
    )
    if not pending:
        return

    future_to_record: dict[Future[GeneratedPart], dict[str, Any]] = {}
    errors: list[str] = []
    completed_now = 0
    last_report = time.monotonic()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for record in pending:
            video_spec = video_specs[str(record["clip_id"])]
            future = executor.submit(
                generate_part,
                record,
                source_video=video_spec.path,
                cache_dir=cache_dir,
                ffmpeg_binary=ffmpeg_binary,
                ffprobe_binary=ffprobe_binary,
                runner=runner,
                probe_runner=probe_runner,
                env=env,
            )
            future_to_record[future] = record

        outstanding = set(future_to_record)
        while outstanding:
            done, outstanding = wait(outstanding, timeout=1.0)
            for future in done:
                record = future_to_record[future]
                try:
                    generated = future.result()
                    if generated.final_path.exists():
                        generated.part_path.unlink(missing_ok=True)
                        raise ContractError(
                            f"cache output appeared concurrently: {generated.final_path}"
                        )
                    os.replace(generated.part_path, generated.final_path)
                    record["status"] = "complete"
                    record["size_bytes"] = generated.size_bytes
                    record["sha256"] = generated.sha256
                    _atomic_write_json(manifest_path, manifest)
                    completed_now += 1
                except Exception as exc:  # failures are aggregated after other workers settle
                    errors.append(f"{record['occurrence_id']}: {exc}")

            now = time.monotonic()
            if now - last_report >= progress_interval_sec:
                logger(
                    f"[progress] generated={completed_now:,}/{len(pending):,}, "
                    f"active={len(outstanding):,}, failures={len(errors):,}"
                )
                last_report = now

    logger(
        f"[progress] generated={completed_now:,}/{len(pending):,}, "
        f"failures={len(errors):,}"
    )
    if errors:
        raise ContractError("clip generation failures:\n" + "\n".join(errors))


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate fixed 900-frame occurrence review clips with one static red "
            "box using GPU 1 and h264_nvenc."
        )
    )
    parser.add_argument("--workers", type=_positive_int, default=2)
    parser.add_argument(
        "--occurrence-id",
        action="append",
        default=[],
        help="Generate one occurrence; repeat for multiple IDs.",
    )
    parser.add_argument(
        "--start-index",
        type=_positive_int,
        default=1,
        help="First 1-based occurrence index in an inclusive batch range.",
    )
    parser.add_argument(
        "--end-index",
        type=_positive_int,
        help="Last 1-based occurrence index in an inclusive batch range.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Build and validate the complete pending manifest without encoding clips.",
    )
    parser.add_argument(
        "--anchor-overrides",
        type=Path,
        help=(
            "Strict JSON mapping selected occurrences to an anchor frame inside their "
            "interval; the same file must be supplied when resuming generation."
        ),
    )
    parser.add_argument(
        "--overrides-only",
        action="store_true",
        help=(
            "Generate only occurrences listed by --anchor-overrides; cannot be "
            "combined with the other selection arguments."
        ),
    )
    parser.add_argument(
        "--expected-override-count",
        type=_positive_int,
        help="Fail unless --anchor-overrides contains exactly this many occurrences.",
    )
    return parser


def validate_cli_selection(
    args: argparse.Namespace, anchor_overrides: Mapping[str, int]
) -> None:
    has_explicit_selection = bool(args.occurrence_id) or (
        args.start_index != 1 or args.end_index is not None
    )
    if args.expected_override_count is not None:
        if args.anchor_overrides is None:
            raise ContractError(
                "--expected-override-count requires --anchor-overrides"
            )
        if len(anchor_overrides) != args.expected_override_count:
            raise ContractError(
                f"expected {args.expected_override_count} anchor overrides, "
                f"found {len(anchor_overrides)}"
            )
    if args.overrides_only:
        if args.anchor_overrides is None:
            raise ContractError("--overrides-only requires --anchor-overrides")
        if has_explicit_selection:
            raise ContractError(
                "--overrides-only cannot be combined with "
                "--occurrence-id/--start-index/--end-index"
            )
        if not anchor_overrides:
            raise ContractError("--overrides-only requires at least one override")
    elif (
        not args.prepare_only
        and args.anchor_overrides is not None
        and not has_explicit_selection
    ):
        raise ContractError(
            "--anchor-overrides with default selection is unsafe; add "
            "--overrides-only or an explicit occurrence/range selection"
        )


def _require_regular_file(path: Path, label: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise ContractError(f"missing fixed {label}: {path}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        _require_regular_file(OCCURRENCES_CSV, "occurrence CSV")
        _require_regular_file(DETECTIONS_CSV, "S6 detections CSV")
        for clip_id, spec in VIDEO_SPECS.items():
            _require_regular_file(spec.path, f"S6 tracked video {clip_id}")

        occurrences = load_occurrences()
        log(f"[occurrences] loaded={len(occurrences):,}")
        anchor_overrides = load_anchor_overrides(args.anchor_overrides, occurrences)
        log(f"[anchors] overrides={len(anchor_overrides):,}")
        validate_cli_selection(args, anchor_overrides)
        boxes = lookup_anchor_bboxes(
            occurrences,
            anchor_frames=anchor_overrides,
        )
        log(f"[index] exact_anchor_bboxes={len(boxes):,}")
        pending_records = build_pending_records(
            occurrences,
            boxes,
            anchor_frames=anchor_overrides,
        )
        manifest = load_or_create_manifest(MANIFEST_PATH, pending_records)
        log(f"[manifest] clips={len(manifest['clips']):,}, path={MANIFEST_PATH}")
        if args.prepare_only:
            log("[done] manifest prepared; no video encoding requested")
            return 0

        if args.overrides_only:
            selected = select_records(
                manifest["clips"], occurrence_ids=list(anchor_overrides)
            )
        else:
            selected = select_records(
                manifest["clips"],
                occurrence_ids=args.occurrence_id,
                start_index=args.start_index,
                end_index=args.end_index,
            )
        encoder_tools = require_nvenc_environment()
        for clip_id, spec in VIDEO_SPECS.items():
            log(f"[preflight] ffprobe source={clip_id}")
            probe_video(
                spec.path,
                ffprobe_binary=encoder_tools.ffprobe_binary,
                expected_frame_count=spec.frame_count,
            )
        generate_selected(
            manifest,
            selected,
            manifest_path=MANIFEST_PATH,
            cache_dir=CACHE_DIR,
            video_specs=VIDEO_SPECS,
            ffmpeg_binary=encoder_tools.ffmpeg_binary,
            ffprobe_binary=encoder_tools.ffprobe_binary,
            workers=args.workers,
        )
        log("[done] selected cache clips are complete")
        return 0
    except ContractError as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
