from __future__ import annotations

import csv
import hashlib
import json
import os
import re
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlencode

import numpy as np
import pandas as pd


APPEARANCE_CACHE_SCHEMA = 5
APPEARANCE_CONTRACT = (
    "webuil-displayed-consecutive-local-frames-canonical-deep-link-v5"
)
WEBUIL_DEEP_LINK_BASE_URL = "http://172.17.6.39:9922/"
TIMECODE_RE = re.compile(
    r"^(?P<hour>\d{2}):(?P<minute>[0-5]\d):(?P<second>[0-5]\d)"
    r"(?P<separator>[:;])(?P<frame>\d{2})$"
)
DISPLAY_ID_RE = re.compile(r"G\d{4}")
TRAJECTORY_COLUMNS = (
    "farm",
    "camera",
    "clip",
    "frame",
    "cow_id",
    "frozen",
    "display_global_id",
    "source_clip_id",
    "source_sample_id",
    "local_frame",
)
TIME_SEQUENCE_COLUMNS = {
    "farm_id",
    "camera_id",
    "sequence_id",
    "sequence_order",
    "source_relative_path",
    "file_name",
    "recording_date",
    "start_timecode",
    "last_frame_timecode",
    "end_timecode_exclusive",
    "frame_count",
    "frame_rate",
}


class DataContractError(RuntimeError):
    pass


@dataclass(frozen=True)
class TimecodeSpec:
    exact_rate: Fraction
    nominal_fps: int
    drop_frames: int
    separator: str

    @property
    def frames_per_24_hours(self) -> int:
        nominal = self.nominal_fps * 24 * 60 * 60
        if self.drop_frames == 0:
            return nominal
        return nominal - self.drop_frames * ((24 * 60) - (24 * 6))


@dataclass(frozen=True)
class ClipClock:
    clip: str
    sequence_order: int
    frame_count: int
    timeline_offset: int
    start_frame: int
    spec: TimecodeSpec

    def hhmmss(self, local_frame: int) -> str:
        if not 0 <= local_frame <= self.frame_count:
            raise DataContractError(
                f"{self.clip}: local frame {local_frame} is outside 0..{self.frame_count}"
            )
        return format_timecode(self.start_frame + local_frame, self.spec)[:8]


@dataclass(frozen=True)
class DeepLinkContext:
    farm_id: str
    camera_label: str
    camera_id: str


def _strict_int(value: object, context: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool):
        raise DataContractError(f"{context} must be an integer")
    text = str(value).strip()
    if re.fullmatch(r"-?\d+", text) is None:
        raise DataContractError(f"{context} must be an integer, got {value!r}")
    result = int(text)
    if result < minimum:
        raise DataContractError(f"{context} must be at least {minimum}, got {result}")
    return result


def _deep_link_context(manifest: Mapping[str, Any]) -> DeepLinkContext:
    farm_id = str(_strict_int(manifest.get("farm"), "manifest farm", minimum=1))
    camera_label = str(manifest.get("camera", "")).strip()
    camera_match = re.fullmatch(r"Gopro(?P<camera_id>\d+)", camera_label)
    if camera_match is None:
        raise DataContractError(
            f"Input manifest has an unsupported camera label: {camera_label!r}"
        )
    camera_id = str(int(camera_match.group("camera_id")))
    return DeepLinkContext(
        farm_id=farm_id,
        camera_label=camera_label,
        camera_id=camera_id,
    )


def _webuil_href(
    context: DeepLinkContext,
    sample_id: str,
    source_clip_id: str,
    segment: int,
    local_frame: int,
) -> str:
    if not sample_id or sample_id != sample_id.strip():
        raise DataContractError("WebUIL deep link sample_id must be non-empty and trimmed")
    segment_id = f"{source_clip_id}-{segment}-{local_frame}"
    query = urlencode(
        [
            ("farmID", context.farm_id),
            ("cameraID", context.camera_id),
            ("clipID", sample_id),
            ("segmentID", segment_id),
            ("frameID", str(local_frame)),
        ]
    )
    return f"{WEBUIL_DEEP_LINK_BASE_URL}?{query}"


def _make_timecode_spec(rate: Fraction, separator: str) -> TimecodeSpec:
    if separator == ";":
        supported = {
            Fraction(30_000, 1_001): (30, 2),
            Fraction(60_000, 1_001): (60, 4),
        }
        if rate not in supported:
            raise DataContractError(f"Unsupported drop-frame rate: {rate}")
        nominal, dropped = supported[rate]
        return TimecodeSpec(rate, nominal, dropped, separator)
    if separator == ":":
        supported = {
            Fraction(24, 1): 24,
            Fraction(24_000, 1_001): 24,
            Fraction(25, 1): 25,
            Fraction(30, 1): 30,
            Fraction(30_000, 1_001): 30,
            Fraction(50, 1): 50,
            Fraction(60, 1): 60,
            Fraction(60_000, 1_001): 60,
        }
        if rate not in supported:
            raise DataContractError(f"Unsupported non-drop frame rate: {rate}")
        return TimecodeSpec(rate, supported[rate], 0, separator)
    raise DataContractError(f"Unsupported timecode separator: {separator!r}")


def _timecode_spec(value: str, rate_text: str, context: str) -> TimecodeSpec:
    match = TIMECODE_RE.fullmatch(value)
    if match is None:
        raise DataContractError(f"{context}: malformed SMPTE timecode {value!r}")
    try:
        rate = Fraction(rate_text)
    except (ValueError, ZeroDivisionError) as exc:
        raise DataContractError(f"{context}: invalid frame rate {rate_text!r}") from exc
    return _make_timecode_spec(rate, match.group("separator"))


def parse_timecode(value: str, spec: TimecodeSpec) -> int:
    match = TIMECODE_RE.fullmatch(value)
    if match is None or match.group("separator") != spec.separator:
        raise DataContractError(f"Malformed or mode-mismatched SMPTE timecode: {value!r}")
    hour = int(match.group("hour"))
    minute = int(match.group("minute"))
    second = int(match.group("second"))
    frame = int(match.group("frame"))
    if hour > 23 or frame >= spec.nominal_fps:
        raise DataContractError(f"Out-of-range SMPTE timecode: {value!r}")
    if (
        spec.drop_frames
        and minute % 10 != 0
        and second == 0
        and frame < spec.drop_frames
    ):
        raise DataContractError(f"SMPTE timecode uses an omitted drop-frame label: {value!r}")
    total_minutes = hour * 60 + minute
    nominal = ((hour * 60 * 60 + minute * 60 + second) * spec.nominal_fps) + frame
    return nominal - spec.drop_frames * (total_minutes - total_minutes // 10)


def format_timecode(frame_index: int, spec: TimecodeSpec) -> str:
    if not 0 <= frame_index < spec.frames_per_24_hours:
        raise DataContractError(
            f"Frame index {frame_index} would wrap the 24-hour SMPTE clock"
        )
    nominal = frame_index
    if spec.drop_frames:
        frames_per_ten = spec.nominal_fps * 60 * 10 - spec.drop_frames * 9
        frames_per_drop_minute = spec.nominal_fps * 60 - spec.drop_frames
        ten_blocks, remainder = divmod(frame_index, frames_per_ten)
        nominal += spec.drop_frames * 9 * ten_blocks
        if remainder >= spec.drop_frames:
            nominal += spec.drop_frames * (
                (remainder - spec.drop_frames) // frames_per_drop_minute
            )
    frame = nominal % spec.nominal_fps
    total_seconds = nominal // spec.nominal_fps
    second = total_seconds % 60
    total_minutes = total_seconds // 60
    minute = total_minutes % 60
    hour = total_minutes // 60
    return f"{hour:02d}:{minute:02d}:{second:02d}{spec.separator}{frame:02d}"


def _load_clip_clocks(
    path: Path,
    expected_farm: str,
    expected_camera: str,
) -> dict[str, ClipClock]:
    if not path.is_file():
        raise FileNotFoundError(f"Embedded-time sequence CSV is missing: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        headers = reader.fieldnames or []
        if len(headers) != len(set(headers)):
            raise DataContractError(f"Time-sequence CSV has duplicate columns: {path}")
        missing = sorted(TIME_SEQUENCE_COLUMNS - set(headers))
        if missing:
            raise DataContractError(f"Time-sequence CSV is missing columns: {missing}")
        rows = list(reader)
    if not rows:
        raise DataContractError(f"Time-sequence CSV contains no clips: {path}")

    parsed: list[tuple[int, str, int, int, TimecodeSpec]] = []
    sequence_ids: set[str] = set()
    seen_clips: set[str] = set()
    recording_dates: set[str] = set()
    for row_number, row in enumerate(rows, start=2):
        context = f"{path}:{row_number}"
        if str(row["farm_id"]).strip() != expected_farm:
            raise DataContractError(f"{context}: unexpected farm_id")
        if str(row["camera_id"]).strip() != expected_camera:
            raise DataContractError(f"{context}: unexpected camera_id")
        sequence_ids.add(str(row["sequence_id"]).strip())
        recording_dates.add(str(row["recording_date"]).strip())
        order = _strict_int(row["sequence_order"], f"{context} sequence_order", minimum=1)
        frame_count = _strict_int(row["frame_count"], f"{context} frame_count", minimum=1)
        file_name = str(row["file_name"]).strip()
        clip = Path(file_name).stem
        if not clip or clip in seen_clips:
            raise DataContractError(f"{context}: empty or duplicate clip {clip!r}")
        seen_clips.add(clip)
        if Path(str(row["source_relative_path"]).strip()).name != file_name:
            raise DataContractError(f"{context}: file_name does not match source_relative_path")
        start_label = str(row["start_timecode"]).strip()
        spec = _timecode_spec(
            start_label,
            str(row["frame_rate"]).strip(),
            context,
        )
        start = parse_timecode(start_label, spec)
        last = parse_timecode(str(row["last_frame_timecode"]).strip(), spec)
        end = parse_timecode(str(row["end_timecode_exclusive"]).strip(), spec)
        if start + frame_count - 1 != last or start + frame_count != end:
            raise DataContractError(
                f"{context}: timecode endpoints do not match frame_count={frame_count}"
            )
        parsed.append((order, clip, frame_count, start, spec))

    if len(sequence_ids) != 1 or "" in sequence_ids:
        raise DataContractError("Time-sequence rows must share one non-empty sequence_id")
    if len(recording_dates) != 1 or "" in recording_dates:
        raise DataContractError("Time-sequence rows must share one non-empty recording_date")
    parsed.sort(key=lambda item: item[0])
    if [item[0] for item in parsed] != list(range(1, len(parsed) + 1)):
        raise DataContractError("Time-sequence sequence_order must be contiguous from 1")

    clocks: dict[str, ClipClock] = {}
    timeline_offset = 0
    previous_end: int | None = None
    previous_spec: TimecodeSpec | None = None
    for order, clip, frame_count, start, spec in parsed:
        if previous_end is not None and (spec != previous_spec or start != previous_end):
            raise DataContractError(
                f"{clip}: embedded timecode does not continue from the previous clip"
            )
        clocks[clip] = ClipClock(
            clip=clip,
            sequence_order=order,
            frame_count=frame_count,
            timeline_offset=timeline_offset,
            start_frame=start,
            spec=spec,
        )
        timeline_offset += frame_count
        previous_end = start + frame_count
        previous_spec = spec
    return clocks


def _file_signature(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required appearance source is missing: {path}")
    stat = path.stat()
    return {
        "path": str(path),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


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


def _manifest_samples(
    manifest: Mapping[str, Any],
    clocks: Mapping[str, ClipClock],
) -> tuple[dict[str, dict[str, Any]], int]:
    samples_value = manifest.get("samples")
    if not isinstance(samples_value, list) or not samples_value:
        raise DataContractError("Input generation manifest has no samples")
    sample_map: dict[str, dict[str, Any]] = {}
    clip_ranges: dict[str, list[tuple[int, int, int, int]]] = {
        clip: [] for clip in clocks
    }
    expected_rows = 0
    canonical_count = 0
    for index, value in enumerate(samples_value):
        if not isinstance(value, dict):
            raise DataContractError(f"Input manifest sample {index} is not an object")
        sample_id = str(value.get("sample_id", "")).strip()
        clip = str(value.get("source_clip_id", "")).strip()
        if not sample_id or sample_id in sample_map:
            raise DataContractError(f"Input manifest has an empty or duplicate sample_id: {sample_id!r}")
        if clip not in clocks:
            raise DataContractError(f"Input manifest sample {sample_id} has unknown clip {clip!r}")
        start = _strict_int(value.get("canonical_start_frame"), f"{sample_id} canonical_start_frame")
        end = _strict_int(value.get("canonical_end_frame"), f"{sample_id} canonical_end_frame")
        count = _strict_int(value.get("canonical_frame_count"), f"{sample_id} canonical_frame_count", minimum=1)
        rows = _strict_int(value.get("trajectory_row_count"), f"{sample_id} trajectory_row_count", minimum=1)
        segment = _strict_int(value.get("shard_index"), f"{sample_id} shard_index", minimum=1)
        segment_count = _strict_int(
            value.get("shard_count"), f"{sample_id} shard_count", minimum=1
        )
        if segment > segment_count:
            raise DataContractError(
                f"{sample_id}: shard_index {segment} exceeds shard_count {segment_count}"
            )
        if end < start or end - start + 1 != count:
            raise DataContractError(f"{sample_id}: canonical frame range/count mismatch")
        if end >= clocks[clip].frame_count:
            raise DataContractError(f"{sample_id}: canonical range exceeds physical clip")
        sample_map[sample_id] = {
            "clip": clip,
            "segment": segment,
            "start": start,
            "end": end,
            "trajectory_rows": rows,
        }
        clip_ranges[clip].append((start, end, segment, segment_count))
        expected_rows += rows
        canonical_count += count

    for clip, ranges in clip_ranges.items():
        if not ranges:
            raise DataContractError(f"Input manifest has no canonical range for clip {clip}")
        ranges.sort(key=lambda item: item[0])
        for previous, current in zip(ranges, ranges[1:]):
            if current[0] != previous[1] + 1:
                raise DataContractError(f"{clip}: canonical sample ranges overlap or have an internal gap")
        segment_numbers = [item[2] for item in ranges]
        segment_counts = {item[3] for item in ranges}
        if (
            len(segment_counts) != 1
            or next(iter(segment_counts)) != len(ranges)
            or segment_numbers != list(range(1, len(ranges) + 1))
        ):
            raise DataContractError(
                f"{clip}: shard indices/count do not form contiguous segments from 1"
            )
    declared_canonical = _strict_int(
        manifest.get("canonical_frame_count"),
        "manifest canonical_frame_count",
        minimum=1,
    )
    if canonical_count != declared_canonical:
        raise DataContractError(
            f"Manifest canonical frame count mismatch: samples={canonical_count}, declared={declared_canonical}"
        )
    declared_rows = _strict_int(
        manifest.get("trajectory_row_count"),
        "manifest trajectory_row_count",
        minimum=1,
    )
    if expected_rows != declared_rows:
        raise DataContractError(
            f"Manifest trajectory row count mismatch: samples={expected_rows}, declared={declared_rows}"
        )
    return sample_map, declared_rows


def _integer_array(series: pd.Series, context: str) -> np.ndarray:
    numeric = pd.to_numeric(series, errors="coerce")
    values = numeric.to_numpy(dtype=float)
    if not np.isfinite(values).all() or not np.array_equal(values, np.rint(values)):
        raise DataContractError(f"Trajectories contain an invalid integer in {context}")
    return np.rint(values).astype(np.int64)


def _canonical_sample_for_frame(
    clip: str,
    frame: int,
    sample_map: Mapping[str, Mapping[str, Any]],
) -> tuple[str, Mapping[str, Any]]:
    matches = [
        (sample_id, spec)
        for sample_id, spec in sample_map.items()
        if str(spec["clip"]) == clip
        and int(spec["start"]) <= frame <= int(spec["end"])
    ]
    if len(matches) != 1:
        raise DataContractError(
            f"{clip}: frame {frame} must resolve to exactly one canonical segment"
        )
    return matches[0]


def _appearance_row(
    clip: str,
    segment: int,
    start: int,
    end: int,
    clock: ClipClock,
    sample_id: str,
    link_context: DeepLinkContext,
) -> dict[str, Any]:
    frame_count = end - start
    if frame_count < 1:
        raise DataContractError(f"{clip}: appearance interval has no frames")
    start_time = clock.hhmmss(start)
    end_time = clock.hhmmss(end)
    return {
        "clip": clip,
        "segment": segment,
        "startFrame": start,
        "endFrameExclusive": end,
        "frameCount": frame_count,
        "startTime": start_time,
        "endTime": end_time,
        "display": f"{start_time}-{end_time}",
        "href": _webuil_href(
            link_context,
            sample_id,
            clip,
            segment,
            start,
        ),
    }


def _build_from_trajectories(
    path: Path,
    manifest: Mapping[str, Any],
    identity_map: Mapping[str, str],
    clocks: Mapping[str, ClipClock],
    sample_map: Mapping[str, Mapping[str, Any]],
    link_context: DeepLinkContext,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, int]]:
    expected_display = {f"G{index:04d}" for index in range(1, 63)}
    if set(identity_map.values()) != expected_display or len(identity_map) != 62:
        raise DataContractError("Identity map must be a bijection onto G0001..G0062")
    if len(set(identity_map.keys())) != len(identity_map):
        raise DataContractError("Identity map contains duplicate raw identities")

    expected_farm = link_context.farm_id
    expected_camera = link_context.camera_label
    expected_combined = str(manifest.get("combined_sample_id", "")).strip()
    if not expected_combined:
        raise DataContractError("Input manifest is missing combined_sample_id")

    intervals: dict[str, list[tuple[str, int, int, int]]] = {
        cow: [] for cow in sorted(expected_display)
    }
    active: dict[str, tuple[str, int, int, int]] = {}
    last_cow_frame: dict[str, int] = {}
    sample_counts = {sample_id: 0 for sample_id in sample_map}
    row_count = 0
    frozen_count = 0
    distinct_frames = 0
    last_global_frame: int | None = None

    string_dtypes = {
        "farm": "string",
        "camera": "string",
        "clip": "string",
        "cow_id": "string",
        "frozen": "string",
        "display_global_id": "string",
        "source_clip_id": "string",
        "source_sample_id": "string",
    }
    try:
        chunks = pd.read_csv(
            path,
            usecols=list(TRAJECTORY_COLUMNS),
            dtype=string_dtypes,
            keep_default_na=False,
            chunksize=250_000,
        )
        for chunk_number, chunk in enumerate(chunks, start=1):
            if chunk.empty:
                continue
            frames = _integer_array(chunk["frame"], "frame")
            local_frames = _integer_array(chunk["local_frame"], "local_frame")
            if (frames < 0).any() or (local_frames < 0).any():
                raise DataContractError("Trajectories contain a negative frame index")
            if (np.diff(frames) < 0).any() or (
                last_global_frame is not None and frames[0] < last_global_frame
            ):
                raise DataContractError("Trajectories are not ordered by global frame")
            distinct_frames += int(np.count_nonzero(np.diff(frames)))
            if last_global_frame is None or frames[0] != last_global_frame:
                distinct_frames += 1
            last_global_frame = int(frames[-1])

            raw_cows = chunk["cow_id"].astype(str)
            displays = chunk["display_global_id"].astype(str)
            mapped = raw_cows.map(identity_map)
            if mapped.isna().any():
                unknown = sorted(set(raw_cows[mapped.isna()].tolist()))
                raise DataContractError(f"Trajectories contain unknown UUID identities: {unknown[:5]}")
            if not np.array_equal(mapped.to_numpy(dtype=str), displays.to_numpy(dtype=str)):
                raise DataContractError(
                    "Trajectory UUID and display_global_id mappings disagree"
                )
            if not displays.map(lambda value: DISPLAY_ID_RE.fullmatch(value) is not None).all():
                raise DataContractError("Trajectories contain a malformed display_global_id")

            if not chunk["farm"].astype(str).eq(expected_farm).all():
                raise DataContractError("Trajectories contain an unexpected farm")
            if not chunk["camera"].astype(str).eq(expected_camera).all():
                raise DataContractError("Trajectories contain an unexpected camera")
            if not chunk["clip"].astype(str).eq(expected_combined).all():
                raise DataContractError("Trajectories contain an unexpected combined sample id")

            source_samples = chunk["source_sample_id"].astype(str)
            sample_specs = source_samples.map(sample_map)
            if sample_specs.isna().any():
                unknown = sorted(set(source_samples[sample_specs.isna()].tolist()))
                raise DataContractError(f"Trajectories contain unknown source_sample_id values: {unknown}")
            source_clips = chunk["source_clip_id"].astype(str)
            expected_clips = source_samples.map(
                {sample_id: str(spec["clip"]) for sample_id, spec in sample_map.items()}
            )
            if not np.array_equal(source_clips.to_numpy(dtype=str), expected_clips.to_numpy(dtype=str)):
                raise DataContractError("source_sample_id and source_clip_id mappings disagree")
            starts = source_samples.map(
                {sample_id: int(spec["start"]) for sample_id, spec in sample_map.items()}
            ).to_numpy(dtype=np.int64)
            ends = source_samples.map(
                {sample_id: int(spec["end"]) for sample_id, spec in sample_map.items()}
            ).to_numpy(dtype=np.int64)
            source_segments = source_samples.map(
                {sample_id: int(spec["segment"]) for sample_id, spec in sample_map.items()}
            ).to_numpy(dtype=np.int64)
            if (local_frames < starts).any() or (local_frames > ends).any():
                raise DataContractError("Trajectory local_frame is outside its canonical sample range")
            clip_counts = source_clips.map(
                {clip: clock.frame_count for clip, clock in clocks.items()}
            )
            offsets = source_clips.map(
                {clip: clock.timeline_offset for clip, clock in clocks.items()}
            )
            if clip_counts.isna().any() or offsets.isna().any():
                raise DataContractError("Trajectories contain a clip absent from the time sequence")
            if (local_frames >= clip_counts.to_numpy(dtype=np.int64)).any():
                raise DataContractError("Trajectory local_frame exceeds its physical clip")
            expected_frames = offsets.to_numpy(dtype=np.int64) + local_frames
            if not np.array_equal(frames, expected_frames):
                raise DataContractError("Global frame does not equal clip offset plus local_frame")

            frozen_values = chunk["frozen"].astype(str).str.lower()
            if not frozen_values.isin({"true", "false"}).all():
                raise DataContractError("Trajectories contain an invalid frozen flag")
            frozen_count += int(frozen_values.eq("true").sum())
            row_count += len(chunk)
            counts = source_samples.value_counts()
            for sample_id, count in counts.items():
                sample_counts[str(sample_id)] += int(count)

            work = pd.DataFrame(
                {
                    "display": displays.to_numpy(dtype=str),
                    "source_clip": source_clips.to_numpy(dtype=str),
                    "source_segment": source_segments,
                    "local_frame": local_frames,
                    "global_frame": frames,
                }
            )
            for cow, rows in work.groupby("display", sort=False):
                local = rows["local_frame"].to_numpy(dtype=np.int64)
                global_values = rows["global_frame"].to_numpy(dtype=np.int64)
                clips = rows["source_clip"].to_numpy(dtype=str)
                source_segment_values = rows["source_segment"].to_numpy(dtype=np.int64)
                previous = last_cow_frame.get(cow)
                if (np.diff(global_values) <= 0).any() or (
                    previous is not None and global_values[0] <= previous
                ):
                    raise DataContractError(f"{cow}: duplicate or out-of-order displayed frame")
                last_cow_frame[cow] = int(global_values[-1])
                breaks = np.flatnonzero(
                    (clips[1:] != clips[:-1]) | (local[1:] != local[:-1] + 1)
                ) + 1
                segment_starts = np.concatenate(([0], breaks))
                segment_ends = np.concatenate((breaks, [len(rows)]))
                for segment_start, segment_end in zip(segment_starts, segment_ends):
                    clip = str(clips[segment_start])
                    source_segment = int(source_segment_values[segment_start])
                    start = int(local[segment_start])
                    end = int(local[segment_end - 1]) + 1
                    current = active.get(cow)
                    if current is not None and current[0] == clip and current[3] == start:
                        active[cow] = (clip, current[1], current[2], end)
                    else:
                        if current is not None:
                            intervals[cow].append(current)
                        active[cow] = (clip, source_segment, start, end)
    except (ValueError, TypeError) as exc:
        raise DataContractError(f"Could not parse WebUIL trajectories: {path}") from exc

    for cow, current in active.items():
        intervals[cow].append(current)

    declared_rows = _strict_int(
        manifest.get("trajectory_row_count"), "manifest trajectory_row_count", minimum=1
    )
    declared_frozen = _strict_int(
        manifest.get("frozen_trajectory_row_count"),
        "manifest frozen_trajectory_row_count",
    )
    declared_frames = _strict_int(
        manifest.get("trajectory_frame_count"),
        "manifest trajectory_frame_count",
        minimum=1,
    )
    if row_count != declared_rows:
        raise DataContractError(
            f"Trajectory row count mismatch: observed={row_count}, declared={declared_rows}"
        )
    if frozen_count != declared_frozen:
        raise DataContractError(
            f"Frozen trajectory row count mismatch: observed={frozen_count}, declared={declared_frozen}"
        )
    if distinct_frames != declared_frames:
        raise DataContractError(
            f"Trajectory frame count mismatch: observed={distinct_frames}, declared={declared_frames}"
        )
    for sample_id, spec in sample_map.items():
        observed = sample_counts[sample_id]
        expected = int(spec["trajectory_rows"])
        if observed != expected:
            raise DataContractError(
                f"{sample_id}: trajectory row count mismatch observed={observed}, expected={expected}"
            )
    if set(active) != expected_display:
        missing = sorted(expected_display - set(active))
        raise DataContractError(f"Trajectories do not display all 62 cattle: missing={missing}")

    result: dict[str, list[dict[str, Any]]] = {}
    for cow in sorted(expected_display):
        values: list[dict[str, Any]] = []
        for clip, segment, start, end in intervals[cow]:
            sample_id, sample_spec = _canonical_sample_for_frame(
                clip,
                start,
                sample_map,
            )
            if int(sample_spec["segment"]) != segment:
                raise DataContractError(
                    f"{clip}: appearance start frame {start} disagrees with its segment"
                )
            values.append(
                _appearance_row(
                    clip,
                    segment,
                    start,
                    end,
                    clocks[clip],
                    sample_id,
                    link_context,
                )
            )
        values.sort(
            key=lambda row: (
                -int(row["frameCount"]),
                clocks[str(row["clip"])].sequence_order,
                int(row["startFrame"]),
            )
        )
        if not values:
            raise DataContractError(f"{cow}: no displayed appearance intervals")
        result[cow] = values
    return result, {
        "sourceRowCount": row_count,
        "frozenRowCount": frozen_count,
        "sourceFrameCount": distinct_frames,
        "intervalCount": sum(len(values) for values in result.values()),
    }


def _validate_cached_index(
    document: Mapping[str, Any],
    generation_id: str,
    clocks: Mapping[str, ClipClock],
    sample_map: Mapping[str, Mapping[str, Any]],
    link_context: DeepLinkContext,
) -> dict[str, tuple[dict[str, Any], ...]]:
    if document.get("schemaVersion") != APPEARANCE_CACHE_SCHEMA:
        raise DataContractError("Appearance cache has an unexpected schemaVersion")
    if document.get("generationId") != generation_id:
        raise DataContractError("Appearance cache generationId does not match")
    cows = document.get("cows")
    expected = {f"G{index:04d}" for index in range(1, 63)}
    if not isinstance(cows, dict) or set(cows) != expected:
        raise DataContractError("Appearance cache must contain exactly G0001..G0062")
    validated: dict[str, tuple[dict[str, Any], ...]] = {}
    expected_keys = {
        "clip",
        "segment",
        "startFrame",
        "endFrameExclusive",
        "frameCount",
        "startTime",
        "endTime",
        "display",
        "href",
    }
    for cow in sorted(expected):
        values = cows[cow]
        if not isinstance(values, list) or not values:
            raise DataContractError(f"Appearance cache has no intervals for {cow}")
        checked: list[dict[str, Any]] = []
        prior_sort: tuple[int, int, int] | None = None
        for row in values:
            if not isinstance(row, dict) or set(row) != expected_keys:
                raise DataContractError(f"Appearance cache has a malformed interval for {cow}")
            clip = row["clip"]
            if not isinstance(clip, str) or clip not in clocks:
                raise DataContractError(f"Appearance cache has an unknown clip for {cow}")
            for key in ("segment", "startFrame", "endFrameExclusive", "frameCount"):
                if isinstance(row[key], bool) or not isinstance(row[key], int):
                    raise DataContractError(f"Appearance cache {cow}.{key} is not an integer")
            start = int(row["startFrame"])
            end = int(row["endFrameExclusive"])
            sample_id, sample_spec = _canonical_sample_for_frame(clip, start, sample_map)
            segment = int(sample_spec["segment"])
            expected_row = _appearance_row(
                clip,
                segment,
                start,
                end,
                clocks[clip],
                sample_id,
                link_context,
            )
            if row != expected_row:
                raise DataContractError(f"Appearance cache interval fields disagree for {cow}")
            sort_key = (-int(row["frameCount"]), clocks[clip].sequence_order, start)
            if prior_sort is not None and sort_key < prior_sort:
                raise DataContractError(f"Appearance cache intervals are not sorted for {cow}")
            prior_sort = sort_key
            checked.append(dict(row))
        validated[cow] = tuple(checked)
    return validated


class AppearanceIndex:
    def __init__(
        self,
        by_cow: Mapping[str, tuple[dict[str, Any], ...]],
        stats: Mapping[str, int],
    ) -> None:
        self._by_cow = dict(by_cow)
        self.stats = dict(stats)

    @classmethod
    def load_or_build(
        cls,
        *,
        trajectories_path: Path,
        manifest_path: Path,
        identity_path: Path,
        time_sequence_path: Path,
        cache_path: Path,
        cache_meta_path: Path,
        generation_id: str,
        identity_map: Mapping[str, str],
    ) -> "AppearanceIndex":
        manifest = _read_json_object(manifest_path, "input generation manifest")
        if str(manifest.get("generation_id", "")).strip() != generation_id:
            raise DataContractError("Appearance source manifest generation_id does not match")
        if _strict_int(manifest.get("global_identity_count"), "global_identity_count", minimum=1) != 62:
            raise DataContractError("Appearance source manifest must declare 62 global identities")
        link_context = _deep_link_context(manifest)
        clocks = _load_clip_clocks(
            time_sequence_path,
            link_context.farm_id,
            link_context.camera_id,
        )
        sample_map, _ = _manifest_samples(manifest, clocks)

        signature = {
            "schemaVersion": APPEARANCE_CACHE_SCHEMA,
            "contract": APPEARANCE_CONTRACT,
            "deepLinkBaseUrl": WEBUIL_DEEP_LINK_BASE_URL,
            "generationId": generation_id,
            "trajectories": _file_signature(trajectories_path),
            "manifest": _file_signature(manifest_path),
            "identities": _file_signature(identity_path),
            "timeSequence": _file_signature(time_sequence_path),
        }
        cache_exists = cache_path.is_file()
        meta_exists = cache_meta_path.is_file()
        # The index is wholly derived from validated sources.  If a process was
        # interrupted between the two atomic replacements, rebuild the pair
        # instead of making that recoverable partial-cache state permanent.
        metadata = (
            _read_json_object(cache_meta_path, "appearance cache metadata")
            if cache_exists and meta_exists
            else None
        )
        if metadata is not None and all(metadata.get(key) == value for key, value in signature.items()):
            checksum = metadata.get("cacheSha256")
            if not isinstance(checksum, str) or re.fullmatch(r"[0-9a-f]{64}", checksum) is None:
                raise DataContractError("Appearance cache metadata has an invalid SHA-256")
            if _sha256(cache_path) != checksum:
                raise DataContractError("Appearance cache checksum does not match its metadata")
            document = _read_json_object(cache_path, "appearance cache")
            validated = _validate_cached_index(
                document,
                generation_id,
                clocks,
                sample_map,
                link_context,
            )
            stats_value = document.get("stats")
            if not isinstance(stats_value, dict) or not all(
                isinstance(value, int) and not isinstance(value, bool) and value >= 0
                for value in stats_value.values()
            ):
                raise DataContractError("Appearance cache has invalid stats")
            if stats_value.get("intervalCount") != sum(len(rows) for rows in validated.values()):
                raise DataContractError("Appearance cache intervalCount does not match its rows")
            return cls(validated, stats_value)

        print(
            f"Building displayed-appearance index from {trajectories_path} ...",
            flush=True,
        )
        by_cow, stats = _build_from_trajectories(
            trajectories_path,
            manifest,
            identity_map,
            clocks,
            sample_map,
            link_context,
        )
        document = {
            "schemaVersion": APPEARANCE_CACHE_SCHEMA,
            "generationId": generation_id,
            "stats": stats,
            "cows": by_cow,
        }
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_tmp = cache_path.with_name(f"{cache_path.name}.tmp")
        meta_tmp = cache_meta_path.with_name(f"{cache_meta_path.name}.tmp")
        try:
            with cache_tmp.open("w", encoding="utf-8") as handle:
                json.dump(document, handle, ensure_ascii=False, separators=(",", ":"))
                handle.write("\n")
            metadata = dict(signature)
            metadata["cacheSha256"] = _sha256(cache_tmp)
            metadata["intervalCount"] = stats["intervalCount"]
            with meta_tmp.open("w", encoding="utf-8") as handle:
                json.dump(metadata, handle, ensure_ascii=False, indent=2, sort_keys=True)
                handle.write("\n")
            os.replace(cache_tmp, cache_path)
            os.replace(meta_tmp, cache_meta_path)
        finally:
            for temporary in (cache_tmp, meta_tmp):
                try:
                    temporary.unlink()
                except FileNotFoundError:
                    pass
        validated = _validate_cached_index(
            document,
            generation_id,
            clocks,
            sample_map,
            link_context,
        )
        print(
            "Displayed-appearance index ready: "
            f"intervals={stats['intervalCount']:,}, path={cache_path}",
            flush=True,
        )
        return cls(validated, stats)

    def for_cow(self, cow_id: str) -> list[dict[str, Any]]:
        try:
            rows = self._by_cow[cow_id]
        except KeyError as exc:
            raise ValueError(f"Unknown cattle identity: {cow_id}") from exc
        return [dict(row) for row in rows]
