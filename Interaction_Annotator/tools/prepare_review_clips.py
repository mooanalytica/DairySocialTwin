from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


WORKSPACE = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOTS = [
    Path(r"F:\V0531_R1_S1_S2"),
    Path(r"F:\V0601_S1_S2"),
    Path(r"F:\V0603_S1_S2"),
]
DEFAULT_SOURCE_ROOT = Path(r"F:\FULLDATA\Dairy Farm Videos")
OUTPUT_ROOT = WORKSPACE / "output"
CACHED_VIDEOS_ROOT = OUTPUT_ROOT / "cached_videos"
CACHE_VIS_ROOT = OUTPUT_ROOT / "cache_vis"
VIDEOS_ROOT = OUTPUT_ROOT / "videos"
ANNOTATIONS_ROOT = OUTPUT_ROOT / "annotations"
LOG_ROOT = OUTPUT_ROOT / "logs"
INDEX_PATH = OUTPUT_ROOT / "index.csv"
PLAN_PATH = OUTPUT_ROOT / "clip_plan.csv"
DISCARDED_PATH = OUTPUT_ROOT / "discarded_clips.txt"
LOG_FILE_HANDLE = None

MIN_CLIP_SECONDS = 15.0
MAX_CLIP_SECONDS = 60.0
DROP_SEGMENT_OVER_SECONDS = 100.0

INDEX_COLUMNS = [
    "clip_id",
    "source_video_id",
    "source_video_root",
    "source_video_rel_path",
    "source_video_path",
    "cache_root",
    "cache_rel_path",
    "cache_path",
    "cache_vis_root",
    "cache_vis_rel_path",
    "cache_vis_path",
    "clip_root",
    "clip_rel_path",
    "clip_path",
    "annotation_root",
    "annotation_rel_path",
    "annotation_path",
    "interaction_csv_root",
    "interaction_csv_rel_path",
    "interaction_class",
    "csv_root",
    "csv_rel_path",
    "source_start_frame",
    "source_end_frame",
    "clip_frame_offset",
    "fps",
    "width",
    "height",
    "source_raw_width",
    "source_raw_height",
    "source_rotation_degrees",
    "selection_pairs",
    "selection_track_ids",
    "duration_s",
    "cache_source_start_frame",
    "cache_source_end_frame",
    "cache_duration_s",
    "selected_cache_start_frame",
    "selected_cache_end_frame",
    "is_saved",
]

ANNOTATION_COLUMNS = [
    "event_id",
    "event_group_id",
    "clip_id",
    "clip_root",
    "clip_rel_path",
    "clip_path",
    "csv_root",
    "csv_rel_path",
    "start_frame",
    "end_frame",
    "roi_x",
    "roi_y",
    "roi_w",
    "roi_h",
    "valence",
    "fine_class",
    "allow_duplicate_pair",
    "fps",
    "width",
    "height",
    "start_time_s",
    "end_time_s",
    "annotator",
    "label_confidence",
    "notes",
    "exclude",
]


def log(message: str) -> None:
    print(message, flush=True)
    if LOG_FILE_HANDLE is not None:
        LOG_FILE_HANDLE.write(message + "\n")
        LOG_FILE_HANDLE.flush()
        os.fsync(LOG_FILE_HANDLE.fileno())


def ensure_dirs() -> None:
    for path in [OUTPUT_ROOT, CACHED_VIDEOS_ROOT, CACHE_VIS_ROOT, VIDEOS_ROOT, ANNOTATIONS_ROOT, LOG_ROOT]:
        path.mkdir(parents=True, exist_ok=True)


def parse_fraction(value: str | None) -> float | None:
    if not value:
        return None
    if "/" in value:
        left, right = value.split("/", 1)
        denominator = float(right)
        if denominator == 0:
            return None
        return float(left) / denominator
    return float(value)


def stream_rotation_degrees(stream: dict[str, Any]) -> int:
    rotate = stream.get("tags", {}).get("rotate")
    if rotate not in (None, ""):
        return int(float(rotate))
    for item in stream.get("side_data_list", []) or []:
        if item.get("rotation") not in (None, ""):
            return int(float(item["rotation"]))
    return 0


def run_ffprobe(path: Path) -> dict[str, Any]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_streams",
        "-show_format",
        str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {path}: {result.stderr.strip()}")
    return json.loads(result.stdout)


def probe_video(path: Path, fallback_fps: float, fallback_duration: float) -> tuple[float, int, int, int, int, int, int]:
    data = run_ffprobe(path)
    video_stream = None
    for stream in data.get("streams", []):
        if stream.get("codec_type") == "video":
            video_stream = stream
            break
    if not video_stream:
        raise RuntimeError(f"No video stream found: {path}")

    fps = (
        parse_fraction(video_stream.get("avg_frame_rate"))
        or parse_fraction(video_stream.get("r_frame_rate"))
        or fallback_fps
    )
    raw_width = int(video_stream["width"])
    raw_height = int(video_stream["height"])
    rotation = stream_rotation_degrees(video_stream)
    width = raw_width
    height = raw_height
    if abs(rotation) % 180 == 90:
        width, height = height, width
    duration = (
        float(video_stream.get("duration") or 0)
        or float(data.get("format", {}).get("duration") or 0)
        or fallback_duration
    )
    frame_count_text = video_stream.get("nb_frames")
    if frame_count_text and str(frame_count_text).isdigit():
        frame_count = int(frame_count_text)
    else:
        frame_count = max(1, int(math.floor(duration * fps)))
    return fps, width, height, frame_count, raw_width, raw_height, rotation


def source_rel_from_manifest(source_path: str) -> Path:
    normalized = source_path.replace("\\", "/")
    marker = "Dairy Farm Videos/"
    if marker not in normalized:
        raise ValueError(f"Manifest source path does not contain {marker!r}: {source_path}")
    rel_posix = normalized.split(marker, 1)[1].lstrip("/")
    parts = PurePosixPath(rel_posix).parts
    return Path(*parts)


def stable_video_id(rel_path: Path) -> str:
    digest = hashlib.sha1(str(rel_path).replace("\\", "/").encode("utf-8")).hexdigest()[:12]
    stem = rel_path.stem.upper()
    return f"SV_{stem}_{digest}"


def safe_rel(path: Path, root: Path) -> str:
    return str(path.resolve().relative_to(root.resolve()))


def read_discarded_clip_ids() -> set[str]:
    if not DISCARDED_PATH.exists():
        return set()
    with DISCARDED_PATH.open("r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip() and not line.lstrip().startswith("#")}


def merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged: list[list[int]] = []
    for start, end in intervals:
        if not merged or start > merged[-1][1] + 1:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(start, end) for start, end in merged]


def expand_interval(start: int, end: int, min_frames: int, total_frames: int) -> tuple[int, int]:
    total_frames = max(1, total_frames)
    start = max(0, min(start, total_frames - 1))
    end = max(start, min(end, total_frames - 1))
    length = end - start + 1
    if length >= min_frames:
        return start, end

    target = min(min_frames, total_frames)
    extra = target - length
    left = extra // 2
    right = extra - left
    new_start = start - left
    new_end = end + right

    if new_start < 0:
        new_end += -new_start
        new_start = 0
    if new_end > total_frames - 1:
        new_start -= new_end - (total_frames - 1)
        new_end = total_frames - 1
    new_start = max(0, new_start)
    new_end = min(total_frames - 1, new_end)
    return new_start, new_end


def split_interval(start: int, end: int, max_frames: int) -> list[tuple[int, int]]:
    length = end - start + 1
    if length <= max_frames:
        return [(start, end)]
    parts = math.ceil(length / max_frames)
    split: list[tuple[int, int]] = []
    for i in range(parts):
        part_start = start + math.floor(i * length / parts)
        part_end = start + math.floor((i + 1) * length / parts) - 1
        split.append((part_start, part_end))
    return split


def intervals_overlap(left_start: int, left_end: int, right_start: int, right_end: int) -> bool:
    return left_start <= right_end and right_start <= left_end


def parse_track_id(value: object) -> int | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


@dataclass(frozen=True)
class InteractionEvent:
    start_frame: int
    end_frame: int
    tid_a: int
    tid_b: int

    @property
    def tids(self) -> tuple[int, int]:
        return self.tid_a, self.tid_b

    @property
    def pair_key(self) -> tuple[int, int]:
        return tuple(sorted((self.tid_a, self.tid_b)))


@dataclass(frozen=True)
class MergedEventGroup:
    start_frame: int
    end_frame: int
    events: tuple[InteractionEvent, ...]


def merge_event_groups(events: list[InteractionEvent]) -> list[MergedEventGroup]:
    if not events:
        return []
    ordered = sorted(events, key=lambda item: (item.start_frame, item.end_frame, item.tid_a, item.tid_b))
    groups: list[MergedEventGroup] = [
        MergedEventGroup(ordered[0].start_frame, ordered[0].end_frame, (ordered[0],))
    ]
    for event in ordered[1:]:
        last = groups[-1]
        if event.start_frame > last.end_frame + 1:
            groups.append(MergedEventGroup(event.start_frame, event.end_frame, (event,)))
        else:
            groups[-1] = MergedEventGroup(
                last.start_frame,
                max(last.end_frame, event.end_frame),
                (*last.events, event),
            )
    return groups


def merge_expanded_groups(groups: list[MergedEventGroup]) -> list[MergedEventGroup]:
    if not groups:
        return []
    ordered = sorted(groups, key=lambda item: (item.start_frame, item.end_frame))
    merged: list[MergedEventGroup] = [ordered[0]]
    for group in ordered[1:]:
        last = merged[-1]
        if group.start_frame > last.end_frame + 1:
            merged.append(group)
        else:
            merged[-1] = MergedEventGroup(
                last.start_frame,
                max(last.end_frame, group.end_frame),
                (*last.events, *group.events),
            )
    return merged


def participant_events_for_part(
    events: tuple[InteractionEvent, ...], part_start: int, part_end: int
) -> tuple[InteractionEvent, ...]:
    selected = [
        event for event in events if intervals_overlap(event.start_frame, event.end_frame, part_start, part_end)
    ]
    return tuple(selected or events)


def unique_selection_pairs(events: tuple[InteractionEvent, ...]) -> list[tuple[int, int]]:
    return sorted({event.pair_key for event in events})


def unique_selection_track_ids(events: tuple[InteractionEvent, ...]) -> list[int]:
    return sorted({tid for event in events for tid in event.tids})


def format_selection_pairs(events: tuple[InteractionEvent, ...]) -> str:
    return ";".join(f"{left}-{right}" for left, right in unique_selection_pairs(events))


def format_selection_track_ids(events: tuple[InteractionEvent, ...]) -> str:
    return ";".join(str(tid) for tid in unique_selection_track_ids(events))


@dataclass(frozen=True)
class DatasetSpec:
    interaction_root: Path
    tracking_root: Path | None = None

    @property
    def csv_base_root(self) -> Path:
        return self.tracking_root or self.interaction_root


@dataclass(frozen=True)
class SourceVideo:
    shard_video_dir: Path
    source_video_id: str
    source_rel_path: Path
    source_path: Path
    interaction_csv_root: Path
    interaction_csv_rel_path: str
    csv_root: Path
    csv_rel_path: str
    fps: float
    width: int
    height: int
    frame_count: int
    raw_width: int
    raw_height: int
    rotation_degrees: int


@dataclass(frozen=True)
class ClipPlan:
    clip_id: str
    interaction_class: str
    source: SourceVideo
    source_start_frame: int
    source_end_frame: int
    cache_path: Path
    cache_vis_path: Path
    clip_path: Path
    annotation_path: Path
    participant_events: tuple[InteractionEvent, ...]

    @property
    def duration_s(self) -> float:
        return (self.source_end_frame - self.source_start_frame + 1) / self.source.fps


def default_dataset_specs() -> list[DatasetSpec]:
    return [
        DatasetSpec(interaction_root=root)
        for root in DEFAULT_DATASET_ROOTS
    ]


def dataset_specs_from_roots(s2_roots: list[Path] | None, tracking_root: Path | None) -> list[DatasetSpec]:
    if not s2_roots:
        return default_dataset_specs()
    return [DatasetSpec(interaction_root=root, tracking_root=tracking_root) for root in s2_roots]


def map_remote_csv_root(remote_csv_root: str, local_root: Path) -> Path:
    normalized = remote_csv_root.replace("\\", "/")
    remote_parts = PurePosixPath(normalized).parts
    local_children = {path.name for path in local_root.iterdir() if path.is_dir()}
    for idx, part in enumerate(remote_parts):
        if part in local_children:
            return local_root / Path(*remote_parts[idx:])
    raise ValueError(f"Cannot map source_csv_root to {local_root}: {remote_csv_root}")


def csv_root_for_manifest(video_dir: Path, manifest_doc: dict[str, Any], dataset: DatasetSpec) -> tuple[Path, str]:
    if dataset.tracking_root is None:
        csv_root = video_dir
    else:
        feature_manifest = manifest_doc.get("feature_csv_manifest") or {}
        source_csv_root = str(feature_manifest.get("source_csv_root") or "").strip()
        if source_csv_root:
            csv_root = map_remote_csv_root(source_csv_root, dataset.tracking_root)
        else:
            csv_root = dataset.tracking_root / video_dir.resolve().relative_to(dataset.interaction_root.resolve())

    boxes_path = csv_root / "tracking_boxes.csv"
    keypoints_path = csv_root / "keypoints.csv"
    if not boxes_path.exists():
        raise FileNotFoundError(f"tracking_boxes.csv not found for {video_dir}: {boxes_path}")
    if not keypoints_path.exists():
        raise FileNotFoundError(f"keypoints.csv not found for {video_dir}: {keypoints_path}")
    return csv_root, safe_rel(csv_root, dataset.csv_base_root)


def load_interval_groups(csv_path: Path, total_frames: int) -> dict[str, list[InteractionEvent]]:
    interval_groups: dict[str, list[InteractionEvent]] = {}
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row.get("start_frame") or not row.get("end_frame"):
                continue
            tid_a = parse_track_id(row.get("tidA"))
            tid_b = parse_track_id(row.get("tidB"))
            if tid_a is None or tid_b is None:
                continue
            interaction_class = str(row.get("class", "")).strip() or "unknown"
            start = int(float(row["start_frame"]))
            end = int(float(row["end_frame"]))
            if end < start:
                start, end = end, start
            start = max(0, start)
            end = min(total_frames - 1, end)
            if end < start:
                continue
            interval_groups.setdefault(interaction_class, []).append(InteractionEvent(start, end, tid_a, tid_b))
    return interval_groups


def safe_name_fragment(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in value.strip())
    return cleaned or "unknown"


def build_plans(
    datasets: list[DatasetSpec], source_root: Path, limit_videos: int | None = None
) -> list[ClipPlan]:
    plans: list[ClipPlan] = []
    clip_number = 1
    processed_videos = 0

    for dataset in datasets:
        manifest_paths = sorted(dataset.interaction_root.rglob("manifest.json"), key=lambda p: str(p).lower())
        for manifest_path in manifest_paths:
            video_dir = manifest_path.parent
            interactions_path = video_dir / "interactions.csv"
            if not interactions_path.exists():
                continue
            with manifest_path.open("r", encoding="utf-8") as f:
                manifest_doc = json.load(f)
            manifest = manifest_doc["video_manifest"]
            csv_root, csv_rel_path = csv_root_for_manifest(video_dir, manifest_doc, dataset)

            source_rel = source_rel_from_manifest(manifest.get("source_path") or manifest["relative_path"])
            source_path = source_root / source_rel
            if not source_path.exists():
                raise FileNotFoundError(f"Mapped source video does not exist: {source_path}")

            fallback_fps = float(manifest.get("fps") or 29.97002997002997)
            fallback_duration = float(manifest.get("duration_sec") or 0)
            fps, width, height, frame_count, raw_width, raw_height, rotation = probe_video(
                source_path, fallback_fps, fallback_duration
            )
            interval_groups = load_interval_groups(interactions_path, frame_count)
            if not interval_groups:
                continue

            min_frames = max(1, int(round(MIN_CLIP_SECONDS * fps)))
            max_frames = max(1, int(math.floor(MAX_CLIP_SECONDS * fps)))
            drop_over_frames = max(1, int(round(DROP_SEGMENT_OVER_SECONDS * fps)))

            source = SourceVideo(
                shard_video_dir=video_dir,
                source_video_id=stable_video_id(source_rel),
                source_rel_path=source_rel,
                source_path=source_path,
                interaction_csv_root=video_dir,
                interaction_csv_rel_path=safe_rel(video_dir, dataset.interaction_root),
                csv_root=csv_root,
                csv_rel_path=csv_rel_path,
                fps=fps,
                width=width,
                height=height,
                frame_count=frame_count,
                raw_width=raw_width,
                raw_height=raw_height,
                rotation_degrees=rotation,
            )

            for interaction_class in sorted(interval_groups):
                raw_merged = merge_event_groups(interval_groups[interaction_class])
                expanded = []
                for group in raw_merged:
                    expanded_start, expanded_end = expand_interval(
                        group.start_frame, group.end_frame, min_frames, frame_count
                    )
                    expanded.append(MergedEventGroup(expanded_start, expanded_end, group.events))
                expanded_merged = merge_expanded_groups(expanded)
                class_fragment = safe_name_fragment(interaction_class)

                for group in expanded_merged:
                    if group.end_frame - group.start_frame + 1 > drop_over_frames:
                        continue
                    for part_start, part_end in split_interval(group.start_frame, group.end_frame, max_frames):
                        clip_id = f"C{clip_number:06d}"
                        cache_folder = CACHED_VIDEOS_ROOT / source.source_video_id
                        cache_vis_folder = CACHE_VIS_ROOT / source.source_video_id
                        valid_folder = VIDEOS_ROOT / source.source_video_id
                        annotation_folder = ANNOTATIONS_ROOT / source.source_video_id
                        filename = (
                            f"{clip_id}_{class_fragment}_{source.source_rel_path.stem}_"
                            f"{part_start}_{part_end}.mp4"
                        )
                        cache_path = cache_folder / filename
                        cache_vis_path = cache_vis_folder / filename
                        clip_path = valid_folder / filename
                        annotation_path = annotation_folder / f"{clip_id}.csv"
                        plans.append(
                            ClipPlan(
                                clip_id=clip_id,
                                interaction_class=interaction_class,
                                source=source,
                                source_start_frame=part_start,
                                source_end_frame=part_end,
                                cache_path=cache_path,
                                cache_vis_path=cache_vis_path,
                                clip_path=clip_path,
                                annotation_path=annotation_path,
                                participant_events=participant_events_for_part(
                                    group.events, part_start, part_end
                                ),
                            )
                        )
                        clip_number += 1

            processed_videos += 1
            if limit_videos is not None and processed_videos >= limit_videos:
                break
        if limit_videos is not None and processed_videos >= limit_videos:
            break

    return plans


def index_row(plan: ClipPlan, source_root: Path) -> dict[str, Any]:
    cache_rel = safe_rel(plan.cache_path, CACHED_VIDEOS_ROOT)
    cache_vis_rel = safe_rel(plan.cache_vis_path, CACHE_VIS_ROOT)
    clip_rel = safe_rel(plan.clip_path, VIDEOS_ROOT)
    annotation_rel = safe_rel(plan.annotation_path, ANNOTATIONS_ROOT)
    return {
        "clip_id": plan.clip_id,
        "source_video_id": plan.source.source_video_id,
        "source_video_root": str(source_root.resolve()),
        "source_video_rel_path": str(plan.source.source_rel_path),
        "source_video_path": str(plan.source.source_path.resolve()),
        "cache_root": str(CACHED_VIDEOS_ROOT.resolve()),
        "cache_rel_path": cache_rel,
        "cache_path": str(plan.cache_path.resolve()),
        "cache_vis_root": str(CACHE_VIS_ROOT.resolve()),
        "cache_vis_rel_path": cache_vis_rel,
        "cache_vis_path": str(plan.cache_vis_path.resolve()),
        "clip_root": str(VIDEOS_ROOT.resolve()),
        "clip_rel_path": clip_rel,
        "clip_path": str(plan.clip_path.resolve()),
        "annotation_root": str(ANNOTATIONS_ROOT.resolve()),
        "annotation_rel_path": annotation_rel,
        "annotation_path": str(plan.annotation_path.resolve()),
        "interaction_csv_root": str(plan.source.interaction_csv_root.resolve()),
        "interaction_csv_rel_path": plan.source.interaction_csv_rel_path,
        "interaction_class": plan.interaction_class,
        "csv_root": str(plan.source.csv_root.resolve()),
        "csv_rel_path": plan.source.csv_rel_path,
        "source_start_frame": plan.source_start_frame,
        "source_end_frame": plan.source_end_frame,
        "clip_frame_offset": plan.source_start_frame,
        "fps": f"{plan.source.fps:.12g}",
        "width": plan.source.width,
        "height": plan.source.height,
        "source_raw_width": plan.source.raw_width,
        "source_raw_height": plan.source.raw_height,
        "source_rotation_degrees": plan.source.rotation_degrees,
        "selection_pairs": format_selection_pairs(plan.participant_events),
        "selection_track_ids": format_selection_track_ids(plan.participant_events),
        "duration_s": f"{plan.duration_s:.6f}",
        "cache_source_start_frame": plan.source_start_frame,
        "cache_source_end_frame": plan.source_end_frame,
        "cache_duration_s": f"{plan.duration_s:.6f}",
        "selected_cache_start_frame": "",
        "selected_cache_end_frame": "",
        "is_saved": "false",
    }


def annotation_header_row(plan: ClipPlan) -> dict[str, Any]:
    clip_rel = safe_rel(plan.clip_path, VIDEOS_ROOT)
    return {
        "event_id": "",
        "event_group_id": "",
        "clip_id": plan.clip_id,
        "clip_root": str(VIDEOS_ROOT.resolve()),
        "clip_rel_path": clip_rel,
        "clip_path": str(plan.clip_path.resolve()),
        "csv_root": str(plan.source.csv_root.resolve()),
        "csv_rel_path": plan.source.csv_rel_path,
        "start_frame": "",
        "end_frame": "",
        "roi_x": "",
        "roi_y": "",
        "roi_w": "",
        "roi_h": "",
        "valence": "",
        "fine_class": "",
        "allow_duplicate_pair": "false",
        "fps": f"{plan.source.fps:.12g}",
        "width": plan.source.width,
        "height": plan.source.height,
        "start_time_s": "",
        "end_time_s": "",
        "annotator": "",
        "label_confidence": "",
        "notes": "",
        "exclude": "",
    }


def write_csv_direct(path: Path, columns: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_existing_index() -> dict[str, dict[str, str]]:
    if not INDEX_PATH.exists():
        return {}
    with INDEX_PATH.open("r", encoding="utf-8-sig", newline="") as f:
        return {row["clip_id"]: row for row in csv.DictReader(f) if row.get("clip_id")}


def preserve_saved_index_fields(row: dict[str, Any], existing: dict[str, str] | None) -> dict[str, Any]:
    if not existing:
        return row
    is_saved = str(existing.get("is_saved", "")).lower() == "true"
    has_selection = bool(existing.get("selected_cache_start_frame") and existing.get("selected_cache_end_frame"))
    if not is_saved and not has_selection:
        return row
    preserved = dict(row)
    for key in [
        "clip_root",
        "clip_rel_path",
        "clip_path",
        "source_start_frame",
        "source_end_frame",
        "clip_frame_offset",
        "duration_s",
        "selected_cache_start_frame",
        "selected_cache_end_frame",
        "is_saved",
    ]:
        if existing.get(key):
            preserved[key] = existing[key]
    return preserved


def write_index_and_annotations(plans: list[ClipPlan], source_root: Path) -> None:
    index_rows = [index_row(plan, source_root) for plan in plans]
    write_csv_direct(INDEX_PATH, INDEX_COLUMNS, index_rows)
    write_csv_direct(PLAN_PATH, INDEX_COLUMNS, index_rows)


def format_seconds(frame: int, fps: float) -> str:
    return f"{frame / fps:.9f}"


def export_clip(plan: ClipPlan, codec: str, crf: int, preset: str, dry_run: bool) -> None:
    plan.cache_path.parent.mkdir(parents=True, exist_ok=True)
    start_s = format_seconds(plan.source_start_frame, plan.source.fps)
    frame_count = plan.source_end_frame - plan.source_start_frame + 1
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-y",
        "-ss",
        start_s,
        "-i",
        str(plan.source.source_path),
        "-map",
        "0:v:0",
        "-frames:v",
        str(frame_count),
        "-c:v",
        codec,
        "-preset",
        preset,
    ]
    if codec == "h264_nvenc":
        cmd += ["-cq", str(crf)]
    else:
        cmd += ["-crf", str(crf)]
    cmd += [
        "-pix_fmt",
        "yuv420p",
        "-an",
        "-movflags",
        "+faststart",
        str(plan.cache_path),
    ]

    if dry_run:
        return

    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed for {plan.clip_id} ({plan.source.source_path}):\n{result.stderr[-4000:]}"
        )


VIS_COLORS_BGR = [
    (36, 80, 255),
    (0, 220, 255),
    (0, 210, 80),
    (255, 170, 30),
    (220, 70, 220),
    (255, 255, 70),
]


def transform_bbox_for_display(
    x: float,
    y: float,
    w: float,
    h: float,
    raw_width: int,
    raw_height: int,
    display_width: int,
    display_height: int,
    rotation_degrees: int,
) -> tuple[float, float, float, float]:
    normalized = rotation_degrees % 360
    if normalized == 270:
        transformed = (raw_height - y - h, x, h, w)
    elif normalized == 90:
        transformed = (y, raw_width - x - w, h, w)
    elif normalized == 180:
        transformed = (raw_width - x - w, raw_height - y - h, w, h)
    else:
        transformed = (x, y, w, h)

    tx, ty, tw, th = transformed
    x1 = max(0.0, min(float(display_width - 1), tx))
    y1 = max(0.0, min(float(display_height - 1), ty))
    x2 = max(x1 + 1.0, min(float(display_width), tx + tw))
    y2 = max(y1 + 1.0, min(float(display_height), ty + th))
    return x1, y1, x2 - x1, y2 - y1


def load_visual_boxes(plan: ClipPlan) -> dict[int, list[tuple[int, float, float, float, float]]]:
    tids = set(unique_selection_track_ids(plan.participant_events))
    if not tids:
        return {}
    boxes_path = plan.source.csv_root / "tracking_boxes.csv"
    source_video_name = plan.source.source_path.name
    boxes_by_frame: dict[int, list[tuple[int, float, float, float, float]]] = {}
    start_frame = plan.source_start_frame
    end_frame = plan.source_end_frame
    saw_target_range = False
    with boxes_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("video") and row["video"] != source_video_name:
                continue
            try:
                frame = int(float(row["frame"]))
            except (KeyError, ValueError):
                continue
            if frame < start_frame:
                continue
            if frame > end_frame:
                if saw_target_range:
                    break
                continue
            saw_target_range = True
            track_id = parse_track_id(row.get("track_id"))
            if track_id is None or track_id not in tids:
                continue
            try:
                x, y, w, h = (float(row["x"]), float(row["y"]), float(row["w"]), float(row["h"]))
            except (KeyError, ValueError):
                continue
            box = transform_bbox_for_display(
                x,
                y,
                w,
                h,
                plan.source.raw_width,
                plan.source.raw_height,
                plan.source.width,
                plan.source.height,
                plan.source.rotation_degrees,
            )
            boxes_by_frame.setdefault(frame - start_frame, []).append((track_id, *box))
    return boxes_by_frame


def start_visual_encoder(
    path: Path, width: int, height: int, fps: float, codec: str, crf: int, preset: str
) -> subprocess.Popen[bytes]:
    path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{width}x{height}",
        "-r",
        f"{fps:.12g}",
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        codec,
        "-preset",
        preset,
    ]
    if codec == "h264_nvenc":
        cmd += ["-cq", str(crf)]
    else:
        cmd += ["-crf", str(crf)]
    cmd += ["-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)


def export_visual_clip(plan: ClipPlan, codec: str, crf: int, preset: str, dry_run: bool) -> None:
    plan.cache_vis_path.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        return
    if not plan.cache_path.exists():
        raise FileNotFoundError(f"clean cache clip not found for visualization: {plan.cache_path}")

    boxes_by_frame = load_visual_boxes(plan)
    if not boxes_by_frame:
        shutil.copyfile(plan.cache_path, plan.cache_vis_path)
        return

    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required to draw cache_vis bbox overlays.") from exc

    capture = cv2.VideoCapture(str(plan.cache_path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open clean cache clip for visualization: {plan.cache_path}")

    ok, frame = capture.read()
    if not ok:
        capture.release()
        raise RuntimeError(f"cannot read first frame from clean cache clip: {plan.cache_path}")

    frame_height, frame_width = frame.shape[:2]
    thickness = max(3, min(frame_width, frame_height) // 360)
    encoder = start_visual_encoder(plan.cache_vis_path, frame_width, frame_height, plan.source.fps, codec, crf, preset)
    frame_idx = 0
    try:
        while ok:
            for track_id, x, y, w, h in boxes_by_frame.get(frame_idx, []):
                color = VIS_COLORS_BGR[track_id % len(VIS_COLORS_BGR)]
                x1 = int(round(max(0.0, min(frame_width - 1.0, x))))
                y1 = int(round(max(0.0, min(frame_height - 1.0, y))))
                x2 = int(round(max(x1 + 1.0, min(float(frame_width), x + w))))
                y2 = int(round(max(y1 + 1.0, min(float(frame_height), y + h))))
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            if encoder.stdin is None:
                raise RuntimeError("visual encoder stdin is not available")
            encoder.stdin.write(frame.tobytes())
            frame_idx += 1
            ok, frame = capture.read()
    except Exception:
        if encoder.stdin is not None:
            encoder.stdin.close()
        encoder.kill()
        raise
    finally:
        capture.release()

    if encoder.stdin is not None:
        encoder.stdin.close()
    stderr = (encoder.stderr.read() if encoder.stderr is not None else b"").decode("utf-8", errors="replace")
    returncode = encoder.wait()
    if returncode != 0:
        raise RuntimeError(f"ffmpeg cache_vis encode failed for {plan.clip_id}:\n{stderr[-4000:]}")


def check_nvenc() -> None:
    result = subprocess.run(
        ["ffmpeg", "-hide_banner", "-encoders"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0 or "h264_nvenc" not in result.stdout:
        raise RuntimeError("h264_nvenc is not available; GPU encoding is expected on this machine.")


def check_log_flush() -> None:
    ensure_dirs()
    probe_path = LOG_ROOT / "flush_probe.log"
    with probe_path.open("w", encoding="utf-8") as f:
        f.write("flush probe line 1\n")
        f.flush()
        os.fsync(f.fileno())
        time.sleep(1)
        f.write("flush probe line 2\n")
        f.flush()
        os.fsync(f.fileno())
    log(f"log flush probe written: {probe_path}")


def progress_iter(items: list[ClipPlan]):
    try:
        from tqdm import tqdm

        yield from tqdm(items, unit="clip", mininterval=1.0)
    except Exception:
        total = len(items)
        last_logged = 0.0
        for idx, item in enumerate(items, start=1):
            now = time.time()
            if now - last_logged >= 10 or idx == 1 or idx == total:
                log(f"progress {idx}/{total}")
                last_logged = now
            yield item


def main() -> int:
    global LOG_FILE_HANDLE
    parser = argparse.ArgumentParser()
    parser.add_argument("--s2-root", dest="s2_roots", type=Path, action="append")
    parser.add_argument("--tracking-root", type=Path)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--limit-videos", type=int)
    parser.add_argument("--limit-clips", type=int)
    parser.add_argument("--start-clip", type=int, default=1)
    parser.add_argument("--next-missing", type=int)
    parser.add_argument("--check-log-flush", action="store_true")
    parser.add_argument("--codec", default="h264_nvenc")
    parser.add_argument("--crf", type=int, default=24)
    parser.add_argument("--preset", default="p4")
    args = parser.parse_args()

    if args.check_log_flush:
        check_log_flush()
        return 0

    ensure_dirs()
    log_path = LOG_ROOT / "prepare_review_clips.log"
    with log_path.open("a", encoding="utf-8") as handle:
        LOG_FILE_HANDLE = handle
        log(f"writing live log to {log_path}")
        dataset_specs = dataset_specs_from_roots(args.s2_roots, args.tracking_root)
        for spec in dataset_specs:
            log(f"reading interactions from {spec.interaction_root}")
            log(f"tracking/keypoint csv root: {spec.csv_base_root}")
        log(f"mapping sources under {args.source_root}")
        log(f"writing cached candidate clips under {CACHED_VIDEOS_ROOT}")
        log(f"writing UI visualization clips under {CACHE_VIS_ROOT}")

        if args.codec == "h264_nvenc":
            check_nvenc()

        plans = build_plans(dataset_specs, args.source_root, args.limit_videos)
        discarded_clip_ids = read_discarded_clip_ids()
        if discarded_clip_ids:
            plans = [plan for plan in plans if plan.clip_id not in discarded_clip_ids]

        total_seconds = sum(plan.duration_s for plan in plans)
        log(f"planned clips: {len(plans)}")
        log(f"discarded clips skipped: {len(discarded_clip_ids)}")
        log(f"planned video duration: {total_seconds / 3600:.3f} hours")
        write_index_and_annotations(plans, args.source_root)
        log(f"wrote {INDEX_PATH}")
        log(f"annotation CSVs will be created only when saved under {ANNOTATIONS_ROOT}")

        if args.dry_run:
            log("dry run complete; no videos exported")
            LOG_FILE_HANDLE = None
            return 0

        if args.next_missing is not None:
            export_plans = [
                plan
                for plan in plans
                if not plan.cache_path.exists() or not plan.cache_vis_path.exists()
            ][: args.next_missing]
        else:
            export_start = max(0, args.start_clip - 1)
            export_plans = plans[export_start:]
            if args.limit_clips is not None:
                export_plans = export_plans[: args.limit_clips]
        log(f"selected clips for export: {len(export_plans)}")

        exported = 0
        visual_exported = 0
        skipped = 0
        visual_skipped = 0
        started = time.time()
        for plan in progress_iter(export_plans):
            cache_exists = plan.cache_path.exists()
            cache_vis_exists = plan.cache_vis_path.exists()
            if cache_exists and not args.force:
                skipped += 1
            else:
                message = (
                    f"{time.strftime('%Y-%m-%d %H:%M:%S')} export clean {plan.clip_id} "
                    f"{plan.source.source_rel_path} frames {plan.source_start_frame}-{plan.source_end_frame}"
                )
                log(message)
                export_clip(plan, args.codec, args.crf, args.preset, dry_run=False)
                exported += 1

            if cache_vis_exists and not args.force:
                visual_skipped += 1
            else:
                log(
                    f"{time.strftime('%Y-%m-%d %H:%M:%S')} export cache_vis {plan.clip_id} "
                    f"pairs={format_selection_pairs(plan.participant_events)}"
                )
                export_visual_clip(plan, args.codec, args.crf, args.preset, dry_run=False)
                visual_exported += 1

            if (exported + visual_exported + skipped + visual_skipped) % 20 == 0:
                elapsed = max(1.0, time.time() - started)
                status = (
                    f"clean_exported={exported} clean_skipped={skipped} "
                    f"vis_exported={visual_exported} vis_skipped={visual_skipped} "
                    f"rate={(exported + visual_exported) / elapsed:.3f} videos/s"
                )
                log(status)

        log(
            f"complete: clean_exported={exported} clean_skipped={skipped} "
            f"vis_exported={visual_exported} vis_skipped={visual_skipped}"
        )
        LOG_FILE_HANDLE = None
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
