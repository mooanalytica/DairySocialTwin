#!/usr/bin/env python3
from __future__ import annotations

import bisect
import csv
import json
import math
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


SOURCE_ROOT = Path("/mnt/data4t/hyw/20260625")
BBOX_VIDEO_ROOT = Path("/mnt/data4t/hyw/20260626V_bbox")
OUTPUT_ROOT = Path(__file__).resolve().parent / "output"

MAX_OUTPUT_LINES = 500_000
MAX_DATA_ROWS = MAX_OUTPUT_LINES - 1
OVERLAP_SECONDS = 10.0
TARGET_CSV_NAMES = ("tracking_boxes.csv", "keypoints.csv")
PLAYBACK_JSON_NAME = "playback_segment.json"
LOG_INTERVAL_SECONDS = 10.0


@dataclass(frozen=True)
class Target:
    farm_id: str
    camera: str
    gx_id: str
    source_dir: Path
    bbox_video: Path


@dataclass(frozen=True)
class FrameStats:
    frames: list[int]
    counts: list[int]
    total_rows: int


@dataclass(frozen=True)
class Shard:
    index: int
    count: int
    base_start_frame: int
    base_end_frame: int
    segment_start_frame: int
    segment_end_frame: int
    expected_data_rows: int


class Log:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.path.open("w", encoding="utf-8", newline="\n")
        self.last_progress_at = 0.0

    def close(self) -> None:
        self.file.close()

    def write(self, message: str) -> None:
        print(message, flush=True)
        self.file.write(message + "\n")
        self.file.flush()

    def progress(self, message: str) -> None:
        now = time.monotonic()
        if now - self.last_progress_at >= LOG_INTERVAL_SECONDS:
            self.last_progress_at = now
            self.write(message)


def fail(message: str) -> None:
    raise RuntimeError(message)


def reset_output_root() -> None:
    if OUTPUT_ROOT.exists():
        if OUTPUT_ROOT.is_dir():
            shutil.rmtree(OUTPUT_ROOT)
        else:
            OUTPUT_ROOT.unlink()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


def parse_bbox_video_name(path: Path) -> tuple[str, str, str]:
    name = path.name
    suffix = "_bbox.mp4"
    if not name.endswith(suffix):
        fail(f"Unexpected bbox video name: {path}")
    stem = name[: -len(suffix)]
    parts = stem.split("_")
    if len(parts) != 3:
        fail(f"Unexpected bbox video name: {path}")
    farm_part, camera, gx_id = parts
    if not farm_part.startswith("F") or not farm_part[1:].isdigit():
        fail(f"Unexpected farm id in bbox video name: {path}")
    if not gx_id.startswith("GX") or not gx_id[2:].isdigit():
        fail(f"Unexpected GX id in bbox video name: {path}")
    return farm_part[1:], camera, gx_id


def discover_targets() -> list[Target]:
    if not SOURCE_ROOT.is_dir():
        fail(f"Missing source root: {SOURCE_ROOT}")
    if not BBOX_VIDEO_ROOT.is_dir():
        fail(f"Missing bbox video root: {BBOX_VIDEO_ROOT}")

    videos = sorted(BBOX_VIDEO_ROOT.glob("*_bbox.mp4"))
    if not videos:
        fail(f"No bbox videos found in {BBOX_VIDEO_ROOT}")

    seen_keys: dict[tuple[str, str, str], Path] = {}
    targets: list[Target] = []
    for video in videos:
        farm_id, camera, gx_id = parse_bbox_video_name(video)
        key = (farm_id, camera, gx_id)
        if key in seen_keys:
            fail(f"Duplicate bbox video key {key}: {seen_keys[key]} and {video}")
        seen_keys[key] = video

        source_dir = SOURCE_ROOT / farm_id / camera / gx_id
        if not source_dir.is_dir():
            fail(f"Missing source directory for {video.name}: {source_dir}")
        for csv_name in TARGET_CSV_NAMES:
            csv_path = source_dir / csv_name
            if not csv_path.is_file():
                fail(f"Missing required CSV for {video.name}: {csv_path}")
        manifest_path = source_dir / "manifest.json"
        if not manifest_path.is_file():
            fail(f"Missing manifest for {video.name}: {manifest_path}")
        targets.append(Target(farm_id, camera, gx_id, source_dir, video))

    return targets


def read_manifest(path: Path) -> dict:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        fail(f"Invalid manifest JSON {path}: {exc}")
    if not isinstance(data, dict):
        fail(f"Manifest root is not an object: {path}")
    return data


def manifest_fps(manifest: dict, path: Path) -> float:
    try:
        fps = manifest["video_manifest"]["fps"]
    except KeyError:
        fail(f"Manifest lacks video_manifest.fps: {path}")
    try:
        fps_float = float(fps)
    except (TypeError, ValueError):
        fail(f"Manifest fps is not numeric in {path}: {fps!r}")
    if fps_float <= 0:
        fail(f"Manifest fps must be positive in {path}: {fps!r}")
    return fps_float


def count_frames(csv_path: Path, log: Log) -> tuple[list[str], FrameStats]:
    counts: dict[int, int] = {}
    total_rows = 0
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            fail(f"CSV is empty: {csv_path}")
        if "frame" not in header:
            fail(f"CSV lacks frame column: {csv_path}")
        frame_idx = header.index("frame")
        for row in reader:
            total_rows += 1
            if frame_idx >= len(row):
                fail(f"Row lacks frame column in {csv_path} at data row {total_rows}")
            try:
                frame = int(row[frame_idx])
            except ValueError:
                fail(f"Non-integer frame in {csv_path} at data row {total_rows}: {row[frame_idx]!r}")
            counts[frame] = counts.get(frame, 0) + 1
            log.progress(f"counting {csv_path}: {total_rows:,} data rows")

    frames = sorted(counts)
    if not frames and total_rows:
        fail(f"Internal frame count mismatch for {csv_path}")
    return header, FrameStats(frames=frames, counts=[counts[f] for f in frames], total_rows=total_rows)


def compare_stats(a_path: Path, a: FrameStats, b_path: Path, b: FrameStats) -> None:
    if a.total_rows != b.total_rows:
        fail(f"Row count mismatch: {a_path} has {a.total_rows}, {b_path} has {b.total_rows}")
    if a.frames != b.frames:
        fail(f"Frame set mismatch between {a_path} and {b_path}")
    if a.counts != b.counts:
        fail(f"Per-frame row count mismatch between {a_path} and {b_path}")


def prefix_sums(values: Iterable[int]) -> list[int]:
    prefix = [0]
    for value in values:
        prefix.append(prefix[-1] + value)
    return prefix


def rows_between(frames: list[int], prefix: list[int], start_frame: int, end_frame: int) -> int:
    start_idx = bisect.bisect_left(frames, start_frame)
    end_idx = bisect.bisect_right(frames, end_frame)
    return prefix[end_idx] - prefix[start_idx]


def make_base_partitions(stats: FrameStats, shard_count: int) -> list[tuple[int, int]]:
    frames = stats.frames
    prefix = prefix_sums(stats.counts)
    partitions: list[tuple[int, int]] = []
    start_idx = 0
    for shard_idx in range(1, shard_count):
        remaining_after = shard_count - shard_idx
        target = stats.total_rows * shard_idx / shard_count
        candidate = bisect.bisect_left(prefix, target)
        if candidate > 0:
            prev_candidate = candidate - 1
            if abs(prefix[prev_candidate] - target) < abs(prefix[candidate] - target):
                candidate = prev_candidate
        min_end = start_idx + 1
        max_end = len(frames) - remaining_after
        end_idx = max(min_end, min(candidate, max_end))
        partitions.append((frames[start_idx], frames[end_idx - 1]))
        start_idx = end_idx
    partitions.append((frames[start_idx], frames[-1]))
    return partitions


def build_shards(stats: FrameStats, fps: float) -> list[Shard]:
    if stats.total_rows <= 0:
        fail("Cannot shard an empty target CSV")
    max_frame_rows = max(stats.counts)
    if max_frame_rows > MAX_DATA_ROWS:
        fail(f"A single frame has {max_frame_rows:,} rows, exceeding the data-row cap {MAX_DATA_ROWS:,}")

    frames = stats.frames
    prefix = prefix_sums(stats.counts)
    overlap_frames = int(round(OVERLAP_SECONDS * fps))
    min_shards = max(1, math.ceil(stats.total_rows / MAX_DATA_ROWS))

    for shard_count in range(min_shards, len(frames) + 1):
        base_ranges = make_base_partitions(stats, shard_count)
        shards: list[Shard] = []
        ok = True
        for index, (base_start, base_end) in enumerate(base_ranges, start=1):
            segment_start = base_start if index == 1 else max(frames[0], base_start - overlap_frames)
            segment_end = base_end if index == shard_count else min(frames[-1], base_end + overlap_frames)
            data_rows = rows_between(frames, prefix, segment_start, segment_end)
            if data_rows > MAX_DATA_ROWS:
                ok = False
                break
            shards.append(
                Shard(
                    index=index,
                    count=shard_count,
                    base_start_frame=base_start,
                    base_end_frame=base_end,
                    segment_start_frame=segment_start,
                    segment_end_frame=segment_end,
                    expected_data_rows=data_rows,
                )
            )
        if ok:
            return shards

    fail("Could not find a shard plan satisfying max lines and overlap constraints")


def output_dir_for(target: Target, shard: Shard) -> Path:
    return OUTPUT_ROOT / target.farm_id / target.camera / f"{target.gx_id}_{shard.index}"


def copy_static_files(target: Target, shards: list[Shard], log: Log) -> None:
    for shard in shards:
        out_dir = output_dir_for(target, shard)
        out_dir.mkdir(parents=True, exist_ok=True)
        for src in sorted(target.source_dir.iterdir()):
            if src.name in TARGET_CSV_NAMES:
                continue
            dst = out_dir / src.name
            if src.is_file():
                shutil.copy2(src, dst)
            elif src.is_dir():
                shutil.copytree(src, dst)
        log.write(f"copied static files for {target.farm_id}/{target.camera}/{target.gx_id}_{shard.index}")


def write_playback_json(target: Target, shards: list[Shard], fps: float) -> None:
    for shard in shards:
        out_dir = output_dir_for(target, shard)
        segment = {
            "farm_id": target.farm_id,
            "camera": target.camera,
            "gx_id": target.gx_id,
            "shard_index": shard.index,
            "shard_count": shard.count,
            "source_dir": str(target.source_dir),
            "bbox_video_path": str(target.bbox_video),
            "fps": fps,
            "overlap_seconds": OVERLAP_SECONDS,
            "overlap_frames": int(round(OVERLAP_SECONDS * fps)),
            "base_start_frame": shard.base_start_frame,
            "base_end_frame": shard.base_end_frame,
            "segment_start_frame": shard.segment_start_frame,
            "segment_end_frame": shard.segment_end_frame,
            "segment_start_seconds": shard.segment_start_frame / fps,
            "segment_end_seconds_exclusive": (shard.segment_end_frame + 1) / fps,
            "expected_csv_data_rows_per_split_file": shard.expected_data_rows,
            "csv_files": list(TARGET_CSV_NAMES),
        }
        with (out_dir / PLAYBACK_JSON_NAME).open("w", encoding="utf-8", newline="\n") as f:
            json.dump(segment, f, ensure_ascii=False, indent=2)
            f.write("\n")


def copy_whole_target_csvs(target: Target, shards: list[Shard], log: Log) -> None:
    if len(shards) != 1:
        fail("copy_whole_target_csvs called with more than one shard")
    out_dir = output_dir_for(target, shards[0])
    for csv_name in TARGET_CSV_NAMES:
        src = target.source_dir / csv_name
        shutil.copy2(src, out_dir / csv_name)
    log.write(f"copied unsplit target CSVs for {target.farm_id}/{target.camera}/{target.gx_id}")


def shard_for_frame(shards: list[Shard], frame: int) -> list[int]:
    hits: list[int] = []
    for pos, shard in enumerate(shards):
        if shard.segment_start_frame <= frame <= shard.segment_end_frame:
            hits.append(pos)
    return hits


def write_split_csv(csv_path: Path, target: Target, shards: list[Shard], expected_header: list[str], log: Log) -> None:
    handles = []
    writers = []
    written_counts = [0 for _ in shards]
    try:
        for shard in shards:
            out_path = output_dir_for(target, shard) / csv_path.name
            handle = out_path.open("w", encoding="utf-8", newline="")
            writer = csv.writer(handle)
            writer.writerow(expected_header)
            handles.append(handle)
            writers.append(writer)

        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            try:
                header = next(reader)
            except StopIteration:
                fail(f"CSV is empty while splitting: {csv_path}")
            if header != expected_header:
                fail(f"Header changed while splitting: {csv_path}")
            frame_idx = header.index("frame")
            scanned = 0
            for row in reader:
                scanned += 1
                try:
                    frame = int(row[frame_idx])
                except (IndexError, ValueError):
                    fail(f"Invalid frame while splitting {csv_path} at data row {scanned}")
                hit_positions = shard_for_frame(shards, frame)
                if not hit_positions:
                    fail(f"Frame {frame} from {csv_path} did not map to any shard")
                for pos in hit_positions:
                    writers[pos].writerow(row)
                    written_counts[pos] += 1
                log.progress(f"splitting {csv_path}: {scanned:,} data rows")
    finally:
        for handle in handles:
            handle.close()

    for pos, shard in enumerate(shards):
        if written_counts[pos] != shard.expected_data_rows:
            fail(
                f"Written row count mismatch for {csv_path.name} shard {shard.index}: "
                f"expected {shard.expected_data_rows}, wrote {written_counts[pos]}"
            )
        if written_counts[pos] + 1 > MAX_OUTPUT_LINES:
            fail(
                f"Output line cap exceeded for {csv_path.name} shard {shard.index}: "
                f"{written_counts[pos] + 1:,} lines"
            )
    log.write(f"wrote split {csv_path.name} for {target.farm_id}/{target.camera}/{target.gx_id}")


def process_target(target: Target, log: Log) -> None:
    log.write(f"processing {target.farm_id}/{target.camera}/{target.gx_id}")
    manifest_path = target.source_dir / "manifest.json"
    fps = manifest_fps(read_manifest(manifest_path), manifest_path)

    tracking_path = target.source_dir / "tracking_boxes.csv"
    keypoints_path = target.source_dir / "keypoints.csv"
    tracking_header, tracking_stats = count_frames(tracking_path, log)
    keypoints_header, keypoints_stats = count_frames(keypoints_path, log)
    compare_stats(tracking_path, tracking_stats, keypoints_path, keypoints_stats)

    shards = build_shards(tracking_stats, fps)
    log.write(
        f"plan {target.farm_id}/{target.camera}/{target.gx_id}: "
        f"{tracking_stats.total_rows:,} data rows, {len(shards)} shard(s)"
    )
    for shard in shards:
        log.write(
            f"  shard {shard.index}/{shard.count}: "
            f"base frames {shard.base_start_frame}-{shard.base_end_frame}, "
            f"segment frames {shard.segment_start_frame}-{shard.segment_end_frame}, "
            f"{shard.expected_data_rows:,} data rows"
        )

    copy_static_files(target, shards, log)
    write_playback_json(target, shards, fps)
    if len(shards) == 1:
        copy_whole_target_csvs(target, shards, log)
    else:
        write_split_csv(tracking_path, target, shards, tracking_header, log)
        write_split_csv(keypoints_path, target, shards, keypoints_header, log)


def main() -> int:
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8")

        reset_output_root()
        log = Log(OUTPUT_ROOT / "split_log.txt")
        try:
            targets = discover_targets()
            log.write(f"found {len(targets)} target bbox videos")
            for target in targets:
                process_target(target, log)
            log.write(f"done: output written to {OUTPUT_ROOT}")
        finally:
            log.close()
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
