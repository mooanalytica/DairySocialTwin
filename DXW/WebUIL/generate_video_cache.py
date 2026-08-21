from __future__ import annotations

import argparse
import concurrent.futures
import os
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app import REQUIRED_CUDA_VISIBLE_DEVICES, load_samples
from trackid_video import (
    TRACK_ID_CACHE_MAX_DIMENSION,
    TRACK_ID_SOURCE_FPS_TEXT,
    TRACK_ID_VIDEO_CACHE_DIR,
    track_id_cache_is_ready,
    validate_track_id_cache,
)


DEFAULT_ENCODER = "h264_nvenc"
DEFAULT_BITRATE = "350k"
DEFAULT_MAXRATE = "500k"
DEFAULT_BUFSIZE = "1000k"


@dataclass(frozen=True)
class VideoCacheTask:
    sample_id: str
    label: str
    source_path: Path
    output_path: Path
    start_frame: int
    end_frame: int
    fps: float

    @property
    def expected_frame_count(self) -> int:
        return self.end_frame - self.start_frame + 1

    @property
    def start_seconds(self) -> float:
        return self.start_frame / self.fps

    @property
    def duration_seconds(self) -> float:
        return self.expected_frame_count / self.fps


LOG_LOCK = threading.Lock()


def log(message: str) -> None:
    with LOG_LOCK:
        print(message, flush=True)


def even_max_expr(axis: str) -> str:
    return f"trunc(min({TRACK_ID_CACHE_MAX_DIMENSION}\\,{axis})/2)*2"


def scale_filter() -> str:
    width_expr = f"if(gte(iw\\,ih)\\,{even_max_expr('iw')}\\,-2)"
    height_expr = f"if(gte(iw\\,ih)\\,-2\\,{even_max_expr('ih')})"
    return f"scale={width_expr}:{height_expr}"


def ffmpeg_command(task: VideoCacheTask, output_path: Path, args: argparse.Namespace) -> list[str]:
    trim_filter = (
        f"trim=start_frame={task.start_frame}:end_frame={task.end_frame + 1},"
        f"setpts=PTS-STARTPTS,{scale_filter()}"
    )
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(task.source_path),
        "-map",
        "0:v:0",
        "-an",
        "-vf",
        trim_filter,
        "-frames:v",
        str(task.expected_frame_count),
        "-r",
        TRACK_ID_SOURCE_FPS_TEXT,
        "-fps_mode",
        "cfr",
        "-c:v",
        args.encoder,
        "-pix_fmt",
        "yuv420p",
    ]
    if args.encoder == "h264_nvenc":
        command.extend(
            [
                "-preset",
                "p4",
                "-rc",
                "vbr",
                "-b:v",
                args.bitrate,
                "-maxrate",
                args.maxrate,
                "-bufsize",
                args.bufsize,
            ]
        )
    elif args.encoder == "libx264":
        command.extend(
            [
                "-preset",
                "veryfast",
                "-crf",
                str(args.crf),
                "-maxrate",
                args.maxrate,
                "-bufsize",
                args.bufsize,
            ]
        )
    else:
        raise RuntimeError(f"Unsupported encoder: {args.encoder}")
    command.extend(
        [
            "-movflags",
            "+faststart",
            "-video_track_timescale",
            "30000",
            "-progress",
            "pipe:1",
            "-nostats",
            str(output_path),
        ]
    )
    return command


def choose_samples(args: argparse.Namespace) -> list[dict[str, Any]]:
    samples = load_samples(require_ready=False)
    if args.sample:
        wanted = set(args.sample)
        samples = [sample for sample in samples if sample["id"] in wanted]
        missing = sorted(wanted - {sample["id"] for sample in samples})
        if missing:
            raise RuntimeError(f"Requested samples are missing from Stage1 segmented candidates: {missing}")
    elif not args.all:
        raise RuntimeError("Use --all or one or more --sample IDs.")

    if args.max_samples > 0:
        samples = samples[: args.max_samples]
    if not samples:
        raise RuntimeError("No samples are available for video cache generation.")
    return samples


def task_for_sample(sample: dict[str, Any]) -> VideoCacheTask:
    video = sample["video"]
    source_path = video.get("trackIdSourcePath")
    output_path = video.get("trackIdCacheTargetPath")
    if not isinstance(source_path, Path) or not source_path.is_file():
        raise RuntimeError(f"Sample {sample['id']} has no QA track-ID source video")
    if not isinstance(output_path, Path):
        raise RuntimeError(f"Sample {sample['id']} has no track-ID cache target path")
    start_frame = int(video["loopStartFrame"])
    end_frame = int(video["loopEndFrame"])
    fps = float(video["fps"])
    task = VideoCacheTask(
        sample["id"], sample["label"], source_path, output_path, start_frame, end_frame, fps
    )
    if task.expected_frame_count != int(video["expectedFrameCount"]):
        raise RuntimeError(f"Sample {sample['id']} has inconsistent track-ID cache frame count")
    if task.expected_frame_count <= 0 or fps <= 0:
        raise RuntimeError(f"Sample {sample['id']} has invalid track-ID cache timing")
    return task


def build_tasks(samples: list[dict[str, Any]]) -> list[VideoCacheTask]:
    tasks = [task_for_sample(sample) for sample in samples]
    tasks.sort(key=lambda task: task.sample_id)
    return tasks


def run_one(task: VideoCacheTask, args: argparse.Namespace) -> str:
    if track_id_cache_is_ready(task.output_path, task.expected_frame_count, task.fps):
        if not args.overwrite:
            return f"[skip] sample {task.sample_id}: existing cache {task.output_path}"

    task.output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = task.output_path.with_name(
        f".{task.output_path.name}.{os.getpid()}.{time.time_ns()}.tmp.mp4"
    )
    command = ffmpeg_command(task, tmp_path, args)
    try:
        log(
            f"[start] sample {task.sample_id}: {task.source_path} "
            f"[frames {task.start_frame}..{task.end_frame}; "
            f"{task.expected_frame_count} frames] -> {task.output_path}"
        )
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        output_tail: deque[str] = deque(maxlen=100)
        current_frame = 0
        last_progress_log = time.monotonic()
        assert process.stdout is not None
        for raw_line in process.stdout:
            line = raw_line.strip()
            if not line:
                continue
            output_tail.append(line)
            if line.startswith("frame="):
                try:
                    current_frame = max(current_frame, int(line.partition("=")[2]))
                except ValueError:
                    pass
            now = time.monotonic()
            if now - last_progress_log >= 10.0:
                percent = min(100.0, 100.0 * current_frame / task.expected_frame_count)
                log(
                    f"[progress] sample {task.sample_id}: {current_frame}/"
                    f"{task.expected_frame_count} frames ({percent:.1f}%)"
                )
                last_progress_log = now
        return_code = process.wait()
        if return_code != 0:
            output_text = "\n".join(output_tail)[-4000:]
            raise RuntimeError(f"ffmpeg failed for sample {task.sample_id}: {output_text}")
        validate_track_id_cache(tmp_path, task.expected_frame_count, task.fps)
        tmp_path.replace(task.output_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
    return f"[done] sample {task.sample_id}: {task.output_path}"


def run_tasks(tasks: list[VideoCacheTask], args: argparse.Namespace) -> None:
    if args.dry_run:
        for task in tasks:
            status = (
                "ready"
                if track_id_cache_is_ready(task.output_path, task.expected_frame_count, task.fps)
                else "pending"
            )
            log(
                f"[dry-run] {status} sample {task.sample_id}: {task.source_path} "
                f"[frames {task.start_frame}..{task.end_frame}; "
                f"{task.expected_frame_count} frames] -> {task.output_path}"
            )
        return

    failures: list[str] = []
    if args.workers <= 1:
        for task in tasks:
            try:
                log(run_one(task, args))
            except Exception as exc:
                failures.append(str(exc))
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_to_task = {executor.submit(run_one, task, args): task for task in tasks}
            for future in concurrent.futures.as_completed(future_to_task):
                try:
                    log(future.result())
                except Exception as exc:
                    failures.append(str(exc))

    if failures:
        for failure in failures:
            log(f"[error] {failure}")
        raise RuntimeError(f"Video cache generation failed for {len(failures)} sample(s)")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate the low-bitrate H.264 WebUI cache from the current QA track-ID videos "
            f"under {TRACK_ID_VIDEO_CACHE_DIR}. Landscape 16:9 videos are encoded at "
            f"{TRACK_ID_CACHE_MAX_DIMENSION}x540."
        )
    )
    parser.add_argument("--all", action="store_true", help="Generate cache for every active Stage1 segmented shard.")
    parser.add_argument("--sample", action="append", default=[], help="Sample id to cache; may be repeated.")
    parser.add_argument("--max-samples", type=int, default=0, help="Limit selected samples for smoke runs; 0 means no limit.")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate cache files that already validate.")
    parser.add_argument("--dry-run", action="store_true", help="List work without running ffmpeg or writing files.")
    parser.add_argument("--workers", type=int, default=1, help="Number of concurrent ffmpeg processes.")
    parser.add_argument("--encoder", choices=["h264_nvenc", "libx264"], default=DEFAULT_ENCODER)
    parser.add_argument("--bitrate", default=DEFAULT_BITRATE)
    parser.add_argument("--maxrate", default=DEFAULT_MAXRATE)
    parser.add_argument("--bufsize", default=DEFAULT_BUFSIZE)
    parser.add_argument("--crf", type=int, default=30, help="Used only with --encoder libx264.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.workers < 1:
        raise RuntimeError("--workers must be >= 1")
    if (
        args.encoder == "h264_nvenc"
        and not args.dry_run
        and os.environ.get("CUDA_VISIBLE_DEVICES", "") != REQUIRED_CUDA_VISIBLE_DEVICES
    ):
        raise RuntimeError(
            f"Refusing NVENC cache generation without CUDA_VISIBLE_DEVICES={REQUIRED_CUDA_VISIBLE_DEVICES}"
        )
    samples = choose_samples(args)
    tasks = build_tasks(samples)
    log(f"[info] video cache dir: {TRACK_ID_VIDEO_CACHE_DIR}")
    log(f"[info] max encoded dimension: {TRACK_ID_CACHE_MAX_DIMENSION}")
    log(f"[info] selected samples: {len(samples)}")
    log(f"[info] tasks: {len(tasks)}")
    run_tasks(tasks, args)
    log("[done] video cache generation finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
