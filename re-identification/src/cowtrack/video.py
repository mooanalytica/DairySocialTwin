from __future__ import annotations

import csv
import json
import subprocess
import time
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Callable

import cv2

from cowtrack.config import ContractError


LogFn = Callable[[str], None]


@dataclass(frozen=True)
class VideoStreamMetadata:
    codec_name: str
    width: int
    height: int
    time_base: Fraction
    average_frame_rate: Fraction
    start_pts: int
    duration_ticks: int
    num_frames: int
    has_b_frames: int
    rotation_degrees: int | None
    timecode: str | None


@dataclass(frozen=True)
class PacketTimeline:
    pts_ticks: tuple[int, ...]
    duration_ticks: tuple[int, ...]


def _run_json(command: list[str]) -> dict:
    completed = subprocess.run(
        command,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )
    if completed.returncode != 0:
        raise ContractError(
            f"command failed ({completed.returncode}): {' '.join(command)}\n"
            f"{completed.stderr.strip()}"
        )
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise ContractError(f"invalid ffprobe JSON for {command[-1]}: {exc}") from exc


def probe_video_stream(video_path: Path, ffprobe_binary: str) -> VideoStreamMetadata:
    payload = _run_json(
        [
            ffprobe_binary,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            (
                "stream=codec_name,width,height,time_base,avg_frame_rate,start_pts,"
                "duration_ts,nb_frames,has_b_frames:stream_tags=timecode:"
                "stream_side_data=rotation"
            ),
            "-of",
            "json",
            str(video_path),
        ]
    )
    streams = payload.get("streams")
    if not isinstance(streams, list) or len(streams) != 1:
        raise ContractError(f"expected exactly one selected video stream: {video_path}")
    stream = streams[0]
    required = (
        "codec_name",
        "width",
        "height",
        "time_base",
        "avg_frame_rate",
        "start_pts",
        "duration_ts",
        "nb_frames",
        "has_b_frames",
    )
    missing = [key for key in required if stream.get(key) in (None, "N/A", "")]
    if missing:
        raise ContractError(f"video stream metadata missing {missing}: {video_path}")
    rotation = None
    side_data = stream.get("side_data_list") or []
    for entry in side_data:
        if "rotation" in entry:
            rotation = int(entry["rotation"])
            break
    metadata = VideoStreamMetadata(
        codec_name=str(stream["codec_name"]),
        width=int(stream["width"]),
        height=int(stream["height"]),
        time_base=Fraction(str(stream["time_base"])),
        average_frame_rate=Fraction(str(stream["avg_frame_rate"])),
        start_pts=int(stream["start_pts"]),
        duration_ticks=int(stream["duration_ts"]),
        num_frames=int(stream["nb_frames"]),
        has_b_frames=int(stream["has_b_frames"]),
        rotation_degrees=rotation,
        timecode=(stream.get("tags") or {}).get("timecode"),
    )
    if metadata.width <= 0 or metadata.height <= 0 or metadata.num_frames <= 0:
        raise ContractError(f"invalid video geometry/frame count: {video_path}")
    if metadata.time_base <= 0 or metadata.average_frame_rate <= 0:
        raise ContractError(f"invalid video time base/frame rate: {video_path}")
    if metadata.has_b_frames != 0:
        raise ContractError(
            f"this fixed packet-to-frame contract requires has_b_frames=0: {video_path}"
        )
    return metadata


def read_packet_timeline(
    video_path: Path,
    metadata: VideoStreamMetadata,
    ffprobe_binary: str,
    progress_interval_sec: float,
    log: LogFn,
) -> PacketTimeline:
    command = [
        ffprobe_binary,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_packets",
        "-show_entries",
        "packet=pts,duration",
        "-of",
        "csv=p=0",
        str(video_path),
    ]
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        bufsize=1,
    )
    if process.stdout is None or process.stderr is None:
        process.kill()
        raise ContractError("failed to create ffprobe pipes")
    pts_values: list[int] = []
    durations: list[int] = []
    last_report = time.monotonic()
    try:
        reader = csv.reader(process.stdout)
        for packet_index, row in enumerate(reader):
            if len(row) != 2 or any(value in ("", "N/A") for value in row):
                raise ContractError(
                    f"packet {packet_index} lacks exact pts/duration in {video_path}: {row}"
                )
            try:
                pts_values.append(int(row[0]))
                durations.append(int(row[1]))
            except ValueError as exc:
                raise ContractError(
                    f"non-integer packet pts/duration at {packet_index}: {row}"
                ) from exc
            now = time.monotonic()
            if now - last_report >= progress_interval_sec:
                log(
                    f"[s00] PTS {video_path.name}: {len(pts_values):,}/"
                    f"{metadata.num_frames:,} packets"
                )
                last_report = now
    except Exception:
        process.kill()
        process.wait()
        raise
    stderr = process.stderr.read()
    return_code = process.wait()
    if return_code != 0:
        raise ContractError(
            f"ffprobe packet scan failed ({return_code}) for {video_path}: {stderr.strip()}"
        )
    if len(pts_values) != metadata.num_frames:
        raise ContractError(
            f"packet/frame mismatch for {video_path}: packets={len(pts_values)}, "
            f"nb_frames={metadata.num_frames}"
        )
    if not pts_values:
        raise ContractError(f"video has no packet timestamps: {video_path}")
    if pts_values[0] != metadata.start_pts:
        raise ContractError(
            f"first packet PTS differs from stream start_pts: {pts_values[0]} != "
            f"{metadata.start_pts}"
        )
    for index, (pts, duration) in enumerate(zip(pts_values, durations)):
        if duration <= 0:
            raise ContractError(f"non-positive packet duration at frame {index}: {duration}")
        if index > 0:
            previous_end = pts_values[index - 1] + durations[index - 1]
            if pts != previous_end:
                raise ContractError(
                    f"non-contiguous packet PTS at frame {index}: {pts} != {previous_end}"
                )
    observed_duration = pts_values[-1] + durations[-1] - pts_values[0]
    if observed_duration != metadata.duration_ticks:
        raise ContractError(
            f"packet duration differs from stream duration_ts: {observed_duration} != "
            f"{metadata.duration_ticks}"
        )
    log(
        f"[s00] PTS {video_path.name}: verified {len(pts_values):,} packets, "
        f"duration={float(metadata.duration_ticks * metadata.time_base):.6f}s"
    )
    return PacketTimeline(tuple(pts_values), tuple(durations))


def open_raw_video_capture(video_path: Path) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise ContractError(f"OpenCV cannot open video: {video_path}")
    if not hasattr(cv2, "CAP_PROP_ORIENTATION_AUTO"):
        capture.release()
        raise ContractError("OpenCV lacks CAP_PROP_ORIENTATION_AUTO")
    if not capture.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0):
        capture.release()
        raise ContractError(f"cannot disable OpenCV auto-rotation: {video_path}")
    if capture.get(cv2.CAP_PROP_ORIENTATION_AUTO) != 0.0:
        capture.release()
        raise ContractError(f"OpenCV auto-rotation remained enabled: {video_path}")
    return capture


def verify_raw_video_geometry(
    video_path: Path, metadata: VideoStreamMetadata
) -> tuple[int, int]:
    capture = open_raw_video_capture(video_path)
    try:
        property_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        property_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        ok, frame = capture.read()
        if not ok or frame is None:
            raise ContractError(f"OpenCV cannot decode first raw frame: {video_path}")
        frame_height, frame_width = frame.shape[:2]
    finally:
        capture.release()
    expected = (metadata.width, metadata.height)
    if (property_width, property_height) != expected:
        raise ContractError(
            f"OpenCV raw property geometry differs from encoded stream for {video_path}: "
            f"{(property_width, property_height)} != {expected}"
        )
    if (frame_width, frame_height) != expected:
        raise ContractError(
            f"OpenCV raw frame geometry differs from encoded stream for {video_path}: "
            f"{(frame_width, frame_height)} != {expected}"
        )
    return expected

