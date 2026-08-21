from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

from app15 import (  # noqa: E402
    CLASSES,
    OUTPUT_ROOT,
    SOURCE_VIDEO_ROOT,
    GroupPaths,
    empty_annotation,
    image_size_from_png,
    read_json,
    utc_now,
    validate_annotation,
    write_convention_doc,
    write_json,
    write_preview_svg,
)


LEGACY_OUTPUT_DIR = ROOT / "outputs"
FARM_RE = re.compile(r"Dairy Farm (?P<farm>\d+) Videos", re.IGNORECASE)
CAMERA_RE = re.compile(r"^gopro(?P<camera>\d+)$", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build output15 reference frames and group manifest.")
    parser.add_argument("--source-root", type=Path, default=SOURCE_VIDEO_ROOT)
    parser.add_argument("--timestamp", default="00:01:00")
    parser.add_argument(
        "--video-index",
        type=int,
        default=1,
        help="Zero-based MP4 index. 1 means the second video after sorting.",
    )
    parser.add_argument("--force-frames", action="store_true", help="Re-extract existing valid frames.")
    parser.add_argument("--no-extract", action="store_true", help="Create dirs/metadata only; do not run ffmpeg.")
    return parser.parse_args()


def farm_id_from_path(path: Path) -> int | None:
    for part in path.parts:
        match = FARM_RE.search(part)
        if match:
            return int(match.group("farm"))
    return None


def discover_mp4s(video_dir: Path) -> list[Path]:
    return sorted(
        (path for path in video_dir.rglob("*") if path.is_file() and path.suffix.lower() == ".mp4"),
        key=lambda path: str(path).lower(),
    )


def discover_groups(source_root: Path, video_index: int) -> list[dict[str, Any]]:
    if not source_root.exists():
        raise FileNotFoundError(f"source root does not exist: {source_root}")

    by_key: dict[tuple[int, int], dict[str, Any]] = {}
    for path in source_root.rglob("*"):
        if not path.is_dir():
            continue
        camera_match = CAMERA_RE.fullmatch(path.name)
        if not camera_match:
            continue
        farm_id = farm_id_from_path(path)
        if farm_id is None:
            continue
        camera_id = int(camera_match.group("camera"))
        group_id = f"farm_ID_{farm_id}_camera_ID_{camera_id}"
        mp4s = discover_mp4s(path)
        if len(mp4s) <= video_index:
            raise FileNotFoundError(
                f"{group_id} needs video index {video_index}, but only found {len(mp4s)} MP4 files under {path}"
            )
        source_video = mp4s[video_index]
        by_key[(farm_id, camera_id)] = {
            "id": group_id,
            "farmId": farm_id,
            "cameraId": camera_id,
            "label": f"Farm {farm_id} / Camera {camera_id}",
            "sourceVideoDir": str(path),
            "sourceVideo": str(source_video),
            "sourceVideoName": source_video.name,
            "videoCount": len(mp4s),
            "videoIndex": video_index,
            "firstVideo": mp4s[0].name if mp4s else None,
            "secondVideo": source_video.name,
            "outputDir": str((OUTPUT_ROOT / group_id).relative_to(ROOT)).replace("\\", "/"),
        }

    groups = [by_key[key] for key in sorted(by_key)]
    expected = {(farm_id, camera_id) for farm_id in range(1, 4) for camera_id in range(1, 6)}
    missing = sorted(expected.difference(by_key))
    if missing:
        missing_text = ", ".join(f"farm {farm} camera {camera}" for farm, camera in missing)
        raise FileNotFoundError(f"missing expected camera directories: {missing_text}")
    return groups


def valid_png(path: Path) -> bool:
    width, height = image_size_from_png(path)
    return bool(width and height)


def copy_overwrite(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as src, target.open("wb") as dst:
        shutil.copyfileobj(src, dst, length=1024 * 1024)


def frame_metadata(group: dict[str, Any], paths: GroupPaths, timestamp: str) -> dict[str, Any]:
    width, height = image_size_from_png(paths.frame)
    return {
        "createdAt": utc_now(),
        "groupId": group["id"],
        "farmId": group["farmId"],
        "cameraId": group["cameraId"],
        "sourceVideoDir": group["sourceVideoDir"],
        "sourceVideo": group["sourceVideo"],
        "sourceVideoName": group["sourceVideoName"],
        "videoIndex": group["videoIndex"],
        "timestamp": timestamp,
        "output": str(paths.frame.relative_to(ROOT)).replace("\\", "/"),
        "outputAbsolute": str(paths.frame),
        "width": width,
        "height": height,
        "format": "png",
        "rotationApplied": False,
        "scaleApplied": False,
        "ffmpegNoAutorotate": True,
        "notes": [
            "The second MP4 is selected with zero-based videoIndex=1.",
            "The output frame is decoded from the original MP4.",
            "No resize, crop, or perspective correction is applied.",
        ],
    }


def extract_frame(group: dict[str, Any], paths: GroupPaths, timestamp: str, force: bool, no_extract: bool) -> None:
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    if valid_png(paths.frame) and not force:
        return
    if no_extract:
        return
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is not available on PATH")
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        timestamp,
        "-noautorotate",
        "-i",
        group["sourceVideo"],
        "-frames:v",
        "1",
        str(paths.frame),
    ]
    subprocess.run(command, check=True)
    if not valid_png(paths.frame):
        raise RuntimeError(f"ffmpeg did not create a valid PNG: {paths.frame}")


def migrate_completed_group(group: dict[str, Any], paths: GroupPaths, timestamp: str) -> None:
    legacy_frame = LEGACY_OUTPUT_DIR / "reference_frame.png"
    legacy_annotation = LEGACY_OUTPUT_DIR / "floorplan_annotation.json"

    if not legacy_frame.exists():
        raise FileNotFoundError(f"completed legacy frame is missing: {legacy_frame}")
    copy_overwrite(legacy_frame, paths.frame)
    write_json(paths.frame_meta, frame_metadata(group, paths, timestamp))

    annotation_data = read_json(legacy_annotation, None)
    if annotation_data is None:
        annotation = empty_annotation(paths)
    else:
        annotation_data["classes"] = CLASSES
        annotation = validate_annotation(annotation_data, paths)
    write_json(paths.annotation, annotation)
    write_preview_svg(annotation, paths)
    write_convention_doc(annotation, paths)


def ensure_empty_annotation(paths: GroupPaths) -> None:
    if paths.annotation.exists():
        try:
            existing = read_json(paths.annotation, None)
            if isinstance(existing, dict):
                return
        except Exception:
            pass
    annotation = empty_annotation(paths)
    write_json(paths.annotation, annotation)
    write_preview_svg(annotation, paths)
    write_convention_doc(annotation, paths)


def write_manifest(groups: list[dict[str, Any]], timestamp: str, source_root: Path, video_index: int) -> None:
    manifest = {
        "schemaVersion": "floorplananno.groups.v1",
        "createdAt": utc_now(),
        "sourceRoot": str(source_root),
        "outputRoot": "output15",
        "timestamp": timestamp,
        "videoIndex": video_index,
        "groups": groups,
    }
    write_json(OUTPUT_ROOT / "floorplan_groups.json", manifest)


def main() -> None:
    args = parse_args()
    groups = discover_groups(args.source_root, args.video_index)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    for group in groups:
        paths = GroupPaths(group["id"])
        if group["id"] == "farm_ID_1_camera_ID_1":
            migrate_completed_group(group, paths, args.timestamp)
            status = "migrated"
        else:
            extract_frame(group, paths, args.timestamp, args.force_frames, args.no_extract)
            if valid_png(paths.frame):
                write_json(paths.frame_meta, frame_metadata(group, paths, args.timestamp))
            ensure_empty_annotation(paths)
            status = "ready" if valid_png(paths.frame) else "metadata-only"
        print(f"{status}: {group['id']} -> {paths.output_dir}")

    write_manifest(groups, args.timestamp, args.source_root, args.video_index)
    print(f"Wrote manifest: {OUTPUT_ROOT / 'floorplan_groups.json'}")


if __name__ == "__main__":
    main()
