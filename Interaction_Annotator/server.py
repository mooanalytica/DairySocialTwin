from __future__ import annotations

import csv
import json
import mimetypes
import os
import re
import subprocess
import sys
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


WORKSPACE = Path(__file__).resolve().parent
WEB_ROOT = WORKSPACE / "web"
INDEX_PATH = WORKSPACE / "output" / "index.csv"
DISCARDED_PATH = WORKSPACE / "output" / "discarded_clips.txt"
PENDING_DELETES_PATH = WORKSPACE / "output" / "pending_deletes.csv"
CACHED_VIDEOS_ROOT = WORKSPACE / "output" / "cached_videos"
CACHE_VIS_ROOT = WORKSPACE / "output" / "cache_vis"
VALID_VIDEOS_ROOT = WORKSPACE / "output" / "videos"
FAKE_INTERACTION_ROOT = WORKSPACE / "output" / "fake_interaction"
DEFAULT_CANDIDATE_ROOT = CACHED_VIDEOS_ROOT

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

FAKE_INTERACTION_VALENCE = "fake_interaction"
VALID_VALENCE = {"friendly", "unfriendly", FAKE_INTERACTION_VALENCE}
VALID_FINE = {"", "Displacement", "Grooming", "Headbutting", "Licking"}
PENDING_DELETE_COLUMNS = [
    "clip_id",
    "kind",
    "root",
    "path",
    "status",
    "attempts",
    "last_error",
    "queued_at",
    "updated_at",
    "deleted_at",
]


def log(message: str) -> None:
    print(message, flush=True)


def read_index() -> list[dict[str, str]]:
    if not INDEX_PATH.exists():
        return []
    with INDEX_PATH.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_index(rows: list[dict[str, str]]) -> None:
    if not INDEX_PATH.exists():
        return
    with INDEX_PATH.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
    with INDEX_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def timestamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def clip_by_id(clip_id: str) -> dict[str, str] | None:
    for row in read_index():
        if row.get("clip_id") == clip_id:
            return row
    return None


def clip_item(row: dict[str, str]) -> dict[str, object]:
    item = dict(row)
    annotation_state, annotation_valence = annotation_summary(row)
    item["annotation_status"] = annotation_state
    item["annotation_valence"] = annotation_valence
    item["clip_exists"] = cache_path_for_row(row).exists()
    item["cache_vis_exists"] = cache_vis_path_for_row(row).exists()
    item["valid_clip_exists"] = valid_path_for_row(row).exists()
    item["review_source_start_frame"] = str(cache_source_start(row))
    item["review_source_end_frame"] = str(cache_source_end(row))
    item["review_frame_count"] = str(cache_frame_count(row))
    item["review_duration_s"] = row.get("cache_duration_s") or row.get("duration_s", "")
    return item


def replace_index_row(updated_row: dict[str, str]) -> None:
    rows = read_index()
    replaced = False
    for idx, row in enumerate(rows):
        if row.get("clip_id") == updated_row.get("clip_id"):
            rows[idx] = {**row, **updated_row}
            replaced = True
            break
    if not replaced:
        raise FileNotFoundError("clip not found")
    write_index(rows)


def cache_path_for_row(row: dict[str, str]) -> Path:
    return Path(row.get("cache_path") or row["clip_path"])


def cache_root_for_row(row: dict[str, str]) -> Path:
    return Path(row.get("cache_root") or row.get("clip_root") or CACHED_VIDEOS_ROOT)


def cache_vis_path_for_row(row: dict[str, str]) -> Path:
    return Path(row.get("cache_vis_path") or row.get("cache_path") or row["clip_path"])


def cache_vis_root_for_row(row: dict[str, str]) -> Path:
    return Path(row.get("cache_vis_root") or row.get("cache_root") or CACHE_VIS_ROOT)


def media_path_for_row(row: dict[str, str]) -> Path:
    cache_vis_path = cache_vis_path_for_row(row)
    if cache_vis_path.exists():
        return cache_vis_path
    return cache_path_for_row(row)


def valid_path_for_row(row: dict[str, str]) -> Path:
    return Path(row["clip_path"])


def valid_root_for_row(row: dict[str, str]) -> Path:
    return Path(row.get("clip_root") or VALID_VIDEOS_ROOT)


def valid_rel_path_for_row(row: dict[str, str]) -> Path:
    rel = row.get("clip_rel_path") or row.get("cache_rel_path") or Path(row["clip_path"]).name
    rel_path = Path(rel)
    if rel_path.is_absolute():
        rel_path = Path(rel_path.name)
    return rel_path


def valid_path_for_root(row: dict[str, str], root: Path) -> Path:
    return root / valid_rel_path_for_row(row)


def cache_source_start(row: dict[str, str]) -> int:
    return int(row.get("cache_source_start_frame") or row["source_start_frame"])


def cache_source_end(row: dict[str, str]) -> int:
    return int(row.get("cache_source_end_frame") or row["source_end_frame"])


def cache_frame_count(row: dict[str, str]) -> int:
    return cache_source_end(row) - cache_source_start(row) + 1


def write_csv_direct(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ANNOTATION_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_annotation(row: dict[str, str]) -> tuple[list[dict[str, str]], bool]:
    path = Path(row["annotation_path"])
    if not path.exists():
        return [], False
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                raise ValueError("missing header")
            rows = list(reader)
        parsed: list[dict[str, str]] = []
        for item in rows:
            if not item.get("event_id"):
                continue
            parsed.append(item)
        return parsed, False
    except Exception as exc:
        log(f"clearing unreadable annotation for {row.get('clip_id')}: {exc}")
        write_csv_direct(path, [])
        return [], True


def annotation_status(row: dict[str, str]) -> str:
    state, _ = annotation_summary(row)
    return state


def annotation_summary(row: dict[str, str]) -> tuple[str, str]:
    rows, cleared = read_annotation(row)
    if cleared:
        return "cleared", ""
    if not rows:
        return "empty", ""

    valences = {str(item.get("valence", "")).strip() for item in rows if item.get("valence")}
    if "friendly" in valences and "unfriendly" in valences:
        return "annotated", "mixed_friendly_unfriendly"
    if FAKE_INTERACTION_VALENCE in valences:
        return "annotated", FAKE_INTERACTION_VALENCE
    if "friendly" in valences:
        return "annotated", "friendly"
    if "unfriendly" in valences:
        return "annotated", "unfriendly"
    return "annotated", ""


def assert_under(path: Path, root: Path) -> Path:
    resolved = path.resolve()
    resolved_root = root.resolve()
    if resolved != resolved_root and resolved_root not in resolved.parents:
        raise ValueError(f"refusing to delete outside allowed root: {resolved}")
    return resolved


def remove_file_if_present(path: Path, root: Path) -> bool:
    resolved = assert_under(path, root)
    if not resolved.exists():
        return False
    if not resolved.is_file():
        raise ValueError(f"refusing to delete non-file path: {resolved}")
    resolved.unlink()
    return True


def read_pending_deletes() -> list[dict[str, str]]:
    if not PENDING_DELETES_PATH.exists():
        return []
    with PENDING_DELETES_PATH.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_pending_deletes(rows: list[dict[str, str]]) -> None:
    PENDING_DELETES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with PENDING_DELETES_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PENDING_DELETE_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def queue_pending_delete(clip_id: str, kind: str, path: Path, root: Path, error: Exception) -> None:
    resolved = assert_under(path, root)
    rows = read_pending_deletes()
    key = (clip_id, kind, str(resolved))
    now = timestamp()
    for row in rows:
        row_key = (row.get("clip_id", ""), row.get("kind", ""), row.get("path", ""))
        if row_key == key:
            row["status"] = "pending"
            row["last_error"] = repr(error)
            row["updated_at"] = now
            write_pending_deletes(rows)
            return
    rows.append(
        {
            "clip_id": clip_id,
            "kind": kind,
            "root": str(root.resolve()),
            "path": str(resolved),
            "status": "pending",
            "attempts": "0",
            "last_error": repr(error),
            "queued_at": now,
            "updated_at": now,
            "deleted_at": "",
        }
    )
    write_pending_deletes(rows)


def remove_or_queue(clip_id: str, kind: str, path: Path, root: Path) -> dict[str, object]:
    resolved = assert_under(path, root)
    if not resolved.exists():
        return {"removed": False, "queued": False, "missing": True}
    if not resolved.is_file():
        raise ValueError(f"refusing to delete non-file path: {resolved}")
    try:
        resolved.unlink()
        return {"removed": True, "queued": False, "missing": False}
    except OSError as exc:
        queue_pending_delete(clip_id, kind, resolved, root, exc)
        return {"removed": False, "queued": True, "missing": False}


def prune_empty_parents(path: Path, root: Path) -> None:
    resolved_root = root.resolve()
    parent = path.resolve().parent
    while parent != resolved_root and resolved_root in parent.parents:
        try:
            parent.rmdir()
        except OSError:
            return
        parent = parent.parent


def mark_discarded(clip_id: str) -> None:
    DISCARDED_PATH.parent.mkdir(parents=True, exist_ok=True)
    existing: set[str] = set()
    if DISCARDED_PATH.exists():
        with DISCARDED_PATH.open("r", encoding="utf-8") as f:
            existing = {line.strip() for line in f if line.strip()}
    if clip_id in existing:
        return
    with DISCARDED_PATH.open("a", encoding="utf-8") as f:
        f.write(clip_id + "\n")


def delete_clip(clip_id: str) -> dict[str, object]:
    rows = read_index()
    target = None
    kept_rows: list[dict[str, str]] = []
    for row in rows:
        if row.get("clip_id") == clip_id:
            target = row
        else:
            kept_rows.append(row)
    if not target:
        raise FileNotFoundError("clip not found")

    cache_root = cache_root_for_row(target)
    cache_vis_root = cache_vis_root_for_row(target)
    valid_root = valid_root_for_row(target)
    annotation_root = Path(target["annotation_root"])
    cache_path = cache_path_for_row(target)
    cache_vis_path = cache_vis_path_for_row(target)
    valid_path = valid_path_for_row(target)
    annotation_path = Path(target["annotation_path"])
    cache_result = remove_or_queue(clip_id, "cache", cache_path, cache_root)
    if target.get("cache_vis_path") and cache_vis_path.resolve() != cache_path.resolve():
        cache_vis_result = remove_or_queue(clip_id, "cache_vis", cache_vis_path, cache_vis_root)
    else:
        cache_vis_result = {"removed": False, "queued": False, "missing": True}
    valid_result = remove_or_queue(clip_id, "valid", valid_path, valid_root)
    annotation_result = remove_or_queue(clip_id, "annotation", annotation_path, annotation_root)
    if cache_result["removed"]:
        prune_empty_parents(cache_path, cache_root)
    if cache_vis_result["removed"]:
        prune_empty_parents(cache_vis_path, cache_vis_root)
    if valid_result["removed"]:
        prune_empty_parents(valid_path, valid_root)
    if annotation_result["removed"]:
        prune_empty_parents(annotation_path, annotation_root)
    mark_discarded(clip_id)
    write_index(kept_rows)
    return {
        "clip_id": clip_id,
        "removed_cache": cache_result["removed"],
        "removed_cache_vis": cache_vis_result["removed"],
        "removed_clip": valid_result["removed"],
        "removed_annotation": annotation_result["removed"],
        "queued_cache": cache_result["queued"],
        "queued_cache_vis": cache_vis_result["queued"],
        "queued_clip": valid_result["queued"],
        "queued_annotation": annotation_result["queued"],
        "removed_from_index": True,
    }


def json_bytes(payload: object) -> bytes:
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")


def clamp_int(value: object, low: int, high: int) -> int:
    number = int(round(float(value)))
    return max(low, min(high, number))


def clamp_float(value: object, low: float, high: float) -> float:
    number = float(value)
    return max(low, min(high, number))


def safe_rel(path: Path, root: Path) -> str:
    return str(path.resolve().relative_to(root.resolve()))


def probe_video_size(path: Path) -> tuple[int, int]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe video size failed for {path}:\n{result.stderr[-2000:]}")
    data = json.loads(result.stdout)
    streams = data.get("streams") or []
    if not streams:
        raise RuntimeError(f"No video stream found: {path}")
    stream = streams[0]
    return int(stream["width"]), int(stream["height"])


def run_ffmpeg_trim(cache_path: Path, valid_path: Path, start_frame: int, end_frame: int) -> None:
    frame_count = end_frame - start_frame + 1
    valid_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-y",
        "-i",
        str(cache_path),
        "-vf",
        f"trim=start_frame={start_frame}:end_frame={end_frame + 1},setpts=PTS-STARTPTS",
        "-frames:v",
        str(frame_count),
        "-an",
        "-c:v",
        "h264_nvenc",
        "-preset",
        "p4",
        "-cq",
        "24",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(valid_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg valid clip export failed:\n{result.stderr[-4000:]}")


def validate_annotation_boxes(payload: dict[str, object]) -> tuple[list[dict[str, object]], Path]:
    boxes = payload.get("boxes")
    if not isinstance(boxes, list):
        raise ValueError("boxes must be a list")
    if not boxes:
        raise ValueError("at least one ROI is required before saving")

    valid_boxes: list[dict[str, object]] = []
    has_fake = False
    has_non_fake = False
    for box in boxes:
        if not isinstance(box, dict):
            continue
        valence = str(box.get("valence", "")).strip()
        fine_class = str(box.get("fine_class", "")).strip()
        if valence not in VALID_VALENCE:
            raise ValueError(f"invalid valence: {valence}")
        if fine_class not in VALID_FINE:
            raise ValueError(f"invalid fine_class: {fine_class}")
        has_fake = has_fake or valence == FAKE_INTERACTION_VALENCE
        has_non_fake = has_non_fake or valence != FAKE_INTERACTION_VALENCE
        valid_boxes.append(box)

    if not valid_boxes:
        raise ValueError("at least one ROI is required before saving")
    if has_fake and has_non_fake:
        raise ValueError("fake_interaction cannot be mixed with friendly/unfriendly in one clip")
    return valid_boxes, FAKE_INTERACTION_ROOT if has_fake else VALID_VIDEOS_ROOT


def save_valid_clip(
    index_row: dict[str, str], selected_start: int, selected_end: int, target_valid_root: Path
) -> dict[str, str]:
    fps = float(index_row["fps"])
    cache_root = cache_root_for_row(index_row)
    cache_path = cache_path_for_row(index_row)
    previous_valid_root = valid_root_for_row(index_row)
    previous_valid_path = valid_path_for_row(index_row)
    valid_root = target_valid_root
    valid_path = valid_path_for_root(index_row, valid_root)
    assert_under(cache_path, cache_root)
    assert_under(valid_path, valid_root)
    if not cache_path.exists():
        raise FileNotFoundError(f"cached video not found: {cache_path}")

    run_ffmpeg_trim(cache_path, valid_path, selected_start, selected_end)
    width, height = probe_video_size(valid_path)
    if previous_valid_path.resolve() != valid_path.resolve() and previous_valid_path.exists():
        previous_result = remove_or_queue(
            index_row["clip_id"], "valid", previous_valid_path, previous_valid_root
        )
        if previous_result["removed"]:
            prune_empty_parents(previous_valid_path, previous_valid_root)

    valid_frame_count = selected_end - selected_start + 1
    source_start = cache_source_start(index_row) + selected_start
    source_end = cache_source_start(index_row) + selected_end
    updated = dict(index_row)
    updated.update(
        {
            "clip_root": str(valid_root.resolve()),
            "clip_rel_path": safe_rel(valid_path, valid_root),
            "clip_path": str(valid_path.resolve()),
            "source_start_frame": str(source_start),
            "source_end_frame": str(source_end),
            "clip_frame_offset": str(source_start),
            "duration_s": f"{valid_frame_count / fps:.6f}",
            "width": str(width),
            "height": str(height),
            "selected_cache_start_frame": str(selected_start),
            "selected_cache_end_frame": str(selected_end),
            "is_saved": "true",
        }
    )
    replace_index_row(updated)
    return updated


def make_annotation_rows(index_row: dict[str, str], payload: dict[str, object]) -> list[dict[str, str]]:
    fps = float(index_row["fps"])
    frame_count = cache_frame_count(index_row)
    selected_start = clamp_int(payload.get("start_frame", 0), 0, frame_count - 1)
    selected_end = clamp_int(payload.get("end_frame", frame_count - 1), selected_start, frame_count - 1)
    valid_frame_count = selected_end - selected_start + 1
    boxes, target_valid_root = validate_annotation_boxes(payload)

    updated_index_row = save_valid_clip(index_row, selected_start, selected_end, target_valid_root)
    width = int(updated_index_row["width"])
    height = int(updated_index_row["height"])
    rows: list[dict[str, str]] = []
    clip_rel_path = updated_index_row["clip_rel_path"]
    annotation_path = Path(updated_index_row["annotation_path"])
    csv_root = str(index_row.get("csv_root", "")).strip()
    csv_rel_path = str(index_row.get("csv_rel_path", "")).strip()
    if not csv_root:
        raise ValueError(f"missing csv_root in index row for {index_row['clip_id']}")

    for idx, box in enumerate(boxes, start=1):
        if not isinstance(box, dict):
            continue
        valence = str(box.get("valence", "")).strip()
        fine_class = str(box.get("fine_class", "")).strip()

        x = clamp_float(box.get("x", 0), 0, max(0, width - 1))
        y = clamp_float(box.get("y", 0), 0, max(0, height - 1))
        w = clamp_float(box.get("w", 1), 1, width - x)
        h = clamp_float(box.get("h", 1), 1, height - y)

        rows.append(
            {
                "event_id": f"{index_row['clip_id']}_E{idx:04d}",
                "event_group_id": str(payload.get("event_group_id", "")),
                "clip_id": index_row["clip_id"],
                "clip_root": updated_index_row["clip_root"],
                "clip_rel_path": clip_rel_path,
                "clip_path": updated_index_row["clip_path"],
                "csv_root": csv_root,
                "csv_rel_path": csv_rel_path,
                "start_frame": "0",
                "end_frame": str(valid_frame_count - 1),
                "roi_x": f"{x:.3f}",
                "roi_y": f"{y:.3f}",
                "roi_w": f"{w:.3f}",
                "roi_h": f"{h:.3f}",
                "valence": valence,
                "fine_class": fine_class,
                "allow_duplicate_pair": "false",
                "fps": updated_index_row["fps"],
                "width": updated_index_row["width"],
                "height": updated_index_row["height"],
                "start_time_s": "0.000000",
                "end_time_s": f"{(valid_frame_count - 1) / fps:.6f}",
                "annotator": str(payload.get("annotator", "")),
                "label_confidence": str(payload.get("label_confidence", "")),
                "notes": str(box.get("notes", "")),
                "exclude": "false",
            }
        )
    write_csv_direct(annotation_path, rows)
    return rows


class Handler(BaseHTTPRequestHandler):
    server_version = "InteractionAnnotator/1.0"

    def log_message(self, fmt: str, *args: object) -> None:
        log(f"{self.address_string()} - {fmt % args}")

    def send_json(self, payload: object, status: int = 200) -> None:
        body = json_bytes(payload)
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_error_json(self, status: int, message: str) -> None:
        self.send_json({"error": message}, status)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        if path == "/":
            return self.serve_file(WEB_ROOT / "index.html")
        if path.startswith("/static/"):
            return self.serve_file(WEB_ROOT / path.removeprefix("/static/"))
        if path == "/api/clips":
            rows = read_index()
            payload = [clip_item(row) for row in rows]
            return self.send_json({"clips": payload})
        if path == "/api/annotation":
            clip_id = parse_qs(parsed.query).get("clip_id", [""])[0]
            row = clip_by_id(clip_id)
            if not row:
                return self.send_error_json(404, "clip not found")
            rows, cleared = read_annotation(row)
            return self.send_json({"rows": rows, "cleared": cleared})
        if path.startswith("/media/"):
            clip_id = Path(path).stem
            row = clip_by_id(clip_id)
            if not row:
                return self.send_error_json(404, "clip not found")
            return self.serve_file(media_path_for_row(row), ranged=True)
        return self.send_error_json(404, "not found")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        if path != "/api/annotation":
            return self.send_error_json(404, "not found")
        length = int(self.headers.get("Content-Length", "0"))
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            clip_id = str(payload.get("clip_id", ""))
            row = clip_by_id(clip_id)
            if not row:
                return self.send_error_json(404, "clip not found")
            rows = make_annotation_rows(row, payload)
            updated_row = clip_by_id(clip_id) or row
        except Exception as exc:
            return self.send_error_json(400, str(exc))
        return self.send_json({"ok": True, "rows": rows, "clip": clip_item(updated_row)})

    def do_DELETE(self) -> None:
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        if path != "/api/clip":
            return self.send_error_json(404, "not found")
        clip_id = parse_qs(parsed.query).get("clip_id", [""])[0]
        try:
            result = delete_clip(clip_id)
        except FileNotFoundError as exc:
            return self.send_error_json(404, str(exc))
        except Exception as exc:
            return self.send_error_json(400, str(exc))
        return self.send_json({"ok": True, **result})

    def serve_file(self, path: Path, ranged: bool = False) -> None:
        try:
            resolved = path.resolve()
            if not resolved.exists() or not resolved.is_file():
                return self.send_error_json(404, "file not found")
            file_size = resolved.stat().st_size
            content_type = mimetypes.guess_type(str(resolved))[0] or "application/octet-stream"
            range_header = self.headers.get("Range") if ranged else None
            start = 0
            end = file_size - 1
            status = HTTPStatus.OK
            if range_header:
                match = re.match(r"bytes=(\d*)-(\d*)", range_header)
                if match:
                    if match.group(1):
                        start = int(match.group(1))
                    if match.group(2):
                        end = int(match.group(2))
                    status = HTTPStatus.PARTIAL_CONTENT
            start = max(0, min(start, file_size - 1))
            end = max(start, min(end, file_size - 1))
            length = end - start + 1
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(length))
            if status == HTTPStatus.PARTIAL_CONTENT:
                self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
            self.end_headers()
            with resolved.open("rb") as f:
                f.seek(start)
                remaining = length
                while remaining > 0:
                    chunk = f.read(min(1024 * 1024, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
        except BrokenPipeError:
            return


def main() -> int:
    host = "127.0.0.1"
    port = int(os.environ.get("INTERACTION_ANNOTATOR_PORT", "8765"))
    server = ThreadingHTTPServer((host, port), Handler)
    log(f"Interaction Annotator running at http://{host}:{port}/")
    log(f"index: {INDEX_PATH}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log("stopping")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
