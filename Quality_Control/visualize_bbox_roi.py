from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from itertools import combinations
from pathlib import Path, PureWindowsPath

import cv2

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - tqdm is optional at runtime
    tqdm = None


WORKDIR = Path(__file__).resolve().parent
ANNOTATOR_ROOT = Path("/home/hyw/Interaction_Annotator-V0601_03_AND_05R")
ANNOTATOR_OUTPUT_ROOT = ANNOTATOR_ROOT / "output"
INDEX_CSV = ANNOTATOR_OUTPUT_ROOT / "index.csv"
ANNOTATION_ROOT = ANNOTATOR_OUTPUT_ROOT / "annotations"
VIDEO_ROOT = ANNOTATOR_OUTPUT_ROOT / "videos"
FAKE_ROOT = ANNOTATOR_OUTPUT_ROOT / "fake_interaction"
STAGE1_ROOT = Path("/home/hyw/V0604_SEG_S1_TI")
OUTPUT_ROOT = WORKDIR / "output"
LOG_PATH = OUTPUT_ROOT / "run.log"

REQUIRED_ANN_COLS = {
    "event_id",
    "clip_id",
    "clip_root",
    "clip_rel_path",
    "clip_path",
    "csv_root",
    "start_frame",
    "end_frame",
    "roi_x",
    "roi_y",
    "roi_w",
    "roi_h",
    "valence",
}
REQUIRED_TRACK_COLS = {"video", "frame", "track_id", "x", "y", "w", "h"}
ALLOWED_VALENCES = {"friendly", "unfriendly", "fake_interaction"}


def configure_stdio() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def log(message: str) -> None:
    print(message, flush=True)
    if LOG_PATH.parent.exists():
        with LOG_PATH.open("a", encoding="utf-8", newline="\n") as f:
            f.write(message + "\n")
            f.flush()


def safe_reset_output() -> None:
    out = OUTPUT_ROOT.resolve()
    root = WORKDIR.resolve()
    try:
        out.relative_to(root)
    except ValueError as exc:
        raise RuntimeError(f"Refusing to clear output outside workspace: {out}") from exc
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "preview_frames").mkdir(parents=True, exist_ok=True)


def norm(path: str | Path) -> str:
    return os.path.normcase(os.path.normpath(str(path)))


def is_under(path: str | Path, root: Path) -> bool:
    try:
        Path(path).resolve().relative_to(root.resolve())
        return True
    except Exception:
        return norm(path).startswith(norm(root) + os.sep)


def read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or []), list(reader)


def split_path_parts(value: str) -> list[str]:
    return [part for part in value.replace("\\", "/").split("/") if part]


def rel_to_path(rel_path: str) -> Path:
    return Path(*split_path_parts(rel_path)) if rel_path else Path()


def basename_any(path: str) -> str:
    if "\\" in path or ":" in path:
        return PureWindowsPath(path).name
    return Path(path).name


def is_true(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def normalized_text_path(path: str) -> str:
    return path.replace("\\", "/").rstrip("/").lower()


def path_category(path: str) -> str:
    text = normalized_text_path(path)
    if text.endswith("/output/videos") or "/output/videos/" in text:
        return "videos"
    if text.endswith("/output/fake_interaction") or "/output/fake_interaction/" in text:
        return "fake_interaction"
    return ""


def clip_rel_key(value: str) -> str:
    return normalized_text_path(value).strip("/")


def stage1_rel_key(value: str) -> str:
    text = clip_rel_key(value)
    if text.startswith("input/"):
        text = text[len("input/") :]
    return text


def path_key(path: str | Path) -> str:
    return normalized_text_path(str(path))


def clip_match_keys(value: str | Path) -> list[str]:
    text = clip_rel_key(str(value))
    parts = split_path_parts(text)
    keys: list[str] = []
    for index, part in enumerate(parts):
        if part.lower() == "videos" and index + 2 < len(parts):
            keys.append("/".join(parts[index + 1 :]))
    for index, part in enumerate(parts):
        if part.lower().startswith("sv_"):
            keys.append("/".join(parts[index:]))
            break
    if len(parts) >= 2:
        keys.append("/".join(parts[-2:]))
    if parts:
        keys.append(parts[-1])
    return ordered_unique(keys)


def ordered_unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def local_clip_path(index_row: dict[str, str]) -> Path:
    clip_root = index_row.get("clip_root", "")
    rel_path = index_row.get("clip_rel_path", "")
    category = path_category(clip_root) or path_category(index_row.get("clip_path", ""))
    if category == "videos":
        return VIDEO_ROOT / rel_to_path(rel_path)
    if category == "fake_interaction":
        return FAKE_ROOT / rel_to_path(rel_path)
    raise RuntimeError(
        f"{index_row.get('clip_id')}: unknown clip_root, cannot resolve clip path: {clip_root}"
    )


def local_annotation_path(index_row: dict[str, str]) -> Path:
    rel_path = index_row.get("annotation_rel_path", "")
    if not rel_path:
        raise RuntimeError(f"{index_row.get('clip_id')}: missing annotation_rel_path in index.csv")
    return ANNOTATION_ROOT / rel_to_path(rel_path)


def load_manifest_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def manifest_source_values(manifest: dict[str, object]) -> list[str]:
    video_manifest = manifest.get("video_manifest")
    feature_manifest = manifest.get("feature_csv_manifest")
    raw_values: list[str] = []
    if isinstance(video_manifest, dict):
        for key in ("source_path", "relative_path"):
            if video_manifest.get(key):
                raw_values.append(str(video_manifest[key]))
    if isinstance(feature_manifest, dict):
        for key in ("source_video_path", "source_video_rel_path"):
            if feature_manifest.get(key):
                raw_values.append(str(feature_manifest[key]))
    return ordered_unique(raw_values)


def localize_annotator_path(value: str) -> Path:
    parts = split_path_parts(value)
    for index, part in enumerate(parts):
        if part.lower() == ANNOTATOR_ROOT.name.lower():
            return ANNOTATOR_ROOT / Path(*parts[index + 1 :])
    return Path(value)


def manifest_source_paths(manifest: dict[str, object]) -> list[Path]:
    paths = [localize_annotator_path(value) for value in manifest_source_values(manifest)]
    return [Path(value) for value in ordered_unique([str(path) for path in paths])]


def manifest_source_path(manifest: dict[str, object]) -> str:
    source_paths = manifest_source_paths(manifest)
    if not source_paths:
        raise RuntimeError("Stage1 manifest missing source_path/source_video_path")
    return str(source_paths[0])


def manifest_rel_keys(manifest: dict[str, object]) -> list[str]:
    keys: list[str] = []
    for value in manifest_source_values(manifest):
        keys.extend(clip_match_keys(value))
    return ordered_unique(keys)


def manifest_dimensions(manifest: dict[str, object]) -> tuple[str, str]:
    video_manifest = manifest.get("video_manifest")
    if not isinstance(video_manifest, dict):
        return "", ""
    return str(video_manifest.get("width", "")), str(video_manifest.get("height", ""))


def build_stage1_manifest_index(index_rows: list[dict[str, str]]) -> dict[str, dict[str, object]]:
    if not STAGE1_ROOT.exists():
        raise FileNotFoundError(f"Stage1 root not found: {STAGE1_ROOT}")

    manifests: list[dict[str, object]] = []
    by_key: dict[str, list[dict[str, object]]] = defaultdict(list)
    errors: list[str] = []
    for manifest_path in sorted(STAGE1_ROOT.rglob("manifest.json")):
        try:
            manifest = load_manifest_json(manifest_path)
            source_path = manifest_source_path(manifest)
            source_keys = manifest_rel_keys(manifest)
            if not source_keys:
                errors.append(f"{manifest_path}: no usable Stage1 source relative path")
                continue
            tracking_csv = manifest_path.parent / "tracking_boxes.csv"
            keypoints_csv = manifest_path.parent / "keypoints.csv"
            width, height = manifest_dimensions(manifest)
            info = {
                "rel_keys": source_keys,
                "rel_key": source_keys[0],
                "source_path": source_path,
                "category": "videos",
                "manifest_path": str(manifest_path),
                "csv_root": str(manifest_path.parent),
                "tracking_csv": str(tracking_csv),
                "keypoints_csv": str(keypoints_csv),
                "width": width,
                "height": height,
            }
            manifests.append(info)
            for rel_key in source_keys:
                by_key[rel_key].append(info)
            if not tracking_csv.exists():
                errors.append(f"{source_keys[0]}: tracking_boxes.csv missing: {tracking_csv}")
            if not keypoints_csv.exists():
                errors.append(f"{source_keys[0]}: keypoints.csv missing: {keypoints_csv}")
        except Exception as exc:
            errors.append(f"{manifest_path}: {exc}")

    index_by_key: dict[str, dict[str, str]] = {}
    index_match_keys: dict[str, list[str]] = {}
    for row in index_rows:
        if not is_true(row.get("is_saved", "true")):
            continue
        category = path_category(row.get("clip_root", "")) or path_category(row.get("clip_path", ""))
        if category == "fake_interaction":
            continue
        if category != "videos":
            errors.append(f"{row.get('clip_id')}: unsupported clip category: {row.get('clip_root')}")
            continue
        try:
            clip_path = local_clip_path(row)
            key = path_key(clip_path)
        except Exception as exc:
            errors.append(f"{row.get('clip_id')}: {exc}")
            continue
        if key in index_by_key:
            errors.append(
                f"duplicate index clip path {key}: {index_by_key[key].get('clip_id')} and {row.get('clip_id')}"
            )
        index_by_key[key] = row
        keys = clip_match_keys(row.get("clip_rel_path", ""))
        keys.extend(clip_match_keys(clip_path))
        index_match_keys[key] = ordered_unique(keys)

    matched_by_key: dict[str, dict[str, object]] = {}
    matched_manifest_paths: set[str] = set()
    for key in sorted(index_by_key):
        row = index_by_key[key]
        candidates = []
        for match_key in index_match_keys.get(key, []):
            candidates.extend(by_key.get(match_key, []))
        unique_candidates: dict[str, dict[str, object]] = {
            str(candidate["manifest_path"]): candidate
            for candidate in candidates
            if candidate.get("category") == "videos"
        }
        candidates = list(unique_candidates.values())
        if not candidates:
            errors.append(
                f"{row.get('clip_id')}: no Stage1 manifest matched clip keys {index_match_keys.get(key, [])}"
            )
            continue
        if len(candidates) > 1:
            paths = ", ".join(str(candidate["manifest_path"]) for candidate in candidates)
            errors.append(f"{row.get('clip_id')}: ambiguous Stage1 manifests for {key}: {paths}")
            continue
        info = candidates[0]
        matched_by_key[key] = info
        matched_manifest_paths.add(str(info["manifest_path"]))
        if str(row.get("width", "")) != str(info.get("width", "")) or str(row.get("height", "")) != str(
            info.get("height", "")
        ):
            errors.append(
                f"{row.get('clip_id')}: index width/height {row.get('width')}x{row.get('height')} "
                f"!= Stage1 manifest {info.get('width')}x{info.get('height')}"
            )

    extra_manifests = [
        info
        for info in manifests
        if info.get("category") == "videos" and str(info["manifest_path"]) not in matched_manifest_paths
    ]
    for info in extra_manifests[:30]:
        errors.append(
            f"Stage1 manifest has no index.csv row: {info['rel_key']} ({info['manifest_path']})"
        )
    if len(extra_manifests) > 30:
        errors.append(f"... and {len(extra_manifests) - 30} more Stage1 manifests without index rows")

    if errors:
        raise RuntimeError("Stage1 manifest matching failed:\n" + "\n".join(errors[:80]))
    skipped_fake_manifests = sum(1 for info in manifests if info.get("category") == "fake_interaction")
    log(f"Stage1 fake manifests skipped: {skipped_fake_manifests}")
    return matched_by_key


def parse_int(value: str, field: str) -> int:
    try:
        return int(float(value))
    except Exception as exc:
        raise ValueError(f"Bad integer field {field}={value!r}") from exc


def parse_float(value: str, field: str) -> float:
    try:
        return float(value)
    except Exception as exc:
        raise ValueError(f"Bad float field {field}={value!r}") from exc


def load_index_rows() -> list[dict[str, str]]:
    if not INDEX_CSV.exists():
        raise FileNotFoundError(f"index.csv not found: {INDEX_CSV}")
    fieldnames, rows = read_csv_rows(INDEX_CSV)
    required = {
        "clip_id",
        "source_video_id",
        "source_video_rel_path",
        "source_video_path",
        "clip_root",
        "clip_rel_path",
        "annotation_rel_path",
        "csv_root",
        "csv_rel_path",
        "source_start_frame",
        "source_end_frame",
        "clip_frame_offset",
        "width",
        "height",
    }
    missing = sorted(required - set(fieldnames))
    if missing:
        raise RuntimeError(f"{INDEX_CSV} missing required columns: {missing}")
    return rows


def load_annotation_rows(index_row: dict[str, str]) -> list[dict[str, str]]:
    annotation_csv = local_annotation_path(index_row)
    if not annotation_csv.exists():
        raise FileNotFoundError(f"{index_row.get('clip_id')}: annotation missing: {annotation_csv}")
    fieldnames, rows = read_csv_rows(annotation_csv)
    missing = sorted(REQUIRED_ANN_COLS - set(fieldnames))
    if missing:
        raise RuntimeError(f"{annotation_csv}: missing required columns: {missing}")
    for line_no, row in enumerate(rows, start=2):
        row["__annotation_csv"] = str(annotation_csv)
        row["__line"] = str(line_no)
    return rows


def build_visualization_index() -> list[dict[str, object]]:
    index_rows = load_index_rows()
    stage1_by_key = build_stage1_manifest_index(index_rows)
    log(f"Index rows total: {len(index_rows)}")
    log(f"Stage1 manifests matched: {len(stage1_by_key)}")
    records: list[dict[str, object]] = []
    errors: list[str] = []
    skipped_unsaved = 0
    skipped_fake = 0
    skipped_unlabeled = 0
    annotation_rows_total = 0
    valence_counts: Counter[str] = Counter()
    for index_row in index_rows:
        clip_id = index_row.get("clip_id", "")
        if not is_true(index_row.get("is_saved", "true")):
            skipped_unsaved += 1
            continue
        category = path_category(index_row.get("clip_root", "")) or path_category(index_row.get("clip_path", ""))
        if category == "fake_interaction":
            skipped_fake += 1
            continue
        if category != "videos":
            errors.append(f"{clip_id}: unsupported clip category: {index_row.get('clip_root')}")
            continue
        try:
            clip_path = local_clip_path(index_row)
            if not clip_path.exists():
                errors.append(f"{clip_id}: clip_path missing: {clip_path}")
                continue
            annotation_rows = load_annotation_rows(index_row)
            annotation_rows_total += len(annotation_rows)
            for row in annotation_rows:
                if row.get("clip_id") != clip_id:
                    errors.append(
                        f"{clip_id}: annotation clip_id mismatch at {row['__annotation_csv']}:{row['__line']}: {row.get('clip_id')}"
                    )
                    continue
                if not row.get("valence", "").strip():
                    skipped_unlabeled += 1
                    continue
                valence = row["valence"].strip()
                if valence not in ALLOWED_VALENCES:
                    errors.append(
                        f"{clip_id}: invalid valence at {row['__annotation_csv']}:{row['__line']}: {valence}"
                    )
                    continue
                if valence == "fake_interaction":
                    skipped_fake += 1
                    continue
                valence_counts[valence] += 1
                stage1_info = stage1_by_key[path_key(clip_path)]
                csv_root = Path(str(stage1_info["csv_root"]))
                tracking_csv = Path(str(stage1_info["tracking_csv"]))
                keypoints_csv = Path(str(stage1_info["keypoints_csv"]))
                if not tracking_csv.exists():
                    errors.append(f"{clip_id}: tracking_boxes.csv missing: {tracking_csv}")
                    continue
                if not keypoints_csv.exists():
                    errors.append(f"{clip_id}: keypoints.csv missing: {keypoints_csv}")
                    continue
                start_frame = parse_int(row["start_frame"], "start_frame")
                end_frame = parse_int(row["end_frame"], "end_frame")
                source_start_frame = parse_int(
                    index_row.get("source_start_frame") or "0", "source_start_frame"
                )
                source_end_frame = parse_int(index_row.get("source_end_frame") or "0", "source_end_frame")
                clip_frame_offset = parse_int(
                    index_row.get("clip_frame_offset") or "0", "clip_frame_offset"
                )
                record = {
                    "clip_id": clip_id,
                    "event_id": row["event_id"],
                    "annotation_csv": row["__annotation_csv"],
                    "annotation_line": row["__line"],
                    "clip_path": str(clip_path),
                    "clip_name": clip_path.name,
                    "source_video_id": index_row["source_video_id"],
                    "source_video_name": basename_any(index_row["source_video_path"]),
                    "tracking_video_name": clip_path.name,
                    "stage1_video": stage1_info["source_path"],
                    "stage1_manifest": stage1_info["manifest_path"],
                    "csv_root": str(csv_root),
                    "tracking_csv": str(tracking_csv),
                    "keypoints_csv": str(keypoints_csv),
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "source_start_frame": source_start_frame,
                    "source_end_frame": source_end_frame,
                    "clip_frame_offset": clip_frame_offset,
                    "stage1_frame_offset": 0,
                    "roi_x": parse_float(row["roi_x"], "roi_x"),
                    "roi_y": parse_float(row["roi_y"], "roi_y"),
                    "roi_w": parse_float(row["roi_w"], "roi_w"),
                    "roi_h": parse_float(row["roi_h"], "roi_h"),
                    "valence": valence,
                    "width": parse_int(row.get("width") or index_row.get("width") or "0", "width"),
                    "height": parse_int(row.get("height") or index_row.get("height") or "0", "height"),
                }
                if record["end_frame"] < record["start_frame"]:
                    errors.append(f"{clip_id}: end_frame < start_frame")
                expected_source_start = start_frame + clip_frame_offset
                expected_source_end = end_frame + clip_frame_offset
                if source_start_frame != expected_source_start or source_end_frame != expected_source_end:
                    errors.append(
                        f"{clip_id}: index source interval {source_start_frame}-{source_end_frame} "
                        f"!= annotation interval plus clip_frame_offset "
                        f"{expected_source_start}-{expected_source_end}"
                    )
                if record["roi_w"] <= 0 or record["roi_h"] <= 0:
                    errors.append(f"{clip_id}: ROI width/height <= 0")
                records.append(record)
        except Exception as exc:
            errors.append(f"{clip_id}: {exc}")
    if errors:
        raise RuntimeError("Index build failed:\n" + "\n".join(errors[:30]))
    log(f"Annotation rows loaded: {annotation_rows_total}")
    log(f"Using labeled annotation rows: {len(records)}")
    log(f"Valence counts: {dict(sorted(valence_counts.items()))}")
    log(f"Skipping fake rows: {skipped_fake}")
    log(f"Skipping unlabeled rows: {skipped_unlabeled}")
    log(f"Skipping unsaved index rows: {skipped_unsaved}")
    records.sort(key=lambda item: str(item["clip_id"]))
    return records


def load_tracking_by_frame(
    tracking_csv: Path,
    video_names: str | list[str] | tuple[str, ...] | set[str],
) -> dict[int, list[dict[str, float | int | str]]]:
    fieldnames, rows = read_csv_rows(tracking_csv)
    missing = sorted(REQUIRED_TRACK_COLS - set(fieldnames))
    if missing:
        raise RuntimeError(f"{tracking_csv} missing required columns: {missing}")
    accepted_names = {video_names} if isinstance(video_names, str) else set(video_names)
    accepted_names = {name for name in accepted_names if name}
    by_frame: dict[int, list[dict[str, float | int | str]]] = defaultdict(list)
    available_video_names: set[str] = set()
    for row in rows:
        row_video = row.get("video", "")
        if row_video:
            available_video_names.add(row_video)
        if accepted_names and row_video and row_video not in accepted_names:
            continue
        frame = parse_int(row["frame"], "frame")
        by_frame[frame].append(
            {
                "frame": frame,
                "track_id": str(row["track_id"]),
                "x": parse_float(row["x"], "x"),
                "y": parse_float(row["y"], "y"),
                "w": parse_float(row["w"], "w"),
                "h": parse_float(row["h"], "h"),
            }
        )
    if accepted_names and not by_frame:
        sample = ", ".join(sorted(available_video_names)[:10])
        raise RuntimeError(
            f"{tracking_csv}: no rows matched video {sorted(accepted_names)}; available video values: {sample}"
        )
    return by_frame


def rect_intersection(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    return iw * ih


def center_distance_norm(a: dict[str, float | int | str], b: dict[str, float | int | str]) -> float:
    ax = float(a["x"]) + float(a["w"]) / 2.0
    ay = float(a["y"]) + float(a["h"]) / 2.0
    bx = float(b["x"]) + float(b["w"]) / 2.0
    by = float(b["y"]) + float(b["h"]) / 2.0
    dist = math.hypot(ax - bx, ay - by)
    adiag = math.hypot(float(a["w"]), float(a["h"]))
    bdiag = math.hypot(float(b["w"]), float(b["h"]))
    denom = max(1.0, (adiag + bdiag) / 2.0)
    return dist / denom


def box_roi_score(
    box: dict[str, float | int | str],
    roi: tuple[float, float, float, float],
) -> tuple[float, dict[str, float]]:
    x = float(box["x"])
    y = float(box["y"])
    w = float(box["w"])
    h = float(box["h"])
    rx, ry, rw, rh = roi
    inter = rect_intersection((x, y, w, h), roi)
    box_area = max(1.0, w * h)
    roi_area = max(1.0, rw * rh)
    overlap_frac = inter / box_area
    roi_coverage = inter / roi_area
    cx = x + w / 2.0
    cy = y + h / 2.0
    rcx = rx + rw / 2.0
    rcy = ry + rh / 2.0
    roi_diag = max(1.0, math.hypot(rw, rh))
    center_dist = math.hypot(cx - rcx, cy - rcy) / roi_diag
    center_inside = 1.0 if rx <= cx <= rx + rw and ry <= cy <= ry + rh else 0.0
    score = (2.0 * overlap_frac) + (3.0 * roi_coverage) + (0.35 * center_inside) - (0.25 * center_dist)
    return score, {
        "intersection": inter,
        "overlap_frac": overlap_frac,
        "roi_coverage": roi_coverage,
        "center_dist": center_dist,
        "center_inside": center_inside,
    }


def select_frame_boxes(
    record: dict[str, object],
    boxes: list[dict[str, float | int | str]],
) -> dict[str, object]:
    roi = (
        float(record["roi_x"]),
        float(record["roi_y"]),
        float(record["roi_w"]),
        float(record["roi_h"]),
    )
    candidates = []
    for index, box in enumerate(boxes):
        score, metrics = box_roi_score(box, roi)
        if metrics["intersection"] <= 0:
            continue
        candidates.append(
            {
                "index": index,
                "box": box,
                "score": score,
                "metrics": metrics,
            }
        )

    if not candidates:
        return {
            "indices": [],
            "track_ids": [],
            "method": "no_roi_intersecting_bbox",
            "candidate_count": 0,
        }
    if len(candidates) == 1:
        item = candidates[0]
        return {
            "indices": [item["index"]],
            "track_ids": [str(item["box"]["track_id"])],
            "method": "single_roi_bbox",
            "candidate_count": 1,
        }

    best_pair = None
    for left, right in combinations(candidates, 2):
        dist = center_distance_norm(left["box"], right["box"])
        score = float(left["score"]) + float(right["score"]) + (1.75 / (1.0 + dist))
        pair = (score, -dist, left, right)
        if best_pair is None or pair > best_pair:
            best_pair = pair

    assert best_pair is not None
    _, neg_dist, left, right = best_pair
    selected = sorted([left, right], key=lambda item: item["index"])
    return {
        "indices": [item["index"] for item in selected],
        "track_ids": [str(item["box"]["track_id"]) for item in selected],
        "method": "framewise_roi_pair_score",
        "candidate_count": len(candidates),
        "mean_norm_distance": -neg_dist,
    }


def track_sort_key(value: str) -> tuple[int, str]:
    return (int(value), "") if value.isdigit() else (10**12, value)


def scale_box(
    box: dict[str, float | int | str],
    scale_x: float,
    scale_y: float,
) -> tuple[int, int, int, int]:
    x1 = int(round(float(box["x"]) * scale_x))
    y1 = int(round(float(box["y"]) * scale_y))
    x2 = int(round((float(box["x"]) + float(box["w"])) * scale_x))
    y2 = int(round((float(box["y"]) + float(box["h"])) * scale_y))
    return x1, y1, x2, y2


def draw_dashed_line(
    frame,
    p1: tuple[int, int],
    p2: tuple[int, int],
    color: tuple[int, int, int],
    thickness: int = 2,
    dash: int = 36,
    gap: int = 24,
) -> None:
    x1, y1 = p1
    x2, y2 = p2
    length = math.hypot(x2 - x1, y2 - y1)
    if length <= 0:
        return
    dx = (x2 - x1) / length
    dy = (y2 - y1) / length
    step = dash + gap
    pos = 0.0
    while pos < length:
        end = min(pos + dash, length)
        start_pt = (int(round(x1 + dx * pos)), int(round(y1 + dy * pos)))
        end_pt = (int(round(x1 + dx * end)), int(round(y1 + dy * end)))
        cv2.line(frame, start_pt, end_pt, color, thickness, cv2.LINE_AA)
        pos += step


def draw_dashed_rectangle(
    frame,
    top_left: tuple[int, int],
    bottom_right: tuple[int, int],
    color: tuple[int, int, int],
    thickness: int = 2,
) -> None:
    x1, y1 = top_left
    x2, y2 = bottom_right
    draw_dashed_line(frame, (x1, y1), (x2, y1), color, thickness)
    draw_dashed_line(frame, (x2, y1), (x2, y2), color, thickness)
    draw_dashed_line(frame, (x2, y2), (x1, y2), color, thickness)
    draw_dashed_line(frame, (x1, y2), (x1, y1), color, thickness)


def draw_label(frame, text: str, x: int, y: int, color: tuple[int, int, int]) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.9
    thickness = 2
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    y = max(th + baseline + 4, y)
    cv2.rectangle(frame, (x, y - th - baseline - 8), (x + tw + 12, y + 4), color, -1)
    cv2.putText(frame, text, (x + 6, y - baseline - 2), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def draw_visuals(
    frame,
    frame_index: int,
    record: dict[str, object],
    boxes: list[dict[str, float | int | str]],
    scale_x: float,
    scale_y: float,
) -> dict[str, object]:
    start = int(record["start_frame"])
    end = int(record["end_frame"])
    if not (start <= frame_index <= end):
        return {
            "indices": [],
            "track_ids": [],
            "method": "outside_event_interval",
            "candidate_count": 0,
        }

    selection = select_frame_boxes(record, boxes)
    selected_indices = set(int(index) for index in selection["indices"])

    for index, box in enumerate(boxes):
        if index in selected_indices:
            continue
        x1, y1, x2, y2 = scale_box(box, scale_x, scale_y)
        draw_dashed_rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)

    roi_x = int(round(float(record["roi_x"]) * scale_x))
    roi_y = int(round(float(record["roi_y"]) * scale_y))
    roi_w = int(round(float(record["roi_w"]) * scale_x))
    roi_h = int(round(float(record["roi_h"]) * scale_y))
    cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (0, 215, 255), 3)
    draw_label(frame, f"{record['clip_id']} {record['valence']}", 20, 46, (40, 40, 40))

    colors = [(40, 90, 255), (40, 200, 80)]
    selected_order = sorted(selected_indices)
    color_by_index = {index: colors[i % len(colors)] for i, index in enumerate(selected_order)}
    for index, box in enumerate(boxes):
        tid = str(box["track_id"])
        if index not in selected_indices:
            continue
        color = color_by_index[index]
        x1, y1, x2, y2 = scale_box(box, scale_x, scale_y)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 5)
        draw_label(frame, f"ID {tid}", x1, y1 - 8, color)
    return selection


def render_record(
    record: dict[str, object],
    output_mp4: Path,
    show_progress: bool = True,
) -> dict[str, object]:
    clip_path = Path(str(record["clip_path"]))
    tracking_csv = Path(str(record["tracking_csv"]))
    by_frame = load_tracking_by_frame(
        tracking_csv,
        str(record["tracking_video_name"]),
    )
    clip_frame_offset = int(record["clip_frame_offset"])
    stage1_frame_offset = int(record.get("stage1_frame_offset", 0))

    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {clip_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 29.97002997
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    source_w = int(record["width"]) if int(record["width"]) > 0 else frame_w
    source_h = int(record["height"]) if int(record["height"]) > 0 else frame_h
    scale_x = frame_w / max(1, source_w)
    scale_y = frame_h / max(1, source_h)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_mp4), fourcc, fps, (frame_w, frame_h))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open VideoWriter: {output_mp4}")

    preview_frame = (int(record["start_frame"]) + int(record["end_frame"])) // 2
    preview_path = OUTPUT_ROOT / "preview_frames" / f"{Path(str(record['clip_name'])).stem}_frame_{preview_frame}.jpg"
    wrote_preview = False
    selected_id_counts: Counter[str] = Counter()
    selected_count_hist: Counter[int] = Counter()
    selection_method_counts: Counter[str] = Counter()
    candidate_count_sum = 0
    selection_frames = 0

    iterable = range(frame_count)
    if show_progress and tqdm is not None:
        iterable = tqdm(iterable, desc=str(record["clip_id"]), mininterval=1.0, unit="frame")

    written = 0
    for frame_index in iterable:
        ok, frame = cap.read()
        if not ok:
            break
        selection = draw_visuals(
            frame,
            frame_index,
            record,
            by_frame.get(frame_index + stage1_frame_offset, []),
            scale_x,
            scale_y,
        )
        if int(record["start_frame"]) <= frame_index <= int(record["end_frame"]):
            track_ids = [str(tid) for tid in selection["track_ids"]]
            selected_count_hist[len(track_ids)] += 1
            selection_method_counts[str(selection["method"])] += 1
            candidate_count_sum += int(selection.get("candidate_count", 0))
            selection_frames += 1
            selected_id_counts.update(track_ids)
        writer.write(frame)
        written += 1
        if not wrote_preview and frame_index >= preview_frame:
            cv2.imwrite(str(preview_path), frame)
            wrote_preview = True

    cap.release()
    writer.release()
    return {
        "clip_id": record["clip_id"],
        "clip_name": record["clip_name"],
        "source_video_id": record["source_video_id"],
        "source_video_name": record["source_video_name"],
        "output_mp4": str(output_mp4),
        "preview_frame": str(preview_path) if wrote_preview else "",
        "selected_track_ids": sorted(selected_id_counts.keys(), key=track_sort_key),
        "selected_track_id_frame_counts": dict(
            sorted(selected_id_counts.items(), key=lambda item: track_sort_key(item[0]))
        ),
        "selection": {
            "method": "framewise_roi_bbox_selection",
            "selected_box_count_hist": dict(sorted(selected_count_hist.items())),
            "selection_method_counts": dict(sorted(selection_method_counts.items())),
            "mean_candidate_count": candidate_count_sum / selection_frames if selection_frames else 0.0,
        },
        "frame_count_input": frame_count,
        "frame_count_written": written,
        "fps": fps,
        "frame_width": frame_w,
        "frame_height": frame_h,
        "scale_x": scale_x,
        "scale_y": scale_y,
        "clip_frame_offset": clip_frame_offset,
        "stage1_frame_offset": stage1_frame_offset,
    }


def render_record_worker(record: dict[str, object], output_mp4: str) -> dict[str, object]:
    return render_record(record, Path(output_mp4), show_progress=False)


def summarize_ids(ids: list[str], limit: int = 12) -> str:
    shown = ids[:limit]
    suffix = "" if len(ids) <= limit else f"...(+{len(ids) - limit})"
    return ",".join(shown) + suffix


def render_records(
    records: list[dict[str, object]],
    workers: int,
) -> list[dict[str, object]]:
    if not records:
        return []
    if workers <= 1:
        results = []
        for record in records:
            log(f"Rendering {record['clip_id']}: {record['clip_name']}")
            output_mp4 = OUTPUT_ROOT / str(record["clip_name"])
            result = render_record(record, output_mp4, show_progress=True)
            results.append(result)
            log(
                "Done {clip_id}: selected IDs {ids}, frames {written}/{total}".format(
                    clip_id=result["clip_id"],
                    ids=summarize_ids(result["selected_track_ids"]),
                    written=result["frame_count_written"],
                    total=result["frame_count_input"],
                )
            )
        return results

    results: list[dict[str, object]] = []
    log(f"Rendering in parallel with {workers} workers")
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {}
        for record in records:
            output_mp4 = OUTPUT_ROOT / str(record["clip_name"])
            future = executor.submit(render_record_worker, record, str(output_mp4))
            futures[future] = record
            log(f"Queued {record['clip_id']}: {record['clip_name']}")

        pending = set(futures)
        total = len(pending)
        while pending:
            done, pending = wait(pending, timeout=10.0, return_when=FIRST_COMPLETED)
            if not done:
                log(f"Still running: {total - len(pending)}/{total} complete, {len(pending)} pending")
                continue
            for future in done:
                record = futures[future]
                result = future.result()
                results.append(result)
                log(
                    "Done {clip_id}: selected IDs {ids}, frames {written}/{total_frames}".format(
                        clip_id=result["clip_id"],
                        ids=summarize_ids(result["selected_track_ids"]),
                        written=result["frame_count_written"],
                        total_frames=result["frame_count_input"],
                    )
                )

    order = {str(record["clip_id"]): i for i, record in enumerate(records)}
    results.sort(key=lambda item: order.get(str(item["clip_id"]), 10**9))
    return results


def write_index_csv(records: list[dict[str, object]], path: Path) -> None:
    fields = [
        "clip_id",
        "event_id",
        "clip_name",
        "source_video_id",
        "source_video_name",
        "annotation_csv",
        "clip_path",
        "tracking_video_name",
        "stage1_video",
        "stage1_manifest",
        "csv_root",
        "tracking_csv",
        "keypoints_csv",
        "start_frame",
        "end_frame",
        "source_start_frame",
        "source_end_frame",
        "clip_frame_offset",
        "stage1_frame_offset",
        "roi_x",
        "roi_y",
        "roi_w",
        "roi_h",
        "valence",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({field: record.get(field, "") for field in fields})


def run_log_test() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    log("log realtime check: line 1")
    time.sleep(1)
    log("log realtime check: line 2")


def main() -> int:
    configure_stdio()
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="Limit rendered rows. 0 means all.")
    parser.add_argument("--index-only", action="store_true", help="Build and validate the index without rendering.")
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Parallel render workers. 0 chooses a conservative automatic value.",
    )
    parser.add_argument("--log-test", action="store_true")
    args = parser.parse_args()

    if args.log_test:
        run_log_test()
        return 0

    safe_reset_output()
    log("BBOX ROI visualization run started")
    records = build_visualization_index()
    write_index_csv(records, OUTPUT_ROOT / "visualization_index.csv")
    if args.index_only:
        log(f"Index-only check complete: {len(records)} rows")
        return 0
    selected = records if args.limit <= 0 else records[: args.limit]
    log(f"Complete visualization index rows: {len(records)}")
    log(f"Rendering rows: {len(selected)}")
    if args.workers <= 0:
        cpu_count = os.cpu_count() or 2
        workers = min(4, len(selected), max(1, cpu_count // 2))
    else:
        workers = min(args.workers, len(selected))
    log(f"Render workers: {workers}")

    render_results = render_records(selected, workers)

    with (OUTPUT_ROOT / "render_results.json").open("w", encoding="utf-8") as f:
        json.dump(render_results, f, ensure_ascii=False, indent=2)
    log("Run complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
