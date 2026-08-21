from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import random
import re
import sys
import time
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable


def _disable_user_site_packages() -> None:
    os.environ.setdefault("PYTHONNOUSERSITE", "1")
    try:
        import site

        candidates = set()
        user_site = getattr(site, "USER_SITE", None)
        if user_site:
            candidates.add(str(Path(user_site).resolve()))
        try:
            candidates.add(str(Path(site.getusersitepackages()).resolve()))
        except Exception:
            pass

        filtered = []
        for entry in sys.path:
            if not entry:
                filtered.append(entry)
                continue
            try:
                resolved = str(Path(entry).resolve())
            except Exception:
                resolved = entry
            if resolved in candidates:
                continue
            if f"{os.sep}.local{os.sep}lib{os.sep}python" in resolved:
                continue
            filtered.append(entry)
        sys.path[:] = filtered
    except Exception:
        pass


_disable_user_site_packages()

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from stage2_pair_features import (
    FEATURE_SCHEMA_VERSION,
    encode_pair_frame,
    expected_in_features as valence_expected_in_features,
    feature_names as valence_feature_names,
    swap_pair_sequence as swap_valence_pair_sequence,
)
from stage2_dynamic_threshold import (
    dynamic_valence_conf_threshold,
    learn_dynamic_valence_threshold,
    predict_with_dynamic_threshold,
)
from stage2_pti_head import PTIHead, PTI_FEATURE_SCHEMA_VERSION


try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


BASE = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = Path(r"D:\DairyCowSNA\output_4_categories")
DEFAULT_V0520_ROOT = Path(r"F:\S1_V0520\output_shards\v0520_4")
DEFAULT_ANNOTATION_ROOT = Path(r"D:\DairyCowSNA\Interaction_Annotator\output\annotations")
DEFAULT_SELECTED_ID_ROOT = Path(r"D:\DairyCowSNA\BBOX_VIS\output_selected_IDs")
DEFAULT_BBOX_GOOD_ROOT = Path(r"D:\DairyCowSNA\BBOX_VIS\GOOD")
DEFAULT_BBOX_GOODTEST_ROOT = Path(r"D:\DairyCowSNA\BBOX_VIS\GOODTEST")
DEFAULT_BBOX_BAD_ROOT = Path(r"D:\DairyCowSNA\BBOX_VIS\BAD")
DEFAULT_STAGE2_INDEX_CSV = BASE / "stage2_index" / "stage2_master_index.csv"
DEFAULT_TEMPLATE_CKPT = BASE / "models_new" / "stage2_valence_transformer_best.pt"
DEFAULT_LOG_DIR = BASE / "run_logs"

CASCADE_TASK = "two_stage_cascade"
GATE_TASK = "interaction_gate_2class"
VALENCE_TASK = "valence_2class"
UNIFIED_TASK = "interaction_3class"

NO_INTERACTION = "no_interaction"
INTERACTION = "interaction"
FRIENDLY = "friendly"
UNFRIENDLY = "unfriendly"
CLASS_TO_VALENCE = {
    "Licking": FRIENDLY,
    "Grooming": FRIENDLY,
    "Displacement": UNFRIENDLY,
    "Headbutting": UNFRIENDLY,
}
VALID_ANNOTATION_VALENCES = {FRIENDLY, UNFRIENDLY}
LABEL_TO_ID = {NO_INTERACTION: 0, FRIENDLY: 1, UNFRIENDLY: 2}
ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}
CASCADE_LABELS = [NO_INTERACTION, FRIENDLY, UNFRIENDLY]
BBOX_VIS_ROOT_TAGS = {"v0520_good", "v0520_goodtest"}

STAGE2_INDEX_REQUIRED_COLUMNS = {
    "index_row_id",
    "source_id",
    "source_name",
    "split",
    "label",
    "bbox_subset",
    "fake_group",
    "clip_name",
    "stage1_folder",
    "stage1_manifest",
    "tracking_csv",
    "keypoints_csv",
    "selected_id_csv",
    "clip_path",
}

SOURCE_REPORTS = {
    "old_20250530_v4c": {
        "split_basis": "stage2_index/stage2_master_index.csv + /mnt/data4t/hyw/20250530/BBOX_VIS-0529-V4C/output/GOOD/GOODTEST",
        "stage1": "/mnt/data4t/hyw/20250530/V4C_S1_S2",
        "bbox": "/mnt/data4t/hyw/20250530/BBOX_VIS-0529-V4C",
        "video": "/mnt/data4t/hyw/20250530/Interaction_Annotator_59/output/videos",
    },
    "old_20250530_sibi": {
        "split_basis": "stage2_index/stage2_master_index.csv + /mnt/data4t/hyw/20250530/BBOX_VIS-0528-SIBI/output/GOOD/GOODTEST",
        "stage1": "/mnt/data4t/hyw/20250530/VSIBI_S1",
        "bbox": "/mnt/data4t/hyw/20250530/BBOX_VIS-0528-SIBI",
        "video": "/mnt/data4t/hyw/20250530/Interaction_Annotator_70/output/videos",
    },
    "old_20250530_0521": {
        "split_basis": "stage2_index/stage2_master_index.csv + /mnt/data4t/hyw/20250530/BBOX_VIS-0521/GOOD/GOODTEST + /mnt/data4t/hyw/20250530/BBOX_FAKE/output/GOOD1",
        "stage1": "/mnt/data4t/hyw/20250530/S1_V0520/output_shards",
        "bbox": "/mnt/data4t/hyw/20250530/BBOX_VIS-0521 + /mnt/data4t/hyw/20250530/BBOX_FAKE/output/GOOD1",
        "video": "/mnt/data4t/hyw/20250530/Interaction_Annotator-0521/output/videos + output/fake_interaction",
    },
    "old_20250530_0527": {
        "split_basis": "stage2_index/stage2_master_index.csv + /mnt/data4t/hyw/20250530/BBOX_VIS-0527-V2/output/GOOD/GOODTEST + /mnt/data4t/hyw/20250530/BBOX_FAKE/output/GOOD2",
        "stage1": "/mnt/data4t/hyw/20250530/V0521_22_seg_S1",
        "bbox": "/mnt/data4t/hyw/20250530/BBOX_VIS-0527-V2 + /mnt/data4t/hyw/20250530/BBOX_FAKE/output/GOOD2",
        "video": "/mnt/data4t/hyw/20250530/Interaction_Annotator-0526/output/videos + output/fake_interaction",
    },
    "new_20260604_0527": {
        "split_basis": "stage2_index/stage2_master_index.csv + /mnt/data4t/hyw/20260604/BBOX_VIS-V0527/GOOD/GOODTEST + /mnt/data4t/hyw/20260604/BBOX_FAKE_0604/output/GOOD1",
        "stage1": "/mnt/data4t/hyw/20260604/V0527_SEG_S1",
        "bbox": "/mnt/data4t/hyw/20260604/BBOX_VIS-V0527 + /mnt/data4t/hyw/20260604/BBOX_FAKE_0604/output/GOOD1",
        "video": "/mnt/data4t/hyw/20260604/Interaction_Annotator-V0527/output/videos + output/fake_interaction",
    },
    "new_20260604_0531": {
        "split_basis": "stage2_index/stage2_master_index.csv + /mnt/data4t/hyw/20260604/BBOX_VIS-V0531/GOOD/GOODTEST + /mnt/data4t/hyw/20260604/BBOX_FAKE_0604/output/GOOD2",
        "stage1": "/mnt/data4t/hyw/20260604/V0531_SEG_S1_FR_UF + /mnt/data4t/hyw/20260604/V0531_SEG_S1_FI",
        "bbox": "/mnt/data4t/hyw/20260604/BBOX_VIS-V0531 + /mnt/data4t/hyw/20260604/BBOX_FAKE_0604/output/GOOD2",
        "video": "/mnt/data4t/hyw/20260604/Interaction_Annotator-V0531/output/videos + output/fake_interaction",
    },
    "new_20260604_0604": {
        "split_basis": "stage2_index/stage2_master_index.csv + /mnt/data4t/hyw/20260604/BBOX_VIS-V0604/GOOD/GOODTEST + /mnt/data4t/hyw/20260604/BBOX_FAKE_0604/output/GOOD3",
        "stage1": "/mnt/data4t/hyw/20260604/V0604_SEG_S1_TI + /mnt/data4t/hyw/20260604/V0604_SEG_S1_FI",
        "bbox": "/mnt/data4t/hyw/20260604/BBOX_VIS-V0604 + /mnt/data4t/hyw/20260604/BBOX_FAKE_0604/output/GOOD3",
        "video": "/mnt/data4t/hyw/20260604/Interaction_Annotator-V0601_03_AND_05R/output/videos + output/fake_interaction",
    },
}


class DataReferenceError(RuntimeError):
    """Raised when required cross-file references are missing or ambiguous."""


def label_maps_for_task(task: str) -> tuple[dict[str, int], dict[int, str]]:
    if task == GATE_TASK:
        mapping = {NO_INTERACTION: 0, INTERACTION: 1}
    elif task == VALENCE_TASK:
        mapping = {FRIENDLY: 0, UNFRIENDLY: 1}
    elif task == UNIFIED_TASK:
        mapping = {NO_INTERACTION: 0, FRIENDLY: 1, UNFRIENDLY: 2}
    else:
        raise ValueError(f"unsupported task: {task}")
    return mapping, {v: k for k, v in mapping.items()}


def set_active_task_labels(task: str) -> None:
    global LABEL_TO_ID, ID_TO_LABEL
    LABEL_TO_ID, ID_TO_LABEL = label_maps_for_task(task)


def output_path_for_task(task: str) -> Path:
    if task == GATE_TASK:
        return BASE / "models_new" / "stage2_interaction_gate_best.pt"
    if task == VALENCE_TASK:
        return BASE / "models_new" / "stage2_valence_transformer_best.pt"
    if task != UNIFIED_TASK:
        raise ValueError(f"unsupported single-model task: {task}")
    return BASE / "models_new" / "stage2_3class_transformer_best.pt"


def split_path_for_task(output_dir: Path, task: str) -> Path:
    if task == GATE_TASK:
        return output_dir / "stage2_interaction_gate_split.csv"
    if task == VALENCE_TASK:
        return output_dir / "stage2_valence_split.csv"
    if task != UNIFIED_TASK:
        raise ValueError(f"unsupported single-model task: {task}")
    return output_dir / "stage2_3class_split.csv"


def split_rule_for_task(task: str) -> str:
    if task == VALENCE_TASK:
        return (
            "authoritative split=stage2_index/stage2_master_index.csv; "
            "valence task uses friendly/unfriendly rows only; no legacy output_4_categories mixing"
        )
    if task in {GATE_TASK, UNIFIED_TASK}:
        return (
            "authoritative split=stage2_index/stage2_master_index.csv; "
            "gate/unified tasks use friendly+unfriendly as interaction and fake groups in the master index as no_interaction; "
            "no legacy output_4_categories mixing"
        )
    raise ValueError(f"unsupported task: {task}")

CONF_MASK_THR = 0.20
INTERACT_DIAG_SIM_RATIO = 0.75
PROX_HARD_CAP = 0.55
Q_PROX = 0.15
Q_WINDOWS_SEC = (1, 2, 4)
GATE_STABLE_M = 12
GATE_STABLE_K = 8
VALENCE_MIN_CONF = 0.60


class TeeLogger:
    def __init__(self, path: Path | None):
        self.path = path
        self.fh = None
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            self.fh = path.open("a", encoding="utf-8", buffering=1)

    def close(self) -> None:
        if self.fh is not None:
            self.fh.flush()
            self.fh.close()
            self.fh = None

    def log(self, message: str) -> None:
        print(message, flush=True)
        if self.fh is not None:
            self.fh.write(message + "\n")
            self.fh.flush()


LOGGER = TeeLogger(None)


def log(message: str) -> None:
    LOGGER.log(message)


def parse_seconds_list(raw: str) -> list[float | str]:
    out: list[float | str] = []
    for part in raw.split(","):
        value = part.strip()
        if not value:
            continue
        if value.lower() == "full":
            out.append("full")
        else:
            out.append(float(value))
    if not out:
        raise ValueError("empty crop seconds list")
    return out


def folder_sort_key(path: Path) -> tuple[int, str]:
    name = path.name
    if name.lower().startswith("v_"):
        try:
            return int(name.split("_", 1)[1]), name
        except Exception:
            pass
    return 10**9, name


def load_json(path: Path) -> dict:
    return normalize_legacy_paths(json.loads(path.read_text(encoding="utf-8")))


def normalize_legacy_path_string(value: str) -> str:
    s = str(value)
    replacements = [
        (
            r"G:\DROPBOX\Isolated_Interaction_Clips\Interaction_Type",
            r"F:\FULLDATA\Isolated_Interaction_Clips\Interaction_Type",
        ),
        (
            "G:/DROPBOX/Isolated_Interaction_Clips/Interaction_Type",
            "F:/FULLDATA/Isolated_Interaction_Clips/Interaction_Type",
        ),
        (r"E:\DROPBOX_LRV\Dairy Farm Videos", r"F:\FULLDATA\Dairy Farm Videos"),
        (r"E:\DROPBOX_LRV\Dairy Farm Video", r"F:\FULLDATA\Dairy Farm Videos"),
        ("E:/DROPBOX_LRV/Dairy Farm Videos", "F:/FULLDATA/Dairy Farm Videos"),
        ("E:/DROPBOX_LRV/Dairy Farm Video", "F:/FULLDATA/Dairy Farm Videos"),
    ]
    for old, new in replacements:
        if s.lower().startswith(old.lower()):
            return new + s[len(old) :]
    return s


def normalize_legacy_paths(obj):
    if isinstance(obj, dict):
        return {k: normalize_legacy_paths(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [normalize_legacy_paths(v) for v in obj]
    if isinstance(obj, str):
        return normalize_legacy_path_string(obj)
    return obj


def _path_parts(value: str) -> list[str]:
    return [p for p in str(value or "").replace("/", "\\").split("\\") if p]


def clip_id_from_stem(stem: str) -> str:
    m = re.match(r"(C\d+)", str(stem or ""), re.IGNORECASE)
    return m.group(1).upper() if m else ""


def clip_key_from_path(value: str) -> tuple[str, str]:
    parts = _path_parts(value)
    if len(parts) >= 2:
        return parts[-2].lower(), Path(parts[-1]).stem.lower()
    if parts:
        return "", Path(parts[-1]).stem.lower()
    return "", ""


def original_class_from_manifest(manifest: dict) -> str:
    known = {k.lower(): k for k in CLASS_TO_VALENCE}
    known["athletic"] = "Athletic"
    vm = manifest.get("video_manifest", {})
    for value in (vm.get("relative_path", ""), vm.get("source_path", "")):
        for part in _path_parts(value):
            cls = known.get(part.lower())
            if cls:
                return cls
    return ""


def parse_bool_false(value) -> bool:
    return str(value or "").strip().lower() in {"", "0", "false", "no", "n"}


def parse_int_field(value, default: int) -> int:
    try:
        return int(float(str(value).strip()))
    except Exception:
        return int(default)


def parse_float_field(value, default: float = 0.0) -> float:
    try:
        out = float(str(value).strip())
    except Exception:
        return float(default)
    return out if math.isfinite(out) else float(default)


def load_annotation_index(annotation_root: Path) -> tuple[dict[tuple[str, str], list[AnnotationEvent]], dict[str, list[AnnotationEvent]], Counter]:
    by_key: dict[tuple[str, str], list[AnnotationEvent]] = defaultdict(list)
    by_clip_id: dict[str, list[AnnotationEvent]] = defaultdict(list)
    stats: Counter = Counter()
    root = Path(annotation_root)
    if not root.is_dir():
        raise FileNotFoundError(f"annotation root not found: {root}")

    for csv_path in sorted(root.rglob("*.csv")):
        stats["files"] += 1
        with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                stats["rows"] += 1
                if not parse_bool_false(row.get("exclude")):
                    stats["excluded_rows"] += 1
                    continue
                valence = str(row.get("valence", "")).strip().lower()
                if valence not in VALID_ANNOTATION_VALENCES:
                    stats[f"ignored_valence:{valence or 'blank'}"] += 1
                    continue
                key = clip_key_from_path(row.get("clip_rel_path") or row.get("clip_path") or csv_path.with_suffix(".mp4").name)
                clip_id = str(row.get("clip_id") or clip_id_from_stem(csv_path.stem)).upper()
                roi = (
                    parse_float_field(row.get("roi_x")),
                    parse_float_field(row.get("roi_y")),
                    parse_float_field(row.get("roi_w")),
                    parse_float_field(row.get("roi_h")),
                )
                event = AnnotationEvent(
                    clip_key=key,
                    clip_id=clip_id,
                    valence=valence,
                    start_frame=parse_int_field(row.get("start_frame"), 0),
                    end_frame=parse_int_field(row.get("end_frame"), 10**12),
                    roi=roi,
                    source_csv=str(csv_path),
                )
                by_key[key].append(event)
                if clip_id:
                    by_clip_id[clip_id].append(event)
                stats[f"valence:{valence}"] += 1

    stats["unique_clip_keys"] = len(by_key)
    stats["unique_clip_ids"] = len(by_clip_id)
    return dict(by_key), dict(by_clip_id), stats


def annotation_events_for_manifest(
    manifest: dict,
    by_key: dict[tuple[str, str], list[AnnotationEvent]],
    by_clip_id: dict[str, list[AnnotationEvent]],
) -> tuple[list[AnnotationEvent], str]:
    vm = manifest.get("video_manifest", {})
    keys = [clip_key_from_path(vm.get("relative_path", "")), clip_key_from_path(vm.get("source_path", ""))]
    for key in keys:
        if key[1] and key in by_key:
            return by_key[key], "annotation_full_path_suffix"

    clip_ids = sorted({clip_id_from_stem(stem) for _parent, stem in keys if clip_id_from_stem(stem)})
    for cid in clip_ids:
        events = by_clip_id.get(cid, [])
        if not events:
            continue
        unique_keys = sorted({ev.clip_key for ev in events})
        if len(unique_keys) == 1:
            return events, "annotation_clip_id_unique"
        raise DataReferenceError(
            f"ambiguous annotation clip_id {cid}: matched multiple source keys {unique_keys[:8]}"
        )

    raise DataReferenceError(f"annotation missing for manifest keys: {keys}")


def annotation_valence(events: list[AnnotationEvent]) -> str:
    vals = sorted({ev.valence for ev in events if ev.valence in VALID_ANNOTATION_VALENCES})
    if len(vals) == 1:
        return vals[0]
    if len(vals) > 1:
        raise DataReferenceError(f"conflicting annotation valences: {vals}")
    return ""


def manifest_contains_fake_interaction(manifest_path: Path, manifest: dict) -> bool:
    if "fake_interaction" in str(manifest_path).lower():
        return True
    try:
        return "fake_interaction" in json.dumps(manifest, ensure_ascii=False).lower()
    except Exception:
        return False


@dataclass
class PairChoice:
    tid_a: int
    tid_b: int
    frames: list[int]
    stable_count: int
    raw_count: int
    common_count: int
    median_dist: float


@dataclass
class PairFrameRecord:
    frame: int
    tid_a: int | None
    tid_b: int | None
    box_a: tuple[float, float, float, float]
    box_b: tuple[float, float, float, float]
    source: str
    kpts_a: np.ndarray | None = None
    kpts_b: np.ndarray | None = None


@dataclass
class Stage2Sample:
    root_tag: str
    folder: str
    manifest_path: str
    source_video: str
    original_class: str
    valence: str
    label_source: str
    label_id: int
    pair: tuple[int, int]
    fps: float
    frames: list[int]
    x: np.ndarray
    split: str = ""
    source_name: str = ""
    index_row_id: str = ""
    bbox_subset: str = ""
    raw_label: str = ""
    clip_name: str = ""
    clip_path: str = ""
    source_set: str = ""
    id_pair_vote_ratio: float = 0.0
    id_pair_vote_count: int = 0
    id_pair_total_rows: int = 0
    real_frames: int = 0
    alias_frames: int = 0
    interpolated_frames: int = 0
    frozen_frames: int = 0
    dropped_frames: int = 0
    fill_ratio: float = 0.0
    pti_keypoints: np.ndarray | None = None
    pti_person_mask: np.ndarray | None = None


@dataclass
class SplitUnit:
    key: str
    samples: list[Stage2Sample]
    is_bbox_vis: bool
    is_multi_segment: bool


@dataclass
class AnnotationEvent:
    clip_key: tuple[str, str]
    clip_id: str
    valence: str
    start_frame: int
    end_frame: int
    roi: tuple[float, float, float, float]
    source_csv: str


def box_center_diag(box: tuple[float, float, float, float]) -> tuple[float, float, float]:
    x, y, w, h = box
    return x + 0.5 * w, y + 0.5 * h, math.hypot(w, h)


def normalized_center_distance(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> tuple[float, float]:
    ax, ay, ad = box_center_diag(a)
    bx, by, bd = box_center_diag(b)
    denom = max(1e-6, 0.5 * (ad + bd))
    dist = math.hypot(ax - bx, ay - by) / denom
    sim = min(ad, bd) / max(ad, bd) if max(ad, bd) > 0 else 0.0
    return float(dist), float(sim)


def boxes_overlap(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    return (min(ax2, bx2) - max(ax1, bx1) > 0.0) and (min(ay2, by2) - max(ay1, by1) > 0.0)


def center_inside_roi(box: tuple[float, float, float, float], roi: tuple[float, float, float, float]) -> bool:
    x, y, w, h = box
    rx, ry, rw, rh = roi
    if rw <= 0 or rh <= 0:
        return False
    cx, cy = x + 0.5 * w, y + 0.5 * h
    return (rx <= cx <= rx + rw) and (ry <= cy <= ry + rh)


def pair_frame_in_annotation_roi(
    frame: int,
    box_a: tuple[float, float, float, float],
    box_b: tuple[float, float, float, float],
    events: list[AnnotationEvent] | None,
) -> bool:
    if not events:
        return True
    for event in events:
        if event.start_frame <= frame <= event.end_frame:
            if center_inside_roi(box_a, event.roi) and center_inside_roi(box_b, event.roi):
                return True
    return False


def load_boxes(path: Path) -> dict[int, dict[int, tuple[float, float, float, float]]]:
    df = pd.read_csv(path)
    required = {"frame", "track_id", "x", "y", "w", "h"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    boxes: dict[int, dict[int, tuple[float, float, float, float]]] = defaultdict(dict)
    for row in df.itertuples(index=False):
        frame = int(getattr(row, "frame"))
        tid = int(getattr(row, "track_id"))
        boxes[tid][frame] = (
            float(getattr(row, "x")),
            float(getattr(row, "y")),
            float(getattr(row, "w")),
            float(getattr(row, "h")),
        )
    return dict(boxes)


def load_keypoints(path: Path, num_kpts: int) -> dict[tuple[int, int], np.ndarray]:
    df = pd.read_csv(path)
    required = {"frame", "track_id"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    cols: list[str] = []
    for i in range(num_kpts):
        cols.extend([f"kpt_{i}_x", f"kpt_{i}_y", f"kpt_{i}_conf"])
    missing_kpts = [c for c in cols if c not in df.columns]
    if missing_kpts:
        raise ValueError(f"{path} missing keypoint columns, first missing: {missing_kpts[:6]}")

    out: dict[tuple[int, int], np.ndarray] = {}
    values = df[cols].to_numpy(dtype=np.float32, copy=True).reshape((-1, num_kpts, 3))
    frames = df["frame"].to_numpy(dtype=np.int64)
    tids = df["track_id"].to_numpy(dtype=np.int64)
    for i in range(len(df)):
        out[(int(frames[i]), int(tids[i]))] = values[i]
    return out


def video_name_from_manifest(manifest: dict) -> str:
    vm = manifest.get("video_manifest", {})
    for value in (vm.get("relative_path", ""), vm.get("source_path", "")):
        parts = _path_parts(value)
        if parts:
            name = parts[-1]
            if name.lower().endswith(".mp4"):
                return name
    return ""


def load_mp4_name_set(root: Path) -> set[str]:
    p = Path(root)
    if not p.is_dir():
        raise FileNotFoundError(f"video list root not found: {p}")
    return {x.name for x in p.glob("*.mp4") if x.is_file()}


def load_selected_id_rows(path: Path) -> tuple[dict[int, tuple[int, int]], Counter]:
    if not path.is_file():
        raise FileNotFoundError(f"selected ID CSV not found: {path}")
    frame_to_pair: dict[int, tuple[int, int]] = {}
    counts: Counter = Counter()
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw = str(row.get("track_ids", "")).strip()
            if not raw or ";" not in raw:
                continue
            try:
                frame = parse_int_field(row.get("frame"), -1)
                parts = [p.strip() for p in raw.split(";") if p.strip()]
                if frame < 0 or len(parts) != 2:
                    continue
                pair = (int(parts[0]), int(parts[1]))
            except Exception:
                continue
            frame_to_pair[frame] = pair
            counts[pair] += 1
    if not counts:
        raise ValueError(f"no usable selected ID rows: {path}")
    return frame_to_pair, counts


def _box_center(box: tuple[float, float, float, float]) -> tuple[float, float]:
    x, y, w, h = box
    return x + 0.5 * w, y + 0.5 * h


def _interp_box(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
    alpha: float,
) -> tuple[float, float, float, float]:
    alpha = float(np.clip(alpha, 0.0, 1.0))
    return tuple(float((1.0 - alpha) * x + alpha * y) for x, y in zip(a, b))


def _copy_kpts(kpts: np.ndarray | None) -> np.ndarray | None:
    if kpts is None:
        return None
    return np.array(kpts, dtype=np.float32, copy=True)


def _interp_kpts(a: np.ndarray | None, b: np.ndarray | None, alpha: float) -> np.ndarray | None:
    if a is None or b is None:
        return None
    if a.shape != b.shape:
        return None
    alpha = float(np.clip(alpha, 0.0, 1.0))
    return np.nan_to_num(((1.0 - alpha) * a + alpha * b).astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def build_sequence_from_records(
    records: list[PairFrameRecord],
    kpts_by_key: dict[tuple[int, int], np.ndarray],
    num_kpts: int,
) -> np.ndarray:
    fdim = valence_expected_in_features(num_kpts)
    x = np.zeros((len(records), fdim), dtype=np.float32)
    for t, record in enumerate(records):
        xa, ya, wa, ha = record.box_a
        xb, yb, wb, hb = record.box_b
        center_a = (xa + 0.5 * wa, ya + 0.5 * ha)
        center_b = (xb + 0.5 * wb, yb + 0.5 * hb)
        scale_a = math.sqrt(max(1.0, wa * ha))
        scale_b = math.sqrt(max(1.0, wb * hb))
        k_a = record.kpts_a if record.kpts_a is not None else (
            kpts_by_key.get((record.frame, int(record.tid_a))) if record.tid_a is not None else None
        )
        k_b = record.kpts_b if record.kpts_b is not None else (
            kpts_by_key.get((record.frame, int(record.tid_b))) if record.tid_b is not None else None
        )
        x[t, :] = encode_pair_frame(
            k_a,
            k_b,
            center_a,
            center_b,
            scale_a,
            scale_b,
            num_kpts,
            conf_thr=CONF_MASK_THR,
        )
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def build_selected_id_records(
    frames: list[int],
    dominant_pair: tuple[int, int],
    frame_to_pair: dict[int, tuple[int, int]],
    boxes_by_tid: dict[int, dict[int, tuple[float, float, float, float]]],
    kpts_by_key: dict[tuple[int, int], np.ndarray],
    fps: float,
    args,
) -> tuple[list[list[PairFrameRecord]], Counter]:
    max_gap_frames = max(1, int(round(float(args.max_id_gap_sec) * fps)))
    stats: Counter = Counter()
    tid_a, tid_b = dominant_pair
    frame_set = set(frames)
    stats["non_dominant_selected_rows"] = sum(1 for frame in frames if frame_to_pair.get(frame) != dominant_pair)

    def real_record(frame: int) -> PairFrameRecord:
        return PairFrameRecord(
            frame,
            tid_a,
            tid_b,
            boxes_by_tid[tid_a][frame],
            boxes_by_tid[tid_b][frame],
            "real",
            _copy_kpts(kpts_by_key.get((frame, tid_a))),
            _copy_kpts(kpts_by_key.get((frame, tid_b))),
        )

    known_frames = [
        frame
        for frame in frames
        if frame in boxes_by_tid.get(tid_a, {}) and frame in boxes_by_tid.get(tid_b, {})
    ]
    stats["real"] = len(known_frames)
    if not known_frames:
        stats["dropped"] = len(frames)
        return [], stats

    first_known = known_frames[0]
    stats["leading_dropped"] = sum(1 for frame in frames if frame < first_known)

    segments: list[list[PairFrameRecord]] = []
    current: list[PairFrameRecord] = [real_record(first_known)]
    for left, right in zip(known_frames, known_frames[1:]):
        gap = right - left
        between = [frame for frame in frames if left < frame < right]
        rec_l = current[-1]
        rec_r = real_record(right)
        if gap <= max_gap_frames:
            for frame in between:
                alpha = (frame - left) / gap
                current.append(
                    PairFrameRecord(
                        frame,
                        None,
                        None,
                        _interp_box(rec_l.box_a, rec_r.box_a, alpha),
                        _interp_box(rec_l.box_b, rec_r.box_b, alpha),
                        "interpolated",
                        _interp_kpts(rec_l.kpts_a, rec_r.kpts_a, alpha),
                        _interp_kpts(rec_l.kpts_b, rec_r.kpts_b, alpha),
                    )
                )
                stats["interpolated"] += 1
            current.append(rec_r)
        else:
            stats["long_gap_splits"] += 1
            stats["dropped_long_gap"] += len(between)
            segments.append(current)
            current = [rec_r]

    last_known = known_frames[-1]
    last_rec = current[-1]
    for frame in [f for f in frames if f > last_known]:
        if frame - last_known <= max_gap_frames:
            current.append(
                PairFrameRecord(
                    frame,
                    None,
                    None,
                    last_rec.box_a,
                    last_rec.box_b,
                    "frozen",
                    _copy_kpts(last_rec.kpts_a),
                    _copy_kpts(last_rec.kpts_b),
                )
            )
            stats["frozen"] += 1
        else:
            stats["dropped_tail"] += 1

    segments.append(current)
    segments = [[rec for rec in segment if rec.frame in frame_set] for segment in segments]
    segments = [segment for segment in segments if segment]
    kept = sum(len(segment) for segment in segments)
    stats["segments"] = len(segments)
    stats["dropped"] = max(0, len(frames) - kept)
    return segments, stats


def gate_frames_for_pair(
    frames: list[int],
    boxes_a: dict[int, tuple[float, float, float, float]],
    boxes_b: dict[int, tuple[float, float, float, float]],
    fps: float,
) -> PairChoice:
    raw_frames: list[int] = []
    stable_frames: list[int] = []
    dists: list[float] = []
    prox_hist = {w: deque(maxlen=max(1, int(round(w * fps)))) for w in Q_WINDOWS_SEC}
    gate_hist: deque[int] = deque(maxlen=GATE_STABLE_M)

    for frame in frames:
        a = boxes_a[frame]
        b = boxes_b[frame]
        dist, sim = normalized_center_distance(a, b)
        overlap = boxes_overlap(a, b)
        diag_ok = sim >= INTERACT_DIAG_SIM_RATIO
        quantile_ok = False

        if overlap and diag_ok:
            for w, hist in prox_hist.items():
                hist.append(dist)
                min_hist = max(3, int(round(0.5 * w * fps)))
                if len(hist) >= min_hist:
                    qv = float(np.quantile(np.asarray(hist, dtype=np.float32), Q_PROX))
                    if qv <= PROX_HARD_CAP:
                        quantile_ok = True
            dists.append(dist)
        else:
            for hist in prox_hist.values():
                hist.clear()

        gate_pass = bool(overlap and diag_ok and (dist <= PROX_HARD_CAP or quantile_ok))
        if gate_pass:
            raw_frames.append(frame)
        gate_hist.append(1 if gate_pass else 0)
        if sum(gate_hist) >= GATE_STABLE_K:
            stable_frames.append(frame)

    chosen = stable_frames if stable_frames else raw_frames
    if not chosen:
        chosen = frames
    median_dist = float(np.median(np.asarray(dists, dtype=np.float32))) if dists else float("inf")
    return PairChoice(
        tid_a=-1,
        tid_b=-1,
        frames=chosen,
        stable_count=len(stable_frames),
        raw_count=len(raw_frames),
        common_count=len(frames),
        median_dist=median_dist,
    )


def choose_pairs(
    boxes_by_tid: dict[int, dict[int, tuple[float, float, float, float]]],
    fps: float,
    min_seq_frames: int,
    samples_per_video: int,
    annotation_events: list[AnnotationEvent] | None = None,
) -> list[PairChoice]:
    tids = sorted(boxes_by_tid)
    choices: list[PairChoice] = []
    for i, tid_a in enumerate(tids):
        for tid_b in tids[i + 1 :]:
            common = sorted(set(boxes_by_tid[tid_a]).intersection(boxes_by_tid[tid_b]))
            if annotation_events:
                common_roi = [
                    frame
                    for frame in common
                    if pair_frame_in_annotation_roi(
                        frame,
                        boxes_by_tid[tid_a][frame],
                        boxes_by_tid[tid_b][frame],
                        annotation_events,
                    )
                ]
                if len(common_roi) >= min_seq_frames:
                    common = common_roi
            if len(common) < min_seq_frames:
                continue
            choice = gate_frames_for_pair(common, boxes_by_tid[tid_a], boxes_by_tid[tid_b], fps)
            if len(choice.frames) < min_seq_frames:
                continue
            choice.tid_a = int(tid_a)
            choice.tid_b = int(tid_b)
            choices.append(choice)

    choices.sort(
        key=lambda c: (
            c.stable_count,
            c.raw_count,
            c.common_count,
            -c.median_dist if np.isfinite(c.median_dist) else -1e9,
        ),
        reverse=True,
    )
    return choices[: max(1, samples_per_video)]


def build_pair_sequence(
    frames: Iterable[int],
    tid_a: int,
    tid_b: int,
    boxes_by_tid: dict[int, dict[int, tuple[float, float, float, float]]],
    kpts_by_key: dict[tuple[int, int], np.ndarray],
    num_kpts: int,
) -> np.ndarray:
    frames = list(frames)
    fdim = valence_expected_in_features(num_kpts)
    x = np.zeros((len(frames), fdim), dtype=np.float32)

    for t, frame in enumerate(frames):
        box_a = boxes_by_tid[tid_a][frame]
        box_b = boxes_by_tid[tid_b][frame]
        xa, ya, wa, ha = box_a
        xb, yb, wb, hb = box_b
        center_a = (xa + 0.5 * wa, ya + 0.5 * ha)
        center_b = (xb + 0.5 * wb, yb + 0.5 * hb)
        scale_a = math.sqrt(max(1.0, wa * ha))
        scale_b = math.sqrt(max(1.0, wb * hb))

        k_a = kpts_by_key.get((frame, tid_a))
        k_b = kpts_by_key.get((frame, tid_b))
        x[t, :] = encode_pair_frame(
            k_a,
            k_b,
            center_a,
            center_b,
            scale_a,
            scale_b,
            num_kpts,
            conf_thr=CONF_MASK_THR,
        )

    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def _clean_pti_person(kpts: np.ndarray | None, num_kpts: int) -> tuple[np.ndarray, bool]:
    out = np.zeros((num_kpts, 3), dtype=np.float32)
    if kpts is None:
        return out, False
    try:
        arr = np.asarray(kpts, dtype=np.float32)
    except Exception:
        return out, False
    if arr.ndim != 2 or arr.shape[0] < num_kpts or arr.shape[1] < 2:
        return out, False
    cols = min(3, arr.shape[1])
    out[:, :cols] = arr[:num_kpts, :cols]
    if cols < 3:
        out[:, 2] = 1.0
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    valid = bool(np.isfinite(arr[:num_kpts, :2]).all())
    return out, valid


def build_pti_sequence_from_records(
    records: list[PairFrameRecord],
    kpts_by_key: dict[tuple[int, int], np.ndarray],
    num_kpts: int,
) -> tuple[np.ndarray, np.ndarray]:
    keypoints = np.zeros((len(records), 2, num_kpts, 3), dtype=np.float32)
    person_mask = np.zeros((len(records), 2), dtype=np.bool_)
    for t, record in enumerate(records):
        k_a = record.kpts_a if record.kpts_a is not None else (
            kpts_by_key.get((record.frame, int(record.tid_a))) if record.tid_a is not None else None
        )
        k_b = record.kpts_b if record.kpts_b is not None else (
            kpts_by_key.get((record.frame, int(record.tid_b))) if record.tid_b is not None else None
        )
        keypoints[t, 0], person_mask[t, 0] = _clean_pti_person(k_a, num_kpts)
        keypoints[t, 1], person_mask[t, 1] = _clean_pti_person(k_b, num_kpts)
    return keypoints, person_mask


def build_pti_sequence_from_frames(
    frames: Iterable[int],
    tid_a: int,
    tid_b: int,
    kpts_by_key: dict[tuple[int, int], np.ndarray],
    num_kpts: int,
) -> tuple[np.ndarray, np.ndarray]:
    records = [
        PairFrameRecord(
            frame=int(frame),
            tid_a=int(tid_a),
            tid_b=int(tid_b),
            box_a=(0.0, 0.0, 1.0, 1.0),
            box_b=(0.0, 0.0, 1.0, 1.0),
            source="real",
        )
        for frame in frames
    ]
    return build_pti_sequence_from_records(records, kpts_by_key, num_kpts)


def load_template_metadata(path: Path) -> tuple[list[str], dict]:
    if not path.is_file():
        names = [f"kpt_{i}" for i in range(27)]
        return names, {
            "transformer": {
                "max_frames": 64,
                "d_model": 64,
                "num_heads": 4,
                "temporal_layers": 2,
                "ffn_dim": 128,
                "dropout": 0.1,
                "causal": False,
            },
            "fps": 60.0,
        }
    ckpt = torch.load(str(path), map_location="cpu")
    if str(ckpt.get("model", "")) != "ValenceTransformer":
        raise RuntimeError(f"template checkpoint is not a ValenceTransformer checkpoint: {path}")
    keypoints = list(ckpt.get("keypoints", [])) or [f"kpt_{i}" for i in range(27)]
    transformer = dict(ckpt.get("transformer", {}))
    meta = {
        "transformer": {
            "max_frames": int(transformer.get("max_frames", ckpt.get("max_frames", 64))),
            "d_model": int(transformer.get("d_model", 64)),
            "num_heads": int(transformer.get("num_heads", 4)),
            "temporal_layers": int(transformer.get("temporal_layers", 2)),
            "ffn_dim": int(transformer.get("ffn_dim", 128)),
            "dropout": float(transformer.get("dropout", 0.1)),
            "causal": bool(transformer.get("causal", False)),
        },
        "fps": float(ckpt.get("fps", 60.0) or 60.0),
    }
    return keypoints, meta


def discover_manifest_folders(root: Path) -> list[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"data root not found: {root}")
    folders = [p.parent for p in root.rglob("manifest.json")]
    return sorted(folders, key=lambda p: (str(p.parent).lower(), folder_sort_key(p)))


def label_for_output4_manifest(manifest: dict) -> tuple[str, str, str]:
    original_class = original_class_from_manifest(manifest)
    if original_class.lower() == "athletic":
        return "", original_class, "exclude_athletic"
    valence = CLASS_TO_VALENCE.get(original_class)
    if not valence:
        return "", original_class, f"unmapped_class:{original_class or 'unknown'}"
    return valence, original_class, "relative_path_class"


def label_for_v0520_manifest(
    manifest_path: Path,
    manifest: dict,
    annotation_by_key: dict[tuple[str, str], list[AnnotationEvent]],
    annotation_by_clip_id: dict[str, list[AnnotationEvent]],
) -> tuple[str, str, str, list[AnnotationEvent]]:
    if manifest_contains_fake_interaction(manifest_path, manifest):
        return NO_INTERACTION, "fake_interaction", "fake_interaction_path", []
    events, source = annotation_events_for_manifest(manifest, annotation_by_key, annotation_by_clip_id)
    if not events:
        return "", "", "missing_annotation", []
    valence = annotation_valence(events)
    if not valence:
        return "", "", "missing_annotation_valence", events
    return valence, valence, source, events


def training_label_for_task(raw_label: str, task: str) -> str:
    if task == GATE_TASK:
        if raw_label == NO_INTERACTION:
            return NO_INTERACTION
        if raw_label in {FRIENDLY, UNFRIENDLY}:
            return INTERACTION
        return ""
    if task == VALENCE_TASK:
        return raw_label if raw_label in {FRIENDLY, UNFRIENDLY} else ""
    if task == UNIFIED_TASK:
        return raw_label if raw_label in {NO_INTERACTION, FRIENDLY, UNFRIENDLY} else ""
    raise ValueError(f"unsupported task: {task}")


def index_path(value: str | Path) -> Path:
    raw = str(value or "").strip()
    if not raw:
        raise ValueError("blank path in Stage2 master index")
    return Path(normalize_legacy_path_string(raw)).expanduser()


def required_index_file(description: str, value: str | Path) -> Path:
    path = index_path(value)
    if not path.is_file():
        raise FileNotFoundError(f"{description} not found: {path}")
    return path


def required_index_dir(description: str, value: str | Path) -> Path:
    path = index_path(value)
    if not path.is_dir():
        raise FileNotFoundError(f"{description} not found: {path}")
    return path


def load_stage2_index_rows(index_csv: Path) -> list[dict[str, str]]:
    if not index_csv.is_file():
        raise FileNotFoundError(f"Stage2 master index CSV not found: {index_csv}")
    with index_csv.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = [dict(row) for row in reader]
        columns = set(reader.fieldnames or [])
    missing = STAGE2_INDEX_REQUIRED_COLUMNS.difference(columns)
    if missing:
        raise ValueError(f"{index_csv} missing required columns: {sorted(missing)}")
    if not rows:
        raise ValueError(f"Stage2 master index is empty: {index_csv}")

    ids = [str(row.get("index_row_id") or "").strip() for row in rows]
    blank_ids = sum(1 for value in ids if not value)
    if blank_ids:
        raise ValueError(f"{index_csv} has {blank_ids} row(s) with blank index_row_id")
    duplicate_ids = [key for key, count in Counter(ids).items() if count > 1]
    if duplicate_ids:
        raise ValueError(f"{index_csv} has duplicate index_row_id values: {duplicate_ids[:8]}")

    for row in rows:
        label = str(row.get("label") or "").strip().lower()
        split = str(row.get("split") or "").strip().lower()
        if label not in CASCADE_LABELS:
            raise ValueError(f"{index_csv}: unsupported label={label!r} for {row.get('index_row_id')}")
        if split not in {"train", "val"}:
            raise ValueError(f"{index_csv}: unsupported split={split!r} for {row.get('index_row_id')}")
    return rows


def row_label(row: dict[str, str]) -> str:
    return str(row.get("label") or row.get("raw_valence") or "").strip().lower()


def row_split(row: dict[str, str]) -> str:
    split = str(row.get("split") or "").strip().lower()
    if split not in {"train", "val"}:
        raise ValueError(f"unsupported split={split!r} for {row.get('index_row_id')}")
    return split


def row_source_name(row: dict[str, str]) -> str:
    value = str(row.get("source_name") or "").strip()
    if not value:
        raise ValueError(f"blank source_name for {row.get('index_row_id')}")
    return value


def row_bbox_subset(row: dict[str, str]) -> str:
    value = str(row.get("bbox_subset") or "").strip()
    if not value:
        raise ValueError(f"blank bbox_subset for {row.get('index_row_id')}")
    return value


def source_report_order(rows: list[dict[str, str]]) -> list[str]:
    seen = {row_source_name(row) for row in rows}
    preferred = [name for name in SOURCE_REPORTS if name in seen]
    extras = sorted(seen.difference(preferred))
    return preferred + extras


def count_rows(rows: list[dict[str, str]], source_name: str, split: str, label: str) -> int:
    return sum(
        1
        for row in rows
        if row_source_name(row) == source_name and row_split(row) == split and row_label(row) == label
    )


def log_canonical_split_report(rows: list[dict[str, str]], index_csv: Path) -> None:
    if getattr(log_canonical_split_report, "_printed", False):
        return
    setattr(log_canonical_split_report, "_printed", True)
    log("[split] canonical Stage2 split")
    log(f"[split] master index: {index_csv}")
    for source_name in source_report_order(rows):
        info = SOURCE_REPORTS.get(source_name, {})
        log(f"[split] {source_name}:")
        log(f"[split]   split basis: {info.get('split_basis', str(index_csv))}")
        log(f"[split]   keypoint trajectory source: {info.get('stage1', 'see stage2_master_index.csv')}")
        log(f"[split]   validated bbox source: {info.get('bbox', 'see stage2_master_index.csv')}")
        log(f"[split]   visualization video source: {info.get('video', 'see stage2_master_index.csv')}")
        for split_name in ("train", "val"):
            prefix = "trn" if split_name == "train" else "val=test"
            log(
                f"[split]   {prefix}: "
                f"friendly={count_rows(rows, source_name, split_name, FRIENDLY)} "
                f"unfriendly={count_rows(rows, source_name, split_name, UNFRIENDLY)} "
                f"fake interaction={count_rows(rows, source_name, split_name, NO_INTERACTION)}"
            )
    totals = Counter((row_split(row), row_label(row)) for row in rows)
    log(
        "[split] total trn: "
        f"friendly={totals.get(('train', FRIENDLY), 0)} "
        f"unfriendly={totals.get(('train', UNFRIENDLY), 0)} "
        f"fake interaction={totals.get(('train', NO_INTERACTION), 0)}"
    )
    log(
        "[split] total val=test: "
        f"friendly={totals.get(('val', FRIENDLY), 0)} "
        f"unfriendly={totals.get(('val', UNFRIENDLY), 0)} "
        f"fake interaction={totals.get(('val', NO_INTERACTION), 0)}"
    )


def attach_index_metadata(samples: list[Stage2Sample], row: dict[str, str], builder_source_set: str) -> None:
    source_name = row_source_name(row)
    bbox_subset = row_bbox_subset(row)
    raw = row_label(row)
    for sample in samples:
        sample.split = row_split(row)
        sample.source_name = source_name
        sample.index_row_id = str(row.get("index_row_id") or "").strip()
        sample.bbox_subset = bbox_subset
        sample.raw_label = raw
        sample.clip_name = str(row.get("clip_name") or "").strip()
        sample.clip_path = str(row.get("clip_path") or "").strip()
        sample.source_set = f"{source_name}:{bbox_subset}"
        if builder_source_set:
            sample.label_source = f"{sample.label_source};builder_source_set={builder_source_set}"


def build_samples_from_folder(
    folder: Path,
    root_tag: str,
    manifest_path: Path,
    manifest: dict,
    raw_label: str,
    original_class: str,
    label_source: str,
    annotation_events: list[AnnotationEvent] | None,
    args,
    num_kpts: int,
) -> list[Stage2Sample]:
    valence = training_label_for_task(raw_label, args.task)
    if not valence:
        raise ValueError(f"label {raw_label!r} is not usable for task {args.task}")
    kp_path = folder / "keypoints.csv"
    box_path = folder / "tracking_boxes.csv"
    if not kp_path.is_file() or not box_path.is_file():
        raise FileNotFoundError("missing tracking_boxes.csv or keypoints.csv")

    fps = float(manifest.get("video_manifest", {}).get("fps", 0.0) or args.fps)
    if fps <= 0:
        fps = float(args.fps)

    boxes_by_tid = load_boxes(box_path)
    if len(boxes_by_tid) < 2:
        raise ValueError("fewer than two tracks")

    choices = choose_pairs(
        boxes_by_tid=boxes_by_tid,
        fps=fps,
        min_seq_frames=args.min_seq_frames,
        samples_per_video=args.samples_per_video,
        annotation_events=annotation_events,
    )
    if not choices:
        raise ValueError("no pair sequence")

    kpts_by_key = load_keypoints(kp_path, num_kpts)
    source_video = str(manifest.get("video_manifest", {}).get("relative_path", ""))
    out: list[Stage2Sample] = []
    for choice in choices:
        x = build_pair_sequence(
            frames=choice.frames,
            tid_a=choice.tid_a,
            tid_b=choice.tid_b,
            boxes_by_tid=boxes_by_tid,
            kpts_by_key=kpts_by_key,
            num_kpts=num_kpts,
        )
        pti_keypoints, pti_person_mask = build_pti_sequence_from_frames(
            frames=choice.frames,
            tid_a=choice.tid_a,
            tid_b=choice.tid_b,
            kpts_by_key=kpts_by_key,
            num_kpts=num_kpts,
        )
        if x.shape[0] < args.min_seq_frames:
            raise ValueError("sequence too short")
        out.append(
            Stage2Sample(
                root_tag=root_tag,
                folder=folder.name,
                manifest_path=str(manifest_path),
                source_video=source_video,
                original_class=original_class,
                valence=valence,
                label_source=label_source,
                label_id=LABEL_TO_ID[valence],
                pair=(choice.tid_a, choice.tid_b),
                fps=fps,
                frames=list(choice.frames),
                x=x,
                source_set=root_tag,
                real_frames=len(choice.frames),
                pti_keypoints=pti_keypoints,
                pti_person_mask=pti_person_mask,
            )
        )
    return out


def build_selected_id_sample_from_folder(
    folder: Path,
    root_tag: str,
    source_set: str,
    manifest_path: Path,
    manifest: dict,
    raw_label: str,
    label_source: str,
    id_csv: Path,
    args,
    num_kpts: int,
) -> list[Stage2Sample]:
    valence = training_label_for_task(raw_label, args.task)
    if not valence:
        raise ValueError(f"label {raw_label!r} is not usable for task {args.task}")

    kp_path = folder / "keypoints.csv"
    box_path = folder / "tracking_boxes.csv"
    if not kp_path.is_file() or not box_path.is_file():
        raise FileNotFoundError("missing tracking_boxes.csv or keypoints.csv")

    fps = float(manifest.get("video_manifest", {}).get("fps", 0.0) or args.fps)
    if fps <= 0:
        fps = float(args.fps)

    frame_to_pair, counts = load_selected_id_rows(id_csv)
    dominant_pair, dominant_count = counts.most_common(1)[0]
    total_votes = int(sum(counts.values()))
    vote_ratio = float(dominant_count / max(1, total_votes))
    strict_ratio = float(args.min_dominant_vote_ratio)
    eval_ratio = float(args.min_eval_dominant_vote_ratio)
    min_ratio = eval_ratio if source_set == "GOODTEST" else strict_ratio
    if vote_ratio < min_ratio:
        log(
            "[warn] dominant pair vote ratio below historical threshold; keeping master-index row: "
            f"dominant pair vote ratio too low: {vote_ratio:.3f} < {min_ratio:.3f} "
            f"pair={dominant_pair} votes={dominant_count}/{total_votes}"
        )

    boxes_by_tid = load_boxes(box_path)
    kpts_by_key = load_keypoints(kp_path, num_kpts)
    frames = sorted(frame_to_pair)
    record_segments, record_stats = build_selected_id_records(
        frames=frames,
        dominant_pair=dominant_pair,
        frame_to_pair=frame_to_pair,
        boxes_by_tid=boxes_by_tid,
        kpts_by_key=kpts_by_key,
        fps=fps,
        args=args,
    )
    usable_segments = [segment for segment in record_segments if len(segment) >= args.min_seq_frames]
    if not usable_segments:
        raise ValueError(
            f"selected-ID sequence too short after dominant-only repair: "
            f"segments={[len(segment) for segment in record_segments]}"
        )

    source_video = str(manifest.get("video_manifest", {}).get("relative_path", ""))
    out: list[Stage2Sample] = []
    max_fill = float(args.max_pair_fill_frac)
    for seg_idx, records in enumerate(usable_segments):
        seg_stats = Counter(r.source for r in records)
        fill_count = int(seg_stats.get("interpolated", 0) + seg_stats.get("frozen", 0))
        fill_ratio = float(fill_count / max(1, len(records)))
        if source_set == "GOOD" and fill_ratio > max_fill:
            log(
                "[warn] pair repair ratio above historical threshold; keeping master-index row: "
                f"{fill_ratio:.3f} > {max_fill:.3f}"
            )
        x = build_sequence_from_records(records, kpts_by_key, num_kpts)
        pti_keypoints, pti_person_mask = build_pti_sequence_from_records(records, kpts_by_key, num_kpts)
        folder_name = folder.name if len(usable_segments) == 1 else f"{folder.name}_seg{seg_idx + 1:02d}"
        out.append(
            Stage2Sample(
                root_tag=root_tag,
                folder=folder_name,
                manifest_path=str(manifest_path),
                source_video=source_video,
                original_class=raw_label,
                valence=valence,
                label_source=label_source,
                label_id=LABEL_TO_ID[valence],
                pair=(int(dominant_pair[0]), int(dominant_pair[1])),
                fps=fps,
                frames=[r.frame for r in records],
                x=x,
                source_set=source_set,
                id_pair_vote_ratio=vote_ratio,
                id_pair_vote_count=int(dominant_count),
                id_pair_total_rows=total_votes,
                real_frames=int(seg_stats.get("real", 0)),
                alias_frames=0,
                interpolated_frames=int(seg_stats.get("interpolated", 0)),
                frozen_frames=int(seg_stats.get("frozen", 0)),
                dropped_frames=int(record_stats.get("dropped", 0)),
                fill_ratio=fill_ratio,
                pti_keypoints=pti_keypoints,
                pti_person_mask=pti_person_mask,
            )
        )
    return out


def discover_samples(args, keypoints: list[str]) -> list[Stage2Sample]:
    index_csv = index_path(args.stage2_index_csv)
    rows = load_stage2_index_rows(index_csv)
    log_canonical_split_report(rows, index_csv)

    if args.max_videos > 0:
        rows = rows[: args.max_videos]

    samples: list[Stage2Sample] = []
    skip_reasons: Counter[str] = Counter()
    label_sources: Counter[str] = Counter()
    num_kpts = len(keypoints)

    pbar = tqdm(rows, desc="build indexed samples", unit="clip", ascii=True, mininterval=1.0, file=sys.stdout)
    for row in pbar:
        raw_label = row_label(row)
        source_name = row_source_name(row)
        bbox_subset = row_bbox_subset(row)
        context = f"{source_name} {bbox_subset} {row.get('clip_name') or row.get('index_row_id')}"
        try:
            training_label = training_label_for_task(raw_label, args.task)
            if not training_label:
                skip_reasons[f"not_used_for_{args.task}:{raw_label}"] += 1
                continue

            folder = required_index_dir(f"{context} Stage1 folder", row.get("stage1_folder") or "")
            manifest_path = required_index_file(f"{context} Stage1 manifest", row.get("stage1_manifest") or "")
            tracking_csv = required_index_file(f"{context} tracking CSV", row.get("tracking_csv") or "")
            keypoints_csv = required_index_file(f"{context} keypoints CSV", row.get("keypoints_csv") or "")
            if tracking_csv.parent != folder or keypoints_csv.parent != folder or manifest_path.parent != folder:
                raise DataReferenceError(
                    f"{context}: Stage1 files do not share one folder: "
                    f"folder={folder} manifest={manifest_path} tracking={tracking_csv} keypoints={keypoints_csv}"
                )
            manifest = load_json(manifest_path)
            root_tag = f"{source_name}_{bbox_subset.lower()}"

            if raw_label in VALID_ANNOTATION_VALENCES:
                selected_id_csv = required_index_file(f"{context} selected-ID CSV", row.get("selected_id_csv") or "")
                builder_source_set = "GOODTEST" if bbox_subset.upper() == "GOODTEST" else "GOOD"
                new_samples = build_selected_id_sample_from_folder(
                    folder=folder,
                    root_tag=root_tag,
                    source_set=builder_source_set,
                    manifest_path=manifest_path,
                    manifest=manifest,
                    raw_label=raw_label,
                    label_source="stage2_master_index+selected_id_pair",
                    id_csv=selected_id_csv,
                    args=args,
                    num_kpts=num_kpts,
                )
            elif raw_label == NO_INTERACTION:
                builder_source_set = str(row.get("fake_group") or bbox_subset)
                new_samples = build_samples_from_folder(
                    folder=folder,
                    root_tag=root_tag,
                    manifest_path=manifest_path,
                    manifest=manifest,
                    raw_label=raw_label,
                    original_class=str(row.get("raw_valence") or NO_INTERACTION).strip() or NO_INTERACTION,
                    label_source="stage2_master_index+fake_interaction",
                    annotation_events=None,
                    args=args,
                    num_kpts=num_kpts,
                )
            else:
                raise ValueError(f"{context}: unsupported label={raw_label!r}")

            attach_index_metadata(new_samples, row, builder_source_set)
            samples.extend(new_samples)
            label_sources[new_samples[0].label_source if new_samples else "empty"] += len(new_samples)
        except Exception as exc:
            log(f"[error] failed to build master-index row for {context}: {exc}")
            raise

    log(f"[info] built {len(samples)} sample(s) from {len(rows)} master-index row(s)")
    log(f"[info] sample labels: {dict(Counter(s.valence for s in samples))}")
    log(f"[info] sample sources: {dict(Counter(s.source_name for s in samples))}")
    log(f"[info] sample source sets: {dict(Counter(s.source_set for s in samples))}")
    log(f"[info] label sources: {dict(label_sources)}")
    if skip_reasons:
        log(f"[info] skipped: {dict(skip_reasons)}")
    return samples


def sample_binary_valence(sample: Stage2Sample) -> str:
    if sample.valence in {FRIENDLY, UNFRIENDLY}:
        return sample.valence
    raw = str(sample.original_class or "").strip()
    if raw in {FRIENDLY, UNFRIENDLY}:
        return raw
    return CLASS_TO_VALENCE.get(raw, "")


def is_bbox_vis_sample(sample: Stage2Sample) -> bool:
    return bool(sample.index_row_id) or sample.root_tag in BBOX_VIS_ROOT_TAGS


def sample_split_group_key(sample: Stage2Sample) -> str:
    if sample.index_row_id:
        return f"stage2_master_index::{sample.index_row_id}"
    if is_bbox_vis_sample(sample):
        return f"bbox_vis_manifest::{sample.manifest_path}"
    return f"{sample.root_tag}::{sample.manifest_path}"


def sample_sort_key(sample: Stage2Sample) -> tuple[str, str, tuple[int, int], int]:
    first_frame = int(sample.frames[0]) if sample.frames else -1
    return sample.manifest_path, sample.folder, sample.pair, first_frame


def split_unit_sort_key(unit: SplitUnit) -> tuple[str, str, str]:
    first = unit.samples[0]
    return first.source_video, first.manifest_path, unit.key


def build_split_units(samples: list[Stage2Sample]) -> list[SplitUnit]:
    by_key: dict[str, list[Stage2Sample]] = defaultdict(list)
    for sample in samples:
        by_key[sample_split_group_key(sample)].append(sample)

    units: list[SplitUnit] = []
    for key, group_samples in by_key.items():
        ordered = sorted(group_samples, key=sample_sort_key)
        is_bbox = any(is_bbox_vis_sample(sample) for sample in ordered)
        if is_bbox and not all(is_bbox_vis_sample(sample) for sample in ordered):
            raise RuntimeError(f"mixed BBOX_VIS/non-BBOX split group: {key}")
        units.append(
            SplitUnit(
                key=key,
                samples=ordered,
                is_bbox_vis=is_bbox,
                is_multi_segment=is_bbox and len(ordered) > 1,
            )
        )
    return sorted(units, key=split_unit_sort_key)


def flatten_split_units(units: list[SplitUnit]) -> list[Stage2Sample]:
    out: list[Stage2Sample] = []
    for unit in sorted(units, key=split_unit_sort_key):
        out.extend(unit.samples)
    return out


def split_unit_binary_valence(unit: SplitUnit) -> str:
    sample = unit.samples[0]
    return sample_binary_valence(sample) or sample.valence


def validate_split_constraints(train: list[Stage2Sample], val: list[Stage2Sample]) -> None:
    membership: dict[str, set[str]] = defaultdict(set)
    for sample in train:
        membership[sample_split_group_key(sample)].add("train")
    for sample in val:
        membership[sample_split_group_key(sample)].add("val")

    mixed = {key: splits for key, splits in membership.items() if len(splits) > 1}
    if mixed:
        first_key = sorted(mixed)[0]
        raise RuntimeError(f"split group appears in multiple splits: {first_key} -> {sorted(mixed[first_key])}")


def deterministic_split(samples: list[Stage2Sample], task: str, seed: int) -> tuple[list[Stage2Sample], list[Stage2Sample]]:
    if task not in {GATE_TASK, VALENCE_TASK, UNIFIED_TASK}:
        raise ValueError(f"unsupported task: {task}")
    del seed

    missing = [s for s in samples if s.split not in {"train", "val"}]
    if missing:
        raise RuntimeError(f"samples missing master-index split, first={missing[0].manifest_path}")

    train = [s for s in samples if s.split == "train"]
    val = [s for s in samples if s.split == "val"]
    return sorted(train, key=sample_sort_key), sorted(val, key=sample_sort_key)


def crop_sequence(
    x: np.ndarray,
    fps: float,
    crop: float | str,
    train: bool,
    rng: random.Random,
    max_frames: int = 0,
) -> np.ndarray:
    if crop == "full":
        seq = x
    else:
        frames = max(2, int(round(float(crop) * fps)))
        if x.shape[0] <= frames:
            seq = x
        else:
            if train:
                start = rng.randint(0, x.shape[0] - frames)
            else:
                start = x.shape[0] - frames
            seq = x[start : start + frames]
    if max_frames > 0 and seq.shape[0] > max_frames:
        start2 = rng.randint(0, seq.shape[0] - max_frames) if train else seq.shape[0] - max_frames
        seq = seq[start2 : start2 + max_frames]
    return seq


def swap_pair_sequence(x: np.ndarray) -> np.ndarray:
    return swap_valence_pair_sequence(x)


class SequenceDataset(Dataset):
    def __init__(
        self,
        samples: list[Stage2Sample],
        crops: list[float | str],
        train: bool,
        seed: int,
        max_frames: int,
    ):
        self.samples = samples
        self.crops = crops
        self.train = train
        self.rng = random.Random(seed)
        self.max_frames = int(max_frames)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        crop = self.rng.choice(self.crops)
        x = crop_sequence(sample.x, sample.fps, crop, train=self.train, rng=self.rng, max_frames=self.max_frames)
        if self.train and self.rng.random() < 0.5:
            x = swap_pair_sequence(x)
        return x.astype(np.float32, copy=False), int(sample.label_id)


def pad_collate(batch):
    lengths = np.asarray([item[0].shape[0] for item in batch], dtype=np.int64)
    max_len = int(lengths.max())
    fdim = int(batch[0][0].shape[1])
    x = np.zeros((len(batch), max_len, fdim), dtype=np.float32)
    y = np.asarray([item[1] for item in batch], dtype=np.int64)
    for i, (seq, _label) in enumerate(batch):
        x[i, : seq.shape[0], :] = seq
    return torch.from_numpy(x), torch.from_numpy(lengths), torch.from_numpy(y)


def crop_pti_sequence(
    keypoints: np.ndarray,
    person_mask: np.ndarray,
    fps: float,
    crop: float | str,
    train: bool,
    rng: random.Random,
    max_frames: int,
) -> tuple[np.ndarray, np.ndarray]:
    if keypoints is None or person_mask is None:
        raise ValueError("PTI sample missing keypoints/person_mask")
    if keypoints.shape[0] != person_mask.shape[0]:
        raise ValueError(f"PTI keypoints/mask length mismatch: {keypoints.shape[0]} vs {person_mask.shape[0]}")

    start = 0
    end = keypoints.shape[0]
    if crop != "full":
        frames = max(2, int(round(float(crop) * fps)))
        if end > frames:
            start = rng.randint(0, end - frames) if train else end - frames
            end = start + frames

    seq = keypoints[start:end]
    mask = person_mask[start:end]
    if seq.shape[0] > max_frames:
        if train:
            start2 = rng.randint(0, seq.shape[0] - max_frames)
        else:
            start2 = seq.shape[0] - max_frames
        end2 = start2 + max_frames
        seq = seq[start2:end2]
        mask = mask[start2:end2]
    return seq, mask


def swap_pti_pair_sequence(keypoints: np.ndarray, person_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.ascontiguousarray(keypoints[:, [1, 0], :, :]), np.ascontiguousarray(person_mask[:, [1, 0]])


class PTIPairDataset(Dataset):
    def __init__(
        self,
        samples: list[Stage2Sample],
        crops: list[float | str],
        train: bool,
        seed: int,
        max_frames: int,
    ):
        self.samples = samples
        self.crops = crops
        self.train = train
        self.rng = random.Random(seed)
        self.max_frames = int(max_frames)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        crop = self.rng.choice(self.crops)
        keypoints, person_mask = crop_pti_sequence(
            sample.pti_keypoints,
            sample.pti_person_mask,
            sample.fps,
            crop,
            train=self.train,
            rng=self.rng,
            max_frames=self.max_frames,
        )
        if self.train and self.rng.random() < 0.5:
            keypoints, person_mask = swap_pti_pair_sequence(keypoints, person_mask)
        label = 1.0 if int(sample.label_id) == int(LABEL_TO_ID[INTERACTION]) else 0.0
        return keypoints.astype(np.float32, copy=False), person_mask.astype(np.bool_, copy=False), np.float32(label)


def pti_pad_collate(batch):
    lengths = np.asarray([item[0].shape[0] for item in batch], dtype=np.int64)
    max_len = int(lengths.max())
    persons = int(batch[0][0].shape[1])
    joints = int(batch[0][0].shape[2])
    channels = int(batch[0][0].shape[3])
    keypoints = np.zeros((len(batch), max_len, persons, joints, channels), dtype=np.float32)
    person_mask = np.zeros((len(batch), max_len, persons), dtype=np.bool_)
    labels = np.asarray([item[2] for item in batch], dtype=np.float32)
    for i, (seq, mask, _label) in enumerate(batch):
        keypoints[i, : seq.shape[0]] = seq
        person_mask[i, : mask.shape[0]] = mask
    return torch.from_numpy(keypoints), torch.from_numpy(person_mask), torch.from_numpy(labels)


def masked_mean_pool_sequence(h: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    _b, t, _c = h.shape
    mask = torch.arange(t, device=h.device).unsqueeze(0) < lengths.unsqueeze(1)
    mask_f = mask.to(dtype=h.dtype).unsqueeze(-1)
    denom = mask_f.sum(dim=1).clamp(min=1.0)
    return (h * mask_f).sum(dim=1) / denom


class ValenceTransformer(nn.Module):
    def __init__(
        self,
        in_features: int,
        num_classes: int,
        max_frames: int = 64,
        d_model: int = 64,
        num_heads: int = 4,
        temporal_layers: int = 2,
        ffn_dim: int = 128,
        dropout: float = 0.1,
        causal: bool = False,
    ):
        super().__init__()
        if int(max_frames) < 2:
            raise ValueError(f"max_frames must be >= 2, got {max_frames}")
        if int(d_model) % int(num_heads) != 0:
            raise ValueError(f"d_model={d_model} must be divisible by num_heads={num_heads}")
        self.in_features = int(in_features)
        self.max_frames = int(max_frames)
        self.d_model = int(d_model)
        self.causal = bool(causal)
        self.input_norm = nn.LayerNorm(self.in_features)
        self.in_proj = nn.Linear(self.in_features, self.d_model)
        self.time_pos = nn.Parameter(torch.zeros(1, self.max_frames, self.d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=int(num_heads),
            dim_feedforward=int(ffn_dim),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=int(temporal_layers))
        self.out_norm = nn.LayerNorm(self.d_model)
        self.head = nn.Sequential(
            nn.Dropout(float(dropout)),
            nn.Linear(self.d_model, int(num_classes)),
        )
        nn.init.trunc_normal_(self.time_pos, std=0.02)

    def _causal_mask(self, size: int, device: torch.device) -> torch.Tensor | None:
        if not self.causal:
            return None
        return torch.triu(torch.ones(size, size, device=device, dtype=torch.bool), diagonal=1)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"expected x shape [B,T,F], got {tuple(x.shape)}")
        batch, frames, features = x.shape
        if int(features) != self.in_features:
            raise ValueError(f"expected {self.in_features} input features, got {features}")
        if int(frames) > self.max_frames:
            raise ValueError(f"sequence length {frames} exceeds max_frames={self.max_frames}")
        lengths = lengths.to(device=x.device, dtype=torch.long).clamp(min=1, max=max(1, int(frames)))
        padding_mask = torch.arange(frames, device=x.device).unsqueeze(0) >= lengths.unsqueeze(1)
        h = self.in_proj(self.input_norm(x))
        h = h + self.time_pos[:, :frames, :].to(device=x.device, dtype=h.dtype)
        h = self.encoder(h, mask=self._causal_mask(int(frames), x.device), src_key_padding_mask=padding_mask)
        h = self.out_norm(h)
        pooled = masked_mean_pool_sequence(h, lengths)
        return self.head(pooled)


def class_weights(samples: list[Stage2Sample], device: torch.device) -> torch.Tensor:
    counts = Counter(s.label_id for s in samples)
    total = float(sum(counts.values()))
    weights = []
    for idx in range(len(LABEL_TO_ID)):
        weights.append(total / max(1.0, len(LABEL_TO_ID) * float(counts.get(idx, 0))))
    return torch.tensor(weights, dtype=torch.float32, device=device)


def macro_f1(y_true: list[int], y_pred: list[int], num_classes: int) -> float:
    f1s = []
    for cls in range(num_classes):
        tp = sum(1 for y, p in zip(y_true, y_pred) if y == cls and p == cls)
        fp = sum(1 for y, p in zip(y_true, y_pred) if y != cls and p == cls)
        fn = sum(1 for y, p in zip(y_true, y_pred) if y == cls and p != cls)
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1s.append(0.0 if precision + recall == 0 else 2.0 * precision * recall / (precision + recall))
    return float(sum(f1s) / max(1, len(f1s)))


def softmax_np(logits: np.ndarray) -> np.ndarray:
    x = logits.astype(np.float32)
    x = x - np.max(x)
    ex = np.exp(x)
    return ex / max(1e-9, float(ex.sum()))


def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    raise RuntimeError("CUDA/GPU is required for Stage2 training; refusing to run on CPU.")


def profile_defaults(profile: str) -> dict[str, int]:
    selected = profile
    if selected == "auto":
        selected = "8gb"
        if torch.cuda.is_available():
            gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            selected = "32gb" if gb >= 30 else "8gb"
    if selected == "32gb":
        return {"batch_size": 48, "eval_batch_size": 96, "num_workers": 0}
    return {"batch_size": 8, "eval_batch_size": 32, "num_workers": 0}


@torch.no_grad()
def logits_for_sequence(model: nn.Module, seq: np.ndarray, device: torch.device) -> np.ndarray:
    x = torch.from_numpy(seq[None, ...]).to(device=device, dtype=torch.float32)
    lengths = torch.tensor([seq.shape[0]], device=device, dtype=torch.long)
    logits = model(x, lengths).squeeze(0)
    return logits.detach().float().cpu().numpy()


@torch.no_grad()
def evaluate(
    model: nn.Module,
    samples: list[Stage2Sample],
    crops: list[float | str],
    device: torch.device,
    criterion: nn.Module,
    bidirectional: bool,
    max_frames: int,
) -> dict:
    model.eval()
    rng = random.Random(12345)
    y_true: list[int] = []
    y_pred: list[int] = []
    losses: list[float] = []
    probs_by_sample = []

    for sample in samples:
        logits_list = []
        for crop in crops:
            seq = crop_sequence(sample.x, sample.fps, crop, train=False, rng=rng, max_frames=max_frames)
            logits_ab = logits_for_sequence(model, seq, device)
            if bidirectional:
                logits_ba = logits_for_sequence(model, swap_pair_sequence(seq), device)
                conf_ab = float(np.max(softmax_np(logits_ab)))
                conf_ba = float(np.max(softmax_np(logits_ba)))
                logits_list.append(logits_ab if conf_ab >= conf_ba else logits_ba)
            else:
                logits_list.append(logits_ab)
        avg_logits = np.mean(np.stack(logits_list, axis=0), axis=0).astype(np.float32)
        target = torch.tensor([sample.label_id], dtype=torch.long, device=device)
        logits_t = torch.from_numpy(avg_logits[None, :]).to(device)
        losses.append(float(criterion(logits_t, target).detach().cpu().item()))
        pred = int(np.argmax(avg_logits))
        y_true.append(sample.label_id)
        y_pred.append(pred)
        probs_by_sample.append(softmax_np(avg_logits).tolist())

    acc = sum(int(y == p) for y, p in zip(y_true, y_pred)) / max(1, len(y_true))
    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "acc": float(acc),
        "macro_f1": macro_f1(y_true, y_pred, len(LABEL_TO_ID)),
        "y_true": y_true,
        "y_pred": y_pred,
        "probs": probs_by_sample,
    }


def pti_config_from_args(args, num_joints: int) -> dict:
    return {
        "num_joints": int(num_joints),
        "max_persons": 2,
        "max_frames": int(args.pti_max_frames),
        "d_model": int(args.pti_d_model),
        "num_heads": int(args.pti_num_heads),
        "temporal_layers": int(args.pti_temporal_layers),
        "pair_layers": int(args.pti_pair_layers),
        "ffn_dim": int(args.pti_ffn_dim),
        "dropout": float(args.pti_dropout),
        "causal": bool(args.pti_causal),
    }


def valence_transformer_config_from_args(args, in_features: int) -> dict:
    return {
        "in_features": int(in_features),
        "max_frames": int(args.pti_max_frames),
        "d_model": int(args.valence_d_model),
        "num_heads": int(args.valence_num_heads),
        "temporal_layers": int(args.valence_temporal_layers),
        "ffn_dim": int(args.valence_ffn_dim),
        "dropout": float(args.valence_dropout),
        "causal": bool(args.valence_causal),
    }


def pti_pos_weight(samples: list[Stage2Sample], device: torch.device) -> torch.Tensor:
    pos_id = int(LABEL_TO_ID[INTERACTION])
    pos = sum(1 for sample in samples if int(sample.label_id) == pos_id)
    neg = max(0, len(samples) - pos)
    weight = float(neg / max(1, pos))
    return torch.tensor([weight], dtype=torch.float32, device=device)


@torch.no_grad()
def pti_logit_for_sequence(
    model: PTIHead,
    keypoints: np.ndarray,
    person_mask: np.ndarray,
    device: torch.device,
) -> float:
    k_t = torch.from_numpy(keypoints[None, ...]).to(device=device, dtype=torch.float32)
    m_t = torch.from_numpy(person_mask[None, ...]).to(device=device, dtype=torch.bool)
    out = model(k_t, m_t)
    return float(out["logit"].detach().float().cpu().item())


@torch.no_grad()
def evaluate_pti_gate(
    model: PTIHead,
    samples: list[Stage2Sample],
    crops: list[float | str],
    device: torch.device,
    criterion: nn.Module,
    bidirectional: bool,
    max_frames: int,
) -> dict:
    model.eval()
    rng = random.Random(12345)
    y_true: list[int] = []
    y_pred: list[int] = []
    losses: list[float] = []
    probs_by_sample = []
    pos_id = int(LABEL_TO_ID[INTERACTION])

    for sample in samples:
        logits: list[float] = []
        for crop in crops:
            keypoints, person_mask = crop_pti_sequence(
                sample.pti_keypoints,
                sample.pti_person_mask,
                sample.fps,
                crop,
                train=False,
                rng=rng,
                max_frames=max_frames,
            )
            logits.append(pti_logit_for_sequence(model, keypoints, person_mask, device))
            if bidirectional:
                swapped_k, swapped_m = swap_pti_pair_sequence(keypoints, person_mask)
                logits.append(pti_logit_for_sequence(model, swapped_k, swapped_m, device))

        avg_logit = float(np.mean(np.asarray(logits, dtype=np.float32)))
        target_value = 1.0 if int(sample.label_id) == pos_id else 0.0
        target = torch.tensor([target_value], dtype=torch.float32, device=device)
        logit_t = torch.tensor([avg_logit], dtype=torch.float32, device=device)
        losses.append(float(criterion(logit_t, target).detach().cpu().item()))
        prob = float(torch.sigmoid(logit_t).detach().cpu().item())
        pred = 1 if prob >= 0.5 else 0
        true = 1 if target_value >= 0.5 else 0
        y_true.append(true)
        y_pred.append(pred)
        probs_by_sample.append([1.0 - prob, prob])

    acc = sum(int(y == p) for y, p in zip(y_true, y_pred)) / max(1, len(y_true))
    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "acc": float(acc),
        "macro_f1": macro_f1(y_true, y_pred, 2),
        "y_true": y_true,
        "y_pred": y_pred,
        "probs": probs_by_sample,
    }


def train_pti_gate(
    args,
    keypoints: list[str],
    train_samples: list[Stage2Sample],
    val_samples: list[Stage2Sample],
    output_path: Path,
    crops: list[float | str],
    device: torch.device,
) -> Path:
    cfg = pti_config_from_args(args, len(keypoints))
    log(f"[pti] training interaction gate with config={cfg}")
    model = PTIHead(**cfg).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pti_pos_weight(train_samples, device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and args.amp))

    train_ds = PTIPairDataset(train_samples, crops=crops, train=True, seed=args.seed, max_frames=cfg["max_frames"])
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=pti_pad_collate,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    best = {"macro_f1": -1.0, "acc": -1.0, "loss": float("inf"), "epoch": 0}
    epochs_without_improve = 0
    last_heartbeat = time.monotonic()
    pos_id = int(LABEL_TO_ID[INTERACTION])

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        train_true: list[int] = []
        train_pred: list[int] = []
        pbar = tqdm(
            train_loader,
            desc=f"pti epoch {epoch}/{args.epochs}",
            unit="batch",
            ascii=True,
            dynamic_ncols=False,
            ncols=100,
            mininterval=1.0,
            file=sys.stdout,
        )
        for kb, mb, yb in pbar:
            kb = kb.to(device=device, dtype=torch.float32, non_blocking=True)
            mb = mb.to(device=device, dtype=torch.bool, non_blocking=True)
            yb = yb.to(device=device, dtype=torch.float32, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda" and args.amp), dtype=torch.float16):
                out = model(kb, mb)
                loss = criterion(out["logit"], yb)
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            batch_n = int(yb.shape[0])
            running_loss += float(loss.detach().cpu().item()) * batch_n
            seen += batch_n
            pred = (out["prob"].detach() >= 0.5).long()
            train_true.extend(yb.detach().long().cpu().tolist())
            train_pred.extend(pred.cpu().tolist())
            pbar.set_postfix(loss=f"{running_loss / max(1, seen):.4f}")

            now = time.monotonic()
            if args.heartbeat_sec > 0 and now - last_heartbeat >= args.heartbeat_sec:
                log(f"[heartbeat] pti epoch={epoch} seen={seen}/{len(train_ds)} loss={running_loss / max(1, seen):.4f}")
                last_heartbeat = now

        scheduler.step()
        train_loss = running_loss / max(1, seen)
        train_acc = sum(int(y == p) for y, p in zip(train_true, train_pred)) / max(1, len(train_true))
        train_f1 = macro_f1(train_true, train_pred, 2)
        val_metrics = evaluate_pti_gate(
            model,
            val_samples,
            crops,
            device,
            criterion,
            bidirectional=args.bidirectional,
            max_frames=cfg["max_frames"],
        )
        improved = (
            val_metrics["macro_f1"] > best["macro_f1"] + 1e-9
            or (
                abs(val_metrics["macro_f1"] - best["macro_f1"]) <= 1e-9
                and val_metrics["loss"] < best["loss"]
            )
        )
        if improved:
            best = {
                "macro_f1": float(val_metrics["macro_f1"]),
                "acc": float(val_metrics["acc"]),
                "loss": float(val_metrics["loss"]),
                "epoch": epoch,
            }
            ckpt = {
                "model": "PTIHead",
                "pipeline_stage": "csv_interaction_analysis",
                "model_role": "interaction_gate_classifier",
                "stage2_task": str(args.task),
                "state_dict": model.state_dict(),
                "feature_schema": PTI_FEATURE_SCHEMA_VERSION,
                "input_shape": "[B,T,2,J,3]",
                "keypoints": keypoints,
                "num_joints": len(keypoints),
                "max_persons": 2,
                "max_frames": int(cfg["max_frames"]),
                "fps": float(args.fps),
                "label_to_id": dict(LABEL_TO_ID),
                "id_to_label": dict(ID_TO_LABEL),
                "no_interaction_label": NO_INTERACTION,
                "interaction_label": INTERACTION,
                "interaction_threshold": 0.50,
                "threshold": 0.50,
                "pti": dict(cfg),
                "pos_weight": criterion.pos_weight.detach().cpu().tolist()
                if getattr(criterion, "pos_weight", None) is not None
                else None,
                "crop_seconds": list(crops),
                "mi_test_seconds": list(crops),
                "mi_test_n": len(crops),
                "seed": int(args.seed),
                "split_seed": int(args.split_seed),
                "best_val_macroF1": float(best["macro_f1"]),
                "best_val_acc": float(best["acc"]),
                "best_val_loss": float(best["loss"]),
                "best_epoch": int(epoch),
                "val_folders": [s.folder for s in val_samples],
                "train_folders": [s.folder for s in train_samples],
                "stage2_index_csv": str(index_path(args.stage2_index_csv)),
                "source_names": sorted({s.source_name for s in train_samples + val_samples if s.source_name}),
                "split_rule": split_rule_for_task(args.task),
                "split_group_rule": "stage2_master_index index_row_id is the split group; clip/event-level split is authoritative",
                "created_at": datetime.now().isoformat(timespec="seconds"),
            }
            torch.save(ckpt, output_path)
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1

        log(
            f"[pti epoch {epoch:03d}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} train_f1={train_f1:.3f} "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['acc']:.3f} "
            f"val_macroF1={val_metrics['macro_f1']:.3f} best_epoch={best['epoch']}"
        )

        if args.patience > 0 and epochs_without_improve >= args.patience:
            log(f"[info] early stopping after {epochs_without_improve} epoch(s) without improvement")
            break

    log(
        f"[done] saved best PTI gate checkpoint: {output_path} "
        f"epoch={best['epoch']} val_macroF1={best['macro_f1']:.3f} val_acc={best['acc']:.3f}"
    )
    return output_path


def save_split_summary(path: Path, train: list[Stage2Sample], val: list[Stage2Sample]) -> None:
    rows = []
    group_sizes = Counter(sample_split_group_key(sample) for sample in train + val)
    for split, samples in (("train", train), ("val", val)):
        for s in samples:
            group_key = sample_split_group_key(s)
            rows.append(
                {
                    "split": split,
                    "split_group": group_key,
                    "split_group_size": int(group_sizes[group_key]),
                    "split_group_is_bbox_vis": bool(is_bbox_vis_sample(s)),
                    "source_name": s.source_name,
                    "index_row_id": s.index_row_id,
                    "bbox_subset": s.bbox_subset,
                    "raw_label": s.raw_label,
                    "clip_name": s.clip_name,
                    "clip_path": s.clip_path,
                    "root_tag": s.root_tag,
                    "folder": s.folder,
                    "manifest_path": s.manifest_path,
                    "source_video": s.source_video,
                    "original_class": s.original_class,
                    "label": s.valence,
                    "label_source": s.label_source,
                    "tidA": s.pair[0],
                    "tidB": s.pair[1],
                    "frames": len(s.frames),
                    "fps": s.fps,
                    "source_set": s.source_set,
                    "id_pair_vote_ratio": s.id_pair_vote_ratio,
                    "id_pair_vote_count": s.id_pair_vote_count,
                    "id_pair_total_rows": s.id_pair_total_rows,
                    "real_frames": s.real_frames,
                    "alias_frames": s.alias_frames,
                    "interpolated_frames": s.interpolated_frames,
                    "frozen_frames": s.frozen_frames,
                    "dropped_frames": s.dropped_frames,
                    "fill_ratio": s.fill_ratio,
                }
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def train(args) -> Path:
    set_active_task_labels(args.task)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device = resolve_device()
    if device.type == "cuda":
        log(f"[info] device: {torch.cuda.get_device_name(0)}")
    else:
        log(f"[info] device: {device}")

    defaults = profile_defaults(args.profile)
    if args.batch_size is None:
        args.batch_size = defaults["batch_size"]
    if args.eval_batch_size is None:
        args.eval_batch_size = defaults["eval_batch_size"]
    if args.num_workers is None:
        args.num_workers = defaults["num_workers"]
    log(
        "[info] profile="
        f"{args.profile} batch_size={args.batch_size} eval_batch_size={args.eval_batch_size} "
        f"num_workers={args.num_workers}"
    )
    log(f"[info] task={args.task} labels={LABEL_TO_ID}")

    keypoints, _template_meta = load_template_metadata(Path(args.template_ckpt))
    in_features = valence_expected_in_features(len(keypoints))
    log(f"[info] keypoints={len(keypoints)} valence_features={in_features} feature_schema={FEATURE_SCHEMA_VERSION}")

    samples = discover_samples(args, keypoints)
    if len(samples) < 4:
        raise RuntimeError(f"too few samples after filtering: {len(samples)}")

    train_samples, val_samples = deterministic_split(samples, args.task, args.split_seed)
    if not train_samples or not val_samples:
        raise RuntimeError("empty train or val split")
    validate_split_constraints(train_samples, val_samples)

    log(f"[info] train labels: {dict(Counter(s.valence for s in train_samples))}")
    log(f"[info] val labels: {dict(Counter(s.valence for s in val_samples))}")
    log(f"[info] train sources: {dict(Counter(s.source_name for s in train_samples))}")
    log(f"[info] val sources: {dict(Counter(s.source_name for s in val_samples))}")
    log(f"[info] train roots: {dict(Counter(s.root_tag for s in train_samples))}")
    log(f"[info] val roots: {dict(Counter(s.root_tag for s in val_samples))}")
    log(
        "[info] split groups: "
        f"train={len(build_split_units(train_samples))} val={len(build_split_units(val_samples))}"
    )
    log(f"[info] val manifests: {[s.manifest_path for s in val_samples]}")

    output_path = Path(args.output) if args.output else output_path_for_task(args.task)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_split_summary(split_path_for_task(output_path.parent, args.task), train_samples, val_samples)

    if args.dry_run:
        log("[info] dry run requested; not training.")
        return output_path

    crops = parse_seconds_list(args.crop_seconds)
    if args.task == GATE_TASK:
        return train_pti_gate(args, keypoints, train_samples, val_samples, output_path, crops, device)

    transformer_cfg = valence_transformer_config_from_args(args, in_features)
    log(f"[info] training valence Transformer with config={transformer_cfg}")
    model = ValenceTransformer(num_classes=len(LABEL_TO_ID), **transformer_cfg).to(device)

    weights = class_weights(train_samples, device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and args.amp))

    train_ds = SequenceDataset(
        train_samples,
        crops=crops,
        train=True,
        seed=args.seed,
        max_frames=transformer_cfg["max_frames"],
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=pad_collate,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    best = {"macro_f1": -1.0, "acc": -1.0, "loss": float("inf"), "epoch": 0}
    epochs_without_improve = 0
    last_heartbeat = time.monotonic()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        train_true: list[int] = []
        train_pred: list[int] = []
        pbar = tqdm(
            train_loader,
            desc=f"epoch {epoch}/{args.epochs}",
            unit="batch",
            ascii=True,
            dynamic_ncols=False,
            ncols=100,
            mininterval=1.0,
            file=sys.stdout,
        )
        for xb, lengths, yb in pbar:
            xb = xb.to(device=device, dtype=torch.float32, non_blocking=True)
            lengths = lengths.to(device=device, dtype=torch.long, non_blocking=True)
            yb = yb.to(device=device, dtype=torch.long, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda" and args.amp), dtype=torch.float16):
                logits = model(xb, lengths)
                loss = criterion(logits, yb)
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            batch_n = int(yb.shape[0])
            running_loss += float(loss.detach().cpu().item()) * batch_n
            seen += batch_n
            pred = torch.argmax(logits.detach(), dim=1)
            train_true.extend(yb.detach().cpu().tolist())
            train_pred.extend(pred.cpu().tolist())
            pbar.set_postfix(loss=f"{running_loss / max(1, seen):.4f}")

            now = time.monotonic()
            if args.heartbeat_sec > 0 and now - last_heartbeat >= args.heartbeat_sec:
                log(f"[heartbeat] epoch={epoch} seen={seen}/{len(train_ds)} loss={running_loss / max(1, seen):.4f}")
                last_heartbeat = now

        scheduler.step()
        train_loss = running_loss / max(1, seen)
        train_acc = sum(int(y == p) for y, p in zip(train_true, train_pred)) / max(1, len(train_true))
        train_f1 = macro_f1(train_true, train_pred, len(LABEL_TO_ID))
        val_metrics = evaluate(
            model,
            val_samples,
            crops,
            device,
            criterion,
            bidirectional=args.bidirectional,
            max_frames=transformer_cfg["max_frames"],
        )

        improved = (
            val_metrics["macro_f1"] > best["macro_f1"] + 1e-9
            or (
                abs(val_metrics["macro_f1"] - best["macro_f1"]) <= 1e-9
                and val_metrics["loss"] < best["loss"]
            )
        )
        if improved:
            best = {
                "macro_f1": float(val_metrics["macro_f1"]),
                "acc": float(val_metrics["acc"]),
                "loss": float(val_metrics["loss"]),
                "epoch": epoch,
            }
            ckpt = {
                "model": "ValenceTransformer",
                "pipeline_stage": "csv_interaction_analysis",
                "model_role": "interaction_gate_classifier" if args.task == GATE_TASK else "valence_classifier" if args.task == VALENCE_TASK else "stage2_interaction_classifier",
                "stage2_task": str(args.task),
                "state_dict": model.state_dict(),
                "in_features": in_features,
                "feature_schema": FEATURE_SCHEMA_VERSION,
                "input_shape": "[B,T,F]",
                "feature_names": valence_feature_names(keypoints),
                "keypoints": keypoints,
                "max_frames": int(transformer_cfg["max_frames"]),
                "fps": float(args.fps),
                "label_to_id": dict(LABEL_TO_ID),
                "id_to_label": dict(ID_TO_LABEL),
                "class_to_valence": dict(CLASS_TO_VALENCE),
                "no_interaction_label": NO_INTERACTION,
                "interaction_label": INTERACTION,
                "interaction_labels": [FRIENDLY, UNFRIENDLY],
                "interaction_threshold": 0.50,
                "transformer": dict(transformer_cfg),
                "dropout": float(transformer_cfg["dropout"]),
                "class_weights": weights.detach().cpu().tolist(),
                "label_smoothing": float(args.label_smoothing),
                "crop_seconds": list(crops),
                "crop_probs": None,
                "mi_test_seconds": list(crops),
                "mi_test_n": len(crops),
                "seed": int(args.seed),
                "split_seed": int(args.split_seed),
                "best_val_macroF1": float(best["macro_f1"]),
                "best_val_acc": float(best["acc"]),
                "best_val_loss": float(best["loss"]),
                "best_epoch": int(epoch),
                "val_folders": [s.folder for s in val_samples],
                "train_folders": [s.folder for s in train_samples],
                "stage2_index_csv": str(index_path(args.stage2_index_csv)),
                "source_names": sorted({s.source_name for s in train_samples + val_samples if s.source_name}),
                "max_id_gap_sec": float(args.max_id_gap_sec),
                "max_pair_fill_frac": float(args.max_pair_fill_frac),
                "min_dominant_vote_ratio": float(args.min_dominant_vote_ratio),
                "min_eval_dominant_vote_ratio": float(args.min_eval_dominant_vote_ratio),
                "dominant_bbox_gap_policy": "short gaps interpolate; short tail gaps freeze; long gaps split/drop",
                "split_rule": split_rule_for_task(args.task),
                "split_group_rule": "stage2_master_index index_row_id is the split group; clip/event-level split is authoritative",
                "created_at": datetime.now().isoformat(timespec="seconds"),
            }
            torch.save(ckpt, output_path)
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1

        log(
            f"[epoch {epoch:03d}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} train_f1={train_f1:.3f} "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['acc']:.3f} "
            f"val_macroF1={val_metrics['macro_f1']:.3f} best_epoch={best['epoch']}"
        )

        if args.patience > 0 and epochs_without_improve >= args.patience:
            log(f"[info] early stopping after {epochs_without_improve} epoch(s) without improvement")
            break

    log(
        f"[done] saved best checkpoint: {output_path} "
        f"epoch={best['epoch']} val_macroF1={best['macro_f1']:.3f} val_acc={best['acc']:.3f}"
    )
    return output_path


def checkpoint_id_to_label(ckpt: dict) -> dict[int, str]:
    raw = ckpt.get("id_to_label", {})
    out: dict[int, str] = {}
    if isinstance(raw, dict):
        for key, value in raw.items():
            try:
                out[int(key)] = str(value)
            except Exception:
                continue
    if out:
        return out
    raw_label_to_id = ckpt.get("label_to_id", {})
    if isinstance(raw_label_to_id, dict):
        for label, idx in raw_label_to_id.items():
            try:
                out[int(idx)] = str(label)
            except Exception:
                continue
    if not out:
        raise RuntimeError("checkpoint has no usable id_to_label/label_to_id mapping")
    return out


def load_valence_transformer_checkpoint(path: Path, device: torch.device) -> tuple[ValenceTransformer, dict[int, str], int]:
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    ckpt = torch.load(str(path), map_location=device)
    if str(ckpt.get("model", "")) != "ValenceTransformer":
        raise RuntimeError(f"checkpoint is not a ValenceTransformer valence model: {path}")
    if str(ckpt.get("feature_schema", "")) != FEATURE_SCHEMA_VERSION:
        raise RuntimeError(
            f"valence checkpoint feature_schema={ckpt.get('feature_schema', '')!r}; "
            f"expected {FEATURE_SCHEMA_VERSION!r}"
        )
    id_to_label = checkpoint_id_to_label(ckpt)
    cfg = dict(ckpt.get("transformer", {}))
    if not cfg:
        raise RuntimeError(f"ValenceTransformer checkpoint missing transformer config: {path}")
    cfg["in_features"] = int(cfg.get("in_features", ckpt.get("in_features", -1)))
    if int(cfg["in_features"]) <= 0:
        raise RuntimeError(f"ValenceTransformer checkpoint missing in_features: {path}")
    if int(cfg["in_features"]) != int(ckpt.get("in_features", cfg["in_features"])):
        raise RuntimeError(
            f"ValenceTransformer in_features mismatch: transformer={cfg['in_features']} "
            f"top_level={ckpt.get('in_features')}"
        )
    model = ValenceTransformer(num_classes=len(id_to_label), **cfg).to(device)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()
    return model, id_to_label, int(cfg["max_frames"])


def load_pti_checkpoint(path: Path, device: torch.device) -> tuple[PTIHead, dict[int, str], float, int]:
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    ckpt = torch.load(str(path), map_location=device)
    if str(ckpt.get("model", "")) != "PTIHead" and str(ckpt.get("feature_schema", "")) != PTI_FEATURE_SCHEMA_VERSION:
        raise RuntimeError(f"checkpoint is not a PTIHead gate: {path}")
    cfg = dict(ckpt.get("pti", {}))
    if not cfg:
        cfg = {
            "num_joints": int(ckpt.get("num_joints", len(ckpt.get("keypoints", [])) or 27)),
            "max_persons": int(ckpt.get("max_persons", 2)),
            "max_frames": int(ckpt.get("max_frames", 64)),
            "d_model": 64,
            "num_heads": 4,
            "temporal_layers": 2,
            "pair_layers": 1,
            "ffn_dim": 128,
            "dropout": 0.1,
            "causal": False,
        }
    model = PTIHead(**cfg).to(device)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()
    id_to_label = checkpoint_id_to_label(ckpt)
    threshold = float(ckpt.get("threshold", ckpt.get("interaction_threshold", 0.5)) or 0.5)
    return model, id_to_label, threshold, int(cfg["max_frames"])


def load_gate_checkpoint(path: Path, device: torch.device):
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    return ("pti", *load_pti_checkpoint(path, device))


@torch.no_grad()
def predict_sample(
    model: nn.Module,
    id_to_label: dict[int, str],
    sample: Stage2Sample,
    crops: list[float | str],
    device: torch.device,
    bidirectional: bool,
    max_frames: int = 0,
) -> tuple[str, list[float]]:
    logits_list = []
    rng = random.Random(12345)
    for crop in crops:
        seq = crop_sequence(sample.x, sample.fps, crop, train=False, rng=rng, max_frames=max_frames)
        logits_ab = logits_for_sequence(model, seq, device)
        if bidirectional:
            logits_ba = logits_for_sequence(model, swap_pair_sequence(seq), device)
            conf_ab = float(np.max(softmax_np(logits_ab)))
            conf_ba = float(np.max(softmax_np(logits_ba)))
            logits_list.append(logits_ab if conf_ab >= conf_ba else logits_ba)
        else:
            logits_list.append(logits_ab)
    avg_logits = np.mean(np.stack(logits_list, axis=0), axis=0).astype(np.float32)
    probs = softmax_np(avg_logits).tolist()
    pred_id = int(np.argmax(avg_logits))
    return id_to_label.get(pred_id, str(pred_id)), probs


@torch.no_grad()
def predict_pti_gate_sample(
    model: PTIHead,
    sample: Stage2Sample,
    crops: list[float | str],
    device: torch.device,
    bidirectional: bool,
    threshold: float,
    max_frames: int,
) -> tuple[str, list[float]]:
    rng = random.Random(12345)
    logits: list[float] = []
    for crop in crops:
        keypoints, person_mask = crop_pti_sequence(
            sample.pti_keypoints,
            sample.pti_person_mask,
            sample.fps,
            crop,
            train=False,
            rng=rng,
            max_frames=max_frames,
        )
        logits.append(pti_logit_for_sequence(model, keypoints, person_mask, device))
        if bidirectional:
            swapped_k, swapped_m = swap_pti_pair_sequence(keypoints, person_mask)
            logits.append(pti_logit_for_sequence(model, swapped_k, swapped_m, device))
    avg_logit = float(np.mean(np.asarray(logits, dtype=np.float32)))
    prob = float(torch.sigmoid(torch.tensor(avg_logit)).item())
    pred = INTERACTION if prob >= float(threshold) else NO_INTERACTION
    return pred, [1.0 - prob, prob]


def confusion_matrix_from_predictions(rows: list[dict[str, str]]) -> dict[str, dict[str, int]]:
    matrix = {true: {pred: 0 for pred in CASCADE_LABELS} for true in CASCADE_LABELS}
    for row in rows:
        true = str(row["true"])
        pred = str(row["pred"])
        if true in matrix and pred in matrix[true]:
            matrix[true][pred] += 1
    return matrix


def write_overall_confusion(path: Path, rows: list[dict[str, str]]) -> None:
    matrix = confusion_matrix_from_predictions(rows)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([""] + CASCADE_LABELS)
        for true in CASCADE_LABELS:
            writer.writerow([true] + [matrix[true][pred] for pred in CASCADE_LABELS])


def write_source_confusions(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sources = sorted({str(row["source_name"]) for row in rows})
    scoped = [("overall", rows)] + [(source, [row for row in rows if row["source_name"] == source]) for source in sources]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["source_name", "true_label", *CASCADE_LABELS, "total"])
        writer.writeheader()
        for source_name, subset in scoped:
            matrix = confusion_matrix_from_predictions(subset)
            for true in CASCADE_LABELS:
                counts = {pred: matrix[true][pred] for pred in CASCADE_LABELS}
                writer.writerow(
                    {
                        "source_name": source_name,
                        "true_label": true,
                        **counts,
                        "total": sum(counts.values()),
                    }
                )


def log_confusions(rows: list[dict[str, str]]) -> None:
    sources = ["overall"] + sorted({str(row["source_name"]) for row in rows})
    for source_name in sources:
        subset = rows if source_name == "overall" else [row for row in rows if row["source_name"] == source_name]
        matrix = confusion_matrix_from_predictions(subset)
        log(f"[cascade-val] confusion {source_name}:")
        log("[cascade-val] true\\pred," + ",".join(CASCADE_LABELS))
        for true in CASCADE_LABELS:
            log("[cascade-val] " + ",".join([true] + [str(matrix[true][pred]) for pred in CASCADE_LABELS]))


def interaction_prob_from_gate_probs(probs: list[float], id_to_label: dict[int, str]) -> float:
    for idx, prob in enumerate(probs):
        if str(id_to_label.get(int(idx), "")).lower() == INTERACTION:
            return float(prob)
    return float(probs[1]) if len(probs) > 1 else 0.0


def valence_label_conf_from_probs(probs: list[float], id_to_label: dict[int, str]) -> tuple[str, float]:
    best_label = ""
    best_conf = 0.0
    for idx, prob in enumerate(probs):
        raw_label = str(id_to_label.get(int(idx), str(idx))).strip()
        label = raw_label.lower()
        if label not in {FRIENDLY, UNFRIENDLY}:
            label = CLASS_TO_VALENCE.get(raw_label, "")
        if label in {FRIENDLY, UNFRIENDLY} and float(prob) > best_conf:
            best_label = label
            best_conf = float(prob)
    return best_label, best_conf


def save_dynamic_threshold_to_valence_checkpoint(path: Path, calibration: dict) -> None:
    ckpt = torch.load(str(path), map_location="cpu")
    ckpt["dynamic_valence_conf_threshold"] = calibration
    torch.save(ckpt, str(path))


def evaluate_trained_cascade(args, gate_path: Path, valence_path: Path) -> None:
    log("[cascade-val] evaluating trained two-stage cascade on master-index val split")
    device = resolve_device()
    keypoints, _meta = load_template_metadata(valence_path)

    eval_args = argparse.Namespace(**vars(args))
    eval_args.task = UNIFIED_TASK
    set_active_task_labels(UNIFIED_TASK)
    samples = discover_samples(eval_args, keypoints)
    val_samples = sorted([sample for sample in samples if sample.split == "val"], key=sample_sort_key)
    if not val_samples:
        raise RuntimeError("empty val split for cascade evaluation")

    gate_kind, gate_model, gate_id_to_label, gate_threshold, gate_max_frames = load_gate_checkpoint(gate_path, device)
    valence_model, valence_id_to_label, valence_max_frames = load_valence_transformer_checkpoint(valence_path, device)
    crops = parse_seconds_list(args.crop_seconds)

    prediction_rows: list[dict[str, str]] = []
    calibration_rows: list[dict[str, object]] = []
    for sample in tqdm(val_samples, desc="cascade val", unit="sample", ascii=True, mininterval=1.0, file=sys.stdout):
        if gate_kind == "pti":
            gate_pred, gate_probs = predict_pti_gate_sample(
                gate_model,
                sample,
                crops,
                device,
                args.bidirectional,
                gate_threshold,
                gate_max_frames,
            )
        else:
            gate_pred, gate_probs = predict_sample(gate_model, gate_id_to_label, sample, crops, device, args.bidirectional)

        gate_prob = interaction_prob_from_gate_probs(gate_probs, gate_id_to_label)
        valence_pred = ""
        valence_probs: list[float] = []
        valence_label = ""
        valence_conf = 0.0
        if gate_prob >= float(gate_threshold):
            valence_pred, valence_probs = predict_sample(
                valence_model,
                valence_id_to_label,
                sample,
                crops,
                device,
                args.bidirectional,
                max_frames=valence_max_frames,
            )
            valence_label, valence_conf = valence_label_conf_from_probs(valence_probs, valence_id_to_label)

        true = sample.raw_label or sample.valence
        calibration_rows.append(
            {
                "true": true,
                "gate_prob": gate_prob,
                "valence_label": valence_label,
                "valence_conf": valence_conf,
            }
        )
        prediction_rows.append(
            {
                "true": true,
                "pred": "",
                "gate_pred": gate_pred,
                "valence_pred": valence_pred,
                "source_name": sample.source_name,
                "index_row_id": sample.index_row_id,
                "clip_name": sample.clip_name,
                "clip_path": sample.clip_path,
                "bbox_subset": sample.bbox_subset,
                "split": sample.split,
                "folder": sample.folder,
                "manifest_path": sample.manifest_path,
                "source_video": sample.source_video,
                "gate_prob_interaction": f"{gate_prob:.6f}",
                "valence_conf": f"{valence_conf:.6f}" if valence_probs else "",
                "valence_conf_thresh": "",
                "gate_probs": ";".join(f"{v:.6f}" for v in gate_probs),
                "valence_probs": ";".join(f"{v:.6f}" for v in valence_probs),
            }
        )

    calibration = learn_dynamic_valence_threshold(
        calibration_rows,
        gate_threshold=float(gate_threshold),
        fixed_threshold=VALENCE_MIN_CONF,
    )
    for row, cal_row in zip(prediction_rows, calibration_rows):
        gate_prob = float(cal_row["gate_prob"])
        row["pred"] = predict_with_dynamic_threshold(
            cal_row,
            calibration,
            gate_threshold=float(gate_threshold),
            fixed_threshold=VALENCE_MIN_CONF,
        )
        if gate_prob >= float(gate_threshold):
            row["valence_conf_thresh"] = f"{dynamic_valence_conf_threshold(gate_prob, calibration, VALENCE_MIN_CONF):.6f}"

    save_dynamic_threshold_to_valence_checkpoint(valence_path, calibration)
    log(
        "[cascade-val] dynamic valence threshold "
        f"base={calibration['base_valence_conf']:.3f} min={calibration['min_valence_conf']:.3f} "
        f"exp={calibration['gate_exponent']:.3f} "
        f"baseline_macroF1={calibration.get('baseline_val_macro_f1', 0.0):.3f} "
        f"best_macroF1={calibration.get('best_val_macro_f1', 0.0):.3f}"
    )
    log(f"[cascade-val] wrote dynamic threshold metadata into {valence_path}")

    output_dir = gate_path.parent
    pred_path = output_dir / "stage2_cascade_val_predictions.csv"
    confusion_path = output_dir / "stage2_cascade_val_confusion_matrix.csv"
    by_source_path = output_dir / "stage2_cascade_val_confusion_by_source.csv"
    with pred_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "true",
            "pred",
            "gate_pred",
            "valence_pred",
            "source_name",
            "index_row_id",
            "clip_name",
            "clip_path",
            "bbox_subset",
            "split",
            "folder",
            "manifest_path",
            "source_video",
            "gate_prob_interaction",
            "valence_conf",
            "valence_conf_thresh",
            "gate_probs",
            "valence_probs",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(prediction_rows)
    write_overall_confusion(confusion_path, prediction_rows)
    write_source_confusions(by_source_path, prediction_rows)
    log_confusions(prediction_rows)
    log(f"[cascade-val] wrote {pred_path}")
    log(f"[cascade-val] wrote {confusion_path}")
    log(f"[cascade-val] wrote {by_source_path}")


def train_two_stage_cascade(args) -> list[Path]:
    if args.output:
        raise RuntimeError("--output is only valid for single-model tasks; two_stage_cascade writes the canonical gate and valence checkpoints.")

    outputs: list[Path] = []
    for task in (GATE_TASK, VALENCE_TASK):
        task_args = argparse.Namespace(**vars(args))
        task_args.task = task
        task_args.output = ""
        log(f"[cascade] training {task}")
        outputs.append(train(task_args))

    if not args.dry_run:
        evaluate_trained_cascade(args, outputs[0], outputs[1])

    log("[cascade] done: " + ", ".join(str(path) for path in outputs))
    return outputs


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the current Stage-2 interaction classifiers from generated CSVs.")
    parser.add_argument(
        "--stage2-index-csv",
        default=str(DEFAULT_STAGE2_INDEX_CSV),
        help="Authoritative Stage2 master index CSV. This CSV owns all train/val split membership.",
    )
    parser.add_argument(
        "--data-root",
        default=str(DEFAULT_DATA_ROOT),
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--v0520-root",
        default=str(DEFAULT_V0520_ROOT),
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--annotation-root",
        default=str(DEFAULT_ANNOTATION_ROOT),
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--selected-id-root", default=str(DEFAULT_SELECTED_ID_ROOT), help=argparse.SUPPRESS)
    parser.add_argument("--bbox-good-root", default=str(DEFAULT_BBOX_GOOD_ROOT), help=argparse.SUPPRESS)
    parser.add_argument("--bbox-goodtest-root", default=str(DEFAULT_BBOX_GOODTEST_ROOT), help=argparse.SUPPRESS)
    parser.add_argument("--bbox-bad-root", default=str(DEFAULT_BBOX_BAD_ROOT), help=argparse.SUPPRESS)
    parser.add_argument(
        "--task",
        choices=[CASCADE_TASK, GATE_TASK, VALENCE_TASK, UNIFIED_TASK],
        default=CASCADE_TASK,
        help="Default trains the two-stage cascade: interaction_gate_2class then valence_2class. Use interaction_3class explicitly for the unified baseline.",
    )
    parser.add_argument("--output", default="", help="Output checkpoint path for single-model tasks. The two_stage_cascade default writes canonical models_new checkpoints.")
    parser.add_argument("--template-ckpt", default=str(DEFAULT_TEMPLATE_CKPT), help="Optional ValenceTransformer checkpoint used for keypoint metadata.")
    parser.add_argument(
        "--eval-cascade-only",
        action="store_true",
        help="Evaluate existing two-stage cascade checkpoints on the master-index val split and exit without training.",
    )
    parser.add_argument(
        "--interaction-gate-ckpt",
        default=str(output_path_for_task(GATE_TASK)),
        help="Interaction gate checkpoint used with --eval-cascade-only.",
    )
    parser.add_argument(
        "--valence-ckpt",
        default=str(output_path_for_task(VALENCE_TASK)),
        help="Valence checkpoint used with --eval-cascade-only.",
    )
    parser.add_argument("--log-file", default="", help="Optional UTF-8 log file path.")
    parser.add_argument("--verify-log-flush", action="store_true", help="Write a few flushed log lines and exit.")
    parser.add_argument("--dry-run", action="store_true", help="Build dataset/split only; do not train.")
    parser.add_argument("--max-videos", type=int, default=0, help="Limit videos for smoke tests; 0 means all.")
    parser.add_argument("--samples-per-video", type=int, default=1, help="How many top candidate pairs to keep per source video.")
    parser.add_argument("--min-seq-frames", type=int, default=12, help="Minimum selected pair frames required for a sample.")
    parser.add_argument("--max-id-gap-sec", type=float, default=0.5, help="Maximum short dominant-bbox gap to repair by interpolation/freeze.")
    parser.add_argument("--max-pair-fill-frac", type=float, default=0.30, help="Maximum repaired-frame fraction for GOOD training clips.")
    parser.add_argument("--min-dominant-vote-ratio", type=float, default=0.60, help="Minimum main selected-ID pair vote ratio for GOOD clips.")
    parser.add_argument("--min-eval-dominant-vote-ratio", type=float, default=0.30, help="Minimum main selected-ID pair vote ratio for GOODTEST clips.")
    parser.add_argument("--fps", type=float, default=60.0, help="Fallback/model FPS.")
    parser.add_argument("--seed", type=int, default=20260515)
    parser.add_argument("--split-seed", type=int, default=20260515)
    parser.add_argument("--profile", choices=["auto", "8gb", "32gb"], default="auto")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=35)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no-amp", action="store_false", dest="amp")
    parser.add_argument("--bidirectional", action="store_true", default=True)
    parser.add_argument("--no-bidirectional", action="store_false", dest="bidirectional")
    parser.add_argument("--crop-seconds", default="1,2,4,full")
    parser.add_argument("--pti-max-frames", type=int, default=64)
    parser.add_argument("--pti-d-model", type=int, default=128)
    parser.add_argument("--pti-num-heads", type=int, default=8)
    parser.add_argument("--pti-temporal-layers", type=int, default=4)
    parser.add_argument("--pti-pair-layers", type=int, default=2)
    parser.add_argument("--pti-ffn-dim", type=int, default=512)
    parser.add_argument("--pti-dropout", type=float, default=0.1)
    parser.add_argument("--pti-causal", action="store_true", default=False)
    parser.add_argument("--valence-d-model", type=int, default=128)
    parser.add_argument("--valence-num-heads", type=int, default=8)
    parser.add_argument("--valence-temporal-layers", type=int, default=4)
    parser.add_argument("--valence-ffn-dim", type=int, default=512)
    parser.add_argument("--valence-dropout", type=float, default=0.1)
    parser.add_argument("--valence-causal", action="store_true", default=False)
    parser.add_argument("--heartbeat-sec", type=float, default=10.0)
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    log_path: Path | None
    if args.log_file:
        log_path = Path(args.log_file)
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        task_name = "stage2_cascade" if args.task == CASCADE_TASK else f"stage2_{args.task}"
        log_path = DEFAULT_LOG_DIR / f"train_{task_name}_{stamp}.log"

    global LOGGER
    LOGGER = TeeLogger(log_path)
    try:
        log(f"[info] log file: {log_path}")
        if args.verify_log_flush:
            for i in range(1, 4):
                log(f"[flush-check] tick={i} time={datetime.now().isoformat(timespec='seconds')}")
                time.sleep(1.0)
            log("[flush-check] ok")
            return 0
        if args.eval_cascade_only:
            if args.task != CASCADE_TASK:
                raise RuntimeError("--eval-cascade-only is only valid with --task two_stage_cascade")
            evaluate_trained_cascade(
                args,
                Path(args.interaction_gate_ckpt),
                Path(args.valence_ckpt),
            )
            return 0
        if args.task == CASCADE_TASK:
            train_two_stage_cascade(args)
        else:
            train(args)
        return 0
    finally:
        LOGGER.close()


if __name__ == "__main__":
    raise SystemExit(main())
