from __future__ import annotations

import argparse
import csv
import json
import math
import os
import queue
import re
import shutil
import sys
import threading
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


APP_VERSION = "0.1.0"
WORKSPACE_ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE_ROOT = Path(r"F:\DROPBOX\Isolated_Interaction_Clips\Interaction_Type")
DEFAULT_OUTPUT_ROOT = WORKSPACE_ROOT / "output"
DEFAULT_ASSET_ROOT = WORKSPACE_ROOT / "dcsna_local"
DEFAULT_DET_WEIGHTS = DEFAULT_ASSET_ROOT / "models" / "Object_Detection_Trained_Model.pt"
DEFAULT_LOG_FILE = WORKSPACE_ROOT / "logs" / "bbox_annotator.log"
DEFAULT_COW_EXCEL = Path(r"D:\OneDrive\F4_COOP\Directed_Interaction_0516_E.xlsx")

ANNOTATION_BY = "Yiwen Huang"
NO_NAME = "NO NAME"
COW_NAMES = ["Bella", "Daisy", "Rosie", "Buttercup", "Marigold"]
COW_OPTIONS = COW_NAMES + [NO_NAME]
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}

VRAM_PROFILES = {
    "8GB": {
        "imgsz": 960,
        "conf": 0.25,
        "iou": 0.50,
        "max_det": 24,
        "description": "Recommended for local 3070/8GB.",
    },
    "40GB": {
        "imgsz": 1280,
        "conf": 0.25,
        "iou": 0.50,
        "max_det": 40,
        "description": "Recommended for A100-40G or other large VRAM GPU.",
    },
}


def configure_utf8() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


class Logger:
    def __init__(self, log_file: Path):
        self.log_file = log_file
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def log(self, message: str) -> None:
        stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{stamp}] {message}"
        with self._lock:
            print(line, flush=True)
            with self.log_file.open("a", encoding="utf-8", newline="") as f:
                f.write(line + "\n")
                f.flush()


LOGGER: Optional[Logger] = None


def log(message: str) -> None:
    if LOGGER is not None:
        LOGGER.log(message)
    else:
        print(message, flush=True)


def commonpath_is_parent(child: Path, parent: Path) -> bool:
    child_resolved = child.resolve(strict=False)
    parent_resolved = parent.resolve(strict=False)
    try:
        common = os.path.commonpath([str(child_resolved), str(parent_resolved)])
    except ValueError:
        return False
    return os.path.normcase(common) == os.path.normcase(str(parent_resolved))


def safe_delete_path(path: Path, allowed_parent: Path) -> None:
    resolved = path.resolve(strict=False)
    allowed = allowed_parent.resolve(strict=False)
    if not commonpath_is_parent(resolved, allowed):
        raise RuntimeError(f"Refusing to delete outside allowed parent: {resolved}")
    if os.path.normcase(str(resolved)) == os.path.normcase(str(allowed)):
        raise RuntimeError(f"Refusing to delete allowed parent itself: {resolved}")
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()
    log(f"deleted: {resolved}")


def clear_output_root(output_root: Path) -> None:
    output_resolved = output_root.resolve(strict=False)
    workspace_resolved = WORKSPACE_ROOT.resolve(strict=False)
    if output_resolved.name.lower() != "output":
        raise RuntimeError(f"Refusing to clear non-output directory: {output_resolved}")
    if not commonpath_is_parent(output_resolved, workspace_resolved):
        raise RuntimeError(f"Refusing to clear output outside workspace: {output_resolved}")
    output_root.mkdir(parents=True, exist_ok=True)
    for child in list(output_root.iterdir()):
        safe_delete_path(child, output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    log(f"cleared output root: {output_resolved}")


def has_any_metadata(output_root: Path) -> bool:
    if not output_root.exists():
        return False
    return any(output_root.rglob("metadata.json"))


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def path_for_json(path: Path) -> str:
    return str(path.resolve(strict=False))


def normalize_interaction_number(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = float(text)
        if parsed.is_integer():
            return str(int(parsed))
    except ValueError:
        pass
    match = re.search(r"\d+", text)
    if not match:
        return None
    return str(int(match.group(0)))


def video_filter_key(rel_path: Path) -> Optional[Tuple[str, str]]:
    if not rel_path.parts:
        return None
    interaction_number = normalize_interaction_number(rel_path.stem)
    if interaction_number is None:
        return None
    category = rel_path.parts[0].strip()
    if not category:
        return None
    return category.lower(), interaction_number


def display_label_for_cow(cow_name: str) -> str:
    return cow_name or NO_NAME


def cow_name_from_display(display_value: str, allowed_names: Sequence[str]) -> str:
    value = (display_value or "").strip()
    if value == NO_NAME:
        return NO_NAME
    allowed = set(allowed_names)
    if value in allowed:
        return value
    for cow_name in allowed_names:
        if value == display_label_for_cow(cow_name):
            return cow_name
    return NO_NAME


def cow_names_in_cells(values: Sequence[Any]) -> List[str]:
    found: List[str] = []
    for value in values:
        if value is None:
            continue
        text = str(value)
        for cow_name in COW_NAMES:
            pattern = rf"(?<![A-Za-z]){re.escape(cow_name)}(?![A-Za-z])"
            if re.search(pattern, text, flags=re.IGNORECASE) and cow_name not in found:
                found.append(cow_name)
    return found


def add_cow_filter_row(
    filters: Dict[Tuple[str, str], List[str]],
    values: Sequence[Any],
) -> None:
    if len(values) < 2:
        return
    category = "" if values[0] is None else str(values[0]).strip()
    interaction_number = normalize_interaction_number(values[1])
    if not category or interaction_number is None:
        return
    names = cow_names_in_cells(values[2:])
    if names:
        filters.setdefault((category.lower(), interaction_number), names)


def load_cow_name_filters_openpyxl(excel_path: Path) -> Dict[Tuple[str, str], List[str]]:
    import openpyxl

    filters: Dict[Tuple[str, str], List[str]] = {}
    workbook = openpyxl.load_workbook(excel_path, read_only=True, data_only=True)
    try:
        for sheet in workbook.worksheets:
            for row in sheet.iter_rows(values_only=True):
                add_cow_filter_row(filters, list(row))
    finally:
        workbook.close()
    return filters


def xlsx_column_index(cell_ref: str) -> int:
    letters = "".join(ch for ch in cell_ref if ch.isalpha()).upper()
    index = 0
    for ch in letters:
        index = index * 26 + (ord(ch) - ord("A") + 1)
    return max(0, index - 1)


def xlsx_text_from_cell(cell: Any, shared_strings: Sequence[str]) -> Optional[str]:
    cell_type = cell.attrib.get("t", "")
    if cell_type == "inlineStr":
        texts = [node.text or "" for node in cell.findall(".//{*}t")]
        return "".join(texts)
    value_node = cell.find("{*}v")
    if value_node is None or value_node.text is None:
        return None
    text = value_node.text
    if cell_type == "s":
        try:
            return shared_strings[int(text)]
        except (ValueError, IndexError):
            return text
    return text


def xlsx_shared_strings(zf: Any) -> List[str]:
    import xml.etree.ElementTree as ET

    if "xl/sharedStrings.xml" not in zf.namelist():
        return []
    root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
    values: List[str] = []
    for item in root.findall("{*}si"):
        texts = [node.text or "" for node in item.findall(".//{*}t")]
        values.append("".join(texts))
    return values


def xlsx_sheet_paths(zf: Any) -> List[str]:
    import xml.etree.ElementTree as ET

    names = set(zf.namelist())
    if "xl/workbook.xml" not in names or "xl/_rels/workbook.xml.rels" not in names:
        return sorted(name for name in names if name.startswith("xl/worksheets/") and name.endswith(".xml"))

    rels_root = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
    rel_targets: Dict[str, str] = {}
    for rel in rels_root.findall("{*}Relationship"):
        rel_id = rel.attrib.get("Id")
        target = rel.attrib.get("Target")
        if not rel_id or not target:
            continue
        if target.startswith("/"):
            path = target.lstrip("/")
        elif target.startswith("xl/"):
            path = target
        else:
            path = f"xl/{target}"
        rel_targets[rel_id] = path

    workbook_root = ET.fromstring(zf.read("xl/workbook.xml"))
    paths: List[str] = []
    for sheet in workbook_root.findall(".//{*}sheet"):
        rel_id = ""
        for attr_name, attr_value in sheet.attrib.items():
            if attr_name.endswith("}id") or attr_name == "id":
                rel_id = attr_value
                break
        path = rel_targets.get(rel_id)
        if path in names:
            paths.append(path)
    return paths or sorted(name for name in names if name.startswith("xl/worksheets/") and name.endswith(".xml"))


def load_cow_name_filters_xlsx_xml(excel_path: Path) -> Dict[Tuple[str, str], List[str]]:
    import zipfile
    import xml.etree.ElementTree as ET

    filters: Dict[Tuple[str, str], List[str]] = {}
    with zipfile.ZipFile(excel_path) as zf:
        shared_strings = xlsx_shared_strings(zf)
        for sheet_path in xlsx_sheet_paths(zf):
            root = ET.fromstring(zf.read(sheet_path))
            for row in root.findall(".//{*}row"):
                values: List[Optional[str]] = []
                for cell in row.findall("{*}c"):
                    cell_ref = cell.attrib.get("r", "")
                    column_index = xlsx_column_index(cell_ref)
                    while len(values) <= column_index:
                        values.append(None)
                    values[column_index] = xlsx_text_from_cell(cell, shared_strings)
                add_cow_filter_row(filters, values)
    return filters


def load_cow_name_filters(excel_path: Path) -> Dict[Tuple[str, str], List[str]]:
    if not excel_path.exists():
        log(f"cow-name Excel not found; using all cow names: {excel_path}")
        return {}
    try:
        filters = load_cow_name_filters_openpyxl(excel_path)
        log(f"loaded {len(filters)} cow-name filter row(s) from {excel_path} via openpyxl")
        return filters
    except Exception as exc:
        log(f"openpyxl Excel read unavailable; trying stdlib xlsx reader: {exc}")

    try:
        filters = load_cow_name_filters_xlsx_xml(excel_path)
        log(f"loaded {len(filters)} cow-name filter row(s) from {excel_path} via stdlib xlsx reader")
        return filters
    except Exception as exc:
        log(f"Excel read failed; using all cow names: {exc}")
        return {}


@dataclass
class VideoEntry:
    path: Path
    rel_path: Path
    out_dir: Path

    @property
    def display_name(self) -> str:
        return self.rel_path.as_posix()

    @property
    def metadata_path(self) -> Path:
        return self.out_dir / "metadata.json"

    @property
    def csv_path(self) -> Path:
        return self.out_dir / "bbox.csv"

    @property
    def vis_path(self) -> Path:
        return self.out_dir / "vis.png"


@dataclass
class BBox:
    box_id: int
    x: float
    y: float
    w: float
    h: float
    cow_name: str = NO_NAME
    score: Optional[float] = None
    source: str = "manual"

    @property
    def x2(self) -> float:
        return self.x + self.w

    @property
    def y2(self) -> float:
        return self.y + self.h

    def as_xyxy(self) -> Tuple[float, float, float, float]:
        return self.x, self.y, self.x2, self.y2

    def set_from_xyxy(self, x1: float, y1: float, x2: float, y2: float) -> None:
        left, right = sorted((float(x1), float(x2)))
        top, bottom = sorted((float(y1), float(y2)))
        self.x = left
        self.y = top
        self.w = max(1.0, right - left)
        self.h = max(1.0, bottom - top)

    def clamp(self, width: int, height: int) -> None:
        x1, y1, x2, y2 = self.as_xyxy()
        x1 = min(max(0.0, x1), max(0.0, width - 1.0))
        y1 = min(max(0.0, y1), max(0.0, height - 1.0))
        x2 = min(max(0.0, x2), float(width))
        y2 = min(max(0.0, y2), float(height))
        if x2 <= x1:
            x2 = min(float(width), x1 + 1.0)
        if y2 <= y1:
            y2 = min(float(height), y1 + 1.0)
        self.set_from_xyxy(x1, y1, x2, y2)


def discover_videos(source_root: Path, output_root: Path) -> List[VideoEntry]:
    if not source_root.exists():
        raise FileNotFoundError(f"Missing source root: {source_root}")
    entries: List[VideoEntry] = []
    for path in source_root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in VIDEO_EXTS:
            continue
        rel_text = os.path.relpath(str(path), str(source_root))
        rel_path = Path(rel_text)
        out_dir = output_root / rel_path.with_suffix("")
        entries.append(VideoEntry(path=path, rel_path=rel_path, out_dir=out_dir))
    entries.sort(key=lambda item: item.rel_path.as_posix().lower())
    return entries


def read_first_frame(video_path: Path) -> Tuple[Any, Dict[str, Any]]:
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")
        ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError(f"Could not read first frame: {video_path}")
        height, width = frame.shape[:2]
        meta = {
            "width": int(width),
            "height": int(height),
            "fps": float(cap.get(cv2.CAP_PROP_FPS) or 0.0),
            "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
            "frame_index": 0,
        }
        return frame, meta
    finally:
        cap.release()


def save_png(path: Path, image_bgr: Any) -> None:
    import cv2

    ok, encoded = cv2.imencode(".png", image_bgr)
    if not ok:
        raise RuntimeError(f"Failed to encode PNG: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded.tofile(str(path))


def box_color(index: int) -> Tuple[int, int, int]:
    palette = [
        (44, 160, 44),
        (31, 119, 180),
        (255, 127, 14),
        (214, 39, 40),
        (148, 103, 189),
        (23, 190, 207),
        (188, 189, 34),
    ]
    return palette[index % len(palette)]


def draw_visualization(frame_bgr: Any, boxes: Sequence[BBox]) -> Any:
    import cv2

    canvas = frame_bgr.copy()
    height, width = canvas.shape[:2]
    font_scale = max(0.55, min(width, height) / 1400.0)
    thickness = max(2, int(round(min(width, height) / 800.0)))
    text_thickness = max(1, thickness - 1)

    for idx, box in enumerate(boxes):
        box = BBox(
            box_id=box.box_id,
            x=box.x,
            y=box.y,
            w=box.w,
            h=box.h,
            cow_name=box.cow_name,
            score=box.score,
            source=box.source,
        )
        box.clamp(width, height)
        x1, y1, x2, y2 = [int(round(v)) for v in box.as_xyxy()]
        color = box_color(idx)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)
        label = f"{idx + 1}: {box.cow_name or NO_NAME}"
        if box.score is not None:
            label += f" {box.score:.2f}"
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness)
        label_y = max(0, y1 - th - baseline - 4)
        cv2.rectangle(canvas, (x1, label_y), (x1 + tw + 8, label_y + th + baseline + 8), color, -1)
        cv2.putText(
            canvas,
            label,
            (x1 + 4, label_y + th + 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            text_thickness,
            cv2.LINE_AA,
        )
    return canvas


def load_saved_boxes(entry: VideoEntry) -> List[BBox]:
    if not entry.metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata: {entry.metadata_path}")
    if not entry.csv_path.exists():
        raise FileNotFoundError(f"Missing bbox CSV: {entry.csv_path}")
    with entry.metadata_path.open("r", encoding="utf-8") as f:
        json.load(f)
    boxes: List[BBox] = []
    with entry.csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or [])
        required = {"x", "y", "w", "h"}
        if not required.issubset(fieldnames):
            raise ValueError(f"bbox.csv missing required columns: {sorted(required - fieldnames)}")
        for idx, row in enumerate(reader, start=1):
            x = float(row["x"])
            y = float(row["y"])
            w = float(row["w"])
            h = float(row["h"])
            if not all(math.isfinite(v) for v in (x, y, w, h)):
                raise ValueError("bbox.csv contains non-finite coordinates")
            if w <= 0 or h <= 0:
                raise ValueError("bbox.csv contains non-positive width/height")
            cow_name = (row.get("cow_name") or NO_NAME).strip() or NO_NAME
            if cow_name not in COW_OPTIONS:
                cow_name = NO_NAME
            score_text = (row.get("score") or "").strip()
            score = float(score_text) if score_text else None
            box_id = int(row.get("bbox_id") or idx)
            source = (row.get("source") or "loaded").strip() or "loaded"
            boxes.append(BBox(box_id=box_id, x=x, y=y, w=w, h=h, cow_name=cow_name, score=score, source=source))
    return boxes


def write_annotation(
    entry: VideoEntry,
    source_root: Path,
    output_root: Path,
    frame_bgr: Any,
    frame_meta: Dict[str, Any],
    boxes: Sequence[BBox],
    detector_info: Dict[str, Any],
    cow_name_options: Sequence[str],
) -> None:
    entry.out_dir.mkdir(parents=True, exist_ok=True)
    height, width = frame_bgr.shape[:2]

    fieldnames = [
        "video",
        "file_name",
        "relative_video",
        "frame",
        "bbox_id",
        "cow_name",
        "x",
        "y",
        "w",
        "h",
        "x2",
        "y2",
        "score",
        "source",
    ]
    with entry.csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for box in boxes:
            box.clamp(width, height)
            writer.writerow(
                {
                    "video": path_for_json(entry.path),
                    "file_name": entry.path.name,
                    "relative_video": entry.rel_path.as_posix(),
                    "frame": 0,
                    "bbox_id": box.box_id,
                    "cow_name": box.cow_name or NO_NAME,
                    "x": f"{box.x:.2f}",
                    "y": f"{box.y:.2f}",
                    "w": f"{box.w:.2f}",
                    "h": f"{box.h:.2f}",
                    "x2": f"{box.x2:.2f}",
                    "y2": f"{box.y2:.2f}",
                    "score": "" if box.score is None else f"{box.score:.6f}",
                    "source": box.source,
                }
            )

    vis = draw_visualization(frame_bgr, boxes)
    save_png(entry.vis_path, vis)

    metadata = {
        "app": "bbox_annotator.py",
        "app_version": APP_VERSION,
        "annotation_by": ANNOTATION_BY,
        "annotated_at": now_iso(),
        "source_root": path_for_json(source_root),
        "output_root": path_for_json(output_root),
        "video_path": path_for_json(entry.path),
        "file_name": entry.path.name,
        "relative_video": entry.rel_path.as_posix(),
        "frame_index": 0,
        "original_width": int(frame_meta.get("width") or width),
        "original_height": int(frame_meta.get("height") or height),
        "fps": float(frame_meta.get("fps") or 0.0),
        "frame_count": int(frame_meta.get("frame_count") or 0),
        "box_count": len(boxes),
        "cow_name_options": list(cow_name_options),
        "bbox_csv": path_for_json(entry.csv_path),
        "visualization": path_for_json(entry.vis_path),
        "detector": detector_info,
        "coordinate_system": {
            "origin": "top-left",
            "units": "original input frame pixels",
            "frame_index_base": "0-based",
            "format": "x,y,w,h with x2=x+w and y2=y+h",
        },
    }
    with entry.metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
        f.write("\n")


class DetectionRunner:
    def __init__(self, weights_path: Path):
        self.weights_path = weights_path
        self.model: Any = None
        self.device: Any = None
        self.cuda_available = False
        self._lock = threading.Lock()

    def ensure_model(self) -> None:
        if self.model is not None:
            return
        if not self.weights_path.exists():
            raise FileNotFoundError(f"Missing detector weights: {self.weights_path}")

        os.environ.setdefault("YOLO_CONFIG_DIR", str(WORKSPACE_ROOT / ".ultralytics"))
        from ultralytics import YOLO

        try:
            import torch

            self.cuda_available = bool(torch.cuda.is_available())
        except Exception:
            self.cuda_available = False

        self.device = 0 if self.cuda_available else "cpu"
        log(f"loading detector: {self.weights_path}")
        log(f"detector device: {'cuda:0' if self.cuda_available else 'cpu'}")
        self.model = YOLO(str(self.weights_path))

    def detect(self, frame_bgr: Any, vram_profile: str) -> List[BBox]:
        profile = VRAM_PROFILES.get(vram_profile, VRAM_PROFILES["8GB"])
        with self._lock:
            self.ensure_model()
            results = self.model.predict(
                source=frame_bgr,
                imgsz=int(profile["imgsz"]),
                conf=float(profile["conf"]),
                iou=float(profile["iou"]),
                max_det=int(profile["max_det"]),
                device=self.device,
                half=bool(self.cuda_available),
                verbose=False,
            )

        if not results:
            return []
        result = results[0]
        boxes_obj = getattr(result, "boxes", None)
        if boxes_obj is None or len(boxes_obj) == 0:
            return []

        xyxy = boxes_obj.xyxy.cpu().numpy().astype(float)
        conf = boxes_obj.conf.cpu().numpy().astype(float)
        height, width = frame_bgr.shape[:2]
        boxes: List[BBox] = []
        for idx, (coords, score) in enumerate(zip(xyxy, conf), start=1):
            x1, y1, x2, y2 = [float(v) for v in coords[:4]]
            if x2 <= x1 or y2 <= y1:
                continue
            box = BBox(
                box_id=idx,
                x=x1,
                y=y1,
                w=x2 - x1,
                h=y2 - y1,
                cow_name=NO_NAME,
                score=float(score),
                source="detector",
            )
            box.clamp(width, height)
            boxes.append(box)
        return boxes

    def info(self, vram_profile: str) -> Dict[str, Any]:
        profile = VRAM_PROFILES.get(vram_profile, VRAM_PROFILES["8GB"])
        return {
            "source": "DCSNA local copy",
            "weights": path_for_json(self.weights_path),
            "requirements": path_for_json(DEFAULT_ASSET_ROOT / "requirements_windows.txt"),
            "vram_profile": vram_profile,
            "recommended_profiles": VRAM_PROFILES,
            "device": "cuda:0" if self.cuda_available else "cpu",
            "imgsz": int(profile["imgsz"]),
            "conf": float(profile["conf"]),
            "iou": float(profile["iou"]),
            "max_det": int(profile["max_det"]),
        }


class BBoxAnnotatorApp:
    def __init__(
        self,
        root: Any,
        source_root: Path,
        output_root: Path,
        detector: DetectionRunner,
        default_vram_profile: str,
        cow_name_filters: Dict[Tuple[str, str], List[str]],
    ):
        import tkinter as tk
        from tkinter import ttk

        self.tk = tk
        self.ttk = ttk
        self.root = root
        self.source_root = source_root
        self.output_root = output_root
        self.detector = detector
        self.cow_name_filters = cow_name_filters

        self.videos: List[VideoEntry] = []
        self.current_index: Optional[int] = None
        self.current_frame: Any = None
        self.current_meta: Dict[str, Any] = {}
        self.boxes: List[BBox] = []
        self.next_box_id = 1
        self.selected_box_id: Optional[int] = None

        self.display_scale = 1.0
        self.display_offset = (0.0, 0.0)
        self.tk_image: Any = None
        self.pil_image_module: Any = None
        self.pil_imagetk_module: Any = None

        self.add_mode = False
        self.drag_mode: Optional[str] = None
        self.drag_box_id: Optional[int] = None
        self.drag_start_img: Optional[Tuple[float, float]] = None
        self.drag_orig_xyxy: Optional[Tuple[float, float, float, float]] = None
        self.temp_add_start: Optional[Tuple[float, float]] = None
        self.temp_add_end: Optional[Tuple[float, float]] = None

        self.detect_queue: "queue.Queue[Tuple[int, Optional[List[BBox]], Optional[str]]]" = queue.Queue()
        self.detect_token = 0
        self.detecting = False

        self.status_var = tk.StringVar(value="Starting...")
        self.vram_profile_var = tk.StringVar(value=default_vram_profile)
        self.add_button: Any = None
        self.detect_button: Any = None
        self.video_listbox: Any = None
        self.canvas: Any = None
        self.box_rows_canvas: Any = None
        self.box_rows_frame: Any = None

        self._build_ui()
        self.root.after(100, self._post_init)
        self.root.after(100, self._poll_detection_queue)

    def _build_ui(self) -> None:
        tk = self.tk
        ttk = self.ttk

        self.root.title("Cow bbox annotator")
        self.root.geometry("1550x920")
        self.root.minsize(1100, 700)
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        toolbar = ttk.Frame(self.root, padding=(8, 6))
        toolbar.grid(row=0, column=0, sticky="ew")
        toolbar.columnconfigure(99, weight=1)

        ttk.Button(toolbar, text="Save", command=self.save_current).grid(row=0, column=0, padx=3)
        self.add_button = ttk.Button(toolbar, text="Add bbox", command=self.toggle_add_mode)
        self.add_button.grid(row=0, column=1, padx=3)
        ttk.Button(toolbar, text="Delete selected", command=self.delete_selected_box).grid(row=0, column=2, padx=3)
        ttk.Button(toolbar, text="Delete all bbox", command=self.delete_all_boxes).grid(row=0, column=3, padx=3)
        self.detect_button = ttk.Button(toolbar, text="Re-detect", command=lambda: self.start_detection(force=True))
        self.detect_button.grid(row=0, column=4, padx=3)
        ttk.Label(toolbar, text="VRAM").grid(row=0, column=5, padx=(14, 3))
        vram_combo = ttk.Combobox(
            toolbar,
            textvariable=self.vram_profile_var,
            values=list(VRAM_PROFILES.keys()),
            width=6,
            state="readonly",
        )
        vram_combo.grid(row=0, column=6, padx=3)
        ttk.Label(toolbar, textvariable=self.status_var).grid(row=0, column=99, padx=(18, 3), sticky="ew")

        panes = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        panes.grid(row=1, column=0, sticky="nsew")

        left = ttk.Frame(panes, padding=(6, 6))
        center = ttk.Frame(panes, padding=(4, 4))
        right = ttk.Frame(panes, padding=(6, 6))
        panes.add(left, weight=1)
        panes.add(center, weight=5)
        panes.add(right, weight=2)

        left.rowconfigure(1, weight=1)
        left.columnconfigure(0, weight=1)
        ttk.Label(left, text="Videos").grid(row=0, column=0, sticky="w")
        video_scroll = ttk.Scrollbar(left, orient=tk.VERTICAL)
        self.video_listbox = tk.Listbox(left, exportselection=False, yscrollcommand=video_scroll.set, width=42)
        video_scroll.config(command=self.video_listbox.yview)
        self.video_listbox.grid(row=1, column=0, sticky="nsew")
        video_scroll.grid(row=1, column=1, sticky="ns")
        self.video_listbox.bind("<<ListboxSelect>>", self.on_video_selected)

        center.rowconfigure(0, weight=1)
        center.columnconfigure(0, weight=1)
        self.canvas = tk.Canvas(center, bg="#202124", highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.canvas.bind("<Configure>", lambda _event: self.redraw_canvas())
        self.canvas.bind("<ButtonPress-1>", self.on_canvas_press)
        self.canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_canvas_release)

        right.rowconfigure(1, weight=1)
        right.columnconfigure(0, weight=1)
        ttk.Label(right, text="Bboxes / cow name").grid(row=0, column=0, sticky="w")
        row_scroll = ttk.Scrollbar(right, orient=tk.VERTICAL)
        self.box_rows_canvas = tk.Canvas(right, highlightthickness=0, yscrollcommand=row_scroll.set)
        row_scroll.config(command=self.box_rows_canvas.yview)
        self.box_rows_canvas.grid(row=1, column=0, sticky="nsew")
        row_scroll.grid(row=1, column=1, sticky="ns")
        self.box_rows_frame = ttk.Frame(self.box_rows_canvas)
        self.box_rows_canvas.create_window((0, 0), window=self.box_rows_frame, anchor="nw")
        self.box_rows_frame.bind(
            "<Configure>",
            lambda _event: self.box_rows_canvas.configure(scrollregion=self.box_rows_canvas.bbox("all")),
        )

    def _post_init(self) -> None:
        from tkinter import messagebox

        try:
            self.videos = discover_videos(self.source_root, self.output_root)
        except Exception as exc:
            messagebox.showerror("Video discovery failed", str(exc))
            self.set_status(f"Video discovery failed: {exc}")
            return

        try:
            if has_any_metadata(self.output_root):
                should_clear = messagebox.askyesno(
                    "Existing annotations found",
                    "Existing metadata.json files were found under ./output.\n\nClear ./output now?",
                )
                if should_clear:
                    clear_output_root(self.output_root)
                else:
                    log("kept existing output annotations")
            else:
                clear_output_root(self.output_root)
        except Exception as exc:
            messagebox.showerror("Output initialization failed", str(exc))
            self.set_status(f"Output initialization failed: {exc}")
            return

        self.refresh_video_list()
        self.set_status(f"Found {len(self.videos)} video(s)")
        if self.videos:
            self.video_listbox.selection_set(0)
            self.video_listbox.activate(0)
            self.load_video(0)

    def set_status(self, message: str) -> None:
        self.status_var.set(message)
        log(message)

    def cow_names_for_entry(self, entry: VideoEntry) -> List[str]:
        key = video_filter_key(entry.rel_path)
        names = self.cow_name_filters.get(key) if key is not None else None
        if not names:
            return list(COW_NAMES)
        filtered = [name for name in names if name in COW_NAMES]
        return filtered or list(COW_NAMES)

    def cow_options_for_entry(self, entry: VideoEntry) -> List[str]:
        return self.cow_names_for_entry(entry) + [NO_NAME]

    def current_cow_options(self) -> List[str]:
        if self.current_index is None:
            return list(COW_OPTIONS)
        return self.cow_options_for_entry(self.videos[self.current_index])

    def current_cow_display_options(self) -> List[str]:
        options = self.current_cow_options()
        return [display_label_for_cow(name) for name in options]

    def display_value_for_current_name(self, cow_name: str) -> str:
        options = self.current_cow_options()
        if cow_name not in options:
            return NO_NAME
        return display_label_for_cow(cow_name)

    def cow_name_from_current_display(self, display_value: str) -> str:
        options = self.current_cow_options()
        return cow_name_from_display(display_value, options)

    def sanitize_boxes_for_current_options(self) -> None:
        options = set(self.current_cow_options())
        for box in self.boxes:
            if box.cow_name not in options:
                box.cow_name = NO_NAME

    def refresh_video_list(self) -> None:
        self.video_listbox.delete(0, self.tk.END)
        for idx, entry in enumerate(self.videos):
            self.video_listbox.insert(self.tk.END, entry.display_name)
            color = "red" if entry.metadata_path.exists() else "black"
            self.video_listbox.itemconfig(idx, fg=color)

    def on_video_selected(self, _event: Any) -> None:
        selection = self.video_listbox.curselection()
        if not selection:
            return
        index = int(selection[0])
        if index == self.current_index:
            return
        self.load_video(index)

    def load_video(self, index: int) -> None:
        from tkinter import messagebox

        self.detect_token += 1
        self.current_index = index
        entry = self.videos[index]
        self.add_mode = False
        self.update_add_button()
        self.boxes = []
        self.selected_box_id = None
        self.current_frame = None
        self.current_meta = {}
        self.rebuild_box_rows()
        self.redraw_canvas()

        try:
            frame, meta = read_first_frame(entry.path)
            self.current_frame = frame
            self.current_meta = meta
            self.set_status(f"Loaded first frame: {entry.display_name}")
        except Exception as exc:
            messagebox.showerror("Read video failed", str(exc))
            self.set_status(f"Read video failed: {exc}")
            return

        if entry.metadata_path.exists():
            try:
                self.boxes = load_saved_boxes(entry)
                self.sanitize_boxes_for_current_options()
                self.next_box_id = max([box.box_id for box in self.boxes] + [0]) + 1
                self.set_status(f"Loaded saved annotation: {entry.display_name}")
            except Exception as exc:
                self.boxes = []
                self.next_box_id = 1
                self.selected_box_id = None
                self.redraw_canvas()
                self.rebuild_box_rows()
                try:
                    safe_delete_path(entry.out_dir, self.output_root)
                    self.refresh_video_list()
                except Exception as delete_exc:
                    messagebox.showerror(
                        "Saved annotation was invalid",
                        f"Could not read saved annotation:\n{exc}\n\nAlso failed to delete it:\n{delete_exc}",
                    )
                    self.set_status(f"Invalid saved annotation; delete failed: {delete_exc}")
                    return
                messagebox.showwarning(
                    "Saved annotation was invalid",
                    f"Could not read saved annotation, so this video's annotation directory was deleted:\n{exc}",
                )
                self.set_status(f"Invalid saved annotation deleted: {entry.display_name}")
                return
        else:
            self.start_detection(force=False)

        self.redraw_canvas()
        self.rebuild_box_rows()

    def start_detection(self, force: bool) -> None:
        if self.current_index is None or self.current_frame is None:
            return
        if self.detecting:
            self.set_status("Detection is already running")
            return

        entry = self.videos[self.current_index]
        if entry.metadata_path.exists() and not force:
            return

        self.detecting = True
        self.detect_token += 1
        token = self.detect_token
        frame_for_worker = self.current_frame.copy()
        profile = self.vram_profile_var.get()
        self.detect_button.configure(state=self.tk.DISABLED)
        self.set_status(f"Running detector ({profile}) on {entry.display_name}")

        def worker() -> None:
            try:
                boxes = self.detector.detect(frame_for_worker, profile)
                self.detect_queue.put((token, boxes, None))
            except Exception:
                self.detect_queue.put((token, None, traceback.format_exc()))

        threading.Thread(target=worker, daemon=True).start()

    def _poll_detection_queue(self) -> None:
        try:
            while True:
                token, boxes, error = self.detect_queue.get_nowait()
                if token != self.detect_token:
                    self.detecting = False
                    self.detect_button.configure(state=self.tk.NORMAL)
                    continue
                self.detecting = False
                self.detect_button.configure(state=self.tk.NORMAL)
                if error:
                    self.boxes = []
                    self.next_box_id = 1
                    self.selected_box_id = None
                    self.rebuild_box_rows()
                    self.redraw_canvas()
                    self.set_status("Detection failed; manual annotation is still available")
                    from tkinter import messagebox

                    messagebox.showerror("Detection failed", error)
                else:
                    self.boxes = boxes or []
                    self.sanitize_boxes_for_current_options()
                    self.next_box_id = max([box.box_id for box in self.boxes] + [0]) + 1
                    self.selected_box_id = self.boxes[0].box_id if self.boxes else None
                    self.rebuild_box_rows()
                    self.redraw_canvas()
                    self.set_status(f"Detection finished: {len(self.boxes)} bbox(es)")
        except queue.Empty:
            pass
        self.root.after(100, self._poll_detection_queue)

    def ensure_pillow(self) -> None:
        if self.pil_image_module is None:
            from PIL import Image, ImageTk

            self.pil_image_module = Image
            self.pil_imagetk_module = ImageTk

    def redraw_canvas(self) -> None:
        self.canvas.delete("all")
        width = max(1, int(self.canvas.winfo_width()))
        height = max(1, int(self.canvas.winfo_height()))

        if self.current_frame is None:
            self.canvas.create_text(width / 2, height / 2, fill="#d0d0d0", text="No frame loaded")
            return

        self.ensure_pillow()
        frame_rgb = self.current_frame[:, :, ::-1]
        image = self.pil_image_module.fromarray(frame_rgb)
        img_w, img_h = image.size
        scale = min(width / img_w, height / img_h)
        scale = max(0.01, scale)
        disp_w = max(1, int(round(img_w * scale)))
        disp_h = max(1, int(round(img_h * scale)))
        resized = image.resize((disp_w, disp_h), self.pil_image_module.Resampling.LANCZOS)
        self.tk_image = self.pil_imagetk_module.PhotoImage(resized)
        offset_x = (width - disp_w) / 2.0
        offset_y = (height - disp_h) / 2.0
        self.display_scale = scale
        self.display_offset = (offset_x, offset_y)
        self.canvas.create_image(offset_x, offset_y, image=self.tk_image, anchor="nw")

        for idx, box in enumerate(self.boxes):
            color = "#ffdd33" if box.box_id == self.selected_box_id else "#24c6dc"
            x1, y1 = self.image_to_canvas(box.x, box.y)
            x2, y2 = self.image_to_canvas(box.x2, box.y2)
            self.canvas.create_rectangle(x1, y1, x2, y2, outline=color, width=3)
            label = f"{idx + 1}: {box.cow_name or NO_NAME}"
            self.canvas.create_rectangle(x1, max(offset_y, y1 - 22), x1 + 10 + len(label) * 7, y1, fill=color, outline=color)
            self.canvas.create_text(x1 + 5, y1 - 11, text=label, fill="#111111", anchor="w")
            if box.box_id == self.selected_box_id:
                self.draw_handles(x1, y1, x2, y2)

        if self.temp_add_start and self.temp_add_end:
            x1, y1 = self.image_to_canvas(*self.temp_add_start)
            x2, y2 = self.image_to_canvas(*self.temp_add_end)
            self.canvas.create_rectangle(x1, y1, x2, y2, outline="#f6f6f6", width=2, dash=(4, 3))

    def draw_handles(self, x1: float, y1: float, x2: float, y2: float) -> None:
        for hx, hy in [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]:
            self.canvas.create_rectangle(hx - 5, hy - 5, hx + 5, hy + 5, fill="#ffdd33", outline="#111111")

    def image_to_canvas(self, x: float, y: float) -> Tuple[float, float]:
        ox, oy = self.display_offset
        return ox + x * self.display_scale, oy + y * self.display_scale

    def canvas_to_image(self, cx: float, cy: float) -> Tuple[float, float]:
        ox, oy = self.display_offset
        return (cx - ox) / self.display_scale, (cy - oy) / self.display_scale

    def clamp_point_to_image(self, x: float, y: float) -> Tuple[float, float]:
        if self.current_frame is None:
            return x, y
        height, width = self.current_frame.shape[:2]
        return min(max(0.0, x), float(width)), min(max(0.0, y), float(height))

    def hit_test(self, cx: float, cy: float) -> Tuple[Optional[int], Optional[str]]:
        handle_radius = 10
        for box in reversed(self.boxes):
            x1, y1 = self.image_to_canvas(box.x, box.y)
            x2, y2 = self.image_to_canvas(box.x2, box.y2)
            handles = {
                "resize_nw": (x1, y1),
                "resize_ne": (x2, y1),
                "resize_sw": (x1, y2),
                "resize_se": (x2, y2),
            }
            for mode, (hx, hy) in handles.items():
                if abs(cx - hx) <= handle_radius and abs(cy - hy) <= handle_radius:
                    return box.box_id, mode
            if min(x1, x2) <= cx <= max(x1, x2) and min(y1, y2) <= cy <= max(y1, y2):
                return box.box_id, "move"
        return None, None

    def on_canvas_press(self, event: Any) -> None:
        if self.current_frame is None:
            return
        ix, iy = self.clamp_point_to_image(*self.canvas_to_image(event.x, event.y))
        if self.add_mode:
            self.temp_add_start = (ix, iy)
            self.temp_add_end = (ix, iy)
            self.redraw_canvas()
            return

        box_id, mode = self.hit_test(event.x, event.y)
        self.selected_box_id = box_id
        self.drag_mode = mode
        self.drag_box_id = box_id
        self.drag_start_img = (ix, iy)
        box = self.get_box(box_id) if box_id is not None else None
        self.drag_orig_xyxy = box.as_xyxy() if box is not None else None
        self.rebuild_box_rows()
        self.redraw_canvas()

    def on_canvas_drag(self, event: Any) -> None:
        if self.current_frame is None:
            return
        ix, iy = self.clamp_point_to_image(*self.canvas_to_image(event.x, event.y))
        if self.add_mode and self.temp_add_start:
            self.temp_add_end = (ix, iy)
            self.redraw_canvas()
            return
        if not self.drag_mode or self.drag_box_id is None or self.drag_start_img is None or self.drag_orig_xyxy is None:
            return
        box = self.get_box(self.drag_box_id)
        if box is None:
            return
        ox1, oy1, ox2, oy2 = self.drag_orig_xyxy
        if self.drag_mode == "move":
            dx = ix - self.drag_start_img[0]
            dy = iy - self.drag_start_img[1]
            box.set_from_xyxy(ox1 + dx, oy1 + dy, ox2 + dx, oy2 + dy)
        else:
            x1, y1, x2, y2 = ox1, oy1, ox2, oy2
            if "n" in self.drag_mode:
                y1 = iy
            if "s" in self.drag_mode:
                y2 = iy
            if "w" in self.drag_mode:
                x1 = ix
            if "e" in self.drag_mode:
                x2 = ix
            box.set_from_xyxy(x1, y1, x2, y2)
        height, width = self.current_frame.shape[:2]
        box.clamp(width, height)
        box.source = "manual_adjusted" if box.source == "detector" else box.source
        self.redraw_canvas()

    def on_canvas_release(self, event: Any) -> None:
        if self.current_frame is None:
            return
        if self.add_mode and self.temp_add_start:
            ix, iy = self.clamp_point_to_image(*self.canvas_to_image(event.x, event.y))
            x1, y1 = self.temp_add_start
            box = BBox(self.next_box_id, x=0, y=0, w=1, h=1, cow_name=NO_NAME, score=None, source="manual")
            box.set_from_xyxy(x1, y1, ix, iy)
            if box.w >= 5 and box.h >= 5:
                height, width = self.current_frame.shape[:2]
                box.clamp(width, height)
                self.boxes.append(box)
                self.next_box_id += 1
                self.selected_box_id = box.box_id
                self.set_status(f"Added bbox #{box.box_id}")
            self.temp_add_start = None
            self.temp_add_end = None
            self.add_mode = False
            self.update_add_button()
            self.rebuild_box_rows()
            self.redraw_canvas()
            return
        self.drag_mode = None
        self.drag_box_id = None
        self.drag_start_img = None
        self.drag_orig_xyxy = None
        self.rebuild_box_rows()

    def get_box(self, box_id: Optional[int]) -> Optional[BBox]:
        if box_id is None:
            return None
        for box in self.boxes:
            if box.box_id == box_id:
                return box
        return None

    def toggle_add_mode(self) -> None:
        self.add_mode = not self.add_mode
        self.temp_add_start = None
        self.temp_add_end = None
        self.update_add_button()
        self.set_status("Add bbox mode on" if self.add_mode else "Add bbox mode off")
        self.redraw_canvas()

    def update_add_button(self) -> None:
        if self.add_button is not None:
            self.add_button.configure(text="Cancel add" if self.add_mode else "Add bbox")

    def delete_selected_box(self) -> None:
        if self.selected_box_id is None:
            return
        before = len(self.boxes)
        self.boxes = [box for box in self.boxes if box.box_id != self.selected_box_id]
        if len(self.boxes) != before:
            self.set_status("Deleted selected bbox")
        self.selected_box_id = self.boxes[0].box_id if self.boxes else None
        self.rebuild_box_rows()
        self.redraw_canvas()

    def delete_all_boxes(self) -> None:
        if not self.boxes:
            return
        from tkinter import messagebox

        if not messagebox.askyesno("Delete all bboxes", "Delete all current bboxes for this video?"):
            return
        self.boxes = []
        self.selected_box_id = None
        self.rebuild_box_rows()
        self.redraw_canvas()
        self.set_status("Deleted all bboxes")

    def select_box(self, box_id: int) -> None:
        self.selected_box_id = box_id
        self.rebuild_box_rows()
        self.redraw_canvas()

    def update_box_name(self, box_id: int, cow_name: str) -> None:
        box = self.get_box(box_id)
        if box is None:
            return
        box.cow_name = self.cow_name_from_current_display(cow_name)
        self.redraw_canvas()

    def delete_box_by_id(self, box_id: int) -> None:
        self.selected_box_id = box_id
        self.delete_selected_box()

    def rebuild_box_rows(self) -> None:
        for child in self.box_rows_frame.winfo_children():
            child.destroy()
        tk = self.tk
        ttk = self.ttk
        if not self.boxes:
            ttk.Label(self.box_rows_frame, text="No bboxes").grid(row=0, column=0, sticky="w", padx=4, pady=4)
            return

        display_options = self.current_cow_display_options()
        for row_idx, box in enumerate(self.boxes):
            row = tk.Frame(self.box_rows_frame, bg="#fff3bf" if box.box_id == self.selected_box_id else "#ffffff")
            row.grid(row=row_idx, column=0, sticky="ew", padx=2, pady=2)
            row.columnconfigure(2, weight=1)
            tk.Label(row, text=f"#{box.box_id}", width=5, bg=row["bg"]).grid(row=0, column=0, padx=2, pady=2)
            name_var = tk.StringVar(value=self.display_value_for_current_name(box.cow_name))
            combo = ttk.Combobox(row, values=display_options, textvariable=name_var, width=18, state="readonly")
            combo.grid(row=0, column=1, padx=2, pady=2)
            combo.bind(
                "<<ComboboxSelected>>",
                lambda _event, bid=box.box_id, var=name_var: self.update_box_name(bid, var.get()),
            )
            coord_text = f"x={box.x:.0f} y={box.y:.0f} w={box.w:.0f} h={box.h:.0f}"
            tk.Label(row, text=coord_text, anchor="w", bg=row["bg"]).grid(row=0, column=2, sticky="ew", padx=2)
            ttk.Button(row, text="Select", command=lambda bid=box.box_id: self.select_box(bid)).grid(row=0, column=3, padx=2)
            ttk.Button(row, text="Del", command=lambda bid=box.box_id: self.delete_box_by_id(bid)).grid(row=0, column=4, padx=2)
        self.box_rows_canvas.configure(scrollregion=self.box_rows_canvas.bbox("all"))

    def save_current(self) -> None:
        from tkinter import messagebox

        if self.current_index is None or self.current_frame is None:
            return
        entry = self.videos[self.current_index]
        try:
            self.sanitize_boxes_for_current_options()
            detector_info = self.detector.info(self.vram_profile_var.get())
            write_annotation(
                entry=entry,
                source_root=self.source_root,
                output_root=self.output_root,
                frame_bgr=self.current_frame,
                frame_meta=self.current_meta,
                boxes=self.boxes,
                detector_info=detector_info,
                cow_name_options=self.current_cow_options(),
            )
            self.refresh_video_list()
            if self.current_index is not None:
                self.video_listbox.selection_clear(0, self.tk.END)
                self.video_listbox.selection_set(self.current_index)
                self.video_listbox.activate(self.current_index)
            self.set_status(f"Saved annotation: {entry.out_dir}")
        except Exception as exc:
            messagebox.showerror("Save failed", str(exc))
            self.set_status(f"Save failed: {exc}")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="First-frame cow bbox annotation GUI.")
    parser.add_argument("--source-root", default=str(DEFAULT_SOURCE_ROOT), help="Video root scanned recursively.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="Annotation output root.")
    parser.add_argument("--det-weights", default=str(DEFAULT_DET_WEIGHTS), help="Local DCSNA detector weights.")
    parser.add_argument("--vram-profile", choices=sorted(VRAM_PROFILES), default="8GB", help="Detection profile.")
    parser.add_argument("--log-file", default=str(DEFAULT_LOG_FILE), help="Realtime log file.")
    parser.add_argument("--cow-excel", default=str(DEFAULT_COW_EXCEL), help="Read-only Excel file for per-video cow-name filters.")
    parser.add_argument("--verify-log-only", action="store_true", help="Verify stdout/file log flushing and exit.")
    return parser.parse_args(argv)


def verify_log_flush() -> int:
    log("log flush check 1/3")
    time.sleep(0.2)
    log("log flush check 2/3")
    time.sleep(0.2)
    log("log flush check 3/3")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    configure_utf8()
    args = parse_args(argv)

    global LOGGER
    LOGGER = Logger(Path(args.log_file))

    if args.verify_log_only:
        return verify_log_flush()

    import tkinter as tk

    source_root = Path(args.source_root)
    output_root = Path(args.output_root)
    detector = DetectionRunner(Path(args.det_weights))
    cow_name_filters = load_cow_name_filters(Path(args.cow_excel))

    root = tk.Tk()
    BBoxAnnotatorApp(
        root=root,
        source_root=source_root,
        output_root=output_root,
        detector=detector,
        default_vram_profile=args.vram_profile,
        cow_name_filters=cow_name_filters,
    )
    log(f"started GUI; source_root={source_root}; output_root={output_root}")
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
