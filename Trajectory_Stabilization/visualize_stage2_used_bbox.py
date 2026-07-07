from __future__ import annotations

import argparse
import csv
import math
import re
import shutil
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

import train_stage2_valence_inception as stage2


try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


DEFAULT_OUT_DIR = stage2.BASE / "output" / "stage2_used_bbox_vis_v0604_good"
VIS_TASKS = [stage2.GATE_TASK, stage2.VALENCE_TASK, stage2.UNIFIED_TASK]
ENCODERS = ["auto", "opencv", "h264_nvenc"]
TEXT_SCALE_MULTIPLIER = 2.0


@dataclass
class CapturedSample:
    sample: stage2.Stage2Sample
    records: list[stage2.PairFrameRecord]
    split: str
    split_index: int
    roi: "RoiBundle | None" = None


@dataclass
class RoiBundle:
    events: list[stage2.AnnotationEvent]
    source: str
    video_path: Path | None = None


def log(message: str) -> None:
    print(message, flush=True)


def normalize_visual_path_string(value: str) -> str:
    text = stage2.normalize_legacy_path_string(str(value or "").strip())
    if not text:
        return text

    if sys.platform != "win32":
        match = re.match(r"^([GgZz]):[\\/]*(.*)$", text)
        if match:
            drive = match.group(1).upper()
            rest = match.group(2).replace("\\", "/")
            root = Path("/home/hyw/UPAN_HYW") if drive == "G" else Path("/home/hyw")
            return str(root / rest)
    else:
        prefixes = [
            ("/home/hyw/UPAN_HYW/", "G:\\"),
            ("/home/hyw/", "Z:\\"),
        ]
        normalized = text.replace("\\", "/")
        for old, new in prefixes:
            if normalized.lower().startswith(old.lower()):
                return new + normalized[len(old) :].replace("/", "\\")

    return text


def visual_path(value: str | Path) -> Path:
    return Path(normalize_visual_path_string(str(value))).expanduser()


def default_visual_path(raw_windows_path: str) -> Path:
    return visual_path(raw_windows_path)


DEFAULT_INDEX_CSV = default_visual_path(r"Z:\BBOX_VIS-V0604\output\visualization_index.csv")
DEFAULT_5090_ANNOTATION_ROOT = default_visual_path(r"Z:\Interaction_Annotator-V0601_03_AND_05R\output\annotations")
DEFAULT_5090_CLEAN_VIDEO_ROOT = default_visual_path(r"Z:\Interaction_Annotator-V0601_03_AND_05R\output\videos")
DEFAULT_5090_STAGE1_ROOT = default_visual_path(r"Z:\V0604_SEG_S1_TI")
DEFAULT_5090_SELECTED_ID_ROOT = default_visual_path(r"Z:\BBOX_VIS-V0604\output_selected_IDs")
DEFAULT_5090_BBOX_GOOD_ROOT = default_visual_path(r"Z:\BBOX_VIS-V0604\GOOD")
DEFAULT_5090_BBOX_GOODTEST_ROOT = default_visual_path(r"Z:\BBOX_VIS-V0604\GOODTEST")
DEFAULT_5090_BBOX_BAD_ROOT = default_visual_path(r"Z:\BBOX_VIS-V0604\BAD")
INDEX_SOURCE_SETS = ["GOOD", "GOODTEST", "all"]


def split_path_parts(value: str | Path) -> list[str]:
    return [part for part in str(value or "").replace("\\", "/").split("/") if part]


def candidate_under_marker(raw: str, marker: str, root: Path) -> Path | None:
    parts = split_path_parts(raw)
    marker_lower = marker.lower()
    for index in range(len(parts) - 1, -1, -1):
        if parts[index].lower() == marker_lower:
            tail = parts[index + 1 :]
            if tail:
                return root.joinpath(*tail)
    return None


def first_existing_file(description: str, candidates: list[Path | None]) -> Path:
    tried: list[str] = []
    for candidate in candidates:
        if candidate is None:
            continue
        path = visual_path(candidate)
        tried.append(str(path))
        if path.is_file():
            return path
    raise FileNotFoundError(f"{description} not found; tried: {tried}")


def first_existing_dir(description: str, candidates: list[Path | None]) -> Path:
    tried: list[str] = []
    for candidate in candidates:
        if candidate is None:
            continue
        path = visual_path(candidate)
        tried.append(str(path))
        if path.is_dir():
            return path
    raise FileNotFoundError(f"{description} not found; tried: {tried}")


def slug(value: str, max_len: int = 80) -> str:
    text = str(value or "").strip()
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    text = text.strip("._")
    return (text or "sample")[:max_len]


def video_name_with_base(video_path: Path) -> str:
    parent = video_path.parent.name
    stem = video_path.stem
    if parent and parent.lower() != stem.lower():
        return slug(f"{parent}_{stem}", 180)
    return slug(stem, 180)


def segment_tag(sample: stage2.Stage2Sample) -> str:
    match = re.search(r"(?:^|_)seg(\d+)$", str(sample.folder), flags=re.IGNORECASE)
    if match:
        return f"seg{int(match.group(1)):02d}"
    return "seg01"


def output_stem_for_sample(run_video_index: int, video_path: Path, sample: stage2.Stage2Sample) -> str:
    return f"{run_video_index:04d}_{video_name_with_base(video_path)}_{segment_tag(sample)}"


def display_label(sample: stage2.Stage2Sample) -> str:
    if sample.valence == stage2.NO_INTERACTION:
        return stage2.NO_INTERACTION
    binary = stage2.sample_binary_valence(sample)
    return binary or sample.valence


def expects_annotation_roi(sample: stage2.Stage2Sample) -> bool:
    return sample.root_tag in {"v0520_good", "v0520_goodtest"} or sample.source_set in {"GOOD", "GOODTEST"}


def has_valid_roi(event: stage2.AnnotationEvent) -> bool:
    _x, _y, w, h = event.roi
    return bool(w > 0 and h > 0)


def same_roi(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
    return all(math.isclose(float(x), float(y), rel_tol=1e-6, abs_tol=1e-3) for x, y in zip(a, b))


def resolve_annotation_video_path(events: list[stage2.AnnotationEvent]) -> Path:
    paths: set[Path] = set()
    missing_paths: list[Path] = []
    for event in events:
        source_csv = Path(event.source_csv)
        if not source_csv.is_file():
            raise FileNotFoundError(f"annotation source CSV not found: {source_csv}")
        with source_csv.open("r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                valence = str(row.get("valence", "")).strip().lower()
                if valence != event.valence:
                    continue
                key = stage2.clip_key_from_path(
                    row.get("clip_rel_path") or row.get("clip_path") or source_csv.with_suffix(".mp4").name
                )
                if key != event.clip_key:
                    continue
                start_frame = stage2.parse_int_field(row.get("start_frame"), -1)
                end_frame = stage2.parse_int_field(row.get("end_frame"), -1)
                roi = (
                    stage2.parse_float_field(row.get("roi_x")),
                    stage2.parse_float_field(row.get("roi_y")),
                    stage2.parse_float_field(row.get("roi_w")),
                    stage2.parse_float_field(row.get("roi_h")),
                )
                if start_frame != event.start_frame or end_frame != event.end_frame or not same_roi(roi, event.roi):
                    continue
                raw_clip_path = str(row.get("clip_path") or "").strip()
                if not raw_clip_path:
                    raise RuntimeError(f"annotation row has no clip_path: {source_csv}")
                clip_path = Path(stage2.normalize_legacy_path_string(raw_clip_path))
                if clip_path.is_file():
                    paths.add(clip_path.resolve())
                else:
                    missing_paths.append(clip_path)
    if len(paths) == 1:
        return next(iter(paths))
    if len(paths) > 1:
        raise RuntimeError(f"annotation events resolve to multiple clean clip paths: {sorted(str(p) for p in paths)}")
    if missing_paths:
        raise FileNotFoundError(f"annotation clip_path not found: {missing_paths[0]}")
    raise RuntimeError("could not resolve annotation clip_path for ROI events")


def resolve_roi_bundle(
    sample: stage2.Stage2Sample,
    annotation_index: tuple[dict[tuple[str, str], list[stage2.AnnotationEvent]], dict[str, list[stage2.AnnotationEvent]], object] | None,
) -> RoiBundle:
    if annotation_index is None:
        return RoiBundle([], "disabled")
    if not expects_annotation_roi(sample):
        return RoiBundle([], "not_annotation_sample")

    manifest = stage2.load_json(Path(sample.manifest_path))
    by_key, by_clip_id, _stats = annotation_index
    events, source = stage2.annotation_events_for_manifest(manifest, by_key, by_clip_id)
    events = [event for event in events if event.valence in stage2.VALID_ANNOTATION_VALENCES and has_valid_roi(event)]
    if not events:
        raise RuntimeError(f"annotation ROI missing for selected-ID sample: {sample.manifest_path}")
    return RoiBundle(events, source, resolve_annotation_video_path(events))


def roi_events_for_frame(roi: RoiBundle, frame: int) -> list[stage2.AnnotationEvent]:
    active = [event for event in roi.events if event.start_frame <= frame <= event.end_frame]
    return active if active else list(roi.events)


def clone_records(records: list[stage2.PairFrameRecord]) -> list[stage2.PairFrameRecord]:
    out: list[stage2.PairFrameRecord] = []
    for record in records:
        out.append(
            stage2.PairFrameRecord(
                frame=int(record.frame),
                tid_a=None if record.tid_a is None else int(record.tid_a),
                tid_b=None if record.tid_b is None else int(record.tid_b),
                box_a=tuple(float(v) for v in record.box_a),
                box_b=tuple(float(v) for v in record.box_b),
                source=str(record.source),
                kpts_a=None,
                kpts_b=None,
            )
        )
    return out


def install_record_capture() -> tuple[list[list[stage2.PairFrameRecord]], Callable[[], None]]:
    captured: list[list[stage2.PairFrameRecord]] = []
    original_build_sequence_from_records = stage2.build_sequence_from_records
    original_build_pair_sequence = stage2.build_pair_sequence

    def capture_build_sequence_from_records(records, kpts_by_key, num_kpts):
        x = original_build_sequence_from_records(records, kpts_by_key, num_kpts)
        captured.append(clone_records(records))
        return x

    def capture_build_pair_sequence(frames, tid_a, tid_b, boxes_by_tid, kpts_by_key, num_kpts):
        frame_list = list(frames)
        x = original_build_pair_sequence(frame_list, tid_a, tid_b, boxes_by_tid, kpts_by_key, num_kpts)
        records = [
            stage2.PairFrameRecord(
                frame=int(frame),
                tid_a=int(tid_a),
                tid_b=int(tid_b),
                box_a=tuple(float(v) for v in boxes_by_tid[int(tid_a)][int(frame)]),
                box_b=tuple(float(v) for v in boxes_by_tid[int(tid_b)][int(frame)]),
                source="real",
                kpts_a=None,
                kpts_b=None,
            )
            for frame in frame_list
        ]
        captured.append(records)
        return x

    stage2.build_sequence_from_records = capture_build_sequence_from_records
    stage2.build_pair_sequence = capture_build_pair_sequence

    def restore() -> None:
        stage2.build_sequence_from_records = original_build_sequence_from_records
        stage2.build_pair_sequence = original_build_pair_sequence

    return captured, restore


def build_train_args(args: argparse.Namespace):
    parser = stage2.build_arg_parser()
    train_args = parser.parse_args([])
    train_args.task = args.task
    train_args.data_root = args.data_root
    train_args.v0520_root = args.v0520_root
    train_args.annotation_root = args.annotation_root
    train_args.selected_id_root = args.selected_id_root
    train_args.bbox_good_root = args.bbox_good_root
    train_args.bbox_goodtest_root = args.bbox_goodtest_root
    train_args.bbox_bad_root = args.bbox_bad_root
    train_args.template_ckpt = args.template_ckpt
    train_args.max_videos = args.max_videos
    train_args.samples_per_video = args.samples_per_video
    train_args.min_seq_frames = args.min_seq_frames
    train_args.max_id_gap_sec = args.max_id_gap_sec
    train_args.max_pair_fill_frac = args.max_pair_fill_frac
    train_args.min_dominant_vote_ratio = args.min_dominant_vote_ratio
    train_args.min_eval_dominant_vote_ratio = args.min_eval_dominant_vote_ratio
    train_args.fps = args.fps
    train_args.seed = args.seed
    train_args.split_seed = args.split_seed
    return train_args


def discover_captured_samples(args: argparse.Namespace) -> list[CapturedSample]:
    train_args = build_train_args(args)
    stage2.set_active_task_labels(train_args.task)
    keypoints, _meta = stage2.load_template_metadata(Path(train_args.template_ckpt))

    captured_records, restore = install_record_capture()
    try:
        samples = stage2.discover_samples(train_args, keypoints)
    finally:
        restore()

    if len(captured_records) != len(samples):
        raise RuntimeError(
            "captured bbox records do not line up with built samples: "
            f"records={len(captured_records)} samples={len(samples)}"
        )

    train_samples, val_samples = stage2.deterministic_split(samples, train_args.task, train_args.split_seed)
    split_by_id = {id(sample): "train" for sample in train_samples}
    split_by_id.update({id(sample): "val" for sample in val_samples})
    index_by_id: dict[int, int] = {}
    for split_samples in (train_samples, val_samples):
        for index, sample in enumerate(split_samples):
            index_by_id[id(sample)] = index

    bundles: list[CapturedSample] = []
    for sample, records in zip(samples, captured_records):
        split = split_by_id.get(id(sample), "unused")
        if args.split != "all" and split != args.split:
            continue
        bundles.append(
            CapturedSample(
                sample=sample,
                records=records,
                split=split,
                split_index=index_by_id.get(id(sample), -1),
            )
        )
    return apply_filters(bundles, args)


def load_index_rows(index_csv: Path) -> list[dict[str, str]]:
    if not index_csv.is_file():
        raise FileNotFoundError(f"visualization index CSV not found: {index_csv}")
    with index_csv.open("r", encoding="utf-8-sig", newline="") as f:
        rows = [dict(row) for row in csv.DictReader(f)]
    required = {"clip_name", "annotation_csv", "clip_path", "tracking_csv", "keypoints_csv", "valence"}
    missing = required.difference(rows[0].keys() if rows else set())
    if missing:
        raise ValueError(f"{index_csv} missing columns: {sorted(missing)}")
    return rows


def row_contains_any(row: dict[str, str], needles: tuple[str, ...]) -> bool:
    text = " ".join(str(value or "") for value in row.values()).lower()
    return any(needle in text for needle in needles)


def source_set_names(root: Path) -> list[str]:
    if not root.is_dir():
        raise FileNotFoundError(f"BBOX_VIS source-set root not found: {root}")
    return [p.name for p in sorted(root.glob("*.mp4"), key=lambda p: p.name.lower()) if p.is_file()]


def group_index_rows_for_source_set(
    rows: list[dict[str, str]],
    source_set: str,
    good_root: Path,
    goodtest_root: Path,
) -> list[tuple[str, list[dict[str, str]]]]:
    by_clip: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        clip_name = str(row.get("clip_name") or "").strip()
        if clip_name:
            by_clip.setdefault(clip_name.lower(), []).append(row)

    if source_set == "all":
        return [
            (group[0].get("clip_name", key), group)
            for key, group in sorted(by_clip.items(), key=lambda item: item[0])
        ]

    root = good_root if source_set == "GOOD" else goodtest_root
    wanted_names = source_set_names(root)
    wanted_lower = {name.lower() for name in wanted_names}
    missing = sorted(name for name in wanted_names if name.lower() not in by_clip)
    if missing:
        raise RuntimeError(f"visualization index missing {source_set} clip(s): {missing}")

    grouped: list[tuple[str, list[dict[str, str]]]] = []
    for name in wanted_names:
        grouped.append((name, by_clip[name.lower()]))

    extras = sorted(
        group[0].get("clip_name", key)
        for key, group in by_clip.items()
        if key not in wanted_lower and source_set in {"GOOD", "GOODTEST"}
    )
    if extras:
        log(f"[info] index has {len(extras)} clip(s) outside {source_set}; they are ignored")
    return grouped


def parse_index_frame(row: dict[str, str], name: str, default: int) -> int:
    return int(stage2.parse_int_field(row.get(name), default))


def annotation_csv_for_index_row(row: dict[str, str], annotation_root: Path) -> Path:
    raw = str(row.get("annotation_csv") or "")
    return first_existing_file(
        "annotation CSV",
        [
            candidate_under_marker(raw, "annotations", annotation_root),
        ],
    )


def clean_video_for_index_rows(rows: list[dict[str, str]], clean_video_root: Path) -> Path:
    paths: set[Path] = set()
    for row in rows:
        raw = str(row.get("clip_path") or "")
        path = first_existing_file(
            "clean annotation video",
            [
                candidate_under_marker(raw, "videos", clean_video_root),
            ],
        )
        paths.add(path.resolve(strict=False))
    if len(paths) != 1:
        raise RuntimeError(f"index rows resolve to multiple clean videos: {sorted(str(p) for p in paths)}")
    return next(iter(paths))


def stage1_file_for_index_row(row: dict[str, str], field: str, expected_name: str, stage1_root: Path) -> Path:
    raw = str(row.get(field) or "")
    path = first_existing_file(
        f"Stage1 {expected_name}",
        [
            candidate_under_marker(raw, stage1_root.name, stage1_root),
        ],
    )
    if path.name.lower() != expected_name.lower():
        raise RuntimeError(f"index {field} resolved to {path}, expected {expected_name}")
    return path


def stage1_folder_for_index_row(row: dict[str, str], stage1_root: Path) -> tuple[Path, Path]:
    tracking_csv = stage1_file_for_index_row(row, "tracking_csv", "tracking_boxes.csv", stage1_root)
    keypoints_csv = stage1_file_for_index_row(row, "keypoints_csv", "keypoints.csv", stage1_root)
    csv_root = tracking_csv.parent
    if keypoints_csv.parent != csv_root:
        raise RuntimeError(f"Stage1 tracking/keypoint CSV roots differ: {tracking_csv} vs {keypoints_csv}")
    manifest = first_existing_file(
        "Stage1 manifest",
        [
            csv_root / "manifest.json",
        ],
    )
    return csv_root, manifest


def selected_id_csv_for_clip(clip_name: str, selected_id_root: Path) -> Path:
    return first_existing_file("selected-ID CSV", [selected_id_root / f"{clip_name}.csv"])


def roi_bundle_from_index_rows(
    rows: list[dict[str, str]],
    annotation_root: Path,
    clean_video_root: Path,
) -> RoiBundle:
    clean_video = clean_video_for_index_rows(rows, clean_video_root)
    events: list[stage2.AnnotationEvent] = []
    for row in rows:
        annotation_csv = annotation_csv_for_index_row(row, annotation_root)
        roi = (
            stage2.parse_float_field(row.get("roi_x")),
            stage2.parse_float_field(row.get("roi_y")),
            stage2.parse_float_field(row.get("roi_w")),
            stage2.parse_float_field(row.get("roi_h")),
        )
        event = stage2.AnnotationEvent(
            clip_key=stage2.clip_key_from_path(row.get("clip_path") or row.get("clip_name") or annotation_csv.name),
            clip_id=str(row.get("clip_id") or stage2.clip_id_from_stem(annotation_csv.stem)).upper(),
            valence=str(row.get("valence") or "").strip().lower(),
            start_frame=parse_index_frame(row, "start_frame", 0),
            end_frame=parse_index_frame(row, "end_frame", 10**12),
            roi=roi,
            source_csv=str(annotation_csv),
        )
        if event.valence in stage2.VALID_ANNOTATION_VALENCES and has_valid_roi(event):
            events.append(event)
    if not events:
        raise RuntimeError(f"no usable ROI event in visualization index rows for {clean_video}")
    return RoiBundle(events, "visualization_index", clean_video)


def discover_index_captured_samples(args: argparse.Namespace, index_csv: Path) -> list[CapturedSample]:
    train_args = build_train_args(args)
    stage2.set_active_task_labels(train_args.task)
    keypoints, _meta = stage2.load_template_metadata(Path(train_args.template_ckpt))
    num_kpts = len(keypoints)

    rows = load_index_rows(index_csv)
    annotation_root = visual_path(args.annotation_root)
    clean_video_root = visual_path(args.clean_video_root)
    selected_id_root = visual_path(args.selected_id_root)
    stage1_root = visual_path(args.v0520_root)
    good_root = visual_path(args.bbox_good_root)
    goodtest_root = visual_path(args.bbox_goodtest_root)
    grouped_rows = group_index_rows_for_source_set(rows, args.index_source_set, good_root, goodtest_root)

    flagged = [
        clip_name
        for clip_name, group in grouped_rows
        if any(row_contains_any(row, ("ambiguous", "review_skipped", "review skipped")) for row in group)
    ]
    if flagged:
        log(f"[warn] {args.index_source_set} contains ambiguous/review rows: {flagged}")

    captured_records, restore = install_record_capture()
    bundles: list[CapturedSample] = []
    skipped_training: list[tuple[str, str]] = []
    try:
        for clip_index, (clip_name, group) in enumerate(grouped_rows):
            if any(row_contains_any(row, ("fake_interaction",)) for row in group):
                skipped_training.append((clip_name, "fake_interaction"))
                log(f"[warn] not training-usable: {clip_name}: fake_interaction")
                continue

            valences = sorted({str(row.get("valence") or "").strip().lower() for row in group if row.get("valence")})
            if len(valences) != 1:
                raise RuntimeError(f"index rows have conflicting valence for {clip_name}: {valences}")
            raw_label = valences[0]

            csv_root, manifest_path = stage1_folder_for_index_row(group[0], stage1_root)
            manifest = stage2.load_json(manifest_path)
            id_csv = selected_id_csv_for_clip(clip_name, selected_id_root)
            roi = roi_bundle_from_index_rows(group, annotation_root, clean_video_root)

            before = len(captured_records)
            try:
                new_samples = stage2.build_selected_id_sample_from_folder(
                    folder=csv_root,
                    root_tag=f"v0520_{args.index_source_set.lower()}",
                    source_set=args.index_source_set,
                    manifest_path=manifest_path,
                    manifest=manifest,
                    raw_label=raw_label,
                    label_source="visualization_index+selected_id_pair",
                    id_csv=id_csv,
                    args=train_args,
                    num_kpts=num_kpts,
                )
            except FileNotFoundError:
                raise
            except ValueError as exc:
                skipped_training.append((clip_name, str(exc)))
                log(f"[warn] not training-usable: {clip_name}: {exc}")
                del captured_records[before:]
                continue
            new_records = captured_records[before:]
            if len(new_records) != len(new_samples):
                raise RuntimeError(
                    f"captured bbox records do not line up for {clip_name}: "
                    f"records={len(new_records)} samples={len(new_samples)}"
                )

            for seg_index, (sample, records) in enumerate(zip(new_samples, new_records)):
                bundles.append(
                    CapturedSample(
                        sample=sample,
                        records=records,
                        split="index",
                        split_index=len(bundles),
                        roi=roi,
                    )
                )
                if len(new_samples) > 1:
                    log(f"[info] {clip_name} segment {seg_index + 1}/{len(new_samples)}: {len(records)} frame(s)")
            log(f"[info] indexed clip {clip_index + 1}/{len(grouped_rows)}: {clip_name}")
    finally:
        restore()

    if skipped_training:
        log("[warn] not training-usable clips:")
        for clip_name, reason in skipped_training:
            log(f"[warn]   {clip_name}: {reason}")
    log(f"[info] training-usable indexed sample(s): {len(bundles)}")
    return apply_filters(bundles, args)


def discover_visualization_samples(args: argparse.Namespace) -> list[CapturedSample]:
    index_csv = visual_path(args.index_csv) if args.index_csv else None
    if args.no_index:
        raise RuntimeError("legacy Stage2 discovery fallback is disabled for this 5090 visualization task")
    if index_csv is None or not index_csv.is_file():
        raise FileNotFoundError(f"visualization index CSV not found: {index_csv}")
    log(f"[info] using visualization index: {index_csv}")
    return discover_index_captured_samples(args, index_csv)


def apply_filters(bundles: list[CapturedSample], args: argparse.Namespace) -> list[CapturedSample]:
    out = bundles
    if args.folder:
        needle = args.folder.lower()
        out = [bundle for bundle in out if needle in bundle.sample.folder.lower()]
    if args.manifest:
        target = str(Path(args.manifest)).lower()
        out = [bundle for bundle in out if str(Path(bundle.sample.manifest_path)).lower() == target]
    if args.source_set:
        wanted = {
            item.strip().lower()
            for item in re.split(r"[,;]", args.source_set)
            if item.strip()
        }
        out = [bundle for bundle in out if bundle.sample.source_set.lower() in wanted]
    if args.exclude_source_set:
        excluded = {
            item.strip().lower()
            for item in re.split(r"[,;]", args.exclude_source_set)
            if item.strip()
        }
        out = [bundle for bundle in out if bundle.sample.source_set.lower() not in excluded]
    if args.label:
        wanted = args.label.lower()
        out = [
            bundle
            for bundle in out
            if wanted
            in {
                bundle.sample.valence.lower(),
                display_label(bundle.sample).lower(),
                str(bundle.sample.original_class or "").lower(),
            }
        ]
    return sorted(out, key=lambda b: (b.split, b.split_index, b.sample.manifest_path, b.sample.folder))


def find_video_for_sample(sample: stage2.Stage2Sample) -> Path:
    manifest_path = Path(sample.manifest_path)
    folder = manifest_path.parent
    manifest = stage2.load_json(manifest_path)
    video_name = stage2.video_name_from_manifest(manifest)
    candidates = sorted(p for p in folder.glob("*.mp4") if p.is_file())
    if video_name:
        exact = [p for p in candidates if p.name.lower() == video_name.lower()]
        if len(exact) == 1:
            return exact[0]
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        source = manifest.get("video_manifest", {}).get("source_path") or manifest.get("video_manifest", {}).get("relative_path")
        if source:
            source_path = Path(stage2.normalize_legacy_path_string(str(source)))
            if source_path.is_file():
                return source_path
        raise FileNotFoundError(f"no mp4 found beside manifest: {manifest_path}")
    raise RuntimeError(f"multiple mp4 candidates for {manifest_path}: {[p.name for p in candidates]}")


def manifest_frame_size(sample: stage2.Stage2Sample) -> tuple[float, float]:
    manifest = stage2.load_json(Path(sample.manifest_path))
    vm = manifest.get("video_manifest", {})
    width = float(vm.get("width") or 0.0)
    height = float(vm.get("height") or 0.0)
    return width, height


def scale_box(
    box: tuple[float, float, float, float],
    scale_x: float,
    scale_y: float,
    frame_w: int,
    frame_h: int,
) -> tuple[int, int, int, int]:
    x, y, w, h = box
    x1 = int(round(x * scale_x))
    y1 = int(round(y * scale_y))
    x2 = int(round((x + w) * scale_x))
    y2 = int(round((y + h) * scale_y))
    x1 = max(0, min(frame_w - 1, x1))
    y1 = max(0, min(frame_h - 1, y1))
    x2 = max(0, min(frame_w - 1, x2))
    y2 = max(0, min(frame_h - 1, y2))
    return x1, y1, x2, y2


def put_text(
    frame: np.ndarray,
    text: str,
    org: tuple[int, int],
    color: tuple[int, int, int] = (255, 255, 255),
    scale: float = 0.55,
    thickness: int = 1,
) -> None:
    scale = float(scale) * TEXT_SCALE_MULTIPLIER
    thickness = max(1, int(round(float(thickness) * TEXT_SCALE_MULTIPLIER)))
    pad = max(2, int(round(2 * TEXT_SCALE_MULTIPLIER)))
    x, y = org
    (w, h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    cv2.rectangle(
        frame,
        (x - pad, y - h - baseline - pad),
        (x + w + pad, y + baseline + pad),
        (0, 0, 0),
        -1,
    )
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def draw_box(
    frame: np.ndarray,
    box: tuple[float, float, float, float],
    label: str,
    color: tuple[int, int, int],
    scale_x: float,
    scale_y: float,
) -> None:
    frame_h, frame_w = frame.shape[:2]
    x1, y1, x2, y2 = scale_box(box, scale_x, scale_y, frame_w, frame_h)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
    put_text(frame, label, (x1 + 4, max(18, y1 + 20)), color=color, scale=0.55, thickness=1)


def draw_roi(
    frame: np.ndarray,
    event: stage2.AnnotationEvent,
    index: int,
    active: bool,
    scale_x: float,
    scale_y: float,
) -> None:
    frame_h, frame_w = frame.shape[:2]
    x1, y1, x2, y2 = scale_box(event.roi, scale_x, scale_y, frame_w, frame_h)
    color = (0, 255, 255) if active else (170, 170, 170)
    thickness = 3 if active else 2
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    label = f"ROI{index} {event.valence} {event.start_frame}-{event.end_frame}"
    put_text(frame, label, (x1 + 4, max(18, y1 + 20)), color=color, scale=0.55, thickness=1)


def draw_hud(frame: np.ndarray, bundle: CapturedSample, record: stage2.PairFrameRecord, roi: RoiBundle) -> None:
    sample = bundle.sample
    pair_a, pair_b = sample.pair
    source_color = {
        "real": (80, 255, 80),
        "interpolated": (0, 220, 255),
        "frozen": (255, 160, 60),
    }.get(record.source, (255, 255, 255))
    lines = [
        (
            f"{bundle.split}[{bundle.split_index}] {sample.folder} "
            f"label={display_label(sample)} task_label={sample.valence} source_set={sample.source_set}"
        ),
        f"frame={record.frame} pair=({pair_a},{pair_b}) bbox_source={record.source}",
        (
            f"real={sample.real_frames} interp={sample.interpolated_frames} "
            f"frozen={sample.frozen_frames} dropped={sample.dropped_frames} fill={sample.fill_ratio:.3f}"
        ),
        f"roi_count={len(roi.events)} roi_source={roi.source}",
        f"video={roi.video_path.name if roi.video_path is not None else 'sample-folder-mp4'}",
    ]
    y = int(round(26 * TEXT_SCALE_MULTIPLIER))
    step = int(round(26 * TEXT_SCALE_MULTIPLIER))
    for index, line in enumerate(lines):
        color = source_color if index == 1 else (255, 255, 255)
        put_text(frame, line, (12, y), color=color, scale=0.62 if index == 0 else 0.55, thickness=1)
        y += step


class OpenCvFrameWriter:
    def __init__(self, path: Path, fps: float, size: tuple[int, int]):
        frame_w, frame_h = size
        self.path = path
        self.writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_w, frame_h))
        if not self.writer.isOpened():
            raise RuntimeError(f"failed to create OpenCV video writer: {path}")

    def write(self, frame: np.ndarray) -> None:
        self.writer.write(frame)

    def close(self) -> None:
        self.writer.release()


class FfmpegNvencFrameWriter:
    def __init__(self, ffmpeg_exe: str, path: Path, fps: float, size: tuple[int, int], cq: int, preset: str):
        frame_w, frame_h = size
        self.path = path
        self.proc = subprocess.Popen(
            [
                ffmpeg_exe,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "bgr24",
                "-s:v",
                f"{frame_w}x{frame_h}",
                "-r",
                f"{fps:.6f}",
                "-i",
                "-",
                "-an",
                "-c:v",
                "h264_nvenc",
                "-preset",
                str(preset),
                "-cq",
                str(cq),
                "-pix_fmt",
                "yuv420p",
                str(path),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        if self.proc.stdin is None:
            raise RuntimeError("failed to open ffmpeg stdin")

    def write(self, frame: np.ndarray) -> None:
        if self.proc.stdin is None:
            raise RuntimeError("ffmpeg stdin is closed")
        try:
            self.proc.stdin.write(np.ascontiguousarray(frame).tobytes())
        except BrokenPipeError as exc:
            raise RuntimeError("ffmpeg NVENC pipe closed while writing frames") from exc

    def close(self) -> None:
        if self.proc.stdin is not None:
            self.proc.stdin.close()
        stderr = b""
        if self.proc.stderr is not None:
            stderr = self.proc.stderr.read()
        code = self.proc.wait()
        if code != 0:
            detail = stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"ffmpeg NVENC failed with exit code {code}: {detail}")


def ffmpeg_has_encoder(ffmpeg_exe: str, encoder: str) -> bool:
    try:
        result = subprocess.run(
            [ffmpeg_exe, "-hide_banner", "-encoders"],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except Exception:
        return False
    return result.returncode == 0 and encoder in result.stdout


def create_frame_writer(
    path: Path,
    fps: float,
    size: tuple[int, int],
    encoder: str,
    nvenc_cq: int,
    nvenc_preset: str,
) -> tuple[object, str]:
    selected = encoder
    if selected == "auto":
        selected = "h264_nvenc"

    if selected == "h264_nvenc":
        ffmpeg_exe = shutil.which("ffmpeg")
        if not ffmpeg_exe:
            raise RuntimeError("ffmpeg not found; cannot use h264_nvenc")
        if not ffmpeg_has_encoder(ffmpeg_exe, "h264_nvenc"):
            raise RuntimeError("ffmpeg does not report h264_nvenc support")
        return FfmpegNvencFrameWriter(ffmpeg_exe, path, fps, size, nvenc_cq, nvenc_preset), "h264_nvenc"

    if selected != "opencv":
        raise ValueError(f"unsupported encoder: {encoder}")
    return OpenCvFrameWriter(path, fps, size), "opencv"


def write_bbox_csv(bundle: CapturedSample, path: Path, roi: RoiBundle) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sample = bundle.sample
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "split",
                "split_index",
                "root_tag",
                "source_set",
                "folder",
                "manifest_path",
                "source_video",
                "label",
                "task_label",
                "original_class",
                "label_source",
                "pair_tidA",
                "pair_tidB",
                "frame",
                "bbox_source",
                "record_tidA",
                "record_tidB",
                "a_x",
                "a_y",
                "a_w",
                "a_h",
                "b_x",
                "b_y",
                "b_w",
                "b_h",
                "roi_count",
                "roi_source",
                "active_roi_count",
                "roi_x",
                "roi_y",
                "roi_w",
                "roi_h",
                "roi_start_frame",
                "roi_end_frame",
                "roi_valence",
                "roi_source_csv",
                "render_video_path",
            ],
        )
        writer.writeheader()
        for record in bundle.records:
            ax, ay, aw, ah = record.box_a
            bx, by, bw, bh = record.box_b
            active_rois = [event for event in roi.events if event.start_frame <= int(record.frame) <= event.end_frame]
            row_rois = active_rois if active_rois else list(roi.events)
            row_roi = row_rois[0] if row_rois else None
            if row_roi is None:
                rx = ry = rw = rh = ""
                roi_start = roi_end = roi_valence = roi_source_csv = ""
            else:
                rx, ry, rw, rh = (f"{v:.3f}" for v in row_roi.roi)
                roi_start = row_roi.start_frame
                roi_end = row_roi.end_frame
                roi_valence = row_roi.valence
                roi_source_csv = row_roi.source_csv
            writer.writerow(
                {
                    "split": bundle.split,
                    "split_index": bundle.split_index,
                    "root_tag": sample.root_tag,
                    "source_set": sample.source_set,
                    "folder": sample.folder,
                    "manifest_path": sample.manifest_path,
                    "source_video": sample.source_video,
                    "label": display_label(sample),
                    "task_label": sample.valence,
                    "original_class": sample.original_class,
                    "label_source": sample.label_source,
                    "pair_tidA": sample.pair[0],
                    "pair_tidB": sample.pair[1],
                    "frame": record.frame,
                    "bbox_source": record.source,
                    "record_tidA": "" if record.tid_a is None else record.tid_a,
                    "record_tidB": "" if record.tid_b is None else record.tid_b,
                    "a_x": f"{ax:.3f}",
                    "a_y": f"{ay:.3f}",
                    "a_w": f"{aw:.3f}",
                    "a_h": f"{ah:.3f}",
                    "b_x": f"{bx:.3f}",
                    "b_y": f"{by:.3f}",
                    "b_w": f"{bw:.3f}",
                    "b_h": f"{bh:.3f}",
                    "roi_count": len(roi.events),
                    "roi_source": roi.source,
                    "active_roi_count": len(active_rois),
                    "roi_x": rx,
                    "roi_y": ry,
                    "roi_w": rw,
                    "roi_h": rh,
                    "roi_start_frame": roi_start,
                    "roi_end_frame": roi_end,
                    "roi_valence": roi_valence,
                    "roi_source_csv": roi_source_csv,
                    "render_video_path": "" if roi.video_path is None else str(roi.video_path),
                }
            )


def render_video(
    bundle: CapturedSample,
    out_dir: Path,
    run_video_index: int,
    every_n: int,
    max_frames: int,
    write_csv: bool,
    roi: RoiBundle,
    encoder: str,
    nvenc_cq: int,
    nvenc_preset: str,
) -> tuple[Path, Path | None]:
    sample = bundle.sample
    out_dir.mkdir(parents=True, exist_ok=True)
    video_path = roi.video_path if roi.video_path is not None else find_video_for_sample(sample)
    stem = output_stem_for_sample(run_video_index, video_path, sample)
    csv_path = out_dir / f"{stem}.csv"
    mp4_path = out_dir / f"{stem}.MP4"
    if write_csv:
        write_bbox_csv(bundle, csv_path, roi)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {video_path}")
    try:
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        in_fps = float(cap.get(cv2.CAP_PROP_FPS) or sample.fps or 30.0)
        if frame_w <= 0 or frame_h <= 0:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"failed to read first frame: {video_path}")
            frame_h, frame_w = frame.shape[:2]
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        manifest_w, manifest_h = manifest_frame_size(sample)
        scale_x = frame_w / manifest_w if manifest_w > 0 else 1.0
        scale_y = frame_h / manifest_h if manifest_h > 0 else 1.0
        out_fps = max(1.0, in_fps / max(1, every_n))
        writer, selected_encoder = create_frame_writer(
            mp4_path,
            out_fps,
            (frame_w, frame_h),
            encoder=encoder,
            nvenc_cq=nvenc_cq,
            nvenc_preset=nvenc_preset,
        )
        log(f"[info] render encoder: {selected_encoder}")
        try:
            records = bundle.records[:: max(1, every_n)]
            if max_frames > 0:
                records = records[:max_frames]
            records = sorted(records, key=lambda r: int(r.frame))
            records_by_frame = {int(record.frame): record for record in records}
            missing = 0
            desc = f"render {bundle.split}[{bundle.split_index}] {sample.folder}"
            if not records_by_frame:
                raise RuntimeError(f"no frames selected for rendering: {sample.manifest_path}")
            first_frame = min(records_by_frame)
            last_frame = max(records_by_frame)
            cap.set(cv2.CAP_PROP_POS_FRAMES, first_frame)
            pbar = tqdm(total=len(records_by_frame), desc=desc, unit="frame", ascii=True, mininterval=1.0, file=sys.stdout)
            for frame_index in range(first_frame, last_frame + 1):
                ok, frame = cap.read()
                if not ok:
                    missing += len([f for f in records_by_frame if f >= frame_index])
                    break
                record = records_by_frame.get(frame_index)
                if record is None:
                    continue
                active_roi_ids = {
                    id(event) for event in roi.events if event.start_frame <= int(record.frame) <= event.end_frame
                }
                for roi_index, event in enumerate(roi.events, start=1):
                    draw_roi(
                        frame,
                        event,
                        roi_index,
                        active=(id(event) in active_roi_ids or not active_roi_ids),
                        scale_x=scale_x,
                        scale_y=scale_y,
                    )
                draw_box(
                    frame,
                    record.box_a,
                    f"A tid={sample.pair[0]}",
                    (255, 255, 0),
                    scale_x,
                    scale_y,
                )
                draw_box(
                    frame,
                    record.box_b,
                    f"B tid={sample.pair[1]}",
                    (255, 0, 255),
                    scale_x,
                    scale_y,
                )
                draw_hud(frame, bundle, record, roi)
                writer.write(frame)
                pbar.update(1)
            pbar.close()
            if missing:
                log(f"[warn] skipped {missing} frame(s) that could not be read from {video_path}")
        finally:
            writer.close()
    finally:
        cap.release()

    return mp4_path, csv_path if write_csv else None


def print_sample_list(bundles: list[CapturedSample], limit: int) -> None:
    log("index,split,split_index,label,task_label,source_set,folder,pair,frames,real,interpolated,frozen,dropped,manifest")
    for index, bundle in enumerate(bundles[:limit]):
        s = bundle.sample
        log(
            ",".join(
                [
                    str(index),
                    bundle.split,
                    str(bundle.split_index),
                    display_label(s),
                    s.valence,
                    s.source_set,
                    s.folder,
                    f"{s.pair[0]}-{s.pair[1]}",
                    str(len(s.frames)),
                    str(s.real_frames),
                    str(s.interpolated_frames),
                    str(s.frozen_frames),
                    str(s.dropped_frames),
                    s.manifest_path,
                ]
            )
        )
    if len(bundles) > limit:
        log(f"[info] listed {limit}/{len(bundles)} sample(s)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize the exact bbox sequence used by Stage2 sample construction, "
            "including dominant selected-ID repair sources: real/interpolated/frozen."
        )
    )
    parser.add_argument("--task", choices=VIS_TASKS, default=stage2.GATE_TASK)
    parser.add_argument("--split", choices=["train", "val", "all"], default="val")
    parser.add_argument("--list-samples", action="store_true", help="List matching samples and exit without rendering.")
    parser.add_argument("--max-list", type=int, default=80)
    parser.add_argument("--sample-index", type=int, default=0, help="Index within the filtered/listed samples.")
    parser.add_argument("--max-samples", type=int, default=0, help="How many filtered samples to render; 0 means all.")
    parser.add_argument("--folder", default="", help="Case-insensitive substring filter on Stage2 sample folder, e.g. V_30_seg01.")
    parser.add_argument("--manifest", default="", help="Exact manifest path filter.")
    parser.add_argument("--source-set", default="", help="Optional source_set filter, e.g. GOOD,GOODTEST or fake_interaction.")
    parser.add_argument("--exclude-source-set", default="", help="Optional comma-separated source_set exclusion, e.g. output_4_categories.")
    parser.add_argument("--label", default="", help="Optional label filter: no_interaction, friendly, or unfriendly.")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument(
        "--index-csv",
        default=str(DEFAULT_INDEX_CSV),
        help="Optional visualization_index.csv that maps C000 clips to Stage1 CSVs, annotations, and clean videos.",
    )
    parser.add_argument(
        "--index-source-set",
        choices=INDEX_SOURCE_SETS,
        default="GOOD",
        help="Source-set folder to visualize from the index. Default is the GOOD folder requested for 5090.",
    )
    parser.add_argument("--no-index", action="store_true", help="Disable visualization_index.csv mode and use legacy Stage2 discovery.")
    parser.add_argument("--clean-video-root", default=str(DEFAULT_5090_CLEAN_VIDEO_ROOT))
    parser.add_argument("--every-n", type=int, default=1, help="Render every Nth used frame.")
    parser.add_argument("--max-frames", type=int, default=0, help="Limit rendered frames per sample; 0 means all.")
    parser.add_argument("--no-csv", action="store_true", help="Do not write the used_bbox CSV sidecar.")
    parser.add_argument("--no-roi", action="store_true", help="Disable annotation ROI overlay.")
    parser.add_argument(
        "--encoder",
        choices=ENCODERS,
        default="h264_nvenc",
        help="Video encoder. Default requires ffmpeg h264_nvenc; pass opencv explicitly to use OpenCV mp4v.",
    )
    parser.add_argument("--nvenc-cq", type=int, default=23, help="NVENC constant-quality value; lower is higher quality/larger file.")
    parser.add_argument("--nvenc-preset", default="p4", help="NVENC preset passed to ffmpeg, e.g. p1 fastest through p7 slowest.")

    parser.add_argument("--data-root", default=str(stage2.DEFAULT_DATA_ROOT))
    parser.add_argument("--v0520-root", default=str(DEFAULT_5090_STAGE1_ROOT))
    parser.add_argument("--annotation-root", default=str(DEFAULT_5090_ANNOTATION_ROOT))
    parser.add_argument("--selected-id-root", default=str(DEFAULT_5090_SELECTED_ID_ROOT))
    parser.add_argument("--bbox-good-root", default=str(DEFAULT_5090_BBOX_GOOD_ROOT))
    parser.add_argument("--bbox-goodtest-root", default=str(DEFAULT_5090_BBOX_GOODTEST_ROOT))
    parser.add_argument("--bbox-bad-root", default=str(DEFAULT_5090_BBOX_BAD_ROOT))
    parser.add_argument("--template-ckpt", default=str(stage2.DEFAULT_TEMPLATE_CKPT))
    parser.add_argument("--max-videos", type=int, default=0)
    parser.add_argument("--samples-per-video", type=int, default=1)
    parser.add_argument("--min-seq-frames", type=int, default=12)
    parser.add_argument("--max-id-gap-sec", type=float, default=0.5)
    parser.add_argument("--max-pair-fill-frac", type=float, default=0.30)
    parser.add_argument("--min-dominant-vote-ratio", type=float, default=0.60)
    parser.add_argument("--min-eval-dominant-vote-ratio", type=float, default=0.30)
    parser.add_argument("--fps", type=float, default=60.0)
    parser.add_argument("--seed", type=int, default=20260515)
    parser.add_argument("--split-seed", type=int, default=20260515)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.every_n < 1:
        raise ValueError("--every-n must be >= 1")
    if args.max_samples < 0:
        raise ValueError("--max-samples must be >= 0")
    if not (0 <= int(args.nvenc_cq) <= 51):
        raise ValueError("--nvenc-cq must be between 0 and 51")

    bundles = discover_visualization_samples(args)
    log(f"[info] matched {len(bundles)} sample(s)")
    if not bundles:
        return 1
    if args.list_samples:
        print_sample_list(bundles, args.max_list)
        return 0

    start = max(0, int(args.sample_index))
    if int(args.max_samples) == 0:
        selected = bundles[start:]
    else:
        selected = bundles[start : start + int(args.max_samples)]
    if not selected:
        raise IndexError(f"sample index {args.sample_index} out of range for {len(bundles)} sample(s)")

    annotation_index = None
    needs_annotation_index = not args.no_roi and any(bundle.roi is None for bundle in selected)
    if needs_annotation_index:
        annotation_index = stage2.load_annotation_index(visual_path(args.annotation_root))

    out_dir = Path(args.out_dir)
    for run_video_index, bundle in enumerate(selected, start=1):
        if bundle.roi is not None:
            if args.no_roi:
                roi = RoiBundle([], "disabled", bundle.roi.video_path)
            else:
                roi = bundle.roi
        else:
            roi = resolve_roi_bundle(bundle.sample, annotation_index)
        if not roi.events and not args.no_roi:
            log(f"[warn] no annotation ROI overlay for {bundle.sample.manifest_path}")
        mp4_path, csv_path = render_video(
            bundle=bundle,
            out_dir=out_dir,
            run_video_index=run_video_index,
            every_n=args.every_n,
            max_frames=args.max_frames,
            write_csv=not args.no_csv,
            roi=roi,
            encoder=args.encoder,
            nvenc_cq=int(args.nvenc_cq),
            nvenc_preset=str(args.nvenc_preset),
        )
        log(f"[done] video: {mp4_path}")
        if csv_path is not None:
            log(f"[done] csv: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
