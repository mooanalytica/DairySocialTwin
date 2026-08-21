from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import itertools
import json
import math
import os
import shutil
import sys
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from tqdm import tqdm
except Exception as exc:  # pragma: no cover - dependency is expected in the target env.
    raise RuntimeError("tqdm is required for SNA progress output") from exc

import app as webui_app
from app import (
    REQUIRED_CUDA_VISIBLE_DEVICES,
    ROOT,
    STAGE2_CODE_DIR,
    build_payload,
    get_geometry,
    get_saved_rotation,
    load_samples,
    realtime_adjusted_payload_point,
)
from dairy_social.config import load_config, write_config
from dairy_social.run import run_pipeline
from reid_index import (
    EXPECTED_CLIP_IDS,
    EXPECTED_GLOBAL_IDENTITY_COUNT,
    EXPECTED_SEQUENCE_FPS,
    REID_SEQUENCE_ID,
    REID_VALIDITY_POLICY,
    reid_global_timeline,
)


COMBINED_SAMPLE_ID = "F1_Gopro1_20250505"
EXPECTED_FARM_ID = "1"
EXPECTED_CAMERA_ID = "Gopro1"
EXPECTED_SAMPLE_COUNT = 12
EXPECTED_ZONE_NETWORKS = {
    "cross_zone",
    "food",
    "path",
    "rest",
    "wait_for_water",
    "water",
}

SNA_INPUT_DIR = ROOT / "sna_inputs"
SNA_OUTPUT_DIR = ROOT / "sna_outputs"
COMMUNITY_WINDOW_S = 300.0
COMMUNITY_STEP_S = 300.0
SNA_METADATA = {
    "cow_id_scope": "global_track_uuid",
    "cow_display_label": "display_global_id",
    "cross_sample_identity_alignment": True,
    "combined_sample_id": COMBINED_SAMPLE_ID,
    "reid_sequence_id": REID_SEQUENCE_ID,
    "reid_validity_policy": REID_VALIDITY_POLICY,
    "reid_filter_stage": "after local-id Stage2 inference and before SNA display statistics",
    "timeline_authority": "reid_global_frame_and_global_time_sec",
    "canonical_frame_policy": "all_playback_base_frames_once_excluding_shard_overlap",
    "opportunity_definition": "simultaneously_visible_stage2_geometry_candidate_and_not_red_crossing",
    "coordinate_rule": (
        "WebUIL simple display coordinates after saved MAP rotation and red/green rectangle exclusion; "
        "semantic warp and UI rotation excluded"
    ),
    "community_window_policy": "fixed_long_video_5min_nonoverlap",
    "community_window_note": (
        "Fixed 300-second non-overlapping windows; cows require at least 30 observed seconds per window, "
        "and windows require at least one positive friendly dyad."
    ),
}

TRAJECTORY_COLUMNS = [
    "farm",
    "camera",
    "clip",
    "frame",
    "time_s",
    "cow_id",
    "anchor_x",
    "anchor_y",
    "track_conf",
    "frozen",
    "display_global_id",
    "source_clip_id",
    "source_sample_id",
    "local_frame",
    "local_track_id",
]

INTERACTION_COLUMNS = [
    "farm",
    "camera",
    "clip",
    "frame",
    "time_s",
    "cow_i",
    "cow_j",
    "p_friendly",
    "p_unfriendly",
    "opportunity_eligible",
    "interaction_conf",
    "display_global_id_i",
    "display_global_id_j",
    "source_clip_id",
    "source_sample_id",
    "local_frame",
    "local_track_id_i",
    "local_track_id_j",
]

CHECKPOINT_FILENAMES = (
    "trajectories.csv",
    "interactions.csv",
    "zones.json",
    "global_identities.json",
    "config.yaml",
    "generation_manifest.json",
)


def log(message: str) -> None:
    print(message, flush=True)


class PayloadProgress:
    def __init__(self, sample_id: str) -> None:
        self.sample_id = sample_id
        self.last_message = ""
        self.last_print = 0.0

    def __call__(self, message: str, current: int | None, total: int | None) -> None:
        now = time.monotonic()
        current_value = 0 if current is None else int(current)
        total_value = None if total is None else int(total)
        finished = total_value is not None and total_value > 0 and current_value >= total_value
        if message != self.last_message or finished or now - self.last_print >= 10.0:
            suffix = (
                f"{current_value:,}/{total_value:,}"
                if total_value is not None and total_value > 0
                else f"{current_value:,}"
            )
            log(f"[payload {self.sample_id}] {message}: {suffix}")
            self.last_message = message
            self.last_print = now


def require_cuda_visible_devices() -> None:
    value = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if value != REQUIRED_CUDA_VISIBLE_DEVICES:
        raise RuntimeError(
            f"Refusing to run Stage2 SNA export without CUDA_VISIBLE_DEVICES={REQUIRED_CUDA_VISIBLE_DEVICES}. "
            f"Run: CUDA_VISIBLE_DEVICES={REQUIRED_CUDA_VISIBLE_DEVICES} "
            "python3 -u generate_sna_precomputed.py --neural-only --overwrite"
        )


def validate_combined_samples(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(samples) != EXPECTED_SAMPLE_COUNT:
        raise RuntimeError(
            f"Expected exactly {EXPECTED_SAMPLE_COUNT} current Farm1/Gopro1 shards; found {len(samples)}"
        )

    ordered = sorted(
        samples,
        key=lambda sample: (
            EXPECTED_CLIP_IDS.index(str(sample["sourceVideoId"])),
            int(sample["shardIndex"]),
        ),
    )
    by_clip: dict[str, list[dict[str, Any]]] = {clip_id: [] for clip_id in EXPECTED_CLIP_IDS}
    source_path_by_clip: dict[str, str] = {}
    clip_by_source_path: dict[str, str] = {}

    for sample in ordered:
        farm_id = str(sample["farmId"])
        camera_id = str(sample["cameraId"])
        clip_id = str(sample["sourceVideoId"])
        if farm_id != EXPECTED_FARM_ID or camera_id != EXPECTED_CAMERA_ID:
            raise RuntimeError(
                f"Unexpected sample scope: {sample['id']} farm={farm_id} camera={camera_id}"
            )
        if clip_id not in by_clip:
            raise RuntimeError(f"Unexpected source clip in current samples: {clip_id}")
        expected_id = f"F{EXPECTED_FARM_ID}_{EXPECTED_CAMERA_ID}_{clip_id}_{int(sample['shardIndex'])}"
        if str(sample["id"]) != expected_id:
            raise RuntimeError(f"Sample id is not canonical: expected={expected_id}, actual={sample['id']}")
        source_path = str(sample["sourcePath"])
        if not source_path or not Path(source_path).is_absolute() or Path(source_path).stem != clip_id:
            raise RuntimeError(f"Sample lacks an authoritative full source path: {sample['id']} -> {source_path!r}")
        old_path = source_path_by_clip.setdefault(clip_id, source_path)
        old_clip = clip_by_source_path.setdefault(source_path, clip_id)
        if old_path != source_path or old_clip != clip_id:
            raise RuntimeError(f"Current clip/full-path mapping is not bijective: {sample['id']}")
        fps = float(sample["video"]["fps"])
        if not math.isclose(fps, EXPECTED_SEQUENCE_FPS, rel_tol=0.0, abs_tol=1e-9):
            raise RuntimeError(f"Sample FPS disagrees with re-ID timeline: {sample['id']} -> {fps}")
        by_clip[clip_id].append(sample)

    for clip_id in EXPECTED_CLIP_IDS:
        clip_samples = sorted(by_clip[clip_id], key=lambda sample: int(sample["shardIndex"]))
        expected_shards = 2 if clip_id == "GX010006" else 1
        if len(clip_samples) != expected_shards:
            raise RuntimeError(
                f"Unexpected shard count for {clip_id}: expected={expected_shards}, actual={len(clip_samples)}"
            )
        if [int(sample["shardIndex"]) for sample in clip_samples] != list(range(1, expected_shards + 1)):
            raise RuntimeError(f"Non-contiguous shard indexes for {clip_id}")
        if any(int(sample["shardCount"]) != expected_shards for sample in clip_samples):
            raise RuntimeError(f"Playback shard_count disagrees for {clip_id}")
        previous_end: int | None = None
        for sample in clip_samples:
            playback = sample["playbackSegment"]
            base_start = int(playback["baseStartFrame"])
            base_end = int(playback["baseEndFrame"])
            segment_start = int(playback["segmentStartFrame"])
            segment_end = int(playback["segmentEndFrame"])
            if not segment_start <= base_start <= base_end <= segment_end:
                raise RuntimeError(f"Invalid canonical/physical range for {sample['id']}")
            if previous_end is not None and base_start != previous_end + 1:
                raise RuntimeError(f"Canonical shard seam is not contiguous for {clip_id}")
            reid_global_timeline(clip_id, base_start)
            reid_global_timeline(clip_id, base_end)
            previous_end = base_end

    if set(source_path_by_clip) != set(EXPECTED_CLIP_IDS):
        raise RuntimeError("Current sample/full-path index does not cover exactly the 11 expected clips")
    return ordered


def choose_combined_samples() -> list[dict[str, Any]]:
    sample_list = load_samples(require_ready=True)
    webui_app.SAMPLE_CACHE[True] = sample_list
    return validate_combined_samples(sample_list)


def require_nonproduction_paths_for_frame_limit(args: argparse.Namespace) -> None:
    if int(args.frame_limit) <= 0:
        return
    inputs_dir = Path(args.inputs_dir).resolve()
    outdir = Path(args.outdir).resolve()
    output_is_production = not args.skip_analysis and outdir == SNA_OUTPUT_DIR.resolve()
    if inputs_dir == SNA_INPUT_DIR.resolve() or output_is_production:
        raise RuntimeError(
            "--frame-limit is only for smoke runs. Use explicit non-production paths, for example "
            "--inputs-dir /tmp/webuil_global_sna_smoke_inputs "
            "--outdir /tmp/webuil_global_sna_smoke_outputs"
        )


def require_nonproduction_analysis_output(frame_limit: int, outdir: Path) -> None:
    if frame_limit > 0 and outdir.resolve() == SNA_OUTPUT_DIR.resolve():
        raise RuntimeError(
            "A smoke input checkpoint cannot replace the production SNA output. "
            "Pass an explicit non-production --outdir."
        )


def combined_input_dir(base_dir: Path) -> Path:
    return base_dir / COMBINED_SAMPLE_ID


def combined_output_dir(base_dir: Path) -> Path:
    return base_dir / COMBINED_SAMPLE_ID


def trajectories_path(sample_dir: Path) -> Path:
    return sample_dir / "trajectories.csv"


def interactions_path(sample_dir: Path) -> Path:
    return sample_dir / "interactions.csv"


def zones_path(sample_dir: Path) -> Path:
    return sample_dir / "zones.json"


def identities_path(sample_dir: Path) -> Path:
    return sample_dir / "global_identities.json"


def config_path(sample_dir: Path) -> Path:
    return sample_dir / "config.yaml"


def generation_manifest_path(sample_dir: Path) -> Path:
    return sample_dir / "generation_manifest.json"


def read_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required checkpoint file is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Checkpoint JSON is invalid: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"Checkpoint JSON must contain an object: {path}")
    return value


def canonical_uuid(value: Any, label: str) -> str:
    text = str(value).strip()
    try:
        parsed = str(uuid.UUID(text))
    except ValueError as exc:
        raise RuntimeError(f"{label} is not a UUID: {text!r}") from exc
    if text != parsed:
        raise RuntimeError(f"{label} is not a canonical UUID: {text!r}")
    return text


def validate_csv_header(path: Path, expected_columns: list[str]) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Required checkpoint file is missing: {path}")
    if path.stat().st_size <= 0:
        raise RuntimeError(f"Checkpoint CSV is empty: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        try:
            header = next(csv.reader(handle))
        except StopIteration as exc:
            raise RuntimeError(f"Checkpoint CSV has no header: {path}") from exc
    if header != expected_columns:
        raise RuntimeError(
            f"Checkpoint CSV header mismatch: {path}: expected={expected_columns}, actual={header}"
        )


def expected_sample_contracts() -> list[tuple[str, str, int, int]]:
    rows: list[tuple[str, str, int, int]] = []
    for clip_id in EXPECTED_CLIP_IDS:
        shard_count = 2 if clip_id == "GX010006" else 1
        for shard_index in range(1, shard_count + 1):
            rows.append(
                (
                    f"F{EXPECTED_FARM_ID}_{EXPECTED_CAMERA_ID}_{clip_id}_{shard_index}",
                    clip_id,
                    shard_index,
                    shard_count,
                )
            )
    return rows


def validate_analysis_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Analysis config is missing: {path}")
    config = load_config(path)

    required_values = {
        ("analysis", "farm"): EXPECTED_FARM_ID,
        ("analysis", "camera"): EXPECTED_CAMERA_ID,
        ("analysis", "clip"): COMBINED_SAMPLE_ID,
        ("analysis", "start_time_s"): None,
        ("analysis", "end_time_s"): None,
        ("metadata", "cow_id_scope"): "global_track_uuid",
        ("metadata", "cow_display_label"): "display_global_id",
        ("metadata", "cross_sample_identity_alignment"): True,
        ("metadata", "combined_sample_id"): COMBINED_SAMPLE_ID,
        ("metadata", "reid_sequence_id"): REID_SEQUENCE_ID,
        ("metadata", "reid_validity_policy"): REID_VALIDITY_POLICY,
        ("metadata", "timeline_authority"): SNA_METADATA["timeline_authority"],
        ("metadata", "canonical_frame_policy"): SNA_METADATA["canonical_frame_policy"],
        ("metadata", "opportunity_definition"): SNA_METADATA["opportunity_definition"],
        ("metadata", "coordinate_rule"): SNA_METADATA["coordinate_rule"],
        ("network", "directed"): False,
    }
    for (section, key), expected in required_values.items():
        actual = config.get(section, {}).get(key)
        if actual != expected:
            raise RuntimeError(
                f"Analysis config cannot change checkpoint binding {section}.{key}: "
                f"expected={expected!r}, actual={actual!r}: {path}"
            )
    fps = config.get("time", {}).get("fps")
    try:
        fps_value = float(fps)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"Analysis config has invalid time.fps={fps!r}: {path}") from exc
    if not math.isclose(fps_value, EXPECTED_SEQUENCE_FPS, rel_tol=0.0, abs_tol=1e-9):
        raise RuntimeError(
            f"Analysis config time.fps disagrees with the checkpoint timeline: "
            f"expected={EXPECTED_SEQUENCE_FPS}, actual={fps_value}: {path}"
        )
    return config


def validate_input_checkpoint(sample_dir: Path) -> tuple[dict[str, Any], int]:
    if not sample_dir.is_dir():
        raise FileNotFoundError(f"SNA input checkpoint directory is missing: {sample_dir}")
    for filename in CHECKPOINT_FILENAMES:
        path = sample_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"Required checkpoint file is missing: {path}")
        if path.stat().st_size <= 0:
            raise RuntimeError(f"Required checkpoint file is empty: {path}")

    validate_csv_header(trajectories_path(sample_dir), TRAJECTORY_COLUMNS)
    validate_csv_header(interactions_path(sample_dir), INTERACTION_COLUMNS)

    manifest = read_json_object(generation_manifest_path(sample_dir))
    if int(manifest.get("schema_version", -1)) != 1:
        raise RuntimeError(
            f"Unsupported checkpoint manifest schema: {manifest.get('schema_version')!r}"
        )
    generation_id = canonical_uuid(manifest.get("generation_id"), "checkpoint generation_id")
    expected_manifest_values = {
        "combined_sample_id": COMBINED_SAMPLE_ID,
        "farm": EXPECTED_FARM_ID,
        "camera": EXPECTED_CAMERA_ID,
        "reid_sequence_id": REID_SEQUENCE_ID,
        "identity_key": "global_track_uuid",
        "display_label": "display_global_id",
        "canonical_frame_policy": SNA_METADATA["canonical_frame_policy"],
    }
    for key, expected in expected_manifest_values.items():
        actual = manifest.get(key)
        if actual != expected:
            raise RuntimeError(
                f"Checkpoint manifest mismatch for {key}: expected={expected!r}, actual={actual!r}"
            )
    try:
        timeline_fps = float(manifest.get("timeline_fps"))
    except (TypeError, ValueError) as exc:
        raise RuntimeError("Checkpoint manifest timeline_fps is invalid") from exc
    if not math.isclose(timeline_fps, EXPECTED_SEQUENCE_FPS, rel_tol=0.0, abs_tol=1e-9):
        raise RuntimeError(
            f"Checkpoint timeline FPS mismatch: expected={EXPECTED_SEQUENCE_FPS}, actual={timeline_fps}"
        )

    raw_frame_limit = manifest.get("frame_limit")
    if raw_frame_limit is None:
        frame_limit = 0
    elif isinstance(raw_frame_limit, bool) or not isinstance(raw_frame_limit, int) or raw_frame_limit <= 0:
        raise RuntimeError(f"Checkpoint manifest frame_limit is invalid: {raw_frame_limit!r}")
    else:
        frame_limit = int(raw_frame_limit)

    integer_fields = (
        "canonical_frame_count",
        "trajectory_frame_count",
        "trajectory_row_count",
        "frozen_trajectory_row_count",
        "interaction_row_count",
        "global_identity_count",
    )
    counts: dict[str, int] = {}
    for key in integer_fields:
        value = manifest.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise RuntimeError(f"Checkpoint manifest {key} is invalid: {value!r}")
        counts[key] = int(value)
    if counts["canonical_frame_count"] <= 0:
        raise RuntimeError("Checkpoint contains no canonical frames")
    if not 0 < counts["trajectory_frame_count"] <= counts["canonical_frame_count"]:
        raise RuntimeError("Checkpoint trajectory_frame_count is outside the canonical timeline")
    if counts["trajectory_row_count"] < counts["trajectory_frame_count"]:
        raise RuntimeError("Checkpoint has fewer trajectory rows than trajectory frames")
    if counts["frozen_trajectory_row_count"] > counts["trajectory_row_count"]:
        raise RuntimeError("Checkpoint frozen trajectory count exceeds all trajectory rows")
    if counts["global_identity_count"] <= 0:
        raise RuntimeError("Checkpoint contains no global identities")
    if frame_limit > 0 and counts["canonical_frame_count"] > frame_limit:
        raise RuntimeError("Checkpoint canonical frame count exceeds its smoke frame_limit")

    samples = manifest.get("samples")
    if not isinstance(samples, list) or not samples:
        raise RuntimeError("Checkpoint manifest samples must be a non-empty list")
    expected_samples = expected_sample_contracts()
    if len(samples) > len(expected_samples):
        raise RuntimeError(f"Checkpoint contains too many sample shards: {len(samples)}")
    if frame_limit <= 0 and len(samples) != EXPECTED_SAMPLE_COUNT:
        raise RuntimeError(
            f"Production checkpoint must contain {EXPECTED_SAMPLE_COUNT} shards; found={len(samples)}"
        )

    sample_canonical_count = 0
    sample_trajectory_rows = 0
    sample_interaction_rows = 0
    for index, item in enumerate(samples):
        if not isinstance(item, dict):
            raise RuntimeError(f"Checkpoint sample row {index} is not an object")
        expected_id, expected_clip, expected_shard, expected_shard_count = expected_samples[index]
        expected_values = {
            "sample_id": expected_id,
            "source_clip_id": expected_clip,
            "shard_index": expected_shard,
            "shard_count": expected_shard_count,
        }
        for key, expected in expected_values.items():
            actual = item.get(key)
            if actual != expected:
                raise RuntimeError(
                    f"Checkpoint sample {index} mismatch for {key}: "
                    f"expected={expected!r}, actual={actual!r}"
                )
        try:
            segment_start = int(item["segment_start_frame"])
            segment_end = int(item["segment_end_frame"])
            canonical_start = int(item["canonical_start_frame"])
            canonical_end = int(item["canonical_end_frame"])
            canonical_count = int(item["canonical_frame_count"])
            trajectory_rows = int(item["trajectory_row_count"])
            interaction_rows = int(item["interaction_row_count"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f"Checkpoint sample {expected_id} has invalid numeric fields") from exc
        if not segment_start <= canonical_start <= canonical_end <= segment_end:
            raise RuntimeError(f"Checkpoint sample {expected_id} has invalid frame bounds")
        if canonical_count != canonical_end - canonical_start + 1:
            raise RuntimeError(f"Checkpoint sample {expected_id} has inconsistent canonical frame count")
        if trajectory_rows < 0 or interaction_rows < 0:
            raise RuntimeError(f"Checkpoint sample {expected_id} has negative row counts")
        sample_canonical_count += canonical_count
        sample_trajectory_rows += trajectory_rows
        sample_interaction_rows += interaction_rows
    if sample_canonical_count != counts["canonical_frame_count"]:
        raise RuntimeError("Checkpoint per-sample canonical counts do not match the manifest total")
    if sample_trajectory_rows != counts["trajectory_row_count"]:
        raise RuntimeError("Checkpoint per-sample trajectory rows do not match the manifest total")
    if sample_interaction_rows != counts["interaction_row_count"]:
        raise RuntimeError("Checkpoint per-sample interaction rows do not match the manifest total")

    identities = read_json_object(identities_path(sample_dir))
    identity_contract = {
        "schema_version": 1,
        "generation_id": generation_id,
        "sequence_id": REID_SEQUENCE_ID,
        "combined_sample_id": COMBINED_SAMPLE_ID,
        "identity_key": "global_track_uuid",
        "display_label": "display_global_id",
    }
    for key, expected in identity_contract.items():
        actual = identities.get(key)
        if actual != expected:
            raise RuntimeError(
                f"Global identity checkpoint mismatch for {key}: expected={expected!r}, actual={actual!r}"
            )
    identity_rows = identities.get("identities")
    if not isinstance(identity_rows, list) or len(identity_rows) != counts["global_identity_count"]:
        raise RuntimeError(
            "Global identity checkpoint count does not match generation_manifest.json"
        )
    global_uuids: set[str] = set()
    global_ids: set[int] = set()
    display_ids: set[str] = set()
    for index, item in enumerate(identity_rows):
        if not isinstance(item, dict):
            raise RuntimeError(f"Global identity row {index} is not an object")
        global_uuid = canonical_uuid(item.get("global_track_uuid"), f"identity row {index} UUID")
        global_id = item.get("global_track_id")
        if isinstance(global_id, bool) or not isinstance(global_id, int) or global_id < 0:
            raise RuntimeError(f"Global identity row {index} has invalid global_track_id={global_id!r}")
        display_id = str(item.get("display_global_id", ""))
        if display_id != f"G{global_id + 1:04d}":
            raise RuntimeError(f"Global identity row {index} has invalid display label={display_id!r}")
        if global_uuid in global_uuids or global_id in global_ids or display_id in display_ids:
            raise RuntimeError(f"Global identity mapping is not bijective at row {index}")
        global_uuids.add(global_uuid)
        global_ids.add(global_id)
        display_ids.add(display_id)
    if frame_limit <= 0:
        if counts["global_identity_count"] != EXPECTED_GLOBAL_IDENTITY_COUNT:
            raise RuntimeError(
                f"Production checkpoint must contain {EXPECTED_GLOBAL_IDENTITY_COUNT} global identities"
            )
        if global_ids != set(range(EXPECTED_GLOBAL_IDENTITY_COUNT)):
            raise RuntimeError("Production checkpoint global IDs are not exactly 0..61")

    zones = read_json_object(zones_path(sample_dir))
    zone_contract = {
        "farm": EXPECTED_FARM_ID,
        "camera": EXPECTED_CAMERA_ID,
        "clip": COMBINED_SAMPLE_ID,
        "outside_zone": "path",
    }
    for key, expected in zone_contract.items():
        actual = zones.get(key)
        if actual != expected:
            raise RuntimeError(
                f"Zone checkpoint mismatch for {key}: expected={expected!r}, actual={actual!r}"
            )
    zone_rows = zones.get("zones")
    if not isinstance(zone_rows, list) or not zone_rows:
        raise RuntimeError("Zone checkpoint contains no polygons")
    zone_ids: set[str] = set()
    for index, item in enumerate(zone_rows):
        if not isinstance(item, dict):
            raise RuntimeError(f"Zone row {index} is not an object")
        zone_id = str(item.get("zone_id", "")).strip()
        zone_type = str(item.get("zone_type", "")).strip()
        polygon = item.get("polygon")
        if not zone_id or zone_id in zone_ids or not zone_type:
            raise RuntimeError(f"Zone row {index} has an invalid or duplicate identity")
        if not isinstance(polygon, list) or len(polygon) < 3:
            raise RuntimeError(f"Zone row {index} has an invalid polygon")
        for point in polygon:
            if (
                not isinstance(point, list)
                or len(point) != 2
                or not all(isinstance(value, (int, float)) and math.isfinite(float(value)) for value in point)
            ):
                raise RuntimeError(f"Zone row {index} has an invalid polygon point")
        zone_ids.add(zone_id)

    validate_analysis_config(config_path(sample_dir))
    return manifest, frame_limit


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def staging_directory(target: Path, label: str) -> Path:
    return target.parent / f".{target.name}.{os.getpid()}.{label}.tmp"


def require_publishable_target(target: Path, overwrite: bool) -> None:
    if target.exists() and not target.is_dir():
        raise RuntimeError(f"Output target exists and is not a directory: {target}")
    if target.exists() and not overwrite:
        raise RuntimeError(f"Output target already exists; pass --overwrite to replace it: {target}")


def require_disjoint_targets(first: Path, second: Path) -> None:
    first_resolved = first.resolve()
    second_resolved = second.resolve()
    if (
        first_resolved == second_resolved
        or first_resolved in second_resolved.parents
        or second_resolved in first_resolved.parents
    ):
        raise RuntimeError(
            f"SNA input and analysis targets must be separate, non-nested directories: "
            f"input={first_resolved}, output={second_resolved}"
        )


def prepare_staging_directory(target: Path, label: str) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = staging_directory(target, label)
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    return staging


def publish_directory(staging: Path, target: Path, overwrite: bool) -> None:
    require_publishable_target(target, overwrite)
    backup = target.parent / f".{target.name}.{os.getpid()}.backup"
    if backup.exists():
        shutil.rmtree(backup)
    if target.exists():
        os.replace(target, backup)
    try:
        os.replace(staging, target)
    except Exception:
        if backup.exists() and not target.exists():
            os.replace(backup, target)
        raise
    if backup.exists():
        shutil.rmtree(backup)


def publish_directory_pair(
    first_staging: Path,
    first_target: Path,
    second_staging: Path,
    second_target: Path,
    overwrite: bool,
) -> None:
    pairs = ((first_staging, first_target), (second_staging, second_target))
    for staging, target in pairs:
        if not staging.is_dir():
            raise RuntimeError(f"Staged output directory is missing: {staging}")
        require_publishable_target(target, overwrite)
    backups = {
        target: target.parent / f".{target.name}.{os.getpid()}.backup"
        for _staging, target in pairs
    }
    for backup in backups.values():
        if backup.exists():
            shutil.rmtree(backup)

    published: list[Path] = []
    try:
        for _staging, target in pairs:
            if target.exists():
                os.replace(target, backups[target])
        for staging, target in pairs:
            os.replace(staging, target)
            published.append(target)
    except Exception:
        for target in reversed(published):
            if target.exists():
                shutil.rmtree(target)
        for _staging, target in pairs:
            backup = backups[target]
            if backup.exists() and not target.exists():
                os.replace(backup, target)
        raise
    for backup in backups.values():
        if backup.exists():
            shutil.rmtree(backup)


def map_display_rotation(
    point: tuple[float, float],
    width: float,
    height: float,
    quarter_turns: int,
) -> tuple[float, float] | None:
    turns = int(quarter_turns) % 4
    if turns == 0:
        return point if 0 <= point[0] < width and 0 <= point[1] < height else None
    cx = width / 2.0
    cy = height / 2.0
    dx = point[0] - cx
    dy = point[1] - cy
    if turns == 1:
        rotated = (cx + dy, cy - dx)
    elif turns == 2:
        rotated = (cx - dx, cy - dy)
    else:
        rotated = (cx - dy, cy + dx)
    if 0 <= rotated[0] < width and 0 <= rotated[1] < height:
        return rotated
    return None


def display_anchor_for_point(
    sample: dict[str, Any],
    payload: dict[str, Any],
    point: dict[str, Any],
    map_quarter_turns: int,
) -> tuple[float, float] | None:
    geometry = get_geometry(sample, int(payload["meta"].get("planRotationQuarterTurns", 0)))
    adjusted = realtime_adjusted_payload_point(point, geometry, "simple")
    return map_display_rotation(
        adjusted,
        float(payload["meta"]["width"]),
        float(payload["meta"]["height"]),
        map_quarter_turns,
    )


def map_quarter_turns_for_payload(sample: dict[str, Any], payload: dict[str, Any]) -> int:
    type_id = str(payload["meta"].get("mappingTypeId", "H"))
    return get_saved_rotation("map", sample["farmId"], sample["cameraId"], type_id)


def active_probability(item: dict[str, Any]) -> tuple[float, float]:
    gate = max(0.0, min(1.0, float(item.get("stage1Prob", 0.0))))
    friendly = max(0.0, min(1.0, float(item.get("friendlyScore", 0.0))))
    unfriendly = max(0.0, min(1.0, float(item.get("unfriendlyScore", 0.0))))
    return gate * friendly, gate * unfriendly


def merge_payload_identities(
    identities: dict[str, dict[str, Any]],
    payload: dict[str, Any],
    source_clip_id: str,
) -> None:
    for item in payload["meta"].get("globalIdentities", []):
        global_id = int(item["globalTrackId"])
        global_uuid = str(item["globalTrackUuid"])
        display_id = str(item["displayGlobalId"])
        expected_display = f"G{global_id + 1:04d}"
        if display_id != expected_display:
            raise RuntimeError(
                f"Global display label mismatch: uuid={global_uuid}, expected={expected_display}, actual={display_id}"
            )
        current = identities.setdefault(
            global_uuid,
            {
                "global_track_uuid": global_uuid,
                "global_track_id": global_id,
                "display_global_id": display_id,
                "id_statuses": set(),
                "local_track_ids_by_clip": {},
            },
        )
        if int(current["global_track_id"]) != global_id or str(current["display_global_id"]) != display_id:
            raise RuntimeError(f"Global identity mapping is not bijective for {global_uuid}")
        current["id_statuses"].update(str(value) for value in item.get("idStatuses", []))
        local_by_clip = current["local_track_ids_by_clip"]
        local_by_clip.setdefault(source_clip_id, set()).update(int(value) for value in item.get("localTrackIds", []))

    by_global_id: dict[int, str] = {}
    by_display: dict[str, str] = {}
    for global_uuid, item in identities.items():
        global_id = int(item["global_track_id"])
        display_id = str(item["display_global_id"])
        if by_global_id.setdefault(global_id, global_uuid) != global_uuid:
            raise RuntimeError(f"global_track_id is not bijective: {global_id}")
        if by_display.setdefault(display_id, global_uuid) != global_uuid:
            raise RuntimeError(f"display_global_id is not bijective: {display_id}")


def point_identity(
    point: dict[str, Any],
    identities: dict[str, dict[str, Any]],
    sample_id: str,
    local_frame: int,
) -> tuple[int, str, str]:
    local_track_id = int(point["trackId"])
    global_uuid = str(point.get("globalTrackUuid", ""))
    display_id = str(point.get("displayGlobalId", ""))
    global_id = int(point.get("globalTrackId", -1))
    identity = identities.get(global_uuid)
    if identity is None:
        raise RuntimeError(
            f"Payload point has no strict global identity: sample={sample_id}, frame={local_frame}, track={local_track_id}"
        )
    if (
        int(identity["global_track_id"]) != global_id
        or str(identity["display_global_id"]) != display_id
        or display_id != f"G{global_id + 1:04d}"
    ):
        raise RuntimeError(
            f"Payload point identity disagrees with the strict mapping: "
            f"sample={sample_id}, frame={local_frame}, track={local_track_id}"
        )
    return local_track_id, global_uuid, display_id


def serialized_identities(
    identities: dict[str, dict[str, Any]],
    production: bool,
    generation_id: str,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for item in sorted(identities.values(), key=lambda value: int(value["global_track_id"])):
        local_by_clip = {
            str(clip_id): sorted(int(track_id) for track_id in track_ids)
            for clip_id, track_ids in sorted(item["local_track_ids_by_clip"].items())
        }
        rows.append(
            {
                "global_track_uuid": str(item["global_track_uuid"]),
                "global_track_id": int(item["global_track_id"]),
                "display_global_id": str(item["display_global_id"]),
                "id_statuses": sorted(str(status) for status in item["id_statuses"]),
                "local_track_ids_by_clip": local_by_clip,
            }
        )
    if production:
        if len(rows) != EXPECTED_GLOBAL_IDENTITY_COUNT:
            raise RuntimeError(
                f"Expected exactly {EXPECTED_GLOBAL_IDENTITY_COUNT} global identities; found {len(rows)}"
            )
        if [row["global_track_id"] for row in rows] != list(range(EXPECTED_GLOBAL_IDENTITY_COUNT)):
            raise RuntimeError("Global identities are not exactly 0..61")
        if [row["display_global_id"] for row in rows] != [
            f"G{index:04d}" for index in range(1, EXPECTED_GLOBAL_IDENTITY_COUNT + 1)
        ]:
            raise RuntimeError("Display global identities are not exactly G0001..G0062")
    return {
        "schema_version": 1,
        "generation_id": generation_id,
        "sequence_id": REID_SEQUENCE_ID,
        "combined_sample_id": COMBINED_SAMPLE_ID,
        "identity_key": "global_track_uuid",
        "display_label": "display_global_id",
        "identities": rows,
    }


def zones_document(payload: dict[str, Any]) -> dict[str, Any]:
    resource = payload["resourceFloorplan"]
    zones = [
        {
            "zone_id": str(zone["zoneId"]),
            "zone_type": str(zone["zoneType"]),
            "label": str(zone.get("label") or zone.get("zoneType") or zone["zoneId"]),
            "color": str(zone.get("color") or "#0b7285"),
            "polygon": zone["points"],
        }
        for zone in resource.get("zones", [])
    ]
    return {
        "farm": EXPECTED_FARM_ID,
        "camera": EXPECTED_CAMERA_ID,
        "clip": COMBINED_SAMPLE_ID,
        "coordinate_system": "webuil_simple_display_floorplan_pixel_map_rotation_no_semantic_warp",
        "source": resource.get("path", ""),
        "outside_zone": "path",
        "zones": zones,
    }


def write_combined_config(path: Path, frame_limit: int) -> None:
    metadata = dict(SNA_METADATA)
    if frame_limit > 0:
        metadata["frame_limit"] = int(frame_limit)
    config = {
        "metadata": metadata,
        "analysis": {
            "farm": EXPECTED_FARM_ID,
            "camera": EXPECTED_CAMERA_ID,
            "clip": COMBINED_SAMPLE_ID,
            "start_time_s": None,
            "end_time_s": None,
        },
        "time": {"fps": EXPECTED_SEQUENCE_FPS, "default_dt_s": None},
        "network": {
            "directed": False,
            "use_probability_weights": True,
            "use_track_confidence": True,
            "use_interaction_confidence": False,
            "pair_zone_mode": "same_or_cross",
            "zone_level": "zone_type",
            "complete_pair_grid_from_trajectories": True,
            "min_opportunity_s": 0.0,
            "min_visible_time_s": 0.0,
            "eps": 1.0e-9,
        },
        "zones": {
            "smoothing_s": 0.0,
            "unknown_policy": "exclude_from_zone_metrics",
            "outside_zone": "path",
        },
        "community": {
            "enabled": True,
            "window_s": COMMUNITY_WINDOW_S,
            "step_s": COMMUNITY_STEP_S,
            "min_visible_time_s": 30.0,
            "min_edges_per_window": 1,
            "algorithm": "louvain_or_greedy",
        },
        "isolation": {
            "alone_definition": "same_zone",
            "weights": {
                "alone_fraction": 0.4,
                "low_friendly_sociality": 0.3,
                "low_partner_diversity": 0.3,
            },
        },
        "report": {
            "time_bin_s": 60.0,
            "write_temporal_outputs": False,
            "edge_top_n": 20,
            "network_layout": "spring",
            "layout_random_seed": 123,
        },
        "events": {
            "enabled": False,
            "event_start_threshold": 0.5,
            "event_end_threshold": 0.3,
            "max_gap_s": 1.0,
            "min_event_duration_s": 2.0,
        },
        "outputs": {"write_adjacency_matrices": True, "write_plots": True},
    }
    write_config(path, config)


def payload_frame_item(payload: dict[str, Any], local_frame: int) -> dict[str, Any]:
    frame_min = int(payload["meta"]["frameMin"])
    frame_max = int(payload["meta"]["frameMax"])
    items = payload["frames"]
    if len(items) != frame_max - frame_min + 1:
        raise RuntimeError("WebUI payload frame timeline is not contiguous")
    index = local_frame - frame_min
    if index < 0 or index >= len(items):
        raise RuntimeError(
            f"Canonical frame is absent from WebUI payload: frame={local_frame}, range=[{frame_min},{frame_max}]"
        )
    item = items[index]
    if int(item["frame"]) != local_frame:
        raise RuntimeError(
            f"WebUI payload frame index mismatch: expected={local_frame}, actual={item['frame']}"
        )
    return item


def build_combined_inputs(
    samples: list[dict[str, Any]],
    resources: Any,
    args: argparse.Namespace,
    staging_dir: Path,
) -> dict[str, Any]:
    from stage2_runtime import Stage2Runtime

    trajectory_file = trajectories_path(staging_dir)
    interaction_file = interactions_path(staging_dir)
    identities: dict[str, dict[str, Any]] = {}
    reference_zones: dict[str, Any] | None = None
    reference_zone_fingerprint: str | None = None
    remaining_frames = int(args.frame_limit) if int(args.frame_limit) > 0 else None
    last_global_frame: int | None = None
    last_global_time: float | None = None
    trajectory_row_count = 0
    interaction_row_count = 0
    canonical_frame_count = 0
    trajectory_frame_count = 0
    frozen_trajectory_row_count = 0
    sample_rows: list[dict[str, Any]] = []
    generation_id = str(uuid.uuid4())

    with trajectory_file.open("w", encoding="utf-8", newline="") as trajectory_handle, interaction_file.open(
        "w", encoding="utf-8", newline=""
    ) as interaction_handle:
        trajectory_writer = csv.DictWriter(trajectory_handle, fieldnames=TRAJECTORY_COLUMNS)
        interaction_writer = csv.DictWriter(interaction_handle, fieldnames=INTERACTION_COLUMNS)
        trajectory_writer.writeheader()
        interaction_writer.writeheader()

        for sample in samples:
            if remaining_frames is not None and remaining_frames <= 0:
                break
            sample_id = str(sample["id"])
            source_clip_id = str(sample["sourceVideoId"])
            playback = sample["playbackSegment"]
            base_start = int(playback["baseStartFrame"])
            base_end = int(playback["baseEndFrame"])
            write_end = base_end
            if remaining_frames is not None:
                write_end = min(write_end, base_start + remaining_frames - 1)
            frame_count = write_end - base_start + 1
            if frame_count <= 0:
                continue

            log(f"[sample] {sample_id} building strict WebUI/re-ID payload")
            payload = build_payload(sample, PayloadProgress(sample_id))
            if str(payload["meta"].get("reidSequenceId")) != REID_SEQUENCE_ID:
                raise RuntimeError(f"Payload re-ID sequence mismatch for {sample_id}")
            if str(payload["meta"].get("reidClipId")) != source_clip_id:
                raise RuntimeError(f"Payload re-ID clip mismatch for {sample_id}")
            if not math.isclose(
                float(payload["meta"]["fps"]), EXPECTED_SEQUENCE_FPS, rel_tol=0.0, abs_tol=1e-9
            ):
                raise RuntimeError(f"Payload FPS mismatch for {sample_id}")
            merge_payload_identities(identities, payload, source_clip_id)

            current_zones = zones_document(payload)
            current_fingerprint = json.dumps(current_zones, ensure_ascii=False, sort_keys=True)
            if reference_zones is None:
                reference_zones = current_zones
                reference_zone_fingerprint = current_fingerprint
            elif current_fingerprint != reference_zone_fingerprint:
                raise RuntimeError(f"Farm1/Gopro1 SNA zone geometry differs across shards: {sample_id}")

            geometry = get_geometry(sample, int(payload["meta"].get("planRotationQuarterTurns", 0)))
            map_quarter_turns = map_quarter_turns_for_payload(sample, payload)
            runtime = Stage2Runtime(
                stage2_code_dir=STAGE2_CODE_DIR,
                source_dir=sample["sourceDir"],
                resources=resources,
            )
            runtime_meta = runtime.meta()
            if not str(runtime_meta.get("device", "")).startswith("cuda"):
                raise RuntimeError(f"Stage2 did not resolve to CUDA for {sample_id}: {runtime_meta.get('device')}")
            if base_start > int(playback["segmentStartFrame"]):
                log(
                    f"[warm-up] {sample_id}: Stage2 will step physical padding "
                    f"{playback['segmentStartFrame']}..{base_start - 1} before canonical output"
                )

            sample_trajectory_start = trajectory_row_count
            sample_interaction_start = interaction_row_count
            with tqdm(
                range(base_start, write_end + 1),
                desc=f"sna-stage2 {sample_id}",
                unit="frame",
                ascii=True,
                dynamic_ncols=False,
                ncols=100,
                mininterval=1.0,
                file=sys.stdout,
            ) as pbar:
                for local_frame in pbar:
                    frame_item = payload_frame_item(payload, local_frame)
                    global_frame, global_time = reid_global_timeline(source_clip_id, local_frame)
                    if last_global_frame is not None and global_frame <= last_global_frame:
                        raise RuntimeError(
                            "Canonical global timeline is duplicated or reversed: "
                            f"previous={last_global_frame}, current={global_frame}, sample={sample_id}"
                        )
                    if last_global_time is not None and global_time <= last_global_time:
                        raise RuntimeError(
                            "Canonical global time is duplicated or reversed: "
                            f"previous={last_global_time}, current={global_time}, sample={sample_id}"
                        )
                    last_global_frame = global_frame
                    last_global_time = global_time
                    canonical_frame_count += 1

                    endpoints: dict[int, dict[str, Any]] = {}
                    frame_global_uuids: set[str] = set()
                    for point in frame_item["points"]:
                        local_track_id, global_uuid, display_id = point_identity(
                            point, identities, sample_id, local_frame
                        )
                        if local_track_id in endpoints:
                            raise RuntimeError(
                                f"Duplicate local track in payload frame: sample={sample_id}, "
                                f"frame={local_frame}, track={local_track_id}"
                            )
                        if global_uuid in frame_global_uuids:
                            raise RuntimeError(
                                f"Duplicate global identity in payload frame: sample={sample_id}, "
                                f"frame={local_frame}, uuid={global_uuid}"
                            )
                        frame_global_uuids.add(global_uuid)
                        anchor = display_anchor_for_point(sample, payload, point, map_quarter_turns)
                        if anchor is None:
                            raise RuntimeError(
                                f"SNA trajectory anchor falls outside the display map: "
                                f"sample={sample_id}, frame={local_frame}, track={local_track_id}"
                            )
                        adjusted = realtime_adjusted_payload_point(point, geometry, "simple")
                        endpoints[local_track_id] = {
                            "local_track_id": local_track_id,
                            "global_uuid": global_uuid,
                            "display_id": display_id,
                            "adjusted": adjusted,
                        }
                        frozen = bool(point.get("frozen", False))
                        trajectory_writer.writerow(
                            {
                                "farm": EXPECTED_FARM_ID,
                                "camera": EXPECTED_CAMERA_ID,
                                "clip": COMBINED_SAMPLE_ID,
                                "frame": global_frame,
                                "time_s": global_time,
                                "cow_id": global_uuid,
                                "anchor_x": float(anchor[0]),
                                "anchor_y": float(anchor[1]),
                                "track_conf": float(point.get("score", 1.0)),
                                "frozen": frozen,
                                "display_global_id": display_id,
                                "source_clip_id": source_clip_id,
                                "source_sample_id": sample_id,
                                "local_frame": local_frame,
                                "local_track_id": local_track_id,
                            }
                        )
                        trajectory_row_count += 1
                        frozen_trajectory_row_count += int(frozen)

                    local_track_ids = sorted(endpoints)
                    if local_track_ids:
                        trajectory_frame_count += 1
                    if len(local_track_ids) < 2:
                        continue

                    active: dict[tuple[int, int], tuple[float, float]] = {}
                    for item in runtime.raw_interactions_for_frame(local_frame):
                        key = tuple(sorted((int(item["tidA"]), int(item["tidB"]))))
                        pf, pu = active_probability(item)
                        old_pf, old_pu = active.get(key, (0.0, 0.0))
                        active[key] = (max(old_pf, pf), max(old_pu, pu))
                    opportunity_pairs = runtime.geometry_candidate_pairs_for_frame(local_frame)

                    for local_i, local_j in itertools.combinations(local_track_ids, 2):
                        local_key = (local_i, local_j)
                        endpoint_i = endpoints[local_i]
                        endpoint_j = endpoints[local_j]
                        red_blocked = geometry.segment_crosses_red(
                            endpoint_i["adjusted"], endpoint_j["adjusted"]
                        )
                        opportunity_eligible = local_key in opportunity_pairs and not red_blocked
                        pf, pu = active.get(local_key, (0.0, 0.0)) if opportunity_eligible else (0.0, 0.0)
                        ordered_endpoints = sorted(
                            (endpoint_i, endpoint_j), key=lambda endpoint: str(endpoint["global_uuid"])
                        )
                        global_i, global_j = ordered_endpoints
                        if global_i["global_uuid"] == global_j["global_uuid"]:
                            raise RuntimeError(
                                f"Local Stage2 pair collapses to one global identity: "
                                f"sample={sample_id}, frame={local_frame}, pair={local_key}"
                            )
                        interaction_writer.writerow(
                            {
                                "farm": EXPECTED_FARM_ID,
                                "camera": EXPECTED_CAMERA_ID,
                                "clip": COMBINED_SAMPLE_ID,
                                "frame": global_frame,
                                "time_s": global_time,
                                "cow_i": global_i["global_uuid"],
                                "cow_j": global_j["global_uuid"],
                                "p_friendly": round(float(pf), 9),
                                "p_unfriendly": round(float(pu), 9),
                                "opportunity_eligible": int(opportunity_eligible),
                                "interaction_conf": 1.0,
                                "display_global_id_i": global_i["display_id"],
                                "display_global_id_j": global_j["display_id"],
                                "source_clip_id": source_clip_id,
                                "source_sample_id": sample_id,
                                "local_frame": local_frame,
                                "local_track_id_i": global_i["local_track_id"],
                                "local_track_id_j": global_j["local_track_id"],
                            }
                        )
                        interaction_row_count += 1

            sample_rows.append(
                {
                    "sample_id": sample_id,
                    "source_clip_id": source_clip_id,
                    "source_path": str(sample["sourcePath"]),
                    "source_dir": str(sample["sourceDir"]),
                    "shard_index": int(sample["shardIndex"]),
                    "shard_count": int(sample["shardCount"]),
                    "segment_start_frame": int(playback["segmentStartFrame"]),
                    "segment_end_frame": int(playback["segmentEndFrame"]),
                    "canonical_start_frame": base_start,
                    "canonical_end_frame": write_end,
                    "canonical_frame_count": frame_count,
                    "trajectory_row_count": trajectory_row_count - sample_trajectory_start,
                    "interaction_row_count": interaction_row_count - sample_interaction_start,
                }
            )
            if remaining_frames is not None:
                remaining_frames -= frame_count
            del runtime, payload
            gc.collect()

        trajectory_handle.flush()
        interaction_handle.flush()

    if reference_zones is None:
        raise RuntimeError("No canonical frames were selected for the combined SNA input")
    production = int(args.frame_limit) <= 0
    if production and len(sample_rows) != EXPECTED_SAMPLE_COUNT:
        raise RuntimeError(
            f"Production combined input did not process all shards: {len(sample_rows)}/{EXPECTED_SAMPLE_COUNT}"
        )
    identity_document = serialized_identities(
        identities,
        production=production,
        generation_id=generation_id,
    )
    write_json(zones_path(staging_dir), reference_zones)
    write_json(identities_path(staging_dir), identity_document)
    write_combined_config(config_path(staging_dir), int(args.frame_limit))

    manifest = {
        "schema_version": 1,
        "generation_id": generation_id,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "combined_sample_id": COMBINED_SAMPLE_ID,
        "farm": EXPECTED_FARM_ID,
        "camera": EXPECTED_CAMERA_ID,
        "reid_sequence_id": REID_SEQUENCE_ID,
        "timeline_fps": EXPECTED_SEQUENCE_FPS,
        "identity_key": "global_track_uuid",
        "display_label": "display_global_id",
        "canonical_frame_policy": SNA_METADATA["canonical_frame_policy"],
        "frame_limit": int(args.frame_limit) if int(args.frame_limit) > 0 else None,
        "canonical_frame_count": canonical_frame_count,
        "trajectory_frame_count": trajectory_frame_count,
        "trajectory_row_count": trajectory_row_count,
        "frozen_trajectory_row_count": frozen_trajectory_row_count,
        "interaction_row_count": interaction_row_count,
        "global_identity_count": len(identity_document["identities"]),
        "samples": sample_rows,
    }
    write_json(staging_dir / "generation_manifest.json", manifest)
    log(
        f"[input-complete] frames={canonical_frame_count:,} trajectories={trajectory_row_count:,} "
        f"interactions={interaction_row_count:,} identities={len(identity_document['identities'])}"
    )
    return manifest


def read_csv_identity_set(path: Path, column: str) -> set[str]:
    values: set[str] = set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if column not in (reader.fieldnames or []):
            raise RuntimeError(f"Required identity column is missing: {path}: {column}")
        for row in reader:
            value = str(row[column]).strip()
            if not value:
                raise RuntimeError(f"Empty identity value in {path}: {column}")
            values.add(value)
    return values


def validate_edge_level(path: Path, production: bool, valid_uuids: set[str]) -> set[str]:
    zones: set[str] = set()
    row_count = 0
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"zone", "expected_seconds", "opportunity_seconds", "normalized_rate", "cow_i", "cow_j"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise RuntimeError(f"edge_level.csv is missing columns: {sorted(missing)}")
        for row in reader:
            row_count += 1
            cow_i = str(row["cow_i"])
            cow_j = str(row["cow_j"])
            if cow_i == cow_j:
                raise RuntimeError(f"Self edge survived SNA validation: {cow_i}")
            if cow_i not in valid_uuids or cow_j not in valid_uuids:
                raise RuntimeError(f"edge_level.csv contains an unknown global UUID: {cow_i}/{cow_j}")
            expected = float(row["expected_seconds"])
            opportunity = float(row["opportunity_seconds"])
            rate = float(row["normalized_rate"])
            if expected < -1e-9 or expected > opportunity + 1e-6:
                raise RuntimeError(
                    f"Invalid expected/opportunity edge values: expected={expected}, opportunity={opportunity}"
                )
            if rate < -1e-6 or rate > 1.0 + 1e-6:
                raise RuntimeError(f"Invalid normalized edge rate: {rate}")
            zones.add(str(row["zone"]))
    if row_count == 0:
        raise RuntimeError("edge_level.csv is empty")
    if production and zones != EXPECTED_ZONE_NETWORKS:
        raise RuntimeError(
            f"Combined SNA does not have the six reference figure zones: "
            f"expected={sorted(EXPECTED_ZONE_NETWORKS)}, actual={sorted(zones)}"
        )
    return zones


def rewrite_analysis_summary_paths(staging: Path, target: Path) -> dict[str, Any]:
    path = staging / "analysis_summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))

    def rewrite(value: Any) -> Any:
        if isinstance(value, str) and value.startswith(str(staging)):
            return str(target) + value[len(str(staging)) :]
        if isinstance(value, dict):
            return {key: rewrite(item) for key, item in value.items()}
        if isinstance(value, list):
            return [rewrite(item) for item in value]
        return value

    rewritten = rewrite(summary)
    write_json(path, rewritten)
    return rewritten


def validate_analysis_outputs(
    staging: Path,
    target: Path,
    input_dir: Path,
    input_manifest: dict[str, Any],
    frame_limit: int,
) -> dict[str, Any]:
    summary = rewrite_analysis_summary_paths(staging, target)
    if int(summary.get("n_dropped_self_pairs", -1)) != 0:
        raise RuntimeError(f"SNA core dropped self pairs: {summary.get('n_dropped_self_pairs')}")
    if int(summary.get("n_dropped_missing_trajectory", -1)) != 0:
        raise RuntimeError(
            f"SNA core dropped interaction rows with missing trajectory: "
            f"{summary.get('n_dropped_missing_trajectory')}"
        )
    if int(summary.get("n_frames", -1)) != int(input_manifest["trajectory_frame_count"]):
        raise RuntimeError(
            f"SNA frame count mismatch: input={input_manifest['trajectory_frame_count']}, "
            f"analysis={summary.get('n_frames')}"
        )
    if int(summary.get("n_trajectory_rows", -1)) != int(input_manifest["trajectory_row_count"]):
        raise RuntimeError(
            f"SNA trajectory row count mismatch: input={input_manifest['trajectory_row_count']}, "
            f"analysis={summary.get('n_trajectory_rows')}"
        )
    if int(summary.get("n_interaction_rows", -1)) != int(input_manifest["interaction_row_count"]):
        raise RuntimeError(
            f"SNA interaction row count mismatch: input={input_manifest['interaction_row_count']}, "
            f"analysis={summary.get('n_interaction_rows')}"
        )
    expected_uuids = read_csv_identity_set(staging / "cow_time_budget.csv", "cow_id")
    node_uuids = read_csv_identity_set(staging / "node_descriptors.csv", "cow_id")
    layout_uuids = read_csv_identity_set(staging / "report_network_layout.csv", "cow_id")
    identity_document = json.loads(identities_path(input_dir).read_text(encoding="utf-8"))
    mapped_uuids = {
        str(item["global_track_uuid"])
        for item in identity_document.get("identities", [])
    }
    if not expected_uuids.issubset(mapped_uuids):
        raise RuntimeError("SNA output contains a cow UUID absent from global_identities.json")
    if expected_uuids != node_uuids or expected_uuids != layout_uuids:
        raise RuntimeError("SNA cow identity sets differ across time-budget/node/layout outputs")
    if frame_limit <= 0 and expected_uuids != mapped_uuids:
        raise RuntimeError(
            f"Combined SNA output does not contain the exact {EXPECTED_GLOBAL_IDENTITY_COUNT}-identity mapping: "
            f"outputs={len(expected_uuids)}, mapping={len(mapped_uuids)}"
        )
    if int(summary.get("n_cows", -1)) != len(expected_uuids):
        raise RuntimeError(
            f"SNA cow count mismatch: summary={summary.get('n_cows')}, outputs={len(expected_uuids)}"
        )
    zones = validate_edge_level(
        staging / "edge_level.csv",
        production=frame_limit <= 0,
        valid_uuids=mapped_uuids,
    )
    return {"summary": summary, "cow_uuids": sorted(expected_uuids), "edge_zones": sorted(zones)}


def build_analysis_output(
    input_dir: Path,
    staging: Path,
    output_target: Path,
    input_manifest: dict[str, Any],
    frame_limit: int,
    analysis_config_path: Path,
    published_input_dir: Path,
    published_analysis_config_path: Path,
) -> None:
    log(f"[analysis] combined sample -> {output_target}")
    resolved_config = validate_analysis_config(analysis_config_path)
    source_config_sha256 = sha256_file(analysis_config_path)
    resolved_config_path = staging / "analysis_config_resolved.yaml"
    write_config(resolved_config_path, resolved_config)
    analysis_started = time.monotonic()
    heartbeat_stop = threading.Event()

    def heartbeat() -> None:
        while not heartbeat_stop.wait(10.0):
            elapsed = time.monotonic() - analysis_started
            log(f"[analysis-heartbeat] running elapsed={elapsed:.0f}s")

    heartbeat_thread = threading.Thread(target=heartbeat, name="combined-sna-heartbeat", daemon=True)
    heartbeat_thread.start()
    try:
        run_pipeline(
            trajectories_path(input_dir),
            interactions_path(input_dir),
            zones_path(input_dir),
            resolved_config_path,
            staging,
        )
    finally:
        heartbeat_stop.set()
        heartbeat_thread.join(timeout=1.0)
    validation = validate_analysis_outputs(
        staging,
        output_target,
        input_dir,
        input_manifest,
        int(frame_limit),
    )
    write_json(
        staging / "generation_manifest.json",
        {
            **input_manifest,
            "input_dir": str(published_input_dir),
            "output_dir": str(output_target),
            "analysis_run_id": str(uuid.uuid4()),
            "analysis_generated_at": datetime.now().isoformat(timespec="seconds"),
            "analysis_config_path": str(published_analysis_config_path),
            "analysis_config_sha256": source_config_sha256,
            "analysis_config_resolved_path": str(output_target / resolved_config_path.name),
            "analysis_config_resolved_sha256": sha256_file(resolved_config_path),
            "analysis_validation": validation,
        },
    )


def run_analysis_only(args: argparse.Namespace) -> int:
    if int(args.frame_limit) != 0:
        raise RuntimeError(
            "--analysis-only reads the frame limit from the checkpoint; do not pass --frame-limit"
        )
    if args.profile != "auto" or args.inference_batch_size is not None:
        raise RuntimeError(
            "--profile and --inference-batch-size belong to neural inference and cannot be used "
            "with --analysis-only"
        )

    input_target = combined_input_dir(Path(args.inputs_dir))
    output_target = combined_output_dir(Path(args.outdir))
    require_disjoint_targets(input_target, output_target)
    input_manifest, checkpoint_frame_limit = validate_input_checkpoint(input_target)
    require_nonproduction_analysis_output(checkpoint_frame_limit, Path(args.outdir))

    selected_config = (
        Path(args.analysis_config).expanduser()
        if args.analysis_config is not None
        else config_path(input_target)
    )
    analysis_config = selected_config.resolve()
    validate_analysis_config(analysis_config)
    output_resolved = output_target.resolve()
    if analysis_config == output_resolved or output_resolved in analysis_config.parents:
        raise RuntimeError(
            f"Analysis config cannot be inside the output directory that will be replaced: {analysis_config}"
        )
    require_publishable_target(output_target, args.overwrite)

    log("[mode] analysis-only (no CUDA or neural model loading)")
    log(f"[info] combined sample: {COMBINED_SAMPLE_ID}")
    log(f"[info] SNA input checkpoint: {input_target}")
    log(f"[info] SNA analysis config: {analysis_config}")
    log(f"[info] SNA output: {output_target}")
    log(
        f"[checkpoint] generation={input_manifest['generation_id']} "
        f"frames={int(input_manifest['canonical_frame_count']):,} "
        f"trajectories={int(input_manifest['trajectory_row_count']):,} "
        f"interactions={int(input_manifest['interaction_row_count']):,}"
    )
    if args.dry_run:
        log("[dry-run] checkpoint and analysis config are valid; analysis was not started")
        return 0

    output_staging = prepare_staging_directory(output_target, "analysis")
    try:
        build_analysis_output(
            input_target,
            output_staging,
            output_target,
            input_manifest,
            frame_limit=checkpoint_frame_limit,
            analysis_config_path=analysis_config,
            published_input_dir=input_target,
            published_analysis_config_path=analysis_config,
        )
        publish_directory(output_staging, output_target, args.overwrite)
    finally:
        if output_staging.exists():
            shutil.rmtree(output_staging)
    log(f"[done] analysis-only combined sample={COMBINED_SAMPLE_ID}")
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate one strict global-ID SNA dataset for all current Farm1/Gopro1 clips."
    )
    parser.add_argument(
        "--frame-limit",
        type=int,
        default=0,
        help="Limit the combined canonical timeline for a smoke run; requires non-production directories.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate/list the fixed combined input without writing files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Atomically replace existing combined SNA inputs/outputs.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--skip-analysis",
        "--neural-only",
        dest="skip_analysis",
        action="store_true",
        help="Run Stage2 neural inference and publish only the reusable SNA input checkpoint.",
    )
    mode.add_argument(
        "--analysis-only",
        action="store_true",
        help="Run only SNA from an existing input checkpoint; CUDA and neural models are not used.",
    )
    parser.add_argument("--inputs-dir", default=str(SNA_INPUT_DIR))
    parser.add_argument("--outdir", default=str(SNA_OUTPUT_DIR))
    parser.add_argument(
        "--analysis-config",
        default=None,
        help="Full YAML/JSON SNA config used by --analysis-only; defaults to checkpoint config.yaml.",
    )
    parser.add_argument("--profile", choices=["auto", "8gb", "40gb"], default="auto")
    parser.add_argument("--inference-batch-size", type=int, default=None)
    parser.add_argument("--verify-log-flush", action="store_true", help="Write flushed log lines and exit.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if int(args.frame_limit) < 0:
        raise RuntimeError("--frame-limit must be non-negative")
    if args.verify_log_flush:
        for index in range(1, 4):
            log(f"[flush-check] tick={index} time={datetime.now().isoformat(timespec='seconds')}")
            time.sleep(1.0)
        log("[flush-check] ok")
        return 0

    if args.analysis_only:
        return run_analysis_only(args)
    if args.analysis_config is not None:
        raise RuntimeError("--analysis-config can only be used with --analysis-only")

    samples = choose_combined_samples()
    input_target = combined_input_dir(Path(args.inputs_dir))
    output_target = combined_output_dir(Path(args.outdir))
    require_disjoint_targets(input_target, output_target)
    log("[mode] neural-only checkpoint generation" if args.skip_analysis else "[mode] full neural + SNA")
    log(f"[info] combined sample: {COMBINED_SAMPLE_ID}")
    log(f"[info] ordered shards: {', '.join(str(sample['id']) for sample in samples)}")
    log(f"[info] SNA input: {input_target}")
    log(f"[info] SNA output: {output_target}")
    if int(args.frame_limit) > 0:
        log(f"[info] smoke frame limit: {args.frame_limit}")
    if args.dry_run:
        return 0

    require_nonproduction_paths_for_frame_limit(args)
    require_publishable_target(input_target, args.overwrite)
    if args.skip_analysis and output_target.exists():
        raise RuntimeError(
            f"--skip-analysis refuses to publish new inputs beside an existing analysis generation: {output_target}"
        )
    if not args.skip_analysis:
        require_publishable_target(output_target, args.overwrite)
    require_cuda_visible_devices()
    from stage2_runtime import load_stage2_resources

    resources = load_stage2_resources(
        STAGE2_CODE_DIR,
        device="cuda",
        profile=args.profile,
        inference_batch_size=args.inference_batch_size,
    )

    input_staging = prepare_staging_directory(input_target, "inputs")
    output_staging: Path | None = None
    try:
        input_manifest = build_combined_inputs(samples, resources, args, input_staging)
        if args.skip_analysis:
            publish_directory(input_staging, input_target, args.overwrite)
        else:
            output_staging = prepare_staging_directory(output_target, "analysis")
            build_analysis_output(
                input_staging,
                output_staging,
                output_target,
                input_manifest,
                frame_limit=int(args.frame_limit),
                analysis_config_path=config_path(input_staging),
                published_input_dir=input_target,
                published_analysis_config_path=config_path(input_target),
            )
            publish_directory_pair(
                input_staging,
                input_target,
                output_staging,
                output_target,
                args.overwrite,
            )
    finally:
        if input_staging.exists():
            shutil.rmtree(input_staging)
        if output_staging is not None and output_staging.exists():
            shutil.rmtree(output_staging)
    log(f"[done] combined sample={COMBINED_SAMPLE_ID}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
