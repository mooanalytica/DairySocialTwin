from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import shutil
import uuid
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".cache" / "matplotlib"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap, to_rgba
from matplotlib.patches import Patch as MplPatch
from matplotlib.patches import Polygon as MplPolygon
import numpy as np
import pandas as pd
from PIL import Image
from scipy.optimize import linear_sum_assignment

from app import floorplan_annotation_file, min_area_rectangle, segment_intersects_polygon
from dairy_social.voc_colors import voc_cls_color_hexes
from reid_index import REID_SEQUENCE_ID


ROOT = Path(__file__).resolve().parent
SNA_INPUT_DIR = ROOT / "sna_inputs"
SNA_OUTPUT_DIR = ROOT / "sna_outputs"
SNA_TF_OUTPUT_DIR = ROOT / "sna_tf_outputs"
COMBINED_SAMPLE_ID = "F1_Gopro1_20250505"
EXPECTED_GLOBAL_IDENTITY_COUNT = 62
EXPECTED_COMMUNITY_WINDOW_COUNT = 92
FIGURE_01_GROUP_SIZE = 10
COW_FRAME_READ_CHUNK_ROWS = 200_000
TRAJECTORY_MAX_POINTS_PER_COW = 5_000
TRAJECTORY_TIME_GAP_FACTOR = 1.5
TRAJECTORY_SEGMENT_COLUMN = "_trajectory_segment"
FIGURE_01_MAX_ADJACENT_DISPLACEMENT_PX = 128.0
FIGURE_01_SAMPLE_MARKER_SIZE = 2.0
FIGURE_NET_DOMINANCE_THRESHOLD = 0.2
REQUIRED_SNA_OUTPUT_KEYS = (
    "identities",
    "input_generation",
    "output_generation",
    "zones",
    "summary",
    "cow_frame",
    "time_budget",
    "edge",
    "node",
    "zone_node",
    "layout",
    "community_windows",
    "community_summary",
    "edge_ranking",
    "events",
)
FIGURE_01_GROUPS = tuple(
    (start, min(start + FIGURE_01_GROUP_SIZE - 1, EXPECTED_GLOBAL_IDENTITY_COUNT))
    for start in range(1, EXPECTED_GLOBAL_IDENTITY_COUNT + 1, FIGURE_01_GROUP_SIZE)
)
FIXED_FIGURE_RELATIVE_PATHS = {
    *{
        f"figures/figure_01/figure_01_cows_{start:02d}_{end:02d}.png"
        for start, end in FIGURE_01_GROUPS
    },
    "figures/figure_02_visibility_zone_time_budget.png",
    "figures/figure_03_interaction_volume_by_zone.png",
    "figures/figure_04A_full_window_friendly_network.png",
    "figures/figure_04B_full_window_unfriendly_network.png",
    "figures/figure_06_adjacency_heatmaps.png",
    "figures/figure_07_descriptor_dashboard.png",
    "figures/figure_08B_community_similarity.png",
    "figures/figure_08_community_stability.png",
    "figures/figure_09_isolation_decomposition.png",
    "figures/figure_10_quality_control.png",
}
FIGURE_ZONE_ORDER = (
    "food",
    "rest",
    "wait_for_water",
    "water",
    "path",
    "cross_zone",
)
MISSING_COMMUNITY_COLOR = "#e9ecef"

ZONE_COLORS = {
    "food": "#2f9e44",
    "rest": "#5f3dc4",
    "wait_for_water": "#f08c00",
    "water": "#1c7ed6",
    "path": "#adb5bd",
    "cross_zone": "#495057",
    "unknown": "#868e96",
}
LAYER_COLORS = {
    "friendly": "#2f9e44",
    "unfriendly": "#e03131",
}
STRUCTURE_COLORS = {
    "obstacle": "#e03131",
    "resource": "#2f9e44",
}


def log(message: str) -> None:
    print(message, flush=True)


def read_csv(path: Path, **kwargs: Any) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Required CSV is missing: {path}")
    try:
        return pd.read_csv(path, **kwargs)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required JSON is missing: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"JSON file must contain an object: {path}")
    return data


def _trajectory_continuation_mask(
    frames: np.ndarray,
    times: np.ndarray,
    dts: np.ndarray,
    previous: tuple[int, float, float] | None,
) -> np.ndarray:
    """Return True where each row continues the preceding raw observation."""
    count = len(frames)
    continuation = np.zeros(count, dtype=bool)
    if count == 0:
        return continuation

    if previous is not None:
        previous_frame, previous_time, previous_dt = previous
        time_delta = float(times[0]) - float(previous_time)
        allowed_delta = TRAJECTORY_TIME_GAP_FACTOR * max(float(previous_dt), float(dts[0]))
        continuation[0] = (
            int(frames[0]) == int(previous_frame) + 1
            and time_delta > 0.0
            and time_delta <= allowed_delta + 1.0e-9
        )

    if count > 1:
        time_deltas = times[1:] - times[:-1]
        allowed_deltas = TRAJECTORY_TIME_GAP_FACTOR * np.maximum(dts[1:], dts[:-1])
        continuation[1:] = (
            (frames[1:] == frames[:-1] + 1)
            & (time_deltas > 0.0)
            & (time_deltas <= allowed_deltas + 1.0e-9)
        )
    return continuation


def _clean_cow_frame_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    out = chunk.copy()
    cow_ids = out["cow_id"].astype("string")
    if cow_ids.isna().any() or cow_ids.str.strip().eq("").any():
        raise ValueError("cow_frame_zone.csv contains an empty cow_id")
    out["cow_id"] = cow_ids.astype(str)
    for column in ("frame", "time_s", "dt_s", "anchor_x", "anchor_y", "visible_flag"):
        out[column] = pd.to_numeric(out[column], errors="coerce")
    finite = np.isfinite(
        out[["frame", "time_s", "dt_s", "anchor_x", "anchor_y"]].to_numpy(dtype=float)
    ).all(axis=1)
    valid = (
        finite
        & (out["dt_s"].to_numpy(dtype=float) > 0.0)
        & (out["visible_flag"].fillna(0).to_numpy(dtype=float) == 1.0)
    )
    out = out.loc[valid].copy()
    if not out.empty:
        out["frame"] = out["frame"].astype(np.int64)
        out["visible_flag"] = out["visible_flag"].astype(np.int8)
    return out


def _cow_frame_chunks(path: Path):
    columns = ["frame", "time_s", "dt_s", "cow_id", "anchor_x", "anchor_y", "visible_flag"]
    header = pd.read_csv(path, nrows=0)
    missing = sorted(set(columns) - set(header.columns))
    if missing:
        raise ValueError(f"Required cow-frame columns are missing from {path}: {missing}")
    yield from pd.read_csv(
        path,
        usecols=columns,
        dtype={"cow_id": str},
        chunksize=COW_FRAME_READ_CHUNK_ROWS,
    )


def _trajectory_segment_lengths(path: Path) -> dict[str, list[int]]:
    lengths: dict[str, list[int]] = {}
    current_lengths: dict[str, int] = {}
    previous_rows: dict[str, tuple[int, float, float]] = {}
    for chunk_index, raw_chunk in enumerate(_cow_frame_chunks(path), start=1):
        chunk = _clean_cow_frame_chunk(raw_chunk)
        for cow_id, group in chunk.groupby("cow_id", sort=False):
            cow = str(cow_id)
            frames = group["frame"].to_numpy(dtype=np.int64)
            times = group["time_s"].to_numpy(dtype=float)
            dts = group["dt_s"].to_numpy(dtype=float)
            continuation = _trajectory_continuation_mask(
                frames, times, dts, previous_rows.get(cow)
            )
            starts = np.r_[0, np.flatnonzero(~continuation[1:]) + 1]
            ends = np.r_[starts[1:], len(group)]
            for run_index, (start, end) in enumerate(zip(starts, ends)):
                run_length = int(end - start)
                continues_previous = run_index == 0 and bool(continuation[0])
                if continues_previous:
                    current_lengths[cow] += run_length
                else:
                    if cow in current_lengths:
                        lengths.setdefault(cow, []).append(current_lengths[cow])
                    current_lengths[cow] = run_length
            previous_rows[cow] = (int(frames[-1]), float(times[-1]), float(dts[-1]))
        if chunk_index % 10 == 0:
            log(
                f"[cow-frame sample] segment scan: "
                f"{chunk_index * COW_FRAME_READ_CHUNK_ROWS:,} source rows"
            )

    for cow, length in current_lengths.items():
        lengths.setdefault(cow, []).append(int(length))
    return lengths


def _segment_sample_offsets(lengths: list[int], max_points: int) -> list[np.ndarray]:
    if max_points <= 0:
        raise ValueError("max_points must be positive")
    if not lengths:
        return []
    lengths_array = np.asarray(lengths, dtype=np.int64)
    if (lengths_array <= 0).any():
        raise ValueError("trajectory segment lengths must be positive")

    minimum = np.where(lengths_array == 1, 1, 2)
    selected = np.ones(len(lengths_array), dtype=bool)
    if int(minimum.sum()) > max_points:
        selected[:] = False
        remaining = int(max_points)
        # Pathological fragmentation: retain complete endpoints for the longest
        # segments and omit whole shorter segments instead of reconnecting them.
        for index in sorted(
            range(len(lengths)), key=lambda item: (-int(lengths_array[item]), item)
        ):
            cost = int(minimum[index])
            if cost <= remaining:
                selected[index] = True
                remaining -= cost

    quotas = np.where(selected, minimum, 0)
    base_points = int(quotas.sum())
    capacities = np.where(selected, lengths_array - minimum, 0)
    extra_target = min(max_points - base_points, int(capacities.sum()))
    if extra_target > 0:
        raw = capacities.astype(float) * (float(extra_target) / float(capacities.sum()))
        extras = np.floor(raw).astype(np.int64)
        extras = np.minimum(extras, capacities)
        missing = int(extra_target - extras.sum())
        order = sorted(
            range(len(lengths)),
            key=lambda item: (-(raw[item] - extras[item]), -int(capacities[item]), item),
        )
        while missing > 0:
            changed = False
            for index in order:
                if extras[index] < capacities[index]:
                    extras[index] += 1
                    missing -= 1
                    changed = True
                    if missing == 0:
                        break
            if not changed:
                raise RuntimeError("Unable to allocate trajectory sampling quota")
        quotas += extras

    offsets: list[np.ndarray] = []
    for length, quota in zip(lengths_array, quotas):
        if quota == 0:
            offsets.append(np.empty(0, dtype=np.int64))
            continue
        selected_offsets = np.rint(
            np.linspace(0, int(length) - 1, num=int(quota))
        ).astype(np.int64)
        if len(np.unique(selected_offsets)) != int(quota):
            raise RuntimeError("Trajectory sampling produced duplicate offsets")
        offsets.append(selected_offsets)
    if sum(len(item) for item in offsets) > max_points:
        raise RuntimeError("Trajectory sampling exceeded its hard point limit")
    return offsets


def read_cow_frame_visual_sample(
    path: Path,
    max_points_per_cow: int = TRAJECTORY_MAX_POINTS_PER_COW,
) -> pd.DataFrame:
    """Read a bounded plot-only sample while preserving raw trajectory gaps."""
    if not path.is_file():
        raise FileNotFoundError(f"Required CSV is missing: {path}")
    segment_lengths = _trajectory_segment_lengths(path)
    offsets = {
        cow: _segment_sample_offsets(lengths, max_points_per_cow)
        for cow, lengths in segment_lengths.items()
    }
    omitted = sum(
        int(len(selected) == 0)
        for cow_offsets in offsets.values()
        for selected in cow_offsets
    )
    if omitted:
        log(
            f"[cow-frame sample] omitted {omitted:,} shortest fragments because their endpoints "
            f"would exceed {max_points_per_cow:,} points/cow"
        )

    previous_rows: dict[str, tuple[int, float, float]] = {}
    segment_indices: dict[str, int] = {}
    segment_positions: dict[str, int] = {}
    sampled: list[pd.DataFrame] = []
    for chunk_index, raw_chunk in enumerate(_cow_frame_chunks(path), start=1):
        chunk = _clean_cow_frame_chunk(raw_chunk)
        for cow_id, group in chunk.groupby("cow_id", sort=False):
            cow = str(cow_id)
            frames = group["frame"].to_numpy(dtype=np.int64)
            times = group["time_s"].to_numpy(dtype=float)
            dts = group["dt_s"].to_numpy(dtype=float)
            continuation = _trajectory_continuation_mask(
                frames, times, dts, previous_rows.get(cow)
            )
            starts = np.r_[0, np.flatnonzero(~continuation[1:]) + 1]
            ends = np.r_[starts[1:], len(group)]
            keep = np.zeros(len(group), dtype=bool)
            plot_segments = np.full(len(group), -1, dtype=np.int64)
            for run_index, (start, end) in enumerate(zip(starts, ends)):
                continues_previous = run_index == 0 and bool(continuation[0])
                if continues_previous:
                    segment_index = segment_indices[cow]
                    start_position = segment_positions[cow] + 1
                else:
                    segment_index = segment_indices.get(cow, -1) + 1
                    start_position = 0
                run_positions = np.arange(
                    start_position,
                    start_position + int(end - start),
                    dtype=np.int64,
                )
                selected_offsets = offsets[cow][segment_index]
                selected = np.isin(run_positions, selected_offsets, assume_unique=True)
                keep[start:end] = selected
                plot_segments[start:end] = segment_index
                segment_indices[cow] = segment_index
                segment_positions[cow] = int(run_positions[-1])
            if keep.any():
                selected_group = group.iloc[np.flatnonzero(keep)].copy()
                selected_group[TRAJECTORY_SEGMENT_COLUMN] = plot_segments[keep]
                sampled.append(selected_group)
            previous_rows[cow] = (int(frames[-1]), float(times[-1]), float(dts[-1]))
        if chunk_index % 10 == 0:
            log(
                f"[cow-frame sample] sampling pass: "
                f"{chunk_index * COW_FRAME_READ_CHUNK_ROWS:,} source rows"
            )

    columns = [
        "frame",
        "time_s",
        "dt_s",
        "cow_id",
        "anchor_x",
        "anchor_y",
        "visible_flag",
        TRAJECTORY_SEGMENT_COLUMN,
    ]
    if not sampled:
        return pd.DataFrame(columns=columns)
    out = pd.concat(sampled, ignore_index=True)
    out = out.sort_values(["frame", "time_s", "cow_id"], kind="stable").reset_index(drop=True)
    counts = out.groupby("cow_id", sort=False).size()
    if (counts > max_points_per_cow).any():
        raise RuntimeError(f"Plot-only cow-frame sample exceeds {max_points_per_cow:,} points/cow")
    log(f"[cow-frame sample] retained {len(out):,} plot rows across {len(counts):,} cows")
    return out[columns]


def output_path(base: Path, filename: str) -> Path:
    base.mkdir(parents=True, exist_ok=True)
    return base / filename


def save_figure(fig: plt.Figure, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def write_table(df: pd.DataFrame, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return str(path)


def zone_color(zone: str) -> str:
    return ZONE_COLORS.get(str(zone), "#748ffc")


def layer_color(layer: str) -> str:
    return LAYER_COLORS.get(str(layer), "#495057")


def contrasting_text_color(color: Any, alpha: float = 1.0) -> str:
    red, green, blue, color_alpha = to_rgba(color)
    effective_alpha = float(alpha) * float(color_alpha)
    if not math.isfinite(effective_alpha) or effective_alpha < 0.0 or effective_alpha > 1.0:
        raise ValueError(f"Text contrast alpha must be within [0, 1]: {effective_alpha}")
    red = red * effective_alpha + (1.0 - effective_alpha)
    green = green * effective_alpha + (1.0 - effective_alpha)
    blue = blue * effective_alpha + (1.0 - effective_alpha)

    def linear(channel: float) -> float:
        return channel / 12.92 if channel <= 0.04045 else ((channel + 0.055) / 1.055) ** 2.4

    luminance = 0.2126 * linear(red) + 0.7152 * linear(green) + 0.0722 * linear(blue)
    return "#000000" if luminance >= 0.42 else "#ffffff"


def marker_label_font_size(marker_size: float, label: str) -> float:
    size = float(marker_size)
    if not math.isfinite(size) or size <= 0:
        raise ValueError(f"Marker size must be finite and positive: {marker_size}")
    base = 0.55 * math.sqrt(size)
    if len(str(label)) > 2:
        base *= 2.0 / len(str(label))
    return max(5.0, min(14.0, base))


def cow_sort_key(cow_id: str) -> tuple[int, int | str]:
    cow = str(cow_id)
    if cow.isdigit():
        return (0, int(cow))
    return (1, cow)


def slug_label(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", str(value).strip()).strip("_").lower()
    return slug or "unknown"


def alphabetic_figure_suffix(index: int) -> str:
    value = int(index)
    if isinstance(index, bool) or value != index or value < 0:
        raise ValueError(f"Figure suffix index must be a non-negative integer: {index}")
    label = ""
    value += 1
    while value:
        value, remainder = divmod(value - 1, 26)
        label = chr(ord("A") + remainder) + label
    return label


def figure_05_key(index: int, zone: str) -> str:
    zone_label = str(zone).strip()
    if not zone_label:
        raise ValueError("Figure 5 zone label must not be empty")
    return f"figure_05{alphabetic_figure_suffix(index)}_{slug_label(zone_label)}_networks"


def expected_figure_relative_paths(zones: list[str]) -> set[str]:
    paths = set(FIXED_FIGURE_RELATIVE_PATHS)
    for index, zone in enumerate(zones):
        paths.add(f"figures/{figure_05_key(index, zone)}.png")
    return paths


def stable_seed(value: str) -> int:
    digest = hashlib.sha256(str(value).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def shuffled_voc_colors(seed: str, count: int) -> list[str]:
    colors = list(voc_cls_color_hexes())
    rng = random.Random(stable_seed(f"sna_tf_cow_voc_colors:{seed}"))
    indices: list[int] = []
    while len(indices) < count:
        pool = list(range(len(colors)))
        rng.shuffle(pool)
        indices.extend(pool)
    return [colors[index] for index in indices[:count]]


def cow_color_map(cows: Any, seed: str) -> dict[str, str]:
    ordered = sorted({str(cow) for cow in cows if pd.notna(cow)}, key=cow_sort_key)
    colors = shuffled_voc_colors(seed, len(ordered))
    return dict(zip(ordered, colors))


def ordered_figure_zones(edge: pd.DataFrame) -> list[str]:
    actual = set(edge["zone"].dropna().astype(str)) if not edge.empty else set()
    expected = set(FIGURE_ZONE_ORDER)
    if actual != expected:
        raise ValueError(
            "Figure edge zones do not match the required pair-zone outputs: "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )
    return list(FIGURE_ZONE_ORDER)


def figure_05_zones(edge: pd.DataFrame) -> list[str]:
    if "zone" not in edge.columns:
        raise ValueError("Figure 5 edge data is missing the zone column")
    if edge.empty:
        raise ValueError("Figure 5 edge data contains no regions")
    if edge["zone"].isna().any():
        raise ValueError("Figure 5 edge data contains a missing zone label")
    raw_zones = edge["zone"].astype(str).tolist()
    for zone in raw_zones:
        if not zone.strip():
            raise ValueError("Figure 5 edge data contains an empty zone label")
        if zone != zone.strip():
            raise ValueError(f"Figure 5 zone label contains surrounding whitespace: {zone!r}")
    # Production analysis and Figure 3 require six reference zones.
    # Figure 5 derives its zone count from the current edge data.
    return sorted(set(raw_zones), key=lambda zone: (zone.casefold(), zone))


def sample_paths(sample_id: str) -> dict[str, Path]:
    outdir = SNA_OUTPUT_DIR / sample_id
    indir = SNA_INPUT_DIR / sample_id
    return {
        "input": indir,
        "output": outdir,
        "identities": indir / "global_identities.json",
        "input_generation": indir / "generation_manifest.json",
        "output_generation": outdir / "generation_manifest.json",
        "zones": indir / "zones.json",
        "config": indir / "config.yaml",
        "summary": outdir / "analysis_summary.json",
        "cow_frame": outdir / "cow_frame_zone.csv",
        "time_budget": outdir / "cow_time_budget.csv",
        "edge": outdir / "edge_level.csv",
        "node": outdir / "node_descriptors.csv",
        "zone_node": outdir / "zone_node_descriptors.csv",
        "layout": outdir / "report_network_layout.csv",
        "community_windows": outdir / "community_windows.csv",
        "community_summary": outdir / "community_stability_summary.csv",
        "edge_ranking": outdir / "edge_ranking.csv",
        "events": outdir / "interaction_events.csv",
    }


def missing_required_sample_files(sample_id: str) -> list[str]:
    paths = sample_paths(sample_id)
    return [str(paths[key]) for key in REQUIRED_SNA_OUTPUT_KEYS if not paths[key].is_file()]


def complete_sna_output_samples() -> list[str]:
    if not SNA_OUTPUT_DIR.is_dir():
        return []
    samples: list[str] = []
    for path in sorted(SNA_OUTPUT_DIR.iterdir(), key=lambda item: item.name):
        if path.is_dir() and not missing_required_sample_files(path.name):
            samples.append(path.name)
    return samples


def load_identity_display_map(path: Path) -> dict[str, str]:
    document = read_json(path)
    if (
        int(document.get("schema_version", -1)) != 1
        or document.get("sequence_id") != REID_SEQUENCE_ID
        or document.get("combined_sample_id") != COMBINED_SAMPLE_ID
        or document.get("identity_key") != "global_track_uuid"
        or document.get("display_label") != "display_global_id"
    ):
        raise ValueError(f"Unexpected global identity mapping contract: {path}")
    identities = document.get("identities")
    if not isinstance(identities, list) or len(identities) != EXPECTED_GLOBAL_IDENTITY_COUNT:
        raise ValueError(
            f"Expected exactly {EXPECTED_GLOBAL_IDENTITY_COUNT} global identities in {path}; "
            f"found={len(identities) if isinstance(identities, list) else 'non-list'}"
        )
    mapping: dict[str, str] = {}
    global_ids: dict[int, str] = {}
    displays: dict[str, str] = {}
    for item in identities:
        if not isinstance(item, dict):
            raise ValueError(f"Identity mapping row is not an object: {path}")
        global_uuid = str(item.get("global_track_uuid", "")).strip()
        display_id = str(item.get("display_global_id", "")).strip()
        global_id = int(item.get("global_track_id", -1))
        try:
            canonical_uuid = str(uuid.UUID(global_uuid))
        except ValueError as exc:
            raise ValueError(f"Invalid global_track_uuid in {path}: {global_uuid!r}") from exc
        if global_uuid != canonical_uuid:
            raise ValueError(f"Non-canonical global_track_uuid in {path}: {global_uuid!r}")
        expected_display = f"G{global_id + 1:04d}"
        if display_id != expected_display:
            raise ValueError(
                f"display_global_id disagrees with global_track_id in {path}: "
                f"expected={expected_display}, actual={display_id}"
            )
        if mapping.setdefault(global_uuid, display_id) != display_id:
            raise ValueError(f"Duplicate UUID mapping in {path}: {global_uuid}")
        if global_ids.setdefault(global_id, global_uuid) != global_uuid:
            raise ValueError(f"Duplicate global_track_id mapping in {path}: {global_id}")
        if displays.setdefault(display_id, global_uuid) != global_uuid:
            raise ValueError(f"Duplicate display_global_id mapping in {path}: {display_id}")
    if set(global_ids) != set(range(EXPECTED_GLOBAL_IDENTITY_COUNT)):
        raise ValueError(f"global_track_id values are not exactly 0..61 in {path}")
    return mapping


def validate_generation_pair(paths: dict[str, Path]) -> str:
    input_generation = read_json(paths["input_generation"])
    output_generation = read_json(paths["output_generation"])
    identity_document = read_json(paths["identities"])
    generation_ids = {
        str(document.get("generation_id", "")).strip()
        for document in (input_generation, output_generation, identity_document)
    }
    if len(generation_ids) != 1:
        raise ValueError(
            "SNA input, analysis output, and identity mapping belong to different generations: "
            f"{sorted(generation_ids)}"
        )
    generation_id = generation_ids.pop()
    try:
        canonical_generation_id = str(uuid.UUID(generation_id))
    except ValueError as exc:
        raise ValueError(f"Invalid SNA generation_id: {generation_id!r}") from exc
    if generation_id != canonical_generation_id:
        raise ValueError(f"Non-canonical SNA generation_id: {generation_id!r}")
    for label, document in (
        ("input", input_generation),
        ("output", output_generation),
    ):
        if (
            int(document.get("schema_version", -1)) != 1
            or document.get("combined_sample_id") != COMBINED_SAMPLE_ID
            or document.get("reid_sequence_id") != REID_SEQUENCE_ID
        ):
            raise ValueError(f"Unexpected {label} SNA generation contract")
    return generation_id


def relabel_identity_column(df: pd.DataFrame, column: str, mapping: dict[str, str], label: str) -> None:
    if column not in df.columns:
        raise ValueError(f"Required identity column is missing from {label}: {column}")
    if df.empty:
        return
    raw = df[column].astype("string")
    if raw.isna().any() or (raw.str.strip() == "").any():
        raise ValueError(f"Empty global identity in {label}.{column}")
    values = raw.astype(str)
    unknown = sorted(set(values) - set(mapping))
    if unknown:
        raise ValueError(f"Unknown global UUIDs in {label}.{column}: {unknown[:5]}")
    df[column] = values.map(mapping)
    if not df[column].astype(str).str.fullmatch(r"G\d{4}").all():
        raise ValueError(f"Relabeled display identities are invalid in {label}.{column}")


def relabel_sample_identities(data: dict[str, Any], mapping: dict[str, str]) -> None:
    columns = {
        "cow_frame": ("cow_id",),
        "time_budget": ("cow_id",),
        "edge": ("cow_i", "cow_j"),
        "node": ("cow_id",),
        "zone_node": ("cow_id",),
        "layout": ("cow_id",),
        "community_windows": ("cow_id",),
        "edge_ranking": ("cow_i", "cow_j"),
        "events": ("cow_i", "cow_j"),
    }
    for label, identity_columns in columns.items():
        frame = data[label]
        for column in identity_columns:
            relabel_identity_column(frame, column, mapping, label)
    expected_displays = set(mapping.values())
    for label in ("time_budget", "node", "layout"):
        actual = set(data[label]["cow_id"].astype(str))
        if actual != expected_displays:
            raise ValueError(
                f"{label} does not contain exactly G0001..G0062: "
                f"expected={len(expected_displays)}, actual={len(actual)}"
            )


def figure_identity_label(value: Any) -> str:
    display_id = str(value).strip()
    match = re.fullmatch(r"G(\d{4})", display_id)
    if match is None:
        raise ValueError(f"Figure identity is not a GXXXX display ID: {display_id!r}")
    identity_number = int(match.group(1))
    if identity_number < 1 or identity_number > EXPECTED_GLOBAL_IDENTITY_COUNT:
        raise ValueError(f"Figure identity is outside 1..{EXPECTED_GLOBAL_IDENTITY_COUNT}: {display_id}")
    return str(identity_number)


def load_sample(sample_id: str) -> dict[str, Any]:
    paths = sample_paths(sample_id)
    validate_generation_pair(paths)
    mapping = load_identity_display_map(paths["identities"])
    data = {
        "sample_id": sample_id,
        "paths": paths,
        "zones": read_json(paths["zones"]),
        "summary": read_json(paths["summary"]),
        "cow_frame": read_cow_frame_visual_sample(paths["cow_frame"]),
        "time_budget": read_csv(paths["time_budget"], dtype={"cow_id": str}),
        "edge": read_csv(paths["edge"], dtype={"cow_i": str, "cow_j": str}),
        "node": read_csv(paths["node"], dtype={"cow_id": str}),
        "zone_node": read_csv(paths["zone_node"], dtype={"cow_id": str}),
        "layout": read_csv(paths["layout"], dtype={"cow_id": str}),
        "community_windows": read_csv(paths["community_windows"], dtype={"cow_id": str}),
        "community_summary": read_csv(paths["community_summary"]),
        "edge_ranking": read_csv(paths["edge_ranking"], dtype={"cow_i": str, "cow_j": str}),
        "events": read_csv(paths["events"], dtype={"cow_i": str, "cow_j": str}),
    }
    relabel_sample_identities(data, mapping)
    return data


def canvas_size(
    zones: dict[str, Any],
    cow_frame: pd.DataFrame,
    layout: pd.DataFrame | None = None,
) -> tuple[float, float]:
    max_x = 3840.0
    max_y = 2160.0
    for zone in zones.get("zones", []):
        for point in zone.get("polygon", []):
            if isinstance(point, list) and len(point) >= 2:
                max_x = max(max_x, float(point[0]))
                max_y = max(max_y, float(point[1]))
    if not cow_frame.empty:
        anchor_x = pd.to_numeric(cow_frame["anchor_x"], errors="coerce")
        anchor_y = pd.to_numeric(cow_frame["anchor_y"], errors="coerce")
        if anchor_x.notna().any():
            max_x = max(max_x, float(anchor_x.max()))
        if anchor_y.notna().any():
            max_y = max(max_y, float(anchor_y.max()))
    if layout is not None and not layout.empty:
        for column, axis in (("median_floorplan_x", "x"), ("median_floorplan_y", "y")):
            if column not in layout.columns:
                continue
            values = pd.to_numeric(layout[column], errors="coerce")
            if values.notna().any():
                if axis == "x":
                    max_x = max(max_x, float(values.max()))
                else:
                    max_y = max(max_y, float(values.max()))
    return max_x, max_y


def draw_zone_polygons(ax: plt.Axes, zones: dict[str, Any]) -> None:
    labels_seen: set[str] = set()
    for zone in zones.get("zones", []):
        polygon = zone.get("polygon", [])
        if not isinstance(polygon, list) or len(polygon) < 3:
            continue
        zone_type = str(zone.get("zone_type", "unknown"))
        points = np.asarray(polygon, dtype=float)
        label = zone_type if zone_type not in labels_seen else None
        labels_seen.add(zone_type)
        patch = MplPolygon(
            points,
            closed=True,
            facecolor=zone_color(zone_type),
            edgecolor=zone_color(zone_type),
            linewidth=1.2,
            alpha=0.22,
            label=label,
        )
        ax.add_patch(patch)


def structure_polygons(data: dict[str, Any]) -> list[dict[str, Any]]:
    analysis = data["summary"].get("config", {}).get("analysis", {})
    farm = str(analysis.get("farm", data["zones"].get("farm", "1")))
    camera = str(analysis.get("camera", data["zones"].get("camera", "Gopro1")))
    annotation = read_json(floorplan_annotation_file(farm, camera))
    out: list[dict[str, Any]] = []
    for polygon in annotation.get("polygons", []):
        class_id = str(polygon.get("classId", ""))
        if class_id not in {"obstacle", "resource"} or not bool(polygon.get("force2DRectangle")):
            continue
        raw_points = polygon.get("points", [])
        if not isinstance(raw_points, list) or len(raw_points) < 3:
            continue
        points = [(float(point[0]), float(point[1])) for point in raw_points]
        out.append(
            {
                "class_id": class_id,
                "points": np.asarray(min_area_rectangle(points), dtype=float),
                "color": STRUCTURE_COLORS[class_id],
                "label": class_id,
            }
        )
    return out


def draw_structure_polygons(ax: plt.Axes, data: dict[str, Any]) -> list[dict[str, Any]]:
    labels_seen: set[str] = set()
    polygons = structure_polygons(data)
    for polygon in polygons:
        label = polygon["label"] if polygon["label"] not in labels_seen else None
        labels_seen.add(polygon["label"])
        patch = MplPolygon(
            polygon["points"],
            closed=True,
            facecolor=polygon["color"],
            edgecolor=polygon["color"],
            linewidth=1.6,
            alpha=0.18,
            label=label,
        )
        ax.add_patch(patch)
    return polygons


def median_floorplan_positions(cow_frame: pd.DataFrame) -> dict[str, tuple[float, float]]:
    if cow_frame.empty:
        return {}
    rows = cow_frame.loc[pd.to_numeric(cow_frame.get("visible_flag", 1), errors="coerce").fillna(0).astype(int) == 1]
    if rows.empty:
        return {}
    grouped = rows.groupby("cow_id")[["anchor_x", "anchor_y"]].median()
    return {str(cow): (float(row["anchor_x"]), float(row["anchor_y"])) for cow, row in grouped.iterrows()}


def layout_floorplan_positions(layout: pd.DataFrame) -> dict[str, tuple[float, float]]:
    required = {"cow_id", "median_floorplan_x", "median_floorplan_y"}
    if layout.empty or not required.issubset(layout.columns):
        return {}
    x_values = pd.to_numeric(layout["median_floorplan_x"], errors="coerce")
    y_values = pd.to_numeric(layout["median_floorplan_y"], errors="coerce")
    valid = np.isfinite(x_values.to_numpy(dtype=float)) & np.isfinite(y_values.to_numpy(dtype=float))
    return {
        str(cow): (float(x), float(y))
        for cow, x, y in zip(layout.loc[valid, "cow_id"], x_values.loc[valid], y_values.loc[valid])
    }


def trajectory_plot_segments(
    group: pd.DataFrame,
    red_obstacle_polygons: list[np.ndarray] | None = None,
    max_displacement_px: float = FIGURE_01_MAX_ADJACENT_DISPLACEMENT_PX,
) -> list[pd.DataFrame]:
    required = {
        TRAJECTORY_SEGMENT_COLUMN,
        "frame",
        "time_s",
        "dt_s",
        "anchor_x",
        "anchor_y",
    }
    missing = sorted(required - set(group.columns))
    if missing:
        raise ValueError(f"Plot-only trajectory sample is missing columns: {missing}")
    max_displacement = float(max_displacement_px)
    if not math.isfinite(max_displacement) or max_displacement <= 0.0:
        raise ValueError(
            f"Figure 1 maximum adjacent displacement must be finite and positive: {max_displacement_px}"
        )

    red_polygons = [] if red_obstacle_polygons is None else red_obstacle_polygons
    segments: list[pd.DataFrame] = []
    for _, segment in group.groupby(TRAJECTORY_SEGMENT_COLUMN, sort=True):
        ordered = segment.sort_values(["frame", "time_s"], kind="stable")
        if ordered.empty:
            continue

        frames = pd.to_numeric(ordered["frame"], errors="coerce").to_numpy(dtype=float)
        times = pd.to_numeric(ordered["time_s"], errors="coerce").to_numpy(dtype=float)
        dts = pd.to_numeric(ordered["dt_s"], errors="coerce").to_numpy(dtype=float)
        coordinates = ordered[["anchor_x", "anchor_y"]].apply(
            pd.to_numeric, errors="coerce"
        ).to_numpy(dtype=float)
        if not (
            np.isfinite(frames).all()
            and np.isfinite(times).all()
            and np.isfinite(dts).all()
            and np.isfinite(coordinates).all()
            and (dts > 0.0).all()
        ):
            raise ValueError("Figure 1 trajectory segment contains invalid numeric values")
        integer_frames = frames.astype(np.int64)
        if not np.array_equal(frames, integer_frames.astype(float)):
            raise ValueError("Figure 1 trajectory segment contains non-integer frame values")

        continuation = _trajectory_continuation_mask(integer_frames, times, dts, None)
        for index in range(1, len(ordered)):
            if not continuation[index]:
                continue
            a = (float(coordinates[index - 1, 0]), float(coordinates[index - 1, 1]))
            b = (float(coordinates[index, 0]), float(coordinates[index, 1]))
            if math.hypot(b[0] - a[0], b[1] - a[1]) > max_displacement:
                continuation[index] = False
                continue
            if any(segment_intersects_polygon(a, b, polygon) for polygon in red_polygons):
                continuation[index] = False

        starts = np.r_[0, np.flatnonzero(~continuation[1:]) + 1]
        ends = np.r_[starts[1:], len(ordered)]
        segments.extend(ordered.iloc[int(start) : int(end)] for start, end in zip(starts, ends))
    return segments


def draw_trajectory_group(
    ax: plt.Axes,
    group: pd.DataFrame,
    color: str,
    red_obstacle_polygons: list[np.ndarray],
) -> None:
    ax.scatter(
        group["anchor_x"],
        group["anchor_y"],
        s=FIGURE_01_SAMPLE_MARKER_SIZE,
        color=color,
        alpha=0.55,
        linewidths=0.0,
        zorder=2.5,
    )
    for segment in trajectory_plot_segments(group, red_obstacle_polygons):
        if len(segment) < 2:
            continue
        ax.plot(
            segment["anchor_x"],
            segment["anchor_y"],
            linewidth=1.0,
            alpha=0.55,
            color=color,
            zorder=2.6,
        )


def circle_positions(cows: list[str]) -> dict[str, tuple[float, float]]:
    if not cows:
        return {}
    return {
        cow: (
            math.cos(2.0 * math.pi * index / len(cows)),
            math.sin(2.0 * math.pi * index / len(cows)),
        )
        for index, cow in enumerate(sorted({str(cow) for cow in cows}, key=cow_sort_key))
    }


def floorplan_axes(ax: plt.Axes, width: float, height: float, labels: bool) -> None:
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_aspect("equal", adjustable="box")
    if labels:
        ax.set_xlabel("floorplan x")
        ax.set_ylabel("floorplan y")
    else:
        ax.axis("off")


def plot_floorplan_trajectories(data: dict[str, Any], figdir: Path) -> dict[str, str]:
    sample_id = data["sample_id"]
    zones = data["zones"]
    cow_frame = data["cow_frame"]
    layout = data["layout"]
    width, height = canvas_size(zones, cow_frame, layout)
    cows = sorted(layout["cow_id"].astype(str).unique(), key=cow_sort_key)
    expected_cows = [f"G{index:04d}" for index in range(1, EXPECTED_GLOBAL_IDENTITY_COUNT + 1)]
    if cows != expected_cows:
        raise ValueError(
            f"Figure 1 requires G0001..G{EXPECTED_GLOBAL_IDENTITY_COUNT:04d}; actual={cows}"
        )
    colors = cow_color_map(cows, seed=sample_id)
    positions = layout_floorplan_positions(layout)
    if not positions:
        positions = median_floorplan_positions(cow_frame)
    missing_positions = sorted(set(cows) - set(positions), key=cow_sort_key)
    if missing_positions:
        raise ValueError(f"Figure 1 identities are missing floorplan positions: {missing_positions}")

    paths: dict[str, str] = {}
    figure_dir = figdir / "figure_01"
    for start, end in FIGURE_01_GROUPS:
        group_cows = [f"G{index:04d}" for index in range(start, end + 1)]
        group_set = set(group_cows)
        group_frame = cow_frame.loc[cow_frame["cow_id"].astype(str).isin(group_set)].copy()
        fig, ax = plt.subplots(figsize=(12, 7))
        draw_zone_polygons(ax, zones)
        structures = draw_structure_polygons(ax, data) or []
        red_obstacle_polygons = [
            polygon["points"] for polygon in structures if polygon["class_id"] == "obstacle"
        ]

        for cow_id, group in group_frame.groupby("cow_id", sort=True):
            cow = str(cow_id)
            draw_trajectory_group(ax, group, colors[cow], red_obstacle_polygons)

        for cow in group_cows:
            x, y = positions[cow]
            figure_label = figure_identity_label(cow)
            ax.scatter(
                [x],
                [y],
                s=72,
                color=colors[cow],
                edgecolor="#212529",
                linewidth=0.7,
                alpha=0.88,
                zorder=4,
            )
            ax.text(x + 28, y, figure_label, fontsize=8, va="center", color="#212529", zorder=5)

        ax.set_title(f"{sample_id} Figure 1. Floorplan trajectories: cows {start}–{end}")
        floorplan_axes(ax, width, height, labels=True)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8, frameon=False)
        key = f"figure_01_cows_{start:02d}_{end:02d}"
        paths[key] = save_figure(fig, output_path(figure_dir, f"{key}.png"))
    return paths


def time_budget_zone_columns(time_budget: pd.DataFrame) -> list[str]:
    return [
        col
        for col in time_budget.columns
        if col.startswith("time_") and col.endswith("_s") and not col.startswith("weighted_")
    ]


def plot_visibility_time_budget(data: dict[str, Any], figdir: Path) -> str:
    sample_id = data["sample_id"]
    budget = data["time_budget"].copy()
    zone_cols = time_budget_zone_columns(budget)
    fig, ax = plt.subplots(figsize=(12.5, 5.5))
    if zone_cols and not budget.empty:
        budget = budget.sort_values("visible_time_s", ascending=False)
        bottoms = np.zeros(len(budget))
        x = np.arange(len(budget))
        for col in zone_cols:
            zone = col.removeprefix("time_").removesuffix("_s")
            values = pd.to_numeric(budget[col], errors="coerce").fillna(0).to_numpy(dtype=float)
            ax.bar(x, values, bottom=bottoms, label=zone, color=zone_color(zone), alpha=0.85)
            bottoms += values
        ax.set_xticks(x)
        ax.set_xticklabels(
            [figure_identity_label(cow) for cow in budget["cow_id"].astype(str)],
            rotation=45,
            ha="right",
        )
        ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8, frameon=False)
    else:
        ax.text(0.5, 0.5, "No time budget rows", ha="center", va="center", transform=ax.transAxes)
    ax.set_ylabel("seconds")
    ax.set_title(f"{sample_id} Figure 2. Cow visibility and zone time budget")
    return save_figure(fig, output_path(figdir, "figure_02_visibility_zone_time_budget.png"))


def zone_layer_summary(edge: pd.DataFrame) -> pd.DataFrame:
    if edge.empty:
        return pd.DataFrame(columns=["zone", "layer", "expected_seconds", "opportunity_seconds", "normalized_rate"])
    rows = []
    for key, group in edge.groupby(["zone", "layer"], sort=True):
        expected = float(pd.to_numeric(group["expected_seconds"], errors="coerce").fillna(0).sum())
        opportunity = float(pd.to_numeric(group["opportunity_seconds"], errors="coerce").fillna(0).sum())
        rows.append(
            {
                "zone": str(key[0]),
                "layer": str(key[1]),
                "expected_seconds": expected,
                "opportunity_seconds": opportunity,
                "normalized_rate": expected / opportunity if opportunity > 0 else 0.0,
            }
        )
    return pd.DataFrame(rows)


def zone_net_layer_summary(
    edge: pd.DataFrame,
    min_dominance: float = FIGURE_NET_DOMINANCE_THRESHOLD,
) -> pd.DataFrame:
    columns = [
        "zone",
        "layer",
        "net_expected_seconds",
        "opportunity_seconds",
        "net_normalized_rate",
    ]
    if edge.empty:
        return pd.DataFrame(columns=columns)

    threshold = float(min_dominance)
    if not math.isfinite(threshold) or threshold < 0.0 or threshold > 1.0:
        raise ValueError(f"Figure net dominance threshold must be within [0, 1]: {min_dominance}")

    required = {"cow_i", "cow_j", "zone", "layer", "expected_seconds", "opportunity_seconds"}
    missing = sorted(required - set(edge.columns))
    if missing:
        raise ValueError(f"Figure 3 edge data is missing columns: {missing}")

    layer_order = ["friendly", "unfriendly"]
    pair_columns = ["cow_i", "cow_j", "zone"]
    rows = edge.loc[edge["layer"].astype(str).isin(layer_order), list(required)].copy()
    if rows.empty:
        return pd.DataFrame(columns=columns)

    rows["layer"] = rows["layer"].astype(str)
    for column in ("expected_seconds", "opportunity_seconds"):
        rows[column] = pd.to_numeric(rows[column], errors="coerce").fillna(0.0)
        values = rows[column].to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values < 0.0).any():
            raise ValueError(f"Figure 3 received invalid {column}")

    grouped = (
        rows.groupby(pair_columns + ["layer"], as_index=False, sort=True)[
            ["expected_seconds", "opportunity_seconds"]
        ]
        .sum()
    )
    expected_wide = (
        grouped.pivot(index=pair_columns, columns="layer", values="expected_seconds")
        .reindex(columns=layer_order)
        .fillna(0.0)
    )
    opportunity_wide = grouped.pivot(index=pair_columns, columns="layer", values="opportunity_seconds").reindex(
        columns=layer_order
    )

    both_layers = opportunity_wide.notna().all(axis=1)
    if bool(both_layers.any()):
        friendly_opportunity = opportunity_wide.loc[both_layers, "friendly"].to_numpy(dtype=float)
        unfriendly_opportunity = opportunity_wide.loc[both_layers, "unfriendly"].to_numpy(dtype=float)
        if not np.isclose(
            friendly_opportunity,
            unfriendly_opportunity,
            rtol=1.0e-9,
            atol=1.0e-9,
        ).all():
            raise ValueError("Figure 3 friendly/unfriendly opportunity seconds differ for the same dyad-zone")

    pair = expected_wide.rename(
        columns={
            "friendly": "friendly_expected_seconds",
            "unfriendly": "unfriendly_expected_seconds",
        }
    )
    pair["opportunity_seconds"] = opportunity_wide.max(axis=1, skipna=True).fillna(0.0)
    pair = pair.reset_index()
    pair["total_expected_seconds"] = (
        pair["friendly_expected_seconds"] + pair["unfriendly_expected_seconds"]
    )
    pair["net_expected_seconds"] = (
        pair["friendly_expected_seconds"] - pair["unfriendly_expected_seconds"]
    )
    positive = pair["total_expected_seconds"] > 0.0
    pair["dominance"] = 0.0
    pair.loc[positive, "dominance"] = (
        pair.loc[positive, "net_expected_seconds"].abs()
        / pair.loc[positive, "total_expected_seconds"]
    )
    decisive = positive & (pair["dominance"] >= threshold)
    pair["friendly_net_expected_seconds"] = np.where(
        decisive & (pair["net_expected_seconds"] > 0.0),
        pair["net_expected_seconds"],
        0.0,
    )
    pair["unfriendly_net_expected_seconds"] = np.where(
        decisive & (pair["net_expected_seconds"] < 0.0),
        -pair["net_expected_seconds"],
        0.0,
    )

    summary_rows: list[dict[str, Any]] = []
    for zone, group in pair.groupby("zone", sort=True):
        opportunity = float(group["opportunity_seconds"].sum())
        for layer in layer_order:
            expected = float(group[f"{layer}_net_expected_seconds"].sum())
            summary_rows.append(
                {
                    "zone": str(zone),
                    "layer": layer,
                    "net_expected_seconds": expected,
                    "opportunity_seconds": opportunity,
                    "net_normalized_rate": expected / opportunity if opportunity > 0.0 else 0.0,
                }
            )
    return pd.DataFrame(summary_rows, columns=columns)


def plot_zone_interaction_volume(data: dict[str, Any], figdir: Path) -> str:
    sample_id = data["sample_id"]
    summary = zone_net_layer_summary(data["edge"])
    zones = ordered_figure_zones(data["edge"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, value_col, title in (
        (axes[0], "net_expected_seconds", "Net expected seconds"),
        (axes[1], "net_normalized_rate", "Net normalized rate"),
    ):
        if summary.empty:
            ax.text(0.5, 0.5, "No edge rows", ha="center", va="center", transform=ax.transAxes)
            continue
        pivot = summary.pivot_table(index="zone", columns="layer", values=value_col, aggfunc="sum", fill_value=0)
        pivot = pivot.reindex(zones, axis=0, fill_value=0.0)
        pivot = pivot.reindex(["friendly", "unfriendly"], axis=1, fill_value=0.0)
        x = np.arange(len(pivot), dtype=float)
        ax.bar(
            x,
            pivot["friendly"].to_numpy(dtype=float),
            color=layer_color("friendly"),
            alpha=0.85,
            label="friendly-dominant",
        )
        ax.bar(
            x,
            -pivot["unfriendly"].to_numpy(dtype=float),
            color=layer_color("unfriendly"),
            alpha=0.85,
            label="unfriendly-dominant",
        )
        ax.axhline(0.0, color="#495057", linewidth=0.8)
        ax.set_title(title)
        ax.set_xlabel("zone")
        ax.set_xticks(x)
        ax.set_xticklabels(pivot.index.astype(str), rotation=35, ha="right")
        ax.set_ylabel("friendly (+) / unfriendly (-)")
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle(
        f"{sample_id} Figure 3. Dyad-exclusive net interaction volume by zone "
        f"(dominance >= {FIGURE_NET_DOMINANCE_THRESHOLD:.2f})"
    )
    return save_figure(fig, output_path(figdir, "figure_03_interaction_volume_by_zone.png"))


def aggregate_network(edge: pd.DataFrame, layer: str, zone: str | None = None) -> pd.DataFrame:
    if edge.empty:
        return pd.DataFrame(columns=["cow_i", "cow_j", "expected_seconds"])
    rows = edge.loc[edge["layer"].astype(str) == layer].copy()
    if zone is not None:
        rows = rows.loc[rows["zone"].astype(str) == zone]
    if rows.empty:
        return pd.DataFrame(columns=["cow_i", "cow_j", "expected_seconds"])
    return (
        rows.groupby(["cow_i", "cow_j"], as_index=False)["expected_seconds"]
        .sum()
        .sort_values("expected_seconds", ascending=False)
    )


def net_dominance_networks(
    edge: pd.DataFrame,
    zone: str | None = None,
    min_dominance: float = FIGURE_NET_DOMINANCE_THRESHOLD,
) -> dict[str, pd.DataFrame]:
    threshold = float(min_dominance)
    if not math.isfinite(threshold) or threshold < 0.0 or threshold > 1.0:
        raise ValueError(f"Figure net dominance threshold must be within [0, 1]: {min_dominance}")

    friendly = aggregate_network(edge, "friendly", zone).rename(
        columns={"expected_seconds": "friendly_expected_seconds"}
    )
    unfriendly = aggregate_network(edge, "unfriendly", zone).rename(
        columns={"expected_seconds": "unfriendly_expected_seconds"}
    )
    pair_columns = ["cow_i", "cow_j"]
    combined = friendly.merge(unfriendly, on=pair_columns, how="outer")
    output_columns = [
        "cow_i",
        "cow_j",
        "expected_seconds",
        "friendly_expected_seconds",
        "unfriendly_expected_seconds",
        "net_expected_seconds",
        "dominance",
        "dominant_layer",
    ]
    empty = pd.DataFrame(columns=output_columns)
    if combined.empty:
        return {"friendly": empty.copy(), "unfriendly": empty.copy()}

    for column in ("friendly_expected_seconds", "unfriendly_expected_seconds"):
        combined[column] = pd.to_numeric(combined[column], errors="coerce").fillna(0.0)
        if (combined[column] < 0.0).any():
            raise ValueError(f"Figure net dominance received negative {column}")

    combined["expected_seconds"] = (
        combined["friendly_expected_seconds"] + combined["unfriendly_expected_seconds"]
    )
    combined["net_expected_seconds"] = (
        combined["friendly_expected_seconds"] - combined["unfriendly_expected_seconds"]
    )
    positive = combined["expected_seconds"] > 0.0
    combined["dominance"] = 0.0
    combined.loc[positive, "dominance"] = (
        combined.loc[positive, "net_expected_seconds"].abs()
        / combined.loc[positive, "expected_seconds"]
    )
    combined["dominant_layer"] = np.where(
        combined["net_expected_seconds"] > 0.0,
        "friendly",
        np.where(combined["net_expected_seconds"] < 0.0, "unfriendly", ""),
    )
    eligible = positive & (combined["dominance"] >= threshold) & combined["dominant_layer"].ne("")
    combined = combined.loc[eligible, output_columns].copy()

    networks: dict[str, pd.DataFrame] = {}
    for layer in ("friendly", "unfriendly"):
        networks[layer] = (
            combined.loc[combined["dominant_layer"] == layer, output_columns]
            .sort_values(["expected_seconds", "cow_i", "cow_j"], ascending=[False, True, True])
            .reset_index(drop=True)
        )
    return networks


def shared_network_max_weight(networks: dict[str, pd.DataFrame]) -> float:
    maxima: list[float] = []
    for network in networks.values():
        if network.empty:
            continue
        value = float(pd.to_numeric(network["expected_seconds"], errors="coerce").max())
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"Figure network maximum weight must be finite and non-negative: {value}")
        maxima.append(value)
    return max(maxima, default=0.0)


def drawable_network_cows(networks: dict[str, pd.DataFrame]) -> list[str]:
    cows: set[str] = set()
    for layer in ("friendly", "unfriendly"):
        if layer not in networks:
            raise ValueError(f"Figure network set is missing layer: {layer}")
        network = networks[layer]
        required = {"cow_i", "cow_j", "expected_seconds"}
        missing = sorted(required - set(network.columns))
        if missing:
            raise ValueError(f"Figure {layer} network is missing columns: {missing}")
        for row in network.loc[:, ["cow_i", "cow_j", "expected_seconds"]].itertuples(index=False):
            weight = float(row.expected_seconds)
            if not math.isfinite(weight) or weight < 0.0:
                raise ValueError(f"Figure network edge weight must be finite and non-negative: {weight}")
            if weight == 0.0:
                continue
            cow_i = str(row.cow_i).strip()
            cow_j = str(row.cow_j).strip()
            if not cow_i or not cow_j:
                raise ValueError("Figure network edge endpoint must not be empty")
            cows.update((cow_i, cow_j))
    return sorted(cows, key=cow_sort_key)


def layout_positions(layout: pd.DataFrame) -> dict[str, tuple[float, float]]:
    return {
        str(row["cow_id"]): (float(row["layout_x"]), float(row["layout_y"]))
        for _, row in layout.iterrows()
    }


def layout_communities(layout: pd.DataFrame) -> dict[str, int]:
    return {
        str(row["cow_id"]): int(row.get("community_id_full_friendly", -1))
        for _, row in layout.iterrows()
    }


def draw_network(
    ax: plt.Axes,
    network: pd.DataFrame,
    positions: dict[str, tuple[float, float]],
    layer: str,
    title: str,
    communities: dict[str, int] | None = None,
    node_colors: dict[str, str] | None = None,
    node_size: float = 220.0,
    label_beside: bool = False,
    max_weight: float | None = None,
    alpha_column: str | None = None,
) -> None:
    communities = communities or {}
    node_colors = node_colors or {}
    if max_weight is None:
        max_weight = float(
            pd.to_numeric(network.get("expected_seconds", pd.Series(dtype=float)), errors="coerce").max() or 0
        )
    max_weight = float(max_weight)
    if not math.isfinite(max_weight) or max_weight < 0.0:
        raise ValueError(f"Figure network maximum weight must be finite and non-negative: {max_weight}")
    for _, row in network.iterrows():
        weight = float(row["expected_seconds"])
        if weight <= 0:
            continue
        cow_i = str(row["cow_i"])
        cow_j = str(row["cow_j"])
        if cow_i not in positions or cow_j not in positions:
            continue
        x1, y1 = positions[cow_i]
        x2, y2 = positions[cow_j]
        width = 0.8 + 5.2 * (weight / max_weight if max_weight > 0 else 0.0)
        edge_alpha = 0.55 if alpha_column is None else float(row[alpha_column])
        if not math.isfinite(edge_alpha) or edge_alpha < 0.0 or edge_alpha > 1.0:
            raise ValueError(f"Figure network alpha must be within [0, 1]: {edge_alpha}")
        ax.plot(
            [x1, x2],
            [y1, y2],
            color=layer_color(layer),
            linewidth=width,
            alpha=edge_alpha,
            zorder=1,
        )
    cmap = plt.get_cmap("tab20")
    for cow, (x, y) in positions.items():
        color = node_colors.get(cow, cmap(communities.get(cow, -1) % 20))
        figure_label = figure_identity_label(cow)
        ax.scatter([x], [y], s=node_size, color=color, edgecolor="#212529", linewidth=0.8, zorder=2)
        if label_beside:
            ax.annotate(
                figure_label,
                (x, y),
                xytext=(10, 3),
                textcoords="offset points",
                fontsize=8,
                zorder=3,
            )
        else:
            ax.text(
                x,
                y,
                figure_label,
                ha="center",
                va="center",
                fontsize=marker_label_font_size(node_size, figure_label),
                color=contrasting_text_color(color),
                zorder=3,
            )
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="datalim")
    ax.axis("off")
    if network.empty or float(pd.to_numeric(network["expected_seconds"], errors="coerce").fillna(0).sum()) <= 0:
        ax.text(0.5, -0.04, "no positive edges", ha="center", va="top", transform=ax.transAxes, fontsize=9, clip_on=False)


def plot_full_networks(data: dict[str, Any], figdir: Path) -> dict[str, str]:
    sample_id = data["sample_id"]
    edge = data["edge"]
    layout = data["layout"]
    cow_frame = data["cow_frame"]
    width, height = canvas_size(data["zones"], cow_frame, layout)
    positions = layout_floorplan_positions(layout)
    if not positions:
        positions = median_floorplan_positions(cow_frame)
    if not positions:
        positions = layout_positions(layout)
    communities = layout_communities(layout)
    colors = cow_color_map(positions.keys(), seed=sample_id)
    networks = net_dominance_networks(edge)
    max_weight = shared_network_max_weight(networks)
    paths: dict[str, str] = {}
    for suffix, layer in (("04A", "friendly"), ("04B", "unfriendly")):
        fig, ax = plt.subplots(figsize=(8, 6))
        draw_structure_polygons(ax, data)
        draw_network(
            ax,
            networks[layer],
            positions,
            layer,
            f"dominant {layer}",
            communities=communities,
            node_colors=colors,
            node_size=250,
            label_beside=True,
            max_weight=max_weight,
            alpha_column="dominance",
        )
        floorplan_axes(ax, width, height, labels=False)
        fig.suptitle(f"{sample_id} Figure {suffix}. Full-window {layer} social network")
        key = f"figure_{suffix}_full_window_{layer}_network"
        paths[key] = save_figure(fig, output_path(figdir, f"{key}.png"))
    return paths


def plot_zone_networks(
    data: dict[str, Any],
    figdir: Path,
    zones: list[str] | None = None,
) -> dict[str, str]:
    sample_id = data["sample_id"]
    edge = data["edge"]
    layout = data["layout"]
    cows = layout["cow_id"].astype(str).tolist()
    communities = layout_communities(layout)
    colors = cow_color_map(cows, seed=sample_id)
    zones = figure_05_zones(edge) if zones is None else list(zones)
    paths: dict[str, str] = {}
    known_cows = set(cows)
    for index, zone in enumerate(zones):
        networks = net_dominance_networks(edge, zone)
        visible_cows = drawable_network_cows(networks)
        unknown_cows = sorted(set(visible_cows) - known_cows, key=cow_sort_key)
        if unknown_cows:
            raise ValueError(f"Figure 5 edge endpoints are absent from the layout: {unknown_cows}")
        positions = circle_positions(visible_cows)
        max_weight = shared_network_max_weight(networks)
        fig, axes = plt.subplots(1, 2, figsize=(9, 4.8), squeeze=False)
        for col, layer in enumerate(["friendly", "unfriendly"]):
            draw_network(
                axes[0][col],
                networks[layer],
                positions,
                layer,
                f"dominant {layer}: {zone}",
                communities=communities,
                node_colors=colors,
                node_size=220,
                label_beside=False,
                max_weight=max_weight,
                alpha_column="dominance",
            )
        fig.suptitle(f"{sample_id} Figure 5. Zone-specific network: {zone}")
        key = figure_05_key(index, zone)
        paths[key] = save_figure(fig, output_path(figdir, f"{key}.png"))
    return paths


def net_adjacency_matrices(
    edge: pd.DataFrame,
    cows: list[str],
    min_dominance: float = FIGURE_NET_DOMINANCE_THRESHOLD,
) -> dict[str, pd.DataFrame]:
    networks = net_dominance_networks(edge, min_dominance=min_dominance)
    matrices: dict[str, pd.DataFrame] = {}
    for layer in ("friendly", "unfriendly"):
        matrix = pd.DataFrame(0.0, index=cows, columns=cows)
        for _, row in networks[layer].iterrows():
            cow_i = str(row["cow_i"])
            cow_j = str(row["cow_j"])
            if cow_i == cow_j:
                raise ValueError(f"Figure 6 received a self dyad: {cow_i}")
            value = abs(float(row["net_expected_seconds"]))
            if not math.isfinite(value):
                raise ValueError("Figure 6 received a non-finite net expected seconds value")
            if cow_i in matrix.index and cow_j in matrix.columns:
                matrix.loc[cow_i, cow_j] += value
                matrix.loc[cow_j, cow_i] += value
        matrices[layer] = matrix

    overlap = (matrices["friendly"] > 0.0) & (matrices["unfriendly"] > 0.0)
    if bool(overlap.to_numpy(dtype=bool).any()):
        raise RuntimeError("Figure 6 net adjacency assigned a dyad to both layers")
    return matrices


def adjacency_heatmap_vmax(matrices: dict[str, pd.DataFrame]) -> float:
    maxima = [
        float(pd.to_numeric(matrix.stack(), errors="coerce").fillna(0.0).max())
        for matrix in matrices.values()
        if not matrix.empty
    ]
    maximum = max(maxima, default=0.0)
    if not math.isfinite(maximum) or maximum < 0.0:
        raise ValueError(f"Figure 6 heatmap maximum must be finite and non-negative: {maximum}")
    return maximum if maximum > 0.0 else 1.0


def plot_adjacency_heatmaps(data: dict[str, Any], figdir: Path) -> str:
    sample_id = data["sample_id"]
    layout = data["layout"].copy()
    layout["community_id_full_friendly"] = pd.to_numeric(layout["community_id_full_friendly"], errors="coerce").fillna(-1)
    layout = layout.sort_values(["community_id_full_friendly", "cow_id"])
    cows = layout["cow_id"].astype(str).tolist()
    matrices = net_adjacency_matrices(data["edge"], cows)
    vmax = adjacency_heatmap_vmax(matrices)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, layer in zip(axes, ["friendly", "unfriendly"]):
        matrix = matrices[layer]
        image = ax.imshow(
            matrix.to_numpy(dtype=float),
            cmap="Greens" if layer == "friendly" else "Reds",
            vmin=0.0,
            vmax=vmax,
        )
        ax.set_title(f"{layer.capitalize()}-dominant net expected seconds")
        ax.set_xticks(range(len(cows)))
        ax.set_xticklabels([figure_identity_label(cow) for cow in cows], rotation=90, fontsize=8)
        ax.set_yticks(range(len(cows)))
        ax.set_yticklabels([figure_identity_label(cow) for cow in cows], fontsize=8)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(
        f"{sample_id} Figure 6. All-zone net adjacency sorted by community "
        f"(dominance >= {FIGURE_NET_DOMINANCE_THRESHOLD:.2f})"
    )
    return save_figure(fig, output_path(figdir, "figure_06_adjacency_heatmaps.png"))


def minmax_frame(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for col in columns:
        values = pd.to_numeric(df[col], errors="coerce")
        min_val = float(values.min()) if values.notna().any() else 0.0
        max_val = float(values.max()) if values.notna().any() else 0.0
        if math.isclose(min_val, max_val):
            out[col] = values.fillna(min_val).map(lambda _: 0.5 if max_val > 0 else 0.0)
        else:
            out[col] = (values - min_val) / (max_val - min_val)
    return out.fillna(0.0)


def padded_numeric_limits(values: pd.Series, left: float, right: float) -> tuple[float, float] | None:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return None
    min_val = float(numeric.min())
    max_val = float(numeric.max())
    span = max_val - min_val
    if math.isclose(span, 0.0):
        span = max(abs(max_val), 1e-3) * 0.2
    return min_val - span * left, max_val + span * right


def plot_descriptor_dashboard(data: dict[str, Any], figdir: Path) -> str:
    sample_id = data["sample_id"]
    node = data["node"].copy().sort_values("cow_id")
    metrics = [
        "friendly_strength_rate",
        "unfriendly_strength_rate",
        "unfriendly_ratio",
        "friendly_partner_diversity",
        "friendly_zone_diversity",
        "unfriendly_zone_diversity",
        "community_stability",
        "alone_fraction",
        "isolation_score",
    ]
    metrics = [col for col in metrics if col in node.columns]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [1, 1.45]})
    ax = axes[0]
    if not node.empty:
        x = pd.to_numeric(node["friendly_strength_rate"], errors="coerce").fillna(0)
        y = pd.to_numeric(node["unfriendly_strength_rate"], errors="coerce").fillna(0)
        isolation = (
            pd.to_numeric(node["isolation_score"], errors="coerce").fillna(0)
            if "isolation_score" in node.columns
            else pd.Series(0.0, index=node.index)
        )
        size = 80 + 320 * isolation
        colors = cow_color_map(node["cow_id"].astype(str).tolist(), seed=sample_id)
        point_colors = [colors.get(str(cow), "#4dabf7") for cow in node["cow_id"]]
        ax.scatter(
            x,
            y,
            s=size,
            color=point_colors,
            edgecolor="#212529",
            alpha=0.82,
        )
        for row_index, (_, row) in enumerate(node.iterrows()):
            label = figure_identity_label(row["cow_id"])
            marker_size = float(size.iloc[row_index])
            face_color = point_colors[row_index]
            ax.text(
                float(x.iloc[row_index]),
                float(y.iloc[row_index]),
                label,
                ha="center",
                va="center",
                fontsize=marker_label_font_size(marker_size, label),
                color=contrasting_text_color(face_color, alpha=0.82),
                zorder=3,
            )
        x_limits = padded_numeric_limits(x, left=0.12, right=0.24)
        y_limits = padded_numeric_limits(y, left=0.14, right=0.14)
        if x_limits:
            ax.set_xlim(*x_limits)
        if y_limits:
            ax.set_ylim(*y_limits)
    ax.set_xlabel("friendly_strength_rate")
    ax.set_ylabel("unfriendly_strength_rate")
    ax.set_title("rate scatter")

    ax = axes[1]
    if metrics and not node.empty:
        norm = minmax_frame(node, metrics)
        image = ax.imshow(norm[metrics].to_numpy(dtype=float), aspect="auto", cmap="viridis", vmin=0, vmax=1)
        ax.set_xticks(range(len(metrics)))
        ax.set_xticklabels(metrics, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(node)))
        ax.set_yticklabels(
            [figure_identity_label(cow) for cow in node["cow_id"].astype(str)],
            fontsize=8,
        )
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="min-max scaled")
    else:
        ax.text(0.5, 0.5, "No node descriptor rows", ha="center", va="center", transform=ax.transAxes)
    ax.set_title("descriptor heatmap")
    fig.suptitle(f"{sample_id} Figure 7. Cow-level descriptor dashboard")
    return save_figure(fig, output_path(figdir, "figure_07_descriptor_dashboard.png"))


def align_community_labels_for_plot(windows: pd.DataFrame) -> pd.DataFrame:
    """Align adjacent-window labels for color continuity without altering metrics."""
    required = {"window_index", "cow_id", "community_id"}
    missing = sorted(required - set(windows.columns))
    if missing:
        raise ValueError(f"Community-window columns are missing: {missing}")
    out = windows.copy()
    out["_plot_community_id"] = pd.Series(pd.NA, index=out.index, dtype="Int64")
    if out.empty:
        return out

    window_values = pd.to_numeric(out["window_index"], errors="raise")
    community_values = pd.to_numeric(out["community_id"], errors="coerce")
    finite_communities = community_values.dropna().to_numpy(dtype=float)
    if len(finite_communities) and not np.allclose(finite_communities, np.rint(finite_communities)):
        raise ValueError("Community IDs must be integer-valued")
    out["_numeric_window_index"] = window_values.astype(np.int64)
    out["_numeric_community_id"] = community_values.astype("Int64")

    next_label = 0
    previous_window_index: int | None = None
    previous_assignment: dict[str, int] = {}
    for window_index in sorted(out["_numeric_window_index"].unique()):
        current_mask = out["_numeric_window_index"] == int(window_index)
        current = out.loc[current_mask & out["_numeric_community_id"].notna()]
        raw_labels = sorted(int(value) for value in current["_numeric_community_id"].unique())
        label_map: dict[int, int] = {}

        if previous_window_index is not None and int(window_index) == previous_window_index + 1 and previous_assignment:
            previous_labels = sorted(set(previous_assignment.values()))
            overlap = np.zeros((len(previous_labels), len(raw_labels)), dtype=np.int64)
            previous_lookup = {label: index for index, label in enumerate(previous_labels)}
            current_lookup = {label: index for index, label in enumerate(raw_labels)}
            for _, row in current.iterrows():
                cow = str(row["cow_id"])
                if cow in previous_assignment:
                    overlap[
                        previous_lookup[previous_assignment[cow]],
                        current_lookup[int(row["_numeric_community_id"])],
                    ] += 1
            if overlap.size:
                previous_indices, current_indices = linear_sum_assignment(-overlap)
                for previous_index, current_index in zip(previous_indices, current_indices):
                    if overlap[previous_index, current_index] > 0:
                        label_map[raw_labels[current_index]] = previous_labels[previous_index]

        for raw_label in raw_labels:
            if raw_label not in label_map:
                label_map[raw_label] = next_label
                next_label += 1

        assigned = current["_numeric_community_id"].map(
            lambda value: label_map[int(value)] if pd.notna(value) else pd.NA
        )
        out.loc[current.index, "_plot_community_id"] = pd.array(assigned, dtype="Int64")
        previous_assignment = {
            str(cow): int(label)
            for cow, label in zip(current["cow_id"], out.loc[current.index, "_plot_community_id"])
            if pd.notna(label)
        }
        previous_window_index = int(window_index)

    return out.drop(columns=["_numeric_window_index", "_numeric_community_id"])


def sparse_tick_positions(count: int, max_ticks: int = 12) -> np.ndarray:
    if count <= 0:
        return np.empty(0, dtype=np.int64)
    if max_ticks <= 0:
        raise ValueError("max_ticks must be positive")
    if count <= max_ticks:
        return np.arange(count, dtype=np.int64)
    return np.unique(np.rint(np.linspace(0, count - 1, num=max_ticks)).astype(np.int64))


def complete_community_window_axis(data: dict[str, Any]) -> pd.DataFrame:
    summary = data["summary"]
    community_config = summary.get("config", {}).get("community", {})
    step_s = float(community_config.get("step_s", 0.0))
    window_s = float(community_config.get("window_s", 0.0))
    start_s = float(summary.get("time_window_start_s"))
    end_s = float(summary.get("time_window_end_s"))
    if not all(math.isfinite(value) for value in (step_s, window_s, start_s, end_s)):
        raise ValueError("Community window axis values must be finite")
    if step_s <= 0 or window_s <= 0 or end_s < start_s:
        raise ValueError(
            f"Invalid community window axis: start={start_s}, end={end_s}, "
            f"window={window_s}, step={step_s}"
        )
    count = int(math.floor((end_s - start_s + 1.0e-9) / step_s)) + 1
    if count != EXPECTED_COMMUNITY_WINDOW_COUNT:
        raise ValueError(
            f"Expected exactly {EXPECTED_COMMUNITY_WINDOW_COUNT} community windows; found={count}"
        )

    expected_pairs = [(index, index + 1) for index in range(count - 1)]
    pair_frame = data["community_summary"]
    required = {"window_a", "window_b"}
    missing = sorted(required - set(pair_frame.columns))
    if missing:
        raise ValueError(f"Community summary is missing window pair columns: {missing}")
    numeric_pairs = pair_frame[["window_a", "window_b"]].apply(pd.to_numeric, errors="coerce")
    if numeric_pairs.isna().any().any():
        raise ValueError("Community summary contains a non-numeric window pair")
    pair_values = numeric_pairs.to_numpy(dtype=float)
    if not np.allclose(pair_values, np.rint(pair_values)):
        raise ValueError("Community summary window pairs must be integer-valued")
    actual_pairs = sorted((int(row[0]), int(row[1])) for row in pair_values)
    if actual_pairs != expected_pairs:
        raise ValueError(
            "Community summary does not contain the complete adjacent-window sequence: "
            f"expected={len(expected_pairs)}, actual={len(actual_pairs)}"
        )

    indices = np.arange(count, dtype=np.int64)
    starts = start_s + indices.astype(float) * step_s
    return pd.DataFrame(
        {
            "window_index": indices,
            "window_start_s": starts,
            "window_end_s": starts + window_s,
        }
    )


def community_plot_matrix(data: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    axis = complete_community_window_axis(data)
    cows = sorted(data["layout"]["cow_id"].astype(str).unique(), key=cow_sort_key)
    expected_cows = [f"G{index:04d}" for index in range(1, EXPECTED_GLOBAL_IDENTITY_COUNT + 1)]
    if cows != expected_cows:
        raise ValueError(
            f"Figure 8 requires G0001..G{EXPECTED_GLOBAL_IDENTITY_COUNT:04d}; actual={cows}"
        )
    window_indices = axis["window_index"].astype(int).tolist()
    matrix = pd.DataFrame(np.nan, index=cows, columns=window_indices, dtype=float)

    windows = align_community_labels_for_plot(data["community_windows"])
    if windows.empty:
        return axis, matrix
    numeric_window = pd.to_numeric(windows["window_index"], errors="coerce")
    if numeric_window.isna().any() or not np.allclose(numeric_window, np.rint(numeric_window)):
        raise ValueError("Community assignment window indices must be integers")
    windows = windows.copy()
    windows["window_index"] = numeric_window.astype(np.int64)
    if windows.duplicated(["cow_id", "window_index"]).any():
        raise ValueError("Community assignments contain duplicate cow/window rows")
    unknown_cows = sorted(set(windows["cow_id"].astype(str)) - set(cows), key=cow_sort_key)
    unknown_windows = sorted(set(windows["window_index"].astype(int)) - set(window_indices))
    if unknown_cows or unknown_windows:
        raise ValueError(
            f"Community assignments are outside the Figure 8 grid: "
            f"unknown_cows={unknown_cows}, unknown_windows={unknown_windows}"
        )
    plot_ids = pd.to_numeric(windows["_plot_community_id"], errors="coerce")
    if plot_ids.isna().any() or (plot_ids < 0).any() or not np.allclose(plot_ids, np.rint(plot_ids)):
        raise ValueError("Aligned plot community IDs must be non-negative integers")
    for cow, window_index, plot_id in zip(
        windows["cow_id"].astype(str),
        windows["window_index"].astype(int),
        plot_ids.astype(int),
    ):
        matrix.loc[cow, window_index] = float(plot_id)
    return axis, matrix


def plot_community_stability(data: dict[str, Any], figdir: Path) -> str:
    sample_id = data["sample_id"]
    axis, matrix_frame = community_plot_matrix(data)
    matrix = matrix_frame.to_numpy(dtype=float)
    values = pd.Series(matrix.ravel()).dropna()
    ticks = sorted(int(value) for value in values.unique()) if not values.empty else []
    colors = [plt.get_cmap("tab20")(index % 20) for index, _ in enumerate(ticks)] or ["#ffffff"]
    cmap = ListedColormap(colors).with_extremes(bad=MISSING_COMMUNITY_COLOR)
    norm = None
    if ticks:
        if len(ticks) == 1:
            boundaries = [ticks[0] - 0.5, ticks[0] + 0.5]
        else:
            boundaries = [ticks[0] - 0.5]
            boundaries.extend((left + right) / 2.0 for left, right in zip(ticks, ticks[1:]))
            boundaries.append(ticks[-1] + 0.5)
        norm = BoundaryNorm(boundaries, cmap.N)

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.imshow(
        np.ma.masked_invalid(matrix),
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        norm=norm,
    )
    tick_positions = sparse_tick_positions(len(axis), max_ticks=12)
    ax.set_xticks(tick_positions)
    origin = float(axis["window_start_s"].iloc[0])
    ax.set_xticklabels(
        [f"{(float(axis['window_start_s'].iloc[position]) - origin) / 3600.0:.1f}h" for position in tick_positions],
        fontsize=8,
    )
    ax.set_yticks(range(len(matrix_frame.index)))
    ax.set_yticklabels(
        [figure_identity_label(cow) for cow in matrix_frame.index.astype(str)],
        fontsize=8,
    )
    ax.legend(
        handles=[
            MplPatch(
                facecolor=MISSING_COMMUNITY_COLOR,
                edgecolor="#adb5bd",
                label="Unavailable / <30 s visible / no eligible friendly edge",
            )
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
        frameon=False,
        fontsize=8,
    )
    ax.set_xlabel("elapsed time (hours; complete 5-minute windows)")
    ax.set_title("community membership")
    fig.suptitle(f"{sample_id} Figure 8. Community stability over time (5-minute windows)")
    return save_figure(fig, output_path(figdir, "figure_08_community_stability.png"))


def plot_community_similarity(data: dict[str, Any], figdir: Path) -> str:
    sample_id = data["sample_id"]
    summary = data["community_summary"].copy()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    if not summary.empty:
        x = np.arange(len(summary))
        plotted = False
        if "network_nmi" in summary.columns:
            values = pd.to_numeric(summary["network_nmi"], errors="coerce")
            if values.notna().any():
                ax.plot(x, values, marker="o", label="NMI")
                plotted = True
        if "network_ari" in summary.columns:
            values = pd.to_numeric(summary["network_ari"], errors="coerce")
            if values.notna().any():
                ax.plot(x, values, marker="s", label="ARI")
                plotted = True
        if plotted:
            ax.set_ylim(-1.05, 1.05)
            ax.set_xlabel("adjacent 5-minute window pair")
            ax.legend(frameon=False)
        else:
            ax.text(0.5, 0.5, "No comparable adjacent windows", ha="center", va="center", transform=ax.transAxes)
    else:
        ax.text(0.5, 0.5, "No stability summary", ha="center", va="center", transform=ax.transAxes)
    ax.set_title("window-to-window similarity")
    fig.suptitle(f"{sample_id} Figure 8B. Community similarity (5-minute windows)")
    return save_figure(fig, output_path(figdir, "figure_08B_community_similarity.png"))


def isolation_weights(summary: dict[str, Any]) -> dict[str, float]:
    weights = summary.get("config", {}).get("isolation", {}).get("weights", {})
    return {
        "alone_fraction": float(weights.get("alone_fraction", 0.4)),
        "low_friendly_sociality": float(weights.get("low_friendly_sociality", 0.3)),
        "low_partner_diversity": float(weights.get("low_partner_diversity", 0.3)),
    }


def plot_isolation_decomposition(data: dict[str, Any], figdir: Path) -> str:
    sample_id = data["sample_id"]
    node = data["node"].copy()
    weights = isolation_weights(data["summary"])
    components = list(weights)
    for col in components:
        if col not in node.columns:
            node[col] = 0.0
        node[f"{col}_component"] = pd.to_numeric(node[col], errors="coerce").fillna(0) * weights[col]
    node = node.sort_values("isolation_score", ascending=False)
    fig, ax = plt.subplots(figsize=(11, 5))
    if not node.empty:
        x = np.arange(len(node))
        bottom = np.zeros(len(node))
        colors = ["#f08c00", "#e03131", "#5c7cfa"]
        for col, color in zip(components, colors):
            values = node[f"{col}_component"].to_numpy(dtype=float)
            ax.bar(x, values, bottom=bottom, label=col, color=color, alpha=0.86)
            bottom += values
        ax.set_xticks(x)
        ax.set_xticklabels(
            [figure_identity_label(cow) for cow in node["cow_id"].astype(str)],
            rotation=45,
            ha="right",
        )
        ax.legend(frameon=False, fontsize=8)
    else:
        ax.text(0.5, 0.5, "No node descriptor rows", ha="center", va="center", transform=ax.transAxes)
    ax.set_ylabel("isolation score contribution")
    ax.set_title(f"{sample_id} Figure 9. Isolation tendency decomposition")
    return save_figure(fig, output_path(figdir, "figure_09_isolation_decomposition.png"))


def plot_quality_control(data: dict[str, Any], figdir: Path) -> str:
    sample_id = data["sample_id"]
    edge = data["edge"].copy()
    fig, ax = plt.subplots(figsize=(7, 5))
    if edge.empty:
        ax.text(0.5, 0.5, "No edge rows", ha="center", va="center", transform=ax.transAxes)
    else:
        for layer in sorted(edge["layer"].astype(str).unique()):
            sub = edge.loc[edge["layer"].astype(str) == layer]
            ax.scatter(
                pd.to_numeric(sub["opportunity_seconds"], errors="coerce"),
                pd.to_numeric(sub["expected_seconds"], errors="coerce"),
                label=layer,
                color=layer_color(layer),
                alpha=0.75,
            )
        ax.set_xlabel("opportunity_seconds")
        ax.set_ylabel("expected_seconds")
        ax.legend(frameon=False, fontsize=8)
        ax.set_title("expected vs opportunity")
    fig.suptitle(f"{sample_id} Figure 10. Expected vs opportunity")
    return save_figure(fig, output_path(figdir, "figure_10_quality_control.png"))


def table_analysis_summary(data: dict[str, Any]) -> pd.DataFrame:
    summary = data["summary"]
    node = data["node"]
    events = data["events"]
    reliable = int(node["cow_reliable"].astype(str).str.lower().isin(["true", "1"]).sum()) if "cow_reliable" in node else 0
    return pd.DataFrame(
        [
            {
                "sample_id": data["sample_id"],
                "farm": summary.get("config", {}).get("analysis", {}).get("farm"),
                "camera": summary.get("config", {}).get("analysis", {}).get("camera"),
                "clip": summary.get("config", {}).get("analysis", {}).get("clip"),
                "time_window_start_s": summary.get("time_window_start_s"),
                "time_window_end_s": summary.get("time_window_end_s"),
                "n_cows": summary.get("n_cows"),
                "n_reliable_cows": reliable,
                "n_frames": summary.get("n_frames"),
                "n_valid_pair_frames": summary.get("n_valid_pair_frames"),
                "n_interaction_rows": summary.get("n_interaction_rows"),
                "n_edge_rows": int(len(data["edge"])),
                "n_events": int(len(events)),
                "n_dropped_self_pairs": summary.get("n_dropped_self_pairs"),
                "n_dropped_missing_trajectory": summary.get("n_dropped_missing_trajectory"),
                "n_unknown_zone_points": summary.get("n_unknown_zone_points"),
                "warnings": "; ".join(str(item) for item in summary.get("warnings", [])),
                "community_stability_policy": summary.get("config", {}).get("metadata", {}).get("community_window_policy"),
                "community_stability_note": summary.get("config", {}).get("metadata", {}).get("community_window_note"),
            }
        ]
    )


def table_top_dyads(data: dict[str, Any], layer: str) -> pd.DataFrame:
    ranking = data["edge_ranking"].copy()
    columns = [
        "zone",
        "rank_by_expected_seconds",
        "rank_by_normalized_rate",
        "cow_i",
        "cow_j",
        "expected_seconds",
        "opportunity_seconds",
        "normalized_rate",
        "mean_distance",
        "num_valid_frames",
        "edge_reliable",
    ]
    if ranking.empty:
        return pd.DataFrame(columns=columns)
    rows = ranking.loc[
        (ranking["layer"].astype(str) == layer)
        & (ranking["ranking_scope"].astype(str) == f"{layer} by zone")
        & (ranking["zone"].astype(str) != "all_zones")
    ].copy()
    if rows.empty:
        return pd.DataFrame(columns=columns)
    rows = rows.loc[pd.to_numeric(rows["expected_seconds"], errors="coerce").fillna(0) > 0].copy()
    if rows.empty:
        return pd.DataFrame(columns=columns)
    rows = rows.sort_values(["zone", "rank_by_expected_seconds", "rank_by_normalized_rate", "cow_i", "cow_j"])
    return rows[columns]


def rank_desc(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0).rank(method="min", ascending=False).astype(int)


def table_cow_descriptors(data: dict[str, Any]) -> pd.DataFrame:
    node = data["node"].copy()
    selected = [
        "cow_id",
        "cow_reliable",
        "visible_time_s",
        "friendly_strength_rate",
        "unfriendly_strength_rate",
        "unfriendly_ratio",
        "friendly_partner_diversity",
        "friendly_zone_diversity",
        "community_stability",
        "alone_fraction",
        "isolation_score",
    ]
    for col in selected:
        if col not in node.columns:
            node[col] = np.nan
    node["rank_friendly_strength_rate"] = rank_desc(node["friendly_strength_rate"])
    node["rank_unfriendly_strength_rate"] = rank_desc(node["unfriendly_strength_rate"])
    node["rank_partner_diversity"] = rank_desc(node["friendly_partner_diversity"])
    node["rank_isolation_score"] = rank_desc(node["isolation_score"])
    return node[
        [
            "cow_id",
            "cow_reliable",
            "visible_time_s",
            "rank_friendly_strength_rate",
            "rank_unfriendly_strength_rate",
            "rank_partner_diversity",
            "rank_isolation_score",
            "friendly_strength_rate",
            "unfriendly_strength_rate",
            "unfriendly_ratio",
            "friendly_partner_diversity",
            "friendly_zone_diversity",
            "community_stability",
            "alone_fraction",
            "isolation_score",
        ]
    ].sort_values(["rank_isolation_score", "cow_id"])


def table_zone_profiles(data: dict[str, Any]) -> pd.DataFrame:
    zone_node = data["zone_node"].copy()
    selected = [
        "cow_id",
        "zone",
        "time_in_zone_s",
        "weighted_time_in_zone_s",
        "friendly_strength_rate_zone",
        "unfriendly_strength_rate_zone",
        "friendly_active_partners_zone",
        "unfriendly_active_partners_zone",
        "alone_fraction_zone",
    ]
    for col in selected:
        if col not in zone_node.columns:
            zone_node[col] = np.nan
    return zone_node[selected].sort_values(["cow_id", "zone"])


def table_community_summary(data: dict[str, Any]) -> pd.DataFrame:
    summary = data["community_summary"].copy()
    metadata = data["summary"].get("config", {}).get("metadata", {})
    if summary.empty:
        return pd.DataFrame(
            columns=[
                "window_a",
                "window_b",
                "n_common_cows",
                "network_nmi",
                "network_ari",
                "community_stability_policy",
                "community_stability_note",
            ]
        )
    summary["community_stability_policy"] = metadata.get(
        "community_window_policy", "fixed_long_video_5min_nonoverlap"
    )
    summary["community_stability_note"] = metadata.get("community_window_note", "")
    return summary


def write_tables(data: dict[str, Any], tabledir: Path) -> dict[str, str]:
    paths = {
        "table_01_analysis_window_quality": write_table(
            table_analysis_summary(data),
            output_path(tabledir, "table_01_analysis_window_quality.csv"),
        ),
        "table_02_top_friendly_dyads_by_zone": write_table(
            table_top_dyads(data, "friendly"),
            output_path(tabledir, "table_02_top_friendly_dyads_by_zone.csv"),
        ),
        "table_03_top_unfriendly_dyads_by_zone": write_table(
            table_top_dyads(data, "unfriendly"),
            output_path(tabledir, "table_03_top_unfriendly_dyads_by_zone.csv"),
        ),
        "table_04_cow_descriptor_ranking": write_table(
            table_cow_descriptors(data),
            output_path(tabledir, "table_04_cow_descriptor_ranking.csv"),
        ),
        "table_05_zone_specific_cow_profiles": write_table(
            table_zone_profiles(data),
            output_path(tabledir, "table_05_zone_specific_cow_profiles.csv"),
        ),
        "table_06_community_stability_summary": write_table(
            table_community_summary(data),
            output_path(tabledir, "table_06_community_stability_summary.csv"),
        ),
    }
    return paths


def write_readme(sample_id: str, outdir: Path, figure_paths: dict[str, str], table_paths: dict[str, str]) -> str:
    lines = [
        f"# SNA_TF outputs for sample {sample_id}",
        "",
        "Figures:",
    ]
    for key, path in sorted(figure_paths.items()):
        lines.append(f"- {key}: {Path(path).name}")
    lines.extend(["", "Tables:"])
    for key, path in sorted(table_paths.items()):
        lines.append(f"- {key}: {Path(path).name}")
    lines.extend(
        [
            "",
            "Notes:",
            "- Zone names are kept as existing data labels: food/rest/wait_for_water/water/path.",
            "- Empty interaction event tables are intentionally preserved when thresholds produce no events.",
            "- Community stability uses fixed, non-overlapping 5-minute windows; see table 06 and analysis_summary metadata.",
            "- Frozen WebUIL display points are included in SNA trajectories by design.",
            "- Cow node and trajectory colors use sample-seeded random indices from dairy_social/voc_colors.py.",
        ]
    )
    path = outdir / "README.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def generate_sample(sample_id: str, outdir: Path) -> dict[str, Any]:
    log(f"[sample] {sample_id}")
    data = load_sample(sample_id)
    zones = figure_05_zones(data["edge"])
    figdir = outdir / "figures"
    log("[figure] 01 floorplan trajectories in seven cow groups")
    figure_paths: dict[str, str] = plot_floorplan_trajectories(data, figdir)
    log("[figure] 02 visibility and zone time budget")
    figure_paths.update({
        "figure_02_visibility_zone_time_budget": plot_visibility_time_budget(data, figdir),
    })
    log("[figure] 03 interaction volume by zone")
    figure_paths.update({
        "figure_03_interaction_volume_by_zone": plot_zone_interaction_volume(data, figdir),
    })
    log("[figure] 04 full-window friendly/unfriendly networks")
    figure_paths.update(plot_full_networks(data, figdir))
    log(f"[figure] 05 zone-specific networks: {len(zones)} regions")
    figure_paths.update(plot_zone_networks(data, figdir, zones))
    log("[figure] 06 adjacency heatmaps")
    figure_paths.update({"figure_06_adjacency_heatmaps": plot_adjacency_heatmaps(data, figdir)})
    log("[figure] 07 descriptor dashboard")
    figure_paths.update({"figure_07_descriptor_dashboard": plot_descriptor_dashboard(data, figdir)})
    log("[figure] 08 community stability and similarity")
    figure_paths.update({
        "figure_08_community_stability": plot_community_stability(data, figdir),
        "figure_08B_community_similarity": plot_community_similarity(data, figdir),
    })
    log("[figure] 09 isolation decomposition")
    figure_paths.update({"figure_09_isolation_decomposition": plot_isolation_decomposition(data, figdir)})
    log("[figure] 10 quality control")
    figure_paths.update({"figure_10_quality_control": plot_quality_control(data, figdir)})
    return {
        "sampleId": sample_id,
        "outdir": str(outdir),
        "figures": figure_paths,
        "expectedFigureRelativePaths": sorted(expected_figure_relative_paths(zones)),
    }


def choose_samples(args: argparse.Namespace) -> list[str]:
    samples = [str(args.sample)]
    if samples != [COMBINED_SAMPLE_ID]:
        raise RuntimeError(
            f"This command only emits the one combined Farm1/Gopro1 figure suite: {COMBINED_SAMPLE_ID}"
        )
    missing = {sample: missing_required_sample_files(sample) for sample in samples}
    missing = {sample: paths for sample, paths in missing.items() if paths}
    if missing:
        details = "; ".join(f"{sample}: {paths}" for sample, paths in missing.items())
        raise RuntimeError(f"SNA output files are missing for samples: {details}")
    return samples


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate the one combined Farm1/Gopro1 SNA figure suite.")
    parser.add_argument("--sample", required=True, help=f"Must be {COMBINED_SAMPLE_ID}.")
    parser.add_argument("--inputs-dir", default=str(SNA_INPUT_DIR), help="Directory containing SNA input sample folders.")
    parser.add_argument("--sna-output-dir", default=str(SNA_OUTPUT_DIR), help="Directory containing SNA analysis output sample folders.")
    parser.add_argument("--outdir", default=str(SNA_TF_OUTPUT_DIR))
    parser.add_argument("--overwrite", action="store_true", help="Atomically replace the existing figure suite.")
    return parser


def validate_figure_suite(sample_dir: Path, expected_relative_paths: set[str]) -> None:
    files = [path for path in sample_dir.rglob("*") if path.is_file()]
    expected_paths = {sample_dir / relative_path for relative_path in expected_relative_paths}
    if set(files) != expected_paths:
        missing = sorted(str(path.relative_to(sample_dir)) for path in expected_paths - set(files))
        extra = sorted(str(path.relative_to(sample_dir)) for path in set(files) - expected_paths)
        raise RuntimeError(f"Figure suite file set is not exact; missing={missing}, extra={extra}")
    for path in sorted(files):
        try:
            with Image.open(path) as image:
                width, height = image.size
                if image.format != "PNG":
                    raise RuntimeError(f"Figure file is not PNG format: {path}: {image.format}")
                image.verify()
        except Exception as exc:
            raise RuntimeError(f"Figure is not a valid PNG: {path}") from exc
        if width <= 0 or height <= 0 or path.stat().st_size <= 24:
            raise RuntimeError(f"Figure PNG is empty or dimensionless: {path}")


def publish_figure_suite(staging: Path, target: Path, overwrite: bool) -> None:
    if target.exists() and not target.is_dir():
        raise RuntimeError(f"Figure target exists and is not a directory: {target}")
    if target.exists() and not overwrite:
        raise RuntimeError(f"Figure target already exists; pass --overwrite to replace it: {target}")
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


def main(argv: list[str] | None = None) -> int:
    global SNA_INPUT_DIR, SNA_OUTPUT_DIR
    args = build_arg_parser().parse_args(argv)
    SNA_INPUT_DIR = Path(args.inputs_dir)
    SNA_OUTPUT_DIR = Path(args.sna_output_dir)
    samples = choose_samples(args)
    base_outdir = Path(args.outdir)
    base_outdir.mkdir(parents=True, exist_ok=True)
    log(f"[info] output: {base_outdir}")
    target = base_outdir / samples[0]
    unexpected = [path for path in base_outdir.iterdir() if path != target]
    if unexpected:
        raise RuntimeError(
            "Figure output root must be dedicated to the one combined suite; unexpected entries: "
            f"{[str(path) for path in unexpected]}"
        )
    if target.exists() and not args.overwrite:
        raise RuntimeError(f"Figure target already exists; pass --overwrite to replace it: {target}")
    staging = base_outdir / f".{target.name}.{os.getpid()}.tmp"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    expected_relative_paths: set[str] | None = None
    try:
        result = generate_sample(samples[0], staging)
        expected_relative_paths = set(result["expectedFigureRelativePaths"])
        validate_figure_suite(staging, expected_relative_paths)
        publish_figure_suite(staging, target, args.overwrite)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    if expected_relative_paths is None:  # pragma: no cover - generation failures raise above.
        raise RuntimeError("Figure output contract was not initialized")
    validate_figure_suite(target, expected_relative_paths)
    log(f"[done] figures={target / 'figures'} count={len(expected_relative_paths)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
