from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .geometry import euclidean_distance


def filter_by_config(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    analysis = config["analysis"]
    out = df.copy()
    out = out.loc[out["farm"].astype(str) == str(analysis["farm"])]
    out = out.loc[out["camera"].astype(str) == str(analysis["camera"])]
    if analysis.get("clip") is not None:
        out = out.loc[out["clip"].astype(str) == str(analysis["clip"])]
    if analysis.get("start_time_s") is not None:
        out = out.loc[out["time_s"].astype(float) >= float(analysis["start_time_s"])]
    if analysis.get("end_time_s") is not None:
        out = out.loc[out["time_s"].astype(float) < float(analysis["end_time_s"])]
    return out.reset_index(drop=True)


def get_frame_durations(traj_df: pd.DataFrame, interactions_df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    fps = config.get("time", {}).get("fps")
    keys = ["farm", "camera", "clip", "frame", "time_s"]
    frames = pd.concat([traj_df[keys], interactions_df[keys]], ignore_index=True).drop_duplicates()
    frames = frames.sort_values(keys).reset_index(drop=True)
    if frames.empty:
        raise ValueError("no frames are available after filtering")
    if fps is not None:
        frames["dt"] = 1.0 / float(fps)
        return frames
    default_dt = config.get("time", {}).get("default_dt_s")
    dts: list[float] = []
    for _, group in frames.groupby(["farm", "camera", "clip"], sort=False):
        times = group["time_s"].to_numpy(dtype=float)
        diffs = np.diff(times)
        positive = diffs[diffs > 0]
        last_dt = float(np.median(positive)) if len(positive) else (float(default_dt) if default_dt is not None else math.nan)
        if not math.isfinite(last_dt) or last_dt <= 0:
            raise ValueError("Cannot infer frame durations; set time.fps or time.default_dt_s")
        vals = [float(delta) if delta > 0 else last_dt for delta in diffs] + [last_dt]
        dts.extend(vals)
    frames["dt"] = dts
    if (frames["dt"] <= 0).any():
        raise ValueError("negative or zero frame duration encountered")
    return frames


def pair_zone(zone_i: str, zone_j: str, directed: bool, mode: str) -> str | None:
    zone_i = str(zone_i)
    zone_j = str(zone_j)
    if mode == "same_or_cross":
        if zone_i == zone_j:
            return zone_i
        if zone_i == "path":
            return zone_j
        if zone_j == "path":
            return zone_i
        return "cross_zone"
    if mode == "same_only":
        return zone_i if zone_i == zone_j else None
    if mode == "zone_pair":
        if directed:
            return f"{zone_i}__to__{zone_j}"
        return "__".join(sorted([zone_i, zone_j]))
    raise ValueError(f"Unsupported pair_zone_mode: {mode}")


def aggregate_edges(traj_df: pd.DataFrame, interactions_df: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    warnings: list[str] = []
    directed = bool(config["network"].get("directed", False))
    mode = str(config["network"].get("pair_zone_mode", "same_or_cross"))
    eps = float(config["network"].get("eps", 1.0e-9))
    min_opportunity = float(config["network"].get("min_opportunity_s", 5.0))

    durations = get_frame_durations(traj_df, interactions_df, config)
    id_cols = ["farm", "camera", "clip", "frame"]
    duration_cols = id_cols + ["time_s", "dt"]
    interactions = interactions_df.merge(durations[duration_cols], on=id_cols + ["time_s"], how="left")

    left = traj_df[id_cols + ["cow_id", "anchor_x", "anchor_y", "track_conf", "zone"]].rename(
        columns={
            "cow_id": "cow_i",
            "anchor_x": "anchor_x_i",
            "anchor_y": "anchor_y_i",
            "track_conf": "track_conf_i",
            "zone": "zone_i",
        }
    )
    right = traj_df[id_cols + ["cow_id", "anchor_x", "anchor_y", "track_conf", "zone"]].rename(
        columns={
            "cow_id": "cow_j",
            "anchor_x": "anchor_x_j",
            "anchor_y": "anchor_y_j",
            "track_conf": "track_conf_j",
            "zone": "zone_j",
        }
    )
    joined = interactions.merge(left, on=id_cols + ["cow_i"], how="left").merge(right, on=id_cols + ["cow_j"], how="left")
    missing = joined[["anchor_x_i", "anchor_x_j"]].isna().any(axis=1)
    dropped_missing = int(missing.sum())
    if dropped_missing:
        warnings.append(f"dropped {dropped_missing} interaction rows with missing trajectory")
        joined = joined.loc[~missing].copy()

    if joined.empty:
        return empty_edge_df(), {
            "warnings": warnings,
            "n_dropped_missing_trajectory": dropped_missing,
            "n_opportunity_pair_frames": 0,
        }

    use_track_conf = bool(config["network"].get("use_track_confidence", True))
    use_interaction_conf = bool(config["network"].get("use_interaction_confidence", True))
    track_conf = joined["track_conf_i"].astype(float) * joined["track_conf_j"].astype(float) if use_track_conf else 1.0
    interaction_conf = joined["interaction_conf"].astype(float) if use_interaction_conf else 1.0
    joined["conf"] = track_conf * interaction_conf
    joined["opportunity_eligible"] = joined["opportunity_eligible"].astype(float) > 0.0
    joined["distance"] = [
        euclidean_distance(x1, y1, x2, y2)
        for x1, y1, x2, y2 in zip(joined["anchor_x_i"], joined["anchor_y_i"], joined["anchor_x_j"], joined["anchor_y_j"])
    ]
    joined["zone"] = [
        pair_zone(zi, zj, directed, mode)
        for zi, zj in zip(joined["zone_i"], joined["zone_j"])
    ]
    joined = joined.loc[joined["zone"].notna()].copy()
    joined = joined.loc[joined["opportunity_eligible"]].copy()

    if joined.empty:
        return empty_edge_df(), {
            "warnings": warnings,
            "n_dropped_missing_trajectory": dropped_missing,
            "n_opportunity_pair_frames": 0,
        }

    n_opportunity_pair_frames = int(len(joined))
    joined["opportunity"] = joined["dt"].astype(float) * joined["conf"].astype(float)
    joined["opportunity_raw"] = joined["dt"].astype(float)

    rows: list[dict[str, Any]] = []
    for layer, prob_col in (("friendly", "p_friendly"), ("unfriendly", "p_unfriendly")):
        current = joined.copy()
        current["layer"] = layer
        current["contribution"] = current["dt"].astype(float) * current[prob_col].astype(float) * current["conf"].astype(float)
        group_cols = ["farm", "camera", "clip", "cow_i", "cow_j", "zone", "layer"]
        for key, group in current.groupby(group_cols, sort=True):
            opportunity = float(group["opportunity"].sum())
            dist_weight = group["opportunity"].to_numpy(dtype=float)
            distances = group["distance"].to_numpy(dtype=float)
            mean_distance = float(np.average(distances, weights=dist_weight)) if dist_weight.sum() > 0 else float(np.mean(distances))
            expected = float(group["contribution"].sum())
            rows.append(
                {
                    "farm": key[0],
                    "camera": key[1],
                    "clip": key[2],
                    "window_start_s": float(group["time_s"].min()),
                    "window_end_s": float((group["time_s"] + group["dt"]).max()),
                    "cow_i": str(key[3]),
                    "cow_j": str(key[4]),
                    "zone": key[5],
                    "layer": key[6],
                    "expected_seconds": expected,
                    "opportunity_seconds": opportunity,
                    "opportunity_raw_seconds": float(group["opportunity_raw"].sum()),
                    "normalized_rate": expected / (opportunity + eps),
                    "mean_distance": mean_distance,
                    "num_valid_frames": int(len(group)),
                    "edge_reliable": bool(opportunity >= min_opportunity),
                }
            )
    edge_df = pd.DataFrame(rows, columns=edge_columns())
    return edge_df, {
        "warnings": warnings,
        "n_dropped_missing_trajectory": dropped_missing,
        "n_opportunity_pair_frames": n_opportunity_pair_frames,
    }


def edge_columns() -> list[str]:
    return [
        "farm",
        "camera",
        "clip",
        "window_start_s",
        "window_end_s",
        "cow_i",
        "cow_j",
        "zone",
        "layer",
        "expected_seconds",
        "opportunity_seconds",
        "opportunity_raw_seconds",
        "normalized_rate",
        "mean_distance",
        "num_valid_frames",
        "edge_reliable",
    ]


def empty_edge_df() -> pd.DataFrame:
    return pd.DataFrame(columns=edge_columns())


def compute_time_budget(traj_df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    durations = get_frame_durations(traj_df, pd.DataFrame(columns=traj_df.columns), config)
    rows = traj_df.merge(durations[["farm", "camera", "clip", "frame", "time_s", "dt"]], on=["farm", "camera", "clip", "frame", "time_s"], how="left")
    rows = rows.loc[rows["anchor_x"].notna() & rows["anchor_y"].notna()].copy()
    rows["weighted_dt"] = rows["dt"].astype(float) * rows["track_conf"].astype(float)
    zones = sorted(rows["zone"].astype(str).unique())
    min_visible = float(config["network"].get("min_visible_time_s", 60.0))
    out_rows: list[dict[str, Any]] = []
    for cow_id, group in rows.groupby("cow_id", sort=True):
        visible = float(group["dt"].sum())
        weighted_visible = float(group["weighted_dt"].sum())
        payload: dict[str, Any] = {
            "cow_id": str(cow_id),
            "visible_time_s": visible,
            "weighted_visible_time_s": weighted_visible,
            "cow_reliable": bool(visible >= min_visible),
        }
        for zone in zones:
            zone_group = group.loc[group["zone"].astype(str) == zone]
            zone_time = float(zone_group["dt"].sum())
            weighted_zone = float(zone_group["weighted_dt"].sum())
            payload[f"time_{safe_name(zone)}_s"] = zone_time
            payload[f"weighted_time_{safe_name(zone)}_s"] = weighted_zone
            payload[f"frac_{safe_name(zone)}"] = zone_time / visible if visible > 0 else 0.0
        out_rows.append(payload)
    return pd.DataFrame(out_rows).fillna(0.0)


def safe_name(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z_]+", "_", str(value)).strip("_") or "zone"


def adjacency_matrix(edge_df: pd.DataFrame, cows: list[str], layer: str, zone: str | None, value_col: str, directed: bool) -> pd.DataFrame:
    matrix = pd.DataFrame(0.0, index=cows, columns=cows)
    if edge_df.empty:
        return matrix
    sub = edge_df.loc[edge_df["layer"] == layer]
    if zone is not None:
        sub = sub.loc[sub["zone"] == zone]
    for _, row in sub.iterrows():
        i = str(row["cow_i"])
        j = str(row["cow_j"])
        value = float(row[value_col])
        if i in matrix.index and j in matrix.columns:
            matrix.loc[i, j] += value
            if not directed:
                matrix.loc[j, i] += value
    return matrix


def write_adjacency_matrices(edge_df: pd.DataFrame, outdir: str | Path, config: dict[str, Any], cows: list[str]) -> None:
    outdir = Path(outdir)
    directed = bool(config["network"].get("directed", False))
    zones = sorted(edge_df["zone"].astype(str).unique()) if not edge_df.empty else []
    for layer in ("friendly", "unfriendly"):
        for value_col in ("expected_seconds", "normalized_rate"):
            adjacency_matrix(edge_df, cows, layer, None, value_col, directed).to_csv(
                outdir / f"adjacency_{layer}_all_zones_{value_col}.csv"
            )
            for zone in zones:
                adjacency_matrix(edge_df, cows, layer, zone, value_col, directed).to_csv(
                    outdir / f"adjacency_{layer}_{safe_name(zone)}_{value_col}.csv"
                )
