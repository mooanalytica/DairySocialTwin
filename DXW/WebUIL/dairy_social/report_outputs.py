from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from .aggregation import aggregate_edges, compute_time_budget, get_frame_durations, pair_zone
from .communities import detect_communities_from_edges
from .descriptors import compute_node_descriptors, finalize_isolation_score
from .geometry import euclidean_distance, point_in_polygon
from .zones import normalized_zones


AggregateEdgesFn = Callable[[pd.DataFrame, pd.DataFrame, dict[str, Any]], tuple[pd.DataFrame, dict[str, Any]]]


REPORT_OUTPUTS = {
    "cow_frame_zone": "cow_frame_zone.csv",
    "zone_occupancy_timeseries": "zone_occupancy_timeseries.csv",
    "temporal_edge_level": "temporal_edge_level.csv",
    "temporal_node_descriptors": "temporal_node_descriptors.csv",
    "interaction_events": "interaction_events.csv",
    "edge_ranking": "edge_ranking.csv",
    "report_network_layout": "report_network_layout.csv",
}


REPORT_OUTPUT_COLUMNS: dict[str, tuple[str, ...]] = {
    "zone_occupancy_timeseries": (
        "farm",
        "camera",
        "clip",
        "bin_start_s",
        "bin_end_s",
        "zone",
        "n_cows",
        "cow_ids",
        "mean_track_conf",
        "total_cow_seconds",
    ),
    "temporal_edge_level": (
        "farm",
        "camera",
        "clip",
        "bin_start_s",
        "bin_end_s",
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
    ),
    "temporal_node_descriptors": (
        "farm",
        "camera",
        "clip",
        "bin_start_s",
        "bin_end_s",
        "cow_id",
        "visible_time_s",
        "friendly_strength_rate",
        "unfriendly_strength_rate",
        "unfriendly_ratio",
        "friendly_partner_diversity",
        "unfriendly_partner_diversity",
        "friendly_active_partners",
        "unfriendly_active_partners",
        "alone_fraction",
        "isolation_score",
        "cow_bin_reliable",
    ),
}


def report_output_columns(output_name: str) -> list[str]:
    try:
        return list(REPORT_OUTPUT_COLUMNS[output_name])
    except KeyError as exc:
        raise ValueError(f"No column schema registered for report output {output_name!r}") from exc


def empty_report_output(output_name: str) -> pd.DataFrame:
    return pd.DataFrame(columns=report_output_columns(output_name))


def write_report_outputs(
    traj_df: pd.DataFrame,
    interactions_df: pd.DataFrame,
    zones: dict[str, Any],
    edge_df: pd.DataFrame,
    node_df: pd.DataFrame,
    time_budget_df: pd.DataFrame,
    outdir: str | Path,
    config: dict[str, Any],
    aggregate_edges_fn: AggregateEdgesFn = aggregate_edges,
) -> tuple[dict[str, str], pd.DataFrame]:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cow_frame_df = build_cow_frame_zone(traj_df, interactions_df, zones, config)
    if bool(config.get("report", {}).get("write_temporal_outputs", True)):
        temporal_edge_df = build_temporal_edge_level(traj_df, interactions_df, config, aggregate_edges_fn)
        temporal_node_df = build_temporal_node_descriptors(traj_df, interactions_df, config, aggregate_edges_fn)
        occupancy_df = build_zone_occupancy_timeseries(cow_frame_df, config)
    else:
        temporal_edge_df = empty_report_output("temporal_edge_level")
        temporal_node_df = empty_report_output("temporal_node_descriptors")
        occupancy_df = empty_report_output("zone_occupancy_timeseries")
    events_df = build_interaction_events(traj_df, interactions_df, config)
    ranking_df = build_edge_ranking(edge_df, config)
    layout_df = build_report_network_layout(cow_frame_df, edge_df, node_df, time_budget_df, config)

    frames = {
        "cow_frame_zone": cow_frame_df,
        "zone_occupancy_timeseries": occupancy_df,
        "temporal_edge_level": temporal_edge_df,
        "temporal_node_descriptors": temporal_node_df,
        "interaction_events": events_df,
        "edge_ranking": ranking_df,
        "report_network_layout": layout_df,
    }
    paths: dict[str, str] = {}
    for key, filename in REPORT_OUTPUTS.items():
        path = outdir / filename
        frames[key].to_csv(path, index=False)
        paths[key] = str(path)
    return paths, layout_df


def report_time_bin_s(config: dict[str, Any]) -> float:
    value = config.get("report", {}).get("time_bin_s")
    if value is None:
        value = config.get("community", {}).get("window_s", 60.0)
    out = float(value)
    if out <= 0:
        raise ValueError("report.time_bin_s must be positive")
    return out


def analysis_identity(traj_df: pd.DataFrame, config: dict[str, Any]) -> dict[str, str]:
    if not traj_df.empty:
        row = traj_df.iloc[0]
        return {"farm": str(row["farm"]), "camera": str(row["camera"]), "clip": str(row["clip"])}
    analysis = config.get("analysis", {})
    return {
        "farm": str(analysis.get("farm", "")),
        "camera": str(analysis.get("camera", "")),
        "clip": str(analysis.get("clip", "")),
    }


def time_bins(traj_df: pd.DataFrame, interactions_df: pd.DataFrame, config: dict[str, Any]) -> list[tuple[float, float]]:
    if traj_df.empty:
        return []
    bin_s = report_time_bin_s(config)
    durations = get_frame_durations(traj_df, interactions_df, config)
    start = config.get("analysis", {}).get("start_time_s")
    end = config.get("analysis", {}).get("end_time_s")
    start_s = float(start) if start is not None else float(durations["time_s"].min())
    end_s = float(end) if end is not None else float((durations["time_s"] + durations["dt"]).max())
    if end_s <= start_s:
        end_s = start_s + bin_s
    bins: list[tuple[float, float]] = []
    cursor = start_s
    while cursor < end_s - 1.0e-9:
        bins.append((float(cursor), float(min(cursor + bin_s, end_s))))
        cursor += bin_s
    return bins


def build_cow_frame_zone(
    traj_df: pd.DataFrame,
    interactions_df: pd.DataFrame,
    zones: dict[str, Any],
    config: dict[str, Any],
) -> pd.DataFrame:
    columns = [
        "farm",
        "camera",
        "clip",
        "frame",
        "time_s",
        "dt_s",
        "cow_id",
        "anchor_x",
        "anchor_y",
        "mapped_x",
        "mapped_y",
        "zone_id",
        "zone_type",
        "zone_final",
        "track_conf",
        "visible_flag",
    ]
    if traj_df.empty:
        return pd.DataFrame(columns=columns)

    durations = get_frame_durations(traj_df, interactions_df, config)
    rows = traj_df.merge(
        durations[["farm", "camera", "clip", "frame", "time_s", "dt"]],
        on=["farm", "camera", "clip", "frame", "time_s"],
        how="left",
    )
    zone_list = normalized_zones(zones, config)
    outside_zone = str(config.get("zones", {}).get("outside_zone", "path"))
    zone_records = [lookup_zone_record(x, y, zone_list, outside_zone) for x, y in zip(rows["anchor_x"], rows["anchor_y"])]
    rows["zone_id"] = [item[0] for item in zone_records]
    rows["zone_type"] = [item[1] for item in zone_records]
    rows["visible_flag"] = (~rows[["anchor_x", "anchor_y"]].isna().any(axis=1)).astype(int)
    out = pd.DataFrame(
        {
            "farm": rows["farm"].astype(str),
            "camera": rows["camera"].astype(str),
            "clip": rows["clip"].astype(str),
            "frame": rows["frame"].astype(int),
            "time_s": rows["time_s"].astype(float),
            "dt_s": rows["dt"].astype(float),
            "cow_id": rows["cow_id"].astype(str),
            "anchor_x": rows["anchor_x"].astype(float),
            "anchor_y": rows["anchor_y"].astype(float),
            "mapped_x": rows["anchor_x"].astype(float),
            "mapped_y": rows["anchor_y"].astype(float),
            "zone_id": rows["zone_id"].astype(str),
            "zone_type": rows["zone_type"].astype(str),
            "zone_final": rows["zone"].astype(str),
            "track_conf": rows["track_conf"].astype(float),
            "visible_flag": rows["visible_flag"].astype(int),
        }
    )
    return out[columns].sort_values(["farm", "camera", "clip", "frame", "cow_id"]).reset_index(drop=True)


def lookup_zone_record(x: float, y: float, zones: list[dict[str, Any]], outside_zone: str) -> tuple[str, str]:
    if pd.isna(x) or pd.isna(y):
        return "unknown", "unknown"
    for zone in zones:
        if point_in_polygon(float(x), float(y), zone["polygon"]):
            return str(zone["zone_id"]), str(zone["zone_type"])
    return outside_zone, outside_zone


def build_zone_occupancy_timeseries(cow_frame_df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    columns = report_output_columns("zone_occupancy_timeseries")
    if cow_frame_df.empty:
        return pd.DataFrame(columns=columns)
    bin_s = report_time_bin_s(config)
    rows = cow_frame_df.loc[cow_frame_df["visible_flag"].astype(int) == 1].copy()
    if rows.empty:
        return pd.DataFrame(columns=columns)
    start_s = float(config.get("analysis", {}).get("start_time_s") or rows["time_s"].min())
    rows["bin_index"] = np.floor((rows["time_s"].astype(float) - start_s) / bin_s).astype(int)
    rows.loc[rows["bin_index"] < 0, "bin_index"] = 0
    rows["bin_start_s"] = start_s + rows["bin_index"].astype(float) * bin_s
    rows["bin_end_s"] = rows["bin_start_s"] + bin_s
    out_rows: list[dict[str, Any]] = []
    for key, group in rows.groupby(["farm", "camera", "clip", "bin_start_s", "bin_end_s", "zone_final"], sort=True):
        cow_ids = sorted(group["cow_id"].astype(str).unique())
        out_rows.append(
            {
                "farm": str(key[0]),
                "camera": str(key[1]),
                "clip": str(key[2]),
                "bin_start_s": float(key[3]),
                "bin_end_s": float(key[4]),
                "zone": str(key[5]),
                "n_cows": int(len(cow_ids)),
                "cow_ids": ";".join(cow_ids),
                "mean_track_conf": float(group["track_conf"].astype(float).mean()) if not group.empty else 0.0,
                "total_cow_seconds": float(group["dt_s"].astype(float).sum()),
            }
        )
    return pd.DataFrame(out_rows, columns=columns)


def build_temporal_edge_level(
    traj_df: pd.DataFrame,
    interactions_df: pd.DataFrame,
    config: dict[str, Any],
    aggregate_edges_fn: AggregateEdgesFn = aggregate_edges,
) -> pd.DataFrame:
    columns = report_output_columns("temporal_edge_level")
    frames: list[pd.DataFrame] = []
    for bin_start, bin_end in time_bins(traj_df, interactions_df, config):
        traj_w, inter_w = slice_window(traj_df, interactions_df, bin_start, bin_end)
        if traj_w.empty:
            continue
        edge_w, _ = aggregate_edges_fn(traj_w, inter_w, config)
        if edge_w.empty:
            continue
        edge_w = edge_w.drop(columns=["window_start_s", "window_end_s"], errors="ignore").copy()
        edge_w.insert(3, "bin_start_s", float(bin_start))
        edge_w.insert(4, "bin_end_s", float(bin_end))
        frames.append(edge_w[columns])
    if not frames:
        return pd.DataFrame(columns=columns)
    return pd.concat(frames, ignore_index=True)[columns]


def build_temporal_node_descriptors(
    traj_df: pd.DataFrame,
    interactions_df: pd.DataFrame,
    config: dict[str, Any],
    aggregate_edges_fn: AggregateEdgesFn = aggregate_edges,
) -> pd.DataFrame:
    columns = report_output_columns("temporal_node_descriptors")
    identity = analysis_identity(traj_df, config)
    frames: list[pd.DataFrame] = []
    min_visible = float(config.get("network", {}).get("min_visible_time_s", 60.0))
    for bin_start, bin_end in time_bins(traj_df, interactions_df, config):
        traj_w, inter_w = slice_window(traj_df, interactions_df, bin_start, bin_end)
        if traj_w.empty:
            continue
        edge_w, _ = aggregate_edges_fn(traj_w, inter_w, config)
        budget_w = compute_time_budget(traj_w, config)
        node_w = compute_node_descriptors(edge_w, budget_w, traj_w, config)
        node_w = finalize_isolation_score(node_w, config)
        if node_w.empty:
            continue
        node_w = node_w.copy()
        node_w.insert(0, "farm", identity["farm"])
        node_w.insert(1, "camera", identity["camera"])
        node_w.insert(2, "clip", identity["clip"])
        node_w.insert(3, "bin_start_s", float(bin_start))
        node_w.insert(4, "bin_end_s", float(bin_end))
        node_w["cow_bin_reliable"] = node_w["visible_time_s"].astype(float) >= min_visible
        frames.append(node_w[columns])
    if not frames:
        return pd.DataFrame(columns=columns)
    return pd.concat(frames, ignore_index=True)[columns]


def slice_window(
    traj_df: pd.DataFrame,
    interactions_df: pd.DataFrame,
    start_s: float,
    end_s: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    traj_w = traj_df.loc[(traj_df["time_s"].astype(float) >= start_s) & (traj_df["time_s"].astype(float) < end_s)].copy()
    inter_w = interactions_df.loc[
        (interactions_df["time_s"].astype(float) >= start_s) & (interactions_df["time_s"].astype(float) < end_s)
    ].copy()
    return traj_w.reset_index(drop=True), inter_w.reset_index(drop=True)


def build_pair_frame_rows(traj_df: pd.DataFrame, interactions_df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    columns = [
        "farm",
        "camera",
        "clip",
        "frame",
        "time_s",
        "dt",
        "cow_i",
        "cow_j",
        "zone",
        "p_friendly",
        "p_unfriendly",
        "conf",
        "distance",
        "opportunity_eligible",
    ]
    if traj_df.empty or interactions_df.empty:
        return pd.DataFrame(columns=columns)

    durations = get_frame_durations(traj_df, interactions_df, config)
    id_cols = ["farm", "camera", "clip", "frame"]
    interactions = interactions_df.merge(
        durations[id_cols + ["time_s", "dt"]],
        on=id_cols + ["time_s"],
        how="left",
    )
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
    joined = joined.loc[~missing].copy()
    if joined.empty:
        return pd.DataFrame(columns=columns)

    use_track_conf = bool(config["network"].get("use_track_confidence", True))
    use_interaction_conf = bool(config["network"].get("use_interaction_confidence", True))
    track_conf = joined["track_conf_i"].astype(float) * joined["track_conf_j"].astype(float) if use_track_conf else 1.0
    interaction_conf = joined["interaction_conf"].astype(float) if use_interaction_conf else 1.0
    joined["conf"] = track_conf * interaction_conf
    joined["distance"] = [
        euclidean_distance(x1, y1, x2, y2)
        for x1, y1, x2, y2 in zip(joined["anchor_x_i"], joined["anchor_y_i"], joined["anchor_x_j"], joined["anchor_y_j"])
    ]
    directed = bool(config["network"].get("directed", False))
    mode = str(config["network"].get("pair_zone_mode", "same_or_cross"))
    joined["zone"] = [
        pair_zone(zi, zj, directed, mode)
        for zi, zj in zip(joined["zone_i"], joined["zone_j"])
    ]
    joined["opportunity_eligible"] = joined["opportunity_eligible"].astype(float) > 0.0
    joined = joined.loc[joined["zone"].notna()].copy()
    return joined[columns].reset_index(drop=True)


def build_interaction_events(traj_df: pd.DataFrame, interactions_df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    columns = [
        "farm",
        "camera",
        "clip",
        "event_id",
        "layer",
        "cow_i",
        "cow_j",
        "dominant_zone",
        "start_frame",
        "end_frame",
        "start_time_s",
        "end_time_s",
        "duration_s",
        "mean_probability",
        "max_probability",
        "expected_seconds",
        "opportunity_seconds",
        "mean_distance",
        "min_distance",
        "n_frames",
        "peak_frame",
        "edge_reliable",
    ]
    if not bool(config.get("events", {}).get("enabled", True)):
        return pd.DataFrame(columns=columns)
    pair_rows = build_pair_frame_rows(traj_df, interactions_df, config)
    pair_rows = pair_rows.loc[pair_rows["opportunity_eligible"]].copy()
    if pair_rows.empty:
        return pd.DataFrame(columns=columns)

    event_cfg = config.get("events", {})
    start_threshold = float(event_cfg.get("event_start_threshold", 0.5))
    end_threshold = float(event_cfg.get("event_end_threshold", 0.3))
    max_gap_s = float(event_cfg.get("max_gap_s", 1.0))
    min_duration_s = float(event_cfg.get("min_event_duration_s", 2.0))
    min_opportunity = float(config.get("network", {}).get("min_opportunity_s", 5.0))
    eps = float(config.get("network", {}).get("eps", 1.0e-9))

    layer_frames: list[pd.DataFrame] = []
    for layer, prob_col in (("friendly", "p_friendly"), ("unfriendly", "p_unfriendly")):
        current = pair_rows.copy()
        current["layer"] = layer
        current["probability"] = current[prob_col].astype(float)
        layer_frames.append(current)
    event_input = pd.concat(layer_frames, ignore_index=True)

    rows: list[dict[str, Any]] = []
    event_index = 1
    group_cols = ["farm", "camera", "clip", "cow_i", "cow_j", "zone", "layer"]
    for key, group in event_input.groupby(group_cols, sort=True):
        group = group.sort_values(["time_s", "frame"]).reset_index(drop=True)
        for event_rows in detect_events_in_group(group, start_threshold, end_threshold, max_gap_s):
            event = pd.DataFrame(event_rows)
            if event.empty:
                continue
            start_time = float(event["time_s"].min())
            end_time = float((event["time_s"] + event["dt"]).max())
            duration = end_time - start_time
            if duration + 1.0e-9 < min_duration_s:
                continue
            event["expected"] = event["dt"].astype(float) * event["probability"].astype(float) * event["conf"].astype(float)
            event["opportunity"] = event["dt"].astype(float) * event["conf"].astype(float)
            expected = float(event["expected"].sum())
            opportunity = float(event["opportunity"].sum())
            weights = event["opportunity"].to_numpy(dtype=float)
            distances = event["distance"].to_numpy(dtype=float)
            mean_distance = float(np.average(distances, weights=weights)) if weights.sum() > 0 else float(np.mean(distances))
            peak_row = event.sort_values(["probability", "time_s"], ascending=[False, True]).iloc[0]
            rows.append(
                {
                    "farm": str(key[0]),
                    "camera": str(key[1]),
                    "clip": str(key[2]),
                    "event_id": f"{key[2]}_{key[6]}_{event_index:05d}",
                    "layer": str(key[6]),
                    "cow_i": str(key[3]),
                    "cow_j": str(key[4]),
                    "dominant_zone": str(key[5]),
                    "start_frame": int(event["frame"].min()),
                    "end_frame": int(event["frame"].max()),
                    "start_time_s": start_time,
                    "end_time_s": end_time,
                    "duration_s": duration,
                    "mean_probability": expected / (opportunity + eps),
                    "max_probability": float(event["probability"].max()),
                    "expected_seconds": expected,
                    "opportunity_seconds": opportunity,
                    "mean_distance": mean_distance,
                    "min_distance": float(event["distance"].min()),
                    "n_frames": int(len(event)),
                    "peak_frame": int(peak_row["frame"]),
                    "edge_reliable": bool(opportunity >= min_opportunity),
                }
            )
            event_index += 1
    return pd.DataFrame(rows, columns=columns)


def detect_events_in_group(
    group: pd.DataFrame,
    start_threshold: float,
    end_threshold: float,
    max_gap_s: float,
) -> list[list[dict[str, Any]]]:
    events: list[list[dict[str, Any]]] = []
    active_rows: list[dict[str, Any]] = []
    gap_rows: list[dict[str, Any]] = []
    last_seen_end: float | None = None

    for row in group.to_dict("records"):
        row_start = float(row["time_s"])
        row_end = row_start + float(row["dt"])
        prob = float(row["probability"])
        if active_rows and last_seen_end is not None and row_start - last_seen_end > max_gap_s:
            events.append(active_rows)
            active_rows = []
            gap_rows = []

        if not active_rows:
            if prob >= start_threshold:
                active_rows = [row]
                gap_rows = []
            last_seen_end = row_end
            continue

        if prob >= end_threshold:
            active_rows.extend(gap_rows)
            gap_rows = []
            active_rows.append(row)
        else:
            gap_rows.append(row)
            last_signal_end = float(active_rows[-1]["time_s"]) + float(active_rows[-1]["dt"])
            if row_end - last_signal_end > max_gap_s:
                events.append(active_rows)
                active_rows = []
                gap_rows = []
        last_seen_end = row_end

    if active_rows:
        events.append(active_rows)
    return events


def build_edge_ranking(edge_df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    columns = [
        "ranking_scope",
        "layer",
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
    if edge_df.empty:
        return pd.DataFrame(columns=columns)
    top_n = int(config.get("report", {}).get("edge_top_n", 20))
    min_opportunity = float(config.get("network", {}).get("min_opportunity_s", 5.0))
    eps = float(config.get("network", {}).get("eps", 1.0e-9))

    scopes: list[pd.DataFrame] = []
    for layer in ("friendly", "unfriendly"):
        sub = edge_df.loc[edge_df["layer"].astype(str) == layer].copy()
        if sub.empty:
            continue
        all_zone = aggregate_all_zone_edges(sub, layer, eps)
        all_zone["ranking_scope"] = f"{layer} all zones"
        scopes.append(all_zone)
        by_zone = sub.copy()
        by_zone["ranking_scope"] = layer + " by zone"
        scopes.append(by_zone)

    rows: list[pd.DataFrame] = []
    for _, scope in pd.concat(scopes, ignore_index=True).groupby(["ranking_scope", "layer", "zone"], sort=True):
        eligible = scope.loc[
            (scope["edge_reliable"].astype(bool))
            & (scope["opportunity_seconds"].astype(float) >= min_opportunity)
        ].copy()
        if eligible.empty:
            continue
        eligible["rank_by_expected_seconds"] = eligible["expected_seconds"].rank(method="first", ascending=False).astype(int)
        eligible["rank_by_normalized_rate"] = eligible["normalized_rate"].rank(method="first", ascending=False).astype(int)
        eligible = eligible.loc[
            (eligible["rank_by_expected_seconds"] <= top_n)
            | (eligible["rank_by_normalized_rate"] <= top_n)
        ].copy()
        rows.append(eligible[columns])
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.concat(rows, ignore_index=True).sort_values(["ranking_scope", "rank_by_expected_seconds", "cow_i", "cow_j"])


def aggregate_all_zone_edges(edge_df: pd.DataFrame, layer: str, eps: float) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for key, group in edge_df.groupby(["farm", "camera", "clip", "cow_i", "cow_j"], sort=True):
        opportunity = float(group["opportunity_seconds"].sum())
        expected = float(group["expected_seconds"].sum())
        weights = group["opportunity_seconds"].to_numpy(dtype=float)
        distances = group["mean_distance"].to_numpy(dtype=float)
        mean_distance = float(np.average(distances, weights=weights)) if weights.sum() > 0 else float(np.mean(distances))
        rows.append(
            {
                "farm": str(key[0]),
                "camera": str(key[1]),
                "clip": str(key[2]),
                "cow_i": str(key[3]),
                "cow_j": str(key[4]),
                "zone": "all_zones",
                "layer": layer,
                "expected_seconds": expected,
                "opportunity_seconds": opportunity,
                "normalized_rate": expected / (opportunity + eps),
                "mean_distance": mean_distance,
                "num_valid_frames": int(group["num_valid_frames"].sum()),
                "edge_reliable": bool(group["edge_reliable"].any()),
            }
        )
    return pd.DataFrame(rows)


def build_report_network_layout(
    cow_frame_df: pd.DataFrame,
    edge_df: pd.DataFrame,
    node_df: pd.DataFrame,
    time_budget_df: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    columns = [
        "cow_id",
        "layout_x",
        "layout_y",
        "median_floorplan_x",
        "median_floorplan_y",
        "visible_time_s",
        "community_id_full_friendly",
    ]
    cows = sorted(time_budget_df["cow_id"].astype(str).unique()) if not time_budget_df.empty else []
    if not cows:
        return pd.DataFrame(columns=columns)

    medians = median_floorplan_positions(cow_frame_df)
    visible = dict(zip(time_budget_df["cow_id"].astype(str), time_budget_df["visible_time_s"].astype(float)))
    communities = detect_communities_from_edges(edge_df, cows, layer="friendly")
    if "community_id_full_friendly" in node_df.columns:
        communities.update(
            {
                str(row["cow_id"]): int(row["community_id_full_friendly"])
                for _, row in node_df[["cow_id", "community_id_full_friendly"]].dropna().iterrows()
            }
        )

    layout_mode = str(config.get("report", {}).get("network_layout", "spring"))
    if layout_mode == "floorplan":
        positions = {cow: medians.get(cow, fallback_circle_position(index, len(cows))) for index, cow in enumerate(cows)}
    else:
        positions = spring_positions(cows, edge_df, int(config.get("report", {}).get("layout_random_seed", 123)))

    rows = []
    for index, cow in enumerate(cows):
        floor_x, floor_y = medians.get(cow, fallback_circle_position(index, len(cows)))
        layout_x, layout_y = positions.get(cow, fallback_circle_position(index, len(cows)))
        rows.append(
            {
                "cow_id": cow,
                "layout_x": float(layout_x),
                "layout_y": float(layout_y),
                "median_floorplan_x": float(floor_x),
                "median_floorplan_y": float(floor_y),
                "visible_time_s": float(visible.get(cow, 0.0)),
                "community_id_full_friendly": int(communities.get(cow, -1)),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def median_floorplan_positions(cow_frame_df: pd.DataFrame) -> dict[str, tuple[float, float]]:
    if cow_frame_df.empty:
        return {}
    visible = cow_frame_df.loc[cow_frame_df["visible_flag"].astype(int) == 1].copy()
    if visible.empty:
        return {}
    med = visible.groupby("cow_id")[["anchor_x", "anchor_y"]].median()
    return {str(cow): (float(row["anchor_x"]), float(row["anchor_y"])) for cow, row in med.iterrows()}


def spring_positions(cows: list[str], edge_df: pd.DataFrame, seed: int) -> dict[str, tuple[float, float]]:
    try:
        import networkx as nx
    except Exception:
        return {cow: fallback_circle_position(index, len(cows)) for index, cow in enumerate(cows)}

    graph = nx.Graph()
    graph.add_nodes_from(cows)
    if not edge_df.empty:
        grouped = edge_df.groupby(["cow_i", "cow_j"], as_index=False)["expected_seconds"].sum()
        for _, row in grouped.iterrows():
            weight = float(row["expected_seconds"])
            if weight > 0:
                graph.add_edge(str(row["cow_i"]), str(row["cow_j"]), weight=weight)
    if graph.number_of_edges() == 0:
        return {cow: fallback_circle_position(index, len(cows)) for index, cow in enumerate(cows)}
    pos = nx.spring_layout(graph, weight="weight", seed=seed)
    return {str(cow): (float(pos[cow][0]), float(pos[cow][1])) for cow in cows}


def fallback_circle_position(index: int, total: int) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0
    angle = 2.0 * math.pi * float(index) / float(total)
    return math.cos(angle), math.sin(angle)
