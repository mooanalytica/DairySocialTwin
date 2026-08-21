from __future__ import annotations

from collections import Counter
from typing import Any

import pandas as pd

from .geometry import point_in_polygon


def normalized_zones(zones: dict[str, Any], config: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, zone in enumerate(zones.get("zones", []), start=1):
        if not isinstance(zone, dict):
            raise ValueError("zones.json zones entries must be objects")
        zone_id = str(zone.get("zone_id") or zone.get("zoneId") or f"zone_{index}")
        if zone_id in seen:
            continue
        seen.add(zone_id)
        polygon = zone.get("polygon") or zone.get("points")
        if not isinstance(polygon, list) or len(polygon) < 3:
            raise ValueError(f"zone has fewer than 3 polygon points: {zone_id}")
        out.append(
            {
                "zone_id": zone_id,
                "zone_type": str(zone.get("zone_type") or zone.get("zoneType") or zone_id),
                "label": str(zone.get("label") or zone.get("zone_type") or zone_id),
                "polygon": polygon,
            }
        )
    return out


def lookup_zone(x: float, y: float, zones: list[dict[str, Any]], outside_zone: str) -> str:
    if pd.isna(x) or pd.isna(y):
        return "unknown"
    for zone in zones:
        if point_in_polygon(float(x), float(y), zone["polygon"]):
            return str(zone["zone_type"])
    return outside_zone


def assign_zones_to_trajectories(traj_df: pd.DataFrame, zones: dict[str, Any], config: dict[str, Any]) -> pd.DataFrame:
    out = traj_df.copy()
    outside_zone = str(config.get("zones", {}).get("outside_zone", "path"))
    zone_list = normalized_zones(zones, config)
    out["zone"] = [
        lookup_zone(x, y, zone_list, outside_zone)
        for x, y in zip(out["anchor_x"], out["anchor_y"])
    ]
    smoothing_s = float(config.get("zones", {}).get("smoothing_s") or 0.0)
    if smoothing_s > 0:
        out = smooth_zones(out, smoothing_s)
    return out


def smooth_zones(traj_df: pd.DataFrame, smoothing_s: float) -> pd.DataFrame:
    if smoothing_s <= 0:
        return traj_df
    out = traj_df.copy().sort_values(["farm", "camera", "clip", "cow_id", "time_s", "frame"])
    half = smoothing_s / 2.0
    smoothed: list[str] = []
    for _, group in out.groupby(["farm", "camera", "clip", "cow_id"], sort=False):
        times = group["time_s"].to_numpy(dtype=float)
        zones = group["zone"].astype(str).tolist()
        for time_s in times:
            labels = [zone for zone, t in zip(zones, times) if abs(float(t) - float(time_s)) <= half]
            known = [label for label in labels if label != "unknown"]
            vote_pool = known if known else labels
            smoothed.append(Counter(vote_pool).most_common(1)[0][0] if vote_pool else "unknown")
    out["zone"] = smoothed
    return out.sort_index()
