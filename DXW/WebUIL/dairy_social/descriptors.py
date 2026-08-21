from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .aggregation import get_frame_durations, safe_name
from .communities import detect_communities_from_edges


def compute_entropy(weights: np.ndarray) -> tuple[float, float]:
    weights = np.asarray(weights, dtype=float)
    weights = weights[weights > 0]
    if weights.size == 0:
        return 0.0, 0.0
    probs = weights / weights.sum()
    raw = float(-(probs * np.log(probs)).sum())
    norm = 0.0 if weights.size <= 1 else raw / float(np.log(weights.size))
    return raw, norm


def compute_node_descriptors(
    edge_df: pd.DataFrame,
    time_budget_df: pd.DataFrame,
    traj_df: pd.DataFrame,
    config: dict[str, Any],
    community_stability_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    cows = sorted(time_budget_df["cow_id"].astype(str).unique())
    community = detect_communities_from_edges(edge_df, cows, layer="friendly")
    isolation = compute_isolation(traj_df, pd.DataFrame({"cow_id": cows}), config)
    stability = community_stability_df if community_stability_df is not None else pd.DataFrame({"cow_id": cows, "community_stability": np.nan})
    rows: list[dict[str, Any]] = []
    for cow in cows:
        budget = time_budget_df.loc[time_budget_df["cow_id"].astype(str) == cow].iloc[0].to_dict()
        friendly_strength = strength_for_cow(edge_df, cow, "friendly")
        unfriendly_strength = strength_for_cow(edge_df, cow, "unfriendly")
        weighted_visible = float(budget.get("weighted_visible_time_s", 0.0))
        eps = float(config["network"].get("eps", 1.0e-9))
        f_raw, f_div = partner_entropy(edge_df, cow, "friendly")
        a_raw, a_div = partner_entropy(edge_df, cow, "unfriendly")
        payload = {
            "cow_id": cow,
            "visible_time_s": float(budget.get("visible_time_s", 0.0)),
            "weighted_visible_time_s": weighted_visible,
            "cow_reliable": bool(budget.get("cow_reliable", False)),
            "friendly_strength": friendly_strength,
            "friendly_strength_rate": friendly_strength / (weighted_visible + eps),
            "unfriendly_strength": unfriendly_strength,
            "unfriendly_strength_rate": unfriendly_strength / (weighted_visible + eps),
            "unfriendly_ratio": unfriendly_strength / (friendly_strength + unfriendly_strength + eps),
            "friendly_partner_diversity": f_div,
            "unfriendly_partner_diversity": a_div,
            "friendly_effective_partners": float(np.exp(f_raw)) if f_raw > 0 else 0.0,
            "unfriendly_effective_partners": float(np.exp(a_raw)) if a_raw > 0 else 0.0,
            "friendly_zone_diversity": zone_entropy(edge_df, cow, "friendly"),
            "unfriendly_zone_diversity": zone_entropy(edge_df, cow, "unfriendly"),
            "friendly_degree_weighted": friendly_strength,
            "unfriendly_degree_weighted": unfriendly_strength,
            "friendly_active_partners": active_partners(edge_df, cow, "friendly"),
            "unfriendly_active_partners": active_partners(edge_df, cow, "unfriendly"),
            "community_id_full_friendly": int(community.get(cow, -1)),
        }
        if bool(config["network"].get("directed", False)):
            payload.update(directed_strengths(edge_df, cow, weighted_visible, eps))
        else:
            payload["unfriendly_involvement_rate"] = payload["unfriendly_strength_rate"]
        iso_row = isolation.loc[isolation["cow_id"].astype(str) == cow]
        if not iso_row.empty:
            payload.update(iso_row.iloc[0].drop(labels=["cow_id"]).to_dict())
        stab_row = stability.loc[stability["cow_id"].astype(str) == cow]
        payload["community_stability"] = float(stab_row.iloc[0]["community_stability"]) if not stab_row.empty else np.nan
        rows.append(payload)
    return pd.DataFrame(rows)


def strength_for_cow(edge_df: pd.DataFrame, cow: str, layer: str) -> float:
    if edge_df.empty:
        return 0.0
    sub = edge_df.loc[(edge_df["layer"] == layer) & ((edge_df["cow_i"].astype(str) == cow) | (edge_df["cow_j"].astype(str) == cow))]
    return float(sub["expected_seconds"].sum())


def partner_entropy(edge_df: pd.DataFrame, cow: str, layer: str) -> tuple[float, float]:
    if edge_df.empty:
        return 0.0, 0.0
    sub = edge_df.loc[(edge_df["layer"] == layer) & ((edge_df["cow_i"].astype(str) == cow) | (edge_df["cow_j"].astype(str) == cow))]
    weights: dict[str, float] = {}
    for _, row in sub.iterrows():
        partner = str(row["cow_j"]) if str(row["cow_i"]) == cow else str(row["cow_i"])
        weights[partner] = weights.get(partner, 0.0) + float(row["expected_seconds"])
    return compute_entropy(np.array(list(weights.values()), dtype=float))


def zone_entropy(edge_df: pd.DataFrame, cow: str, layer: str) -> float:
    if edge_df.empty:
        return 0.0
    sub = edge_df.loc[(edge_df["layer"] == layer) & ((edge_df["cow_i"].astype(str) == cow) | (edge_df["cow_j"].astype(str) == cow))]
    weights = sub.groupby("zone")["expected_seconds"].sum().to_numpy(dtype=float)
    return compute_entropy(weights)[1]


def active_partners(edge_df: pd.DataFrame, cow: str, layer: str) -> int:
    if edge_df.empty:
        return 0
    sub = edge_df.loc[(edge_df["layer"] == layer) & ((edge_df["cow_i"].astype(str) == cow) | (edge_df["cow_j"].astype(str) == cow))]
    partners = {
        str(row["cow_j"]) if str(row["cow_i"]) == cow else str(row["cow_i"])
        for _, row in sub.iterrows()
        if float(row["expected_seconds"]) > 0
    }
    return len(partners)


def directed_strengths(edge_df: pd.DataFrame, cow: str, visible: float, eps: float) -> dict[str, float]:
    out: dict[str, float] = {}
    for layer in ("friendly", "unfriendly"):
        sub = edge_df.loc[edge_df["layer"] == layer] if not edge_df.empty else edge_df
        out_strength = float(sub.loc[sub["cow_i"].astype(str) == cow, "expected_seconds"].sum()) if not sub.empty else 0.0
        in_strength = float(sub.loc[sub["cow_j"].astype(str) == cow, "expected_seconds"].sum()) if not sub.empty else 0.0
        out[f"{layer}_out_strength"] = out_strength
        out[f"{layer}_in_strength"] = in_strength
    out["unfriendly_exposure_in"] = out["unfriendly_in_strength"] / (visible + eps)
    out["unfriendly_output"] = out["unfriendly_out_strength"] / (visible + eps)
    return out


def compute_isolation(traj_df: pd.DataFrame, node_df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    durations = get_frame_durations(traj_df, pd.DataFrame(columns=traj_df.columns), config)
    rows = traj_df.merge(durations[["farm", "camera", "clip", "frame", "time_s", "dt"]], on=["farm", "camera", "clip", "frame", "time_s"], how="left")
    rows = rows.loc[rows["zone"].astype(str) != "unknown"].copy()
    alone_time: dict[str, float] = {}
    observable: dict[str, float] = {}
    zone_alone: dict[tuple[str, str], float] = {}
    zone_observable: dict[tuple[str, str], float] = {}
    for _, group in rows.groupby(["farm", "camera", "clip", "frame"], sort=False):
        counts = group["zone"].astype(str).value_counts().to_dict()
        for _, row in group.iterrows():
            cow = str(row["cow_id"])
            zone = str(row["zone"])
            dt = float(row["dt"])
            observable[cow] = observable.get(cow, 0.0) + dt
            zone_observable[(cow, zone)] = zone_observable.get((cow, zone), 0.0) + dt
            if counts.get(zone, 0) <= 1:
                alone_time[cow] = alone_time.get(cow, 0.0) + dt
                zone_alone[(cow, zone)] = zone_alone.get((cow, zone), 0.0) + dt
    cows = sorted(node_df["cow_id"].astype(str).unique())
    return pd.DataFrame(
        {
            "cow_id": cow,
            "alone_fraction": alone_time.get(cow, 0.0) / (observable.get(cow, 0.0) + float(config["network"].get("eps", 1.0e-9))),
        }
        for cow in cows
    )


def finalize_isolation_score(node_df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    out = node_df.copy()
    max_rate = float(out["friendly_strength_rate"].max()) if not out.empty else 0.0
    eps = float(config["network"].get("eps", 1.0e-9))
    out["low_friendly_sociality"] = 1.0 - (out["friendly_strength_rate"] / max_rate if max_rate > eps else 0.0)
    out["low_partner_diversity"] = 1.0 - out["friendly_partner_diversity"]
    weights = config["isolation"]["weights"]
    out["isolation_score"] = (
        float(weights["alone_fraction"]) * out["alone_fraction"]
        + float(weights["low_friendly_sociality"]) * out["low_friendly_sociality"]
        + float(weights["low_partner_diversity"]) * out["low_partner_diversity"]
    )
    return out


def compute_zone_node_descriptors(edge_df: pd.DataFrame, time_budget_df: pd.DataFrame, traj_df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    zones = sorted(traj_df["zone"].astype(str).unique())
    cows = sorted(time_budget_df["cow_id"].astype(str).unique())
    rows: list[dict[str, Any]] = []
    eps = float(config["network"].get("eps", 1.0e-9))
    zone_alone, zone_observable = zone_isolation_times(traj_df, config)
    for cow in cows:
        budget = time_budget_df.loc[time_budget_df["cow_id"].astype(str) == cow].iloc[0]
        for zone in zones:
            safe = safe_name(zone)
            time_in_zone = float(budget.get(f"time_{safe}_s", 0.0))
            weighted_time = float(budget.get(f"weighted_time_{safe}_s", time_in_zone))
            row = {
                "cow_id": cow,
                "zone": zone,
                "time_in_zone_s": time_in_zone,
                "weighted_time_in_zone_s": weighted_time,
            }
            for layer in ("friendly", "unfriendly"):
                sub = edge_df.loc[
                    (edge_df["layer"] == layer)
                    & (edge_df["zone"].astype(str) == zone)
                    & ((edge_df["cow_i"].astype(str) == cow) | (edge_df["cow_j"].astype(str) == cow))
                ] if not edge_df.empty else edge_df
                strength = float(sub["expected_seconds"].sum()) if not sub.empty else 0.0
                partners = {
                    str(item["cow_j"]) if str(item["cow_i"]) == cow else str(item["cow_i"])
                    for _, item in sub.iterrows()
                    if float(item["expected_seconds"]) > 0
                } if not sub.empty else set()
                row[f"{layer}_strength_zone"] = strength
                row[f"{layer}_strength_rate_zone"] = strength / (weighted_time + eps)
                row[f"{layer}_active_partners_zone"] = len(partners)
            row["alone_fraction_zone"] = zone_alone.get((cow, zone), 0.0) / (zone_observable.get((cow, zone), 0.0) + eps)
            rows.append(row)
    return pd.DataFrame(rows)


def zone_isolation_times(traj_df: pd.DataFrame, config: dict[str, Any]) -> tuple[dict[tuple[str, str], float], dict[tuple[str, str], float]]:
    durations = get_frame_durations(traj_df, pd.DataFrame(columns=traj_df.columns), config)
    rows = traj_df.merge(durations[["farm", "camera", "clip", "frame", "time_s", "dt"]], on=["farm", "camera", "clip", "frame", "time_s"], how="left")
    rows = rows.loc[rows["zone"].astype(str) != "unknown"].copy()
    alone: dict[tuple[str, str], float] = {}
    observable: dict[tuple[str, str], float] = {}
    for _, group in rows.groupby(["farm", "camera", "clip", "frame"], sort=False):
        counts = group["zone"].astype(str).value_counts().to_dict()
        for _, row in group.iterrows():
            key = (str(row["cow_id"]), str(row["zone"]))
            dt = float(row["dt"])
            observable[key] = observable.get(key, 0.0) + dt
            if counts.get(key[1], 0) <= 1:
                alone[key] = alone.get(key, 0.0) + dt
    return alone, observable
