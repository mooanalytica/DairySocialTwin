from __future__ import annotations

from typing import Any

import pandas as pd


def _as_string(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna("").astype(str)


def validate_and_clean_trajectories(df: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    warnings: list[str] = []
    df = df.copy()
    required = {"farm", "camera", "clip", "frame", "time_s", "cow_id"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"trajectories.csv is missing columns: {missing}")

    if ("anchor_x" not in df.columns or "anchor_y" not in df.columns) and {
        "bbox_x1",
        "bbox_y1",
        "bbox_x2",
        "bbox_y2",
    }.issubset(df.columns):
        df["anchor_x"] = (pd.to_numeric(df["bbox_x1"], errors="coerce") + pd.to_numeric(df["bbox_x2"], errors="coerce")) / 2.0
        df["anchor_y"] = pd.to_numeric(df["bbox_y2"], errors="coerce")

    if "anchor_x" not in df.columns or "anchor_y" not in df.columns:
        raise ValueError("trajectories.csv needs anchor_x/anchor_y or bbox_x1/bbox_y1/bbox_x2/bbox_y2")

    for column in ("farm", "camera", "clip", "cow_id"):
        df[column] = _as_string(df[column])
    df["frame"] = pd.to_numeric(df["frame"], errors="raise").astype(int)
    df["time_s"] = pd.to_numeric(df["time_s"], errors="raise").astype(float)
    df["anchor_x"] = pd.to_numeric(df["anchor_x"], errors="coerce")
    df["anchor_y"] = pd.to_numeric(df["anchor_y"], errors="coerce")
    if "track_conf" not in df.columns:
        df["track_conf"] = 1.0
    df["track_conf"] = pd.to_numeric(df["track_conf"], errors="coerce").fillna(1.0).clip(0.0, 1.0)
    return df, {"warnings": warnings}


def validate_and_clean_interactions(df: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    warnings: list[str] = []
    df = df.copy()
    required = {
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
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"interactions.csv is missing columns: {missing}")

    for column in ("farm", "camera", "clip", "cow_i", "cow_j"):
        df[column] = _as_string(df[column])
    df["frame"] = pd.to_numeric(df["frame"], errors="raise").astype(int)
    df["time_s"] = pd.to_numeric(df["time_s"], errors="raise").astype(float)
    for column in ("p_friendly", "p_unfriendly"):
        before = pd.to_numeric(df[column], errors="coerce")
        clipped = before.clip(0.0, 1.0)
        if not before.fillna(0.0).equals(clipped.fillna(0.0)):
            warnings.append(f"{column} clipped to [0, 1]")
        df[column] = clipped.fillna(0.0)
    before = pd.to_numeric(df["opportunity_eligible"], errors="coerce")
    clipped = before.clip(0.0, 1.0)
    if not before.fillna(0.0).equals(clipped.fillna(0.0)):
        warnings.append("opportunity_eligible clipped to [0, 1]")
    df["opportunity_eligible"] = clipped.fillna(0.0)
    if "interaction_conf" not in df.columns:
        df["interaction_conf"] = 1.0
    df["interaction_conf"] = pd.to_numeric(df["interaction_conf"], errors="coerce").fillna(1.0).clip(0.0, 1.0)

    self_mask = df["cow_i"] == df["cow_j"]
    dropped_self = int(self_mask.sum())
    if dropped_self:
        warnings.append(f"dropped {dropped_self} self-pair interaction rows")
        df = df.loc[~self_mask].copy()

    if not bool(config["network"].get("directed", False)):
        cow_a = df[["cow_i", "cow_j"]].min(axis=1)
        cow_b = df[["cow_i", "cow_j"]].max(axis=1)
        df["cow_i"] = cow_a
        df["cow_j"] = cow_b

    group_cols = ["farm", "camera", "clip", "frame", "time_s", "cow_i", "cow_j"]
    df = (
        df.groupby(group_cols, as_index=False)
        .agg(
            {
                "p_friendly": "mean",
                "p_unfriendly": "mean",
                "opportunity_eligible": "max",
                "interaction_conf": "mean",
            }
        )
        .sort_values(group_cols)
        .reset_index(drop=True)
    )
    return df, {"warnings": warnings, "n_dropped_self_pairs": dropped_self}
