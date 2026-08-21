from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd

from .aggregation import get_frame_durations


def _time_indexed_frame(df: pd.DataFrame, label: str) -> tuple[pd.DataFrame, np.ndarray]:
    """Return a time-monotonic frame and its numeric ``time_s`` index."""
    times = pd.to_numeric(df["time_s"], errors="raise").to_numpy(dtype=float, copy=False)
    if not bool(np.isfinite(times).all()):
        raise ValueError(f"{label}.time_s must contain only finite values")
    if len(times) < 2 or bool(np.all(times[:-1] <= times[1:])):
        return df, times

    order = np.argsort(times, kind="stable")
    return df.iloc[order], times[order]


def _networkx_graph(edge_df: pd.DataFrame, cows: list[str], layer: str):
    import networkx as nx

    graph = nx.Graph()
    graph.add_nodes_from(cows)
    if not edge_df.empty:
        sub = edge_df.loc[edge_df["layer"] == layer]
        grouped = sub.groupby(["cow_i", "cow_j"], as_index=False)["expected_seconds"].sum()
        for _, row in grouped.iterrows():
            weight = float(row["expected_seconds"])
            if weight > 0:
                graph.add_edge(str(row["cow_i"]), str(row["cow_j"]), weight=weight)
    return graph


def detect_communities_from_edges(edge_df: pd.DataFrame, cows: list[str], layer: str = "friendly") -> dict[str, int]:
    cows = sorted(str(cow) for cow in cows)
    try:
        import networkx as nx
    except Exception:
        return {cow: index for index, cow in enumerate(cows)}

    graph = _networkx_graph(edge_df, cows, layer)
    if graph.number_of_edges() == 0:
        return {cow: index for index, cow in enumerate(cows)}
    try:
        communities = nx.algorithms.community.louvain_communities(graph, weight="weight", seed=0)
    except Exception:
        communities = nx.algorithms.community.greedy_modularity_communities(graph, weight="weight")
    out: dict[str, int] = {}
    for community_id, community in enumerate(communities):
        for cow in community:
            out[str(cow)] = int(community_id)
    for cow in cows:
        out.setdefault(cow, len(out))
    return out


def compute_community_stability(
    traj_df: pd.DataFrame,
    interactions_df: pd.DataFrame,
    config: dict[str, Any],
    aggregate_edges_fn,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    window_columns = [
        "window_index",
        "window_start_s",
        "window_end_s",
        "cow_id",
        "community_id",
        "visible_time_s_in_window",
    ]
    summary_columns = ["window_a", "window_b", "n_common_cows", "network_nmi", "network_ari"]
    cows = sorted(traj_df["cow_id"].astype(str).unique())
    enabled = bool(config.get("community", {}).get("enabled", True))
    if not enabled or traj_df.empty:
        empty_windows = pd.DataFrame(columns=window_columns)
        empty_summary = pd.DataFrame(columns=summary_columns)
        empty_node = pd.DataFrame({"cow_id": cows, "community_stability": np.nan})
        return empty_windows, empty_summary, empty_node

    window_s = float(config["community"].get("window_s", 300.0))
    step_s = float(config["community"].get("step_s", window_s))
    min_visible_time_s = float(config["community"].get("min_visible_time_s", 0.0))
    raw_min_edges = config["community"].get("min_edges_per_window", 1)
    min_edges_per_window = int(raw_min_edges)
    if not np.isfinite(window_s) or not np.isfinite(step_s) or window_s <= 0 or step_s <= 0:
        raise ValueError("community.window_s and community.step_s must be finite and positive")
    if not np.isfinite(min_visible_time_s) or min_visible_time_s < 0:
        raise ValueError("community.min_visible_time_s must be finite and non-negative")
    if isinstance(raw_min_edges, bool) or float(raw_min_edges) != min_edges_per_window or min_edges_per_window < 0:
        raise ValueError("community.min_edges_per_window must be a non-negative integer")

    frame_keys = ["farm", "camera", "clip", "frame", "time_s"]
    traj_by_time, traj_times = _time_indexed_frame(traj_df, "trajectory")
    interactions_by_time, interaction_times = _time_indexed_frame(interactions_df, "interaction")
    configured_fps = config.get("time", {}).get("fps")
    fixed_dt: float | None = None
    frame_durations: pd.DataFrame | None = None
    if configured_fps is not None:
        fps = float(configured_fps)
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError("time.fps must be finite and positive")
        fixed_dt = 1.0 / fps
    else:
        frame_durations = get_frame_durations(traj_df, interactions_df, config)[frame_keys + ["dt"]]
    start = float(traj_times[0])
    end = float(traj_times[-1])
    windows: list[tuple[int, float, float]] = []
    cursor = start
    index = 0
    while cursor <= end + 1.0e-9:
        windows.append((index, cursor, cursor + window_s))
        cursor += step_s
        index += 1

    window_rows: list[dict[str, Any]] = []
    assignments: dict[int, dict[str, int]] = {}
    present_by_window: dict[int, set[str]] = {}
    window_starts = np.fromiter((window[1] for window in windows), dtype=float, count=len(windows))
    window_ends = np.fromiter((window[2] for window in windows), dtype=float, count=len(windows))
    traj_left = np.searchsorted(traj_times, window_starts, side="left")
    traj_right = np.searchsorted(traj_times, window_ends, side="left")
    interaction_left = np.searchsorted(interaction_times, window_starts, side="left")
    interaction_right = np.searchsorted(interaction_times, window_ends, side="left")
    for position, (window_index, window_start, window_end) in enumerate(windows):
        traj_w = traj_by_time.iloc[int(traj_left[position]) : int(traj_right[position])].copy()
        inter_w = interactions_by_time.iloc[
            int(interaction_left[position]) : int(interaction_right[position])
        ].copy()
        present_by_window[window_index] = set()
        if traj_w.empty:
            continue

        if fixed_dt is not None:
            visible = traj_w.assign(_cow_id=traj_w["cow_id"].astype(str)).groupby(
                "_cow_id", sort=True
            ).size().astype(float) * fixed_dt
        else:
            if frame_durations is None:  # pragma: no cover - guarded by the branches above.
                raise RuntimeError("frame duration source was not initialized")
            traj_with_dt = traj_w.merge(frame_durations, on=frame_keys, how="left", validate="many_to_one")
            if traj_with_dt["dt"].isna().any():
                raise ValueError("missing frame duration while computing community visibility")
            visible = (
                traj_with_dt.assign(_cow_id=traj_with_dt["cow_id"].astype(str))
                .groupby("_cow_id", sort=True)["dt"]
                .sum()
            )
        present = set(visible.loc[visible >= min_visible_time_s].index.astype(str))
        if not present:
            continue

        traj_w = traj_w.loc[traj_w["cow_id"].astype(str).isin(present)].copy()
        inter_w = inter_w.loc[
            inter_w["cow_i"].astype(str).isin(present)
            & inter_w["cow_j"].astype(str).isin(present)
        ].copy()
        edge_w, _ = aggregate_edges_fn(traj_w, inter_w, config)
        positive_friendly = edge_w.loc[
            (edge_w["layer"].astype(str) == "friendly")
            & (pd.to_numeric(edge_w["expected_seconds"], errors="coerce") > 0.0),
            ["cow_i", "cow_j"],
        ]
        positive_dyads = {
            tuple(sorted((str(row.cow_i), str(row.cow_j))))
            for row in positive_friendly.itertuples(index=False)
            if str(row.cow_i) != str(row.cow_j)
        }
        if len(positive_dyads) < min_edges_per_window:
            continue

        assignment = detect_communities_from_edges(edge_w, sorted(present), layer="friendly")
        assignments[window_index] = assignment
        present_by_window[window_index] = set(assignment)
        for cow in sorted(present):
            window_rows.append(
                {
                    "window_index": window_index,
                    "window_start_s": window_start,
                    "window_end_s": window_end,
                    "cow_id": str(cow),
                    "community_id": int(assignment.get(str(cow), -1)),
                    "visible_time_s_in_window": float(visible.get(str(cow), 0.0)),
                }
            )

    summary_rows: list[dict[str, Any]] = []
    per_cow_scores: dict[str, list[float]] = {cow: [] for cow in cows}
    for (idx_a, _, _), (idx_b, _, _) in zip(windows, windows[1:]):
        assign_a = assignments.get(idx_a, {})
        assign_b = assignments.get(idx_b, {})
        common = sorted(present_by_window.get(idx_a, set()) & present_by_window.get(idx_b, set()))
        nmi, ari = clustering_similarity(assign_a, assign_b, common)
        summary_rows.append({"window_a": idx_a, "window_b": idx_b, "n_common_cows": len(common), "network_nmi": nmi, "network_ari": ari})
        if len(common) < 2:
            continue
        for cow in common:
            others = [item for item in common if item != cow]
            diffs = [
                abs(int(assign_a.get(cow) == assign_a.get(other)) - int(assign_b.get(cow) == assign_b.get(other)))
                for other in others
            ]
            if diffs:
                per_cow_scores[cow].append(1.0 - float(np.mean(diffs)))

    node_rows = [
        {"cow_id": cow, "community_stability": float(np.mean(scores)) if scores else np.nan}
        for cow, scores in per_cow_scores.items()
    ]
    return (
        pd.DataFrame(window_rows, columns=window_columns),
        pd.DataFrame(summary_rows, columns=summary_columns),
        pd.DataFrame(node_rows),
    )


def clustering_similarity(assign_a: dict[str, int], assign_b: dict[str, int], cows: list[str]) -> tuple[float, float]:
    if len(cows) < 2:
        return np.nan, np.nan
    labels_a = [assign_a.get(cow, -1) for cow in cows]
    labels_b = [assign_b.get(cow, -1) for cow in cows]
    try:
        from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    except Exception:
        return np.nan, np.nan
    return float(normalized_mutual_info_score(labels_a, labels_b)), float(adjusted_rand_score(labels_a, labels_b))


def comembership_pairs(assign: dict[str, int], cows: list[str]) -> dict[tuple[str, str], int]:
    return {
        tuple(sorted((a, b))): int(assign.get(a) == assign.get(b))
        for a, b in combinations(cows, 2)
    }
