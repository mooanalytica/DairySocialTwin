from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .aggregation import aggregate_edges, compute_time_budget, filter_by_config, write_adjacency_matrices
from .communities import compute_community_stability
from .config import load_config
from .descriptors import compute_node_descriptors, compute_zone_node_descriptors, finalize_isolation_score
from .io import load_interactions, load_trajectories, load_zones
from .plots import write_plots
from .report_outputs import write_report_outputs
from .validation import validate_and_clean_interactions, validate_and_clean_trajectories
from .zones import assign_zones_to_trajectories


def run_pipeline(
    trajectories_path: str | Path,
    interactions_path: str | Path,
    zones_path: str | Path,
    config_path: str | Path,
    outdir: str | Path,
) -> dict[str, Any]:
    config = load_config(config_path)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    raw_traj = load_trajectories(trajectories_path)
    raw_interactions = load_interactions(interactions_path)
    zones = load_zones(zones_path)
    traj, traj_report = validate_and_clean_trajectories(raw_traj, config)
    interactions, interaction_report = validate_and_clean_interactions(raw_interactions, config)
    traj = filter_by_config(traj, config)
    interactions = filter_by_config(interactions, config)
    traj = assign_zones_to_trajectories(traj, zones, config)

    edge_df, edge_report = aggregate_edges(traj, interactions, config)
    time_budget_df = compute_time_budget(traj, config)
    cows = sorted(time_budget_df["cow_id"].astype(str).unique())
    community_windows, community_summary, community_node = compute_community_stability(
        traj, interactions, config, aggregate_edges
    )
    node_df = compute_node_descriptors(edge_df, time_budget_df, traj, config, community_node)
    node_df = finalize_isolation_score(node_df, config)
    zone_node_df = compute_zone_node_descriptors(edge_df, time_budget_df, traj, config)

    edge_df.to_csv(outdir / "edge_level.csv", index=False)
    node_df.to_csv(outdir / "node_descriptors.csv", index=False)
    zone_node_df.to_csv(outdir / "zone_node_descriptors.csv", index=False)
    time_budget_df.to_csv(outdir / "cow_time_budget.csv", index=False)
    community_windows.to_csv(outdir / "community_windows.csv", index=False)
    community_summary.to_csv(outdir / "community_stability_summary.csv", index=False)
    report_paths, layout_df = write_report_outputs(
        traj,
        interactions,
        zones,
        edge_df,
        node_df,
        time_budget_df,
        outdir,
        config,
        aggregate_edges,
    )
    if bool(config.get("outputs", {}).get("write_adjacency_matrices", True)):
        write_adjacency_matrices(edge_df, outdir, config, cows)
    write_plots(edge_df, node_df, time_budget_df, outdir, config, layout_df=layout_df)

    warnings = []
    warnings.extend(traj_report.get("warnings", []))
    warnings.extend(interaction_report.get("warnings", []))
    warnings.extend(edge_report.get("warnings", []))
    warnings.extend(validate_outputs(edge_df))
    summary = {
        "config": config,
        "n_trajectory_rows": int(len(traj)),
        "n_interaction_rows": int(len(interactions)),
        "n_cows": int(len(cows)),
        "n_frames": int(traj[["farm", "camera", "clip", "frame"]].drop_duplicates().shape[0]),
        "n_valid_pair_frames": int(edge_report.get("n_opportunity_pair_frames", 0)),
        "n_dropped_self_pairs": int(interaction_report.get("n_dropped_self_pairs", 0)),
        "n_dropped_missing_trajectory": int(edge_report.get("n_dropped_missing_trajectory", 0)),
        "n_unknown_zone_points": int((traj["zone"].astype(str) == "unknown").sum()),
        "time_window_start_s": float(traj["time_s"].min()) if not traj.empty else None,
        "time_window_end_s": float(traj["time_s"].max()) if not traj.empty else None,
        "warnings": warnings,
        "outputs": {
            "edge_level": str(outdir / "edge_level.csv"),
            "node_descriptors": str(outdir / "node_descriptors.csv"),
            "zone_node_descriptors": str(outdir / "zone_node_descriptors.csv"),
            **report_paths,
        },
    }
    (outdir / "analysis_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    return summary


def validate_outputs(edge_df: pd.DataFrame) -> list[str]:
    warnings: list[str] = []
    if edge_df.empty:
        warnings.append("edge_level.csv is empty")
        return warnings
    tolerance = 1.0e-6
    bad_expected = edge_df["expected_seconds"] > edge_df["opportunity_seconds"] + tolerance
    if bool(bad_expected.any()):
        warnings.append(f"{int(bad_expected.sum())} edge rows have expected_seconds > opportunity_seconds")
    bad_rate = (edge_df["normalized_rate"] < -tolerance) | (edge_df["normalized_rate"] > 1.0 + tolerance)
    if bool(bad_rate.any()):
        warnings.append(f"{int(bad_rate.sum())} edge rows have normalized_rate outside [0, 1]")
    return warnings


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run zone-aware dairy cattle social network analysis.")
    parser.add_argument("--trajectories", required=True)
    parser.add_argument("--interactions", required=True)
    parser.add_argument("--zones", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--outdir", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    summary = run_pipeline(args.trajectories, args.interactions, args.zones, args.config, args.outdir)
    print(
        f"[done] cows={summary['n_cows']} frames={summary['n_frames']} "
        f"edge_rows={Path(summary['outputs']['edge_level']).stat().st_size}B outdir={args.outdir}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
