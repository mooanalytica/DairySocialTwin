from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd


def write_plots(
    edge_df: pd.DataFrame,
    node_df: pd.DataFrame,
    time_budget_df: pd.DataFrame,
    outdir: str | Path,
    config: dict[str, Any],
    layout_df: pd.DataFrame | None = None,
) -> None:
    if not bool(config.get("outputs", {}).get("write_plots", True)):
        return
    mpl_config_dir = Path.cwd() / ".cache" / "matplotlib"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))
    import matplotlib.pyplot as plt

    plot_dir = Path(outdir) / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    for layer in ("friendly", "unfriendly"):
        fig, ax = plt.subplots(figsize=(8, 5))
        sub = edge_df.loc[edge_df["layer"] == layer] if not edge_df.empty else edge_df
        if layout_df is not None and not layout_df.empty:
            draw_network_plot(ax, sub, layout_df, layer)
        elif sub.empty:
            ax.text(0.5, 0.5, "no edges", ha="center", va="center")
        else:
            totals = sub.groupby(["cow_i", "cow_j"])["expected_seconds"].sum().sort_values(ascending=False).head(20)
            totals.plot(kind="bar", ax=ax)
        ax.set_title(f"{layer} network edge weights")
        ax.set_ylabel("expected_seconds")
        fig.tight_layout()
        fig.savefig(plot_dir / f"network_{layer}_all_zones.png")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    if edge_df.empty:
        ax.text(0.5, 0.5, "no edges", ha="center", va="center")
    else:
        edge_df.groupby(["layer", "zone"])["expected_seconds"].sum().unstack(0).fillna(0.0).plot(kind="bar", ax=ax)
    ax.set_ylabel("expected_seconds")
    fig.tight_layout()
    fig.savefig(plot_dir / "zone_interaction_barplot.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    zone_cols = [col for col in time_budget_df.columns if col.startswith("time_") and col.endswith("_s")]
    if zone_cols:
        time_budget_df.set_index("cow_id")[zone_cols].plot(kind="bar", stacked=True, ax=ax)
    fig.tight_layout()
    fig.savefig(plot_dir / "cow_time_budget_by_zone.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    if "isolation_score" in node_df.columns:
        node_df.set_index("cow_id")["isolation_score"].sort_values(ascending=False).plot(kind="bar", ax=ax)
    fig.tight_layout()
    fig.savefig(plot_dir / "isolation_score_barplot.png")
    plt.close(fig)


def draw_network_plot(ax, edge_df: pd.DataFrame, layout_df: pd.DataFrame, layer: str) -> None:
    positions = {
        str(row["cow_id"]): (float(row["layout_x"]), float(row["layout_y"]))
        for _, row in layout_df.iterrows()
    }
    color = "#2f9e44" if layer == "friendly" else "#e03131"
    grouped = (
        edge_df.groupby(["cow_i", "cow_j"], as_index=False)["expected_seconds"].sum()
        if not edge_df.empty
        else pd.DataFrame(columns=["cow_i", "cow_j", "expected_seconds"])
    )
    max_weight = float(grouped["expected_seconds"].max()) if not grouped.empty else 0.0
    for _, row in grouped.iterrows():
        weight = float(row["expected_seconds"])
        if weight <= 0:
            continue
        cow_i = str(row["cow_i"])
        cow_j = str(row["cow_j"])
        if cow_i not in positions or cow_j not in positions:
            continue
        x1, y1 = positions[cow_i]
        x2, y2 = positions[cow_j]
        width = 0.8 + 4.0 * (weight / max_weight if max_weight > 0 else 0.0)
        ax.plot([x1, x2], [y1, y2], color=color, alpha=0.45, linewidth=width, solid_capstyle="round")
    xs = [positions[cow][0] for cow in sorted(positions)]
    ys = [positions[cow][1] for cow in sorted(positions)]
    ax.scatter(xs, ys, s=130, color="#f8f9fa", edgecolor="#343a40", linewidth=1.2, zorder=3)
    for cow in sorted(positions):
        x, y = positions[cow]
        ax.text(x, y, cow, ha="center", va="center", fontsize=8, zorder=4)
    ax.set_aspect("equal", adjustable="datalim")
    ax.axis("off")
