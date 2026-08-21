from __future__ import annotations

import io
import math
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch, Polygon

import webuil_figure_source as source
from dashboard_data import DashboardData


_RENDER_DPI = 145.0
_TIGHT_BBOX_PAD_INCHES = 0.1
_HIT_SPECS_ATTRIBUTE = "_dashboard_hit_specs"


class RenderSupersededError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class RenderedFigure:
    image: bytes
    width: int
    height: int
    regions: tuple[dict[str, object], ...]


def _set_hit_specs(figure: object, specs: list[dict[str, object]]) -> None:
    setattr(figure, _HIT_SPECS_ATTRIBUTE, tuple(specs))


def _ellipse_hit_spec(
    axis: object,
    cow_id: object,
    x: object,
    y: object,
    marker_size: object,
    *,
    marker_linewidth: object = 0.0,
    pick: str | None = None,
) -> dict[str, object]:
    spec: dict[str, object] = {
        "shape": "ellipse",
        "axis": axis,
        "cowId": str(cow_id),
        "x": float(x),
        "y": float(y),
        "markerSize": float(marker_size),
        "markerLinewidth": float(marker_linewidth),
    }
    if pick is not None:
        spec["pick"] = pick
    return spec


def _rect_hit_spec(
    axis: object,
    cow_id: object,
    x0: object,
    y0: object,
    x1: object,
    y1: object,
) -> dict[str, object]:
    return {
        "shape": "rect",
        "axis": axis,
        "cowId": str(cow_id),
        "x0": float(x0),
        "y0": float(y0),
        "x1": float(x1),
        "y1": float(y1),
    }


def _stacked_bar_hit_specs(
    axis: object,
    cow_ids: list[str],
    containers: list[object],
) -> list[dict[str, object]]:
    specs: list[dict[str, object]] = []
    for index, cow_id in enumerate(cow_ids):
        patches = [container.patches[index] for container in containers]
        x_values = [
            coordinate
            for patch in patches
            for coordinate in (patch.get_x(), patch.get_x() + patch.get_width())
        ]
        y_values = [
            coordinate
            for patch in patches
            for coordinate in (patch.get_y(), patch.get_y() + patch.get_height())
        ]
        specs.append(
            _rect_hit_spec(
                axis,
                cow_id,
                min(x_values),
                min(y_values),
                max(x_values),
                max(y_values),
            )
        )
    return specs


class FigureRenderer:
    def __init__(self, data: DashboardData, max_cache_entries: int = 96) -> None:
        if max_cache_entries < 1:
            raise ValueError("max_cache_entries must be positive")
        self.data = data
        self.max_cache_entries = max_cache_entries
        self._render_lock = threading.Lock()
        self._cache: OrderedDict[
            tuple[str, tuple[str, ...]],
            RenderedFigure,
        ] = OrderedDict()
        self._figure_builders: dict[str, Callable[[list[str]], object]] = {
            "01": self._figure_01,
            "02": self._figure_02,
            "03": self._figure_03,
            "04A": lambda selected: self._figure_04(selected, "friendly", "04A"),
            "04B": lambda selected: self._figure_04(selected, "unfriendly", "04B"),
            "06": self._figure_06,
            "07": self._figure_07,
            "08": self._figure_08,
            "09": self._figure_09,
            "10": self._figure_10,
        }
        for figure_key, zone in self.data.figure_zone.items():
            self._figure_builders[figure_key] = (
                lambda selected, figure_key=figure_key, zone=zone: self._figure_05(
                    selected,
                    figure_key,
                    zone,
                )
            )

    @property
    def figure_keys(self) -> set[str]:
        return set(self._figure_builders)

    def render(
        self,
        figure_key: str,
        selected: list[str],
        should_cancel: Callable[[], bool] | None = None,
    ) -> bytes:
        return self.render_result(figure_key, selected, should_cancel).image

    def render_result(
        self,
        figure_key: str,
        selected: list[str],
        should_cancel: Callable[[], bool] | None = None,
    ) -> RenderedFigure:
        if figure_key not in self._figure_builders:
            raise ValueError(f"Unknown figure: {figure_key}")
        selection = tuple(sorted(set(selected), key=source.cow_sort_key))
        cache_key = (figure_key, selection)
        if should_cancel is not None and should_cancel():
            raise RenderSupersededError("A newer figure request superseded this render")
        with self._render_lock:
            if should_cancel is not None and should_cancel():
                raise RenderSupersededError("A newer figure request superseded this render")
            cached = self._cache.get(cache_key)
            if cached is not None:
                self._cache.move_to_end(cache_key)
                return cached
            if should_cancel is not None and should_cancel():
                raise RenderSupersededError("A newer figure request superseded this render")
            figure = self._figure_builders[figure_key](list(selection))
            result = self._encode_figure_result(figure)
            self._cache[cache_key] = result
            self._cache.move_to_end(cache_key)
            while len(self._cache) > self.max_cache_entries:
                self._cache.popitem(last=False)
            return result

    def _encode_figure(self, figure: object) -> bytes:
        return self._encode_figure_result(figure).image

    def _encode_figure_result(self, figure: object) -> RenderedFigure:
        buffer = io.BytesIO()
        try:
            figure.patch.set_facecolor("#fbfbf8")
            figure.set_dpi(_RENDER_DPI)
            figure.tight_layout(rect=(0.0, 0.015, 1.0, 0.96))
            figure.canvas.draw()
            renderer = figure.canvas.get_renderer()
            tight_bbox = figure.get_tightbbox(renderer)
            if tight_bbox is None:
                raise RuntimeError("Unable to determine the figure's tight bounding box")
            padded_bbox = tight_bbox.padded(_TIGHT_BBOX_PAD_INCHES)
            if (
                not np.isfinite(
                    [
                        padded_bbox.x0,
                        padded_bbox.y0,
                        padded_bbox.x1,
                        padded_bbox.y1,
                    ]
                ).all()
                or padded_bbox.width <= 0.0
                or padded_bbox.height <= 0.0
            ):
                raise RuntimeError("The figure's padded tight bounding box is invalid")

            pixel_specs = self._hit_specs_in_output_pixels(
                figure,
                padded_bbox,
                _RENDER_DPI,
            )
            figure.savefig(
                buffer,
                format="png",
                dpi=_RENDER_DPI,
                bbox_inches=padded_bbox,
                pad_inches=0.0,
                facecolor="#fbfbf8",
            )
        finally:
            source.plt.close(figure)
        image = buffer.getvalue()
        width, height = self._png_dimensions(image)
        regions = self._normalize_hit_regions(pixel_specs, width, height)
        return RenderedFigure(
            image=image,
            width=width,
            height=height,
            regions=regions,
        )

    @staticmethod
    def _png_dimensions(image: bytes) -> tuple[int, int]:
        if (
            len(image) < 24
            or image[:8] != b"\x89PNG\r\n\x1a\n"
            or image[12:16] != b"IHDR"
        ):
            raise RuntimeError("Rendered figure is not a PNG with an IHDR header")
        width = int.from_bytes(image[16:20], "big")
        height = int.from_bytes(image[20:24], "big")
        if width <= 0 or height <= 0:
            raise RuntimeError("Rendered PNG has invalid dimensions")
        return width, height

    @staticmethod
    def _hit_specs_in_output_pixels(
        figure: object,
        padded_bbox: object,
        dpi: float,
    ) -> list[dict[str, object]]:
        specs = getattr(figure, _HIT_SPECS_ATTRIBUTE, ())
        output: list[dict[str, object]] = []
        crop_x0 = float(padded_bbox.x0) * dpi
        crop_y1 = float(padded_bbox.y1) * dpi
        for spec in specs:
            axis = spec["axis"]
            shape = str(spec["shape"])
            if shape == "ellipse":
                marker_size = float(spec["markerSize"])
                marker_linewidth = float(spec["markerLinewidth"])
                if (
                    not math.isfinite(marker_size)
                    or marker_size <= 0.0
                    or not math.isfinite(marker_linewidth)
                    or marker_linewidth < 0.0
                ):
                    continue
                center = axis.transData.transform(
                    (float(spec["x"]), float(spec["y"]))
                )
                center_x = float(center[0]) - crop_x0
                center_y = crop_y1 - float(center[1])
                radius = (
                    0.5 * (math.sqrt(marker_size) + marker_linewidth) * dpi / 72.0
                )
                values = (center_x, center_y, radius)
                if not all(math.isfinite(value) for value in values) or radius <= 0.0:
                    continue
                region: dict[str, object] = {
                    "shape": "ellipse",
                    "cowId": str(spec["cowId"]),
                    "cx": center_x,
                    "cy": center_y,
                    "rx": radius,
                    "ry": radius,
                }
                if "pick" in spec:
                    region["pick"] = str(spec["pick"])
                output.append(region)
                continue

            if shape != "rect":
                raise RuntimeError(f"Unsupported internal hit-region shape: {shape}")
            corners = axis.transData.transform(
                np.asarray(
                    [
                        [float(spec["x0"]), float(spec["y0"])],
                        [float(spec["x0"]), float(spec["y1"])],
                        [float(spec["x1"]), float(spec["y0"])],
                        [float(spec["x1"]), float(spec["y1"])],
                    ],
                    dtype=float,
                )
            )
            x_values = corners[:, 0] - crop_x0
            y_values = crop_y1 - corners[:, 1]
            bounds = (
                float(np.min(x_values)),
                float(np.min(y_values)),
                float(np.max(x_values)),
                float(np.max(y_values)),
            )
            if not all(math.isfinite(value) for value in bounds):
                continue
            output.append(
                {
                    "shape": "rect",
                    "cowId": str(spec["cowId"]),
                    "x0": bounds[0],
                    "y0": bounds[1],
                    "x1": bounds[2],
                    "y1": bounds[3],
                }
            )
        return output

    @staticmethod
    def _normalize_hit_regions(
        pixel_regions: list[dict[str, object]],
        width: int,
        height: int,
    ) -> tuple[dict[str, object], ...]:
        def clamp(value: float, maximum: int) -> float:
            return min(max(value, 0.0), float(maximum)) / float(maximum)

        regions: list[dict[str, object]] = []
        for pixel_region in pixel_regions:
            if pixel_region["shape"] == "ellipse":
                region: dict[str, object] = {
                    "shape": "ellipse",
                    "cowId": str(pixel_region["cowId"]),
                    "cx": clamp(float(pixel_region["cx"]), width),
                    "cy": clamp(float(pixel_region["cy"]), height),
                    "rx": clamp(float(pixel_region["rx"]), width),
                    "ry": clamp(float(pixel_region["ry"]), height),
                }
                if float(region["rx"]) <= 0.0 or float(region["ry"]) <= 0.0:
                    continue
                if "pick" in pixel_region:
                    region["pick"] = str(pixel_region["pick"])
                regions.append(region)
                continue

            x0 = clamp(float(pixel_region["x0"]), width)
            y0 = clamp(float(pixel_region["y0"]), height)
            x1 = clamp(float(pixel_region["x1"]), width)
            y1 = clamp(float(pixel_region["y1"]), height)
            if x1 <= x0 or y1 <= y0:
                continue
            regions.append(
                {
                    "shape": "rect",
                    "cowId": str(pixel_region["cowId"]),
                    "x0": x0,
                    "y0": y0,
                    "x1": x1,
                    "y1": y1,
                }
            )
        return tuple(regions)

    def _draw_structures(self, axis: object) -> list[dict[str, object]]:
        labels_seen: set[str] = set()
        for item in self.data.structures:
            label = str(item["label"])
            display_label = label if label not in labels_seen else None
            labels_seen.add(label)
            axis.add_patch(
                Polygon(
                    item["points"],
                    closed=True,
                    facecolor=item["color"],
                    edgecolor=item["color"],
                    linewidth=1.6,
                    alpha=0.18,
                    label=display_label,
                )
            )
        return self.data.structures

    def _selected_edge_frame(
        self,
        frame: pd.DataFrame,
        selected: set[str],
    ) -> pd.DataFrame:
        if frame.empty or not selected:
            return frame.iloc[0:0].copy()
        mask = (
            frame["cow_i"].astype(str).isin(selected)
            & frame["cow_j"].astype(str).isin(selected)
        )
        return frame.loc[mask].copy()

    def _figure_01(self, selected: list[str]) -> object:
        data = self.data.data
        cow_frame = self.data.cow_frame()
        layout = data["layout"]
        width, height = source.canvas_size(data["zones"], cow_frame, layout)
        figure, axis = source.plt.subplots(figsize=(12, 7))
        source.draw_zone_polygons(axis, data["zones"])
        structures = self._draw_structures(axis)
        obstacle_polygons = [
            item["points"]
            for item in structures
            if item["class_id"] == "obstacle"
        ]

        selected_set = set(selected)
        rows = cow_frame.loc[cow_frame["cow_id"].astype(str).isin(selected_set)]
        for cow_id, group in rows.groupby("cow_id", sort=True):
            cow = str(cow_id)
            source.draw_trajectory_group(
                axis,
                group,
                self.data.colors[cow],
                obstacle_polygons,
            )

        hit_specs: list[dict[str, object]] = []
        for cow in selected:
            x, y = self.data.positions[cow]
            label = source.figure_identity_label(cow)
            axis.scatter(
                [x],
                [y],
                s=72,
                color=self.data.colors[cow],
                edgecolor="#212529",
                linewidth=0.7,
                alpha=0.88,
                zorder=4,
            )
            axis.text(
                x + 28,
                y,
                label,
                fontsize=8,
                va="center",
                color="#212529",
                zorder=5,
            )
            hit_specs.append(
                _ellipse_hit_spec(
                    axis,
                    cow,
                    x,
                    y,
                    72.0,
                    marker_linewidth=0.7,
                )
            )
        if not selected:
            axis.text(
                0.5,
                0.5,
                "No cattle selected",
                ha="center",
                va="center",
                transform=axis.transAxes,
                fontsize=12,
                color="#6b7280",
                zorder=6,
            )
        axis.set_title(
            f"{self.data.data['sample_id']} Figure 1. Floorplan trajectories "
            f"({len(selected)} selected)"
        )
        source.floorplan_axes(axis, width, height, labels=True)
        handles, labels = axis.get_legend_handles_labels()
        if handles:
            axis.legend(
                handles,
                labels,
                loc="center left",
                bbox_to_anchor=(1.01, 0.5),
                fontsize=8,
                frameon=False,
            )
        _set_hit_specs(figure, hit_specs)
        return figure

    def _figure_02(self, selected: list[str]) -> object:
        full = self.data.data["time_budget"].copy()
        full["visible_time_s"] = pd.to_numeric(full["visible_time_s"], errors="raise")
        full = full.sort_values(
            ["visible_time_s", "cow_id"],
            ascending=[False, True],
            kind="stable",
        )
        selected_set = set(selected)
        budget = full.loc[full["cow_id"].astype(str).isin(selected_set)].copy()
        zone_columns = source.time_budget_zone_columns(full)
        global_height = (
            full[zone_columns]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0.0)
            .sum(axis=1)
            .max()
            if zone_columns
            else 1.0
        )
        global_height = max(float(global_height), 1.0)

        figure, axis = source.plt.subplots(figsize=(12.5, 5.5))
        bar_containers: list[object] = []
        if zone_columns and not budget.empty:
            bottoms = np.zeros(len(budget))
            x_values = np.arange(len(budget))
            for column in zone_columns:
                zone = column.removeprefix("time_").removesuffix("_s")
                values = (
                    pd.to_numeric(budget[column], errors="coerce")
                    .fillna(0.0)
                    .to_numpy(dtype=float)
                )
                bar_containers.append(
                    axis.bar(
                        x_values,
                        values,
                        bottom=bottoms,
                        label=zone,
                        color=source.zone_color(zone),
                        alpha=0.85,
                    )
                )
                bottoms += values
            axis.set_xticks(x_values)
            axis.set_xticklabels(
                [
                    source.figure_identity_label(cow)
                    for cow in budget["cow_id"].astype(str)
                ],
                rotation=45,
                ha="right",
            )
            axis.legend(
                loc="center left",
                bbox_to_anchor=(1.01, 0.5),
                fontsize=8,
                frameon=False,
            )
        else:
            axis.text(
                0.5,
                0.5,
                "No cattle selected",
                ha="center",
                va="center",
                transform=axis.transAxes,
                color="#6b7280",
            )
        axis.set_ylim(0.0, global_height * 1.05)
        axis.set_ylabel("seconds")
        axis.set_title(
            f"{self.data.data['sample_id']} Figure 2. Cow visibility and zone time budget"
        )
        _set_hit_specs(
            figure,
            _stacked_bar_hit_specs(
                axis,
                budget["cow_id"].astype(str).tolist(),
                bar_containers,
            )
            if bar_containers
            else [],
        )
        return figure

    def _figure_03(self, _selected: list[str]) -> object:
        data = self.data.data
        summary = source.zone_net_layer_summary(data["edge"])
        zones = source.ordered_figure_zones(data["edge"])
        figure, axes = source.plt.subplots(1, 2, figsize=(12, 5))
        for axis, value_column, title in (
            (axes[0], "net_expected_seconds", "Net expected seconds"),
            (axes[1], "net_normalized_rate", "Net normalized rate"),
        ):
            pivot = summary.pivot_table(
                index="zone",
                columns="layer",
                values=value_column,
                aggfunc="sum",
                fill_value=0,
            )
            pivot = pivot.reindex(zones, axis=0, fill_value=0.0)
            pivot = pivot.reindex(
                ["friendly", "unfriendly"],
                axis=1,
                fill_value=0.0,
            )
            x_values = np.arange(len(pivot), dtype=float)
            axis.bar(
                x_values,
                pivot["friendly"].to_numpy(dtype=float),
                color=source.layer_color("friendly"),
                alpha=0.85,
                label="friendly-dominant",
            )
            axis.bar(
                x_values,
                -pivot["unfriendly"].to_numpy(dtype=float),
                color=source.layer_color("unfriendly"),
                alpha=0.85,
                label="unfriendly-dominant",
            )
            axis.axhline(0.0, color="#495057", linewidth=0.8)
            axis.set_title(title)
            axis.set_xlabel("zone")
            axis.set_xticks(x_values)
            axis.set_xticklabels(pivot.index.astype(str), rotation=35, ha="right")
            axis.set_ylabel("friendly (+) / unfriendly (-)")
            axis.legend(frameon=False, fontsize=8)
        figure.suptitle(
            f"{data['sample_id']} Figure 3. Dyad-exclusive net interaction volume by zone "
            f"(dominance >= {source.FIGURE_NET_DOMINANCE_THRESHOLD:.2f})"
        )
        return figure

    def _figure_04(self, selected: list[str], layer: str, suffix: str) -> object:
        data = self.data.data
        selected_set = set(selected)
        network = self._selected_edge_frame(
            self.data.full_networks[layer],
            selected_set,
        )
        positions = {cow: self.data.positions[cow] for cow in selected}
        width, height = source.canvas_size(
            data["zones"],
            self.data.cow_frame(),
            data["layout"],
        )
        figure, axis = source.plt.subplots(figsize=(8, 6))
        self._draw_structures(axis)
        source.draw_network(
            axis,
            network,
            positions,
            layer,
            f"dominant {layer}",
            communities=self.data.communities,
            node_colors=self.data.colors,
            node_size=250,
            label_beside=True,
            max_weight=self.data.full_network_max,
            alpha_column="dominance",
        )
        _set_hit_specs(
            figure,
            [
                _ellipse_hit_spec(
                    axis,
                    cow,
                    x,
                    y,
                    250.0,
                    marker_linewidth=0.8,
                )
                for cow, (x, y) in positions.items()
            ],
        )
        source.floorplan_axes(axis, width, height, labels=False)
        figure.suptitle(
            f"{data['sample_id']} Figure {suffix}. Full-window {layer} social network "
            f"({len(selected)} selected)"
        )
        return figure

    def _figure_05(
        self,
        selected: list[str],
        figure_key: str,
        zone: str,
    ) -> object:
        selected_set = set(selected)
        positions = source.circle_positions(selected)
        networks = {
            layer: self._selected_edge_frame(network, selected_set)
            for layer, network in self.data.zone_networks[zone].items()
        }
        figure, axes = source.plt.subplots(1, 2, figsize=(9, 4.8), squeeze=False)
        hit_specs: list[dict[str, object]] = []
        for column, layer in enumerate(("friendly", "unfriendly")):
            axis = axes[0][column]
            source.draw_network(
                axis,
                networks[layer],
                positions,
                layer,
                f"dominant {layer}: {zone}",
                communities=self.data.communities,
                node_colors=self.data.colors,
                node_size=220,
                label_beside=False,
                max_weight=self.data.zone_network_max[zone],
                alpha_column="dominance",
            )
            hit_specs.extend(
                _ellipse_hit_spec(
                    axis,
                    cow,
                    x,
                    y,
                    220.0,
                    marker_linewidth=0.8,
                )
                for cow, (x, y) in positions.items()
            )
            axis.set_aspect("equal", adjustable="box")
            axis.set_xlim(-1.22, 1.22)
            axis.set_ylim(-1.22, 1.22)
        figure.suptitle(
            f"{self.data.data['sample_id']} Figure {figure_key.removeprefix('0')}. "
            f"Zone-specific network: {zone} ({len(selected)} selected)"
        )
        _set_hit_specs(figure, hit_specs)
        return figure

    def _figure_06(self, selected: list[str]) -> object:
        ordered = [cow for cow in self.data.matrix_cow_order if cow in set(selected)]
        figure, axes = source.plt.subplots(1, 2, figsize=(12, 5))
        for axis, layer in zip(axes, ("friendly", "unfriendly")):
            if ordered:
                matrix = self.data.full_matrices[layer].loc[ordered, ordered]
                image = axis.imshow(
                    matrix.to_numpy(dtype=float),
                    cmap="Greens" if layer == "friendly" else "Reds",
                    vmin=0.0,
                    vmax=self.data.full_matrix_vmax,
                )
                fontsize = 8 if len(ordered) <= 20 else 6
                axis.set_xticks(range(len(ordered)))
                axis.set_xticklabels(
                    [source.figure_identity_label(cow) for cow in ordered],
                    rotation=90,
                    fontsize=fontsize,
                )
                axis.set_yticks(range(len(ordered)))
                axis.set_yticklabels(
                    [source.figure_identity_label(cow) for cow in ordered],
                    fontsize=fontsize,
                )
                figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
            else:
                axis.text(
                    0.5,
                    0.5,
                    "No cattle selected",
                    ha="center",
                    va="center",
                    transform=axis.transAxes,
                    color="#6b7280",
                )
                axis.set_xticks([])
                axis.set_yticks([])
            axis.set_title(f"{layer.capitalize()}-dominant net expected seconds")
        figure.suptitle(
            f"{self.data.data['sample_id']} Figure 6. Selected net adjacency "
            f"(global vmax; dominance >= {source.FIGURE_NET_DOMINANCE_THRESHOLD:.2f})"
        )
        return figure

    def _figure_07(self, selected: list[str]) -> object:
        full = self.data.data["node"].copy().sort_values("cow_id", kind="stable")
        selected_set = set(selected)
        node = full.loc[full["cow_id"].astype(str).isin(selected_set)].copy()
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
        metrics = [column for column in metrics if column in full.columns]
        figure, axes = source.plt.subplots(
            1,
            2,
            figsize=(14, 6),
            gridspec_kw={"width_ratios": [1, 1.45]},
        )
        hit_specs: list[dict[str, object]] = []

        global_x = pd.to_numeric(full["friendly_strength_rate"], errors="coerce").fillna(0)
        global_y = pd.to_numeric(full["unfriendly_strength_rate"], errors="coerce").fillna(0)
        axis = axes[0]
        if not node.empty:
            x_values = pd.to_numeric(
                node["friendly_strength_rate"],
                errors="coerce",
            ).fillna(0)
            y_values = pd.to_numeric(
                node["unfriendly_strength_rate"],
                errors="coerce",
            ).fillna(0)
            isolation = pd.to_numeric(node["isolation_score"], errors="coerce").fillna(0)
            sizes = 80 + 320 * isolation
            colors = [self.data.colors[str(cow)] for cow in node["cow_id"]]
            point_collection = axis.scatter(
                x_values,
                y_values,
                s=sizes,
                color=colors,
                edgecolor="#212529",
                alpha=0.82,
            )
            point_linewidths = point_collection.get_linewidths()
            point_linewidth = (
                float(point_linewidths[0])
                if len(point_linewidths) > 0
                else 0.0
            )
            for row_index, (_, row) in enumerate(node.iterrows()):
                label = source.figure_identity_label(row["cow_id"])
                axis.text(
                    float(x_values.iloc[row_index]),
                    float(y_values.iloc[row_index]),
                    label,
                    ha="center",
                    va="center",
                    fontsize=source.marker_label_font_size(
                        float(sizes.iloc[row_index]),
                        label,
                    ),
                    color=source.contrasting_text_color(
                        colors[row_index],
                        alpha=0.82,
                    ),
                    zorder=3,
                )
                hit_specs.append(
                    _ellipse_hit_spec(
                        axis,
                        row["cow_id"],
                        x_values.iloc[row_index],
                        y_values.iloc[row_index],
                        sizes.iloc[row_index],
                        marker_linewidth=point_linewidth,
                        pick="nearest",
                    )
                )
        else:
            axis.text(
                0.5,
                0.5,
                "No cattle selected",
                ha="center",
                va="center",
                transform=axis.transAxes,
                color="#6b7280",
            )
        x_limits = source.padded_numeric_limits(global_x, left=0.12, right=0.24)
        y_limits = source.padded_numeric_limits(global_y, left=0.14, right=0.14)
        if x_limits:
            axis.set_xlim(*x_limits)
        if y_limits:
            axis.set_ylim(*y_limits)
        axis.set_xlabel("friendly_strength_rate")
        axis.set_ylabel("unfriendly_strength_rate")
        axis.set_title("rate scatter")

        axis = axes[1]
        if metrics and not node.empty:
            normalized_full = source.minmax_frame(full, metrics)
            normalized = normalized_full.loc[node.index, metrics]
            image = axis.imshow(
                normalized.to_numpy(dtype=float),
                aspect="auto",
                cmap="viridis",
                vmin=0.0,
                vmax=1.0,
            )
            axis.set_xticks(range(len(metrics)))
            axis.set_xticklabels(metrics, rotation=45, ha="right", fontsize=8)
            axis.set_yticks(range(len(node)))
            axis.set_yticklabels(
                [
                    source.figure_identity_label(cow)
                    for cow in node["cow_id"].astype(str)
                ],
                fontsize=8,
            )
            figure.colorbar(
                image,
                ax=axis,
                fraction=0.046,
                pad=0.04,
                label="global min-max scaled",
            )
            hit_specs.extend(
                _rect_hit_spec(
                    axis,
                    cow,
                    -0.5,
                    row_index - 0.5,
                    len(metrics) - 0.5,
                    row_index + 0.5,
                )
                for row_index, cow in enumerate(node["cow_id"].astype(str))
            )
        else:
            axis.text(
                0.5,
                0.5,
                "No cattle selected",
                ha="center",
                va="center",
                transform=axis.transAxes,
                color="#6b7280",
            )
            axis.set_xticks([])
            axis.set_yticks([])
        axis.set_title("descriptor heatmap")
        figure.suptitle(
            f"{self.data.data['sample_id']} Figure 7. Cow-level descriptor dashboard "
            f"({len(selected)} selected)"
        )
        _set_hit_specs(figure, hit_specs)
        return figure

    def _figure_08(self, selected: list[str]) -> object:
        axis_frame = self.data.community_axis
        full_matrix = self.data.community_matrix
        selected_order = [cow for cow in full_matrix.index if cow in set(selected)]
        values = pd.Series(full_matrix.to_numpy(dtype=float).ravel()).dropna()
        ticks = sorted(int(value) for value in values.unique()) if not values.empty else []
        colors = [
            source.plt.get_cmap("tab20")(index % 20)
            for index, _ in enumerate(ticks)
        ] or ["#ffffff"]
        color_map = ListedColormap(colors).with_extremes(
            bad=source.MISSING_COMMUNITY_COLOR
        )
        normalization = None
        if ticks:
            if len(ticks) == 1:
                boundaries = [ticks[0] - 0.5, ticks[0] + 0.5]
            else:
                boundaries = [ticks[0] - 0.5]
                boundaries.extend(
                    (left + right) / 2.0
                    for left, right in zip(ticks, ticks[1:])
                )
                boundaries.append(ticks[-1] + 0.5)
            normalization = BoundaryNorm(boundaries, color_map.N)

        figure_height = max(5.0, min(8.2, 3.5 + len(selected_order) * 0.19))
        figure, axis = source.plt.subplots(figsize=(12, figure_height))
        hit_specs: list[dict[str, object]] = []
        if selected_order:
            matrix_frame = full_matrix.loc[selected_order]
            matrix = matrix_frame.to_numpy(dtype=float)
            axis.imshow(
                np.ma.masked_invalid(matrix),
                aspect="auto",
                interpolation="nearest",
                cmap=color_map,
                norm=normalization,
            )
            axis.set_yticks(range(len(matrix_frame.index)))
            axis.set_yticklabels(
                [
                    source.figure_identity_label(cow)
                    for cow in matrix_frame.index.astype(str)
                ],
                fontsize=8,
            )
            hit_specs.extend(
                _rect_hit_spec(
                    axis,
                    cow,
                    -0.5,
                    row_index - 0.5,
                    len(axis_frame) - 0.5,
                    row_index + 0.5,
                )
                for row_index, cow in enumerate(matrix_frame.index.astype(str))
            )
        else:
            empty_matrix = np.full((1, len(axis_frame)), np.nan, dtype=float)
            axis.imshow(
                np.ma.masked_invalid(empty_matrix),
                aspect="auto",
                interpolation="nearest",
                cmap=color_map,
                norm=normalization,
            )
            axis.set_yticks([])
            axis.text(
                0.5,
                0.5,
                "No cattle selected",
                ha="center",
                va="center",
                transform=axis.transAxes,
                color="#6b7280",
            )

        tick_positions = source.sparse_tick_positions(len(axis_frame), max_ticks=12)
        axis.set_xticks(tick_positions)
        origin = float(axis_frame["window_start_s"].iloc[0])
        axis.set_xticklabels(
            [
                f"{(float(axis_frame['window_start_s'].iloc[position]) - origin) / 3600.0:.1f}h"
                for position in tick_positions
            ],
            fontsize=8,
        )
        axis.legend(
            handles=[
                Patch(
                    facecolor=source.MISSING_COMMUNITY_COLOR,
                    edgecolor="#adb5bd",
                    label="Unavailable / <30 s visible / no eligible friendly edge",
                )
            ],
            loc="upper center",
            bbox_to_anchor=(0.5, -0.08),
            frameon=False,
            fontsize=8,
        )
        axis.set_xlabel("elapsed time (hours; complete 5-minute windows)")
        axis.set_title("community membership")
        figure.suptitle(
            f"{self.data.data['sample_id']} Figure 8. Existing community stability "
            f"({len(selected)} selected; no recomputation)"
        )
        _set_hit_specs(figure, hit_specs)
        return figure

    def _figure_09(self, selected: list[str]) -> object:
        node = self.data.data["node"].copy()
        weights = source.isolation_weights(self.data.data["summary"])
        components = list(weights)
        for column in components:
            if column not in node.columns:
                node[column] = 0.0
            node[f"{column}_component"] = (
                pd.to_numeric(node[column], errors="coerce").fillna(0.0)
                * weights[column]
            )
        global_total = node[
            [f"{column}_component" for column in components]
        ].sum(axis=1)
        global_height = max(float(global_total.max()), 1e-6)
        selected_set = set(selected)
        node = node.loc[node["cow_id"].astype(str).isin(selected_set)]
        node = node.sort_values(
            ["isolation_score", "cow_id"],
            ascending=[False, True],
            kind="stable",
        )

        figure, axis = source.plt.subplots(figsize=(11, 5))
        bar_containers: list[object] = []
        if not node.empty:
            x_values = np.arange(len(node))
            bottom = np.zeros(len(node))
            colors = ["#f08c00", "#e03131", "#5c7cfa"]
            for column, color in zip(components, colors):
                values = node[f"{column}_component"].to_numpy(dtype=float)
                bar_containers.append(
                    axis.bar(
                        x_values,
                        values,
                        bottom=bottom,
                        label=column,
                        color=color,
                        alpha=0.86,
                    )
                )
                bottom += values
            axis.set_xticks(x_values)
            axis.set_xticklabels(
                [
                    source.figure_identity_label(cow)
                    for cow in node["cow_id"].astype(str)
                ],
                rotation=45,
                ha="right",
            )
            axis.legend(frameon=False, fontsize=8)
        else:
            axis.text(
                0.5,
                0.5,
                "No cattle selected",
                ha="center",
                va="center",
                transform=axis.transAxes,
                color="#6b7280",
            )
        axis.set_ylim(0.0, global_height * 1.05)
        axis.set_ylabel("isolation score contribution")
        axis.set_title(
            f"{self.data.data['sample_id']} Figure 9. Isolation tendency decomposition"
        )
        _set_hit_specs(
            figure,
            _stacked_bar_hit_specs(
                axis,
                node["cow_id"].astype(str).tolist(),
                bar_containers,
            )
            if bar_containers
            else [],
        )
        return figure

    def _figure_10(self, _selected: list[str]) -> object:
        edge = self.data.data["edge"].copy()
        figure, axis = source.plt.subplots(figsize=(7, 5))
        for layer in sorted(edge["layer"].astype(str).unique()):
            subset = edge.loc[edge["layer"].astype(str) == layer]
            axis.scatter(
                pd.to_numeric(subset["opportunity_seconds"], errors="coerce"),
                pd.to_numeric(subset["expected_seconds"], errors="coerce"),
                label=layer,
                color=source.layer_color(layer),
                alpha=0.75,
            )
        axis.set_xlabel("opportunity_seconds")
        axis.set_ylabel("expected_seconds")
        axis.legend(frameon=False, fontsize=8)
        axis.set_title("expected vs opportunity")
        figure.suptitle(
            f"{self.data.data['sample_id']} Figure 10. Expected vs opportunity"
        )
        return figure
