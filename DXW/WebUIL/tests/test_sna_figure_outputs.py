from __future__ import annotations

import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import generate_sna_tf_outputs as figures


def community_data() -> dict[str, object]:
    cows = [f"G{index:04d}" for index in range(1, figures.EXPECTED_GLOBAL_IDENTITY_COUNT + 1)]
    return {
        "summary": {
            "time_window_start_s": 0.0,
            "time_window_end_s": 27_577.2497,
            "config": {"community": {"window_s": 300.0, "step_s": 300.0}},
        },
        "community_summary": pd.DataFrame(
            {
                "window_a": list(range(figures.EXPECTED_COMMUNITY_WINDOW_COUNT - 1)),
                "window_b": list(range(1, figures.EXPECTED_COMMUNITY_WINDOW_COUNT)),
            }
        ),
        "community_windows": pd.DataFrame(
            [
                {
                    "window_index": 0,
                    "window_start_s": 0.0,
                    "window_end_s": 300.0,
                    "cow_id": "G0001",
                    "community_id": 7,
                    "visible_time_s_in_window": 50.0,
                },
                {
                    "window_index": 1,
                    "window_start_s": 300.0,
                    "window_end_s": 600.0,
                    "cow_id": "G0001",
                    "community_id": 3,
                    "visible_time_s_in_window": 50.0,
                },
                {
                    "window_index": 3,
                    "window_start_s": 900.0,
                    "window_end_s": 1200.0,
                    "cow_id": "G0001",
                    "community_id": 3,
                    "visible_time_s_in_window": 50.0,
                },
            ]
        ),
        "layout": pd.DataFrame({"cow_id": cows}),
        "sample_id": figures.COMBINED_SAMPLE_ID,
    }


class FigureOutputContractTest(unittest.TestCase):
    def test_figure_01_groups_and_dynamic_output_contract(self) -> None:
        self.assertEqual(
            [end - start + 1 for start, end in figures.FIGURE_01_GROUPS],
            [10, 10, 10, 10, 10, 10, 2],
        )
        zones = figures.figure_05_zones(
            pd.DataFrame(
                {"zone": ["food", "rest", "wait_for_water", "water", "path", "cross_zone"]}
            )
        )
        expected_paths = figures.expected_figure_relative_paths(zones)
        figure_01_paths = sorted(
            path for path in expected_paths if path.startswith("figures/figure_01/")
        )
        self.assertEqual(len(figure_01_paths), 7)
        self.assertEqual(len(expected_paths), len(figures.FIXED_FIGURE_RELATIVE_PATHS) + len(zones))
        self.assertIn("figures/figure_05A_cross_zone_networks.png", expected_paths)
        self.assertIn("figures/figure_05F_water_networks.png", expected_paths)
        self.assertNotIn(
            "figures/figure_01_floorplan_trajectories_edges.png",
            expected_paths,
        )

    def test_figure_05_suffixes_and_region_count_are_dynamic(self) -> None:
        self.assertEqual(figures.alphabetic_figure_suffix(0), "A")
        self.assertEqual(figures.alphabetic_figure_suffix(25), "Z")
        self.assertEqual(figures.alphabetic_figure_suffix(26), "AA")
        self.assertEqual(figures.alphabetic_figure_suffix(27), "AB")
        with self.assertRaises(ValueError):
            figures.alphabetic_figure_suffix(-1)

        zones = ["alpha", "beta", "gamma"]
        paths = figures.expected_figure_relative_paths(zones)
        self.assertEqual(len(paths), len(figures.FIXED_FIGURE_RELATIVE_PATHS) + 3)
        self.assertIn("figures/figure_05A_alpha_networks.png", paths)
        self.assertIn("figures/figure_05C_gamma_networks.png", paths)

    def test_figure_identity_label_is_display_only_and_strict(self) -> None:
        self.assertEqual(figures.figure_identity_label("G0001"), "1")
        self.assertEqual(figures.figure_identity_label("G0062"), "62")
        for invalid in ("1", "G001", "G0063", "G0000", ""):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                figures.figure_identity_label(invalid)

    def test_node_label_style_uses_contrast_and_marker_area(self) -> None:
        self.assertEqual(figures.contrasting_text_color("#000000"), "#ffffff")
        self.assertEqual(figures.contrasting_text_color("#ffffff"), "#000000")
        self.assertEqual(figures.contrasting_text_color("#000000", alpha=0.1), "#000000")
        small = figures.marker_label_font_size(80.0, "12")
        large = figures.marker_label_font_size(400.0, "12")
        self.assertGreater(large, small)
        self.assertGreaterEqual(small, 5.0)
        self.assertLessEqual(large, 14.0)

    def test_pair_zone_figures_keep_current_six_region_contract(self) -> None:
        edge = pd.DataFrame({"zone": list(figures.FIGURE_ZONE_ORDER)})
        self.assertEqual(figures.ordered_figure_zones(edge), list(figures.FIGURE_ZONE_ORDER))
        with self.assertRaisesRegex(ValueError, "missing"):
            figures.ordered_figure_zones(edge.loc[edge["zone"] != "cross_zone"])

    def test_figure_05_discovers_regions_from_edge_data(self) -> None:
        edge = pd.DataFrame({"zone": ["water", "food", "new_region", "food", None]})
        with self.assertRaisesRegex(ValueError, "missing zone"):
            figures.figure_05_zones(edge)

        edge = edge.dropna().copy()
        self.assertEqual(figures.figure_05_zones(edge), ["food", "new_region", "water"])
        self.assertEqual(figures.figure_05_zones(edge.loc[edge["zone"] != "water"]), ["food", "new_region"])
        with self.assertRaisesRegex(ValueError, "zone column"):
            figures.figure_05_zones(pd.DataFrame({"layer": ["friendly"]}))
        with self.assertRaisesRegex(ValueError, "empty zone"):
            figures.figure_05_zones(pd.DataFrame({"zone": ["food", " "]}))
        with self.assertRaisesRegex(ValueError, "surrounding whitespace"):
            figures.figure_05_zones(pd.DataFrame({"zone": ["food", " water"]}))

    def test_zone_net_layer_summary_removes_same_dyad_companion_mass(self) -> None:
        edge = pd.DataFrame(
            [
                {
                    "cow_i": "G0001",
                    "cow_j": "G0002",
                    "zone": "food",
                    "layer": "friendly",
                    "expected_seconds": 6.0,
                    "opportunity_seconds": 10.0,
                },
                {
                    "cow_i": "G0001",
                    "cow_j": "G0002",
                    "zone": "food",
                    "layer": "unfriendly",
                    "expected_seconds": 4.0,
                    "opportunity_seconds": 10.0,
                },
                {
                    "cow_i": "G0003",
                    "cow_j": "G0004",
                    "zone": "food",
                    "layer": "friendly",
                    "expected_seconds": 3.0,
                    "opportunity_seconds": 20.0,
                },
                {
                    "cow_i": "G0003",
                    "cow_j": "G0004",
                    "zone": "food",
                    "layer": "unfriendly",
                    "expected_seconds": 9.0,
                    "opportunity_seconds": 20.0,
                },
                {
                    "cow_i": "G0005",
                    "cow_j": "G0006",
                    "zone": "food",
                    "layer": "friendly",
                    "expected_seconds": 5.5,
                    "opportunity_seconds": 30.0,
                },
                {
                    "cow_i": "G0005",
                    "cow_j": "G0006",
                    "zone": "food",
                    "layer": "unfriendly",
                    "expected_seconds": 4.5,
                    "opportunity_seconds": 30.0,
                },
            ]
        )

        summary = figures.zone_net_layer_summary(edge)
        friendly = summary.loc[
            (summary["zone"] == "food") & (summary["layer"] == "friendly")
        ].iloc[0]
        unfriendly = summary.loc[
            (summary["zone"] == "food") & (summary["layer"] == "unfriendly")
        ].iloc[0]

        self.assertAlmostEqual(float(friendly["net_expected_seconds"]), 2.0)
        self.assertAlmostEqual(float(unfriendly["net_expected_seconds"]), 6.0)
        self.assertAlmostEqual(float(friendly["opportunity_seconds"]), 60.0)
        self.assertAlmostEqual(float(unfriendly["opportunity_seconds"]), 60.0)
        self.assertAlmostEqual(float(friendly["net_normalized_rate"]), 2.0 / 60.0)
        self.assertAlmostEqual(float(unfriendly["net_normalized_rate"]), 6.0 / 60.0)

    def test_zone_net_layer_summary_keeps_zone_assignment_before_dominance(self) -> None:
        edge = pd.DataFrame(
            [
                {
                    "cow_i": "G0001",
                    "cow_j": "G0002",
                    "zone": "food",
                    "layer": "friendly",
                    "expected_seconds": 6.0,
                    "opportunity_seconds": 12.0,
                },
                {
                    "cow_i": "G0001",
                    "cow_j": "G0002",
                    "zone": "food",
                    "layer": "unfriendly",
                    "expected_seconds": 4.0,
                    "opportunity_seconds": 12.0,
                },
                {
                    "cow_i": "G0001",
                    "cow_j": "G0002",
                    "zone": "rest",
                    "layer": "friendly",
                    "expected_seconds": 0.0,
                    "opportunity_seconds": 15.0,
                },
                {
                    "cow_i": "G0001",
                    "cow_j": "G0002",
                    "zone": "rest",
                    "layer": "unfriendly",
                    "expected_seconds": 10.0,
                    "opportunity_seconds": 15.0,
                },
            ]
        )

        summary = figures.zone_net_layer_summary(edge)
        values = summary.set_index(["zone", "layer"])["net_expected_seconds"]
        self.assertAlmostEqual(float(values.loc[("food", "friendly")]), 2.0)
        self.assertAlmostEqual(float(values.loc[("food", "unfriendly")]), 0.0)
        self.assertAlmostEqual(float(values.loc[("rest", "friendly")]), 0.0)
        self.assertAlmostEqual(float(values.loc[("rest", "unfriendly")]), 10.0)

    def test_zone_net_layer_summary_rejects_layer_opportunity_mismatch(self) -> None:
        edge = pd.DataFrame(
            [
                {
                    "cow_i": "G0001",
                    "cow_j": "G0002",
                    "zone": "food",
                    "layer": "friendly",
                    "expected_seconds": 6.0,
                    "opportunity_seconds": 10.0,
                },
                {
                    "cow_i": "G0001",
                    "cow_j": "G0002",
                    "zone": "food",
                    "layer": "unfriendly",
                    "expected_seconds": 4.0,
                    "opportunity_seconds": 11.0,
                },
            ]
        )

        with self.assertRaisesRegex(ValueError, "opportunity seconds differ"):
            figures.zone_net_layer_summary(edge)

    def test_zone_net_layer_summary_handles_duplicates_missing_layer_and_zero_mass(self) -> None:
        edge = pd.DataFrame(
            [
                {
                    "cow_i": "G0001",
                    "cow_j": "G0002",
                    "zone": "food",
                    "layer": "friendly",
                    "expected_seconds": 3.0,
                    "opportunity_seconds": 5.0,
                },
                {
                    "cow_i": "G0001",
                    "cow_j": "G0002",
                    "zone": "food",
                    "layer": "friendly",
                    "expected_seconds": 3.0,
                    "opportunity_seconds": 5.0,
                },
                {
                    "cow_i": "G0001",
                    "cow_j": "G0002",
                    "zone": "food",
                    "layer": "unfriendly",
                    "expected_seconds": 2.0,
                    "opportunity_seconds": 5.0,
                },
                {
                    "cow_i": "G0001",
                    "cow_j": "G0002",
                    "zone": "food",
                    "layer": "unfriendly",
                    "expected_seconds": 2.0,
                    "opportunity_seconds": 5.0,
                },
                {
                    "cow_i": "G0003",
                    "cow_j": "G0004",
                    "zone": "food",
                    "layer": "friendly",
                    "expected_seconds": 6.0,
                    "opportunity_seconds": 10.0,
                },
                {
                    "cow_i": "G0005",
                    "cow_j": "G0006",
                    "zone": "food",
                    "layer": "friendly",
                    "expected_seconds": 0.0,
                    "opportunity_seconds": 4.0,
                },
                {
                    "cow_i": "G0005",
                    "cow_j": "G0006",
                    "zone": "food",
                    "layer": "unfriendly",
                    "expected_seconds": 0.0,
                    "opportunity_seconds": 4.0,
                },
                {
                    "cow_i": "G0007",
                    "cow_j": "G0008",
                    "zone": "food",
                    "layer": "friendly",
                    "expected_seconds": 4.0,
                    "opportunity_seconds": 8.0,
                },
                {
                    "cow_i": "G0007",
                    "cow_j": "G0008",
                    "zone": "food",
                    "layer": "unfriendly",
                    "expected_seconds": 4.0,
                    "opportunity_seconds": 8.0,
                },
            ]
        )

        summary = figures.zone_net_layer_summary(edge).set_index(["zone", "layer"])
        self.assertAlmostEqual(float(summary.loc[("food", "friendly"), "net_expected_seconds"]), 8.0)
        self.assertAlmostEqual(float(summary.loc[("food", "unfriendly"), "net_expected_seconds"]), 0.0)
        self.assertAlmostEqual(float(summary.loc[("food", "friendly"), "opportunity_seconds"]), 32.0)
        self.assertAlmostEqual(float(summary.loc[("food", "friendly"), "net_normalized_rate"]), 0.25)

    def test_net_dominance_networks_assign_each_dyad_to_one_layer(self) -> None:
        edge = pd.DataFrame(
            [
                {"cow_i": "G0001", "cow_j": "G0002", "zone": "food", "layer": "friendly", "expected_seconds": 6.0},
                {"cow_i": "G0001", "cow_j": "G0002", "zone": "rest", "layer": "friendly", "expected_seconds": 4.0},
                {"cow_i": "G0001", "cow_j": "G0002", "zone": "food", "layer": "unfriendly", "expected_seconds": 5.0},
                {"cow_i": "G0003", "cow_j": "G0004", "zone": "food", "layer": "friendly", "expected_seconds": 3.0},
                {"cow_i": "G0003", "cow_j": "G0004", "zone": "food", "layer": "unfriendly", "expected_seconds": 9.0},
                {"cow_i": "G0005", "cow_j": "G0006", "zone": "food", "layer": "friendly", "expected_seconds": 5.5},
                {"cow_i": "G0005", "cow_j": "G0006", "zone": "food", "layer": "unfriendly", "expected_seconds": 4.5},
                {"cow_i": "G0007", "cow_j": "G0008", "zone": "food", "layer": "friendly", "expected_seconds": 6.0},
                {"cow_i": "G0007", "cow_j": "G0008", "zone": "food", "layer": "unfriendly", "expected_seconds": 4.0},
            ]
        )

        networks = figures.net_dominance_networks(edge)
        friendly_pairs = set(zip(networks["friendly"]["cow_i"], networks["friendly"]["cow_j"]))
        unfriendly_pairs = set(zip(networks["unfriendly"]["cow_i"], networks["unfriendly"]["cow_j"]))
        self.assertEqual(friendly_pairs, {("G0001", "G0002"), ("G0007", "G0008")})
        self.assertEqual(unfriendly_pairs, {("G0003", "G0004")})
        self.assertTrue(friendly_pairs.isdisjoint(unfriendly_pairs))

        friendly_row = networks["friendly"].loc[
            (networks["friendly"]["cow_i"] == "G0001")
            & (networks["friendly"]["cow_j"] == "G0002")
        ].iloc[0]
        self.assertAlmostEqual(float(friendly_row["expected_seconds"]), 15.0)
        self.assertAlmostEqual(float(friendly_row["net_expected_seconds"]), 5.0)
        self.assertAlmostEqual(float(friendly_row["dominance"]), 1.0 / 3.0)

        threshold_row = networks["friendly"].loc[
            (networks["friendly"]["cow_i"] == "G0007")
            & (networks["friendly"]["cow_j"] == "G0008")
        ].iloc[0]
        self.assertAlmostEqual(float(threshold_row["dominance"]), 0.2)

    def test_net_dominance_networks_filter_zone_before_assignment(self) -> None:
        edge = pd.DataFrame(
            [
                {"cow_i": "G0001", "cow_j": "G0002", "zone": "food", "layer": "friendly", "expected_seconds": 6.0},
                {"cow_i": "G0001", "cow_j": "G0002", "zone": "food", "layer": "unfriendly", "expected_seconds": 4.0},
                {"cow_i": "G0001", "cow_j": "G0002", "zone": "rest", "layer": "friendly", "expected_seconds": 0.0},
                {"cow_i": "G0001", "cow_j": "G0002", "zone": "rest", "layer": "unfriendly", "expected_seconds": 10.0},
            ]
        )

        food = figures.net_dominance_networks(edge, "food")
        rest = figures.net_dominance_networks(edge, "rest")
        self.assertEqual(len(food["friendly"]), 1)
        self.assertTrue(food["unfriendly"].empty)
        self.assertTrue(rest["friendly"].empty)
        self.assertEqual(len(rest["unfriendly"]), 1)

    def test_drawable_network_cows_uses_only_positive_display_edges(self) -> None:
        networks = {
            "friendly": pd.DataFrame(
                [
                    {"cow_i": "G0001", "cow_j": "G0004", "expected_seconds": 8.0},
                    {"cow_i": "G0002", "cow_j": "G0003", "expected_seconds": 0.0},
                ]
            ),
            "unfriendly": pd.DataFrame(
                [{"cow_i": "G0001", "cow_j": "G0006", "expected_seconds": 10.0}]
            ),
        }
        self.assertEqual(figures.drawable_network_cows(networks), ["G0001", "G0004", "G0006"])

    def test_figure_05_relayouts_union_of_edges_drawn_in_both_panels(self) -> None:
        cows = [f"G{index:04d}" for index in range(1, 8)]
        data = {
            "sample_id": figures.COMBINED_SAMPLE_ID,
            "layout": pd.DataFrame({"cow_id": cows}),
            "edge": pd.DataFrame(
                [
                    {"cow_i": "G0001", "cow_j": "G0004", "zone": "food", "layer": "friendly", "expected_seconds": 8.0},
                    {"cow_i": "G0001", "cow_j": "G0004", "zone": "food", "layer": "unfriendly", "expected_seconds": 2.0},
                    {"cow_i": "G0001", "cow_j": "G0006", "zone": "food", "layer": "friendly", "expected_seconds": 1.0},
                    {"cow_i": "G0001", "cow_j": "G0006", "zone": "food", "layer": "unfriendly", "expected_seconds": 9.0},
                    {"cow_i": "G0002", "cow_j": "G0003", "zone": "food", "layer": "friendly", "expected_seconds": 5.5},
                    {"cow_i": "G0002", "cow_j": "G0003", "zone": "food", "layer": "unfriendly", "expected_seconds": 4.5},
                ]
            ),
        }
        draw_calls: list[dict[str, object]] = []

        def capture_draw(_ax, _network, positions, layer, _title, **kwargs):
            draw_calls.append(
                {
                    "layer": layer,
                    "positions": dict(positions),
                    "node_colors": dict(kwargs["node_colors"]),
                }
            )

        def close_and_return(fig, path):
            figures.plt.close(fig)
            return str(path)

        with tempfile.TemporaryDirectory(prefix="sna-figure-05-") as temporary:
            with patch.object(figures, "draw_network", side_effect=capture_draw):
                with patch.object(figures, "save_figure", side_effect=close_and_return):
                    paths = figures.plot_zone_networks(data, Path(temporary))

        self.assertEqual(set(paths), {"figure_05A_food_networks"})
        self.assertEqual([call["layer"] for call in draw_calls], ["friendly", "unfriendly"])
        expected_positions = figures.circle_positions(["G0001", "G0004", "G0006"])
        expected_colors = figures.cow_color_map(cows, seed=figures.COMBINED_SAMPLE_ID)
        for call in draw_calls:
            self.assertEqual(call["positions"], expected_positions)
            self.assertEqual(call["node_colors"], expected_colors)

    def test_figure_05_keeps_region_with_no_decisive_edges_and_no_nodes(self) -> None:
        data = {
            "sample_id": figures.COMBINED_SAMPLE_ID,
            "layout": pd.DataFrame({"cow_id": ["G0001", "G0002"]}),
            "edge": pd.DataFrame(
                [
                    {"cow_i": "G0001", "cow_j": "G0002", "zone": "food", "layer": "friendly", "expected_seconds": 5.5},
                    {"cow_i": "G0001", "cow_j": "G0002", "zone": "food", "layer": "unfriendly", "expected_seconds": 4.5},
                ]
            ),
        }
        positions_seen: list[dict[str, tuple[float, float]]] = []

        def capture_draw(_ax, network, positions, _layer, _title, **_kwargs):
            self.assertTrue(network.empty)
            positions_seen.append(dict(positions))

        def close_and_return(fig, path):
            figures.plt.close(fig)
            return str(path)

        with tempfile.TemporaryDirectory(prefix="sna-figure-05-empty-") as temporary:
            with patch.object(figures, "draw_network", side_effect=capture_draw):
                with patch.object(figures, "save_figure", side_effect=close_and_return):
                    paths = figures.plot_zone_networks(data, Path(temporary))

        self.assertEqual(set(paths), {"figure_05A_food_networks"})
        self.assertEqual(positions_seen, [{}, {}])

    def test_dynamic_figure_suite_validation_rejects_missing_and_extra_files(self) -> None:
        expected = {"figures/figure_05A_food_networks.png"}
        with tempfile.TemporaryDirectory(prefix="sna-figure-contract-") as temporary:
            sample_dir = Path(temporary)
            figure_dir = sample_dir / "figures"
            figure_dir.mkdir()
            expected_file = figure_dir / "figure_05A_food_networks.png"
            figures.Image.new("RGB", (2, 2), color="white").save(expected_file)
            figures.validate_figure_suite(sample_dir, expected)

            extra_file = figure_dir / "figure_05_food_networks.png"
            figures.Image.new("RGB", (2, 2), color="white").save(extra_file)
            with self.assertRaisesRegex(RuntimeError, "extra"):
                figures.validate_figure_suite(sample_dir, expected)
            extra_file.unlink()
            expected_file.unlink()
            with self.assertRaisesRegex(RuntimeError, "missing"):
                figures.validate_figure_suite(sample_dir, expected)

    def test_draw_network_uses_dominance_for_edge_alpha(self) -> None:
        network = pd.DataFrame(
            [
                {
                    "cow_i": "G0001",
                    "cow_j": "G0002",
                    "expected_seconds": 15.0,
                    "dominance": 1.0 / 3.0,
                }
            ]
        )
        fig, ax = figures.plt.subplots()
        try:
            figures.draw_network(
                ax,
                network,
                {"G0001": (0.0, 0.0), "G0002": (1.0, 1.0)},
                "friendly",
                "friendly",
                max_weight=15.0,
                alpha_column="dominance",
            )
            self.assertEqual(len(ax.lines), 1)
            self.assertAlmostEqual(float(ax.lines[0].get_alpha()), 1.0 / 3.0)
            self.assertAlmostEqual(float(ax.lines[0].get_linewidth()), 6.0)
        finally:
            figures.plt.close(fig)

    def test_figure_06_net_adjacency_assigns_each_dyad_to_one_panel(self) -> None:
        edge = pd.DataFrame(
            [
                {"cow_i": "G0001", "cow_j": "G0002", "zone": "food", "layer": "friendly", "expected_seconds": 6.0},
                {"cow_i": "G0001", "cow_j": "G0002", "zone": "food", "layer": "unfriendly", "expected_seconds": 4.0},
                {"cow_i": "G0003", "cow_j": "G0004", "zone": "food", "layer": "friendly", "expected_seconds": 5.5},
                {"cow_i": "G0003", "cow_j": "G0004", "zone": "food", "layer": "unfriendly", "expected_seconds": 4.5},
                {"cow_i": "G0005", "cow_j": "G0006", "zone": "food", "layer": "friendly", "expected_seconds": 3.0},
                {"cow_i": "G0005", "cow_j": "G0006", "zone": "food", "layer": "unfriendly", "expected_seconds": 9.0},
            ]
        )
        cows = [f"G{index:04d}" for index in range(1, 7)]
        matrices = figures.net_adjacency_matrices(edge, cows)
        friendly = matrices["friendly"]
        unfriendly = matrices["unfriendly"]

        self.assertAlmostEqual(float(friendly.loc["G0001", "G0002"]), 2.0)
        self.assertAlmostEqual(float(unfriendly.loc["G0001", "G0002"]), 0.0)
        self.assertAlmostEqual(float(friendly.loc["G0003", "G0004"]), 0.0)
        self.assertAlmostEqual(float(unfriendly.loc["G0003", "G0004"]), 0.0)
        self.assertAlmostEqual(float(friendly.loc["G0005", "G0006"]), 0.0)
        self.assertAlmostEqual(float(unfriendly.loc["G0005", "G0006"]), 6.0)
        self.assertTrue(np.allclose(friendly.to_numpy(dtype=float), friendly.to_numpy(dtype=float).T))
        self.assertTrue(np.allclose(unfriendly.to_numpy(dtype=float), unfriendly.to_numpy(dtype=float).T))
        self.assertTrue(np.allclose(np.diag(friendly.to_numpy(dtype=float)), 0.0))
        self.assertTrue(np.allclose(np.diag(unfriendly.to_numpy(dtype=float)), 0.0))
        self.assertFalse(
            bool(((friendly > 0.0) & (unfriendly > 0.0)).to_numpy(dtype=bool).any())
        )
        self.assertAlmostEqual(figures.adjacency_heatmap_vmax(matrices), 6.0)

    def test_figure_06_net_adjacency_aggregates_all_zones_before_dominance(self) -> None:
        edge = pd.DataFrame(
            [
                {"cow_i": "G0001", "cow_j": "G0002", "zone": "food", "layer": "friendly", "expected_seconds": 6.0},
                {"cow_i": "G0001", "cow_j": "G0002", "zone": "food", "layer": "unfriendly", "expected_seconds": 4.0},
                {"cow_i": "G0001", "cow_j": "G0002", "zone": "rest", "layer": "friendly", "expected_seconds": 0.0},
                {"cow_i": "G0001", "cow_j": "G0002", "zone": "rest", "layer": "unfriendly", "expected_seconds": 10.0},
            ]
        )

        matrices = figures.net_adjacency_matrices(edge, ["G0001", "G0002"])
        self.assertAlmostEqual(float(matrices["friendly"].loc["G0001", "G0002"]), 0.0)
        self.assertAlmostEqual(float(matrices["unfriendly"].loc["G0001", "G0002"]), 8.0)

    def test_community_grid_is_complete_and_preserves_assignment_gaps(self) -> None:
        data = community_data()
        axis, matrix = figures.community_plot_matrix(data)
        self.assertEqual(axis.shape, (92, 3))
        self.assertEqual(matrix.shape, (62, 92))
        self.assertEqual(float(axis.iloc[-1]["window_start_s"]), 27_300.0)
        self.assertEqual(float(axis.iloc[-1]["window_end_s"]), 27_600.0)
        self.assertEqual(matrix.loc["G0001", 0], matrix.loc["G0001", 1])
        self.assertTrue(math.isnan(matrix.loc["G0001", 2]))
        self.assertNotEqual(matrix.loc["G0001", 1], matrix.loc["G0001", 3])
        self.assertTrue(matrix.loc[:, 2].isna().all())

    def test_community_axis_rejects_missing_adjacent_pair(self) -> None:
        data = community_data()
        data["community_summary"] = data["community_summary"].iloc[:-1].copy()
        with self.assertRaisesRegex(ValueError, "complete adjacent-window sequence"):
            figures.complete_community_window_axis(data)

    def test_figure_01_does_not_require_or_draw_interaction_edges(self) -> None:
        cows = [f"G{index:04d}" for index in range(1, figures.EXPECTED_GLOBAL_IDENTITY_COUNT + 1)]
        cow_frame = pd.DataFrame(
            {
                "cow_id": cows,
                "frame": np.arange(len(cows), dtype=np.int64),
                "time_s": np.arange(len(cows), dtype=float),
                "dt_s": np.full(len(cows), 1.0 / 30.0),
                "anchor_x": np.linspace(100.0, 3700.0, len(cows)),
                "anchor_y": np.linspace(100.0, 2000.0, len(cows)),
                "visible_flag": np.ones(len(cows), dtype=np.int8),
                figures.TRAJECTORY_SEGMENT_COLUMN: np.arange(len(cows), dtype=np.int64),
            }
        )
        layout = pd.DataFrame(
            {
                "cow_id": cows,
                "median_floorplan_x": cow_frame["anchor_x"],
                "median_floorplan_y": cow_frame["anchor_y"],
            }
        )
        data = {
            "sample_id": figures.COMBINED_SAMPLE_ID,
            "zones": {"zones": []},
            "cow_frame": cow_frame,
            "layout": layout,
        }
        with tempfile.TemporaryDirectory(prefix="sna-figure-01-") as temporary:
            with patch.object(figures, "draw_structure_polygons", return_value=None):
                paths = figures.plot_floorplan_trajectories(data, Path(temporary))
            self.assertEqual(len(paths), 7)
            self.assertTrue(all(Path(path).is_file() for path in paths.values()))

    def test_figure_01_trajectory_segments_break_unsafe_connections(self) -> None:
        red_square = np.asarray(
            [[40.0, 40.0], [60.0, 40.0], [60.0, 60.0], [40.0, 60.0]],
            dtype=float,
        )

        cases = (
            (
                "non_adjacent_frame",
                [10, 12],
                [0.0, 1.0 / 30.0],
                [(0.0, 0.0), (1.0, 0.0)],
                [],
                128.0,
                [[10], [12]],
            ),
            (
                "time_gap",
                [10, 11],
                [0.0, 1.0 / 15.0],
                [(0.0, 0.0), (1.0, 0.0)],
                [],
                128.0,
                [[10], [11]],
            ),
            (
                "abnormal_displacement",
                [10, 11],
                [0.0, 1.0 / 30.0],
                [(0.0, 0.0), (11.0, 0.0)],
                [],
                10.0,
                [[10], [11]],
            ),
            (
                "red_intersection",
                [10, 11],
                [0.0, 1.0 / 30.0],
                [(0.0, 50.0), (100.0, 50.0)],
                [red_square],
                128.0,
                [[10], [11]],
            ),
            (
                "safe_at_displacement_limit",
                [10, 11],
                [0.0, 1.0 / 30.0],
                [(0.0, 0.0), (3.0, 4.0)],
                [red_square],
                5.0,
                [[10, 11]],
            ),
        )
        for name, frames, times, points, red_polygons, threshold, expected in cases:
            with self.subTest(name=name):
                group = pd.DataFrame(
                    {
                        "frame": frames,
                        "time_s": times,
                        "dt_s": np.full(len(frames), 1.0 / 30.0),
                        "anchor_x": [point[0] for point in points],
                        "anchor_y": [point[1] for point in points],
                        figures.TRAJECTORY_SEGMENT_COLUMN: np.zeros(len(frames), dtype=np.int64),
                    }
                )
                segments = figures.trajectory_plot_segments(
                    group,
                    red_polygons,
                    max_displacement_px=threshold,
                )
                self.assertEqual([segment["frame"].tolist() for segment in segments], expected)

    def test_figure_01_draws_samples_when_all_connections_break(self) -> None:
        group = pd.DataFrame(
            {
                "frame": [10, 12, 14],
                "time_s": [0.0, 2.0 / 30.0, 4.0 / 30.0],
                "dt_s": np.full(3, 1.0 / 30.0),
                "anchor_x": [10.0, 20.0, 30.0],
                "anchor_y": [40.0, 50.0, 60.0],
                figures.TRAJECTORY_SEGMENT_COLUMN: np.zeros(3, dtype=np.int64),
            }
        )
        fig, ax = figures.plt.subplots()
        try:
            figures.draw_trajectory_group(ax, group, "#123456", [])
            self.assertEqual(len(ax.lines), 0)
            self.assertEqual(len(ax.collections), 1)
            np.testing.assert_allclose(
                ax.collections[0].get_offsets(),
                group[["anchor_x", "anchor_y"]].to_numpy(dtype=float),
            )
        finally:
            figures.plt.close(fig)

    def test_figure_08_has_one_axes_numeric_labels_and_missing_legend(self) -> None:
        data = community_data()
        captured: dict[str, object] = {}

        def capture(fig, path):
            captured["fig"] = fig
            captured["path"] = path
            return str(path)

        with tempfile.TemporaryDirectory(prefix="sna-figure-08-") as temporary:
            with patch.object(figures, "save_figure", side_effect=capture):
                figures.plot_community_stability(data, Path(temporary))
        fig = captured["fig"]
        try:
            self.assertEqual(len(fig.axes), 1)
            ax = fig.axes[0]
            self.assertEqual(ax.get_yticklabels()[0].get_text(), "1")
            self.assertEqual(ax.get_yticklabels()[-1].get_text(), "62")
            legend = ax.get_legend()
            self.assertIsNotNone(legend)
            self.assertIn("Unavailable", legend.get_texts()[0].get_text())
        finally:
            figures.plt.close(fig)


if __name__ == "__main__":
    unittest.main()
