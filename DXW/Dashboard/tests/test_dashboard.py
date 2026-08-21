from __future__ import annotations

import hashlib
import io
import json
import random
import re
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from urllib.parse import parse_qsl, urlsplit

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from PIL import Image

import dashboard_plotting
from appearance_data import _canonical_sample_for_frame
from dashboard_data import (
    DashboardData,
    DataContractError,
    FIGURE_06_EXTREME_CELL_COUNT,
    ROOT,
    TRAJECTORY_CACHE,
    TRAJECTORY_CACHE_META,
    TRAJECTORY_SAMPLER_CONTRACT,
    _file_sha256,
    validate_edge_contract,
)
from dashboard_plotting import FigureRenderer
from server import DashboardHandler, build_argument_parser


EXPECTED_GENERATION_ID = "99f9311a-6c60-4b1f-a74a-f9f7fa2d2ce3"
EXPECTED_FIGURES = [
    "01",
    "02",
    "03",
    "04A",
    "04B",
    "05A",
    "05B",
    "05C",
    "05D",
    "05E",
    "05F",
    "06",
    "07",
    "08",
    "09",
    "10",
]
EXPECTED_FIGURE_02_HIGH = {"G0004", "G0035", "G0046", "G0001", "G0020"}
EXPECTED_FIGURE_02_LOW = {"G0057", "G0033", "G0009", "G0027", "G0010"}
EXPECTED_FIGURE_09_HIGH = {"G0022", "G0002", "G0035", "G0024", "G0058"}
EXPECTED_FIGURE_09_LOW = {"G0017", "G0060", "G0055", "G0046", "G0041"}
EXPECTED_FIGURE_04 = {
    "G0002",
    "G0004",
    "G0005",
    "G0008",
    "G0013",
    "G0014",
    "G0021",
    "G0022",
    "G0029",
    "G0031",
    "G0032",
    "G0034",
    "G0035",
    "G0038",
    "G0039",
    "G0044",
    "G0045",
    "G0049",
    "G0052",
    "G0056",
}
EXPECTED_FIGURE_06_HIGH_CELLS = [
    ("unfriendly", "G0047", "G0055"),
    ("unfriendly", "G0008", "G0041"),
    ("unfriendly", "G0018", "G0055"),
    ("friendly", "G0006", "G0023"),
    ("unfriendly", "G0017", "G0046"),
    ("friendly", "G0021", "G0037"),
    ("unfriendly", "G0018", "G0041"),
]
EXPECTED_FIGURE_06_LOW_CELLS = [
    ("friendly", "G0008", "G0054"),
    ("friendly", "G0016", "G0028"),
    ("friendly", "G0025", "G0054"),
    ("unfriendly", "G0026", "G0028"),
    ("unfriendly", "G0017", "G0032"),
    ("unfriendly", "G0001", "G0035"),
    ("unfriendly", "G0052", "G0054"),
]


class DashboardDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data = DashboardData()

    def test_generation_and_identity_contract(self) -> None:
        self.assertEqual(self.data.generation_id, EXPECTED_GENERATION_ID)
        self.assertEqual(self.data.all_cows, [f"G{index:04d}" for index in range(1, 63)])
        catalog = self.data.cattle_catalog()
        self.assertEqual(len(catalog), 62)
        self.assertEqual(catalog[0]["label"], "1")
        self.assertEqual(catalog[-1]["label"], "62")
        self.assertEqual(len({item["uuid"] for item in catalog}), 62)
        colors = {item["color"] for item in catalog}
        self.assertEqual(len(colors), 20)
        self.assertTrue(all(re.fullmatch(r"#[0-9a-f]{6}", color) for color in colors))

    def test_cow_appearance_payload_contract(self) -> None:
        manifest = json.loads(
            self.data.paths["input_generation"].read_text(encoding="utf-8")
        )
        camera_match = re.fullmatch(r"Gopro(\d+)", str(manifest["camera"]))
        self.assertIsNotNone(camera_match)
        expected_fields = {
            "clip",
            "segment",
            "startFrame",
            "endFrameExclusive",
            "frameCount",
            "startTime",
            "endTime",
            "display",
            "href",
        }
        for cow_id in self.data.all_cows:
            with self.subTest(cow_id=cow_id):
                appearances = self.data.appearances_for_cow(cow_id)
                self.assertIsInstance(appearances, list)
                self.assertTrue(appearances)
                self.assertEqual(
                    [item["frameCount"] for item in appearances],
                    sorted(
                        (item["frameCount"] for item in appearances),
                        reverse=True,
                    ),
                )
                seen_intervals: set[tuple[str, int, int]] = set()
                for item in appearances:
                    self.assertEqual(set(item), expected_fields)
                    self.assertRegex(item["clip"], r"^GX\d{6}$")
                    self.assertIsInstance(item["segment"], int)
                    self.assertIsInstance(item["startFrame"], int)
                    self.assertIsInstance(item["endFrameExclusive"], int)
                    self.assertIsInstance(item["frameCount"], int)
                    self.assertGreaterEqual(item["segment"], 1)
                    self.assertGreaterEqual(item["startFrame"], 0)
                    self.assertGreater(
                        item["endFrameExclusive"],
                        item["startFrame"],
                    )
                    self.assertEqual(
                        item["frameCount"],
                        item["endFrameExclusive"] - item["startFrame"],
                    )
                    for field in ("startTime", "endTime"):
                        self.assertRegex(
                            item[field],
                            r"^(?:[01]\d|2[0-3]):[0-5]\d:[0-5]\d$",
                        )
                    self.assertEqual(
                        item["display"],
                        f"{item['startTime']}-{item['endTime']}",
                    )
                    matching_samples = [
                        sample
                        for sample in manifest["samples"]
                        if sample["source_clip_id"] == item["clip"]
                        and sample["canonical_start_frame"]
                        <= item["startFrame"]
                        <= sample["canonical_end_frame"]
                    ]
                    self.assertEqual(len(matching_samples), 1)
                    sample = matching_samples[0]
                    self.assertEqual(item["segment"], sample["shard_index"])
                    parsed_href = urlsplit(item["href"])
                    self.assertEqual(parsed_href.scheme, "http")
                    self.assertEqual(parsed_href.netloc, "172.17.6.39:9922")
                    self.assertEqual(parsed_href.path, "/")
                    self.assertEqual(parsed_href.fragment, "")
                    self.assertIsNone(parsed_href.username)
                    self.assertIsNone(parsed_href.password)
                    self.assertEqual(
                        parse_qsl(
                            parsed_href.query,
                            keep_blank_values=True,
                            strict_parsing=True,
                        ),
                        [
                            ("farmID", str(int(manifest["farm"]))),
                            ("cameraID", camera_match.group(1)),
                            ("clipID", sample["sample_id"]),
                            (
                                "segmentID",
                                f"{item['clip']}-{item['segment']}-{item['startFrame']}",
                            ),
                            ("frameID", str(item["startFrame"])),
                        ],
                    )
                    interval_key = (
                        item["clip"],
                        item["startFrame"],
                        item["endFrameExclusive"],
                    )
                    self.assertNotIn(interval_key, seen_intervals)
                    seen_intervals.add(interval_key)

        first_g0001 = self.data.appearances_for_cow("G0001")[0]
        self.assertEqual(first_g0001["clip"], "GX010006")
        self.assertEqual(first_g0001["segment"], 1)
        self.assertEqual(first_g0001["startFrame"], 0)
        self.assertEqual(first_g0001["endFrameExclusive"], 33138)
        self.assertEqual(first_g0001["frameCount"], 33138)
        self.assertEqual(first_g0001["startTime"], "10:20:54")
        self.assertEqual(first_g0001["endTime"], "10:39:20")
        self.assertEqual(first_g0001["display"], "10:20:54-10:39:20")
        self.assertEqual(
            first_g0001["href"],
            "http://172.17.6.39:9922/?farmID=1&cameraID=1&"
            "clipID=F1_Gopro1_GX010006_1&segmentID=GX010006-1-0&frameID=0",
        )

        second_segment_g0001 = next(
            item
            for item in self.data.appearances_for_cow("G0001")
            if item["clip"] == "GX010006" and item["startFrame"] == 82066
        )
        self.assertEqual(second_segment_g0001["segment"], 2)
        self.assertEqual(
            second_segment_g0001["display"],
            "11:06:32-11:10:01",
        )
        self.assertEqual(
            second_segment_g0001["href"],
            "http://172.17.6.39:9922/?farmID=1&cameraID=1&"
            "clipID=F1_Gopro1_GX010006_2&"
            "segmentID=GX010006-2-82066&frameID=82066",
        )

        cross_segment_g0005 = next(
            item
            for item in self.data.appearances_for_cow("G0005")
            if item["clip"] == "GX010006" and item["startFrame"] == 42199
        )
        self.assertEqual(cross_segment_g0005["segment"], 1)
        self.assertEqual(cross_segment_g0005["endFrameExclusive"], 44400)
        self.assertEqual(
            cross_segment_g0005["href"],
            "http://172.17.6.39:9922/?farmID=1&cameraID=1&"
            "clipID=F1_Gopro1_GX010006_1&"
            "segmentID=GX010006-1-42199&frameID=42199",
        )

    def test_canonical_appearance_sample_resolution_is_strict(self) -> None:
        sample_map = {
            "sample-1": {"clip": "GX000001", "start": 0, "end": 99},
            "sample-2": {"clip": "GX000001", "start": 100, "end": 199},
        }
        sample_id, _ = _canonical_sample_for_frame("GX000001", 100, sample_map)
        self.assertEqual(sample_id, "sample-2")

        with self.assertRaises(DataContractError):
            _canonical_sample_for_frame("GX000001", 200, sample_map)

        overlapping = dict(sample_map)
        overlapping["sample-overlap"] = {
            "clip": "GX000001",
            "start": 90,
            "end": 110,
        }
        with self.assertRaises(DataContractError):
            _canonical_sample_for_frame("GX000001", 100, overlapping)

    def test_cow_photo_catalog_contract(self) -> None:
        photos = [self.data.cow_photo_for_cow(cow_id) for cow_id in self.data.all_cows]
        self.assertEqual(len(photos), 62)
        self.assertEqual([photo.cow_id for photo in photos], self.data.all_cows)
        self.assertEqual(len({photo.path for photo in photos}), 62)

        photo_directories = {photo.path.parent for photo in photos}
        self.assertEqual(len(photo_directories), 1)
        photo_directory = next(iter(photo_directories))
        self.assertEqual(
            sorted(photo_directory.glob("*.jpg")),
            sorted(photo.path for photo in photos),
        )

        for cow_id, photo in zip(self.data.all_cows, photos, strict=True):
            with self.subTest(cow_id=cow_id):
                first_appearance = self.data.appearances_for_cow(cow_id)[0]
                self.assertEqual(photo.path.name, f"{cow_id}.jpg")
                self.assertEqual(photo.clip, first_appearance["clip"])
                self.assertEqual(photo.local_frame, first_appearance["startFrame"])
                self.assertEqual(photo.width, 640)
                self.assertEqual(photo.height, 360)
                self.assertEqual(photo.mime_type, "image/jpeg")
                self.assertRegex(photo.sha256, r"^[0-9a-f]{64}$")
                self.assertTrue(photo.path.is_file())

                image_bytes = photo.path.read_bytes()
                self.assertEqual(hashlib.sha256(image_bytes).hexdigest(), photo.sha256)
                self.assertEqual(image_bytes[:2], b"\xff\xd8")
                self.assertEqual(image_bytes[-2:], b"\xff\xd9")
                with Image.open(io.BytesIO(image_bytes)) as image:
                    self.assertEqual(image.format, "JPEG")
                    self.assertEqual(image.size, (640, 360))
                    image.verify()

    def test_figure_catalog_excludes_08b(self) -> None:
        definitions = self.data.figure_definitions(random.Random(4))
        keys = [item["key"] for item in definitions]
        self.assertEqual(keys, EXPECTED_FIGURES)
        self.assertNotIn("08B", keys)
        self.assertEqual(
            [item["zone"] for item in definitions if item["key"].startswith("05")],
            ["cross_zone", "food", "path", "rest", "wait_for_water", "water"],
        )
        modes = {item["key"]: item["cattleMode"] for item in definitions}
        self.assertEqual(modes["03"], "none")
        self.assertEqual(modes["10"], "none")
        self.assertTrue(all(modes[key] == "filter" for key in keys if key not in {"03", "10"}))

    def test_initial_selection_contract(self) -> None:
        selections = self.data.initial_selections(random.Random(7))
        self.assertEqual(selections["01"], [f"G{index:04d}" for index in range(1, 6)])
        self.assertEqual(len(selections["02"]), 15)
        self.assertTrue(EXPECTED_FIGURE_02_HIGH <= set(selections["02"]))
        self.assertTrue(EXPECTED_FIGURE_02_LOW <= set(selections["02"]))
        self.assertEqual(set(selections["04A"]), EXPECTED_FIGURE_04)
        self.assertEqual(selections["04A"], selections["04B"])
        self.assertEqual(len(selections["06"]), 20)
        self.assertEqual(len(selections["07"]), 10)
        self.assertEqual(len(selections["08"]), 20)
        self.assertEqual(len(selections["09"]), 15)
        self.assertTrue(EXPECTED_FIGURE_09_HIGH <= set(selections["09"]))
        self.assertTrue(EXPECTED_FIGURE_09_LOW <= set(selections["09"]))
        expected_figure_05_counts = {
            "05A": 17,
            "05B": 20,
            "05C": 20,
            "05D": 20,
            "05E": 20,
            "05F": 20,
        }
        for key, count in expected_figure_05_counts.items():
            self.assertEqual(len(selections[key]), count)
            zone = self.data.figure_zone[key]
            candidates = set(self.data.zone_candidates[zone])
            self.assertTrue(set(selections[key]) <= candidates)
            if len(candidates) <= 20:
                self.assertEqual(set(selections[key]), candidates)
        self.assertEqual(
            {
                zone: len(candidates)
                for zone, candidates in self.data.zone_candidates.items()
            },
            {
                "cross_zone": 17,
                "food": 23,
                "path": 24,
                "rest": 21,
                "wait_for_water": 60,
                "water": 36,
            },
        )
        for key, values in selections.items():
            self.assertEqual(len(values), len(set(values)), key)
            self.assertTrue(set(values) <= set(self.data.all_cows), key)

    def test_figure_06_extreme_cell_default(self) -> None:
        high_cells = self.data.figure_06_high_cells
        low_cells = self.data.figure_06_low_cells
        self.assertEqual(len(high_cells), FIGURE_06_EXTREME_CELL_COUNT)
        self.assertEqual(len(low_cells), FIGURE_06_EXTREME_CELL_COUNT)
        self.assertEqual(
            [(layer, cow_i, cow_j) for _, layer, cow_i, cow_j in high_cells],
            EXPECTED_FIGURE_06_HIGH_CELLS,
        )
        self.assertEqual(
            [(layer, cow_i, cow_j) for _, layer, cow_i, cow_j in low_cells],
            EXPECTED_FIGURE_06_LOW_CELLS,
        )

        selected = set(self.data.figure_06_default)
        expected_selected = {
            cow
            for _, cow_i, cow_j in (
                EXPECTED_FIGURE_06_HIGH_CELLS + EXPECTED_FIGURE_06_LOW_CELLS
            )
            for cow in (cow_i, cow_j)
        }
        self.assertEqual(selected, expected_selected)
        self.assertEqual(len(selected), 20)

        for score, layer, cow_i, cow_j in high_cells + low_cells:
            self.assertGreater(score, 0.0)
            self.assertAlmostEqual(
                float(self.data.full_matrices[layer].loc[cow_i, cow_j]),
                score,
                places=12,
            )
            self.assertAlmostEqual(
                float(self.data.full_matrices[layer].loc[cow_j, cow_i]),
                score,
                places=12,
            )
            self.assertIn(cow_i, selected)
            self.assertIn(cow_j, selected)

        first = self.data.initial_selections(random.Random(1))["06"]
        second = self.data.initial_selections(random.Random(999))["06"]
        self.assertEqual(first, self.data.figure_06_default)
        self.assertEqual(second, self.data.figure_06_default)

    def test_time_budget_and_isolation_formulas(self) -> None:
        budget = self.data.data["time_budget"]
        zone_columns = [
            "time_food_s",
            "time_path_s",
            "time_rest_s",
            "time_wait_for_water_s",
            "time_water_s",
        ]
        zone_sum = budget[zone_columns].apply(pd.to_numeric, errors="raise").sum(axis=1)
        visible = pd.to_numeric(budget["visible_time_s"], errors="raise")
        np.testing.assert_allclose(zone_sum, visible, rtol=1e-10, atol=1e-8)

        node = self.data.data["node"]
        expected_score = (
            0.4 * pd.to_numeric(node["alone_fraction"], errors="raise")
            + 0.3 * pd.to_numeric(node["low_friendly_sociality"], errors="raise")
            + 0.3 * pd.to_numeric(node["low_partner_diversity"], errors="raise")
        )
        actual_score = pd.to_numeric(node["isolation_score"], errors="raise")
        np.testing.assert_allclose(expected_score, actual_score, rtol=1e-12, atol=1e-12)

    def test_network_and_community_scale_contract(self) -> None:
        self.assertEqual(len(self.data.data["edge"]), 2152)
        self.assertAlmostEqual(self.data.full_network_max, 452.242416791892, places=9)
        self.assertAlmostEqual(self.data.full_matrix_vmax, 121.378096088217, places=9)
        self.assertEqual(len(self.data.community_axis), 92)
        self.assertEqual(list(self.data.community_axis["window_index"]), list(range(92)))
        self.assertEqual(self.data.community_matrix.shape, (62, 92))
        self.assertEqual(len(self.data.data["community_summary"]), 91)
        summary_pairs = list(
            self.data.data["community_summary"][["window_a", "window_b"]]
            .astype(int)
            .itertuples(index=False, name=None)
        )
        self.assertEqual(summary_pairs, [(index, index + 1) for index in range(91)])
        assigned_windows = set(
            pd.to_numeric(
                self.data.data["community_windows"]["window_index"],
                errors="raise",
            ).astype(int)
        )
        expected_empty = {
            13,
            14,
            15,
            16,
            17,
            18,
            24,
            25,
            26,
            27,
            29,
            30,
            52,
            54,
            58,
            60,
            61,
            62,
            63,
            64,
            66,
            67,
            68,
            69,
            70,
            71,
            72,
            73,
            74,
            75,
            76,
            81,
        }
        self.assertEqual(set(range(92)) - assigned_windows, expected_empty)
        self.assertTrue(
            self.data.community_matrix.loc[:, sorted(expected_empty)].isna().all().all()
        )

    def test_expected_figure_07_missing_values(self) -> None:
        node = self.data.data["node"].set_index("cow_id")
        missing = set(node.index[node["community_stability"].isna()])
        self.assertEqual(missing, {"G0009", "G0024"})

    def test_edge_validator_rejects_contract_mutations(self) -> None:
        expected = set(self.data.all_cows)
        edge = self.data.data["edge"].copy()
        invalid_layer = edge.copy()
        invalid_layer.loc[invalid_layer.index[0], "layer"] = "other"
        with self.assertRaises(DataContractError):
            validate_edge_contract(invalid_layer, expected)

        invalid_weight = edge.copy()
        invalid_weight.loc[invalid_weight.index[0], "expected_seconds"] = -1.0
        with self.assertRaises(DataContractError):
            validate_edge_contract(invalid_weight, expected)

        invalid_order = edge.copy()
        first_index = invalid_order.index[0]
        left = invalid_order.loc[first_index, "cow_i"]
        invalid_order.loc[first_index, "cow_i"] = invalid_order.loc[first_index, "cow_j"]
        invalid_order.loc[first_index, "cow_j"] = left
        with self.assertRaises(DataContractError):
            validate_edge_contract(invalid_order, expected)

    def test_trajectory_cache_integrity_contract(self) -> None:
        metadata = json.loads(TRAJECTORY_CACHE_META.read_text(encoding="utf-8"))
        frame = self.data.cow_frame()
        self.assertEqual(metadata["sampler_contract"], TRAJECTORY_SAMPLER_CONTRACT)
        self.assertEqual(metadata["row_count"], len(frame))
        self.assertEqual(metadata["cache_sha256"], _file_sha256(TRAJECTORY_CACHE))
        counts = frame["cow_id"].value_counts()
        self.assertEqual(len(counts), 62)
        self.assertTrue((counts <= 5000).all())
        self.assertTrue(np.isfinite(frame[["time_s", "dt_s", "anchor_x", "anchor_y"]]).all().all())
        self.assertTrue((pd.to_numeric(frame["dt_s"], errors="raise") > 0).all())


class FigureRendererTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data = DashboardData()
        cls.renderer = FigureRenderer(cls.data, max_cache_entries=24)
        cls.selections = cls.data.initial_selections(random.Random(12))

    def assert_png(self, image: bytes) -> None:
        self.assertGreater(len(image), 10_000)
        self.assertEqual(image[:8], b"\x89PNG\r\n\x1a\n")

    def test_non_trajectory_figures_render(self) -> None:
        for key in EXPECTED_FIGURES:
            if key == "01":
                continue
            with self.subTest(figure=key):
                image = self.renderer.render(key, self.selections[key])
                self.assert_png(image)

    def test_empty_selections_render(self) -> None:
        for key in ("02", "04A", "05A", "06", "07", "08", "09"):
            with self.subTest(figure=key):
                self.assert_png(self.renderer.render(key, []))

    def test_figure_06_default_renders_all_extreme_cells(self) -> None:
        imshow_calls: list[tuple[np.ndarray, object]] = []
        original_imshow = Axes.imshow

        def capture_imshow(axis, values, *args, **kwargs):
            imshow_calls.append(
                (np.asarray(values, dtype=float).copy(), kwargs.get("vmax"))
            )
            return original_imshow(axis, values, *args, **kwargs)

        with patch.object(Axes, "imshow", new=capture_imshow):
            figure = self.renderer._figure_06(self.data.figure_06_default)
            dashboard_plotting.source.plt.close(figure)

        self.assertEqual(len(imshow_calls), 2)
        self.assertEqual(
            [values.shape for values, _ in imshow_calls],
            [(20, 20), (20, 20)],
        )
        self.assertEqual(
            [vmax for _, vmax in imshow_calls],
            [self.data.full_matrix_vmax, self.data.full_matrix_vmax],
        )

        selected = set(self.data.figure_06_default)
        ordered = [
            cow
            for cow in self.data.matrix_cow_order
            if cow in selected
        ]
        position = {cow: index for index, cow in enumerate(ordered)}
        layer_index = {"friendly": 0, "unfriendly": 1}
        for score, layer, cow_i, cow_j in (
            self.data.figure_06_high_cells + self.data.figure_06_low_cells
        ):
            plotted = imshow_calls[layer_index[layer]][0]
            self.assertAlmostEqual(
                float(plotted[position[cow_i], position[cow_j]]),
                score,
                places=12,
            )
            self.assertAlmostEqual(
                float(plotted[position[cow_j], position[cow_i]]),
                score,
                places=12,
            )

    def test_render_cache_returns_same_bytes(self) -> None:
        first = self.renderer.render("04A", self.selections["04A"])
        second = self.renderer.render("04A", list(reversed(self.selections["04A"])))
        self.assertIs(first, second)

    def test_interactive_figure_hit_region_contract(self) -> None:
        selected = ["G0001", "G0002"]
        expected = {
            "01": {"ellipse": 2},
            "02": {"rect": 2},
            "04A": {"ellipse": 2},
            "04B": {"ellipse": 2},
            "05A": {"ellipse": 4},
            "07": {"ellipse": 2, "rect": 2},
            "08": {"rect": 2},
            "09": {"rect": 2},
        }
        for figure_key, expected_shapes in expected.items():
            with self.subTest(figure=figure_key):
                result = self.renderer.render_result(figure_key, selected)
                self.assert_png(result.image)
                self.assertGreater(result.width, 0)
                self.assertGreater(result.height, 0)
                self.assertEqual(
                    result.width,
                    int.from_bytes(result.image[16:20], "big"),
                )
                self.assertEqual(
                    result.height,
                    int.from_bytes(result.image[20:24], "big"),
                )
                actual_shapes = {
                    shape: sum(region["shape"] == shape for region in result.regions)
                    for shape in {region["shape"] for region in result.regions}
                }
                self.assertEqual(actual_shapes, expected_shapes)
                self.assertEqual(
                    {str(region["cowId"]) for region in result.regions},
                    set(selected),
                )
                for region in result.regions:
                    coordinate_names = (
                        ("cx", "cy", "rx", "ry")
                        if region["shape"] == "ellipse"
                        else ("x0", "y0", "x1", "y1")
                    )
                    for name in coordinate_names:
                        self.assertGreaterEqual(float(region[name]), 0.0)
                        self.assertLessEqual(float(region[name]), 1.0)
                    if region["shape"] == "ellipse":
                        self.assertGreater(float(region["rx"]), 0.0)
                        self.assertGreater(float(region["ry"]), 0.0)
                    else:
                        self.assertLess(float(region["x0"]), float(region["x1"]))
                        self.assertLess(float(region["y0"]), float(region["y1"]))

        figure_07 = self.renderer.render_result("07", selected)
        nearest = [
            region
            for region in figure_07.regions
            if region["shape"] == "ellipse"
        ]
        self.assertTrue(nearest)
        self.assertTrue(all(region.get("pick") == "nearest" for region in nearest))

    def test_noninteractive_figures_have_no_hit_regions(self) -> None:
        for figure_key, selected in (
            ("03", []),
            ("06", ["G0001", "G0002"]),
            ("10", []),
        ):
            with self.subTest(figure=figure_key):
                result = self.renderer.render_result(figure_key, selected)
                self.assert_png(result.image)
                self.assertEqual(result.regions, ())

    def test_figure_01_hit_centers_align_with_rendered_markers(self) -> None:
        selected = ["G0001", "G0002"]
        result = self.renderer.render_result("01", selected)
        image = Image.open(io.BytesIO(result.image)).convert("RGB")
        self.assertEqual(image.size, (result.width, result.height))
        for region in result.regions:
            cow_id = str(region["cowId"])
            self.assertGreater(
                float(region["rx"]) * result.width,
                0.5 * (72.0 ** 0.5) * 145.0 / 72.0,
            )
            center_x = min(
                result.width - 1,
                max(0, int(round(float(region["cx"]) * result.width))),
            )
            center_y = min(
                result.height - 1,
                max(0, int(round(float(region["cy"]) * result.height))),
            )
            radius = max(2, int(round(float(region["rx"]) * result.width * 0.7)))
            expected_hex = self.data.colors[cow_id].lstrip("#")
            expected = tuple(
                int(expected_hex[offset : offset + 2], 16)
                for offset in (0, 2, 4)
            )
            pixels = [
                image.getpixel((x, y))
                for x in range(max(0, center_x - radius), min(result.width, center_x + radius + 1))
                for y in range(max(0, center_y - radius), min(result.height, center_y + radius + 1))
            ]
            closest_distance = min(
                sum((actual[channel] - expected[channel]) ** 2 for channel in range(3))
                for actual in pixels
            )
            self.assertLess(closest_distance, 2_500, cow_id)

    def test_figure_01_trajectory_cache_and_render(self) -> None:
        original = dashboard_plotting.source.draw_trajectory_group
        plotted: list[str] = []

        def capture(axis, group, color, obstacles):
            plotted.extend(group["cow_id"].astype(str).unique())
            return original(axis, group, color, obstacles)

        with patch.object(
            dashboard_plotting.source,
            "draw_trajectory_group",
            side_effect=capture,
        ):
            image = self.renderer.render("01", self.selections["01"])
        self.assert_png(image)
        self.assertEqual(set(plotted), set(self.selections["01"]))
        self.assertIsNotNone(self.data.cow_frame())
        self.assertEqual(
            set(self.data.cow_frame()["cow_id"].astype(str)),
            set(self.data.all_cows),
        )

    def test_network_render_inputs_preserve_nodes_edges_and_global_scales(self) -> None:
        original = dashboard_plotting.source.draw_network
        captures: list[dict[str, object]] = []

        def capture(axis, network, positions, layer, title, **kwargs):
            captures.append(
                {
                    "edges": network.copy(),
                    "nodes": set(positions),
                    "max_weight": kwargs["max_weight"],
                    "layer": layer,
                }
            )
            return original(axis, network, positions, layer, title, **kwargs)

        with patch.object(
            dashboard_plotting.source,
            "draw_network",
            side_effect=capture,
        ):
            figure = self.renderer._figure_04(["G0001"], "friendly", "04A")
            dashboard_plotting.source.plt.close(figure)
        self.assertEqual(len(captures), 1)
        self.assertEqual(captures[0]["nodes"], {"G0001"})
        self.assertTrue(captures[0]["edges"].empty)
        self.assertEqual(captures[0]["max_weight"], self.data.full_network_max)

        captures.clear()
        with patch.object(
            dashboard_plotting.source,
            "draw_network",
            side_effect=capture,
        ):
            figure = self.renderer._figure_05(
                ["G0002"],
                "05E",
                "wait_for_water",
            )
            dashboard_plotting.source.plt.close(figure)
        self.assertEqual(len(captures), 2)
        for item in captures:
            self.assertEqual(item["nodes"], {"G0002"})
            self.assertTrue(item["edges"].empty)
            self.assertEqual(
                item["max_weight"],
                self.data.zone_network_max["wait_for_water"],
            )

        selected = set(self.selections["05E"])
        for network in self.data.zone_networks["wait_for_water"].values():
            filtered = self.renderer._selected_edge_frame(network, selected)
            endpoints = set(filtered["cow_i"].astype(str)) | set(
                filtered["cow_j"].astype(str)
            )
            self.assertTrue(endpoints <= selected)

    def test_global_scale_and_filter_order_are_render_inputs(self) -> None:
        imshow_calls: list[tuple[tuple[int, ...], object]] = []
        original_imshow = Axes.imshow

        def capture_imshow(axis, values, *args, **kwargs):
            imshow_calls.append((np.asarray(values).shape, kwargs.get("vmax")))
            return original_imshow(axis, values, *args, **kwargs)

        with patch.object(Axes, "imshow", new=capture_imshow):
            figure = self.renderer._figure_06(["G0001", "G0002", "G0003"])
            dashboard_plotting.source.plt.close(figure)
        self.assertEqual(
            imshow_calls,
            [
                ((3, 3), self.data.full_matrix_vmax),
                ((3, 3), self.data.full_matrix_vmax),
            ],
        )

        minmax_row_counts: list[int] = []
        original_minmax = dashboard_plotting.source.minmax_frame

        def capture_minmax(frame, columns):
            minmax_row_counts.append(len(frame))
            return original_minmax(frame, columns)

        with patch.object(
            dashboard_plotting.source,
            "minmax_frame",
            side_effect=capture_minmax,
        ):
            figure = self.renderer._figure_07(["G0001", "G0002"])
            dashboard_plotting.source.plt.close(figure)
        self.assertEqual(minmax_row_counts, [62])

        imshow_calls.clear()
        with patch.object(Axes, "imshow", new=capture_imshow):
            figure = self.renderer._figure_08(["G0001", "G0002", "G0003"])
            dashboard_plotting.source.plt.close(figure)
        self.assertEqual(imshow_calls[0][0], (3, 92))

        figure_02 = self.renderer._figure_02(["G0010"])
        full_budget_height = float(
            self.data.data["time_budget"][
                [
                    "time_food_s",
                    "time_path_s",
                    "time_rest_s",
                    "time_wait_for_water_s",
                    "time_water_s",
                ]
            ].sum(axis=1).max()
        )
        self.assertAlmostEqual(
            figure_02.axes[0].get_ylim()[1],
            full_budget_height * 1.05,
            places=8,
        )
        dashboard_plotting.source.plt.close(figure_02)


class StaticContractTests(unittest.TestCase):
    def test_dom_references_exist(self) -> None:
        html = (ROOT / "static" / "index.html").read_text(encoding="utf-8")
        javascript = (ROOT / "static" / "app.js").read_text(encoding="utf-8")
        html_ids = set(re.findall(r'\bid="([^"]+)"', html))
        referenced_ids = set(re.findall(r'getElementById\("([^"]+)"\)', javascript))
        self.assertEqual(referenced_ids - html_ids, set())
        self.assertGreater(len(referenced_ids), 20)

    def test_frontend_is_same_origin_and_has_no_08b(self) -> None:
        html = (ROOT / "static" / "index.html").read_text(encoding="utf-8")
        javascript = (ROOT / "static" / "app.js").read_text(encoding="utf-8")
        combined = html + javascript
        self.assertNotRegex(combined, r"https?://")
        self.assertNotIn("08B", combined)
        self.assertIn("/api/bootstrap", javascript)
        self.assertIn("/api/figure", javascript)

    def test_embedded_figure_click_card_contract(self) -> None:
        html = (ROOT / "static" / "index.html").read_text(encoding="utf-8")
        javascript = (ROOT / "static" / "app.js").read_text(encoding="utf-8")
        css = (ROOT / "static" / "app.css").read_text(encoding="utf-8")
        self.assertIn('id="cowInfoCard"', html)
        self.assertIn('id="cowInfoClose"', html)
        self.assertRegex(
            html,
            r'<img\b(?=[^>]*\bid="cowInfoPhoto")[^>]*>',
        )
        self.assertIn('id="cowInfoAppearances"', html)
        self.assertIn("/api/figure-map", javascript)
        self.assertIn("/api/cow-appearances?", javascript)
        self.assertIn("/api/cow-photo?", javascript)
        self.assertIn("new AbortController()", javascript)
        self.assertIn('document.createElement("a")', javascript)
        self.assertIn("link.href = appearance.href", javascript)
        self.assertIn('link.target = "_blank"', javascript)
        self.assertIn('link.rel = "noopener noreferrer"', javascript)
        self.assertIn("link.textContent = appearance.display", javascript)
        self.assertIn("requireAppearanceHref", javascript)
        self.assertIn('url.host !== "172.17.6.39:9922"', javascript)
        self.assertIn(".blob()", javascript)
        self.assertIn("decodedImage.decode()", javascript)
        self.assertIn("URL.createObjectURL", javascript)
        self.assertIn("URL.revokeObjectURL", javascript)
        self.assertIn("infoPhotoUrl", javascript)
        self.assertIn('figureImage.addEventListener("click"', javascript)
        self.assertIn("INFO_CARD_LIFETIME_MS = 30_000", javascript)
        self.assertIn("const decodedImage = new Image()", javascript)
        self.assertNotIn("elements.figureImage.onload", javascript)
        self.assertIn(".cow-info-card", css)
        self.assertIn(".cow-info-photo-frame", css)
        self.assertIn(".cow-info-photo", css)
        self.assertRegex(
            html,
            r'<img\b(?=[^>]*\bid="cowInfoPhoto")'
            r'(?=[^>]*\bwidth="640")(?=[^>]*\bheight="360")[^>]*>',
        )
        photo_frame_rules = re.search(
            r"\.cow-info-photo-frame\s*\{(.*?)\}",
            css,
            re.DOTALL,
        )
        self.assertIsNotNone(photo_frame_rules)
        self.assertIn("aspect-ratio: 16 / 9", photo_frame_rules.group(1))
        self.assertIn("overflow: hidden", photo_frame_rules.group(1))
        photo_rules = re.search(
            r"\.cow-info-photo\s*\{(.*?)\}",
            css,
            re.DOTALL,
        )
        self.assertIsNotNone(photo_rules)
        self.assertIn("width: 100%", photo_rules.group(1))
        self.assertIn("height: 100%", photo_rules.group(1))
        self.assertIn("object-fit: contain", photo_rules.group(1))
        appearance_list_rules = re.search(
            r"\.cow-info-appearances\s*\{(.*?)\}",
            css,
            re.DOTALL,
        )
        self.assertIsNotNone(appearance_list_rules)
        self.assertIn("max-height: 15rem", appearance_list_rules.group(1))
        self.assertIn("overflow-y: auto", appearance_list_rules.group(1))
        appearance_row_rules = re.search(
            r"\.cow-info-appearance\s*\{(.*?)\}",
            css,
            re.DOTALL,
        )
        self.assertIsNotNone(appearance_row_rules)
        self.assertIn("line-height: 1.5rem", appearance_row_rules.group(1))
        appearance_link_rules = re.search(
            r"\.cow-info-appearance-link\s*\{(.*?)\}",
            css,
            re.DOTALL,
        )
        self.assertIsNotNone(appearance_link_rules)
        self.assertIn("display: block", appearance_link_rules.group(1))
        self.assertIn("width: 100%", appearance_link_rules.group(1))
        self.assertIn("color: var(--teal)", appearance_link_rules.group(1))
        self.assertIn("text-decoration-line: underline", appearance_link_rules.group(1))
        self.assertIn("text-overflow: ellipsis", appearance_link_rules.group(1))
        interactive_match = re.search(
            r"const INTERACTIVE_FIGURES = new Set\(\[(.*?)\]\);",
            javascript,
            re.DOTALL,
        )
        self.assertIsNotNone(interactive_match)
        interactive = set(re.findall(r'"([0-9A-Z]+)"', interactive_match.group(1)))
        self.assertEqual(
            interactive,
            {
                "01",
                "02",
                "04A",
                "04B",
                "05A",
                "05B",
                "05C",
                "05D",
                "05E",
                "05F",
                "07",
                "08",
                "09",
            },
        )

    def test_default_dashboard_port_is_2299(self) -> None:
        self.assertEqual(build_argument_parser().parse_args([]).port, 2299)


class HttpContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data = DashboardData()
        cls.renderer = FigureRenderer(cls.data, max_cache_entries=8)
        cls.latest_requests: dict[str, int] = {}

        def register(client_id: str, request_version: int) -> bool:
            current = cls.latest_requests.get(client_id, -1)
            if request_version < current:
                return False
            cls.latest_requests[client_id] = request_version
            return True

        def is_current(client_id: str, request_version: int) -> bool:
            return cls.latest_requests.get(client_id) == request_version

        cls.fake_server = SimpleNamespace(
            dashboard_data=cls.data,
            figure_renderer=cls.renderer,
            register_figure_request=register,
            figure_request_is_current=is_current,
        )

    def request(
        self,
        path: str,
        method: str = "GET",
    ) -> tuple[int, dict[str, str], bytes]:
        handler = DashboardHandler.__new__(DashboardHandler)
        handler.server = self.fake_server
        handler.path = path
        handler.command = method
        handler.wfile = io.BytesIO()
        result: dict[str, object] = {
            "status": None,
            "headers": {},
        }

        def send_response(status: int) -> None:
            result["status"] = status

        def send_header(name: str, value: str) -> None:
            result["headers"][name] = value

        handler.send_response = send_response
        handler.send_header = send_header
        handler.end_headers = lambda: None
        if method == "GET":
            handler.do_GET()
        elif method == "POST":
            handler.do_POST()
        else:
            raise ValueError(f"Unsupported test method: {method}")
        return (
            int(result["status"]),
            dict(result["headers"]),
            handler.wfile.getvalue(),
        )

    def test_static_bootstrap_and_health(self) -> None:
        status, headers, body = self.request("/")
        self.assertEqual(status, 200)
        self.assertIn("text/html", headers["Content-Type"])
        self.assertIn(b"Social Network", body)

        status, _, body = self.request("/api/bootstrap")
        self.assertEqual(status, 200)
        payload = json.loads(body)
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["dataset"]["generationId"], EXPECTED_GENERATION_ID)
        self.assertEqual(len(payload["cattle"]), 62)
        self.assertEqual([item["key"] for item in payload["figures"]], EXPECTED_FIGURES)

        status, _, body = self.request("/api/health")
        self.assertEqual(status, 200)
        payload = json.loads(body)
        self.assertEqual(payload["status"], "ready")

    def test_cow_appearances_endpoint(self) -> None:
        path = (
            "/api/cow-appearances?cow=G0001"
            f"&generation={EXPECTED_GENERATION_ID}"
        )
        status, headers, body = self.request(path)
        self.assertEqual(status, 200)
        self.assertIn("application/json", headers["Content-Type"])
        payload = json.loads(body)
        self.assertEqual(
            set(payload),
            {"ok", "cowId", "appearances"},
        )
        self.assertIs(payload["ok"], True)
        self.assertEqual(payload["cowId"], "G0001")
        self.assertEqual(
            payload["appearances"],
            self.data.appearances_for_cow("G0001"),
        )

    def test_invalid_cow_appearances_queries_are_rejected(self) -> None:
        invalid_paths = {
            "missing generation": "/api/cow-appearances?cow=G0001",
            "duplicate generation": (
                "/api/cow-appearances?cow=G0001"
                f"&generation={EXPECTED_GENERATION_ID}"
                f"&generation={EXPECTED_GENERATION_ID}"
            ),
            "stale generation": (
                "/api/cow-appearances?cow=G0001&generation=stale"
            ),
            "missing cow": (
                f"/api/cow-appearances?generation={EXPECTED_GENERATION_ID}"
            ),
            "duplicate cow": (
                "/api/cow-appearances?cow=G0001&cow=G0002"
                f"&generation={EXPECTED_GENERATION_ID}"
            ),
            "unknown cow": (
                "/api/cow-appearances?cow=G9999"
                f"&generation={EXPECTED_GENERATION_ID}"
            ),
        }
        for label, path in invalid_paths.items():
            with self.subTest(case=label):
                status, headers, body = self.request(path)
                self.assertEqual(status, 400)
                self.assertIn("application/json", headers["Content-Type"])
                payload = json.loads(body)
                self.assertIs(payload["ok"], False)
                self.assertEqual(payload["error"]["code"], "INVALID_REQUEST")

    def test_cow_photo_endpoint(self) -> None:
        path = (
            "/api/cow-photo?cow=G0001"
            f"&generation={EXPECTED_GENERATION_ID}"
        )
        status, headers, body = self.request(path)
        photo = self.data.cow_photo_for_cow("G0001")
        self.assertEqual(status, 200)
        self.assertEqual(headers["Content-Type"], "image/jpeg")
        self.assertEqual(int(headers["Content-Length"]), len(body))
        self.assertEqual(body, photo.path.read_bytes())
        self.assertEqual(hashlib.sha256(body).hexdigest(), photo.sha256)
        with Image.open(io.BytesIO(body)) as image:
            self.assertEqual(image.format, "JPEG")
            self.assertEqual(image.size, (640, 360))
            image.verify()

    def test_invalid_cow_photo_queries_are_rejected(self) -> None:
        invalid_paths = {
            "missing generation": "/api/cow-photo?cow=G0001",
            "duplicate generation": (
                "/api/cow-photo?cow=G0001"
                f"&generation={EXPECTED_GENERATION_ID}"
                f"&generation={EXPECTED_GENERATION_ID}"
            ),
            "stale generation": "/api/cow-photo?cow=G0001&generation=stale",
            "missing cow": (
                f"/api/cow-photo?generation={EXPECTED_GENERATION_ID}"
            ),
            "duplicate cow": (
                "/api/cow-photo?cow=G0001&cow=G0002"
                f"&generation={EXPECTED_GENERATION_ID}"
            ),
            "unknown cow": (
                "/api/cow-photo?cow=G9999"
                f"&generation={EXPECTED_GENERATION_ID}"
            ),
            "unexpected parameter": (
                "/api/cow-photo?cow=G0001"
                f"&generation={EXPECTED_GENERATION_ID}&extra=1"
            ),
        }
        for label, path in invalid_paths.items():
            with self.subTest(case=label):
                status, headers, body = self.request(path)
                self.assertEqual(status, 400)
                self.assertIn("application/json", headers["Content-Type"])
                payload = json.loads(body)
                self.assertIs(payload["ok"], False)
                self.assertEqual(payload["error"]["code"], "INVALID_REQUEST")

    def test_figure_endpoint(self) -> None:
        path = (
            "/api/figure?figure=02&cattle=G0001,G0002"
            f"&generation={EXPECTED_GENERATION_ID}&client=testclient&request=1"
        )
        status, headers, image = self.request(path)
        self.assertEqual(status, 200)
        self.assertEqual(headers["Content-Type"], "image/png")
        self.assertEqual(headers["X-Dashboard-Figure"], "02")
        self.assertEqual(headers["X-Selected-Cattle"], "2")
        self.assertEqual(image[:8], b"\x89PNG\r\n\x1a\n")

    def test_figure_map_endpoint_matches_the_png_request_contract(self) -> None:
        query = (
            "figure=07&cattle=G0001,G0002"
            f"&generation={EXPECTED_GENERATION_ID}&client=pairedclient&request=1"
        )
        image_status, _, image = self.request(f"/api/figure?{query}")
        map_status, headers, body = self.request(f"/api/figure-map?{query}")
        self.assertEqual(image_status, 200)
        self.assertEqual(image[:8], b"\x89PNG\r\n\x1a\n")
        self.assertEqual(map_status, 200)
        self.assertIn("application/json", headers["Content-Type"])
        payload = json.loads(body)
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["figure"], "07")
        self.assertEqual(payload["width"], int.from_bytes(image[16:20], "big"))
        self.assertEqual(payload["height"], int.from_bytes(image[20:24], "big"))
        self.assertEqual(len(payload["regions"]), 4)
        self.assertEqual(
            {region["cowId"] for region in payload["regions"]},
            {"G0001", "G0002"},
        )

    def test_invalid_identity_is_rejected(self) -> None:
        path = (
            "/api/figure?figure=02&cattle=G9999"
            f"&generation={EXPECTED_GENERATION_ID}&client=testclient&request=2"
        )
        status, _, body = self.request(path)
        self.assertEqual(status, 400)
        payload = json.loads(body)
        self.assertEqual(payload["error"]["code"], "INVALID_REQUEST")

    def test_missing_or_stale_generation_is_rejected(self) -> None:
        status, _, _ = self.request("/api/figure?figure=02&cattle=G0001")
        self.assertEqual(status, 400)
        status, _, _ = self.request(
            "/api/figure?figure=02&cattle=G0001&generation=stale"
        )
        self.assertEqual(status, 400)

    def test_superseded_request_is_rejected_without_rendering(self) -> None:
        current_path = (
            "/api/figure?figure=02&cattle=G0001"
            f"&generation={EXPECTED_GENERATION_ID}&client=queueclient&request=4"
        )
        status, _, _ = self.request(current_path)
        self.assertEqual(status, 200)
        stale_path = (
            "/api/figure?figure=02&cattle=G0002"
            f"&generation={EXPECTED_GENERATION_ID}&client=queueclient&request=3"
        )
        status, _, body = self.request(stale_path)
        self.assertEqual(status, 409)
        payload = json.loads(body)
        self.assertEqual(payload["error"]["code"], "RENDER_SUPERSEDED")


if __name__ == "__main__":
    unittest.main()
