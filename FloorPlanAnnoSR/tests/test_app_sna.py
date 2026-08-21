from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import app_sna


def write_png_header(path: Path, width: int = 3840, height: int = 2160) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        b"\x89PNG\r\n\x1a\n"
        + (13).to_bytes(4, "big")
        + b"IHDR"
        + width.to_bytes(4, "big")
        + height.to_bytes(4, "big")
    )
    path.write_bytes(header)


class SnaGroupIsolationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory(prefix=".sna_test_", dir=app_sna.ROOT)
        temp_path = Path(self.temp_dir.name).resolve()
        if app_sna.ROOT.resolve() not in temp_path.parents:
            self.temp_dir.cleanup()
            raise RuntimeError(f"refusing to use unexpected test path: {temp_path}")
        self.addCleanup(self.temp_dir.cleanup)
        root = temp_path
        self.output_root = root / "local_output15"
        self.source_root = root / "source_output15"
        self.manifest_file = self.source_root / "floorplan_groups.json"
        self.patch = patch.multiple(
            app_sna,
            OUTPUT_ROOT=self.output_root,
            SOURCE_OUTPUT_ROOT=self.source_root,
            SOURCE_MANIFEST_FILE=self.manifest_file,
        )
        self.patch.start()
        self.addCleanup(self.patch.stop)

        for group in app_sna.expected_group_records():
            paths = app_sna.GroupPaths(group["id"])
            write_png_header(paths.frame)
            app_sna.write_json(
                paths.frame_meta,
                {
                    "groupId": group["id"],
                    "farmId": group["farmId"],
                    "cameraId": group["cameraId"],
                    "rotationApplied": False,
                    "scaleApplied": False,
                },
            )

        existing_paths = app_sna.GroupPaths("farm_ID_1_camera_ID_1")
        existing = app_sna.base_zone_doc(existing_paths, "2026-01-01T00:00:00Z")
        existing["zones"] = [
            {
                "zone_id": f"existing_{index + 1}",
                "zone_type": "resource_area",
                "label": f"Existing {index + 1}",
                "color": "#0b7285",
                "polygon": [[index * 20, 0], [index * 20 + 10, 0], [index * 20 + 10, 10]],
            }
            for index in range(5)
        ]
        app_sna.write_json(existing_paths.zones, existing)
        self.existing_paths = existing_paths
        self.existing_bytes = existing_paths.zones.read_bytes()

    def test_expected_groups_are_three_farms_by_five_cameras(self) -> None:
        records = app_sna.expected_group_records()
        ids = [record["id"] for record in records]
        self.assertEqual(len(records), 15)
        self.assertEqual(len(set(ids)), 15)
        self.assertEqual(ids[0], "farm_ID_1_camera_ID_1")
        self.assertEqual(ids[-1], "farm_ID_3_camera_ID_5")

    def test_group_paths_are_unique_and_group_scoped(self) -> None:
        records = app_sna.expected_group_records()
        zone_paths = [app_sna.GroupPaths(record["id"]).zones for record in records]
        self.assertEqual(len(zone_paths), len(set(zone_paths)))
        for record, zone_path in zip(records, zone_paths, strict=True):
            self.assertEqual(zone_path.parent.name, record["id"])
            self.assertEqual(zone_path.name, app_sna.ZONES_NAME)

    def test_existing_group_loads_five_zones_and_other_groups_are_blank(self) -> None:
        existing = app_sna.load_zones(self.existing_paths)
        self.assertEqual(len(existing["zones"]), 5)
        for record in app_sna.expected_group_records()[1:]:
            annotation = app_sna.load_zones(app_sna.GroupPaths(record["id"]))
            self.assertEqual(annotation["groupId"], record["id"])
            self.assertEqual(annotation["zones"], [])

    def test_saving_one_blank_group_does_not_change_other_groups(self) -> None:
        target = app_sna.GroupPaths("farm_ID_2_camera_ID_3")
        neighbor = app_sna.GroupPaths("farm_ID_2_camera_ID_4")
        payload = {
            "zones": [
                {
                    "zone_id": "water_1",
                    "zone_type": "water",
                    "label": "Water",
                    "color": "#1864ab",
                    "polygon": [[100, 100], [200, 100], [200, 200], [100, 200]],
                }
            ]
        }

        saved = app_sna.validate_zones(payload, target)
        app_sna.write_json(target.zones, saved)
        app_sna.write_preview_svg(saved, target)
        app_sna.write_convention_doc(saved, target)

        self.assertTrue(target.zones.exists())
        self.assertTrue(target.preview.exists())
        self.assertTrue(target.convention.exists())
        self.assertFalse(neighbor.local_output_dir.exists())
        self.assertEqual(self.existing_paths.zones.read_bytes(), self.existing_bytes)

        reloaded = json.loads(target.zones.read_text(encoding="utf-8"))
        self.assertEqual(reloaded["schemaVersion"], app_sna.SCHEMA_VERSION)
        self.assertEqual(reloaded["groupId"], "farm_ID_2_camera_ID_3")
        self.assertEqual(reloaded["farm"], "2")
        self.assertEqual(reloaded["camera"], "Gopro3")
        self.assertEqual(reloaded["coordinate_system"], app_sna.COORDINATE_SYSTEM)
        self.assertEqual(reloaded["zones"][0]["zone_id"], "water_1")
        self.assertIn("farm_ID_2_camera_ID_3", reloaded["image"]["file"])

    def test_group_id_rejects_path_traversal(self) -> None:
        for value in ("../farm_ID_1_camera_ID_1", "farm_ID_1_camera_ID_1/..", "farm_1_camera_1"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    app_sna.GroupPaths(value)


if __name__ == "__main__":
    unittest.main()
