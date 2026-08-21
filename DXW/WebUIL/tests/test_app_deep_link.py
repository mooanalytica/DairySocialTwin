from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import app


SOURCE_CLIP_ID = "GX010006"
SOURCE_PATH = "/authoritative/videos/GX010006.MP4"


def manifest(farm_id: str = "1", camera_id: str = "Gopro1") -> dict[str, Any]:
    return {
        "schema_version": 1,
        "farm": farm_id,
        "camera": camera_id,
        "combined_sample_id": f"F{farm_id}_{camera_id}_20250505",
        "samples": [
            {
                "sample_id": f"F{farm_id}_{camera_id}_{SOURCE_CLIP_ID}_1",
                "source_clip_id": SOURCE_CLIP_ID,
                "source_path": SOURCE_PATH,
                "shard_index": 1,
                "shard_count": 2,
                "segment_start_frame": 0,
                "segment_end_frame": 42652,
                "canonical_start_frame": 0,
                "canonical_end_frame": 42352,
                "canonical_frame_count": 42353,
            },
            {
                "sample_id": f"F{farm_id}_{camera_id}_{SOURCE_CLIP_ID}_2",
                "source_clip_id": SOURCE_CLIP_ID,
                "source_path": SOURCE_PATH,
                "shard_index": 2,
                "shard_count": 2,
                "segment_start_frame": 42053,
                "segment_end_frame": 88319,
                "canonical_start_frame": 42353,
                "canonical_end_frame": 88319,
                "canonical_frame_count": 45967,
            },
        ],
    }


def selectable_sample(
    shard_index: int,
    farm_id: str = "1",
    camera_id: str = "Gopro1",
    video_available: bool = True,
) -> dict[str, Any]:
    if shard_index == 1:
        segment_start, segment_end = 0, 42652
        base_start, base_end = 0, 42352
    elif shard_index == 2:
        segment_start, segment_end = 42053, 88319
        base_start, base_end = 42353, 88319
    else:
        raise ValueError(f"unsupported fixture shard: {shard_index}")
    return {
        "id": f"F{farm_id}_{camera_id}_{SOURCE_CLIP_ID}_{shard_index}",
        "farmId": farm_id,
        "cameraId": camera_id,
        "goproId": app.camera_number(camera_id),
        "sourceVideoId": SOURCE_CLIP_ID,
        "sourcePath": SOURCE_PATH,
        "shardIndex": shard_index,
        "shardCount": 2,
        "playbackSegment": {
            "baseStartFrame": base_start,
            "baseEndFrame": base_end,
            "segmentStartFrame": segment_start,
            "segmentEndFrame": segment_end,
        },
        "video": {"available": video_available},
    }


def query_params(
    shard_index: int = 2,
    frame_id: int = 82066,
    farm_id: str = "1",
    camera_id: str = "1",
    clip_id: str | None = None,
    segment_frame: int | None = None,
) -> dict[str, list[str]]:
    camera_label = f"Gopro{int(camera_id)}" if camera_id.isdigit() else camera_id
    selected_clip_id = clip_id or f"F{farm_id}_{camera_label}_{SOURCE_CLIP_ID}_{shard_index}"
    embedded_frame = frame_id if segment_frame is None else segment_frame
    return {
        "farmID": [farm_id],
        "cameraID": [camera_id],
        "clipID": [selected_clip_id],
        "segmentID": [f"{SOURCE_CLIP_ID}-{shard_index}-{embedded_frame}"],
        "frameID": [str(frame_id)],
    }


class DeepLinkManifestIndexTest(unittest.TestCase):
    def load_index(self, document: dict[str, Any]) -> dict[str, Any]:
        with patch.object(app, "read_json", return_value=document):
            return app.load_deep_link_manifest_index()

    def test_adjacent_inclusive_canonical_ranges_are_valid(self) -> None:
        index = self.load_index(manifest())

        first, second = index["bySourceClipId"][SOURCE_CLIP_ID]
        self.assertEqual((first["canonicalStartFrame"], first["canonicalEndFrame"]), (0, 42352))
        self.assertEqual((second["canonicalStartFrame"], second["canonicalEndFrame"]), (42353, 88319))

    def test_current_generation_manifest_builds_the_expected_index(self) -> None:
        index = app.load_deep_link_manifest_index()

        self.assertEqual(index["farmId"], "1")
        self.assertEqual(app.camera_number(index["cameraId"]), "1")
        self.assertEqual(len(index["bySampleId"]), 12)
        gx01 = index["bySourceClipId"][SOURCE_CLIP_ID]
        self.assertEqual(
            [(entry["shardIndex"], entry["canonicalStartFrame"], entry["canonicalEndFrame"]) for entry in gx01],
            [(1, 0, 42352), (2, 42353, 88319)],
        )

    def test_overlapping_canonical_ranges_are_a_data_error(self) -> None:
        document = manifest()
        document["samples"][0]["canonical_end_frame"] = 42353
        document["samples"][0]["canonical_frame_count"] = 42354

        with patch.object(app, "read_json", return_value=document):
            with self.assertRaisesRegex(ValueError, "canonical ranges overlap"):
                app.load_deep_link_manifest_index()

    def test_source_clip_to_full_path_must_be_bijective(self) -> None:
        document = manifest()
        document["samples"][1]["source_path"] = "/different/videos/GX010006.MP4"

        with patch.object(app, "read_json", return_value=document):
            with self.assertRaisesRegex(ValueError, "full-path mapping is not bijective"):
                app.load_deep_link_manifest_index()


class DeepLinkResolutionTest(unittest.TestCase):
    def resolve(
        self,
        params: dict[str, list[str]],
        samples: list[dict[str, Any]],
        document: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        source_document = document or manifest()
        with patch.object(app, "read_json", return_value=source_document):
            index = app.load_deep_link_manifest_index()
        with (
            patch.object(app, "load_deep_link_manifest_index", return_value=index),
            patch.object(app, "load_samples", return_value=samples),
        ):
            return app.resolve_deep_link(params)

    def test_valid_target_resolves_dropdown_sample_and_original_clip_frame(self) -> None:
        result = self.resolve(query_params(), [selectable_sample(1), selectable_sample(2)])

        self.assertEqual(
            result,
            {
                "ok": True,
                "valid": True,
                "sampleId": "F1_Gopro1_GX010006_2",
                "frameId": 82066,
            },
        )

    def test_canonical_seam_maps_to_exactly_one_shard(self) -> None:
        samples = [selectable_sample(1), selectable_sample(2)]
        cases = (
            (1, 42352, "F1_Gopro1_GX010006_1"),
            (2, 42353, "F1_Gopro1_GX010006_2"),
        )
        for shard_index, frame_id, expected_sample_id in cases:
            with self.subTest(frame=frame_id):
                result = self.resolve(query_params(shard_index, frame_id), samples)
                self.assertTrue(result["valid"])
                self.assertEqual(result["sampleId"], expected_sample_id)
                self.assertEqual(result["frameId"], frame_id)

    def test_farm_and_camera_are_matched_from_data_not_forced_to_one(self) -> None:
        document = manifest(farm_id="2", camera_id="Gopro2")
        samples = [
            selectable_sample(1, farm_id="2", camera_id="Gopro2"),
            selectable_sample(2, farm_id="2", camera_id="Gopro2"),
        ]

        result = self.resolve(query_params(farm_id="2", camera_id="2"), samples, document)

        self.assertTrue(result["valid"])
        self.assertEqual(result["sampleId"], "F2_Gopro2_GX010006_2")

    def test_partial_duplicate_or_empty_parameters_are_ignored_before_index_load(self) -> None:
        cases = [
            {"farmID": ["1"]},
            {**query_params(), "frameID": ["82066", "82067"]},
            {**query_params(), "clipID": [" "]},
        ]
        for params in cases:
            with self.subTest(params=params):
                with patch.object(app, "load_deep_link_manifest_index") as load_index:
                    self.assertEqual(app.resolve_deep_link(params), {"ok": True, "valid": False})
                    load_index.assert_not_called()

    def test_inconsistent_or_unavailable_targets_are_ignored(self) -> None:
        samples = [selectable_sample(1), selectable_sample(2)]
        cases = [
            query_params(segment_frame=82065),
            query_params(shard_index=1, frame_id=82066),
            query_params(clip_id="F1_Gopro1_GX010006_1"),
            query_params(clip_id=SOURCE_CLIP_ID),
            query_params(clip_id="F1_Gopro1_20250505"),
            query_params(shard_index=2, frame_id=42300),
            query_params(frame_id=90000),
            query_params(farm_id="2", clip_id="F1_Gopro1_GX010006_2"),
        ]
        for params in cases:
            with self.subTest(params=params):
                self.assertEqual(
                    self.resolve(params, samples),
                    {"ok": True, "valid": False},
                )

        unavailable = [selectable_sample(1), selectable_sample(2, video_available=False)]
        self.assertEqual(
            self.resolve(query_params(), unavailable),
            {"ok": True, "valid": False},
        )

    def test_manifest_and_selectable_sample_mismatch_is_a_data_error(self) -> None:
        mismatched = selectable_sample(2)
        mismatched["playbackSegment"] = copy.deepcopy(mismatched["playbackSegment"])
        mismatched["playbackSegment"]["baseEndFrame"] = 88318

        with self.assertRaisesRegex(RuntimeError, "canonical end"):
            self.resolve(query_params(), [selectable_sample(1), mismatched])

    def test_selectable_sample_missing_from_manifest_is_a_data_error(self) -> None:
        missing = selectable_sample(2)
        missing["id"] = "F1_Gopro1_GX010006_2_missing"
        params = query_params(clip_id=missing["id"])

        with self.assertRaisesRegex(RuntimeError, "absent from the generation manifest"):
            self.resolve(params, [selectable_sample(1), missing])


if __name__ == "__main__":
    unittest.main()
