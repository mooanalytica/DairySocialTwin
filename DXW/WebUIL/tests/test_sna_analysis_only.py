from __future__ import annotations

import csv
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import generate_sna_precomputed as generator


class AnalysisOnlyTest(unittest.TestCase):
    def _write_checkpoint(self, base: Path) -> tuple[Path, str]:
        sample_dir = base / generator.COMBINED_SAMPLE_ID
        sample_dir.mkdir(parents=True)
        generation_id = "11111111-1111-4111-8111-111111111111"
        cow_ids = (
            "00000000-0000-4000-8000-000000000001",
            "00000000-0000-4000-8000-000000000002",
        )
        frame_count = 400

        with generator.trajectories_path(sample_dir).open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=generator.TRAJECTORY_COLUMNS)
            writer.writeheader()
            for frame in range(frame_count):
                for global_id, cow_id in enumerate(cow_ids):
                    writer.writerow(
                        {
                            "farm": generator.EXPECTED_FARM_ID,
                            "camera": generator.EXPECTED_CAMERA_ID,
                            "clip": generator.COMBINED_SAMPLE_ID,
                            "frame": frame,
                            "time_s": frame / generator.EXPECTED_SEQUENCE_FPS,
                            "cow_id": cow_id,
                            "anchor_x": 25.0 + global_id * 10.0,
                            "anchor_y": 25.0,
                            "track_conf": 1.0,
                            "frozen": 0,
                            "display_global_id": f"G{global_id + 1:04d}",
                            "source_clip_id": "GX010006",
                            "source_sample_id": "F1_Gopro1_GX010006_1",
                            "local_frame": frame,
                            "local_track_id": global_id + 1,
                        }
                    )

        with generator.interactions_path(sample_dir).open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=generator.INTERACTION_COLUMNS)
            writer.writeheader()
            for frame in range(frame_count):
                writer.writerow(
                    {
                        "farm": generator.EXPECTED_FARM_ID,
                        "camera": generator.EXPECTED_CAMERA_ID,
                        "clip": generator.COMBINED_SAMPLE_ID,
                        "frame": frame,
                        "time_s": frame / generator.EXPECTED_SEQUENCE_FPS,
                        "cow_i": cow_ids[0],
                        "cow_j": cow_ids[1],
                        "p_friendly": 0.75,
                        "p_unfriendly": 0.1,
                        "opportunity_eligible": 1,
                        "interaction_conf": 1.0,
                        "display_global_id_i": "G0001",
                        "display_global_id_j": "G0002",
                        "source_clip_id": "GX010006",
                        "source_sample_id": "F1_Gopro1_GX010006_1",
                        "local_frame": frame,
                        "local_track_id_i": 1,
                        "local_track_id_j": 2,
                    }
                )

        generator.write_json(
            generator.zones_path(sample_dir),
            {
                "farm": generator.EXPECTED_FARM_ID,
                "camera": generator.EXPECTED_CAMERA_ID,
                "clip": generator.COMBINED_SAMPLE_ID,
                "coordinate_system": "test_floorplan_pixels",
                "outside_zone": "path",
                "source": "synthetic-test",
                "zones": [
                    {
                        "zone_id": "rest_test",
                        "zone_type": "rest",
                        "label": "rest",
                        "color": "#000000",
                        "polygon": [[0.0, 0.0], [100.0, 0.0], [100.0, 100.0], [0.0, 100.0]],
                    }
                ],
            },
        )
        generator.write_json(
            generator.identities_path(sample_dir),
            {
                "schema_version": 1,
                "generation_id": generation_id,
                "sequence_id": generator.REID_SEQUENCE_ID,
                "combined_sample_id": generator.COMBINED_SAMPLE_ID,
                "identity_key": "global_track_uuid",
                "display_label": "display_global_id",
                "identities": [
                    {
                        "global_track_uuid": cow_id,
                        "global_track_id": global_id,
                        "display_global_id": f"G{global_id + 1:04d}",
                        "id_statuses": ["synthetic"],
                        "local_track_ids_by_clip": {"GX010006": [global_id + 1]},
                    }
                    for global_id, cow_id in enumerate(cow_ids)
                ],
            },
        )
        generator.write_combined_config(generator.config_path(sample_dir), frame_count)
        generator.write_json(
            generator.generation_manifest_path(sample_dir),
            {
                "schema_version": 1,
                "generation_id": generation_id,
                "generated_at": "2026-01-01T00:00:00",
                "combined_sample_id": generator.COMBINED_SAMPLE_ID,
                "farm": generator.EXPECTED_FARM_ID,
                "camera": generator.EXPECTED_CAMERA_ID,
                "reid_sequence_id": generator.REID_SEQUENCE_ID,
                "timeline_fps": generator.EXPECTED_SEQUENCE_FPS,
                "identity_key": "global_track_uuid",
                "display_label": "display_global_id",
                "canonical_frame_policy": generator.SNA_METADATA["canonical_frame_policy"],
                "frame_limit": frame_count,
                "canonical_frame_count": frame_count,
                "trajectory_frame_count": frame_count,
                "trajectory_row_count": frame_count * 2,
                "frozen_trajectory_row_count": 0,
                "interaction_row_count": frame_count,
                "global_identity_count": len(cow_ids),
                "samples": [
                    {
                        "sample_id": "F1_Gopro1_GX010006_1",
                        "source_clip_id": "GX010006",
                        "source_path": "/unused/GX010006.MP4",
                        "source_dir": "/unused/GX010006_1",
                        "shard_index": 1,
                        "shard_count": 2,
                        "segment_start_frame": 0,
                        "segment_end_frame": frame_count - 1,
                        "canonical_start_frame": 0,
                        "canonical_end_frame": frame_count - 1,
                        "canonical_frame_count": frame_count,
                        "trajectory_row_count": frame_count * 2,
                        "interaction_row_count": frame_count,
                    }
                ],
            },
        )
        return sample_dir, generation_id

    def test_analysis_only_runs_without_cuda_or_stage2_and_preserves_checkpoint(self) -> None:
        old_cuda = os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        stage2_module_before = sys.modules.get("stage2_runtime")
        try:
            with tempfile.TemporaryDirectory(prefix="sna-analysis-only-real-") as temporary:
                root = Path(temporary)
                inputs_base = root / "inputs"
                outputs_base = root / "outputs"
                sample_dir, generation_id = self._write_checkpoint(inputs_base)
                tuned_config_path = root / "tuned_sna.yaml"
                tuned_config = generator.load_config(generator.config_path(sample_dir))
                tuned_config["community"]["min_visible_time_s"] = 1.0
                generator.write_config(tuned_config_path, tuned_config)
                before = {
                    path.name: (path.stat().st_ino, path.stat().st_size, path.stat().st_mtime_ns)
                    for path in sample_dir.iterdir()
                }

                result = generator.main(
                    [
                        "--analysis-only",
                        "--inputs-dir",
                        str(inputs_base),
                        "--outdir",
                        str(outputs_base),
                        "--analysis-config",
                        str(tuned_config_path),
                        "--overwrite",
                    ]
                )

                self.assertEqual(result, 0)
                self.assertIs(sys.modules.get("stage2_runtime"), stage2_module_before)
                after = {
                    path.name: (path.stat().st_ino, path.stat().st_size, path.stat().st_mtime_ns)
                    for path in sample_dir.iterdir()
                }
                self.assertEqual(before, after)
                output_dir = outputs_base / generator.COMBINED_SAMPLE_ID
                output_manifest = json.loads(
                    (output_dir / "generation_manifest.json").read_text(encoding="utf-8")
                )
                self.assertEqual(output_manifest["generation_id"], generation_id)
                self.assertEqual(output_manifest["frame_limit"], 400)
                self.assertTrue(output_manifest["analysis_run_id"])
                self.assertEqual(output_manifest["analysis_config_path"], str(tuned_config_path))
                self.assertTrue((output_dir / "analysis_config_resolved.yaml").is_file())
                self.assertTrue((output_dir / "edge_level.csv").is_file())
                resolved_config = generator.load_config(output_dir / "analysis_config_resolved.yaml")
                self.assertEqual(resolved_config["community"]["min_visible_time_s"], 1.0)
        finally:
            if old_cuda is not None:
                os.environ["CUDA_VISIBLE_DEVICES"] = old_cuda


if __name__ == "__main__":
    unittest.main()
