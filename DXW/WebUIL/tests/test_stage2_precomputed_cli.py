from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app import (
    REQUIRED_CUDA_VISIBLE_DEVICES,
    ROOT as APP_ROOT,
    STAGE2_PRECOMPUTED_DIR,
    stage2_precomputed_index_file,
)
from generate_stage2_precomputed import build_arg_parser, sample_output_file
from trackid_video import TRACK_ID_VIDEO_CACHE_DIR


class Stage2PrecomputedCliTest(unittest.TestCase):
    def test_generated_caches_are_project_local(self) -> None:
        self.assertEqual(STAGE2_PRECOMPUTED_DIR, APP_ROOT / ".cache" / "stage2_precomputed")
        self.assertEqual(TRACK_ID_VIDEO_CACHE_DIR, APP_ROOT / ".cache" / "track_id_videos")
        self.assertEqual(stage2_precomputed_index_file(), STAGE2_PRECOMPUTED_DIR / "index.json")
        self.assertEqual(sample_output_file("sample"), STAGE2_PRECOMPUTED_DIR / "sample.json")

    def test_cuda_visible_devices_default_matches_webui_gpu(self) -> None:
        args = build_arg_parser().parse_args([])
        self.assertEqual(args.cuda_visible_devices, REQUIRED_CUDA_VISIBLE_DEVICES)
        self.assertEqual(args.cuda_visible_devices, "1")

        overridden = build_arg_parser().parse_args(["--cuda-visible-devices", "3"])
        self.assertEqual(overridden.cuda_visible_devices, "3")


if __name__ == "__main__":
    unittest.main()
