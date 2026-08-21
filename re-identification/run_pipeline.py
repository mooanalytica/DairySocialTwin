"""One-command entry point for the fixed unattended CowTrack pipeline."""

from __future__ import annotations

import os
import sys
from pathlib import Path


PROJECT_ROOT = Path("/home/hyw/re-identification")
if Path(__file__).resolve().parent != PROJECT_ROOT:
    raise SystemExit(f"run_pipeline.py must remain at {PROJECT_ROOT}")

# Select physical GPU 1 before any CowTrack/PyTorch module can be imported.
visible = os.environ.get("CUDA_VISIBLE_DEVICES")
if visible not in (None, "1"):
    raise SystemExit(
        "CUDA_VISIBLE_DEVICES must be unset or exactly '1'; refusing to select another GPU"
    )
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cowtrack.pipeline import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
