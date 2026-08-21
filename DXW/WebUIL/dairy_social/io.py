from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .config import load_config


def load_trajectories(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"farm": str, "camera": str, "clip": str, "cow_id": str})


def load_interactions(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"farm": str, "camera": str, "clip": str, "cow_i": str, "cow_j": str})


def load_zones(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Zones file is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


__all__ = ["load_trajectories", "load_interactions", "load_zones", "load_config"]
