from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any


DEFAULT_CONFIG: dict[str, Any] = {
    "metadata": {
        "cow_id_scope": None,
        "cross_sample_identity_alignment": False,
        "opportunity_definition": "simultaneously_visible_and_opportunity_eligible",
        "community_window_policy": "adjustable",
        "community_window_note": "Set community.window_s and community.step_s to fixed research values when specified.",
    },
    "analysis": {
        "farm": "1",
        "camera": "Gopro1",
        "clip": None,
        "start_time_s": None,
        "end_time_s": None,
    },
    "time": {"fps": None, "default_dt_s": None},
    "network": {
        "directed": False,
        "use_probability_weights": True,
        "use_track_confidence": True,
        "use_interaction_confidence": True,
        "pair_zone_mode": "same_or_cross",
        "zone_level": "zone_type",
        "complete_pair_grid_from_trajectories": False,
        "min_opportunity_s": 5.0,
        "min_visible_time_s": 60.0,
        "eps": 1.0e-9,
    },
    "zones": {
        "smoothing_s": 1.0,
        "unknown_policy": "exclude_from_zone_metrics",
        "outside_zone": "path",
    },
    "community": {
        "enabled": True,
        "window_s": 300.0,
        "step_s": 300.0,
        "min_visible_time_s": 0.0,
        "min_edges_per_window": 1,
        "algorithm": "louvain_or_greedy",
    },
    "isolation": {
        "alone_definition": "same_zone",
        "weights": {
            "alone_fraction": 0.4,
            "low_friendly_sociality": 0.3,
            "low_partner_diversity": 0.3,
        },
    },
    "report": {
        "time_bin_s": 60.0,
        "write_temporal_outputs": True,
        "edge_top_n": 20,
        "network_layout": "spring",
        "layout_random_seed": 123,
    },
    "events": {
        "enabled": True,
        "event_start_threshold": 0.5,
        "event_end_threshold": 0.3,
        "max_gap_s": 1.0,
        "min_event_duration_s": 2.0,
    },
    "outputs": {"write_adjacency_matrices": True, "write_plots": True},
}


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return copy.deepcopy(DEFAULT_CONFIG)
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Config file is missing: {path}")
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        data = json.loads(text)
    else:
        try:
            import yaml
        except Exception as exc:  # pragma: no cover - depends on target env.
            raise RuntimeError("PyYAML is required to read YAML config files") from exc
        data = yaml.safe_load(text) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return deep_update(DEFAULT_CONFIG, data)


def write_config(path: str | Path, config: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".json":
        path.write_text(json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return
    try:
        import yaml
    except Exception as exc:  # pragma: no cover - depends on target env.
        raise RuntimeError("PyYAML is required to write YAML config files") from exc
    path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
