from __future__ import annotations

from pathlib import Path

import pytest

from cowtrack.config import ContractError, load_microtrack_config


def _write_config(path: Path, *, center_weight: float = 0.55) -> None:
    path.write_text(
        f"""
pipeline:
  schema_version: "1.0"
  random_seed: 7
  use_keypoints: false
  use_legacy_tracking_id: false
microtrack:
  cost_weights:
    center_distance: {center_weight}
    iou: 0.30
    size: 0.15
  max_time_gap_sec: auto
  max_time_gap_multiplier: 1.5
  center_distance_gate: 0.75
  max_area_ratio: 1.8
  min_iou: 0.01
  alternate_center_gate: 0.35
  ambiguity_margin: 0.08
  bidirectional_edges_only: true
  bridge_missing_detections: false
  min_length_detections: 3
  velocity_history_detections: 4
  fisheye_motion_grid: [12, 8]
  motion_prior_percentile: 99.0
  motion_prior_gate_floor: 0.10
  motion_prior_min_edges_per_cell: 8
  parquet_compression: zstd
  progress_interval_sec: 10.0
""".lstrip(),
        encoding="utf-8",
    )


def test_microtrack_config_loads_auto_gap_and_fixed_contract(tmp_path: Path) -> None:
    path = tmp_path / "s01.yaml"
    _write_config(path)
    config, payload, config_hash = load_microtrack_config(path)
    assert config.max_time_gap_sec is None
    assert config.max_time_gap_multiplier == 1.5
    assert config.fisheye_grid_width == 12
    assert config.fisheye_grid_height == 8
    assert config.motion_prior_min_edges_per_cell == 8
    assert payload["pipeline"]["use_legacy_tracking_id"] is False
    assert len(config_hash) == 64


def test_repository_s01_config_matches_specification() -> None:
    path = Path(__file__).resolve().parents[1] / "configs" / "s01_microtrack.yaml"
    config, _, _ = load_microtrack_config(path)
    assert (
        config.center_distance_weight,
        config.iou_weight,
        config.size_weight,
    ) == (0.55, 0.30, 0.15)
    assert config.center_distance_gate == 0.75
    assert config.max_area_ratio == 1.8
    assert config.ambiguity_margin == 0.08
    assert config.motion_prior_percentile == 99.0


def test_microtrack_config_rejects_weights_that_do_not_sum_to_one(
    tmp_path: Path,
) -> None:
    path = tmp_path / "bad.yaml"
    _write_config(path, center_weight=0.50)
    with pytest.raises(ContractError, match="weights must sum to 1"):
        load_microtrack_config(path)


def test_microtrack_config_rejects_legacy_id_dependency(tmp_path: Path) -> None:
    path = tmp_path / "legacy.yaml"
    _write_config(path)
    text = path.read_text(encoding="utf-8").replace(
        "use_legacy_tracking_id: false", "use_legacy_tracking_id: true"
    )
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ContractError, match="use_legacy_tracking_id=false"):
        load_microtrack_config(path)


@pytest.mark.parametrize(
    ("original", "replacement"),
    [
        ("center_distance_gate: 0.75", "center_distance_gate: .nan"),
        ("max_area_ratio: 1.8", "max_area_ratio: .inf"),
        ("ambiguity_margin: 0.08", "ambiguity_margin: .nan"),
        ("progress_interval_sec: 10.0", "progress_interval_sec: .inf"),
    ],
)
def test_microtrack_config_rejects_nonfinite_values(
    tmp_path: Path, original: str, replacement: str
) -> None:
    path = tmp_path / "nonfinite.yaml"
    _write_config(path)
    path.write_text(
        path.read_text(encoding="utf-8").replace(original, replacement),
        encoding="utf-8",
    )
    with pytest.raises(ContractError, match="finite number"):
        load_microtrack_config(path)


def test_microtrack_config_rejects_unsupported_schema(tmp_path: Path) -> None:
    path = tmp_path / "schema.yaml"
    _write_config(path)
    path.write_text(
        path.read_text(encoding="utf-8").replace(
            'schema_version: "1.0"', 'schema_version: "2.0"'
        ),
        encoding="utf-8",
    )
    with pytest.raises(ContractError, match="schema_version=1.0"):
        load_microtrack_config(path)
