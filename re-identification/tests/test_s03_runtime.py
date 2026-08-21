from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import numpy as np

from cowtrack.config import ContractError
from cowtrack.linking.config import load_link_calibration_config
from cowtrack.linking.runtime import (
    FileFingerprint,
    ProductionInputBundle,
    load_s03_runtime_config,
    validate_s03_input_fingerprints,
    load_production_inputs,
)


CONFIG = Path(__file__).resolve().parents[1] / "configs" / "s03_calibration.yaml"


def _write_runtime_config(directory: Path, *, input_records=None) -> str:
    _, payload, config_hash = load_link_calibration_config(CONFIG)
    effective = directory / "effective_config.json"
    effective.write_text(
        json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    raw = effective.read_bytes()
    fingerprint = {
        "path": "effective_config.json",
        "size_bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }
    (directory / "_SUCCESS.json").write_text(
        json.dumps(
            {
                "stage": "S03",
                "config_hash": config_hash,
                "input_fingerprints": input_records or [],
                "output_fingerprints": [fingerprint],
            }
        ),
        encoding="utf-8",
    )
    return config_hash


def test_runtime_config_recomputes_canonical_hash_and_verifies_bytes(tmp_path) -> None:
    expected_hash = _write_runtime_config(tmp_path)
    loaded = load_s03_runtime_config(tmp_path)
    assert loaded.config_hash == expected_hash
    assert loaded.config.short_gap_max_sec == 5.0

    with (tmp_path / "effective_config.json").open("a", encoding="utf-8") as handle:
        handle.write(" ")
    with pytest.raises(ContractError, match="runtime artifact changed"):
        load_s03_runtime_config(tmp_path)


def test_runtime_inputs_must_equal_s03_recorded_bytes(tmp_path) -> None:
    names = (
        "_SUCCESS.json",
        "frames.parquet",
        "detections.parquet",
        "_SUCCESS.json",
        "det_to_micro.parquet",
        "microtracklets.parquet",
        "_SUCCESS.json",
        "appearance_samples.parquet",
        "sample_embeddings.f16.npy",
        "appearance_exclusions.parquet",
        "micro_appearance.parquet",
        "appearance_report.json",
        "encoder_choice.json",
        "effective_config.json",
    )
    paths = tuple(tmp_path / f"part-{index}" / name for index, name in enumerate(names))
    fingerprints = tuple(
        FileFingerprint(str(path), index + 1, f"{index:064x}")
        for index, path in enumerate(paths)
    )
    records = [
        {"path": "/source/s03_calibration.yaml", "size_bytes": 1, "sha256": "f" * 64}
    ] + [
        {
            "path": f"/source/part-{index}/{path.name}",
            "size_bytes": fingerprint.size_bytes,
            "sha256": fingerprint.sha256,
        }
        for index, (path, fingerprint) in enumerate(zip(paths, fingerprints, strict=True))
    ]
    _write_runtime_config(tmp_path, input_records=records)
    bundle = ProductionInputBundle(
        calibration_input=None,  # type: ignore[arg-type]
        detections=None,  # type: ignore[arg-type]
        micro_paths={},
        endpoints={},
        micro_appearance={},
        appearance_report={},
        encoder_choice={},
        consumed_paths=paths,
        input_fingerprints=fingerprints,
    )
    validate_s03_input_fingerprints(tmp_path, bundle)

    changed = list(fingerprints)
    changed[-1] = FileFingerprint(changed[-1].path, changed[-1].size_bytes, "a" * 64)
    changed_bundle = ProductionInputBundle(
        calibration_input=bundle.calibration_input,
        detections=bundle.detections,
        micro_paths=bundle.micro_paths,
        endpoints=bundle.endpoints,
        micro_appearance=bundle.micro_appearance,
        appearance_report=bundle.appearance_report,
        encoder_choice=bundle.encoder_choice,
        consumed_paths=bundle.consumed_paths,
        input_fingerprints=tuple(changed),
    )
    with pytest.raises(ContractError, match="differs from S03 marker bytes"):
        validate_s03_input_fingerprints(tmp_path, changed_bundle)


def test_public_loader_builds_canonical_paths_and_endpoints(
    tmp_path, monkeypatch
) -> None:
    import cowtrack.linking.runtime as runtime

    count = 6
    micros = np.asarray([10, 10, 10, 20, 20, 20], dtype=np.int64)
    clips = np.asarray(["GX040006"] * 3 + ["GX050006"] * 3, dtype=object)
    data = {
        "det_id": np.arange(100, 106, dtype=np.int64),
        "global_frame": np.arange(count, dtype=np.int64),
        "global_time_sec": np.arange(count, dtype=np.float64),
        "clip_id": clips,
        "x1": np.arange(count, dtype=np.float64),
        "y1": np.arange(count, dtype=np.float64) + 1.0,
        "x2": np.arange(count, dtype=np.float64) + 10.0,
        "y2": np.arange(count, dtype=np.float64) + 11.0,
        "cx_norm": np.full(count, 0.5),
        "cy_norm": np.full(count, 0.5),
        "w_norm": np.full(count, 0.2),
        "h_norm": np.full(count, 0.2),
        "micro_id_by_detection": micros,
        "order_in_micro": np.asarray([0, 1, 2, 0, 1, 2], dtype=np.int64),
        "other_bbox_max_iou": np.zeros(count, dtype=np.float32),
        "boundary_distance": np.ones(count, dtype=np.float32),
        "review_excluded": np.zeros(count, dtype=np.bool_),
        "paths": {
            10: np.asarray([0, 1, 2], dtype=np.int64),
            20: np.asarray([3, 4, 5], dtype=np.int64),
        },
        "micro_summary": {
            "micro_id": np.asarray([10, 20], dtype=np.int64),
            "status": np.asarray(["valid", "valid"], dtype=object),
            "num_detections": np.asarray([3, 3], dtype=np.int64),
            "local_purity_score": np.ones(2),
            "bidirectional_agreement": np.ones(2),
        },
        "sample_id": np.arange(count, dtype=np.int64),
        "sample_det_id": np.arange(100, 106, dtype=np.int64),
        "sample_micro_id": micros,
        "sample_crop_quality": np.ones(count, dtype=np.float32),
        "sample_other_bbox_max_iou": np.zeros(count, dtype=np.float32),
        "sample_s02_inlier": np.ones(count, dtype=np.bool_),
        "sample_embedding_row": np.arange(count, dtype=np.int64),
        "sample_embeddings": np.ones((count, 2), dtype=np.float16),
        "micro_appearance": {
            "micro_id": np.asarray([20, 10], dtype=np.int64),
            "internal_cosine_p10": np.asarray([0.9, 0.8]),
            "appearance_usable": np.asarray([True, True]),
        },
        "frame_clip_id": clips,
        "frame_global_time_sec": np.arange(count, dtype=np.float64),
        "appearance_report": {"warnings": []},
        "encoder_choice": {"profile": "synthetic"},
    }
    monkeypatch.setattr(runtime, "_load_canonical_data", lambda *args, **kwargs: data)
    monkeypatch.setattr(runtime, "_verify_upstream_marker", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        runtime,
        "fingerprint_file",
        lambda path: FileFingerprint(str(path.resolve()), 1, "0" * 64),
    )
    bundle = load_production_inputs(
        tmp_path / "00", tmp_path / "01", tmp_path / "02"
    )
    assert tuple(bundle.micro_paths) == (10, 20)
    assert bundle.endpoints[10].start_det_id == 100
    assert bundle.endpoints[10].end_det_id == 102
    assert bundle.endpoints[20].start_clip_id == "GX050006"
    assert bundle.calibration_input.parent_internal_cosine_p10.tolist() == [0.8, 0.9]
    assert len(bundle.consumed_paths) == len(bundle.input_fingerprints) == 14
