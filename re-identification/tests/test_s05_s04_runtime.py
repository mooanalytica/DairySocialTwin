from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from cowtrack.config import ContractError
from cowtrack.linking.finalize_config import load_s04_finalize_config
from cowtrack.linking.runtime import FileFingerprint
import cowtrack.linking.s04_runtime as s04_runtime
from cowtrack.schemas.s04 import (
    DET_TO_STABLE_SCHEMA,
    MICRO_TO_STABLE_SCHEMA,
    STABLE_TRACKLETS_SCHEMA,
)
from cowtrack.schemas.stable_appearance import STABLE_APPEARANCE_SCHEMA


ROOT = Path(__file__).parents[1]
S04_CONFIG = ROOT / "configs" / "s04_finalize.yaml"


def _fingerprint(path: Path, *, relative_to: Path | None = None) -> dict[str, object]:
    payload = path.read_bytes()
    return {
        "path": str(path.relative_to(relative_to)) if relative_to else str(path.resolve()),
        "size_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _patch_small_contract(monkeypatch: pytest.MonkeyPatch, config_hash: str) -> None:
    monkeypatch.setattr(s04_runtime, "EXPECTED_EMBEDDING_DIM", 4)


def _mapping_rows() -> list[dict[str, object]]:
    common = {
        "stable_id": 0,
        "component_num_microtracklets": 2,
        "component_num_proposal_edges": 1,
        "component_min_proposal_probability": 0.99,
        "component_mean_proposal_probability": 0.99,
        "component_max_proposal_probability": 0.99,
    }
    return [
        {
            **common,
            "micro_id": 0,
            "order_in_stable": 0,
            "predecessor_micro_id": None,
            "predecessor_edge_id": None,
            "predecessor_link_probability": None,
        },
        {
            **common,
            "micro_id": 1,
            "order_in_stable": 1,
            "predecessor_micro_id": 0,
            "predecessor_edge_id": "edge-1",
            "predecessor_link_probability": 0.99,
        },
    ]


def _stable_rows() -> list[dict[str, object]]:
    return [
        {
            "stable_id": 0,
            "first_micro_id": 0,
            "last_micro_id": 1,
            "start_det_id": -10,
            "end_det_id": 5,
            "start_clip_id": "GX040006",
            "end_clip_id": "GX040006",
            "start_global_frame": 0,
            "end_global_frame": 3,
            "start_time_sec": 0.0,
            "end_time_sec": 0.3,
            "num_microtracklets": 2,
            "num_detections": 4,
            "num_proposal_edges": 1,
            "min_proposal_probability": 0.99,
            "mean_proposal_probability": 0.99,
            "max_proposal_probability": 0.99,
            "is_singleton": False,
        }
    ]


def _detection_rows() -> list[dict[str, object]]:
    return [
        {
            "det_id": det_id,
            "micro_id": micro_id,
            "stable_id": 0,
            "order_in_stable": micro_id,
            "order_in_micro": order_in_micro,
            "order_in_stable_detection": stable_order,
        }
        for stable_order, (det_id, micro_id, order_in_micro) in enumerate(
            ((-10, 0, 0), (-9, 0, 1), (4, 1, 0), (5, 1, 1))
        )
    ]


def _appearance_rows() -> list[dict[str, object]]:
    return [
        {
            "stable_id": 0,
            "prototype_row": 0,
            "constituent_micro_ids": [0, 1],
            "num_input_samples": 3,
            "num_s02_inliers": 3,
            "num_clean_candidates": 3,
            "num_clean_inliers": 3,
            "num_valid_prototypes": 1,
            "appearance_usable": True,
            "missing_reason": None,
            "clean_sample_ids": [0, 1, 2],
            "clean_det_ids": [-10, -9, 4],
            "clean_embedding_rows": [0, 1, 2],
            "medoid_sample_id": 0,
            "appearance_quality": 0.9,
            "internal_cosine_p10": 1.0,
            "internal_cosine_p50": 1.0,
            "internal_cosine_min": 1.0,
            "num_overlap_rejected": 0,
            "num_review_excluded": 0,
            "num_local_outliers": 0,
            "max_other_bbox_iou": 0.1,
            "max_clean_other_bbox_iou": 0.1,
        }
    ]


def _stats() -> dict[str, int]:
    return {
        "microtracklets": 2,
        "detections": 4,
        "proposals_consumed": 1,
        "proposal_connected_microtracklets": 2,
        "stable_tracklets": 1,
        "non_singleton_components": 1,
        "singleton_components": 0,
        "max_component_size": 2,
        "stable_appearance_usable": 1,
        "stable_appearance_missing": 0,
    }


def _refresh_output_fingerprints(directory: Path) -> None:
    marker_path = directory / "_SUCCESS.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["output_fingerprints"] = [
        _fingerprint(directory / name, relative_to=directory)
        for name in s04_runtime._OUTPUT_NAMES
    ]
    marker_path.write_text(json.dumps(marker), encoding="utf-8")


def _finalized_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Path:
    directory = tmp_path / "04_short_stable"
    directory.mkdir()
    config_payload = yaml.safe_load(S04_CONFIG.read_text(encoding="utf-8"))
    (directory / "effective_config.json").write_text(
        json.dumps(config_payload), encoding="utf-8"
    )
    _, _, config_hash = load_s04_finalize_config(directory / "effective_config.json")
    _patch_small_contract(monkeypatch, config_hash)

    pq.write_table(
        pa.Table.from_pylist(_mapping_rows(), schema=MICRO_TO_STABLE_SCHEMA),
        directory / "micro_to_stable.parquet",
    )
    pq.write_table(
        pa.Table.from_pylist(_stable_rows(), schema=STABLE_TRACKLETS_SCHEMA),
        directory / "stable_tracklets.parquet",
    )
    pq.write_table(
        pa.Table.from_pylist(_detection_rows(), schema=DET_TO_STABLE_SCHEMA),
        directory / "det_to_stable.parquet",
    )
    pq.write_table(
        pa.Table.from_pylist(_appearance_rows(), schema=STABLE_APPEARANCE_SCHEMA),
        directory / "stable_appearance.parquet",
    )
    prototypes = np.zeros((1, 3, 4), dtype=np.float16)
    prototypes[0, 0, 0] = np.float16(1.0)
    mask = np.asarray([[True, False, False]], dtype=np.bool_)
    np.save(directory / "stable_prototypes.f16.npy", prototypes, allow_pickle=False)
    np.save(directory / "stable_prototype_mask.npy", mask, allow_pickle=False)

    upstream = tmp_path / "upstream.bin"
    upstream.write_bytes(b"immutable-upstream")
    input_fingerprints = [_fingerprint(upstream)]
    report = {
        "schema_version": "1.0",
        "stage": "S04_FINALIZE",
        "execution_mode": "operator_approved_component_union",
        "operator_approved": True,
        "merge_policy": "all_provisional_undirected_components",
        "graph_node_identity": "actual_micro_id",
        "automatic_merge_allowed": False,
        "num_automatic_merges": 0,
        "num_operator_approved_merges": 1,
        "read_review_labels": False,
        "solver_used": False,
        "counts": _stats(),
        "component_size_distribution": {"2": 1},
        "input_fingerprints": input_fingerprints,
    }
    (directory / "finalize_report.json").write_text(
        json.dumps(report), encoding="utf-8"
    )
    marker = {
        "schema_version": "1.0",
        "stage": "S04_FINALIZE",
        "config_hash": config_hash,
        "execution_mode": "operator_approved_component_union",
        "operator_approved": True,
        "merge_policy": "all_provisional_undirected_components",
        "automatic_merge_allowed": False,
        "num_automatic_merges": 0,
        "num_operator_approved_merges": 1,
        "read_review_labels": False,
        "solver_used": False,
        "input_fingerprints": input_fingerprints,
        "output_fingerprints": [
            _fingerprint(directory / name, relative_to=directory)
            for name in s04_runtime._OUTPUT_NAMES
        ],
        "stats": _stats(),
        "elapsed_sec": 1.0,
    }
    (directory / "_SUCCESS.json").write_text(json.dumps(marker), encoding="utf-8")
    return directory


def test_load_s04_finalized_returns_immutable_bijective_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = _finalized_fixture(tmp_path, monkeypatch)

    bundle = s04_runtime.load_s04_finalized(directory)

    assert bundle.micro_to_stable == {0: 0, 1: 0}
    assert bundle.micro_order_in_stable == {0: 0, 1: 1}
    assert bundle.stable_to_micros == {0: (0, 1)}
    assert bundle.det_ids.tolist() == [-10, -9, 4, 5]
    assert bundle.det_micro_ids.tolist() == [0, 0, 1, 1]
    assert bundle.det_stable_ids.tolist() == [0, 0, 0, 0]
    assert bundle.stable_tracklets[0].start_det_id == -10
    assert bundle.stable_appearance[0].clean_det_ids == (-10, -9, 4)
    assert bundle.stable_prototypes.shape == (1, 3, 4)
    assert len(bundle.consumed_paths) == len(s04_runtime._CONSUMED_NAMES)
    assert len(bundle.input_fingerprints) == len(bundle.consumed_paths)
    with pytest.raises(TypeError):
        bundle.micro_to_stable[2] = 0  # type: ignore[index]
    with pytest.raises(ValueError):
        bundle.det_ids[0] = 7
    with pytest.raises(TypeError):
        bundle.success_marker["stage"] = "changed"  # type: ignore[index]


def test_load_s04_finalized_rebinds_recorded_input_without_old_path_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = _finalized_fixture(tmp_path, monkeypatch)
    historical = (tmp_path / "upstream.bin").resolve()
    current = (tmp_path / "current" / "upstream.bin").resolve()
    current.parent.mkdir()
    current.write_bytes(historical.read_bytes())
    historical.unlink()
    messages: list[str] = []

    bundle = s04_runtime.load_s04_finalized(
        directory,
        recorded_input_relocations={str(historical): current},
        logger=messages.append,
    )

    assert bundle.stable_ids.tolist() == [0]
    assert not historical.exists()
    assert any("pass 1/2" in item for item in messages)
    assert any("pass 2/2" in item for item in messages)


def test_load_s04_finalized_rejects_source_effective_config_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = _finalized_fixture(tmp_path, monkeypatch)
    source_payload = yaml.safe_load(S04_CONFIG.read_text(encoding="utf-8"))
    source_payload["finalize"]["progress_interval_sec"] = 11.0
    source = tmp_path / "s04_finalize.yaml"
    source.write_text(yaml.safe_dump(source_payload), encoding="utf-8")

    with pytest.raises(ContractError, match="source/effective config provenance"):
        s04_runtime.load_s04_finalized(
            directory,
            source_config_path=source,
        )


def test_load_s04_finalized_rejects_changed_output_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = _finalized_fixture(tmp_path, monkeypatch)
    path = directory / "stable_prototype_mask.npy"
    path.write_bytes(path.read_bytes() + b"tampered")

    with pytest.raises(ContractError, match="artifact changed"):
        s04_runtime.load_s04_finalized(directory)


def test_load_s04_finalized_rejects_changed_recorded_input_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = _finalized_fixture(tmp_path, monkeypatch)
    (tmp_path / "upstream.bin").write_bytes(b"changed-upstream")

    with pytest.raises(ContractError, match="recorded S04 input changed"):
        s04_runtime.load_s04_finalized(directory)


def test_load_s04_finalized_rejects_exact_schema_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = _finalized_fixture(tmp_path, monkeypatch)
    rows = _detection_rows()
    wrong_schema = DET_TO_STABLE_SCHEMA.set(
        DET_TO_STABLE_SCHEMA.get_field_index("det_id"),
        pa.field("det_id", pa.uint64(), nullable=False),
    )
    for row in rows:
        row["det_id"] = int(row["det_id"]) % (1 << 64)
    pq.write_table(
        pa.Table.from_pylist(rows, schema=wrong_schema),
        directory / "det_to_stable.parquet",
    )
    _refresh_output_fingerprints(directory)

    with pytest.raises(ContractError, match="schema mismatch"):
        s04_runtime.load_s04_finalized(directory)


def test_load_s04_finalized_rejects_det_micro_stable_join_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = _finalized_fixture(tmp_path, monkeypatch)
    rows = _detection_rows()
    rows[-1]["micro_id"] = 99
    pq.write_table(
        pa.Table.from_pylist(rows, schema=DET_TO_STABLE_SCHEMA),
        directory / "det_to_stable.parquet",
    )
    _refresh_output_fingerprints(directory)

    with pytest.raises(ContractError, match="cover every micro"):
        s04_runtime.load_s04_finalized(directory)


def test_load_s04_finalized_detects_load_time_byte_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = _finalized_fixture(tmp_path, monkeypatch)
    original = s04_runtime._fingerprint_file
    calls = 0

    def changed_after_load(path: Path) -> FileFingerprint:
        nonlocal calls
        calls += 1
        result = original(path)
        if calls > len(s04_runtime._CONSUMED_NAMES) and path.name == "det_to_stable.parquet":
            return FileFingerprint(result.path, result.size_bytes, "0" * 64)
        return result

    monkeypatch.setattr(s04_runtime, "_fingerprint_file", changed_after_load)
    with pytest.raises(ContractError, match="changed while being loaded"):
        s04_runtime.load_s04_finalized(directory)
