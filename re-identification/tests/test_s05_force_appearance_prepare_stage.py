from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from cowtrack.config import ContractError
from cowtrack.linking.forced_appearance import (
    SupplementalStableEmbeddings,
    build_forced_appearance_base,
    complete_forced_appearance_with_supplemental,
)
from cowtrack.linking.forced_appearance_config import load_forced_appearance_config
from cowtrack.linking.runtime import fingerprint_file
import cowtrack.stages.s05_force_appearance as stage


CONFIG_PATH = Path("configs/s05_force_appearance.yaml")


def _unit(*values: float) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float32)
    return vector / np.linalg.norm(vector)


def _config(*, no_s02: int | None = 1):
    config, payload, config_hash = load_forced_appearance_config(CONFIG_PATH)
    return (
        replace(
            config,
            expected_stable_track_count=2,
            expected_microtrack_count=2,
            expected_clean_appearance_count=0,
            expected_missing_clean_appearance_count=2,
            expected_no_s02_sample_count=no_s02,
            embedding_dim=3,
        ),
        payload,
        config_hash,
    )


def _prepared_fixture():
    base = build_forced_appearance_base(
        stable_ids=np.arange(2, dtype=np.int64),
        existing_prototypes=np.zeros((2, 3, 3), dtype=np.float16),
        existing_prototype_mask=np.zeros((2, 3), dtype=np.bool_),
        existing_usable=np.zeros(2, dtype=np.bool_),
        sample_ids=np.asarray([10], dtype=np.int64),
        sample_embeddings=np.stack([_unit(1.0, 0.2, 0.0)]),
        sample_stable_ids=np.asarray([0], dtype=np.int64),
        sample_quality=np.asarray([0.8], dtype=np.float32),
        sample_other_bbox_max_iou=np.asarray([0.1], dtype=np.float32),
        sample_s02_inlier=np.asarray([True], dtype=np.bool_),
    )
    rescue_embedding = _unit(0.0, 1.0, 0.2)
    appearance = complete_forced_appearance_with_supplemental(
        base,
        (
            SupplementalStableEmbeddings(
                stable_id=1,
                sample_ids=np.asarray([1_000_000], dtype=np.int64),
                embeddings=rescue_embedding[None, :],
                quality=np.asarray([0.7], dtype=np.float32),
            ),
        ),
    )
    rescue_rows = [
        {
            "stable_id": 1,
            "micro_id": 1,
            "det_id": 101,
            "clip_id": "A",
            "local_frame": 2,
            "global_frame": 2,
            "global_time_sec": 2.0,
            "x1": 1.0,
            "y1": 2.0,
            "x2": 11.0,
            "y2": 12.0,
            "crop_quality": 0.7,
            "other_bbox_max_iou": 0.1,
            "clipped_fraction": 0.0,
            "bbox_area_percentile": 0.5,
            "blur_score": 0.8,
            "distance_to_image_boundary": 0.9,
            "review_excluded": False,
            "quality_gate_passed": True,
            "selected_for_descriptor": True,
            "selection_reason": "quality_ranked",
            "embedding_row": 0,
        }
    ]
    return appearance, rescue_rows, rescue_embedding[None, :]


def _write_bundle(output: Path):
    config, payload, config_hash = _config()
    appearance, rescue_rows, rescue_embeddings = _prepared_fixture()
    inputs = [fingerprint_file(CONFIG_PATH).as_dict()]
    marker = stage._write_prepared_output(
        output,
        config=config,
        config_payload=payload,
        config_hash=config_hash,
        input_fingerprints=inputs,
        appearance=appearance,
        rescue_rows=rescue_rows,
        rescue_embeddings=rescue_embeddings,
        elapsed_sec=1.25,
    )
    return config, payload, config_hash, inputs, appearance, marker


def _make_solver_drift_source(output: Path):
    config, payload, config_hash, inputs, appearance, marker = _write_bundle(
        output
    )
    source_payload = deepcopy(payload)
    source_payload["solver"] = {"superseded_solve_policy": True}
    names = stage._prepared_artifact_names(config)
    stage._write_json(output / names["effective_config"], source_payload)
    source_hash = "a" * 64
    marker["config_hash"] = source_hash
    config_record = next(
        item
        for item in marker["input_fingerprints"]
        if item["path"] == str(CONFIG_PATH.resolve())
    )
    config_record["size_bytes"] = 1
    config_record["sha256"] = source_hash
    marker["output_fingerprints"] = [
        stage._output_fingerprint(output / name, output)
        for name in sorted(names.values())
    ]
    stage._write_json(output / stage._PREPARE_MARKER, marker)
    return config, payload, config_hash, inputs, appearance, marker, source_payload


def test_prepared_bundle_round_trips_exact_float32_centers(tmp_path: Path) -> None:
    output = tmp_path / "prepared"
    config, payload, config_hash, inputs, appearance, marker = _write_bundle(output)

    restored, loaded_marker = stage._load_prepared_appearance(
        output,
        config=config,
        config_payload=payload,
        config_hash=config_hash,
        input_fingerprints=inputs,
    )

    assert loaded_marker == marker
    assert np.array_equal(
        restored.appearance.descriptor_centers, appearance.descriptor_centers
    )
    assert restored.appearance.descriptor_centers.dtype == np.float32
    assert restored.no_s02_sample_count == 1
    assert restored.rescue_embeddings.shape == (1, 3)
    assert restored.rescue_embeddings.dtype == np.float16


def test_prepared_bundle_rejects_byte_tamper(tmp_path: Path) -> None:
    output = tmp_path / "prepared"
    config, payload, config_hash, inputs, _, _ = _write_bundle(output)
    centers = output / stage._PREPARED_DESCRIPTOR_CENTERS
    with centers.open("ab") as handle:
        handle.write(b"tamper")

    with pytest.raises(ContractError, match="prepared artifact changed"):
        stage._load_prepared_appearance(
            output,
            config=config,
            config_payload=payload,
            config_hash=config_hash,
            input_fingerprints=inputs,
        )


def test_prepared_bundle_semantics_reject_wrong_center_dtype_after_rehash(
    tmp_path: Path,
) -> None:
    output = tmp_path / "prepared"
    config, payload, config_hash, inputs, _, marker = _write_bundle(output)
    names = stage._prepared_artifact_names(config)
    center_path = output / names["descriptor_centers"]
    centers = np.load(center_path, allow_pickle=False)
    np.save(center_path, centers.astype(np.float16), allow_pickle=False)
    marker["output_fingerprints"] = [
        stage._output_fingerprint(output / name, output)
        for name in sorted(names.values())
    ]
    stage._write_json(output / stage._PREPARE_MARKER, marker)

    with pytest.raises(ContractError, match="restored prototype tensors"):
        stage._load_prepared_appearance(
            output,
            config=config,
            config_payload=payload,
            config_hash=config_hash,
            input_fingerprints=inputs,
        )


def test_repackage_preserves_descriptor_bytes_and_target_strict_loads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    target = tmp_path / "target"
    (
        config,
        payload,
        config_hash,
        _inputs,
        appearance,
        source_marker,
        _source_payload,
    ) = _make_solver_drift_source(source)
    monkeypatch.setattr(
        stage,
        "load_forced_appearance_config",
        lambda _path: (config, payload, config_hash),
    )
    names = stage._prepared_artifact_names(config)
    descriptor_names = sorted(
        name
        for key, name in names.items()
        if key != "effective_config"
    )
    source_bytes = {
        name: (source / name).read_bytes() for name in descriptor_names
    }

    marker = stage.run_s05_force_appearance_repackage_prepared(
        source,
        CONFIG_PATH,
        target,
        logger=lambda _message: None,
    )

    assert marker["config_hash"] == config_hash
    assert marker["stats"] == source_marker["stats"]
    assert stage._read_json(
        target / names["effective_config"], "target effective config"
    ) == payload
    assert {
        name: (target / name).read_bytes() for name in descriptor_names
    } == source_bytes
    assert {
        name: (source / name).read_bytes() for name in descriptor_names
    } == source_bytes
    assert marker["output_fingerprints"] == [
        stage._output_fingerprint(target / name, target)
        for name in sorted(names.values())
    ]
    current_config_record = next(
        item
        for item in marker["input_fingerprints"]
        if item["path"] == str(CONFIG_PATH.resolve())
    )
    assert current_config_record == fingerprint_file(CONFIG_PATH).as_dict()
    restored, loaded_marker = stage._load_prepared_appearance(
        target,
        config=config,
        config_payload=payload,
        config_hash=config_hash,
        input_fingerprints=marker["input_fingerprints"],
    )
    assert loaded_marker == marker
    assert np.array_equal(
        restored.appearance.descriptor_centers,
        appearance.descriptor_centers,
    )


@pytest.mark.parametrize("section", ("pipeline", "inputs", "appearance", "runtime"))
def test_repackage_rejects_non_solver_config_drift(
    tmp_path: Path,
    section: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    target = tmp_path / "target"
    config, payload, config_hash, _, _, marker, source_payload = (
        _make_solver_drift_source(source)
    )
    monkeypatch.setattr(
        stage,
        "load_forced_appearance_config",
        lambda _path: (config, payload, config_hash),
    )
    changed = deepcopy(source_payload)
    changed[section][next(iter(changed[section]))] = "tampered"
    names = stage._prepared_artifact_names(config)
    stage._write_json(source / names["effective_config"], changed)
    marker["output_fingerprints"] = [
        stage._output_fingerprint(source / name, source)
        for name in sorted(names.values())
    ]
    stage._write_json(source / stage._PREPARE_MARKER, marker)

    with pytest.raises(
        ContractError,
        match="allows changes only within the solver section",
    ):
        stage.run_s05_force_appearance_repackage_prepared(
            source,
            CONFIG_PATH,
            target,
            logger=lambda _message: None,
        )
    assert not target.exists()


def test_repackage_rejects_source_artifact_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    target = tmp_path / "target"
    config, payload, config_hash, *_ = _make_solver_drift_source(source)
    monkeypatch.setattr(
        stage,
        "load_forced_appearance_config",
        lambda _path: (config, payload, config_hash),
    )
    with (source / stage._PREPARED_DESCRIPTOR_CENTERS).open("ab") as handle:
        handle.write(b"tamper")

    with pytest.raises(ContractError, match="prepared artifact changed"):
        stage.run_s05_force_appearance_repackage_prepared(
            source,
            CONFIG_PATH,
            target,
            logger=lambda _message: None,
        )
    assert not target.exists()


def test_repackage_rejects_source_output_overlap(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _make_solver_drift_source(source)

    with pytest.raises(ContractError, match="output overlaps input"):
        stage.run_s05_force_appearance_repackage_prepared(
            source,
            CONFIG_PATH,
            source,
            logger=lambda _message: None,
        )


def test_prepare_orchestration_never_enters_candidate_or_solver(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, payload, config_hash = _config(no_s02=0)
    base = build_forced_appearance_base(
        stable_ids=np.arange(2, dtype=np.int64),
        existing_prototypes=np.zeros((2, 3, 3), dtype=np.float16),
        existing_prototype_mask=np.zeros((2, 3), dtype=np.bool_),
        existing_usable=np.zeros(2, dtype=np.bool_),
        sample_ids=np.asarray([10, 11], dtype=np.int64),
        sample_embeddings=np.stack([_unit(1, 0, 0), _unit(0, 1, 0)]),
        sample_stable_ids=np.asarray([0, 1], dtype=np.int64),
        sample_quality=np.asarray([0.8, 0.8], dtype=np.float32),
        sample_other_bbox_max_iou=np.asarray([0.1, 0.1], dtype=np.float32),
        sample_s02_inlier=np.asarray([True, True], dtype=np.bool_),
    )
    inputs = (fingerprint_file(CONFIG_PATH).as_dict(),)
    common = stage._LoadedForcedInputs(
        config=config,
        config_payload=payload,
        config_hash=config_hash,
        production=SimpleNamespace(),
        stable=SimpleNamespace(),
        identity={},
        video_paths={},
        input_fingerprints=inputs,
        video_path_set=frozenset(),
    )
    monkeypatch.setattr(stage, "_load_common_forced_inputs", lambda *_a, **_k: common)
    monkeypatch.setattr(stage, "_build_base_appearance", lambda *_a, **_k: base)
    monkeypatch.setattr(stage, "_stable_by_detection", lambda *_a, **_k: np.empty(0))
    monkeypatch.setattr(stage, "_plan_rescue_candidates", lambda *_a, **_k: [])
    monkeypatch.setattr(stage, "_verify_regular_fingerprints", lambda *_a, **_k: None)
    monkeypatch.setattr(
        stage,
        "_build_graph_and_cover",
        lambda *_a, **_k: pytest.fail("prepare entered candidate/solver code"),
    )
    output = tmp_path / "prepared"
    args = (
        tmp_path / "manifest.csv",
        tmp_path / "ingest",
        tmp_path / "micro",
        tmp_path / "appearance",
        tmp_path / "stable",
        CONFIG_PATH,
        "cuda:0",
        output,
    )

    marker = stage.run_s05_force_appearance_prepare(*args, logger=lambda _m: None)
    assert marker["stage"] == stage._PREPARE_STAGE
    assert (output / stage._PREPARE_MARKER).is_file()

    monkeypatch.setattr(
        stage,
        "_build_base_appearance",
        lambda *_a, **_k: pytest.fail("valid prepared resume rebuilt descriptors"),
    )
    resumed = stage.run_s05_force_appearance_prepare(*args, logger=lambda _m: None)
    assert resumed == marker
