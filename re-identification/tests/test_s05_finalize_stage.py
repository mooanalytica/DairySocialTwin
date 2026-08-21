from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pyarrow as pa
import pytest

from cowtrack.config import ContractError
from cowtrack.linking.path_cover import (
    GlobalProposalEdge,
    GlobalStableNode,
    PathCoverResult,
    solve_operator_approved_path_cover,
)
from cowtrack.linking.s04_runtime import StableTracklet
from cowtrack.linking.s05_finalize_config import load_s05_finalize_config
from cowtrack.schemas.s05_finalize import (
    DET_TO_GLOBAL_SCHEMA,
    GLOBAL_CANDIDATE_EDGES_SCHEMA,
    GLOBAL_TRACKS_SCHEMA,
    STABLE_TO_GLOBAL_SCHEMA,
)
import cowtrack.stages.s05_finalize as stage


CONFIG = Path("configs/s05_finalize.yaml")


def _small_config():
    config, payload, digest = load_s05_finalize_config(CONFIG)
    return (
        replace(
            config,
            expected_sequence_id="seq",
            expected_stable_track_count=2,
            expected_microtrack_count=2,
            expected_detection_count=3,
            expected_invalid_detections=0,
            expected_frame_count=201,
            expected_candidate_count=1,
            expected_proposal_count=1,
            clip_order=("A", "B"),
            frame_counts_by_clip=(2, 199),
            expected_valid_detections_by_clip=(2, 1),
            expected_invalid_detections_by_clip=(0, 0),
        ),
        payload,
        digest,
    )


def _stable_fixture() -> SimpleNamespace:
    tracks = {
        0: StableTracklet(
            0,
            0,
            0,
            10,
            11,
            "A",
            "A",
            0,
            1,
            0.0,
            0.1,
            1,
            2,
            0,
            None,
            None,
            None,
            True,
        ),
        1: StableTracklet(
            1,
            1,
            1,
            12,
            12,
            "B",
            "B",
            200,
            200,
            6.2,
            6.2,
            1,
            1,
            0,
            None,
            None,
            None,
            True,
        ),
    }
    return SimpleNamespace(
        stable_ids=np.asarray([0, 1], dtype=np.int64),
        stable_tracklets=tracks,
        det_ids=np.asarray([12, 10, 11], dtype=np.int64),
        det_micro_ids=np.asarray([1, 0, 0], dtype=np.int64),
        det_stable_ids=np.asarray([1, 0, 0], dtype=np.int64),
        det_order_in_micro=np.asarray([0, 0, 1], dtype=np.int64),
        det_order_in_stable=np.asarray([0, 0, 0], dtype=np.int64),
        det_order_in_stable_detection=np.asarray([0, 0, 1], dtype=np.int64),
    )


def _result() -> PathCoverResult:
    nodes = [
        GlobalStableNode(0, "A", "A", 0, 1, 0.0, 0.1, 1, 2),
        GlobalStableNode(1, "B", "B", 200, 200, 6.2, 6.2, 1, 1),
    ]
    edge = GlobalProposalEdge(
        "s05p-000000-000001",
        "s05c-000000-000001",
        0,
        1,
        0.9,
        -0.1,
        2,
        3,
        False,
        0.8,
        6.1,
    )
    return solve_operator_approved_path_cover(nodes, [edge])


def test_global_and_detection_rows_cover_both_clips_without_certification() -> None:
    config, _, _ = _small_config()
    stable = _stable_fixture()
    result = _result()

    mapping, global_rows, by_stable = stage._build_global_rows(
        stable, result, config
    )
    assert len(mapping) == 2
    assert len(global_rows) == 1
    assert [row["link_type"] for row in mapping] == ["PATH_START", "FILE_BOUNDARY"]
    assert all(row["id_status"] == "provisional" for row in mapping)
    assert global_rows[0]["clip_ids"] == ["A", "B"]
    assert global_rows[0]["num_detections"] == 3

    identity = {
        "det_id": np.asarray([10, 11, 12], dtype=np.int64),
        "sequence_id": np.asarray(["seq", "seq", "seq"], dtype=object),
        "clip_id": np.asarray(["A", "A", "B"], dtype=object),
        "local_frame": np.asarray([0, 1, 0], dtype=np.int32),
        "global_frame": np.asarray([0, 1, 200], dtype=np.int64),
        "global_time_sec": np.asarray([0.0, 0.1, 6.2], dtype=np.float64),
    }
    table = stage._build_detection_table(
        identity, stable, by_stable, global_rows, config
    )
    assert table.schema.equals(DET_TO_GLOBAL_SCHEMA, check_metadata=False)
    assert table.num_rows == 3
    assert table["global_track_id"].to_pylist() == [0, 0, 0]
    assert table["order_in_global_detection"].to_pylist() == [0, 1, 2]


def test_detection_mapping_rejects_same_global_id_twice_in_one_frame() -> None:
    config, _, _ = _small_config()
    stable = _stable_fixture()
    mapping, global_rows, by_stable = stage._build_global_rows(
        stable, _result(), config
    )
    assert mapping
    identity = {
        "det_id": np.asarray([10, 11, 12], dtype=np.int64),
        "sequence_id": np.asarray(["seq", "seq", "seq"], dtype=object),
        "clip_id": np.asarray(["A", "A", "B"], dtype=object),
        "local_frame": np.asarray([0, 0, 0], dtype=np.int32),
        "global_frame": np.asarray([0, 0, 200], dtype=np.int64),
        "global_time_sec": np.asarray([0.0, 0.0, 6.2], dtype=np.float64),
    }
    with pytest.raises(ContractError, match="appears twice"):
        stage._build_detection_table(identity, stable, by_stable, global_rows, config)


def _empty_tables() -> dict[str, pa.Table]:
    return {
        "candidate_edges": pa.Table.from_pylist([], schema=GLOBAL_CANDIDATE_EDGES_SCHEMA),
        "stable_to_global": pa.Table.from_pylist([], schema=STABLE_TO_GLOBAL_SCHEMA),
        "global_tracks": pa.Table.from_pylist([], schema=GLOBAL_TRACKS_SCHEMA),
        "det_to_global": pa.Table.from_pylist([], schema=DET_TO_GLOBAL_SCHEMA),
    }


def _runner_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    config, payload, digest = load_s05_finalize_config(CONFIG)
    config_path = tmp_path / "s05_finalize.yaml"
    config_path.write_text("fixed synthetic config\n", encoding="utf-8")
    dirs = {
        name: tmp_path / name
        for name in (
            "00_ingest",
            "01_microtrack",
            "02_appearance",
            "04_short_stable",
            "05_long_calibration",
            "05_long_proposals",
        )
    }
    for directory in dirs.values():
        directory.mkdir()
    (dirs["05_long_calibration"] / "_SUCCESS.json").write_text(
        "{}\n", encoding="utf-8"
    )
    production = SimpleNamespace(input_fingerprints=())
    stable = SimpleNamespace(
        input_fingerprints=(),
        stable_ids=np.empty(0, dtype=np.int64),
        micro_ids=np.empty(0, dtype=np.int64),
    )
    long_runtime = SimpleNamespace(
        directory=dirs["05_long_calibration"],
        input_fingerprints=(),
        output_fingerprints=(),
    )
    proposal = SimpleNamespace(
        input_fingerprints=(),
        consumed_fingerprints=(),
        candidates=(),
        proposals=(),
    )
    result = PathCoverResult(
        selected_edges=(),
        paths=(),
        predecessor_by_stable=MappingProxyType({}),
        successor_by_stable=MappingProxyType({}),
        solver_cost_by_candidate=MappingProxyType({}),
        max_cardinality=0,
    )
    stats = {"num_selected_links": 0, "num_global_tracks": 0}
    report = {"counts": stats}
    tables = _empty_tables()

    monkeypatch.setattr(
        stage, "load_s05_finalize_config", lambda _path: (config, payload, digest)
    )
    monkeypatch.setattr(stage, "load_production_inputs", lambda *a, **k: production)
    monkeypatch.setattr(stage, "load_s04_finalized", lambda *a, **k: stable)
    monkeypatch.setattr(stage, "load_s05_long_calibration", lambda *a, **k: long_runtime)
    monkeypatch.setattr(stage, "load_s05_proposals", lambda *a, **k: proposal)
    monkeypatch.setattr(stage, "_validate_fixed_inputs", lambda *a, **k: None)
    monkeypatch.setattr(stage, "_recompute_complete_graph", lambda *a, **k: ([], []))
    monkeypatch.setattr(stage, "_load_ingest_identity", lambda *a, **k: {})
    monkeypatch.setattr(
        stage,
        "_build_expected_outputs",
        lambda **kwargs: (tables, report, result),
    )
    output = tmp_path / "05_global_link"

    def run(target: Path = output):
        return stage.run_s05_finalize(
            dirs["00_ingest"],
            dirs["01_microtrack"],
            dirs["02_appearance"],
            dirs["04_short_stable"],
            dirs["05_long_calibration"],
            dirs["05_long_proposals"],
            config_path,
            target,
            logger=lambda _message: None,
        )

    return output, run, config


def test_stage_atomically_writes_and_fully_revalidates_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output, run, config = _runner_fixture(tmp_path, monkeypatch)
    marker = run()
    assert marker["stage"] == "S05_FINALIZE"
    assert marker["operator_approved"] is True
    assert marker["certification_claimed"] is False
    assert {path.name for path in output.iterdir()} == {
        config.artifacts.candidate_edges,
        config.artifacts.stable_to_global,
        config.artifacts.global_tracks,
        config.artifacts.det_to_global,
        config.artifacts.report,
        config.artifacts.effective_config,
        config.artifacts.success,
    }
    assert run() == marker

    candidate_path = output / config.artifacts.candidate_edges
    candidate_path.write_bytes(candidate_path.read_bytes() + b"tamper")
    with pytest.raises(ContractError, match="artifact changed"):
        run()


def test_stage_partial_write_failure_leaves_no_output_or_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output, run, _ = _runner_fixture(tmp_path, monkeypatch)
    real_write = stage._write_table
    count = 0

    def fail(*args, **kwargs):
        nonlocal count
        count += 1
        if count == 2:
            raise ContractError("synthetic write failure")
        return real_write(*args, **kwargs)

    monkeypatch.setattr(stage, "_write_table", fail)
    with pytest.raises(ContractError, match="synthetic write failure"):
        run()
    assert not output.exists()
    assert not list(tmp_path.glob(".05_global_link.staging-*"))


def test_stage_rejects_nonempty_output_without_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output, run, _ = _runner_fixture(tmp_path, monkeypatch)
    output.mkdir()
    (output / "partial.bin").write_bytes(b"partial")
    with pytest.raises(ContractError, match="non-empty without _SUCCESS"):
        run()
