from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cowtrack.config import ContractError
from cowtrack.linking.features import LONG_FEATURE_SCHEMA
from cowtrack.linking.long_proposal_config import load_long_proposal_config
from cowtrack.linking.model import LinkResult
from cowtrack.linking.runtime import fingerprint_file
from cowtrack.linking.s05_proposal_runtime import load_s05_proposals
from cowtrack.schemas.s05_proposals import (
    LONG_CANDIDATE_EDGES_SCHEMA,
    LONG_LINK_PROPOSALS_SCHEMA,
)
import cowtrack.stages.s05_propose as stage


ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "configs" / "s05_proposals.yaml"
EXPECTED_LONG_CALIBRATION_CONFIG_HASH = load_long_proposal_config(
    CONFIG
)[0].expected_long_calibration_config_hash


def _segment(
    row: dict[str, Any],
    prefix: str,
    *,
    stable_id: int,
    start_time: float,
    end_time: float,
) -> None:
    base = 1_000 + stable_id * 10
    row.update(
        {
            f"{prefix}_stable_id": stable_id,
            f"{prefix}_constituent_micro_ids": [100 + stable_id],
            f"{prefix}_start_det_id": base,
            f"{prefix}_end_det_id": base + 1,
            f"{prefix}_start_clip_id": "GX040006",
            f"{prefix}_end_clip_id": "GX040006",
            f"{prefix}_start_global_frame": int(start_time * 10),
            f"{prefix}_end_global_frame": int(end_time * 10),
            f"{prefix}_start_time_sec": start_time,
            f"{prefix}_end_time_sec": end_time,
            f"{prefix}_num_detections": 2,
        }
    )


def _features(*, gap: float) -> dict[str, float]:
    values = {name: 0.0 for name in LONG_FEATURE_SCHEMA}
    values.update(
        {
            "prototype_cosine_max": 0.9,
            "prototype_cosine_top3_mean": 0.85,
            "medoid_cosine": 0.8,
            "mutual_prototype_score": 0.87,
            "appearance_quality_min": 0.9,
            "appearance_quality_mean": 0.9,
            "gap_sec": gap,
            "log1p_gap_sec": math.log1p(gap),
            "src_end_boundary_distance": 1.0,
            "dst_start_boundary_distance": 1.0,
        }
    )
    return values


def _gallery_provenance(
    row: dict[str, Any], prefix: str, *, token: int, present: bool
) -> None:
    if not present:
        row.update(
            {
                f"{prefix}_{suffix}": None
                for suffix in stage._GALLERY_PROVENANCE_SUFFIXES
            }
        )
        return
    sample_ids = [token + 1, token + 2, token + 3]
    row.update(
        {
            f"{prefix}_sample_ids": sample_ids,
            f"{prefix}_gallery_det_ids": [token + 101, token + 102, token + 103],
            f"{prefix}_embedding_rows": [token + 201, token + 202, token + 203],
            f"{prefix}_medoid_sample_id": sample_ids[0],
            f"{prefix}_appearance_quality": 0.9,
            f"{prefix}_internal_cosine_p10": 0.8,
            f"{prefix}_internal_cosine_p50": 0.9,
            f"{prefix}_internal_cosine_min": 0.75,
            f"{prefix}_gallery_num_input_samples": 3,
            f"{prefix}_gallery_num_overlap_rejected": 0,
            f"{prefix}_gallery_num_review_excluded": 0,
            f"{prefix}_gallery_num_local_outliers": 0,
            f"{prefix}_gallery_max_other_bbox_iou": 0.1,
            f"{prefix}_gallery_max_clean_other_bbox_iou": 0.1,
        }
    )


def _candidate_rows() -> list[dict[str, Any]]:
    provisional: dict[str, Any] = {
        "candidate_id": "s05c-000000-000001",
        "source_stable_id": 0,
        "target_stable_id": 1,
        "temporal_gap_sec": 9.0,
        "temporally_nonoverlapping": True,
        "appearance_present": True,
        "source_gallery_present": True,
        "target_gallery_present": True,
        "source_endpoint_review_excluded": False,
        "target_endpoint_review_excluded": False,
        "high_overlap": False,
        "selected_by_appearance_topk": True,
        "selected_by_temporal_nearest": True,
        "appearance_rank_out": 1,
        "appearance_rank_in": 1,
        "best_margin_out": 0.2,
        "best_margin_in": 0.1,
        "gallery_score_max": 0.9,
        "gallery_score_top3": 0.85,
        "gallery_score_src_to_dst": 0.88,
        "gallery_score_dst_to_src": 0.86,
        "gallery_score_mutual": 0.87,
        "model_probability": 0.8,
        "model_raw_score": math.log(4.0),
        "candidate_margin": 0.1,
        "provisional_threshold": 0.5,
        "selected_probability_threshold": 0.75,
        "selected_margin_threshold": 0.05,
        "passes_provisional_threshold": True,
        "passes_selected_probability_gate": True,
        "passes_selected_margin_gate": True,
        "passes_selected_gate": True,
        "decision": "provisional",
        "decision_reason": "selected_gate_evidence_uncertified",
        "selected_by_solver": False,
        "confirmed": False,
        "merge_applied": False,
        "proposed_for_review": True,
    }
    _segment(
        provisional,
        "source",
        stable_id=0,
        start_time=0.0,
        end_time=1.0,
    )
    _segment(
        provisional,
        "target",
        stable_id=1,
        start_time=10.0,
        end_time=11.0,
    )
    _gallery_provenance(provisional, "source", token=10, present=True)
    _gallery_provenance(provisional, "target", token=20, present=True)
    provisional.update(_features(gap=9.0))

    rejected: dict[str, Any] = {
        "candidate_id": "s05c-000001-000002",
        "source_stable_id": 1,
        "target_stable_id": 2,
        "temporal_gap_sec": 9.0,
        "temporally_nonoverlapping": True,
        "appearance_present": False,
        "source_gallery_present": True,
        "target_gallery_present": False,
        "source_endpoint_review_excluded": False,
        "target_endpoint_review_excluded": False,
        "high_overlap": False,
        "selected_by_appearance_topk": False,
        "selected_by_temporal_nearest": True,
        "appearance_rank_out": None,
        "appearance_rank_in": None,
        "best_margin_out": None,
        "best_margin_in": None,
        "gallery_score_max": None,
        "gallery_score_top3": None,
        "gallery_score_src_to_dst": None,
        "gallery_score_dst_to_src": None,
        "gallery_score_mutual": None,
        "model_probability": None,
        "model_raw_score": None,
        "candidate_margin": None,
        "provisional_threshold": 0.5,
        "selected_probability_threshold": 0.75,
        "selected_margin_threshold": 0.05,
        "passes_provisional_threshold": False,
        "passes_selected_probability_gate": False,
        "passes_selected_margin_gate": False,
        "passes_selected_gate": False,
        "decision": "reject",
        "decision_reason": "target_gallery_missing",
        "selected_by_solver": False,
        "confirmed": False,
        "merge_applied": False,
        "proposed_for_review": False,
    }
    _segment(
        rejected,
        "source",
        stable_id=1,
        start_time=10.0,
        end_time=11.0,
    )
    _segment(
        rejected,
        "target",
        stable_id=2,
        start_time=20.0,
        end_time=21.0,
    )
    _gallery_provenance(rejected, "source", token=30, present=True)
    _gallery_provenance(rejected, "target", token=40, present=False)
    rejected.update({name: None for name in LONG_FEATURE_SCHEMA})
    return [provisional, rejected]


def _replace_output_fingerprint(output: Path, name: str) -> None:
    marker_path = output / "_SUCCESS.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    payload = (output / name).read_bytes()
    replacement = {
        "path": name,
        "size_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    marker["output_fingerprints"] = [
        replacement if item["path"] == name else item
        for item in marker["output_fingerprints"]
    ]
    marker_path.write_text(json.dumps(marker), encoding="utf-8")


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    directories = {
        name: tmp_path / name
        for name in (
            "00_ingest",
            "01_microtrack",
            "02_appearance",
            "04_short_stable",
            "05_long_calibration",
        )
    }
    for directory in directories.values():
        directory.mkdir()
    (directories["05_long_calibration"] / "_SUCCESS.json").write_text(
        '{"synthetic":true}\n', encoding="utf-8"
    )
    transitive_input = directories["00_ingest"] / "transitive-input.bin"
    transitive_input.write_bytes(b"immutable-transitive-input")
    long_output = directories["05_long_calibration"] / "link-model.bin"
    long_output.write_bytes(b"immutable-long-model")

    production = SimpleNamespace(input_fingerprints=())
    stable = SimpleNamespace(
        input_fingerprints=(),
        stable_ids=tuple(range(3_769)),
        micro_to_stable={0: 0},
        micro_order_in_stable={0: 0},
    )
    selected_gate = SimpleNamespace(
        probability_threshold=0.75,
        margin_threshold=0.05,
        certified=False,
        executable=False,
        false_accepts=0,
        certification_hard_negative_count=140,
        false_accept_upper=0.021,
        certification_true_accepts=42,
    )
    calls = {"store": 0, "retrieve": 0, "score": 0, "rescore": 0}

    class ReplayScorer:
        def score_features(
            self,
            feature_values: dict[str, float] | None,
            *,
            appearance_present: bool,
            high_overlap: bool,
            candidate_margin: float | None = None,
        ) -> LinkResult:
            calls["rescore"] += 1
            if not appearance_present:
                return LinkResult(
                    None,
                    None,
                    "reject",
                    None,
                    "appearance_missing",
                    appearance_present=False,
                    high_overlap=high_overlap,
                )
            assert feature_values is not None
            return LinkResult(
                0.8,
                math.log(4.0),
                "provisional",
                dict(feature_values),
                None,
                appearance_present=True,
                high_overlap=high_overlap,
            )

    runtime = SimpleNamespace(
        config_hash=EXPECTED_LONG_CALIBRATION_CONFIG_HASH,
        model_enabled=True,
        confirmed_enabled=False,
        effective_config={
            "split": {
                "train_fraction": 0.60,
                "threshold_selection_fraction": 0.10,
                "certification_fraction": 0.10,
                "audit_fraction": 0.20,
            },
            "pseudo_positive": {"min_gap_sec_exclusive": 5.0},
            "appearance": {
                "clean_max_other_bbox_iou_exclusive": 0.25,
                "min_clean_samples_per_side": 3,
                "outlier_medoid_cosine": 0.65,
                "outlier_support_cosine": 0.70,
                "new_prototype_cosine": 0.92,
                "min_side_internal_cosine_p10": 0.70,
            },
        },
        thresholds=SimpleNamespace(
            provisional_threshold=0.5,
            confirmed_disabled_reason=(
                "certification_false_accept_upper_exceeds_target"
            ),
        ),
        selected_gate_evidence=selected_gate,
        input_fingerprints=(fingerprint_file(transitive_input),),
        output_fingerprints=(fingerprint_file(long_output),),
        scorer=ReplayScorer(),
    )

    class FakeStore:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            calls["store"] += 1

        def retrieval_descriptors(self, **kwargs: Any) -> tuple[object, ...]:
            return (SimpleNamespace(prototypes=None),)

    def retrieve(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        calls["retrieve"] += 1
        return [{"unscored": True}]

    def score(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        calls["score"] += 1
        return _candidate_rows()

    monkeypatch.setattr(stage, "load_production_inputs", lambda *a, **k: production)
    monkeypatch.setattr(stage, "load_s04_finalized", lambda *a, **k: stable)
    monkeypatch.setattr(stage, "load_s05_long_calibration", lambda *a, **k: runtime)
    monkeypatch.setattr(stage, "_validate_input_alignment", lambda *a, **k: None)
    monkeypatch.setattr(stage, "StablePathFeatureStore", FakeStore)
    monkeypatch.setattr(stage, "enumerate_long_candidates", retrieve)
    monkeypatch.setattr(stage, "score_long_candidates", score)

    output = tmp_path / "05_long_proposals"

    def run(target: Path = output) -> dict[str, Any]:
        return stage.run_s05_propose(
            directories["00_ingest"],
            directories["01_microtrack"],
            directories["02_appearance"],
            directories["04_short_stable"],
            directories["05_long_calibration"],
            CONFIG,
            target,
            logger=lambda message: None,
        )

    run.runtime = runtime  # type: ignore[attr-defined]

    return output, run, calls


def test_public_proposal_loader_consumes_complete_graph_not_review_sample(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output, run, _ = _fixture(tmp_path, monkeypatch)
    run()
    bundle = load_s05_proposals(
        output,
        run.runtime,  # type: ignore[attr-defined]
        expected_stable_count=3_769,
    )
    assert len(bundle.candidates) == 2
    assert len(bundle.proposals) == 1
    assert bundle.proposals[0]["candidate_id"] == "s05c-000000-000001"

    report = output / "s05_proposal_report.json"
    report.write_bytes(report.read_bytes() + b"tamper")
    with pytest.raises(ContractError, match="artifact changed"):
        load_s05_proposals(
            output,
            run.runtime,  # type: ignore[attr-defined]
            expected_stable_count=3_769,
        )


def test_stage_atomically_commits_candidates_proposals_report_and_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output, run, calls = _fixture(tmp_path, monkeypatch)

    marker = run()

    assert calls == {"store": 1, "retrieve": 1, "score": 1, "rescore": 2}
    assert output.is_dir()
    assert not list(tmp_path.glob(".05_long_proposals.staging-*"))
    assert {path.name for path in output.iterdir()} == {
        "long_candidate_edges.parquet",
        "long_link_proposals.parquet",
        "s05_review_manifest.json",
        "s05_proposal_report.json",
        "effective_config.json",
        "_SUCCESS.json",
    }
    candidates = pq.ParquetFile(output / "long_candidate_edges.parquet")
    proposals = pq.ParquetFile(output / "long_link_proposals.parquet")
    assert candidates.schema_arrow.equals(
        LONG_CANDIDATE_EDGES_SCHEMA, check_metadata=False
    )
    assert proposals.schema_arrow.equals(
        LONG_LINK_PROPOSALS_SCHEMA, check_metadata=False
    )
    assert candidates.metadata.num_rows == 2
    assert proposals.metadata.num_rows == 1
    candidate_rows = candidates.read().to_pylist()
    proposal_rows = proposals.read().to_pylist()
    assert candidate_rows[0]["source_sample_ids"] == [11, 12, 13]
    assert candidate_rows[0]["target_sample_ids"] == [21, 22, 23]
    assert candidate_rows[1]["target_sample_ids"] is None
    assert proposal_rows[0]["evidence_status"] == "selected_gate_uncertified"
    report = json.loads((output / "s05_proposal_report.json").read_text())
    assert report["execution_mode"] == "long_proposal_only"
    assert report["counts"]["candidates"] == 2
    assert report["counts"]["provisional_review_proposals"] == 1
    assert report["counts"]["confirmed"] == 0
    assert report["safety_boundary"] == {
        **report["safety_boundary"],
        "automatic_merge_allowed": False,
        "confirmed_links_allowed": False,
        "solver_used": False,
        "path_cover_used": False,
        "global_identity_artifacts_emitted": False,
    }
    assert marker["execution_mode"] == "long_proposal_only"
    assert marker["automatic_merge_allowed"] is False
    assert marker["confirmed_links_allowed"] is False
    assert marker["solver_used"] is False
    assert marker["path_cover_used"] is False
    assert marker["num_merges"] == 0
    fingerprint_names = {
        Path(item["path"]).name for item in marker["input_fingerprints"]
    }
    assert {"transitive-input.bin", "link-model.bin"} <= fingerprint_names
    assert marker["stats"] == {
        "num_candidates": 2,
        "num_proposals": 1,
        "num_rejects": 1,
        "num_confirmed": 0,
        "num_solver_selected": 0,
        "num_merges": 0,
    }
    assert not (output / "stable_to_global.parquet").exists()


def test_partial_write_failure_leaves_no_output_or_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output, run, _ = _fixture(tmp_path, monkeypatch)
    real_write = stage._write_parquet
    calls = 0

    def fail_second_write(*args: Any, **kwargs: Any) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ContractError("synthetic proposal write failure")
        real_write(*args, **kwargs)

    monkeypatch.setattr(stage, "_write_parquet", fail_second_write)
    with pytest.raises(ContractError, match="synthetic proposal write failure"):
        run()
    assert not output.exists()
    assert not list(tmp_path.glob(".05_long_proposals.staging-*"))


def test_completed_rerun_rebuilds_exact_retrieval_and_scoring(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, run, calls = _fixture(tmp_path, monkeypatch)
    first = run()
    assert run() == first
    assert calls == {"store": 2, "retrieve": 2, "score": 2, "rescore": 3}


def test_completed_rerun_rejects_candidate_omission_with_refreshed_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output, run, calls = _fixture(tmp_path, monkeypatch)
    run()

    candidates_path = output / "long_candidate_edges.parquet"
    rows = pq.read_table(candidates_path).to_pylist()
    pq.write_table(
        pa.Table.from_pylist(rows[:-1], schema=LONG_CANDIDATE_EDGES_SCHEMA),
        candidates_path,
    )
    _replace_output_fingerprint(output, "long_candidate_edges.parquet")
    with pytest.raises(ContractError, match="exact retrieval/model recomputation"):
        run()
    assert calls == {"store": 2, "retrieve": 2, "score": 2, "rescore": 2}


def test_completed_rerun_rejects_gallery_provenance_tamper_with_refreshed_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output, run, _ = _fixture(tmp_path, monkeypatch)
    run()

    candidates_path = output / "long_candidate_edges.parquet"
    rows = pq.read_table(candidates_path).to_pylist()
    rows[0]["source_appearance_quality"] = 0.85
    pq.write_table(
        pa.Table.from_pylist(rows, schema=LONG_CANDIDATE_EDGES_SCHEMA),
        candidates_path,
    )
    _replace_output_fingerprint(output, "long_candidate_edges.parquet")
    with pytest.raises(ContractError, match="exact retrieval/model recomputation"):
        run()


@pytest.mark.parametrize("field", ["model_probability", "model_raw_score"])
def test_completed_rerun_rejects_infinite_model_values_with_refreshed_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    output, run, _ = _fixture(tmp_path, monkeypatch)
    run()
    candidates_path = output / "long_candidate_edges.parquet"
    rows = pq.read_table(candidates_path).to_pylist()
    rows[0][field] = math.inf
    pq.write_table(
        pa.Table.from_pylist(rows, schema=LONG_CANDIDATE_EDGES_SCHEMA),
        candidates_path,
    )
    _replace_output_fingerprint(output, "long_candidate_edges.parquet")
    with pytest.raises(ContractError, match="exact retrieval/model recomputation"):
        run()


def test_completed_rerun_rejects_byte_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output, run, _ = _fixture(tmp_path, monkeypatch)
    run()
    report = output / "s05_proposal_report.json"
    report.write_bytes(report.read_bytes() + b" ")
    with pytest.raises(ContractError, match="artifact changed"):
        run()


def test_nonempty_or_file_output_without_success_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, run, _ = _fixture(tmp_path, monkeypatch)
    nonempty = tmp_path / "nonempty"
    nonempty.mkdir()
    (nonempty / "partial.tmp").write_bytes(b"partial")
    with pytest.raises(ContractError, match="non-empty without _SUCCESS"):
        run(nonempty)

    file_output = tmp_path / "ordinary-file"
    file_output.write_bytes(b"not a directory")
    with pytest.raises(ContractError, match="output path is not a directory"):
        run(file_output)
