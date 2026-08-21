from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from cowtrack.config import ContractError
from cowtrack.linking.features import SHORT_FEATURE_SCHEMA
from cowtrack.linking.proposal_config import load_short_proposal_config
from cowtrack.linking.proposals import (
    MicroEndpoint,
    build_proposals,
    enumerate_short_candidates,
    review_manifest,
    score_short_candidates,
)
from cowtrack.schemas.s04 import (
    DET_TO_STABLE_SCHEMA,
    MICRO_TO_STABLE_SCHEMA,
    SHORT_CANDIDATE_EDGES_SCHEMA,
    SHORT_LINK_PROPOSALS_SCHEMA,
    STABLE_TRACKLETS_SCHEMA,
)
from cowtrack.schemas.stable_appearance import STABLE_APPEARANCE_SCHEMA
import cowtrack.stages.s04_finalize as finalize_stage
from cowtrack.stages.s04_finalize import run_s04_finalize


ROOT = Path(__file__).parents[1]
PROPOSAL_CONFIG = ROOT / "configs" / "s04_proposals.yaml"
FINALIZE_CONFIG = ROOT / "configs" / "s04_finalize.yaml"
S03_HASH = "ee6730ca12bde6ffc173ce5bd3a3b327591060deee8124c8a6e99cffeb5a17ad"


def _fingerprint(path: Path, *, relative_to: Path | None = None) -> dict[str, object]:
    payload = path.read_bytes()
    return {
        "path": str(path.relative_to(relative_to)) if relative_to else str(path.resolve()),
        "size_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _gallery(seed: int) -> SimpleNamespace:
    return SimpleNamespace(
        present=True,
        reason=None,
        provenance={
            "sample_ids": [seed],
            "gallery_det_ids": [seed + 1],
            "embedding_rows": [seed + 2],
            "medoid_sample_id": seed,
            "appearance_quality": 0.9,
            "internal_cosine_p10": 0.8,
            "internal_cosine_p50": 0.9,
            "internal_cosine_min": 0.7,
            "gallery_num_input_samples": 3,
            "gallery_num_overlap_rejected": 0,
            "gallery_num_review_excluded": 0,
            "gallery_num_local_outliers": 0,
            "gallery_max_other_bbox_iou": 0.1,
            "gallery_max_clean_other_bbox_iou": 0.1,
        },
    )


class _Scorer:
    def score_pair(self, source: int, target: int, mode: str) -> SimpleNamespace:
        return SimpleNamespace(
            probability=0.999,
            raw_score=1.0,
            decision="provisional",
            feature_values={
                name: 1.0 if name == "gap_sec" else 0.2
                for name in SHORT_FEATURE_SCHEMA
            },
            reason=None,
            appearance_present=True,
            high_overlap=False,
            source_gallery=_gallery(10),
            target_gallery=_gallery(20),
        )


def _proposal_output(
    directory: Path,
    upstream: list[dict[str, object]],
    *,
    with_edge: bool,
) -> None:
    directory.mkdir()
    if with_edge:
        endpoints = [
            MicroEndpoint(1, -10, -9, "GX040006", "GX040006", 0, 1, 0.0, 1.0),
            MicroEndpoint(2, -8, -7, "GX040006", "GX040006", 2, 3, 2.0, 3.0),
        ]
        candidates = score_short_candidates(
            enumerate_short_candidates(endpoints), _Scorer()
        )
        proposals = build_proposals(candidates)
    else:
        candidates = []
        proposals = []
    pq.write_table(
        pa.Table.from_pylist(candidates, schema=SHORT_CANDIDATE_EDGES_SCHEMA),
        directory / "short_candidate_edges.parquet",
    )
    pq.write_table(
        pa.Table.from_pylist(proposals, schema=SHORT_LINK_PROPOSALS_SCHEMA),
        directory / "short_link_proposals.parquet",
    )
    (directory / "review_manifest.json").write_text(
        json.dumps(review_manifest(proposals)), encoding="utf-8"
    )
    # Deliberately contains a label; finalize must never open or apply it.
    (directory / "review_labels.csv").write_text(
        "proposal_id,review_label,reviewer,notes\nignored,ACCEPT,x,x\n",
        encoding="utf-8",
    )
    (directory / "s04_proposal_report.json").write_text("{}", encoding="utf-8")
    payload = yaml.safe_load(PROPOSAL_CONFIG.read_text(encoding="utf-8"))
    (directory / "effective_config.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    _, _, config_hash = load_short_proposal_config(directory / "effective_config.json")
    output_names = (
        "short_candidate_edges.parquet",
        "short_link_proposals.parquet",
        "review_manifest.json",
        "review_labels.csv",
        "s04_proposal_report.json",
        "effective_config.json",
    )
    marker = {
        "stage": "S04_PROPOSE",
        "config_hash": config_hash,
        "execution_mode": "proposal_only",
        "accepted_decision_for_review": "provisional",
        "automatic_merge_allowed": False,
        "human_labels_applied": False,
        "num_confirmed_edges": 0,
        "num_automatic_merges": 0,
        "input_fingerprints": upstream,
        "output_fingerprints": [
            _fingerprint(directory / name, relative_to=directory) for name in output_names
        ],
    }
    (directory / "_SUCCESS.json").write_text(json.dumps(marker), encoding="utf-8")


def _appearance_row(stable_id: int, micros: list[int]) -> dict[str, object]:
    return {
        "stable_id": stable_id,
        "prototype_row": stable_id,
        "constituent_micro_ids": micros,
        "num_input_samples": 0,
        "num_s02_inliers": 0,
        "num_clean_candidates": 0,
        "num_clean_inliers": 0,
        "num_valid_prototypes": 0,
        "appearance_usable": False,
        "missing_reason": "clean_gallery_missing",
        "clean_sample_ids": [],
        "clean_det_ids": [],
        "clean_embedding_rows": [],
        "medoid_sample_id": None,
        "appearance_quality": None,
        "internal_cosine_p10": None,
        "internal_cosine_p50": None,
        "internal_cosine_min": None,
        "num_overlap_rejected": 0,
        "num_review_excluded": 0,
        "num_local_outliers": None,
        "max_other_bbox_iou": 0.0,
        "max_clean_other_bbox_iou": None,
    }


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, with_edge: bool):
    ingest, micro, appearance = (tmp_path / name for name in ("00", "01", "02"))
    for directory in (ingest, micro, appearance):
        directory.mkdir()
    upstream_file = ingest / "immutable.bin"
    upstream_file.write_bytes(b"upstream")
    calibration = tmp_path / "03"
    calibration.mkdir()
    for name in (
        "_SUCCESS.json",
        "effective_config.json",
        "link_model_short.joblib",
        "link_model_long.joblib",
        "thresholds.json",
        "pair_feature_schema.json",
    ):
        (calibration / name).write_bytes(f"cal-{name}".encode("ascii"))
    upstream = [_fingerprint(upstream_file)] + [
        _fingerprint(calibration / name)
        for name in (
            "_SUCCESS.json",
            "effective_config.json",
            "link_model_short.joblib",
            "link_model_long.joblib",
            "thresholds.json",
            "pair_feature_schema.json",
        )
    ]
    proposals = tmp_path / "04_proposals"
    _proposal_output(proposals, upstream, with_edge=with_edge)
    endpoints = {
        1: SimpleNamespace(
            start_det_id=-10,
            end_det_id=-9,
            start_clip_id="GX040006",
            end_clip_id="GX040006",
            start_global_frame=0,
            end_global_frame=1,
            start_global_time_sec=0.0,
            end_global_time_sec=1.0,
        ),
        2: SimpleNamespace(
            start_det_id=-8,
            end_det_id=-7,
            start_clip_id="GX040006",
            end_clip_id="GX040006",
            start_global_frame=2,
            end_global_frame=3,
            start_global_time_sec=2.0,
            end_global_time_sec=3.0,
        ),
    }
    bundle = SimpleNamespace(
        endpoints=endpoints,
        micro_paths={1: np.asarray([0, 1]), 2: np.asarray([2, 3])},
        detections=SimpleNamespace(
            det_ids=np.asarray([-10, -9, -8, -7], dtype=np.int64),
            micro_ids=np.asarray([1, 1, 2, 2], dtype=np.int64),
            order_in_micro=np.asarray([0, 1, 0, 1], dtype=np.int64),
        ),
        calibration_input=object(),
        input_fingerprints=(_fingerprint(upstream_file),),
    )

    def appearance_builder(data, mapping, config):
        stable_count = len(set(mapping.values()))
        members = {
            stable_id: sorted(micro for micro, value in mapping.items() if value == stable_id)
            for stable_id in range(stable_count)
        }
        return SimpleNamespace(
            stable_ids=np.arange(stable_count, dtype=np.int64),
            stable_prototypes=np.zeros((stable_count, 3, 4), dtype=np.float16),
            stable_prototype_mask=np.zeros((stable_count, 3), dtype=np.bool_),
            stable_appearance_rows=tuple(
                _appearance_row(stable_id, members[stable_id])
                for stable_id in range(stable_count)
            ),
        )

    output = tmp_path / "04_final"

    def invoke(*, builder=appearance_builder):
        return run_s04_finalize(
            ingest,
            micro,
            appearance,
            calibration,
            proposals,
            FINALIZE_CONFIG,
            output,
            logger=lambda _: None,
            runtime_loader=lambda *args, **kwargs: bundle,
            runtime_config_loader=lambda path: SimpleNamespace(
                config=object(), config_hash=S03_HASH
            ),
            runtime_input_validator=lambda path, value: None,
            appearance_builder=builder,
            appearance_schema=STABLE_APPEARANCE_SCHEMA,
        )

    return output, proposals, invoke


def test_finalize_stage_writes_total_component_union_and_appearance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output, _, invoke = _fixture(tmp_path, monkeypatch, with_edge=True)
    marker = invoke()
    assert marker["stage"] == "S04_FINALIZE"
    assert marker["operator_approved"] is True
    assert marker["read_review_labels"] is False
    assert marker["solver_used"] is False
    mapping = pq.read_table(output / "micro_to_stable.parquet")
    stable = pq.read_table(output / "stable_tracklets.parquet")
    detections = pq.read_table(output / "det_to_stable.parquet")
    assert mapping.schema.equals(MICRO_TO_STABLE_SCHEMA, check_metadata=False)
    assert stable.schema.equals(STABLE_TRACKLETS_SCHEMA, check_metadata=False)
    assert detections.schema.equals(DET_TO_STABLE_SCHEMA, check_metadata=False)
    assert mapping.num_rows == 2
    assert stable.num_rows == 1
    assert detections.num_rows == 4
    assert np.load(output / "stable_prototypes.f16.npy").shape == (1, 3, 4)
    report = json.loads((output / "finalize_report.json").read_text(encoding="utf-8"))
    assert report["counts"]["proposal_connected_microtracklets"] == 2
    assert report["counts"]["non_singleton_components"] == 1
    # Resume validates bytes and never rebuilds appearance.
    assert invoke(
        builder=lambda *args: (_ for _ in ()).throw(AssertionError("must not rebuild"))
    ) == marker


def test_finalize_empty_proposals_keeps_all_micros_as_singletons(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output, _, invoke = _fixture(tmp_path, monkeypatch, with_edge=False)
    marker = invoke()
    assert marker["stats"]["proposals_consumed"] == 0
    assert marker["stats"]["stable_tracklets"] == 2
    mapping = pq.read_table(output / "micro_to_stable.parquet").to_pylist()
    assert {(row["micro_id"], row["stable_id"]) for row in mapping} == {(1, 0), (2, 1)}


def test_appearance_membership_is_canonical_when_id_order_differs_from_time() -> None:
    result = SimpleNamespace(
        stable_ids=np.asarray([0], dtype=np.int64),
        stable_prototypes=np.zeros((1, 3, 4), dtype=np.float16),
        stable_prototype_mask=np.zeros((1, 3), dtype=np.bool_),
        stable_appearance_rows=(_appearance_row(0, [10, 20]),),
    )
    _, _, rows = finalize_stage._validate_appearance_result(result, ((20, 10),))
    assert rows[0]["constituent_micro_ids"] == [10, 20]


def test_finalize_rejects_changed_proposal_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, proposals, invoke = _fixture(tmp_path, monkeypatch, with_edge=False)
    path = proposals / "short_link_proposals.parquet"
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ContractError, match="artifact changed"):
        invoke()


def test_finalize_rejects_null_candidate_gap_even_with_matching_fingerprint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, proposals, invoke = _fixture(tmp_path, monkeypatch, with_edge=True)
    candidate_path = proposals / "short_candidate_edges.parquet"
    table = pq.read_table(candidate_path)
    gap_index = table.schema.get_field_index("gap_sec")
    table = table.set_column(
        gap_index,
        table.schema.field(gap_index),
        pa.array([None] * table.num_rows, type=pa.float64()),
    )
    pq.write_table(table, candidate_path)
    marker_path = proposals / "_SUCCESS.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["output_fingerprints"] = [
        _fingerprint(candidate_path, relative_to=proposals)
        if row["path"] == candidate_path.name
        else row
        for row in marker["output_fingerprints"]
    ]
    marker_path.write_text(json.dumps(marker), encoding="utf-8")
    with pytest.raises(ContractError, match="candidate gap_sec"):
        invoke()


def test_finalize_resume_rejects_tampered_marker_statistics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output, _, invoke = _fixture(tmp_path, monkeypatch, with_edge=True)
    invoke()
    marker_path = output / "_SUCCESS.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["num_operator_approved_merges"] = 999
    marker["stats"]["stable_tracklets"] = 999
    marker_path.write_text(json.dumps(marker), encoding="utf-8")
    with pytest.raises(ContractError, match="report/marker statistics differ"):
        invoke()
