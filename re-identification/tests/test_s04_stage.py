from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pyarrow.parquet as pq
import pytest

from cowtrack.config import ContractError
from cowtrack.linking.features import SHORT_FEATURE_SCHEMA
from cowtrack.schemas.s04 import (
    SHORT_CANDIDATE_EDGES_SCHEMA,
    SHORT_LINK_PROPOSALS_SCHEMA,
)
from cowtrack.stages.s04_propose import run_s04_propose


CONFIG = Path(__file__).parents[1] / "configs" / "s04_proposals.yaml"
S03_HASH = "ee6730ca12bde6ffc173ce5bd3a3b327591060deee8124c8a6e99cffeb5a17ad"


def _fingerprint(path: Path) -> dict[str, object]:
    payload = path.read_bytes()
    return {
        "path": str(path.resolve()),
        "size_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _calibration_files(directory: Path) -> None:
    directory.mkdir()
    for name in (
        "_SUCCESS.json",
        "effective_config.json",
        "link_model_short.joblib",
        "link_model_long.joblib",
        "thresholds.json",
        "pair_feature_schema.json",
    ):
        (directory / name).write_bytes(f"synthetic-{name}".encode("ascii"))
    (directory / "thresholds.json").write_text(
        json.dumps(
            {
                "aggressive_global_merge_allowed": False,
                "short": {
                    "model_enabled": True,
                    "confirmed_enabled": False,
                    "confirmed_threshold": None,
                    "provisional_threshold": 0.9958425067703355,
                },
                "long": {"model_enabled": False, "confirmed_enabled": False},
            }
        ),
        encoding="utf-8",
    )


def _input(num_micros: int) -> SimpleNamespace:
    det_ids = []
    micro_ids = []
    orders = []
    clips = []
    frames = []
    times = []
    for micro_id in range(1, num_micros + 1):
        start = float((micro_id - 1) * 2)
        for order in (0, 1):
            det_ids.append(micro_id * 10 + order)
            micro_ids.append(micro_id)
            orders.append(order)
            clips.append("GX040006")
            frames.append((micro_id - 1) * 2 + order)
            times.append(start + order)
    return SimpleNamespace(
        det_ids=det_ids,
        det_micro_ids=micro_ids,
        det_order_in_micro=orders,
        det_clip_ids=clips,
        det_global_frames=frames,
        det_global_time_sec=times,
        parent_micro_ids=list(range(1, num_micros + 1)),
    )


def _provenance(seed: int) -> dict[str, object]:
    return {
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
    }


class _Scorer:
    def score_pair(self, source: int, target: int, mode: str) -> SimpleNamespace:
        assert (source, target, mode) == (1, 2, "short")
        gallery_a = SimpleNamespace(
            present=True, reason=None, provenance=_provenance(100)
        )
        gallery_b = SimpleNamespace(
            present=True, reason=None, provenance=_provenance(200)
        )
        return SimpleNamespace(
            probability=0.999,
            raw_score=1.5,
            decision="provisional",
            feature_values={
                name: 1.0 if name == "gap_sec" else 0.2
                for name in SHORT_FEATURE_SCHEMA
            },
            reason=None,
            appearance_present=True,
            high_overlap=True,
            source_gallery=gallery_a,
            target_gallery=gallery_b,
        )


def _run(tmp_path: Path, num_micros: int):
    ingest = tmp_path / "00"
    micro = tmp_path / "01"
    appearance = tmp_path / "02"
    for directory in (ingest, micro, appearance):
        directory.mkdir()
    upstream = ingest / "input.bin"
    upstream.write_bytes(b"immutable")
    calibration = tmp_path / "03"
    _calibration_files(calibration)
    data = _input(num_micros)
    bundle = SimpleNamespace(
        calibration_input=data,
        input_fingerprints=(_fingerprint(upstream),),
    )
    validated = []

    def validator(directory: Path, received: object) -> None:
        validated.append((directory, received))

    output = tmp_path / "04"
    def invoke(
        *,
        scorer_factory=lambda directory, provider: _Scorer(),
        runtime_config_loader=lambda directory: SimpleNamespace(
            config=object(), config_hash=S03_HASH
        ),
    ):
        return run_s04_propose(
            ingest,
            micro,
            appearance,
            calibration,
            CONFIG,
            output,
            logger=lambda _: None,
            runtime_loader=lambda *args, **kwargs: bundle,
            runtime_config_loader=runtime_config_loader,
            runtime_input_validator=validator,
            feature_store_factory=lambda data, config: object(),
            scorer_factory=scorer_factory,
        )

    marker = invoke()
    assert validated == [(calibration.resolve(), bundle)]
    return output, marker, invoke


def test_stage_writes_proposal_only_artifacts_and_label_per_proposal(
    tmp_path: Path,
) -> None:
    output, marker, invoke = _run(tmp_path, 2)
    assert marker["stage"] == "S04_PROPOSE"
    assert marker["automatic_merge_allowed"] is False
    assert marker["human_labels_applied"] is False
    assert marker["num_confirmed_edges"] == 0
    assert marker["num_automatic_merges"] == 0
    assert not any("stable" in path.name or "solver" in path.name for path in output.iterdir())
    candidates = pq.read_table(output / "short_candidate_edges.parquet")
    proposals = pq.read_table(output / "short_link_proposals.parquet")
    assert candidates.schema.equals(SHORT_CANDIDATE_EDGES_SCHEMA, check_metadata=False)
    assert proposals.schema.equals(SHORT_LINK_PROPOSALS_SCHEMA, check_metadata=False)
    assert candidates.num_rows == proposals.num_rows == 1
    with (output / "review_labels.csv").open(encoding="utf-8", newline="") as handle:
        labels = list(csv.DictReader(handle))
    assert len(labels) == 1
    assert labels[0]["proposal_id"] == proposals["proposal_id"][0].as_py()
    assert labels[0]["review_label"] == ""
    manifest = json.loads((output / "review_manifest.json").read_text(encoding="utf-8"))
    assert manifest["num_proposals"] == 1
    assert manifest["proposals"][0]["proposal_id"] == labels[0]["proposal_id"]
    report = json.loads((output / "s04_proposal_report.json").read_text(encoding="utf-8"))
    assert report["coverage_metric_name"] == "proposal_coverage"
    assert report["counts"]["high_overlap_proposals"] == 1
    # Completed output revalidation must not instantiate or call a scorer.
    revalidated = invoke(
        scorer_factory=lambda directory, provider: (_ for _ in ()).throw(
            AssertionError("completed stage must not score again")
        )
    )
    assert revalidated == marker


def test_empty_candidate_set_still_writes_typed_artifacts(tmp_path: Path) -> None:
    output, marker, _ = _run(tmp_path, 1)
    assert marker["stats"]["num_candidates"] == 0
    assert marker["stats"]["num_proposals"] == 0
    assert pq.read_table(output / "short_candidate_edges.parquet").num_rows == 0
    assert pq.read_table(output / "short_link_proposals.parquet").num_rows == 0
    assert (output / "review_labels.csv").read_text(encoding="utf-8") == (
        "proposal_id,review_label,reviewer,notes\n"
    )


def test_completed_output_rejects_changed_input_fingerprint(tmp_path: Path) -> None:
    _, _, invoke = _run(tmp_path, 1)
    (tmp_path / "00" / "input.bin").write_bytes(b"changed")
    with pytest.raises(ContractError, match="input changed"):
        invoke()


def test_stage_rejects_s03_config_or_threshold_mismatch(tmp_path: Path) -> None:
    _, _, invoke = _run(tmp_path, 1)
    with pytest.raises(ContractError, match="config hash"):
        invoke(
            runtime_config_loader=lambda directory: SimpleNamespace(
                config=object(), config_hash="not-a-sha256"
            )
        )
    thresholds = tmp_path / "03" / "thresholds.json"
    payload = json.loads(thresholds.read_text(encoding="utf-8"))
    payload["short"]["provisional_threshold"] = 1.5
    thresholds.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ContractError, match="provisional threshold"):
        invoke()


def test_stage_rejects_output_tree_overlap(tmp_path: Path) -> None:
    ingest = tmp_path / "input"
    ingest.mkdir()
    with pytest.raises(ContractError, match="overlap"):
        run_s04_propose(
            ingest,
            tmp_path / "micro",
            tmp_path / "appearance",
            tmp_path / "calibration",
            CONFIG,
            ingest,
            logger=lambda _: None,
            runtime_loader=lambda *args, **kwargs: None,
            runtime_config_loader=lambda directory: None,
            runtime_input_validator=lambda directory, bundle: None,
        )
