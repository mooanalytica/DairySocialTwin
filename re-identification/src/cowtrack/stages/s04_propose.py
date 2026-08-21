"""S04 proposal-only short-link stage.

The stage emits review candidates and never builds a matching graph, invokes a
solver, reads human labels, or writes stable-track artifacts.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

from cowtrack.config import ContractError
from cowtrack.linking.proposal_config import (
    ShortProposalConfig,
    load_short_proposal_config,
)
from cowtrack.linking.proposals import (
    build_proposals,
    endpoints_from_calibration_input,
    enumerate_short_candidates,
    review_manifest,
    score_short_candidates,
)
from cowtrack.schemas.s04 import (
    SHORT_CANDIDATE_EDGES_SCHEMA,
    SHORT_LINK_PROPOSALS_SCHEMA,
)


LogFn = Callable[[str], None]
def log(message: str) -> None:
    print(message, flush=True)


def _read_json(path: Path, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read {label} {path}: {exc}") from exc


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    try:
        path.write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
    except OSError as exc:
        raise ContractError(f"cannot write S04 JSON {path}: {exc}") from exc


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise ContractError(f"cannot fingerprint S04 file {path}: {exc}") from exc
    return digest.hexdigest()


def _fingerprint(path: Path, *, relative_to: Path | None = None) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise ContractError(f"required S04 file does not exist: {path}")
    display = str(path.relative_to(relative_to)) if relative_to else str(path)
    return {
        "path": display,
        "size_bytes": int(path.stat().st_size),
        "sha256": _sha256(path),
    }


def _canonical_fingerprints(records: Sequence[Any]) -> list[dict[str, Any]]:
    by_path: dict[str, dict[str, Any]] = {}
    for record in records:
        if hasattr(record, "as_dict"):
            record = record.as_dict()
        elif not isinstance(record, Mapping):
            record = {
                "path": getattr(record, "path", None),
                "size_bytes": getattr(record, "size_bytes", None),
                "sha256": getattr(record, "sha256", None),
            }
        path = record.get("path")
        size = record.get("size_bytes")
        sha = record.get("sha256")
        if (
            not isinstance(path, str)
            or not path
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
            or not isinstance(sha, str)
            or len(sha) != 64
        ):
            raise ContractError("S04 input fingerprint record is invalid")
        resolved = str(Path(path).resolve())
        normalized = {"path": resolved, "size_bytes": size, "sha256": sha}
        if resolved in by_path and by_path[resolved] != normalized:
            raise ContractError("S04 input fingerprint path is duplicated inconsistently")
        by_path[resolved] = normalized
    return [by_path[path] for path in sorted(by_path)]


def _write_parquet(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    schema: pa.Schema,
    compression: str,
) -> None:
    try:
        table = pa.Table.from_pylist([dict(row) for row in rows], schema=schema)
        pq.write_table(table, path, compression=compression, version="2.6")
    except (OSError, pa.ArrowException, TypeError, ValueError) as exc:
        raise ContractError(f"cannot write S04 Parquet {path}: {exc}") from exc


def _write_labels(path: Path, proposals: Sequence[Mapping[str, Any]]) -> None:
    try:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=("proposal_id", "review_label", "reviewer", "notes"),
                lineterminator="\n",
            )
            writer.writeheader()
            for row in sorted(proposals, key=lambda item: str(item["proposal_id"])):
                writer.writerow(
                    {
                        "proposal_id": row["proposal_id"],
                        "review_label": "",
                        "reviewer": "",
                        "notes": "",
                    }
                )
    except (OSError, csv.Error) as exc:
        raise ContractError(f"cannot write S04 review label template: {exc}") from exc


def _clip_distribution(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts = Counter(
        f"{row['source_end_clip_id']}->{row['target_start_clip_id']}" for row in rows
    )
    return dict(sorted(counts.items()))


def _validate_proposal_threshold_policy(
    calibration_dir: Path, s03_runtime: Any
) -> None:
    config_hash = getattr(s03_runtime, "config_hash", None)
    if (
        not isinstance(config_hash, str)
        or len(config_hash) != 64
        or any(character not in "0123456789abcdef" for character in config_hash)
    ):
        raise ContractError("S04 S03 effective config hash is invalid")
    payload = _read_json(calibration_dir / "thresholds.json", "S03 thresholds")
    short = payload.get("short") if isinstance(payload, dict) else None
    long = payload.get("long") if isinstance(payload, dict) else None
    if (
        not isinstance(short, dict)
        or not isinstance(long, dict)
        or payload.get("aggressive_global_merge_allowed") is not False
        or short.get("model_enabled") is not True
        or short.get("confirmed_enabled") is not False
        or short.get("confirmed_threshold") is not None
        or long.get("model_enabled") is not False
        or long.get("confirmed_enabled") is not False
    ):
        raise ContractError(
            "S04 proposal milestone requires enabled provisional-only short "
            "and disabled long/automatic merge"
        )
    threshold = short.get("provisional_threshold")
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not math.isfinite(float(threshold))
        or not 0.0 <= float(threshold) <= 1.0
    ):
        raise ContractError("S04 short provisional threshold is invalid")


def _reject_output_input_overlap(
    output_dir: Path,
    inputs: Sequence[Path],
) -> None:
    output = output_dir.resolve()
    for raw in inputs:
        item = raw.resolve()
        if output == item or output in item.parents or item in output.parents:
            raise ContractError(f"S04 output/input paths overlap: {output} and {item}")


def _report(
    candidates: Sequence[Mapping[str, Any]],
    proposals: Sequence[Mapping[str, Any]],
    input_fingerprints: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    proposal_edges = {str(row["edge_id"]) for row in proposals}
    selected_candidates = [
        row for row in candidates if str(row["edge_id"]) in proposal_edges
    ]
    reasons = Counter(str(row["decision_reason"]) for row in candidates)
    conflict_groups: dict[str, tuple[int, int]] = {}
    for row in proposals:
        conflict_groups[str(row["conflict_group_id"])] = (
            int(row["conflict_group_edge_count"]),
            int(row["conflict_group_node_count"]),
        )
    missing_gallery = sum(
        not bool(row["source_gallery_present"])
        or not bool(row["target_gallery_present"])
        for row in candidates
    )
    missing_history = sum(
        row["motion_history_present"] is False for row in candidates
    )
    review_endpoint = sum(
        "review_excluded_endpoint" in str(row["decision_reason"])
        for row in candidates
    )
    return {
        "schema_version": "1.0",
        "stage": "S04_PROPOSE",
        "execution_mode": "proposal_only",
        "accepted_decision_for_review": "provisional",
        "automatic_merge_allowed": False,
        "human_labels_applied": False,
        "num_confirmed_edges": 0,
        "num_automatic_merges": 0,
        "coverage_metric_name": "proposal_coverage",
        "counts": {
            "candidates": len(candidates),
            "provisional": len(proposals),
            "reject": sum(row["decision"] == "reject" for row in candidates),
            "missing": sum(row["probability"] is None for row in candidates),
            "missing_gallery": missing_gallery,
            "missing_motion_history": missing_history,
            "review_excluded_endpoint": review_endpoint,
            "high_overlap_candidates": sum(
                bool(row["high_overlap"]) for row in candidates
            ),
            "high_overlap_proposals": sum(
                bool(row["high_overlap"]) for row in proposals
            ),
            "cross_clip_candidates": sum(
                row["source_end_clip_id"] != row["target_start_clip_id"]
                for row in candidates
            ),
            "cross_clip_proposals": sum(
                row["source_end_clip_id"] != row["target_start_clip_id"]
                for row in selected_candidates
            ),
        },
        "candidate_decision_reasons": dict(sorted(reasons.items())),
        "candidate_clip_distribution": _clip_distribution(candidates),
        "proposal_clip_distribution": _clip_distribution(selected_candidates),
        "conflict_groups": [
            {
                "conflict_group_id": group_id,
                "edge_count": sizes[0],
                "node_count": sizes[1],
            }
            for group_id, sizes in sorted(conflict_groups.items())
        ],
        "upstream_fingerprints": list(input_fingerprints),
    }


def _expected_names(config: ShortProposalConfig) -> set[str]:
    artifacts = config.artifacts
    return {
        artifacts.candidate_edges,
        artifacts.proposals,
        artifacts.review_manifest,
        artifacts.review_labels,
        artifacts.report,
        artifacts.effective_config,
    }


def _verify_inputs_unchanged(records: Sequence[Mapping[str, Any]]) -> None:
    for record in records:
        current = _fingerprint(Path(str(record["path"])))
        if any(current[key] != record[key] for key in ("size_bytes", "sha256")):
            raise ContractError(f"S04 input changed during proposal stage: {record['path']}")


def _validate_completed(
    output_dir: Path,
    marker: Mapping[str, Any],
    config_hash: str,
    inputs: Sequence[Mapping[str, Any]],
    config: ShortProposalConfig,
) -> None:
    if (
        marker.get("stage") != "S04_PROPOSE"
        or marker.get("config_hash") != config_hash
        or marker.get("execution_mode") != "proposal_only"
        or marker.get("automatic_merge_allowed") is not False
        or marker.get("human_labels_applied") is not False
        or marker.get("num_confirmed_edges") != 0
        or marker.get("num_automatic_merges") != 0
    ):
        raise ContractError("completed S04 proposal marker policy/config differs")
    recorded_inputs = _canonical_fingerprints(marker.get("input_fingerprints", []))
    if recorded_inputs != list(inputs):
        raise ContractError("completed S04 proposal input fingerprints differ")
    records = marker.get("output_fingerprints")
    if not isinstance(records, list):
        raise ContractError("completed S04 proposal marker lacks output fingerprints")
    by_name = {str(item.get("path")): item for item in records if isinstance(item, dict)}
    expected = _expected_names(config)
    if set(by_name) != expected or len(by_name) != len(records):
        raise ContractError("completed S04 proposal output fingerprint set differs")
    actual = {
        str(path.relative_to(output_dir))
        for path in output_dir.rglob("*")
        if path.is_file() and path.name != config.artifacts.success
    }
    if actual != expected:
        raise ContractError("completed S04 proposal artifact set differs")
    for name, record in by_name.items():
        current = _fingerprint(output_dir / name, relative_to=output_dir)
        if any(current[key] != record.get(key) for key in ("size_bytes", "sha256")):
            raise ContractError(f"completed S04 artifact changed: {name}")
    schemas = (
        (config.artifacts.candidate_edges, SHORT_CANDIDATE_EDGES_SCHEMA),
        (config.artifacts.proposals, SHORT_LINK_PROPOSALS_SCHEMA),
    )
    for name, schema in schemas:
        if not pq.ParquetFile(output_dir / name).schema_arrow.equals(
            schema, check_metadata=False
        ):
            raise ContractError(f"completed S04 Parquet schema changed: {name}")


def run_s04_propose(
    ingest_dir: Path,
    microtrack_dir: Path,
    appearance_dir: Path,
    calibration_dir: Path,
    config_path: Path,
    output_dir: Path,
    *,
    logger: LogFn = log,
    runtime_loader: Callable[..., Any] | None = None,
    runtime_config_loader: Callable[..., Any] | None = None,
    runtime_input_validator: Callable[[Path, Any], None] | None = None,
    scorer_factory: Callable[[Path, Any], Any] | None = None,
    feature_store_factory: Callable[[Any, Any], Any] | None = None,
) -> dict[str, Any]:
    """Generate all structural candidates and provisional review proposals."""

    started = time.monotonic()
    ingest_dir = ingest_dir.resolve()
    microtrack_dir = microtrack_dir.resolve()
    appearance_dir = appearance_dir.resolve()
    calibration_dir = calibration_dir.resolve()
    config_path = config_path.resolve()
    output_dir = output_dir.resolve()
    config, config_payload, config_hash = load_short_proposal_config(config_path)
    _reject_output_input_overlap(
        output_dir,
        (ingest_dir, microtrack_dir, appearance_dir, calibration_dir, config_path),
    )

    if (
        runtime_loader is None
        or runtime_config_loader is None
        or runtime_input_validator is None
    ):
        from cowtrack.linking.runtime import (
            load_production_inputs,
            load_s03_runtime_config,
            validate_s03_input_fingerprints,
        )

        runtime_loader = runtime_loader or load_production_inputs
        runtime_config_loader = runtime_config_loader or load_s03_runtime_config
        runtime_input_validator = (
            runtime_input_validator or validate_s03_input_fingerprints
        )
    logger("[s04-propose] validating immutable S00/S01/S02/S03 inputs")
    bundle = runtime_loader(
        ingest_dir, microtrack_dir, appearance_dir, logger=logger
    )
    s03_runtime = runtime_config_loader(calibration_dir)
    if not hasattr(s03_runtime, "config"):
        raise ContractError("S04 public S03 runtime config lacks validated config")
    runtime_input_validator(calibration_dir, bundle)
    s03_config = s03_runtime.config
    _validate_proposal_threshold_policy(calibration_dir, s03_runtime)
    runtime_fingerprints = getattr(bundle, "input_fingerprints", None)
    if runtime_fingerprints is None:
        raise ContractError("S04 public runtime bundle lacks input fingerprints")
    calibration_paths = (
        calibration_dir / "_SUCCESS.json",
        calibration_dir / "effective_config.json",
        calibration_dir / "link_model_short.joblib",
        calibration_dir / "link_model_long.joblib",
        calibration_dir / "thresholds.json",
        calibration_dir / "pair_feature_schema.json",
    )
    inputs = _canonical_fingerprints(
        [*runtime_fingerprints, _fingerprint(config_path)]
        + [_fingerprint(path) for path in calibration_paths]
    )
    _verify_inputs_unchanged(inputs)
    success_path = output_dir / config.artifacts.success
    if success_path.is_file():
        marker = _read_json(success_path, "S04 proposal success marker")
        if not isinstance(marker, dict):
            raise ContractError("S04 proposal success marker must be an object")
        _validate_completed(output_dir, marker, config_hash, inputs, config)
        logger(f"[s04-propose] already complete and revalidated: {success_path}")
        return dict(marker)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ContractError(
            f"output directory is non-empty without _SUCCESS.json: {output_dir}"
        )

    if feature_store_factory is None:
        from cowtrack.linking.pseudo_pairs import ProductionFeatureStore

        feature_store_factory = ProductionFeatureStore
    if scorer_factory is None:
        from cowtrack.linking.scorer import CalibratedLinkScorer

        scorer_factory = CalibratedLinkScorer.from_directory
    provider = feature_store_factory(bundle.calibration_input, s03_config)
    scorer = scorer_factory(calibration_dir, provider)
    endpoints = endpoints_from_calibration_input(bundle.calibration_input)
    logger(f"[s04-propose] enumerating short windows for {len(endpoints):,} micros")
    structural = enumerate_short_candidates(
        endpoints, max_gap_sec=config.max_gap_sec, clip_order=config.clip_order
    )
    logger(f"[s04-propose] scoring {len(structural):,} structural candidates")
    candidates = score_short_candidates(
        structural,
        scorer,
        progress=lambda completed, total: logger(
            f"[s04-propose] scoring progress: {completed:,}/{total:,}"
        ),
        progress_interval_sec=config.progress_interval_sec,
    )
    proposals = build_proposals(candidates)
    if any(row["decision"] == "confirmed" for row in candidates):
        raise ContractError("fixed S04 artifact unexpectedly produced a confirmed edge")
    if any(row["decision"] != "provisional" for row in candidates if row["proposed_for_review"]):
        raise ContractError("S04 review proposal includes a non-provisional decision")
    manifest = review_manifest(proposals)
    report = _report(candidates, proposals, inputs)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = output_dir.parent / f".{output_dir.name}.staging-{os.getpid()}"
    if staging.exists():
        raise ContractError(f"S04 staging directory already exists: {staging}")
    staging.mkdir(parents=False)
    try:
        artifacts = config.artifacts
        _write_parquet(
            staging / artifacts.candidate_edges,
            candidates,
            SHORT_CANDIDATE_EDGES_SCHEMA,
            config.parquet_compression,
        )
        _write_parquet(
            staging / artifacts.proposals,
            proposals,
            SHORT_LINK_PROPOSALS_SCHEMA,
            config.parquet_compression,
        )
        _write_json(staging / artifacts.review_manifest, manifest)
        _write_labels(staging / artifacts.review_labels, proposals)
        _write_json(staging / artifacts.report, report)
        _write_json(staging / artifacts.effective_config, config_payload)
        _verify_inputs_unchanged(inputs)
        outputs = [
            _fingerprint(staging / name, relative_to=staging)
            for name in sorted(_expected_names(config))
        ]
        marker = {
            "schema_version": config.schema_version,
            "stage": "S04_PROPOSE",
            "config_hash": config_hash,
            "execution_mode": "proposal_only",
            "accepted_decision_for_review": "provisional",
            "automatic_merge_allowed": False,
            "human_labels_applied": False,
            "num_confirmed_edges": 0,
            "num_automatic_merges": 0,
            "input_fingerprints": inputs,
            "output_fingerprints": outputs,
            "stats": {
                "num_candidates": len(candidates),
                "num_proposals": len(proposals),
                "num_rejects": sum(row["decision"] == "reject" for row in candidates),
            },
            "elapsed_sec": float(time.monotonic() - started),
        }
        _write_json(staging / artifacts.success, marker)
        if output_dir.exists():
            try:
                output_dir.rmdir()
            except OSError as exc:
                raise ContractError(
                    f"S04 output directory cannot be atomically committed: {output_dir}"
                ) from exc
        os.replace(staging, output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    logger(f"[s04-propose] complete: {output_dir}")
    return marker


__all__ = ["run_s04_propose"]
