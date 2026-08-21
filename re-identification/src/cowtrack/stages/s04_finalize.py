"""Operator-approved S04 component-union finalization stage."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from cowtrack.config import ContractError
from cowtrack.linking.finalize import (
    FinalizeMicro,
    FinalizeProposal,
    build_component_union,
    build_det_to_stable_rows,
)
from cowtrack.linking.finalize_config import (
    S04FinalizeConfig,
    load_s04_finalize_config,
)
from cowtrack.linking.proposal_config import load_short_proposal_config
from cowtrack.schemas.s04 import (
    DET_TO_STABLE_SCHEMA,
    MICRO_TO_STABLE_SCHEMA,
    SHORT_CANDIDATE_EDGES_SCHEMA,
    SHORT_LINK_PROPOSALS_SCHEMA,
    STABLE_TRACKLETS_SCHEMA,
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
        raise ContractError(f"cannot write S04 finalize JSON {path}: {exc}") from exc


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise ContractError(f"cannot fingerprint S04 finalize file {path}: {exc}") from exc
    return digest.hexdigest()


def _fingerprint(path: Path, *, relative_to: Path | None = None) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise ContractError(f"required S04 finalize file does not exist: {path}")
    return {
        "path": str(path.relative_to(relative_to)) if relative_to else str(path),
        "size_bytes": int(path.stat().st_size),
        "sha256": _sha256(path),
    }


def _normalize_fingerprints(records: Sequence[Any]) -> list[dict[str, Any]]:
    by_path: dict[str, dict[str, Any]] = {}
    for value in records:
        if hasattr(value, "as_dict"):
            value = value.as_dict()
        if not isinstance(value, Mapping):
            raise ContractError("S04 finalize fingerprint record is invalid")
        path = value.get("path")
        size = value.get("size_bytes")
        sha = value.get("sha256")
        if (
            not isinstance(path, str)
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
            or not isinstance(sha, str)
            or len(sha) != 64
        ):
            raise ContractError("S04 finalize fingerprint fields are invalid")
        resolved = str(Path(path).resolve())
        row = {"path": resolved, "size_bytes": size, "sha256": sha}
        if resolved in by_path and by_path[resolved] != row:
            raise ContractError("S04 finalize fingerprint path is inconsistent")
        by_path[resolved] = row
    return [by_path[path] for path in sorted(by_path)]


def _verify_unchanged(records: Sequence[Mapping[str, Any]]) -> None:
    for expected in records:
        current = _fingerprint(Path(str(expected["path"])))
        if any(current[key] != expected[key] for key in ("size_bytes", "sha256")):
            raise ContractError(f"S04 finalize input changed: {expected['path']}")


def _reject_overlap(output_dir: Path, inputs: Sequence[Path]) -> None:
    output = output_dir.resolve()
    for raw in inputs:
        item = raw.resolve()
        if output == item or output in item.parents or item in output.parents:
            raise ContractError(f"S04 finalize output/input paths overlap: {output}, {item}")


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
        raise ContractError(f"cannot write S04 finalize Parquet {path}: {exc}") from exc


def _read_parquet(path: Path, schema: pa.Schema, label: str) -> pa.Table:
    try:
        parquet = pq.ParquetFile(path)
        if not parquet.schema_arrow.equals(schema, check_metadata=False):
            raise ContractError(f"{label} schema mismatch: {path}")
        return pq.read_table(path)
    except ContractError:
        raise
    except (OSError, pa.ArrowException, TypeError, ValueError) as exc:
        raise ContractError(f"cannot read {label} {path}: {exc}") from exc


def _column(table: pa.Table, name: str) -> list[Any]:
    return table[name].combine_chunks().to_pylist()


def _proposal_inputs(
    proposal_dir: Path,
    current_upstream: Sequence[Mapping[str, Any]],
) -> tuple[list[FinalizeProposal], list[dict[str, Any]]]:
    marker_path = proposal_dir / "_SUCCESS.json"
    marker = _read_json(marker_path, "S04 proposal success marker")
    if (
        not isinstance(marker, dict)
        or marker.get("stage") != "S04_PROPOSE"
        or marker.get("execution_mode") != "proposal_only"
        or marker.get("accepted_decision_for_review") != "provisional"
        or marker.get("automatic_merge_allowed") is not False
        or marker.get("human_labels_applied") is not False
        or marker.get("num_confirmed_edges") != 0
        or marker.get("num_automatic_merges") != 0
    ):
        raise ContractError("S04 finalize proposal marker policy differs")
    effective = proposal_dir / "effective_config.json"
    _, _, proposal_config_hash = load_short_proposal_config(effective)
    if marker.get("config_hash") != proposal_config_hash:
        raise ContractError("S04 proposal effective config hash differs")

    records = marker.get("output_fingerprints")
    if not isinstance(records, list):
        raise ContractError("S04 proposal marker lacks output fingerprints")
    by_name = {
        str(record.get("path")): record for record in records if isinstance(record, dict)
    }
    # review_labels.csv is intentionally neither opened nor fingerprinted.
    consumed_names = (
        "short_candidate_edges.parquet",
        "short_link_proposals.parquet",
        "review_manifest.json",
        "s04_proposal_report.json",
        "effective_config.json",
    )
    if any(name not in by_name for name in consumed_names):
        raise ContractError("S04 proposal marker lacks a finalize input artifact")
    consumed = [_fingerprint(marker_path)]
    for name in consumed_names:
        current = _fingerprint(proposal_dir / name)
        if any(
            current[key] != by_name[name].get(key) for key in ("size_bytes", "sha256")
        ):
            raise ContractError(f"completed S04 proposal artifact changed: {name}")
        consumed.append(current)

    recorded_inputs = {
        str(Path(str(record.get("path"))).resolve()): record
        for record in marker.get("input_fingerprints", [])
        if isinstance(record, dict) and isinstance(record.get("path"), str)
    }
    for current in current_upstream:
        recorded = recorded_inputs.get(str(current["path"]))
        if recorded is None or any(
            current[key] != recorded.get(key) for key in ("size_bytes", "sha256")
        ):
            raise ContractError(
                "S04 proposal was not built from current S00/S01/S02/S03 bytes"
            )

    candidate_table = _read_parquet(
        proposal_dir / "short_candidate_edges.parquet",
        SHORT_CANDIDATE_EDGES_SCHEMA,
        "S04 candidate edges",
    )
    proposal_table = _read_parquet(
        proposal_dir / "short_link_proposals.parquet",
        SHORT_LINK_PROPOSALS_SCHEMA,
        "S04 proposals",
    )
    candidate_values = candidate_table.to_pylist()
    for row in candidate_values:
        gap = row["gap_sec"]
        if (
            gap is None
            or isinstance(gap, bool)
            or not math.isfinite(float(gap))
            or not 0.0 < float(gap) <= 5.0
        ):
            raise ContractError("S04 candidate gap_sec must be finite and in (0, 5]")
    candidate_rows = {str(row["edge_id"]): row for row in candidate_values}
    if len(candidate_rows) != candidate_table.num_rows:
        raise ContractError("S04 candidate edge IDs are not unique")
    expected_edges = {
        edge_id
        for edge_id, row in candidate_rows.items()
        if row["decision"] == "provisional" and row["proposed_for_review"] is True
    }
    proposals: list[FinalizeProposal] = []
    seen_edges: set[str] = set()
    seen_ids: set[str] = set()
    for row in proposal_table.to_pylist():
        proposal_id = str(row["proposal_id"])
        edge_id = str(row["edge_id"])
        if (
            proposal_id in seen_ids
            or edge_id in seen_edges
            or row["review_status"] != "pending"
        ):
            raise ContractError("S04 proposal rows are duplicated or non-pending")
        seen_ids.add(proposal_id)
        seen_edges.add(edge_id)
        candidate = candidate_rows.get(edge_id)
        if (
            candidate is None
            or candidate["decision"] != "provisional"
            or candidate["proposed_for_review"] is not True
            or int(candidate["source_micro_id"]) != int(row["source_micro_id"])
            or int(candidate["target_micro_id"]) != int(row["target_micro_id"])
            or float(candidate["probability"]) != float(row["probability"])
        ):
            raise ContractError("S04 proposal/candidate evidence differs")
        proposals.append(
            FinalizeProposal(
                proposal_id,
                edge_id,
                int(row["source_micro_id"]),
                int(row["target_micro_id"]),
                float(row["probability"]),
            )
        )
    if seen_edges != expected_edges:
        raise ContractError("S04 finalize must consume every provisional proposal")
    proposals.sort(key=lambda proposal: proposal.proposal_id)
    return proposals, consumed


def _micros_from_bundle(bundle: Any) -> list[FinalizeMicro]:
    micros: list[FinalizeMicro] = []
    endpoints = bundle.endpoints
    paths = bundle.micro_paths
    if set(endpoints) != set(paths):
        raise ContractError("S04 finalize runtime endpoint/path ID sets differ")
    for micro_id in sorted(endpoints):
        endpoint = endpoints[micro_id]
        micros.append(
            FinalizeMicro(
                micro_id=int(micro_id),
                start_det_id=int(endpoint.start_det_id),
                end_det_id=int(endpoint.end_det_id),
                start_clip_id=str(endpoint.start_clip_id),
                end_clip_id=str(endpoint.end_clip_id),
                start_global_frame=int(endpoint.start_global_frame),
                end_global_frame=int(endpoint.end_global_frame),
                start_time_sec=float(endpoint.start_global_time_sec),
                end_time_sec=float(endpoint.end_global_time_sec),
                num_detections=len(paths[micro_id]),
            )
        )
    return micros


def _expected_names(config: S04FinalizeConfig) -> set[str]:
    artifacts = config.artifacts
    return {
        artifacts.micro_to_stable,
        artifacts.stable_tracklets,
        artifacts.det_to_stable,
        artifacts.stable_prototypes,
        artifacts.stable_prototype_mask,
        artifacts.stable_appearance,
        artifacts.report,
        artifacts.effective_config,
    }


def _validate_appearance_result(
    result: Any, components: Sequence[Sequence[int]]
) -> tuple[np.ndarray, np.ndarray, Sequence[Mapping[str, Any]]]:
    stable_count = len(components)
    stable_ids = np.asarray(result.stable_ids)
    prototypes = np.asarray(result.stable_prototypes)
    mask = np.asarray(result.stable_prototype_mask)
    rows = result.stable_appearance_rows
    if not np.array_equal(stable_ids, np.arange(stable_count, dtype=np.int64)):
        raise ContractError("S04 stable appearance IDs are not dense/canonical")
    if (
        prototypes.dtype != np.float16
        or prototypes.ndim != 3
        or prototypes.shape[0] != stable_count
        or prototypes.shape[1] != 3
        or prototypes.shape[2] <= 0
        or not np.isfinite(prototypes).all()
        or mask.dtype != np.bool_
        or mask.shape != prototypes.shape[:2]
        or len(rows) != stable_count
    ):
        raise ContractError("S04 stable appearance array/row contract differs")
    row_ids = [int(row["stable_id"]) for row in rows]
    if (
        row_ids != list(range(stable_count))
        or [int(row["prototype_row"]) for row in rows] != list(range(stable_count))
        or [tuple(map(int, row["constituent_micro_ids"])) for row in rows]
        != [tuple(sorted(map(int, component))) for component in components]
        or any(
            int(row["num_valid_prototypes"]) != int(np.count_nonzero(mask[index]))
            or bool(row["appearance_usable"]) != bool(np.any(mask[index]))
            for index, row in enumerate(rows)
        )
    ):
        raise ContractError("S04 stable appearance rows/membership are not canonical")
    if np.any(prototypes[~mask] != np.float16(0.0)):
        raise ContractError("S04 invalid stable prototype slots must be zero")
    if np.any(mask):
        norms = np.linalg.norm(prototypes.astype(np.float32), axis=2)[mask]
        if not np.allclose(norms, 1.0, rtol=0.0, atol=2e-3):
            raise ContractError("S04 valid stable prototypes must be L2-normalized")
    return prototypes, mask, rows


def _report(
    *,
    union: Any,
    proposals: Sequence[FinalizeProposal],
    appearance_rows: Sequence[Mapping[str, Any]],
    inputs: Sequence[Mapping[str, Any]],
    detection_count: int,
) -> dict[str, Any]:
    sizes = [len(component) for component in union.components]
    connected = {
        micro_id
        for proposal in proposals
        for micro_id in (proposal.source_micro_id, proposal.target_micro_id)
    }
    usable = sum(bool(row.get("appearance_usable")) for row in appearance_rows)
    return {
        "schema_version": "1.0",
        "stage": "S04_FINALIZE",
        "execution_mode": "operator_approved_component_union",
        "operator_approved": True,
        "merge_policy": "all_provisional_undirected_components",
        "graph_node_identity": "actual_micro_id",
        "automatic_merge_allowed": False,
        "num_automatic_merges": 0,
        "num_operator_approved_merges": len(union.micro_to_stable) - len(union.components),
        "read_review_labels": False,
        "solver_used": False,
        "counts": {
            "microtracklets": len(union.micro_to_stable),
            "detections": detection_count,
            "proposals_consumed": len(proposals),
            "proposal_connected_microtracklets": len(connected),
            "stable_tracklets": len(union.components),
            "non_singleton_components": sum(size > 1 for size in sizes),
            "singleton_components": sum(size == 1 for size in sizes),
            "max_component_size": max(sizes, default=0),
            "stable_appearance_usable": usable,
            "stable_appearance_missing": len(appearance_rows) - usable,
        },
        "component_size_distribution": dict(
            sorted((str(size), count) for size, count in Counter(sizes).items())
        ),
        "input_fingerprints": list(inputs),
    }


def _validate_completed(
    output_dir: Path,
    marker: Mapping[str, Any],
    config_hash: str,
    inputs: Sequence[Mapping[str, Any]],
    config: S04FinalizeConfig,
    appearance_schema: pa.Schema,
) -> None:
    if (
        marker.get("stage") != "S04_FINALIZE"
        or marker.get("config_hash") != config_hash
        or marker.get("execution_mode") != "operator_approved_component_union"
        or marker.get("operator_approved") is not True
        or marker.get("merge_policy") != "all_provisional_undirected_components"
        or marker.get("automatic_merge_allowed") is not False
        or marker.get("num_automatic_merges") != 0
        or marker.get("read_review_labels") is not False
        or marker.get("solver_used") is not False
    ):
        raise ContractError("completed S04 finalize marker policy/config differs")
    if _normalize_fingerprints(marker.get("input_fingerprints", [])) != list(inputs):
        raise ContractError("completed S04 finalize input fingerprints differ")
    records = marker.get("output_fingerprints")
    if not isinstance(records, list):
        raise ContractError("completed S04 finalize marker lacks fingerprints")
    by_name = {str(row.get("path")): row for row in records if isinstance(row, dict)}
    expected = _expected_names(config)
    if set(by_name) != expected or len(by_name) != len(records):
        raise ContractError("completed S04 finalize output set differs")
    actual = {
        str(path.relative_to(output_dir))
        for path in output_dir.rglob("*")
        if path.is_file() and path.name != config.artifacts.success
    }
    if actual != expected:
        raise ContractError("completed S04 finalize artifact tree differs")
    for name, record in by_name.items():
        current = _fingerprint(output_dir / name, relative_to=output_dir)
        if any(current[key] != record.get(key) for key in ("size_bytes", "sha256")):
            raise ContractError(f"completed S04 finalize artifact changed: {name}")
    for name, schema in (
        (config.artifacts.micro_to_stable, MICRO_TO_STABLE_SCHEMA),
        (config.artifacts.stable_tracklets, STABLE_TRACKLETS_SCHEMA),
        (config.artifacts.det_to_stable, DET_TO_STABLE_SCHEMA),
        (config.artifacts.stable_appearance, appearance_schema),
    ):
        if not pq.ParquetFile(output_dir / name).schema_arrow.equals(
            schema, check_metadata=False
        ):
            raise ContractError(f"completed S04 finalize schema changed: {name}")
    mapping = pq.read_table(
        output_dir / config.artifacts.micro_to_stable,
        columns=[
            "micro_id",
            "stable_id",
            "order_in_stable",
            "component_num_proposal_edges",
        ],
    ).to_pylist()
    stable = pq.read_table(
        output_dir / config.artifacts.stable_tracklets,
        columns=["stable_id", "num_microtracklets", "num_proposal_edges"],
    ).to_pylist()
    detections = pq.read_table(
        output_dir / config.artifacts.det_to_stable,
        columns=["det_id", "micro_id", "stable_id"],
    ).to_pylist()
    appearance = pq.read_table(
        output_dir / config.artifacts.stable_appearance,
        columns=[
            "stable_id",
            "prototype_row",
            "constituent_micro_ids",
            "num_valid_prototypes",
            "appearance_usable",
        ],
    ).to_pylist()
    stable_ids = list(range(len(stable)))
    mapping_members: dict[int, list[tuple[int, int]]] = {}
    for row in mapping:
        mapping_members.setdefault(int(row["stable_id"]), []).append(
            (int(row["order_in_stable"]), int(row["micro_id"]))
        )
    expected_members = [
        sorted(micro for _, micro in mapping_members[stable_id])
        for stable_id in stable_ids
    ]
    if (
        [int(row["stable_id"]) for row in stable] != stable_ids
        or [int(row["stable_id"]) for row in appearance] != stable_ids
        or [int(row["prototype_row"]) for row in appearance] != stable_ids
        or [list(map(int, row["constituent_micro_ids"])) for row in appearance]
        != expected_members
        or not mapping
        or not detections
        or len({int(row["micro_id"]) for row in mapping}) != len(mapping)
        or set(int(row["stable_id"]) for row in mapping) != set(stable_ids)
        or sum(int(row["num_microtracklets"]) for row in stable) != len(mapping)
        or len({int(row["det_id"]) for row in detections}) != len(detections)
        or set(int(row["micro_id"]) for row in detections)
        != {int(row["micro_id"]) for row in mapping}
        or set(int(row["stable_id"]) for row in detections) != set(stable_ids)
    ):
        raise ContractError("completed S04 finalize mapping invariants differ")
    try:
        prototypes = np.load(
            output_dir / config.artifacts.stable_prototypes,
            mmap_mode="r",
            allow_pickle=False,
        )
        mask = np.load(
            output_dir / config.artifacts.stable_prototype_mask,
            mmap_mode="r",
            allow_pickle=False,
        )
    except (OSError, ValueError) as exc:
        raise ContractError(f"cannot reload completed stable appearance arrays: {exc}") from exc
    if (
        prototypes.dtype != np.float16
        or prototypes.ndim != 3
        or prototypes.shape[0] != len(stable_ids)
        or prototypes.shape[1] != 3
        or prototypes.shape[2] <= 0
        or not np.isfinite(prototypes).all()
        or mask.dtype != np.bool_
        or mask.shape != prototypes.shape[:2]
        or any(
            int(row["num_valid_prototypes"]) != int(np.count_nonzero(mask[index]))
            or bool(row["appearance_usable"]) != bool(np.any(mask[index]))
            for index, row in enumerate(appearance)
        )
        or np.any(prototypes[~mask] != np.float16(0.0))
    ):
        raise ContractError("completed S04 stable appearance arrays differ")
    if np.any(mask):
        norms = np.linalg.norm(prototypes.astype(np.float32), axis=2)[mask]
        if not np.allclose(norms, 1.0, rtol=0.0, atol=2e-3):
            raise ContractError("completed S04 stable prototypes are not normalized")
    sizes = [int(row["num_microtracklets"]) for row in stable]
    computed_stats = {
        "microtracklets": len(mapping),
        "detections": len(detections),
        "proposals_consumed": sum(int(row["num_proposal_edges"]) for row in stable),
        "proposal_connected_microtracklets": sum(
            int(row["component_num_proposal_edges"]) > 0 for row in mapping
        ),
        "stable_tracklets": len(stable),
        "non_singleton_components": sum(size > 1 for size in sizes),
        "singleton_components": sum(size == 1 for size in sizes),
        "max_component_size": max(sizes, default=0),
        "stable_appearance_usable": sum(
            bool(row["appearance_usable"]) for row in appearance
        ),
        "stable_appearance_missing": sum(
            not bool(row["appearance_usable"]) for row in appearance
        ),
    }
    approved_merges = len(mapping) - len(stable)
    report = _read_json(
        output_dir / config.artifacts.report, "S04 finalize report"
    )
    if (
        marker.get("num_operator_approved_merges") != approved_merges
        or marker.get("stats") != computed_stats
        or not isinstance(report, dict)
        or report.get("stage") != "S04_FINALIZE"
        or report.get("execution_mode") != "operator_approved_component_union"
        or report.get("operator_approved") is not True
        or report.get("merge_policy")
        != "all_provisional_undirected_components"
        or report.get("automatic_merge_allowed") is not False
        or report.get("num_automatic_merges") != 0
        or report.get("num_operator_approved_merges") != approved_merges
        or report.get("read_review_labels") is not False
        or report.get("solver_used") is not False
        or report.get("counts") != computed_stats
        or _normalize_fingerprints(report.get("input_fingerprints", []))
        != list(inputs)
        or report.get("component_size_distribution")
        != dict(sorted((str(size), count) for size, count in Counter(sizes).items()))
    ):
        raise ContractError("completed S04 finalize report/marker statistics differ")


def run_s04_finalize(
    ingest_dir: Path,
    microtrack_dir: Path,
    appearance_dir: Path,
    calibration_dir: Path,
    proposal_dir: Path,
    config_path: Path,
    output_dir: Path,
    *,
    logger: LogFn = log,
    runtime_loader: Callable[..., Any] | None = None,
    runtime_config_loader: Callable[[Path], Any] | None = None,
    runtime_input_validator: Callable[[Path, Any], None] | None = None,
    appearance_builder: Callable[[Any, Mapping[int, int], Any], Any] | None = None,
    appearance_schema: pa.Schema | None = None,
) -> dict[str, Any]:
    """Merge every provisional connected component and rebuild appearance."""

    started = time.monotonic()
    ingest_dir = ingest_dir.resolve()
    microtrack_dir = microtrack_dir.resolve()
    appearance_dir = appearance_dir.resolve()
    calibration_dir = calibration_dir.resolve()
    proposal_dir = proposal_dir.resolve()
    config_path = config_path.resolve()
    output_dir = output_dir.resolve()
    config, config_payload, config_hash = load_s04_finalize_config(config_path)
    _reject_overlap(
        output_dir,
        (
            ingest_dir,
            microtrack_dir,
            appearance_dir,
            calibration_dir,
            proposal_dir,
            config_path,
        ),
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
        runtime_input_validator = runtime_input_validator or validate_s03_input_fingerprints
    use_default_appearance_builder = appearance_builder is None
    if appearance_builder is None or appearance_schema is None:
        from cowtrack.linking.stable_appearance import build_stable_appearance
        from cowtrack.schemas.stable_appearance import STABLE_APPEARANCE_SCHEMA

        appearance_builder = appearance_builder or build_stable_appearance
        appearance_schema = appearance_schema or STABLE_APPEARANCE_SCHEMA

    logger("[s04-finalize] validating immutable S00/S01/S02/S03 inputs")
    bundle = runtime_loader(ingest_dir, microtrack_dir, appearance_dir, logger=logger)
    s03_runtime = runtime_config_loader(calibration_dir)
    s03_hash = getattr(s03_runtime, "config_hash", None)
    if (
        not isinstance(s03_hash, str)
        or len(s03_hash) != 64
        or any(character not in "0123456789abcdef" for character in s03_hash)
    ):
        raise ContractError("S04 finalize S03 config hash is invalid")
    runtime_input_validator(calibration_dir, bundle)
    calibration_paths = (
        calibration_dir / "_SUCCESS.json",
        calibration_dir / "effective_config.json",
        calibration_dir / "link_model_short.joblib",
        calibration_dir / "link_model_long.joblib",
        calibration_dir / "thresholds.json",
        calibration_dir / "pair_feature_schema.json",
    )
    upstream = _normalize_fingerprints(
        [*bundle.input_fingerprints]
        + [_fingerprint(path) for path in calibration_paths]
    )
    proposals, proposal_inputs = _proposal_inputs(proposal_dir, upstream)
    inputs = _normalize_fingerprints(
        [*upstream, *proposal_inputs, _fingerprint(config_path)]
    )
    _verify_unchanged(inputs)
    success_path = output_dir / config.artifacts.success
    if success_path.is_file():
        marker = _read_json(success_path, "S04 finalize success marker")
        if not isinstance(marker, dict):
            raise ContractError("S04 finalize marker must be an object")
        _validate_completed(
            output_dir, marker, config_hash, inputs, config, appearance_schema
        )
        logger(f"[s04-finalize] already complete and revalidated: {success_path}")
        return dict(marker)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ContractError(
            f"output directory is non-empty without _SUCCESS.json: {output_dir}"
        )

    micros = _micros_from_bundle(bundle)
    if not micros:
        raise ContractError("S04 finalize requires at least one microtracklet")
    detection = bundle.detections
    if not len(detection.det_ids):
        raise ContractError("S04 finalize requires at least one valid detection")
    expected_by_micro = {micro.micro_id: micro.num_detections for micro in micros}
    observed_by_micro = Counter(map(int, detection.micro_ids))
    if observed_by_micro != Counter(expected_by_micro):
        raise ContractError(
            "S04 finalize endpoint paths and det-to-micro counts differ"
        )
    logger(
        f"[s04-finalize] unioning {len(proposals):,} proposals across "
        f"{len(micros):,} micros"
    )
    union = build_component_union(micros, proposals)
    det_rows = build_det_to_stable_rows(
        det_ids=detection.det_ids,
        micro_ids=detection.micro_ids,
        order_in_micro=detection.order_in_micro,
        micro_to_stable_rows=union.micro_to_stable_rows,
    )
    if (
        len(det_rows) != len(detection.det_ids)
        or {int(row["det_id"]) for row in det_rows}
        != set(map(int, detection.det_ids))
    ):
        raise ContractError("S04 finalize detection mapping is not an exact bijection")
    logger(f"[s04-finalize] rebuilding {len(union.components):,} stable appearances")
    if use_default_appearance_builder:
        appearance_result = appearance_builder(
            bundle.calibration_input,
            union.micro_to_stable,
            s03_runtime.config,
            progress_callback=lambda completed, total: logger(
                f"[s04-finalize] appearance progress: {completed:,}/{total:,}"
            ),
            progress_interval_sec=config.progress_interval_sec,
        )
    else:
        appearance_result = appearance_builder(
            bundle.calibration_input, union.micro_to_stable, s03_runtime.config
        )
    prototypes, prototype_mask, appearance_rows = _validate_appearance_result(
        appearance_result, union.components
    )
    report = _report(
        union=union,
        proposals=proposals,
        appearance_rows=appearance_rows,
        inputs=inputs,
        detection_count=len(detection.det_ids),
    )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = output_dir.parent / f".{output_dir.name}.staging-{os.getpid()}"
    if staging.exists():
        raise ContractError(f"S04 finalize staging directory exists: {staging}")
    staging.mkdir(parents=False)
    try:
        artifacts = config.artifacts
        _write_parquet(
            staging / artifacts.micro_to_stable,
            union.micro_to_stable_rows,
            MICRO_TO_STABLE_SCHEMA,
            config.parquet_compression,
        )
        _write_parquet(
            staging / artifacts.stable_tracklets,
            union.stable_tracklet_rows,
            STABLE_TRACKLETS_SCHEMA,
            config.parquet_compression,
        )
        _write_parquet(
            staging / artifacts.det_to_stable,
            det_rows,
            DET_TO_STABLE_SCHEMA,
            config.parquet_compression,
        )
        _write_parquet(
            staging / artifacts.stable_appearance,
            appearance_rows,
            appearance_schema,
            config.parquet_compression,
        )
        try:
            np.save(staging / artifacts.stable_prototypes, prototypes, allow_pickle=False)
            np.save(
                staging / artifacts.stable_prototype_mask,
                prototype_mask,
                allow_pickle=False,
            )
        except (OSError, ValueError) as exc:
            raise ContractError(f"cannot write S04 stable appearance arrays: {exc}") from exc
        _write_json(staging / artifacts.report, report)
        _write_json(staging / artifacts.effective_config, config_payload)
        _verify_unchanged(inputs)
        outputs = [
            _fingerprint(staging / name, relative_to=staging)
            for name in sorted(_expected_names(config))
        ]
        marker = {
            "schema_version": config.schema_version,
            "stage": "S04_FINALIZE",
            "config_hash": config_hash,
            "execution_mode": "operator_approved_component_union",
            "operator_approved": True,
            "merge_policy": "all_provisional_undirected_components",
            "automatic_merge_allowed": False,
            "num_automatic_merges": 0,
            "num_operator_approved_merges": (
                len(union.micro_to_stable) - len(union.components)
            ),
            "read_review_labels": False,
            "solver_used": False,
            "input_fingerprints": inputs,
            "output_fingerprints": outputs,
            "stats": report["counts"],
            "elapsed_sec": float(time.monotonic() - started),
        }
        _write_json(staging / artifacts.success, marker)
        if output_dir.exists():
            try:
                output_dir.rmdir()
            except OSError as exc:
                raise ContractError(
                    f"S04 finalize output cannot be atomically committed: {output_dir}"
                ) from exc
        os.replace(staging, output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    logger(f"[s04-finalize] complete: {output_dir}")
    return marker


__all__ = ["run_s04_finalize"]
