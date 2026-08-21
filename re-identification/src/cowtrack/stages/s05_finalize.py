"""Operator-approved full-graph S05 global path-cover finalization.

This stage deliberately does not consume the bounded 50-video review output.
It revalidates and exactly recomputes the complete S05 proposal graph, applies
the persisted blanket operator strategy to every provisional proposal, and
emits a total stable/detection-to-global mapping for the full clip sequence.

The operator strategy is not an independent statistical certificate.  The
original proposal ``confirmed == false`` evidence is preserved and every
resulting global identity remains explicitly provisional.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import stat
import time
import uuid
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from cowtrack.config import ContractError
from cowtrack.linking.path_cover import (
    GlobalProposalEdge,
    GlobalStableNode,
    PathCoverResult,
    solve_operator_approved_path_cover,
)
from cowtrack.linking.runtime import (
    FileFingerprint,
    fingerprint_file,
    load_production_inputs,
)
from cowtrack.linking.s04_runtime import S04FinalizedBundle, load_s04_finalized
from cowtrack.linking.s05_finalize_config import (
    S05FinalizeConfig,
    load_s05_finalize_config,
)
from cowtrack.linking.s05_proposal_runtime import (
    S05ProposalBundle,
    load_s05_proposals,
)
from cowtrack.linking.s05_runtime import (
    S05LongCalibrationBundle,
    load_s05_long_calibration,
)
from cowtrack.schemas.detections import DETECTIONS_SCHEMA
from cowtrack.schemas.frames import FRAMES_SCHEMA
from cowtrack.schemas.s05_finalize import (
    DET_TO_GLOBAL_SCHEMA,
    GLOBAL_CANDIDATE_EDGES_SCHEMA,
    GLOBAL_TRACKS_SCHEMA,
    STABLE_TO_GLOBAL_SCHEMA,
)
from cowtrack.schemas.s05_proposals import (
    LONG_CANDIDATE_EDGES_SCHEMA,
    LONG_LINK_PROPOSALS_SCHEMA,
)
from cowtrack.stages.s05_propose import (
    generate_s05_candidate_outputs,
    validate_s05_proposal_input_alignment,
)


LogFn = Callable[[str], None]
_STAGE = "S05_FINALIZE"
_FRAME_DURATION_SEC = 1001.0 / 30000.0


def log(message: str) -> None:
    print(message, flush=True)


def _read_json(path: Path, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read {label} {path}: {exc}") from exc


def _write_json(path: Path, payload: Any) -> None:
    try:
        path.write_text(
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
    except (OSError, TypeError, ValueError) as exc:
        raise ContractError(f"cannot write S05 finalize JSON {path}: {exc}") from exc


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise ContractError(f"cannot fingerprint S05 finalize artifact {path}: {exc}") from exc
    return digest.hexdigest()


def _output_fingerprint(path: Path, directory: Path) -> dict[str, Any]:
    resolved = path.resolve()
    root = directory.resolve()
    if not resolved.is_file() or resolved.is_symlink() or resolved.parent != root:
        raise ContractError(f"required S05 finalize artifact is not a regular file: {resolved}")
    return {
        "path": str(resolved.relative_to(root)),
        "size_bytes": int(resolved.stat().st_size),
        "sha256": _sha256(resolved),
    }


def _normalize_fingerprints(
    records: Sequence[FileFingerprint | Mapping[str, Any]],
) -> list[dict[str, Any]]:
    by_path: dict[str, dict[str, Any]] = {}
    for record in records:
        value = record.as_dict() if isinstance(record, FileFingerprint) else dict(record)
        raw_path, size, sha256 = (
            value.get("path"),
            value.get("size_bytes"),
            value.get("sha256"),
        )
        if (
            not isinstance(raw_path, str)
            or not raw_path
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
            or not isinstance(sha256, str)
            or len(sha256) != 64
            or any(character not in "0123456789abcdef" for character in sha256)
        ):
            raise ContractError("S05 finalize input fingerprint is invalid")
        path = str(Path(raw_path).resolve())
        normalized = {"path": path, "size_bytes": size, "sha256": sha256}
        if path in by_path and by_path[path] != normalized:
            raise ContractError("S05 finalize input fingerprint conflicts by path")
        by_path[path] = normalized
    if not by_path:
        raise ContractError("S05 finalize input fingerprint set cannot be empty")
    return [by_path[path] for path in sorted(by_path)]


def _verify_unchanged(records: Sequence[Mapping[str, Any]]) -> None:
    for expected in records:
        current = fingerprint_file(Path(str(expected["path"]))).as_dict()
        if current != dict(expected):
            raise ContractError(f"S05 finalize input changed: {expected['path']}")


def _reject_path_overlap(output_dir: Path, inputs: Sequence[Path]) -> None:
    output = output_dir.resolve()
    for raw in inputs:
        item = raw.resolve()
        if output == item or output in item.parents or item in output.parents:
            raise ContractError(f"S05 finalize output/input paths overlap: {output}, {item}")


def _write_table(path: Path, table: pa.Table, schema: pa.Schema, compression: str) -> None:
    if not table.schema.equals(schema, check_metadata=False):
        raise ContractError(f"S05 finalize table schema differs before write: {path.name}")
    try:
        pq.write_table(table, path, compression=compression, version="2.6")
    except (OSError, pa.ArrowException, TypeError, ValueError) as exc:
        raise ContractError(f"cannot write S05 finalize Parquet {path}: {exc}") from exc


def _rows_table(
    rows: Sequence[Mapping[str, Any]], schema: pa.Schema, label: str
) -> pa.Table:
    try:
        return pa.Table.from_pylist([dict(row) for row in rows], schema=schema)
    except (pa.ArrowException, TypeError, ValueError) as exc:
        raise ContractError(f"cannot canonicalize {label}: {exc}") from exc


def _read_table(path: Path, schema: pa.Schema, label: str) -> pa.Table:
    try:
        parquet = pq.ParquetFile(path)
        if not parquet.schema_arrow.equals(schema, check_metadata=False):
            raise ContractError(f"{label} schema mismatch: {path}")
        return pq.read_table(path)
    except ContractError:
        raise
    except (OSError, pa.ArrowException, TypeError, ValueError) as exc:
        raise ContractError(f"cannot read {label} {path}: {exc}") from exc


def _artifact_names(config: S05FinalizeConfig) -> dict[str, str]:
    artifacts = config.artifacts
    return {
        "candidate_edges": artifacts.candidate_edges,
        "stable_to_global": artifacts.stable_to_global,
        "global_tracks": artifacts.global_tracks,
        "det_to_global": artifacts.det_to_global,
        "report": artifacts.report,
        "effective_config": artifacts.effective_config,
    }


def _validate_fixed_inputs(
    production: Any,
    stable: S04FinalizedBundle,
    long_runtime: S05LongCalibrationBundle,
    proposals: S05ProposalBundle,
    config: S05FinalizeConfig,
) -> None:
    if stable.success_marker.get("config_hash") != config.expected_s04_finalize_config_hash:
        raise ContractError("S05 finalize S04 config hash differs")
    if long_runtime.config_hash != config.expected_long_calibration_config_hash:
        raise ContractError("S05 finalize long calibration config hash differs")
    if proposals.config_hash != config.expected_long_proposal_config_hash:
        raise ContractError("S05 finalize proposal config hash differs")
    if (
        len(stable.stable_ids) != config.expected_stable_track_count
        or len(stable.micro_ids) != config.expected_microtrack_count
        or len(stable.det_ids) != config.expected_detection_count
        or len(proposals.candidates) != config.expected_candidate_count
        or len(proposals.proposals) != config.expected_proposal_count
    ):
        raise ContractError("S05 finalize fixed input counts differ")
    clips = tuple(map(str, production.calibration_input.timeline_clip_ids))
    if clips != config.clip_order:
        raise ContractError(f"S05 finalize clip order differs: {clips!r}")
    if not long_runtime.model_enabled or long_runtime.confirmed_enabled:
        raise ContractError(
            "S05 finalize fixed override requires enabled scoring and disabled confirmation"
        )
    if config.certification_claim_allowed:
        raise ContractError("S05 finalize must not claim independent certification")
    validate_s05_proposal_input_alignment(
        production,
        stable,
        long_runtime,
        proposals.config,
    )
    recorded = {item.path: item for item in proposals.input_fingerprints}
    current = (
        *production.input_fingerprints,
        *stable.input_fingerprints,
        *long_runtime.input_fingerprints,
        *long_runtime.output_fingerprints,
        fingerprint_file(long_runtime.directory / "_SUCCESS.json"),
    )
    for fingerprint in current:
        if recorded.get(fingerprint.path) != fingerprint:
            raise ContractError(
                "S05 proposal graph was not built from the supplied current inputs: "
                f"{fingerprint.path}"
            )


def _recompute_complete_graph(
    production: Any,
    stable: S04FinalizedBundle,
    long_runtime: S05LongCalibrationBundle,
    proposal_bundle: S05ProposalBundle,
    *,
    logger: LogFn,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidates, proposals = generate_s05_candidate_outputs(
        production,
        stable,
        long_runtime,
        proposal_bundle.config,
        logger=logger,
    )
    expected_candidates = _rows_table(
        candidates, LONG_CANDIDATE_EDGES_SCHEMA, "recomputed S05 candidates"
    )
    expected_proposals = _rows_table(
        proposals, LONG_LINK_PROPOSALS_SCHEMA, "recomputed S05 proposals"
    )
    persisted_candidates = _rows_table(
        proposal_bundle.candidates,
        LONG_CANDIDATE_EDGES_SCHEMA,
        "persisted S05 candidates",
    )
    persisted_proposals = _rows_table(
        proposal_bundle.proposals,
        LONG_LINK_PROPOSALS_SCHEMA,
        "persisted S05 proposals",
    )
    if not expected_candidates.equals(persisted_candidates) or not expected_proposals.equals(
        persisted_proposals
    ):
        raise ContractError(
            "S05 finalize proposal graph differs from exact retrieval/model recomputation"
        )
    return candidates, proposals


def _load_ingest_identity(
    ingest_dir: Path,
    production: Any,
    config: S05FinalizeConfig,
) -> dict[str, np.ndarray]:
    frames = _read_table(ingest_dir / "frames.parquet", FRAMES_SCHEMA, "S05 frames")
    detections = _read_table(
        ingest_dir / "detections.parquet", DETECTIONS_SCHEMA, "S05 detections"
    )
    frame_clips = np.asarray(frames["clip_id"].to_pylist(), dtype=object)
    frame_orders = np.asarray(
        frames["clip_order"].combine_chunks().to_numpy(zero_copy_only=False),
        dtype=np.int16,
    )
    frame_global = np.asarray(
        frames["global_frame"].combine_chunks().to_numpy(zero_copy_only=False),
        dtype=np.int64,
    )
    if (
        len(frames) != config.expected_frame_count
        or not np.array_equal(frame_global, np.arange(len(frames), dtype=np.int64))
        or tuple(dict.fromkeys(map(str, frame_clips))) != config.clip_order
        or any(
            int(np.count_nonzero(frame_clips == clip_id)) != expected
            for clip_id, expected in zip(
                config.clip_order, config.frame_counts_by_clip, strict=True
            )
        )
        or any(
            np.unique(frame_orders[frame_clips == clip_id]).tolist() != [order]
            for order, clip_id in enumerate(config.clip_order)
        )
    ):
        raise ContractError("S05 finalize frame coverage differs from the fixed sequence")

    valid = np.asarray(
        detections["valid"].combine_chunks().to_numpy(zero_copy_only=False),
        dtype=np.bool_,
    )
    clip_all = np.asarray(detections["clip_id"].to_pylist(), dtype=object)
    valid_counts = tuple(
        int(np.count_nonzero(valid & (clip_all == clip_id)))
        for clip_id in config.clip_order
    )
    invalid_counts = tuple(
        int(np.count_nonzero(~valid & (clip_all == clip_id)))
        for clip_id in config.clip_order
    )
    if (
        int(np.count_nonzero(valid)) != config.expected_detection_count
        or int(np.count_nonzero(~valid)) != config.expected_invalid_detections
        or valid_counts != config.expected_valid_detections_by_clip
        or invalid_counts != config.expected_invalid_detections_by_clip
    ):
        raise ContractError("S05 finalize valid/invalid detection coverage differs")

    def numeric(name: str, dtype: Any) -> np.ndarray:
        return np.asarray(
            detections[name].combine_chunks().to_numpy(zero_copy_only=False),
            dtype=dtype,
        )[valid]

    det_ids = numeric("det_id", np.int64)
    global_frames = numeric("global_frame", np.int64)
    order = np.lexsort((det_ids, global_frames))
    sequence_all = np.asarray(detections["sequence_id"].to_pylist(), dtype=object)[valid]
    result = {
        "det_id": det_ids[order],
        "sequence_id": sequence_all[order],
        "clip_id": clip_all[valid][order],
        "local_frame": numeric("local_frame", np.int32)[order],
        "global_frame": global_frames[order],
        "global_time_sec": numeric("global_time_sec", np.float64)[order],
    }
    runtime = production.detections
    if (
        len(set(map(str, result["sequence_id"]))) != 1
        or str(result["sequence_id"][0]) != config.expected_sequence_id
        or not np.array_equal(result["det_id"], runtime.det_ids)
        or not np.array_equal(result["clip_id"], runtime.clip_ids)
        or not np.array_equal(result["global_frame"], runtime.global_frames)
        or not np.array_equal(result["global_time_sec"], runtime.global_time_sec)
    ):
        raise ContractError("S05 finalize S00 detection identity differs from runtime")
    return result


def _build_graph_inputs(
    stable: S04FinalizedBundle,
    proposals: Sequence[Mapping[str, Any]],
    config: S05FinalizeConfig,
) -> tuple[list[GlobalStableNode], list[GlobalProposalEdge]]:
    nodes: list[GlobalStableNode] = []
    for stable_id in map(int, stable.stable_ids):
        tracklet = stable.stable_tracklets.get(stable_id)
        if tracklet is None:
            raise ContractError(f"S05 finalize stable metadata missing: {stable_id}")
        nodes.append(
            GlobalStableNode(
                stable_id=stable_id,
                start_clip_id=str(tracklet.start_clip_id),
                end_clip_id=str(tracklet.end_clip_id),
                start_global_frame=int(tracklet.start_global_frame),
                end_global_frame=int(tracklet.end_global_frame),
                start_time_sec=float(tracklet.start_time_sec),
                end_time_sec=float(tracklet.end_time_sec),
                num_microtracklets=int(tracklet.num_microtracklets),
                num_detections=int(tracklet.num_detections),
            )
        )
    edges: list[GlobalProposalEdge] = []
    previous = ""
    for row in proposals:
        proposal_id = str(row["proposal_id"])
        candidate_id = str(row["candidate_id"])
        if (
            proposal_id <= previous
            or row["decision"] != config.required_proposal_decision
            or row["review_status"] != config.required_proposal_review_status
            or row["selected_by_solver"] is not False
            or row["confirmed"] is not False
            or row["merge_applied"] is not False
            or row["appearance_present"] is not True
            or row["temporally_nonoverlapping"] is not True
        ):
            raise ContractError("S05 finalize proposal authorization contract differs")
        previous = proposal_id
        rank_out, rank_in = row["appearance_rank_out"], row["appearance_rank_in"]
        probability, mutual = row["model_probability"], row["gallery_score_mutual"]
        if rank_out is None or rank_in is None or probability is None or mutual is None:
            raise ContractError("S05 finalize provisional proposal lacks solver evidence")
        edges.append(
            GlobalProposalEdge(
                proposal_id=proposal_id,
                candidate_id=candidate_id,
                source_stable_id=int(row["source_stable_id"]),
                target_stable_id=int(row["target_stable_id"]),
                probability=float(probability),
                candidate_margin=(
                    None
                    if row["candidate_margin"] is None
                    else float(row["candidate_margin"])
                ),
                rank_out=int(rank_out),
                rank_in=int(rank_in),
                high_overlap=bool(row["high_overlap"]),
                gallery_score_mutual=float(mutual),
                temporal_gap_sec=float(row["temporal_gap_sec"]),
            )
        )
    if len(nodes) != config.expected_stable_track_count or len(edges) != config.expected_proposal_count:
        raise ContractError("S05 finalize graph node/edge counts differ")
    return nodes, edges


def _selected_edge_maps(
    result: PathCoverResult,
) -> tuple[
    dict[str, GlobalProposalEdge],
    dict[tuple[int, int], GlobalProposalEdge],
]:
    by_candidate: dict[str, GlobalProposalEdge] = {}
    by_pair: dict[tuple[int, int], GlobalProposalEdge] = {}
    for edge in result.selected_edges:
        if edge.candidate_id in by_candidate:
            raise ContractError("S05 finalize solver selected a candidate twice")
        pair = (edge.source_stable_id, edge.target_stable_id)
        if pair in by_pair:
            raise ContractError("S05 finalize solver selected duplicate stable pair")
        by_candidate[edge.candidate_id] = edge
        by_pair[pair] = edge
    return by_candidate, by_pair


def _global_link_id(edge: GlobalProposalEdge) -> str:
    return f"s05l-{edge.source_stable_id:06d}-{edge.target_stable_id:06d}"


def _build_candidate_audit_rows(
    candidates: Sequence[Mapping[str, Any]],
    proposals: Sequence[Mapping[str, Any]],
    result: PathCoverResult,
    config: S05FinalizeConfig,
) -> list[dict[str, Any]]:
    proposal_by_candidate = {str(row["candidate_id"]): row for row in proposals}
    if len(proposal_by_candidate) != len(proposals):
        raise ContractError("S05 finalize proposal candidate IDs are not unique")
    selected, _ = _selected_edge_maps(result)
    costs = dict(result.solver_cost_by_candidate)
    rows: list[dict[str, Any]] = []
    previous = ""
    for source in candidates:
        candidate_id = str(source["candidate_id"])
        if candidate_id <= previous:
            raise ContractError("S05 finalize candidate rows are not canonical")
        previous = candidate_id
        proposal = proposal_by_candidate.get(candidate_id)
        eligible = proposal is not None
        chosen = selected.get(candidate_id)
        if chosen is not None and not eligible:
            raise ContractError("S05 finalize solver selected an ineligible candidate")
        if eligible:
            final_decision = "selected" if chosen is not None else "deferred"
            final_reason = (
                "operator_approved_maximum_cardinality_path_cover"
                if chosen is not None
                else "not_selected_by_global_one_to_one_path_cover"
            )
            authorization_basis = config.authorization_basis
        else:
            final_decision = "rejected"
            final_reason = str(source["decision_reason"])
            authorization_basis = "none"
        solver_cost = costs.get(candidate_id) if eligible else None
        if eligible and solver_cost is None:
            raise ContractError("S05 finalize eligible proposal lacks a solver cost")
        row = {
            "candidate_id": candidate_id,
            "proposal_id": None if proposal is None else str(proposal["proposal_id"]),
            "source_stable_id": int(source["source_stable_id"]),
            "target_stable_id": int(source["target_stable_id"]),
            "source_end_clip_id": str(source["source_end_clip_id"]),
            "target_start_clip_id": str(source["target_start_clip_id"]),
            "source_end_global_frame": int(source["source_end_global_frame"]),
            "target_start_global_frame": int(source["target_start_global_frame"]),
            "source_end_time_sec": float(source["source_end_time_sec"]),
            "target_start_time_sec": float(source["target_start_time_sec"]),
            "temporal_gap_sec": float(source["temporal_gap_sec"]),
            "temporally_nonoverlapping": bool(source["temporally_nonoverlapping"]),
            "appearance_present": bool(source["appearance_present"]),
            "high_overlap": bool(source["high_overlap"]),
            "selected_by_appearance_topk": bool(source["selected_by_appearance_topk"]),
            "selected_by_temporal_nearest": bool(source["selected_by_temporal_nearest"]),
            "appearance_rank_out": source["appearance_rank_out"],
            "appearance_rank_in": source["appearance_rank_in"],
            "best_margin_out": source["best_margin_out"],
            "best_margin_in": source["best_margin_in"],
            "gallery_score_mutual": source["gallery_score_mutual"],
            "model_probability": source["model_probability"],
            "model_raw_score": source["model_raw_score"],
            "candidate_margin": source["candidate_margin"],
            "provisional_threshold": float(source["provisional_threshold"]),
            "selected_probability_threshold": float(
                source["selected_probability_threshold"]
            ),
            "selected_margin_threshold": float(source["selected_margin_threshold"]),
            "passes_provisional_threshold": bool(source["passes_provisional_threshold"]),
            "passes_selected_probability_gate": bool(
                source["passes_selected_probability_gate"]
            ),
            "passes_selected_margin_gate": bool(source["passes_selected_margin_gate"]),
            "passes_selected_gate": bool(source["passes_selected_gate"]),
            "proposal_decision": str(source["decision"]),
            "proposal_decision_reason": str(source["decision_reason"]),
            "proposal_evidence_status": (
                None if proposal is None else str(proposal["evidence_status"])
            ),
            "proposal_review_status": (
                None if proposal is None else str(proposal["review_status"])
            ),
            "proposal_selected_by_solver": bool(source["selected_by_solver"]),
            "proposal_confirmed": bool(source["confirmed"]),
            "proposal_merge_applied": bool(source["merge_applied"]),
            "operator_approved": eligible,
            "approval_strategy": config.approval_strategy,
            "authorization_basis": authorization_basis,
            "eligible_for_solver": eligible,
            "selected_by_solver": chosen is not None,
            "solver_iteration": 1 if eligible else 0,
            "solver_cost_int": None if solver_cost is None else int(solver_cost),
            "global_link_id": None if chosen is None else _global_link_id(chosen),
            "final_decision": final_decision,
            "final_decision_reason": final_reason,
        }
        rows.append(row)
    if (
        len(rows) != len(candidates)
        or sum(bool(row["eligible_for_solver"]) for row in rows) != len(proposals)
        or sum(bool(row["selected_by_solver"]) for row in rows)
        != len(result.selected_edges)
    ):
        raise ContractError("S05 finalize candidate audit coverage differs")
    return rows


def _path_sort_key(path: Sequence[int], stable: S04FinalizedBundle) -> tuple[Any, ...]:
    first = stable.stable_tracklets[int(path[0])]
    return (
        float(first.start_time_sec),
        int(first.start_global_frame),
        int(first.stable_id),
        tuple(map(int, path)),
    )


def _path_clip_ids(
    path: Sequence[int], stable: S04FinalizedBundle, clip_order: Sequence[str]
) -> list[str]:
    positions = {clip_id: index for index, clip_id in enumerate(clip_order)}
    used: set[str] = set()
    for stable_id in path:
        tracklet = stable.stable_tracklets[int(stable_id)]
        if tracklet.start_clip_id not in positions or tracklet.end_clip_id not in positions:
            raise ContractError("S05 finalize stable path references an unknown clip")
        left, right = positions[tracklet.start_clip_id], positions[tracklet.end_clip_id]
        if left > right:
            raise ContractError("S05 finalize stable path reverses clip order")
        used.update(clip_order[left : right + 1])
    return [clip_id for clip_id in clip_order if clip_id in used]


def _build_global_rows(
    stable: S04FinalizedBundle,
    result: PathCoverResult,
    config: S05FinalizeConfig,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[int, dict[str, Any]],
]:
    _, selected_by_pair = _selected_edge_maps(result)
    paths = [tuple(map(int, path)) for path in result.paths]
    paths.sort(key=lambda path: _path_sort_key(path, stable))
    flattened = [stable_id for path in paths for stable_id in path]
    expected_ids = list(map(int, stable.stable_ids))
    if (
        sorted(flattened) != expected_ids
        or len(flattened) != len(set(flattened))
        or len(paths) != len(expected_ids) - len(result.selected_edges)
        or int(result.max_cardinality) != len(result.selected_edges)
    ):
        raise ContractError("S05 finalize path cover is not total or P=N-L differs")
    population_warning = len(paths) > config.population_warning_threshold
    mapping_rows: list[dict[str, Any]] = []
    global_rows: list[dict[str, Any]] = []
    by_stable: dict[int, dict[str, Any]] = {}
    for global_track_id, path in enumerate(paths):
        if not path:
            raise ContractError("S05 finalize path cover contains an empty path")
        tracklets = [stable.stable_tracklets[stable_id] for stable_id in path]
        links: list[GlobalProposalEdge] = []
        for source_id, target_id in zip(path, path[1:]):
            edge = selected_by_pair.get((source_id, target_id))
            if edge is None:
                raise ContractError("S05 finalize path adjacency lacks its selected edge")
            links.append(edge)
        for previous, current in zip(tracklets, tracklets[1:]):
            if (
                previous.end_global_frame >= current.start_global_frame
                or previous.end_time_sec >= current.start_time_sec
            ):
                raise ContractError("S05 finalize global path contains temporal overlap")
        path_token = ",".join(map(str, path))
        global_uuid = str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"cowtrack://{config.expected_sequence_id}/s05/{path_token}",
            )
        )
        display_id = f"G{global_track_id + 1:04d}"
        probabilities = [float(edge.probability) for edge in links]
        margins = [
            float(edge.candidate_margin)
            for edge in links
            if edge.candidate_margin is not None
        ]
        mutual_scores = [float(edge.gallery_score_mutual) for edge in links]
        minimum = min(probabilities) if probabilities else None
        mean = float(np.mean(probabilities)) if probabilities else None
        maximum = max(probabilities) if probabilities else None
        component_detections = sum(int(tracklet.num_detections) for tracklet in tracklets)
        identity_basis = (
            config.authorization_basis if links else "s04_stable_path_only"
        )
        clip_ids = _path_clip_ids(path, stable, config.clip_order)
        global_rows.append(
            {
                "global_track_id": global_track_id,
                "global_track_uuid": global_uuid,
                "display_global_id": display_id,
                "sequence_id": config.expected_sequence_id,
                "first_stable_id": path[0],
                "last_stable_id": path[-1],
                "start_det_id": int(tracklets[0].start_det_id),
                "end_det_id": int(tracklets[-1].end_det_id),
                "start_clip_id": str(tracklets[0].start_clip_id),
                "end_clip_id": str(tracklets[-1].end_clip_id),
                "clip_ids": clip_ids,
                "start_global_frame": int(tracklets[0].start_global_frame),
                "end_global_frame": int(tracklets[-1].end_global_frame),
                "start_time_sec": float(tracklets[0].start_time_sec),
                "end_time_sec": float(tracklets[-1].end_time_sec),
                "num_stable_tracklets": len(path),
                "num_microtracklets": sum(
                    int(tracklet.num_microtracklets) for tracklet in tracklets
                ),
                "num_detections": component_detections,
                "num_long_links": len(links),
                "duration_visible_sec": component_detections * _FRAME_DURATION_SEC,
                "min_link_probability": minimum,
                "p10_link_probability": (
                    None
                    if not probabilities
                    else float(np.quantile(probabilities, 0.1, method="linear"))
                ),
                "mean_link_probability": mean,
                "max_link_probability": maximum,
                "min_link_margin": min(margins) if margins else None,
                "appearance_consistency": (
                    float(np.mean(mutual_scores)) if mutual_scores else None
                ),
                "identity_basis": identity_basis,
                "id_status": "provisional",
                "spans_multiple_clips": len(clip_ids) > 1,
                "population_warning": population_warning,
            }
        )
        for order, (stable_id, tracklet) in enumerate(zip(path, tracklets, strict=True)):
            predecessor = links[order - 1] if order else None
            cross_clip = bool(
                predecessor is not None
                and stable.stable_tracklets[predecessor.source_stable_id].end_clip_id
                != stable.stable_tracklets[predecessor.target_stable_id].start_clip_id
            )
            mapping = {
                "stable_id": stable_id,
                "global_track_id": global_track_id,
                "global_track_uuid": global_uuid,
                "display_global_id": display_id,
                "order_in_global_path": order,
                "predecessor_stable_id": (
                    None if predecessor is None else predecessor.source_stable_id
                ),
                "predecessor_candidate_id": (
                    None if predecessor is None else predecessor.candidate_id
                ),
                "predecessor_proposal_id": (
                    None if predecessor is None else predecessor.proposal_id
                ),
                "predecessor_global_link_id": (
                    None if predecessor is None else _global_link_id(predecessor)
                ),
                "predecessor_link_probability": (
                    None if predecessor is None else float(predecessor.probability)
                ),
                "predecessor_link_margin": (
                    None
                    if predecessor is None or predecessor.candidate_margin is None
                    else float(predecessor.candidate_margin)
                ),
                "predecessor_authorization_basis": (
                    None if predecessor is None else config.authorization_basis
                ),
                "link_type": (
                    "PATH_START"
                    if predecessor is None
                    else "FILE_BOUNDARY"
                    if cross_clip
                    else "LONG_GAP_APPEARANCE"
                ),
                "cross_clip_boundary": cross_clip,
                "component_num_stable_tracklets": len(path),
                "component_num_detections": component_detections,
                "component_num_long_links": len(links),
                "component_min_link_probability": minimum,
                "component_mean_link_probability": mean,
                "component_max_link_probability": maximum,
                "identity_basis": identity_basis,
                "id_status": "provisional",
            }
            mapping_rows.append(mapping)
            by_stable[stable_id] = mapping
    mapping_rows.sort(key=lambda row: int(row["stable_id"]))
    if (
        len(mapping_rows) != config.expected_stable_track_count
        or set(by_stable) != set(expected_ids)
        or sum(int(row["num_stable_tracklets"]) for row in global_rows)
        != config.expected_stable_track_count
        or sum(int(row["num_microtracklets"]) for row in global_rows)
        != config.expected_microtrack_count
        or sum(int(row["num_detections"]) for row in global_rows)
        != config.expected_detection_count
        or sum(int(row["num_long_links"]) for row in global_rows)
        != len(result.selected_edges)
    ):
        raise ContractError("S05 finalize global aggregate coverage differs")
    return mapping_rows, global_rows, by_stable


def _join_stable_detection_mapping(
    det_ids: np.ndarray, stable: S04FinalizedBundle
) -> tuple[np.ndarray, ...]:
    source_ids = np.asarray(stable.det_ids, dtype=np.int64)
    order = np.argsort(source_ids, kind="stable")
    sorted_ids = source_ids[order]
    positions = np.searchsorted(sorted_ids, det_ids)
    if (
        len(det_ids) != len(source_ids)
        or np.any(positions >= len(sorted_ids))
        or not np.array_equal(sorted_ids[positions], det_ids)
        or len(np.unique(source_ids)) != len(source_ids)
    ):
        raise ContractError("S05 finalize S00/S04 detection IDs are not bijective")
    selected = order[positions]
    return tuple(
        np.asarray(values)[selected]
        for values in (
            stable.det_micro_ids,
            stable.det_stable_ids,
            stable.det_order_in_micro,
            stable.det_order_in_stable,
            stable.det_order_in_stable_detection,
        )
    )


def _group_order(
    global_ids: np.ndarray, global_frames: np.ndarray, det_ids: np.ndarray
) -> np.ndarray:
    order = np.lexsort((det_ids, global_frames, global_ids))
    sorted_ids = global_ids[order]
    starts = np.empty(len(order), dtype=np.int64)
    if len(order):
        is_start = np.r_[True, sorted_ids[1:] != sorted_ids[:-1]]
        start_positions = np.flatnonzero(is_start)
        starts[:] = np.repeat(start_positions, np.diff(np.r_[start_positions, len(order)]))
    ranks = np.arange(len(order), dtype=np.int64) - starts
    result = np.empty(len(order), dtype=np.int64)
    result[order] = ranks
    return result


def _build_detection_table(
    identity: Mapping[str, np.ndarray],
    stable: S04FinalizedBundle,
    mapping_by_stable: Mapping[int, Mapping[str, Any]],
    global_rows: Sequence[Mapping[str, Any]],
    config: S05FinalizeConfig,
) -> pa.Table:
    det_ids = np.asarray(identity["det_id"], dtype=np.int64)
    global_frames = np.asarray(identity["global_frame"], dtype=np.int64)
    (
        micro_ids,
        stable_ids,
        order_in_micro,
        order_in_stable,
        order_in_stable_detection,
    ) = _join_stable_detection_mapping(det_ids, stable)
    micro_ids = np.asarray(micro_ids, dtype=np.int64)
    stable_ids = np.asarray(stable_ids, dtype=np.int64)
    order_in_micro = np.asarray(order_in_micro, dtype=np.int32)
    order_in_stable = np.asarray(order_in_stable, dtype=np.int32)
    order_in_stable_detection = np.asarray(order_in_stable_detection, dtype=np.int64)
    if (
        np.any(stable_ids < 0)
        or np.any(stable_ids >= config.expected_stable_track_count)
        or set(map(int, stable_ids)) != set(mapping_by_stable)
    ):
        raise ContractError("S05 finalize detection stable IDs are incomplete")
    global_by_stable = np.full(config.expected_stable_track_count, -1, dtype=np.int64)
    global_order_by_stable = np.full(
        config.expected_stable_track_count, -1, dtype=np.int32
    )
    for stable_id, row in mapping_by_stable.items():
        global_by_stable[stable_id] = int(row["global_track_id"])
        global_order_by_stable[stable_id] = int(row["order_in_global_path"])
    global_ids = global_by_stable[stable_ids]
    global_stable_order = global_order_by_stable[stable_ids]
    if np.any(global_ids < 0) or np.any(global_stable_order < 0):
        raise ContractError("S05 finalize detection global mapping contains holes")

    simultaneous = np.lexsort((global_frames, global_ids))
    sorted_global = global_ids[simultaneous]
    sorted_frames = global_frames[simultaneous]
    if len(simultaneous) > 1 and np.any(
        (sorted_global[1:] == sorted_global[:-1])
        & (sorted_frames[1:] == sorted_frames[:-1])
    ):
        raise ContractError("S05 finalize same global ID appears twice in one frame")
    global_detection_order = _group_order(global_ids, global_frames, det_ids)

    global_uuid = np.asarray(
        [str(row["global_track_uuid"]) for row in global_rows], dtype=object
    )
    display_id = np.asarray(
        [str(row["display_global_id"]) for row in global_rows], dtype=object
    )
    identity_basis = np.asarray(
        [str(row["identity_basis"]) for row in global_rows], dtype=object
    )
    id_status = np.asarray(
        [str(row["id_status"]) for row in global_rows], dtype=object
    )
    clip_lookup = {clip_id: index for index, clip_id in enumerate(config.clip_order)}
    clip_ids = np.asarray(identity["clip_id"], dtype=object)
    try:
        clip_orders = np.asarray([clip_lookup[str(value)] for value in clip_ids], dtype=np.int16)
    except KeyError as exc:
        raise ContractError(f"S05 finalize detection references unknown clip: {exc}") from exc
    columns: dict[str, Any] = {
        "det_id": det_ids,
        "sequence_id": np.asarray(identity["sequence_id"], dtype=object),
        "clip_id": clip_ids,
        "clip_order": clip_orders,
        "local_frame": np.asarray(identity["local_frame"], dtype=np.int32),
        "global_frame": global_frames,
        "global_time_sec": np.asarray(identity["global_time_sec"], dtype=np.float64),
        "valid": np.ones(len(det_ids), dtype=np.bool_),
        "micro_id": micro_ids,
        "stable_id": stable_ids,
        "global_track_id": global_ids,
        "global_track_uuid": global_uuid[global_ids],
        "display_global_id": display_id[global_ids],
        "order_in_micro": order_in_micro,
        "order_in_stable": order_in_stable,
        "order_in_stable_detection": order_in_stable_detection,
        "order_in_global_stable": global_stable_order,
        "order_in_global_detection": global_detection_order,
        "identity_basis": identity_basis[global_ids],
        "id_status": id_status[global_ids],
    }
    try:
        table = pa.Table.from_pydict(columns, schema=DET_TO_GLOBAL_SCHEMA)
    except (pa.ArrowException, TypeError, ValueError) as exc:
        raise ContractError(f"cannot build S05 detection-to-global table: {exc}") from exc
    if (
        table.num_rows != config.expected_detection_count
        or len(np.unique(det_ids)) != len(det_ids)
        or sum(int(row["num_detections"]) for row in global_rows) != table.num_rows
        or Counter(map(str, clip_ids))
        != Counter(
            {
                clip_id: expected
                for clip_id, expected in zip(
                    config.clip_order,
                    config.expected_valid_detections_by_clip,
                    strict=True,
                )
            }
        )
    ):
        raise ContractError("S05 finalize detection-to-global coverage differs")
    observed_counts = np.bincount(global_ids, minlength=len(global_rows))
    expected_counts = np.asarray(
        [int(row["num_detections"]) for row in global_rows], dtype=np.int64
    )
    if not np.array_equal(observed_counts, expected_counts):
        raise ContractError("S05 finalize per-global detection aggregates differ")
    return table


def build_detection_to_global_table(
    identity: Mapping[str, np.ndarray],
    stable: S04FinalizedBundle,
    mapping_by_stable: Mapping[int, Mapping[str, Any]],
    global_rows: Sequence[Mapping[str, Any]],
    config: S05FinalizeConfig,
) -> pa.Table:
    """Public shared builder for a validated S04-path to global-ID mapping.

    The config argument is structural (expected counts and clip order); callers
    may use another strict stage config that provides the same fields.
    """

    return _build_detection_table(
        identity, stable, mapping_by_stable, global_rows, config
    )


def _stats(
    *,
    config: S05FinalizeConfig,
    result: PathCoverResult,
    candidate_rows: Sequence[Mapping[str, Any]],
    global_rows: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    selected_rows = [row for row in candidate_rows if bool(row["selected_by_solver"])]
    return {
        "num_frames": config.expected_frame_count,
        "num_valid_detections": config.expected_detection_count,
        "num_invalid_detections_excluded": config.expected_invalid_detections,
        "num_microtracklets": config.expected_microtrack_count,
        "num_stable_tracklets": config.expected_stable_track_count,
        "num_candidates_audited": len(candidate_rows),
        "num_proposals_eligible": sum(
            bool(row["eligible_for_solver"]) for row in candidate_rows
        ),
        "num_selected_links": len(result.selected_edges),
        "num_global_tracks": len(global_rows),
        "num_singleton_global_tracks": sum(
            int(row["num_stable_tracklets"]) == 1 for row in global_rows
        ),
        "num_cross_clip_links": sum(
            row["source_end_clip_id"] != row["target_start_clip_id"]
            for row in selected_rows
        ),
        "num_selected_high_overlap_links": sum(
            bool(row["high_overlap"]) for row in selected_rows
        ),
        "num_selected_gate_uncertified_links": sum(
            row["proposal_evidence_status"] == "selected_gate_uncertified"
            for row in selected_rows
        ),
        "num_certified_auto_links": 0,
        "num_review_cases_consumed": 0,
        "same_frame_same_global_id_violations": 0,
    }


def _build_report(
    *,
    config: S05FinalizeConfig,
    config_hash: str,
    stable: S04FinalizedBundle,
    long_runtime: S05LongCalibrationBundle,
    proposal_bundle: S05ProposalBundle,
    result: PathCoverResult,
    candidate_rows: Sequence[Mapping[str, Any]],
    global_rows: Sequence[Mapping[str, Any]],
    input_fingerprints: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    stats = _stats(
        config=config,
        result=result,
        candidate_rows=candidate_rows,
        global_rows=global_rows,
    )
    selected_rows = [row for row in candidate_rows if bool(row["selected_by_solver"])]
    population_warning = len(global_rows) > config.population_warning_threshold
    stable_span_counts = Counter(
        f"{tracklet.start_clip_id}->{tracklet.end_clip_id}"
        for tracklet in stable.stable_tracklets.values()
    )
    return {
        "schema_version": "1.0",
        "stage": _STAGE,
        "config_hash": config_hash,
        "execution_mode": config.execution_mode,
        "sequence_id": config.expected_sequence_id,
        "clip_order": list(config.clip_order),
        "operator_approval": {
            "operator_approved": True,
            "strategy": config.approval_strategy,
            "authorization_basis": config.authorization_basis,
            "scope": "all_current_provisional_proposals_in_fixed_11_clip_sequence",
            "review_output_consumed": False,
            "review_labels_consumed": False,
            "bounded_review_case_limit_applied": False,
        },
        "evidence_semantics": {
            "long_model_enabled": bool(long_runtime.model_enabled),
            "long_confirmed_enabled": bool(long_runtime.confirmed_enabled),
            "confirmed_threshold": long_runtime.thresholds.confirmed_threshold,
            "provisional_threshold": long_runtime.thresholds.provisional_threshold,
            "confirmed_disabled_reason": long_runtime.thresholds.confirmed_disabled_reason,
            "confirmed_false_accept_upper": (
                long_runtime.selected_gate_evidence.false_accept_upper
            ),
            "confirmed_far_target": long_runtime.thresholds.confirmed_far_target,
            "certification_claimed": False,
            "original_proposal_confirmed_state_preserved": True,
            "all_global_ids_status": "provisional",
        },
        "proposal_graph": {
            "config_hash": proposal_bundle.config_hash,
            "candidate_rows": len(candidate_rows),
            "provisional_proposal_rows": stats["num_proposals_eligible"],
            "review_sample_rows_consumed": 0,
            "selected_gate_uncertified_proposals": sum(
                row["proposal_evidence_status"] == "selected_gate_uncertified"
                for row in candidate_rows
            ),
            "proposal_clip_distribution": dict(
                sorted(
                    Counter(
                        f"{row['source_end_clip_id']}->{row['target_start_clip_id']}"
                        for row in candidate_rows
                        if bool(row["eligible_for_solver"])
                    ).items()
                )
            ),
        },
        "solver": {
            "algorithm": config.solver,
            "objective": config.objective,
            "candidate_policy": config.candidate_policy,
            "evidence_cost": config.evidence_cost,
            "deterministic_tie_break": config.deterministic_tie_break,
            "runtime_python_version": config.expected_python_version,
            "runtime_scipy_version": config.expected_scipy_version,
            "iterations": 1,
            "maximum_cardinality": int(result.max_cardinality),
            "selected_links": len(result.selected_edges),
            "global_paths": len(global_rows),
            "p_equals_n_minus_l": (
                len(global_rows)
                == config.expected_stable_track_count - len(result.selected_edges)
            ),
            "max_incoming_links": 1,
            "max_outgoing_links": 1,
            "selected_clip_distribution": dict(
                sorted(
                    Counter(
                        f"{row['source_end_clip_id']}->{row['target_start_clip_id']}"
                        for row in selected_rows
                    ).items()
                )
            ),
            "selected_high_overlap_links": stats["num_selected_high_overlap_links"],
            "population_prior_used_by_solver": False,
            "threshold_adapted": False,
        },
        "coverage": {
            "frames_by_clip": dict(
                zip(config.clip_order, config.frame_counts_by_clip, strict=True)
            ),
            "valid_detections_by_clip": dict(
                zip(
                    config.clip_order,
                    config.expected_valid_detections_by_clip,
                    strict=True,
                )
            ),
            "invalid_detections_excluded_by_clip": dict(
                zip(
                    config.clip_order,
                    config.expected_invalid_detections_by_clip,
                    strict=True,
                )
            ),
            "stable_span_distribution": dict(sorted(stable_span_counts.items())),
            "all_stable_paths_mapped_once": True,
            "all_valid_detections_mapped_once": True,
            "invalid_detections_receive_global_id": False,
            "same_frame_same_global_id_violations": 0,
        },
        "population": {
            "soft_max_global_ids": config.soft_max_global_ids,
            "overflow_allowance": config.overflow_allowance,
            "warning_threshold": config.population_warning_threshold,
            "observed_global_paths": len(global_rows),
            "warning": population_warning,
            "policy": "warning_only",
            "forced_merges": 0,
        },
        "counts": stats,
        "global_path_size_distribution": dict(
            sorted(
                (str(size), count)
                for size, count in Counter(
                    int(row["num_stable_tracklets"]) for row in global_rows
                ).items()
            )
        ),
        "input_fingerprints": list(input_fingerprints),
    }


def _build_expected_outputs(
    *,
    identity: Mapping[str, np.ndarray],
    stable: S04FinalizedBundle,
    long_runtime: S05LongCalibrationBundle,
    proposal_bundle: S05ProposalBundle,
    candidates: Sequence[Mapping[str, Any]],
    proposals: Sequence[Mapping[str, Any]],
    config: S05FinalizeConfig,
    config_hash: str,
    input_fingerprints: Sequence[Mapping[str, Any]],
    logger: LogFn,
) -> tuple[dict[str, pa.Table], dict[str, Any], PathCoverResult]:
    nodes, edges = _build_graph_inputs(stable, proposals, config)
    logger(
        f"[s05-finalize] solving all {len(edges):,} provisional proposals across "
        f"{len(nodes):,} stable paths"
    )
    result = solve_operator_approved_path_cover(nodes, edges)
    candidate_rows = _build_candidate_audit_rows(candidates, proposals, result, config)
    mapping_rows, global_rows, mapping_by_stable = _build_global_rows(
        stable, result, config
    )
    detection_table = _build_detection_table(
        identity, stable, mapping_by_stable, global_rows, config
    )
    tables = {
        "candidate_edges": _rows_table(
            candidate_rows,
            GLOBAL_CANDIDATE_EDGES_SCHEMA,
            "S05 global candidate edges",
        ),
        "stable_to_global": _rows_table(
            mapping_rows, STABLE_TO_GLOBAL_SCHEMA, "S05 stable-to-global"
        ),
        "global_tracks": _rows_table(
            global_rows, GLOBAL_TRACKS_SCHEMA, "S05 global tracks"
        ),
        "det_to_global": detection_table,
    }
    report = _build_report(
        config=config,
        config_hash=config_hash,
        stable=stable,
        long_runtime=long_runtime,
        proposal_bundle=proposal_bundle,
        result=result,
        candidate_rows=candidate_rows,
        global_rows=global_rows,
        input_fingerprints=input_fingerprints,
    )
    if (
        tables["candidate_edges"].num_rows != config.expected_candidate_count
        or tables["stable_to_global"].num_rows != config.expected_stable_track_count
        or tables["det_to_global"].num_rows != config.expected_detection_count
        or report["counts"]["num_selected_links"] != len(result.selected_edges)
        or report["counts"]["num_global_tracks"]
        != config.expected_stable_track_count - len(result.selected_edges)
    ):
        raise ContractError("S05 finalize expected output counts differ")
    return tables, report, result


_SUCCESS_KEYS = {
    "schema_version",
    "stage",
    "config_hash",
    "execution_mode",
    "operator_approved",
    "approval_strategy",
    "authorization_basis",
    "certification_claimed",
    "solver_used",
    "path_cover_used",
    "population_affects_solver",
    "input_fingerprints",
    "output_fingerprints",
    "stats",
    "elapsed_sec",
}


def _validate_completed_output(
    directory: Path,
    marker: Mapping[str, Any],
    *,
    config: S05FinalizeConfig,
    config_payload: Mapping[str, Any],
    config_hash: str,
    input_fingerprints: Sequence[Mapping[str, Any]],
    expected_tables: Mapping[str, pa.Table],
    expected_report: Mapping[str, Any],
) -> None:
    consumed_names = (
        config.artifacts.success,
        *sorted(_artifact_names(config).values()),
    )
    before = tuple(
        _output_fingerprint(directory / name, directory) for name in consumed_names
    )
    persisted_marker = _read_json(
        directory / config.artifacts.success, "S05 finalize success marker"
    )
    if persisted_marker != dict(marker):
        raise ContractError("completed S05 finalize marker changed before validation")
    if not isinstance(marker, Mapping) or set(marker) != _SUCCESS_KEYS:
        raise ContractError("completed S05 finalize marker fields differ")
    elapsed = marker.get("elapsed_sec")
    if (
        marker.get("schema_version") != "1.0"
        or marker.get("stage") != _STAGE
        or marker.get("config_hash") != config_hash
        or marker.get("execution_mode") != config.execution_mode
        or marker.get("operator_approved") is not True
        or marker.get("approval_strategy") != config.approval_strategy
        or marker.get("authorization_basis") != config.authorization_basis
        or marker.get("certification_claimed") is not False
        or marker.get("solver_used") is not True
        or marker.get("path_cover_used") is not True
        or marker.get("population_affects_solver") is not False
        or marker.get("input_fingerprints") != list(input_fingerprints)
        or marker.get("stats") != expected_report["counts"]
        or isinstance(elapsed, bool)
        or not isinstance(elapsed, (int, float))
        or not math.isfinite(float(elapsed))
        or float(elapsed) < 0.0
    ):
        raise ContractError("completed S05 finalize marker policy/input/stats differ")
    names = _artifact_names(config)
    expected_files = {*names.values(), config.artifacts.success}
    entries = tuple(directory.iterdir())
    try:
        all_regular = all(
            stat.S_ISREG(path.lstat().st_mode)
            and not stat.S_ISLNK(path.lstat().st_mode)
            for path in entries
        )
    except OSError as exc:
        raise ContractError(f"cannot inspect S05 finalize output tree: {exc}") from exc
    if {path.name for path in entries} != expected_files or not all_regular:
        raise ContractError("completed S05 finalize artifact tree differs")
    records = marker.get("output_fingerprints")
    expected_output_names = sorted(names.values())
    if (
        not isinstance(records, list)
        or [item.get("path") for item in records if isinstance(item, dict)]
        != expected_output_names
    ):
        raise ContractError("completed S05 finalize output fingerprints differ")
    for item in records:
        if (
            not isinstance(item, dict)
            or set(item) != {"path", "size_bytes", "sha256"}
            or isinstance(item["size_bytes"], bool)
            or not isinstance(item["size_bytes"], int)
            or item["size_bytes"] < 0
            or not isinstance(item["sha256"], str)
            or len(item["sha256"]) != 64
        ):
            raise ContractError("completed S05 finalize output fingerprint is invalid")
        current = _output_fingerprint(directory / item["path"], directory)
        if current != item:
            raise ContractError(
                f"completed S05 finalize artifact changed: {item['path']}"
            )
    persisted_config, persisted_payload, persisted_hash = load_s05_finalize_config(
        directory / config.artifacts.effective_config
    )
    persisted_config = replace(
        persisted_config,
        expected_stable_track_count=config.expected_stable_track_count,
        expected_microtrack_count=config.expected_microtrack_count,
        expected_candidate_count=config.expected_candidate_count,
        expected_proposal_count=config.expected_proposal_count,
    )
    if (
        persisted_config != config
        or persisted_payload != dict(config_payload)
        or persisted_hash != config_hash
    ):
        raise ContractError("completed S05 finalize effective config differs")
    schemas = {
        "candidate_edges": GLOBAL_CANDIDATE_EDGES_SCHEMA,
        "stable_to_global": STABLE_TO_GLOBAL_SCHEMA,
        "global_tracks": GLOBAL_TRACKS_SCHEMA,
        "det_to_global": DET_TO_GLOBAL_SCHEMA,
    }
    for key, schema in schemas.items():
        observed = _read_table(
            directory / names[key], schema, f"completed S05 finalize {key}"
        )
        if not expected_tables[key].equals(observed, check_metadata=False):
            raise ContractError(f"completed S05 finalize {key} rows differ")
    report = _read_json(directory / config.artifacts.report, "S05 finalize report")
    if report != dict(expected_report):
        raise ContractError("completed S05 finalize report differs")
    _verify_unchanged(input_fingerprints)
    after = tuple(
        _output_fingerprint(directory / name, directory) for name in consumed_names
    )
    if before != after:
        raise ContractError("S05 finalize artifacts changed while being validated")


def run_s05_finalize(
    ingest_dir: Path,
    microtrack_dir: Path,
    appearance_dir: Path,
    stable_dir: Path,
    long_calibration_dir: Path,
    proposals_dir: Path,
    config_path: Path,
    output_dir: Path,
    *,
    logger: LogFn = log,
) -> dict[str, Any]:
    """Finalize the complete full-sequence S05 proposal graph into global IDs."""

    if not callable(logger):
        raise ContractError("S05 finalize logger must be callable")
    started = time.monotonic()
    (
        ingest_dir,
        microtrack_dir,
        appearance_dir,
        stable_dir,
        long_calibration_dir,
        proposals_dir,
        config_path,
        output_dir,
    ) = (
        path.resolve()
        for path in (
            ingest_dir,
            microtrack_dir,
            appearance_dir,
            stable_dir,
            long_calibration_dir,
            proposals_dir,
            config_path,
            output_dir,
        )
    )
    _reject_path_overlap(
        output_dir,
        (
            ingest_dir,
            microtrack_dir,
            appearance_dir,
            stable_dir,
            long_calibration_dir,
            proposals_dir,
            config_path,
        ),
    )
    config_fingerprint = fingerprint_file(config_path)
    config, config_payload, config_hash = load_s05_finalize_config(config_path)
    if fingerprint_file(config_path) != config_fingerprint:
        raise ContractError("S05 finalize config changed while being loaded")

    logger("[s05-finalize] strict-loading immutable S00/S01/S02 inputs")
    production = load_production_inputs(
        ingest_dir, microtrack_dir, appearance_dir, logger=logger
    )
    logger("[s05-finalize] strict-loading finalized S04 stable paths")
    stable = load_s04_finalized(stable_dir)
    observed_stable_count = len(stable.stable_ids)
    observed_micro_count = len(stable.micro_ids)
    if (
        config.expected_stable_track_count not in (None, observed_stable_count)
        or config.expected_microtrack_count not in (None, observed_micro_count)
    ):
        raise ContractError("S05 finalize configured S04 counts differ")
    config = replace(
        config,
        expected_stable_track_count=observed_stable_count,
        expected_microtrack_count=observed_micro_count,
    )
    logger("[s05-finalize] strict-loading S05A model and uncertified thresholds")
    long_runtime = load_s05_long_calibration(long_calibration_dir)
    logger("[s05-finalize] strict-loading complete S05 proposal graph, not review sample")
    proposal_bundle = load_s05_proposals(
        proposals_dir,
        long_runtime,
        expected_stable_count=observed_stable_count,
    )
    observed_candidate_count = len(proposal_bundle.candidates)
    observed_proposal_count = len(proposal_bundle.proposals)
    if (
        config.expected_candidate_count not in (None, observed_candidate_count)
        or config.expected_proposal_count not in (None, observed_proposal_count)
    ):
        raise ContractError("S05 finalize configured proposal counts differ")
    config = replace(
        config,
        expected_candidate_count=observed_candidate_count,
        expected_proposal_count=observed_proposal_count,
    )
    _validate_fixed_inputs(
        production, stable, long_runtime, proposal_bundle, config
    )
    input_fingerprints = _normalize_fingerprints(
        [
            config_fingerprint,
            *production.input_fingerprints,
            *stable.input_fingerprints,
            *long_runtime.input_fingerprints,
            *long_runtime.output_fingerprints,
            fingerprint_file(long_calibration_dir / "_SUCCESS.json"),
            *proposal_bundle.input_fingerprints,
            *proposal_bundle.consumed_fingerprints,
        ]
    )
    _verify_unchanged(input_fingerprints)
    logger("[s05-finalize] recomputing the exact complete candidate graph for audit")
    candidates, proposals = _recompute_complete_graph(
        production,
        stable,
        long_runtime,
        proposal_bundle,
        logger=logger,
    )
    identity = _load_ingest_identity(ingest_dir, production, config)
    expected_tables, report, result = _build_expected_outputs(
        identity=identity,
        stable=stable,
        long_runtime=long_runtime,
        proposal_bundle=proposal_bundle,
        candidates=candidates,
        proposals=proposals,
        config=config,
        config_hash=config_hash,
        input_fingerprints=input_fingerprints,
        logger=logger,
    )
    success_path = output_dir / config.artifacts.success
    if success_path.is_file():
        marker = _read_json(success_path, "S05 finalize success marker")
        if not isinstance(marker, Mapping):
            raise ContractError("S05 finalize success marker must be an object")
        _validate_completed_output(
            output_dir,
            marker,
            config=config,
            config_payload=config_payload,
            config_hash=config_hash,
            input_fingerprints=input_fingerprints,
            expected_tables=expected_tables,
            expected_report=report,
        )
        logger(f"[s05-finalize] already complete and fully revalidated: {success_path}")
        return dict(marker)
    if output_dir.exists():
        if not output_dir.is_dir():
            raise ContractError(f"S05 finalize output is not a directory: {output_dir}")
        if any(output_dir.iterdir()):
            raise ContractError(
                f"S05 finalize output is non-empty without _SUCCESS.json: {output_dir}"
            )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = output_dir.parent / f".{output_dir.name}.staging-{os.getpid()}"
    if staging.exists():
        raise ContractError(f"S05 finalize staging directory already exists: {staging}")
    staging.mkdir(parents=False)
    try:
        names = _artifact_names(config)
        schemas = {
            "candidate_edges": GLOBAL_CANDIDATE_EDGES_SCHEMA,
            "stable_to_global": STABLE_TO_GLOBAL_SCHEMA,
            "global_tracks": GLOBAL_TRACKS_SCHEMA,
            "det_to_global": DET_TO_GLOBAL_SCHEMA,
        }
        for key, schema in schemas.items():
            _write_table(
                staging / names[key],
                expected_tables[key],
                schema,
                config.parquet_compression,
            )
        _write_json(staging / config.artifacts.report, report)
        _write_json(staging / config.artifacts.effective_config, config_payload)
        _verify_unchanged(input_fingerprints)
        output_fingerprints = [
            _output_fingerprint(staging / name, staging)
            for name in sorted(names.values())
        ]
        marker = {
            "schema_version": "1.0",
            "stage": _STAGE,
            "config_hash": config_hash,
            "execution_mode": config.execution_mode,
            "operator_approved": True,
            "approval_strategy": config.approval_strategy,
            "authorization_basis": config.authorization_basis,
            "certification_claimed": False,
            "solver_used": True,
            "path_cover_used": True,
            "population_affects_solver": False,
            "input_fingerprints": input_fingerprints,
            "output_fingerprints": output_fingerprints,
            "stats": report["counts"],
            "elapsed_sec": float(time.monotonic() - started),
        }
        _write_json(staging / config.artifacts.success, marker)
        _validate_completed_output(
            staging,
            marker,
            config=config,
            config_payload=config_payload,
            config_hash=config_hash,
            input_fingerprints=input_fingerprints,
            expected_tables=expected_tables,
            expected_report=report,
        )
        if output_dir.exists():
            try:
                output_dir.rmdir()
            except OSError as exc:
                raise ContractError(
                    f"S05 finalize output cannot be atomically committed: {output_dir}"
                ) from exc
        os.replace(staging, output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    logger(
        f"[s05-finalize] complete: {len(result.selected_edges):,} links, "
        f"{len(result.paths):,} global paths, {output_dir}"
    )
    return marker


__all__ = ["run_s05_finalize"]
