"""Exact, proposal-only long-gap retrieval over finalized stable paths.

This module deliberately has no graph, solver, path-cover, population-driven
threshold adjustment, or merge operation.  It enumerates a bounded union of
exact appearance neighbours and temporally nearest future paths, then applies
the independently persisted S05A model as provisional review evidence only.
"""

from __future__ import annotations

import math
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Protocol

import numpy as np

from cowtrack.config import ContractError
from cowtrack.linking.features import LONG_FEATURE_SCHEMA
from cowtrack.linking.model import LinkResult
from cowtrack.linking.stable_long_pairs import (
    StablePathPairFeatures,
    StablePathRetrievalDescriptor,
)


LogFn = Callable[[str], None]


class LongPathFeatureProvider(Protocol):
    def build_pair(
        self, source_path_id: int, target_path_id: int
    ) -> StablePathPairFeatures: ...


class LongFeatureScorer(Protocol):
    def score_features(
        self,
        feature_values: Mapping[str, float] | None,
        *,
        appearance_present: bool,
        high_overlap: bool,
        candidate_margin: float | None = None,
    ) -> LinkResult: ...


def _positive_integer(value: Any, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ContractError(f"S05 {name} must be an integer")
    result = int(value)
    if result < 1:
        raise ContractError(f"S05 {name} must be positive")
    return result


def _gap_threshold(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ContractError("S05 minimum gap must be numeric")
    result = float(value)
    if not math.isfinite(result) or not math.isclose(
        result, 5.0, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ContractError("S05 long retrieval requires a gap strictly greater than 5s")
    return result


def _validated_descriptors(
    descriptors: Sequence[StablePathRetrievalDescriptor],
) -> list[StablePathRetrievalDescriptor]:
    if isinstance(descriptors, (str, bytes)):
        raise ContractError("S05 path descriptors must be a sequence")
    rows = sorted(descriptors, key=lambda item: int(item.stable_id))
    if not rows:
        raise ContractError("S05 path descriptors cannot be empty")
    ids = [int(item.stable_id) for item in rows]
    if len(ids) != len(set(ids)) or ids != list(range(len(ids))):
        raise ContractError("S05 stable/path IDs must be unique and dense from zero")
    dimensions: set[int] = set()
    for item in rows:
        if not isinstance(item, StablePathRetrievalDescriptor):
            raise ContractError("S05 path descriptor has the wrong type")
        segment = item.segment
        if segment.stable_id != item.stable_id:
            raise ContractError("S05 path descriptor stable ID differs from segment")
        if (
            not math.isfinite(float(segment.start_time_sec))
            or not math.isfinite(float(segment.end_time_sec))
            or float(segment.start_time_sec) > float(segment.end_time_sec)
            or segment.start_global_frame > segment.end_global_frame
        ):
            raise ContractError("S05 path descriptor has invalid temporal bounds")
        if (item.prototypes is None) != (item.gallery is None):
            raise ContractError("S05 descriptor gallery/prototype availability differs")
        if item.prototypes is None:
            if item.medoid_embedding is not None or item.missing_reason is None:
                raise ContractError("S05 missing gallery descriptor is inconsistent")
            continue
        prototypes = np.asarray(item.prototypes)
        medoid = np.asarray(item.medoid_embedding)
        if (
            prototypes.ndim != 2
            or not 1 <= len(prototypes) <= 3
            or prototypes.shape[1] <= 0
            or prototypes.dtype.kind != "f"
            or medoid.shape != (prototypes.shape[1],)
            or medoid.dtype.kind != "f"
            or not np.all(np.isfinite(prototypes))
            or not np.all(np.isfinite(medoid))
            or not np.allclose(
                np.linalg.norm(prototypes, axis=1), 1.0, rtol=0.0, atol=2e-3
            )
            or not math.isclose(
                float(np.linalg.norm(medoid)), 1.0, rel_tol=0.0, abs_tol=2e-3
            )
            or item.missing_reason is not None
        ):
            raise ContractError("S05 path descriptor has invalid retrieval embeddings")
        dimensions.add(int(prototypes.shape[1]))
    if len(dimensions) > 1:
        raise ContractError("S05 retrieval descriptor dimensions differ")
    return rows


def _exact_gallery_matrix(
    descriptors: Sequence[StablePathRetrievalDescriptor],
    *,
    logger: LogFn | None = None,
    progress_interval_sec: float = 10.0,
) -> tuple[tuple[int, ...], np.ndarray]:
    """Return exact max-prototype cosine for every available gallery pair."""

    usable = [item for item in descriptors if item.prototypes is not None]
    if not usable:
        return (), np.empty((0, 0), dtype=np.float32)
    ids = tuple(int(item.stable_id) for item in usable)
    arrays = [np.asarray(item.prototypes, dtype=np.float32) for item in usable]
    concatenated = np.concatenate(arrays, axis=0)
    offsets = np.cumsum([0, *[len(array) for array in arrays[:-1]]])
    matrix = np.empty((len(usable), len(usable)), dtype=np.float32)
    last_report = time.monotonic()
    for row_index, source in enumerate(arrays):
        per_target_prototype = np.max(source @ concatenated.T, axis=0)
        matrix[row_index] = np.maximum.reduceat(
            per_target_prototype, offsets
        )
        now = time.monotonic()
        if logger is not None and now - last_report >= progress_interval_sec:
            logger(
                f"[s05-propose] exact cosine rows {row_index + 1:,}/{len(usable):,}"
            )
            last_report = now
    np.clip(matrix, -1.0, 1.0, out=matrix)
    if not np.all(np.isfinite(matrix)):
        raise ContractError("S05 exact cosine retrieval produced non-finite scores")
    return ids, matrix


def _gallery_scores(
    source: StablePathRetrievalDescriptor,
    target: StablePathRetrievalDescriptor,
) -> dict[str, float] | None:
    if source.prototypes is None or target.prototypes is None:
        return None
    similarity = np.clip(
        np.asarray(source.prototypes) @ np.asarray(target.prototypes).T,
        -1.0,
        1.0,
    ).astype(np.float64, copy=False)
    flattened = np.sort(similarity, axis=None)[::-1]
    top_count = min(3, len(flattened))
    source_to_target = float(np.mean(np.max(similarity, axis=1)))
    target_to_source = float(np.mean(np.max(similarity, axis=0)))
    return {
        "gallery_score_max": float(flattened[0]),
        "gallery_score_top3": float(np.mean(flattened[:top_count])),
        "gallery_score_src_to_dst": source_to_target,
        "gallery_score_dst_to_src": target_to_source,
        "gallery_score_mutual": 0.5 * (source_to_target + target_to_source),
    }


def _segment_columns(
    prefix: str, descriptor: StablePathRetrievalDescriptor
) -> dict[str, Any]:
    return descriptor.segment.prefixed_row(prefix)


def _best_other_margin(
    ranked: Sequence[tuple[int, float]], candidate_id: int
) -> tuple[int | None, float | None]:
    rank: int | None = None
    score: float | None = None
    for index, (path_id, value) in enumerate(ranked):
        if path_id == candidate_id:
            rank = index + 1
            score = float(value)
            break
    if rank is None or score is None:
        return None, None
    competitors = [value for path_id, value in ranked if path_id != candidate_id]
    if not competitors:
        return rank, None
    return rank, score - float(max(competitors))


def enumerate_long_candidates(
    descriptors: Sequence[StablePathRetrievalDescriptor],
    *,
    appearance_topk: int = 20,
    temporal_nearest_k: int = 5,
    min_gap_sec_exclusive: float = 5.0,
    logger: LogFn | None = None,
    progress_interval_sec: float = 10.0,
) -> list[dict[str, Any]]:
    """Enumerate exact top-k plus nearest-future proposal candidates.

    Appearance ranks are over all structurally legal, role-eligible clean
    galleries—not merely over the emitted top-k union.  Inbound ranks use the
    exact reverse candidate set of all legal past sources.
    """

    topk = _positive_integer(appearance_topk, name="appearance_topk")
    temporal_k = _positive_integer(
        temporal_nearest_k, name="temporal_nearest_k"
    )
    gap_floor = _gap_threshold(min_gap_sec_exclusive)
    if (
        isinstance(progress_interval_sec, bool)
        or not math.isfinite(float(progress_interval_sec))
        or float(progress_interval_sec) <= 0.0
    ):
        raise ContractError("S05 retrieval progress interval must be positive")
    ordered = _validated_descriptors(descriptors)
    by_id = {int(item.stable_id): item for item in ordered}
    gallery_ids, gallery_matrix = _exact_gallery_matrix(
        ordered,
        logger=logger,
        progress_interval_sec=progress_interval_sec,
    )
    gallery_index = {path_id: index for index, path_id in enumerate(gallery_ids)}

    candidate_flags: dict[tuple[int, int], dict[str, bool]] = {}
    out_rankings: dict[int, list[tuple[int, float]]] = {}
    last_report = time.monotonic()
    for completed, source in enumerate(ordered, start=1):
        future = [
            target
            for target in ordered
            if float(target.segment.start_time_sec)
            - float(source.segment.end_time_sec)
            > gap_floor
        ]
        temporal = sorted(
            future,
            key=lambda target: (
                float(target.segment.start_time_sec)
                - float(source.segment.end_time_sec),
                int(target.stable_id),
            ),
        )[:temporal_k]
        appearance: list[StablePathRetrievalDescriptor] = []
        ranked: list[tuple[int, float]] = []
        source_index = gallery_index.get(int(source.stable_id))
        if (
            source_index is not None
            and not source.source_endpoint_review_excluded
        ):
            for target in future:
                target_index = gallery_index.get(int(target.stable_id))
                if (
                    target_index is None
                    or target.target_endpoint_review_excluded
                ):
                    continue
                ranked.append(
                    (
                        int(target.stable_id),
                        float(gallery_matrix[source_index, target_index]),
                    )
                )
            ranked.sort(
                key=lambda item: (
                    -item[1],
                    float(by_id[item[0]].segment.start_time_sec)
                    - float(source.segment.end_time_sec),
                    item[0],
                )
            )
            appearance = [by_id[path_id] for path_id, _ in ranked[:topk]]
            out_rankings[int(source.stable_id)] = ranked
        appearance_ids = {int(item.stable_id) for item in appearance}
        temporal_ids = {int(item.stable_id) for item in temporal}
        for target_id in sorted(appearance_ids | temporal_ids):
            candidate_flags[(int(source.stable_id), target_id)] = {
                "selected_by_appearance_topk": target_id in appearance_ids,
                "selected_by_temporal_nearest": target_id in temporal_ids,
            }
        now = time.monotonic()
        if logger is not None and now - last_report >= progress_interval_sec:
            logger(
                f"[s05-propose] retrieval sources {completed:,}/{len(ordered):,}; "
                f"union={len(candidate_flags):,}"
            )
            last_report = now

    candidates_by_target: dict[int, list[int]] = defaultdict(list)
    candidates_by_source: dict[int, list[int]] = defaultdict(list)
    for source_id, target_id in candidate_flags:
        candidates_by_source[source_id].append(target_id)
        candidates_by_target[target_id].append(source_id)

    in_rankings: dict[int, list[tuple[int, float]]] = {}
    for target_id in sorted(candidates_by_target):
        target = by_id[target_id]
        target_index = gallery_index.get(target_id)
        ranked: list[tuple[int, float]] = []
        if (
            target_index is not None
            and not target.target_endpoint_review_excluded
        ):
            for source in ordered:
                source_id = int(source.stable_id)
                source_index = gallery_index.get(source_id)
                if (
                    source_index is None
                    or source.source_endpoint_review_excluded
                    or float(target.segment.start_time_sec)
                    - float(source.segment.end_time_sec)
                    <= gap_floor
                ):
                    continue
                ranked.append(
                    (source_id, float(gallery_matrix[source_index, target_index]))
                )
            ranked.sort(
                key=lambda item: (
                    -item[1],
                    float(target.segment.start_time_sec)
                    - float(by_id[item[0]].segment.end_time_sec),
                    item[0],
                )
            )
        in_rankings[target_id] = ranked

    rows: list[dict[str, Any]] = []
    for source_id, target_id in sorted(candidate_flags):
        source, target = by_id[source_id], by_id[target_id]
        gap = float(target.segment.start_time_sec - source.segment.end_time_sec)
        if gap <= gap_floor:
            raise ContractError("S05 emitted an overlapping or <=5s candidate")
        appearance_rank_out, best_margin_out = _best_other_margin(
            out_rankings.get(source_id, ()), target_id
        )
        appearance_rank_in, best_margin_in = _best_other_margin(
            in_rankings.get(target_id, ()), source_id
        )
        row: dict[str, Any] = {
            "candidate_id": f"s05c-{source_id:06d}-{target_id:06d}",
            "source_stable_id": source_id,
            "target_stable_id": target_id,
            "gap_sec": gap,
            "temporal_gap_sec": gap,
            "temporally_nonoverlapping": True,
            **candidate_flags[(source_id, target_id)],
            "appearance_rank_out": appearance_rank_out,
            "appearance_rank_in": appearance_rank_in,
            "best_margin_out": best_margin_out,
            "best_margin_in": best_margin_in,
            "source_gallery_present": source.gallery is not None,
            "target_gallery_present": target.gallery is not None,
            "source_endpoint_review_excluded": bool(
                source.source_endpoint_review_excluded
            ),
            "target_endpoint_review_excluded": bool(
                target.target_endpoint_review_excluded
            ),
        }
        row.update(_segment_columns("source", source))
        row.update(_segment_columns("target", target))
        scores = _gallery_scores(source, target)
        if scores is None:
            row.update(
                {
                    "gallery_score_max": None,
                    "gallery_score_top3": None,
                    "gallery_score_src_to_dst": None,
                    "gallery_score_dst_to_src": None,
                    "gallery_score_mutual": None,
                }
            )
        else:
            row.update(scores)
        rows.append(row)
    if len({row["candidate_id"] for row in rows}) != len(rows):
        raise ContractError("S05 deterministic candidate IDs are not unique")
    return rows


def _gallery_provenance_columns(
    prefix: str, value: Any | None
) -> dict[str, Any]:
    names = (
        "sample_ids",
        "gallery_det_ids",
        "embedding_rows",
        "medoid_sample_id",
        "appearance_quality",
        "internal_cosine_p10",
        "internal_cosine_p50",
        "internal_cosine_min",
        "gallery_num_input_samples",
        "gallery_num_overlap_rejected",
        "gallery_num_review_excluded",
        "gallery_num_local_outliers",
        "gallery_max_other_bbox_iou",
        "gallery_max_clean_other_bbox_iou",
    )
    if value is None:
        return {f"{prefix}_{name}": None for name in names}
    return value.prefixed_row(prefix)


def score_long_candidates(
    candidates: Sequence[Mapping[str, Any]],
    feature_provider: LongPathFeatureProvider,
    scorer: LongFeatureScorer,
    *,
    provisional_threshold: float,
    selected_probability_threshold: float,
    selected_margin_threshold: float,
    worker_count: int = 1,
    logger: LogFn | None = None,
    progress_interval_sec: float = 10.0,
) -> list[dict[str, Any]]:
    """Score retrieved edges while preserving the uncertified safety boundary."""

    thresholds = {
        "provisional_threshold": provisional_threshold,
        "selected_probability_threshold": selected_probability_threshold,
        "selected_margin_threshold": selected_margin_threshold,
    }
    checked: dict[str, float] = {}
    for name, raw in thresholds.items():
        if isinstance(raw, bool) or not isinstance(
            raw, (int, float, np.integer, np.floating)
        ):
            raise ContractError(f"S05 {name} must be numeric")
        value = float(raw)
        if not math.isfinite(value):
            raise ContractError(f"S05 {name} must be finite")
        checked[name] = value
    if not 0.0 <= checked["provisional_threshold"] <= 1.0:
        raise ContractError("S05 provisional threshold must be in [0, 1]")
    if not 0.0 <= checked["selected_probability_threshold"] <= 1.0:
        raise ContractError("S05 selected probability threshold must be in [0, 1]")
    if checked["selected_margin_threshold"] < 0.0:
        raise ContractError("S05 selected margin threshold cannot be negative")
    if (
        isinstance(progress_interval_sec, bool)
        or not math.isfinite(float(progress_interval_sec))
        or float(progress_interval_sec) <= 0.0
    ):
        raise ContractError("S05 scoring progress interval must be positive")
    workers = _positive_integer(worker_count, name="worker_count")

    ordered = sorted(candidates, key=lambda row: str(row["candidate_id"]))
    if len({str(row["candidate_id"]) for row in ordered}) != len(ordered):
        raise ContractError("S05 candidate IDs are duplicated")
    def score_one(source: Mapping[str, Any]) -> dict[str, Any]:
        source_id = int(source["source_stable_id"])
        target_id = int(source["target_stable_id"])
        built = feature_provider.build_pair(source_id, target_id)
        out_margin = source.get("best_margin_out")
        in_margin = source.get("best_margin_in")
        candidate_margin = (
            min(float(out_margin), float(in_margin))
            if out_margin is not None and in_margin is not None
            else None
        )
        if built.feature_values is None:
            result = scorer.score_features(
                None,
                appearance_present=False,
                high_overlap=bool(built.high_overlap),
                candidate_margin=candidate_margin,
            )
        else:
            if tuple(built.feature_values) != LONG_FEATURE_SCHEMA:
                raise ContractError("S05 candidate feature order differs")
            expected_max = source.get("gallery_score_max")
            if expected_max is None or not math.isclose(
                float(built.feature_values["prototype_cosine_max"]),
                float(expected_max),
                rel_tol=0.0,
                abs_tol=2e-6,
            ):
                raise ContractError("S05 retrieval and pair gallery scores differ")
            result = scorer.score_features(
                built.feature_values,
                appearance_present=True,
                high_overlap=bool(built.high_overlap),
                candidate_margin=candidate_margin,
            )
        if result.decision == "confirmed":
            raise ContractError("S05 proposal stage cannot emit a confirmed decision")
        probability = result.probability
        probability_pass = bool(
            probability is not None
            and float(probability) >= checked["provisional_threshold"]
        )
        selected_probability_pass = bool(
            probability is not None
            and float(probability) >= checked["selected_probability_threshold"]
        )
        selected_margin_pass = bool(
            candidate_margin is not None
            and float(candidate_margin) >= checked["selected_margin_threshold"]
        )
        selected_evidence = bool(
            selected_probability_pass
            and selected_margin_pass
            and not built.high_overlap
        )
        proposed = result.decision == "provisional"
        if proposed != probability_pass:
            raise ContractError("S05 scorer/provisional threshold decisions differ")
        if built.feature_values is None:
            reason = built.reason or result.reason or "features_unavailable"
        elif proposed and selected_evidence:
            reason = "selected_gate_evidence_uncertified"
        elif proposed and built.high_overlap:
            reason = "provisional_high_overlap"
        elif proposed:
            reason = "provisional_threshold_met"
        else:
            reason = result.reason or "below_provisional_threshold"
        row = dict(source)
        row.update(_gallery_provenance_columns("source", built.source_gallery))
        row.update(_gallery_provenance_columns("target", built.target_gallery))
        for name in LONG_FEATURE_SCHEMA:
            row[name] = (
                None
                if built.feature_values is None
                else float(built.feature_values[name])
            )
        row.update(
            {
                "appearance_present": bool(built.appearance_present),
                "high_overlap": bool(built.high_overlap),
                "model_probability": (
                    None if probability is None else float(probability)
                ),
                "model_raw_score": (
                    None if result.raw_score is None else float(result.raw_score)
                ),
                "candidate_margin": candidate_margin,
                "provisional_threshold": checked["provisional_threshold"],
                "selected_probability_threshold": checked[
                    "selected_probability_threshold"
                ],
                "selected_margin_threshold": checked["selected_margin_threshold"],
                "passes_provisional_threshold": probability_pass,
                "passes_selected_probability_gate": selected_probability_pass,
                "passes_selected_margin_gate": selected_margin_pass,
                "passes_selected_gate": selected_evidence,
                # Readable alias retained in ordinary Python/JSON reports;
                # the exact Arrow contract uses passes_selected_gate.
                "selected_gate_evidence": selected_evidence,
                "decision": result.decision,
                "decision_reason": reason,
                "proposed_for_review": proposed,
                "selected_by_solver": False,
                "confirmed": False,
                "merge_applied": False,
            }
        )
        return row

    rows: list[dict[str, Any]] = []
    last_report = time.monotonic()
    if workers == 1:
        iterator = map(score_one, ordered)
        executor = None
    else:
        executor = ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="s05-score"
        )
        iterator = executor.map(score_one, ordered)
    try:
        for completed, row in enumerate(iterator, start=1):
            rows.append(row)
            now = time.monotonic()
            if logger is not None and (
                now - last_report >= progress_interval_sec
                or completed == len(ordered)
            ):
                logger(
                    f"[s05-propose] scoring {completed:,}/{len(ordered):,}; "
                    f"provisional={sum(item['proposed_for_review'] for item in rows):,}"
                )
                last_report = now
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
    return rows


def build_long_proposals(
    candidates: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Copy only provisional edges into an explicitly pending review table."""

    rows: list[dict[str, Any]] = []
    for candidate in sorted(candidates, key=lambda row: str(row["candidate_id"])):
        if candidate.get("decision") != "provisional" or candidate.get(
            "proposed_for_review"
        ) is not True:
            continue
        probability = candidate.get("model_probability")
        if probability is None:
            raise ContractError("S05 provisional candidate lacks probability")
        row = dict(candidate)
        row.update(
            {
                "proposal_id": f"s05p-{str(candidate['candidate_id'])[5:]}",
                "review_status": "pending",
                "evidence_status": (
                    "selected_gate_uncertified"
                    if candidate.get("passes_selected_gate") is True
                    else "provisional"
                ),
            }
        )
        rows.append(row)
    if len({str(row["proposal_id"]) for row in rows}) != len(rows):
        raise ContractError("S05 deterministic proposal IDs are not unique")
    return rows


def long_review_manifest(
    proposals: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Return JSON-safe proposal references without asserting identity links."""

    rows = [
        {
            "proposal_id": str(row["proposal_id"]),
            "candidate_id": str(row["candidate_id"]),
            "source_stable_id": int(row["source_stable_id"]),
            "target_stable_id": int(row["target_stable_id"]),
            "gap_sec": float(row["gap_sec"]),
            "model_probability": float(row["model_probability"]),
            "candidate_margin": (
                None
                if row.get("candidate_margin") is None
                else float(row["candidate_margin"])
            ),
            "evidence_status": str(
                row.get(
                    "evidence_status",
                    "selected_gate_uncertified"
                    if row.get("passes_selected_gate") is True
                    else "provisional",
                )
            ),
            "review_status": "pending",
        }
        for row in sorted(proposals, key=lambda item: str(item["proposal_id"]))
    ]
    return {
        "schema_version": "1.0",
        "stage": "S05_PROPOSE",
        "execution_mode": "long_proposal_only",
        "confirmed_enabled": False,
        "automatic_merge_allowed": False,
        "solver_used": False,
        "path_cover_used": False,
        "num_proposals": len(rows),
        "proposals": rows,
    }


__all__ = [
    "LongFeatureScorer",
    "LongPathFeatureProvider",
    "build_long_proposals",
    "enumerate_long_candidates",
    "long_review_manifest",
    "score_long_candidates",
]
