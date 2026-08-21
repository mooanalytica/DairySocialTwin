"""Deterministic, quality-weighted S02 microtrack prototypes."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from cowtrack.config import ContractError


@dataclass(frozen=True)
class PrototypeBatch:
    """Dense prototype tensor plus per-microtrack and per-sample metadata."""

    micro_ids: np.ndarray
    prototypes: np.ndarray
    prototype_mask: np.ndarray
    sample_inlier_mask: np.ndarray
    sample_outlier_mask: np.ndarray
    medoid_sample_indices: np.ndarray
    num_samples: np.ndarray
    appearance_quality: np.ndarray
    internal_cosine_p10: np.ndarray
    internal_cosine_p50: np.ndarray
    internal_cosine_min: np.ndarray
    outlier_count: np.ndarray


def _validate_inputs(
    embeddings: np.ndarray,
    sample_micro_ids: np.ndarray,
    crop_quality: np.ndarray,
    all_micro_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    embeddings = np.asarray(embeddings)
    if embeddings.ndim != 2 or embeddings.shape[1] <= 0:
        raise ContractError("S02 sample embeddings must have shape [N, D] with D > 0")
    if embeddings.dtype.kind != "f":
        raise ContractError("S02 sample embeddings must have a floating dtype")
    embeddings32 = embeddings.astype(np.float32, copy=False)
    if not np.all(np.isfinite(embeddings32)):
        raise ContractError("S02 sample embeddings must be finite")

    sample_micro_ids = np.asarray(sample_micro_ids)
    crop_quality = np.asarray(crop_quality)
    all_micro_ids = np.asarray(all_micro_ids)
    count = len(embeddings32)
    if sample_micro_ids.ndim != 1 or len(sample_micro_ids) != count:
        raise ContractError("S02 sample_micro_ids must match sample embeddings")
    if crop_quality.ndim != 1 or len(crop_quality) != count:
        raise ContractError("S02 crop_quality must match sample embeddings")
    if all_micro_ids.ndim != 1:
        raise ContractError("S02 all_micro_ids must be one-dimensional")
    if sample_micro_ids.dtype.kind not in "iu" or all_micro_ids.dtype.kind not in "iu":
        raise ContractError("S02 micro IDs must be integer arrays")
    if np.unique(all_micro_ids).size != len(all_micro_ids):
        raise ContractError("S02 all_micro_ids must be unique")
    quality = crop_quality.astype(np.float32, copy=False)
    if not np.all(np.isfinite(quality)) or not np.all(
        (quality >= 0.0) & (quality <= 1.0)
    ):
        raise ContractError("S02 crop_quality must be finite and in [0, 1]")
    if count:
        known = np.isin(sample_micro_ids, all_micro_ids)
        if not np.all(known):
            raise ContractError("S02 sample references an unknown micro_id")
        norms = np.linalg.norm(embeddings32, axis=1)
        if not np.all(np.isfinite(norms)) or not np.allclose(
            norms, 1.0, rtol=0.0, atol=2e-3
        ):
            raise ContractError("S02 sample embeddings must be L2-normalized")
    return (
        embeddings32,
        sample_micro_ids.astype(np.int64, copy=False),
        quality,
        all_micro_ids.astype(np.int64, copy=False),
    )


def _medoid(similarity: np.ndarray, sample_rows: np.ndarray) -> int:
    """Return a local medoid index with global input row as the tie break."""

    if similarity.ndim != 2 or similarity.shape[0] != similarity.shape[1]:
        raise ContractError("S02 internal similarity matrix must be square")
    if not len(similarity):
        raise ContractError("S02 cannot compute a medoid from zero samples")
    scores = np.mean(similarity, axis=1, dtype=np.float64)
    best_score = float(np.max(scores))
    tied = np.flatnonzero(np.isclose(scores, best_score, rtol=0.0, atol=1e-12))
    return int(min((int(index) for index in tied), key=lambda index: int(sample_rows[index])))


def _seed_medoids(
    similarity: np.ndarray,
    sample_rows: np.ndarray,
    *,
    max_prototypes: int,
    new_prototype_cosine: float,
) -> list[int]:
    seeds = [_medoid(similarity, sample_rows)]
    while len(seeds) < max_prototypes:
        coverage = np.max(similarity[:, seeds], axis=1)
        candidate_coverage = float(np.min(coverage))
        if candidate_coverage >= new_prototype_cosine:
            break
        tied = np.flatnonzero(
            np.isclose(coverage, candidate_coverage, rtol=0.0, atol=1e-12)
        )
        candidate = min(
            (int(index) for index in tied), key=lambda index: int(sample_rows[index])
        )
        if candidate in seeds:
            raise ContractError("S02 internal prototype seeding repeated a medoid")
        seeds.append(candidate)
    return seeds


def _refine_medoids(
    similarity: np.ndarray, sample_rows: np.ndarray, seeds: list[int]
) -> tuple[np.ndarray, list[int]]:
    """Run deterministic spherical k-medoids to a fixed point."""

    maximum_iterations = max(1, len(sample_rows) * len(seeds) + 1)
    assignments = np.zeros(len(sample_rows), dtype=np.int64)
    for _ in range(maximum_iterations):
        assignments = np.argmax(similarity[:, seeds], axis=1).astype(
            np.int64, copy=False
        )
        updated: list[int] = []
        for cluster_index in range(len(seeds)):
            members = np.flatnonzero(assignments == cluster_index)
            if not len(members):
                raise ContractError("S02 k-medoids produced an empty prototype cluster")
            local = similarity[np.ix_(members, members)]
            member_medoid = _medoid(local, sample_rows[members])
            updated.append(int(members[member_medoid]))
        if updated == seeds:
            return assignments, seeds
        seeds = updated
    raise ContractError("S02 k-medoids failed to converge deterministically")


def _internal_statistics(similarity: np.ndarray) -> tuple[float, float, float]:
    if len(similarity) < 2:
        return 0.0, 0.0, 0.0
    values = similarity[np.triu_indices(len(similarity), k=1)]
    if not len(values):
        return 0.0, 0.0, 0.0
    return (
        float(np.percentile(values, 10.0)),
        float(np.percentile(values, 50.0)),
        float(np.min(values)),
    )


def build_tracklet_prototypes(
    embeddings: np.ndarray,
    sample_micro_ids: np.ndarray,
    crop_quality: np.ndarray,
    all_micro_ids: np.ndarray,
    *,
    min_samples: int = 3,
    max_prototypes: int = 3,
    outlier_medoid_cosine: float = 0.65,
    outlier_support_cosine: float | None = None,
    new_prototype_cosine: float = 0.92,
) -> PrototypeBatch:
    """Build at most three prototypes for every requested microtrack.

    Samples with zero quality are unusable.  An embedding below the medoid
    threshold is an outlier.  If ``outlier_support_cosine`` is supplied,
    a low-medoid sample is retained only when another sample supports it at or
    above that cosine, allowing a coherent secondary view to survive.  Tracks
    with fewer than ``min_samples`` inliers keep all-zero prototypes and masks.
    """

    embeddings, sample_micro_ids, quality, all_micro_ids = _validate_inputs(
        embeddings, sample_micro_ids, crop_quality, all_micro_ids
    )
    if isinstance(min_samples, bool) or not isinstance(min_samples, (int, np.integer)):
        raise ContractError("S02 prototype min_samples must be an integer")
    if isinstance(max_prototypes, bool) or not isinstance(
        max_prototypes, (int, np.integer)
    ):
        raise ContractError("S02 prototype max_prototypes must be an integer")
    if int(min_samples) != 3:
        raise ContractError("S02 prototype min_samples is fixed at 3")
    if int(max_prototypes) != 3:
        raise ContractError("S02 max_prototypes is fixed at 3")
    thresholds = [outlier_medoid_cosine, new_prototype_cosine]
    if outlier_support_cosine is not None:
        thresholds.append(outlier_support_cosine)
    if not all(math.isfinite(float(value)) and -1.0 <= float(value) <= 1.0 for value in thresholds):
        raise ContractError("S02 prototype cosine thresholds must be in [-1, 1]")

    micro_count = len(all_micro_ids)
    embedding_dim = embeddings.shape[1]
    prototypes = np.zeros(
        (micro_count, int(max_prototypes), embedding_dim), dtype=np.float32
    )
    prototype_mask = np.zeros((micro_count, int(max_prototypes)), dtype=np.bool_)
    medoid_sample_indices = np.full(
        (micro_count, int(max_prototypes)), -1, dtype=np.int64
    )
    sample_inlier_mask = np.zeros(len(embeddings), dtype=np.bool_)
    sample_outlier_mask = np.zeros(len(embeddings), dtype=np.bool_)
    num_samples = np.zeros(micro_count, dtype=np.int16)
    appearance_quality = np.zeros(micro_count, dtype=np.float32)
    internal_cosine_p10 = np.zeros(micro_count, dtype=np.float32)
    internal_cosine_p50 = np.zeros(micro_count, dtype=np.float32)
    internal_cosine_min = np.zeros(micro_count, dtype=np.float32)
    outlier_count = np.zeros(micro_count, dtype=np.int16)

    usable_rows_by_micro: dict[int, list[int]] = {}
    for sample_row in np.flatnonzero(quality > 0.0):
        usable_rows_by_micro.setdefault(
            int(sample_micro_ids[sample_row]), []
        ).append(int(sample_row))

    for micro_row, micro_id in enumerate(all_micro_ids):
        rows = np.asarray(
            usable_rows_by_micro.get(int(micro_id), ()), dtype=np.int64
        )
        if len(rows) > np.iinfo(np.int16).max:
            raise ContractError("S02 microtrack has too many appearance samples for int16")
        num_samples[micro_row] = len(rows)
        if not len(rows):
            continue
        local_embeddings = embeddings[rows]
        similarity = np.clip(
            local_embeddings @ local_embeddings.T, -1.0, 1.0
        ).astype(np.float32, copy=False)
        medoid = _medoid(similarity, rows)
        low_medoid = similarity[:, medoid] < float(outlier_medoid_cosine)
        if outlier_support_cosine is None or len(rows) < 2:
            outliers = low_medoid
        else:
            support = similarity.copy()
            np.fill_diagonal(support, -np.inf)
            outliers = low_medoid & (
                np.max(support, axis=1) < float(outlier_support_cosine)
            )
        sample_outlier_mask[rows[outliers]] = True
        inlier_rows = rows[~outliers]
        sample_inlier_mask[inlier_rows] = True
        outlier_count[micro_row] = int(np.count_nonzero(outliers))
        if len(inlier_rows):
            appearance_quality[micro_row] = float(np.mean(quality[inlier_rows]))
        inlier_similarity = similarity[np.ix_(~outliers, ~outliers)]
        p10, p50, minimum = _internal_statistics(inlier_similarity)
        internal_cosine_p10[micro_row] = p10
        internal_cosine_p50[micro_row] = p50
        internal_cosine_min[micro_row] = minimum
        if len(inlier_rows) < int(min_samples):
            continue

        seeds = _seed_medoids(
            inlier_similarity,
            inlier_rows,
            max_prototypes=int(max_prototypes),
            new_prototype_cosine=float(new_prototype_cosine),
        )
        assignments, seeds = _refine_medoids(
            inlier_similarity, inlier_rows, seeds
        )
        for prototype_index, seed in enumerate(seeds):
            members = np.flatnonzero(assignments == prototype_index)
            member_rows = inlier_rows[members]
            weights = quality[member_rows].astype(np.float64)
            weighted = np.sum(
                embeddings[member_rows].astype(np.float64) * weights[:, None],
                axis=0,
            )
            norm = float(np.linalg.norm(weighted))
            if not math.isfinite(norm) or norm <= np.finfo(np.float64).eps:
                raise ContractError("S02 quality-weighted prototype has zero norm")
            prototype = (weighted / norm).astype(np.float32)
            if not np.all(np.isfinite(prototype)):
                raise ContractError("S02 quality-weighted prototype is non-finite")
            prototypes[micro_row, prototype_index] = prototype
            prototype_mask[micro_row, prototype_index] = True
            medoid_sample_indices[micro_row, prototype_index] = int(
                inlier_rows[seed]
            )

    if np.any(prototypes[~prototype_mask] != 0.0):
        raise ContractError("S02 unused prototype slots must remain zero")
    valid_prototypes = prototypes[prototype_mask]
    if len(valid_prototypes) and not np.allclose(
        np.linalg.norm(valid_prototypes, axis=1), 1.0, rtol=0.0, atol=2e-6
    ):
        raise ContractError("S02 generated prototypes are not L2-normalized")
    return PrototypeBatch(
        micro_ids=all_micro_ids.copy(),
        prototypes=prototypes,
        prototype_mask=prototype_mask,
        sample_inlier_mask=sample_inlier_mask,
        sample_outlier_mask=sample_outlier_mask,
        medoid_sample_indices=medoid_sample_indices,
        num_samples=num_samples,
        appearance_quality=appearance_quality,
        internal_cosine_p10=internal_cosine_p10,
        internal_cosine_p50=internal_cosine_p50,
        internal_cosine_min=internal_cosine_min,
        outlier_count=outlier_count,
    )


def build_micro_prototypes(
    embeddings: np.ndarray,
    sample_micro_ids: np.ndarray,
    crop_quality: np.ndarray,
    all_micro_ids: np.ndarray,
    **kwargs: object,
) -> PrototypeBatch:
    """Compatibility name for the microtrack-specific public operation."""

    return build_tracklet_prototypes(
        embeddings,
        sample_micro_ids,
        crop_quality,
        all_micro_ids,
        **kwargs,
    )


__all__ = ["PrototypeBatch", "build_micro_prototypes", "build_tracklet_prototypes"]
