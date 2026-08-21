"""Leakage-safe long-link pairs and features for finalized stable paths.

This module is the S05A boundary between the finalized S04 paths and long-link
calibration.  A stable ID is never passed to the micro-only S03 feature store:
stable paths are rebuilt from their constituent detections and every gallery is
rebuilt directly from the corresponding S02 sample embeddings.

The public input deliberately couples a strict :class:`ProductionInputBundle`
to the two anonymous S04 mappings needed to reproduce a path.  A stage may
adapt its strict S04 loader bundle to this small type without introducing a
dependency on a particular loader implementation here.
"""

from __future__ import annotations

import math
import time
from collections.abc import Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any, Callable, Literal

import numpy as np

from cowtrack.config import ContractError
from cowtrack.linking.features import (
    LONG_FEATURE_SCHEMA,
    CleanGallery,
    EndpointContext,
    EndpointGeometry,
    build_clean_gallery,
    has_high_endpoint_overlap,
    pair_features,
    validate_disjoint_sides,
)
from cowtrack.linking.pseudo_pairs import GalleryProvenance
from cowtrack.linking.runtime import ProductionInputBundle


StablePartition = Literal[
    "train", "threshold_selection", "certification", "audit"
]
LogFn = Callable[[str], None]

_PARTITIONS: tuple[StablePartition, ...] = (
    "train",
    "threshold_selection",
    "certification",
    "audit",
)
_PARTITION_RANK = {name: rank for rank, name in enumerate(_PARTITIONS)}


@dataclass(frozen=True)
class StableLongPolicy:
    """Fixed S05A path, gallery, and split policy.

    The defaults mirror ``configs/s05_long_calibration.yaml``.  Keeping the
    small policy explicit lets the stage construct it from its strict config
    while synthetic callers do not need to fabricate an S03 config object.
    """

    train_fraction: float = 0.60
    threshold_selection_fraction: float = 0.10
    certification_fraction: float = 0.10
    audit_fraction: float = 0.20
    min_gap_sec_exclusive: float = 5.0
    clean_max_other_bbox_iou_exclusive: float = 0.25
    min_clean_samples_per_side: int = 3
    outlier_medoid_cosine: float = 0.65
    outlier_support_cosine: float = 0.70
    new_prototype_cosine: float = 0.92
    min_side_internal_cosine_p10: float = 0.70
    high_overlap_iou_threshold: float = 0.25


@dataclass(frozen=True)
class StableLongInput:
    """Strict S00/S01/S02 bytes plus finalized S04 path mappings.

    Both mappings must cover exactly the production micro IDs.  Stable IDs are
    dense from zero and ``micro_order_in_stable`` is contiguous independently
    within every stable path.
    """

    production: ProductionInputBundle
    micro_to_stable: Mapping[int, int]
    micro_order_in_stable: Mapping[int, int]


@dataclass(frozen=True)
class StableSegmentProvenance:
    """Inclusive temporal extent of one stable/path segment."""

    stable_id: int
    constituent_micro_ids: tuple[int, ...]
    start_det_id: int
    end_det_id: int
    start_global_frame: int
    end_global_frame: int
    start_time_sec: float
    end_time_sec: float
    start_clip_id: str
    end_clip_id: str
    num_detections: int

    def prefixed_row(self, prefix: str) -> dict[str, Any]:
        return {
            f"{prefix}_stable_id": self.stable_id,
            f"{prefix}_constituent_micro_ids": list(
                self.constituent_micro_ids
            ),
            f"{prefix}_start_det_id": self.start_det_id,
            f"{prefix}_end_det_id": self.end_det_id,
            f"{prefix}_start_global_frame": self.start_global_frame,
            f"{prefix}_end_global_frame": self.end_global_frame,
            f"{prefix}_start_time_sec": self.start_time_sec,
            f"{prefix}_end_time_sec": self.end_time_sec,
            f"{prefix}_start_clip_id": self.start_clip_id,
            f"{prefix}_end_clip_id": self.end_clip_id,
            f"{prefix}_num_detections": self.num_detections,
        }


@dataclass(frozen=True)
class StableLongPair:
    """One positive or independently grouped simultaneous hard negative."""

    pair_id: str
    candidate_group_id: str
    parent_group_id: str
    partition: StablePartition
    label: bool
    pair_kind: str
    parent_stable_id: int
    source: StableSegmentProvenance
    target: StableSegmentProvenance
    source_gallery: GalleryProvenance
    target_gallery: GalleryProvenance
    cooccurrence_clip_id: str | None
    cooccurrence_global_frame: int | None
    cooccurrence_parent_det_id: int | None
    cooccurrence_other_det_id: int | None
    hard_negative_rank: int
    candidate_margin: float | None
    high_overlap: bool
    features: Mapping[str, float]

    @property
    def split(self) -> str:
        if self.partition == "train":
            return "train"
        if self.partition in {"threshold_selection", "certification"}:
            return "calibration"
        return "audit"

    @property
    def calibration_role(self) -> str:
        if self.partition in {"threshold_selection", "certification"}:
            return self.partition
        return "not_applicable"

    def as_row(self) -> dict[str, Any]:
        """Return a deterministic JSON/Arrow-ready provenance row."""

        if tuple(self.features) != LONG_FEATURE_SCHEMA:
            raise ContractError(
                "S05A stable pair feature order does not match long schema"
            )
        row: dict[str, Any] = {
            "pair_id": self.pair_id,
            "candidate_group_id": self.candidate_group_id,
            "parent_group_id": self.parent_group_id,
            "partition": self.partition,
            "split": self.split,
            "calibration_role": self.calibration_role,
            "mode": "long",
            "label": self.label,
            "pair_kind": self.pair_kind,
            "parent_stable_id": self.parent_stable_id,
            "source_clip_id": self.source.end_clip_id,
            "target_clip_id": self.target.start_clip_id,
            "hard_negative_rank": self.hard_negative_rank,
            "candidate_margin": self.candidate_margin,
            "appearance_present": True,
            "high_overlap": self.high_overlap,
            "cooccurrence_clip_id": self.cooccurrence_clip_id,
            "cooccurrence_global_frame": self.cooccurrence_global_frame,
            "cooccurrence_parent_det_id": self.cooccurrence_parent_det_id,
            "cooccurrence_other_det_id": self.cooccurrence_other_det_id,
            # Retain the exact ordered mapping for model-core callers, while
            # also exposing flat schema columns for Parquet serialization.
            "features": {
                name: float(self.features[name]) for name in LONG_FEATURE_SCHEMA
            },
        }
        row.update(self.source.prefixed_row("source"))
        row.update(self.target.prefixed_row("target"))
        row.update(self.source_gallery.prefixed_row("source"))
        row.update(self.target_gallery.prefixed_row("target"))
        row.update(
            (name, float(self.features[name])) for name in LONG_FEATURE_SCHEMA
        )
        return row

    to_dict = as_row


@dataclass(frozen=True)
class StablePathPairFeatures:
    """Fail-closed result from :class:`StablePathFeatureStore`."""

    feature_values: Mapping[str, float] | None
    appearance_present: bool
    high_overlap: bool
    reason: str | None
    source_gallery: GalleryProvenance | None
    target_gallery: GalleryProvenance | None


@dataclass(frozen=True)
class StablePathRetrievalDescriptor:
    """Whole-path metadata and clean S02 prototypes for exact retrieval.

    The descriptor exposes only anonymous stable/path IDs and clean gallery
    material rebuilt by this module.  Arrays are immutable copies so callers
    cannot alter the feature-store cache or the production embedding matrix.
    Endpoint flags are role-specific because a path contributes its end as a
    source and its start as a target.
    """

    stable_id: int
    segment: StableSegmentProvenance
    gallery: GalleryProvenance | None
    prototypes: np.ndarray | None
    medoid_embedding: np.ndarray | None
    source_endpoint_review_excluded: bool
    target_endpoint_review_excluded: bool
    source_high_overlap: bool
    target_high_overlap: bool
    missing_reason: str | None


@dataclass(frozen=True)
class _Segment:
    stable_id: int
    det_positions: np.ndarray


@dataclass(frozen=True)
class _Prepared:
    data: Any
    stable_ids: tuple[int, ...]
    stable_paths: Mapping[int, np.ndarray]
    stable_to_micros: Mapping[int, tuple[int, ...]]
    det_stable_ids: np.ndarray
    det_path_orders: np.ndarray
    det_row_by_id: Mapping[int, int]
    frame_rows: Mapping[tuple[str, int], tuple[int, ...]]
    sample_det_positions: np.ndarray
    sample_stable_ids: np.ndarray
    sample_path_orders: np.ndarray
    sample_rows_by_stable: Mapping[int, np.ndarray]
    timeline_bounds: Mapping[str, tuple[float, float]]


@dataclass(frozen=True)
class _Positive:
    pair: StableLongPair
    source: _Segment
    target: _Segment
    source_gallery: CleanGallery
    target_gallery: CleanGallery


def _validate_policy(policy: StableLongPolicy) -> None:
    if not isinstance(policy, StableLongPolicy):
        raise ContractError("S05A stable long policy has the wrong type")
    fractions = (
        policy.train_fraction,
        policy.threshold_selection_fraction,
        policy.certification_fraction,
        policy.audit_fraction,
    )
    if any(
        isinstance(value, bool)
        or not math.isfinite(float(value))
        or float(value) <= 0.0
        for value in fractions
    ) or not math.isclose(sum(fractions), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ContractError("S05A partition fractions must be positive and sum to one")
    fixed = (
        math.isclose(
            float(policy.min_gap_sec_exclusive), 5.0, rel_tol=0.0, abs_tol=1e-12
        )
        and math.isclose(
            float(policy.clean_max_other_bbox_iou_exclusive),
            0.25,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        and policy.min_clean_samples_per_side == 3
        and math.isclose(
            float(policy.min_side_internal_cosine_p10),
            0.70,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    )
    if not fixed:
        raise ContractError(
            "S05A requires gap >5s, IoU <0.25, three samples, and p10 >=0.70"
        )
    for name in (
        "outlier_medoid_cosine",
        "outlier_support_cosine",
        "new_prototype_cosine",
        "high_overlap_iou_threshold",
    ):
        value = getattr(policy, name)
        if (
            isinstance(value, bool)
            or not math.isfinite(float(value))
            or not -1.0 <= float(value) <= 1.0
        ):
            raise ContractError(f"S05A {name} is invalid")


def _integer_vector(
    values: object, *, name: str, length: int | None = None
) -> np.ndarray:
    result = np.asarray(values)
    if result.ndim != 1 or result.dtype.kind not in "iu":
        raise ContractError(f"S05A {name} must be a one-dimensional integer array")
    if length is not None and len(result) != length:
        raise ContractError(f"S05A {name} has an inconsistent length")
    if result.dtype.kind == "u" and len(result) and int(np.max(result)) > np.iinfo(
        np.int64
    ).max:
        raise ContractError(f"S05A {name} exceeds signed int64")
    return result.astype(np.int64, copy=False)


def _float_vector(values: object, *, name: str, length: int) -> np.ndarray:
    result = np.asarray(values)
    if result.ndim != 1 or len(result) != length or result.dtype.kind not in "fiu":
        raise ContractError(f"S05A {name} must be a one-dimensional numeric array")
    result = result.astype(np.float64, copy=False)
    if not np.all(np.isfinite(result)):
        raise ContractError(f"S05A {name} must be finite")
    return result


def _mapping(
    values: Mapping[int, int], *, name: str, expected_keys: set[int]
) -> dict[int, int]:
    if not isinstance(values, Mapping):
        raise ContractError(f"S05A {name} must be a mapping")
    result: dict[int, int] = {}
    for raw_key, raw_value in values.items():
        if (
            isinstance(raw_key, (bool, np.bool_))
            or not isinstance(raw_key, (int, np.integer))
            or isinstance(raw_value, (bool, np.bool_))
            or not isinstance(raw_value, (int, np.integer))
        ):
            raise ContractError(f"S05A {name} keys and values must be integers")
        key, value = int(raw_key), int(raw_value)
        if key in result or value < 0:
            raise ContractError(f"S05A {name} contains invalid IDs")
        result[key] = value
    if set(result) != expected_keys:
        raise ContractError(f"S05A {name} must exactly cover production micro IDs")
    return result


def _prepare(inputs: StableLongInput, policy: StableLongPolicy) -> _Prepared:
    _validate_policy(policy)
    if not isinstance(inputs, StableLongInput):
        raise ContractError("S05A stable long input has the wrong type")
    if not isinstance(inputs.production, ProductionInputBundle):
        raise ContractError("S05A requires a strict ProductionInputBundle")
    data = inputs.production.calibration_input

    parent_ids = _integer_vector(data.parent_micro_ids, name="parent_micro_ids")
    if len(parent_ids) == 0 or len(np.unique(parent_ids)) != len(parent_ids):
        raise ContractError("S05A production parent micro IDs must be non-empty/unique")
    parent_set = set(map(int, parent_ids))
    micro_to_stable = _mapping(
        inputs.micro_to_stable,
        name="micro_to_stable",
        expected_keys=parent_set,
    )
    micro_order = _mapping(
        inputs.micro_order_in_stable,
        name="micro_order_in_stable",
        expected_keys=parent_set,
    )
    stable_ids_array = np.asarray(
        sorted(set(micro_to_stable.values())), dtype=np.int64
    )
    if not np.array_equal(
        stable_ids_array, np.arange(len(stable_ids_array), dtype=np.int64)
    ):
        raise ContractError("S05A stable IDs must be dense from zero")
    stable_ids = tuple(map(int, stable_ids_array))

    det_ids = _integer_vector(data.det_ids, name="det_ids")
    det_count = len(det_ids)
    if det_count == 0 or len(np.unique(det_ids)) != det_count:
        raise ContractError("S05A detection IDs must be non-empty and unique")
    det_micro = _integer_vector(
        data.det_micro_ids, name="det_micro_ids", length=det_count
    )
    det_order = _integer_vector(
        data.det_order_in_micro, name="det_order_in_micro", length=det_count
    )
    det_frames = _integer_vector(
        data.det_global_frames, name="det_global_frames", length=det_count
    )
    det_times = _float_vector(
        data.det_global_time_sec, name="det_global_time_sec", length=det_count
    )
    clips = np.asarray(data.det_clip_ids, dtype=object)
    if (
        clips.ndim != 1
        or len(clips) != det_count
        or any(not isinstance(value, str) or not value for value in clips)
    ):
        raise ContractError("S05A detection clip IDs must be non-empty strings")
    if set(map(int, np.unique(det_micro))) != parent_set:
        raise ContractError("S05A detection micro IDs differ from production parents")
    for name in (
        "det_cx_norm",
        "det_cy_norm",
        "det_w_norm",
        "det_h_norm",
        "det_other_bbox_max_iou",
        "det_boundary_distance",
    ):
        _float_vector(getattr(data, name), name=name, length=det_count)
    review = np.asarray(data.det_review_excluded)
    if review.ndim != 1 or len(review) != det_count or review.dtype != np.bool_:
        raise ContractError("S05A det_review_excluded must be a boolean vector")
    det_row_by_id = {int(det_id): row for row, det_id in enumerate(det_ids)}

    if not isinstance(inputs.production.micro_paths, Mapping) or set(
        map(int, inputs.production.micro_paths)
    ) != parent_set:
        raise ContractError("S05A production micro paths differ from parent IDs")
    canonical_micro_paths: dict[int, np.ndarray] = {}
    covered_positions = np.zeros(det_count, dtype=np.bool_)
    for micro_id in sorted(parent_set):
        positions = _integer_vector(
            inputs.production.micro_paths[micro_id],
            name=f"micro_paths[{micro_id}]",
        )
        if (
            len(positions) == 0
            or np.any(positions < 0)
            or np.any(positions >= det_count)
            or np.any(covered_positions[positions])
            or not np.all(det_micro[positions] == micro_id)
            or not np.array_equal(
                det_order[positions], np.arange(len(positions), dtype=np.int64)
            )
        ):
            raise ContractError(
                f"S05A production micro path {micro_id} disagrees with columns"
            )
        if len(positions) > 1 and np.any(np.diff(det_times[positions]) <= 0.0):
            raise ContractError(f"S05A micro {micro_id} path is not time ordered")
        covered_positions[positions] = True
        canonical_micro_paths[micro_id] = positions
    if not np.all(covered_positions):
        raise ContractError("S05A production micro paths do not cover detections")

    stable_to_micros: dict[int, tuple[int, ...]] = {}
    stable_paths: dict[int, np.ndarray] = {}
    det_stable = np.full(det_count, -1, dtype=np.int64)
    det_path_order = np.full(det_count, -1, dtype=np.int64)
    micros_by_stable: dict[int, list[int]] = {
        stable_id: [] for stable_id in stable_ids
    }
    for micro_id in sorted(parent_set):
        micros_by_stable[micro_to_stable[micro_id]].append(micro_id)
    for stable_id in stable_ids:
        micros = micros_by_stable[stable_id]
        micros.sort(key=lambda micro_id: (micro_order[micro_id], micro_id))
        if [micro_order[micro_id] for micro_id in micros] != list(
            range(len(micros))
        ):
            raise ContractError(
                f"S05A stable {stable_id} micro order is not contiguous"
            )
        path = np.concatenate([canonical_micro_paths[micro_id] for micro_id in micros])
        if len(path) > 1 and np.any(np.diff(det_times[path]) <= 0.0):
            raise ContractError(
                f"S05A stable {stable_id} constituent paths overlap or reverse time"
            )
        keys = [(str(clips[pos]), int(det_frames[pos])) for pos in path]
        if len(keys) != len(set(keys)):
            raise ContractError(
                f"S05A stable {stable_id} appears more than once in a frame"
            )
        stable_to_micros[stable_id] = tuple(micros)
        stable_paths[stable_id] = path
        det_stable[path] = stable_id
        det_path_order[path] = np.arange(len(path), dtype=np.int64)
    if np.any(det_stable < 0) or np.any(det_path_order < 0):
        raise ContractError("S05A stable paths do not partition detections")

    frame_lists: dict[tuple[str, int], list[int]] = {}
    for position in range(det_count):
        frame_lists.setdefault(
            (str(clips[position]), int(det_frames[position])), []
        ).append(position)
    frame_rows: dict[tuple[str, int], tuple[int, ...]] = {}
    for key, positions in frame_lists.items():
        positions.sort(
            key=lambda pos: (int(det_stable[pos]), int(det_ids[pos]))
        )
        frame_rows[key] = tuple(positions)

    sample_ids = _integer_vector(data.sample_ids, name="sample_ids")
    sample_count = len(sample_ids)
    sample_micro = _integer_vector(
        data.sample_micro_ids, name="sample_micro_ids", length=sample_count
    )
    sample_det = _integer_vector(
        data.sample_det_ids, name="sample_det_ids", length=sample_count
    )
    embedding_rows = _integer_vector(
        data.sample_embedding_rows,
        name="sample_embedding_rows",
        length=sample_count,
    )
    if (
        len(np.unique(sample_ids)) != sample_count
        or len(np.unique(sample_det)) != sample_count
        or len(np.unique(embedding_rows)) != sample_count
    ):
        raise ContractError("S05A sample IDs/detections/embedding rows must be unique")
    embeddings = np.asarray(data.embeddings)
    if (
        embeddings.ndim != 2
        or embeddings.shape[1] <= 0
        or embeddings.dtype.kind != "f"
        or np.any(embedding_rows < 0)
        or np.any(embedding_rows >= len(embeddings))
        or not np.all(np.isfinite(embeddings))
    ):
        raise ContractError("S05A sample embeddings/mapping are invalid")
    if len(embedding_rows):
        selected_embeddings = embeddings[embedding_rows].astype(np.float32, copy=False)
        if not np.allclose(
            np.linalg.norm(selected_embeddings, axis=1),
            1.0,
            rtol=0.0,
            atol=2e-3,
        ):
            raise ContractError("S05A sample embeddings must be L2-normalized")
    for name in ("sample_crop_quality", "sample_other_bbox_max_iou"):
        values = _float_vector(getattr(data, name), name=name, length=sample_count)
        if np.any((values < 0.0) | (values > 1.0)):
            raise ContractError(f"S05A {name} must be in [0, 1]")
    inlier = np.asarray(data.sample_s02_inlier)
    if inlier.ndim != 1 or len(inlier) != sample_count or inlier.dtype != np.bool_:
        raise ContractError("S05A sample_s02_inlier must be a boolean vector")

    sample_det_positions = np.empty(sample_count, dtype=np.int64)
    for row, (det_id, micro_id) in enumerate(
        zip(sample_det, sample_micro, strict=True)
    ):
        position = det_row_by_id.get(int(det_id))
        if position is None or int(det_micro[position]) != int(micro_id):
            raise ContractError("S05A sample/detection/micro join is inconsistent")
        sample_det_positions[row] = position
    sample_stable = det_stable[sample_det_positions]
    sample_path_order = det_path_order[sample_det_positions]
    sample_rows_by_stable: dict[int, np.ndarray] = {}
    canonical_sample_rows = np.lexsort(
        (
            sample_ids,
            sample_det,
            embedding_rows,
            sample_path_order,
            sample_stable,
        )
    )
    ordered_sample_stable = sample_stable[canonical_sample_rows]
    for stable_id in stable_ids:
        start = int(np.searchsorted(ordered_sample_stable, stable_id, side="left"))
        end = int(np.searchsorted(ordered_sample_stable, stable_id, side="right"))
        rows = canonical_sample_rows[start:end]
        sample_rows_by_stable[stable_id] = rows

    timeline_clips = np.asarray(data.timeline_clip_ids, dtype=object)
    timeline_count = len(timeline_clips)
    timeline_starts = _float_vector(
        data.timeline_clip_start_time_sec,
        name="timeline_clip_start_time_sec",
        length=timeline_count,
    )
    timeline_ends = _float_vector(
        data.timeline_clip_end_time_sec,
        name="timeline_clip_end_time_sec",
        length=timeline_count,
    )
    if (
        timeline_clips.ndim != 1
        or timeline_count == 0
        or any(not isinstance(value, str) or not value for value in timeline_clips)
        or len(set(map(str, timeline_clips))) != timeline_count
        or np.any(timeline_ends <= timeline_starts)
    ):
        raise ContractError("S05A clip timeline is invalid")
    timeline_bounds = {
        str(clip): (float(start), float(end))
        for clip, start, end in zip(
            timeline_clips, timeline_starts, timeline_ends, strict=True
        )
    }
    if set(map(str, np.unique(clips))) != set(timeline_bounds):
        raise ContractError("S05A detection clips differ from clip timeline")
    for clip, (start, end) in timeline_bounds.items():
        values = det_times[clips == clip]
        if not len(values) or np.any((values < start) | (values > end)):
            raise ContractError(f"S05A detections fall outside clip timeline {clip}")

    return _Prepared(
        data=data,
        stable_ids=stable_ids,
        stable_paths=MappingProxyType(stable_paths),
        stable_to_micros=MappingProxyType(stable_to_micros),
        det_stable_ids=det_stable,
        det_path_orders=det_path_order,
        det_row_by_id=MappingProxyType(det_row_by_id),
        frame_rows=MappingProxyType(frame_rows),
        sample_det_positions=sample_det_positions,
        sample_stable_ids=sample_stable,
        sample_path_orders=sample_path_order,
        sample_rows_by_stable=MappingProxyType(sample_rows_by_stable),
        timeline_bounds=MappingProxyType(timeline_bounds),
    )


def _partition_for_time(
    time_sec: float,
    clip_start: float,
    clip_end: float,
    policy: StableLongPolicy,
) -> StablePartition:
    if not clip_start <= time_sec <= clip_end or clip_end <= clip_start:
        raise ContractError("S05A partition time is outside its clip")
    duration = clip_end - clip_start
    train_end = clip_start + policy.train_fraction * duration
    selection_end = train_end + policy.threshold_selection_fraction * duration
    certification_end = (
        selection_end + policy.certification_fraction * duration
    )
    if time_sec < train_end:
        return "train"
    if time_sec < selection_end:
        return "threshold_selection"
    if time_sec < certification_end:
        return "certification"
    return "audit"


def _assign_partitions_prepared(
    prepared: _Prepared, policy: StableLongPolicy
) -> dict[int, StablePartition]:
    data = prepared.data
    result: dict[int, StablePartition] = {}
    for stable_id in prepared.stable_ids:
        path = prepared.stable_paths[stable_id]
        choices: list[StablePartition] = []
        for clip in sorted({str(data.det_clip_ids[pos]) for pos in path}):
            clip_positions = path[
                np.asarray(data.det_clip_ids, dtype=object)[path] == clip
            ]
            start, end = prepared.timeline_bounds[clip]
            choices.extend(
                (
                    _partition_for_time(
                        float(np.min(data.det_global_time_sec[clip_positions])),
                        start,
                        end,
                        policy,
                    ),
                    _partition_for_time(
                        float(np.max(data.det_global_time_sec[clip_positions])),
                        start,
                        end,
                        policy,
                    ),
                )
            )
        if not choices:
            raise ContractError(f"S05A stable {stable_id} is absent from all clips")
        result[stable_id] = max(choices, key=_PARTITION_RANK.__getitem__)
    return result


def assign_stable_partitions(
    inputs: StableLongInput, policy: StableLongPolicy | None = None
) -> dict[int, StablePartition]:
    """Assign stable parents before mining with held-out-wins precedence."""

    actual_policy = StableLongPolicy() if policy is None else policy
    prepared = _prepare(inputs, actual_policy)
    return _assign_partitions_prepared(prepared, actual_policy)


def _gallery(
    prepared: _Prepared,
    segment: _Segment,
    policy: StableLongPolicy,
    cache: dict[tuple[int, int, int], CleanGallery | None],
) -> CleanGallery | None:
    path = prepared.stable_paths[segment.stable_id]
    positions = segment.det_positions
    if len(positions) == 0:
        raise ContractError("S05A cannot build a gallery for an empty segment")
    start_order = int(prepared.det_path_orders[positions[0]])
    end_order = int(prepared.det_path_orders[positions[-1]])
    if not np.array_equal(positions, path[start_order : end_order + 1]):
        raise ContractError("S05A gallery segment is not contiguous in its path")
    key = (segment.stable_id, start_order, end_order)
    if key in cache:
        return cache[key]
    rows = prepared.sample_rows_by_stable[segment.stable_id]
    if len(rows):
        sample_orders = prepared.sample_path_orders[rows]
        rows = rows[(sample_orders >= start_order) & (sample_orders <= end_order)]
    data = prepared.data
    review = data.det_review_excluded[prepared.sample_det_positions[rows]]
    gallery = build_clean_gallery(
        np.asarray(data.embeddings)[data.sample_embedding_rows[rows]],
        sample_ids=data.sample_ids[rows],
        det_ids=data.sample_det_ids[rows],
        embedding_rows=data.sample_embedding_rows[rows],
        crop_quality=np.asarray(data.sample_crop_quality[rows], dtype=np.float32),
        other_bbox_max_iou=np.asarray(
            data.sample_other_bbox_max_iou[rows], dtype=np.float32
        ),
        s02_inlier_mask=data.sample_s02_inlier[rows],
        review_excluded_mask=review,
        max_other_bbox_iou=policy.clean_max_other_bbox_iou_exclusive,
        min_clean_inliers=policy.min_clean_samples_per_side,
        outlier_medoid_cosine=policy.outlier_medoid_cosine,
        outlier_support_cosine=policy.outlier_support_cosine,
        new_prototype_cosine=policy.new_prototype_cosine,
    )
    if gallery is not None and (
        float(gallery.internal_cosine_p10)
        < float(policy.min_side_internal_cosine_p10)
    ):
        gallery = None
    cache[key] = gallery
    return gallery


def _endpoint_geometry(prepared: _Prepared, position: int) -> EndpointGeometry:
    data = prepared.data
    return EndpointGeometry(
        time_sec=float(data.det_global_time_sec[position]),
        cx_norm=float(data.det_cx_norm[position]),
        cy_norm=float(data.det_cy_norm[position]),
        w_norm=float(data.det_w_norm[position]),
        h_norm=float(data.det_h_norm[position]),
    )


def _endpoint_context(prepared: _Prepared, position: int) -> EndpointContext:
    data = prepared.data
    return EndpointContext(
        max_other_iou=float(data.det_other_bbox_max_iou[position]),
        boundary_distance=float(data.det_boundary_distance[position]),
        clip_id=str(data.det_clip_ids[position]),
    )


def _build_features(
    prepared: _Prepared,
    source: _Segment,
    target: _Segment,
    source_gallery: CleanGallery,
    target_gallery: CleanGallery,
    policy: StableLongPolicy,
) -> tuple[dict[str, float], bool]:
    validate_disjoint_sides(source_gallery, target_gallery)
    source_end = int(source.det_positions[-1])
    target_start = int(target.det_positions[0])
    gap = float(
        prepared.data.det_global_time_sec[target_start]
        - prepared.data.det_global_time_sec[source_end]
    )
    if gap <= policy.min_gap_sec_exclusive:
        raise ContractError("S05A long features require a gap strictly greater than 5s")
    source_context = _endpoint_context(prepared, source_end)
    target_context = _endpoint_context(prepared, target_start)
    values = pair_features(
        source_gallery,
        target_gallery,
        _endpoint_geometry(prepared, source_end),
        _endpoint_geometry(prepared, target_start),
        source_context,
        target_context,
        mode="long",
    )
    high_overlap = has_high_endpoint_overlap(
        source_context,
        target_context,
        threshold=policy.high_overlap_iou_threshold,
    ) or bool(
        source_gallery.max_other_bbox_iou >= policy.high_overlap_iou_threshold
        or target_gallery.max_other_bbox_iou >= policy.high_overlap_iou_threshold
    )
    return values, high_overlap


def _provenance(prepared: _Prepared, segment: _Segment) -> StableSegmentProvenance:
    data = prepared.data
    positions = segment.det_positions
    first, last = int(positions[0]), int(positions[-1])
    micros = tuple(
        dict.fromkeys(int(data.det_micro_ids[position]) for position in positions)
    )
    return StableSegmentProvenance(
        stable_id=segment.stable_id,
        constituent_micro_ids=micros,
        start_det_id=int(data.det_ids[first]),
        end_det_id=int(data.det_ids[last]),
        start_global_frame=int(data.det_global_frames[first]),
        end_global_frame=int(data.det_global_frames[last]),
        start_time_sec=float(data.det_global_time_sec[first]),
        end_time_sec=float(data.det_global_time_sec[last]),
        start_clip_id=str(data.det_clip_ids[first]),
        end_clip_id=str(data.det_clip_ids[last]),
        num_detections=len(positions),
    )


def _positive_for_stable(
    prepared: _Prepared,
    stable_id: int,
    partition: StablePartition,
    policy: StableLongPolicy,
    cache: dict[tuple[int, int, int], CleanGallery | None],
) -> _Positive | None:
    path = prepared.stable_paths[stable_id]
    rows = prepared.sample_rows_by_stable[stable_id]
    if len(rows) < 2 * policy.min_clean_samples_per_side:
        return None
    candidate_orders = np.unique(prepared.sample_path_orders[rows])
    source: _Segment | None = None
    source_gallery: CleanGallery | None = None
    for end_order in candidate_orders:
        candidate = _Segment(stable_id, path[: int(end_order) + 1])
        gallery = _gallery(prepared, candidate, policy, cache)
        if gallery is not None:
            source, source_gallery = candidate, gallery
            break
    target: _Segment | None = None
    target_gallery: CleanGallery | None = None
    for start_order in candidate_orders[::-1]:
        candidate = _Segment(stable_id, path[int(start_order) :])
        gallery = _gallery(prepared, candidate, policy, cache)
        if gallery is not None:
            target, target_gallery = candidate, gallery
            break
    if (
        source is None
        or target is None
        or source_gallery is None
        or target_gallery is None
    ):
        return None
    source_end = int(source.det_positions[-1])
    target_start = int(target.det_positions[0])
    if int(prepared.det_path_orders[source_end]) >= int(
        prepared.det_path_orders[target_start]
    ):
        return None
    gap = float(
        prepared.data.det_global_time_sec[target_start]
        - prepared.data.det_global_time_sec[source_end]
    )
    if gap <= policy.min_gap_sec_exclusive:
        return None
    validate_disjoint_sides(source_gallery, target_gallery)
    features, high_overlap = _build_features(
        prepared,
        source,
        target,
        source_gallery,
        target_gallery,
        policy,
    )
    source_provenance = _provenance(prepared, source)
    target_provenance = _provenance(prepared, target)
    group = f"partition-{partition}:parent-stable-{stable_id}"
    pair = StableLongPair(
        pair_id=(
            f"{group}:positive:a-{source_provenance.end_det_id}:"
            f"b-{target_provenance.start_det_id}"
        ),
        candidate_group_id=group,
        parent_group_id=f"stable-{stable_id}",
        partition=partition,
        label=True,
        pair_kind="stable_path_pseudo_positive",
        parent_stable_id=stable_id,
        source=source_provenance,
        target=target_provenance,
        source_gallery=GalleryProvenance.from_gallery(source_gallery),
        target_gallery=GalleryProvenance.from_gallery(target_gallery),
        cooccurrence_clip_id=None,
        cooccurrence_global_frame=None,
        cooccurrence_parent_det_id=None,
        cooccurrence_other_det_id=None,
        hard_negative_rank=0,
        candidate_margin=None,
        high_overlap=high_overlap,
        features=MappingProxyType(features),
    )
    return _Positive(pair, source, target, source_gallery, target_gallery)


def _negative_candidates(
    prepared: _Prepared,
    positive: _Positive,
    partitions: Mapping[int, StablePartition],
    policy: StableLongPolicy,
    cache: dict[tuple[int, int, int], CleanGallery | None],
) -> list[StableLongPair]:
    data = prepared.data
    target_start = int(positive.target.det_positions[0])
    frame_key = (
        str(data.det_clip_ids[target_start]),
        int(data.det_global_frames[target_start]),
    )
    parent = positive.pair.parent_stable_id
    result: list[StableLongPair] = []
    for position in prepared.frame_rows.get(frame_key, ()):
        other = int(prepared.det_stable_ids[position])
        if other == parent or partitions[other] != positive.pair.partition:
            continue
        if bool(data.det_review_excluded[position]):
            continue
        start_order = int(prepared.det_path_orders[position])
        target = _Segment(other, prepared.stable_paths[other][start_order:])
        target_gallery = _gallery(prepared, target, policy, cache)
        if target_gallery is None:
            continue
        features, high_overlap = _build_features(
            prepared,
            positive.source,
            target,
            positive.source_gallery,
            target_gallery,
            policy,
        )
        source_provenance = _provenance(prepared, positive.source)
        target_provenance = _provenance(prepared, target)
        low, high = sorted((parent, other))
        group = (
            f"partition-{positive.pair.partition}:stable-pair-{low}-{high}"
        )
        result.append(
            StableLongPair(
                pair_id=(
                    f"{group}:negative:parent-{parent}:"
                    f"a-{source_provenance.end_det_id}:"
                    f"b-{target_provenance.start_det_id}"
                ),
                candidate_group_id=group,
                parent_group_id=f"stable-{parent}",
                partition=positive.pair.partition,
                label=False,
                pair_kind="simultaneous_stable_hard_negative",
                parent_stable_id=parent,
                source=source_provenance,
                target=target_provenance,
                source_gallery=GalleryProvenance.from_gallery(
                    positive.source_gallery
                ),
                target_gallery=GalleryProvenance.from_gallery(target_gallery),
                cooccurrence_clip_id=frame_key[0],
                cooccurrence_global_frame=frame_key[1],
                cooccurrence_parent_det_id=int(data.det_ids[target_start]),
                cooccurrence_other_det_id=int(data.det_ids[position]),
                hard_negative_rank=0,
                candidate_margin=None,
                high_overlap=high_overlap,
                features=MappingProxyType(features),
            )
        )
    return result


def _negative_sort_key(pair: StableLongPair) -> tuple[Any, ...]:
    return (
        -float(pair.features["prototype_cosine_max"]),
        -float(pair.features["prototype_cosine_top3_mean"]),
        -float(pair.features["medoid_cosine"]),
        pair.parent_stable_id,
        pair.source.end_det_id,
        pair.target.stable_id,
        pair.target.start_det_id,
        pair.pair_id,
    )


def generate_stable_long_pairs(
    inputs: StableLongInput,
    policy: StableLongPolicy | None = None,
    *,
    logger: LogFn | None = None,
    progress_interval_sec: float = 10.0,
) -> tuple[StableLongPair, ...]:
    """Build one positive per eligible stable parent and deduplicated negatives.

    Partition assignment is completed before any gallery or candidate mining.
    Within a partition, an unordered pair of distinct stable IDs can contribute
    at most one negative regardless of how many frame rows or orientations
    prove simultaneity.
    """

    actual_policy = StableLongPolicy() if policy is None else policy
    if logger is not None and not callable(logger):
        raise ContractError("S05A logger must be callable")
    if (
        isinstance(progress_interval_sec, bool)
        or not isinstance(progress_interval_sec, (int, float))
        or not math.isfinite(float(progress_interval_sec))
        or float(progress_interval_sec) <= 0.0
    ):
        raise ContractError("S05A progress interval must be positive and finite")
    interval = float(progress_interval_sec)
    prepared = _prepare(inputs, actual_policy)
    partitions = _assign_partitions_prepared(prepared, actual_policy)
    cache: dict[tuple[int, int, int], CleanGallery | None] = {}
    positives: list[_Positive] = []
    last_progress = time.monotonic()
    total_stables = len(prepared.stable_ids)
    for completed, stable_id in enumerate(prepared.stable_ids, start=1):
        candidate = _positive_for_stable(
            prepared,
            stable_id,
            partitions[stable_id],
            actual_policy,
            cache,
        )
        if candidate is not None:
            positives.append(candidate)
        now = time.monotonic()
        if logger is not None and (
            completed == total_stables or now - last_progress >= interval
        ):
            logger(
                f"[s05a] positive mining: stable {completed:,}/"
                f"{total_stables:,}, accepted={len(positives):,}"
            )
            last_progress = now
    if logger is not None:
        logger(
            f"[s05a] stable pseudo-positives: {len(positives):,}/"
            f"{len(prepared.stable_ids):,}"
        )

    negative_pool: list[StableLongPair] = []
    last_progress = time.monotonic()
    total_positives = len(positives)
    for completed, positive in enumerate(positives, start=1):
        negative_pool.extend(
            _negative_candidates(
                prepared,
                positive,
                partitions,
                actual_policy,
                cache,
            )
        )
        now = time.monotonic()
        if logger is not None and (
            completed == total_positives or now - last_progress >= interval
        ):
            logger(
                f"[s05a] negative mining: positive {completed:,}/"
                f"{total_positives:,}, raw_candidates={len(negative_pool):,}"
            )
            last_progress = now
    negative_pool.sort(key=_negative_sort_key)
    # The candidate group encodes partition + unordered stable pair.  Taking
    # the first item after the total sort is deterministic and frame-row-count
    # invariant.
    selected_by_group: dict[str, StableLongPair] = {}
    for pair in negative_pool:
        selected_by_group.setdefault(pair.candidate_group_id, pair)
    negatives = list(selected_by_group.values())

    by_parent: dict[int, list[StableLongPair]] = {}
    for pair in negatives:
        by_parent.setdefault(pair.parent_stable_id, []).append(pair)
    ranked_negatives: list[StableLongPair] = []
    for parent in sorted(by_parent):
        candidates = sorted(by_parent[parent], key=_negative_sort_key)
        positive_score = float(
            next(
                item.pair.features["prototype_cosine_max"]
                for item in positives
                if item.pair.parent_stable_id == parent
            )
        )
        candidate_scores = [
            float(pair.features["prototype_cosine_max"])
            for pair in candidates
        ]
        for rank, pair in enumerate(candidates, start=1):
            score = candidate_scores[rank - 1]
            other_scores = [positive_score] + [
                value
                for index, value in enumerate(candidate_scores)
                if index != rank - 1
            ]
            ranked_negatives.append(
                replace(
                    pair,
                    hard_negative_rank=rank,
                    # A candidate margin is always own score minus the best
                    # competing candidate in the same positive-parent group.
                    candidate_margin=score - max(other_scores),
                )
            )
    hardest_by_parent = {
        parent: max(
            float(pair.features["prototype_cosine_max"])
            for pair in candidates
        )
        for parent, candidates in by_parent.items()
    }
    positive_pairs = [
        replace(
            item.pair,
            candidate_margin=(
                float(item.pair.features["prototype_cosine_max"])
                - hardest_by_parent[item.pair.parent_stable_id]
                if item.pair.parent_stable_id in hardest_by_parent
                # No independently proven competitor supplies no separation
                # evidence.  Explicit zero is the conservative, all-rows-
                # finite representation consumed by the calibration core.
                else 0.0
            ),
        )
        for item in positives
    ]

    output = positive_pairs + ranked_negatives
    output.sort(
        key=lambda pair: (
            _PARTITION_RANK[pair.partition],
            pair.parent_stable_id,
            0 if pair.label else 1,
            pair.hard_negative_rank,
            pair.target.stable_id,
            pair.pair_id,
        )
    )
    if len({pair.pair_id for pair in output}) != len(output):
        raise ContractError("S05A pair IDs are not unique")
    positive_parents = [pair.parent_stable_id for pair in output if pair.label]
    if len(positive_parents) != len(set(positive_parents)):
        raise ContractError("S05A emitted more than one positive per stable parent")
    negative_groups = [
        (pair.partition, pair.candidate_group_id)
        for pair in output
        if not pair.label
    ]
    if len(negative_groups) != len(set(negative_groups)):
        raise ContractError("S05A negative stable-pair groups are not unique")
    if logger is not None:
        logger(
            f"[s05a] long pairs complete: positives={len(positive_pairs):,}, "
            f"hard_negatives={len(ranked_negatives):,}"
        )
    return tuple(output)


def stable_long_pairs_as_rows(
    pairs: tuple[StableLongPair, ...] | list[StableLongPair],
) -> list[dict[str, Any]]:
    """Flatten pairs without changing deterministic order."""

    return [pair.as_row() for pair in pairs]


class StablePathFeatureStore:
    """Long-only feature provider addressed exclusively by stable/path ID."""

    def __init__(
        self,
        inputs: StableLongInput,
        policy: StableLongPolicy | None = None,
    ) -> None:
        self._policy = StableLongPolicy() if policy is None else policy
        self._prepared = _prepare(inputs, self._policy)
        self._gallery_cache: dict[tuple[int, int, int], CleanGallery | None] = {}

    @property
    def path_ids(self) -> tuple[int, ...]:
        return self._prepared.stable_ids

    def retrieval_descriptor(
        self, path_id: int
    ) -> StablePathRetrievalDescriptor:
        """Return immutable whole-path material for exact candidate retrieval."""

        if isinstance(path_id, (bool, np.bool_)) or not isinstance(
            path_id, (int, np.integer)
        ):
            raise ContractError("S05 path_id must be an integer stable/path ID")
        stable_id = int(path_id)
        positions = self._prepared.stable_paths.get(stable_id)
        if positions is None:
            raise ContractError("S05 retrieval references an unknown path ID")
        segment = _Segment(stable_id, positions)
        gallery = _gallery(
            self._prepared,
            segment,
            self._policy,
            self._gallery_cache,
        )
        provenance = (
            GalleryProvenance.from_gallery(gallery)
            if gallery is not None
            else None
        )
        first, last = int(positions[0]), int(positions[-1])
        source_context = _endpoint_context(self._prepared, last)
        target_context = _endpoint_context(self._prepared, first)
        gallery_overlap = bool(
            gallery is not None
            and gallery.max_other_bbox_iou
            >= self._policy.high_overlap_iou_threshold
        )
        prototypes: np.ndarray | None = None
        medoid: np.ndarray | None = None
        if gallery is not None:
            prototypes = np.array(gallery.prototypes, dtype=np.float32, copy=True)
            medoid = np.array(
                gallery.medoid_embedding, dtype=np.float32, copy=True
            )
            prototypes.setflags(write=False)
            medoid.setflags(write=False)
        return StablePathRetrievalDescriptor(
            stable_id=stable_id,
            segment=_provenance(self._prepared, segment),
            gallery=provenance,
            prototypes=prototypes,
            medoid_embedding=medoid,
            source_endpoint_review_excluded=bool(
                self._prepared.data.det_review_excluded[last]
            ),
            target_endpoint_review_excluded=bool(
                self._prepared.data.det_review_excluded[first]
            ),
            source_high_overlap=gallery_overlap
            or bool(
                source_context.max_other_iou
                >= self._policy.high_overlap_iou_threshold
            ),
            target_high_overlap=gallery_overlap
            or bool(
                target_context.max_other_iou
                >= self._policy.high_overlap_iou_threshold
            ),
            missing_reason=(None if gallery is not None else "gallery_missing"),
        )

    def retrieval_descriptors(
        self,
        *,
        logger: LogFn | None = None,
        progress_interval_sec: float = 10.0,
    ) -> tuple[StablePathRetrievalDescriptor, ...]:
        """Build every descriptor deterministically with bounded progress logs."""

        if (
            isinstance(progress_interval_sec, bool)
            or not math.isfinite(float(progress_interval_sec))
            or float(progress_interval_sec) <= 0.0
        ):
            raise ContractError("S05 descriptor progress interval must be positive")
        result: list[StablePathRetrievalDescriptor] = []
        last_report = time.monotonic()
        total = len(self.path_ids)
        for completed, stable_id in enumerate(self.path_ids, start=1):
            result.append(self.retrieval_descriptor(stable_id))
            now = time.monotonic()
            if logger is not None and now - last_report >= progress_interval_sec:
                logger(f"[s05-propose] clean galleries {completed:,}/{total:,}")
                last_report = now
        return tuple(result)

    def build_pair(
        self, source_path_id: int, target_path_id: int
    ) -> StablePathPairFeatures:
        """Build LONG features; absent appearance returns a rejected result."""

        for value, name in (
            (source_path_id, "source_path_id"),
            (target_path_id, "target_path_id"),
        ):
            if isinstance(value, (bool, np.bool_)) or not isinstance(
                value, (int, np.integer)
            ):
                raise ContractError(f"S05A {name} must be an integer stable/path ID")
        source_id, target_id = int(source_path_id), int(target_path_id)
        if source_id == target_id:
            raise ContractError("S05A source and target stable/path IDs must differ")
        if (
            source_id not in self._prepared.stable_paths
            or target_id not in self._prepared.stable_paths
        ):
            raise ContractError("S05A feature request references an unknown path ID")
        source = _Segment(source_id, self._prepared.stable_paths[source_id])
        target = _Segment(target_id, self._prepared.stable_paths[target_id])
        source_end, target_start = (
            int(source.det_positions[-1]),
            int(target.det_positions[0]),
        )
        gap = float(
            self._prepared.data.det_global_time_sec[target_start]
            - self._prepared.data.det_global_time_sec[source_end]
        )
        if gap <= self._policy.min_gap_sec_exclusive:
            raise ContractError(
                "S05A production paths overlap, reverse, or have gap <=5s"
            )
        source_gallery = _gallery(
            self._prepared,
            source,
            self._policy,
            self._gallery_cache,
        )
        target_gallery = _gallery(
            self._prepared,
            target,
            self._policy,
            self._gallery_cache,
        )
        source_provenance = (
            GalleryProvenance.from_gallery(source_gallery)
            if source_gallery is not None
            else None
        )
        target_provenance = (
            GalleryProvenance.from_gallery(target_gallery)
            if target_gallery is not None
            else None
        )
        source_context = _endpoint_context(self._prepared, source_end)
        target_context = _endpoint_context(self._prepared, target_start)
        high_overlap = has_high_endpoint_overlap(
            source_context,
            target_context,
            threshold=self._policy.high_overlap_iou_threshold,
        )
        if source_gallery is not None:
            high_overlap = high_overlap or bool(
                source_gallery.max_other_bbox_iou
                >= self._policy.high_overlap_iou_threshold
            )
        if target_gallery is not None:
            high_overlap = high_overlap or bool(
                target_gallery.max_other_bbox_iou
                >= self._policy.high_overlap_iou_threshold
            )
        if bool(self._prepared.data.det_review_excluded[source_end]) or bool(
            self._prepared.data.det_review_excluded[target_start]
        ):
            return StablePathPairFeatures(
                None,
                False,
                high_overlap,
                "review_excluded_endpoint",
                source_provenance,
                target_provenance,
            )
        if source_gallery is None or target_gallery is None:
            side = (
                "source_target"
                if source_gallery is None and target_gallery is None
                else "source"
                if source_gallery is None
                else "target"
            )
            return StablePathPairFeatures(
                None,
                False,
                high_overlap,
                f"{side}_gallery_missing",
                source_provenance,
                target_provenance,
            )
        features, feature_high_overlap = _build_features(
            self._prepared,
            source,
            target,
            source_gallery,
            target_gallery,
            self._policy,
        )
        if high_overlap != feature_high_overlap:
            raise ContractError("S05A feature-store overlap assessments disagree")
        return StablePathPairFeatures(
            MappingProxyType(features),
            True,
            high_overlap,
            None,
            source_provenance,
            target_provenance,
        )


build_stable_long_pairs = generate_stable_long_pairs


__all__ = [
    "StableLongInput",
    "StableLongPair",
    "StableLongPolicy",
    "StablePartition",
    "StablePathFeatureStore",
    "StablePathPairFeatures",
    "StablePathRetrievalDescriptor",
    "StableSegmentProvenance",
    "assign_stable_partitions",
    "build_stable_long_pairs",
    "generate_stable_long_pairs",
    "stable_long_pairs_as_rows",
]
