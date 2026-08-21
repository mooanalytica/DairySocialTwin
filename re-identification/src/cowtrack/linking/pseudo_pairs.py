"""Deterministic, leakage-safe pseudo-pair construction for S03.

The public input is deliberately columnar: the real calibration input contains
hundreds of thousands of detections, so making one Python object per detection
would be needlessly expensive.  The only Python objects produced per example
are the relatively small number of :class:`PseudoPair` records.

No legacy tracking identity, review reason/case identity, keypoint, or raw
absolute-position field is accepted by this module.  Review information is a
single per-detection exclusion boolean and is used only as a reliability mask.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Callable, Literal, Mapping

import numpy as np

from cowtrack.config import ContractError
from cowtrack.linking.config import LinkCalibrationConfig
from cowtrack.linking.features import (
    CleanGallery,
    EndpointContext,
    EndpointGeometry,
    LinkMode,
    MotionHistory,
    build_clean_gallery,
    feature_schema,
    has_high_endpoint_overlap,
    pair_features,
    validate_disjoint_sides,
)


SplitName = Literal["train", "calibration", "audit"]
CalibrationRole = Literal["not_applicable", "threshold_selection", "certification"]
ParentPartition = Literal[
    "train", "threshold_selection", "certification", "audit"
]
LogFn = Callable[[str], None]

_SPLITS: tuple[SplitName, ...] = ("train", "calibration", "audit")
_SPLIT_RANK = {name: rank for rank, name in enumerate(_SPLITS)}
_PARTITIONS: tuple[ParentPartition, ...] = (
    "train",
    "threshold_selection",
    "certification",
    "audit",
)
_PARTITION_RANK = {name: rank for rank, name in enumerate(_PARTITIONS)}


@dataclass(frozen=True)
class CalibrationInput:
    """Columnar S00/S01/S02 inputs required to construct pseudo-pairs.

    ``det_*`` columns contain valid S00 detections joined to S01
    ``det_to_micro``.  ``det_other_bbox_max_iou`` and
    ``det_boundary_distance`` are anonymous geometry/context values computed
    for every detection; the stage must not substitute legacy IDs for either.
    ``det_review_excluded`` is the already-resolved S02 review exclusion mask,
    with no case ID or reason attached.

    ``sample_*`` columns correspond to S02 appearance samples.  Embeddings are
    addressed only by the stable ``sample_embedding_rows`` values.
    """

    timeline_clip_ids: np.ndarray
    timeline_clip_start_time_sec: np.ndarray
    timeline_clip_end_time_sec: np.ndarray

    parent_micro_ids: np.ndarray
    parent_status: np.ndarray
    parent_num_detections: np.ndarray
    parent_local_purity_score: np.ndarray
    parent_bidirectional_agreement: np.ndarray
    parent_internal_cosine_p10: np.ndarray

    det_ids: np.ndarray
    det_micro_ids: np.ndarray
    det_order_in_micro: np.ndarray
    det_clip_ids: np.ndarray
    det_global_frames: np.ndarray
    det_global_time_sec: np.ndarray
    det_cx_norm: np.ndarray
    det_cy_norm: np.ndarray
    det_w_norm: np.ndarray
    det_h_norm: np.ndarray
    det_other_bbox_max_iou: np.ndarray
    det_boundary_distance: np.ndarray
    det_review_excluded: np.ndarray

    sample_ids: np.ndarray
    sample_micro_ids: np.ndarray
    sample_det_ids: np.ndarray
    sample_crop_quality: np.ndarray
    sample_other_bbox_max_iou: np.ndarray
    sample_s02_inlier: np.ndarray
    sample_embedding_rows: np.ndarray
    embeddings: np.ndarray


@dataclass(frozen=True)
class GalleryProvenance:
    """Stable identifiers and quality statistics for one rebuilt gallery."""

    sample_ids: tuple[int, ...]
    det_ids: tuple[int, ...]
    embedding_rows: tuple[int, ...]
    medoid_sample_id: int
    appearance_quality: float
    internal_cosine_p10: float
    internal_cosine_p50: float
    internal_cosine_min: float
    num_input_samples: int
    num_overlap_rejected: int
    num_review_excluded: int
    num_local_outliers: int
    max_other_bbox_iou: float
    max_clean_other_bbox_iou: float

    @classmethod
    def from_gallery(cls, gallery: CleanGallery) -> "GalleryProvenance":
        return cls(
            sample_ids=tuple(int(value) for value in gallery.sample_ids),
            det_ids=tuple(int(value) for value in gallery.det_ids),
            embedding_rows=tuple(int(value) for value in gallery.embedding_rows),
            medoid_sample_id=int(gallery.medoid_sample_id),
            appearance_quality=float(gallery.appearance_quality),
            internal_cosine_p10=float(gallery.internal_cosine_p10),
            internal_cosine_p50=float(gallery.internal_cosine_p50),
            internal_cosine_min=float(gallery.internal_cosine_min),
            num_input_samples=int(gallery.num_input_samples),
            num_overlap_rejected=int(gallery.num_overlap_rejected),
            num_review_excluded=int(gallery.num_review_excluded),
            num_local_outliers=int(gallery.num_local_outliers),
            max_other_bbox_iou=float(gallery.max_other_bbox_iou),
            max_clean_other_bbox_iou=float(gallery.max_clean_other_bbox_iou),
        )

    def prefixed_row(self, prefix: str) -> dict[str, Any]:
        return {
            f"{prefix}_sample_ids": list(self.sample_ids),
            f"{prefix}_gallery_det_ids": list(self.det_ids),
            f"{prefix}_embedding_rows": list(self.embedding_rows),
            f"{prefix}_medoid_sample_id": self.medoid_sample_id,
            f"{prefix}_appearance_quality": self.appearance_quality,
            f"{prefix}_internal_cosine_p10": self.internal_cosine_p10,
            f"{prefix}_internal_cosine_p50": self.internal_cosine_p50,
            f"{prefix}_internal_cosine_min": self.internal_cosine_min,
            f"{prefix}_gallery_num_input_samples": self.num_input_samples,
            f"{prefix}_gallery_num_overlap_rejected": self.num_overlap_rejected,
            f"{prefix}_gallery_num_review_excluded": self.num_review_excluded,
            f"{prefix}_gallery_num_local_outliers": self.num_local_outliers,
            f"{prefix}_gallery_max_other_bbox_iou": self.max_other_bbox_iou,
            f"{prefix}_gallery_max_clean_other_bbox_iou": (
                self.max_clean_other_bbox_iou
            ),
        }


@dataclass(frozen=True)
class SegmentProvenance:
    """Inclusive temporal extent of a source or target pseudo segment."""

    micro_id: int
    start_det_id: int
    end_det_id: int
    start_global_frame: int
    end_global_frame: int
    start_time_sec: float
    end_time_sec: float
    start_clip_id: str
    end_clip_id: str
    num_detections: int


@dataclass(frozen=True)
class PseudoPair:
    """One model-ready positive or simultaneous hard-negative example."""

    pair_id: str
    candidate_group_id: str
    parent_group_id: str
    split: SplitName
    calibration_role: CalibrationRole
    mode: LinkMode
    label: bool
    pair_kind: str
    stratum: str
    parent_micro_id: int
    source: SegmentProvenance
    target: SegmentProvenance
    source_gallery: GalleryProvenance
    target_gallery: GalleryProvenance
    gap_target_sec: float
    gap_error_sec: float
    hard_negative_rank: int
    appearance_present: bool
    high_overlap: bool
    features: Mapping[str, float]

    def as_row(self) -> dict[str, Any]:
        """Return a deterministic flat row ready for stage serialization.

        The canonical schema columns are present directly.  Additional gallery
        detection IDs and side-local quality statistics retain enough
        provenance to audit every no-sharing and clean-gallery decision; a
        stage writing the compact Arrow schema may explicitly select its schema
        fields after preserving these diagnostics elsewhere.
        """

        expected = feature_schema(self.mode)
        if tuple(self.features) != expected:
            raise ContractError("S03 pseudo-pair feature order does not match schema")
        row: dict[str, Any] = {
            "pair_id": self.pair_id,
            "candidate_group_id": self.candidate_group_id,
            "parent_group_id": self.parent_group_id,
            "split": self.split,
            "calibration_role": self.calibration_role,
            "mode": self.mode,
            "label": self.label,
            "pair_kind": self.pair_kind,
            "stratum": self.stratum,
            "parent_micro_id": self.parent_micro_id,
            "source_micro_id": self.source.micro_id,
            "target_micro_id": self.target.micro_id,
            "source_start_det_id": self.source.start_det_id,
            "source_end_det_id": self.source.end_det_id,
            "target_start_det_id": self.target.start_det_id,
            "target_end_det_id": self.target.end_det_id,
            "source_start_global_frame": self.source.start_global_frame,
            "source_end_global_frame": self.source.end_global_frame,
            "target_start_global_frame": self.target.start_global_frame,
            "target_end_global_frame": self.target.end_global_frame,
            "source_start_time_sec": self.source.start_time_sec,
            "source_end_time_sec": self.source.end_time_sec,
            "target_start_time_sec": self.target.start_time_sec,
            "target_end_time_sec": self.target.end_time_sec,
            "source_clip_id": self.source.end_clip_id,
            "target_clip_id": self.target.start_clip_id,
            "source_segment_start_clip_id": self.source.start_clip_id,
            "target_segment_end_clip_id": self.target.end_clip_id,
            "source_num_detections": self.source.num_detections,
            "target_num_detections": self.target.num_detections,
            "gap_target_sec": self.gap_target_sec,
            "gap_error_sec": self.gap_error_sec,
            "hard_negative_rank": self.hard_negative_rank,
            "appearance_present": self.appearance_present,
            "high_overlap": self.high_overlap,
        }
        row.update(self.source_gallery.prefixed_row("source"))
        row.update(self.target_gallery.prefixed_row("target"))
        row.update((name, float(self.features[name])) for name in expected)
        return row

    # Both spellings are convenient at stage call-sites.
    to_dict = as_row


@dataclass(frozen=True)
class _PreparedInput:
    timeline_clip_ids: np.ndarray
    timeline_clip_start_time_sec: np.ndarray
    timeline_clip_end_time_sec: np.ndarray
    timeline_bounds: Mapping[str, tuple[float, float]]

    parent_micro_ids: np.ndarray
    parent_status: np.ndarray
    parent_num_detections: np.ndarray
    parent_local_purity_score: np.ndarray
    parent_bidirectional_agreement: np.ndarray
    parent_internal_cosine_p10: np.ndarray
    parent_row: Mapping[int, int]

    det_ids: np.ndarray
    det_micro_ids: np.ndarray
    det_order_in_micro: np.ndarray
    det_clip_ids: np.ndarray
    det_global_frames: np.ndarray
    det_global_time_sec: np.ndarray
    det_cx_norm: np.ndarray
    det_cy_norm: np.ndarray
    det_w_norm: np.ndarray
    det_h_norm: np.ndarray
    det_other_bbox_max_iou: np.ndarray
    det_boundary_distance: np.ndarray
    det_review_excluded: np.ndarray
    det_row: Mapping[int, int]
    paths: Mapping[int, np.ndarray]
    frame_rows: Mapping[tuple[str, int], tuple[int, ...]]

    sample_ids: np.ndarray
    sample_micro_ids: np.ndarray
    sample_det_ids: np.ndarray
    sample_crop_quality: np.ndarray
    sample_other_bbox_max_iou: np.ndarray
    sample_s02_inlier: np.ndarray
    sample_embedding_rows: np.ndarray
    sample_det_positions: np.ndarray
    sample_rows: Mapping[int, np.ndarray]
    embeddings: np.ndarray


@dataclass(frozen=True)
class _Segment:
    micro_id: int
    det_positions: np.ndarray


@dataclass(frozen=True)
class _PositiveCandidate:
    source: _Segment
    target: _Segment
    source_gallery: CleanGallery
    target_gallery: CleanGallery
    mode: LinkMode
    gap_target_sec: float


@dataclass(frozen=True)
class TrackletPairFeatures:
    """Production feature result using the exact clean-gallery calibration policy."""

    feature_values: Mapping[str, float] | None
    appearance_present: bool
    high_overlap: bool
    reason: str | None
    source_gallery: "GalleryAssessment | None" = None
    target_gallery: "GalleryAssessment | None" = None


@dataclass(frozen=True)
class GalleryAssessment:
    """Outcome and audit provenance for one attempted production gallery.

    ``provenance`` is populated only when the exact calibration-time clean
    gallery policy succeeds.  A missing gallery is deliberately represented
    as an assessment, rather than by a synthetic/empty prototype.
    """

    present: bool
    reason: str | None
    provenance: GalleryProvenance | None

    @classmethod
    def from_gallery(cls, gallery: CleanGallery | None) -> "GalleryAssessment":
        if gallery is None:
            return cls(False, "clean_gallery_missing", None)
        return cls(True, None, GalleryProvenance.from_gallery(gallery))


def _array(
    values: object,
    *,
    name: str,
    length: int | None = None,
    kinds: str,
) -> np.ndarray:
    result = np.asarray(values)
    if result.ndim != 1 or result.dtype.kind not in kinds:
        raise ContractError(f"S03 {name} must be a one-dimensional {kinds!r} array")
    if length is not None and len(result) != length:
        raise ContractError(f"S03 {name} has an inconsistent length")
    return result


def _integers(values: object, *, name: str, length: int | None = None) -> np.ndarray:
    return _array(values, name=name, length=length, kinds="iu").astype(
        np.int64, copy=False
    )


def _floats(values: object, *, name: str, length: int) -> np.ndarray:
    result = _array(values, name=name, length=length, kinds="fiu").astype(
        np.float64, copy=False
    )
    if not np.all(np.isfinite(result)):
        raise ContractError(f"S03 {name} must be finite")
    return result


def _booleans(values: object, *, name: str, length: int) -> np.ndarray:
    return _array(values, name=name, length=length, kinds="b")


def _strings(values: object, *, name: str, length: int) -> np.ndarray:
    result = np.asarray(values)
    if result.ndim != 1 or len(result) != length:
        raise ContractError(f"S03 {name} must be a one-dimensional string array")
    converted = result.astype(object, copy=False)
    if any(not isinstance(value, str) or not value for value in converted):
        raise ContractError(f"S03 {name} must contain non-empty strings")
    return converted


def _unique(values: np.ndarray, *, name: str) -> None:
    if len(np.unique(values)) != len(values):
        raise ContractError(f"S03 {name} must be unique")


def _prepare(data: CalibrationInput) -> _PreparedInput:
    if not isinstance(data, CalibrationInput):
        raise ContractError("S03 pseudo-pair input must be CalibrationInput")

    timeline_clips_raw = np.asarray(data.timeline_clip_ids)
    if timeline_clips_raw.ndim != 1:
        raise ContractError("S03 timeline_clip_ids must be one-dimensional")
    timeline_count = len(timeline_clips_raw)
    if timeline_count == 0:
        raise ContractError("S03 calibration input has no clip timeline")
    timeline_clips = _strings(
        timeline_clips_raw, name="timeline_clip_ids", length=timeline_count
    )
    _unique(timeline_clips, name="timeline_clip_ids")
    timeline_starts = _floats(
        data.timeline_clip_start_time_sec,
        name="timeline_clip_start_time_sec",
        length=timeline_count,
    )
    timeline_ends = _floats(
        data.timeline_clip_end_time_sec,
        name="timeline_clip_end_time_sec",
        length=timeline_count,
    )
    if np.any(timeline_ends <= timeline_starts):
        raise ContractError("S03 clip timelines must have positive duration")
    timeline_bounds = {
        str(clip_id): (float(start), float(end))
        for clip_id, start, end in zip(
            timeline_clips, timeline_starts, timeline_ends, strict=True
        )
    }

    parent_ids = _integers(data.parent_micro_ids, name="parent_micro_ids")
    parent_count = len(parent_ids)
    if parent_count == 0:
        raise ContractError("S03 calibration input has no parent micros")
    _unique(parent_ids, name="parent_micro_ids")
    parent_status = _strings(
        data.parent_status, name="parent_status", length=parent_count
    )
    parent_num = _integers(
        data.parent_num_detections,
        name="parent_num_detections",
        length=parent_count,
    )
    if np.any(parent_num <= 0):
        raise ContractError("S03 parent_num_detections must be positive")
    parent_purity = _floats(
        data.parent_local_purity_score,
        name="parent_local_purity_score",
        length=parent_count,
    )
    parent_bidir = _floats(
        data.parent_bidirectional_agreement,
        name="parent_bidirectional_agreement",
        length=parent_count,
    )
    parent_internal = _floats(
        data.parent_internal_cosine_p10,
        name="parent_internal_cosine_p10",
        length=parent_count,
    )
    for values, name in (
        (parent_purity, "parent_local_purity_score"),
        (parent_bidir, "parent_bidirectional_agreement"),
    ):
        if np.any((values < 0.0) | (values > 1.0)):
            raise ContractError(f"S03 {name} must be in [0, 1]")
    if np.any((parent_internal < -1.0) | (parent_internal > 1.0)):
        raise ContractError("S03 parent_internal_cosine_p10 must be in [-1, 1]")
    parent_row = {int(value): row for row, value in enumerate(parent_ids)}

    det_ids = _integers(data.det_ids, name="det_ids")
    det_count = len(det_ids)
    if det_count == 0:
        raise ContractError("S03 calibration input has no detections")
    _unique(det_ids, name="det_ids")
    det_micro = _integers(
        data.det_micro_ids, name="det_micro_ids", length=det_count
    )
    det_order = _integers(
        data.det_order_in_micro, name="det_order_in_micro", length=det_count
    )
    det_clips = _strings(data.det_clip_ids, name="det_clip_ids", length=det_count)
    det_frames = _integers(
        data.det_global_frames, name="det_global_frames", length=det_count
    )
    det_times = _floats(
        data.det_global_time_sec, name="det_global_time_sec", length=det_count
    )
    det_cx = _floats(data.det_cx_norm, name="det_cx_norm", length=det_count)
    det_cy = _floats(data.det_cy_norm, name="det_cy_norm", length=det_count)
    det_w = _floats(data.det_w_norm, name="det_w_norm", length=det_count)
    det_h = _floats(data.det_h_norm, name="det_h_norm", length=det_count)
    det_iou = _floats(
        data.det_other_bbox_max_iou,
        name="det_other_bbox_max_iou",
        length=det_count,
    )
    det_boundary = _floats(
        data.det_boundary_distance,
        name="det_boundary_distance",
        length=det_count,
    )
    det_review = _booleans(
        data.det_review_excluded,
        name="det_review_excluded",
        length=det_count,
    )
    if set(map(str, np.unique(det_clips))) != set(timeline_bounds):
        raise ContractError("S03 detection clips differ from the authoritative timeline")
    for clip_id, (start, end) in timeline_bounds.items():
        clip_times = det_times[det_clips == clip_id]
        if not len(clip_times) or np.any((clip_times < start) | (clip_times > end)):
            raise ContractError(f"S03 detections fall outside clip timeline {clip_id}")
    if np.any((det_cx < 0.0) | (det_cx > 1.0)) or np.any(
        (det_cy < 0.0) | (det_cy > 1.0)
    ):
        raise ContractError("S03 normalized detection centers must be in [0, 1]")
    if np.any((det_w <= 0.0) | (det_w > 1.0)) or np.any(
        (det_h <= 0.0) | (det_h > 1.0)
    ):
        raise ContractError("S03 normalized detection sizes must be in (0, 1]")
    for values, name in (
        (det_iou, "det_other_bbox_max_iou"),
        (det_boundary, "det_boundary_distance"),
    ):
        if np.any((values < 0.0) | (values > 1.0)):
            raise ContractError(f"S03 {name} must be in [0, 1]")
    unknown = sorted(set(int(value) for value in det_micro) - set(parent_row))
    if unknown:
        raise ContractError(f"S03 detections contain unknown parent micros: {unknown[:5]}")
    det_row = {int(value): row for row, value in enumerate(det_ids)}

    paths: dict[int, np.ndarray] = {}
    # Group the 745k-row production table once.  Repeated ``flatnonzero`` per
    # parent would be O(num_parents * num_detections).
    grouped_detection_order = np.lexsort((det_ids, det_order, det_micro))
    grouped_detection_micro = det_micro[grouped_detection_order]
    grouped_ids, grouped_starts, grouped_counts = np.unique(
        grouped_detection_micro, return_index=True, return_counts=True
    )
    grouped_positions = {
        int(micro_id): grouped_detection_order[start : start + count]
        for micro_id, start, count in zip(
            grouped_ids, grouped_starts, grouped_counts, strict=True
        )
    }
    if set(grouped_positions) != set(parent_row):
        missing = sorted(set(parent_row) - set(grouped_positions))
        raise ContractError(f"S03 parent micros have no detections: {missing[:5]}")
    for micro_id in sorted(parent_row):
        positions = grouped_positions[micro_id]
        if len(positions) != int(parent_num[parent_row[micro_id]]):
            raise ContractError(
                f"S03 parent {micro_id} detection count disagrees with S01"
            )
        if not np.array_equal(det_order[positions], np.arange(len(positions))):
            raise ContractError(
                f"S03 parent {micro_id} order_in_micro must be contiguous from zero"
            )
        if len(positions) > 1 and np.any(np.diff(det_times[positions]) <= 0.0):
            raise ContractError(
                f"S03 parent {micro_id} detection times must be strictly increasing"
            )
        keys = [(str(det_clips[pos]), int(det_frames[pos])) for pos in positions]
        if len(set(keys)) != len(keys):
            raise ContractError(
                f"S03 parent {micro_id} has duplicate detections in a frame"
            )
        paths[micro_id] = positions

    frame_lists: dict[tuple[str, int], list[int]] = {}
    for position in range(det_count):
        key = (str(det_clips[position]), int(det_frames[position]))
        frame_lists.setdefault(key, []).append(position)
    frame_rows: dict[tuple[str, int], tuple[int, ...]] = {}
    for key, positions in frame_lists.items():
        positions.sort(key=lambda pos: (int(det_micro[pos]), int(det_ids[pos])))
        frame_rows[key] = tuple(positions)

    sample_ids = _integers(data.sample_ids, name="sample_ids")
    sample_count = len(sample_ids)
    _unique(sample_ids, name="sample_ids")
    sample_micro = _integers(
        data.sample_micro_ids, name="sample_micro_ids", length=sample_count
    )
    sample_det = _integers(
        data.sample_det_ids, name="sample_det_ids", length=sample_count
    )
    _unique(sample_det, name="sample_det_ids")
    sample_quality = _floats(
        data.sample_crop_quality, name="sample_crop_quality", length=sample_count
    )
    sample_iou = _floats(
        data.sample_other_bbox_max_iou,
        name="sample_other_bbox_max_iou",
        length=sample_count,
    )
    sample_inlier = _booleans(
        data.sample_s02_inlier, name="sample_s02_inlier", length=sample_count
    )
    embedding_rows = _integers(
        data.sample_embedding_rows,
        name="sample_embedding_rows",
        length=sample_count,
    )
    _unique(embedding_rows, name="sample_embedding_rows")
    if np.any((sample_quality < 0.0) | (sample_quality > 1.0)):
        raise ContractError("S03 sample_crop_quality must be in [0, 1]")
    if np.any((sample_iou < 0.0) | (sample_iou > 1.0)):
        raise ContractError("S03 sample_other_bbox_max_iou must be in [0, 1]")

    embeddings = np.asarray(data.embeddings)
    if embeddings.ndim != 2 or embeddings.shape[1] <= 0 or embeddings.dtype.kind != "f":
        raise ContractError("S03 embeddings must be a floating [N, D] array")
    if not np.all(np.isfinite(embeddings)):
        raise ContractError("S03 embeddings must be finite")
    norms = np.linalg.norm(embeddings.astype(np.float32, copy=False), axis=1)
    if len(embeddings) and not np.allclose(norms, 1.0, rtol=0.0, atol=2e-3):
        raise ContractError("S03 embeddings must be L2-normalized and nonzero")
    if np.any(embedding_rows < 0) or np.any(embedding_rows >= len(embeddings)):
        raise ContractError("S03 sample_embedding_rows are outside embeddings")

    sample_det_positions = np.empty(sample_count, dtype=np.int64)
    for row in range(sample_count):
        det_id = int(sample_det[row])
        position = det_row.get(det_id)
        if position is None:
            raise ContractError(f"S03 appearance sample references unknown det_id {det_id}")
        if int(sample_micro[row]) != int(det_micro[position]):
            raise ContractError("S03 appearance sample micro/detection join is inconsistent")
        sample_det_positions[row] = position
    # The same one-pass grouping keeps S02 sample preparation linearithmic.
    sample_rows: dict[int, np.ndarray] = {
        micro_id: np.empty(0, dtype=np.int64) for micro_id in parent_row
    }
    if sample_count:
        grouped_sample_order = np.lexsort(
            (sample_ids, sample_det, embedding_rows, sample_micro)
        )
        grouped_sample_micro = sample_micro[grouped_sample_order]
        grouped_ids, grouped_starts, grouped_counts = np.unique(
            grouped_sample_micro, return_index=True, return_counts=True
        )
        unknown_samples = sorted(
            set(int(value) for value in grouped_ids) - set(parent_row)
        )
        if unknown_samples:
            raise ContractError(
                f"S03 samples contain unknown parent micros: {unknown_samples[:5]}"
            )
        for micro_id, start, count in zip(
            grouped_ids, grouped_starts, grouped_counts, strict=True
        ):
            sample_rows[int(micro_id)] = grouped_sample_order[start : start + count]

    return _PreparedInput(
        timeline_clip_ids=timeline_clips,
        timeline_clip_start_time_sec=timeline_starts,
        timeline_clip_end_time_sec=timeline_ends,
        timeline_bounds=timeline_bounds,
        parent_micro_ids=parent_ids,
        parent_status=parent_status,
        parent_num_detections=parent_num,
        parent_local_purity_score=parent_purity,
        parent_bidirectional_agreement=parent_bidir,
        parent_internal_cosine_p10=parent_internal,
        parent_row=parent_row,
        det_ids=det_ids,
        det_micro_ids=det_micro,
        det_order_in_micro=det_order,
        det_clip_ids=det_clips,
        det_global_frames=det_frames,
        det_global_time_sec=det_times,
        det_cx_norm=det_cx,
        det_cy_norm=det_cy,
        det_w_norm=det_w,
        det_h_norm=det_h,
        det_other_bbox_max_iou=det_iou,
        det_boundary_distance=det_boundary,
        det_review_excluded=det_review,
        det_row=det_row,
        paths=paths,
        frame_rows=frame_rows,
        sample_ids=sample_ids,
        sample_micro_ids=sample_micro,
        sample_det_ids=sample_det,
        sample_crop_quality=sample_quality,
        sample_other_bbox_max_iou=sample_iou,
        sample_s02_inlier=sample_inlier,
        sample_embedding_rows=embedding_rows,
        sample_det_positions=sample_det_positions,
        sample_rows=sample_rows,
        embeddings=embeddings,
    )


def _partition_for_time(
    time_sec: float,
    clip_start_sec: float,
    clip_end_sec: float,
    config: LinkCalibrationConfig,
) -> ParentPartition:
    if (
        not math.isfinite(time_sec)
        or not math.isfinite(clip_start_sec)
        or not math.isfinite(clip_end_sec)
        or clip_end_sec <= clip_start_sec
        or time_sec < clip_start_sec
        or time_sec > clip_end_sec
    ):
        raise ContractError("S03 continuous time-block split bounds are invalid")
    duration = clip_end_sec - clip_start_sec
    train_end = clip_start_sec + config.train_fraction * duration
    selection_end = clip_start_sec + (
        config.train_fraction
        + config.calibration_fraction * config.calibration_selection_fraction
    ) * duration
    certification_end = clip_start_sec + (
        config.train_fraction + config.calibration_fraction
    ) * duration
    if time_sec < train_end:
        return "train"
    if time_sec < selection_end:
        return "threshold_selection"
    if time_sec < certification_end:
        return "certification"
    return "audit"


def _partition_split(partition: ParentPartition) -> SplitName:
    if partition == "train":
        return "train"
    if partition in {"threshold_selection", "certification"}:
        return "calibration"
    if partition == "audit":
        return "audit"
    raise ContractError(f"S03 unknown parent partition: {partition}")


def _partition_calibration_role(partition: ParentPartition) -> CalibrationRole:
    if partition in {"train", "audit"}:
        return "not_applicable"
    if partition == "threshold_selection":
        return "threshold_selection"
    if partition == "certification":
        return "certification"
    raise ContractError(f"S03 unknown parent partition: {partition}")


def _assign_parent_partitions_prepared(
    data: _PreparedInput, config: LinkCalibrationConfig
) -> dict[int, ParentPartition]:
    per_clip_parent_positions: dict[str, dict[int, list[int]]] = {}
    for position in range(len(data.det_ids)):
        clip_id = str(data.det_clip_ids[position])
        micro_id = int(data.det_micro_ids[position])
        per_clip_parent_positions.setdefault(clip_id, {}).setdefault(
            micro_id, []
        ).append(position)

    per_clip: dict[str, dict[int, ParentPartition]] = {}
    for micro_id, positions in data.paths.items():
        for clip_id in sorted({str(data.det_clip_ids[pos]) for pos in positions}):
            parent_positions = np.asarray(
                per_clip_parent_positions[clip_id][micro_id], dtype=np.int64
            )
            start, end = data.timeline_bounds[clip_id]
            # A parent is an indivisible leakage group.  If it touches more than
            # one contiguous time block, the more held-out block wins.
            parent_start_partition = _partition_for_time(
                float(np.min(data.det_global_time_sec[parent_positions])),
                start,
                end,
                config,
            )
            parent_end_partition = _partition_for_time(
                float(np.max(data.det_global_time_sec[parent_positions])),
                start,
                end,
                config,
            )
            per_clip.setdefault(clip_id, {})[micro_id] = max(
                (parent_start_partition, parent_end_partition),
                key=_PARTITION_RANK.__getitem__,
            )

    proposals: dict[int, list[ParentPartition]] = {
        micro_id: [] for micro_id in data.paths
    }
    for clip_id in sorted(per_clip):
        for micro_id in sorted(per_clip[clip_id]):
            proposals[micro_id].append(per_clip[clip_id][micro_id])

    assignments: dict[int, ParentPartition] = {}
    for micro_id in sorted(proposals):
        choices = proposals[micro_id]
        if not choices:
            raise ContractError(f"S03 parent {micro_id} is absent from all clips")
        # A cross-clip micro is a single leakage group.  If its per-clip time
        # blocks disagree, the more held-out assignment wins; this is the
        # conservative deterministic resolution and can never leak audit into
        # fitting.
        assignments[micro_id] = max(choices, key=_PARTITION_RANK.__getitem__)

    if config.split_require_all_clips_in_audit:
        for clip_id, groups in per_clip.items():
            if not any(assignments[micro_id] == "audit" for micro_id in groups):
                raise ContractError(f"S03 split lacks audit coverage for clip {clip_id}")
    return assignments


def _assign_parent_splits_prepared(
    data: _PreparedInput, config: LinkCalibrationConfig
) -> dict[int, SplitName]:
    return {
        micro_id: _partition_split(partition)
        for micro_id, partition in _assign_parent_partitions_prepared(
            data, config
        ).items()
    }


def assign_parent_splits(
    data: CalibrationInput, config: LinkCalibrationConfig
) -> dict[int, SplitName]:
    """Assign each parent micro to one time-block split.

    Assignment uses contiguous 60/20/20 time ranges independently within each
    clip; calibration is pre-partitioned 50/50 for threshold selection and
    independent certification. A micro touching a later range, or present in
    multiple clips, is kept intact using conservative held-out precedence.
    """

    if not isinstance(config, LinkCalibrationConfig):
        raise ContractError("S03 split config must be LinkCalibrationConfig")
    return _assign_parent_splits_prepared(_prepare(data), config)


def _eligible_parents(
    data: _PreparedInput, config: LinkCalibrationConfig
) -> dict[int, bool]:
    result: dict[int, bool] = {}
    for micro_id, row in data.parent_row.items():
        result[micro_id] = bool(
            str(data.parent_status[row]) == config.required_parent_status
            and int(data.parent_num_detections[row]) >= config.min_parent_detections
            and float(data.parent_local_purity_score[row])
            >= config.min_local_purity_score
            and float(data.parent_bidirectional_agreement[row])
            >= config.min_bidirectional_agreement
            and float(data.parent_internal_cosine_p10[row])
            >= config.min_internal_cosine_p10
        )
    return result


def _gallery(
    data: _PreparedInput,
    segment: _Segment,
    config: LinkCalibrationConfig,
    cache: dict[tuple[int, int, int], CleanGallery | None],
) -> CleanGallery | None:
    positions = segment.det_positions
    if len(positions) == 0:
        raise ContractError("S03 cannot build a gallery for an empty segment")
    start_order = int(data.det_order_in_micro[positions[0]])
    end_order = int(data.det_order_in_micro[positions[-1]])
    key = (segment.micro_id, start_order, end_order)
    if key in cache:
        return cache[key]
    rows = data.sample_rows[segment.micro_id]
    if len(rows):
        sample_orders = data.det_order_in_micro[data.sample_det_positions[rows]]
        rows = rows[(sample_orders >= start_order) & (sample_orders <= end_order)]
    review = data.det_review_excluded[data.sample_det_positions[rows]]
    if not config.exclude_review_detections:
        review = np.zeros(len(rows), dtype=np.bool_)
    original_inlier = data.sample_s02_inlier[rows]
    if not config.require_s02_prototype_inlier:
        original_inlier = np.ones(len(rows), dtype=np.bool_)
    result = build_clean_gallery(
        data.embeddings[data.sample_embedding_rows[rows]],
        sample_ids=data.sample_ids[rows],
        det_ids=data.sample_det_ids[rows],
        embedding_rows=data.sample_embedding_rows[rows],
        crop_quality=data.sample_crop_quality[rows].astype(np.float32, copy=False),
        other_bbox_max_iou=data.sample_other_bbox_max_iou[rows].astype(
            np.float32, copy=False
        ),
        s02_inlier_mask=original_inlier,
        review_excluded_mask=review,
        max_other_bbox_iou=config.clean_max_other_bbox_iou,
        min_clean_inliers=config.clean_min_samples_per_side,
        outlier_medoid_cosine=config.outlier_medoid_cosine,
        outlier_support_cosine=config.outlier_support_cosine,
        new_prototype_cosine=config.new_prototype_cosine,
    )
    if result is not None and (
        result.internal_cosine_p10 < config.min_internal_cosine_p10
    ):
        result = None
    cache[key] = result
    return result


def _segment_provenance(data: _PreparedInput, segment: _Segment) -> SegmentProvenance:
    positions = segment.det_positions
    first = int(positions[0])
    last = int(positions[-1])
    return SegmentProvenance(
        micro_id=segment.micro_id,
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


def _endpoint_geometry(data: _PreparedInput, position: int) -> EndpointGeometry:
    return EndpointGeometry(
        time_sec=float(data.det_global_time_sec[position]),
        cx_norm=float(data.det_cx_norm[position]),
        cy_norm=float(data.det_cy_norm[position]),
        w_norm=float(data.det_w_norm[position]),
        h_norm=float(data.det_h_norm[position]),
    )


def _endpoint_context(data: _PreparedInput, position: int) -> EndpointContext:
    return EndpointContext(
        max_other_iou=float(data.det_other_bbox_max_iou[position]),
        boundary_distance=float(data.det_boundary_distance[position]),
        clip_id=str(data.det_clip_ids[position]),
    )


def _history(
    data: _PreparedInput, segment: _Segment, config: LinkCalibrationConfig
) -> MotionHistory | None:
    positions = segment.det_positions[-config.velocity_history_detections :]
    if len(positions) < config.min_velocity_detections:
        return None
    return MotionHistory(
        time_sec=data.det_global_time_sec[positions],
        cx_norm=data.det_cx_norm[positions].astype(np.float32, copy=False),
        cy_norm=data.det_cy_norm[positions].astype(np.float32, copy=False),
        w_norm=data.det_w_norm[positions].astype(np.float32, copy=False),
        h_norm=data.det_h_norm[positions].astype(np.float32, copy=False),
    )


def _build_features(
    data: _PreparedInput,
    source: _Segment,
    target: _Segment,
    source_gallery: CleanGallery,
    target_gallery: CleanGallery,
    mode: LinkMode,
    config: LinkCalibrationConfig,
) -> tuple[dict[str, float], bool] | None:
    validate_disjoint_sides(source_gallery, target_gallery)
    source_end = int(source.det_positions[-1])
    target_start = int(target.det_positions[0])
    history = _history(data, source, config) if mode == "short" else None
    if mode == "short" and history is None:
        return None
    source_context = _endpoint_context(data, source_end)
    target_context = _endpoint_context(data, target_start)
    values = pair_features(
        source_gallery,
        target_gallery,
        _endpoint_geometry(data, source_end),
        _endpoint_geometry(data, target_start),
        source_context,
        target_context,
        mode=mode,
        source_history=history,
    )
    high_overlap = has_high_endpoint_overlap(
        source_context,
        target_context,
        threshold=config.hard_negative_high_overlap_iou_threshold,
    ) or bool(
        source_gallery.max_other_bbox_iou
        >= config.hard_negative_high_overlap_iou_threshold
        or target_gallery.max_other_bbox_iou
        >= config.hard_negative_high_overlap_iou_threshold
    )
    return values, high_overlap


class ProductionFeatureStore:
    """Cache clean full-tracklet galleries for downstream S04/S05 scoring.

    This deliberately shares `_gallery` and `_build_features` with pseudo-pair
    calibration so production cannot silently return to contaminated S02 whole
    micro prototypes.
    """

    def __init__(self, data: CalibrationInput, config: LinkCalibrationConfig) -> None:
        if not isinstance(config, LinkCalibrationConfig):
            raise ContractError("S03 production feature config is invalid")
        self._data = _prepare(data)
        self._config = config
        self._gallery_cache: dict[tuple[int, int, int], CleanGallery | None] = {}

    def build_pair(
        self, source_tracklet_id: int, target_tracklet_id: int, mode: LinkMode
    ) -> TrackletPairFeatures:
        source_id = int(source_tracklet_id)
        target_id = int(target_tracklet_id)
        if source_id == target_id:
            raise ContractError("S03 source and target tracklet IDs must differ")
        if source_id not in self._data.paths or target_id not in self._data.paths:
            raise ContractError("S03 score_pair references an unknown tracklet ID")
        source = _Segment(source_id, self._data.paths[source_id])
        target = _Segment(target_id, self._data.paths[target_id])
        source_end = int(source.det_positions[-1])
        target_start = int(target.det_positions[0])
        if (
            float(self._data.det_global_time_sec[target_start])
            <= float(self._data.det_global_time_sec[source_end])
        ):
            raise ContractError("S03 production candidate gap must be positive")

        # Reliability is assessed independently of feature availability.  In
        # particular S04 must retain endpoint/raw-sample overlap diagnostics
        # even when a review exclusion, missing gallery, or missing motion
        # history makes scoring fail closed.
        threshold = self._config.hard_negative_high_overlap_iou_threshold
        source_context = _endpoint_context(self._data, source_end)
        target_context = _endpoint_context(self._data, target_start)
        source_sample_rows = self._data.sample_rows[source_id]
        target_sample_rows = self._data.sample_rows[target_id]
        high_overlap = has_high_endpoint_overlap(
            source_context, target_context, threshold=threshold
        ) or bool(
            (
                len(source_sample_rows)
                and float(
                    np.max(
                        self._data.sample_other_bbox_max_iou[source_sample_rows]
                    )
                )
                >= threshold
            )
            or (
                len(target_sample_rows)
                and float(
                    np.max(
                        self._data.sample_other_bbox_max_iou[target_sample_rows]
                    )
                )
                >= threshold
            )
        )
        source_gallery = _gallery(
            self._data, source, self._config, self._gallery_cache
        )
        target_gallery = _gallery(
            self._data, target, self._config, self._gallery_cache
        )
        source_assessment = GalleryAssessment.from_gallery(source_gallery)
        target_assessment = GalleryAssessment.from_gallery(target_gallery)
        if self._config.exclude_review_detections:
            source_review = bool(self._data.det_review_excluded[source_end])
            target_review = bool(self._data.det_review_excluded[target_start])
            if source_review or target_review:
                side = (
                    "source_target"
                    if source_review and target_review
                    else "source"
                    if source_review
                    else "target"
                )
                return TrackletPairFeatures(
                    None,
                    False,
                    high_overlap,
                    f"{side}_review_excluded_endpoint",
                    source_assessment,
                    target_assessment,
                )
        if source_gallery is None or target_gallery is None:
            side = (
                "source_target"
                if source_gallery is None and target_gallery is None
                else "source"
                if source_gallery is None
                else "target"
            )
            return TrackletPairFeatures(
                None,
                False,
                high_overlap,
                f"{side}_gallery_missing",
                source_assessment,
                target_assessment,
            )
        built = _build_features(
            self._data,
            source,
            target,
            source_gallery,
            target_gallery,
            mode,
            self._config,
        )
        if built is None:
            return TrackletPairFeatures(
                None,
                True,
                high_overlap,
                "source_motion_history_missing",
                source_assessment,
                target_assessment,
            )
        values, feature_high_overlap = built
        if feature_high_overlap != high_overlap:
            raise ContractError("S03 production overlap assessments disagree")
        return TrackletPairFeatures(
            values,
            True,
            high_overlap,
            None,
            source_assessment,
            target_assessment,
        )


def _endpoint_candidates(
    data: _PreparedInput,
    path: np.ndarray,
    *,
    mode: LinkMode,
    target_gap: float,
    config: LinkCalibrationConfig,
) -> list[tuple[float, int, int]]:
    times = data.det_global_time_sec[path]
    tolerance = config.gap_target_tolerance_sec
    candidates: list[tuple[float, int, int]] = []
    for source_index in range(len(path) - 1):
        desired = float(times[source_index] + target_gap)
        lower = int(np.searchsorted(times, desired - tolerance, side="left"))
        upper = int(np.searchsorted(times, desired + tolerance, side="right"))
        lower = max(lower, source_index + 1)
        for target_index in range(lower, upper):
            gap = float(times[target_index] - times[source_index])
            if gap <= 0.0:
                continue
            if mode == "short" and gap > config.short_gap_max_sec:
                continue
            if mode == "long" and gap <= config.short_gap_max_sec:
                continue
            source_position = int(path[source_index])
            target_position = int(path[target_index])
            if config.exclude_review_detections and (
                bool(data.det_review_excluded[source_position])
                or bool(data.det_review_excluded[target_position])
            ):
                continue
            candidates.append((abs(gap - target_gap), source_index, target_index))
    candidates.sort(
        key=lambda item: (
            item[0],
            int(data.det_ids[path[item[1]]]),
            int(data.det_ids[path[item[2]]]),
        )
    )
    return candidates


def _positive_candidates(
    data: _PreparedInput,
    config: LinkCalibrationConfig,
    eligible: Mapping[int, bool],
    gallery_cache: dict[tuple[int, int, int], CleanGallery | None],
    *,
    logger: LogFn | None,
    progress_interval_sec: float,
) -> list[_PositiveCandidate]:
    result: list[_PositiveCandidate] = []
    targets: tuple[tuple[LinkMode, float], ...] = tuple(
        ("short", float(value)) for value in config.short_gap_targets_sec
    ) + tuple(("long", float(value)) for value in config.long_gap_targets_sec)
    parent_ids = sorted(data.paths)
    last_report = time.monotonic()
    completed_targets = 0
    total_targets = sum(bool(eligible[micro_id]) for micro_id in parent_ids) * len(
        targets
    )
    for parent_number, micro_id in enumerate(parent_ids, start=1):
        if not eligible[micro_id]:
            continue
        path = data.paths[micro_id]
        for mode, target_gap in targets:
            accepted = 0
            for _, source_index, target_index in _endpoint_candidates(
                data,
                path,
                mode=mode,
                target_gap=target_gap,
                config=config,
            ):
                source = _Segment(micro_id, path[: source_index + 1])
                target = _Segment(micro_id, path[target_index:])
                source_gallery = _gallery(data, source, config, gallery_cache)
                target_gallery = _gallery(data, target, config, gallery_cache)
                if source_gallery is None or target_gallery is None:
                    continue
                validate_disjoint_sides(source_gallery, target_gallery)
                if mode == "short" and _history(data, source, config) is None:
                    continue
                result.append(
                    _PositiveCandidate(
                        source=source,
                        target=target,
                        source_gallery=source_gallery,
                        target_gallery=target_gallery,
                        mode=mode,
                        gap_target_sec=target_gap,
                    )
                )
                accepted += 1
                if accepted >= config.max_positive_pairs_per_parent_per_gap:
                    break
            completed_targets += 1
            now = time.monotonic()
            if logger is not None and now - last_report >= progress_interval_sec:
                logger(
                    f"[s03] pseudo-positive mining: parent {parent_number:,}/"
                    f"{len(parent_ids):,}, gap targets {completed_targets:,}/"
                    f"{total_targets:,}, accepted {len(result):,}"
                )
                last_report = now
    if logger is not None:
        logger(
            f"[s03] pseudo-positive mining complete: parents {len(parent_ids):,}, "
            f"gap targets {completed_targets:,}/{total_targets:,}, "
            f"accepted {len(result):,}"
        )
    return result


def _simultaneous_count(
    data: _PreparedInput, left: _Segment, right: _Segment
) -> int:
    left_keys = {
        (str(data.det_clip_ids[pos]), int(data.det_global_frames[pos]))
        for pos in left.det_positions
    }
    right_keys = {
        (str(data.det_clip_ids[pos]), int(data.det_global_frames[pos]))
        for pos in right.det_positions
    }
    return len(left_keys & right_keys)


def _group_id(data: _PreparedInput, candidate: _PositiveCandidate) -> str:
    source_end = int(data.det_ids[candidate.source.det_positions[-1]])
    target_start = int(data.det_ids[candidate.target.det_positions[0]])
    return (
        f"parent-{candidate.source.micro_id}:mode-{candidate.mode}:"
        f"gap-{candidate.gap_target_sec:.6f}:a-{source_end}:b-{target_start}"
    )


def _make_pair(
    data: _PreparedInput,
    config: LinkCalibrationConfig,
    *,
    candidate_group_id: str,
    split: SplitName,
    calibration_role: CalibrationRole,
    parent_micro_id: int,
    source: _Segment,
    target: _Segment,
    source_gallery: CleanGallery,
    target_gallery: CleanGallery,
    mode: LinkMode,
    gap_target_sec: float,
    label: bool,
    hard_negative_rank: int,
) -> PseudoPair | None:
    built = _build_features(
        data,
        source,
        target,
        source_gallery,
        target_gallery,
        mode,
        config,
    )
    if built is None:
        return None
    values, high_overlap = built
    source_end = int(source.det_positions[-1])
    target_start = int(target.det_positions[0])
    gap = float(
        data.det_global_time_sec[target_start] - data.det_global_time_sec[source_end]
    )
    if gap <= 0.0:
        raise ContractError("S03 pseudo-pair construction produced a non-positive gap")
    if label:
        pair_kind = "pseudo_positive"
        stratum = "positive_high_overlap" if high_overlap else "positive_clean"
        suffix = "positive"
    else:
        pair_kind = "hard_negative_simultaneous"
        stratum = "high_overlap_stress" if high_overlap else "hard_negative"
        suffix = (
            f"negative-{target.micro_id}-{int(data.det_ids[target_start])}"
        )
    return PseudoPair(
        pair_id=f"{candidate_group_id}:{suffix}",
        candidate_group_id=candidate_group_id,
        parent_group_id=f"parent-{parent_micro_id}",
        split=split,
        calibration_role=calibration_role,
        mode=mode,
        label=label,
        pair_kind=pair_kind,
        stratum=stratum,
        parent_micro_id=parent_micro_id,
        source=_segment_provenance(data, source),
        target=_segment_provenance(data, target),
        source_gallery=GalleryProvenance.from_gallery(source_gallery),
        target_gallery=GalleryProvenance.from_gallery(target_gallery),
        gap_target_sec=float(gap_target_sec),
        gap_error_sec=float(gap - gap_target_sec),
        hard_negative_rank=hard_negative_rank,
        appearance_present=True,
        high_overlap=high_overlap,
        features=values,
    )


def _negative_pairs(
    data: _PreparedInput,
    config: LinkCalibrationConfig,
    positive: _PositiveCandidate,
    positive_pair: PseudoPair,
    partitions: Mapping[int, ParentPartition],
    gallery_cache: dict[tuple[int, int, int], CleanGallery | None],
) -> list[PseudoPair]:
    if not config.hard_negative_enabled:
        return []
    target_start = int(positive.target.det_positions[0])
    source_end = int(positive.source.det_positions[-1])
    frame_key = (
        str(data.det_clip_ids[target_start]),
        int(data.det_global_frames[target_start]),
    )
    positive_gap = float(
        data.det_global_time_sec[target_start] - data.det_global_time_sec[source_end]
    )
    candidates: list[PseudoPair] = []
    for position in data.frame_rows.get(frame_key, ()):
        target_micro = int(data.det_micro_ids[position])
        if target_micro == positive.source.micro_id:
            continue
        if partitions[target_micro] != partitions[positive.source.micro_id]:
            continue
        if config.exclude_review_detections and bool(data.det_review_excluded[position]):
            continue
        path = data.paths[target_micro]
        start_order = int(data.det_order_in_micro[position])
        target = _Segment(target_micro, path[start_order:])
        if len(target.det_positions) == 0 or int(target.det_positions[0]) != position:
            raise ContractError("S03 negative target segment did not start at B's frame")
        negative_gap = float(
            data.det_global_time_sec[position] - data.det_global_time_sec[source_end]
        )
        if negative_gap <= 0.0:
            continue
        if abs(negative_gap - positive_gap) > config.hard_negative_gap_match_tolerance_sec:
            continue
        # The exact shared start frame already proves one simultaneous frame,
        # which is the fixed production policy.  Only build full segment sets
        # if a stricter synthetic/configured policy asks for more.
        if config.hard_negative_min_simultaneous_frames > 1 and (
            _simultaneous_count(data, positive.target, target)
            < config.hard_negative_min_simultaneous_frames
        ):
            continue
        target_gallery = _gallery(data, target, config, gallery_cache)
        if target_gallery is None:
            continue
        pair = _make_pair(
            data,
            config,
            candidate_group_id=positive_pair.candidate_group_id,
            split=positive_pair.split,
            calibration_role=positive_pair.calibration_role,
            parent_micro_id=positive.source.micro_id,
            source=positive.source,
            target=target,
            source_gallery=positive.source_gallery,
            target_gallery=target_gallery,
            mode=positive.mode,
            gap_target_sec=positive.gap_target_sec,
            label=False,
            hard_negative_rank=0,
        )
        if pair is not None:
            candidates.append(pair)

    # Appearance-hardest first, with artifact IDs as total deterministic ties.
    candidates.sort(
        key=lambda pair: (
            -float(pair.features["prototype_cosine_max"]),
            -float(pair.features["prototype_cosine_top3_mean"]),
            -float(pair.features["medoid_cosine"]),
            pair.target.micro_id,
            pair.target.start_det_id,
        )
    )
    ranked = [
        PseudoPair(
            **{
                **pair.__dict__,
                "hard_negative_rank": rank,
            }
        )
        for rank, pair in enumerate(candidates, start=1)
    ]
    pool = ranked[: config.hard_negative_max_candidates_per_positive]
    selected_ids = {
        pair.pair_id
        for pair in pool[: config.hard_negative_appearance_top_k_per_positive]
    }
    if config.hard_negative_retain_high_overlap_stress:
        # Stress examples are diagnostics, not an optimization shortlist.  Keep
        # every valid high-overlap candidate even when it falls outside top-k
        # (or the ordinary candidate pool) so contamination cannot disappear.
        selected_ids.update(pair.pair_id for pair in ranked if pair.high_overlap)
    return [pair for pair in ranked if pair.pair_id in selected_ids]


def generate_pseudo_pairs(
    data: CalibrationInput,
    config: LinkCalibrationConfig,
    *,
    logger: LogFn | None = None,
    progress_interval_sec: float | None = None,
    clean_gallery_status: dict[int, bool] | None = None,
) -> tuple[PseudoPair, ...]:
    """Generate deterministic positive and simultaneous hard-negative pairs.

    Every emitted example has a positive production-domain gap and two
    independently rebuilt clean galleries.  Hard-negative mining happens only
    among micros already assigned to the positive's pre-mining partition, so no
    candidate or ranking crosses train, calibration-selection, independent
    certification, or audit boundaries.
    """

    if not isinstance(config, LinkCalibrationConfig):
        raise ContractError("S03 pseudo-pair config must be LinkCalibrationConfig")
    if logger is not None and not callable(logger):
        raise ContractError("S03 pseudo-pair logger must be callable")
    interval = (
        float(config.progress_interval_sec)
        if progress_interval_sec is None
        else float(progress_interval_sec)
    )
    if not math.isfinite(interval) or interval <= 0.0:
        raise ContractError("S03 pseudo-pair progress interval must be positive")
    prepared = _prepare(data)
    partitions = _assign_parent_partitions_prepared(prepared, config)
    eligible = _eligible_parents(prepared, config)
    gallery_cache: dict[tuple[int, int, int], CleanGallery | None] = {}
    if clean_gallery_status is not None:
        if not isinstance(clean_gallery_status, dict) or clean_gallery_status:
            raise ContractError("S03 clean-gallery status output must be an empty dict")
        last_gallery_report = time.monotonic()
        for completed, micro_id in enumerate(sorted(prepared.paths), start=1):
            full_segment = _Segment(micro_id, prepared.paths[micro_id])
            clean_gallery_status[micro_id] = (
                _gallery(prepared, full_segment, config, gallery_cache) is not None
            )
            now = time.monotonic()
            if logger is not None and now - last_gallery_report >= interval:
                logger(
                    f"[s03] clean production galleries: {completed:,}/"
                    f"{len(prepared.paths):,}"
                )
                last_gallery_report = now
        if logger is not None:
            present = sum(clean_gallery_status.values())
            logger(
                f"[s03] clean production galleries complete: present={present:,}, "
                f"missing={len(clean_gallery_status) - present:,}"
            )
    positives = _positive_candidates(
        prepared,
        config,
        eligible,
        gallery_cache,
        logger=logger,
        progress_interval_sec=interval,
    )
    output: list[PseudoPair] = []
    last_report = time.monotonic()
    for group_number, candidate in enumerate(positives, start=1):
        partition = partitions[candidate.source.micro_id]
        split = _partition_split(partition)
        calibration_role = _partition_calibration_role(partition)
        group_id = _group_id(prepared, candidate)
        positive_pair = _make_pair(
            prepared,
            config,
            candidate_group_id=group_id,
            split=split,
            calibration_role=calibration_role,
            parent_micro_id=candidate.source.micro_id,
            source=candidate.source,
            target=candidate.target,
            source_gallery=candidate.source_gallery,
            target_gallery=candidate.target_gallery,
            mode=candidate.mode,
            gap_target_sec=candidate.gap_target_sec,
            label=True,
            hard_negative_rank=0,
        )
        if positive_pair is None:
            continue
        negatives = _negative_pairs(
            prepared,
            config,
            candidate,
            positive_pair,
            partitions,
            gallery_cache,
        )
        # A positive without an independently proven simultaneous negative is
        # not a calibration group and must not leak into fitting by itself.
        if not negatives:
            now = time.monotonic()
            if logger is not None and now - last_report >= interval:
                logger(
                    f"[s03] pseudo-pair groups: {group_number:,}/"
                    f"{len(positives):,}, emitted {len(output):,} rows"
                )
                last_report = now
            continue
        output.append(positive_pair)
        output.extend(negatives)
        now = time.monotonic()
        if logger is not None and now - last_report >= interval:
            logger(
                f"[s03] pseudo-pair groups: {group_number:,}/"
                f"{len(positives):,}, emitted {len(output):,} rows"
            )
            last_report = now

    if logger is not None:
        logger(
            f"[s03] pseudo-pair group mining complete: {len(positives):,}/"
            f"{len(positives):,}, emitted {len(output):,} rows before sort"
        )

    output.sort(
        key=lambda pair: (
            _SPLIT_RANK[pair.split],
            pair.calibration_role,
            0 if pair.mode == "short" else 1,
            pair.parent_micro_id,
            pair.gap_target_sec,
            pair.source.end_det_id,
            pair.candidate_group_id,
            0 if pair.label else 1,
            pair.hard_negative_rank,
            pair.target.start_det_id,
            pair.pair_id,
        )
    )
    if len({pair.pair_id for pair in output}) != len(output):
        raise ContractError("S03 pseudo-pair IDs are not unique")
    if logger is not None:
        logger(f"[s03] pseudo-pair construction complete: {len(output):,} rows")
    return tuple(output)


def pseudo_pairs_as_rows(pairs: tuple[PseudoPair, ...] | list[PseudoPair]) -> list[dict[str, Any]]:
    """Flatten generated pairs without changing their deterministic order."""

    return [pair.as_row() for pair in pairs]


# Concise alias for stage code and downstream callers.
build_pseudo_pairs = generate_pseudo_pairs


__all__ = [
    "CalibrationInput",
    "GalleryAssessment",
    "GalleryProvenance",
    "PseudoPair",
    "ProductionFeatureStore",
    "SegmentProvenance",
    "SplitName",
    "TrackletPairFeatures",
    "assign_parent_splits",
    "build_pseudo_pairs",
    "generate_pseudo_pairs",
    "pseudo_pairs_as_rows",
]
