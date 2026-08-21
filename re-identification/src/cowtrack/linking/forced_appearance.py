"""Pure graded appearance descriptors for forced S05 identity assignment.

This module deliberately performs no video decoding, model inference, graph
construction, or identity merge.  It combines three explicitly distinguished
sources of appearance evidence:

``A_CLEAN``
    Existing, usable S04 clean-gallery prototypes.  Their float16 values and
    masks are copied without numerical reconstruction.
``B_EXISTING_DEGRADED``
    Existing S02 sample embeddings for an S04 path without usable clean
    prototypes.  Samples are selected by a fixed, progressively relaxed policy.
``C_SUPPLEMENTAL``
    Externally supplied embeddings for paths that had no S02 samples at all.

The base builder is allowed to return explicit C gaps.  The completion builder
is fail-closed: every stable ID must have a usable descriptor when it returns.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Any

import numpy as np

from cowtrack.appearance.prototypes import build_tracklet_prototypes
from cowtrack.config import ContractError


GRADE_A_CLEAN = "A_CLEAN"
GRADE_B_EXISTING_DEGRADED = "B_EXISTING_DEGRADED"
GRADE_C_SUPPLEMENTAL = "C_REENCODED_DEGRADED"
GRADE_MISSING = "MISSING"

_B_CLEAN_INLIER = "inlier_iou_lt_0_25"
_B_ALL_INLIER = "all_inlier"
_B_ALL_SAMPLES = "all_samples"
_A_EXISTING = "existing_s04_clean"
_C_SUPPLEMENTAL = "supplemental_embeddings"
_CLEAN_IOU_THRESHOLD = 0.25
_PROTOTYPE_SLOTS = 3
_NEW_PROTOTYPE_COSINE = 0.92
_NORM_ATOL = 2e-3


@dataclass(frozen=True)
class ForcedAppearanceBundle:
    """Dense graded descriptor tensors plus one provenance row per stable ID."""

    stable_ids: np.ndarray
    prototypes: np.ndarray
    prototype_mask: np.ndarray
    descriptor_centers: np.ndarray
    rows: tuple[Mapping[str, Any], ...]
    zero_sample_stable_ids: tuple[int, ...]


@dataclass(frozen=True)
class SupplementalStableEmbeddings:
    """Deterministically identified supplemental embeddings for one stable ID."""

    stable_id: int
    sample_ids: np.ndarray
    embeddings: np.ndarray
    quality: np.ndarray


def _integer_vector(
    values: object, *, name: str, length: int | None = None
) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or array.dtype.kind not in "iu":
        raise ContractError(f"forced appearance {name} must be an integer vector")
    if length is not None and len(array) != length:
        raise ContractError(f"forced appearance {name} has an inconsistent length")
    if array.dtype.kind == "u" and len(array):
        if int(np.max(array)) > np.iinfo(np.int64).max:
            raise ContractError(f"forced appearance {name} exceeds signed int64")
    return array.astype(np.int64, copy=False)


def _boolean_vector(values: object, *, name: str, length: int) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or len(array) != length or array.dtype != np.bool_:
        raise ContractError(f"forced appearance {name} must be a boolean vector")
    return array


def _unit_embeddings(
    values: object,
    *,
    name: str,
    expected_dim: int | None = None,
    allow_empty: bool = True,
) -> np.ndarray:
    array = np.asarray(values)
    if (
        array.ndim != 2
        or array.dtype.kind != "f"
        or array.shape[1] <= 0
        or (expected_dim is not None and array.shape[1] != expected_dim)
    ):
        raise ContractError(
            f"forced appearance {name} must be floating [num_samples, D]"
        )
    if not allow_empty and len(array) == 0:
        raise ContractError(f"forced appearance {name} cannot be empty")
    result = array.astype(np.float32, copy=False)
    if not np.all(np.isfinite(result)):
        raise ContractError(f"forced appearance {name} must be finite")
    if len(result):
        norms = np.linalg.norm(result, axis=1)
        if not np.allclose(norms, 1.0, rtol=0.0, atol=_NORM_ATOL):
            raise ContractError(f"forced appearance {name} must be L2-normalized")
    return result


def _quality_vector(values: object, *, name: str, length: int) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or len(array) != length or array.dtype.kind not in "fiu":
        raise ContractError(f"forced appearance {name} must be a numeric vector")
    result = array.astype(np.float32, copy=False)
    if not np.all(np.isfinite(result)) or np.any((result <= 0.0) | (result > 1.0)):
        raise ContractError(f"forced appearance {name} must be finite and in (0, 1]")
    return result


def _overlap_vector(values: object, *, length: int) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or len(array) != length or array.dtype.kind not in "fiu":
        raise ContractError("forced appearance sample IoU must be a numeric vector")
    result = array.astype(np.float32, copy=False)
    if not np.all(np.isfinite(result)) or np.any((result < 0.0) | (result > 1.0)):
        raise ContractError("forced appearance sample IoU must be finite and in [0, 1]")
    return result


def _normalized_mean(prototypes: np.ndarray) -> np.ndarray:
    mean = np.mean(np.asarray(prototypes, dtype=np.float64), axis=0)
    norm = float(np.linalg.norm(mean))
    if not math.isfinite(norm) or norm <= np.finfo(np.float64).eps:
        raise ContractError("forced appearance descriptor center has zero norm")
    center = (mean / norm).astype(np.float32)
    if not np.all(np.isfinite(center)):
        raise ContractError("forced appearance descriptor center is non-finite")
    return center


def _relaxed_prototypes(
    embeddings: np.ndarray,
    quality: np.ndarray,
    sample_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, tuple[int, ...]]:
    """Return at most three prototypes after canonical sample-ID ordering.

    One- and two-sample descriptors retain each normalized observation.  For
    three or more observations the public deterministic prototype builder is
    reused with outlier rejection disabled, while retaining its spherical
    k-medoids and quality-weighted prototype construction.
    """

    count, dimension = embeddings.shape
    if count == 0:
        raise ContractError("forced appearance cannot build prototypes from zero samples")
    order = np.argsort(sample_ids, kind="stable")
    ordered_ids = sample_ids[order]
    if len(np.unique(ordered_ids)) != count:
        raise ContractError("forced appearance selected sample IDs must be unique")
    ordered_embeddings = embeddings[order].astype(np.float32, copy=True)
    # Persisted S02 embeddings are float16 and are valid within the upstream
    # 2e-3 tolerance.  Re-normalize before a one/two-sample row becomes a
    # prototype so the new float32 construction has a strict unit-vector
    # contract instead of preserving quantization drift.
    norms = np.linalg.norm(ordered_embeddings, axis=1, keepdims=True)
    if not np.all(np.isfinite(norms)) or np.any(norms <= 1e-12):
        raise ContractError("forced appearance selected embedding has zero norm")
    ordered_embeddings /= norms
    ordered_quality = quality[order]

    values = np.zeros((_PROTOTYPE_SLOTS, dimension), dtype=np.float32)
    mask = np.zeros(_PROTOTYPE_SLOTS, dtype=np.bool_)
    if count <= 2:
        values[:count] = ordered_embeddings
        mask[:count] = True
    else:
        built = build_tracklet_prototypes(
            ordered_embeddings,
            np.zeros(count, dtype=np.int64),
            ordered_quality,
            np.asarray([0], dtype=np.int64),
            min_samples=3,
            max_prototypes=3,
            outlier_medoid_cosine=-1.0,
            outlier_support_cosine=None,
            new_prototype_cosine=_NEW_PROTOTYPE_COSINE,
        )
        values[:] = built.prototypes[0]
        mask[:] = built.prototype_mask[0]
        if not np.any(mask):
            raise ContractError("forced appearance relaxed prototype builder returned empty")

    valid = values[mask]
    if not len(valid) or not np.allclose(
        np.linalg.norm(valid, axis=1), 1.0, rtol=0.0, atol=2e-6
    ):
        raise ContractError("forced appearance relaxed prototypes are not normalized")
    return values, mask, tuple(map(int, ordered_ids))


def _base_row(
    *,
    stable_id: int,
    grade: str,
    usable: bool,
    selection_policy: str | None,
    prototype_count: int,
    input_sample_ids: np.ndarray,
    selected_sample_ids: Sequence[int],
    selected_quality: np.ndarray,
    selected_iou: np.ndarray,
    selected_inlier: np.ndarray,
    missing_reason: str | None,
) -> Mapping[str, Any]:
    if len(selected_quality):
        quality_min: float | None = float(np.min(selected_quality))
        quality_mean: float | None = float(np.mean(selected_quality, dtype=np.float64))
        maximum_iou: float | None = float(np.max(selected_iou))
    else:
        quality_min = None
        quality_mean = None
        maximum_iou = None
    row = {
        "stable_id": stable_id,
        "evidence_grade": grade,
        "descriptor_usable": usable,
        "selection_policy": selection_policy,
        "num_valid_prototypes": prototype_count,
        "num_input_samples": len(input_sample_ids),
        "num_selected_samples": len(selected_sample_ids),
        "input_sample_ids": tuple(map(int, input_sample_ids)),
        "selected_sample_ids": tuple(map(int, selected_sample_ids)),
        "selected_quality_min": quality_min,
        "selected_quality_mean": quality_mean,
        "selected_max_other_bbox_iou": maximum_iou,
        "used_high_overlap": bool(
            len(selected_iou) and np.any(selected_iou >= _CLEAN_IOU_THRESHOLD)
        ),
        "used_s02_outlier": bool(len(selected_inlier) and np.any(~selected_inlier)),
        "missing_reason": missing_reason,
    }
    return MappingProxyType(row)


def _freeze_array(values: np.ndarray) -> np.ndarray:
    result = np.ascontiguousarray(values)
    result.setflags(write=False)
    return result


def _validate_bundle(bundle: ForcedAppearanceBundle, *, require_total: bool) -> None:
    if not isinstance(bundle, ForcedAppearanceBundle):
        raise ContractError("forced appearance bundle has the wrong type")
    stable_ids = np.asarray(bundle.stable_ids)
    prototypes = np.asarray(bundle.prototypes)
    mask = np.asarray(bundle.prototype_mask)
    centers = np.asarray(bundle.descriptor_centers)
    if (
        stable_ids.ndim != 1
        or stable_ids.dtype != np.int64
        or not np.array_equal(stable_ids, np.arange(len(stable_ids), dtype=np.int64))
    ):
        raise ContractError("forced appearance bundle stable IDs are not dense")
    if (
        prototypes.dtype != np.float16
        or prototypes.ndim != 3
        or prototypes.shape[:2] != (len(stable_ids), _PROTOTYPE_SLOTS)
        or prototypes.shape[2] <= 0
        or mask.dtype != np.bool_
        or mask.shape != prototypes.shape[:2]
        or centers.dtype != np.float32
        or centers.shape != (len(stable_ids), prototypes.shape[2])
        or len(bundle.rows) != len(stable_ids)
    ):
        raise ContractError("forced appearance bundle tensor contract differs")
    if not np.all(np.isfinite(prototypes)) or not np.all(np.isfinite(centers)):
        raise ContractError("forced appearance bundle contains non-finite values")
    if np.any(prototypes[~mask] != np.float16(0.0)):
        raise ContractError("forced appearance unused prototype slots must be zero")
    counts = np.count_nonzero(mask, axis=1)
    usable = counts > 0
    if np.any(mask[:, 1:] & ~mask[:, :-1]):
        raise ContractError("forced appearance prototype masks must be prefix masks")
    valid = prototypes[mask].astype(np.float32, copy=False)
    if len(valid) and not np.allclose(
        np.linalg.norm(valid, axis=1), 1.0, rtol=0.0, atol=_NORM_ATOL
    ):
        raise ContractError("forced appearance persisted prototypes are not normalized")
    if np.any(centers[~usable] != 0.0):
        raise ContractError("forced appearance missing centers must be zero")
    if np.any(usable) and not np.allclose(
        np.linalg.norm(centers[usable], axis=1), 1.0, rtol=0.0, atol=2e-6
    ):
        raise ContractError("forced appearance descriptor centers are not normalized")
    row_ids = [int(row["stable_id"]) for row in bundle.rows]
    row_usable = [bool(row["descriptor_usable"]) for row in bundle.rows]
    row_counts = [int(row["num_valid_prototypes"]) for row in bundle.rows]
    if (
        row_ids != list(range(len(stable_ids)))
        or row_usable != list(map(bool, usable))
        or row_counts != list(map(int, counts))
    ):
        raise ContractError("forced appearance bundle row/tensor provenance differs")
    missing = tuple(map(int, np.flatnonzero(~usable)))
    if tuple(bundle.zero_sample_stable_ids) != missing:
        raise ContractError("forced appearance C-gap IDs differ from tensor coverage")
    if require_total and missing:
        raise ContractError(
            "forced appearance completion left stable IDs without descriptors: "
            f"{list(missing[:10])}"
        )


def build_forced_appearance_base(
    *,
    stable_ids: np.ndarray,
    existing_prototypes: np.ndarray,
    existing_prototype_mask: np.ndarray,
    existing_usable: np.ndarray,
    sample_ids: np.ndarray,
    sample_embeddings: np.ndarray,
    sample_stable_ids: np.ndarray,
    sample_quality: np.ndarray,
    sample_other_bbox_max_iou: np.ndarray,
    sample_s02_inlier: np.ndarray,
) -> ForcedAppearanceBundle:
    """Build A/B descriptors and return explicit zero-sample stable IDs for C."""

    ids = _integer_vector(stable_ids, name="stable_ids")
    if not len(ids) or not np.array_equal(ids, np.arange(len(ids), dtype=np.int64)):
        raise ContractError("forced appearance stable IDs must be dense from zero")
    prototypes_in = np.asarray(existing_prototypes)
    if (
        prototypes_in.dtype != np.float16
        or prototypes_in.ndim != 3
        or prototypes_in.shape[:2] != (len(ids), _PROTOTYPE_SLOTS)
        or prototypes_in.shape[2] <= 0
    ):
        raise ContractError(
            "forced appearance existing prototypes must be float16 [N, 3, D]"
        )
    dimension = int(prototypes_in.shape[2])
    if not np.all(np.isfinite(prototypes_in)):
        raise ContractError("forced appearance existing prototypes must be finite")
    mask_in = np.asarray(existing_prototype_mask)
    if mask_in.dtype != np.bool_ or mask_in.shape != prototypes_in.shape[:2]:
        raise ContractError("forced appearance existing prototype mask differs")
    usable_in = _boolean_vector(
        existing_usable, name="existing_usable", length=len(ids)
    )
    if not np.array_equal(usable_in, np.any(mask_in, axis=1)):
        raise ContractError("forced appearance existing usable/mask values differ")
    if np.any(mask_in[:, 1:] & ~mask_in[:, :-1]):
        raise ContractError("forced appearance existing masks must be prefix masks")
    if np.any(prototypes_in[~mask_in] != np.float16(0.0)):
        raise ContractError("forced appearance existing unused prototypes must be zero")
    existing_valid = prototypes_in[mask_in].astype(np.float32, copy=False)
    if len(existing_valid) and not np.allclose(
        np.linalg.norm(existing_valid, axis=1), 1.0, rtol=0.0, atol=_NORM_ATOL
    ):
        raise ContractError("forced appearance existing prototypes are not normalized")

    embeddings = _unit_embeddings(
        sample_embeddings, name="sample_embeddings", expected_dim=dimension
    )
    sample_count = len(embeddings)
    samples = _integer_vector(sample_ids, name="sample_ids", length=sample_count)
    if len(np.unique(samples)) != sample_count:
        raise ContractError("forced appearance sample IDs must be unique")
    sample_stable = _integer_vector(
        sample_stable_ids, name="sample_stable_ids", length=sample_count
    )
    if len(sample_stable) and (
        np.any(sample_stable < 0) or np.any(sample_stable >= len(ids))
    ):
        raise ContractError("forced appearance sample references an unknown stable ID")
    quality = _quality_vector(sample_quality, name="sample_quality", length=sample_count)
    overlap = _overlap_vector(sample_other_bbox_max_iou, length=sample_count)
    inlier = _boolean_vector(
        sample_s02_inlier, name="sample_s02_inlier", length=sample_count
    )

    prototypes = np.zeros_like(prototypes_in)
    mask = np.zeros_like(mask_in)
    centers = np.zeros((len(ids), dimension), dtype=np.float32)
    rows: list[Mapping[str, Any]] = []
    missing: list[int] = []

    for stable_id in map(int, ids):
        positions = np.flatnonzero(sample_stable == stable_id)
        if len(positions):
            position_order = np.argsort(samples[positions], kind="stable")
            positions = positions[position_order]
        input_ids = samples[positions]
        if usable_in[stable_id]:
            prototypes[stable_id] = prototypes_in[stable_id]
            mask[stable_id] = mask_in[stable_id]
            centers[stable_id] = _normalized_mean(
                prototypes[stable_id, mask[stable_id]].astype(np.float32)
            )
            rows.append(
                _base_row(
                    stable_id=stable_id,
                    grade=GRADE_A_CLEAN,
                    usable=True,
                    selection_policy=_A_EXISTING,
                    prototype_count=int(np.count_nonzero(mask[stable_id])),
                    input_sample_ids=input_ids,
                    selected_sample_ids=(),
                    selected_quality=np.empty(0, dtype=np.float32),
                    selected_iou=np.empty(0, dtype=np.float32),
                    selected_inlier=np.empty(0, dtype=np.bool_),
                    missing_reason=None,
                )
            )
            continue

        if not len(positions):
            missing.append(stable_id)
            rows.append(
                _base_row(
                    stable_id=stable_id,
                    grade=GRADE_MISSING,
                    usable=False,
                    selection_policy=None,
                    prototype_count=0,
                    input_sample_ids=input_ids,
                    selected_sample_ids=(),
                    selected_quality=np.empty(0, dtype=np.float32),
                    selected_iou=np.empty(0, dtype=np.float32),
                    selected_inlier=np.empty(0, dtype=np.bool_),
                    missing_reason="no_existing_s02_samples",
                )
            )
            continue

        local_inlier = inlier[positions]
        local_overlap = overlap[positions]
        clean = local_inlier & (local_overlap < np.float32(_CLEAN_IOU_THRESHOLD))
        if np.any(clean):
            selected = positions[clean]
            policy = _B_CLEAN_INLIER
        elif np.any(local_inlier):
            selected = positions[local_inlier]
            policy = _B_ALL_INLIER
        else:
            selected = positions
            policy = _B_ALL_SAMPLES
        values, local_mask, selected_ids = _relaxed_prototypes(
            embeddings[selected], quality[selected], samples[selected]
        )
        prototypes[stable_id] = values.astype(np.float16)
        mask[stable_id] = local_mask
        centers[stable_id] = _normalized_mean(values[local_mask])
        rows.append(
            _base_row(
                stable_id=stable_id,
                grade=GRADE_B_EXISTING_DEGRADED,
                usable=True,
                selection_policy=policy,
                prototype_count=int(np.count_nonzero(local_mask)),
                input_sample_ids=input_ids,
                selected_sample_ids=selected_ids,
                selected_quality=quality[selected],
                selected_iou=overlap[selected],
                selected_inlier=inlier[selected],
                missing_reason=None,
            )
        )

    bundle = ForcedAppearanceBundle(
        stable_ids=_freeze_array(ids.astype(np.int64, copy=True)),
        prototypes=_freeze_array(prototypes),
        prototype_mask=_freeze_array(mask),
        descriptor_centers=_freeze_array(centers),
        rows=tuple(rows),
        zero_sample_stable_ids=tuple(missing),
    )
    _validate_bundle(bundle, require_total=False)
    return bundle


def complete_forced_appearance_with_supplemental(
    base: ForcedAppearanceBundle,
    supplemental: Sequence[SupplementalStableEmbeddings],
) -> ForcedAppearanceBundle:
    """Fill every explicit C gap and require total stable descriptor coverage."""

    _validate_bundle(base, require_total=False)
    if isinstance(supplemental, (str, bytes)):
        raise ContractError("forced appearance supplemental input must be a sequence")
    try:
        records = tuple(supplemental)
    except TypeError as exc:
        raise ContractError(
            "forced appearance supplemental input must be a sequence"
        ) from exc
    if any(not isinstance(item, SupplementalStableEmbeddings) for item in records):
        raise ContractError("forced appearance supplemental record has the wrong type")
    by_stable: dict[int, SupplementalStableEmbeddings] = {}
    global_sample_ids: set[int] = set()
    dimension = int(base.prototypes.shape[2])
    for item in records:
        if isinstance(item.stable_id, (bool, np.bool_)) or not isinstance(
            item.stable_id, (int, np.integer)
        ):
            raise ContractError("forced appearance supplemental stable_id must be an integer")
        stable_id = int(item.stable_id)
        if stable_id in by_stable:
            raise ContractError("forced appearance supplemental stable IDs are duplicated")
        embeddings = _unit_embeddings(
            item.embeddings,
            name="supplemental_embeddings",
            expected_dim=dimension,
            allow_empty=False,
        )
        sample_ids = _integer_vector(
            item.sample_ids,
            name="supplemental_sample_ids",
            length=len(embeddings),
        )
        if len(np.unique(sample_ids)) != len(sample_ids):
            raise ContractError("forced appearance supplemental sample IDs are duplicated")
        overlap = global_sample_ids.intersection(map(int, sample_ids))
        if overlap:
            raise ContractError(
                "forced appearance supplemental sample IDs must be globally unique"
            )
        global_sample_ids.update(map(int, sample_ids))
        _quality_vector(
            item.quality, name="supplemental_quality", length=len(embeddings)
        )
        by_stable[stable_id] = item

    expected = set(map(int, base.zero_sample_stable_ids))
    if set(by_stable) != expected:
        missing = sorted(expected - set(by_stable))
        extra = sorted(set(by_stable) - expected)
        raise ContractError(
            "forced appearance supplemental stable IDs must exactly cover C gaps: "
            f"missing={missing[:10]}, extra={extra[:10]}"
        )

    prototypes = np.array(base.prototypes, copy=True)
    mask = np.array(base.prototype_mask, copy=True)
    centers = np.array(base.descriptor_centers, copy=True)
    rows = list(base.rows)
    for stable_id in sorted(expected):
        item = by_stable[stable_id]
        embeddings = _unit_embeddings(
            item.embeddings,
            name="supplemental_embeddings",
            expected_dim=dimension,
            allow_empty=False,
        )
        sample_ids = _integer_vector(
            item.sample_ids,
            name="supplemental_sample_ids",
            length=len(embeddings),
        )
        quality = _quality_vector(
            item.quality, name="supplemental_quality", length=len(embeddings)
        )
        values, local_mask, selected_ids = _relaxed_prototypes(
            embeddings, quality, sample_ids
        )
        prototypes[stable_id] = values.astype(np.float16)
        mask[stable_id] = local_mask
        centers[stable_id] = _normalized_mean(values[local_mask])
        rows[stable_id] = _base_row(
            stable_id=stable_id,
            grade=GRADE_C_SUPPLEMENTAL,
            usable=True,
            selection_policy=_C_SUPPLEMENTAL,
            prototype_count=int(np.count_nonzero(local_mask)),
            input_sample_ids=sample_ids,
            selected_sample_ids=selected_ids,
            selected_quality=quality,
            selected_iou=np.zeros(len(embeddings), dtype=np.float32),
            selected_inlier=np.ones(len(embeddings), dtype=np.bool_),
            missing_reason=None,
        )

    result = ForcedAppearanceBundle(
        stable_ids=_freeze_array(np.array(base.stable_ids, copy=True)),
        prototypes=_freeze_array(prototypes),
        prototype_mask=_freeze_array(mask),
        descriptor_centers=_freeze_array(centers),
        rows=tuple(rows),
        zero_sample_stable_ids=(),
    )
    _validate_bundle(result, require_total=True)
    return result


def restore_completed_forced_appearance(
    *,
    stable_ids: np.ndarray,
    prototypes: np.ndarray,
    prototype_mask: np.ndarray,
    descriptor_centers: np.ndarray,
    rows: Sequence[Mapping[str, Any]],
) -> ForcedAppearanceBundle:
    """Restore and fully validate a persisted total appearance bundle.

    Descriptor centers are persisted separately because B/C centers are built
    from float32 prototypes before those prototypes are quantized to float16.
    Reconstructing them from the audit prototypes could therefore perturb a
    top-k tie or integer appearance cost on resume.
    """

    ids = _integer_vector(stable_ids, name="restored_stable_ids")
    values = np.asarray(prototypes)
    mask = np.asarray(prototype_mask)
    centers = np.asarray(descriptor_centers)
    if (
        values.dtype != np.float16
        or values.ndim != 3
        or values.shape[:2] != (len(ids), _PROTOTYPE_SLOTS)
        or values.shape[2] <= 0
        or mask.dtype != np.bool_
        or mask.shape != values.shape[:2]
        or centers.dtype != np.float32
        or centers.shape != (len(ids), values.shape[2])
    ):
        raise ContractError("forced appearance restored prototype tensors differ")
    try:
        supplied_rows = tuple(rows)
    except TypeError as exc:
        raise ContractError("forced appearance restored rows must be a sequence") from exc
    if len(supplied_rows) != len(ids) or any(
        not isinstance(row, Mapping) for row in supplied_rows
    ):
        raise ContractError("forced appearance restored row coverage differs")
    if not np.all(np.any(mask, axis=1)):
        raise ContractError("forced appearance restored bundle is not total")

    canonical_rows: list[Mapping[str, Any]] = []
    for row in supplied_rows:
        payload = dict(row)
        for key in ("input_sample_ids", "selected_sample_ids"):
            if key in payload:
                payload[key] = tuple(map(int, payload[key]))
        canonical_rows.append(MappingProxyType(payload))
    bundle = ForcedAppearanceBundle(
        stable_ids=_freeze_array(ids.astype(np.int64, copy=True)),
        prototypes=_freeze_array(np.array(values, copy=True)),
        prototype_mask=_freeze_array(np.array(mask, copy=True)),
        descriptor_centers=_freeze_array(np.array(centers, copy=True)),
        rows=tuple(canonical_rows),
        zero_sample_stable_ids=(),
    )
    _validate_bundle(bundle, require_total=True)
    return bundle


__all__ = [
    "ForcedAppearanceBundle",
    "GRADE_A_CLEAN",
    "GRADE_B_EXISTING_DEGRADED",
    "GRADE_C_SUPPLEMENTAL",
    "GRADE_MISSING",
    "SupplementalStableEmbeddings",
    "build_forced_appearance_base",
    "complete_forced_appearance_with_supplemental",
    "restore_completed_forced_appearance",
]
