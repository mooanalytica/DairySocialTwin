"""Strict, read-only runtime loader for the fixed finalized S04 output.

S05 consumes this module instead of reaching into the S04 stage's private
resume validator.  The loader validates the complete finalized artifact as a
single immutable snapshot: marker policy, canonical config hash, output
fingerprints, exact Arrow schemas, fixed production counts, mapping
bijections, and stable-appearance array alignment.
"""

from __future__ import annotations

import hashlib
import json
import math
import stat
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from cowtrack.config import ContractError
from cowtrack.linking.finalize_config import load_s04_finalize_config
from cowtrack.linking.runtime import FileFingerprint
from cowtrack.schemas.s04 import (
    DET_TO_STABLE_SCHEMA,
    MICRO_TO_STABLE_SCHEMA,
    STABLE_TRACKLETS_SCHEMA,
)
from cowtrack.schemas.stable_appearance import STABLE_APPEARANCE_SCHEMA


EXPECTED_PROTOTYPE_SLOTS = 3
EXPECTED_EMBEDDING_DIM = 1_536

LogFn = Callable[[str], None]

_OUTPUT_NAMES = (
    "det_to_stable.parquet",
    "effective_config.json",
    "finalize_report.json",
    "micro_to_stable.parquet",
    "stable_appearance.parquet",
    "stable_prototype_mask.npy",
    "stable_prototypes.f16.npy",
    "stable_tracklets.parquet",
)
_CONSUMED_NAMES = ("_SUCCESS.json", *_OUTPUT_NAMES)
_MARKER_KEYS = {
    "schema_version",
    "stage",
    "config_hash",
    "execution_mode",
    "operator_approved",
    "merge_policy",
    "automatic_merge_allowed",
    "num_automatic_merges",
    "num_operator_approved_merges",
    "read_review_labels",
    "solver_used",
    "input_fingerprints",
    "output_fingerprints",
    "stats",
    "elapsed_sec",
}
_REPORT_KEYS = {
    "schema_version",
    "stage",
    "execution_mode",
    "operator_approved",
    "merge_policy",
    "graph_node_identity",
    "automatic_merge_allowed",
    "num_automatic_merges",
    "num_operator_approved_merges",
    "read_review_labels",
    "solver_used",
    "counts",
    "component_size_distribution",
    "input_fingerprints",
}


@dataclass(frozen=True)
class StableTracklet:
    """One canonical finalized stable tracklet."""

    stable_id: int
    first_micro_id: int
    last_micro_id: int
    start_det_id: int
    end_det_id: int
    start_clip_id: str
    end_clip_id: str
    start_global_frame: int
    end_global_frame: int
    start_time_sec: float
    end_time_sec: float
    num_microtracklets: int
    num_detections: int
    num_proposal_edges: int
    min_proposal_probability: float | None
    mean_proposal_probability: float | None
    max_proposal_probability: float | None
    is_singleton: bool


@dataclass(frozen=True)
class StableAppearance:
    """Immutable S04 appearance provenance for one stable tracklet."""

    stable_id: int
    prototype_row: int
    constituent_micro_ids: tuple[int, ...]
    num_input_samples: int
    num_s02_inliers: int
    num_clean_candidates: int
    num_clean_inliers: int
    num_valid_prototypes: int
    appearance_usable: bool
    missing_reason: str | None
    clean_sample_ids: tuple[int, ...]
    clean_det_ids: tuple[int, ...]
    clean_embedding_rows: tuple[int, ...]
    medoid_sample_id: int | None
    appearance_quality: float | None
    internal_cosine_p10: float | None
    internal_cosine_p50: float | None
    internal_cosine_min: float | None
    num_overlap_rejected: int
    num_review_excluded: int
    num_local_outliers: int | None
    max_other_bbox_iou: float
    max_clean_other_bbox_iou: float | None


@dataclass(frozen=True)
class S04FinalizedBundle:
    """Public immutable view of the exact operator-approved S04 snapshot."""

    directory: Path
    stable_ids: np.ndarray
    micro_ids: np.ndarray
    micro_to_stable: Mapping[int, int]
    micro_order_in_stable: Mapping[int, int]
    stable_to_micros: Mapping[int, tuple[int, ...]]
    stable_tracklets: Mapping[int, StableTracklet]
    stable_appearance: Mapping[int, StableAppearance]
    det_ids: np.ndarray
    det_micro_ids: np.ndarray
    det_stable_ids: np.ndarray
    det_order_in_micro: np.ndarray
    det_order_in_stable: np.ndarray
    det_order_in_stable_detection: np.ndarray
    stable_prototypes: np.ndarray
    stable_prototype_mask: np.ndarray
    effective_config: Mapping[str, Any]
    report: Mapping[str, Any]
    success_marker: Mapping[str, Any]
    consumed_paths: tuple[Path, ...]
    input_fingerprints: tuple[FileFingerprint, ...]


def _read_json(path: Path, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read {label} {path}: {exc}") from exc


def _fingerprint_file(path: Path) -> FileFingerprint:
    resolved = path.resolve()
    if not resolved.is_file():
        raise ContractError(f"required S04 finalized artifact does not exist: {resolved}")
    digest = hashlib.sha256()
    try:
        with resolved.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
        size = int(resolved.stat().st_size)
    except OSError as exc:
        raise ContractError(f"cannot fingerprint S04 finalized artifact {resolved}: {exc}") from exc
    return FileFingerprint(str(resolved), size, digest.hexdigest())


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


def _column(table: pa.Table, name: str, dtype: Any) -> np.ndarray:
    values = table[name].combine_chunks().to_numpy(zero_copy_only=False)
    return np.asarray(values, dtype=dtype)


def _readonly(values: Any, dtype: Any) -> np.ndarray:
    result = np.array(values, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def _freeze_json(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({str(key): _freeze_json(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _fingerprint_records(value: Any, label: str) -> tuple[tuple[str, int, str], ...]:
    if not isinstance(value, list):
        raise ContractError(f"{label} must be a list")
    records: list[tuple[str, int, str]] = []
    for item in value:
        if not isinstance(item, dict) or set(item) != {"path", "size_bytes", "sha256"}:
            raise ContractError(f"{label} contains an invalid fingerprint record")
        path, size, sha256 = item["path"], item["size_bytes"], item["sha256"]
        if (
            not isinstance(path, str)
            or not path
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
            or not isinstance(sha256, str)
            or len(sha256) != 64
            or any(character not in "0123456789abcdef" for character in sha256)
        ):
            raise ContractError(f"{label} contains invalid fingerprint fields")
        records.append((path, size, sha256))
    if len({path for path, _, _ in records}) != len(records):
        raise ContractError(f"{label} contains duplicate paths")
    return tuple(records)


def _recorded_input_snapshot(
    value: Any,
    label: str,
    *,
    path_replacements: Mapping[str, Path] | None = None,
    logger: LogFn = lambda _message: None,
    phase: str = "",
) -> tuple[FileFingerprint, ...]:
    """Recompute every absolute regular file recorded by an S04 marker."""

    records = _fingerprint_records(value, label)
    if not records:
        raise ContractError(f"{label} cannot be empty")
    if path_replacements is not None and set(path_replacements) != {
        raw_path for raw_path, _, _ in records
    }:
        raise ContractError(f"{label} relocation map differs from recorded paths")
    snapshot: list[FileFingerprint] = []
    for index, (raw_path, expected_size, expected_sha256) in enumerate(
        records, start=1
    ):
        recorded_path = Path(raw_path)
        if not recorded_path.is_absolute():
            raise ContractError(f"{label} path must be absolute: {raw_path}")
        path = (
            Path(path_replacements[raw_path])
            if path_replacements is not None
            else recorded_path
        )
        phase_label = f" {phase}" if phase else ""
        logger(
            f"[s04-runtime] fingerprint recorded input{phase_label} "
            f"{index:,}/{len(records):,}: {path.name}"
        )
        try:
            resolved = path.resolve(strict=True)
            mode = path.stat().st_mode
        except OSError as exc:
            raise ContractError(
                f"cannot resolve recorded S04 input {path}: {exc}"
            ) from exc
        if resolved != path or not stat.S_ISREG(mode):
            raise ContractError(
                f"{label} path must be a canonical absolute regular file: {path}"
            )
        current = _fingerprint_file(path)
        if (
            current.size_bytes != expected_size
            or current.sha256 != expected_sha256
        ):
            raise ContractError(f"recorded S04 input changed: {path}")
        snapshot.append(current)
    return tuple(snapshot)


def _verify_output_fingerprints(
    marker: Mapping[str, Any], before: Sequence[FileFingerprint]
) -> None:
    records = _fingerprint_records(marker.get("output_fingerprints"), "S04 output_fingerprints")
    if tuple(path for path, _, _ in records) != _OUTPUT_NAMES:
        raise ContractError("S04 finalized output fingerprint set/order differs")
    actual = {Path(item.path).name: item for item in before}
    for name, size, sha256 in records:
        current = actual.get(name)
        if current is None or current.size_bytes != size or current.sha256 != sha256:
            raise ContractError(f"completed S04 finalized artifact changed: {name}")


def _validate_policy(marker: Mapping[str, Any]) -> None:
    if set(marker) != _MARKER_KEYS:
        raise ContractError("S04 finalized success marker keys differ")
    expected = {
        "schema_version": "1.0",
        "stage": "S04_FINALIZE",
        "execution_mode": "operator_approved_component_union",
        "operator_approved": True,
        "merge_policy": "all_provisional_undirected_components",
        "automatic_merge_allowed": False,
        "num_automatic_merges": 0,
        "read_review_labels": False,
        "solver_used": False,
    }
    if any(marker.get(key) != value or type(marker.get(key)) is not type(value) for key, value in expected.items()):
        raise ContractError("S04 finalized marker policy/config differs")
    config_hash = marker.get("config_hash")
    approved = marker.get("num_operator_approved_merges")
    if (
        not isinstance(config_hash, str)
        or len(config_hash) != 64
        or any(character not in "0123456789abcdef" for character in config_hash)
        or isinstance(approved, bool)
        or not isinstance(approved, int)
        or approved < 0
    ):
        raise ContractError("S04 finalized marker dynamic fields are invalid")
    elapsed = marker.get("elapsed_sec")
    if (
        isinstance(elapsed, bool)
        or not isinstance(elapsed, (int, float))
        or not math.isfinite(float(elapsed))
        or float(elapsed) < 0.0
    ):
        raise ContractError("S04 finalized marker elapsed_sec is invalid")


def _optional_probability(value: Any, label: str) -> float | None:
    if value is None:
        return None
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ContractError(f"S04 finalized {label} is invalid")
    return result


def _build_mappings(
    mapping_table: pa.Table,
) -> tuple[
    np.ndarray,
    dict[int, int],
    dict[int, int],
    dict[int, tuple[int, ...]],
    dict[int, Mapping[str, Any]],
    int,
]:
    micro_count = mapping_table.num_rows
    if micro_count < 1:
        raise ContractError("S04 finalized mapping is empty")
    rows = mapping_table.to_pylist()
    micro_ids = _column(mapping_table, "micro_id", np.int64)
    if not np.array_equal(micro_ids, np.arange(micro_count, dtype=np.int64)):
        raise ContractError("S04 finalized micro IDs are not dense/canonical")
    stable_values = _column(mapping_table, "stable_id", np.int64)
    stable_count = int(np.max(stable_values)) + 1
    if (
        np.any(stable_values < 0)
        or set(map(int, stable_values)) != set(range(stable_count))
    ):
        raise ContractError("S04 finalized micro-to-stable mapping is not total/dense")
    micro_to_stable = dict(zip(map(int, micro_ids), map(int, stable_values), strict=True))
    micro_order = {
        int(row["micro_id"]): int(row["order_in_stable"]) for row in rows
    }
    grouped: dict[int, list[dict[str, Any]]] = {
        stable_id: [] for stable_id in range(stable_count)
    }
    for row in rows:
        grouped[int(row["stable_id"])].append(row)
    stable_to_micros: dict[int, tuple[int, ...]] = {}
    component_rows: dict[int, Mapping[str, Any]] = {}
    for stable_id, members in grouped.items():
        members.sort(key=lambda row: int(row["order_in_stable"]))
        count = len(members)
        if not count or [int(row["order_in_stable"]) for row in members] != list(range(count)):
            raise ContractError("S04 finalized order_in_stable is not contiguous")
        micros = tuple(int(row["micro_id"]) for row in members)
        stable_to_micros[stable_id] = micros
        reference = members[0]
        component_rows[stable_id] = reference
        edge_count = int(reference["component_num_proposal_edges"])
        if int(reference["component_num_microtracklets"]) != count or edge_count < 0:
            raise ContractError("S04 finalized component counts are inconsistent")
        probability_names = (
            "component_min_proposal_probability",
            "component_mean_proposal_probability",
            "component_max_proposal_probability",
        )
        probabilities = tuple(
            _optional_probability(reference[name], name) for name in probability_names
        )
        if (edge_count == 0) != all(value is None for value in probabilities):
            raise ContractError("S04 finalized component probability nullability differs")
        if edge_count and (any(value is None for value in probabilities) or not probabilities[0] <= probabilities[1] <= probabilities[2]):
            raise ContractError("S04 finalized component probability ordering differs")
        for order, row in enumerate(members):
            if (
                int(row["component_num_microtracklets"]) != count
                or int(row["component_num_proposal_edges"]) != edge_count
                or tuple(row[name] for name in probability_names)
                != tuple(reference[name] for name in probability_names)
            ):
                raise ContractError("S04 finalized component metadata differs between micros")
            predecessor = row["predecessor_micro_id"]
            edge_id = row["predecessor_edge_id"]
            edge_probability = row["predecessor_link_probability"]
            if order == 0:
                if predecessor is not None or edge_id is not None or edge_probability is not None:
                    raise ContractError("S04 finalized first micro has predecessor evidence")
            else:
                if predecessor is None or int(predecessor) != micros[order - 1]:
                    raise ContractError("S04 finalized predecessor chain differs")
                if (edge_id is None) != (edge_probability is None):
                    raise ContractError("S04 finalized predecessor edge evidence differs")
                if edge_id is not None:
                    if not isinstance(edge_id, str) or not edge_id:
                        raise ContractError("S04 finalized predecessor edge ID is invalid")
                    _optional_probability(edge_probability, "predecessor_link_probability")
    return (
        micro_ids,
        micro_to_stable,
        micro_order,
        stable_to_micros,
        component_rows,
        stable_count,
    )


def _build_stable_tracklets(
    table: pa.Table,
    stable_to_micros: Mapping[int, tuple[int, ...]],
    component_rows: Mapping[int, Mapping[str, Any]],
) -> dict[int, StableTracklet]:
    stable_count = len(stable_to_micros)
    if table.num_rows != stable_count:
        raise ContractError("S04 finalized stable-tracklet count differs")
    rows = table.to_pylist()
    if [int(row["stable_id"]) for row in rows] != list(range(stable_count)):
        raise ContractError("S04 finalized stable IDs are not dense/canonical")
    result: dict[int, StableTracklet] = {}
    for row in rows:
        stable_id = int(row["stable_id"])
        micros = stable_to_micros[stable_id]
        component = component_rows[stable_id]
        count = len(micros)
        edge_count = int(row["num_proposal_edges"])
        probability_values = tuple(
            _optional_probability(row[name], name)
            for name in (
                "min_proposal_probability",
                "mean_proposal_probability",
                "max_proposal_probability",
            )
        )
        component_probability_values = tuple(
            _optional_probability(component[name], name)
            for name in (
                "component_min_proposal_probability",
                "component_mean_proposal_probability",
                "component_max_proposal_probability",
            )
        )
        start_time, end_time = float(row["start_time_sec"]), float(row["end_time_sec"])
        if (
            int(row["first_micro_id"]) != micros[0]
            or int(row["last_micro_id"]) != micros[-1]
            or int(row["num_microtracklets"]) != count
            or edge_count != int(component["component_num_proposal_edges"])
            or probability_values != component_probability_values
            or bool(row["is_singleton"]) != (count == 1)
            or int(row["num_detections"]) < count
            or not str(row["start_clip_id"])
            or not str(row["end_clip_id"])
            or int(row["start_global_frame"]) < 0
            or int(row["end_global_frame"]) < int(row["start_global_frame"])
            or not math.isfinite(start_time)
            or not math.isfinite(end_time)
            or end_time < start_time
        ):
            raise ContractError("S04 finalized stable-tracklet summary differs")
        result[stable_id] = StableTracklet(
            stable_id=stable_id,
            first_micro_id=int(row["first_micro_id"]),
            last_micro_id=int(row["last_micro_id"]),
            start_det_id=int(row["start_det_id"]),
            end_det_id=int(row["end_det_id"]),
            start_clip_id=str(row["start_clip_id"]),
            end_clip_id=str(row["end_clip_id"]),
            start_global_frame=int(row["start_global_frame"]),
            end_global_frame=int(row["end_global_frame"]),
            start_time_sec=start_time,
            end_time_sec=end_time,
            num_microtracklets=count,
            num_detections=int(row["num_detections"]),
            num_proposal_edges=edge_count,
            min_proposal_probability=probability_values[0],
            mean_proposal_probability=probability_values[1],
            max_proposal_probability=probability_values[2],
            is_singleton=bool(row["is_singleton"]),
        )
    return result


def _build_detection_mapping(
    table: pa.Table,
    micro_to_stable: Mapping[int, int],
    micro_order: Mapping[int, int],
    stable_to_micros: Mapping[int, tuple[int, ...]],
    stable_tracklets: Mapping[int, StableTracklet],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if table.num_rows < 1:
        raise ContractError("S04 finalized detection mapping is empty")
    micro_count = len(micro_to_stable)
    stable_count = len(stable_to_micros)
    det_ids = _column(table, "det_id", np.int64)
    micro_ids = _column(table, "micro_id", np.int64)
    stable_ids = _column(table, "stable_id", np.int64)
    order_in_stable = _column(table, "order_in_stable", np.int64)
    order_in_micro = _column(table, "order_in_micro", np.int64)
    stable_detection_order = _column(table, "order_in_stable_detection", np.int64)
    # det_id is deliberately signed int64; negative IDs are valid and preserved.
    if len(np.unique(det_ids)) != len(det_ids) or (
        len(det_ids) > 1 and np.any(det_ids[1:] <= det_ids[:-1])
    ):
        raise ContractError("S04 finalized signed det_id mapping is not unique/canonical")
    if np.any(micro_ids < 0) or np.any(micro_ids >= micro_count):
        raise ContractError("S04 finalized detections do not cover every micro exactly")
    micro_counts = np.bincount(micro_ids, minlength=micro_count)
    if len(micro_counts) != micro_count or np.any(micro_counts == 0):
        raise ContractError("S04 finalized detections do not cover every micro exactly")
    stable_lookup = np.asarray(
        [micro_to_stable[micro_id] for micro_id in range(micro_count)],
        dtype=np.int64,
    )
    micro_order_lookup = np.asarray(
        [micro_order[micro_id] for micro_id in range(micro_count)],
        dtype=np.int64,
    )
    expected_stable = stable_lookup[micro_ids]
    expected_micro_order = micro_order_lookup[micro_ids]
    if not np.array_equal(stable_ids, expected_stable) or not np.array_equal(order_in_stable, expected_micro_order):
        raise ContractError("S04 finalized det-to-micro-to-stable join differs")
    by_micro: dict[int, np.ndarray] = {}
    micro_sort = np.lexsort((det_ids, order_in_micro, micro_ids))
    micro_offsets = np.r_[0, np.cumsum(micro_counts, dtype=np.int64)]
    for micro_id in range(micro_count):
        positions = micro_sort[micro_offsets[micro_id] : micro_offsets[micro_id + 1]]
        if not np.array_equal(order_in_micro[positions], np.arange(len(positions), dtype=np.int64)):
            raise ContractError("S04 finalized order_in_micro is not contiguous")
        by_micro[micro_id] = positions
    stable_counts = np.bincount(stable_ids, minlength=stable_count)
    if len(stable_counts) != stable_count or np.any(stable_counts == 0):
        raise ContractError("S04 finalized detections do not cover every stable ID")
    stable_sort = np.lexsort((stable_detection_order, stable_ids))
    stable_offsets = np.r_[0, np.cumsum(stable_counts, dtype=np.int64)]
    for stable_id, micros in stable_to_micros.items():
        positions = stable_sort[
            stable_offsets[stable_id] : stable_offsets[stable_id + 1]
        ]
        if not np.array_equal(stable_detection_order[positions], np.arange(len(positions), dtype=np.int64)):
            raise ContractError("S04 finalized stable detection order is not contiguous")
        expected_positions = np.concatenate([by_micro[micro_id] for micro_id in micros])
        if not np.array_equal(positions, expected_positions):
            raise ContractError("S04 finalized stable/micro detection chronology differs")
        tracklet = stable_tracklets[stable_id]
        if (
            len(positions) != tracklet.num_detections
            or int(det_ids[positions[0]]) != tracklet.start_det_id
            or int(det_ids[positions[-1]]) != tracklet.end_det_id
        ):
            raise ContractError("S04 finalized stable detection endpoints/count differ")
    return det_ids, micro_ids, stable_ids, order_in_micro, order_in_stable, stable_detection_order


def _optional_finite(value: Any, label: str) -> float | None:
    if value is None:
        return None
    result = float(value)
    if not math.isfinite(result):
        raise ContractError(f"S04 finalized stable appearance {label} is non-finite")
    return result


def _build_stable_appearance(
    table: pa.Table,
    stable_to_micros: Mapping[int, tuple[int, ...]],
    prototypes: np.ndarray,
    mask: np.ndarray,
) -> dict[int, StableAppearance]:
    stable_count = len(stable_to_micros)
    if table.num_rows != stable_count:
        raise ContractError("S04 finalized stable-appearance count differs")
    rows = table.to_pylist()
    expected_ids = list(range(stable_count))
    if (
        [int(row["stable_id"]) for row in rows] != expected_ids
        or [int(row["prototype_row"]) for row in rows] != expected_ids
    ):
        raise ContractError("S04 finalized stable appearance IDs/rows differ")
    if (
        prototypes.dtype != np.float16
        or prototypes.shape
        != (stable_count, EXPECTED_PROTOTYPE_SLOTS, EXPECTED_EMBEDDING_DIM)
        or not np.isfinite(prototypes).all()
        or mask.dtype != np.bool_
        or mask.shape != (stable_count, EXPECTED_PROTOTYPE_SLOTS)
        or np.any(prototypes[~mask] != np.float16(0.0))
    ):
        raise ContractError("S04 finalized stable appearance array contract differs")
    if np.any(mask):
        norms = np.linalg.norm(prototypes.astype(np.float32), axis=2)[mask]
        if not np.allclose(norms, 1.0, rtol=0.0, atol=2e-3):
            raise ContractError("S04 finalized stable prototypes are not L2-normalized")
    result: dict[int, StableAppearance] = {}
    usable_count = 0
    for row in rows:
        stable_id = int(row["stable_id"])
        constituents = tuple(map(int, row["constituent_micro_ids"]))
        clean_sample_ids = tuple(map(int, row["clean_sample_ids"]))
        clean_det_ids = tuple(map(int, row["clean_det_ids"]))
        clean_embedding_rows = tuple(map(int, row["clean_embedding_rows"]))
        usable = bool(row["appearance_usable"])
        valid_count = int(row["num_valid_prototypes"])
        clean_count = int(row["num_clean_inliers"])
        input_count = int(row["num_input_samples"])
        clean_candidate_count = int(row["num_clean_candidates"])
        expected_mask = np.arange(EXPECTED_PROTOTYPE_SLOTS) < valid_count
        if (
            constituents != tuple(sorted(stable_to_micros[stable_id]))
            or valid_count != int(np.count_nonzero(mask[stable_id]))
            or not np.array_equal(mask[stable_id], expected_mask)
            or usable != bool(np.any(mask[stable_id]))
            or len(clean_sample_ids) != clean_count
            or len(clean_det_ids) != clean_count
            or len(clean_embedding_rows) != clean_count
            or len(set(clean_sample_ids)) != clean_count
            or len(set(clean_det_ids)) != clean_count
            or len(set(clean_embedding_rows)) != clean_count
            or any(int(row[name]) < 0 for name in (
                "num_input_samples", "num_s02_inliers", "num_clean_candidates",
                "num_clean_inliers", "num_overlap_rejected", "num_review_excluded",
            ))
            or not 0 <= int(row["num_s02_inliers"]) <= input_count
            or not 0 <= clean_count <= clean_candidate_count <= input_count
            or not 0 <= int(row["num_overlap_rejected"]) <= input_count
            or not 0 <= int(row["num_review_excluded"]) <= input_count
            or valid_count < 0
            or valid_count > EXPECTED_PROTOTYPE_SLOTS
            or (usable and row["missing_reason"] is not None)
            or (not usable and (not isinstance(row["missing_reason"], str) or not row["missing_reason"]))
        ):
            raise ContractError("S04 finalized stable appearance row contract differs")
        local_outliers = row["num_local_outliers"]
        if local_outliers is not None and int(local_outliers) < 0:
            raise ContractError("S04 finalized stable appearance outlier count is invalid")
        max_iou = _optional_finite(row["max_other_bbox_iou"], "max_other_bbox_iou")
        assert max_iou is not None
        optional_values = {
            name: _optional_finite(row[name], name)
            for name in (
                "appearance_quality",
                "internal_cosine_p10",
                "internal_cosine_p50",
                "internal_cosine_min",
                "max_clean_other_bbox_iou",
            )
        }
        if not 0.0 <= max_iou <= 1.0 or any(
            value is not None and not -1.0 <= value <= 1.0
            for name, value in optional_values.items()
            if name.startswith("internal_cosine")
        ):
            raise ContractError("S04 finalized stable appearance numeric range differs")
        quality = optional_values["appearance_quality"]
        clean_iou = optional_values["max_clean_other_bbox_iou"]
        if (quality is not None and not 0.0 <= quality <= 1.0) or (
            clean_iou is not None and not 0.0 <= clean_iou <= 1.0
        ):
            raise ContractError("S04 finalized stable appearance quality/IoU differs")
        medoid = row["medoid_sample_id"]
        gallery_values = (
            quality,
            optional_values["internal_cosine_p10"],
            optional_values["internal_cosine_p50"],
            optional_values["internal_cosine_min"],
            clean_iou,
        )
        if clean_count == 0:
            if medoid is not None or local_outliers is not None or any(
                value is not None for value in gallery_values
            ):
                raise ContractError("S04 finalized missing gallery provenance differs")
        elif (
            medoid is None
            or int(medoid) not in clean_sample_ids
            or local_outliers is None
            or any(value is None for value in gallery_values)
        ):
            raise ContractError("S04 finalized present gallery provenance differs")
        usable_count += int(usable)
        result[stable_id] = StableAppearance(
            stable_id=stable_id,
            prototype_row=int(row["prototype_row"]),
            constituent_micro_ids=constituents,
            num_input_samples=int(row["num_input_samples"]),
            num_s02_inliers=int(row["num_s02_inliers"]),
            num_clean_candidates=int(row["num_clean_candidates"]),
            num_clean_inliers=clean_count,
            num_valid_prototypes=valid_count,
            appearance_usable=usable,
            missing_reason=row["missing_reason"],
            clean_sample_ids=clean_sample_ids,
            clean_det_ids=clean_det_ids,
            clean_embedding_rows=clean_embedding_rows,
            medoid_sample_id=(None if medoid is None else int(medoid)),
            appearance_quality=quality,
            internal_cosine_p10=optional_values["internal_cosine_p10"],
            internal_cosine_p50=optional_values["internal_cosine_p50"],
            internal_cosine_min=optional_values["internal_cosine_min"],
            num_overlap_rejected=int(row["num_overlap_rejected"]),
            num_review_excluded=int(row["num_review_excluded"]),
            num_local_outliers=(None if local_outliers is None else int(local_outliers)),
            max_other_bbox_iou=max_iou,
            max_clean_other_bbox_iou=clean_iou,
        )
    return result


def _computed_stats(
    stable_tracklets: Mapping[int, StableTracklet],
    stable_appearance: Mapping[int, StableAppearance],
    *,
    micro_count: int,
    detection_count: int,
) -> dict[str, int]:
    sizes = [row.num_microtracklets for row in stable_tracklets.values()]
    return {
        "microtracklets": micro_count,
        "detections": detection_count,
        "proposals_consumed": sum(row.num_proposal_edges for row in stable_tracklets.values()),
        "proposal_connected_microtracklets": sum(
            row.num_microtracklets for row in stable_tracklets.values() if row.num_proposal_edges > 0
        ),
        "stable_tracklets": len(stable_tracklets),
        "non_singleton_components": sum(size > 1 for size in sizes),
        "singleton_components": sum(size == 1 for size in sizes),
        "max_component_size": max(sizes, default=0),
        "stable_appearance_usable": sum(row.appearance_usable for row in stable_appearance.values()),
        "stable_appearance_missing": sum(not row.appearance_usable for row in stable_appearance.values()),
    }


def _validate_report_and_marker(
    report: Mapping[str, Any],
    marker: Mapping[str, Any],
    stats: Mapping[str, int],
    stable_tracklets: Mapping[int, StableTracklet],
) -> None:
    if set(report) != _REPORT_KEYS:
        raise ContractError("S04 finalized report keys differ")
    policy = {
        "schema_version": "1.0",
        "stage": "S04_FINALIZE",
        "execution_mode": "operator_approved_component_union",
        "operator_approved": True,
        "merge_policy": "all_provisional_undirected_components",
        "graph_node_identity": "actual_micro_id",
        "automatic_merge_allowed": False,
        "num_automatic_merges": 0,
        "num_operator_approved_merges": int(marker["num_operator_approved_merges"]),
        "read_review_labels": False,
        "solver_used": False,
    }
    if any(report.get(key) != value or type(report.get(key)) is not type(value) for key, value in policy.items()):
        raise ContractError("S04 finalized report policy differs")
    if marker.get("stats") != dict(stats) or report.get("counts") != dict(stats):
        raise ContractError("S04 finalized marker/report statistics differ")
    distribution = dict(
        sorted(
            (str(size), count)
            for size, count in Counter(
                row.num_microtracklets for row in stable_tracklets.values()
            ).items()
        )
    )
    if report.get("component_size_distribution") != distribution:
        raise ContractError("S04 finalized component-size distribution differs")
    marker_inputs = _fingerprint_records(marker.get("input_fingerprints"), "S04 marker input_fingerprints")
    report_inputs = _fingerprint_records(report.get("input_fingerprints"), "S04 report input_fingerprints")
    if marker_inputs != report_inputs:
        raise ContractError("S04 finalized report/marker input fingerprints differ")


def load_s04_finalized(
    output_dir: Path,
    *,
    recorded_input_relocations: Mapping[str, Path] | None = None,
    source_config_path: Path | None = None,
    logger: LogFn = lambda _message: None,
) -> S04FinalizedBundle:
    """Load and fully validate the fixed finalized S04 snapshot without writes."""

    directory = output_dir.resolve()
    if not directory.is_dir():
        raise ContractError(f"S04 finalized directory does not exist: {directory}")
    actual_names = {path.name for path in directory.iterdir() if path.is_file()}
    if actual_names != set(_CONSUMED_NAMES) or any(path.is_dir() for path in directory.iterdir()):
        raise ContractError("S04 finalized artifact tree differs")
    consumed_paths = tuple(directory / name for name in _CONSUMED_NAMES)
    before_list: list[FileFingerprint] = []
    for index, path in enumerate(consumed_paths, start=1):
        logger(
            f"[s04-runtime] fingerprint finalized output pass 1/2 "
            f"{index:,}/{len(consumed_paths):,}: {path.name}"
        )
        before_list.append(_fingerprint_file(path))
    before = tuple(before_list)

    marker = _read_json(directory / "_SUCCESS.json", "S04 finalized success marker")
    if not isinstance(marker, dict):
        raise ContractError("S04 finalized success marker must be an object")
    _validate_policy(marker)
    _verify_output_fingerprints(marker, before)
    recorded_inputs_before = _recorded_input_snapshot(
        marker.get("input_fingerprints"),
        "S04 marker input_fingerprints",
        path_replacements=recorded_input_relocations,
        logger=logger,
        phase="pass 1/2",
    )

    config, config_payload, config_hash = load_s04_finalize_config(
        directory / "effective_config.json"
    )
    if source_config_path is not None:
        source_config, source_payload, source_hash = load_s04_finalize_config(
            source_config_path
        )
        if (
            source_hash != marker["config_hash"]
            or source_config != config
            or source_payload != config_payload
        ):
            raise ContractError(
                "S04 finalized source/effective config provenance differs"
            )
    if (
        marker["config_hash"] != config_hash
        or tuple(sorted(vars(config.artifacts).values()))
        != tuple(sorted((*_OUTPUT_NAMES, "_SUCCESS.json")))
    ):
        raise ContractError("S04 finalized effective config/hash differs")

    logger("[s04-runtime] loading finalized mapping and appearance artifacts")
    mapping_table = _read_parquet(
        directory / config.artifacts.micro_to_stable,
        MICRO_TO_STABLE_SCHEMA,
        "S04 finalized micro-to-stable",
    )
    stable_table = _read_parquet(
        directory / config.artifacts.stable_tracklets,
        STABLE_TRACKLETS_SCHEMA,
        "S04 finalized stable tracklets",
    )
    detection_table = _read_parquet(
        directory / config.artifacts.det_to_stable,
        DET_TO_STABLE_SCHEMA,
        "S04 finalized det-to-stable",
    )
    appearance_table = _read_parquet(
        directory / config.artifacts.stable_appearance,
        STABLE_APPEARANCE_SCHEMA,
        "S04 finalized stable appearance",
    )
    try:
        prototypes = np.load(
            directory / config.artifacts.stable_prototypes, allow_pickle=False
        )
        prototype_mask = np.load(
            directory / config.artifacts.stable_prototype_mask, allow_pickle=False
        )
    except (OSError, ValueError) as exc:
        raise ContractError(f"cannot load S04 finalized appearance arrays: {exc}") from exc

    (
        micro_ids,
        micro_to_stable,
        micro_order,
        stable_to_micros,
        component_rows,
        stable_count,
    ) = _build_mappings(mapping_table)
    stable_tracklets = _build_stable_tracklets(
        stable_table, stable_to_micros, component_rows
    )
    detection_values = _build_detection_mapping(
        detection_table,
        micro_to_stable,
        micro_order,
        stable_to_micros,
        stable_tracklets,
    )
    stable_appearance = _build_stable_appearance(
        appearance_table, stable_to_micros, prototypes, prototype_mask
    )
    report = _read_json(directory / config.artifacts.report, "S04 finalized report")
    if not isinstance(report, dict):
        raise ContractError("S04 finalized report must be an object")
    stats = _computed_stats(
        stable_tracklets,
        stable_appearance,
        micro_count=len(micro_ids),
        detection_count=len(detection_values[0]),
    )
    if len(micro_ids) - stable_count != marker["num_operator_approved_merges"]:
        raise ContractError("S04 operator-approved merge count is inconsistent")
    _validate_report_and_marker(report, marker, stats, stable_tracklets)

    after_list: list[FileFingerprint] = []
    for index, path in enumerate(consumed_paths, start=1):
        logger(
            f"[s04-runtime] fingerprint finalized output pass 2/2 "
            f"{index:,}/{len(consumed_paths):,}: {path.name}"
        )
        after_list.append(_fingerprint_file(path))
    after = tuple(after_list)
    if before != after:
        raise ContractError("S04 finalized artifacts changed while being loaded")
    recorded_inputs_after = _recorded_input_snapshot(
        marker.get("input_fingerprints"),
        "S04 marker input_fingerprints",
        path_replacements=recorded_input_relocations,
        logger=logger,
        phase="pass 2/2",
    )
    if recorded_inputs_before != recorded_inputs_after:
        raise ContractError("recorded S04 inputs changed while being loaded")

    readonly_prototypes = _readonly(prototypes, np.float16)
    readonly_mask = _readonly(prototype_mask, np.bool_)
    readonly_detections = tuple(
        _readonly(values, np.int64) for values in detection_values
    )
    return S04FinalizedBundle(
        directory=directory,
        stable_ids=_readonly(np.arange(stable_count), np.int64),
        micro_ids=_readonly(micro_ids, np.int64),
        micro_to_stable=MappingProxyType(dict(micro_to_stable)),
        micro_order_in_stable=MappingProxyType(dict(micro_order)),
        stable_to_micros=MappingProxyType(dict(stable_to_micros)),
        stable_tracklets=MappingProxyType(dict(stable_tracklets)),
        stable_appearance=MappingProxyType(dict(stable_appearance)),
        det_ids=readonly_detections[0],
        det_micro_ids=readonly_detections[1],
        det_stable_ids=readonly_detections[2],
        det_order_in_micro=readonly_detections[3],
        det_order_in_stable=readonly_detections[4],
        det_order_in_stable_detection=readonly_detections[5],
        stable_prototypes=readonly_prototypes,
        stable_prototype_mask=readonly_mask,
        effective_config=_freeze_json(config_payload),
        report=_freeze_json(report),
        success_marker=_freeze_json(marker),
        consumed_paths=consumed_paths,
        input_fingerprints=before,
    )


__all__ = [
    "EXPECTED_EMBEDDING_DIM",
    "EXPECTED_PROTOTYPE_SLOTS",
    "S04FinalizedBundle",
    "StableAppearance",
    "StableTracklet",
    "load_s04_finalized",
]
