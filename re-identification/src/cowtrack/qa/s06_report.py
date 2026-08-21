"""Pure, deterministic data products for the S06 QA/export stage.

The functions in this module deliberately do no file or video I/O.  They turn
the already validated S00/S01/S05-forced Arrow artifacts into the CSV
tables and independently recompute the small set of structural invariants
that S06 must not trust from an upstream report.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from cowtrack.config import ContractError
from cowtrack.qa.s06_config import APPEARANCE_GRADES, S06ExportConfig
from cowtrack.schemas.detections import (
    BBoxQAFlag,
    DETECTIONS_SCHEMA,
    INVALID_BBOX_MASK,
)
from cowtrack.schemas.s05_finalize import (
    DET_TO_GLOBAL_SCHEMA,
    GLOBAL_TRACKS_SCHEMA,
    STABLE_TO_GLOBAL_SCHEMA,
)
from cowtrack.schemas.s05_forced import (
    FORCED_CANDIDATE_EDGES_SCHEMA,
    GRADED_STABLE_APPEARANCE_SCHEMA,
)
from cowtrack.schemas.s06 import (
    DETECTIONS_WITH_GLOBAL_ID_SCHEMA,
    GLOBAL_TRACK_SUMMARY_SCHEMA,
)
from cowtrack.schemas.tracklets import MICROTRACKLETS_SCHEMA


_AUTHORIZATION_BASIS = "operator_forced_appearance_exact_62"
_ID_STATUS = "forced_provisional"
_PROBABILITY_REASON = "forced_appearance_cosine_is_not_a_calibrated_probability"
_IDENTITY_FIELDS = (
    "global_track_id",
    "global_track_uuid",
    "display_global_id",
    "id_status",
    "identity_basis",
    "micro_id",
    "stable_id",
    "order_in_micro",
    "order_in_stable",
    "order_in_stable_detection",
    "order_in_global_stable",
    "order_in_global_detection",
    "local_purity_score",
    "assignment_confidence",
    "incoming_link_probability",
    "outgoing_link_probability",
)
_MAPPED_FIELDS = (
    "global_track_id",
    "global_track_uuid",
    "display_global_id",
    "id_status",
    "identity_basis",
    "micro_id",
    "stable_id",
    "order_in_micro",
    "order_in_stable",
    "order_in_stable_detection",
    "order_in_global_stable",
    "order_in_global_detection",
    "local_purity_score",
)


def _require_exact_schema(table: pa.Table, schema: pa.Schema, label: str) -> pa.Table:
    if not isinstance(table, pa.Table):
        raise ContractError(f"S06 {label} must be an Arrow table")
    if not table.schema.equals(schema, check_metadata=False):
        raise ContractError(
            f"S06 {label} schema differs; expected={schema}, actual={table.schema}"
        )
    for field in schema:
        if not field.nullable and table[field.name].null_count:
            raise ContractError(
                f"S06 {label}.{field.name} contains nulls in a non-nullable field"
            )
    return table.combine_chunks()


def _as_numpy(table: pa.Table, name: str, dtype: np.dtype[Any] | type) -> np.ndarray:
    try:
        return np.asarray(
            table[name].combine_chunks().to_numpy(zero_copy_only=False), dtype=dtype
        )
    except (pa.ArrowException, TypeError, ValueError) as exc:
        raise ContractError(f"S06 cannot read {name}: {exc}") from exc


def _all_true(value: pa.Array | pa.ChunkedArray | pa.Scalar) -> bool:
    result = pc.all(pc.fill_null(value, False)).as_py()
    return bool(result)


def _take(column: pa.ChunkedArray, indices: np.ndarray) -> pa.Array:
    return pc.take(column.combine_chunks(), pa.array(indices, type=pa.int64()))


def _assert_unique(values: np.ndarray, label: str) -> None:
    if len(values) > 1:
        ordered = np.sort(values, kind="stable")
        if np.any(ordered[1:] == ordered[:-1]):
            raise ContractError(f"S06 {label} values are duplicated")


def _search_exact(
    sorted_values: np.ndarray, query: np.ndarray, label: str
) -> np.ndarray:
    positions = np.searchsorted(sorted_values, query)
    safe = np.minimum(positions, max(0, len(sorted_values) - 1))
    if not len(sorted_values) or np.any(positions >= len(sorted_values)) or np.any(
        sorted_values[safe] != query
    ):
        raise ContractError(f"S06 {label} references an unknown ID")
    return positions.astype(np.int64, copy=False)


def _decode_invalid_reason(raw_flags: int) -> str:
    known_mask = 0
    names: list[str] = []
    for flag in BBoxQAFlag:
        known_mask |= int(flag)
        if raw_flags & int(flag):
            names.append(flag.name)
    if raw_flags & ~known_mask:
        raise ContractError("S06 detection qa_flags contains an unknown bit")
    if not raw_flags & INVALID_BBOX_MASK:
        raise ContractError("S06 invalid detection lacks an invalid-bbox QA flag")
    return "|".join(names)


def _nullable_mapped(
    mapping: pa.Table,
    name: str,
    lookup: np.ndarray,
    present: np.ndarray,
    output_type: pa.DataType,
) -> pa.Array:
    if mapping.num_rows == 0:
        return pa.nulls(len(present), type=output_type)
    safe = np.minimum(lookup, max(0, mapping.num_rows - 1)).astype(np.int64)
    values = _take(mapping[name], safe)
    return pc.if_else(
        pa.array(present, type=pa.bool_()),
        values,
        pa.nulls(len(present), type=output_type),
    )


def _validate_clip_order(clip_order: Sequence[str]) -> tuple[str, ...]:
    if isinstance(clip_order, (str, bytes)):
        raise ContractError("S06 clip_order must be a sequence of clip IDs")
    try:
        result = tuple(clip_order)
    except TypeError as exc:
        raise ContractError("S06 clip_order must be a sequence of clip IDs") from exc
    if (
        not result
        or any(not isinstance(value, str) or not value for value in result)
        or len(set(result)) != len(result)
    ):
        raise ContractError("S06 clip_order must contain unique non-empty clip IDs")
    return result


def build_detection_export_table(
    detections: pa.Table,
    det_to_global: pa.Table,
    microtracklets: pa.Table,
    *,
    clip_order: Sequence[str],
) -> pa.Table:
    """Left-join forced identities onto every S00 row.

    The join is keyed only by ``det_id`` and the final ordering is always
    ``(clip_order, csv_row_index)``.  Missing valid mappings remain null so the
    independent structural audit can report them.  A mapping for an invalid
    or unknown detection, however, is an upstream contract violation.
    """

    detections = _require_exact_schema(detections, DETECTIONS_SCHEMA, "detections")
    mapping = _require_exact_schema(
        det_to_global, DET_TO_GLOBAL_SCHEMA, "forced det_to_global"
    )
    microtracklets = _require_exact_schema(
        microtracklets, MICROTRACKLETS_SCHEMA, "microtracklets"
    )
    clips = _validate_clip_order(clip_order)
    if detections.num_rows == 0:
        raise ContractError("S06 detections table must not be empty")

    det_ids_input = _as_numpy(detections, "det_id", np.int64)
    _assert_unique(det_ids_input, "detection det_id")
    csv_rows_input = _as_numpy(detections, "csv_row_index", np.int64)
    if np.any(csv_rows_input < 0):
        raise ContractError("S06 csv_row_index must be non-negative")
    clip_values_input = detections["clip_id"].combine_chunks().to_pylist()
    clip_lookup = {clip_id: index for index, clip_id in enumerate(clips)}
    try:
        clip_indices_input = np.asarray(
            [clip_lookup[str(value)] for value in clip_values_input], dtype=np.int16
        )
    except KeyError as exc:
        raise ContractError(f"S06 detection references unknown clip {exc.args[0]!r}") from exc
    duplicate_order = np.lexsort((csv_rows_input, clip_indices_input))
    if len(duplicate_order) > 1:
        left = duplicate_order[:-1]
        right = duplicate_order[1:]
        if np.any(
            (clip_indices_input[left] == clip_indices_input[right])
            & (csv_rows_input[left] == csv_rows_input[right])
        ):
            raise ContractError("S06 (clip_id, csv_row_index) keys are duplicated")

    qa_flags_input = _as_numpy(detections, "qa_flags", np.uint32)
    valid_input = _as_numpy(detections, "valid", np.bool_)
    legacy_input = detections["legacy_track_id"].combine_chunks().to_pylist()
    if any(
        bool(is_valid)
        and (value is None or not isinstance(value, str) or not value.strip())
        for value, is_valid in zip(legacy_input, valid_input, strict=True)
    ):
        raise ContractError(
            "S06 valid detection lacks legacy_track_id for old-to-new ID audit"
        )
    known_mask = sum(int(flag) for flag in BBoxQAFlag)
    if np.any(qa_flags_input.astype(np.uint64) & np.uint64(~known_mask & 0xFFFFFFFF)):
        raise ContractError("S06 detection qa_flags contains an unknown bit")
    expected_valid = (qa_flags_input & np.uint32(INVALID_BBOX_MASK)) == 0
    if not np.array_equal(valid_input, expected_valid):
        raise ContractError("S06 detection valid flag differs from its QA flags")

    order = np.lexsort((csv_rows_input, clip_indices_input)).astype(np.int64)
    detections = detections.take(pa.array(order, type=pa.int64()))
    det_ids = det_ids_input[order]
    csv_rows = csv_rows_input[order]
    clip_indices = clip_indices_input[order]
    valid = valid_input[order]
    qa_flags = qa_flags_input[order]

    map_det_ids = _as_numpy(mapping, "det_id", np.int64)
    _assert_unique(map_det_ids, "forced mapping det_id")
    if mapping.num_rows:
        map_order = np.argsort(map_det_ids, kind="stable").astype(np.int64)
        mapping = mapping.take(pa.array(map_order, type=pa.int64()))
        map_det_ids = map_det_ids[map_order]
    det_ids_by_id = np.sort(det_ids, kind="stable")
    if mapping.num_rows:
        _search_exact(det_ids_by_id, map_det_ids, "forced mapping det_id")
    lookup = np.searchsorted(map_det_ids, det_ids)
    safe_lookup = np.minimum(lookup, max(0, mapping.num_rows - 1))
    present = (lookup < mapping.num_rows)
    if mapping.num_rows:
        present &= map_det_ids[safe_lookup] == det_ids
    if np.any(present & ~valid):
        raise ContractError("S06 forced mapping assigns an invalid detection")

    if np.any(present):
        det_positions = np.flatnonzero(present).astype(np.int64)
        map_positions = lookup[present].astype(np.int64)
        for name in (
            "det_id",
            "sequence_id",
            "clip_id",
            "local_frame",
            "global_frame",
            "global_time_sec",
            "valid",
        ):
            if not _all_true(
                pc.equal(
                    _take(detections[name], det_positions),
                    _take(mapping[name], map_positions),
                )
            ):
                raise ContractError(f"S06 forced mapping {name} differs from S00")
        mapped_clip_order = _as_numpy(mapping, "clip_order", np.int16)[map_positions]
        if not np.array_equal(mapped_clip_order, clip_indices[det_positions]):
            raise ContractError("S06 forced mapping clip_order differs from S00")
        if not _all_true(pc.equal(mapping["id_status"], _ID_STATUS)):
            raise ContractError("S06 forced mapping must remain forced_provisional")
        if not _all_true(
            pc.equal(mapping["identity_basis"], _AUTHORIZATION_BASIS)
        ):
            raise ContractError("S06 forced mapping identity basis differs")
        for name in (
            "global_track_id",
            "micro_id",
            "stable_id",
            "order_in_micro",
            "order_in_stable",
            "order_in_stable_detection",
            "order_in_global_stable",
            "order_in_global_detection",
        ):
            if np.any(_as_numpy(mapping, name, np.int64) < 0):
                raise ContractError(f"S06 forced mapping {name} must be non-negative")

    micro_ids = _as_numpy(microtracklets, "micro_id", np.int64)
    _assert_unique(micro_ids, "microtracklet micro_id")
    micro_order = np.argsort(micro_ids, kind="stable").astype(np.int64)
    microtracklets = microtracklets.take(pa.array(micro_order, type=pa.int64()))
    micro_ids = micro_ids[micro_order]
    purities = _as_numpy(microtracklets, "local_purity_score", np.float32)
    if np.any(~np.isfinite(purities)) or np.any((purities < 0.0) | (purities > 1.0)):
        raise ContractError("S06 microtracklet local_purity_score is invalid")
    if mapping.num_rows:
        mapped_micro = _as_numpy(mapping, "micro_id", np.int64)
        purity_lookup = _search_exact(
            micro_ids, mapped_micro, "forced mapping micro_id"
        )
        mapped_purity = pa.array(purities[purity_lookup], type=pa.float32())
    else:
        mapped_purity = pa.array([], type=pa.float32())

    invalid_reasons: list[str | None] = [None] * detections.num_rows
    for row in np.flatnonzero(~valid):
        invalid_reasons[int(row)] = _decode_invalid_reason(int(qa_flags[row]))

    arrays: list[pa.Array | pa.ChunkedArray] = []
    direct = {
        "sequence_id": detections["sequence_id"],
        "clip_id": detections["clip_id"],
        "det_id": detections["det_id"],
        "csv_row_index": detections["csv_row_index"],
        "legacy_track_id": detections["legacy_track_id"],
        "local_frame": detections["local_frame"],
        "global_frame": detections["global_frame"],
        "global_time_sec": detections["global_time_sec"],
        "x1": detections["x1"],
        "y1": detections["y1"],
        "x2": detections["x2"],
        "y2": detections["y2"],
        "bbox_confidence": detections["bbox_confidence"],
        "valid": detections["valid"],
        "qa_flags": detections["qa_flags"],
    }
    mapped_names = set(DET_TO_GLOBAL_SCHEMA.names)
    for field in DETECTIONS_WITH_GLOBAL_ID_SCHEMA:
        name = field.name
        if name in direct:
            arrays.append(direct[name])
        elif name == "clip_order":
            arrays.append(pa.array(clip_indices, type=pa.int16()))
        elif name == "invalid_reason":
            arrays.append(pa.array(invalid_reasons, type=pa.string()))
        elif name == "local_purity_score":
            if mapping.num_rows == 0:
                arrays.append(pa.nulls(detections.num_rows, type=pa.float32()))
            else:
                safe = np.minimum(lookup, mapping.num_rows - 1).astype(np.int64)
                values = pc.take(mapped_purity, pa.array(safe, type=pa.int64()))
                arrays.append(
                    pc.if_else(
                        pa.array(present),
                        values,
                        pa.nulls(detections.num_rows, type=pa.float32()),
                    )
                )
        elif name in (
            "assignment_confidence",
            "incoming_link_probability",
            "outgoing_link_probability",
        ):
            arrays.append(pa.nulls(detections.num_rows, type=field.type))
        elif name in mapped_names:
            arrays.append(
                _nullable_mapped(mapping, name, lookup, present, field.type)
            )
        else:  # pragma: no cover - protects future schema drift
            raise ContractError(f"S06 export builder does not handle field {name}")
    try:
        result = pa.Table.from_arrays(arrays, schema=DETECTIONS_WITH_GLOBAL_ID_SCHEMA)
    except (pa.ArrowException, TypeError, ValueError) as exc:
        raise ContractError(f"cannot build S06 detection export: {exc}") from exc
    if result.num_rows != detections.num_rows:
        raise ContractError("S06 detection export row count differs")
    return result


@dataclass(frozen=True, slots=True)
class S06StructuralMetrics:
    """Independently recomputed structural violation counts.

    ``cycle_count`` is the number of cyclic strongly connected components in
    the selected stable-link graph.  Same-frame violations count excess rows
    after the first row for each ``(global_track_id, global_frame)`` key.
    """

    same_frame_violation_count: int
    temporal_overlap_violation_count: int
    cycle_count: int
    unassigned_valid_detection_count: int

    @property
    def same_frame_duplicate_count(self) -> int:
        return self.same_frame_violation_count

    @property
    def selected_temporal_overlap_count(self) -> int:
        return self.temporal_overlap_violation_count

    def as_dict(self) -> dict[str, int]:
        return {
            "same_frame_violation_count": self.same_frame_violation_count,
            "temporal_overlap_violation_count": self.temporal_overlap_violation_count,
            "cycle_count": self.cycle_count,
            "unassigned_valid_detection_count": self.unassigned_valid_detection_count,
        }


# Concise public name used by the stage orchestration layer.
StructuralMetrics = S06StructuralMetrics


def _cycle_component_count(
    nodes: np.ndarray, sources: np.ndarray, targets: np.ndarray
) -> int:
    node_values = [int(value) for value in nodes]
    adjacency = {node: [] for node in node_values}
    reverse = {node: [] for node in node_values}
    self_loops: set[int] = set()
    for source_raw, target_raw in zip(sources, targets, strict=True):
        source, target = int(source_raw), int(target_raw)
        adjacency[source].append(target)
        reverse[target].append(source)
        if source == target:
            self_loops.add(source)
    for values in (*adjacency.values(), *reverse.values()):
        values.sort()

    visited: set[int] = set()
    finish: list[int] = []
    for root in sorted(node_values):
        if root in visited:
            continue
        visited.add(root)
        stack: list[tuple[int, int]] = [(root, 0)]
        while stack:
            node, offset = stack[-1]
            neighbours = adjacency[node]
            if offset < len(neighbours):
                target = neighbours[offset]
                stack[-1] = (node, offset + 1)
                if target not in visited:
                    visited.add(target)
                    stack.append((target, 0))
            else:
                finish.append(node)
                stack.pop()

    visited.clear()
    cyclic = 0
    for root in reversed(finish):
        if root in visited:
            continue
        component: list[int] = []
        stack = [root]
        visited.add(root)
        while stack:
            node = stack.pop()
            component.append(node)
            for target in reverse[node]:
                if target not in visited:
                    visited.add(target)
                    stack.append(target)
        if len(component) > 1 or component[0] in self_loops:
            cyclic += 1
    return cyclic


def recompute_structural_metrics(
    export_table: pa.Table,
    stable_to_global: pa.Table,
    candidate_edges: pa.Table,
) -> S06StructuralMetrics:
    """Recompute assignment, simultaneity, overlap and cycle invariants."""

    export_table = _require_exact_schema(
        export_table, DETECTIONS_WITH_GLOBAL_ID_SCHEMA, "detection export"
    )
    stable_to_global = _require_exact_schema(
        stable_to_global, STABLE_TO_GLOBAL_SCHEMA, "stable_to_global"
    )
    candidate_edges = _require_exact_schema(
        candidate_edges, FORCED_CANDIDATE_EDGES_SCHEMA, "candidate_edges"
    )
    stable_ids = _as_numpy(stable_to_global, "stable_id", np.int64)
    _assert_unique(stable_ids, "stable_to_global stable_id")
    stable_sorted = np.sort(stable_ids, kind="stable")

    valid = _as_numpy(export_table, "valid", np.bool_)
    global_column = export_table["global_track_id"].combine_chunks()
    assigned = np.asarray(
        pc.is_valid(global_column).to_numpy(zero_copy_only=False), dtype=np.bool_
    )
    unassigned = int(np.count_nonzero(valid & ~assigned))
    assigned_positions = np.flatnonzero(valid & assigned).astype(np.int64)
    duplicate_count = 0
    if len(assigned_positions):
        global_ids = np.asarray(
            _take(export_table["global_track_id"], assigned_positions).to_numpy(),
            dtype=np.int64,
        )
        frames = _as_numpy(export_table, "global_frame", np.int64)[assigned_positions]
        order = np.lexsort((frames, global_ids))
        if len(order) > 1:
            duplicate_count = int(
                np.count_nonzero(
                    (global_ids[order][1:] == global_ids[order][:-1])
                    & (frames[order][1:] == frames[order][:-1])
                )
            )

    candidate_ids = candidate_edges["candidate_id"].combine_chunks().to_pylist()
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ContractError("S06 candidate IDs are duplicated")
    selected = _as_numpy(candidate_edges, "selected_by_solver", np.bool_)
    selected_rows = candidate_edges.filter(pa.array(selected))
    sources = _as_numpy(selected_rows, "source_stable_id", np.int64)
    targets = _as_numpy(selected_rows, "target_stable_id", np.int64)
    if len(sources):
        _search_exact(stable_sorted, sources, "selected source stable_id")
        _search_exact(stable_sorted, targets, "selected target stable_id")
    pairs = np.rec.fromarrays([sources, targets], names="source,target")
    if len(pairs) > 1:
        ordered_pairs = np.sort(pairs, kind="stable")
        if np.any(ordered_pairs[1:] == ordered_pairs[:-1]):
            raise ContractError("S06 selected stable edge pairs are duplicated")

    strict_flag = _as_numpy(selected_rows, "strictly_nonoverlapping", np.bool_)
    source_frames = _as_numpy(selected_rows, "source_end_global_frame", np.int64)
    target_frames = _as_numpy(selected_rows, "target_start_global_frame", np.int64)
    source_times = _as_numpy(selected_rows, "source_end_time_sec", np.float64)
    target_times = _as_numpy(selected_rows, "target_start_time_sec", np.float64)
    finite = np.isfinite(source_times) & np.isfinite(target_times)
    overlap = (
        ~strict_flag
        | (source_frames >= target_frames)
        | (source_times >= target_times)
        | ~finite
    )
    overlap_count = int(np.count_nonzero(overlap))
    cycle_count = _cycle_component_count(stable_sorted, sources, targets)
    return S06StructuralMetrics(
        same_frame_violation_count=duplicate_count,
        temporal_overlap_violation_count=overlap_count,
        cycle_count=cycle_count,
        unassigned_valid_detection_count=unassigned,
    )


def _require_constant_string(table: pa.Table, name: str, value: str, label: str) -> None:
    if table.num_rows == 0:
        return
    if not _all_true(pc.equal(table[name], value)):
        raise ContractError(f"S06 {label}.{name} must equal {value!r}")


def _require_all_null(table: pa.Table, names: Sequence[str], label: str) -> None:
    for name in names:
        if table[name].null_count != table.num_rows:
            raise ContractError(f"S06 {label}.{name} must be entirely null")


def _validate_export_for_summary(
    table: pa.Table, config: S06ExportConfig
) -> tuple[pa.Table, str]:
    table = _require_exact_schema(
        table, DETECTIONS_WITH_GLOBAL_ID_SCHEMA, "detection export"
    )
    sequence_values = table["sequence_id"].combine_chunks().to_pylist()
    sequences = {
        value
        for value in sequence_values
        if isinstance(value, str) and value
    }
    if len(sequences) != 1 or len(sequence_values) != sum(
        isinstance(value, str) and bool(value) for value in sequence_values
    ):
        raise ContractError("S06 detection export must contain one non-empty sequence")
    observed_sequence_id = next(iter(sequences))
    valid_mask = _as_numpy(table, "valid", np.bool_)
    invalid = table.filter(pa.array(~valid_mask))
    _require_all_null(invalid, _IDENTITY_FIELDS, "invalid detection")
    valid = table.filter(pa.array(valid_mask))
    for name in _MAPPED_FIELDS:
        if valid[name].null_count:
            raise ContractError(f"S06 valid detection {name} is unassigned")
    _require_all_null(
        table,
        (
            "assignment_confidence",
            "incoming_link_probability",
            "outgoing_link_probability",
        ),
        "detection export",
    )
    _require_constant_string(valid, "id_status", config.id_status, "valid detection")
    _require_constant_string(
        valid, "identity_basis", config.authorization_basis, "valid detection"
    )
    if config.id_status != _ID_STATUS or config.authorization_basis != _AUTHORIZATION_BASIS:
        raise ContractError("S06 config attempts to upgrade or relabel forced identities")
    if config.certification_claimed or config.probability_from_cosine_allowed:
        raise ContractError("S06 forced export cannot claim certification/probability")
    clips = _validate_clip_order(config.clip_order)
    clip_indices = _as_numpy(table, "clip_order", np.int16)
    csv_rows = _as_numpy(table, "csv_row_index", np.int64)
    if np.any((clip_indices < 0) | (clip_indices >= len(clips))):
        raise ContractError("S06 export clip_order index is out of range")
    order = np.lexsort((csv_rows, clip_indices))
    if not np.array_equal(order, np.arange(table.num_rows)):
        raise ContractError("S06 export is not sorted by (clip_order, csv_row_index)")
    clip_values = table["clip_id"].combine_chunks().to_pylist()
    if any(
        clips[int(index)] != value
        for index, value in zip(clip_indices, clip_values, strict=True)
    ):
        raise ContractError("S06 export clip_id and clip_order disagree")
    return valid, observed_sequence_id


def _validate_stable_mapping(
    table: pa.Table, config: S06ExportConfig
) -> tuple[pa.Table, np.ndarray, np.ndarray, np.ndarray]:
    table = _require_exact_schema(table, STABLE_TO_GLOBAL_SCHEMA, "stable_to_global")
    stable_ids = _as_numpy(table, "stable_id", np.int64)
    _assert_unique(stable_ids, "stable_to_global stable_id")
    global_ids = _as_numpy(table, "global_track_id", np.int64)
    expected_globals = np.arange(config.expected_global_track_count, dtype=np.int64)
    if set(map(int, global_ids)) != set(map(int, expected_globals)):
        raise ContractError("S06 stable mapping global IDs differ from config")
    order_in_path = _as_numpy(table, "order_in_global_path", np.int64)
    _require_constant_string(table, "id_status", config.id_status, "stable mapping")
    _require_constant_string(
        table, "identity_basis", config.authorization_basis, "stable mapping"
    )
    _require_all_null(
        table,
        (
            "predecessor_link_probability",
            "predecessor_link_margin",
            "component_min_link_probability",
            "component_mean_link_probability",
            "component_max_link_probability",
        ),
        "stable mapping",
    )
    for global_id in expected_globals:
        observed = np.sort(order_in_path[global_ids == global_id])
        if not np.array_equal(observed, np.arange(len(observed))):
            raise ContractError("S06 stable path order is not contiguous")
    return table, stable_ids, global_ids, order_in_path


def _global_metadata(
    global_tracks: pa.Table, config: S06ExportConfig
) -> dict[int, dict[str, Any]]:
    global_tracks = _require_exact_schema(
        global_tracks, GLOBAL_TRACKS_SCHEMA, "global_tracks"
    )
    if global_tracks.num_rows != config.expected_global_track_count:
        raise ContractError("S06 global track count differs from config")
    _require_all_null(
        global_tracks,
        (
            "min_link_probability",
            "p10_link_probability",
            "mean_link_probability",
            "max_link_probability",
            "min_link_margin",
        ),
        "global tracks",
    )
    rows = global_tracks.to_pylist()
    by_id: dict[int, dict[str, Any]] = {}
    for row in rows:
        global_id = int(row["global_track_id"])
        if global_id in by_id:
            raise ContractError("S06 global_track_id values are duplicated")
        by_id[global_id] = row
    if set(by_id) != set(range(config.expected_global_track_count)):
        raise ContractError("S06 global_track_id values are not contiguous")
    return by_id


def _stable_endpoints(
    valid: pa.Table, stable_ids_sorted: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    detection_stable = _as_numpy(valid, "stable_id", np.int64)
    stable_index = _search_exact(
        stable_ids_sorted, detection_stable, "detection stable_id"
    )
    frames = _as_numpy(valid, "global_frame", np.int64)
    times = _as_numpy(valid, "global_time_sec", np.float64)
    det_ids = _as_numpy(valid, "det_id", np.int64)
    clips = np.asarray(valid["clip_id"].combine_chunks().to_pylist(), dtype=object)
    sort_order = np.lexsort((det_ids, frames, stable_index))
    sorted_index = stable_index[sort_order]
    starts = np.r_[True, sorted_index[1:] != sorted_index[:-1]]
    ends = np.r_[sorted_index[1:] != sorted_index[:-1], True]
    start_rows = sort_order[starts]
    end_rows = sort_order[ends]
    if len(start_rows) != len(stable_ids_sorted) or not np.array_equal(
        stable_index[start_rows], np.arange(len(stable_ids_sorted))
    ):
        raise ContractError("S06 at least one stable track has no detection")
    return (
        frames[start_rows],
        frames[end_rows],
        times[start_rows],
        times[end_rows],
        clips[start_rows],
        clips[end_rows],
    )


def build_global_track_summary(
    export_table: pa.Table,
    stable_to_global: pa.Table,
    global_tracks: pa.Table,
    candidate_edges: pa.Table,
    graded_stable_appearance: pa.Table,
    *,
    config: S06ExportConfig,
) -> pa.Table:
    """Build the strict 62-row (or synthetic replacement) global QA summary."""

    if not isinstance(config, S06ExportConfig):
        raise ContractError("S06 summary config has the wrong type")
    valid, observed_sequence_id = _validate_export_for_summary(export_table, config)
    stable_to_global, stable_ids, stable_globals, path_orders = _validate_stable_mapping(
        stable_to_global, config
    )
    globals_by_id = _global_metadata(global_tracks, config)
    candidate_edges = _require_exact_schema(
        candidate_edges, FORCED_CANDIDATE_EDGES_SCHEMA, "candidate_edges"
    )
    graded = _require_exact_schema(
        graded_stable_appearance,
        GRADED_STABLE_APPEARANCE_SCHEMA,
        "graded_stable_appearance",
    )
    stable_sort = np.argsort(stable_ids, kind="stable")
    stable_ids_sorted = stable_ids[stable_sort]
    stable_globals_sorted = stable_globals[stable_sort]
    path_orders_sorted = path_orders[stable_sort]
    stable_table_sorted = stable_to_global.take(pa.array(stable_sort, type=pa.int64()))
    grade_ids = _as_numpy(graded, "stable_id", np.int64)
    _assert_unique(grade_ids, "graded appearance stable_id")
    grade_sort = np.argsort(grade_ids, kind="stable")
    if not np.array_equal(grade_ids[grade_sort], stable_ids_sorted):
        raise ContractError("S06 graded appearance stable coverage differs")
    graded = graded.take(pa.array(grade_sort, type=pa.int64()))
    grade_values = np.asarray(
        graded["evidence_grade"].combine_chunks().to_pylist(), dtype=object
    )
    if not set(map(str, grade_values)) <= set(APPEARANCE_GRADES):
        raise ContractError("S06 graded appearance contains an unknown grade")
    if not _all_true(graded["descriptor_usable"]):
        raise ContractError("S06 forced graded appearance must be usable")

    detection_stable = _as_numpy(valid, "stable_id", np.int64)
    detection_stable_index = _search_exact(
        stable_ids_sorted, detection_stable, "detection stable_id"
    )
    detection_global = _as_numpy(valid, "global_track_id", np.int64)
    if not np.array_equal(
        detection_global, stable_globals_sorted[detection_stable_index]
    ):
        raise ContractError("S06 detection global mapping differs from stable mapping")
    detection_global_order = _as_numpy(valid, "order_in_global_stable", np.int64)
    if not np.array_equal(
        detection_global_order, path_orders_sorted[detection_stable_index]
    ):
        raise ContractError("S06 detection path order differs from stable mapping")
    if np.any((detection_global < 0) | (detection_global >= config.expected_global_track_count)):
        raise ContractError("S06 detection global_track_id is out of range")

    stable_uuid = stable_table_sorted["global_track_uuid"].combine_chunks().to_pylist()
    stable_display = stable_table_sorted["display_global_id"].combine_chunks().to_pylist()
    detection_uuid = np.asarray(
        valid["global_track_uuid"].combine_chunks().to_pylist(), dtype=object
    )
    detection_display = np.asarray(
        valid["display_global_id"].combine_chunks().to_pylist(), dtype=object
    )
    if not np.array_equal(
        detection_uuid,
        np.asarray(stable_uuid, dtype=object)[detection_stable_index],
    ) or not np.array_equal(
        detection_display,
        np.asarray(stable_display, dtype=object)[detection_stable_index],
    ):
        raise ContractError("S06 detection identity metadata differs from stable mapping")
    for index, stable_id in enumerate(stable_ids_sorted):
        global_id = int(stable_globals_sorted[index])
        metadata = globals_by_id[global_id]
        if (
            stable_uuid[index] != metadata["global_track_uuid"]
            or stable_display[index] != metadata["display_global_id"]
        ):
            raise ContractError(
                f"S06 stable/global metadata differs for stable {int(stable_id)}"
            )

    endpoint_data = _stable_endpoints(valid, stable_ids_sorted)
    (
        stable_start_frame,
        stable_end_frame,
        stable_start_time,
        stable_end_time,
        stable_start_clip,
        stable_end_clip,
    ) = endpoint_data

    candidate_ids = candidate_edges["candidate_id"].combine_chunks().to_pylist()
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ContractError("S06 candidate IDs are duplicated")
    sources = _as_numpy(candidate_edges, "source_stable_id", np.int64)
    targets = _as_numpy(candidate_edges, "target_stable_id", np.int64)
    source_index = _search_exact(stable_ids_sorted, sources, "candidate source_stable_id")
    target_index = _search_exact(stable_ids_sorted, targets, "candidate target_stable_id")
    pair_order = np.lexsort((targets, sources))
    if len(pair_order) > 1 and np.any(
        (sources[pair_order][1:] == sources[pair_order][:-1])
        & (targets[pair_order][1:] == targets[pair_order][:-1])
    ):
        raise ContractError("S06 candidate stable edge pairs are duplicated")
    if not np.array_equal(
        _as_numpy(candidate_edges, "source_end_global_frame", np.int64),
        stable_end_frame[source_index],
    ) or not np.array_equal(
        _as_numpy(candidate_edges, "target_start_global_frame", np.int64),
        stable_start_frame[target_index],
    ):
        raise ContractError("S06 candidate frame endpoints differ from detections")
    candidate_source_time = _as_numpy(
        candidate_edges, "source_end_time_sec", np.float64
    )
    candidate_target_time = _as_numpy(
        candidate_edges, "target_start_time_sec", np.float64
    )
    if not np.array_equal(
        candidate_source_time, stable_end_time[source_index]
    ) or not np.array_equal(candidate_target_time, stable_start_time[target_index]):
        raise ContractError("S06 candidate time endpoints differ from detections")
    source_clips = np.asarray(
        candidate_edges["source_end_clip_id"].combine_chunks().to_pylist(), dtype=object
    )
    target_clips = np.asarray(
        candidate_edges["target_start_clip_id"].combine_chunks().to_pylist(), dtype=object
    )
    if not np.array_equal(source_clips, stable_end_clip[source_index]) or not np.array_equal(
        target_clips, stable_start_clip[target_index]
    ):
        raise ContractError("S06 candidate clip endpoints differ from detections")
    recomputed_gaps = candidate_target_time - candidate_source_time
    declared_gaps = _as_numpy(candidate_edges, "temporal_gap_sec", np.float64)
    if not np.allclose(declared_gaps, recomputed_gaps, rtol=0.0, atol=1e-12):
        raise ContractError("S06 candidate temporal gaps differ from endpoints")
    strict = (stable_end_frame[source_index] < stable_start_frame[target_index]) & (
        candidate_source_time < candidate_target_time
    )
    if not np.array_equal(
        _as_numpy(candidate_edges, "strictly_nonoverlapping", np.bool_), strict
    ):
        raise ContractError("S06 candidate non-overlap flags differ from endpoints")
    cosine = _as_numpy(candidate_edges, "appearance_cosine", np.float64)
    if np.any(~np.isfinite(cosine)) or np.any((cosine < -1.0) | (cosine > 1.0)):
        raise ContractError("S06 candidate appearance cosine is invalid")
    source_grades = np.asarray(
        candidate_edges["source_evidence_grade"].combine_chunks().to_pylist(), dtype=object
    )
    target_grades = np.asarray(
        candidate_edges["target_evidence_grade"].combine_chunks().to_pylist(), dtype=object
    )
    if not np.array_equal(source_grades, grade_values[source_index]) or not np.array_equal(
        target_grades, grade_values[target_index]
    ):
        raise ContractError("S06 candidate evidence grades differ from graded appearance")

    selected = _as_numpy(candidate_edges, "selected_by_solver", np.bool_)
    selected_global_link = candidate_edges["global_link_id"].combine_chunks()
    if pc.any(pc.and_(pa.array(selected), pc.is_null(selected_global_link))).as_py():
        raise ContractError("S06 selected candidate lacks global_link_id")
    if pc.any(pc.and_(pa.array(~selected), pc.is_valid(selected_global_link))).as_py():
        raise ContractError("S06 unselected candidate has global_link_id")
    selected_rows = candidate_edges.filter(pa.array(selected))
    _require_constant_string(
        candidate_edges,
        "authorization_basis",
        config.authorization_basis,
        "candidate",
    )
    _require_constant_string(
        selected_rows, "authorization_basis", config.authorization_basis, "selected candidate"
    )
    _require_constant_string(selected_rows, "id_status", config.id_status, "selected candidate")

    expected_pairs: set[tuple[int, int]] = set()
    paths_by_global: dict[int, list[int]] = {}
    for global_id in range(config.expected_global_track_count):
        indices = np.flatnonzero(stable_globals_sorted == global_id)
        ordered = indices[np.argsort(path_orders_sorted[indices], kind="stable")]
        path = [int(stable_ids_sorted[index]) for index in ordered]
        paths_by_global[global_id] = path
        expected_pairs.update(zip(path, path[1:], strict=False))
    selected_pairs = {
        (int(source), int(target))
        for source, target in zip(sources[selected], targets[selected], strict=True)
    }
    if selected_pairs != expected_pairs or len(selected_pairs) != int(
        np.count_nonzero(selected)
    ):
        raise ContractError("S06 selected candidates differ from stable path adjacency")
    if np.any(
        stable_globals_sorted[source_index[selected]]
        != stable_globals_sorted[target_index[selected]]
    ):
        raise ContractError("S06 selected candidate crosses global paths")

    structural = recompute_structural_metrics(
        export_table, stable_to_global, candidate_edges
    )
    expected_structural = S06StructuralMetrics(0, 0, 0, 0)
    if structural != expected_structural:
        raise ContractError(
            f"S06 structural metrics differ: observed={structural}, expected={expected_structural}"
        )

    valid_frames = _as_numpy(valid, "global_frame", np.int64)
    valid_times = _as_numpy(valid, "global_time_sec", np.float64)
    valid_det_ids = _as_numpy(valid, "det_id", np.int64)
    valid_micro = _as_numpy(valid, "micro_id", np.int64)
    valid_clip_indices = _as_numpy(valid, "clip_order", np.int16)
    valid_clips = np.asarray(valid["clip_id"].combine_chunks().to_pylist(), dtype=object)
    selected_source_indices = source_index[selected]
    selected_global = stable_globals_sorted[selected_source_indices]
    selected_cosine = cosine[selected]
    selected_gaps = recomputed_gaps[selected]
    selected_source_grades = source_grades[selected]
    selected_target_grades = target_grades[selected]
    selected_source_topk = _as_numpy(
        candidate_edges, "selected_by_source_topk", np.bool_
    )[selected]
    selected_target_topk = _as_numpy(
        candidate_edges, "selected_by_target_topk", np.bool_
    )[selected]
    selected_backbone = _as_numpy(
        candidate_edges, "temporal_backbone", np.bool_
    )[selected]
    selected_prior = _as_numpy(candidate_edges, "prior_global_link", np.bool_)[selected]

    actual_overflow = max(
        0, config.expected_global_track_count - config.population_soft_max
    )
    if (
        config.num_confirmed_ids != 0
        or config.num_provisional_ids != config.expected_global_track_count
    ):
        raise ContractError("S06 confirmed/provisional identity counts are inconsistent")
    population_warning = actual_overflow > 0
    frame_duration_sec = config.fps_denominator / config.fps_numerator
    rows: list[dict[str, Any]] = []
    for global_id in range(config.expected_global_track_count):
        positions = np.flatnonzero(detection_global == global_id)
        if not len(positions):
            raise ContractError(f"S06 global track {global_id} has no detections")
        temporal_order = np.lexsort(
            (valid_det_ids[positions], valid_frames[positions])
        )
        first_position = positions[temporal_order[0]]
        last_position = positions[temporal_order[-1]]
        path = paths_by_global[global_id]
        link_positions = np.flatnonzero(selected_global == global_id)
        scores = selected_cosine[link_positions]
        gaps = selected_gaps[link_positions]
        endpoints = np.concatenate(
            (
                selected_source_grades[link_positions],
                selected_target_grades[link_positions],
            )
        )
        used_clips = [
            clip
            for clip_index, clip in enumerate(config.clip_order)
            if np.any(valid_clip_indices[positions] == clip_index)
        ]
        metadata = globals_by_id[global_id]
        expected_display = (
            f"{config.display_id_prefix}{global_id + 1:0{config.display_id_width}d}"
        )
        recomputed = {
            "global_track_uuid": str(metadata["global_track_uuid"]),
            "display_global_id": expected_display,
            "start_clip_id": str(valid_clips[first_position]),
            "end_clip_id": str(valid_clips[last_position]),
            "clip_ids": used_clips,
            "start_global_frame": int(valid_frames[first_position]),
            "end_global_frame": int(valid_frames[last_position]),
            "start_time_sec": float(valid_times[first_position]),
            "end_time_sec": float(valid_times[last_position]),
            "num_stable_tracklets": len(path),
            "num_microtracklets": int(len(np.unique(valid_micro[positions]))),
            "num_detections": int(len(positions)),
            "selected_link_count": int(len(link_positions)),
        }
        comparisons = {
            "global_track_uuid": metadata["global_track_uuid"],
            "display_global_id": metadata["display_global_id"],
            "start_clip_id": metadata["start_clip_id"],
            "end_clip_id": metadata["end_clip_id"],
            "clip_ids": metadata["clip_ids"],
            "start_global_frame": metadata["start_global_frame"],
            "end_global_frame": metadata["end_global_frame"],
            "start_time_sec": metadata["start_time_sec"],
            "end_time_sec": metadata["end_time_sec"],
            "num_stable_tracklets": metadata["num_stable_tracklets"],
            "num_microtracklets": metadata["num_microtracklets"],
            "num_detections": metadata["num_detections"],
            "selected_link_count": metadata["num_long_links"],
        }
        if comparisons != recomputed:
            raise ContractError(
                f"S06 upstream global track {global_id} differs from recomputation"
            )
        if (
            int(metadata["first_stable_id"]) != path[0]
            or int(metadata["last_stable_id"]) != path[-1]
            or int(metadata["start_det_id"]) != int(valid_det_ids[first_position])
            or int(metadata["end_det_id"]) != int(valid_det_ids[last_position])
            or
            metadata["sequence_id"] != observed_sequence_id
            or metadata["id_status"] != config.id_status
            or metadata["identity_basis"] != config.authorization_basis
            or bool(metadata["spans_multiple_clips"]) != (len(used_clips) > 1)
        ):
            raise ContractError(f"S06 upstream global track {global_id} provenance differs")
        rows.append(
            {
                "global_track_id": global_id,
                "global_track_uuid": recomputed["global_track_uuid"],
                "display_global_id": expected_display,
                "sequence_id": observed_sequence_id,
                "id_status": config.id_status,
                "identity_basis": config.authorization_basis,
                "authorization_basis": config.authorization_basis,
                "certification_claimed": False,
                "start_clip_id": recomputed["start_clip_id"],
                "end_clip_id": recomputed["end_clip_id"],
                "clip_ids": "|".join(used_clips),
                "start_global_frame": recomputed["start_global_frame"],
                "end_global_frame": recomputed["end_global_frame"],
                "start_time_sec": recomputed["start_time_sec"],
                "end_time_sec": recomputed["end_time_sec"],
                "duration_visible_sec": len(positions) * frame_duration_sec,
                "num_stable_tracklets": len(path),
                "num_microtracklets": recomputed["num_microtracklets"],
                "num_detections": len(positions),
                "selected_link_count": len(link_positions),
                "spans_multiple_clips": len(used_clips) > 1,
                "selected_link_cosine_min": float(np.min(scores)) if len(scores) else None,
                "selected_link_cosine_p10": (
                    float(np.quantile(scores, 0.1, method="linear")) if len(scores) else None
                ),
                "selected_link_cosine_mean": float(np.mean(scores)) if len(scores) else None,
                "selected_link_cosine_max": float(np.max(scores)) if len(scores) else None,
                "longest_gap_sec": float(np.max(gaps)) if len(gaps) else None,
                "num_links_cosine_below_0_3": int(np.count_nonzero(scores < 0.3)),
                "num_links_cosine_below_0_4": int(np.count_nonzero(scores < 0.4)),
                "num_links_cosine_below_0_5": int(np.count_nonzero(scores < 0.5)),
                "num_links_cosine_below_0_6": int(np.count_nonzero(scores < 0.6)),
                "num_link_endpoints_grade_a_clean": int(
                    np.count_nonzero(endpoints == "A_CLEAN")
                ),
                "num_link_endpoints_grade_b_existing_degraded": int(
                    np.count_nonzero(endpoints == "B_EXISTING_DEGRADED")
                ),
                "num_link_endpoints_grade_c_reencoded_degraded": int(
                    np.count_nonzero(endpoints == "C_REENCODED_DEGRADED")
                ),
                "num_links_selected_by_source_topk": int(
                    np.count_nonzero(selected_source_topk[link_positions])
                ),
                "num_links_selected_by_target_topk": int(
                    np.count_nonzero(selected_target_topk[link_positions])
                ),
                "num_links_temporal_backbone": int(
                    np.count_nonzero(selected_backbone[link_positions])
                ),
                "num_links_prior_global": int(
                    np.count_nonzero(selected_prior[link_positions])
                ),
                "min_link_probability": None,
                "p10_link_probability": None,
                "mean_link_probability": None,
                "max_link_probability": None,
                "probability_not_available_reason": _PROBABILITY_REASON,
                "population_soft_max": config.population_soft_max,
                "population_overflow": actual_overflow,
                "population_warning": population_warning,
            }
        )
    try:
        result = pa.Table.from_pylist(rows, schema=GLOBAL_TRACK_SUMMARY_SCHEMA)
    except (pa.ArrowException, TypeError, ValueError) as exc:
        raise ContractError(f"cannot build S06 global summary: {exc}") from exc
    return result


__all__ = [
    "S06StructuralMetrics",
    "StructuralMetrics",
    "build_detection_export_table",
    "build_global_track_summary",
    "recompute_structural_metrics",
]
