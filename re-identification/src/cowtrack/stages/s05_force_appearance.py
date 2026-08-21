"""Solve all finalized S04 fragments into 62 appearance-based global IDs.

This is an explicitly operator-authorized, dataset-specific stage.  It does
not reinterpret its result as certified identity: every output ID remains
``forced_provisional``.  Same-frame/temporal overlap, reverse time, cycles,
and non-bijective mappings remain hard failures.  The complete sequence builds
one strict-future candidate graph and runs one exact fixed-cardinality min-cost
flow.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import time
import uuid
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from cowtrack.appearance.encoders import (
    build_encoder,
    embed_four_views,
    encoder_profile_by_name,
)
from cowtrack.appearance.quality import (
    bbox_area_percentiles,
    combine_crop_quality,
    crop_with_padding,
    laplacian_blur_score,
    soft_border_mask,
)
from cowtrack.appearance.sampling import select_candidate_pool_indices
from cowtrack.config import ContractError, load_manifest
from cowtrack.linking.forced_appearance import (
    GRADE_A_CLEAN,
    GRADE_B_EXISTING_DEGRADED,
    GRADE_C_SUPPLEMENTAL,
    ForcedAppearanceBundle,
    SupplementalStableEmbeddings,
    build_forced_appearance_base,
    complete_forced_appearance_with_supplemental,
    restore_completed_forced_appearance,
)
from cowtrack.linking.forced_appearance_config import (
    ForcedAppearanceConfig,
    load_forced_appearance_config,
)
from cowtrack.linking.forced_candidates import (
    ForcedCandidateGraph,
    build_forced_candidate_graph,
)
from cowtrack.linking.forced_path_cover import (
    ForcedPathCoverResult,
    solve_forced_fixed_path_cover,
)
from cowtrack.linking.path_cover import GlobalStableNode
from cowtrack.linking.runtime import (
    FileFingerprint,
    ProductionInputBundle,
    fingerprint_file,
    load_production_inputs,
)
from cowtrack.linking.s04_runtime import S04FinalizedBundle, load_s04_finalized
from cowtrack.schemas.detections import DETECTIONS_SCHEMA
from cowtrack.schemas.s05_finalize import (
    DET_TO_GLOBAL_SCHEMA,
    GLOBAL_TRACKS_SCHEMA,
    STABLE_TO_GLOBAL_SCHEMA,
)
from cowtrack.schemas.s05_forced import (
    FORCED_CANDIDATE_EDGES_SCHEMA,
    GRADED_STABLE_APPEARANCE_SCHEMA,
    RESCUE_SAMPLES_SCHEMA,
)
from cowtrack.video import open_raw_video_capture
from cowtrack.stages.s05_finalize import build_detection_to_global_table


LogFn = Callable[[str], None]
_STAGE = "S05_FORCE_APPEARANCE"
_PREPARE_STAGE = "S05_FORCE_APPEARANCE_PREPARE"
_PREPARE_MARKER = "_PREPARED.json"
_PREPARE_ALGORITHM = "graded_appearance_rescue_v1"
_PREPARED_DESCRIPTOR_CENTERS = "graded_descriptor_centers.f32.npy"
_FRAME_DURATION_SEC = 1001.0 / 30000.0
_RAW_FRAME_SHAPE = (2160, 3840, 3)
_AUTHORIZATION = "operator_forced_appearance_exact_62"
_ID_STATUS = "forced_provisional"
_SOLVE_MODE = "full_sequence_single_solve"
_SOLVER_ALGORITHM = "deterministic_full_sequence_fixed_cardinality_min_cost_flow"
_SOLVER_OBJECTIVE = (
    "exact_62_global_paths_then_minimum_full_sequence_appearance_cost"
)
_ASSIGNMENT_ALGORITHM = "exact_fixed_flow_min_cost_network_without_dummies"


def log(message: str) -> None:
    print(message, flush=True)


@dataclass
class _RescueCandidate:
    position: int
    stable_id: int
    micro_id: int
    det_id: int
    clip_id: str
    local_frame: int
    global_frame: int
    global_time_sec: float
    bbox: tuple[float, float, float, float]
    other_iou: float
    area_percentile: float
    review_excluded: bool
    crop_quality: float = 0.0
    clipped_fraction: float = 0.0
    blur_score: float = 0.0
    boundary_distance: float = 0.0
    selected: bool = False
    embedding_row: int | None = None
    crop_rgb: np.ndarray | None = None


@dataclass(frozen=True, slots=True)
class _SolvePassSummary:
    """The single complete-sequence fixed-cardinality solve."""

    pass_index: int
    source_clip_ids: tuple[str, ...]
    solver_nodes: int
    candidate_count: int
    interval_width: int
    backbone_chain_count: int
    maximum_feasible_links: int
    required_links: int
    selected_links: int
    total_appearance_cost_int: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "pass_index": self.pass_index,
            "source_clip_ids": list(self.source_clip_ids),
            "solver_nodes": self.solver_nodes,
            "candidate_count": self.candidate_count,
            "interval_width": self.interval_width,
            "backbone_chain_count": self.backbone_chain_count,
            "maximum_feasible_links": self.maximum_feasible_links,
            "required_links": self.required_links,
            "selected_links": self.selected_links,
            "total_appearance_cost_int": self.total_appearance_cost_int,
        }


@dataclass(frozen=True, slots=True)
class _LoadedForcedInputs:
    """Strict common inputs shared by preparation and exact linking."""

    config: ForcedAppearanceConfig
    config_payload: Mapping[str, Any]
    config_hash: str
    production: ProductionInputBundle
    stable: S04FinalizedBundle
    identity: Mapping[str, np.ndarray] | None
    video_paths: Mapping[str, Path]
    input_fingerprints: tuple[Mapping[str, Any], ...]
    video_path_set: frozenset[str]


@dataclass(frozen=True, slots=True)
class _PreparedAppearance:
    """Validated immutable preparation bundle consumed by the CPU solve."""

    appearance: ForcedAppearanceBundle
    rescue_rows: tuple[Mapping[str, Any], ...]
    rescue_embeddings: np.ndarray
    no_s02_sample_count: int
    input_fingerprints: tuple[FileFingerprint, ...]


def _read_json(path: Path, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read {label} {path}: {exc}") from exc


def _write_json(path: Path, payload: Any) -> None:
    try:
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except (OSError, TypeError, ValueError) as exc:
        raise ContractError(f"cannot write forced appearance JSON {path}: {exc}") from exc


def _write_table(path: Path, table: pa.Table, schema: pa.Schema, compression: str) -> None:
    if not table.schema.equals(schema, check_metadata=False):
        raise ContractError(f"forced appearance table schema differs: {path.name}")
    try:
        pq.write_table(table, path, compression=compression, version="2.6")
    except (OSError, pa.ArrowException) as exc:
        raise ContractError(f"cannot write forced appearance Parquet {path}: {exc}") from exc


def _rows_table(rows: Sequence[Mapping[str, Any]], schema: pa.Schema, label: str) -> pa.Table:
    try:
        return pa.Table.from_pylist([dict(row) for row in rows], schema=schema)
    except (pa.ArrowException, TypeError, ValueError) as exc:
        raise ContractError(f"cannot build {label}: {exc}") from exc


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise ContractError(f"cannot fingerprint forced appearance artifact {path}: {exc}") from exc
    return digest.hexdigest()


def _output_fingerprint(path: Path, directory: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(directory)),
        "size_bytes": int(path.stat().st_size),
        "sha256": _sha256(path),
    }


def _normalize_fingerprints(records: Sequence[FileFingerprint]) -> list[dict[str, Any]]:
    by_path: dict[str, FileFingerprint] = {}
    for item in records:
        previous = by_path.get(item.path)
        if previous is not None and previous != item:
            raise ContractError("forced appearance input fingerprint conflicts by path")
        by_path[item.path] = item
    if not by_path:
        raise ContractError("forced appearance input fingerprint set cannot be empty")
    return [by_path[path].as_dict() for path in sorted(by_path)]


def _verify_regular_fingerprints(records: Sequence[Mapping[str, Any]], video_paths: set[str]) -> None:
    for item in records:
        path = str(item["path"])
        source = Path(path)
        if path in video_paths:
            try:
                stat = source.stat()
            except OSError as exc:
                raise ContractError(f"cannot stat source video {source}: {exc}") from exc
            if int(stat.st_size) != int(item["size_bytes"]):
                raise ContractError(f"source video size changed: {source}")
            continue
        current = fingerprint_file(source).as_dict()
        if current != dict(item):
            raise ContractError(f"forced appearance input changed: {source}")


def _reject_path_overlap(output_dir: Path, inputs: Sequence[Path]) -> None:
    output = output_dir.resolve()
    for source in inputs:
        resolved = source.resolve()
        if output == resolved or output in resolved.parents or resolved in output.parents:
            raise ContractError(f"forced appearance output overlaps input: {resolved}")


def _load_identity(
    ingest_dir: Path,
    production: ProductionInputBundle,
    config: ForcedAppearanceConfig,
) -> dict[str, np.ndarray]:
    path = ingest_dir / "detections.parquet"
    try:
        parquet = pq.ParquetFile(path)
        if not parquet.schema_arrow.equals(DETECTIONS_SCHEMA, check_metadata=False):
            raise ContractError("forced appearance S00 detections schema differs")
        table = pq.read_table(
            path,
            columns=[
                "det_id", "sequence_id", "clip_id", "local_frame", "global_frame",
                "global_time_sec", "x1", "y1", "x2", "y2", "valid",
            ],
            filters=[("valid", "=", True)],
        )
    except ContractError:
        raise
    except (OSError, pa.ArrowException, TypeError, ValueError) as exc:
        raise ContractError(f"cannot read forced appearance S00 detections: {exc}") from exc
    if table.num_rows != config.expected_valid_detection_count:
        raise ContractError("forced appearance valid detection count differs")
    det_ids = np.asarray(table["det_id"].combine_chunks().to_numpy(), dtype=np.int64)
    frames = np.asarray(table["global_frame"].combine_chunks().to_numpy(), dtype=np.int64)
    canonical = np.lexsort((det_ids, frames))

    def numeric(name: str, dtype: Any) -> np.ndarray:
        return np.asarray(table[name].combine_chunks().to_numpy(), dtype=dtype)[canonical]

    result = {
        "det_id": det_ids[canonical],
        "sequence_id": np.asarray(table["sequence_id"].to_pylist(), dtype=object)[canonical],
        "clip_id": np.asarray(table["clip_id"].to_pylist(), dtype=object)[canonical],
        "local_frame": numeric("local_frame", np.int32),
        "global_frame": frames[canonical],
        "global_time_sec": numeric("global_time_sec", np.float64),
        "x1": numeric("x1", np.float64),
        "y1": numeric("y1", np.float64),
        "x2": numeric("x2", np.float64),
        "y2": numeric("y2", np.float64),
    }
    detections = production.detections
    comparisons = (
        (result["det_id"], detections.det_ids),
        (result["clip_id"], detections.clip_ids),
        (result["global_frame"], detections.global_frames),
        (result["global_time_sec"], detections.global_time_sec),
        (result["x1"], detections.x1), (result["y1"], detections.y1),
        (result["x2"], detections.x2), (result["y2"], detections.y2),
    )
    if any(not np.array_equal(left, np.asarray(right)) for left, right in comparisons):
        raise ContractError("forced appearance S00/runtime detection join differs")
    if set(map(str, result["sequence_id"])) != {config.expected_sequence_id}:
        raise ContractError("forced appearance detection sequence differs")
    return result


def _stable_by_detection(
    production: ProductionInputBundle,
    stable: S04FinalizedBundle,
) -> np.ndarray:
    target_ids = np.asarray(production.detections.det_ids, dtype=np.int64)
    source_ids = np.asarray(stable.det_ids, dtype=np.int64)
    order = np.argsort(source_ids, kind="stable")
    sorted_ids = source_ids[order]
    locations = np.searchsorted(sorted_ids, target_ids)
    if (
        len(target_ids) != len(source_ids)
        or np.any(locations >= len(sorted_ids))
        or not np.array_equal(sorted_ids[locations], target_ids)
        or len(np.unique(source_ids)) != len(source_ids)
    ):
        raise ContractError("forced appearance S00/S04 detection mapping is not bijective")
    return np.asarray(stable.det_stable_ids, dtype=np.int64)[order[locations]]


def _sample_stable_ids(
    production: ProductionInputBundle,
    stable: S04FinalizedBundle,
) -> np.ndarray:
    calibration = production.calibration_input
    try:
        result = np.asarray(
            [stable.micro_to_stable[int(value)] for value in calibration.sample_micro_ids],
            dtype=np.int64,
        )
    except KeyError as exc:
        raise ContractError(f"forced appearance sample references unknown micro: {exc}") from exc
    return result


def _build_base_appearance(
    production: ProductionInputBundle,
    stable: S04FinalizedBundle,
    config: ForcedAppearanceConfig,
) -> ForcedAppearanceBundle:
    calibration = production.calibration_input
    existing_usable = np.asarray(
        [stable.stable_appearance[int(stable_id)].appearance_usable for stable_id in stable.stable_ids],
        dtype=np.bool_,
    )
    if int(np.count_nonzero(existing_usable)) != config.expected_clean_appearance_count:
        raise ContractError("forced appearance clean S04 count differs")
    return build_forced_appearance_base(
        stable_ids=np.asarray(stable.stable_ids, dtype=np.int64),
        existing_prototypes=np.asarray(stable.stable_prototypes),
        existing_prototype_mask=np.asarray(stable.stable_prototype_mask),
        existing_usable=existing_usable,
        sample_ids=np.asarray(calibration.sample_ids, dtype=np.int64),
        sample_embeddings=np.asarray(calibration.embeddings),
        sample_stable_ids=_sample_stable_ids(production, stable),
        sample_quality=np.asarray(calibration.sample_crop_quality, dtype=np.float32),
        sample_other_bbox_max_iou=np.asarray(
            calibration.sample_other_bbox_max_iou, dtype=np.float32
        ),
        sample_s02_inlier=np.asarray(calibration.sample_s02_inlier, dtype=np.bool_),
    )


def _manifest_and_video_fingerprints(
    manifest_path: Path,
    ingest_dir: Path,
    config: ForcedAppearanceConfig,
) -> tuple[dict[str, Path], tuple[FileFingerprint, ...]]:
    rows, manifest_hash = load_manifest(manifest_path)
    if (
        [row.sequence_id for row in rows]
        != [config.expected_sequence_id] * len(config.clip_order)
        or [row.clip_id for row in rows] != list(config.clip_order)
        or [row.clip_order for row in rows] != list(range(len(config.clip_order)))
    ):
        raise ContractError("forced appearance live manifest clip contract differs")
    resolved = _read_json(ingest_dir / "resolved_manifest.json", "S00 resolved manifest")
    expected = [
        {
            "sequence_id": row.sequence_id,
            "clip_order": row.clip_order,
            "clip_id": row.clip_id,
            "video_path": str(row.video_path),
            "bbox_csv_path": str(row.bbox_csv_path),
            "frame_index_base": row.frame_index_base,
            "bbox_format": row.bbox_format,
        }
        for row in rows
    ]
    if resolved != expected:
        raise ContractError("forced appearance live/S00 resolved manifests differ")
    s00 = _read_json(ingest_dir / "_SUCCESS.json", "S00 success marker")
    records = s00.get("input_fingerprints") if isinstance(s00, dict) else None
    if not isinstance(records, list) or s00.get("manifest_hash") != manifest_hash:
        raise ContractError("forced appearance S00 manifest fingerprint differs")
    by_path = {
        str(item.get("path")): item for item in records if isinstance(item, dict)
    }
    fingerprints: list[FileFingerprint] = [fingerprint_file(manifest_path)]
    video_paths: dict[str, Path] = {}
    for row in rows:
        path = row.video_path.resolve()
        record = by_path.get(str(path))
        if not isinstance(record, dict) or set(record) != {
            "path", "size_bytes", "mtime_ns", "sha256"
        }:
            raise ContractError(f"S00 lacks source video fingerprint: {path}")
        try:
            stat = path.stat()
        except OSError as exc:
            raise ContractError(f"cannot stat source video {path}: {exc}") from exc
        if (
            int(stat.st_size) != record["size_bytes"]
            or int(stat.st_mtime_ns) != record["mtime_ns"]
        ):
            raise ContractError(f"source video metadata changed since S00: {path}")
        fingerprints.append(
            FileFingerprint(str(path), int(record["size_bytes"]), str(record["sha256"]))
        )
        video_paths[row.clip_id] = path
    return video_paths, tuple(fingerprints)


def _load_prior_pairs(
    prior_dir: Path,
    stable: S04FinalizedBundle,
    config: ForcedAppearanceConfig,
) -> tuple[set[tuple[int, int]], tuple[FileFingerprint, ...]]:
    marker_path = prior_dir / "_SUCCESS.json"
    marker = _read_json(marker_path, "prior S05 finalize success marker")
    if not isinstance(marker, dict) or marker.get("stage") != "S05_FINALIZE":
        raise ContractError("prior global directory is not completed S05_FINALIZE")
    records = marker.get("output_fingerprints")
    if not isinstance(records, list):
        raise ContractError("prior S05 finalize marker lacks output fingerprints")
    by_name = {
        str(item.get("path")): item for item in records if isinstance(item, dict)
    }
    name = "stable_to_global.parquet"
    record = by_name.get(name)
    if not isinstance(record, dict) or set(record) != {"path", "size_bytes", "sha256"}:
        raise ContractError("prior S05 finalize marker lacks stable_to_global")
    path = prior_dir / name
    current = _output_fingerprint(path, prior_dir)
    if current != record:
        raise ContractError("prior stable_to_global fingerprint differs")
    try:
        parquet = pq.ParquetFile(path)
        if not parquet.schema_arrow.equals(STABLE_TO_GLOBAL_SCHEMA, check_metadata=False):
            raise ContractError("prior stable_to_global schema differs")
        table = pq.read_table(
            path,
            columns=["stable_id", "predecessor_stable_id", "id_status"],
        )
    except ContractError:
        raise
    except (OSError, pa.ArrowException, TypeError, ValueError) as exc:
        raise ContractError(f"cannot read prior stable_to_global: {exc}") from exc
    rows = table.to_pylist()
    if (
        len(rows) != config.expected_stable_track_count
        or [int(row["stable_id"]) for row in rows]
        != list(range(config.expected_stable_track_count))
        or set(str(row["id_status"]) for row in rows) != {"provisional"}
    ):
        raise ContractError("prior stable_to_global row contract differs")
    pairs: set[tuple[int, int]] = set()
    for row in rows:
        target = int(row["stable_id"])
        predecessor = row["predecessor_stable_id"]
        if predecessor is None:
            continue
        source = int(predecessor)
        left = stable.stable_tracklets[source]
        right = stable.stable_tracklets[target]
        if (
            left.end_global_frame >= right.start_global_frame
            or left.end_time_sec >= right.start_time_sec
        ):
            raise ContractError("prior S05 link violates strict temporal order")
        pairs.add((source, target))
    if (
        config.expected_prior_link_count is not None
        and len(pairs) != config.expected_prior_link_count
    ):
        raise ContractError(
            f"prior link count differs: {len(pairs)} != {config.expected_prior_link_count}"
        )
    return pairs, (fingerprint_file(marker_path), fingerprint_file(path))


def _plan_rescue_candidates(
    identity: Mapping[str, np.ndarray],
    production: ProductionInputBundle,
    stable_by_detection: np.ndarray,
    target_stable_ids: Sequence[int],
    config: ForcedAppearanceConfig,
) -> list[_RescueCandidate]:
    targets = np.asarray(tuple(target_stable_ids), dtype=np.int64)
    if not len(targets):
        return []
    selected_positions = np.flatnonzero(np.isin(stable_by_detection, targets))
    if not len(selected_positions):
        raise ContractError("forced appearance rescue targets have no detections")
    target_group = stable_by_detection[selected_positions]
    review = np.asarray(production.detections.review_excluded, dtype=np.bool_)[
        selected_positions
    ].copy()
    # Explicitly authorized degraded fallback: only a stable whose every row is
    # excluded may use excluded rows.  The original flag remains in the audit.
    for stable_id in targets:
        local = target_group == stable_id
        if not np.any(local):
            raise ContractError(f"rescue stable {stable_id} has no valid detection")
        if np.all(review[local]):
            review[local] = False
    local_selected = select_candidate_pool_indices(
        target_group,
        np.asarray(production.detections.global_time_sec)[selected_positions],
        np.asarray(production.detections.global_frames)[selected_positions],
        np.asarray(production.detections.other_bbox_max_iou)[selected_positions],
        review,
        period_sec=config.rescue_candidate_period_sec,
        max_samples=config.rescue_candidate_max_samples,
        endpoint_samples=config.rescue_candidate_endpoint_samples,
        preferred_max_iou=config.clean_max_other_bbox_iou,
    )
    positions = selected_positions[local_selected]
    observed = set(map(int, stable_by_detection[positions]))
    if observed != set(map(int, targets)):
        raise ContractError(
            f"rescue candidate plan did not cover every C stable: missing={sorted(set(map(int, targets))-observed)}"
        )
    boxes = np.column_stack(
        [np.asarray(identity[name], dtype=np.float64) for name in ("x1", "y1", "x2", "y2")]
    )
    area_percentile = bbox_area_percentiles(boxes)
    rows: list[_RescueCandidate] = []
    for position in map(int, positions):
        rows.append(
            _RescueCandidate(
                position=position,
                stable_id=int(stable_by_detection[position]),
                micro_id=int(production.detections.micro_ids[position]),
                det_id=int(production.detections.det_ids[position]),
                clip_id=str(production.detections.clip_ids[position]),
                local_frame=int(identity["local_frame"][position]),
                global_frame=int(production.detections.global_frames[position]),
                global_time_sec=float(production.detections.global_time_sec[position]),
                bbox=tuple(map(float, boxes[position])),
                other_iou=float(production.detections.other_bbox_max_iou[position]),
                area_percentile=float(area_percentile[position]),
                review_excluded=bool(production.detections.review_excluded[position]),
            )
        )
    rows.sort(key=lambda item: (item.clip_id, item.local_frame, item.stable_id, item.det_id))
    return rows


def _quality_one(
    crop_rgb: np.ndarray,
    candidate: _RescueCandidate,
    clipped: float,
    boundary: float,
    config: ForcedAppearanceConfig,
) -> tuple[float, float]:
    resized = cv2.resize(
        crop_rgb,
        (config.blur_measurement_size, config.blur_measurement_size),
        interpolation=cv2.INTER_AREA,
    )
    blur = laplacian_blur_score(np.ascontiguousarray(resized))
    quality = combine_crop_quality(
        np.asarray([candidate.other_iou], dtype=np.float64),
        np.asarray([clipped], dtype=np.float64),
        np.asarray([candidate.area_percentile], dtype=np.float64),
        np.asarray([blur], dtype=np.float64),
        np.asarray([boundary], dtype=np.float64),
        weights=config.quality_weights,
        blur_scale=config.blur_scale,
    )
    return float(quality[0]), float(blur)


def _decode_rescue_candidates(
    candidates: list[_RescueCandidate],
    video_paths: Mapping[str, Path],
    config: ForcedAppearanceConfig,
    *,
    logger: LogFn,
) -> None:
    by_clip: dict[str, dict[int, list[_RescueCandidate]]] = {}
    for row in candidates:
        by_clip.setdefault(row.clip_id, {}).setdefault(row.local_frame, []).append(row)
    for clip_id in config.clip_order:
        by_frame = by_clip.get(clip_id)
        if not by_frame:
            continue
        last_frame = max(by_frame)
        capture = open_raw_video_capture(video_paths[clip_id])
        last_report = time.monotonic()
        try:
            for local_frame in range(last_frame + 1):
                ok, frame_bgr = capture.read()
                if not ok or frame_bgr is None:
                    raise ContractError(
                        f"forced appearance cannot decode {clip_id} frame {local_frame}"
                    )
                if frame_bgr.shape != _RAW_FRAME_SHAPE or frame_bgr.dtype != np.uint8:
                    raise ContractError(
                        f"forced appearance decoded unexpected frame {clip_id}/{local_frame}: "
                        f"{frame_bgr.shape}/{frame_bgr.dtype}"
                    )
                for row in sorted(by_frame.get(local_frame, ()), key=lambda item: item.det_id):
                    crop_bgr, clipped, boundary = crop_with_padding(
                        frame_bgr, row.bbox, config.bbox_padding_ratio
                    )
                    crop_rgb = np.ascontiguousarray(crop_bgr[:, :, ::-1])
                    quality, blur = _quality_one(
                        crop_rgb, row, clipped, boundary, config
                    )
                    row.crop_quality = quality
                    row.clipped_fraction = float(clipped)
                    row.boundary_distance = float(boundary)
                    row.blur_score = blur
                    row.crop_rgb = crop_rgb
                now = time.monotonic()
                if now - last_report >= config.progress_interval_sec:
                    logger(
                        f"[s05-force-appearance-prepare] decode {clip_id}: "
                        f"{local_frame + 1:,}/{last_frame + 1:,} frames"
                    )
                    last_report = now
        finally:
            capture.release()
        logger(
            f"[s05-force-appearance-prepare] decoded {clip_id} once through "
            f"frame {last_frame:,}"
        )


def _select_and_embed_rescue(
    candidates: list[_RescueCandidate],
    config: ForcedAppearanceConfig,
    *,
    logger: LogFn,
) -> tuple[list[SupplementalStableEmbeddings], np.ndarray]:
    grouped: dict[int, list[_RescueCandidate]] = {}
    for row in candidates:
        if row.crop_rgb is None:
            raise ContractError("forced appearance rescue crop was not decoded")
        grouped.setdefault(row.stable_id, []).append(row)
    selected: list[_RescueCandidate] = []
    for stable_id in sorted(grouped):
        options = grouped[stable_id]
        passed = [row for row in options if row.crop_quality >= config.min_clean_crop_quality]
        pool = passed if passed else options
        if not pool:
            raise ContractError(f"forced appearance rescue stable {stable_id} has no crop")
        ranked = sorted(
            pool,
            key=lambda row: (
                -row.crop_quality,
                row.review_excluded,
                row.other_iou,
                row.global_frame,
                row.det_id,
            ),
        )[: config.rescue_samples_per_stable]
        for row in ranked:
            row.selected = True
        selected.extend(ranked)
    selected.sort(key=lambda row: (row.stable_id, row.global_frame, row.det_id))
    if not selected:
        raise ContractError("forced appearance selected no rescue crops")
    profile = encoder_profile_by_name(config.selected_encoder)
    if profile.embedding_dim != config.embedding_dim:
        raise ContractError("forced appearance encoder dimension differs")
    logger(
        f"[s05-force-appearance-prepare] loading {profile.name} on logical "
        f"{config.device}; "
        f"embedding {len(selected):,} rescue crops"
    )
    encoder = build_encoder(profile, device=config.device)
    raw = [row.crop_rgb for row in selected]
    if any(crop is None for crop in raw):
        raise ContractError("forced appearance selected rescue crop is missing")
    raw_crops = [np.asarray(crop) for crop in raw]
    masked = [soft_border_mask(crop) for crop in raw_crops]
    embeddings = embed_four_views(
        encoder, raw_crops, masked, batch_size=config.batch_size
    )
    if embeddings.shape != (len(selected), config.embedding_dim):
        raise ContractError("forced appearance rescue embedding shape differs")
    for embedding_row, row in enumerate(selected):
        row.embedding_row = embedding_row
        row.crop_rgb = None
    selected_by_stable: dict[int, list[int]] = {}
    for index, row in enumerate(selected):
        selected_by_stable.setdefault(row.stable_id, []).append(index)
    supplemental: list[SupplementalStableEmbeddings] = []
    for stable_id in sorted(selected_by_stable):
        indices = np.asarray(selected_by_stable[stable_id], dtype=np.int64)
        supplemental.append(
            SupplementalStableEmbeddings(
                stable_id=stable_id,
                sample_ids=1_000_000 + indices,
                embeddings=embeddings[indices],
                quality=np.asarray(
                    [selected[int(index)].crop_quality for index in indices],
                    dtype=np.float32,
                ),
            )
        )
    for row in candidates:
        row.crop_rgb = None
    return supplemental, embeddings.astype(np.float32, copy=False)


def _rescue_rows(candidates: Sequence[_RescueCandidate]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in sorted(candidates, key=lambda row: (row.stable_id, row.global_frame, row.det_id)):
        rows.append(
            {
                "stable_id": item.stable_id,
                "micro_id": item.micro_id,
                "det_id": item.det_id,
                "clip_id": item.clip_id,
                "local_frame": item.local_frame,
                "global_frame": item.global_frame,
                "global_time_sec": item.global_time_sec,
                "x1": item.bbox[0], "y1": item.bbox[1],
                "x2": item.bbox[2], "y2": item.bbox[3],
                "crop_quality": item.crop_quality,
                "other_bbox_max_iou": item.other_iou,
                "clipped_fraction": item.clipped_fraction,
                "bbox_area_percentile": item.area_percentile,
                "blur_score": item.blur_score,
                "distance_to_image_boundary": item.boundary_distance,
                "review_excluded": item.review_excluded,
                "quality_gate_passed": item.crop_quality >= 0.45,
                "selected_for_descriptor": item.selected,
                "selection_reason": (
                    "quality_ranked" if item.selected and item.crop_quality >= 0.45
                    else "best_degraded_fallback" if item.selected
                    else "candidate_not_selected"
                ),
                "embedding_row": item.embedding_row,
            }
        )
    if any(row["selected_for_descriptor"] != (row["embedding_row"] is not None) for row in rows):
        raise ContractError("forced appearance rescue selection/embedding mapping differs")
    return rows


def _solver_nodes(stable: S04FinalizedBundle) -> list[GlobalStableNode]:
    nodes: list[GlobalStableNode] = []
    for stable_id in map(int, stable.stable_ids):
        row = stable.stable_tracklets[stable_id]
        nodes.append(
            GlobalStableNode(
                stable_id=stable_id,
                start_clip_id=str(row.start_clip_id),
                end_clip_id=str(row.end_clip_id),
                start_global_frame=int(row.start_global_frame),
                end_global_frame=int(row.end_global_frame),
                start_time_sec=float(row.start_time_sec),
                end_time_sec=float(row.end_time_sec),
                num_microtracklets=int(row.num_microtracklets),
                num_detections=int(row.num_detections),
            )
        )
    return nodes


def _maximum_concurrent(nodes: Sequence[GlobalStableNode]) -> int:
    events = [
        event
        for node in nodes
        for event in (
            (int(node.start_global_frame), 0),
            (int(node.end_global_frame), 1),
        )
    ]
    concurrent = 0
    maximum = 0
    for _, kind in sorted(events):
        if kind == 0:
            concurrent += 1
            maximum = max(maximum, concurrent)
        else:
            concurrent -= 1
    if concurrent != 0:
        raise ContractError("forced full-sequence interval sweep is unbalanced")
    return maximum


def _solve_full_sequence_graph(
    nodes: Sequence[GlobalStableNode],
    center_embeddings: np.ndarray,
    grades: Sequence[str],
    prior_pairs: set[tuple[int, int]],
    config: ForcedAppearanceConfig,
    *,
    logger: LogFn,
) -> tuple[ForcedCandidateGraph, ForcedPathCoverResult]:
    label = (
        f"full-sequence ranks[0:{len(nodes)}) "
        f"clips={','.join(config.clip_order)}"
    )
    logger(
        f"[s05-force-appearance] {label}: retrieving top-"
        f"{config.source_top_k} strict-future appearance neighbors for all "
        f"{len(nodes):,} solver nodes"
    )
    with ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="forced-full-candidates"
    ) as pool:
        future = pool.submit(
            build_forced_candidate_graph,
            nodes,
            center_embeddings,
            grades,
            prior_pairs,
            top_k=config.source_top_k,
            cost_scale=config.appearance_cost_scale,
            grade_penalties=config.grade_penalty,
            prior_bonus=config.prior_link_bonus,
        )
        elapsed = 0.0
        while True:
            try:
                graph = future.result(timeout=config.progress_interval_sec)
                break
            except FutureTimeoutError:
                elapsed += config.progress_interval_sec
                logger(
                    f"[s05-force-appearance] {label}: candidate retrieval "
                    f"running; {elapsed:.0f}s elapsed"
                )
    if graph.max_concurrent > config.target_global_track_count:
        raise ContractError(
            "forced full-sequence exact target violates the concurrency lower "
            f"bound: {graph.max_concurrent} > "
            f"{config.target_global_track_count}"
        )
    logger(
        f"[s05-force-appearance] {label}: solving "
        f"{len(graph.candidates):,} candidates for exactly "
        f"{config.target_global_track_count} global paths; "
        f"hard lower bound={graph.max_concurrent}"
    )
    progress_label = (
        f"[s05-force-appearance] {label}: exact-"
        f"{config.target_global_track_count} fixed-flow min-cost solver"
    )
    result = solve_forced_fixed_path_cover(
        nodes,
        graph.edges,
        target_num_paths=config.target_global_track_count,
        certified_path_cover=graph.backbone_chains,
        progress_interval_sec=config.progress_interval_sec,
        progress_logger=logger,
        progress_label=progress_label,
    )
    logger(
        f"[s05-force-appearance] {label}: complete; "
        f"{result.num_selected_links:,} links, {result.num_paths} paths"
    )
    return graph, result


def _build_graph_and_cover(
    stable: S04FinalizedBundle,
    appearance: ForcedAppearanceBundle,
    prior_pairs: set[tuple[int, int]],
    config: ForcedAppearanceConfig,
    *,
    logger: LogFn,
) -> tuple[
    ForcedCandidateGraph,
    ForcedPathCoverResult,
    tuple[_SolvePassSummary, ...],
]:
    """Build one complete candidate graph and solve one exact-62 flow."""

    nodes = _solver_nodes(stable)
    if not config.clip_order:
        raise ContractError("forced full-sequence solve requires a clip order")
    stable_ids = np.asarray(appearance.stable_ids, dtype=np.int64)
    descriptor_centers = np.asarray(
        appearance.descriptor_centers, dtype=np.float32
    )
    if (
        len(stable_ids) != len(nodes)
        or not np.array_equal(
            stable_ids, np.asarray(stable.stable_ids, dtype=np.int64)
        )
        or len(appearance.rows) != len(nodes)
        or descriptor_centers.shape[0] != len(nodes)
    ):
        raise ContractError(
            "forced full-sequence appearance/stable alignment differs"
        )
    grades: list[str] = []
    for index, row in enumerate(appearance.rows):
        stable_id = int(row.get("stable_id", stable_ids[index]))
        if stable_id != int(stable_ids[index]):
            raise ContractError(
                "forced full-sequence appearance rows are not stable-ID aligned"
            )
        grades.append(str(row["evidence_grade"]))

    clip_index = {
        clip_id: index for index, clip_id in enumerate(config.clip_order)
    }
    if len(clip_index) != len(config.clip_order):
        raise ContractError("forced full-sequence clip order contains duplicates")
    for node in nodes:
        start_index = clip_index.get(str(node.start_clip_id))
        end_index = clip_index.get(str(node.end_clip_id))
        if start_index is None or end_index is None or start_index > end_index:
            raise ContractError(
                "forced full-sequence stable clip ownership is invalid"
            )

    target = config.target_global_track_count
    maximum_concurrent = _maximum_concurrent(nodes)
    if maximum_concurrent > target:
        raise ContractError(
            "forced full-sequence exact target violates the global concurrency "
            f"lower bound: {maximum_concurrent} > {target}"
        )
    graph, result = _solve_full_sequence_graph(
        nodes,
        descriptor_centers,
        grades,
        prior_pairs,
        config,
        logger=logger,
    )
    flattened_backbone = [
        stable_id for chain in graph.backbone_chains for stable_id in chain
    ]
    if (
        graph.max_concurrent != maximum_concurrent
        or graph.chain_count != maximum_concurrent
        or sorted(flattened_backbone) != list(map(int, stable_ids))
        or result.target_num_paths != target
        or result.required_links != len(nodes) - target
        or result.num_selected_links != result.required_links
        or result.maximum_feasible_links != len(nodes) - maximum_concurrent
    ):
        raise ContractError("forced full-sequence graph/cover contract differs")
    summary = _SolvePassSummary(
        pass_index=0,
        source_clip_ids=config.clip_order,
        solver_nodes=len(nodes),
        candidate_count=len(graph.candidates),
        interval_width=graph.max_concurrent,
        backbone_chain_count=graph.chain_count,
        maximum_feasible_links=result.maximum_feasible_links,
        required_links=result.required_links,
        selected_links=result.num_selected_links,
        total_appearance_cost_int=result.total_appearance_cost_int,
    )
    return graph, result, (summary,)


def _global_link_id(source_id: int, target_id: int) -> str:
    return f"s05fl-{source_id:06d}-{target_id:06d}"


def _candidate_rows(
    graph: ForcedCandidateGraph,
    result: ForcedPathCoverResult,
    stable: S04FinalizedBundle,
) -> list[dict[str, Any]]:
    selected = {
        (int(edge.source_stable_id), int(edge.target_stable_id))
        for edge in result.selected_edges
    }
    costs = dict(result.solver_cost_by_edge)
    rows: list[dict[str, Any]] = []
    for record in graph.candidates:
        edge = record.edge
        source_id = int(edge.source_stable_id)
        target_id = int(edge.target_stable_id)
        source = stable.stable_tracklets[source_id]
        target = stable.stable_tracklets[target_id]
        chosen = (source_id, target_id) in selected
        solver_cost = costs.get(edge.edge_id)
        if solver_cost is None:
            raise ContractError("forced appearance candidate lacks solver cost")
        rows.append(
            {
                "candidate_id": edge.edge_id,
                "source_stable_id": source_id,
                "target_stable_id": target_id,
                "source_end_clip_id": str(source.end_clip_id),
                "target_start_clip_id": str(target.start_clip_id),
                "source_end_global_frame": int(source.end_global_frame),
                "target_start_global_frame": int(target.start_global_frame),
                "source_end_time_sec": float(source.end_time_sec),
                "target_start_time_sec": float(target.start_time_sec),
                "temporal_gap_sec": float(target.start_time_sec - source.end_time_sec),
                "strictly_nonoverlapping": True,
                "appearance_cosine": float(record.cosine_similarity),
                "source_evidence_grade": record.source_grade,
                "target_evidence_grade": record.target_grade,
                "selected_by_source_topk": record.selected_by_source_topk,
                "selected_by_target_topk": record.selected_by_target_topk,
                "temporal_backbone": record.selected_by_backbone,
                "prior_global_link": record.selected_by_prior,
                "appearance_cost_int": int(record.base_cost_int),
                "solver_cost_int": int(solver_cost),
                "selected_by_solver": chosen,
                "global_link_id": _global_link_id(source_id, target_id) if chosen else None,
                "authorization_basis": _AUTHORIZATION,
                "id_status": _ID_STATUS if chosen else "candidate_only",
            }
        )
    if (
        len({row["candidate_id"] for row in rows}) != len(rows)
        or sum(bool(row["selected_by_solver"]) for row in rows) != result.required_links
    ):
        raise ContractError("forced appearance candidate audit coverage differs")
    return rows


def _path_clip_ids(
    path: Sequence[int], stable: S04FinalizedBundle, clip_order: Sequence[str]
) -> list[str]:
    order = {clip_id: index for index, clip_id in enumerate(clip_order)}
    used: set[str] = set()
    for stable_id in path:
        row = stable.stable_tracklets[int(stable_id)]
        left, right = order[row.start_clip_id], order[row.end_clip_id]
        if left > right:
            raise ContractError("forced appearance stable reverses clip order")
        used.update(clip_order[left : right + 1])
    return [clip_id for clip_id in clip_order if clip_id in used]


def _build_global_rows(
    stable: S04FinalizedBundle,
    graph: ForcedCandidateGraph,
    result: ForcedPathCoverResult,
    config: ForcedAppearanceConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[int, dict[str, Any]]]:
    records = {
        (int(record.edge.source_stable_id), int(record.edge.target_stable_id)): record
        for record in graph.candidates
    }
    selected_pairs = {
        (int(edge.source_stable_id), int(edge.target_stable_id))
        for edge in result.selected_edges
    }
    # ``result.paths`` is deterministically ordered by the full-sequence solver.
    # Any input-set change is a new global optimization and may renumber paths.
    paths = [tuple(map(int, path)) for path in result.paths]
    flattened = [stable_id for path in paths for stable_id in path]
    if (
        sorted(flattened) != list(map(int, stable.stable_ids))
        or len(flattened) != len(set(flattened))
        or len(paths) != config.target_global_track_count
    ):
        raise ContractError("forced appearance path cover is not total exact-62")
    mapping_rows: list[dict[str, Any]] = []
    global_rows: list[dict[str, Any]] = []
    by_stable: dict[int, dict[str, Any]] = {}
    for global_id, path in enumerate(paths):
        tracklets = [stable.stable_tracklets[stable_id] for stable_id in path]
        links = []
        for pair in zip(path, path[1:], strict=False):
            if pair not in selected_pairs or pair not in records:
                raise ContractError("forced appearance path adjacency lacks selected evidence")
            links.append(records[pair])
        for left, right in zip(tracklets, tracklets[1:], strict=False):
            if (
                left.end_global_frame >= right.start_global_frame
                or left.end_time_sec >= right.start_time_sec
            ):
                raise ContractError("forced appearance global path contains overlap")
        # Bind UUIDs to the complete-sequence optimization result.
        global_uuid = str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"cowtrack://{config.expected_sequence_id}/s05-force/full-sequence/"
                f"{global_id}/{path[0]}",
            )
        )
        display_id = f"G{global_id + 1:04d}"
        scores = [float(record.cosine_similarity) for record in links]
        detections = sum(int(row.num_detections) for row in tracklets)
        clip_ids = _path_clip_ids(path, stable, config.clip_order)
        global_rows.append(
            {
                "global_track_id": global_id,
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
                "num_microtracklets": sum(int(row.num_microtracklets) for row in tracklets),
                "num_detections": detections,
                "num_long_links": len(links),
                "duration_visible_sec": detections * _FRAME_DURATION_SEC,
                "min_link_probability": None,
                "p10_link_probability": None,
                "mean_link_probability": None,
                "max_link_probability": None,
                "min_link_margin": None,
                "appearance_consistency": float(np.mean(scores)) if scores else None,
                "identity_basis": _AUTHORIZATION,
                "id_status": _ID_STATUS,
                "spans_multiple_clips": len(clip_ids) > 1,
                "population_warning": False,
            }
        )
        for path_order, (stable_id, tracklet) in enumerate(zip(path, tracklets, strict=True)):
            predecessor_id = path[path_order - 1] if path_order else None
            predecessor = (
                None if predecessor_id is None else records[(predecessor_id, stable_id)]
            )
            cross_clip = bool(
                predecessor_id is not None
                and stable.stable_tracklets[predecessor_id].end_clip_id
                != tracklet.start_clip_id
            )
            mapping = {
                "stable_id": stable_id,
                "global_track_id": global_id,
                "global_track_uuid": global_uuid,
                "display_global_id": display_id,
                "order_in_global_path": path_order,
                "predecessor_stable_id": predecessor_id,
                "predecessor_candidate_id": None if predecessor is None else predecessor.edge.edge_id,
                "predecessor_proposal_id": None,
                "predecessor_global_link_id": (
                    None if predecessor is None else _global_link_id(predecessor_id, stable_id)
                ),
                "predecessor_link_probability": None,
                "predecessor_link_margin": None,
                "predecessor_authorization_basis": None if predecessor is None else _AUTHORIZATION,
                "link_type": (
                    "PATH_START" if predecessor is None
                    else "FORCED_FILE_BOUNDARY" if cross_clip
                    else "FORCED_APPEARANCE"
                ),
                "cross_clip_boundary": cross_clip,
                "component_num_stable_tracklets": len(path),
                "component_num_detections": detections,
                "component_num_long_links": len(links),
                "component_min_link_probability": None,
                "component_mean_link_probability": None,
                "component_max_link_probability": None,
                "identity_basis": _AUTHORIZATION,
                "id_status": _ID_STATUS,
            }
            mapping_rows.append(mapping)
            by_stable[stable_id] = mapping
    mapping_rows.sort(key=lambda row: int(row["stable_id"]))
    if (
        len(mapping_rows) != config.expected_stable_track_count
        or len(global_rows) != config.target_global_track_count
        or sum(int(row["num_stable_tracklets"]) for row in global_rows)
        != config.expected_stable_track_count
        or sum(int(row["num_microtracklets"]) for row in global_rows)
        != config.expected_microtrack_count
        or sum(int(row["num_detections"]) for row in global_rows)
        != config.expected_valid_detection_count
        or sum(int(row["num_long_links"]) for row in global_rows) != result.required_links
    ):
        raise ContractError("forced appearance global aggregate coverage differs")
    return mapping_rows, global_rows, by_stable


def _build_report(
    *,
    config: ForcedAppearanceConfig,
    config_hash: str,
    appearance: ForcedAppearanceBundle,
    rescue_rows: Sequence[Mapping[str, Any]],
    graph: ForcedCandidateGraph,
    result: ForcedPathCoverResult,
    solve_passes: Sequence[_SolvePassSummary],
    candidate_rows: Sequence[Mapping[str, Any]],
    global_rows: Sequence[Mapping[str, Any]],
    input_fingerprints: Sequence[Mapping[str, Any]],
    elapsed_sec: float,
) -> dict[str, Any]:
    grade_counts = Counter(str(row["evidence_grade"]) for row in appearance.rows)
    selected_candidates = [row for row in candidate_rows if row["selected_by_solver"]]
    selected_scores = [float(row["appearance_cosine"]) for row in selected_candidates]
    path_sizes = Counter(int(row["num_stable_tracklets"]) for row in global_rows)
    if (
        len(solve_passes) != 1
        or solve_passes[0].solver_nodes != len(appearance.stable_ids)
        or solve_passes[0].candidate_count != len(candidate_rows)
        or solve_passes[0].selected_links != result.num_selected_links
    ):
        raise ContractError("forced full-sequence report solve-pass contract differs")
    return {
        "schema_version": "1.0",
        "stage": _STAGE,
        "config_hash": config_hash,
        "execution_mode": config.execution_mode,
        "sequence_id": config.expected_sequence_id,
        "operator_approval": {
            "operator_approved": True,
            "authorization_basis": _AUTHORIZATION,
            "target_global_track_count": config.target_global_track_count,
            "existing_degraded_embeddings_allowed": True,
            "missing_embeddings_reencoded": True,
            "best_degraded_crop_allowed": True,
            "any_strictly_nonoverlapping_gap_allowed": True,
            "prior_links_locked": False,
            "prior_links_soft_preference_only": True,
        },
        "evidence_semantics": {
            "encoder": config.selected_encoder,
            "embedding_dim": config.embedding_dim,
            "certification_claimed": False,
            "all_global_ids_status": _ID_STATUS,
            "appearance_cosine_is_not_probability": True,
            "model_threshold_used": False,
        },
        "graded_appearance": {
            "counts_by_grade": dict(sorted(grade_counts.items())),
            "all_stable_tracks_have_descriptor": not appearance.zero_sample_stable_ids,
            "num_stable_tracks": len(appearance.stable_ids),
            "prototype_slots": int(appearance.prototypes.shape[1]),
        },
        "rescue": {
            "zero_s02_sample_stable_tracks": config.expected_no_s02_sample_count,
            "candidate_crops_decoded": len(rescue_rows),
            "selected_crops_encoded": sum(bool(row["selected_for_descriptor"]) for row in rescue_rows),
            "selected_best_degraded_crops": sum(
                row["selection_reason"] == "best_degraded_fallback" for row in rescue_rows
            ),
            "selected_review_excluded_crops": sum(
                bool(row["selected_for_descriptor"]) and bool(row["review_excluded"])
                for row in rescue_rows
            ),
        },
        "solver": {
            "solve_mode": _SOLVE_MODE,
            "algorithm": _SOLVER_ALGORITHM,
            "objective": _SOLVER_OBJECTIVE,
            "candidate_scope": "complete_sequence",
            "assignment": _ASSIGNMENT_ALGORITHM,
            "cardinality_certificate": "minimum_width_interval_backbone",
            "candidate_count": len(candidate_rows),
            "source_top_k": config.source_top_k,
            "target_top_k": config.target_top_k,
            "interval_width": graph.max_concurrent,
            "backbone_chain_count": graph.chain_count,
            "maximum_feasible_links": result.maximum_feasible_links,
            "required_links": result.required_links,
            "selected_links": result.num_selected_links,
            "global_paths": result.num_paths,
            "total_appearance_cost_int": result.total_appearance_cost_int,
            "input_prior_links": config.expected_prior_link_count,
            "candidate_prior_links": sum(
                bool(row["prior_global_link"]) for row in candidate_rows
            ),
            "selected_prior_links": sum(bool(row["prior_global_link"]) for row in selected_candidates),
            "selected_appearance_cosine": {
                "min": min(selected_scores),
                "p10": float(np.quantile(selected_scores, 0.1, method="linear")),
                "mean": float(np.mean(selected_scores)),
                "max": max(selected_scores),
            },
            "input_set_change_triggers_global_reoptimization": True,
            "stable_node_atomicity": "one_complete_sequence_graph_node_per_stable_id",
            "solve_pass_count": len(solve_passes),
            "solve_passes": [solve_pass.as_dict() for solve_pass in solve_passes],
        },
        "coverage": {
            "stable_tracks_mapped_once": config.expected_stable_track_count,
            "microtracklets_mapped_once": config.expected_microtrack_count,
            "valid_detections_mapped_once": config.expected_valid_detection_count,
            "invalid_detections_excluded": config.expected_invalid_detection_count,
            "same_frame_same_global_id_violations": 0,
            "temporal_overlap_link_violations": 0,
            "cycles": 0,
            "exact_global_track_count": len(global_rows),
        },
        "global_path_size_distribution": {
            str(size): count for size, count in sorted(path_sizes.items())
        },
        "input_fingerprints": list(input_fingerprints),
        "elapsed_sec": float(elapsed_sec),
    }


def _artifact_names(config: ForcedAppearanceConfig) -> dict[str, str]:
    return {
        name: getattr(config.artifacts, name)
        for name in config.artifacts.__dataclass_fields__
    }


def _validate_completed_output(
    output_dir: Path,
    marker: Mapping[str, Any],
    *,
    config: ForcedAppearanceConfig,
    config_hash: str,
    input_fingerprints: Sequence[Mapping[str, Any]],
) -> None:
    required_marker = {
        "schema_version", "stage", "config_hash", "execution_mode",
        "operator_approved", "authorization_basis", "certification_claimed",
        "target_global_track_count", "stats", "input_fingerprints",
        "output_fingerprints", "elapsed_sec",
    }
    if set(marker) != required_marker:
        raise ContractError("completed forced appearance marker keys differ")
    if (
        marker.get("schema_version") != "1.0"
        or marker.get("stage") != _STAGE
        or marker.get("config_hash") != config_hash
        or marker.get("execution_mode") != config.execution_mode
        or marker.get("operator_approved") is not True
        or marker.get("authorization_basis") != _AUTHORIZATION
        or marker.get("certification_claimed") is not False
        or marker.get("target_global_track_count") != config.target_global_track_count
        or marker.get("input_fingerprints") != list(input_fingerprints)
    ):
        raise ContractError("completed forced appearance marker contract differs")
    records = marker.get("output_fingerprints")
    if not isinstance(records, list):
        raise ContractError("completed forced appearance marker lacks output fingerprints")
    names = _artifact_names(config)
    expected_names = set(names.values()) - {config.artifacts.success}
    by_name = {
        str(item.get("path")): item for item in records if isinstance(item, dict)
    }
    if set(by_name) != expected_names:
        raise ContractError("completed forced appearance output artifact set differs")
    for name in sorted(expected_names):
        if _output_fingerprint(output_dir / name, output_dir) != by_name[name]:
            raise ContractError(f"completed forced appearance artifact changed: {name}")
    parquet_contracts = {
        config.artifacts.rescue_samples: (RESCUE_SAMPLES_SCHEMA, None),
        config.artifacts.graded_stable_appearance: (
            GRADED_STABLE_APPEARANCE_SCHEMA, config.expected_stable_track_count
        ),
        config.artifacts.candidate_edges: (FORCED_CANDIDATE_EDGES_SCHEMA, None),
        config.artifacts.stable_to_global: (STABLE_TO_GLOBAL_SCHEMA, config.expected_stable_track_count),
        config.artifacts.global_tracks: (GLOBAL_TRACKS_SCHEMA, config.target_global_track_count),
        config.artifacts.det_to_global: (DET_TO_GLOBAL_SCHEMA, config.expected_valid_detection_count),
    }
    for name, (schema, expected_rows) in parquet_contracts.items():
        parquet = pq.ParquetFile(output_dir / name)
        if not parquet.schema_arrow.equals(schema, check_metadata=False) or (
            expected_rows is not None and parquet.metadata.num_rows != expected_rows
        ):
            raise ContractError(f"completed forced appearance Parquet differs: {name}")
    prototypes = np.load(output_dir / config.artifacts.graded_prototypes, mmap_mode="r")
    mask = np.load(output_dir / config.artifacts.graded_prototype_mask, mmap_mode="r")
    rescue = np.load(output_dir / config.artifacts.rescue_embeddings, mmap_mode="r")
    if (
        prototypes.shape != (config.expected_stable_track_count, 3, config.embedding_dim)
        or prototypes.dtype != np.float16
        or mask.shape != (config.expected_stable_track_count, 3)
        or mask.dtype != np.bool_
        or rescue.ndim != 2
        or rescue.shape[1] != config.embedding_dim
        or rescue.dtype != np.float16
        or not np.all(np.any(mask, axis=1))
    ):
        raise ContractError("completed forced appearance array contract differs")
    report = _read_json(output_dir / config.artifacts.report, "forced appearance report")
    solver = report.get("solver") if isinstance(report, dict) else None
    if (
        not isinstance(report, dict)
        or report.get("stage") != _STAGE
        or report.get("config_hash") != config_hash
        or report.get("coverage", {}).get("exact_global_track_count")
        != config.target_global_track_count
        or not isinstance(solver, dict)
        or solver.get("solve_mode") != _SOLVE_MODE
        or solver.get("algorithm") != _SOLVER_ALGORITHM
        or solver.get("objective") != _SOLVER_OBJECTIVE
        or solver.get("candidate_scope") != "complete_sequence"
        or solver.get("assignment") != _ASSIGNMENT_ALGORITHM
        or solver.get("cardinality_certificate")
        != "minimum_width_interval_backbone"
        or solver.get("solve_pass_count") != 1
    ):
        raise ContractError("completed forced appearance report differs")


def _validate_fixed_inputs(
    production: ProductionInputBundle,
    stable: S04FinalizedBundle,
    config: ForcedAppearanceConfig,
) -> None:
    if (
        len(production.detections.det_ids) != config.expected_valid_detection_count
        or len(stable.stable_ids) != config.expected_stable_track_count
        or len(stable.micro_ids) != config.expected_microtrack_count
        or len(stable.det_ids) != config.expected_valid_detection_count
        or sum(row.appearance_usable for row in stable.stable_appearance.values())
        != config.expected_clean_appearance_count
    ):
        raise ContractError("forced appearance fixed upstream counts differ")
    choice = production.encoder_choice
    if (
        choice.get("selected_profile") != config.selected_encoder
        or choice.get("embedding_dim") != config.embedding_dim
        or choice.get("no_runtime_fallback") is not True
    ):
        raise ContractError("forced appearance S02 encoder contract differs")


def _load_common_forced_inputs(
    manifest_path: Path,
    ingest_dir: Path,
    microtrack_dir: Path,
    appearance_dir: Path,
    stable_dir: Path,
    config_path: Path,
    *,
    device: str | None,
    load_identity: bool,
    logger: LogFn,
) -> _LoadedForcedInputs:
    """Load and fingerprint every input that affects graded descriptors.

    ``device`` is required only by the preparation phase.  Passing ``None``
    keeps the exact-link solve independent of CUDA after the immutable
    descriptor cache has been validated.
    """

    config_fingerprint = fingerprint_file(config_path)
    config, config_payload, config_hash = load_forced_appearance_config(config_path)
    if fingerprint_file(config_path) != config_fingerprint:
        raise ContractError("forced appearance config changed while loading")
    if device is not None:
        if device != config.device:
            raise ContractError(f"forced appearance preparation requires --device {config.device}")
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
            raise ContractError(
                "forced appearance preparation requires command environment "
                "CUDA_VISIBLE_DEVICES=1"
            )

    logger("[s05-force-appearance] validating manifest and immutable source videos")
    video_paths, video_fingerprints = _manifest_and_video_fingerprints(
        manifest_path, ingest_dir, config
    )
    logger("[s05-force-appearance] strict-loading immutable S00/S01/S02 inputs")
    production = load_production_inputs(
        ingest_dir, microtrack_dir, appearance_dir, logger=logger
    )
    logger("[s05-force-appearance] strict-loading all finalized S04 stable tracks")
    stable = load_s04_finalized(stable_dir)
    observed_stable_count = len(stable.stable_ids)
    observed_micro_count = len(stable.micro_ids)
    observed_clean_count = sum(
        row.appearance_usable for row in stable.stable_appearance.values()
    )
    observed_missing_count = observed_stable_count - observed_clean_count
    configured_counts = (
        (config.expected_stable_track_count, observed_stable_count),
        (config.expected_microtrack_count, observed_micro_count),
        (config.expected_clean_appearance_count, observed_clean_count),
        (config.expected_missing_clean_appearance_count, observed_missing_count),
    )
    if any(expected not in (None, observed) for expected, observed in configured_counts):
        raise ContractError("forced appearance configured S04 counts differ")
    config = replace(
        config,
        expected_stable_track_count=observed_stable_count,
        expected_microtrack_count=observed_micro_count,
        expected_clean_appearance_count=observed_clean_count,
        expected_missing_clean_appearance_count=observed_missing_count,
    )
    _validate_fixed_inputs(production, stable, config)

    encoder_profile = encoder_profile_by_name(config.selected_encoder)
    encoder_fingerprint = fingerprint_file(encoder_profile.checkpoint_path)
    if (
        str(encoder_profile.checkpoint_path.resolve())
        != str(production.encoder_choice.get("checkpoint_path"))
        or encoder_fingerprint.sha256
        != production.encoder_choice.get("checkpoint_sha256")
    ):
        raise ContractError("forced appearance local encoder checkpoint differs from S02")
    identity = _load_identity(ingest_dir, production, config) if load_identity else None
    input_fingerprints = tuple(
        _normalize_fingerprints(
            (
                config_fingerprint,
                *video_fingerprints,
                *production.input_fingerprints,
                *stable.input_fingerprints,
                encoder_fingerprint,
                fingerprint_file(ingest_dir / "resolved_manifest.json"),
            )
        )
    )
    video_path_set = frozenset(str(path.resolve()) for path in video_paths.values())
    _verify_regular_fingerprints(input_fingerprints, set(video_path_set))
    return _LoadedForcedInputs(
        config=config,
        config_payload=config_payload,
        config_hash=config_hash,
        production=production,
        stable=stable,
        identity=identity,
        video_paths=MappingProxyType(dict(video_paths)),
        input_fingerprints=input_fingerprints,
        video_path_set=video_path_set,
    )


def _prepared_artifact_names(config: ForcedAppearanceConfig) -> dict[str, str]:
    names = {
        "rescue_samples": config.artifacts.rescue_samples,
        "rescue_embeddings": config.artifacts.rescue_embeddings,
        "graded_stable_appearance": config.artifacts.graded_stable_appearance,
        "graded_prototypes": config.artifacts.graded_prototypes,
        "graded_prototype_mask": config.artifacts.graded_prototype_mask,
        "descriptor_centers": _PREPARED_DESCRIPTOR_CENTERS,
        "effective_config": config.artifacts.effective_config,
    }
    if len(set(names.values())) != len(names) or _PREPARE_MARKER in names.values():
        raise ContractError("forced appearance prepared artifact names are not unique")
    return names


def _prepared_stats(
    appearance: ForcedAppearanceBundle,
    rescue_rows: Sequence[Mapping[str, Any]],
    rescue_embeddings: np.ndarray,
) -> dict[str, int]:
    c_ids = {
        int(row["stable_id"])
        for row in appearance.rows
        if str(row["evidence_grade"]) == GRADE_C_SUPPLEMENTAL
    }
    return {
        "num_stable_tracks": len(appearance.stable_ids),
        "num_rescue_stable_tracks": len(c_ids),
        "num_rescue_candidate_rows": len(rescue_rows),
        "num_rescue_embeddings": len(rescue_embeddings),
    }


def _load_prepared_appearance(
    prepared_dir: Path,
    *,
    config: ForcedAppearanceConfig,
    config_payload: Mapping[str, Any],
    config_hash: str,
    input_fingerprints: Sequence[Mapping[str, Any]],
) -> tuple[_PreparedAppearance, dict[str, Any]]:
    """Strictly restore a complete preparation cache with no recomputation."""

    marker_path = prepared_dir / _PREPARE_MARKER
    marker = _read_json(marker_path, "forced appearance prepared marker")
    required_marker = {
        "schema_version",
        "stage",
        "algorithm",
        "config_hash",
        "execution_mode",
        "identity_assignment_performed",
        "stats",
        "input_fingerprints",
        "output_fingerprints",
        "elapsed_sec",
    }
    if not isinstance(marker, dict) or set(marker) != required_marker:
        raise ContractError("forced appearance prepared marker keys differ")
    elapsed = marker.get("elapsed_sec")
    if (
        marker.get("schema_version") != "1.0"
        or marker.get("stage") != _PREPARE_STAGE
        or marker.get("algorithm") != _PREPARE_ALGORITHM
        or marker.get("config_hash") != config_hash
        or marker.get("execution_mode") != config.execution_mode
        or marker.get("identity_assignment_performed") is not False
        or marker.get("input_fingerprints") != list(input_fingerprints)
        or isinstance(elapsed, bool)
        or not isinstance(elapsed, (int, float))
        or not math.isfinite(float(elapsed))
        or float(elapsed) < 0.0
    ):
        raise ContractError("forced appearance prepared marker contract differs")

    names = _prepared_artifact_names(config)
    records = marker.get("output_fingerprints")
    if not isinstance(records, list) or any(
        not isinstance(item, dict)
        or set(item) != {"path", "size_bytes", "sha256"}
        for item in records
    ):
        raise ContractError("forced appearance prepared fingerprints differ")
    record_names = [str(item["path"]) for item in records]
    by_name = {str(item["path"]): item for item in records}
    if (
        record_names != sorted(names.values())
        or len(by_name) != len(records)
        or set(by_name) != set(names.values())
    ):
        raise ContractError("forced appearance prepared artifact set differs")
    for name in sorted(by_name):
        if _output_fingerprint(prepared_dir / name, prepared_dir) != by_name[name]:
            raise ContractError(f"forced appearance prepared artifact changed: {name}")

    if _read_json(
        prepared_dir / names["effective_config"],
        "forced appearance prepared effective config",
    ) != dict(config_payload):
        raise ContractError("forced appearance prepared effective config differs")
    try:
        rescue_file = pq.ParquetFile(prepared_dir / names["rescue_samples"])
        graded_file = pq.ParquetFile(prepared_dir / names["graded_stable_appearance"])
        if not rescue_file.schema_arrow.equals(
            RESCUE_SAMPLES_SCHEMA, check_metadata=False
        ) or not graded_file.schema_arrow.equals(
            GRADED_STABLE_APPEARANCE_SCHEMA, check_metadata=False
        ):
            raise ContractError("forced appearance prepared Parquet schema differs")
        rescue_rows = tuple(
            pq.read_table(prepared_dir / names["rescue_samples"]).to_pylist()
        )
        graded_rows = tuple(
            pq.read_table(prepared_dir / names["graded_stable_appearance"]).to_pylist()
        )
        prototypes = np.load(
            prepared_dir / names["graded_prototypes"],
            allow_pickle=False,
            mmap_mode="r",
        )
        prototype_mask = np.load(
            prepared_dir / names["graded_prototype_mask"],
            allow_pickle=False,
            mmap_mode="r",
        )
        descriptor_centers = np.load(
            prepared_dir / names["descriptor_centers"],
            allow_pickle=False,
            mmap_mode="r",
        )
        stored_rescue_embeddings = np.load(
            prepared_dir / names["rescue_embeddings"],
            allow_pickle=False,
            mmap_mode="r",
        )
    except ContractError:
        raise
    except (OSError, ValueError, pa.ArrowException) as exc:
        raise ContractError(f"cannot load forced appearance prepared artifacts: {exc}") from exc

    if graded_file.metadata.num_rows != config.expected_stable_track_count:
        raise ContractError("forced appearance prepared graded row count differs")
    stable_ids = np.asarray(
        [int(row["stable_id"]) for row in graded_rows], dtype=np.int64
    )
    appearance = restore_completed_forced_appearance(
        stable_ids=stable_ids,
        prototypes=np.asarray(prototypes),
        prototype_mask=np.asarray(prototype_mask),
        descriptor_centers=np.asarray(descriptor_centers),
        rows=graded_rows,
    )
    allowed_grades = {
        GRADE_A_CLEAN,
        GRADE_B_EXISTING_DEGRADED,
        GRADE_C_SUPPLEMENTAL,
    }
    if (
        {str(row["evidence_grade"]) for row in graded_rows} - allowed_grades
        or any(
            not bool(row["descriptor_usable"])
            or row["missing_reason"] is not None
            for row in graded_rows
        )
    ):
        raise ContractError("forced appearance prepared evidence grades differ")
    rescue_embeddings = np.array(stored_rescue_embeddings, copy=True)
    if (
        rescue_embeddings.dtype != np.float16
        or rescue_embeddings.ndim != 2
        or rescue_embeddings.shape[1] != config.embedding_dim
        or not np.all(np.isfinite(rescue_embeddings))
    ):
        raise ContractError("forced appearance prepared rescue embeddings differ")
    if len(rescue_embeddings) and not np.allclose(
        np.linalg.norm(rescue_embeddings.astype(np.float32), axis=1),
        1.0,
        rtol=0.0,
        atol=2e-3,
    ):
        raise ContractError("forced appearance prepared rescue embeddings are not normalized")

    c_ids = {
        int(row["stable_id"])
        for row in graded_rows
        if str(row["evidence_grade"]) == GRADE_C_SUPPLEMENTAL
    }
    rescue_ids = {int(row["stable_id"]) for row in rescue_rows}
    selected_rows = [
        row for row in rescue_rows if bool(row["selected_for_descriptor"])
    ]
    embedding_rows = [int(row["embedding_row"]) for row in selected_rows]
    c_selected_ids = {
        int(row["stable_id"]): tuple(
            sorted(int(sample_id) for sample_id in row["selected_sample_ids"])
        )
        for row in graded_rows
        if str(row["evidence_grade"]) == GRADE_C_SUPPLEMENTAL
    }
    rescue_selected_ids = {
        stable_id: tuple(
            sorted(
                1_000_000 + int(row["embedding_row"])
                for row in selected_rows
                if int(row["stable_id"]) == stable_id
            )
        )
        for stable_id in c_ids
    }
    if (
        any(
            bool(row["selected_for_descriptor"])
            != (row["embedding_row"] is not None)
            for row in rescue_rows
        )
        or sorted(embedding_rows) != list(range(len(rescue_embeddings)))
        or rescue_ids != c_ids
        or {int(row["stable_id"]) for row in selected_rows} != c_ids
        or c_selected_ids != rescue_selected_ids
    ):
        raise ContractError("forced appearance prepared rescue mapping differs")
    no_s02_count = len(c_ids)
    if config.expected_no_s02_sample_count not in (None, no_s02_count):
        raise ContractError("forced appearance prepared C-grade count differs")
    expected_stats = _prepared_stats(appearance, rescue_rows, rescue_embeddings)
    if marker.get("stats") != expected_stats:
        raise ContractError("forced appearance prepared statistics differ")

    consumed = tuple(
        fingerprint_file(prepared_dir / name)
        for name in sorted(names.values())
    ) + (fingerprint_file(marker_path),)
    return (
        _PreparedAppearance(
            appearance=appearance,
            rescue_rows=tuple(MappingProxyType(dict(row)) for row in rescue_rows),
            rescue_embeddings=rescue_embeddings,
            no_s02_sample_count=no_s02_count,
            input_fingerprints=consumed,
        ),
        marker,
    )


def _write_prepared_output(
    output_dir: Path,
    *,
    config: ForcedAppearanceConfig,
    config_payload: Mapping[str, Any],
    config_hash: str,
    input_fingerprints: Sequence[Mapping[str, Any]],
    appearance: ForcedAppearanceBundle,
    rescue_rows: Sequence[Mapping[str, Any]],
    rescue_embeddings: np.ndarray,
    elapsed_sec: float,
) -> dict[str, Any]:
    """Atomically publish and then semantically revalidate preparation data."""

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = output_dir.parent / f".{output_dir.name}.staging-{os.getpid()}"
    if staging.exists():
        raise ContractError(f"forced appearance preparation staging exists: {staging}")
    staging.mkdir(parents=False)
    try:
        names = _prepared_artifact_names(config)
        _write_table(
            staging / names["rescue_samples"],
            _rows_table(rescue_rows, RESCUE_SAMPLES_SCHEMA, "forced rescue samples"),
            RESCUE_SAMPLES_SCHEMA,
            config.parquet_compression,
        )
        _write_table(
            staging / names["graded_stable_appearance"],
            _rows_table(
                appearance.rows,
                GRADED_STABLE_APPEARANCE_SCHEMA,
                "forced graded stable appearance",
            ),
            GRADED_STABLE_APPEARANCE_SCHEMA,
            config.parquet_compression,
        )
        np.save(
            staging / names["rescue_embeddings"],
            np.asarray(rescue_embeddings, dtype=np.float16),
            allow_pickle=False,
        )
        np.save(
            staging / names["graded_prototypes"],
            np.asarray(appearance.prototypes, dtype=np.float16),
            allow_pickle=False,
        )
        np.save(
            staging / names["graded_prototype_mask"],
            np.asarray(appearance.prototype_mask, dtype=np.bool_),
            allow_pickle=False,
        )
        np.save(
            staging / names["descriptor_centers"],
            np.asarray(appearance.descriptor_centers, dtype=np.float32),
            allow_pickle=False,
        )
        _write_json(staging / names["effective_config"], config_payload)
        output_fingerprints = [
            _output_fingerprint(staging / name, staging)
            for name in sorted(names.values())
        ]
        marker = {
            "schema_version": "1.0",
            "stage": _PREPARE_STAGE,
            "algorithm": _PREPARE_ALGORITHM,
            "config_hash": config_hash,
            "execution_mode": config.execution_mode,
            "identity_assignment_performed": False,
            "stats": _prepared_stats(appearance, rescue_rows, rescue_embeddings),
            "input_fingerprints": list(input_fingerprints),
            "output_fingerprints": output_fingerprints,
            "elapsed_sec": float(elapsed_sec),
        }
        _write_json(staging / _PREPARE_MARKER, marker)
        _load_prepared_appearance(
            staging,
            config=config,
            config_payload=config_payload,
            config_hash=config_hash,
            input_fingerprints=input_fingerprints,
        )
        try:
            os.replace(staging, output_dir)
        except OSError as exc:
            raise ContractError(
                f"forced appearance preparation cannot be atomically committed: {exc}"
            ) from exc
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return marker


def _canonical_preparation_config(payload: Any, label: str) -> str:
    """Return a typed canonical config projection that excludes solve policy."""

    if not isinstance(payload, dict) or "solver" not in payload:
        raise ContractError(f"{label} must be a complete config object")
    if not isinstance(payload["solver"], dict):
        raise ContractError(f"{label} solver section must be an object")
    projection = {
        key: value for key, value in payload.items() if key != "solver"
    }
    try:
        return json.dumps(
            projection,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise ContractError(f"cannot canonicalize {label}: {exc}") from exc


def _rekey_prepared_input_fingerprints(
    records: Any,
    *,
    config_path: Path,
    source_config_hash: str,
    target_config_fingerprint: FileFingerprint,
) -> list[dict[str, Any]]:
    """Replace exactly the config record while preserving every upstream byte ID."""

    if not isinstance(records, list) or not records:
        raise ContractError("prepared repackage source input fingerprints differ")
    normalized: list[dict[str, Any]] = []
    for item in records:
        if (
            not isinstance(item, dict)
            or set(item) != {"path", "size_bytes", "sha256"}
            or not isinstance(item["path"], str)
            or not item["path"]
            or isinstance(item["size_bytes"], bool)
            or not isinstance(item["size_bytes"], int)
            or int(item["size_bytes"]) < 0
            or not isinstance(item["sha256"], str)
            or len(item["sha256"]) != 64
            or any(character not in "0123456789abcdef" for character in item["sha256"])
        ):
            raise ContractError(
                "prepared repackage source input fingerprint record differs"
            )
        normalized.append(dict(item))
    paths = [str(item["path"]) for item in normalized]
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise ContractError(
            "prepared repackage source input fingerprint ordering differs"
        )
    resolved_config = str(config_path.resolve())
    matching = [
        index
        for index, item in enumerate(normalized)
        if str(item["path"]) == resolved_config
    ]
    if len(matching) != 1:
        raise ContractError(
            "prepared repackage must find exactly one source config fingerprint"
        )
    index = matching[0]
    if str(normalized[index]["sha256"]) != source_config_hash:
        raise ContractError(
            "prepared repackage source config fingerprint/hash differs"
        )
    current = target_config_fingerprint.as_dict()
    if str(current["path"]) != resolved_config:
        raise ContractError("prepared repackage target config path differs")
    normalized[index] = current
    if [str(item["path"]) for item in normalized] != sorted(
        str(item["path"]) for item in normalized
    ):
        raise ContractError("prepared repackage target input ordering differs")
    return normalized


def run_s05_force_appearance_repackage_prepared(
    source_prepared_dir: Path,
    config_path: Path,
    output_dir: Path,
    *,
    logger: LogFn = log,
) -> dict[str, Any]:
    """Atomically re-key immutable descriptors after a solve-only config change."""

    if not callable(logger):
        raise ContractError("forced appearance prepared repackage logger must be callable")
    source_prepared_dir, config_path, output_dir = (
        path.resolve()
        for path in (source_prepared_dir, config_path, output_dir)
    )
    _reject_path_overlap(output_dir, (source_prepared_dir, config_path))
    if output_dir.exists():
        raise ContractError(
            "forced appearance prepared repackage output must not already exist: "
            f"{output_dir}"
        )

    target_config_fingerprint = fingerprint_file(config_path)
    config, target_payload, target_config_hash = load_forced_appearance_config(
        config_path
    )
    if fingerprint_file(config_path) != target_config_fingerprint:
        raise ContractError(
            "forced appearance config changed while preparing repackage"
        )

    marker_path = source_prepared_dir / _PREPARE_MARKER
    if marker_path.is_symlink():
        raise ContractError("prepared repackage source marker cannot be a symlink")
    source_marker = _read_json(
        marker_path, "forced appearance prepared repackage source marker"
    )
    if not isinstance(source_marker, dict):
        raise ContractError("prepared repackage source marker must be an object")
    source_config_hash = source_marker.get("config_hash")
    stats = source_marker.get("stats")
    expected_stats = {
        "num_stable_tracks",
        "num_rescue_stable_tracks",
        "num_rescue_candidate_rows",
        "num_rescue_embeddings",
    }
    if (
        not isinstance(source_config_hash, str)
        or len(source_config_hash) != 64
        or any(
            character not in "0123456789abcdef"
            for character in source_config_hash
        )
        or source_config_hash == target_config_hash
        or not isinstance(stats, dict)
        or set(stats) != expected_stats
        or any(
            isinstance(stats[name], bool)
            or not isinstance(stats[name], int)
            or int(stats[name]) < 0
            for name in expected_stats
        )
    ):
        raise ContractError("prepared repackage source marker contract differs")

    names = _prepared_artifact_names(config)
    for name in (*names.values(), _PREPARE_MARKER):
        path = source_prepared_dir / name
        if path.is_symlink() or not path.is_file():
            raise ContractError(
                f"prepared repackage source artifact is not a regular file: {name}"
            )
    source_payload = _read_json(
        source_prepared_dir / names["effective_config"],
        "forced appearance prepared repackage source config",
    )
    if (
        not isinstance(source_payload, dict)
        or set(source_payload) != set(target_payload)
        or _canonical_preparation_config(source_payload, "source prepared config")
        != _canonical_preparation_config(target_payload, "target prepared config")
    ):
        raise ContractError(
            "prepared repackage allows changes only within the solver section"
        )

    source_inputs = source_marker.get("input_fingerprints")
    target_inputs = _rekey_prepared_input_fingerprints(
        source_inputs,
        config_path=config_path,
        source_config_hash=source_config_hash,
        target_config_fingerprint=target_config_fingerprint,
    )
    validation_config = replace(
        config,
        expected_stable_track_count=int(stats["num_stable_tracks"]),
        expected_no_s02_sample_count=int(stats["num_rescue_stable_tracks"]),
    )
    logger(
        "[s05-force-appearance-repackage-prepared] validating source hashes, "
        "schemas, arrays, and descriptor semantics"
    )
    source_prepared, validated_source_marker = _load_prepared_appearance(
        source_prepared_dir,
        config=validation_config,
        config_payload=source_payload,
        config_hash=source_config_hash,
        input_fingerprints=source_inputs,
    )
    if validated_source_marker != source_marker or (
        source_prepared.appearance.descriptor_centers.ndim != 2
        or source_prepared.appearance.descriptor_centers.shape[1]
        != config.embedding_dim
        or source_prepared.appearance.prototypes.shape[2]
        != config.embedding_dim
    ):
        raise ContractError("prepared repackage source descriptor contract differs")

    source_records = source_marker.get("output_fingerprints")
    if not isinstance(source_records, list):
        raise ContractError("prepared repackage source output fingerprints differ")
    source_by_name = {
        str(item["path"]): dict(item)
        for item in source_records
        if isinstance(item, dict) and "path" in item
    }
    descriptor_names = sorted(
        name
        for key, name in names.items()
        if key != "effective_config"
    )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = output_dir.parent / f".{output_dir.name}.staging-{os.getpid()}"
    if staging.exists():
        raise ContractError(
            f"forced appearance prepared repackage staging exists: {staging}"
        )
    staging.mkdir(parents=False)
    try:
        logger(
            "[s05-force-appearance-repackage-prepared] copying six validated "
            "descriptor artifacts without recomputation"
        )
        for name in descriptor_names:
            shutil.copyfile(source_prepared_dir / name, staging / name)
            copied = _output_fingerprint(staging / name, staging)
            if copied != source_by_name.get(name):
                raise ContractError(
                    f"prepared repackage descriptor bytes changed while copying: {name}"
                )
        _write_json(staging / names["effective_config"], target_payload)
        output_fingerprints = [
            _output_fingerprint(staging / name, staging)
            for name in sorted(names.values())
        ]
        target_marker = {
            "schema_version": "1.0",
            "stage": _PREPARE_STAGE,
            "algorithm": _PREPARE_ALGORITHM,
            "config_hash": target_config_hash,
            "execution_mode": config.execution_mode,
            "identity_assignment_performed": False,
            "stats": dict(stats),
            "input_fingerprints": target_inputs,
            "output_fingerprints": output_fingerprints,
            "elapsed_sec": float(source_marker["elapsed_sec"]),
        }
        _write_json(staging / _PREPARE_MARKER, target_marker)
        logger(
            "[s05-force-appearance-repackage-prepared] validating repackaged "
            "bundle with the strict production loader"
        )
        target_prepared, loaded_target_marker = _load_prepared_appearance(
            staging,
            config=validation_config,
            config_payload=target_payload,
            config_hash=target_config_hash,
            input_fingerprints=target_inputs,
        )
        if loaded_target_marker != target_marker or (
            target_prepared.appearance.descriptor_centers.shape[1]
            != config.embedding_dim
        ):
            raise ContractError("prepared repackage target descriptor contract differs")
        for name in descriptor_names:
            if (
                _output_fingerprint(source_prepared_dir / name, source_prepared_dir)
                != source_by_name[name]
            ):
                raise ContractError(
                    f"prepared repackage source changed while copying: {name}"
                )
        if fingerprint_file(config_path) != target_config_fingerprint:
            raise ContractError(
                "forced appearance config changed during prepared repackage"
            )
        try:
            os.replace(staging, output_dir)
        except OSError as exc:
            raise ContractError(
                "forced appearance prepared repackage cannot be atomically "
                f"committed: {exc}"
            ) from exc
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    logger(
        "[s05-force-appearance-repackage-prepared] complete: descriptor bytes "
        f"preserved; output={output_dir}"
    )
    return target_marker


def run_s05_force_appearance_prepare(
    manifest_path: Path,
    ingest_dir: Path,
    microtrack_dir: Path,
    appearance_dir: Path,
    stable_dir: Path,
    config_path: Path,
    device: str,
    output_dir: Path,
    *,
    logger: LogFn = log,
) -> dict[str, Any]:
    """Decode/embed once and atomically persist the exact solver descriptors."""

    if not callable(logger):
        raise ContractError("forced appearance preparation logger must be callable")
    started = time.monotonic()
    (
        manifest_path,
        ingest_dir,
        microtrack_dir,
        appearance_dir,
        stable_dir,
        config_path,
        output_dir,
    ) = (
        path.resolve()
        for path in (
            manifest_path,
            ingest_dir,
            microtrack_dir,
            appearance_dir,
            stable_dir,
            config_path,
            output_dir,
        )
    )
    _reject_path_overlap(
        output_dir,
        (manifest_path, ingest_dir, microtrack_dir, appearance_dir, stable_dir, config_path),
    )
    common = _load_common_forced_inputs(
        manifest_path,
        ingest_dir,
        microtrack_dir,
        appearance_dir,
        stable_dir,
        config_path,
        device=device,
        load_identity=True,
        logger=logger,
    )
    marker_path = output_dir / _PREPARE_MARKER
    if marker_path.is_file():
        _, marker = _load_prepared_appearance(
            output_dir,
            config=common.config,
            config_payload=common.config_payload,
            config_hash=common.config_hash,
            input_fingerprints=common.input_fingerprints,
        )
        logger(
            f"[s05-force-appearance-prepare] already complete and revalidated: {marker_path}"
        )
        return marker
    if output_dir.exists():
        if not output_dir.is_dir():
            raise ContractError(
                f"forced appearance preparation output is not a directory: {output_dir}"
            )
        if any(output_dir.iterdir()):
            raise ContractError(
                "forced appearance preparation output is non-empty without "
                f"{_PREPARE_MARKER}: {output_dir}"
            )

    config = common.config
    logger("[s05-force-appearance-prepare] building A-clean and B-degraded descriptors")
    base = _build_base_appearance(common.production, common.stable, config)
    no_s02_count = len(base.zero_sample_stable_ids)
    if config.expected_no_s02_sample_count not in (None, no_s02_count):
        raise ContractError(
            "forced appearance zero-S02 stable count differs: "
            f"{no_s02_count} != {config.expected_no_s02_sample_count}"
        )
    config = replace(config, expected_no_s02_sample_count=no_s02_count)
    stable_by_detection = _stable_by_detection(common.production, common.stable)
    if common.identity is None:
        raise ContractError("forced appearance preparation identity input was not loaded")
    rescue_candidates = _plan_rescue_candidates(
        common.identity,
        common.production,
        stable_by_detection,
        base.zero_sample_stable_ids,
        config,
    )
    logger(
        f"[s05-force-appearance-prepare] decoding {len(rescue_candidates):,} "
        f"candidate crops for {no_s02_count:,} C-grade stable tracks"
    )
    if rescue_candidates:
        _decode_rescue_candidates(
            rescue_candidates, common.video_paths, config, logger=logger
        )
        supplemental, rescue_embeddings = _select_and_embed_rescue(
            rescue_candidates, config, logger=logger
        )
    else:
        supplemental = []
        rescue_embeddings = np.empty((0, config.embedding_dim), dtype=np.float32)
    graded = complete_forced_appearance_with_supplemental(base, supplemental)
    rescue_rows = _rescue_rows(rescue_candidates)
    _verify_regular_fingerprints(
        common.input_fingerprints, set(common.video_path_set)
    )
    marker = _write_prepared_output(
        output_dir,
        config=config,
        config_payload=common.config_payload,
        config_hash=common.config_hash,
        input_fingerprints=common.input_fingerprints,
        appearance=graded,
        rescue_rows=rescue_rows,
        rescue_embeddings=rescue_embeddings,
        elapsed_sec=time.monotonic() - started,
    )
    _verify_regular_fingerprints(
        common.input_fingerprints, set(common.video_path_set)
    )
    logger(
        "[s05-force-appearance-prepare] complete: "
        f"{no_s02_count:,} C-grade stable tracks, "
        f"{len(rescue_embeddings):,} embeddings; output={output_dir}"
    )
    return marker


def run_s05_force_appearance(
    manifest_path: Path,
    ingest_dir: Path,
    microtrack_dir: Path,
    appearance_dir: Path,
    stable_dir: Path,
    prior_global_dir: Path,
    config_path: Path,
    prepared_dir: Path,
    output_dir: Path,
    *,
    logger: LogFn = log,
) -> dict[str, Any]:
    """Solve exact-62 paths from a required, fully validated prepared cache."""

    if not callable(logger):
        raise ContractError("forced appearance logger must be callable")
    started = time.monotonic()
    (
        manifest_path,
        ingest_dir,
        microtrack_dir,
        appearance_dir,
        stable_dir,
        prior_global_dir,
        config_path,
        prepared_dir,
        output_dir,
    ) = (
        path.resolve()
        for path in (
            manifest_path,
            ingest_dir,
            microtrack_dir,
            appearance_dir,
            stable_dir,
            prior_global_dir,
            config_path,
            prepared_dir,
            output_dir,
        )
    )
    upstream_paths = (
        manifest_path,
        ingest_dir,
        microtrack_dir,
        appearance_dir,
        stable_dir,
        prior_global_dir,
        config_path,
    )
    _reject_path_overlap(
        output_dir,
        (*upstream_paths, prepared_dir),
    )
    _reject_path_overlap(prepared_dir, upstream_paths)
    common = _load_common_forced_inputs(
        manifest_path,
        ingest_dir,
        microtrack_dir,
        appearance_dir,
        stable_dir,
        config_path,
        device=None,
        load_identity=False,
        logger=logger,
    )
    config = common.config
    logger("[s05-force-appearance] loading current S05 links as soft preference only")
    prior_pairs, prior_fingerprints = _load_prior_pairs(
        prior_global_dir, common.stable, config
    )
    if config.expected_prior_link_count not in (None, len(prior_pairs)):
        raise ContractError("forced appearance configured prior-link count differs")
    config = replace(config, expected_prior_link_count=len(prior_pairs))
    common_fingerprints = tuple(
        FileFingerprint(
            path=str(item["path"]),
            size_bytes=int(item["size_bytes"]),
            sha256=str(item["sha256"]),
        )
        for item in common.input_fingerprints
    )
    input_fingerprints = _normalize_fingerprints(
        (*common_fingerprints, *prior_fingerprints)
    )
    _verify_regular_fingerprints(input_fingerprints, set(common.video_path_set))

    success_path = output_dir / config.artifacts.success
    if success_path.is_file():
        marker = _read_json(success_path, "forced appearance success marker")
        if not isinstance(marker, Mapping):
            raise ContractError("forced appearance success marker must be an object")
        _validate_completed_output(
            output_dir,
            marker,
            config=config,
            config_hash=common.config_hash,
            input_fingerprints=input_fingerprints,
        )
        logger(f"[s05-force-appearance] already complete and revalidated: {success_path}")
        return dict(marker)
    if output_dir.exists():
        if not output_dir.is_dir():
            raise ContractError(f"forced appearance output is not a directory: {output_dir}")
        if any(output_dir.iterdir()):
            raise ContractError(
                f"forced appearance output is non-empty without _SUCCESS.json: {output_dir}"
            )

    logger(
        "[s05-force-appearance] loading required immutable prepared descriptors; "
        "video decode and embedding are disabled in solve"
    )
    prepared, _prepared_marker = _load_prepared_appearance(
        prepared_dir,
        config=config,
        config_payload=common.config_payload,
        config_hash=common.config_hash,
        input_fingerprints=common.input_fingerprints,
    )
    config = replace(
        config, expected_no_s02_sample_count=prepared.no_s02_sample_count
    )
    graded = prepared.appearance
    rescue_rows = prepared.rescue_rows
    rescue_embeddings = prepared.rescue_embeddings
    prepared_fingerprints = tuple(
        item.as_dict() for item in prepared.input_fingerprints
    )
    _verify_regular_fingerprints(prepared_fingerprints, set())
    graph, result, solve_passes = _build_graph_and_cover(
        common.stable, graded, prior_pairs, config, logger=logger
    )
    candidate_rows = _candidate_rows(graph, result, common.stable)
    mapping_rows, global_rows, mapping_by_stable = _build_global_rows(
        common.stable, graph, result, config
    )
    # Do not retain the 3.3-million-row identity projection while the complete
    # candidate graph and HiGHS flow workspace are live.  Load it only after
    # the exact cover succeeds, immediately before final detection mapping.
    identity = _load_identity(ingest_dir, common.production, config)
    detection_table = build_detection_to_global_table(  # type: ignore[arg-type]
        identity, common.stable, mapping_by_stable, global_rows, config
    )
    tables = {
        "rescue_samples": _rows_table(
            rescue_rows, RESCUE_SAMPLES_SCHEMA, "forced rescue samples"
        ),
        "graded_stable_appearance": _rows_table(
            graded.rows,
            GRADED_STABLE_APPEARANCE_SCHEMA,
            "forced graded stable appearance",
        ),
        "candidate_edges": _rows_table(
            candidate_rows,
            FORCED_CANDIDATE_EDGES_SCHEMA,
            "forced candidate edges",
        ),
        "stable_to_global": _rows_table(
            mapping_rows, STABLE_TO_GLOBAL_SCHEMA, "forced stable-to-global"
        ),
        "global_tracks": _rows_table(
            global_rows, GLOBAL_TRACKS_SCHEMA, "forced global tracks"
        ),
        "det_to_global": detection_table,
    }
    report = _build_report(
        config=config,
        config_hash=common.config_hash,
        appearance=graded,
        rescue_rows=rescue_rows,
        graph=graph,
        result=result,
        solve_passes=solve_passes,
        candidate_rows=candidate_rows,
        global_rows=global_rows,
        input_fingerprints=input_fingerprints,
        elapsed_sec=time.monotonic() - started,
    )
    _verify_regular_fingerprints(input_fingerprints, set(common.video_path_set))
    _verify_regular_fingerprints(prepared_fingerprints, set())

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = output_dir.parent / f".{output_dir.name}.staging-{os.getpid()}"
    if staging.exists():
        raise ContractError(f"forced appearance staging directory exists: {staging}")
    staging.mkdir(parents=False)
    try:
        names = _artifact_names(config)
        schemas = {
            "rescue_samples": RESCUE_SAMPLES_SCHEMA,
            "graded_stable_appearance": GRADED_STABLE_APPEARANCE_SCHEMA,
            "candidate_edges": FORCED_CANDIDATE_EDGES_SCHEMA,
            "stable_to_global": STABLE_TO_GLOBAL_SCHEMA,
            "global_tracks": GLOBAL_TRACKS_SCHEMA,
            "det_to_global": DET_TO_GLOBAL_SCHEMA,
        }
        for key, schema in schemas.items():
            _write_table(
                staging / names[key], tables[key], schema, config.parquet_compression
            )
        np.save(
            staging / config.artifacts.rescue_embeddings,
            rescue_embeddings.astype(np.float16),
            allow_pickle=False,
        )
        np.save(
            staging / config.artifacts.graded_prototypes,
            np.asarray(graded.prototypes, dtype=np.float16),
            allow_pickle=False,
        )
        np.save(
            staging / config.artifacts.graded_prototype_mask,
            np.asarray(graded.prototype_mask, dtype=np.bool_),
            allow_pickle=False,
        )
        _write_json(staging / config.artifacts.report, report)
        _write_json(staging / config.artifacts.effective_config, common.config_payload)
        _verify_regular_fingerprints(input_fingerprints, set(common.video_path_set))
        _verify_regular_fingerprints(prepared_fingerprints, set())
        output_names = sorted(set(names.values()) - {config.artifacts.success})
        output_fingerprints = [
            _output_fingerprint(staging / name, staging) for name in output_names
        ]
        marker = {
            "schema_version": "1.0",
            "stage": _STAGE,
            "config_hash": common.config_hash,
            "execution_mode": config.execution_mode,
            "operator_approved": True,
            "authorization_basis": _AUTHORIZATION,
            "certification_claimed": False,
            "target_global_track_count": config.target_global_track_count,
            "stats": {
                "num_stable_tracks": config.expected_stable_track_count,
                "num_selected_links": result.required_links,
                "num_global_tracks": result.num_paths,
                "num_rescue_stable_tracks": config.expected_no_s02_sample_count,
                "num_rescue_embeddings": len(rescue_embeddings),
                "num_candidate_edges": len(candidate_rows),
                "max_concurrent_stable_tracks": graph.max_concurrent,
            },
            "input_fingerprints": input_fingerprints,
            "output_fingerprints": output_fingerprints,
            "elapsed_sec": float(time.monotonic() - started),
        }
        _write_json(staging / config.artifacts.success, marker)
        _validate_completed_output(
            staging,
            marker,
            config=config,
            config_hash=common.config_hash,
            input_fingerprints=input_fingerprints,
        )
        try:
            os.replace(staging, output_dir)
        except OSError as exc:
            raise ContractError(
                f"forced appearance output cannot be atomically committed: {exc}"
            ) from exc
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    logger(
        f"[s05-force-appearance] complete: {result.required_links:,} links -> "
        f"{result.num_paths} forced-provisional IDs; output={output_dir}"
    )
    return marker


__all__ = [
    "run_s05_force_appearance",
    "run_s05_force_appearance_prepare",
    "run_s05_force_appearance_repackage_prepared",
]
