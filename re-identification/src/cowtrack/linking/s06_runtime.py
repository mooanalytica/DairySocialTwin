"""Strict read-only inputs for the S06 forced-provisional export.

This module deliberately does more than load a few Parquet files.  S06 is the
last consumer of the identity graph, so it re-establishes the complete chain
of custody from the live manifest through S00, S01, finalized S04 and the
operator-authorized ``S05_FORCE_APPEARANCE`` snapshot.  It never reads the
older ``05_global_link`` directory as an identity result.

The C-grade overlap fields written by S05 are known to be wrong.  The loader
therefore derives those fields from ``rescue_samples.parquet`` and exposes the
derived, immutable provenance together with an explicit warning.
"""

from __future__ import annotations

import hashlib
import json
import math
import stat as stat_module
import time
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Callable, Mapping, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from cowtrack.config import (
    ContractError,
    load_config,
    load_manifest,
    load_microtrack_config,
)
from cowtrack.linking.dataset_contract import (
    EXPECTED_CLIP_ORDER,
    EXPECTED_FRAME_COUNT,
    EXPECTED_FRAME_COUNTS,
    EXPECTED_SEQUENCE_ID,
    MAX_GLOBAL_TRACK_COUNT,
)
from cowtrack.linking.forced_appearance_config import (
    ForcedAppearanceConfig,
    load_forced_appearance_config,
)
from cowtrack.linking.runtime import FileFingerprint
from cowtrack.linking.s04_runtime import S04FinalizedBundle, load_s04_finalized
from cowtrack.qa.s06_config import (
    EXPECTED_INVALID_DETECTIONS,
    EXPECTED_INVALID_DETECTIONS_BY_CLIP,
    EXPECTED_TOTAL_DETECTIONS,
    EXPECTED_VALID_DETECTIONS,
    EXPECTED_VALID_DETECTIONS_BY_CLIP,
)
from cowtrack.schemas.detections import (
    DETECTIONS_SCHEMA,
    INVALID_BBOX_MASK,
)
from cowtrack.schemas.edges import DET_EDGES_SCHEMA
from cowtrack.schemas.frames import FRAMES_SCHEMA
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
from cowtrack.schemas.tracklets import DET_TO_MICRO_SCHEMA, MICROTRACKLETS_SCHEMA

if TYPE_CHECKING:
    from cowtrack.qa.s06_config import S06ExportConfig


LogFn = Callable[[str], None]

EXPECTED_VALID_BY_CLIP = EXPECTED_VALID_DETECTIONS_BY_CLIP
EXPECTED_INVALID_BY_CLIP = EXPECTED_INVALID_DETECTIONS_BY_CLIP
EXPECTED_FRAMES = EXPECTED_FRAME_COUNT
EXPECTED_GLOBAL_TRACKS = MAX_GLOBAL_TRACK_COUNT

_APPEARANCE_GRADES = frozenset(
    {"A_CLEAN", "B_EXISTING_DEGRADED", "C_REENCODED_DEGRADED"}
)

AUTHORIZATION_BASIS = "operator_forced_appearance_exact_62"
FORCED_ID_STATUS = "forced_provisional"
FORCED_STAGE = "S05_FORCE_APPEARANCE"

_S00_BASE_OUTPUTS = (
    "detections.parquet",
    "effective_config.json",
    "frames.parquet",
    "ingest_report.json",
    "overlay_manifest.json",
    "resolved_manifest.json",
)
_S01_OUTPUTS = (
    "det_edges.parquet",
    "det_to_micro.parquet",
    "effective_config.json",
    "microtrack_report.json",
    "microtracklets.parquet",
    "motion_prior.npz",
)
_S01_S00_INPUT_NAMES = (
    "_SUCCESS.json",
    "detections.parquet",
    "frames.parquet",
    "ingest_report.json",
    "resolved_manifest.json",
)
_S04_RECORDED_INPUT_ROLES = MappingProxyType(
    {
        "configs": ("s04_finalize.yaml",),
        "00_ingest": ("_SUCCESS.json", "detections.parquet", "frames.parquet"),
        "01_microtrack": (
            "_SUCCESS.json",
            "det_to_micro.parquet",
            "microtracklets.parquet",
        ),
        "02_appearance": (
            "_SUCCESS.json",
            "appearance_exclusions.parquet",
            "appearance_report.json",
            "appearance_samples.parquet",
            "effective_config.json",
            "encoder_choice.json",
            "micro_appearance.parquet",
            "sample_embeddings.f16.npy",
        ),
        "03_calibration": (
            "_SUCCESS.json",
            "effective_config.json",
            "link_model_long.joblib",
            "link_model_short.joblib",
            "pair_feature_schema.json",
            "thresholds.json",
        ),
        "04_short_proposals": (
            "_SUCCESS.json",
            "effective_config.json",
            "review_manifest.json",
            "s04_proposal_report.json",
            "short_candidate_edges.parquet",
            "short_link_proposals.parquet",
        ),
    }
)
_FORCED_OUTPUTS = (
    "det_to_global.parquet",
    "effective_config.json",
    "forced_candidate_edges.parquet",
    "global_tracks.parquet",
    "graded_prototype_mask.npy",
    "graded_prototypes.f16.npy",
    "graded_stable_appearance.parquet",
    "rescue_embeddings.f16.npy",
    "rescue_samples.parquet",
    "s05_force_appearance_report.json",
    "stable_to_global.parquet",
)

_S00_MARKER_KEYS = {
    "stage", "schema_version", "config_hash", "manifest_hash",
    "program_commit_hash", "input_fingerprints", "output_fingerprints", "stats",
}
_S01_MARKER_KEYS = {
    "stage", "schema_version", "config_hash", "program_commit_hash",
    "input_fingerprints", "output_fingerprints", "stats",
}
_FORCED_MARKER_KEYS = {
    "schema_version", "stage", "config_hash", "execution_mode",
    "operator_approved", "authorization_basis", "certification_claimed",
    "target_global_track_count", "stats", "input_fingerprints",
    "output_fingerprints", "elapsed_sec",
}
_FORCED_REPORT_KEYS = {
    "schema_version", "stage", "config_hash", "execution_mode", "sequence_id",
    "operator_approval", "evidence_semantics", "graded_appearance", "rescue",
    "solver", "coverage", "global_path_size_distribution", "input_fingerprints",
    "elapsed_sec",
}


@dataclass(frozen=True)
class CGradeProvenance:
    """Authoritative C-grade descriptor provenance rebuilt from rescue rows."""

    stable_id: int
    num_candidate_crops: int
    num_selected_crops: int
    selected_max_other_bbox_iou: float
    used_high_overlap: bool
    selected_det_ids: tuple[int, ...]
    selected_embedding_rows: tuple[int, ...]


@dataclass(frozen=True)
class S06ForcedTables:
    """Immutable Arrow tables from the exact forced S05 snapshot."""

    rescue_samples: pa.Table
    graded_stable_appearance: pa.Table
    candidate_edges: pa.Table
    stable_to_global: pa.Table
    global_tracks: pa.Table
    det_to_global: pa.Table


@dataclass(frozen=True)
class S06InputBundle:
    """Fully validated, immutable input snapshot used by S06."""

    manifest_path: Path
    manifest_hash: str
    ingest_dir: Path
    microtrack_dir: Path
    forced_dir: Path
    frames: pa.Table
    detections: pa.Table
    det_to_micro: pa.Table
    microtracklets: pa.Table
    s04: S04FinalizedBundle
    forced: S06ForcedTables
    c_grade_provenance: Mapping[int, CGradeProvenance]
    forced_report: Mapping[str, Any]
    forced_success_marker: Mapping[str, Any]
    video_paths: Mapping[str, Path]
    warnings: tuple[str, ...]
    consumed_paths: tuple[Path, ...]
    input_fingerprints: tuple[FileFingerprint, ...]
    input_stat_tokens: Mapping[str, tuple[int, int, int]]


@dataclass(frozen=True)
class _Expectations:
    sequence_id: str = EXPECTED_SEQUENCE_ID
    clip_order: tuple[str, ...] = EXPECTED_CLIP_ORDER
    frame_counts: tuple[int, ...] = EXPECTED_FRAME_COUNTS
    valid_by_clip: tuple[int, ...] = EXPECTED_VALID_BY_CLIP
    invalid_by_clip: tuple[int, ...] = EXPECTED_INVALID_BY_CLIP
    frames: int = EXPECTED_FRAMES
    total_detections: int = EXPECTED_TOTAL_DETECTIONS
    valid_detections: int = EXPECTED_VALID_DETECTIONS
    invalid_detections: int = EXPECTED_INVALID_DETECTIONS
    microtracks: int | None = None
    stable_tracks: int | None = None
    global_tracks: int = EXPECTED_GLOBAL_TRACKS


@dataclass(frozen=True)
class _StatToken:
    size: int
    mtime_ns: int
    inode: int


def _expectations(config: S06ExportConfig | None) -> _Expectations:
    expected = _Expectations()
    if config is None:
        return expected
    checks = {
        "expected_frame_count": expected.frames,
        "expected_total_detection_count": expected.total_detections,
        "expected_valid_detection_count": expected.valid_detections,
        "expected_invalid_detection_count": expected.invalid_detections,
        "expected_global_track_count": expected.global_tracks,
        "expected_cycle_count": 0,
        "expected_same_frame_violation_count": 0,
        "expected_temporal_overlap_violation_count": 0,
        "expected_unassigned_valid_detection_count": 0,
        "num_confirmed_ids": 0,
        "num_provisional_ids": expected.global_tracks,
        "population_soft_max": 57,
        "population_overflow": 5,
    }
    for name, value in checks.items():
        if getattr(config, name, value) != value:
            raise ContractError(f"S06 config/runtime fixed value differs: {name}")
    for name in ("expected_microtrack_count", "expected_stable_track_count"):
        value = getattr(config, name, None)
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value < 1
        ):
            raise ContractError(
                f"S06 config/runtime upstream-derived value is invalid: {name}"
            )
    for name, value in {
        "clip_order": expected.clip_order,
        "frame_counts_by_clip": expected.frame_counts,
        "expected_total_detections_by_clip": tuple(
            left + right
            for left, right in zip(
                expected.valid_by_clip, expected.invalid_by_clip, strict=True
            )
        ),
        "expected_valid_detections_by_clip": expected.valid_by_clip,
        "expected_invalid_detections_by_clip": expected.invalid_by_clip,
    }.items():
        actual = getattr(config, name, value)
        if tuple(actual) != tuple(value):
            raise ContractError(f"S06 config/runtime fixed value differs: {name}")
    for name, value in {
        "expected_sequence_id": expected.sequence_id,
        "required_upstream_stage": FORCED_STAGE,
        "identity_source": "05_forced_appearance_only",
        "authorization_basis": AUTHORIZATION_BASIS,
        "id_status": FORCED_ID_STATUS,
        "certification_claimed": False,
        "probability_from_cosine_allowed": False,
    }.items():
        if getattr(config, name, value) != value:
            raise ContractError(f"S06 config/runtime fixed value differs: {name}")
    return replace(
        expected,
        microtracks=getattr(config, "expected_microtrack_count", None),
        stable_tracks=getattr(config, "expected_stable_track_count", None),
    )


def _resolve_expected_count(
    configured: int | None, observed: int, label: str
) -> int:
    if observed < 1:
        raise ContractError(f"S06 observed {label} must be positive")
    if configured is not None and configured != observed:
        raise ContractError(
            f"S06 configured {label} differs from validated upstream artifacts"
        )
    return observed


def _s00_output_names(expected: _Expectations) -> tuple[str, ...]:
    overlays = tuple(
        f"qa/overlays/{clip}_{region}_f{frame:06d}.jpg"
        for clip, count in zip(
            expected.clip_order, expected.frame_counts, strict=True
        )
        for region, frame in (
            ("front", 0),
            ("middle", (count - 1) // 2),
            ("back", count - 1),
        )
    )
    return (*_S00_BASE_OUTPUTS, *overlays)


def _freeze_json(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({str(key): _freeze_json(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _read_json(path: Path, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read {label} {path}: {exc}") from exc


def _token(path: Path) -> _StatToken:
    try:
        info = path.stat()
    except OSError as exc:
        raise ContractError(f"cannot stat S06 input {path}: {exc}") from exc
    return _StatToken(int(info.st_size), int(info.st_mtime_ns), int(info.st_ino))


def _fingerprint_uncached(
    path: Path, *, logger: LogFn
) -> tuple[FileFingerprint, _StatToken]:
    path = path.resolve()
    try:
        if path.is_symlink() or not stat_module.S_ISREG(path.stat().st_mode):
            raise ContractError(f"S06 input is not a regular file: {path}")
    except OSError as exc:
        raise ContractError(f"cannot resolve S06 input {path}: {exc}") from exc
    before = _token(path)
    digest = hashlib.sha256()
    completed_bytes = 0
    last_report = time.monotonic()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
                completed_bytes += len(block)
                now = time.monotonic()
                if now - last_report >= 10.0:
                    logger(
                        f"[s06-input] hashing {path.name}: "
                        f"{completed_bytes:,}/{before.size:,} bytes"
                    )
                    last_report = now
    except OSError as exc:
        raise ContractError(f"cannot fingerprint S06 input {path}: {exc}") from exc
    after = _token(path)
    if before != after:
        raise ContractError(f"S06 input changed while being fingerprinted: {path}")
    return FileFingerprint(str(path), before.size, digest.hexdigest()), before


def _fingerprint(
    path: Path,
    cache: dict[Path, tuple[FileFingerprint, _StatToken]],
    *,
    logger: LogFn = lambda _message: None,
) -> FileFingerprint:
    canonical = path.resolve()
    cached = cache.get(canonical)
    if cached is None:
        cached = _fingerprint_uncached(canonical, logger=logger)
        cache[canonical] = cached
    return cached[0]


def _valid_digest(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_fingerprint_record(
    record: Any,
    *,
    label: str,
    with_mtime: bool,
) -> tuple[str, int, str, int | None]:
    keys = {"path", "size_bytes", "sha256"} | ({"mtime_ns"} if with_mtime else set())
    if not isinstance(record, dict) or set(record) != keys:
        raise ContractError(f"{label} contains an invalid fingerprint record")
    path, size, digest = record["path"], record["size_bytes"], record["sha256"]
    mtime = record.get("mtime_ns")
    if (
        not isinstance(path, str)
        or not path
        or isinstance(size, bool)
        or not isinstance(size, int)
        or size < 0
        or not _valid_digest(digest)
        or (with_mtime and (isinstance(mtime, bool) or not isinstance(mtime, int) or mtime < 0))
    ):
        raise ContractError(f"{label} contains invalid fingerprint fields")
    return path, size, digest, mtime


def _safe_relative_name(raw: str, label: str) -> str:
    relative = Path(raw)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ContractError(f"{label} contains unsafe output path: {raw}")
    return relative.as_posix()


def _snapshot_output_tree(
    directory: Path,
    marker: Mapping[str, Any],
    expected_names: Sequence[str],
    cache: dict[Path, tuple[FileFingerprint, _StatToken]],
    *,
    label: str,
    logger: LogFn = lambda _message: None,
) -> tuple[Path, ...]:
    directory = directory.resolve()
    if not directory.is_dir() or directory.is_symlink():
        raise ContractError(f"{label} directory does not exist or is unsafe: {directory}")
    expected = tuple(sorted(expected_names))
    records = marker.get("output_fingerprints")
    if not isinstance(records, list):
        raise ContractError(f"{label} marker lacks output_fingerprints")
    parsed = [
        _validate_fingerprint_record(item, label=f"{label} output_fingerprints", with_mtime=False)
        for item in records
    ]
    names = tuple(_safe_relative_name(item[0], label) for item in parsed)
    if names != expected or len(set(names)) != len(names):
        raise ContractError(f"{label} output fingerprint set/order differs")
    actual: set[str] = set()
    actual_directories: set[str] = set()
    for path in directory.rglob("*"):
        if path.is_symlink():
            raise ContractError(f"{label} artifact tree contains a symlink: {path}")
        relative = path.relative_to(directory).as_posix()
        if path.is_dir():
            actual_directories.add(relative)
        elif path.is_file() and relative != "_SUCCESS.json":
            actual.add(relative)
    if actual != set(expected):
        raise ContractError(f"{label} artifact tree differs")
    expected_directories = {
        Path(name).parent.as_posix()
        for name in expected
        if Path(name).parent != Path(".")
    }
    expected_directories |= {
        parent.as_posix()
        for name in tuple(expected_directories)
        for parent in Path(name).parents
        if parent != Path(".")
    }
    if actual_directories != expected_directories:
        raise ContractError(f"{label} artifact directory tree differs")
    paths = (directory / "_SUCCESS.json", *(directory / name for name in expected))
    _fingerprint(paths[0], cache, logger=logger)
    for (name, size, digest, _), path in zip(parsed, paths[1:], strict=True):
        if name != path.relative_to(directory).as_posix():
            raise ContractError(f"{label} output path ordering differs")
        current = _fingerprint(path, cache, logger=logger)
        if current.size_bytes != size or current.sha256 != digest:
            raise ContractError(f"completed {label} artifact changed: {name}")
    return tuple(paths)


def _verify_recorded_inputs(
    records: Any,
    *,
    label: str,
    with_mtime: bool,
    cache: dict[Path, tuple[FileFingerprint, _StatToken]],
    logger: LogFn,
    path_replacements: Mapping[str, Path] | None = None,
) -> tuple[FileFingerprint, ...]:
    parsed = _recorded_input_fingerprints(
        records,
        label=label,
        with_mtime=with_mtime,
    )
    assert isinstance(records, list)
    if path_replacements is not None and set(path_replacements) != {
        item.path for item in parsed
    }:
        raise ContractError(f"{label} relocation map differs from recorded paths")
    result: list[FileFingerprint] = []
    for index, (recorded, raw_record) in enumerate(
        zip(parsed, records, strict=True), start=1
    ):
        raw_path = recorded.path
        size = recorded.size_bytes
        digest = recorded.sha256
        recorded_mtime = raw_record.get("mtime_ns")
        recorded_path = Path(raw_path)
        path = (
            Path(path_replacements[raw_path])
            if path_replacements is not None
            else recorded_path
        )
        try:
            canonical = path.resolve(strict=True)
        except OSError as exc:
            raise ContractError(f"cannot resolve recorded S06 input {path}: {exc}") from exc
        if canonical != path or path.is_symlink() or not stat_module.S_ISREG(path.stat().st_mode):
            raise ContractError(f"{label} path is not canonical regular file: {path}")
        if path != recorded_path:
            logger(
                f"[s06-input] accepted byte-identical provenance relocation: "
                f"{recorded_path} -> {path}"
            )
        logger(f"[s06-input] fingerprint provenance {index:,}/{len(parsed):,}: {path.name}")
        current = _fingerprint(path, cache, logger=logger)
        if current.size_bytes != size or current.sha256 != digest:
            raise ContractError(f"recorded S06 upstream input changed: {path}")
        # Historical mtimes are provenance metadata, not content identity.  A
        # file may be restored with byte-identical contents and a different
        # timestamp; current-run stat tokens still guard every cached input
        # against changes while S06 is loading and exporting.
        if (
            with_mtime
            and path == recorded_path
            and cache[canonical][1].mtime_ns != recorded_mtime
        ):
            logger(
                "[s06-input] warning: recorded mtime differs but content hash "
                f"matches: {path}"
            )
        result.append(current)
    return tuple(result)


def _recorded_input_fingerprints(
    records: Any,
    *,
    label: str,
    with_mtime: bool,
) -> tuple[FileFingerprint, ...]:
    """Parse provenance metadata without touching the recorded files."""

    if not isinstance(records, list) or not records:
        raise ContractError(f"{label} cannot be empty")
    validated = [
        _validate_fingerprint_record(item, label=label, with_mtime=with_mtime)
        for item in records
    ]
    raw_paths = [item[0] for item in validated]
    if len(set(raw_paths)) != len(raw_paths):
        raise ContractError(f"{label} contains duplicate paths")
    result: list[FileFingerprint] = []
    for raw_path, size, digest, _recorded_mtime in validated:
        path = Path(raw_path)
        if not path.is_absolute():
            raise ContractError(f"{label} path must be absolute: {raw_path}")
        result.append(FileFingerprint(raw_path, size, digest))
    return tuple(result)


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


def _column(table: pa.Table, name: str, dtype: Any) -> np.ndarray:
    return np.asarray(
        table[name].combine_chunks().to_numpy(zero_copy_only=False), dtype=dtype
    )


def _locate(known: np.ndarray, requested: np.ndarray, label: str) -> np.ndarray:
    known = np.asarray(known, dtype=np.int64)
    requested = np.asarray(requested, dtype=np.int64)
    if len(np.unique(known)) != len(known):
        raise ContractError(f"{label} source IDs are not unique")
    order = np.argsort(known, kind="stable")
    sorted_ids = known[order]
    positions = np.searchsorted(sorted_ids, requested)
    if len(positions) and (
        np.any(positions >= len(sorted_ids))
        or not np.array_equal(sorted_ids[positions], requested)
    ):
        raise ContractError(f"{label} references an unknown ID")
    return order[positions]


def _finite_nonnegative(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ContractError(f"{label} must be finite and nonnegative")
    return result


def _validate_s00_recorded_input_roles(
    recorded: Sequence[FileFingerprint],
    rows: Sequence[Any],
    manifest_hash: str,
) -> tuple[Path, Path]:
    """Identify relocated manifest/config records without relaxing media paths."""

    by_path = {item.path: item for item in recorded}
    if len(by_path) != len(recorded):
        raise ContractError("S00 input fingerprints contain duplicate paths")
    required_media_paths = {
        *(str(row.video_path.resolve()) for row in rows),
        *(str(row.bbox_csv_path.resolve()) for row in rows),
    }
    if not required_media_paths.issubset(by_path):
        raise ContractError(
            "S00 video/bbox fingerprint paths differ from live manifest"
        )
    repository_local_paths = set(by_path) - required_media_paths
    if len(repository_local_paths) != 2:
        raise ContractError(
            "S00 input fingerprints must contain one manifest and one config"
        )
    manifest_paths = {
        path
        for path in repository_local_paths
        if by_path[path].sha256 == manifest_hash
    }
    if len(manifest_paths) != 1:
        raise ContractError(
            "S00 recorded manifest is not uniquely identified by content hash"
        )
    manifest_path = next(iter(manifest_paths))
    config_paths = repository_local_paths - manifest_paths
    if len(config_paths) != 1:
        raise ContractError("S00 recorded config path is not unique")
    config_path = next(iter(config_paths))
    manifest_role = Path(manifest_path)
    config_role = Path(config_path)
    if (
        manifest_role.name != "manifest.csv"
        or manifest_role.parent.name != "data"
        or config_role.name != "production.yaml"
        or config_role.parent.name != "configs"
        or manifest_role.parent.parent != config_role.parent.parent
    ):
        raise ContractError("S00 recorded manifest/config roles differ")
    return manifest_role, config_role


def _validate_s01_recorded_input_roles(
    recorded: Sequence[FileFingerprint],
    current_s00: Mapping[str, FileFingerprint],
) -> Path:
    """Bind a relocated S01 marker to the validated live S00 snapshot."""

    if len(recorded) != len(_S01_S00_INPUT_NAMES) + 1:
        raise ContractError("S01 input fingerprint count differs")
    if len({item.path for item in recorded}) != len(recorded):
        raise ContractError("S01 input fingerprints contain duplicate paths")
    if set(current_s00) != set(_S01_S00_INPUT_NAMES):
        raise ContractError("live S00 inputs required by S01 are incomplete")

    matched_paths: set[str] = set()
    matched_parents: set[Path] = set()
    for name in _S01_S00_INPUT_NAMES:
        matches = [item for item in recorded if Path(item.path).name == name]
        if len(matches) != 1:
            raise ContractError(f"S01 recorded {name} role is not unique")
        recorded_item = matches[0]
        current_item = current_s00[name]
        if (
            recorded_item.size_bytes != current_item.size_bytes
            or recorded_item.sha256 != current_item.sha256
        ):
            raise ContractError(f"S01/S00 {name} byte provenance differs")
        matched_paths.add(recorded_item.path)
        matched_parents.add(Path(recorded_item.path).parent)
    if len(matched_parents) != 1:
        raise ContractError("S01 recorded S00 inputs do not share one snapshot root")

    config_records = [item for item in recorded if item.path not in matched_paths]
    if len(config_records) != 1:
        raise ContractError("S01 recorded source config role is not unique")
    config_path = Path(config_records[0].path)
    if (
        next(iter(matched_parents)).name != "00_ingest"
        or config_path.name != "s01_microtrack.yaml"
        or config_path.parent.name != "configs"
    ):
        raise ContractError("S01 recorded input role suffixes differ")
    return config_path


def _s04_recorded_input_relocations(
    stable_dir: Path,
    repository_root: Path,
) -> Mapping[str, Path]:
    """Map the fixed S04 provenance roles into the current repository copy."""

    marker = _read_json(
        stable_dir / "_SUCCESS.json",
        "S04 finalized success marker",
    )
    if not isinstance(marker, dict):
        raise ContractError("S04 finalized success marker must be an object")
    recorded = _recorded_input_fingerprints(
        marker.get("input_fingerprints"),
        label="S04 marker input_fingerprints",
        with_mtime=False,
    )
    expected_roles = {
        (directory, name)
        for directory, names in _S04_RECORDED_INPUT_ROLES.items()
        for name in names
    }
    observed_roles = {
        (Path(item.path).parent.name, Path(item.path).name) for item in recorded
    }
    if (
        len(recorded) != len(expected_roles)
        or observed_roles != expected_roles
    ):
        raise ContractError("S04 recorded input role set differs")

    current_sequence_root = stable_dir.parent.resolve()
    recorded_sequence_roots = {
        Path(item.path).parent.parent
        for item in recorded
        if Path(item.path).parent.name != "configs"
    }
    if (
        len(recorded_sequence_roots) != 1
        or next(iter(recorded_sequence_roots)).name
        != current_sequence_root.name
    ):
        raise ContractError("S04 recorded inputs mix sequence snapshot roots")
    current_repository_root = repository_root.resolve()
    replacements: dict[str, Path] = {}
    for item in recorded:
        recorded_path = Path(item.path)
        directory = recorded_path.parent.name
        if directory == "configs":
            current = current_repository_root / "configs" / recorded_path.name
        else:
            current = current_sequence_root / directory / recorded_path.name
        replacements[item.path] = current.resolve()
    if len(set(replacements.values())) != len(replacements):
        raise ContractError("S04 recorded input relocation targets collide")
    return MappingProxyType(replacements)


def _validate_manifest_and_s00(
    manifest_path: Path,
    ingest_dir: Path,
    expected: _Expectations,
    cache: dict[Path, tuple[FileFingerprint, _StatToken]],
    *,
    logger: LogFn,
) -> tuple[str, Mapping[str, Path], pa.Table, pa.Table, tuple[Path, ...], Mapping[str, Any]]:
    manifest_path = manifest_path.resolve()
    rows, manifest_hash = load_manifest(manifest_path)
    if (
        [row.sequence_id for row in rows] != [expected.sequence_id] * len(expected.clip_order)
        or tuple(row.clip_id for row in rows) != expected.clip_order
        or tuple(row.clip_order for row in rows) != tuple(range(len(expected.clip_order)))
    ):
        raise ContractError("S06 manifest sequence/clip order differs")
    expected_resolved = [
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
    resolved = _read_json(ingest_dir / "resolved_manifest.json", "S00 resolved manifest")
    if resolved != expected_resolved:
        raise ContractError("S06 live manifest/S00 resolved manifest bijection differs")
    marker = _read_json(ingest_dir / "_SUCCESS.json", "S00 success marker")
    if not isinstance(marker, dict) or set(marker) != _S00_MARKER_KEYS:
        raise ContractError("S00 success marker keys differ")
    if (
        marker.get("stage") != "S00"
        or marker.get("schema_version") != "1.0"
        or marker.get("manifest_hash") != manifest_hash
        or not _valid_digest(marker.get("config_hash"))
        or marker.get("program_commit_hash") is not None
    ):
        raise ContractError("S00 success marker policy/hash differs")
    stats = marker.get("stats")
    if not isinstance(stats, dict) or {
        key: stats.get(key)
        for key in ("num_clips", "num_frames", "num_input_boxes", "num_valid_boxes", "num_invalid_boxes")
    } != {
        "num_clips": len(expected.clip_order),
        "num_frames": expected.frames,
        "num_input_boxes": expected.total_detections,
        "num_valid_boxes": expected.valid_detections,
        "num_invalid_boxes": expected.invalid_detections,
    }:
        raise ContractError("S00 success marker fixed statistics differ")
    recorded_s00_inputs = _recorded_input_fingerprints(
        marker.get("input_fingerprints"),
        label="S00 input_fingerprints",
        with_mtime=True,
    )
    recorded_manifest_path, recorded_config_path = (
        _validate_s00_recorded_input_roles(
            recorded_s00_inputs,
            rows,
            manifest_hash,
        )
    )
    repository_root = manifest_path.parent.parent
    current_config_path = (
        repository_root / "configs" / recorded_config_path.name
    ).resolve()
    s00_input_paths = {
        item.path: Path(item.path) for item in recorded_s00_inputs
    }
    s00_input_paths[str(recorded_manifest_path)] = manifest_path
    s00_input_paths[str(recorded_config_path)] = current_config_path
    paths = _snapshot_output_tree(
        ingest_dir,
        marker,
        _s00_output_names(expected),
        cache,
        label="S00",
        logger=logger,
    )
    _verify_recorded_inputs(
        marker.get("input_fingerprints"),
        label="S00 input_fingerprints",
        with_mtime=True,
        cache=cache,
        logger=logger,
        path_replacements=s00_input_paths,
    )
    recorded_by_path = {item.path: item for item in recorded_s00_inputs}
    if recorded_by_path[str(recorded_manifest_path)].sha256 != manifest_hash:
        raise ContractError("S00 manifest byte hash differs")
    _s00_config, s00_config_payload, s00_config_hash = load_config(
        current_config_path
    )
    if (
        s00_config_hash != marker.get("config_hash")
        or _read_json(
            ingest_dir / "effective_config.json", "S00 effective config"
        )
        != s00_config_payload
    ):
        raise ContractError("S00 recorded config hash/payload differs")
    report = _read_json(ingest_dir / "ingest_report.json", "S00 ingest report")
    if (
        not isinstance(report, dict)
        or report.get("sequence_id") != expected.sequence_id
        or report.get("coordinate_system") != "raw_encoded_landscape_no_autorotate"
        or report.get("source_rows_retained") is not True
        or report.get("legacy_track_id_used") is not False
        or report.get("num_frames") != expected.frames
        or report.get("num_input_boxes") != expected.total_detections
        or report.get("num_valid_boxes") != expected.valid_detections
        or report.get("num_invalid_boxes") != expected.invalid_detections
    ):
        raise ContractError("S00 ingest report fixed contract differs")
    frames = _read_table(ingest_dir / "frames.parquet", FRAMES_SCHEMA, "S00 frames")
    detections = _read_table(
        ingest_dir / "detections.parquet", DETECTIONS_SCHEMA, "S00 detections"
    )
    _validate_frames(frames, expected)
    _validate_detections(detections, frames, expected)
    video_paths = MappingProxyType({row.clip_id: row.video_path.resolve() for row in rows})
    return manifest_hash, video_paths, frames, detections, paths, marker


def _validate_frames(frames: pa.Table, expected: _Expectations) -> None:
    if frames.num_rows != expected.frames:
        raise ContractError("S00 frame row count differs")
    sequence = np.asarray(frames["sequence_id"].to_pylist(), dtype=object)
    clips = np.asarray(frames["clip_id"].to_pylist(), dtype=object)
    clip_order = _column(frames, "clip_order", np.int16)
    local = _column(frames, "local_frame", np.int64)
    global_frame = _column(frames, "global_frame", np.int64)
    pts = _column(frames, "pts_sec", np.float64)
    times = _column(frames, "global_time_sec", np.float64)
    width = _column(frames, "width", np.int64)
    height = _column(frames, "height", np.int64)
    if (
        set(map(str, sequence)) != {expected.sequence_id}
        or not np.array_equal(global_frame, np.arange(expected.frames, dtype=np.int64))
        or not np.all(np.isfinite(pts))
        or not np.all(np.isfinite(times))
        or np.any(np.diff(times) <= 0.0)
        or not np.all(width == 3840)
        or not np.all(height == 2160)
    ):
        raise ContractError("S00 frame timeline/geometry differs")
    offset = 0
    for order, (clip, count) in enumerate(zip(expected.clip_order, expected.frame_counts, strict=True)):
        positions = np.arange(offset, offset + count)
        if (
            not np.all(clips[positions] == clip)
            or not np.all(clip_order[positions] == order)
            or not np.array_equal(local[positions], np.arange(count, dtype=np.int64))
        ):
            raise ContractError("S00 frame clip partition/order differs")
        offset += count


def _validate_detections(
    detections: pa.Table, frames: pa.Table, expected: _Expectations
) -> None:
    if detections.num_rows != expected.total_detections:
        raise ContractError("S00 detection row count differs")
    ids = _column(detections, "det_id", np.int64)
    clips = np.asarray(detections["clip_id"].to_pylist(), dtype=object)
    sequence = np.asarray(detections["sequence_id"].to_pylist(), dtype=object)
    local = _column(detections, "local_frame", np.int64)
    global_frame = _column(detections, "global_frame", np.int64)
    global_time = _column(detections, "global_time_sec", np.float64)
    csv_row = _column(detections, "csv_row_index", np.int64)
    valid = _column(detections, "valid", np.bool_)
    flags = _column(detections, "qa_flags", np.uint32)
    if (
        len(np.unique(ids)) != len(ids)
        or set(map(str, sequence)) != {expected.sequence_id}
        or np.any(global_frame < 0)
        or np.any(global_frame >= expected.frames)
        or not np.array_equal(
            global_time, _column(frames, "global_time_sec", np.float64)[global_frame]
        )
        or not np.array_equal(local, _column(frames, "local_frame", np.int64)[global_frame])
        or not np.array_equal(clips, np.asarray(frames["clip_id"].to_pylist(), dtype=object)[global_frame])
        or not np.array_equal(valid, (flags & np.uint32(INVALID_BBOX_MASK)) == 0)
    ):
        raise ContractError("S00 detection identity/frame/validity contract differs")
    clip_rank = np.asarray(
        [{clip: index for index, clip in enumerate(expected.clip_order)}.get(str(value), -1) for value in clips],
        dtype=np.int16,
    )
    if np.any(clip_rank < 0) or not np.array_equal(
        np.lexsort((csv_row, clip_rank)), np.arange(len(ids), dtype=np.int64)
    ):
        raise ContractError("S00 detections are not in canonical source-row order")
    for clip, valid_count, invalid_count in zip(
        expected.clip_order, expected.valid_by_clip, expected.invalid_by_clip, strict=True
    ):
        selected = clips == clip
        observed_rows = csv_row[selected]
        if (
            int(np.count_nonzero(selected & valid)) != valid_count
            or int(np.count_nonzero(selected & ~valid)) != invalid_count
            or not np.array_equal(observed_rows, np.arange(len(observed_rows), dtype=np.int64))
        ):
            raise ContractError(f"S00 per-clip detection coverage differs: {clip}")
    for name in ("x1", "y1", "x2", "y2"):
        values = _column(detections, name, np.float64)[valid]
        if not np.all(np.isfinite(values)):
            raise ContractError(f"S00 valid detection {name} is non-finite")
    x1, y1, x2, y2 = (
        _column(detections, name, np.float64)[valid] for name in ("x1", "y1", "x2", "y2")
    )
    if np.any(x2 <= x1) or np.any(y2 <= y1):
        raise ContractError("S00 valid detection has non-positive bbox area")


def _validate_s01(
    microtrack_dir: Path,
    detections: pa.Table,
    s00_paths: Sequence[Path],
    repository_root: Path,
    expected: _Expectations,
    cache: dict[Path, tuple[FileFingerprint, _StatToken]],
    *,
    logger: LogFn,
) -> tuple[pa.Table, pa.Table, tuple[Path, ...], _Expectations]:
    marker = _read_json(microtrack_dir / "_SUCCESS.json", "S01 success marker")
    if not isinstance(marker, dict) or set(marker) != _S01_MARKER_KEYS:
        raise ContractError("S01 success marker keys differ")
    if (
        marker.get("stage") != "S01"
        or marker.get("schema_version") != "1.0"
        or marker.get("program_commit_hash") is not None
        or not _valid_digest(marker.get("config_hash"))
    ):
        raise ContractError("S01 success marker policy/hash differs")
    stats = marker.get("stats")
    if not isinstance(stats, dict):
        raise ContractError("S01 success marker statistics must be an object")
    recorded_inputs = _recorded_input_fingerprints(
        marker.get("input_fingerprints"),
        label="S01 input_fingerprints",
        with_mtime=True,
    )
    current_s00_paths = {
        path.name: path.resolve()
        for path in s00_paths
        if path.name in _S01_S00_INPUT_NAMES
    }
    current_s00 = {
        name: _fingerprint(path, cache, logger=logger)
        for name, path in current_s00_paths.items()
    }
    recorded_config_path = _validate_s01_recorded_input_roles(
        recorded_inputs,
        current_s00,
    )
    current_config_path = (
        repository_root / "configs" / recorded_config_path.name
    ).resolve()
    s01_input_paths: dict[str, Path] = {}
    for item in recorded_inputs:
        name = Path(item.path).name
        s01_input_paths[item.path] = current_s00_paths.get(
            name,
            current_config_path,
        )
    paths = _snapshot_output_tree(
        microtrack_dir,
        marker,
        _S01_OUTPUTS,
        cache,
        label="S01",
        logger=logger,
    )
    _verify_recorded_inputs(
        marker.get("input_fingerprints"), label="S01 input_fingerprints",
        with_mtime=True, cache=cache, logger=logger,
        path_replacements=s01_input_paths,
    )
    source_config, source_payload, source_hash = load_microtrack_config(
        current_config_path
    )
    effective_config, effective_payload, effective_hash = load_microtrack_config(
        microtrack_dir / "effective_config.json"
    )
    if (
        source_hash != marker.get("config_hash")
        or effective_hash != marker.get("config_hash")
        or source_config != effective_config
        or source_payload != effective_payload
    ):
        raise ContractError("S01 source/effective config provenance differs")
    edges = _read_table(microtrack_dir / "det_edges.parquet", DET_EDGES_SCHEMA, "S01 edges")
    mapping = _read_table(
        microtrack_dir / "det_to_micro.parquet", DET_TO_MICRO_SCHEMA, "S01 det-to-micro"
    )
    micros = _read_table(
        microtrack_dir / "microtracklets.parquet", MICROTRACKLETS_SCHEMA, "S01 microtracklets"
    )
    microtrack_count = _resolve_expected_count(
        expected.microtracks, micros.num_rows, "microtrack count"
    )
    expected = replace(expected, microtracks=microtrack_count)
    if (
        stats.get("num_frames") != expected.frames
        or stats.get("num_valid_detections") != expected.valid_detections
        or stats.get("num_microtracklets") != microtrack_count
        or stats.get("num_accepted_edges")
        != expected.valid_detections - microtrack_count
        or edges.num_rows != stats.get("num_accepted_edges")
    ):
        raise ContractError("S01 success marker statistics differ")
    report = _read_json(microtrack_dir / "microtrack_report.json", "S01 report")
    if (
        not isinstance(report, dict)
        or report.get("stage") != "S01"
        or report.get("sequence_id") != expected.sequence_id
        or report.get("input_rows_modified") is not False
        or report.get("legacy_track_id_used") is not False
        or report.get("stats") != stats
    ):
        raise ContractError("S01 report contract differs")
    _validate_micro_mapping(mapping, micros, detections, expected)
    return mapping, micros, paths, expected


def _validate_micro_mapping(
    mapping: pa.Table,
    micros: pa.Table,
    detections: pa.Table,
    expected: _Expectations,
) -> None:
    if expected.microtracks is None:
        raise ContractError("S01 microtrack count was not resolved")
    if mapping.num_rows != expected.valid_detections or micros.num_rows != expected.microtracks:
        raise ContractError("S01 mapping/micro row count differs")
    all_det = _column(detections, "det_id", np.int64)
    valid = _column(detections, "valid", np.bool_)
    valid_ids = all_det[valid]
    mapping_det = _column(mapping, "det_id", np.int64)
    micro_id = _column(mapping, "micro_id", np.int64)
    order = _column(mapping, "order_in_micro", np.int64)
    if (
        len(np.unique(mapping_det)) != len(mapping_det)
        or set(map(int, mapping_det)) != set(map(int, valid_ids))
        or np.any(micro_id < 0)
        or np.any(micro_id >= expected.microtracks)
        or set(map(int, micro_id)) != set(range(expected.microtracks))
    ):
        raise ContractError("S01 det-to-micro is not a total bijection")
    micro_rows = _column(micros, "micro_id", np.int64)
    if not np.array_equal(micro_rows, np.arange(expected.microtracks, dtype=np.int64)):
        raise ContractError("S01 micro IDs are not dense/canonical")
    sort = np.lexsort((mapping_det, order, micro_id))
    counts = np.bincount(micro_id, minlength=expected.microtracks)
    offsets = np.r_[0, np.cumsum(counts)]
    positions = _locate(all_det, mapping_det, "S01 mapping/detections")
    global_frame = _column(detections, "global_frame", np.int64)[positions]
    global_time = _column(detections, "global_time_sec", np.float64)[positions]
    for current in range(expected.microtracks):
        chosen = sort[offsets[current] : offsets[current + 1]]
        if (
            not len(chosen)
            or not np.array_equal(order[chosen], np.arange(len(chosen), dtype=np.int64))
            or np.any(np.diff(global_frame[chosen]) <= 0)
            or int(micros["num_detections"][current].as_py()) != len(chosen)
            or int(micros["start_global_frame"][current].as_py()) != int(global_frame[chosen[0]])
            or int(micros["end_global_frame"][current].as_py()) != int(global_frame[chosen[-1]])
            or float(micros["start_time_sec"][current].as_py()) != float(global_time[chosen[0]])
            or float(micros["end_time_sec"][current].as_py()) != float(global_time[chosen[-1]])
        ):
            raise ContractError(f"S01 micro path summary/order differs: {current}")


def _validate_s04_cross_stage(
    stable: S04FinalizedBundle,
    mapping: pa.Table,
    expected: _Expectations,
) -> None:
    if expected.microtracks is None or expected.stable_tracks is None:
        raise ContractError("S04 upstream-derived counts were not resolved")
    if (
        len(stable.stable_ids) != expected.stable_tracks
        or len(stable.micro_ids) != expected.microtracks
        or len(stable.det_ids) != expected.valid_detections
    ):
        raise ContractError("S04 fixed row counts differ from S06 contract")
    mapping_det = _column(mapping, "det_id", np.int64)
    positions = _locate(mapping_det, stable.det_ids, "S04/S01 detection mapping")
    if (
        not np.array_equal(_column(mapping, "micro_id", np.int64)[positions], stable.det_micro_ids)
        or not np.array_equal(_column(mapping, "order_in_micro", np.int64)[positions], stable.det_order_in_micro)
    ):
        raise ContractError("S04/S01 detection-to-micro mapping differs")


def _forced_artifact_names(config: ForcedAppearanceConfig) -> tuple[str, ...]:
    names = tuple(
        sorted(
            getattr(config.artifacts, name)
            for name in config.artifacts.__dataclass_fields__
            if name != "success"
        )
    )
    if names != _FORCED_OUTPUTS:
        raise ContractError("forced S05 effective artifact names differ")
    return names


def _validate_forced_config_provenance(
    marker: Mapping[str, Any],
    effective_config: ForcedAppearanceConfig,
    effective_payload: Mapping[str, Any],
) -> str:
    """Bind the source-config byte hash to the semantic effective config."""

    config_hash = marker.get("config_hash")
    if not _valid_digest(config_hash):
        raise ContractError("forced S05 marker config_hash is invalid")
    records = marker.get("input_fingerprints")
    if not isinstance(records, list) or not records:
        raise ContractError("forced S05 marker lacks input fingerprints")
    parsed = [
        _validate_fingerprint_record(
            item, label="forced S05 input_fingerprints", with_mtime=False
        )
        for item in records
    ]
    matches = [item for item in parsed if item[2] == config_hash]
    if len(matches) != 1:
        raise ContractError(
            "forced S05 marker config_hash must identify exactly one source config"
        )
    source_path = Path(matches[0][0])
    if not source_path.is_absolute():
        raise ContractError("forced S05 source config path must be absolute")
    source_config, source_payload, source_hash = load_forced_appearance_config(
        source_path
    )
    if (
        source_hash != config_hash
        or source_config != effective_config
        or source_payload != effective_payload
    ):
        raise ContractError("forced S05 source/effective config provenance differs")
    return config_hash


def _validate_forced_marker(
    marker: Mapping[str, Any],
    report: Mapping[str, Any],
    forced_config: ForcedAppearanceConfig,
    effective_payload: Mapping[str, Any],
    expected: _Expectations,
) -> None:
    if set(marker) != _FORCED_MARKER_KEYS:
        raise ContractError("forced S05 success marker keys differ")
    if (
        marker.get("schema_version") != "1.0"
        or marker.get("stage") != FORCED_STAGE
        or marker.get("execution_mode") != forced_config.execution_mode
        or marker.get("operator_approved") is not True
        or marker.get("authorization_basis") != AUTHORIZATION_BASIS
        or marker.get("certification_claimed") is not False
        or marker.get("target_global_track_count") != expected.global_tracks
    ):
        raise ContractError("forced S05 marker policy/config differs")
    config_hash = _validate_forced_config_provenance(
        marker, forced_config, effective_payload
    )
    _finite_nonnegative(marker.get("elapsed_sec"), "forced S05 marker elapsed_sec")
    if not isinstance(marker.get("stats"), dict):
        raise ContractError("forced S05 marker statistics must be an object")
    if set(report) != _FORCED_REPORT_KEYS:
        raise ContractError("forced S05 report keys differ")
    if (
        report.get("schema_version") != "1.0"
        or report.get("stage") != FORCED_STAGE
        or report.get("config_hash") != config_hash
        or report.get("execution_mode") != forced_config.execution_mode
        or report.get("sequence_id") != expected.sequence_id
        or report.get("input_fingerprints") != marker.get("input_fingerprints")
    ):
        raise ContractError("forced S05 report identity/provenance differs")
    approval = report.get("operator_approval")
    semantics = report.get("evidence_semantics")
    if (
        not isinstance(approval, dict)
        or approval.get("operator_approved") is not True
        or approval.get("authorization_basis") != AUTHORIZATION_BASIS
        or approval.get("target_global_track_count") != expected.global_tracks
        or not isinstance(semantics, dict)
        or semantics.get("certification_claimed") is not False
        or semantics.get("all_global_ids_status") != FORCED_ID_STATUS
        or semantics.get("appearance_cosine_is_not_probability") is not True
        or semantics.get("model_threshold_used") is not False
    ):
        raise ContractError("forced S05 report authorization/evidence semantics differ")


def _load_forced_tables(
    directory: Path,
    forced_config: ForcedAppearanceConfig,
) -> S06ForcedTables:
    artifacts = forced_config.artifacts
    return S06ForcedTables(
        rescue_samples=_read_table(
            directory / artifacts.rescue_samples, RESCUE_SAMPLES_SCHEMA,
            "forced rescue samples",
        ),
        graded_stable_appearance=_read_table(
            directory / artifacts.graded_stable_appearance,
            GRADED_STABLE_APPEARANCE_SCHEMA, "forced graded stable appearance",
        ),
        candidate_edges=_read_table(
            directory / artifacts.candidate_edges, FORCED_CANDIDATE_EDGES_SCHEMA,
            "forced candidate edges",
        ),
        stable_to_global=_read_table(
            directory / artifacts.stable_to_global, STABLE_TO_GLOBAL_SCHEMA,
            "forced stable-to-global",
        ),
        global_tracks=_read_table(
            directory / artifacts.global_tracks, GLOBAL_TRACKS_SCHEMA,
            "forced global tracks",
        ),
        det_to_global=_read_table(
            directory / artifacts.det_to_global, DET_TO_GLOBAL_SCHEMA,
            "forced det-to-global",
        ),
    )


def _validate_forced_arrays(
    directory: Path,
    forced_config: ForcedAppearanceConfig,
    graded: pa.Table,
    rescue: pa.Table,
    expected: _Expectations,
) -> int:
    try:
        prototypes = np.load(directory / forced_config.artifacts.graded_prototypes, allow_pickle=False)
        mask = np.load(directory / forced_config.artifacts.graded_prototype_mask, allow_pickle=False)
        rescue_embeddings = np.load(
            directory / forced_config.artifacts.rescue_embeddings, allow_pickle=False
        )
    except (OSError, ValueError) as exc:
        raise ContractError(f"cannot load forced S05 arrays: {exc}") from exc
    selected_rescue = _column(rescue, "selected_for_descriptor", np.bool_)
    embedding_values = rescue["embedding_row"].to_pylist()
    embedding_rows = [int(value) for value in embedding_values if value is not None]
    observed_embedding_count = int(np.count_nonzero(selected_rescue))
    if (
        prototypes.dtype != np.float16
        or prototypes.shape
        != (
            expected.stable_tracks,
            forced_config.max_prototypes,
            forced_config.embedding_dim,
        )
        or mask.dtype != np.bool_
        or mask.shape != (expected.stable_tracks, forced_config.max_prototypes)
        or rescue_embeddings.dtype != np.float16
        or rescue_embeddings.shape
        != (observed_embedding_count, forced_config.embedding_dim)
        or not np.isfinite(prototypes).all()
        or not np.isfinite(rescue_embeddings).all()
        or np.any(prototypes[~mask] != np.float16(0.0))
        or np.any(~np.any(mask, axis=1))
    ):
        raise ContractError("forced S05 array shape/dtype/value contract differs")
    prototype_norms = np.linalg.norm(prototypes.astype(np.float32), axis=2)[mask]
    rescue_norms = np.linalg.norm(rescue_embeddings.astype(np.float32), axis=1)
    if (
        not np.allclose(prototype_norms, 1.0, rtol=0.0, atol=2e-3)
        or not np.allclose(rescue_norms, 1.0, rtol=0.0, atol=2e-3)
        or not np.array_equal(
            np.count_nonzero(mask, axis=1), _column(graded, "num_valid_prototypes", np.int64)
        )
    ):
        raise ContractError("forced S05 descriptor normalization/alignment differs")
    if (
        any(
            (value is None) == bool(is_selected)
            for value, is_selected in zip(
                embedding_values, selected_rescue, strict=True
            )
        )
        or sorted(embedding_rows) != list(range(observed_embedding_count))
    ):
        raise ContractError("forced S05 rescue embedding-row bijection differs")
    return int(prototypes.shape[1])


def _derive_c_grade_provenance(
    rescue: pa.Table,
    graded: pa.Table,
    *,
    overlap_threshold: float,
) -> tuple[Mapping[int, CGradeProvenance], tuple[str, ...]]:
    grade_rows = graded.to_pylist()
    c_rows = {
        int(row["stable_id"]): row
        for row in grade_rows
        if row["evidence_grade"] == "C_REENCODED_DEGRADED"
    }
    grouped: dict[int, list[dict[str, Any]]] = {stable_id: [] for stable_id in c_rows}
    for row in rescue.to_pylist():
        stable_id = int(row["stable_id"])
        if stable_id not in grouped:
            raise ContractError("forced rescue row references a non-C-grade stable track")
        grouped[stable_id].append(row)
    if set(grouped) != set(c_rows) or any(not rows for rows in grouped.values()):
        raise ContractError("forced rescue/C-grade stable coverage differs")
    result: dict[int, CGradeProvenance] = {}
    mismatch_ids: list[int] = []
    high_overlap_rows = 0
    for stable_id in sorted(grouped):
        candidates = grouped[stable_id]
        selected = [row for row in candidates if bool(row["selected_for_descriptor"])]
        if not selected:
            raise ContractError(f"C-grade stable has no selected rescue descriptor: {stable_id}")
        max_iou = max(float(row["other_bbox_max_iou"]) for row in selected)
        used_high = any(float(row["other_bbox_max_iou"]) >= overlap_threshold for row in selected)
        high_overlap_rows += sum(
            float(row["other_bbox_max_iou"]) >= overlap_threshold for row in selected
        )
        grade = c_rows[stable_id]
        qualities = [float(row["crop_quality"]) for row in selected]
        if (
            int(grade["num_input_samples"]) != len(selected)
            or int(grade["num_selected_samples"]) != len(selected)
            or not math.isclose(float(grade["selected_quality_min"]), min(qualities), rel_tol=0.0, abs_tol=1e-6)
            or not math.isclose(float(grade["selected_quality_mean"]), float(np.mean(qualities)), rel_tol=0.0, abs_tol=1e-6)
        ):
            raise ContractError(f"C-grade rescue selection/quality provenance differs: {stable_id}")
        stored_max = float(grade["selected_max_other_bbox_iou"])
        stored_high = bool(grade["used_high_overlap"])
        if not math.isclose(stored_max, max_iou, rel_tol=0.0, abs_tol=1e-7) or stored_high != used_high:
            mismatch_ids.append(stable_id)
        result[stable_id] = CGradeProvenance(
            stable_id=stable_id,
            num_candidate_crops=len(candidates),
            num_selected_crops=len(selected),
            selected_max_other_bbox_iou=max_iou,
            used_high_overlap=used_high,
            selected_det_ids=tuple(int(row["det_id"]) for row in selected),
            selected_embedding_rows=tuple(int(row["embedding_row"]) for row in selected),
        )
    warnings: list[str] = []
    if mismatch_ids:
        warnings.append(
            "UPSTREAM_C_GRADE_IOU_PROVENANCE_MISMATCH: rescue_samples.parquet is "
            f"authoritative; {high_overlap_rows} selected crops use IoU >= "
            f"{overlap_threshold:g}; affected stored rows={','.join(map(str, mismatch_ids))}"
        )
    return MappingProxyType(result), tuple(warnings)


def _validate_grade_and_rescue(
    tables: S06ForcedTables,
    stable: S04FinalizedBundle,
    detections: pa.Table,
    forced_config: ForcedAppearanceConfig,
    expected: _Expectations,
) -> tuple[Mapping[int, CGradeProvenance], tuple[str, ...]]:
    graded, rescue = tables.graded_stable_appearance, tables.rescue_samples
    if graded.num_rows != expected.stable_tracks:
        raise ContractError("forced graded-appearance row count differs")
    stable_ids = _column(graded, "stable_id", np.int64)
    grades = np.asarray(graded["evidence_grade"].to_pylist(), dtype=object)
    usable = _column(graded, "descriptor_usable", np.bool_)
    if (
        not np.array_equal(stable_ids, np.arange(expected.stable_tracks, dtype=np.int64))
        or not np.all(usable)
        or not set(map(str, grades)).issubset(_APPEARANCE_GRADES)
        or any(value is not None for value in graded["missing_reason"].to_pylist())
    ):
        raise ContractError("forced graded appearance ID/grade/usable contract differs")
    rescue_stable = _column(rescue, "stable_id", np.int64)
    rescue_micro = _column(rescue, "micro_id", np.int64)
    rescue_det = _column(rescue, "det_id", np.int64)
    if len(np.unique(rescue_det)) != len(rescue_det):
        raise ContractError("forced rescue rows repeat a detection")
    det_ids = _column(detections, "det_id", np.int64)
    det_positions = _locate(det_ids, rescue_det, "forced rescue/detection join")
    stable_positions = _locate(stable.det_ids, rescue_det, "forced rescue/S04 join")
    if (
        not np.all(_column(detections, "valid", np.bool_)[det_positions])
        or not np.array_equal(stable.det_stable_ids[stable_positions], rescue_stable)
        or not np.array_equal(stable.det_micro_ids[stable_positions], rescue_micro)
        or not np.array_equal(
            np.asarray(detections["clip_id"].to_pylist(), dtype=object)[det_positions],
            np.asarray(rescue["clip_id"].to_pylist(), dtype=object),
        )
        or not np.array_equal(
            _column(detections, "global_frame", np.int64)[det_positions],
            _column(rescue, "global_frame", np.int64),
        )
    ):
        raise ContractError("forced rescue detection/micro/stable provenance differs")
    for name in ("crop_quality", "other_bbox_max_iou", "clipped_fraction", "bbox_area_percentile"):
        values = _column(rescue, name, np.float64)
        if not np.all(np.isfinite(values)) or np.any((values < 0.0) | (values > 1.0)):
            raise ContractError(f"forced rescue {name} range differs")
    return _derive_c_grade_provenance(
        rescue, graded, overlap_threshold=forced_config.clean_max_other_bbox_iou
    )


def _all_null(table: pa.Table, names: Sequence[str], label: str) -> None:
    for name in names:
        if table[name].null_count != table.num_rows:
            raise ContractError(f"{label} must remain entirely null: {name}")


def _validate_path_graph(
    stable_ids: np.ndarray,
    global_ids: np.ndarray,
    path_order: np.ndarray,
    predecessors: Sequence[int | None],
    stable: S04FinalizedBundle,
    *,
    expected_global_count: int,
) -> tuple[int, int]:
    """Independently recompute cycle and temporal-overlap violations."""

    stable_ids = np.asarray(stable_ids, dtype=np.int64)
    global_ids = np.asarray(global_ids, dtype=np.int64)
    path_order = np.asarray(path_order, dtype=np.int64)
    if len({int(value) for value in stable_ids}) != len(stable_ids):
        raise ContractError("forced path graph repeats a stable ID")
    by_id = {int(stable_id): index for index, stable_id in enumerate(stable_ids)}
    predecessor_map: dict[int, int] = {}
    temporal_violations = 0
    predecessor_order_differs = False
    for index, value in enumerate(predecessors):
        stable_id = int(stable_ids[index])
        if value is None:
            if int(path_order[index]) != 0:
                raise ContractError("forced path non-start lacks a predecessor")
            continue
        predecessor = int(value)
        if predecessor not in by_id or predecessor == stable_id:
            raise ContractError("forced path predecessor is unknown/self")
        pred_index = by_id[predecessor]
        if int(global_ids[pred_index]) != int(global_ids[index]):
            raise ContractError("forced path predecessor global/order differs")
        predecessor_order_differs |= (
            int(path_order[pred_index]) + 1 != int(path_order[index])
        )
        predecessor_map[stable_id] = predecessor
        left = stable.stable_tracklets[predecessor]
        right = stable.stable_tracklets[stable_id]
        temporal_violations += int(
            left.end_global_frame >= right.start_global_frame
            or left.end_time_sec >= right.start_time_sec
        )
    cycles = 0
    state: dict[int, int] = {}

    def visit(node: int) -> None:
        nonlocal cycles
        status = state.get(node, 0)
        if status == 1:
            cycles += 1
            return
        if status == 2:
            return
        state[node] = 1
        predecessor = predecessor_map.get(node)
        if predecessor is not None:
            visit(predecessor)
        state[node] = 2

    for stable_id in map(int, stable_ids):
        visit(stable_id)
    if set(map(int, global_ids)) != set(range(expected_global_count)):
        raise ContractError("forced global IDs are not dense")
    for global_id in range(expected_global_count):
        orders = sorted(map(int, path_order[global_ids == global_id]))
        if orders != list(range(len(orders))):
            raise ContractError("forced path order is not contiguous")
    # A directed cycle necessarily makes at least one predecessor order
    # impossible.  Return the independently measured cycle count so callers
    # reject it as an invariant violation rather than masking it as metadata.
    if predecessor_order_differs and cycles == 0:
        raise ContractError("forced path predecessor global/order differs")
    return cycles, temporal_violations


def _validate_global_graph(
    tables: S06ForcedTables,
    stable: S04FinalizedBundle,
    detections: pa.Table,
    expected: _Expectations,
) -> tuple[int, int, int]:
    candidate = tables.candidate_edges
    mapping = tables.stable_to_global
    globals_table = tables.global_tracks
    det_global = tables.det_to_global
    if (
        mapping.num_rows != expected.stable_tracks
        or globals_table.num_rows != expected.global_tracks
        or det_global.num_rows != expected.valid_detections
    ):
        raise ContractError("forced graph table row count differs")
    source = _column(candidate, "source_stable_id", np.int64)
    target = _column(candidate, "target_stable_id", np.int64)
    selected = _column(candidate, "selected_by_solver", np.bool_)
    candidate_ids = candidate["candidate_id"].to_pylist()
    link_ids = candidate["global_link_id"].to_pylist()
    observed_selected_links = int(np.count_nonzero(selected))
    required_selected_links = mapping.num_rows - globals_table.num_rows
    if (
        len(set(candidate_ids)) != len(candidate_ids)
        or len(set(zip(map(int, source), map(int, target), strict=True))) != len(source)
        or np.any(source < 0)
        or np.any(source >= expected.stable_tracks)
        or np.any(target < 0)
        or np.any(target >= expected.stable_tracks)
        or np.any(source == target)
        or observed_selected_links != required_selected_links
        or any((link_id is None) == bool(is_selected) for link_id, is_selected in zip(link_ids, selected, strict=True))
        or len({str(link_ids[index]) for index in np.flatnonzero(selected)})
        != observed_selected_links
        or not np.all(_column(candidate, "strictly_nonoverlapping", np.bool_))
        or set(candidate["authorization_basis"].to_pylist()) != {AUTHORIZATION_BASIS}
    ):
        raise ContractError("forced candidate identity/selection contract differs")
    statuses = np.asarray(candidate["id_status"].to_pylist(), dtype=object)
    if not np.all(statuses[selected] == FORCED_ID_STATUS) or not np.all(statuses[~selected] == "candidate_only"):
        raise ContractError("forced candidate id_status semantics differ")
    stable_tracklets = stable.stable_tracklets
    expected_source_frame = np.asarray(
        [stable_tracklets[int(value)].end_global_frame for value in source], dtype=np.int64
    )
    expected_target_frame = np.asarray(
        [stable_tracklets[int(value)].start_global_frame for value in target], dtype=np.int64
    )
    expected_source_time = np.asarray(
        [stable_tracklets[int(value)].end_time_sec for value in source], dtype=np.float64
    )
    expected_target_time = np.asarray(
        [stable_tracklets[int(value)].start_time_sec for value in target], dtype=np.float64
    )
    if (
        not np.array_equal(_column(candidate, "source_end_global_frame", np.int64), expected_source_frame)
        or not np.array_equal(_column(candidate, "target_start_global_frame", np.int64), expected_target_frame)
        or not np.array_equal(_column(candidate, "source_end_time_sec", np.float64), expected_source_time)
        or not np.array_equal(_column(candidate, "target_start_time_sec", np.float64), expected_target_time)
        or not np.allclose(
            _column(candidate, "temporal_gap_sec", np.float64),
            expected_target_time - expected_source_time, rtol=0.0, atol=1e-12,
        )
        or np.any(expected_source_frame >= expected_target_frame)
        or np.any(expected_source_time >= expected_target_time)
    ):
        raise ContractError("forced candidate temporal endpoint provenance differs")
    cosine = _column(candidate, "appearance_cosine", np.float64)
    if not np.all(np.isfinite(cosine)) or np.any((cosine < -1.0) | (cosine > 1.0)):
        raise ContractError("forced candidate appearance cosine range differs")
    grade_by_stable = np.asarray(
        tables.graded_stable_appearance["evidence_grade"].to_pylist(),
        dtype=object,
    )
    if (
        not np.array_equal(
            np.asarray(candidate["source_evidence_grade"].to_pylist(), dtype=object),
            grade_by_stable[source],
        )
        or not np.array_equal(
            np.asarray(candidate["target_evidence_grade"].to_pylist(), dtype=object),
            grade_by_stable[target],
        )
    ):
        raise ContractError("forced candidate evidence grades differ from appearance table")

    stable_ids = _column(mapping, "stable_id", np.int64)
    global_ids = _column(mapping, "global_track_id", np.int64)
    path_order = _column(mapping, "order_in_global_path", np.int64)
    predecessors = mapping["predecessor_stable_id"].to_pylist()
    if not np.array_equal(stable_ids, np.arange(expected.stable_tracks, dtype=np.int64)):
        raise ContractError("forced stable-to-global stable IDs are not dense/canonical")
    _all_null(
        mapping,
        (
            "predecessor_link_probability", "predecessor_link_margin",
            "component_min_link_probability", "component_mean_link_probability",
            "component_max_link_probability",
        ),
        "forced stable-to-global probability",
    )
    if (
        set(mapping["identity_basis"].to_pylist()) != {AUTHORIZATION_BASIS}
        or set(mapping["id_status"].to_pylist()) != {FORCED_ID_STATUS}
    ):
        raise ContractError("forced stable mapping identity semantics differ")
    cycles, temporal = _validate_path_graph(
        stable_ids, global_ids, path_order, predecessors, stable,
        expected_global_count=expected.global_tracks,
    )
    selected_by_pair = {
        (int(source[index]), int(target[index])): index
        for index in np.flatnonzero(selected)
    }
    predecessor_candidate = mapping["predecessor_candidate_id"].to_pylist()
    predecessor_link = mapping["predecessor_global_link_id"].to_pylist()
    for index, predecessor in enumerate(predecessors):
        if predecessor is None:
            if (
                predecessor_candidate[index] is not None
                or predecessor_link[index] is not None
                or mapping["predecessor_authorization_basis"][index].as_py() is not None
            ):
                raise ContractError("forced path start contains predecessor evidence")
            continue
        pair = (int(predecessor), int(stable_ids[index]))
        candidate_index = selected_by_pair.get(pair)
        if (
            candidate_index is None
            or predecessor_candidate[index] != candidate_ids[candidate_index]
            or predecessor_link[index] != link_ids[candidate_index]
            or mapping["predecessor_authorization_basis"][index].as_py() != AUTHORIZATION_BASIS
        ):
            raise ContractError("forced selected candidate/path predecessor bijection differs")
    if len(selected_by_pair) != sum(value is not None for value in predecessors):
        raise ContractError("forced selected candidates/path links are not bijective")

    global_track_ids = _column(globals_table, "global_track_id", np.int64)
    if not np.array_equal(global_track_ids, np.arange(expected.global_tracks, dtype=np.int64)):
        raise ContractError("forced global-track IDs are not dense/canonical")
    _all_null(
        globals_table,
        (
            "min_link_probability", "p10_link_probability", "mean_link_probability",
            "max_link_probability", "min_link_margin",
        ),
        "forced global-track probability",
    )
    if (
        set(globals_table["identity_basis"].to_pylist()) != {AUTHORIZATION_BASIS}
        or set(globals_table["id_status"].to_pylist()) != {FORCED_ID_STATUS}
        or globals_table["global_track_uuid"].null_count
        or len(set(globals_table["global_track_uuid"].to_pylist())) != expected.global_tracks
    ):
        raise ContractError("forced global-track identity semantics differ")
    mapping_uuid = mapping["global_track_uuid"].to_pylist()
    mapping_display = mapping["display_global_id"].to_pylist()
    global_uuid = globals_table["global_track_uuid"].to_pylist()
    global_display = globals_table["display_global_id"].to_pylist()
    if global_display != [f"G{index + 1:04d}" for index in range(expected.global_tracks)]:
        raise ContractError("forced display global IDs differ")
    for index in range(expected.stable_tracks):
        global_id = int(global_ids[index])
        if mapping_uuid[index] != global_uuid[global_id] or mapping_display[index] != global_display[global_id]:
            raise ContractError("forced stable/global UUID or display-ID join differs")
    for global_id in range(expected.global_tracks):
        selected_stables = stable_ids[global_ids == global_id]
        row = globals_table.slice(global_id, 1).to_pylist()[0]
        tracklets = [stable_tracklets[int(value)] for value in selected_stables]
        if (
            int(row["num_stable_tracklets"]) != len(tracklets)
            or int(row["num_microtracklets"]) != sum(item.num_microtracklets for item in tracklets)
            or int(row["num_detections"]) != sum(item.num_detections for item in tracklets)
            or int(row["num_long_links"]) != len(tracklets) - 1
            or int(row["first_stable_id"]) != int(selected_stables[np.argmin(path_order[global_ids == global_id])])
            or int(row["last_stable_id"]) != int(selected_stables[np.argmax(path_order[global_ids == global_id])])
        ):
            raise ContractError("forced global-track aggregate differs")

    det_ids = _column(det_global, "det_id", np.int64)
    raw_ids = _column(detections, "det_id", np.int64)
    raw_valid = _column(detections, "valid", np.bool_)
    if len(np.unique(det_ids)) != len(det_ids) or set(map(int, det_ids)) != set(map(int, raw_ids[raw_valid])):
        raise ContractError("forced det-to-global is not a bijection over valid detections")
    raw_positions = _locate(raw_ids, det_ids, "forced det-to-global/S00 join")
    stable_positions = _locate(stable.det_ids, det_ids, "forced det-to-global/S04 join")
    if (
        not np.all(_column(det_global, "valid", np.bool_))
        or not np.array_equal(_column(det_global, "micro_id", np.int64), stable.det_micro_ids[stable_positions])
        or not np.array_equal(_column(det_global, "stable_id", np.int64), stable.det_stable_ids[stable_positions])
        or not np.array_equal(
            _column(det_global, "global_track_id", np.int64),
            global_ids[_column(det_global, "stable_id", np.int64)],
        )
        or not np.array_equal(
            _column(det_global, "global_frame", np.int64),
            _column(detections, "global_frame", np.int64)[raw_positions],
        )
        or not np.array_equal(
            np.asarray(det_global["clip_id"].to_pylist(), dtype=object),
            np.asarray(detections["clip_id"].to_pylist(), dtype=object)[raw_positions],
        )
        or set(det_global["identity_basis"].to_pylist()) != {AUTHORIZATION_BASIS}
        or set(det_global["id_status"].to_pylist()) != {FORCED_ID_STATUS}
    ):
        raise ContractError("forced detection identity/join semantics differ")
    frame = _column(det_global, "global_frame", np.int64)
    det_gid = _column(det_global, "global_track_id", np.int64)
    composite = frame * expected.global_tracks + det_gid
    same_frame = len(composite) - len(np.unique(composite))
    return cycles, temporal, same_frame


def _maximum_concurrent_stable_tracks(stable: S04FinalizedBundle) -> int:
    events = [
        event
        for tracklet in stable.stable_tracklets.values()
        for event in (
            (int(tracklet.start_global_frame), 0),
            (int(tracklet.end_global_frame), 1),
        )
    ]
    if not events:
        raise ContractError("forced S05 has no stable-track intervals")
    concurrent = 0
    maximum = 0
    for _frame, kind in sorted(events):
        if kind == 0:
            concurrent += 1
            maximum = max(maximum, concurrent)
        else:
            concurrent -= 1
        if concurrent < 0:
            raise ContractError("forced stable-track interval events are unbalanced")
    if concurrent != 0 or maximum <= 0:
        raise ContractError("forced stable-track interval events are unbalanced")
    return maximum


def _validate_forced_report_counts(
    marker: Mapping[str, Any],
    report: Mapping[str, Any],
    tables: S06ForcedTables,
    stable: S04FinalizedBundle,
    cycles: int,
    temporal: int,
    same_frame: int,
    expected: _Expectations,
    forced_config: ForcedAppearanceConfig,
    prototype_slots: int,
) -> None:
    """Cross-check S05 metadata against the current persisted tables only."""

    coverage = report.get("coverage")
    grades = report.get("graded_appearance")
    rescue = report.get("rescue")
    solver = report.get("solver")
    grade_values = list(
        map(str, tables.graded_stable_appearance["evidence_grade"].to_pylist())
    )
    grade_counts = dict(sorted(Counter(grade_values).items()))
    c_grade_count = grade_counts.get("C_REENCODED_DEGRADED", 0)
    rescue_selected = _column(
        tables.rescue_samples, "selected_for_descriptor", np.bool_
    )
    rescue_reasons = np.asarray(
        tables.rescue_samples["selection_reason"].to_pylist(), dtype=object
    )
    rescue_review_excluded = _column(
        tables.rescue_samples, "review_excluded", np.bool_
    )
    selected = _column(tables.candidate_edges, "selected_by_solver", np.bool_)
    observed_selected_links = int(np.count_nonzero(selected))
    observed_candidate_edges = tables.candidate_edges.num_rows
    observed_rescue_embeddings = int(np.count_nonzero(rescue_selected))
    observed_global_tracks = tables.global_tracks.num_rows
    max_concurrent = _maximum_concurrent_stable_tracks(stable)
    observed_marker_stats = {
        "num_stable_tracks": tables.stable_to_global.num_rows,
        "num_selected_links": observed_selected_links,
        "num_global_tracks": observed_global_tracks,
        "num_rescue_stable_tracks": c_grade_count,
        "num_rescue_embeddings": observed_rescue_embeddings,
        "num_candidate_edges": observed_candidate_edges,
        "max_concurrent_stable_tracks": max_concurrent,
    }
    if marker.get("stats") != observed_marker_stats:
        raise ContractError("forced S05 marker statistics differ from current tables")
    expected_coverage = {
        "stable_tracks_mapped_once": expected.stable_tracks,
        "microtracklets_mapped_once": expected.microtracks,
        "valid_detections_mapped_once": expected.valid_detections,
        "invalid_detections_excluded": expected.invalid_detections,
        "same_frame_same_global_id_violations": same_frame,
        "temporal_overlap_link_violations": temporal,
        "cycles": cycles,
        "exact_global_track_count": expected.global_tracks,
    }
    if coverage != expected_coverage or cycles or temporal or same_frame:
        raise ContractError("forced S05 independently recomputed graph invariants differ")
    expected_grades = {
        "counts_by_grade": grade_counts,
        "all_stable_tracks_have_descriptor": True,
        "num_stable_tracks": tables.graded_stable_appearance.num_rows,
        "prototype_slots": prototype_slots,
    }
    if grades != expected_grades:
        raise ContractError(
            "forced S05 report graded appearance differs from current tables"
        )
    expected_rescue = {
        "zero_s02_sample_stable_tracks": c_grade_count,
        "candidate_crops_decoded": tables.rescue_samples.num_rows,
        "selected_crops_encoded": observed_rescue_embeddings,
        "selected_best_degraded_crops": int(
            np.count_nonzero(rescue_reasons == "best_degraded_fallback")
        ),
        "selected_review_excluded_crops": int(
            np.count_nonzero(rescue_selected & rescue_review_excluded)
        ),
    }
    if rescue != expected_rescue:
        raise ContractError("forced S05 report rescue counts differ from current tables")
    if not isinstance(solver, dict):
        raise ContractError("forced S05 report solver must be an object")
    expected_solve_pass = {
        "pass_index": 0,
        "source_clip_ids": list(expected.clip_order),
        "solver_nodes": expected.stable_tracks,
        "candidate_count": observed_candidate_edges,
        "interval_width": max_concurrent,
        "backbone_chain_count": max_concurrent,
        "maximum_feasible_links": expected.stable_tracks - max_concurrent,
        "required_links": expected.stable_tracks - observed_global_tracks,
        "selected_links": observed_selected_links,
        "total_appearance_cost_int": int(
            np.sum(
                _column(
                    tables.candidate_edges, "appearance_cost_int", np.int64
                )[selected],
                dtype=np.int64,
            )
        ),
    }
    solve_passes = solver.get("solve_passes")
    if (
        solver.get("solve_pass_count") != 1
        or not isinstance(solve_passes, list)
        or len(solve_passes) != 1
        or solve_passes[0] != expected_solve_pass
    ):
        raise ContractError(
            "forced S05 report must contain exactly one validated full-sequence "
            "solve pass"
        )
    observed_solver = {
        "solve_mode": "full_sequence_single_solve",
        "algorithm": "deterministic_full_sequence_fixed_cardinality_min_cost_flow",
        "objective": (
            "exact_62_global_paths_then_minimum_full_sequence_appearance_cost"
        ),
        "candidate_scope": "complete_sequence",
        "assignment": "exact_fixed_flow_min_cost_network_without_dummies",
        "cardinality_certificate": "minimum_width_interval_backbone",
        "solve_pass_count": 1,
        "solve_passes": [expected_solve_pass],
        "candidate_count": observed_candidate_edges,
        "source_top_k": forced_config.source_top_k,
        "target_top_k": forced_config.target_top_k,
        "interval_width": max_concurrent,
        "backbone_chain_count": max_concurrent,
        "maximum_feasible_links": expected.stable_tracks - max_concurrent,
        "required_links": expected.stable_tracks - observed_global_tracks,
        "selected_links": observed_selected_links,
        "global_paths": observed_global_tracks,
        "total_appearance_cost_int": int(
            np.sum(
                _column(tables.candidate_edges, "appearance_cost_int", np.int64)[
                    selected
                ],
                dtype=np.int64,
            )
        ),
        "selected_prior_links": int(
            np.count_nonzero(
                _column(tables.candidate_edges, "prior_global_link", np.bool_)[
                    selected
                ]
            )
        ),
    }
    if any(solver.get(name) != value for name, value in observed_solver.items()):
        raise ContractError("forced S05 report solver differs from current tables")
    values = _column(tables.candidate_edges, "appearance_cosine", np.float64)[selected]
    if not len(values):
        raise ContractError("forced S05 selected-cosine summary has no selected links")
    summary = solver.get("selected_appearance_cosine")
    computed = {
        "min": float(np.min(values)),
        "p10": float(np.quantile(values, 0.1, method="linear")),
        "mean": float(np.mean(values)),
        "max": float(np.max(values)),
    }
    if not isinstance(summary, dict) or any(
        not math.isclose(float(summary.get(name, math.nan)), value, rel_tol=0.0, abs_tol=1e-12)
        for name, value in computed.items()
    ):
        raise ContractError(
            "forced S05 selected-cosine summary differs from current tables"
        )
    sizes = Counter(map(int, tables.global_tracks["num_stable_tracklets"].to_pylist()))
    distribution = {str(size): count for size, count in sorted(sizes.items())}
    if report.get("global_path_size_distribution") != distribution:
        raise ContractError("forced S05 global path-size distribution differs")


def _verify_required_forced_provenance(
    records: Sequence[FileFingerprint],
    *,
    config_hash: str,
    manifest_path: Path,
    video_paths: Mapping[str, Path],
    s00_paths: Sequence[Path],
    s01_paths: Sequence[Path],
    stable: S04FinalizedBundle,
) -> None:
    by_path = {Path(item.path): item for item in records}
    config_matches = [item for item in records if item.sha256 == config_hash]
    if len(config_matches) != 1:
        raise ContractError(
            "forced S05 source config role is not uniquely identified"
        )
    sequence_root = stable.directory.parent.resolve()
    appearance_dir = sequence_root / "02_appearance"
    encoder_choice = _read_json(
        appearance_dir / "encoder_choice.json",
        "S02 encoder choice",
    )
    if not isinstance(encoder_choice, dict) or not isinstance(
        encoder_choice.get("checkpoint_path"), str
    ):
        raise ContractError("S02 encoder choice lacks checkpoint_path")
    checkpoint_path = Path(encoder_choice["checkpoint_path"])
    if not checkpoint_path.is_absolute():
        raise ContractError("S02 encoder checkpoint path must be absolute")
    required = {
        Path(config_matches[0].path),
        manifest_path.resolve(),
        *(path.resolve() for path in video_paths.values()),
        *(path.resolve() for path in s00_paths if path.name in {"_SUCCESS.json", "frames.parquet", "detections.parquet", "resolved_manifest.json"}),
        *(path.resolve() for path in s01_paths if path.name in {"_SUCCESS.json", "det_to_micro.parquet", "microtracklets.parquet"}),
        *(path.resolve() for path in stable.consumed_paths),
        *(
            (appearance_dir / name).resolve()
            for name in _S04_RECORDED_INPUT_ROLES["02_appearance"]
        ),
        (sequence_root / "05_global_link" / "_SUCCESS.json").resolve(),
        (
            sequence_root
            / "05_global_link"
            / "stable_to_global.parquet"
        ).resolve(),
        checkpoint_path.resolve(),
    }
    if set(by_path) != required:
        missing = sorted(str(path) for path in required - set(by_path))
        extra = sorted(str(path) for path in set(by_path) - required)
        raise ContractError(
            "forced S05 marker provenance role set differs; "
            f"missing={missing}, extra={extra}"
        )


def _verify_stable_cache(cache: Mapping[Path, tuple[FileFingerprint, _StatToken]]) -> None:
    for path, (_, before) in cache.items():
        if _token(path) != before:
            raise ContractError(f"S06 input changed while being loaded: {path}")


def load_s06_inputs(
    manifest_path: Path,
    ingest_dir: Path,
    microtrack_dir: Path,
    stable_dir: Path,
    forced_dir: Path,
    *,
    config: S06ExportConfig | None = None,
    logger: LogFn = lambda _message: None,
) -> S06InputBundle:
    """Strict-load the fixed forced-provisional S06 inputs without any writes."""

    if not callable(logger):
        raise ContractError("S06 input logger must be callable")
    expected = _expectations(config)
    manifest_path, ingest_dir, microtrack_dir, stable_dir, forced_dir = (
        path.resolve()
        for path in (manifest_path, ingest_dir, microtrack_dir, stable_dir, forced_dir)
    )
    if len({manifest_path, ingest_dir, microtrack_dir, stable_dir, forced_dir}) != 5:
        raise ContractError("S06 input paths overlap by identity")
    cache: dict[Path, tuple[FileFingerprint, _StatToken]] = {}
    _fingerprint(manifest_path, cache, logger=logger)

    logger("[s06-input] validating manifest and complete S00 snapshot")
    manifest_hash, video_paths, frames, detections, s00_paths, _s00_marker = (
        _validate_manifest_and_s00(
            manifest_path, ingest_dir, expected, cache, logger=logger
        )
    )
    logger("[s06-input] validating complete S01 snapshot and micro bijection")
    det_to_micro, microtracklets, s01_paths, expected = _validate_s01(
        microtrack_dir,
        detections,
        s00_paths,
        manifest_path.parent.parent,
        expected,
        cache,
        logger=logger,
    )
    logger("[s06-input] strict-loading finalized S04 snapshot")
    s04_relocations = _s04_recorded_input_relocations(
        stable_dir,
        manifest_path.parent.parent,
    )
    logger(
        "[s06-input] validating 27 S04 provenance roles against the current "
        "repository copy"
    )
    stable = load_s04_finalized(
        stable_dir,
        recorded_input_relocations=s04_relocations,
        source_config_path=(
            manifest_path.parent.parent / "configs" / "s04_finalize.yaml"
        ),
        logger=logger,
    )
    for path, fingerprint in zip(stable.consumed_paths, stable.input_fingerprints, strict=True):
        current = _fingerprint(path, cache, logger=logger)
        if current != fingerprint:
            raise ContractError(f"S04 bundle fingerprint changed: {path}")
    stable_track_count = _resolve_expected_count(
        expected.stable_tracks, len(stable.stable_ids), "stable-track count"
    )
    expected = replace(expected, stable_tracks=stable_track_count)
    _validate_s04_cross_stage(stable, det_to_micro, expected)

    logger("[s06-input] validating forced S05 marker, tree and provenance")
    marker = _read_json(forced_dir / "_SUCCESS.json", "forced S05 success marker")
    if not isinstance(marker, dict):
        raise ContractError("forced S05 success marker must be an object")
    forced_paths = _snapshot_output_tree(
        forced_dir,
        marker,
        _FORCED_OUTPUTS,
        cache,
        label="S05_FORCE_APPEARANCE",
        logger=logger,
    )
    forced_config, effective_payload, _effective_file_hash = load_forced_appearance_config(
        forced_dir / "effective_config.json"
    )
    _forced_artifact_names(forced_config)
    report = _read_json(
        forced_dir / forced_config.artifacts.report, "forced S05 report"
    )
    if not isinstance(report, dict):
        raise ContractError("forced S05 report must be an object")
    _validate_forced_marker(
        marker, report, forced_config, effective_payload, expected
    )
    recorded_forced_metadata = _recorded_input_fingerprints(
        marker.get("input_fingerprints"),
        label="forced S05 input_fingerprints",
        with_mtime=False,
    )
    _verify_required_forced_provenance(
        recorded_forced_metadata,
        config_hash=str(marker["config_hash"]),
        manifest_path=manifest_path,
        video_paths=video_paths,
        s00_paths=s00_paths, s01_paths=s01_paths, stable=stable,
    )
    _verify_recorded_inputs(
        marker.get("input_fingerprints"),
        label="forced S05 input_fingerprints",
        with_mtime=False,
        cache=cache,
        logger=logger,
    )
    tables = _load_forced_tables(forced_dir, forced_config)
    prototype_slots = _validate_forced_arrays(
        forced_dir, forced_config, tables.graded_stable_appearance,
        tables.rescue_samples, expected,
    )
    c_provenance, provenance_warnings = _validate_grade_and_rescue(
        tables, stable, detections, forced_config, expected
    )
    cycles, temporal, same_frame = _validate_global_graph(
        tables, stable, detections, expected
    )
    _validate_forced_report_counts(
        marker,
        report,
        tables,
        stable,
        cycles,
        temporal,
        same_frame,
        expected,
        forced_config,
        prototype_slots,
    )
    warnings = provenance_warnings

    _verify_stable_cache(cache)
    ordered = tuple(sorted(cache))
    fingerprints = tuple(cache[path][0] for path in ordered)
    return S06InputBundle(
        manifest_path=manifest_path,
        manifest_hash=manifest_hash,
        ingest_dir=ingest_dir,
        microtrack_dir=microtrack_dir,
        forced_dir=forced_dir,
        frames=frames,
        detections=detections,
        det_to_micro=det_to_micro,
        microtracklets=microtracklets,
        s04=stable,
        forced=tables,
        c_grade_provenance=c_provenance,
        forced_report=_freeze_json(report),
        forced_success_marker=_freeze_json(marker),
        video_paths=MappingProxyType(dict(video_paths)),
        warnings=tuple(warnings),
        consumed_paths=ordered,
        input_fingerprints=fingerprints,
        input_stat_tokens=MappingProxyType(
            {
                str(path): (token.size, token.mtime_ns, token.inode)
                for path, (_, token) in sorted(cache.items())
            }
        ),
    )


__all__ = [
    "AUTHORIZATION_BASIS",
    "CGradeProvenance",
    "FORCED_ID_STATUS",
    "FORCED_STAGE",
    "S06ForcedTables",
    "S06InputBundle",
    "load_s06_inputs",
]
