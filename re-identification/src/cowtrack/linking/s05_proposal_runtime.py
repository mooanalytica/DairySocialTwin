"""Strict read-only loader for one completed S05 proposal graph.

The bounded review renderer consumes only a sample of this graph.  Global
finalization must instead consume and validate the complete candidate and
proposal Parquets, including their transitive input-byte provenance.
"""

from __future__ import annotations

import json
import math
import stat
from dataclasses import dataclass
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

from cowtrack.config import ContractError
from cowtrack.linking.long_proposal_config import (
    LongProposalConfig,
    load_long_proposal_config,
)
from cowtrack.linking.long_proposals import long_review_manifest
from cowtrack.linking.runtime import FileFingerprint, fingerprint_file
from cowtrack.linking.s05_runtime import S05LongCalibrationBundle
from cowtrack.schemas.s05_proposals import (
    LONG_CANDIDATE_EDGES_SCHEMA,
    LONG_LINK_PROPOSALS_SCHEMA,
)
from cowtrack.stages.s05_propose import validate_s05_proposal_rows


_STAGE = "S05_PROPOSE"
_OUTPUT_NAMES = (
    "effective_config.json",
    "long_candidate_edges.parquet",
    "long_link_proposals.parquet",
    "s05_proposal_report.json",
    "s05_review_manifest.json",
)
_CONSUMED_NAMES = ("_SUCCESS.json", *_OUTPUT_NAMES)
_MARKER_KEYS = {
    "schema_version",
    "stage",
    "config_hash",
    "execution_mode",
    "automatic_merge_allowed",
    "confirmed_links_allowed",
    "solver_used",
    "path_cover_used",
    "num_merges",
    "input_fingerprints",
    "output_fingerprints",
    "stats",
    "elapsed_sec",
}
_STATS_KEYS = {
    "num_candidates",
    "num_proposals",
    "num_rejects",
    "num_confirmed",
    "num_solver_selected",
    "num_merges",
}


@dataclass(frozen=True)
class S05ProposalBundle:
    directory: Path
    config: LongProposalConfig
    config_hash: str
    effective_config: Mapping[str, Any]
    candidates: tuple[Mapping[str, Any], ...]
    proposals: tuple[Mapping[str, Any], ...]
    review_manifest: Mapping[str, Any]
    report: Mapping[str, Any]
    success_marker: Mapping[str, Any]
    consumed_paths: tuple[Path, ...]
    consumed_fingerprints: tuple[FileFingerprint, ...]
    input_fingerprints: tuple[FileFingerprint, ...]
    output_fingerprints: tuple[FileFingerprint, ...]


def _read_json(path: Path, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read {label} {path}: {exc}") from exc


def _read_parquet(path: Path, schema: pa.Schema, label: str) -> list[dict[str, Any]]:
    try:
        parquet = pq.ParquetFile(path)
        if not parquet.schema_arrow.equals(schema, check_metadata=False):
            raise ContractError(f"{label} schema mismatch: {path}")
        return pq.read_table(path).to_pylist()
    except ContractError:
        raise
    except (OSError, pa.ArrowException, TypeError, ValueError) as exc:
        raise ContractError(f"cannot read {label} {path}: {exc}") from exc


def _validate_hex_digest(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ContractError(f"{label} is not a lowercase SHA-256 digest")
    return value


def _fingerprint_records(
    value: Any, *, relative: bool, label: str
) -> tuple[tuple[str, int, str], ...]:
    if not isinstance(value, list) or not value:
        raise ContractError(f"{label} must be a non-empty list")
    result: list[tuple[str, int, str]] = []
    for item in value:
        if not isinstance(item, dict) or set(item) != {"path", "size_bytes", "sha256"}:
            raise ContractError(f"{label} contains an invalid record")
        raw_path, size = item["path"], item["size_bytes"]
        if (
            not isinstance(raw_path, str)
            or not raw_path
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
        ):
            raise ContractError(f"{label} contains invalid path/size fields")
        path = Path(raw_path)
        if relative:
            if path.is_absolute() or path.name != raw_path or raw_path in {".", ".."}:
                raise ContractError(f"{label} path is not a canonical artifact name")
        elif not path.is_absolute() or path.resolve() != path:
            raise ContractError(f"{label} path is not canonical absolute")
        result.append((raw_path, size, _validate_hex_digest(item["sha256"], label)))
    if [path for path, _, _ in result] != sorted(path for path, _, _ in result):
        raise ContractError(f"{label} order is not canonical")
    if len({path for path, _, _ in result}) != len(result):
        raise ContractError(f"{label} contains duplicate paths")
    return tuple(result)


def _recorded_input_snapshot(value: Any) -> tuple[FileFingerprint, ...]:
    records = _fingerprint_records(
        value, relative=False, label="S05 proposal input_fingerprints"
    )
    result: list[FileFingerprint] = []
    for raw_path, size, sha256 in records:
        path = Path(raw_path)
        try:
            mode = path.stat().st_mode
        except OSError as exc:
            raise ContractError(f"cannot stat recorded S05 proposal input {path}: {exc}") from exc
        if not stat.S_ISREG(mode) or stat.S_ISLNK(mode):
            raise ContractError(f"recorded S05 proposal input is not a regular file: {path}")
        current = fingerprint_file(path)
        if current.size_bytes != size or current.sha256 != sha256:
            raise ContractError(f"recorded S05 proposal input changed: {path}")
        result.append(current)
    return tuple(result)


def _validate_marker(marker: Any) -> Mapping[str, Any]:
    if not isinstance(marker, dict) or set(marker) != _MARKER_KEYS:
        raise ContractError("S05 proposal success marker fields differ")
    elapsed = marker.get("elapsed_sec")
    if (
        marker.get("schema_version") != "1.0"
        or marker.get("stage") != _STAGE
        or marker.get("execution_mode") != "long_proposal_only"
        or marker.get("automatic_merge_allowed") is not False
        or marker.get("confirmed_links_allowed") is not False
        or marker.get("solver_used") is not False
        or marker.get("path_cover_used") is not False
        or marker.get("num_merges") != 0
        or isinstance(elapsed, bool)
        or not isinstance(elapsed, (int, float))
        or not math.isfinite(float(elapsed))
        or float(elapsed) < 0.0
    ):
        raise ContractError("S05 proposal success marker policy differs")
    _validate_hex_digest(marker.get("config_hash"), "S05 proposal config_hash")
    stats = marker.get("stats")
    if not isinstance(stats, dict) or set(stats) != _STATS_KEYS:
        raise ContractError("S05 proposal success marker stats fields differ")
    if any(type(value) is not int or value < 0 for value in stats.values()):
        raise ContractError("S05 proposal success marker stats are invalid")
    if (
        stats["num_candidates"] != stats["num_proposals"] + stats["num_rejects"]
        or any(
            stats[name] != 0
            for name in ("num_confirmed", "num_solver_selected", "num_merges")
        )
    ):
        raise ContractError("S05 proposal success marker stats are inconsistent")
    return marker


def _verify_output_fingerprints(
    directory: Path,
    marker: Mapping[str, Any],
    current: Sequence[FileFingerprint],
) -> tuple[FileFingerprint, ...]:
    records = _fingerprint_records(
        marker.get("output_fingerprints"),
        relative=True,
        label="S05 proposal output_fingerprints",
    )
    if tuple(name for name, _, _ in records) != _OUTPUT_NAMES:
        raise ContractError("S05 proposal output fingerprint set differs")
    by_name = {Path(item.path).name: item for item in current}
    result: list[FileFingerprint] = []
    for name, size, sha256 in records:
        item = by_name.get(name)
        if item is None or item.size_bytes != size or item.sha256 != sha256:
            raise ContractError(f"completed S05 proposal artifact changed: {name}")
        if Path(item.path).parent != directory:
            raise ContractError("S05 proposal artifact resolved outside its directory")
        result.append(item)
    return tuple(result)


def load_s05_proposals(
    output_dir: Path,
    long_runtime: S05LongCalibrationBundle,
    *,
    expected_stable_count: int,
) -> S05ProposalBundle:
    """Fully validate and return the complete proposal graph, never its review sample."""

    directory = output_dir.resolve()
    if not directory.is_dir():
        raise ContractError(f"S05 proposal directory does not exist: {directory}")
    entries = tuple(directory.iterdir())
    try:
        all_regular = all(
            stat.S_ISREG(path.lstat().st_mode)
            and not stat.S_ISLNK(path.lstat().st_mode)
            for path in entries
        )
    except OSError as exc:
        raise ContractError(f"cannot inspect S05 proposal artifact tree: {exc}") from exc
    if {path.name for path in entries} != set(_CONSUMED_NAMES) or not all_regular:
        raise ContractError("S05 proposal artifact tree differs")
    consumed_paths = tuple(directory / name for name in _CONSUMED_NAMES)
    before = tuple(fingerprint_file(path) for path in consumed_paths)
    marker = _validate_marker(
        _read_json(directory / "_SUCCESS.json", "S05 proposal success marker")
    )
    output_fingerprints = _verify_output_fingerprints(directory, marker, before)
    inputs_before = _recorded_input_snapshot(marker["input_fingerprints"])

    config, config_payload, config_hash = load_long_proposal_config(
        directory / "effective_config.json"
    )
    if (
        marker["config_hash"] != config_hash
        or config.expected_stable_track_count not in (None, expected_stable_count)
        or tuple(sorted(vars(config.artifacts).values()))
        != tuple(sorted(_CONSUMED_NAMES))
    ):
        raise ContractError("S05 proposal effective config/hash differs")
    config = replace(
        config, expected_stable_track_count=expected_stable_count
    )
    candidates = _read_parquet(
        directory / config.artifacts.candidate_edges,
        LONG_CANDIDATE_EDGES_SCHEMA,
        "S05 candidate edges",
    )
    proposals = _read_parquet(
        directory / config.artifacts.proposals,
        LONG_LINK_PROPOSALS_SCHEMA,
        "S05 link proposals",
    )
    stats = validate_s05_proposal_rows(
        candidates,
        proposals,
        long_runtime,
        stable_count=expected_stable_count,
    )
    if marker["stats"] != stats:
        raise ContractError("S05 proposal marker stats differ from Parquet rows")
    manifest = _read_json(
        directory / config.artifacts.review_manifest, "S05 review manifest"
    )
    if manifest != long_review_manifest(proposals):
        raise ContractError("S05 proposal review manifest differs from complete proposals")
    report = _read_json(directory / config.artifacts.report, "S05 proposal report")
    if (
        not isinstance(report, dict)
        or report.get("stage") != _STAGE
        or report.get("config_hash") != config_hash
        or report.get("execution_mode") != "long_proposal_only"
        or report.get("counts", {}).get("stable_paths") != expected_stable_count
        or report.get("counts", {}).get("candidates") != len(candidates)
        or report.get("counts", {}).get("provisional_review_proposals")
        != len(proposals)
        or report.get("safety_boundary", {}).get("automatic_merge_allowed") is not False
        or report.get("safety_boundary", {}).get("path_cover_used") is not False
    ):
        raise ContractError("S05 proposal report policy/counts differ")

    after = tuple(fingerprint_file(path) for path in consumed_paths)
    if before != after:
        raise ContractError("S05 proposal artifacts changed while being loaded")
    inputs_after = _recorded_input_snapshot(marker["input_fingerprints"])
    if inputs_before != inputs_after:
        raise ContractError("recorded S05 proposal inputs changed while being loaded")
    return S05ProposalBundle(
        directory=directory,
        config=config,
        config_hash=config_hash,
        effective_config=MappingProxyType(dict(config_payload)),
        candidates=tuple(MappingProxyType(dict(row)) for row in candidates),
        proposals=tuple(MappingProxyType(dict(row)) for row in proposals),
        review_manifest=MappingProxyType(dict(manifest)),
        report=MappingProxyType(dict(report)),
        success_marker=MappingProxyType(dict(marker)),
        consumed_paths=consumed_paths,
        consumed_fingerprints=before,
        input_fingerprints=inputs_before,
        output_fingerprints=output_fingerprints,
    )


__all__ = ["S05ProposalBundle", "load_s05_proposals"]
