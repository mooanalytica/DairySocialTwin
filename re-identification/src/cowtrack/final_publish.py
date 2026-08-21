"""Publish the user-facing artifacts without rewriting source detections.

The final detection CSV is deliberately built from the manifest's original
``tracking_boxes.csv`` files. S00/S06 data contributes only the five identity
columns. Consequently numeric spelling, empty strings, source row order, and
all ten source columns survive the publication step unchanged.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import time
from typing import Callable, Iterator, Sequence
import uuid

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from cowtrack.config import ClipManifest, ContractError, load_manifest
from cowtrack.schemas.detections import DETECTIONS_SCHEMA
from cowtrack.schemas.s05_finalize import DET_TO_GLOBAL_SCHEMA
from cowtrack.schemas.s06 import DETECTIONS_WITH_GLOBAL_ID_SCHEMA


FINAL_CSV_NAME = "detections_with_global_id.csv"
IDENTITY_COLUMNS = (
    "global_track_id",
    "global_track_uuid",
    "display_global_id",
    "id_status",
    "identity_basis",
)
EXPECTED_SOURCE_COLUMN_COUNT = 10
DEFAULT_EXPECTED_INVALID_COUNT = 4_796
DEFAULT_EXPECTED_GLOBAL_TRACK_COUNT = 62
EXPECTED_ID_STATUS = "forced_provisional"
EXPECTED_IDENTITY_BASIS = "operator_forced_appearance_exact_62"
PROGRESS_INTERVAL_SEC = 10.0

LogFn = Callable[[str], None]


def log(message: str) -> None:
    print(message, flush=True)


@dataclass(frozen=True)
class FinalPublishResult:
    """Validated publication result suitable for a work-side success marker."""

    deliverables_dir: Path
    detections_csv: Path
    videos_by_clip: dict[str, Path]
    row_count: int
    valid_count: int
    invalid_count: int
    resumed: bool

    def marker_payload(self) -> dict[str, object]:
        return {
            "deliverables_dir": str(self.deliverables_dir),
            "detections_csv": str(self.detections_csv),
            "videos_by_clip": {
                clip: str(path) for clip, path in self.videos_by_clip.items()
            },
            "row_count": self.row_count,
            "valid_count": self.valid_count,
            "invalid_count": self.invalid_count,
            "resumed": self.resumed,
        }


@dataclass(frozen=True)
class _IdentityRow:
    clip_id: str
    csv_row_index: int
    values: tuple[str, str, str, str, str]
    valid: bool


@dataclass(frozen=True)
class _MergeStats:
    row_count: int
    valid_count: int
    invalid_count: int


def _require_regular_file(path: Path, label: str, *, nonempty: bool = False) -> Path:
    if path.is_symlink():
        raise ContractError(f"{label} must not be a symlink: {path}")
    try:
        canonical = path.resolve(strict=True)
    except OSError as exc:
        raise ContractError(f"{label} does not exist: {path}") from exc
    if not canonical.is_file():
        raise ContractError(f"{label} must be a regular file: {path}")
    try:
        size = canonical.stat().st_size
    except OSError as exc:
        raise ContractError(f"cannot stat {label}: {canonical}") from exc
    if nonempty and size <= 0:
        raise ContractError(f"{label} is empty: {canonical}")
    return canonical


def _read_source_header(path: Path) -> list[str]:
    _require_regular_file(path, "source tracking CSV", nonempty=True)
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader)
    except StopIteration as exc:
        raise ContractError(f"source tracking CSV is empty: {path}") from exc
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ContractError(f"cannot read source tracking CSV header {path}: {exc}") from exc
    if len(header) != EXPECTED_SOURCE_COLUMN_COUNT:
        raise ContractError(
            f"source tracking CSV must have exactly {EXPECTED_SOURCE_COLUMN_COUNT} "
            f"columns, got {len(header)}: {path}"
        )
    if len(set(header)) != len(header):
        raise ContractError(f"source tracking CSV has duplicate header names: {path}")
    overlap = sorted(set(header).intersection(IDENTITY_COLUMNS))
    if overlap:
        raise ContractError(
            f"source tracking CSV already contains final identity columns {overlap}: {path}"
        )
    return header


def _validate_manifest(rows: Sequence[ClipManifest]) -> tuple[list[str], dict[str, int]]:
    if not rows:
        raise ContractError("final publication requires at least one clip")
    expected_header: list[str] | None = None
    for clip in rows:
        header = _read_source_header(clip.bbox_csv_path)
        if expected_header is None:
            expected_header = header
        elif header != expected_header:
            raise ContractError(
                f"source tracking CSV header/order differs for clip {clip.clip_id}"
            )
    assert expected_header is not None
    return expected_header, {clip.clip_id: clip.clip_order for clip in rows}


def _iter_source_rows(
    rows: Sequence[ClipManifest], expected_header: Sequence[str]
) -> Iterator[tuple[str, int, list[str]]]:
    for clip in rows:
        count = 0
        try:
            with clip.bbox_csv_path.open(
                "r", encoding="utf-8-sig", newline=""
            ) as handle:
                reader = csv.reader(handle)
                try:
                    header = next(reader)
                except StopIteration as exc:
                    raise ContractError(
                        f"source tracking CSV is empty: {clip.bbox_csv_path}"
                    ) from exc
                if header != list(expected_header):
                    raise ContractError(
                        f"source tracking CSV header changed for clip {clip.clip_id}"
                    )
                for csv_row_index, source_values in enumerate(reader):
                    if len(source_values) != len(expected_header):
                        raise ContractError(
                            "source tracking CSV column count mismatch at "
                            f"({clip.clip_id}, {csv_row_index}): "
                            f"{len(source_values)} != {len(expected_header)}"
                        )
                    count += 1
                    yield clip.clip_id, csv_row_index, source_values
        except ContractError:
            raise
        except (OSError, UnicodeError, csv.Error) as exc:
            raise ContractError(
                f"cannot read source tracking CSV {clip.bbox_csv_path}: {exc}"
            ) from exc
        if count == 0:
            raise ContractError(
                f"source tracking CSV has no data rows: {clip.bbox_csv_path}"
            )


def _parse_bool(value: str, *, row_number: int) -> bool:
    normalized = value.casefold()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ContractError(f"derived S06 valid is not boolean at data row {row_number}")


def _iter_derived_identities(
    path: Path,
    rows: Sequence[ClipManifest],
    clip_orders: dict[str, int],
) -> Iterator[_IdentityRow]:
    path = _require_regular_file(path, "derived S06 detection CSV", nonempty=True)
    expected_sequence_id = rows[0].sequence_id
    previous: tuple[int, int] | None = None
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != DETECTIONS_WITH_GLOBAL_ID_SCHEMA.names:
                raise ContractError("derived S06 detection CSV header differs")
            for row_number, row in enumerate(reader, start=1):
                if None in row or any(value is None for value in row.values()):
                    raise ContractError(
                        f"derived S06 column count differs at data row {row_number}"
                    )
                try:
                    clip_id = row["clip_id"]
                    csv_row_index = int(row["csv_row_index"])
                    clip_order = int(row["clip_order"])
                except (KeyError, TypeError, ValueError) as exc:
                    raise ContractError(
                        f"derived S06 key is invalid at data row {row_number}"
                    ) from exc
                expected_order = clip_orders.get(clip_id)
                if expected_order is None or clip_order != expected_order:
                    raise ContractError(
                        f"derived S06 clip/order differs at data row {row_number}"
                    )
                if row.get("sequence_id") != expected_sequence_id:
                    raise ContractError(
                        f"derived S06 sequence_id differs at data row {row_number}"
                    )
                if csv_row_index < 0:
                    raise ContractError(
                        f"derived S06 csv_row_index is negative at data row {row_number}"
                    )
                key = (clip_order, csv_row_index)
                if previous is not None and key <= previous:
                    raise ContractError(
                        "derived S06 (clip_id, csv_row_index) keys are duplicated "
                        "or not in manifest order"
                    )
                previous = key
                valid = _parse_bool(row.get("valid", ""), row_number=row_number)
                identity = tuple(row.get(name, "") for name in IDENTITY_COLUMNS)
                if valid:
                    if any(value == "" for value in identity):
                        raise ContractError(
                            f"derived S06 valid row lacks identity at data row {row_number}"
                        )
                    try:
                        global_id = int(identity[0])
                    except (TypeError, ValueError) as exc:
                        raise ContractError(
                            f"derived S06 global_track_id is invalid at data row {row_number}"
                        ) from exc
                    if global_id < 0:
                        raise ContractError(
                            f"derived S06 global_track_id is negative at data row {row_number}"
                        )
                elif any(value != "" for value in identity):
                    raise ContractError(
                        f"derived S06 invalid row has identity at data row {row_number}"
                    )
                yield _IdentityRow(
                    clip_id=clip_id,
                    csv_row_index=csv_row_index,
                    values=identity,  # type: ignore[arg-type]
                    valid=valid,
                )
    except ContractError:
        raise
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ContractError(f"cannot read derived S06 detection CSV {path}: {exc}") from exc


def _read_exact_parquet(path: Path, schema: pa.Schema, label: str) -> pa.Table:
    path = _require_regular_file(path, label, nonempty=True)
    try:
        table = pq.read_table(path)
    except (OSError, pa.ArrowException) as exc:
        raise ContractError(f"cannot read {label} {path}: {exc}") from exc
    if not table.schema.equals(schema, check_metadata=False):
        raise ContractError(f"{label} schema differs")
    return table


def _iter_parquet_identities(
    detections_path: Path,
    forced_mapping_path: Path,
    rows: Sequence[ClipManifest],
    clip_orders: dict[str, int],
) -> Iterator[_IdentityRow]:
    detections = _read_exact_parquet(
        detections_path, DETECTIONS_SCHEMA, "S00 detections Parquet"
    )
    mapping = _read_exact_parquet(
        forced_mapping_path, DET_TO_GLOBAL_SCHEMA, "forced detection mapping Parquet"
    )
    if mapping.num_rows == 0:
        raise ContractError("forced detection mapping Parquet is empty")
    if mapping["valid"].null_count or not all(mapping["valid"].to_pylist()):
        raise ContractError("forced detection mapping contains an invalid detection")
    order = pc.sort_indices(mapping, sort_keys=[("det_id", "ascending")])
    mapping = mapping.take(order)
    mapping_ids = mapping["det_id"].combine_chunks().to_numpy(zero_copy_only=False)
    if np.any(mapping_ids[1:] <= mapping_ids[:-1]):
        raise ContractError("forced detection mapping det_id values are duplicated")
    used = np.zeros(mapping.num_rows, dtype=np.bool_)
    expected_sequence_id = rows[0].sequence_id
    valid_count = 0
    row_number = 0
    for batch in detections.to_batches(max_chunksize=65_536):
        for record in batch.to_pylist():
            row_number += 1
            clip_id = str(record["clip_id"])
            csv_row_index = int(record["csv_row_index"])
            if record["sequence_id"] != expected_sequence_id or clip_id not in clip_orders:
                raise ContractError(
                    f"S00 detection manifest identity differs at data row {row_number}"
                )
            det_id = int(record["det_id"])
            position = int(np.searchsorted(mapping_ids, det_id))
            present = position < len(mapping_ids) and int(mapping_ids[position]) == det_id
            valid = bool(record["valid"])
            if not valid:
                if present:
                    raise ContractError("forced mapping assigns an invalid S00 detection")
                values = ("", "", "", "", "")
            else:
                valid_count += 1
                if not present:
                    raise ContractError("forced mapping lacks a valid S00 detection")
                if used[position]:
                    raise ContractError("forced mapping is ambiguous for S00 detections")
                used[position] = True
                mapped = mapping.slice(position, 1).to_pylist()[0]
                if (
                    mapped["sequence_id"] != expected_sequence_id
                    or mapped["clip_id"] != clip_id
                    or int(mapped["clip_order"]) != clip_orders[clip_id]
                    or not bool(mapped["valid"])
                ):
                    raise ContractError("forced mapping identity differs from S00 detection")
                values = (
                    str(mapped["global_track_id"]),
                    str(mapped["global_track_uuid"]),
                    str(mapped["display_global_id"]),
                    str(mapped["id_status"]),
                    str(mapped["identity_basis"]),
                )
                if any(value == "" or value == "None" for value in values):
                    raise ContractError("forced mapping contains a null identity field")
            yield _IdentityRow(clip_id, csv_row_index, values, valid)
    if valid_count != mapping.num_rows or not bool(np.all(used)):
        raise ContractError("forced mapping and valid S00 detections are not a bijection")


def _identity_iterator(
    *,
    detections_path: Path,
    forced_mapping_path: Path | None,
    rows: Sequence[ClipManifest],
    clip_orders: dict[str, int],
) -> Iterator[_IdentityRow]:
    if forced_mapping_path is None:
        if detections_path.suffix.casefold() != ".csv":
            raise ContractError(
                "detections_path must be the derived S06 CSV when forced_mapping_path is omitted"
            )
        return _iter_derived_identities(detections_path, rows, clip_orders)
    if detections_path.suffix.casefold() != ".parquet":
        raise ContractError(
            "detections_path must be S00 detections Parquet when forced_mapping_path is provided"
        )
    return _iter_parquet_identities(
        detections_path, forced_mapping_path, rows, clip_orders
    )


def _merge_rows(
    *,
    rows: Sequence[ClipManifest],
    source_header: Sequence[str],
    detections_path: Path,
    forced_mapping_path: Path | None,
    clip_orders: dict[str, int],
    consume: Callable[[list[str]], None],
    expected_invalid_count: int | None,
    expected_global_track_count: int | None,
    logger: LogFn,
    progress_label: str,
) -> _MergeStats:
    identities = _identity_iterator(
        detections_path=detections_path,
        forced_mapping_path=forced_mapping_path,
        rows=rows,
        clip_orders=clip_orders,
    )
    total = valid_count = invalid_count = 0
    seen_global_ids: set[int] = set()
    metadata_by_global_id: dict[int, tuple[str, str, str, str]] = {}
    global_id_by_uuid: dict[str, int] = {}
    last_progress = time.monotonic()
    logger(f"[final-publish] {progress_label} source/identity rows")
    for clip_id, csv_row_index, source_values in _iter_source_rows(
        rows, source_header
    ):
        try:
            identity = next(identities)
        except StopIteration as exc:
            raise ContractError(
                "identity input is missing source key "
                f"({clip_id}, {csv_row_index})"
            ) from exc
        if (identity.clip_id, identity.csv_row_index) != (clip_id, csv_row_index):
            raise ContractError(
                "source/identity (clip_id, csv_row_index) bijection failed: "
                f"expected ({clip_id}, {csv_row_index}), got "
                f"({identity.clip_id}, {identity.csv_row_index})"
            )
        if identity.valid:
            valid_count += 1
            global_id = int(identity.values[0])
            global_uuid, display_id, id_status, identity_basis = identity.values[1:]
            if display_id != f"G{global_id + 1:04d}":
                raise ContractError(
                    f"final display_global_id differs for global_track_id={global_id}"
                )
            if id_status != EXPECTED_ID_STATUS:
                raise ContractError(
                    f"final id_status differs for global_track_id={global_id}"
                )
            if identity_basis != EXPECTED_IDENTITY_BASIS:
                raise ContractError(
                    f"final identity_basis differs for global_track_id={global_id}"
                )
            metadata = (global_uuid, display_id, id_status, identity_basis)
            previous_metadata = metadata_by_global_id.setdefault(global_id, metadata)
            if previous_metadata != metadata:
                raise ContractError(
                    f"final identity metadata is inconsistent for global_track_id={global_id}"
                )
            previous_global_id = global_id_by_uuid.setdefault(global_uuid, global_id)
            if previous_global_id != global_id:
                raise ContractError("final global_track_uuid is shared by multiple IDs")
            seen_global_ids.add(global_id)
        else:
            invalid_count += 1
        consume(source_values + list(identity.values))
        total += 1
        now = time.monotonic()
        if now - last_progress >= PROGRESS_INTERVAL_SEC:
            logger(
                f"[final-publish] {progress_label}: {total:,} rows; "
                f"valid={valid_count:,}; invalid={invalid_count:,}"
            )
            last_progress = now
    try:
        extra = next(identities)
    except StopIteration:
        extra = None
    if extra is not None:
        raise ContractError(
            "identity input has an extra key not present in source CSVs: "
            f"({extra.clip_id}, {extra.csv_row_index})"
        )
    if expected_invalid_count is not None and invalid_count != expected_invalid_count:
        raise ContractError(
            f"final invalid row count differs: {invalid_count} != {expected_invalid_count}"
        )
    if expected_global_track_count is not None and seen_global_ids != set(
        range(expected_global_track_count)
    ):
        raise ContractError(
            "final global_track_id coverage differs from contiguous range "
            f"0..{expected_global_track_count - 1}"
        )
    logger(
        f"[final-publish] {progress_label} complete: {total:,} rows; "
        f"valid={valid_count:,}; invalid={invalid_count:,}"
    )
    return _MergeStats(total, valid_count, invalid_count)


def _write_csv_atomic(
    path: Path,
    *,
    rows: Sequence[ClipManifest],
    source_header: Sequence[str],
    detections_path: Path,
    forced_mapping_path: Path | None,
    clip_orders: dict[str, int],
    expected_invalid_count: int | None,
    expected_global_track_count: int | None,
    logger: LogFn,
) -> _MergeStats:
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    try:
        with temporary.open("x", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow([*source_header, *IDENTITY_COLUMNS])
            stats = _merge_rows(
                rows=rows,
                source_header=source_header,
                detections_path=detections_path,
                forced_mapping_path=forced_mapping_path,
                clip_orders=clip_orders,
                consume=writer.writerow,
                expected_invalid_count=expected_invalid_count,
                expected_global_track_count=expected_global_track_count,
                logger=logger,
                progress_label="writing final CSV",
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        return stats
    except ContractError:
        raise
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ContractError(f"cannot atomically write final detection CSV {path}: {exc}") from exc
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def _validate_final_csv(
    path: Path,
    *,
    rows: Sequence[ClipManifest],
    source_header: Sequence[str],
    detections_path: Path,
    forced_mapping_path: Path | None,
    clip_orders: dict[str, int],
    expected_invalid_count: int | None,
    expected_global_track_count: int | None,
    logger: LogFn,
) -> _MergeStats:
    path = _require_regular_file(path, "final detection CSV", nonempty=True)
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            try:
                header = next(reader)
            except StopIteration as exc:
                raise ContractError("final detection CSV is empty") from exc
            expected_header = [*source_header, *IDENTITY_COLUMNS]
            if header != expected_header:
                raise ContractError("final detection CSV header/order differs")
            output_row_number = 0

            def compare(expected: list[str]) -> None:
                nonlocal output_row_number
                try:
                    observed = next(reader)
                except StopIteration as exc:
                    raise ContractError(
                        f"final detection CSV ends before data row {output_row_number}"
                    ) from exc
                if observed != expected:
                    raise ContractError(
                        "final detection CSV does not exactly preserve source/identity "
                        f"values at data row {output_row_number}"
                    )
                output_row_number += 1

            stats = _merge_rows(
                rows=rows,
                source_header=source_header,
                detections_path=detections_path,
                forced_mapping_path=forced_mapping_path,
                clip_orders=clip_orders,
                consume=compare,
                expected_invalid_count=expected_invalid_count,
                expected_global_track_count=expected_global_track_count,
                logger=logger,
                progress_label="validating final CSV",
            )
            try:
                extra = next(reader)
            except StopIteration:
                extra = None
            if extra is not None:
                raise ContractError("final detection CSV has extra data rows")
            return stats
    except ContractError:
        raise
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ContractError(f"cannot validate final detection CSV {path}: {exc}") from exc


def _video_sources(
    s06_output_dir: Path, rows: Sequence[ClipManifest]
) -> dict[str, Path]:
    if s06_output_dir.is_symlink() or not s06_output_dir.is_dir():
        raise ContractError(f"S06 output must be a real directory: {s06_output_dir}")
    result: dict[str, Path] = {}
    for clip in rows:
        name = f"{clip.clip_id}_tracked.mp4"
        result[clip.clip_id] = _require_regular_file(
            s06_output_dir / "qa" / "videos" / name,
            f"S06 full overlay for {clip.clip_id}",
            nonempty=True,
        )
    return result


def _create_video_hardlinks(directory: Path, videos: dict[str, Path]) -> None:
    for source in videos.values():
        destination = directory / source.name
        try:
            os.link(source, destination, follow_symlinks=False)
        except OSError as exc:
            raise ContractError(
                "cannot create the required same-filesystem final video hard link "
                f"{destination}: {exc}"
            ) from exc


def _validate_video_hardlinks(directory: Path, videos: dict[str, Path]) -> None:
    for source in videos.values():
        published = directory / source.name
        if published.is_symlink():
            raise ContractError(f"final video must be a regular file, not a symlink: {published}")
        try:
            canonical = published.resolve(strict=True)
            same_file = os.path.samefile(canonical, source)
        except OSError as exc:
            raise ContractError(f"cannot validate final video hard link {published}: {exc}") from exc
        if not canonical.is_file() or canonical.stat().st_size <= 0 or not same_file:
            raise ContractError(
                f"final video is not the validated S06 video hard link: {published}"
            )


def _fsync_directory(directory: Path) -> None:
    try:
        descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    except OSError as exc:
        raise ContractError(f"cannot open directory for durable publication: {directory}") from exc
    try:
        os.fsync(descriptor)
    except OSError as exc:
        raise ContractError(f"cannot fsync publication directory: {directory}") from exc
    finally:
        os.close(descriptor)


def _expected_names(videos: dict[str, Path]) -> set[str]:
    return {FINAL_CSV_NAME, *(source.name for source in videos.values())}


def _validate_directory_entries(directory: Path, videos: dict[str, Path]) -> None:
    if directory.is_symlink() or not directory.is_dir():
        raise ContractError(f"final deliverables path must be a real directory: {directory}")
    try:
        names = {entry.name for entry in directory.iterdir()}
    except OSError as exc:
        raise ContractError(f"cannot enumerate final deliverables directory: {exc}") from exc
    expected = _expected_names(videos)
    if names != expected:
        raise ContractError(
            f"final deliverables must contain exactly {sorted(expected)}, got {sorted(names)}"
        )


def publish_final_deliverables(
    manifest_path: Path,
    s06_output_dir: Path,
    deliverables_dir: Path,
    *,
    detections_path: Path | None = None,
    forced_mapping_path: Path | None = None,
    expected_invalid_count: int | None = DEFAULT_EXPECTED_INVALID_COUNT,
    expected_global_track_count: int | None = DEFAULT_EXPECTED_GLOBAL_TRACK_COUNT,
    logger: LogFn = log,
) -> FinalPublishResult:
    """Create or strictly revalidate the CSV plus one video hard link per clip.

    With the default arguments identities are read from S06's derived CSV.
    Supplying both ``detections_path`` (S00 Parquet) and
    ``forced_mapping_path`` uses the equivalent upstream identity join.
    ``deliverables_dir`` never receives a success marker.
    """

    if not callable(logger):
        raise ContractError("final publisher logger must be callable")
    manifest_path = Path(manifest_path).resolve()
    raw_s06_output = Path(s06_output_dir)
    if raw_s06_output.is_symlink():
        raise ContractError(f"S06 output must not be a symlink: {raw_s06_output}")
    s06_output_dir = raw_s06_output.resolve()
    raw_deliverables = Path(deliverables_dir)
    if raw_deliverables.is_symlink():
        raise ContractError(f"final deliverables path must not be a symlink: {raw_deliverables}")
    deliverables_dir = raw_deliverables.resolve()
    rows, _ = load_manifest(manifest_path)
    source_header, clip_orders = _validate_manifest(rows)
    videos = _video_sources(s06_output_dir, rows)
    if detections_path is None:
        detections_path = s06_output_dir / FINAL_CSV_NAME
    detections_path = Path(detections_path)
    if forced_mapping_path is not None:
        forced_mapping_path = Path(forced_mapping_path)
    if expected_invalid_count is not None and expected_invalid_count < 0:
        raise ContractError("expected_invalid_count must be non-negative or None")
    if expected_global_track_count is not None and expected_global_track_count <= 0:
        raise ContractError("expected_global_track_count must be positive or None")
    if (
        deliverables_dir == s06_output_dir
        or deliverables_dir in s06_output_dir.parents
        or s06_output_dir in deliverables_dir.parents
    ):
        raise ContractError("final deliverables directory must not overlap S06 output")

    final_csv = deliverables_dir / FINAL_CSV_NAME
    final_videos = {
        clip.clip_id: deliverables_dir / f"{clip.clip_id}_tracked.mp4" for clip in rows
    }
    if deliverables_dir.exists():
        _validate_directory_entries(deliverables_dir, videos)
        _validate_video_hardlinks(deliverables_dir, videos)
        stats = _validate_final_csv(
            final_csv,
            rows=rows,
            source_header=source_header,
            detections_path=detections_path,
            forced_mapping_path=forced_mapping_path,
            clip_orders=clip_orders,
            expected_invalid_count=expected_invalid_count,
            expected_global_track_count=expected_global_track_count,
            logger=logger,
        )
        logger(f"[final-publish] existing deliverables fully revalidated: {deliverables_dir}")
        return FinalPublishResult(
            deliverables_dir,
            final_csv,
            final_videos,
            stats.row_count,
            stats.valid_count,
            stats.invalid_count,
            True,
        )

    try:
        deliverables_dir.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ContractError(f"cannot create final deliverables parent: {exc}") from exc
    stale_staging = sorted(
        deliverables_dir.parent.glob(f".{deliverables_dir.name}.staging-*")
    )
    if stale_staging:
        raise ContractError(
            f"stale final publication staging directories require inspection: {stale_staging}"
        )
    staging = deliverables_dir.with_name(
        f".{deliverables_dir.name}.staging-{uuid.uuid4().hex}"
    )
    try:
        staging.mkdir()
        _write_csv_atomic(
            staging / FINAL_CSV_NAME,
            rows=rows,
            source_header=source_header,
            detections_path=detections_path,
            forced_mapping_path=forced_mapping_path,
            clip_orders=clip_orders,
            expected_invalid_count=expected_invalid_count,
            expected_global_track_count=expected_global_track_count,
            logger=logger,
        )
        logger(
            f"[final-publish] creating {len(videos)} same-filesystem video hard links"
        )
        _create_video_hardlinks(staging, videos)
        _validate_directory_entries(staging, videos)
        _validate_video_hardlinks(staging, videos)
        stats = _validate_final_csv(
            staging / FINAL_CSV_NAME,
            rows=rows,
            source_header=source_header,
            detections_path=detections_path,
            forced_mapping_path=forced_mapping_path,
            clip_orders=clip_orders,
            expected_invalid_count=expected_invalid_count,
            expected_global_track_count=expected_global_track_count,
            logger=logger,
        )
        _fsync_directory(staging)
        os.replace(staging, deliverables_dir)
        _fsync_directory(deliverables_dir.parent)
        logger(f"[final-publish] atomically published: {deliverables_dir}")
    except ContractError:
        raise
    except OSError as exc:
        raise ContractError(f"cannot atomically publish final deliverables: {exc}") from exc
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
    return FinalPublishResult(
        deliverables_dir,
        final_csv,
        final_videos,
        stats.row_count,
        stats.valid_count,
        stats.invalid_count,
        False,
    )


__all__ = [
    "DEFAULT_EXPECTED_GLOBAL_TRACK_COUNT",
    "DEFAULT_EXPECTED_INVALID_COUNT",
    "FINAL_CSV_NAME",
    "IDENTITY_COLUMNS",
    "FinalPublishResult",
    "publish_final_deliverables",
]
