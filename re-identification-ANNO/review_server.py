#!/usr/bin/env python3
"""Strict, standard-library review server for occurrence-level QC."""

from __future__ import annotations

import argparse
import csv
import json
import mimetypes
import os
import re
import tempfile
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping
from urllib.parse import unquote, urlsplit

from review_lock import ReviewFileLock, ReviewLockUnavailable


BASE_DIR = Path(__file__).resolve().parent
OCCURRENCES_PATH = BASE_DIR / "occurrence_segments.csv"
MANIFEST_PATH = BASE_DIR / "cached_clips" / "manifest.json"
REVIEWS_PATH = BASE_DIR / "occurrence_reviews.csv"
WEB_ROOT = BASE_DIR / "web"
SCHEMA_VERSION = "1.0"
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 9922
MAX_REQUEST_BYTES = 16 * 1024
EXPECTED_OCCURRENCE_COUNT = 2_414
REVIEW_CLIP_FRAME_COUNT = 900
REVIEW_FPS_NUMERATOR = 30_000
REVIEW_FPS_DENOMINATOR = 1_001
PLAYBACK_PREROLL_SECONDS = 3.0

SOURCE_FIELDS = (
    "occurrence_id", "clip_order", "clip_id", "display_global_id",
    "legacy_track_id", "start_frame", "end_frame", "start_time_sec",
    "end_time_sec_inclusive", "end_time_sec_exclusive", "span_duration_sec",
    "num_valid_detections", "num_missing_frames",
    "max_internal_gap_missing_frames", "start_det_id", "end_det_id",
)
REVIEW_FIELDS = (
    "review_action", "reviewed_global_id", "invalid_multiple_cows",
    "reviewed_at_utc",
)
REVIEW_ACTIONS = frozenset({"accept", "invalid_multiple_cows", "update_id"})
CLIP_FRAME_COUNTS = {"GX040006": 84_480, "GX050006": 78_720}
CLIP_ORDERS = {"GX040006": 0, "GX050006": 1}
MANIFEST_BASE_FIELDS = frozenset({
    "occurrence_id", "relative_path", "status", "clip_id", "anchor_frame",
    "window_start_frame", "window_end_frame", "frame_count",
    "anchor_offset_frame", "red_box", "display_global_id", "legacy_track_id",
})
MANIFEST_COMPLETE_FIELDS = MANIFEST_BASE_FIELDS | {"size_bytes", "sha256"}
MANIFEST_STATUSES = frozenset({"pending", "complete"})
_GID_RE = re.compile(r"G[0-9]{4}\Z")
_OCCURRENCE_RE = re.compile(r"O[0-9]{6}\Z")
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_REVIEW_ROUTE_RE = re.compile(r"/api/reviews/(O[0-9]{6})\Z")
_CLIP_ROUTE_RE = re.compile(r"/clips/(O[0-9]{6})\.mp4\Z")


class ContractError(RuntimeError):
    """A persisted input or request violates the fixed review contract."""


class ReviewLockedError(ContractError):
    """The corrected exporter currently owns the review file."""


class RangeNotSatisfiable(ValueError):
    pass


def _integer(value: Any, label: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool):
        raise ContractError(f"{label} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"{label} must be an integer") from exc
    if isinstance(value, float) or (isinstance(value, str) and str(result) != value):
        raise ContractError(f"{label} must be a canonical integer")
    if minimum is not None and result < minimum:
        raise ContractError(f"{label} must be >= {minimum}")
    return result


def _strict_json(path: Path) -> Any:
    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ContractError(f"duplicate JSON key {key!r} in {path}")
            result[key] = value
        return result

    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(
                handle,
                object_pairs_hook=object_pairs,
                parse_constant=lambda value: (_ for _ in ()).throw(
                    ContractError(f"non-finite JSON number {value!r} in {path}")
                ),
            )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read valid JSON {path}: {exc}") from exc


def _validate_gid(value: str, label: str) -> None:
    if (
        not isinstance(value, str)
        or _GID_RE.fullmatch(value) is None
        or not 1 <= int(value[1:]) <= 62
    ):
        raise ContractError(f"{label} must be G0001-G0062")


@dataclass(frozen=True)
class OccurrenceCatalog:
    rows: tuple[dict[str, str], ...]
    by_id: Mapping[str, dict[str, str]]

    @classmethod
    def load(
        cls,
        path: Path,
        *,
        expected_count: int | None = EXPECTED_OCCURRENCE_COUNT,
    ) -> "OccurrenceCatalog":
        try:
            handle = path.open("r", encoding="utf-8", newline="")
        except OSError as exc:
            raise ContractError(f"cannot open occurrence CSV {path}: {exc}") from exc
        rows: list[dict[str, str]] = []
        by_id: dict[str, dict[str, str]] = {}
        with handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != SOURCE_FIELDS:
                raise ContractError("occurrence CSV header differs from the fixed schema")
            for index, source in enumerate(reader, 1):
                if set(source) != set(SOURCE_FIELDS) or any(v is None for v in source.values()):
                    raise ContractError(f"malformed occurrence CSV row {index}")
                row = {field: source[field] for field in SOURCE_FIELDS}
                occurrence_id = row["occurrence_id"]
                if occurrence_id != f"O{index:06d}" or _OCCURRENCE_RE.fullmatch(occurrence_id) is None:
                    raise ContractError(f"non-sequential occurrence_id at row {index}")
                clip_id = row["clip_id"]
                if clip_id not in CLIP_FRAME_COUNTS:
                    raise ContractError(f"unknown clip_id at {occurrence_id}")
                if _integer(row["clip_order"], f"{occurrence_id}.clip_order", minimum=0) != CLIP_ORDERS[clip_id]:
                    raise ContractError(f"clip_order mismatch at {occurrence_id}")
                _validate_gid(row["display_global_id"], f"{occurrence_id}.display_global_id")
                start = _integer(row["start_frame"], f"{occurrence_id}.start_frame", minimum=0)
                end = _integer(row["end_frame"], f"{occurrence_id}.end_frame", minimum=start)
                if end >= CLIP_FRAME_COUNTS[clip_id]:
                    raise ContractError(f"frame interval outside clip at {occurrence_id}")
                detections = _integer(row["num_valid_detections"], f"{occurrence_id}.num_valid_detections", minimum=1)
                missing = _integer(row["num_missing_frames"], f"{occurrence_id}.num_missing_frames", minimum=0)
                gap = _integer(row["max_internal_gap_missing_frames"], f"{occurrence_id}.max_internal_gap_missing_frames", minimum=0)
                if detections + missing != end - start + 1 or gap > 30:
                    raise ContractError(f"span accounting mismatch at {occurrence_id}")
                for field in ("legacy_track_id", "start_det_id", "end_det_id"):
                    if not row[field]:
                        raise ContractError(f"empty {field} at {occurrence_id}")
                for field in ("start_time_sec", "end_time_sec_inclusive", "end_time_sec_exclusive", "span_duration_sec"):
                    try:
                        value = Decimal(row[field])
                    except InvalidOperation as exc:
                        raise ContractError(f"invalid {field} at {occurrence_id}") from exc
                    if not value.is_finite() or value < 0:
                        raise ContractError(f"invalid {field} at {occurrence_id}")
                rows.append(row)
                by_id[occurrence_id] = row
        if not rows:
            raise ContractError("occurrence CSV contains no rows")
        if expected_count is not None and len(rows) != expected_count:
            raise ContractError(
                f"expected {expected_count:,} occurrences, found {len(rows):,}"
            )
        return cls(tuple(rows), by_id)


@dataclass(frozen=True)
class ClipEntry:
    occurrence_id: str
    relative_path: str
    status: str
    path: Path
    anchor_offset_frame: int

    @property
    def ready(self) -> bool:
        return self.status == "complete"


@dataclass(frozen=True)
class ClipManifest:
    entries: Mapping[str, ClipEntry]

    @classmethod
    def load(cls, path: Path, catalog: OccurrenceCatalog) -> "ClipManifest":
        payload = _strict_json(path)
        if not isinstance(payload, dict) or set(payload) != {"schema_version", "clips"}:
            raise ContractError("clip manifest must contain only schema_version and clips")
        if payload["schema_version"] != SCHEMA_VERSION or not isinstance(payload["clips"], list):
            raise ContractError("clip manifest schema_version/clips mismatch")
        if len(payload["clips"]) != len(catalog.rows):
            raise ContractError("clip manifest occurrence count mismatch")
        root = path.parent.resolve()
        entries: dict[str, ClipEntry] = {}
        used_paths: set[str] = set()
        for index, (item, occurrence) in enumerate(zip(payload["clips"], catalog.rows, strict=True)):
            if not isinstance(item, dict):
                raise ContractError(f"clip manifest item {index} must be an object")
            status = item.get("status")
            expected_fields = MANIFEST_COMPLETE_FIELDS if status == "complete" else MANIFEST_BASE_FIELDS
            if set(item) != expected_fields or status not in MANIFEST_STATUSES:
                raise ContractError(f"clip manifest fields/status mismatch at item {index}")
            occurrence_id = item["occurrence_id"]
            if occurrence_id != occurrence["occurrence_id"]:
                raise ContractError(f"clip manifest order/identity mismatch at item {index}")
            if item["clip_id"] != occurrence["clip_id"]:
                raise ContractError(f"clip_id mismatch for {occurrence_id}")
            if item["display_global_id"] != occurrence["display_global_id"]:
                raise ContractError(f"display_global_id mismatch for {occurrence_id}")
            if item["legacy_track_id"] != occurrence["legacy_track_id"]:
                raise ContractError(f"legacy_track_id mismatch for {occurrence_id}")
            anchor = _integer(item["anchor_frame"], f"{occurrence_id}.anchor_frame", minimum=0)
            window_start = _integer(item["window_start_frame"], f"{occurrence_id}.window_start_frame", minimum=0)
            window_end = _integer(item["window_end_frame"], f"{occurrence_id}.window_end_frame", minimum=window_start)
            frame_count = _integer(item["frame_count"], f"{occurrence_id}.frame_count", minimum=1)
            offset = _integer(item["anchor_offset_frame"], f"{occurrence_id}.anchor_offset_frame", minimum=0)
            occurrence_start = int(occurrence["start_frame"])
            occurrence_end = int(occurrence["end_frame"])
            if not occurrence_start <= anchor <= occurrence_end:
                raise ContractError(
                    f"anchor_frame outside occurrence interval for {occurrence_id}"
                )
            if window_end >= CLIP_FRAME_COUNTS[occurrence["clip_id"]] or not window_start <= anchor <= window_end:
                raise ContractError(f"invalid clip window for {occurrence_id}")
            expected_start = max(
                0,
                min(
                    anchor - REVIEW_CLIP_FRAME_COUNT // 2,
                    CLIP_FRAME_COUNTS[occurrence["clip_id"]]
                    - REVIEW_CLIP_FRAME_COUNT,
                ),
            )
            if (
                frame_count != REVIEW_CLIP_FRAME_COUNT
                or window_start != expected_start
                or window_end != expected_start + REVIEW_CLIP_FRAME_COUNT - 1
                or offset != anchor - window_start
            ):
                raise ContractError(f"clip frame accounting mismatch for {occurrence_id}")
            red_box = item["red_box"]
            if not isinstance(red_box, dict) or set(red_box) != {"x", "y", "width", "height"}:
                raise ContractError(f"invalid red_box for {occurrence_id}")
            x = _integer(red_box["x"], f"{occurrence_id}.red_box.x", minimum=0)
            y = _integer(red_box["y"], f"{occurrence_id}.red_box.y", minimum=0)
            width = _integer(red_box["width"], f"{occurrence_id}.red_box.width", minimum=1)
            height = _integer(red_box["height"], f"{occurrence_id}.red_box.height", minimum=1)
            if x + width > 1920 or y + height > 1080:
                raise ContractError(f"red_box outside output frame for {occurrence_id}")
            relative = item["relative_path"]
            if not isinstance(relative, str) or "\\" in relative:
                raise ContractError(f"invalid relative_path for {occurrence_id}")
            pure = PurePosixPath(relative)
            if pure.is_absolute() or str(pure) != relative or any(part in {"", ".", ".."} for part in pure.parts) or pure.suffix != ".mp4":
                raise ContractError(f"unsafe relative_path for {occurrence_id}")
            if relative in used_paths:
                raise ContractError(f"duplicate relative_path {relative!r}")
            if relative != f"{occurrence_id}.mp4":
                raise ContractError(f"noncanonical relative_path for {occurrence_id}")
            candidate = (root / Path(*pure.parts)).resolve()
            try:
                candidate.relative_to(root)
            except ValueError as exc:
                raise ContractError(f"relative_path escapes cached_clips for {occurrence_id}") from exc
            if status == "complete":
                size = _integer(item["size_bytes"], f"{occurrence_id}.size_bytes", minimum=1)
                if not isinstance(item["sha256"], str) or _SHA256_RE.fullmatch(item["sha256"]) is None:
                    raise ContractError(f"invalid sha256 for {occurrence_id}")
                try:
                    actual_size = candidate.stat().st_size
                except OSError as exc:
                    raise ContractError(f"missing completed clip for {occurrence_id}: {exc}") from exc
                if not candidate.is_file() or actual_size != size:
                    raise ContractError(f"completed clip size mismatch for {occurrence_id}")
            entries[occurrence_id] = ClipEntry(
                occurrence_id,
                relative,
                status,
                candidate,
                offset,
            )
            used_paths.add(relative)
        return cls(entries)


def playback_start_seconds(anchor_offset_frame: int) -> float:
    anchor_seconds = (
        anchor_offset_frame * REVIEW_FPS_DENOMINATOR / REVIEW_FPS_NUMERATOR
    )
    return max(0.0, anchor_seconds - PLAYBACK_PREROLL_SECONDS)


def normalize_new_id(value: Any) -> str:
    if isinstance(value, bool):
        raise ContractError("new_id must be a number from 1 to 62")
    if isinstance(value, int):
        number = value
    elif isinstance(value, str) and re.fullmatch(r"[0-9]+", value.strip()):
        number = int(value.strip())
    else:
        raise ContractError("new_id must be a numeric string or integer")
    if not 1 <= number <= 62:
        raise ContractError("new_id must be in the range 1-62")
    return f"G{number:04d}"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class ReviewStore:
    def __init__(self, path: Path, catalog: OccurrenceCatalog, *, clock: Callable[[], datetime] = _utc_now):
        self.path = path
        self.file_lock_path = path.with_name(f".{path.name}.lock")
        self.catalog = catalog
        self.clock = clock
        self.lock = threading.RLock()
        self._reviews: dict[str, dict[str, str]] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            handle = self.path.open("r", encoding="utf-8", newline="")
        except OSError as exc:
            raise ContractError(f"cannot open review CSV {self.path}: {exc}") from exc
        previous_index = 0
        with handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != SOURCE_FIELDS + REVIEW_FIELDS:
                raise ContractError("review CSV header differs from the fixed schema")
            for source in reader:
                if set(source) != set(SOURCE_FIELDS + REVIEW_FIELDS) or any(v is None for v in source.values()):
                    raise ContractError("malformed review CSV row")
                occurrence_id = source["occurrence_id"]
                original = self.catalog.by_id.get(occurrence_id)
                if original is None or any(source[field] != original[field] for field in SOURCE_FIELDS):
                    raise ContractError(f"review source identity mismatch for {occurrence_id}")
                current_index = int(occurrence_id[1:])
                if occurrence_id in self._reviews or current_index <= previous_index:
                    raise ContractError("duplicate or unordered review occurrence")
                review = {field: source[field] for field in REVIEW_FIELDS}
                self._validate_persisted(review, original)
                self._reviews[occurrence_id] = review
                previous_index = current_index

    @staticmethod
    def _validate_persisted(review: Mapping[str, str], original: Mapping[str, str]) -> None:
        action = review["review_action"]
        if action not in REVIEW_ACTIONS:
            raise ContractError("invalid persisted review action")
        _validate_gid(review["reviewed_global_id"], "reviewed_global_id")
        expected_invalid = "true" if action == "invalid_multiple_cows" else "false"
        if review["invalid_multiple_cows"] != expected_invalid:
            raise ContractError("persisted invalid_multiple_cows flag mismatch")
        if action != "update_id" and review["reviewed_global_id"] != original["display_global_id"]:
            raise ContractError("persisted retained identity mismatch")
        stamp = review["reviewed_at_utc"]
        if not stamp.endswith("Z"):
            raise ContractError("reviewed_at_utc must be UTC with Z suffix")
        try:
            parsed = datetime.fromisoformat(stamp[:-1] + "+00:00")
        except ValueError as exc:
            raise ContractError("invalid reviewed_at_utc") from exc
        if parsed.utcoffset() != timezone.utc.utcoffset(parsed):
            raise ContractError("reviewed_at_utc is not UTC")

    def snapshot(self) -> dict[str, dict[str, str]]:
        with self.lock:
            return {key: dict(value) for key, value in self._reviews.items()}

    def apply(self, occurrence_id: str, action: str, *, new_id: Any = None, new_id_present: bool = False) -> dict[str, str]:
        if occurrence_id not in self.catalog.by_id:
            raise KeyError(occurrence_id)
        if action not in REVIEW_ACTIONS:
            raise ContractError("action must be accept, invalid_multiple_cows, or update_id")
        original = self.catalog.by_id[occurrence_id]
        if action == "update_id":
            if not new_id_present:
                raise ContractError("update_id requires new_id")
            reviewed_id = normalize_new_id(new_id)
        else:
            if new_id_present:
                raise ContractError(f"{action} must not include new_id")
            reviewed_id = original["display_global_id"]
        now = self.clock()
        if not isinstance(now, datetime) or now.tzinfo is None:
            raise ContractError("review clock must return a timezone-aware datetime")
        stamp = now.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
        review = {
            "review_action": action,
            "reviewed_global_id": reviewed_id,
            "invalid_multiple_cows": "true" if action == "invalid_multiple_cows" else "false",
            "reviewed_at_utc": stamp,
        }
        with self.lock:
            try:
                file_lock = ReviewFileLock(self.file_lock_path, blocking=False)
                with file_lock:
                    previous = self._reviews.get(occurrence_id)
                    self._reviews[occurrence_id] = review
                    try:
                        self._write_locked()
                    except Exception:
                        if previous is None:
                            self._reviews.pop(occurrence_id, None)
                        else:
                            self._reviews[occurrence_id] = previous
                        raise
            except ReviewLockUnavailable as exc:
                raise ReviewLockedError(
                    "reviews are frozen while the corrected export is running"
                ) from exc
        return dict(review)

    def _write_locked(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                "w", encoding="utf-8", newline="", dir=self.path.parent,
                prefix=f".{self.path.name}.", suffix=".tmp", delete=False,
            ) as handle:
                temporary = handle.name
                writer = csv.DictWriter(handle, fieldnames=SOURCE_FIELDS + REVIEW_FIELDS, lineterminator="\n")
                writer.writeheader()
                for occurrence in self.catalog.rows:
                    review = self._reviews.get(occurrence["occurrence_id"])
                    if review is not None:
                        writer.writerow({**occurrence, **review})
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
            temporary = None
        finally:
            if temporary is not None:
                Path(temporary).unlink(missing_ok=True)


class ReviewApplication:
    def __init__(self, catalog: OccurrenceCatalog, manifest: ClipManifest, store: ReviewStore, web_root: Path):
        self.catalog = catalog
        self.manifest = manifest
        self.store = store
        self.web_root = web_root.resolve()
        if not (self.web_root / "index.html").is_file():
            raise ContractError(f"missing web/index.html in {self.web_root}")

    def summary(self, reviews: Mapping[str, Any] | None = None) -> dict[str, Any]:
        current = self.store.snapshot() if reviews is None else reviews
        actions = {action: 0 for action in sorted(REVIEW_ACTIONS)}
        for review in current.values():
            actions[review["review_action"]] += 1
        ready = sum(entry.ready for entry in self.manifest.entries.values())
        return {
            "total_occurrences": len(self.catalog.rows),
            "reviewed_occurrences": len(current),
            "unreviewed_occurrences": len(self.catalog.rows) - len(current),
            "clips_ready": ready,
            "clips_not_ready": len(self.catalog.rows) - ready,
            "actions": actions,
        }

    def state(self) -> dict[str, Any]:
        reviews = self.store.snapshot()
        occurrences: list[dict[str, Any]] = []
        for source in self.catalog.rows:
            occurrence_id = source["occurrence_id"]
            entry = self.manifest.entries[occurrence_id]
            row: dict[str, Any] = dict(source)
            row.update({
                "clip_url": f"/clips/{occurrence_id}.mp4",
                "clip_ready": entry.ready,
                "playback_start_sec": playback_start_seconds(
                    entry.anchor_offset_frame
                ),
                "review": None if occurrence_id not in reviews else {
                    "action": reviews[occurrence_id]["review_action"],
                    "reviewed_global_id": reviews[occurrence_id]["reviewed_global_id"],
                    "invalid_multiple_cows": reviews[occurrence_id]["invalid_multiple_cows"] == "true",
                    "reviewed_at_utc": reviews[occurrence_id]["reviewed_at_utc"],
                },
            })
            occurrences.append(row)
        return {"schema_version": SCHEMA_VERSION, "summary": self.summary(reviews), "occurrences": occurrences}

    def submit(self, occurrence_id: str, body: Mapping[str, Any]) -> dict[str, Any]:
        entry = self.manifest.entries.get(occurrence_id)
        if entry is None:
            raise KeyError(occurrence_id)
        if not entry.ready:
            raise ContractError("cached clip is not complete for this occurrence")
        if not isinstance(body, dict) or set(body) - {"action", "new_id"} or "action" not in body:
            raise ContractError("body must contain action and optional new_id only")
        action = body["action"]
        if not isinstance(action, str):
            raise ContractError("action must be a string")
        review = self.store.apply(occurrence_id, action, new_id=body.get("new_id"), new_id_present="new_id" in body)
        return {
            "schema_version": SCHEMA_VERSION,
            "occurrence_id": occurrence_id,
            "review": {
                "action": review["review_action"],
                "reviewed_global_id": review["reviewed_global_id"],
                "invalid_multiple_cows": review["invalid_multiple_cows"] == "true",
                "reviewed_at_utc": review["reviewed_at_utc"],
            },
            "summary": self.summary(),
        }


def parse_single_range(value: str | None, size: int) -> tuple[int, int] | None:
    if value is None:
        return None
    if size <= 0 or not value.startswith("bytes=") or "," in value:
        raise RangeNotSatisfiable
    spec = value[6:].strip()
    match = re.fullmatch(r"([0-9]*)-([0-9]*)", spec)
    if match is None or not any(match.groups()):
        raise RangeNotSatisfiable
    first, last = match.groups()
    if first:
        start = int(first)
        if start >= size:
            raise RangeNotSatisfiable
        end = size - 1 if not last else min(int(last), size - 1)
        if end < start:
            raise RangeNotSatisfiable
        return start, end
    suffix = int(last)
    if suffix <= 0:
        raise RangeNotSatisfiable
    return max(0, size - suffix), size - 1


class ReviewHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], application: ReviewApplication):
        self.application = application
        super().__init__(address, ReviewRequestHandler)


class ReviewRequestHandler(BaseHTTPRequestHandler):
    server: ReviewHTTPServer
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[http] {self.address_string()} {fmt % args}", flush=True)

    def do_GET(self) -> None:
        self._route(send_body=True)

    def do_HEAD(self) -> None:
        self._route(send_body=False)

    def do_POST(self) -> None:
        parsed = urlsplit(self.path)
        match = _REVIEW_ROUTE_RE.fullmatch(parsed.path)
        if match is None or parsed.query:
            self._json_error(HTTPStatus.NOT_FOUND, "not_found", "unknown endpoint")
            return
        content_type = self.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
        if content_type != "application/json":
            self._json_error(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, "content_type", "Content-Type must be application/json")
            return
        if self.headers.get("Transfer-Encoding"):
            self._json_error(HTTPStatus.BAD_REQUEST, "transfer_encoding", "chunked request bodies are not supported")
            return
        try:
            length = int(self.headers.get("Content-Length", ""))
        except ValueError:
            length = -1
        if not 1 <= length <= MAX_REQUEST_BYTES:
            self._json_error(HTTPStatus.BAD_REQUEST, "content_length", "invalid Content-Length")
            return
        try:
            raw = self.rfile.read(length)
            body = json.loads(raw.decode("utf-8"))
            result = self.server.application.submit(match.group(1), body)
        except KeyError:
            self._json_error(HTTPStatus.NOT_FOUND, "unknown_occurrence", "unknown occurrence_id")
            return
        except ReviewLockedError as exc:
            self._json_error(HTTPStatus.LOCKED, "reviews_locked", str(exc))
            return
        except (UnicodeError, json.JSONDecodeError, ContractError) as exc:
            self._json_error(HTTPStatus.BAD_REQUEST, "invalid_review", str(exc))
            return
        except OSError:
            self._json_error(HTTPStatus.INTERNAL_SERVER_ERROR, "persistence_failed", "could not persist review")
            return
        self._json(HTTPStatus.OK, result)

    def _route(self, *, send_body: bool) -> None:
        parsed = urlsplit(self.path)
        if parsed.path == "/api/state" and not parsed.query:
            if not send_body:
                self.send_error(HTTPStatus.METHOD_NOT_ALLOWED)
            else:
                self._json(HTTPStatus.OK, self.server.application.state())
            return
        clip_match = _CLIP_ROUTE_RE.fullmatch(parsed.path)
        if clip_match is not None and not parsed.query:
            entry = self.server.application.manifest.entries.get(clip_match.group(1))
            if entry is None or not entry.ready:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            self._file(entry.path, "video/mp4", send_body=send_body, allow_range=True)
            return
        if parsed.path == "/":
            path = self.server.application.web_root / "index.html"
        elif parsed.path in {"/web", "/web/"}:
            path = self.server.application.web_root / "index.html"
        elif parsed.path.startswith("/web/"):
            try:
                decoded = unquote(parsed.path[5:], errors="strict")
            except UnicodeError:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            if "\\" in decoded or "\x00" in decoded:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            path = (self.server.application.web_root / decoded).resolve()
            try:
                path.relative_to(self.server.application.web_root)
            except ValueError:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
        else:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        if not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        if content_type.startswith("text/") or content_type in {"application/javascript", "application/json"}:
            content_type += "; charset=utf-8"
        self._file(path, content_type, send_body=send_body, allow_range=False)

    def _file(self, path: Path, content_type: str, *, send_body: bool, allow_range: bool) -> None:
        try:
            size = path.stat().st_size
            byte_range = parse_single_range(self.headers.get("Range"), size) if allow_range else None
        except (OSError, RangeNotSatisfiable):
            if allow_range:
                self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                self.send_header("Content-Range", f"bytes */{size if 'size' in locals() else 0}")
                self.send_header("Content-Length", "0")
                self.end_headers()
            else:
                self.send_error(HTTPStatus.NOT_FOUND)
            return
        start, end = (0, size - 1) if byte_range is None else byte_range
        length = 0 if size == 0 else end - start + 1
        self.send_response(HTTPStatus.OK if byte_range is None else HTTPStatus.PARTIAL_CONTENT)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(length))
        self.send_header("Cache-Control", "no-store")
        if allow_range:
            self.send_header("Accept-Ranges", "bytes")
        if byte_range is not None:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        if not send_body or length == 0:
            return
        try:
            with path.open("rb") as handle:
                handle.seek(start)
                remaining = length
                while remaining:
                    chunk = handle.read(min(1024 * 1024, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
        except (BrokenPipeError, ConnectionResetError):
            return

    def _json(self, status: HTTPStatus, payload: Any) -> None:
        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(encoded)

    def _json_error(self, status: HTTPStatus, code: str, message: str) -> None:
        self._json(status, {"error": {"code": code, "message": message}})


def create_application(*, occurrences_path: Path = OCCURRENCES_PATH, manifest_path: Path = MANIFEST_PATH, reviews_path: Path = REVIEWS_PATH, web_root: Path = WEB_ROOT, clock: Callable[[], datetime] = _utc_now, expected_occurrence_count: int | None = EXPECTED_OCCURRENCE_COUNT) -> ReviewApplication:
    catalog = OccurrenceCatalog.load(
        occurrences_path, expected_count=expected_occurrence_count
    )
    manifest = ClipManifest.load(manifest_path, catalog)
    store = ReviewStore(reviews_path, catalog, clock=clock)
    return ReviewApplication(catalog, manifest, store, web_root)


def create_server(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT, **application_paths: Any) -> ReviewHTTPServer:
    return ReviewHTTPServer((host, port), create_application(**application_paths))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args()
    server = create_server(args.host, args.port)
    print(f"[ready] http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
