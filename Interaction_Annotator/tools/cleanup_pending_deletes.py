from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


WORKSPACE = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = WORKSPACE / "output"
CACHED_VIDEOS_ROOT = OUTPUT_ROOT / "cached_videos"
CACHE_VIS_ROOT = OUTPUT_ROOT / "cache_vis"
VIDEOS_ROOT = OUTPUT_ROOT / "videos"
ANNOTATIONS_ROOT = OUTPUT_ROOT / "annotations"
INDEX_PATH = OUTPUT_ROOT / "index.csv"
DISCARDED_PATH = OUTPUT_ROOT / "discarded_clips.txt"
PENDING_DELETES_PATH = OUTPUT_ROOT / "pending_deletes.csv"

PENDING_DELETE_COLUMNS = [
    "clip_id",
    "kind",
    "root",
    "path",
    "status",
    "attempts",
    "last_error",
    "queued_at",
    "updated_at",
    "deleted_at",
]
DATASET_DELETE_KINDS = {"cache", "cache_vis", "valid", "annotation"}


def log(message: str) -> None:
    print(message, flush=True)


def timestamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.exists():
        return [], []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return reader.fieldnames or [], list(reader)


def write_csv(path: Path, columns: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def is_under(path: Path, root: Path) -> bool:
    resolved = path.resolve()
    resolved_root = root.resolve()
    return resolved == resolved_root or resolved_root in resolved.parents


def assert_allowed(path: Path) -> Path:
    resolved = path.resolve()
    allowed_roots = [
        CACHED_VIDEOS_ROOT.resolve(),
        CACHE_VIS_ROOT.resolve(),
        VIDEOS_ROOT.resolve(),
        ANNOTATIONS_ROOT.resolve(),
    ]
    if not any(resolved == root or root in resolved.parents for root in allowed_roots):
        raise ValueError(f"refusing outside output roots: {resolved}")
    return resolved


def read_discarded() -> set[str]:
    if not DISCARDED_PATH.exists():
        return set()
    with DISCARDED_PATH.open("r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def write_discarded(discarded: set[str]) -> None:
    DISCARDED_PATH.parent.mkdir(parents=True, exist_ok=True)
    with DISCARDED_PATH.open("w", encoding="utf-8") as f:
        for clip_id in sorted(discarded):
            f.write(clip_id + "\n")


def remove_from_index(clip_ids: set[str]) -> int:
    columns, rows = read_csv(INDEX_PATH)
    if not columns or not rows:
        return 0
    kept = [row for row in rows if row.get("clip_id") not in clip_ids]
    removed = len(rows) - len(kept)
    if removed:
        write_csv(INDEX_PATH, columns, kept)
    return removed


def cleanup_pending(dry_run: bool, keep_deleted: bool) -> int:
    _, rows = read_csv(PENDING_DELETES_PATH)
    if not rows:
        log(f"no pending delete list found: {PENDING_DELETES_PATH}")
        return 0

    discarded = read_discarded()
    touched_clip_ids = {
        row.get("clip_id", "")
        for row in rows
        if row.get("clip_id") and row.get("kind") in DATASET_DELETE_KINDS
    }
    discarded.update(touched_clip_ids)
    if not dry_run:
        write_discarded(discarded)
        removed_index_rows = remove_from_index(touched_clip_ids)
    else:
        removed_index_rows = 0

    deleted = 0
    still_pending = 0
    invalid = 0
    now = timestamp()
    updated_rows: list[dict[str, str]] = []

    for row in rows:
        status = row.get("status", "pending")
        if status == "deleted" and not keep_deleted:
            continue

        row = {column: row.get(column, "") for column in PENDING_DELETE_COLUMNS}
        row["updated_at"] = now
        row["attempts"] = str(int(row.get("attempts") or "0") + 1)

        try:
            target = assert_allowed(Path(row["path"]))
            if not target.exists():
                row["status"] = "deleted"
                row["last_error"] = ""
                row["deleted_at"] = now
                deleted += 1
            elif not target.is_file():
                raise ValueError(f"refusing non-file path: {target}")
            elif dry_run:
                row["status"] = "pending"
                row["last_error"] = "dry run"
                still_pending += 1
            else:
                target.unlink()
                row["status"] = "deleted"
                row["last_error"] = ""
                row["deleted_at"] = now
                deleted += 1
        except Exception as exc:
            row["status"] = "pending"
            row["last_error"] = repr(exc)
            still_pending += 1
            if isinstance(exc, ValueError):
                invalid += 1

        if keep_deleted or row["status"] != "deleted":
            updated_rows.append(row)

    if not dry_run:
        write_csv(PENDING_DELETES_PATH, PENDING_DELETE_COLUMNS, updated_rows)

    log(f"deleted or already missing: {deleted}")
    log(f"still pending: {still_pending}")
    log(f"invalid/refused paths: {invalid}")
    log(f"index rows removed for queued clip ids: {removed_index_rows}")
    log(f"pending list: {PENDING_DELETES_PATH}")
    return 0 if invalid == 0 else 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keep-deleted", action="store_true")
    args = parser.parse_args()
    return cleanup_pending(args.dry_run, args.keep_deleted)


if __name__ == "__main__":
    raise SystemExit(main())
