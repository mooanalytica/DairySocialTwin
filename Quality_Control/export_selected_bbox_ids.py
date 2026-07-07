from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.dont_write_bytecode = True

import visualize_bbox_roi as vis

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - tqdm is optional at runtime
    tqdm = None


WORKDIR = Path(__file__).resolve().parent
OUTPUT_ROOT = WORKDIR / "output_selected_IDs"
LOG_PATH = OUTPUT_ROOT / "run.log"


def configure_stdio() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def log(message: str) -> None:
    print(message, flush=True)
    if LOG_PATH.parent.exists():
        with LOG_PATH.open("a", encoding="utf-8", newline="\n") as f:
            f.write(message + "\n")
            f.flush()


def safe_reset_output() -> None:
    out = OUTPUT_ROOT.resolve()
    root = WORKDIR.resolve()
    try:
        out.relative_to(root)
    except ValueError as exc:
        raise RuntimeError(f"Refusing to clear output outside workspace: {out}") from exc
    if out.exists():
        try:
            shutil.rmtree(out)
        except PermissionError:
            clear_output_with_powershell(out, root)
    out.mkdir(parents=True, exist_ok=True)


def ps_quote(path: Path) -> str:
    return "'" + str(path).replace("'", "''") + "'"


def clear_output_with_powershell(out: Path, root: Path) -> None:
    script = f"""
$target = Resolve-Path -LiteralPath {ps_quote(out)}
$root = Resolve-Path -LiteralPath {ps_quote(root)}
if (-not ($target.Path -like "$($root.Path)\\*")) {{
    throw "Refusing to clear output outside workspace: $($target.Path)"
}}
Get-ChildItem -LiteralPath $target.Path -Force | ForEach-Object {{
    Remove-Item -LiteralPath $_.FullName -Recurse -Force -ErrorAction Stop
}}
"""
    subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", script],
        check=True,
    )


def write_log_test() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    log("log realtime check: line 1")
    time.sleep(1)
    log("log realtime check: line 2")


def parse_frame(value: object) -> int:
    return int(value)


def clip_frame_count(record: dict[str, object]) -> int:
    clip_path = Path(str(record["clip_path"]))
    cap = vis.cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video for frame count: {clip_path}")
    try:
        frame_count = int(cap.get(vis.cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()
    if frame_count <= 0:
        raise RuntimeError(f"Could not determine frame count for: {clip_path}")
    return frame_count


def dedupe_in_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def group_records(records: list[dict[str, object]]) -> dict[str, list[dict[str, object]]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in records:
        grouped[str(record["clip_name"])].append(record)
    for clip_records in grouped.values():
        clip_records.sort(key=lambda item: (str(item["clip_id"]), str(item["event_id"])))
    return dict(sorted(grouped.items()))


def selected_ids_for_frame(
    frame: int,
    records: list[dict[str, object]],
    by_stage1_frame: dict[int, list[dict[str, float | int | str]]],
) -> list[str]:
    selected: list[str] = []
    for record in records:
        stage1_frame = frame + parse_frame(record.get("stage1_frame_offset", 0))
        boxes = by_stage1_frame.get(stage1_frame, [])
        selection = vis.select_frame_boxes(record, boxes)
        selected.extend(str(track_id) for track_id in selection["track_ids"])
    return dedupe_in_order(selected)


def write_clip_selected_ids(
    clip_name: str,
    records: list[dict[str, object]],
    output_csv: Path,
) -> dict[str, object]:
    tracking_csv = Path(str(records[0]["tracking_csv"]))
    by_stage1_frame = vis.load_tracking_by_frame(
        tracking_csv,
        str(records[0]["tracking_video_name"]),
    )

    frame_count = clip_frame_count(records[0])
    frames = list(range(frame_count))
    missing_source_frames = 0
    for record in records:
        missing_source_frames += sum(
            1
            for frame in frames
            if frame + parse_frame(record.get("stage1_frame_offset", 0)) not in by_stage1_frame
        )

    rows_written = 0
    empty_rows = 0
    one_id_rows = 0
    multi_id_rows = 0
    unique_ids: set[str] = set()

    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["frame", "track_ids"])
        writer.writeheader()
        for frame in frames:
            ids = selected_ids_for_frame(frame, records, by_stage1_frame)
            unique_ids.update(ids)
            if not ids:
                empty_rows += 1
            elif len(ids) == 1:
                one_id_rows += 1
            else:
                multi_id_rows += 1
            writer.writerow({"frame": frame, "track_ids": ";".join(ids)})
            rows_written += 1

    return {
        "clip_name": clip_name,
        "output_csv": str(output_csv),
        "tracking_csv": str(tracking_csv),
        "annotation_rows": len(records),
        "stage1_frame_count": len(by_stage1_frame),
        "source_frame_count": frame_count,
        "rows_written": rows_written,
        "empty_rows": empty_rows,
        "one_id_rows": one_id_rows,
        "multi_id_rows": multi_id_rows,
        "unique_selected_track_ids": sorted(unique_ids, key=vis.track_sort_key),
        "missing_source_frames": missing_source_frames,
    }


def export_selected_ids(limit: int | None = None) -> list[dict[str, object]]:
    vis.LOG_PATH = LOG_PATH
    vis.log = log

    safe_reset_output()
    log("Selected bbox ID export started")
    records = vis.build_visualization_index()
    grouped = group_records(records)
    items = list(grouped.items())
    if limit is not None:
        items = items[:limit]

    log(f"Input annotation rows: {len(records)}")
    log(f"Output MP4 CSV files: {len(items)}")

    iterator = items
    if tqdm is not None:
        iterator = tqdm(items, desc="export", mininterval=1.0, unit="clip")

    results: list[dict[str, object]] = []
    last_log = time.monotonic()
    for index, (clip_name, clip_records) in enumerate(iterator, start=1):
        output_csv = OUTPUT_ROOT / f"{clip_name}.csv"
        result = write_clip_selected_ids(clip_name, clip_records, output_csv)
        results.append(result)
        now = time.monotonic()
        if now - last_log >= 10.0 or index == len(items):
            log(f"Progress: {index}/{len(items)} CSV files written")
            last_log = now

    with (OUTPUT_ROOT / "run_summary.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    log("Selected bbox ID export complete")
    return results


def main() -> int:
    configure_stdio()
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="Limit number of MP4 CSV files. 0 means all.")
    parser.add_argument("--log-test", action="store_true")
    args = parser.parse_args()

    if args.log_test:
        write_log_test()
        return 0

    export_selected_ids(limit=args.limit if args.limit > 0 else None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
