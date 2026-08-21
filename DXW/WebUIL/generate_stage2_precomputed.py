from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from tqdm import tqdm
except Exception as exc:  # pragma: no cover - dependency is expected in the target env.
    raise RuntimeError("tqdm is required for Stage2 precompute progress output") from exc

from app import (
    REQUIRED_CUDA_VISIBLE_DEVICES,
    ROOT,
    STAGE2_CODE_DIR,
    STAGE2_PRECOMPUTED_DIR,
    STAGE2_PRECOMPUTED_SCHEMA_VERSION,
    load_samples,
    require_under_directory,
    stage2_precomputed_index_file,
)
from stage2_runtime import Stage2Runtime, load_stage2_resources


def log(message: str) -> None:
    print(message, flush=True)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        tmp_path = Path(handle.name)
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    tmp_path.replace(path)


def require_cuda_visible_devices(expected: str) -> None:
    expected_value = str(expected).strip()
    value = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if value != expected_value:
        raise RuntimeError(
            f"Refusing to precompute Stage2 without CUDA_VISIBLE_DEVICES={expected_value}. "
            f"Run: CUDA_VISIBLE_DEVICES={expected_value} python3 generate_stage2_precomputed.py "
            f"--all --cuda-visible-devices {expected_value}"
        )


def read_index() -> dict[str, Any]:
    path = stage2_precomputed_index_file()
    if not path.is_file():
        return {
            "schemaVersion": STAGE2_PRECOMPUTED_SCHEMA_VERSION,
            "createdAt": datetime.now().isoformat(timespec="seconds"),
            "updatedAt": "",
            "samples": {},
        }
    data = json.loads(path.read_text(encoding="utf-8"))
    if int(data.get("schemaVersion", -1)) != STAGE2_PRECOMPUTED_SCHEMA_VERSION:
        raise RuntimeError(f"Unsupported Stage2 precomputed index schema: {path}")
    if not isinstance(data.get("samples"), dict):
        raise RuntimeError(f"Stage2 precomputed index samples must be an object: {path}")
    return data


def write_index(index: dict[str, Any]) -> None:
    index["schemaVersion"] = STAGE2_PRECOMPUTED_SCHEMA_VERSION
    index["updatedAt"] = datetime.now().isoformat(timespec="seconds")
    atomic_write_json(stage2_precomputed_index_file(), index)


def sample_output_file(sample_id: str) -> Path:
    return STAGE2_PRECOMPUTED_DIR / f"{sample_id}.json"


def validate_existing_output(path: Path, sample: dict[str, Any]) -> dict[str, Any]:
    sample_id = sample["id"]
    resolved = require_under_directory(path, STAGE2_PRECOMPUTED_DIR)
    data = json.loads(resolved.read_text(encoding="utf-8"))
    if int(data.get("schemaVersion", -1)) != STAGE2_PRECOMPUTED_SCHEMA_VERSION:
        raise RuntimeError(f"Existing Stage2 output has unsupported schema: {resolved}")
    if str(data.get("sampleId", "")) != sample_id:
        raise RuntimeError(f"Existing Stage2 output sample mismatch for {sample_id}: {resolved}")
    if str(data.get("sourceDir", "")) != str(sample["sourceDir"]):
        raise RuntimeError(f"Existing Stage2 output source mismatch for {sample_id}: {resolved}")
    if not data.get("complete", False):
        raise RuntimeError(f"Existing Stage2 output is incomplete for {sample_id}: {resolved}")
    if not isinstance(data.get("frames", {}), dict):
        raise RuntimeError(f"Existing Stage2 output frames must be an object for {sample_id}: {resolved}")
    return {
        "sampleId": sample_id,
        "path": resolved.name,
        "sourceDir": str(sample["sourceDir"]),
        "trackingCsv": str(sample["trackingCsv"]),
        "manifest": str(sample["manifest"]),
        "frameMin": int(data.get("frameMin", 0)),
        "frameTotal": int(data.get("frameTotal", 0)),
        "framesWithInteractions": int(data.get("framesWithInteractions", len(data.get("frames", {})))),
        "complete": True,
        "generatedAt": str(data.get("generatedAt", "")),
    }


def existing_complete(index: dict[str, Any], sample: dict[str, Any]) -> bool:
    sample_id = sample["id"]
    entry = index.get("samples", {}).get(sample_id)
    if not isinstance(entry, dict) or not entry.get("complete", False):
        return False
    path = require_under_directory(STAGE2_PRECOMPUTED_DIR / str(entry.get("path", "")), STAGE2_PRECOMPUTED_DIR)
    if not path.is_file():
        return False
    validate_existing_output(path, sample)
    return True


def choose_samples(args: argparse.Namespace) -> list[dict[str, Any]]:
    candidates = load_samples(require_ready=False)
    if args.sample:
        wanted = set(args.sample)
        candidates = [sample for sample in candidates if sample["id"] in wanted]
        missing = sorted(wanted - {sample["id"] for sample in candidates})
        if missing:
            raise RuntimeError(f"Requested samples are missing from Stage1 segmented candidates: {missing}")
    elif not args.all:
        raise RuntimeError("Use --all or one or more --sample IDs.")

    if args.max_samples > 0:
        candidates = candidates[: args.max_samples]
    if not candidates:
        raise RuntimeError("No samples are available for Stage2 precompute.")
    return candidates


def precompute_sample(sample: dict[str, Any], resources: Any, overwrite: bool) -> dict[str, Any]:
    sample_id = sample["id"]
    out_path = sample_output_file(sample_id)
    if out_path.is_file() and not overwrite:
        entry = validate_existing_output(out_path, sample)
        log(f"[skip] sample {sample_id}: existing {out_path}")
        entry["skipped"] = True
        return entry

    runtime = Stage2Runtime(
        stage2_code_dir=STAGE2_CODE_DIR,
        source_dir=sample["sourceDir"],
        resources=resources,
    )
    frames: dict[str, list[dict[str, Any]]] = {}
    frame_start = int(runtime.frame_min)
    frame_stop = int(runtime.frame_total)
    pbar = tqdm(
        range(frame_start, frame_stop),
        desc=f"sample {sample_id}",
        unit="frame",
        ascii=True,
        dynamic_ncols=False,
        ncols=100,
        mininterval=1.0,
        file=sys.stdout,
    )
    for frame in pbar:
        interactions = runtime.interactions_for_frame(frame)
        if interactions:
            frames[str(int(frame))] = interactions
    pbar.close()

    payload = {
        "schemaVersion": STAGE2_PRECOMPUTED_SCHEMA_VERSION,
        "sampleId": sample_id,
        "sourceDir": str(sample["sourceDir"]),
        "trackingCsv": str(sample["trackingCsv"]),
        "keypointsCsv": str(sample["keypointsCsv"]),
        "manifest": str(sample["manifest"]),
        "generatedAt": datetime.now().isoformat(timespec="seconds"),
        "complete": True,
        "frameMin": frame_start,
        "frameTotal": frame_stop,
        "framesWithInteractions": len(frames),
        "frames": frames,
        "meta": runtime.meta(),
    }
    atomic_write_json(out_path, payload)
    log(
        f"[done] sample {sample_id}: frames {frame_start}-{frame_stop - 1}, "
        f"{len(frames)} frames with visible Stage2 links -> {out_path}"
    )
    return {
        "sampleId": sample_id,
        "path": out_path.name,
        "sourceDir": str(sample["sourceDir"]),
        "trackingCsv": str(sample["trackingCsv"]),
        "manifest": str(sample["manifest"]),
        "frameMin": frame_start,
        "frameTotal": frame_stop,
        "framesWithInteractions": len(frames),
        "complete": True,
        "generatedAt": payload["generatedAt"],
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Precompute WebUI Stage2 display results into the project-local .cache directory."
    )
    parser.add_argument("--all", action="store_true", help="Precompute every active Stage1 segmented shard.")
    parser.add_argument("--sample", action="append", default=[], help="Sample id to precompute; may be repeated.")
    parser.add_argument("--max-samples", type=int, default=0, help="Limit selected samples for smoke runs; 0 means no limit.")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate existing sample JSON files.")
    parser.add_argument("--dry-run", action="store_true", help="List selected samples without loading Stage2 models.")
    parser.add_argument("--profile", choices=["auto", "8gb", "40gb"], default="auto")
    parser.add_argument("--inference-batch-size", type=int, default=None)
    parser.add_argument(
        "--cuda-visible-devices",
        default=REQUIRED_CUDA_VISIBLE_DEVICES,
        help=f"Expected CUDA_VISIBLE_DEVICES value for Stage2 (default: {REQUIRED_CUDA_VISIBLE_DEVICES}).",
    )
    parser.add_argument("--verify-log-flush", action="store_true", help="Write flushed log lines and exit.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.verify_log_flush:
        for index in range(1, 4):
            log(f"[flush-check] tick={index} time={datetime.now().isoformat(timespec='seconds')}")
            time.sleep(1.0)
        log("[flush-check] ok")
        return 0

    samples = choose_samples(args)
    log(f"[info] WebUI root: {ROOT}")
    log(f"[info] Stage2 output: {STAGE2_PRECOMPUTED_DIR}")
    log(f"[info] selected samples: {len(samples)}")
    for sample in samples:
        log(f"[sample] {sample['id']} {sample['label']}")
    if args.dry_run:
        return 0

    require_cuda_visible_devices(args.cuda_visible_devices)
    STAGE2_PRECOMPUTED_DIR.mkdir(parents=True, exist_ok=True)
    index = read_index()
    resources = load_stage2_resources(
        STAGE2_CODE_DIR,
        device="cuda",
        profile=args.profile,
        inference_batch_size=args.inference_batch_size,
    )

    completed = 0
    skipped = 0
    for sample in samples:
        sample_id = sample["id"]
        if existing_complete(index, sample) and not args.overwrite:
            skipped += 1
            log(f"[skip] sample {sample_id}: already indexed")
            continue
        entry = precompute_sample(sample, resources, args.overwrite)
        index.setdefault("samples", {})[sample_id] = entry
        write_index(index)
        completed += 1

    log(f"[done] completed={completed} skipped={skipped} index={stage2_precomputed_index_file()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
