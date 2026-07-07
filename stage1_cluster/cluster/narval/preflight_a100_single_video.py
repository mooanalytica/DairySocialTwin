#!/usr/bin/env python3
"""Lightweight Narval/A100 preflight for one-video DCSNA jobs."""

from __future__ import annotations

import argparse
import ast
import importlib
import os
import sys
import time
from pathlib import Path


def configure_stdio() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def log(message: str) -> None:
    print(message, flush=True)


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise SystemExit(f"[fatal] missing {label}: {path}")


def require_dir(path: Path, label: str) -> None:
    if not path.is_dir():
        raise SystemExit(f"[fatal] missing {label}: {path}")


def iter_mp4s(root: Path) -> list[Path]:
    return sorted(
        (p for p in root.rglob("*") if p.is_file() and p.suffix.lower() == ".mp4"),
        key=lambda p: str(p).lower(),
    )


def log_flush_check(total_seconds: int, interval_seconds: int) -> None:
    total_seconds = max(0, int(total_seconds))
    interval_seconds = max(1, int(interval_seconds))
    if total_seconds <= 0:
        return

    log("[log-check] starting stdout flush check; confirm these lines appear live in the Slurm .out file")
    elapsed = 0
    heartbeat = 1
    while elapsed < total_seconds:
        log(f"[log-check] heartbeat={heartbeat} elapsed={elapsed}s")
        sleep_for = min(interval_seconds, total_seconds - elapsed)
        time.sleep(sleep_for)
        elapsed += sleep_for
        heartbeat += 1
    log(f"[log-check] complete elapsed={elapsed}s")


def ast_check(path: Path) -> None:
    log(f"[check] AST parse {path}")
    source = path.read_text(encoding="utf-8")
    ast.parse(source, filename=str(path))
    log(f"[ok] AST parse {path.name}")


def import_check(module_name: str):
    start = time.time()
    log(f"[check] import {module_name}")
    module = importlib.import_module(module_name)
    log(f"[ok] import {module_name}; sec={time.time() - start:.2f}")
    return module


def main(argv: list[str] | None = None) -> int:
    configure_stdio()
    parser = argparse.ArgumentParser(description="Preflight checks for the one-video DCSNA Narval/A100 job.")
    parser.add_argument("--project-root", default="/work")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--expected-video", required=True)
    parser.add_argument("--require-device-substring", default="A100")
    parser.add_argument("--log-check-seconds", type=int, default=20)
    parser.add_argument("--log-check-interval", type=int, default=10)
    args = parser.parse_args(argv)

    project_root = Path(args.project_root).resolve()
    input_dir = Path(args.input_dir).resolve()
    expected_video = str(args.expected_video).strip()

    log("[boot] DCSNA one-video Narval/A100 preflight started")
    log(f"[info] pid={os.getpid()}")
    log(f"[info] project_root={project_root}")
    log(f"[info] input_dir={input_dir}")
    log(f"[info] expected_video={expected_video}")

    log_flush_check(args.log_check_seconds, args.log_check_interval)

    require_dir(project_root, "project root")
    require_dir(input_dir, "input directory")

    required_files = [
        project_root / "run_full_interaction_pipeline.py",
        project_root / "stage2_pair_features.py",
        project_root / "models" / "Object_Detection_Trained_Model.pt",
        project_root / "models" / "Identification_Model_Trained.pt",
        project_root / "models" / "Keypoint_Model_Trained.pth",
        project_root / "models_new" / "stage2_interaction_gate_best.pt",
        project_root / "models_new" / "stage2_valence_inception_best.pt",
        project_root
        / "vendor"
        / "ZebraPoseViTPose"
        / "ZebraPose"
        / "configs"
        / "animal"
        / "2d_kpt_sview_rgb_img"
        / "topdown_heatmap"
        / "MAE_pret_syn"
        / "s_zebras_old_adam.py",
    ]
    for path in required_files:
        require_file(path, "required project file")
    log(f"[ok] required project/model files present: {len(required_files)}")

    ast_check(project_root / "run_full_interaction_pipeline.py")

    videos = iter_mp4s(input_dir)
    log(f"[info] discovered MP4 files: {len(videos)}")
    for path in videos:
        log(f"[video] {path}")
    if len(videos) != 1:
        raise SystemExit(f"[fatal] expected exactly 1 MP4 under {input_dir}, got {len(videos)}")
    if videos[0].name != expected_video:
        raise SystemExit(f"[fatal] expected video {expected_video}, got {videos[0].name}")

    sys.path.insert(0, str(project_root))
    sys.path.insert(0, str(project_root / "vendor" / "ByteTrack"))
    sys.path.insert(0, str(project_root / "vendor" / "ZebraPoseViTPose"))

    for module_name in [
        "cv2",
        "numpy",
        "pandas",
        "torch",
        "torchvision",
        "ultralytics",
        "mmcv",
        "mmpose",
    ]:
        import_check(module_name)

    torch = importlib.import_module("torch")
    log(f"[ok] torch={torch.__version__}; cuda={torch.version.cuda}")
    if not torch.cuda.is_available():
        raise SystemExit("[fatal] CUDA is required but torch.cuda.is_available() is False")
    gpu_name = torch.cuda.get_device_name(0)
    log(f"[ok] gpu={gpu_name}")
    required = str(args.require_device_substring or "").strip()
    if required and required.lower() not in gpu_name.lower():
        raise SystemExit(f"[fatal] expected GPU name containing {required!r}, got {gpu_name!r}")

    log("[ok] DCSNA one-video Narval/A100 preflight complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
