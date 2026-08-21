"""Fixed, unattended orchestration for the 11-clip CowTrack production run.

The scientific stages remain independent and keep their existing strict
``_SUCCESS.json`` contracts.  This module only wires those stages together,
protects any unmanaged output, and publishes the combined CSV plus 11 tracked
videos requested for this dataset.
"""

from __future__ import annotations

import contextlib
import csv
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import sys
import threading
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, TextIO

from cowtrack.config import ContractError


SCHEMA_VERSION = "1.0"
SEQUENCE_ID = "dairy_farm_1_gopro1_20250505"
RUN_ID = f"{SEQUENCE_ID}_all11_y20260716"
WORK_ID = f"{SEQUENCE_ID}_all11"
EXPECTED_PROJECT_ROOT = Path("/home/hyw/re-identification")
EXPECTED_VENV_ROOT = Path("/home/hyw/.venvs/trackID")
EXPECTED_PYTHON_VERSION = "3.14.4"
EXPECTED_SCIPY_VERSION = "1.18.0"
EXPECTED_VISIBLE_GPU = "1"
EXPECTED_CLIPS = tuple((index - 1, f"GX{index:02d}0006") for index in range(1, 12))
EXPECTED_FRAME_COUNTS = (
    88_320,
    84_480,
    78_720,
    84_480,
    78_720,
    78_720,
    76_800,
    71_040,
    72_960,
    65_280,
    46_972,
)
EXPECTED_VIDEO_ROOT = Path(
    "/home/hyw/UPAN_HYW/May 5 2025 Dairy Farm 1 Videos/Gopro1/100GOPRO"
)
EXPECTED_BBOX_ROOT = Path("/home/hyw/UPAN_HYW/stage1_output/1/Gopro1")
SELECTED_CHECKPOINT = Path(
    "/home/hyw/re-identification-models/MegaDescriptor-L-384/pytorch_model.bin"
)
EXPECTED_CHECKPOINT_SHA256 = (
    "ccfe757f50f7984a115ffe00921cd4c09e260e645215463b23814586040227a3"
)
EXPECTED_MODEL_REVISION = "33b3c6f4ee0c386a4126cc3dcd23843920613fa1"
WORKSPACE_MARKER = "_AUTOMATED_PIPELINE.json"


@dataclass(frozen=True, slots=True)
class PipelineLayout:
    project_root: Path
    venv_root: Path
    manifest: Path
    work_root: Path
    reference_root: Path
    deliverables_root: Path
    log_path: Path

    @classmethod
    def fixed(cls) -> "PipelineLayout":
        root = EXPECTED_PROJECT_ROOT
        return cls(
            project_root=root,
            venv_root=EXPECTED_VENV_ROOT,
            manifest=root / "data" / "manifest.csv",
            work_root=root / "work" / WORK_ID,
            reference_root=root / "work" / f"{WORK_ID}_inspection_backup_20260715",
            deliverables_root=root / "final" / RUN_ID,
            log_path=root / "logs" / "run_pipeline_all11_y20260716.log",
        )

    def config(self, name: str) -> Path:
        return self.project_root / "configs" / name

    def stage(self, name: str) -> Path:
        return self.work_root / name


@dataclass(frozen=True, slots=True)
class PipelineRunners:
    s00: Callable[..., Any]
    s01: Callable[..., Any]
    s01_plan: Callable[..., Any]
    s02: Callable[..., Any]
    s03: Callable[..., Any]
    s04_propose: Callable[..., Any]
    s04_finalize: Callable[..., Any]
    s05_calibrate_long: Callable[..., Any]
    s05_propose: Callable[..., Any]
    s05_finalize: Callable[..., Any]
    s05_force_prepare: Callable[..., Any]
    s05_force: Callable[..., Any]
    s06: Callable[..., Any]
    publish: Callable[..., Any]


class _TeeStream:
    """Mirror every Python text write to the console and a flushed log file."""

    def __init__(self, primary: TextIO, logfile: TextIO, lock: threading.Lock) -> None:
        self._primary = primary
        self._logfile = logfile
        self._lock = lock
        self.encoding = getattr(primary, "encoding", "utf-8")
        self.errors = getattr(primary, "errors", "strict")

    def write(self, text: str) -> int:
        if not isinstance(text, str):
            raise TypeError("pipeline log stream accepts text only")
        with self._lock:
            written = self._primary.write(text)
            self._primary.flush()
            self._logfile.write(text)
            self._logfile.flush()
        return len(text) if written is None else int(written)

    def flush(self) -> None:
        with self._lock:
            self._primary.flush()
            self._logfile.flush()

    def isatty(self) -> bool:
        return bool(getattr(self._primary, "isatty", lambda: False)())

    def fileno(self) -> int:
        return self._primary.fileno()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except OSError as exc:
        temporary.unlink(missing_ok=True)
        raise ContractError(f"cannot atomically write pipeline marker {path}: {exc}") from exc


@contextlib.contextmanager
def realtime_pipeline_log(path: Path) -> Iterator[None]:
    """Install a stdout/stderr tee after proving that the log flushes to disk."""

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        logfile = path.open("a", encoding="utf-8", buffering=1, newline="\n")
    except OSError as exc:
        raise ContractError(f"cannot open pipeline log {path}: {exc}") from exc
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    lock = threading.Lock()
    try:
        before = path.stat().st_size
        probe = f"[pipeline] realtime log probe pid={os.getpid()}\n"
        logfile.write(probe)
        logfile.flush()
        os.fsync(logfile.fileno())
        if path.stat().st_size < before + len(probe.encode("utf-8")):
            raise ContractError(f"pipeline log did not flush immediately: {path}")
        sys.stdout = _TeeStream(original_stdout, logfile, lock)  # type: ignore[assignment]
        sys.stderr = _TeeStream(original_stderr, logfile, lock)  # type: ignore[assignment]
        print(f"[pipeline] realtime log verified: {path}", flush=True)
        yield
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        logfile.flush()
        logfile.close()


def _validate_fixed_manifest(path: Path) -> None:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except OSError as exc:
        raise ContractError(f"cannot read fixed manifest {path}: {exc}") from exc
    if len(rows) != len(EXPECTED_CLIPS):
        raise ContractError(
            f"fixed pipeline manifest must contain exactly {len(EXPECTED_CLIPS)} clips"
        )
    for row, (expected_order, expected_clip) in zip(rows, EXPECTED_CLIPS, strict=True):
        if (
            row.get("sequence_id") != SEQUENCE_ID
            or row.get("clip_order") != str(expected_order)
            or row.get("clip_id") != expected_clip
        ):
            raise ContractError("fixed pipeline manifest clip identity/order differs")
        expected_paths = {
            "video_path": EXPECTED_VIDEO_ROOT / f"{expected_clip}.MP4",
            "bbox_csv_path": EXPECTED_BBOX_ROOT / expected_clip / "tracking_boxes.csv",
        }
        for key, expected_path in expected_paths.items():
            value = row.get(key)
            if not value or Path(value).resolve() != expected_path:
                raise ContractError(
                    f"fixed manifest {key} differs for {expected_clip}: {value}"
                )
            if not expected_path.is_file():
                raise ContractError(f"fixed manifest {key} does not exist: {value}")
        if row.get("frame_index_base") != "0" or row.get("bbox_format") != "xywh":
            raise ContractError(
                f"fixed manifest coordinate contract differs for {expected_clip}"
            )


def _validate_time_sequence(path: Path) -> None:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except OSError as exc:
        raise ContractError(f"cannot read fixed time-sequence CSV {path}: {exc}") from exc
    if len(rows) != len(EXPECTED_CLIPS):
        raise ContractError("time-sequence CSV does not contain the fixed 11 clips")
    for row, (clip_order, clip_id), frame_count in zip(
        rows, EXPECTED_CLIPS, EXPECTED_FRAME_COUNTS, strict=True
    ):
        if (
            row.get("sequence_order") != str(clip_order + 1)
            or row.get("file_name") != f"{clip_id}.MP4"
            or row.get("frame_count") != str(frame_count)
            or row.get("frame_rate") != "30000/1001"
        ):
            raise ContractError(
                f"time-sequence identity/cadence differs at clip {clip_order}: {clip_id}"
            )


def validate_fixed_layout(layout: PipelineLayout) -> None:
    if layout.project_root.resolve() != EXPECTED_PROJECT_ROOT:
        raise ContractError(
            f"fixed pipeline must run from {EXPECTED_PROJECT_ROOT}, got {layout.project_root}"
        )
    required = (
        layout.manifest,
        layout.project_root / "gopro_time_sequence.csv",
        layout.config("production.yaml"),
        layout.config("s01_microtrack.yaml"),
        layout.config("s01_review.yaml"),
        layout.config("s02_appearance.yaml"),
        layout.config("s03_calibration.yaml"),
        layout.config("s04_proposals.yaml"),
        layout.config("s04_finalize.yaml"),
        layout.config("s05_long_calibration.yaml"),
        layout.config("s05_proposals.yaml"),
        layout.config("s05_finalize.yaml"),
        layout.config("s05_force_appearance.yaml"),
        layout.config("s06_export.yaml"),
        SELECTED_CHECKPOINT,
        SELECTED_CHECKPOINT.parent / "config.json",
        SELECTED_CHECKPOINT.parent / "README.md",
        SELECTED_CHECKPOINT.parent
        / ".cache"
        / "huggingface"
        / "download"
        / f"{SELECTED_CHECKPOINT.name}.metadata",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise ContractError(f"fixed pipeline inputs are missing: {missing}")
    _validate_fixed_manifest(layout.manifest)
    _validate_time_sequence(layout.project_root / "gopro_time_sequence.csv")
    for left, right, label in (
        (layout.work_root, layout.reference_root, "work/reference"),
        (layout.work_root, layout.deliverables_root, "work/deliverables"),
        (layout.reference_root, layout.deliverables_root, "reference/deliverables"),
    ):
        a = left.resolve()
        b = right.resolve()
        if a == b or a in b.parents or b in a.parents:
            raise ContractError(f"pipeline {label} paths overlap")


def _validate_model_bundle() -> None:
    metadata_path = (
        SELECTED_CHECKPOINT.parent
        / ".cache"
        / "huggingface"
        / "download"
        / f"{SELECTED_CHECKPOINT.name}.metadata"
    )
    try:
        revision = metadata_path.read_text(encoding="utf-8").splitlines()[0]
    except (OSError, IndexError) as exc:
        raise ContractError(f"cannot read model revision metadata: {metadata_path}") from exc
    if revision != EXPECTED_MODEL_REVISION:
        raise ContractError(
            f"model revision differs: {revision} != {EXPECTED_MODEL_REVISION}"
        )
    digest = hashlib.sha256()
    try:
        with SELECTED_CHECKPOINT.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise ContractError(f"cannot fingerprint selected checkpoint: {exc}") from exc
    observed = digest.hexdigest()
    if observed != EXPECTED_CHECKPOINT_SHA256:
        raise ContractError(
            f"selected checkpoint hash differs: {observed} != {EXPECTED_CHECKPOINT_SHA256}"
        )
    print(
        f"[pipeline] locked model bundle verified: {EXPECTED_MODEL_REVISION}",
        flush=True,
    )


def preflight_runtime(layout: PipelineLayout) -> None:
    """Fail before touching the historical work tree if runtime/GPU is wrong."""

    if os.environ.get("CUDA_VISIBLE_DEVICES") != EXPECTED_VISIBLE_GPU:
        raise ContractError("run_pipeline requires CUDA_VISIBLE_DEVICES=1")
    if os.environ.get("HF_HUB_OFFLINE") != "1":
        raise ContractError("run_pipeline requires HF_HUB_OFFLINE=1")
    if Path(sys.prefix).resolve() != layout.venv_root.resolve():
        raise ContractError(
            f"run_pipeline requires existing venv {layout.venv_root}; sys.prefix={sys.prefix}"
        )
    actual_python = platform.python_version()
    if actual_python != EXPECTED_PYTHON_VERSION:
        raise ContractError(
            f"run_pipeline requires Python {EXPECTED_PYTHON_VERSION}, got {actual_python}"
        )
    try:
        scipy_version = importlib.metadata.version("scipy")
    except importlib.metadata.PackageNotFoundError as exc:
        raise ContractError("run_pipeline requires SciPy 1.18.0") from exc
    if scipy_version != EXPECTED_SCIPY_VERSION:
        raise ContractError(
            f"run_pipeline requires SciPy {EXPECTED_SCIPY_VERSION}, got {scipy_version}"
        )

    _validate_model_bundle()

    try:
        torch = importlib.import_module("torch")
    except Exception as exc:
        raise ContractError(f"cannot import the existing PyTorch runtime: {exc}") from exc
    try:
        cuda_available = bool(torch.cuda.is_available())
        device_count = int(torch.cuda.device_count())
        device_name = str(torch.cuda.get_device_name(0)) if device_count else ""
    except Exception as exc:
        raise ContractError(f"cannot query required GPU 1 through PyTorch: {exc}") from exc
    if not cuda_available or device_count != 1:
        raise ContractError(
            "CUDA_VISIBLE_DEVICES=1 must expose exactly one usable logical CUDA device"
        )
    print(f"[pipeline] GPU preflight passed: logical cuda:0 = {device_name}", flush=True)

    # Exercise one real NVENC frame now, rather than discovering an unavailable
    # encoder only after all CPU/GPU identity stages have completed.
    from cowtrack.qa.s06_config import load_s06_export_config
    from cowtrack.stages.s06_export import _preflight_nvenc

    s06_config, _, _ = load_s06_export_config(layout.config("s06_export.yaml"))
    _preflight_nvenc(s06_config)
    print("[pipeline] GPU 1 NVENC preflight passed", flush=True)


def _workspace_payload(layout: PipelineLayout) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "sequence_id": SEQUENCE_ID,
        "workspace_kind": "fresh_11_clip_automated_pipeline_with_stage_resume",
        "historical_output_reused": False,
        "inspection_backup": str(layout.reference_root),
    }


def prepare_fresh_workspace(layout: PipelineLayout) -> None:
    """Rename the old inspection tree once, then create the resumable work root."""

    marker = layout.work_root / WORKSPACE_MARKER
    expected = _workspace_payload(layout)
    if layout.work_root.exists():
        if not layout.work_root.is_dir() or layout.work_root.is_symlink():
            raise ContractError(f"pipeline work root is not a safe directory: {layout.work_root}")
        if marker.is_file():
            try:
                observed = json.loads(marker.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ContractError(f"cannot read pipeline workspace marker: {exc}") from exc
            if observed != expected:
                raise ContractError("pipeline workspace marker differs")
            print(f"[pipeline] resuming managed work root: {layout.work_root}", flush=True)
            return
        if layout.reference_root.exists():
            raise ContractError(
                "unmanaged work root and inspection backup both exist; refusing ambiguity"
            )
        try:
            os.replace(layout.work_root, layout.reference_root)
        except OSError as exc:
            raise ContractError(
                f"cannot rename inspection-only output to {layout.reference_root}: {exc}"
            ) from exc
        print(
            f"[pipeline] renamed inspection-only output: {layout.reference_root}",
            flush=True,
        )
    try:
        layout.work_root.mkdir(parents=True, exist_ok=False)
    except OSError as exc:
        raise ContractError(f"cannot create fresh pipeline work root: {exc}") from exc
    _atomic_write_json(marker, expected)
    print(f"[pipeline] created fresh resumable work root: {layout.work_root}", flush=True)


def load_default_runners() -> PipelineRunners:
    """Import heavy stage modules only after GPU/environment selection is fixed."""

    from cowtrack.final_publish import publish_final_deliverables
    from cowtrack.qa.s01_review_plan_only import run_s01_review_plan_only
    from cowtrack.stages.s00_ingest import run_s00
    from cowtrack.stages.s01_microtrack import run_s01
    from cowtrack.stages.s02_appearance import run_s02
    from cowtrack.stages.s03_calibrate import run_s03
    from cowtrack.stages.s04_finalize import run_s04_finalize
    from cowtrack.stages.s04_propose import run_s04_propose
    from cowtrack.stages.s05_calibrate_long import run_s05_calibrate_long
    from cowtrack.stages.s05_finalize import run_s05_finalize
    from cowtrack.stages.s05_force_appearance import (
        run_s05_force_appearance,
        run_s05_force_appearance_prepare,
    )
    from cowtrack.stages.s05_propose import run_s05_propose
    from cowtrack.stages.s06_export import run_s06

    return PipelineRunners(
        s00=run_s00,
        s01=run_s01,
        s01_plan=run_s01_review_plan_only,
        s02=run_s02,
        s03=run_s03,
        s04_propose=run_s04_propose,
        s04_finalize=run_s04_finalize,
        s05_calibrate_long=run_s05_calibrate_long,
        s05_propose=run_s05_propose,
        s05_finalize=run_s05_finalize,
        s05_force_prepare=run_s05_force_appearance_prepare,
        s05_force=run_s05_force_appearance,
        s06=run_s06,
        publish=publish_final_deliverables,
    )


def _stage(label: str, callback: Callable[..., Any], *args: Any) -> Any:
    print(f"[pipeline] >>> {label}", flush=True)
    result = callback(*args)
    print(f"[pipeline] <<< {label} complete", flush=True)
    return result


def execute_stages(layout: PipelineLayout, runners: PipelineRunners) -> None:
    """Execute/revalidate every required stage in the fixed unattended order."""

    ingest = layout.stage("00_ingest")
    micro = layout.stage("01_microtrack")
    review_plan = layout.stage("01_review_plan")
    appearance = layout.stage("02_appearance")
    calibration = layout.stage("03_calibration")
    short_proposals = layout.stage("04_short_proposals")
    stable = layout.stage("04_short_stable")
    long_calibration = layout.stage("05_long_calibration")
    long_proposals = layout.stage("05_long_proposals")
    prior_global = layout.stage("05_global_link")
    forced_appearance_prepare = layout.stage("05_forced_appearance_prepare")
    forced_global = layout.stage("05_forced_appearance")
    export = layout.stage("06_export")

    _stage(
        "S00 ingest",
        runners.s00,
        layout.manifest,
        layout.config("production.yaml"),
        ingest,
    )
    _stage(
        "S01 microtrack",
        runners.s01,
        ingest,
        layout.config("s01_microtrack.yaml"),
        micro,
    )
    _stage(
        "S01 review plan only",
        runners.s01_plan,
        ingest,
        micro,
        layout.config("s01_review.yaml"),
        review_plan,
    )
    _stage(
        "S02 locked MegaDescriptor appearance",
        runners.s02,
        layout.manifest,
        ingest,
        micro,
        review_plan / "review_manifest.json",
        layout.config("s02_appearance.yaml"),
        "cuda:0",
        appearance,
    )
    _stage(
        "S03 calibration",
        runners.s03,
        ingest,
        micro,
        appearance,
        layout.config("s03_calibration.yaml"),
        calibration,
    )
    _stage(
        "S04 short proposals",
        runners.s04_propose,
        ingest,
        micro,
        appearance,
        calibration,
        layout.config("s04_proposals.yaml"),
        short_proposals,
    )
    _stage(
        "S04 blanket-approved finalize",
        runners.s04_finalize,
        ingest,
        micro,
        appearance,
        calibration,
        short_proposals,
        layout.config("s04_finalize.yaml"),
        stable,
    )
    _stage(
        "S05 long calibration",
        runners.s05_calibrate_long,
        ingest,
        micro,
        appearance,
        calibration,
        stable,
        layout.config("s05_long_calibration.yaml"),
        long_calibration,
    )
    _stage(
        "S05 long proposals",
        runners.s05_propose,
        ingest,
        micro,
        appearance,
        stable,
        long_calibration,
        layout.config("s05_proposals.yaml"),
        long_proposals,
    )
    _stage(
        "S05 blanket-approved global path cover",
        runners.s05_finalize,
        ingest,
        micro,
        appearance,
        stable,
        long_calibration,
        long_proposals,
        layout.config("s05_finalize.yaml"),
        prior_global,
    )
    _stage(
        "S05 forced appearance preparation",
        runners.s05_force_prepare,
        layout.manifest,
        ingest,
        micro,
        appearance,
        stable,
        layout.config("s05_force_appearance.yaml"),
        "cuda:0",
        forced_appearance_prepare,
    )
    _stage(
        "S05 forced exact-62 appearance linking",
        runners.s05_force,
        layout.manifest,
        ingest,
        micro,
        appearance,
        stable,
        prior_global,
        layout.config("s05_force_appearance.yaml"),
        forced_appearance_prepare,
        forced_global,
    )
    _stage(
        "S06 internal QA and overlay export",
        runners.s06,
        layout.manifest,
        ingest,
        micro,
        stable,
        forced_global,
        layout.config("s06_export.yaml"),
        export,
    )
    _stage(
        "publish combined CSV and 11 tracked videos",
        runners.publish,
        layout.manifest,
        export,
        layout.deliverables_root,
    )
    print("[pipeline] all stages complete", flush=True)
    print(f"[pipeline] final deliverables: {layout.deliverables_root}", flush=True)


def run_pipeline(
    *,
    layout: PipelineLayout | None = None,
    runners: PipelineRunners | None = None,
    preflight: Callable[[PipelineLayout], None] = preflight_runtime,
) -> None:
    selected_layout = PipelineLayout.fixed() if layout is None else layout
    with realtime_pipeline_log(selected_layout.log_path):
        _run_pipeline_body(selected_layout, runners=runners, preflight=preflight)


def _run_pipeline_body(
    layout: PipelineLayout,
    *,
    runners: PipelineRunners | None,
    preflight: Callable[[PipelineLayout], None],
) -> None:
    validate_fixed_layout(layout)
    preflight(layout)
    # Import every production stage before moving the inspection-only work tree.
    # Missing dependencies or import-time incompatibilities must leave it in place.
    selected_runners = load_default_runners() if runners is None else runners
    prepare_fresh_workspace(layout)
    execute_stages(layout, selected_runners)


def main() -> int:
    layout = PipelineLayout.fixed()
    try:
        with realtime_pipeline_log(layout.log_path):
            try:
                _run_pipeline_body(
                    layout,
                    runners=None,
                    preflight=preflight_runtime,
                )
            except ContractError as exc:
                print(f"[fatal] {exc}", file=sys.stderr, flush=True)
                return 2
            except Exception as exc:
                print(f"[fatal] unexpected pipeline failure: {exc}", file=sys.stderr, flush=True)
                traceback.print_exc(file=sys.stderr)
                return 1
    except ContractError as exc:
        # Opening/verifying the log itself can fail before the tee is installed.
        print(f"[fatal] {exc}", file=sys.stderr, flush=True)
        return 2
    except Exception as exc:
        print(f"[fatal] cannot start pipeline logging: {exc}", file=sys.stderr, flush=True)
        traceback.print_exc(file=sys.stderr)
        return 1
    return 0


__all__ = [
    "PipelineLayout",
    "PipelineRunners",
    "execute_stages",
    "main",
    "prepare_fresh_workspace",
    "preflight_runtime",
    "realtime_pipeline_log",
    "run_pipeline",
    "validate_fixed_layout",
]


if __name__ == "__main__":
    raise SystemExit(main())
