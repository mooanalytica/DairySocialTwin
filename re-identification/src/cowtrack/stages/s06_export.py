"""Final forced-provisional audit export for the fixed 11-clip dataset.

S06 is intentionally an export/QA stage, not another identity solver.  Its
only global-identity input is the strictly validated ``S05_FORCE_APPEARANCE``
snapshot.  Appearance cosine remains evidence (never a probability), invalid
S00 detections remain unassigned, and all 62 identities remain
``forced_provisional``.
"""

from __future__ import annotations

import csv
import hashlib
import html
import json
import math
import os
import shutil
import subprocess
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from fractions import Fraction
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.csv as pacsv

from cowtrack.config import ContractError
from cowtrack.frame_observation import (
    frame_observation_policy,
    summarize_frame_observation,
)
from cowtrack.linking.runtime import FileFingerprint, fingerprint_file
from cowtrack.linking.s06_runtime import S06InputBundle, load_s06_inputs
from cowtrack.qa.ffprobe import validate_qa_mp4
from cowtrack.qa.nvenc import NvencVideoWriter
from cowtrack.qa.s06_config import (
    S06ExportConfig,
    load_s06_export_config,
)
from cowtrack.qa.s06_plan import (
    CandidateMetric,
    ContactSheetPlan,
    CropRequest,
    CropRequestIndex,
    DetectionCrop,
    build_contact_sheet_plan,
    build_crop_request_index,
    build_low_confidence_crop_requests,
    compute_directional_candidate_metrics,
    select_low_confidence_links,
)
from cowtrack.qa.s06_render import (
    OverlayDetection,
    extract_detection_crop,
    render_contact_sheet,
    render_s06_overlay_frame,
)
from cowtrack.qa.s06_report import (
    StructuralMetrics,
    build_detection_export_table,
    build_global_track_summary,
    recompute_structural_metrics,
)
from cowtrack.schemas.s06 import (
    DETECTIONS_WITH_GLOBAL_ID_SCHEMA,
    GLOBAL_TRACK_SUMMARY_SCHEMA,
)
from cowtrack.video import open_raw_video_capture


LogFn = Callable[[str], None]
_STAGE = "S06_EXPORT"
_RAW_SHAPE = (2160, 3840, 3)
_EXACT_FPS = Fraction(30000, 1001)
_FRAME_DURATION_SEC = Fraction(1001, 30000)


def log(message: str) -> None:
    print(message, flush=True)


def _read_json(path: Path, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read {label} {path}: {exc}") from exc


def _write_json(path: Path, payload: Any) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except (OSError, TypeError, ValueError) as exc:
        raise ContractError(f"cannot write S06 JSON {path}: {exc}") from exc


def _sha256(
    path: Path,
    *,
    logger: LogFn = lambda _message: None,
    context: str = "S06 artifact",
) -> str:
    digest = hashlib.sha256()
    try:
        size = int(path.stat().st_size)
    except OSError as exc:
        raise ContractError(f"cannot stat {context} {path}: {exc}") from exc
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
                        f"[s06] hashing {context} {path.name}: "
                        f"{completed_bytes:,}/{size:,} bytes"
                    )
                    last_report = now
    except OSError as exc:
        raise ContractError(f"cannot fingerprint S06 artifact {path}: {exc}") from exc
    return digest.hexdigest()


def _fingerprint_record(
    path: Path,
    root: Path,
    *,
    logger: LogFn = lambda _message: None,
) -> dict[str, Any]:
    try:
        size = int(path.stat().st_size)
    except OSError as exc:
        raise ContractError(f"cannot stat S06 artifact {path}: {exc}") from exc
    return {
        "path": path.relative_to(root).as_posix(),
        "size_bytes": size,
        "sha256": _sha256(path, logger=logger, context="output"),
    }


def _official_output_files(root: Path, success_name: str) -> tuple[Path, ...]:
    try:
        all_paths = tuple(root.rglob("*"))
        if any(path.is_symlink() for path in all_paths):
            raise ContractError("S06 output tree must not contain symlinks")
        files = tuple(
            sorted(
                (
                    path
                    for path in all_paths
                    if path.is_file() and path.relative_to(root).as_posix() != success_name
                ),
                key=lambda path: path.relative_to(root).as_posix(),
            )
        )
    except OSError as exc:
        raise ContractError(f"cannot enumerate S06 output tree {root}: {exc}") from exc
    if not files:
        raise ContractError("S06 output tree contains no official artifacts")
    return files


def _output_fingerprints(
    root: Path,
    success_name: str,
    *,
    logger: LogFn = lambda _message: None,
) -> list[dict[str, Any]]:
    return [
        _fingerprint_record(path, root, logger=logger)
        for path in _official_output_files(root, success_name)
    ]


def _normalize_input_fingerprints(
    values: Sequence[FileFingerprint],
) -> list[dict[str, Any]]:
    by_path: dict[str, FileFingerprint] = {}
    for value in values:
        previous = by_path.get(value.path)
        if previous is not None and previous != value:
            raise ContractError(f"S06 input fingerprint conflict: {value.path}")
        by_path[value.path] = value
    if not by_path:
        raise ContractError("S06 input fingerprint set cannot be empty")
    return [by_path[path].as_dict() for path in sorted(by_path)]


def _verify_input_fingerprints(
    records: Sequence[Mapping[str, Any]],
    *,
    source_videos: set[str],
    logger: LogFn = lambda _message: None,
) -> None:
    """Recheck regular inputs; avoid hashing each multi-GB video a second time."""

    for record in records:
        if set(record) != {"path", "size_bytes", "sha256"}:
            raise ContractError("S06 input fingerprint record keys differ")
        path = Path(str(record["path"]))
        if str(path) in source_videos:
            try:
                if not path.is_file() or int(path.stat().st_size) != int(
                    record["size_bytes"]
                ):
                    raise ContractError(f"S06 source video changed: {path}")
            except OSError as exc:
                raise ContractError(f"cannot stat S06 source video {path}: {exc}") from exc
            continue
        try:
            size = int(path.stat().st_size)
        except OSError as exc:
            raise ContractError(f"cannot stat S06 input {path}: {exc}") from exc
        digest = _sha256(path, logger=logger, context="input recheck")
        if size != int(record["size_bytes"]) or digest != str(record["sha256"]):
            raise ContractError(f"S06 input changed while exporting: {path}")


def _input_stat_snapshot(
    records: Sequence[Mapping[str, Any]],
    *,
    expected_tokens: Mapping[str, tuple[int, int, int]] | None = None,
) -> dict[Path, tuple[int, int, int]]:
    result: dict[Path, tuple[int, int, int]] = {}
    for record in records:
        path = Path(str(record["path"]))
        try:
            canonical = path.resolve(strict=True)
            stat = path.stat()
        except OSError as exc:
            raise ContractError(f"cannot snapshot S06 input {path}: {exc}") from exc
        if canonical != path or not path.is_file() or path.is_symlink():
            raise ContractError(f"S06 input must be a canonical regular file: {path}")
        token = (int(stat.st_size), int(stat.st_mtime_ns), int(stat.st_ino))
        if token[0] != int(record["size_bytes"]):
            raise ContractError(f"S06 input size differs after validation: {path}")
        if expected_tokens is not None:
            expected = expected_tokens.get(str(path))
            if expected is None or tuple(expected) != token:
                raise ContractError(
                    f"S06 input stat token changed after fingerprinting: {path}"
                )
        result[path] = token
    if expected_tokens is not None and set(expected_tokens) != {
        str(path) for path in result
    }:
        raise ContractError("S06 input stat-token path set differs")
    return result


def _path_stat_token(path: Path) -> tuple[int, int, int]:
    try:
        stat = path.stat()
    except OSError as exc:
        raise ContractError(f"cannot stat S06 input {path}: {exc}") from exc
    return (int(stat.st_size), int(stat.st_mtime_ns), int(stat.st_ino))


def _verify_input_stat_snapshot(
    snapshot: Mapping[Path, tuple[int, int, int]]
) -> None:
    for path, expected in snapshot.items():
        try:
            stat = path.stat()
        except OSError as exc:
            raise ContractError(f"cannot re-stat S06 input {path}: {exc}") from exc
        actual = (int(stat.st_size), int(stat.st_mtime_ns), int(stat.st_ino))
        if actual != expected:
            raise ContractError(f"S06 input changed while exporting: {path}")


def _reject_path_overlap(output_dir: Path, inputs: Sequence[Path]) -> None:
    output = output_dir.resolve()
    for source in inputs:
        resolved = source.resolve()
        if output == resolved or output in resolved.parents or resolved in output.parents:
            raise ContractError(f"S06 output overlaps input: {resolved}")


def _resolve_output_path(path: Path) -> Path:
    try:
        if path.is_symlink():
            raise ContractError(f"S06 output path must not be a symlink: {path}")
        return path.resolve()
    except OSError as exc:
        raise ContractError(f"cannot resolve S06 output path {path}: {exc}") from exc


def _preflight_nvenc(config: S06ExportConfig) -> None:
    """Prove that the approved physical GPU 1 NVENC path is available."""

    if os.environ.get("CUDA_VISIBLE_DEVICES") != config.required_cuda_visible_devices:
        raise ContractError(
            "S06 requires CUDA_VISIBLE_DEVICES exactly equal to "
            f"{config.required_cuda_visible_devices!r}"
        )
    if shutil.which(config.ffmpeg_binary) is None:
        raise ContractError(f"FFmpeg executable not found: {config.ffmpeg_binary}")
    if shutil.which(config.ffprobe_binary) is None:
        raise ContractError(f"ffprobe executable not found: {config.ffprobe_binary}")
    command = [
        config.ffmpeg_binary,
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-f",
        "rawvideo",
        "-pixel_format",
        "bgr24",
        "-video_size",
        f"{config.output_width}x{config.output_height}",
        "-framerate",
        f"{config.fps_numerator}/{config.fps_denominator}",
        "-i",
        "pipe:0",
        "-frames:v",
        "1",
        "-c:v",
        config.codec,
        "-gpu",
        str(config.logical_gpu),
        "-preset",
        config.preset,
        "-tune",
        "hq",
        "-rc:v",
        "vbr",
        "-cq:v",
        str(config.cq),
        "-b:v",
        "0",
        "-pix_fmt",
        config.pixel_format,
        "-f",
        "null",
        "-",
    ]
    try:
        completed = subprocess.run(
            command,
            input=bytes(config.output_width * config.output_height * 3),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            check=False,
            shell=False,
        )
    except OSError as exc:
        raise ContractError(f"cannot start S06 NVENC preflight: {exc}") from exc
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise ContractError(
            f"GPU 1 NVENC preflight failed ({completed.returncode}): "
            f"{detail or 'no FFmpeg diagnostics'}"
        )


def _column_numpy(table: pa.Table, name: str, dtype: Any) -> np.ndarray:
    try:
        return np.asarray(
            table[name].combine_chunks().to_numpy(zero_copy_only=False), dtype=dtype
        )
    except (KeyError, pa.ArrowException, TypeError, ValueError) as exc:
        raise ContractError(f"cannot materialize S06 column {name}: {exc}") from exc


def _maximum_concurrent_stable_tracks(bundle: S06InputBundle) -> int:
    events = [
        event
        for tracklet in bundle.s04.stable_tracklets.values()
        for event in (
            (int(tracklet.start_global_frame), 0),
            (int(tracklet.end_global_frame), 1),
        )
    ]
    if not events:
        raise ContractError("S06 cannot resolve an empty stable-track population")
    concurrent = 0
    maximum = 0
    for _frame, event_kind in sorted(events):
        if event_kind == 0:
            concurrent += 1
            maximum = max(maximum, concurrent)
        else:
            concurrent -= 1
        if concurrent < 0:
            raise ContractError("S06 stable-track interval events are unbalanced")
    if concurrent != 0 or maximum < 1:
        raise ContractError("S06 stable-track interval events are unbalanced")
    return maximum


def _resolve_config_from_bundle(
    config: S06ExportConfig, bundle: S06InputBundle
) -> S06ExportConfig:
    """Bind all null upstream-derived fields to the validated S05 snapshot."""

    candidates = bundle.forced.candidate_edges
    rescue = bundle.forced.rescue_samples
    selected = _column_numpy(candidates, "selected_by_solver", np.bool_)
    prior = _column_numpy(candidates, "prior_global_link", np.bool_)
    rescue_selected = _column_numpy(rescue, "selected_for_descriptor", np.bool_)
    rescue_review_excluded = _column_numpy(rescue, "review_excluded", np.bool_)
    rescue_reasons = np.asarray(rescue["selection_reason"].to_pylist(), dtype=object)
    grade_counts = Counter(
        map(str, bundle.forced.graded_stable_appearance["evidence_grade"].to_pylist())
    )
    stable_count = int(bundle.forced.stable_to_global.num_rows)
    maximum_concurrent = _maximum_concurrent_stable_tracks(bundle)
    observed = {
        "expected_microtrack_count": int(bundle.microtracklets.num_rows),
        "expected_stable_track_count": stable_count,
        "expected_selected_link_count": int(np.count_nonzero(selected)),
        "expected_candidate_edge_count": int(candidates.num_rows),
        "expected_maximum_feasible_link_count": stable_count - maximum_concurrent,
        "expected_max_concurrent_stable_track_count": maximum_concurrent,
        # The full-sequence interval backbone is a minimum-width path cover,
        # so its chain count is exactly the interval-concurrency lower bound.
        "expected_backbone_chain_count": maximum_concurrent,
        "expected_selected_prior_link_count": int(
            np.count_nonzero(selected & prior)
        ),
        "expected_rescue_candidate_count": int(rescue.num_rows),
        "expected_rescue_embedding_count": int(
            np.count_nonzero(rescue_selected)
        ),
        "expected_rescue_stable_track_count": int(
            grade_counts.get("C_REENCODED_DEGRADED", 0)
        ),
        "expected_selected_best_degraded_crop_count": int(
            np.count_nonzero(rescue_reasons == "best_degraded_fallback")
        ),
        "expected_selected_review_excluded_crop_count": int(
            np.count_nonzero(rescue_selected & rescue_review_excluded)
        ),
        "expected_grade_a_clean_count": int(grade_counts.get("A_CLEAN", 0)),
        "expected_grade_b_existing_degraded_count": int(
            grade_counts.get("B_EXISTING_DEGRADED", 0)
        ),
        "expected_grade_c_reencoded_degraded_count": int(
            grade_counts.get("C_REENCODED_DEGRADED", 0)
        ),
    }
    for name, value in observed.items():
        configured = getattr(config, name)
        if configured is not None and configured != value:
            raise ContractError(
                f"S06 configured {name} differs from validated upstream artifacts"
            )
    if sum(grade_counts.values()) != stable_count:
        raise ContractError("S06 appearance grades do not cover all stable tracks")
    return replace(config, **observed)


def _write_csv(path: Path, table: pa.Table, schema: pa.Schema) -> None:
    if not table.schema.equals(schema, check_metadata=False):
        raise ContractError(f"S06 CSV table schema differs for {path.name}")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        pacsv.write_csv(table, path)
    except (OSError, pa.ArrowException) as exc:
        raise ContractError(f"cannot write S06 CSV {path}: {exc}") from exc


def _write_jpeg(path: Path, image: np.ndarray, *, quality: int = 92) -> None:
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
        raise ContractError(f"S06 JPEG image has invalid type: {path}")
    if image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) <= 0:
        raise ContractError(f"S06 JPEG image has invalid geometry: {path}")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        ok = cv2.imwrite(
            str(path), image, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
        )
    except cv2.error as exc:
        raise ContractError(f"cannot encode S06 JPEG {path}: {exc}") from exc
    if not ok or not path.is_file() or path.stat().st_size <= 0:
        raise ContractError(f"cannot publish S06 JPEG {path}")


def _candidate_metrics(table: pa.Table) -> tuple[CandidateMetric, ...]:
    required = (
        "candidate_id",
        "source_stable_id",
        "target_stable_id",
        "source_end_clip_id",
        "target_start_clip_id",
        "source_end_global_frame",
        "target_start_global_frame",
        "source_end_time_sec",
        "target_start_time_sec",
        "temporal_gap_sec",
        "strictly_nonoverlapping",
        "appearance_cosine",
        "source_evidence_grade",
        "target_evidence_grade",
        "selected_by_source_topk",
        "selected_by_target_topk",
        "temporal_backbone",
        "prior_global_link",
        "selected_by_solver",
        "global_link_id",
        "authorization_basis",
        "id_status",
    )
    try:
        columns = {name: table[name].to_pylist() for name in required}
    except (KeyError, pa.ArrowException) as exc:
        raise ContractError(f"cannot build S06 candidate QA view: {exc}") from exc
    return compute_directional_candidate_metrics(columns)


def _valid_detection_crops(table: pa.Table) -> tuple[DetectionCrop, ...]:
    valid = _column_numpy(table, "valid", np.bool_)
    positions = np.flatnonzero(valid)
    det_id = _column_numpy(table, "det_id", np.int64)
    local_frame = _column_numpy(table, "local_frame", np.int64)
    global_frame = _column_numpy(table, "global_frame", np.int64)
    global_time = _column_numpy(table, "global_time_sec", np.float64)
    x1 = _column_numpy(table, "x1", np.float64)
    y1 = _column_numpy(table, "y1", np.float64)
    x2 = _column_numpy(table, "x2", np.float64)
    y2 = _column_numpy(table, "y2", np.float64)
    clips = table["clip_id"].to_pylist()
    stable = table["stable_id"].to_pylist()
    global_ids = table["global_track_id"].to_pylist()
    display_ids = table["display_global_id"].to_pylist()
    statuses = table["id_status"].to_pylist()
    result: list[DetectionCrop] = []
    for position in positions:
        index = int(position)
        if any(
            value is None
            for value in (
                stable[index], global_ids[index], display_ids[index], statuses[index]
            )
        ):
            raise ContractError("S06 valid detection lacks a QA identity mapping")
        result.append(
            DetectionCrop(
                det_id=int(det_id[index]),
                clip_id=str(clips[index]),
                local_frame=int(local_frame[index]),
                global_frame=int(global_frame[index]),
                global_time_sec=float(global_time[index]),
                stable_id=int(stable[index]),
                global_track_id=int(global_ids[index]),
                display_global_id=str(display_ids[index]),
                id_status=str(statuses[index]),
                x1=float(x1[index]),
                y1=float(y1[index]),
                x2=float(x2[index]),
                y2=float(y2[index]),
            )
        )
    return tuple(result)


def _build_crop_and_contact_plans(
    export_table: pa.Table,
    selected_metrics: Sequence[CandidateMetric],
    low_links: Sequence[CandidateMetric],
    config: S06ExportConfig,
) -> tuple[
    tuple[ContactSheetPlan, ...],
    tuple[CropRequest, ...],
    tuple[CropRequest, ...],
    CropRequestIndex,
]:
    detections = _valid_detection_crops(export_table)
    by_global: dict[int, list[DetectionCrop]] = defaultdict(list)
    for detection in detections:
        by_global[detection.global_track_id].append(detection)
    if set(by_global) != set(range(config.expected_global_track_count)):
        raise ContractError("S06 contact-sheet global-ID coverage differs")
    contacts = tuple(
        build_contact_sheet_plan(
            f"{config.display_id_prefix}{global_id + 1:0{config.display_id_width}d}",
            by_global[global_id],
            selected_metrics,
        )
        for global_id in range(config.expected_global_track_count)
    )
    low_requests = build_low_confidence_crop_requests(
        low_links,
        detections,
        crops_per_endpoint=config.crops_per_link_endpoint,
    )
    contact_requests = tuple(
        request for plan in contacts for request in plan.crop_requests()
    )
    index = build_crop_request_index((*low_requests, *contact_requests))
    if index.num_uses != len(low_requests) + len(contact_requests):
        raise ContractError("S06 crop-request use count differs after indexing")
    return contacts, low_requests, contact_requests, index


def _stable_global_mapping(table: pa.Table) -> dict[int, int]:
    stable_ids = _column_numpy(table, "stable_id", np.int64)
    global_ids = _column_numpy(table, "global_track_id", np.int64)
    if len(np.unique(stable_ids)) != len(stable_ids):
        raise ContractError("S06 stable/global mapping repeats stable_id")
    return {
        int(stable_id): int(global_id)
        for stable_id, global_id in zip(stable_ids, global_ids, strict=True)
    }


def _link_global_id(metric: CandidateMetric, by_stable: Mapping[int, int]) -> int:
    try:
        source = int(by_stable[metric.source_stable_id])
        target = int(by_stable[metric.target_stable_id])
    except KeyError as exc:
        raise ContractError(
            f"S06 selected candidate references unknown stable ID: {metric.candidate_id}"
        ) from exc
    if source != target:
        raise ContractError(
            f"S06 selected candidate crosses global IDs: {metric.candidate_id}"
        )
    return source


def _asset_relative_path(det_id: int) -> str:
    return f"low_confidence_assets/det_{det_id:09d}.jpg"


def _requests_by_consumer(
    requests: Sequence[CropRequest],
) -> dict[str, tuple[CropRequest, ...]]:
    grouped: dict[str, list[CropRequest]] = defaultdict(list)
    for request in requests:
        grouped[request.use.consumer_id].append(request)
    return {
        key: tuple(
            sorted(
                values,
                key=lambda request: (request.use.role, request.source.det_id),
            )
        )
        for key, values in grouped.items()
    }


def _format_optional(value: float | None, digits: int = 6) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _render_low_confidence_html(
    path: Path,
    links: Sequence[CandidateMetric],
    low_requests: Sequence[CropRequest],
    stable_to_global: pa.Table,
    config: S06ExportConfig,
    *,
    candidate_count: int,
) -> None:
    if candidate_count < len(links):
        raise ContractError("S06 candidate graph cannot be smaller than its QA subset")
    by_stable = _stable_global_mapping(stable_to_global)
    by_consumer = _requests_by_consumer(low_requests)
    cards: list[str] = []
    for order, metric in enumerate(links, start=1):
        global_id = _link_global_id(metric, by_stable)
        display = f"{config.display_id_prefix}{global_id + 1:0{config.display_id_width}d}"
        consumer = f"low_confidence_link:{metric.candidate_id}"
        requests = by_consumer.get(consumer, ())
        expected_maximum = config.crops_per_link_endpoint * 2
        if not requests or len(requests) > expected_maximum:
            raise ContractError(
                f"S06 low-confidence crop plan differs for {metric.candidate_id}"
            )
        figures: list[str] = []
        for request in requests:
            relative = _asset_relative_path(request.source.det_id)
            figures.append(
                "<figure><img loading=\"lazy\" src=\""
                + html.escape(relative, quote=True)
                + "\" alt=\""
                + html.escape(request.use.label, quote=True)
                + "\"><figcaption>"
                + html.escape(f"{request.use.role} | {request.use.label}")
                + "</figcaption></figure>"
            )
        priority = (
            f"priority: cosine &lt; {config.html_priority_threshold:g}"
            if metric.appearance_cosine < config.html_priority_threshold
            else f"review: cosine &lt; {config.html_maximum_threshold:g}"
        )
        fields = (
            f"cosine={metric.appearance_cosine:.9f}; gap={metric.temporal_gap_sec:.6f}s; "
            f"outgoing rank={metric.outgoing_rank}; incoming rank={metric.incoming_rank}; "
            f"outgoing cosine margin={_format_optional(metric.outgoing_margin)}; "
            f"incoming cosine margin={_format_optional(metric.incoming_margin)}; "
            f"conservative cosine margin={_format_optional(metric.conservative_margin)}"
        )
        provenance = (
            f"grades={metric.source_evidence_grade}→{metric.target_evidence_grade}; "
            f"source_topk={str(metric.selected_by_source_topk).lower()}; "
            f"target_topk={str(metric.selected_by_target_topk).lower()}; "
            f"backbone={str(metric.temporal_backbone).lower()}; "
            f"prior={str(metric.prior_global_link).lower()}"
        )
        cards.append(
            f"<article><h2>{order:03d}. {html.escape(display)} — "
            f"{html.escape(metric.candidate_id)}</h2>"
            f"<p><strong>{priority}</strong> | stable "
            f"{metric.source_stable_id} → {metric.target_stable_id}</p>"
            f"<p>{html.escape(fields)}</p><p>{html.escape(provenance)}</p>"
            f"<div class=\"crops\">{''.join(figures)}</div></article>"
        )
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>S06 low-confidence forced links</title>
<style>
body{{font:15px system-ui,sans-serif;margin:24px;background:#111;color:#eee}}
h1,h2{{color:#9ed0ff}} article{{border-top:1px solid #555;padding:18px 0}}
.notice{{background:#332b12;padding:14px;border:1px solid #8b7430}}
.crops{{display:flex;flex-wrap:wrap;gap:10px}} figure{{margin:0;width:220px}}
img{{width:220px;height:160px;object-fit:contain;background:#222}}
figcaption{{font-size:12px;color:#bbb;overflow-wrap:anywhere}}
</style></head><body>
<h1>S06 forced-provisional low-cosine links ({len(links)})</h1>
<p class="notice">Appearance cosine and cosine margins are not probabilities or
identity certification. Ranks cover only the persisted {candidate_count:,}-edge candidate graph.
All displayed IDs remain forced_provisional.</p>
{''.join(cards)}
</body></html>
"""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(document, encoding="utf-8")
    except OSError as exc:
        raise ContractError(f"cannot write S06 HTML {path}: {exc}") from exc


def _compute_observed_qa(
    metrics: Sequence[CandidateMetric],
    structural: StructuralMetrics,
    config: S06ExportConfig,
) -> dict[str, Any]:
    """Compute QA from this run without requiring a historical reference run."""

    all_scores = np.asarray(
        [metric.appearance_cosine for metric in metrics], dtype=np.float64
    )
    if np.any(~np.isfinite(all_scores)) or np.any(
        (all_scores < -1.0) | (all_scores > 1.0)
    ):
        raise ContractError("S06 candidate cosine values must be finite and in [-1, 1]")
    selected = tuple(metric for metric in metrics if metric.selected_by_solver)
    scores = np.asarray(
        [metric.appearance_cosine for metric in selected], dtype=np.float64
    )
    summary: dict[str, float | None]
    if len(scores):
        summary = {
            "min": float(np.min(scores)),
            "p10": float(np.quantile(scores, 0.1, method="linear")),
            "mean": float(np.mean(scores)),
            "max": float(np.max(scores)),
        }
    else:
        summary = {"min": None, "p10": None, "mean": None, "max": None}
    below = tuple(
        int(np.count_nonzero(scores < threshold))
        for threshold in config.low_confidence_thresholds
    )
    if structural != StructuralMetrics(0, 0, 0, 0):
        raise ContractError(
            f"S06 independently recomputed structure is invalid: {structural}"
        )
    return {
        "reference_mode": config.reference_mode,
        "reference_available": False,
        "reference_comparison_performed": False,
        "selected_link_count": len(selected),
        "appearance_cosine": summary,
        "counts_below_threshold": {
            format(threshold, ".1f"): count
            for threshold, count in zip(
                config.low_confidence_thresholds, below, strict=True
            )
        },
    }


def _provenance_audit(
    bundle: S06InputBundle,
    global_summary: pa.Table,
    config: S06ExportConfig,
) -> tuple[list[str], dict[str, Any]]:
    rescue = bundle.forced.rescue_samples
    selected = _column_numpy(rescue, "selected_for_descriptor", np.bool_)
    overlap = _column_numpy(rescue, "other_bbox_max_iou", np.float64)
    stable_ids = _column_numpy(rescue, "stable_id", np.int64)
    high = selected & (overlap >= config.c_grade_high_overlap_threshold)
    high_count = int(np.count_nonzero(high))
    high_stable_ids = tuple(sorted(set(map(int, stable_ids[high]))))
    exact_duration = _column_numpy(
        global_summary, "duration_visible_sec", np.float64
    )
    upstream_duration = _column_numpy(
        bundle.forced.global_tracks, "duration_visible_sec", np.float64
    )
    if len(exact_duration) != len(upstream_duration):
        raise ContractError("S06 upstream/exact global duration row counts differ")
    mismatch = ~np.isclose(
        exact_duration, upstream_duration, rtol=0.0, atol=1e-12
    )
    duration_mismatch_count = int(np.count_nonzero(mismatch))
    upstream_population = _column_numpy(
        bundle.forced.global_tracks, "population_warning", np.bool_
    )
    warnings = list(bundle.warnings)
    if duration_mismatch_count:
        warnings.append(
            "UPSTREAM_VISIBLE_DURATION_CADENCE_MISMATCH: S05 visible durations "
            "differ from S06's exact 1001/30000 cadence for "
            f"{duration_mismatch_count} global tracks"
        )
    expected_population_warning = global_summary.num_rows > config.population_soft_max
    population_disagreement_count = int(
        np.count_nonzero(upstream_population != expected_population_warning)
    )
    if population_disagreement_count:
        warnings.append(
            "UPSTREAM_POPULATION_WARNING_MISMATCH: S06 derives the population "
            f"warning from the current ID count; disagreeing rows="
            f"{population_disagreement_count}"
        )
    return warnings, {
        "c_grade_rescue_samples_authoritative": True,
        "c_grade_selected_high_overlap_crop_count": high_count,
        "c_grade_high_overlap_stable_ids": list(high_stable_ids),
        "s05_stored_c_grade_iou_fields_trusted": False,
        "s05_visible_duration_mismatch_global_track_count": duration_mismatch_count,
        "s06_visible_duration_frame_cadence": "1001/30000",
        "s05_population_warning_disagreement_count": population_disagreement_count,
        "s05_population_warning_trusted": population_disagreement_count == 0,
    }


def _build_frame_observation_report(
    frames: pa.Table,
    detections: pa.Table,
    clip_order: Sequence[str],
) -> dict[str, Any]:
    """Recompute no-row frame coverage from authoritative S00/S06 tables."""

    ordered_clips = tuple(str(clip) for clip in clip_order)
    if not ordered_clips or len(set(ordered_clips)) != len(ordered_clips):
        raise ContractError("S06 frame observation clip order is invalid")
    frame_clips = np.asarray(frames["clip_id"].to_pylist(), dtype=object)
    frame_local = _column_numpy(frames, "local_frame", np.int64)
    detection_clips = np.asarray(detections["clip_id"].to_pylist(), dtype=object)
    detection_local = _column_numpy(detections, "local_frame", np.int64)
    expected_clip_set = set(ordered_clips)
    if set(map(str, frame_clips)) != expected_clip_set:
        raise ContractError("S06 frame observation frame clip set differs")
    unknown_detection_clips = set(map(str, detection_clips)) - expected_clip_set
    if unknown_detection_clips:
        raise ContractError(
            "S06 frame observation detection clip set differs: "
            f"{sorted(unknown_detection_clips)}"
        )

    per_clip: dict[str, dict[str, Any]] = {}
    observed_total = 0
    unobserved_total = 0
    for clip in ordered_clips:
        frame_positions = np.flatnonzero(frame_clips == clip)
        local_values = frame_local[frame_positions]
        if not np.array_equal(
            np.sort(local_values, kind="stable"),
            np.arange(len(frame_positions), dtype=np.int64),
        ):
            raise ContractError(
                f"S06 frame observation local frame timeline differs for {clip}"
            )
        detection_positions = np.flatnonzero(detection_clips == clip)
        summary = summarize_frame_observation(
            len(frame_positions),
            detection_local[detection_positions],
        )
        per_clip[clip] = summary
        observed_total += int(summary["num_observed_frames"])
        unobserved_total += int(summary["num_unobserved_frames"])

    report: dict[str, Any] = {
        **frame_observation_policy(),
        "num_frames": int(frames.num_rows),
        "num_observed_frames": observed_total,
        "num_unobserved_frames": unobserved_total,
        "per_clip": per_clip,
    }
    if observed_total + unobserved_total != frames.num_rows:
        raise ContractError("S06 frame observation totals differ")
    return report


def _build_qa_metrics(
    bundle: S06InputBundle,
    export_table: pa.Table,
    global_summary: pa.Table,
    candidate_metrics: Sequence[CandidateMetric],
    structural: StructuralMetrics,
    config: S06ExportConfig,
    *,
    config_hash: str,
) -> dict[str, Any]:
    observed_qa = _compute_observed_qa(candidate_metrics, structural, config)
    warnings, provenance = _provenance_audit(bundle, global_summary, config)
    valid = _column_numpy(export_table, "valid", np.bool_)
    clip_values = np.asarray(export_table["clip_id"].to_pylist(), dtype=object)
    total_by_clip = {
        clip: int(np.count_nonzero(clip_values == clip)) for clip in config.clip_order
    }
    valid_by_clip = {
        clip: int(np.count_nonzero((clip_values == clip) & valid))
        for clip in config.clip_order
    }
    invalid_by_clip = {
        clip: total_by_clip[clip] - valid_by_clip[clip]
        for clip in config.clip_order
    }
    frame_clip_values = np.asarray(bundle.frames["clip_id"].to_pylist(), dtype=object)
    frames_by_clip = {
        clip: int(np.count_nonzero(frame_clip_values == clip))
        for clip in config.clip_order
    }
    if sum(frames_by_clip.values()) != bundle.frames.num_rows:
        raise ContractError("S06 frames contain a clip outside the configured order")
    frame_observation = _build_frame_observation_report(
        bundle.frames,
        export_table,
        config.clip_order,
    )
    if frame_observation["num_unobserved_frames"] > 0:
        warnings.append(
            f"{frame_observation['num_unobserved_frames']:,} source-video frames "
            "have no source bbox CSV row; they are treated as unobserved, not "
            "as empty-scene evidence, and are rendered without identity overlays."
        )
    num_valid = int(np.count_nonzero(valid))
    num_invalid = int(np.count_nonzero(~valid))
    num_microtracklets = int(bundle.microtracklets.num_rows)
    num_stable_tracklets = int(bundle.forced.stable_to_global.num_rows)
    num_candidate_edges = len(candidate_metrics)
    num_selected_links = int(observed_qa["selected_link_count"])
    num_global_ids = int(global_summary.num_rows)
    num_rescue_candidates = int(bundle.forced.rescue_samples.num_rows)
    num_rescue_embeddings = int(
        np.count_nonzero(
            _column_numpy(
                bundle.forced.rescue_samples, "selected_for_descriptor", np.bool_
            )
        )
    )
    population_overflow = max(0, num_global_ids - config.population_soft_max)
    unavailable_reason = (
        "No independently confirmed identity ground truth is available for this "
        "forced-provisional export."
    )
    selected_prior = sum(
        metric.selected_by_solver and metric.prior_global_link
        for metric in candidate_metrics
    )
    structural_payload = {
        "same_frame_same_id_violations": structural.same_frame_violation_count,
        "temporal_overlap_violations": structural.temporal_overlap_violation_count,
        "cycles": structural.cycle_count,
        "unassigned_valid_detections": structural.unassigned_valid_detection_count,
    }
    return {
        "schema_version": "1.0",
        "stage": _STAGE,
        "config_hash": config_hash,
        "sequence_id": config.expected_sequence_id,
        "identity_source": config.identity_source,
        "authorization_basis": config.authorization_basis,
        "certification_claimed": False,
        "id_status": config.id_status,
        "qa_reference_mode": config.reference_mode,
        "reference_available": False,
        "reference_comparison_performed": False,
        "num_total_detection_rows": export_table.num_rows,
        "num_valid_detections": num_valid,
        "num_invalid_detections": num_invalid,
        "num_microtracklets": num_microtracklets,
        "num_stable_tracklets": num_stable_tracklets,
        "num_global_ids": num_global_ids,
        "num_confirmed_ids": 0,
        "num_provisional_ids": num_global_ids,
        "num_junk_tracks": None,
        "same_frame_same_id_violations": structural.same_frame_violation_count,
        "temporal_overlap_violations": structural.temporal_overlap_violation_count,
        "cycles": structural.cycle_count,
        "unassigned_valid_detections": structural.unassigned_valid_detection_count,
        "short_gap_recovery_rate": None,
        "long_gap_recovery_rate": None,
        "hard_negative_false_accept_rate": None,
        "unavailable_metrics_not_available_reason": unavailable_reason,
        "population_soft_max": config.population_soft_max,
        "population_overflow": population_overflow,
        "counts": {
            "total_detection_rows": export_table.num_rows,
            "valid_detections": num_valid,
            "invalid_detections": num_invalid,
            "frames": int(bundle.frames.num_rows),
            "frames_by_clip": frames_by_clip,
            "observed_frames": frame_observation["num_observed_frames"],
            "unobserved_frames": frame_observation["num_unobserved_frames"],
            "observed_frames_by_clip": {
                clip: frame_observation["per_clip"][clip]["num_observed_frames"]
                for clip in config.clip_order
            },
            "unobserved_frames_by_clip": {
                clip: frame_observation["per_clip"][clip]["num_unobserved_frames"]
                for clip in config.clip_order
            },
            "detections_by_clip": total_by_clip,
            "valid_detections_by_clip": valid_by_clip,
            "invalid_detections_by_clip": invalid_by_clip,
            "microtracklets": num_microtracklets,
            "stable_tracklets": num_stable_tracklets,
            "candidate_edges": num_candidate_edges,
            "selected_links": num_selected_links,
            "global_ids": num_global_ids,
            "confirmed_ids": 0,
            "forced_provisional_ids": num_global_ids,
            "rescue_candidates": num_rescue_candidates,
            "rescue_embeddings": num_rescue_embeddings,
        },
        "structural_invariants": structural_payload,
        "selected_link_evidence": {
            **observed_qa,
            "score_semantics": "appearance_cosine_not_probability",
            "candidate_rank_scope": config.candidate_rank_scope,
            "candidate_rank_tie_break": config.candidate_rank_tie_break,
            "margin_semantics": config.margin_semantics,
            "selected_prior_link_count": selected_prior,
        },
        "population": {
            "population_soft_max": config.population_soft_max,
            "population_overflow": population_overflow,
            "warning": population_overflow > 0,
            "per_id_non_cow_classification_available": False,
            "note": "Overflow is dataset-level only; no G ID is labelled non-cow.",
        },
        "unavailable_identity_metrics": {
            "num_junk_tracks": None,
            "short_gap_recovery_rate": None,
            "long_gap_recovery_rate": None,
            "hard_negative_false_accept_rate": None,
            "not_available_reason": unavailable_reason,
        },
        "probability_semantics": {
            "assignment_confidence": None,
            "incoming_link_probability": None,
            "outgoing_link_probability": None,
            "not_available_reason": (
                "Forced appearance cosine is not a calibrated probability and is "
                "never converted into confidence."
            ),
        },
        "frame_observation": frame_observation,
        "render_contract": {
            "full_video_content": "valid_bbox_and_display_global_id_only",
            "all_source_video_frames_preserved": True,
            "zero_source_row_frame_policy": (
                "unobserved_passthrough_without_identity_overlay"
            ),
            "zero_source_row_frames_are_empty_scene_evidence": False,
            "invalid_identity_drawn": False,
            "output_size": [config.output_width, config.output_height],
            "fps": f"{config.fps_numerator}/{config.fps_denominator}",
            "codec": config.codec,
            "pixel_format": config.pixel_format,
            "audio": False,
            "autorotate": False,
            "bbox_color_semantics": "deterministic_visual_distinction_only",
        },
        "upstream_provenance_audit": provenance,
        "warnings": warnings,
    }


@dataclass(frozen=True)
class _OverlayIndex:
    positions_and_offsets_by_clip: Mapping[str, tuple[np.ndarray, np.ndarray]]
    det_id: np.ndarray
    local_frame: np.ndarray
    x1: np.ndarray
    y1: np.ndarray
    x2: np.ndarray
    y2: np.ndarray
    global_track_id: tuple[int | None, ...]
    display_global_id: tuple[str | None, ...]

    def detections_at(self, clip_id: str, local_frame: int) -> list[OverlayDetection]:
        try:
            positions, offsets = self.positions_and_offsets_by_clip[clip_id]
        except KeyError as exc:
            raise ContractError(f"S06 overlay index lacks clip {clip_id}") from exc
        start, stop = int(offsets[local_frame]), int(offsets[local_frame + 1])
        output: list[OverlayDetection] = []
        for raw_position in positions[start:stop]:
            position = int(raw_position)
            global_id = self.global_track_id[position]
            display = self.display_global_id[position]
            if global_id is None or display is None:
                raise ContractError("S06 valid overlay row lacks global identity")
            output.append(
                OverlayDetection(
                    det_id=int(self.det_id[position]),
                    valid=True,
                    x1=float(self.x1[position]),
                    y1=float(self.y1[position]),
                    x2=float(self.x2[position]),
                    y2=float(self.y2[position]),
                    global_track_id=int(global_id),
                    display_global_id=str(display),
                )
            )
        return output


def _build_overlay_index(
    export_table: pa.Table,
    config: S06ExportConfig,
    *,
    frame_counts: Mapping[str, int] | None = None,
) -> _OverlayIndex:
    valid = _column_numpy(export_table, "valid", np.bool_)
    clips = np.asarray(export_table["clip_id"].to_pylist(), dtype=object)
    local = _column_numpy(export_table, "local_frame", np.int64)
    det_id = _column_numpy(export_table, "det_id", np.int64)
    by_clip: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    resolved_frame_counts = (
        dict(zip(config.clip_order, config.frame_counts_by_clip, strict=True))
        if frame_counts is None
        else dict(frame_counts)
    )
    if set(resolved_frame_counts) != set(config.clip_order):
        raise ContractError("S06 overlay frame-count clip set differs")
    for clip in config.clip_order:
        frame_count = int(resolved_frame_counts[clip])
        if frame_count < 1:
            raise ContractError(f"S06 overlay frame count is invalid for {clip}")
        positions = np.flatnonzero(valid & (clips == clip)).astype(np.int64)
        frame_values = local[positions]
        if np.any((frame_values < 0) | (frame_values >= frame_count)):
            raise ContractError(f"S06 overlay local frame is out of range for {clip}")
        order = np.lexsort((det_id[positions], frame_values))
        ordered_positions = positions[order]
        counts = np.bincount(frame_values, minlength=frame_count)
        offsets = np.concatenate(
            (np.asarray([0], dtype=np.int64), np.cumsum(counts, dtype=np.int64))
        )
        if int(offsets[-1]) != len(ordered_positions):
            raise ContractError(f"S06 overlay frame offsets differ for {clip}")
        by_clip[clip] = (ordered_positions, offsets)
    return _OverlayIndex(
        positions_and_offsets_by_clip=by_clip,
        det_id=det_id,
        local_frame=local,
        x1=_column_numpy(export_table, "x1", np.float64),
        y1=_column_numpy(export_table, "y1", np.float64),
        x2=_column_numpy(export_table, "x2", np.float64),
        y2=_column_numpy(export_table, "y2", np.float64),
        global_track_id=tuple(export_table["global_track_id"].to_pylist()),
        display_global_id=tuple(export_table["display_global_id"].to_pylist()),
    )


def _frame_pts_by_clip(
    frames: pa.Table, config: S06ExportConfig
) -> dict[str, np.ndarray]:
    clips = np.asarray(frames["clip_id"].to_pylist(), dtype=object)
    local = _column_numpy(frames, "local_frame", np.int64)
    pts = _column_numpy(frames, "pts_sec", np.float64)
    result: dict[str, np.ndarray] = {}
    for clip in config.clip_order:
        positions = np.flatnonzero(clips == clip)
        expected_count = len(positions)
        order = np.argsort(local[positions], kind="stable")
        positions = positions[order]
        if (
            len(positions) != expected_count
            or not np.array_equal(
                local[positions], np.arange(expected_count, dtype=np.int64)
            )
            or not np.all(np.isfinite(pts[positions]))
        ):
            raise ContractError(f"S06 frame cadence/index differs for {clip}")
        result[clip] = np.asarray(pts[positions], dtype=np.float64)
    if sum(len(values) for values in result.values()) != frames.num_rows:
        raise ContractError("S06 frame table contains an unknown clip")
    return result


def _crop_frames_by_clip(
    index: CropRequestIndex, config: S06ExportConfig
) -> dict[str, dict[int, Any]]:
    result: dict[str, dict[int, Any]] = {clip: {} for clip in config.clip_order}
    for frame in index.frames:
        if frame.clip_id not in result:
            raise ContractError(f"S06 crop plan references unknown clip {frame.clip_id}")
        if frame.local_frame in result[frame.clip_id]:
            raise ContractError("S06 crop index repeats a clip/frame")
        result[frame.clip_id][frame.local_frame] = frame
    return result


def _decode_render_and_extract(
    bundle: S06InputBundle,
    export_table: pa.Table,
    crop_index: CropRequestIndex,
    low_requests: Sequence[CropRequest],
    contact_requests: Sequence[CropRequest],
    staging: Path,
    config: S06ExportConfig,
    *,
    logger: LogFn,
) -> dict[int, Path]:
    frame_pts = _frame_pts_by_clip(bundle.frames, config)
    frame_counts = {clip: len(values) for clip, values in frame_pts.items()}
    overlay = _build_overlay_index(
        export_table, config, frame_counts=frame_counts
    )
    crop_frames = _crop_frames_by_clip(crop_index, config)
    low_det_ids = {request.source.det_id for request in low_requests}
    contact_det_ids = {request.source.det_id for request in contact_requests}
    requested_det_ids = low_det_ids | contact_det_ids
    if len(requested_det_ids) != crop_index.num_unique_crops:
        raise ContractError("S06 crop index unique detection count differs")
    contact_cache = staging / ".contact_crop_cache"
    contact_cache.mkdir(parents=False, exist_ok=False)
    low_assets = staging / config.artifacts.low_confidence_assets_dir
    low_assets.mkdir(parents=True, exist_ok=False)
    crop_paths: dict[int, Path] = {}
    processed: set[int] = set()
    videos = {
        clip: staging / relative
        for clip, relative in zip(
            config.clip_order, config.artifacts.videos_by_clip, strict=True
        )
    }
    for clip in config.clip_order:
        expected_frames = frame_counts[clip]
        video_path = bundle.video_paths[clip]
        output_path = videos[clip]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        capture = open_raw_video_capture(video_path)
        last_report = time.monotonic()
        try:
            with NvencVideoWriter(
                output_path,
                width=config.output_width,
                height=config.output_height,
                fps=_EXACT_FPS,
                expected_frame_count=expected_frames,
                ffmpeg_binary=config.ffmpeg_binary,
                preset=config.preset,
                cq=config.cq,
                logical_gpu=config.logical_gpu,
                pixel_format=config.pixel_format,
            ) as writer:
                for local_frame in range(expected_frames):
                    ok, raw = capture.read()
                    if not ok or raw is None:
                        raise ContractError(
                            f"cannot decode S06 source frame {clip}:{local_frame}"
                        )
                    if raw.shape != _RAW_SHAPE or raw.dtype != np.uint8:
                        raise ContractError(
                            f"S06 source frame geometry/type differs at "
                            f"{clip}:{local_frame}"
                        )
                    position_after = int(round(capture.get(cv2.CAP_PROP_POS_FRAMES)))
                    if position_after != local_frame + 1:
                        raise ContractError(
                            f"S06 source frame position differs at {clip}:{local_frame}"
                        )
                    if not hasattr(cv2, "CAP_PROP_PTS"):
                        raise ContractError("OpenCV lacks CAP_PROP_PTS for exact S06 decode")
                    decoded_pts_frame = int(round(capture.get(cv2.CAP_PROP_PTS)))
                    if decoded_pts_frame != local_frame:
                        raise ContractError(
                            f"S06 decoded PTS frame differs at {clip}:{local_frame}"
                        )
                    decoded_msec = float(capture.get(cv2.CAP_PROP_POS_MSEC))
                    expected_msec = float(frame_pts[clip][local_frame]) * 1000.0
                    if not math.isclose(
                        decoded_msec, expected_msec, rel_tol=0.0, abs_tol=0.1
                    ):
                        raise ContractError(
                            f"S06 decoded PTS time differs at {clip}:{local_frame}: "
                            f"{decoded_msec} != {expected_msec} ms"
                        )
                    crop_frame = crop_frames[clip].get(local_frame)
                    if crop_frame is not None:
                        for indexed in crop_frame.crops:
                            det_id = indexed.source.det_id
                            if det_id in processed:
                                raise ContractError(
                                    f"S06 decoded crop repeated det_id {det_id}"
                                )
                            try:
                                crop = extract_detection_crop(raw, indexed.source)
                            except (TypeError, ValueError, cv2.error) as exc:
                                raise ContractError(
                                    f"cannot extract S06 crop det_id={det_id}: {exc}"
                                ) from exc
                            if det_id in low_det_ids:
                                asset_path = low_assets / f"det_{det_id:09d}.jpg"
                                _write_jpeg(asset_path, crop)
                                crop_paths[det_id] = asset_path
                            elif det_id in contact_det_ids:
                                cache_path = contact_cache / f"det_{det_id:09d}.jpg"
                                _write_jpeg(cache_path, crop)
                                crop_paths[det_id] = cache_path
                            else:  # pragma: no cover - guarded by plan count check
                                raise ContractError("S06 crop has no official consumer")
                            processed.add(det_id)
                    try:
                        annotated = render_s06_overlay_frame(
                            raw,
                            overlay.detections_at(clip, local_frame),
                            output_width=config.output_width,
                            output_height=config.output_height,
                        )
                    except (TypeError, ValueError, cv2.error) as exc:
                        raise ContractError(
                            f"cannot render S06 overlay {clip}:{local_frame}: {exc}"
                        ) from exc
                    writer.write(np.ascontiguousarray(annotated))
                    now = time.monotonic()
                    if now - last_report >= config.progress_interval_sec:
                        logger(
                            f"[s06] {clip}: {local_frame + 1:,}/{expected_frames:,} "
                            "frames"
                        )
                        last_report = now
        finally:
            capture.release()
        validate_qa_mp4(
            output_path,
            expected_frame_rate=_EXACT_FPS,
            expected_frame_count=expected_frames,
            ffprobe_binary=config.ffprobe_binary,
        )
        logger(f"[s06] validated full overlay: {output_path}")
    if processed != requested_det_ids or set(crop_paths) != requested_det_ids:
        missing = sorted(requested_det_ids - processed)
        raise ContractError(f"S06 did not decode every planned crop: {missing[:10]}")
    return crop_paths


def _render_contact_sheets(
    plans: Sequence[ContactSheetPlan],
    crop_paths: Mapping[int, Path],
    staging: Path,
    config: S06ExportConfig,
) -> None:
    output = staging / config.artifacts.contact_sheets_dir
    output.mkdir(parents=True, exist_ok=False)
    if len(plans) != config.expected_global_track_count:
        raise ContractError("S06 contact-sheet plan count differs")
    for plan in plans:
        crops: dict[int, np.ndarray] = {}
        for slot in plan.slots:
            if slot.source is None:
                continue
            path = crop_paths.get(slot.source.det_id)
            if path is None:
                raise ContractError(
                    f"S06 contact crop path missing det_id={slot.source.det_id}"
                )
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                raise ContractError(f"cannot read S06 contact crop {path}")
            crops[slot.source.det_id] = image
        try:
            sheet = render_contact_sheet(
                plan,
                crops,
                columns=config.contact_sheet_columns,
            )
        except (TypeError, ValueError, cv2.error) as exc:
            raise ContractError(
                f"cannot render S06 contact sheet {plan.display_global_id}: {exc}"
            ) from exc
        _write_jpeg(output / f"{plan.display_global_id}.jpg", sheet, quality=94)
    expected = {
        f"{config.display_id_prefix}{global_id + 1:0{config.display_id_width}d}.jpg"
        for global_id in range(config.expected_global_track_count)
    }
    actual = {path.name for path in output.glob("*.jpg") if path.is_file()}
    if actual != expected:
        raise ContractError("S06 contact-sheet output set differs")


def _validate_detection_csv(
    path: Path,
    config: S06ExportConfig,
    *,
    expected_total: int | None = None,
    expected_valid: int | None = None,
    expected_invalid: int | None = None,
    expected_global_ids: int | None = None,
) -> None:
    expected_header = DETECTIONS_WITH_GLOBAL_ID_SCHEMA.names
    identity_fields = (
        "global_track_id",
        "global_track_uuid",
        "display_global_id",
        "id_status",
        "identity_basis",
        "micro_id",
        "stable_id",
        "order_in_micro",
        "order_in_stable",
        "order_in_stable_detection",
        "order_in_global_stable",
        "order_in_global_detection",
        "local_purity_score",
        "assignment_confidence",
        "incoming_link_probability",
        "outgoing_link_probability",
    )
    probability_fields = (
        "assignment_confidence",
        "incoming_link_probability",
        "outgoing_link_probability",
    )
    count = 0
    valid_count = 0
    invalid_count = 0
    previous: tuple[int, int] | None = None
    seen_global_ids: set[int] = set()
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != expected_header:
                raise ContractError("completed S06 detection CSV header differs")
            for row in reader:
                count += 1
                try:
                    key = (int(row["clip_order"]), int(row["csv_row_index"]))
                except (KeyError, TypeError, ValueError) as exc:
                    raise ContractError(
                        f"completed S06 detection CSV ordering field is invalid at row {count}"
                    ) from exc
                if previous is not None and key <= previous:
                    raise ContractError("completed S06 detection CSV order is not strict")
                previous = key
                value = row.get("valid", "").casefold()
                if value not in {"true", "false"}:
                    raise ContractError(
                        f"completed S06 detection CSV valid value differs at row {count}"
                    )
                is_valid = value == "true"
                if is_valid:
                    valid_count += 1
                    if row.get("legacy_track_id", "").strip() == "":
                        raise ContractError(
                            f"completed S06 valid row lacks old track ID at row {count}"
                        )
                    if any(row.get(name, "") == "" for name in identity_fields[:-3]):
                        raise ContractError(
                            f"completed S06 valid identity is null at row {count}"
                        )
                    if row["id_status"] != config.id_status:
                        raise ContractError(
                            f"completed S06 valid id_status differs at row {count}"
                        )
                    global_id = int(row["global_track_id"])
                    expected_display = (
                        f"{config.display_id_prefix}"
                        f"{global_id + 1:0{config.display_id_width}d}"
                    )
                    if row["display_global_id"] != expected_display:
                        raise ContractError(
                            f"completed S06 display ID differs at row {count}"
                        )
                    seen_global_ids.add(global_id)
                else:
                    invalid_count += 1
                    if any(row.get(name, "") != "" for name in identity_fields):
                        raise ContractError(
                            f"completed S06 invalid row has identity at row {count}"
                        )
                    if row.get("invalid_reason", "") == "":
                        raise ContractError(
                            f"completed S06 invalid row lacks reason at row {count}"
                        )
                if any(row.get(name, "") != "" for name in probability_fields):
                    raise ContractError(
                        f"completed S06 fabricated probability at row {count}"
                    )
    except ContractError:
        raise
    except (OSError, UnicodeError, csv.Error, KeyError, TypeError, ValueError) as exc:
        raise ContractError(f"cannot validate completed S06 CSV {path}: {exc}") from exc
    total_target = (
        config.expected_total_detection_count
        if expected_total is None
        else expected_total
    )
    valid_target = (
        config.expected_valid_detection_count
        if expected_valid is None
        else expected_valid
    )
    invalid_target = (
        config.expected_invalid_detection_count
        if expected_invalid is None
        else expected_invalid
    )
    global_target = (
        config.expected_global_track_count
        if expected_global_ids is None
        else expected_global_ids
    )
    if (
        count != total_target
        or valid_count != valid_target
        or invalid_count != invalid_target
        or seen_global_ids != set(range(global_target))
    ):
        raise ContractError("completed S06 detection CSV counts/ID coverage differ")


def _validate_global_summary_csv(
    path: Path,
    config: S06ExportConfig,
    *,
    expected_global_ids: int | None = None,
) -> None:
    count = 0
    seen: list[int] = []
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != GLOBAL_TRACK_SUMMARY_SCHEMA.names:
                raise ContractError("completed S06 global summary header differs")
            for row in reader:
                count += 1
                global_id = int(row["global_track_id"])
                seen.append(global_id)
                if (
                    row["id_status"] != config.id_status
                    or row["certification_claimed"].casefold() != "false"
                    or row["population_warning"].casefold() != "true"
                    or int(row["population_overflow"]) != config.population_overflow
                    or any(
                        row[name] != ""
                        for name in (
                            "min_link_probability",
                            "p10_link_probability",
                            "mean_link_probability",
                            "max_link_probability",
                        )
                    )
                ):
                    raise ContractError(
                        f"completed S06 global summary semantics differ at row {count}"
                    )
                expected_display = (
                    f"{config.display_id_prefix}"
                    f"{global_id + 1:0{config.display_id_width}d}"
                )
                if row["display_global_id"] != expected_display:
                    raise ContractError(
                        f"completed S06 global summary display ID differs at row {count}"
                    )
    except ContractError:
        raise
    except (OSError, UnicodeError, csv.Error, TypeError, ValueError) as exc:
        raise ContractError(
            f"cannot validate completed S06 global summary {path}: {exc}"
        ) from exc
    global_target = (
        config.expected_global_track_count
        if expected_global_ids is None
        else expected_global_ids
    )
    if count != global_target or seen != list(range(global_target)):
        raise ContractError("completed S06 global summary ID rows differ")


def _validate_completed_output(
    output_dir: Path,
    marker: Mapping[str, Any],
    *,
    config: S06ExportConfig,
    config_payload: Mapping[str, Any],
    config_hash: str,
    input_fingerprints: Sequence[Mapping[str, Any]],
    logger: LogFn = lambda _message: None,
) -> None:
    required_keys = {
        "schema_version",
        "stage",
        "config_hash",
        "execution_mode",
        "identity_source",
        "authorization_basis",
        "certification_claimed",
        "id_status",
        "stats",
        "input_fingerprints",
        "output_fingerprints",
        "elapsed_sec",
    }
    if set(marker) != required_keys:
        raise ContractError("completed S06 marker keys differ")
    if (
        marker.get("schema_version") != "1.0"
        or marker.get("stage") != _STAGE
        or marker.get("config_hash") != config_hash
        or marker.get("execution_mode") != config.execution_mode
        or marker.get("identity_source") != config.identity_source
        or marker.get("authorization_basis") != config.authorization_basis
        or marker.get("certification_claimed") is not False
        or marker.get("id_status") != config.id_status
        or marker.get("input_fingerprints") != list(input_fingerprints)
        or isinstance(marker.get("elapsed_sec"), bool)
        or not isinstance(marker.get("elapsed_sec"), (int, float))
        or not math.isfinite(float(marker["elapsed_sec"]))
        or float(marker["elapsed_sec"]) < 0.0
    ):
        raise ContractError("completed S06 marker policy/provenance differs")
    records = marker.get("output_fingerprints")
    if not isinstance(records, list) or not records:
        raise ContractError("completed S06 marker lacks output fingerprints")
    names: list[str] = []
    for record in records:
        if (
            not isinstance(record, dict)
            or set(record) != {"path", "size_bytes", "sha256"}
            or not isinstance(record["path"], str)
            or not isinstance(record["size_bytes"], int)
            or record["size_bytes"] < 0
            or not isinstance(record["sha256"], str)
            or len(record["sha256"]) != 64
        ):
            raise ContractError("completed S06 output fingerprint record differs")
        names.append(record["path"])
    if names != sorted(names) or len(names) != len(set(names)):
        raise ContractError("completed S06 output fingerprints are not canonical")
    actual = _official_output_files(output_dir, config.artifacts.success)
    actual_names = [path.relative_to(output_dir).as_posix() for path in actual]
    if actual_names != names:
        raise ContractError("completed S06 artifact tree differs")
    expected_directories = {
        parent.as_posix()
        for name in (*names, config.artifacts.success)
        for parent in Path(name).parents
        if parent != Path(".")
    }
    expected_directories.update(
        {
            config.artifacts.low_confidence_assets_dir,
            config.artifacts.contact_sheets_dir,
            config.artifacts.videos_dir,
        }
    )
    actual_directories = {
        path.relative_to(output_dir).as_posix()
        for path in output_dir.rglob("*")
        if path.is_dir()
    }
    if actual_directories != expected_directories:
        raise ContractError("completed S06 artifact directory tree differs")
    for path, expected in zip(actual, records, strict=True):
        if _fingerprint_record(path, output_dir, logger=logger) != expected:
            raise ContractError(
                f"completed S06 artifact changed: {expected['path']}"
            )

    required_files = {
        config.artifacts.detections_csv,
        config.artifacts.qa_metrics,
        config.artifacts.global_track_summary,
        config.artifacts.low_confidence_links,
        config.artifacts.effective_config,
        *config.artifacts.videos_by_clip,
    }
    if not required_files.issubset(names):
        raise ContractError("completed S06 required artifact set differs")
    contact_prefix = config.artifacts.contact_sheets_dir + "/"
    contacts = {name for name in names if name.startswith(contact_prefix)}
    expected_contacts = {
        f"{contact_prefix}{config.display_id_prefix}"
        f"{global_id + 1:0{config.display_id_width}d}.jpg"
        for global_id in range(config.expected_global_track_count)
    }
    if contacts != expected_contacts:
        raise ContractError("completed S06 contact-sheet artifact set differs")
    assets_prefix = config.artifacts.low_confidence_assets_dir + "/"
    assets = [name for name in names if name.startswith(assets_prefix)]
    if any(
        not Path(name).name.startswith("det_") or Path(name).suffix != ".jpg"
        for name in assets
    ):
        raise ContractError("completed S06 low-confidence asset set differs")
    video_prefix = config.artifacts.videos_dir + "/"
    videos = {name for name in names if name.startswith(video_prefix)}
    if videos != set(config.artifacts.videos_by_clip):
        raise ContractError("completed S06 video artifact set differs")

    effective = _read_json(
        output_dir / config.artifacts.effective_config, "S06 effective config"
    )
    if effective != dict(config_payload):
        raise ContractError("completed S06 effective config differs")
    metrics = _read_json(output_dir / config.artifacts.qa_metrics, "S06 QA metrics")
    if (
        not isinstance(metrics, dict)
        or metrics.get("stage") != _STAGE
        or metrics.get("config_hash") != config_hash
        or metrics.get("certification_claimed") is not False
        or metrics.get("id_status") != config.id_status
        or metrics.get("qa_reference_mode") != config.reference_mode
        or metrics.get("reference_available") is not False
        or metrics.get("reference_comparison_performed") is not False
        or metrics.get("counts", {}).get("global_ids")
        != config.expected_global_track_count
    ):
        raise ContractError("completed S06 QA metrics differ")
    counts = metrics.get("counts")
    evidence = metrics.get("selected_link_evidence")
    if not isinstance(counts, dict) or not isinstance(evidence, dict):
        raise ContractError("completed S06 observed QA counts are missing")
    required_observed_counts = (
        "total_detection_rows",
        "valid_detections",
        "invalid_detections",
        "microtracklets",
        "stable_tracklets",
        "candidate_edges",
        "selected_links",
        "global_ids",
    )
    if any(
        isinstance(counts.get(name), bool)
        or not isinstance(counts.get(name), int)
        or int(counts[name]) < 0
        for name in required_observed_counts
    ):
        raise ContractError("completed S06 observed counts are invalid")
    if counts["total_detection_rows"] != (
        counts["valid_detections"] + counts["invalid_detections"]
    ):
        raise ContractError("completed S06 observed detection partition differs")
    if counts["global_ids"] != config.expected_global_track_count:
        raise ContractError("completed S06 exact global-ID policy differs")
    _validate_detection_csv(
        output_dir / config.artifacts.detections_csv,
        config,
        expected_total=counts["total_detection_rows"],
        expected_valid=counts["valid_detections"],
        expected_invalid=counts["invalid_detections"],
        expected_global_ids=counts["global_ids"],
    )
    _validate_global_summary_csv(
        output_dir / config.artifacts.global_track_summary,
        config,
        expected_global_ids=counts["global_ids"],
    )
    try:
        html_text = (output_dir / config.artifacts.low_confidence_links).read_text(
            encoding="utf-8"
        )
    except (OSError, UnicodeError) as exc:
        raise ContractError(f"cannot read completed S06 HTML: {exc}") from exc
    if (
        "appearance_cosine_not_probability" not in html_text
        and "not probabilities" not in html_text
    ) or "forced_provisional" not in html_text:
        raise ContractError("completed S06 HTML evidence semantics differ")
    frames_by_clip = counts.get("frames_by_clip")
    if (
        not isinstance(frames_by_clip, dict)
        or set(frames_by_clip) != set(config.clip_order)
        or any(
            isinstance(frames_by_clip[clip], bool)
            or not isinstance(frames_by_clip[clip], int)
            or frames_by_clip[clip] < 1
            for clip in config.clip_order
        )
    ):
        raise ContractError("completed S06 observed frame counts are invalid")
    for clip, relative in zip(
        config.clip_order, config.artifacts.videos_by_clip, strict=True
    ):
        validate_qa_mp4(
            output_dir / relative,
            expected_frame_rate=_EXACT_FPS,
            expected_frame_count=frames_by_clip[clip],
            ffprobe_binary=config.ffprobe_binary,
        )
    threshold_counts = evidence.get("counts_below_threshold")
    threshold_key = format(config.html_maximum_threshold, ".1f")
    if not isinstance(threshold_counts, dict) or not isinstance(
        threshold_counts.get(threshold_key), int
    ):
        raise ContractError("completed S06 observed low-confidence count is missing")
    stats = marker.get("stats")
    expected_stats = {
        "num_total_detection_rows": counts.get("total_detection_rows"),
        "num_valid_detections": counts.get("valid_detections"),
        "num_invalid_detections": counts.get("invalid_detections"),
        "num_microtracklets": counts.get("microtracklets"),
        "num_stable_tracklets": counts.get("stable_tracklets"),
        "num_candidate_edges": counts.get("candidate_edges"),
        "num_selected_links": counts.get("selected_links"),
        "num_global_ids": counts.get("global_ids"),
        "num_low_confidence_links": threshold_counts[threshold_key],
        "num_low_confidence_assets": len(assets),
        "num_contact_sheets": config.expected_global_track_count,
        "num_full_overlay_videos": len(config.clip_order),
    }
    if stats != expected_stats:
        raise ContractError("completed S06 marker statistics differ")


def run_s06(
    manifest_path: Path,
    ingest_dir: Path,
    microtrack_dir: Path,
    stable_dir: Path,
    global_dir: Path,
    config_path: Path,
    output_dir: Path,
    *,
    logger: LogFn = log,
) -> dict[str, Any]:
    """Build the fixed 11-video S06 CSV, QA evidence and simple overlays."""

    if not callable(logger):
        raise ContractError("S06 logger must be callable")
    started = time.monotonic()
    input_paths = tuple(
        path.resolve()
        for path in (
            manifest_path,
            ingest_dir,
            microtrack_dir,
            stable_dir,
            global_dir,
            config_path,
        )
    )
    (
        manifest_path,
        ingest_dir,
        microtrack_dir,
        stable_dir,
        global_dir,
        config_path,
    ) = input_paths
    output_dir = _resolve_output_path(output_dir)
    _reject_path_overlap(
        output_dir,
        (
            manifest_path,
            ingest_dir,
            microtrack_dir,
            stable_dir,
            global_dir,
            config_path,
        ),
    )
    config_stat_token = _path_stat_token(config_path)
    config_fingerprint = fingerprint_file(config_path)
    config, config_payload, config_hash = load_s06_export_config(config_path)
    if (
        fingerprint_file(config_path) != config_fingerprint
        or _path_stat_token(config_path) != config_stat_token
    ):
        raise ContractError("S06 config changed while loading")

    success_path = output_dir / config.artifacts.success
    resume_requested = success_path.is_file()
    if output_dir.exists():
        if not output_dir.is_dir() or output_dir.is_symlink():
            raise ContractError(f"S06 output is not a safe directory: {output_dir}")
        if not resume_requested and any(output_dir.iterdir()):
            raise ContractError(
                f"S06 output is non-empty without {config.artifacts.success}: "
                f"{output_dir}"
            )
    if not resume_requested:
        _preflight_nvenc(config)

    logger("[s06] strict-loading S00/S01/S04 and forced S05 snapshot")
    bundle = load_s06_inputs(
        manifest_path,
        ingest_dir,
        microtrack_dir,
        stable_dir,
        global_dir,
        config=config,
        logger=logger,
    )
    config = _resolve_config_from_bundle(config, bundle)
    input_fingerprints = _normalize_input_fingerprints(
        (*bundle.input_fingerprints, config_fingerprint)
    )
    expected_stat_tokens = dict(bundle.input_stat_tokens)
    expected_stat_tokens[str(config_path)] = config_stat_token
    input_stats = _input_stat_snapshot(
        input_fingerprints, expected_tokens=expected_stat_tokens
    )

    if resume_requested:
        marker = _read_json(success_path, "S06 success marker")
        if not isinstance(marker, Mapping):
            raise ContractError("S06 success marker must be an object")
        _validate_completed_output(
            output_dir,
            marker,
            config=config,
            config_payload=config_payload,
            config_hash=config_hash,
            input_fingerprints=input_fingerprints,
            logger=logger,
        )
        _verify_input_stat_snapshot(input_stats)
        logger(f"[s06] already complete and revalidated: {success_path}")
        return dict(marker)

    logger("[s06] building deterministic all-row identity CSV join")
    export_table = build_detection_export_table(
        bundle.detections,
        bundle.forced.det_to_global,
        bundle.microtracklets,
        clip_order=config.clip_order,
    )
    structural = recompute_structural_metrics(
        export_table,
        bundle.forced.stable_to_global,
        bundle.forced.candidate_edges,
    )
    candidate_metrics = _candidate_metrics(bundle.forced.candidate_edges)
    selected_metrics = tuple(
        metric for metric in candidate_metrics if metric.selected_by_solver
    )
    low_links = select_low_confidence_links(
        candidate_metrics, threshold=config.html_maximum_threshold
    )
    logger("[s06] building global summary and single-pass crop plan")
    global_summary = build_global_track_summary(
        export_table,
        bundle.forced.stable_to_global,
        bundle.forced.global_tracks,
        bundle.forced.candidate_edges,
        bundle.forced.graded_stable_appearance,
        config=config,
    )
    contacts, low_requests, contact_requests, crop_index = (
        _build_crop_and_contact_plans(
            export_table,
            selected_metrics,
            low_links,
            config,
        )
    )
    qa_metrics = _build_qa_metrics(
        bundle,
        export_table,
        global_summary,
        candidate_metrics,
        structural,
        config,
        config_hash=config_hash,
    )

    try:
        output_dir.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ContractError(f"cannot create S06 output parent: {exc}") from exc
    staging = output_dir.parent / f".{output_dir.name}.staging-{os.getpid()}"
    if staging.exists():
        raise ContractError(f"S06 staging directory already exists: {staging}")
    try:
        staging.mkdir(parents=False)
    except OSError as exc:
        raise ContractError(f"cannot create S06 staging directory: {exc}") from exc
    try:
        _write_csv(
            staging / config.artifacts.detections_csv,
            export_table,
            DETECTIONS_WITH_GLOBAL_ID_SCHEMA,
        )
        _write_csv(
            staging / config.artifacts.global_track_summary,
            global_summary,
            GLOBAL_TRACK_SUMMARY_SCHEMA,
        )
        _write_json(staging / config.artifacts.qa_metrics, qa_metrics)
        # The byte-derived config hash and effective-config artifact describe
        # the operator-authored null-sentinel contract.  Resolved observations
        # are reported in QA/marker counts, not written back into that payload.
        _write_json(staging / config.artifacts.effective_config, config_payload)
        crop_paths = _decode_render_and_extract(
            bundle,
            export_table,
            crop_index,
            low_requests,
            contact_requests,
            staging,
            config,
            logger=logger,
        )
        _render_contact_sheets(contacts, crop_paths, staging, config)
        _render_low_confidence_html(
            staging / config.artifacts.low_confidence_links,
            low_links,
            low_requests,
            bundle.forced.stable_to_global,
            config,
            candidate_count=len(candidate_metrics),
        )
        cache = staging / ".contact_crop_cache"
        shutil.rmtree(cache)
        _verify_input_stat_snapshot(input_stats)
        _verify_input_fingerprints(
            input_fingerprints,
            source_videos={str(path) for path in bundle.video_paths.values()},
            logger=logger,
        )
        _verify_input_stat_snapshot(input_stats)
        output_fingerprints = _output_fingerprints(
            staging, config.artifacts.success, logger=logger
        )
        asset_prefix = config.artifacts.low_confidence_assets_dir + "/"
        num_assets = sum(
            str(record["path"]).startswith(asset_prefix)
            for record in output_fingerprints
        )
        observed_counts = qa_metrics.get("counts")
        if not isinstance(observed_counts, dict):
            raise ContractError("S06 observed QA counts are missing before commit")
        marker = {
            "schema_version": "1.0",
            "stage": _STAGE,
            "config_hash": config_hash,
            "execution_mode": config.execution_mode,
            "identity_source": config.identity_source,
            "authorization_basis": config.authorization_basis,
            "certification_claimed": False,
            "id_status": config.id_status,
            "stats": {
                "num_total_detection_rows": observed_counts.get(
                    "total_detection_rows"
                ),
                "num_valid_detections": observed_counts.get("valid_detections"),
                "num_invalid_detections": observed_counts.get("invalid_detections"),
                "num_microtracklets": observed_counts.get("microtracklets"),
                "num_stable_tracklets": observed_counts.get("stable_tracklets"),
                "num_candidate_edges": observed_counts.get("candidate_edges"),
                "num_selected_links": observed_counts.get("selected_links"),
                "num_global_ids": observed_counts.get("global_ids"),
                "num_low_confidence_links": len(low_links),
                "num_low_confidence_assets": num_assets,
                "num_contact_sheets": len(contacts),
                "num_full_overlay_videos": len(config.clip_order),
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
            config_payload=config_payload,
            config_hash=config_hash,
            input_fingerprints=input_fingerprints,
            logger=logger,
        )
        _verify_input_stat_snapshot(input_stats)
        if output_dir.exists():
            try:
                output_dir.rmdir()
            except OSError as exc:
                raise ContractError(
                    f"cannot remove empty S06 output directory: {exc}"
                ) from exc
        try:
            os.replace(staging, output_dir)
        except OSError as exc:
            raise ContractError(f"cannot atomically commit S06 output: {exc}") from exc
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    logger(
        f"[s06] complete: {int(marker['stats']['num_total_detection_rows']):,} CSV rows, "
        f"{int(marker['stats']['num_global_ids'])} forced-provisional IDs, "
        f"{len(config.clip_order)} full overlays; output={output_dir}"
    )
    return marker


__all__ = ["run_s06"]
