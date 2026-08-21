from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from cowtrack.bbox import normalize_xywh, stable_det_id, suppress_duplicate_boxes
from cowtrack.config import (
    ClipManifest,
    ContractError,
    IngestConfig,
    load_config,
    load_manifest,
)
from cowtrack.frame_observation import (
    frame_observation_policy,
    summarize_frame_observation,
)
from cowtrack.schemas.detections import (
    BBoxQAFlag,
    DETECTIONS_SCHEMA,
    INVALID_BBOX_MASK,
    QA_FLAG_DEFINITIONS,
)
from cowtrack.schemas.frames import FRAMES_SCHEMA
from cowtrack.video import (
    PacketTimeline,
    VideoStreamMetadata,
    open_raw_video_capture,
    probe_video_stream,
    read_packet_timeline,
    verify_raw_video_geometry,
)


def log(message: str) -> None:
    print(message, flush=True)


@dataclass
class ClipRuntime:
    manifest: ClipManifest
    metadata: VideoStreamMetadata
    packets: PacketTimeline
    global_frame_offset: int
    global_time_offset: Fraction
    frame_step: Fraction
    pts_seconds: np.ndarray
    global_times: np.ndarray
    frames_with_boxes: set[int] = field(default_factory=set)


def _atomic_write_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _file_fingerprint(
    path: Path,
    *,
    progress_interval_sec: float,
    logger: Callable[[str], None],
) -> dict[str, Any]:
    before = path.stat()
    digest = hashlib.sha256()
    processed = 0
    last_report = time.monotonic()
    with path.open("rb") as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
            processed += len(block)
            now = time.monotonic()
            if now - last_report >= progress_interval_sec:
                percent = 100.0 * processed / before.st_size if before.st_size else 100.0
                logger(
                    f"[s00] fingerprint {path.name}: {processed / (1024**3):.2f}/"
                    f"{before.st_size / (1024**3):.2f} GiB ({percent:.1f}%)"
                )
                last_report = now
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ContractError(f"input changed while fingerprinting: {path}")
    return {
        "path": str(path.resolve()),
        "size_bytes": int(before.st_size),
        "mtime_ns": int(before.st_mtime_ns),
        "sha256": digest.hexdigest(),
    }


def _verify_inputs_unchanged(fingerprints: list[dict[str, Any]]) -> None:
    for fingerprint in fingerprints:
        path = Path(fingerprint["path"])
        current = path.stat()
        if current.st_size != fingerprint["size_bytes"] or current.st_mtime_ns != fingerprint[
            "mtime_ns"
        ]:
            raise ContractError(f"input changed during S00: {path}")


def _acceptance_int(config: IngestConfig, key: str) -> int:
    if key not in config.acceptance:
        raise ContractError(f"missing acceptance.{key}")
    return int(config.acceptance[key])


def _build_clip_runtimes(
    clips: list[ClipManifest], config: IngestConfig, logger: Callable[[str], None]
) -> tuple[list[ClipRuntime], Fraction]:
    runtimes: list[ClipRuntime] = []
    frame_offset = 0
    time_offset = Fraction(0, 1)
    expected_step: Fraction | None = None
    expected_clip_frames = config.acceptance.get("expected_clip_frames")
    if not isinstance(expected_clip_frames, dict):
        raise ContractError("acceptance.expected_clip_frames must be a mapping")

    for clip in clips:
        logger(f"[s00] probing {clip.clip_id}: {clip.video_path}")
        metadata = probe_video_stream(clip.video_path, config.ffprobe_binary)
        expected_width = _acceptance_int(config, "expected_width")
        expected_height = _acceptance_int(config, "expected_height")
        if (metadata.width, metadata.height) != (expected_width, expected_height):
            raise ContractError(
                f"unexpected encoded geometry for {clip.clip_id}: "
                f"{(metadata.width, metadata.height)} != {(expected_width, expected_height)}"
            )
        expected_frames = int(expected_clip_frames.get(clip.clip_id, -1))
        if metadata.num_frames != expected_frames:
            raise ContractError(
                f"unexpected frame count for {clip.clip_id}: "
                f"{metadata.num_frames} != {expected_frames}"
            )
        if metadata.time_base != Fraction(
            1, _acceptance_int(config, "expected_time_base_denominator")
        ):
            raise ContractError(f"unexpected time_base for {clip.clip_id}: {metadata.time_base}")
        verify_raw_video_geometry(clip.video_path, metadata)
        packets = read_packet_timeline(
            clip.video_path,
            metadata,
            config.ffprobe_binary,
            config.progress_interval_sec,
            logger,
        )
        unique_durations = set(packets.duration_ticks)
        expected_duration_ticks = _acceptance_int(
            config, "expected_frame_duration_ticks"
        )
        if unique_durations != {expected_duration_ticks}:
            raise ContractError(
                f"unexpected packet durations for {clip.clip_id}: {sorted(unique_durations)}"
            )
        frame_step = expected_duration_ticks * metadata.time_base
        if expected_step is None:
            expected_step = frame_step
        elif frame_step != expected_step:
            raise ContractError(
                f"clip frame cadence mismatch: {frame_step} != {expected_step}"
            )

        pts_seconds = np.fromiter(
            (float(pts * metadata.time_base) for pts in packets.pts_ticks),
            dtype=np.float64,
            count=metadata.num_frames,
        )
        global_times = np.fromiter(
            (
                float(time_offset + (pts - metadata.start_pts) * metadata.time_base)
                for pts in packets.pts_ticks
            ),
            dtype=np.float64,
            count=metadata.num_frames,
        )
        if metadata.num_frames > 1 and not np.all(np.diff(global_times) > 0.0):
            raise ContractError(f"non-monotonic global time inside {clip.clip_id}")
        runtime = ClipRuntime(
            manifest=clip,
            metadata=metadata,
            packets=packets,
            global_frame_offset=frame_offset,
            global_time_offset=time_offset,
            frame_step=frame_step,
            pts_seconds=pts_seconds,
            global_times=global_times,
        )
        runtimes.append(runtime)
        frame_offset += metadata.num_frames
        time_offset += metadata.duration_ticks * metadata.time_base

    if frame_offset != _acceptance_int(config, "expected_num_frames"):
        raise ContractError(
            f"sequence frame count mismatch: {frame_offset} != "
            f"{_acceptance_int(config, 'expected_num_frames')}"
        )
    expected_end = float(config.acceptance["expected_sequence_end_time_sec"])
    if not math.isclose(float(time_offset), expected_end, rel_tol=0.0, abs_tol=1e-9):
        raise ContractError(f"sequence duration mismatch: {float(time_offset)} != {expected_end}")
    for previous, current in zip(runtimes, runtimes[1:]):
        expected_start = previous.global_times[-1] + float(previous.frame_step)
        if not math.isclose(
            float(current.global_times[0]), expected_start, rel_tol=0.0, abs_tol=1e-9
        ):
            raise ContractError(
                f"non-contiguous clip boundary: {previous.manifest.clip_id} -> "
                f"{current.manifest.clip_id}"
            )
    return runtimes, time_offset


def _write_frames(
    runtimes: list[ClipRuntime], output_path: Path, config: IngestConfig
) -> int:
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    writer = pq.ParquetWriter(
        temporary,
        FRAMES_SCHEMA,
        compression=config.parquet_compression,
        use_dictionary=["sequence_id", "clip_id"],
    )
    total = 0
    try:
        for runtime in runtimes:
            count = runtime.metadata.num_frames
            local_frames = np.arange(count, dtype=np.int32)
            table = pa.Table.from_arrays(
                [
                    pa.array([runtime.manifest.sequence_id] * count, type=pa.string()),
                    pa.array([runtime.manifest.clip_id] * count, type=pa.string()),
                    pa.array(
                        np.full(count, runtime.manifest.clip_order, dtype=np.int16),
                        type=pa.int16(),
                    ),
                    pa.array(local_frames, type=pa.int32()),
                    pa.array(
                        local_frames.astype(np.int64) + runtime.global_frame_offset,
                        type=pa.int64(),
                    ),
                    pa.array(runtime.pts_seconds, type=pa.float64()),
                    pa.array(runtime.global_times, type=pa.float64()),
                    pa.array(
                        np.full(count, runtime.metadata.width, dtype=np.int32),
                        type=pa.int32(),
                    ),
                    pa.array(
                        np.full(count, runtime.metadata.height, dtype=np.int32),
                        type=pa.int32(),
                    ),
                ],
                schema=FRAMES_SCHEMA,
            )
            writer.write_table(table)
            total += count
    finally:
        writer.close()
    os.replace(temporary, output_path)
    return total


def _parse_float(value: str, *, column: str, row_index: int, csv_path: Path) -> float:
    try:
        return float(value)
    except ValueError as exc:
        raise ContractError(
            f"cannot parse {column} at data row {row_index} in {csv_path}: {value!r}"
        ) from exc


def _parse_frame(value: str, *, row_index: int, csv_path: Path) -> int:
    try:
        return int(value)
    except ValueError as exc:
        raise ContractError(
            f"frame is not an integer at data row {row_index} in {csv_path}: {value!r}"
        ) from exc


def _flush_detection_batch(
    writer: pq.ParquetWriter, batch: list[dict[str, Any]]
) -> None:
    if not batch:
        return
    writer.write_table(pa.Table.from_pylist(batch, schema=DETECTIONS_SCHEMA))
    batch.clear()


def _write_detections(
    runtimes: list[ClipRuntime],
    output_path: Path,
    config: IngestConfig,
    logger: Callable[[str], None],
) -> dict[str, Any]:
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    writer = pq.ParquetWriter(
        temporary,
        DETECTIONS_SCHEMA,
        compression=config.parquet_compression,
        use_dictionary=["sequence_id", "clip_id", "legacy_track_id"],
    )
    seen_det_ids: set[int] = set()
    batch: list[dict[str, Any]] = []
    totals: Counter[str] = Counter()
    total_flag_counts: Counter[str] = Counter()
    per_clip: list[dict[str, Any]] = []

    try:
        for runtime in runtimes:
            clip = runtime.manifest
            csv_config = config.input_csv
            required_columns = [
                csv_config.video_column,
                csv_config.frame_column,
                csv_config.x_column,
                csv_config.y_column,
                csv_config.width_column,
                csv_config.height_column,
                csv_config.confidence_column,
                csv_config.legacy_track_id_column,
            ]
            clip_counts: Counter[str] = Counter()
            clip_flag_counts: Counter[str] = Counter()
            current_frame: int | None = None
            frame_rows: list[dict[str, Any]] = []
            previous_frame = -1
            last_report = time.monotonic()

            def finalize_frame() -> None:
                nonlocal frame_rows
                if not frame_rows:
                    return
                suppressed = suppress_duplicate_boxes(
                    frame_rows,
                    iou_threshold=config.duplicate_iou_threshold,
                    minimum_area_similarity=config.duplicate_area_similarity_min,
                )
                clip_counts["num_duplicate_boxes_removed"] += suppressed
                for record in frame_rows:
                    flags = int(record["qa_flags"])
                    if bool(record["valid"]) != ((flags & INVALID_BBOX_MASK) == 0):
                        raise ContractError("valid/qa_flags invariant failed before Parquet write")
                    clip_counts["num_valid_boxes" if record["valid"] else "num_invalid_boxes"] += 1
                    for flag in BBoxQAFlag:
                        if flags & int(flag):
                            clip_flag_counts[flag.name] += 1
                    batch.append(record)
                runtime.frames_with_boxes.add(int(frame_rows[0]["local_frame"]))
                frame_rows = []
                if len(batch) >= config.parquet_batch_rows:
                    _flush_detection_batch(writer, batch)

            logger(f"[s00] ingesting bbox CSV for {clip.clip_id}: {clip.bbox_csv_path}")
            with clip.bbox_csv_path.open("r", newline="", encoding="utf-8-sig") as handle:
                reader = csv.reader(handle)
                try:
                    header = next(reader)
                except StopIteration as exc:
                    raise ContractError(f"empty bbox CSV: {clip.bbox_csv_path}") from exc
                if len(header) != len(set(header)):
                    raise ContractError(f"duplicate CSV header names: {clip.bbox_csv_path}")
                missing = [name for name in required_columns if name not in header]
                if missing:
                    raise ContractError(
                        f"bbox CSV missing required columns {missing}: {clip.bbox_csv_path}"
                    )
                column = {name: header.index(name) for name in required_columns}

                for csv_row_index, row in enumerate(reader):
                    if len(row) != len(header):
                        raise ContractError(
                            f"CSV column count mismatch at data row {csv_row_index} in "
                            f"{clip.bbox_csv_path}: {len(row)} != {len(header)}"
                        )
                    video_name = row[column[csv_config.video_column]]
                    if video_name != clip.video_path.name:
                        raise ContractError(
                            f"video/CSV bijection failed at data row {csv_row_index}: "
                            f"{video_name!r} != {clip.video_path.name!r}"
                        )
                    source_frame = _parse_frame(
                        row[column[csv_config.frame_column]],
                        row_index=csv_row_index,
                        csv_path=clip.bbox_csv_path,
                    )
                    local_frame = source_frame - clip.frame_index_base
                    if not 0 <= local_frame < runtime.metadata.num_frames:
                        raise ContractError(
                            f"frame outside video at data row {csv_row_index}: "
                            f"{local_frame} not in [0, {runtime.metadata.num_frames})"
                        )
                    if local_frame < previous_frame:
                        raise ContractError(
                            f"CSV frames must be non-decreasing: {local_frame} after "
                            f"{previous_frame} in {clip.bbox_csv_path}"
                        )
                    if current_frame is None:
                        current_frame = local_frame
                    elif local_frame != current_frame:
                        finalize_frame()
                        current_frame = local_frame
                    previous_frame = local_frame

                    x = _parse_float(
                        row[column[csv_config.x_column]],
                        column=csv_config.x_column,
                        row_index=csv_row_index,
                        csv_path=clip.bbox_csv_path,
                    )
                    y = _parse_float(
                        row[column[csv_config.y_column]],
                        column=csv_config.y_column,
                        row_index=csv_row_index,
                        csv_path=clip.bbox_csv_path,
                    )
                    width = _parse_float(
                        row[column[csv_config.width_column]],
                        column=csv_config.width_column,
                        row_index=csv_row_index,
                        csv_path=clip.bbox_csv_path,
                    )
                    height = _parse_float(
                        row[column[csv_config.height_column]],
                        column=csv_config.height_column,
                        row_index=csv_row_index,
                        csv_path=clip.bbox_csv_path,
                    )
                    geometry = normalize_xywh(
                        x,
                        y,
                        width,
                        height,
                        frame_width=runtime.metadata.width,
                        frame_height=runtime.metadata.height,
                        minimum_area=config.minimum_bbox_area_pixels,
                        minimum_retained_fraction=config.minimum_retained_area_fraction,
                    )
                    confidence_text = row[column[csv_config.confidence_column]].strip()
                    confidence: float | None
                    qa_flags = int(geometry.qa_flags)
                    if confidence_text == "":
                        confidence = None
                    else:
                        confidence_value = _parse_float(
                            confidence_text,
                            column=csv_config.confidence_column,
                            row_index=csv_row_index,
                            csv_path=clip.bbox_csv_path,
                        )
                        if not math.isfinite(confidence_value) or not 0.0 <= confidence_value <= 1.0:
                            confidence = None
                            qa_flags |= int(BBoxQAFlag.INVALID_CONFIDENCE)
                        else:
                            confidence = confidence_value
                    legacy_text = row[column[csv_config.legacy_track_id_column]].strip()
                    legacy_track_id = legacy_text if legacy_text else None
                    det_id = stable_det_id(
                        clip.sequence_id, clip.clip_id, local_frame, csv_row_index
                    )
                    if det_id in seen_det_ids:
                        raise ContractError(f"det_id collision: {det_id}")
                    seen_det_ids.add(det_id)

                    frame_rows.append(
                        {
                            "det_id": det_id,
                            "sequence_id": clip.sequence_id,
                            "clip_id": clip.clip_id,
                            "local_frame": local_frame,
                            "global_frame": runtime.global_frame_offset + local_frame,
                            "global_time_sec": float(runtime.global_times[local_frame]),
                            "x1": geometry.x1,
                            "y1": geometry.y1,
                            "x2": geometry.x2,
                            "y2": geometry.y2,
                            "cx_norm": geometry.cx_norm,
                            "cy_norm": geometry.cy_norm,
                            "w_norm": geometry.w_norm,
                            "h_norm": geometry.h_norm,
                            "area_norm": geometry.area_norm,
                            "bbox_confidence": confidence,
                            "legacy_track_id": legacy_track_id,
                            "csv_row_index": csv_row_index,
                            "valid": geometry.valid,
                            "qa_flags": qa_flags,
                        }
                    )
                    clip_counts["num_input_boxes"] += 1
                    now = time.monotonic()
                    if now - last_report >= config.progress_interval_sec:
                        logger(
                            f"[s00] CSV {clip.clip_id}: {clip_counts['num_input_boxes']:,} rows, "
                            f"frame={local_frame:,}"
                        )
                        last_report = now
            finalize_frame()
            if clip_counts["num_input_boxes"] == 0:
                raise ContractError(f"bbox CSV has no data rows: {clip.bbox_csv_path}")
            for key, value in clip_counts.items():
                totals[key] += value
            for key, value in clip_flag_counts.items():
                total_flag_counts[key] += value
            frame_observation = summarize_frame_observation(
                runtime.metadata.num_frames,
                runtime.frames_with_boxes,
            )
            per_clip.append(
                {
                    "clip_id": clip.clip_id,
                    "num_input_boxes": clip_counts["num_input_boxes"],
                    "num_valid_boxes": clip_counts["num_valid_boxes"],
                    "num_invalid_boxes": clip_counts["num_invalid_boxes"],
                    "num_duplicate_boxes_removed": clip_counts[
                        "num_duplicate_boxes_removed"
                    ],
                    "num_clamped_boxes": clip_flag_counts["CLAMPED_TO_IMAGE"],
                    "num_frames_with_boxes": len(runtime.frames_with_boxes),
                    "num_frames_without_boxes": runtime.metadata.num_frames
                    - len(runtime.frames_with_boxes),
                    "frame_observation": frame_observation,
                    "qa_flag_counts": dict(sorted(clip_flag_counts.items())),
                }
            )
            logger(
                f"[s00] CSV {clip.clip_id}: rows={clip_counts['num_input_boxes']:,}, "
                f"valid={clip_counts['num_valid_boxes']:,}, "
                f"duplicates={clip_counts['num_duplicate_boxes_removed']:,}"
            )
        _flush_detection_batch(writer, batch)
    finally:
        writer.close()
    os.replace(temporary, output_path)

    if len(seen_det_ids) != totals["num_input_boxes"]:
        raise ContractError("det_id uniqueness/count invariant failed")
    return {
        "num_input_boxes": totals["num_input_boxes"],
        "num_valid_boxes": totals["num_valid_boxes"],
        "num_invalid_boxes": totals["num_invalid_boxes"],
        "num_duplicate_boxes_removed": totals["num_duplicate_boxes_removed"],
        "num_clamped_boxes": total_flag_counts["CLAMPED_TO_IMAGE"],
        "qa_flag_counts": dict(sorted(total_flag_counts.items())),
        "per_clip": per_clip,
    }


def _render_overlay_samples(
    runtimes: list[ClipRuntime],
    detections_path: Path,
    output_dir: Path,
    config: IngestConfig,
) -> dict[str, Any]:
    overlay_dir = output_dir / "qa" / "overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, Any]] = []
    for runtime in runtimes:
        count = runtime.metadata.num_frames
        sample_frames = [0, (count - 1) // 2, count - 1]
        table = pq.read_table(
            detections_path,
            filters=[
                ("clip_id", "=", runtime.manifest.clip_id),
                ("local_frame", "in", sample_frames),
            ],
        )
        records_by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for record in table.to_pylist():
            records_by_frame[int(record["local_frame"])].append(record)

        capture = open_raw_video_capture(runtime.manifest.video_path)
        try:
            for region, local_frame in zip(("front", "middle", "back"), sample_frames):
                if not capture.set(cv2.CAP_PROP_POS_FRAMES, int(local_frame)):
                    raise ContractError(
                        f"cannot seek overlay frame {local_frame}: {runtime.manifest.video_path}"
                    )
                ok, frame = capture.read()
                if not ok or frame is None:
                    raise ContractError(
                        f"cannot decode overlay frame {local_frame}: {runtime.manifest.video_path}"
                    )
                frame_height, frame_width = frame.shape[:2]
                if (frame_width, frame_height) != (
                    runtime.metadata.width,
                    runtime.metadata.height,
                ):
                    raise ContractError(
                        f"overlay frame was rotated/resized: {(frame_width, frame_height)}"
                    )
                position_after = int(round(capture.get(cv2.CAP_PROP_POS_FRAMES)))
                if position_after != local_frame + 1:
                    raise ContractError(
                        f"OpenCV seek landed on wrong frame: {position_after - 1} != {local_frame}"
                    )
                if not hasattr(cv2, "CAP_PROP_PTS"):
                    raise ContractError("OpenCV lacks CAP_PROP_PTS for overlay identity check")
                decoded_pts_frame = int(round(capture.get(cv2.CAP_PROP_PTS)))
                if decoded_pts_frame != local_frame:
                    raise ContractError(
                        f"OpenCV decoded PTS frame mismatch: {decoded_pts_frame} != {local_frame}"
                    )
                decoded_pts_msec = float(capture.get(cv2.CAP_PROP_POS_MSEC))
                expected_pts_msec = float(runtime.pts_seconds[local_frame]) * 1000.0
                if not math.isclose(
                    decoded_pts_msec, expected_pts_msec, rel_tol=0.0, abs_tol=0.1
                ):
                    raise ContractError(
                        f"OpenCV decoded PTS time mismatch: {decoded_pts_msec} != "
                        f"{expected_pts_msec} ms"
                    )
                frame_records = records_by_frame.get(local_frame, [])
                valid_count = 0
                invalid_count = 0
                duplicate_count = 0
                for record in frame_records:
                    flags = int(record["qa_flags"])
                    if bool(record["valid"]):
                        color = (0, 255, 0)
                        valid_count += 1
                    elif flags & int(BBoxQAFlag.HIGH_IOU_DUPLICATE):
                        color = (0, 165, 255)
                        invalid_count += 1
                        duplicate_count += 1
                    else:
                        color = (0, 0, 255)
                        invalid_count += 1
                    values = [record[key] for key in ("x1", "y1", "x2", "y2")]
                    if not all(math.isfinite(float(value)) for value in values):
                        continue
                    x1 = min(max(int(round(float(record["x1"]))), 0), frame_width - 1)
                    y1 = min(max(int(round(float(record["y1"]))), 0), frame_height - 1)
                    x2 = min(max(int(round(float(record["x2"]))), 0), frame_width - 1)
                    y2 = min(max(int(round(float(record["y2"]))), 0), frame_height - 1)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4)
                    cv2.putText(
                        frame,
                        f"row={int(record['csv_row_index'])}",
                        (x1, max(30, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        color,
                        2,
                        cv2.LINE_AA,
                    )
                global_frame = runtime.global_frame_offset + local_frame
                global_time = float(runtime.global_times[local_frame])
                title = (
                    f"{runtime.manifest.clip_id} local={local_frame} global={global_frame} "
                    f"t={global_time:.6f}s raw={frame_width}x{frame_height}"
                )
                cv2.rectangle(frame, (0, 0), (min(frame_width - 1, 1900), 70), (0, 0, 0), -1)
                cv2.putText(
                    frame,
                    title,
                    (20, 48),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (255, 255, 255),
                    3,
                    cv2.LINE_AA,
                )
                filename = (
                    f"{runtime.manifest.clip_id}_{region}_f{local_frame:06d}.jpg"
                )
                image_path = overlay_dir / filename
                if not cv2.imwrite(
                    str(image_path),
                    frame,
                    [cv2.IMWRITE_JPEG_QUALITY, config.overlay_jpeg_quality],
                ):
                    raise ContractError(f"failed to write overlay image: {image_path}")
                manifest_rows.append(
                    {
                        "clip_id": runtime.manifest.clip_id,
                        "region": region,
                        "local_frame": local_frame,
                        "global_frame": global_frame,
                        "pts_sec": float(runtime.pts_seconds[local_frame]),
                        "global_time_sec": global_time,
                        "width": frame_width,
                        "height": frame_height,
                        "valid_boxes": valid_count,
                        "invalid_boxes": invalid_count,
                        "duplicate_boxes": duplicate_count,
                        "path": str(image_path.relative_to(output_dir)),
                    }
                )
        finally:
            capture.release()
    payload = {
        "coordinate_system": "raw_encoded_landscape_no_autorotate",
        "human_alignment_confirmed": False,
        "machine_frame_pts_verified": True,
        "visual_review_status": "pending_user_review",
        "samples": manifest_rows,
    }
    _atomic_write_json(output_dir / "overlay_manifest.json", payload)
    return payload


def _clip_boundaries(runtimes: list[ClipRuntime]) -> list[dict[str, Any]]:
    boundaries: list[dict[str, Any]] = []
    for runtime in runtimes:
        last = runtime.metadata.num_frames - 1
        boundaries.append(
            {
                "clip_id": runtime.manifest.clip_id,
                "clip_order": runtime.manifest.clip_order,
                "num_frames": runtime.metadata.num_frames,
                "start_global_frame": runtime.global_frame_offset,
                "end_global_frame_inclusive": runtime.global_frame_offset + last,
                "start_pts_sec": float(runtime.pts_seconds[0]),
                "end_pts_sec_inclusive": float(runtime.pts_seconds[-1]),
                "start_global_time_sec": float(runtime.global_times[0]),
                "end_global_time_sec_inclusive": float(runtime.global_times[-1]),
                "end_global_time_sec_exclusive": float(
                    runtime.global_time_offset
                    + runtime.metadata.duration_ticks * runtime.metadata.time_base
                ),
                "width": runtime.metadata.width,
                "height": runtime.metadata.height,
                "average_frame_rate": str(runtime.metadata.average_frame_rate),
                "time_base": str(runtime.metadata.time_base),
                "frame_duration_ticks": runtime.packets.duration_ticks[0],
                "rotation_metadata_degrees": runtime.metadata.rotation_degrees,
                "opencv_auto_rotate": False,
            }
        )
    return boundaries


def _validate_outputs(
    frames_path: Path,
    detections_path: Path,
    report: dict[str, Any],
    config: IngestConfig,
) -> None:
    frames_file = pq.ParquetFile(frames_path)
    detections_file = pq.ParquetFile(detections_path)
    if not frames_file.schema_arrow.equals(FRAMES_SCHEMA, check_metadata=False):
        raise ContractError("frames.parquet schema mismatch")
    if not detections_file.schema_arrow.equals(DETECTIONS_SCHEMA, check_metadata=False):
        raise ContractError("detections.parquet schema mismatch")
    expected_frames = _acceptance_int(config, "expected_num_frames")
    expected_boxes = _acceptance_int(config, "expected_num_input_boxes")
    if frames_file.metadata.num_rows != expected_frames:
        raise ContractError("frames.parquet row count mismatch")
    if detections_file.metadata.num_rows != expected_boxes:
        raise ContractError("detections.parquet row count mismatch")

    frames = pq.read_table(frames_path, columns=["global_frame", "global_time_sec"])
    global_frames = frames["global_frame"].to_numpy(zero_copy_only=False)
    frame_times = frames["global_time_sec"].to_numpy(zero_copy_only=False)
    if not np.array_equal(global_frames, np.arange(expected_frames, dtype=np.int64)):
        raise ContractError("global_frame is not contiguous 0..N-1")
    if not np.all(np.diff(frame_times) > 0.0):
        raise ContractError("frames.global_time_sec is not strictly increasing")

    detections = pq.read_table(
        detections_path,
        columns=[
            "det_id",
            "clip_id",
            "csv_row_index",
            "local_frame",
            "global_frame",
            "global_time_sec",
            "x1",
            "y1",
            "x2",
            "y2",
            "valid",
            "qa_flags",
        ],
    )
    det_ids = detections["det_id"].to_numpy(zero_copy_only=False)
    if np.unique(det_ids).size != expected_boxes:
        raise ContractError("det_id values are not unique")
    clip_ids = np.asarray(detections["clip_id"].to_pylist(), dtype=object)
    csv_row_indices = detections["csv_row_index"].to_numpy(zero_copy_only=False)
    det_local_frames = detections["local_frame"].to_numpy(zero_copy_only=False)
    for clip_report in report["per_clip"]:
        clip_id = str(clip_report["clip_id"])
        observed = csv_row_indices[clip_ids == clip_id]
        expected_clip_rows = int(clip_report["num_input_boxes"])
        if not np.array_equal(observed, np.arange(expected_clip_rows, dtype=np.int64)):
            raise ContractError(
                f"csv_row_index is not complete and contiguous for {clip_id}"
            )
    if "frame_observation_policy" in report:
        if report.get("frame_observation_policy") != frame_observation_policy():
            raise ContractError("frame observation policy differs from the contract")
        boundary_frames = {
            str(row["clip_id"]): int(row["num_frames"])
            for row in report["clip_boundaries"]
        }
        observed_total = 0
        unobserved_total = 0
        for clip_report in report["per_clip"]:
            clip_id = str(clip_report["clip_id"])
            expected_observation = summarize_frame_observation(
                boundary_frames[clip_id],
                det_local_frames[clip_ids == clip_id],
            )
            if clip_report.get("frame_observation") != expected_observation:
                raise ContractError(
                    f"frame observation report differs for {clip_id}"
                )
            if (
                int(clip_report["num_frames_with_boxes"])
                != expected_observation["num_observed_frames"]
                or int(clip_report["num_frames_without_boxes"])
                != expected_observation["num_unobserved_frames"]
            ):
                raise ContractError(
                    f"legacy frame-with-box counts differ for {clip_id}"
                )
            observed_total += expected_observation["num_observed_frames"]
            unobserved_total += expected_observation["num_unobserved_frames"]
        if (
            report.get("num_observed_frames") != observed_total
            or report.get("num_unobserved_frames") != unobserved_total
            or observed_total + unobserved_total != expected_frames
        ):
            raise ContractError("frame observation totals differ")
    det_global_frames = detections["global_frame"].to_numpy(zero_copy_only=False)
    det_times = detections["global_time_sec"].to_numpy(zero_copy_only=False)
    if not np.array_equal(det_times, frame_times[det_global_frames]):
        raise ContractError("detection global_time does not exactly match frames join")
    valid = detections["valid"].to_numpy(zero_copy_only=False)
    flags = detections["qa_flags"].to_numpy(zero_copy_only=False)
    if not np.array_equal(valid, (flags & INVALID_BBOX_MASK) == 0):
        raise ContractError("valid does not equal qa_flags invalid mask")
    x1 = detections["x1"].to_numpy(zero_copy_only=False)
    y1 = detections["y1"].to_numpy(zero_copy_only=False)
    x2 = detections["x2"].to_numpy(zero_copy_only=False)
    y2 = detections["y2"].to_numpy(zero_copy_only=False)
    width = _acceptance_int(config, "expected_width")
    height = _acceptance_int(config, "expected_height")
    valid_geometry = (
        np.isfinite(x1)
        & np.isfinite(y1)
        & np.isfinite(x2)
        & np.isfinite(y2)
        & (x1 >= 0.0)
        & (y1 >= 0.0)
        & (x2 <= width)
        & (y2 <= height)
        & (x2 > x1)
        & (y2 > y1)
        & (((x2 - x1) * (y2 - y1)) >= config.minimum_bbox_area_pixels)
    )
    if not np.all(valid_geometry[valid]):
        raise ContractError("at least one valid bbox violates geometry invariants")

    expected_mapping = {
        "num_clips": _acceptance_int(config, "expected_num_clips"),
        "num_frames": expected_frames,
        "num_input_boxes": expected_boxes,
        "num_valid_boxes": _acceptance_int(config, "expected_num_valid_boxes"),
        "num_invalid_boxes": _acceptance_int(config, "expected_num_invalid_boxes"),
        "num_duplicate_boxes_removed": _acceptance_int(
            config, "expected_num_duplicate_boxes_removed"
        ),
        "num_clamped_boxes": _acceptance_int(config, "expected_num_clamped_boxes"),
    }
    for key, expected in expected_mapping.items():
        if int(report[key]) != expected:
            raise ContractError(f"acceptance mismatch for {key}: {report[key]} != {expected}")
    duplicate_bit_count = int(
        np.count_nonzero(flags & int(BBoxQAFlag.HIGH_IOU_DUPLICATE))
    )
    if duplicate_bit_count != report["num_duplicate_boxes_removed"]:
        raise ContractError("duplicate flag/report count mismatch")
    if int(np.count_nonzero(valid)) != report["num_valid_boxes"]:
        raise ContractError("valid flag/report count mismatch")


def _output_fingerprint(path: Path, output_dir: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return {
        "path": str(path.relative_to(output_dir)),
        "size_bytes": int(path.stat().st_size),
        "sha256": digest.hexdigest(),
    }


def _unique_input_paths(
    manifest_path: Path, config_path: Path, clips: list[ClipManifest]
) -> list[Path]:
    candidates: list[Path] = [manifest_path, config_path]
    for clip in clips:
        candidates.extend([clip.video_path, clip.bbox_csv_path])
    result: list[Path] = []
    seen: set[Path] = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved not in seen:
            result.append(resolved)
            seen.add(resolved)
    return result


def _validate_completed_output(
    output_dir: Path,
    existing: dict[str, Any],
    input_paths: list[Path],
    config: IngestConfig,
) -> None:
    recorded_inputs = {
        str(Path(item["path"]).resolve()): item
        for item in existing.get("input_fingerprints", [])
    }
    if set(recorded_inputs) != {str(path.resolve()) for path in input_paths}:
        raise ContractError("completed input fingerprint path set no longer matches manifest")
    for path in input_paths:
        current = _file_fingerprint(
            path,
            progress_interval_sec=config.progress_interval_sec,
            logger=log,
        )
        recorded = recorded_inputs[str(path.resolve())]
        for key in ("size_bytes", "sha256"):
            if current[key] != recorded.get(key):
                raise ContractError(
                    f"completed input fingerprint mismatch for {path} ({key})"
                )

    recorded_outputs = existing.get("output_fingerprints", [])
    if not isinstance(recorded_outputs, list) or not recorded_outputs:
        raise ContractError("_SUCCESS.json has no output fingerprints")
    for recorded in recorded_outputs:
        artifact_path = output_dir / str(recorded["path"])
        if not artifact_path.is_file():
            raise ContractError(f"completed output artifact is missing: {recorded['path']}")
        current = _output_fingerprint(artifact_path, output_dir)
        for key in ("size_bytes", "sha256"):
            if current[key] != recorded.get(key):
                raise ContractError(
                    f"completed output fingerprint mismatch for {recorded['path']} ({key})"
                )

    report_path = output_dir / "ingest_report.json"
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read completed ingest report: {exc}") from exc
    _validate_outputs(
        output_dir / "frames.parquet",
        output_dir / "detections.parquet",
        report,
        config,
    )
    if existing.get("stats") != {
        key: report[key]
        for key in (
            "num_clips",
            "num_frames",
            "num_input_boxes",
            "num_valid_boxes",
            "num_invalid_boxes",
            "num_duplicate_boxes_removed",
            "num_clamped_boxes",
        )
    }:
        raise ContractError("_SUCCESS stats do not match ingest_report.json")


def run_s00(manifest_path: Path, config_path: Path, output_dir: Path) -> dict[str, Any]:
    manifest_path = manifest_path.resolve()
    config_path = config_path.resolve()
    output_dir = output_dir.resolve()
    config, config_payload, config_hash = load_config(config_path)
    clips, manifest_hash = load_manifest(manifest_path)
    if len(clips) != _acceptance_int(config, "expected_num_clips"):
        raise ContractError("manifest clip count does not match acceptance contract")

    input_paths = _unique_input_paths(manifest_path, config_path, clips)

    success_path = output_dir / "_SUCCESS.json"
    if success_path.is_file():
        existing = json.loads(success_path.read_text(encoding="utf-8"))
        if existing.get("config_hash") != config_hash or existing.get(
            "manifest_hash"
        ) != manifest_hash:
            raise ContractError("existing _SUCCESS.json belongs to different inputs/config")
        _validate_completed_output(output_dir, existing, input_paths, config)
        log(f"[s00] already complete and fully revalidated: {success_path}")
        return existing
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ContractError(
            f"output directory is non-empty without _SUCCESS.json: {output_dir}"
        )
    final_output_dir = output_dir
    final_output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = final_output_dir.parent / (
        f".{final_output_dir.name}.staging-{os.getpid()}"
    )
    if staging_dir.exists():
        raise ContractError(f"staging directory already exists: {staging_dir}")
    staging_dir.mkdir(parents=False)
    output_dir = staging_dir
    success_path = output_dir / "_SUCCESS.json"

    input_fingerprints = [
        _file_fingerprint(
            path,
            progress_interval_sec=config.progress_interval_sec,
            logger=log,
        )
        for path in input_paths
    ]

    runtimes, sequence_end_time = _build_clip_runtimes(clips, config, log)
    frames_path = output_dir / "frames.parquet"
    detections_path = output_dir / "detections.parquet"
    num_frames = _write_frames(runtimes, frames_path, config)
    log(f"[s00] wrote frames.parquet: {num_frames:,} rows")
    detection_stats = _write_detections(runtimes, detections_path, config, log)
    log(
        f"[s00] wrote detections.parquet: {detection_stats['num_input_boxes']:,} rows"
    )

    overlay_payload: dict[str, Any] | None = None
    if config.create_overlay_samples:
        overlay_payload = _render_overlay_samples(
            runtimes, detections_path, output_dir, config
        )
        log(f"[s00] wrote {len(overlay_payload['samples'])} full-resolution overlays")

    resolved_manifest = [
        {
            "sequence_id": clip.sequence_id,
            "clip_order": clip.clip_order,
            "clip_id": clip.clip_id,
            "video_path": str(clip.video_path),
            "bbox_csv_path": str(clip.bbox_csv_path),
            "frame_index_base": clip.frame_index_base,
            "bbox_format": clip.bbox_format,
        }
        for clip in clips
    ]
    _atomic_write_json(output_dir / "resolved_manifest.json", resolved_manifest)
    _atomic_write_json(output_dir / "effective_config.json", config_payload)

    report = {
        "schema_version": config.schema_version,
        "sequence_id": clips[0].sequence_id,
        "num_clips": len(clips),
        "num_frames": num_frames,
        "num_input_boxes": detection_stats["num_input_boxes"],
        "num_valid_boxes": detection_stats["num_valid_boxes"],
        "num_invalid_boxes": detection_stats["num_invalid_boxes"],
        "num_duplicate_boxes_removed": detection_stats[
            "num_duplicate_boxes_removed"
        ],
        "num_clamped_boxes": detection_stats["num_clamped_boxes"],
        "sequence_end_time_sec_exclusive": float(sequence_end_time),
        "source_rows_retained": True,
        "keypoints_used": False,
        "legacy_track_id_used": False,
        "frame_observation_policy": frame_observation_policy(),
        "num_observed_frames": sum(
            int(row["frame_observation"]["num_observed_frames"])
            for row in detection_stats["per_clip"]
        ),
        "num_unobserved_frames": sum(
            int(row["frame_observation"]["num_unobserved_frames"])
            for row in detection_stats["per_clip"]
        ),
        "coordinate_system": "raw_encoded_landscape_no_autorotate",
        "qa_flag_definitions": QA_FLAG_DEFINITIONS,
        "qa_flag_counts": detection_stats["qa_flag_counts"],
        "clip_boundaries": _clip_boundaries(runtimes),
        "per_clip": detection_stats["per_clip"],
        "overlay_samples_created": 0
        if overlay_payload is None
        else len(overlay_payload["samples"]),
    }
    _atomic_write_json(output_dir / "ingest_report.json", report)
    _validate_outputs(frames_path, detections_path, report, config)
    _verify_inputs_unchanged(input_fingerprints)

    artifact_paths = sorted(
        path
        for path in output_dir.rglob("*")
        if path.is_file() and path.name != "_SUCCESS.json"
    )
    output_fingerprints = [
        _output_fingerprint(path, output_dir) for path in artifact_paths
    ]
    success = {
        "stage": "S00",
        "schema_version": config.schema_version,
        "config_hash": config_hash,
        "manifest_hash": manifest_hash,
        "program_commit_hash": None,
        "input_fingerprints": input_fingerprints,
        "output_fingerprints": output_fingerprints,
        "stats": {
            "num_clips": report["num_clips"],
            "num_frames": report["num_frames"],
            "num_input_boxes": report["num_input_boxes"],
            "num_valid_boxes": report["num_valid_boxes"],
            "num_invalid_boxes": report["num_invalid_boxes"],
            "num_duplicate_boxes_removed": report[
                "num_duplicate_boxes_removed"
            ],
            "num_clamped_boxes": report["num_clamped_boxes"],
        },
    }
    _atomic_write_json(success_path, success)
    if final_output_dir.exists():
        if any(final_output_dir.iterdir()):
            raise ContractError(
                f"final output became non-empty during staging: {final_output_dir}"
            )
        final_output_dir.rmdir()
    os.replace(staging_dir, final_output_dir)
    success_path = final_output_dir / "_SUCCESS.json"
    log(f"[s00] complete: {success_path}")
    return success
