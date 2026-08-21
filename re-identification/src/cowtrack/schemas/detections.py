from __future__ import annotations

from enum import IntFlag

import pyarrow as pa


class BBoxQAFlag(IntFlag):
    """Stable uint32 bit assignments for S00 bbox QA."""

    CLAMPED_TO_IMAGE = 1 << 0
    NONFINITE_NUMERIC = 1 << 1
    NONPOSITIVE_SOURCE_SIZE = 1 << 2
    EMPTY_AFTER_CLAMP = 1 << 3
    AREA_BELOW_MINIMUM = 1 << 4
    MOSTLY_OUTSIDE = 1 << 5
    HIGH_IOU_DUPLICATE = 1 << 6
    INVALID_CONFIDENCE = 1 << 7


INVALID_BBOX_MASK = int(
    BBoxQAFlag.NONFINITE_NUMERIC
    | BBoxQAFlag.NONPOSITIVE_SOURCE_SIZE
    | BBoxQAFlag.EMPTY_AFTER_CLAMP
    | BBoxQAFlag.AREA_BELOW_MINIMUM
    | BBoxQAFlag.MOSTLY_OUTSIDE
    | BBoxQAFlag.HIGH_IOU_DUPLICATE
)


QA_FLAG_DEFINITIONS = {flag.name: int(flag) for flag in BBoxQAFlag}


DETECTIONS_SCHEMA = pa.schema(
    [
        pa.field("det_id", pa.int64(), nullable=False),
        pa.field("sequence_id", pa.string(), nullable=False),
        pa.field("clip_id", pa.string(), nullable=False),
        pa.field("local_frame", pa.int32(), nullable=False),
        pa.field("global_frame", pa.int64(), nullable=False),
        pa.field("global_time_sec", pa.float64(), nullable=False),
        pa.field("x1", pa.float32(), nullable=False),
        pa.field("y1", pa.float32(), nullable=False),
        pa.field("x2", pa.float32(), nullable=False),
        pa.field("y2", pa.float32(), nullable=False),
        pa.field("cx_norm", pa.float32(), nullable=False),
        pa.field("cy_norm", pa.float32(), nullable=False),
        pa.field("w_norm", pa.float32(), nullable=False),
        pa.field("h_norm", pa.float32(), nullable=False),
        pa.field("area_norm", pa.float32(), nullable=False),
        pa.field("bbox_confidence", pa.float32(), nullable=True),
        pa.field("legacy_track_id", pa.string(), nullable=True),
        pa.field("csv_row_index", pa.int64(), nullable=False),
        pa.field("valid", pa.bool_(), nullable=False),
        pa.field("qa_flags", pa.uint32(), nullable=False),
    ]
)

