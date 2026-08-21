"""Stable Arrow schema for S01 detection-edge diagnostics."""

from __future__ import annotations

import pyarrow as pa


DET_EDGES_SCHEMA = pa.schema(
    [
        pa.field("src_det_id", pa.int64(), nullable=False),
        pa.field("dst_det_id", pa.int64(), nullable=False),
        pa.field("delta_time_sec", pa.float32(), nullable=False),
        pa.field("forward_cost", pa.float32(), nullable=False),
        pa.field("backward_cost", pa.float32(), nullable=False),
        pa.field("forward_rank", pa.int16(), nullable=False),
        pa.field("backward_rank", pa.int16(), nullable=False),
        pa.field("bidirectional_agree", pa.bool_(), nullable=False),
        pa.field("accepted", pa.bool_(), nullable=False),
        pa.field("reject_reason", pa.string(), nullable=False),
    ]
)
