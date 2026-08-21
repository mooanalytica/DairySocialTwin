from __future__ import annotations

import pyarrow as pa


FRAMES_SCHEMA = pa.schema(
    [
        pa.field("sequence_id", pa.string(), nullable=False),
        pa.field("clip_id", pa.string(), nullable=False),
        pa.field("clip_order", pa.int16(), nullable=False),
        pa.field("local_frame", pa.int32(), nullable=False),
        pa.field("global_frame", pa.int64(), nullable=False),
        pa.field("pts_sec", pa.float64(), nullable=False),
        pa.field("global_time_sec", pa.float64(), nullable=False),
        pa.field("width", pa.int32(), nullable=False),
        pa.field("height", pa.int32(), nullable=False),
    ]
)

