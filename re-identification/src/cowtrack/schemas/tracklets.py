"""Stable Arrow schemas for S01 micro-tracklet outputs."""

from __future__ import annotations

import pyarrow as pa


MICROTRACKLET_STATUS_VALID = "valid"
MICROTRACKLET_STATUS_SHORT_FRAGMENT = "short_fragment"
MICROTRACKLET_STATUS_JUNK_CANDIDATE = "junk_candidate"
MICROTRACKLET_STATUSES = frozenset(
    {
        MICROTRACKLET_STATUS_VALID,
        MICROTRACKLET_STATUS_SHORT_FRAGMENT,
        MICROTRACKLET_STATUS_JUNK_CANDIDATE,
    }
)


DET_TO_MICRO_SCHEMA = pa.schema(
    [
        pa.field("det_id", pa.int64(), nullable=False),
        pa.field("micro_id", pa.int64(), nullable=False),
        pa.field("order_in_micro", pa.int32(), nullable=False),
        pa.field("incoming_edge_score", pa.float32(), nullable=True),
    ]
)


MICROTRACKLETS_SCHEMA = pa.schema(
    [
        pa.field("micro_id", pa.int64(), nullable=False),
        pa.field("start_global_frame", pa.int64(), nullable=False),
        pa.field("end_global_frame", pa.int64(), nullable=False),
        pa.field("start_time_sec", pa.float64(), nullable=False),
        pa.field("end_time_sec", pa.float64(), nullable=False),
        pa.field("num_detections", pa.int32(), nullable=False),
        pa.field("duration_sec", pa.float32(), nullable=False),
        pa.field("start_x1", pa.float32(), nullable=False),
        pa.field("start_y1", pa.float32(), nullable=False),
        pa.field("start_x2", pa.float32(), nullable=False),
        pa.field("start_y2", pa.float32(), nullable=False),
        pa.field("end_x1", pa.float32(), nullable=False),
        pa.field("end_y1", pa.float32(), nullable=False),
        pa.field("end_x2", pa.float32(), nullable=False),
        pa.field("end_y2", pa.float32(), nullable=False),
        pa.field("end_vx_norm_per_sec", pa.float32(), nullable=False),
        pa.field("end_vy_norm_per_sec", pa.float32(), nullable=False),
        pa.field("start_vx_norm_per_sec", pa.float32(), nullable=False),
        pa.field("start_vy_norm_per_sec", pa.float32(), nullable=False),
        pa.field("max_internal_center_jump", pa.float32(), nullable=False),
        pa.field("median_internal_cost", pa.float32(), nullable=False),
        pa.field("bidirectional_agreement", pa.float32(), nullable=False),
        pa.field("local_purity_score", pa.float32(), nullable=False),
        pa.field("status", pa.string(), nullable=False),
    ]
)
