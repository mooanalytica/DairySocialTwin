from __future__ import annotations

from cowtrack.schemas.edges import DET_EDGES_SCHEMA
from cowtrack.schemas.tracklets import DET_TO_MICRO_SCHEMA, MICROTRACKLETS_SCHEMA


def test_s01_output_schema_columns_are_stable() -> None:
    assert DET_EDGES_SCHEMA.names == [
        "src_det_id",
        "dst_det_id",
        "delta_time_sec",
        "forward_cost",
        "backward_cost",
        "forward_rank",
        "backward_rank",
        "bidirectional_agree",
        "accepted",
        "reject_reason",
    ]
    assert DET_TO_MICRO_SCHEMA.names == [
        "det_id",
        "micro_id",
        "order_in_micro",
        "incoming_edge_score",
    ]
    assert MICROTRACKLETS_SCHEMA.names[:7] == [
        "micro_id",
        "start_global_frame",
        "end_global_frame",
        "start_time_sec",
        "end_time_sec",
        "num_detections",
        "duration_sec",
    ]


def test_only_incoming_edge_score_is_nullable() -> None:
    assert [field.name for field in DET_EDGES_SCHEMA if field.nullable] == []
    assert [field.name for field in DET_TO_MICRO_SCHEMA if field.nullable] == [
        "incoming_edge_score"
    ]
    assert [field.name for field in MICROTRACKLETS_SCHEMA if field.nullable] == []
