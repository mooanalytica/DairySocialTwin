from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from cowtrack.config import ContractError
from cowtrack.linking.finalize import (
    FinalizeMicro,
    FinalizeProposal,
    build_component_union,
    build_det_to_stable_rows,
)
from cowtrack.linking.finalize_config import load_s04_finalize_config


CONFIG = Path(__file__).parents[1] / "configs" / "s04_finalize.yaml"


def _micro(micro_id: int, start: float, end: float) -> FinalizeMicro:
    start_frame = int(start * 10)
    end_frame = int(end * 10)
    return FinalizeMicro(
        micro_id=micro_id,
        start_det_id=micro_id * 10,
        end_det_id=micro_id * 10 + 1,
        start_clip_id="GX040006",
        end_clip_id="GX040006",
        start_global_frame=start_frame,
        end_global_frame=end_frame,
        start_time_sec=start,
        end_time_sec=end,
        num_detections=2,
    )


def _edge(name: str, source: int, target: int, probability: float) -> FinalizeProposal:
    return FinalizeProposal(name, f"edge-{name}", source, target, probability)


def test_component_union_is_undirected_total_dense_and_shuffle_invariant() -> None:
    micros = [
        _micro(1, 0.0, 1.0),
        _micro(2, 2.0, 3.0),
        _micro(3, 4.0, 5.0),
        _micro(4, 6.0, 7.0),
    ]
    # 1--3--2 is one actual-micro component even though the evidence does not
    # connect chronological neighbors 2 and 3.
    proposals = [_edge("a", 1, 3, 0.99), _edge("b", 1, 2, 0.98)]
    result = build_component_union(micros, proposals)
    shuffled = build_component_union(list(reversed(micros)), list(reversed(proposals)))
    assert result == shuffled
    assert result.components == ((1, 2, 3), (4,))
    assert result.micro_to_stable == {1: 0, 2: 0, 3: 0, 4: 1}
    rows = {int(row["micro_id"]): row for row in result.micro_to_stable_rows}
    assert rows[2]["predecessor_micro_id"] == 1
    assert rows[2]["predecessor_edge_id"] == "edge-b"
    assert rows[3]["predecessor_micro_id"] == 2
    assert rows[3]["predecessor_edge_id"] is None
    assert rows[3]["predecessor_link_probability"] is None
    assert rows[3]["component_num_proposal_edges"] == 2
    assert result.stable_tracklet_rows[1]["is_singleton"] is True


def test_empty_proposals_make_every_micro_a_singleton() -> None:
    result = build_component_union(
        [_micro(2, 2.0, 3.0), _micro(1, 0.0, 1.0)], []
    )
    assert result.components == ((1,), (2,))
    assert result.micro_to_stable == {1: 0, 2: 1}
    assert all(row["num_proposal_edges"] == 0 for row in result.stable_tracklet_rows)


def test_component_overlap_fails_closed_even_when_each_edge_is_legal() -> None:
    micros = [
        _micro(1, 0.0, 2.0),
        _micro(2, 1.0, 3.0),
        _micro(3, 4.0, 5.0),
    ]
    proposals = [_edge("a", 1, 3, 0.9), _edge("b", 2, 3, 0.8)]
    with pytest.raises(ContractError, match="overlapping"):
        build_component_union(micros, proposals)


@pytest.mark.parametrize(
    "proposal",
    [
        _edge("self", 1, 1, 0.9),
        _edge("unknown", 1, 99, 0.9),
    ],
)
def test_invalid_proposal_identity_fails_closed(proposal: FinalizeProposal) -> None:
    with pytest.raises(ContractError):
        build_component_union([_micro(1, 0.0, 1.0)], [proposal])


def test_detection_mapping_is_total_and_chronological() -> None:
    union = build_component_union(
        [_micro(1, 0.0, 1.0), _micro(2, 2.0, 3.0)],
        [_edge("a", 1, 2, 0.9)],
    )
    rows = build_det_to_stable_rows(
        det_ids=[20, 10, 21, 11],
        micro_ids=[2, 1, 2, 1],
        order_in_micro=[0, 0, 1, 1],
        micro_to_stable_rows=union.micro_to_stable_rows,
    )
    by_det = {int(row["det_id"]): row for row in rows}
    assert [by_det[det]["order_in_stable_detection"] for det in (10, 11, 20, 21)] == [
        0,
        1,
        2,
        3,
    ]


def test_signed_detection_ids_are_preserved() -> None:
    micro = FinalizeMicro(
        micro_id=1,
        start_det_id=-99,
        end_det_id=-2,
        start_clip_id="GX040006",
        end_clip_id="GX040006",
        start_global_frame=0,
        end_global_frame=1,
        start_time_sec=0.0,
        end_time_sec=0.1,
        num_detections=2,
    )
    union = build_component_union([micro], [])
    assert union.stable_tracklet_rows[0]["start_det_id"] == -99
    rows = build_det_to_stable_rows(
        det_ids=[-99, -2],
        micro_ids=[1, 1],
        order_in_micro=[0, 1],
        micro_to_stable_rows=union.micro_to_stable_rows,
    )
    assert [row["det_id"] for row in rows] == [-99, -2]


def test_finalize_config_is_strict(tmp_path: Path) -> None:
    config, _, digest = load_s04_finalize_config(CONFIG)
    assert config.operator_approved is True
    assert config.read_review_labels is False
    assert config.solver == "none"
    assert len(digest) == 64
    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    payload["finalize"]["solver"] = "matching"
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ContractError):
        load_s04_finalize_config(bad)


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("pipeline", "operator_approved", 1),
        ("finalize", "require_strict_non_overlap", 1),
        ("finalize", "read_review_labels", 0),
    ],
)
def test_finalize_config_rejects_integer_booleans(
    tmp_path: Path, section: str, field: str, value: int
) -> None:
    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    payload[section][field] = value
    bad = tmp_path / f"bad-{field}.yaml"
    bad.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ContractError, match="must be a boolean"):
        load_s04_finalize_config(bad)
