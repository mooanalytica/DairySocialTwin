from __future__ import annotations

from dataclasses import fields, replace
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

from cowtrack.config import ContractError
from cowtrack.linking.config import load_link_calibration_config
from cowtrack.linking.features import build_clean_gallery
from cowtrack.linking.pseudo_pairs import CalibrationInput
from cowtrack.linking.stable_appearance import build_stable_appearance
from cowtrack.schemas.stable_appearance import STABLE_APPEARANCE_SCHEMA


CONFIG = Path(__file__).parents[1] / "configs" / "s03_calibration.yaml"


def _unit_angle(degrees: float) -> np.ndarray:
    radians = np.deg2rad(degrees)
    return np.asarray([np.cos(radians), np.sin(radians)], dtype=np.float32)


def _input() -> CalibrationInput:
    parent_ids = np.asarray([-10, 20, 30], dtype=np.int64)
    det_ids = np.asarray([-101, -100, -99, 200, 201, 202, 300, 301], dtype=np.int64)
    det_micro_ids = np.asarray([-10, -10, -10, 20, 20, 20, 30, 30], dtype=np.int64)
    det_count = len(det_ids)
    sample_count = det_count
    review = np.zeros(det_count, dtype=np.bool_)
    review[4] = True
    overlap = np.zeros(sample_count, dtype=np.float32)
    overlap[5] = 0.30
    s02_inlier = np.ones(sample_count, dtype=np.bool_)
    s02_inlier[2] = False
    embeddings = np.stack([_unit_angle(value) for value in range(sample_count)])
    return CalibrationInput(
        timeline_clip_ids=np.asarray(["clip-a"], dtype=object),
        timeline_clip_start_time_sec=np.asarray([0.0], dtype=np.float64),
        timeline_clip_end_time_sec=np.asarray([10.0], dtype=np.float64),
        parent_micro_ids=parent_ids,
        parent_status=np.asarray(["valid"] * 3, dtype=object),
        parent_num_detections=np.asarray([3, 3, 2], dtype=np.int32),
        parent_local_purity_score=np.ones(3, dtype=np.float32),
        parent_bidirectional_agreement=np.ones(3, dtype=np.float32),
        parent_internal_cosine_p10=np.ones(3, dtype=np.float32),
        det_ids=det_ids,
        det_micro_ids=det_micro_ids,
        det_order_in_micro=np.asarray([0, 1, 2, 0, 1, 2, 0, 1], dtype=np.int32),
        det_clip_ids=np.asarray(["clip-a"] * det_count, dtype=object),
        det_global_frames=np.arange(det_count, dtype=np.int64),
        det_global_time_sec=np.arange(det_count, dtype=np.float64),
        det_cx_norm=np.full(det_count, 0.5, dtype=np.float32),
        det_cy_norm=np.full(det_count, 0.5, dtype=np.float32),
        det_w_norm=np.full(det_count, 0.2, dtype=np.float32),
        det_h_norm=np.full(det_count, 0.3, dtype=np.float32),
        det_other_bbox_max_iou=overlap.copy(),
        det_boundary_distance=np.full(det_count, 0.5, dtype=np.float32),
        det_review_excluded=review,
        sample_ids=np.arange(1_000, 1_000 + sample_count, dtype=np.int64),
        sample_micro_ids=det_micro_ids.copy(),
        sample_det_ids=det_ids.copy(),
        sample_crop_quality=np.ones(sample_count, dtype=np.float32),
        sample_other_bbox_max_iou=overlap,
        sample_s02_inlier=s02_inlier,
        sample_embedding_rows=np.arange(sample_count, dtype=np.int64),
        embeddings=embeddings,
    )


def _config():
    return load_link_calibration_config(CONFIG)[0]


def _mapping() -> dict[int, int]:
    return {-10: 0, 20: 0, 30: 1}


def _shuffle(data: CalibrationInput) -> CalibrationInput:
    parent_order = np.asarray([2, 0, 1])
    det_order = np.asarray([4, 0, 7, 2, 6, 1, 5, 3])
    sample_order = np.asarray([5, 1, 7, 0, 6, 3, 2, 4])
    values: dict[str, object] = {}
    for field in fields(CalibrationInput):
        value = getattr(data, field.name)
        if field.name == "embeddings" or field.name.startswith("timeline_"):
            values[field.name] = value
        elif field.name.startswith("parent_"):
            values[field.name] = value[parent_order]
        elif field.name.startswith("det_"):
            values[field.name] = value[det_order]
        elif field.name.startswith("sample_"):
            values[field.name] = value[sample_order]
        else:  # pragma: no cover
            raise AssertionError(field.name)
    return CalibrationInput(**values)  # type: ignore[arg-type]


def test_rebuilds_pooled_stable_gallery_and_retains_missing_row(monkeypatch) -> None:
    observed_counts: list[int] = []

    def spy(*args, **kwargs):
        observed_counts.append(len(args[0]))
        return build_clean_gallery(*args, **kwargs)

    monkeypatch.setattr(
        "cowtrack.linking.stable_appearance.build_clean_gallery", spy
    )
    result = build_stable_appearance(_input(), _mapping(), _config())

    assert observed_counts == [6, 2]
    np.testing.assert_array_equal(result.stable_ids, [0, 1])
    assert result.stable_prototypes.shape == (2, 3, 2)
    assert result.stable_prototypes.dtype == np.float16
    assert result.stable_prototype_mask.dtype == np.bool_
    assert result.stable_prototype_mask[0].tolist() == [True, False, False]
    assert result.stable_prototype_mask[1].tolist() == [False, False, False]
    assert not np.any(result.stable_prototypes[1])

    usable, missing = result.stable_appearance_rows
    assert usable["stable_id"] == usable["prototype_row"] == 0
    assert usable["constituent_micro_ids"] == [-10, 20]
    assert usable["num_input_samples"] == 6
    assert usable["num_s02_inliers"] == 5
    assert usable["num_clean_candidates"] == 3
    assert usable["num_clean_inliers"] == 3
    assert usable["num_overlap_rejected"] == 1
    assert usable["num_review_excluded"] == 1
    assert usable["clean_sample_ids"] == [1000, 1001, 1003]
    assert usable["appearance_usable"] is True
    assert usable["missing_reason"] is None
    assert missing["stable_id"] == missing["prototype_row"] == 1
    assert missing["constituent_micro_ids"] == [30]
    assert missing["num_input_samples"] == 2
    assert missing["appearance_usable"] is False
    assert missing["missing_reason"] == "clean_gallery_missing"
    assert missing["clean_sample_ids"] == []

    table = pa.Table.from_pylist(
        list(result.stable_appearance_rows), schema=STABLE_APPEARANCE_SCHEMA
    )
    assert table.schema.equals(STABLE_APPEARANCE_SCHEMA)


def test_stable_appearance_reports_final_progress() -> None:
    progress: list[tuple[int, int]] = []
    build_stable_appearance(
        _input(),
        _mapping(),
        _config(),
        progress_callback=lambda completed, total: progress.append(
            (completed, total)
        ),
        progress_interval_sec=60.0,
    )
    assert progress == [(2, 2)]


def test_result_is_invariant_to_all_input_row_orders() -> None:
    data = _input()
    first = build_stable_appearance(data, _mapping(), _config())
    second = build_stable_appearance(
        _shuffle(data), dict(reversed(tuple(_mapping().items()))), _config()
    )

    np.testing.assert_array_equal(first.stable_ids, second.stable_ids)
    np.testing.assert_array_equal(first.stable_prototypes, second.stable_prototypes)
    np.testing.assert_array_equal(
        first.stable_prototype_mask, second.stable_prototype_mask
    )
    assert first.stable_appearance_rows == second.stable_appearance_rows


def test_row_prototype_count_matches_multiple_valid_tensor_slots() -> None:
    data = _input()
    embeddings = data.embeddings.copy()
    embeddings[:6] = np.stack(
        [_unit_angle(value) for value in (0.0, 1.0, -1.0, 40.0, 41.0, 39.0)]
    )
    clean = replace(
        data,
        embeddings=embeddings,
        sample_s02_inlier=np.ones(8, dtype=np.bool_),
        sample_other_bbox_max_iou=np.zeros(8, dtype=np.float32),
        det_review_excluded=np.zeros(8, dtype=np.bool_),
    )

    result = build_stable_appearance(clean, _mapping(), _config())

    assert result.stable_prototype_mask[0].tolist() == [True, True, False]
    assert result.stable_appearance_rows[0]["num_valid_prototypes"] == 2
    assert result.stable_appearance_rows[0]["num_valid_prototypes"] == int(
        np.count_nonzero(result.stable_prototype_mask[0])
    )


@pytest.mark.parametrize(
    "mapping",
    [
        {-10: 0, 20: 0},
        {-10: 1, 20: 1, 30: 2},
        {-10: 0, 20: 0, 30: 2},
        {-10: 0, 20: 0, 30: True},
    ],
)
def test_rejects_non_bijective_or_non_dense_component_mapping(mapping) -> None:
    with pytest.raises(ContractError, match="mapping|dense"):
        build_stable_appearance(_input(), mapping, _config())


def test_rejects_non_bijective_embedding_rows() -> None:
    data = _input()
    embedding_rows = data.sample_embedding_rows.copy()
    embedding_rows[-1] = embedding_rows[0]

    with pytest.raises(ContractError, match="sample_embedding_rows values"):
        build_stable_appearance(
            replace(data, sample_embedding_rows=embedding_rows),
            _mapping(),
            _config(),
        )


@pytest.mark.parametrize("kind", ["non_normalized", "non_finite"])
def test_rejects_invalid_embeddings(kind: str) -> None:
    data = _input()
    embeddings = data.embeddings.copy()
    if kind == "non_normalized":
        embeddings[0] *= 2.0
    else:
        embeddings[0, 0] = np.nan

    with pytest.raises(ContractError, match="embeddings"):
        build_stable_appearance(
            replace(data, embeddings=embeddings), _mapping(), _config()
        )


def test_rejects_changed_clean_policy() -> None:
    with pytest.raises(ContractError, match="exact fixed S03 clean policy"):
        build_stable_appearance(
            _input(),
            _mapping(),
            replace(_config(), clean_max_other_bbox_iou=0.3),
        )
