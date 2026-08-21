from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from cowtrack.linking.config import load_link_calibration_config
from cowtrack.linking.features import feature_schema
from cowtrack.linking.pseudo_pairs import CalibrationInput, ProductionFeatureStore


CONFIG = Path(__file__).resolve().parents[1] / "configs" / "s03_calibration.yaml"


def _data(*, missing_target: bool = False) -> CalibrationInput:
    det_ids = np.arange(6, dtype=np.int64)
    micros = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64)
    order = np.asarray([0, 1, 2, 0, 1, 2], dtype=np.int32)
    embeddings = np.asarray(
        [[1.0, 0.0]] * 3 + [[0.99, 0.14106736]] * 3, dtype=np.float32
    )
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    inlier = np.ones(6, dtype=np.bool_)
    if missing_target:
        inlier[-1] = False
    return CalibrationInput(
        timeline_clip_ids=np.asarray(["GX040006"], dtype=object),
        timeline_clip_start_time_sec=np.asarray([0.0], dtype=np.float64),
        timeline_clip_end_time_sec=np.asarray([6.0], dtype=np.float64),
        parent_micro_ids=np.asarray([0, 1], dtype=np.int64),
        parent_status=np.asarray(["valid", "valid"], dtype=object),
        parent_num_detections=np.asarray([3, 3], dtype=np.int32),
        parent_local_purity_score=np.ones(2, dtype=np.float32),
        parent_bidirectional_agreement=np.ones(2, dtype=np.float32),
        parent_internal_cosine_p10=np.ones(2, dtype=np.float32),
        det_ids=det_ids,
        det_micro_ids=micros,
        det_order_in_micro=order,
        det_clip_ids=np.asarray(["GX040006"] * 6, dtype=object),
        det_global_frames=np.arange(6, dtype=np.int64),
        det_global_time_sec=np.arange(6, dtype=np.float64),
        det_cx_norm=np.linspace(0.2, 0.25, 6, dtype=np.float32),
        det_cy_norm=np.full(6, 0.5, dtype=np.float32),
        det_w_norm=np.full(6, 0.2, dtype=np.float32),
        det_h_norm=np.full(6, 0.3, dtype=np.float32),
        det_other_bbox_max_iou=np.zeros(6, dtype=np.float32),
        det_boundary_distance=np.full(6, 0.5, dtype=np.float32),
        det_review_excluded=np.zeros(6, dtype=np.bool_),
        sample_ids=np.arange(6, dtype=np.int64),
        sample_micro_ids=micros,
        sample_det_ids=det_ids,
        sample_crop_quality=np.ones(6, dtype=np.float32),
        sample_other_bbox_max_iou=np.zeros(6, dtype=np.float32),
        sample_s02_inlier=inlier,
        sample_embedding_rows=np.arange(6, dtype=np.int64),
        embeddings=embeddings,
    )


def _config():
    config, _, _ = load_link_calibration_config(CONFIG)
    return replace(config, min_parent_detections=3)


def test_production_uses_same_clean_gallery_and_feature_schema() -> None:
    built = ProductionFeatureStore(_data(), _config()).build_pair(0, 1, "short")
    assert built.appearance_present is True
    assert built.high_overlap is False
    assert tuple(built.feature_values or {}) == feature_schema("short")


def test_production_missing_clean_gallery_rejects_without_sentinel() -> None:
    built = ProductionFeatureStore(
        _data(missing_target=True), _config()
    ).build_pair(0, 1, "short")
    assert built.appearance_present is False
    assert built.feature_values is None
    assert built.reason == "target_gallery_missing"
    assert built.source_gallery.present is True
    assert built.source_gallery.provenance is not None
    assert built.target_gallery.present is False
    assert built.target_gallery.provenance is None


def test_missing_gallery_preserves_raw_overlap_and_missing_side() -> None:
    data = _data(missing_target=True)
    overlap = data.sample_other_bbox_max_iou.copy()
    overlap[-1] = 0.8
    built = ProductionFeatureStore(
        replace(data, sample_other_bbox_max_iou=overlap), _config()
    ).build_pair(0, 1, "short")
    assert built.feature_values is None
    assert built.high_overlap is True
    assert built.reason == "target_gallery_missing"
