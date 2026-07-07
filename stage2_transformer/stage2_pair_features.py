from __future__ import annotations

import math
from typing import Iterable

import numpy as np


FEATURE_SCHEMA_VERSION = "stage2_pair_local_v1"
FEATURES_PER_KEYPOINT = 8
GLOBAL_FEATURES = 2


def expected_in_features(num_keypoints: int) -> int:
    return FEATURES_PER_KEYPOINT * int(num_keypoints) + GLOBAL_FEATURES


def feature_names(keypoints: Iterable[str]) -> list[str]:
    names: list[str] = []
    for kp in keypoints:
        names.extend(
            [
                f"{kp}_A_local_x",
                f"{kp}_A_local_y",
                f"{kp}_A_conf",
                f"{kp}_B_local_x",
                f"{kp}_B_local_y",
                f"{kp}_B_conf",
                f"{kp}_AminusB_x",
                f"{kp}_AminusB_y",
            ]
        )
    names.extend(["pair_center_AminusB_x", "pair_center_AminusB_y"])
    return names


def _clean_center(center) -> tuple[tuple[float, float], bool]:
    if center is None:
        return (0.0, 0.0), False
    try:
        x = float(center[0])
        y = float(center[1])
    except Exception:
        return (0.0, 0.0), False
    if not (math.isfinite(x) and math.isfinite(y)):
        return (0.0, 0.0), False
    return (x, y), True


def _clean_scale(scale) -> tuple[float, bool]:
    try:
        value = float(scale)
    except Exception:
        return 1.0, False
    if not math.isfinite(value) or value <= 0:
        return 1.0, False
    return value, True


def encode_pair_frame(
    kpts_a,
    kpts_b,
    center_a,
    center_b,
    scale_a,
    scale_b,
    num_keypoints: int,
    conf_thr: float = 0.20,
) -> np.ndarray:
    """Encode one pair frame without absolute image-origin or image-size features."""

    num_keypoints = int(num_keypoints)
    out = np.zeros((expected_in_features(num_keypoints),), dtype=np.float32)

    (ca_x, ca_y), valid_center_a = _clean_center(center_a)
    (cb_x, cb_y), valid_center_b = _clean_center(center_b)
    s_a, valid_scale_a = _clean_scale(scale_a)
    s_b, valid_scale_b = _clean_scale(scale_b)
    valid_geom_a = valid_center_a and valid_scale_a
    valid_geom_b = valid_center_b and valid_scale_b
    pair_scale = max(1e-6, 0.5 * (s_a + s_b))

    col = 0
    for ki in range(num_keypoints):
        ax = ay = ac = bx = by = bc = 0.0
        valid_a = False
        valid_b = False

        if kpts_a is not None and ki < len(kpts_a):
            try:
                ax = float(kpts_a[ki, 0])
                ay = float(kpts_a[ki, 1])
                ac = float(kpts_a[ki, 2])
                valid_a = math.isfinite(ax) and math.isfinite(ay) and ac >= conf_thr and valid_geom_a
            except Exception:
                ax = ay = ac = 0.0

        if kpts_b is not None and ki < len(kpts_b):
            try:
                bx = float(kpts_b[ki, 0])
                by = float(kpts_b[ki, 1])
                bc = float(kpts_b[ki, 2])
                valid_b = math.isfinite(bx) and math.isfinite(by) and bc >= conf_thr and valid_geom_b
            except Exception:
                bx = by = bc = 0.0

        a_local_x = (ax - ca_x) / s_a if valid_a else 0.0
        a_local_y = (ay - ca_y) / s_a if valid_a else 0.0
        b_local_x = (bx - cb_x) / s_b if valid_b else 0.0
        b_local_y = (by - cb_y) / s_b if valid_b else 0.0
        delta_x = (ax - bx) / pair_scale if valid_a and valid_b else 0.0
        delta_y = (ay - by) / pair_scale if valid_a and valid_b else 0.0

        out[col : col + FEATURES_PER_KEYPOINT] = [
            a_local_x,
            a_local_y,
            ac,
            b_local_x,
            b_local_y,
            bc,
            delta_x,
            delta_y,
        ]
        col += FEATURES_PER_KEYPOINT

    if valid_geom_a and valid_geom_b:
        out[col] = (ca_x - cb_x) / pair_scale
        out[col + 1] = (ca_y - cb_y) / pair_scale

    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def swap_pair_sequence(x_np: np.ndarray) -> np.ndarray:
    if x_np is None:
        return None
    if x_np.ndim != 2 or x_np.shape[1] < GLOBAL_FEATURES:
        return np.array(x_np, copy=True)
    if (x_np.shape[1] - GLOBAL_FEATURES) % FEATURES_PER_KEYPOINT != 0:
        return np.array(x_np, copy=True)

    out = np.empty_like(x_np)
    num_keypoints = (x_np.shape[1] - GLOBAL_FEATURES) // FEATURES_PER_KEYPOINT
    for j in range(num_keypoints):
        c = FEATURES_PER_KEYPOINT * j
        out[:, c : c + 6] = x_np[:, [c + 3, c + 4, c + 5, c + 0, c + 1, c + 2]]
        out[:, c + 6] = -x_np[:, c + 6]
        out[:, c + 7] = -x_np[:, c + 7]

    out[:, -2] = -x_np[:, -2]
    out[:, -1] = -x_np[:, -1]
    return out
