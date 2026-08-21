from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def apply_homography(points: np.ndarray, h_matrix: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    h_matrix = np.asarray(h_matrix, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must have shape (n, 2)")
    if h_matrix.shape != (3, 3):
        raise ValueError("homography matrix must have shape (3, 3)")
    ones = np.ones((points.shape[0], 1), dtype=float)
    homogeneous = np.concatenate([points, ones], axis=1)
    mapped = homogeneous @ h_matrix.T
    denom = mapped[:, 2:3]
    denom[np.abs(denom) < 1.0e-12] = np.nan
    return mapped[:, :2] / denom


def point_on_segment(x: float, y: float, a: tuple[float, float], b: tuple[float, float], tol: float = 1.0e-9) -> bool:
    ax, ay = a
    bx, by = b
    cross = (x - ax) * (by - ay) - (y - ay) * (bx - ax)
    if abs(cross) > tol:
        return False
    dot = (x - ax) * (x - bx) + (y - ay) * (y - by)
    return dot <= tol


def point_in_polygon(x: float, y: float, polygon: Iterable[Iterable[float]]) -> bool:
    pts = [(float(px), float(py)) for px, py in polygon]
    if len(pts) < 3 or not math.isfinite(x) or not math.isfinite(y):
        return False
    inside = False
    for idx, a in enumerate(pts):
        b = pts[(idx + 1) % len(pts)]
        if point_on_segment(x, y, a, b):
            return True
        xi, yi = a
        xj, yj = b
        if (yi > y) != (yj > y):
            x_at_y = (xj - xi) * (y - yi) / ((yj - yi) or 1.0e-12) + xi
            if x < x_at_y:
                inside = not inside
    return inside


def euclidean_distance(x1: float, y1: float, x2: float, y2: float) -> float:
    return float(math.hypot(float(x1) - float(x2), float(y1) - float(y2)))
