from __future__ import annotations

import math
from pathlib import Path


FLOORPLAN_ROOT = Path("/home/hyw/FloorPlanAnnoEN/output15")
Point = tuple[float, float]


def camera_number(camera_id: str) -> str:
    value = str(camera_id).strip()
    if value.lower().startswith("gopro"):
        value = value[5:]
    if not value or not value.isdigit():
        raise ValueError(f"Unsupported camera id: {camera_id}")
    return str(int(value))


def floorplan_group_id(farm_id: str, camera_id: str) -> str:
    farm = str(farm_id).strip()
    if not farm or not farm.isdigit():
        raise ValueError(f"Unsupported farm id: {farm_id}")
    return f"farm_ID_{int(farm)}_camera_ID_{camera_number(camera_id)}"


def floorplan_annotation_file(farm_id: str, camera_id: str) -> Path:
    return FLOORPLAN_ROOT / floorplan_group_id(farm_id, camera_id) / "floorplan_annotation.json"


def convex_hull(points: list[Point]) -> list[Point]:
    unique = sorted(set(points))
    if len(unique) <= 1:
        return unique

    def cross(origin: Point, first: Point, second: Point) -> float:
        return (
            (first[0] - origin[0]) * (second[1] - origin[1])
            - (first[1] - origin[1]) * (second[0] - origin[0])
        )

    lower: list[Point] = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)

    upper: list[Point] = []
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return lower[:-1] + upper[:-1]


def rotate_point(point: Point, angle: float) -> Point:
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    return point[0] * cos_a - point[1] * sin_a, point[0] * sin_a + point[1] * cos_a


def min_area_rectangle(points: list[Point]) -> list[Point]:
    hull = convex_hull(points)
    if len(hull) < 3:
        return points

    best_area = math.inf
    best_rectangle: list[Point] | None = None
    for index, point in enumerate(hull):
        next_point = hull[(index + 1) % len(hull)]
        angle = -math.atan2(next_point[1] - point[1], next_point[0] - point[0])
        rotated = [rotate_point(item, angle) for item in hull]
        min_x = min(item[0] for item in rotated)
        max_x = max(item[0] for item in rotated)
        min_y = min(item[1] for item in rotated)
        max_y = max(item[1] for item in rotated)
        area = (max_x - min_x) * (max_y - min_y)
        if area < best_area:
            best_area = area
            corners = [
                (min_x, min_y),
                (max_x, min_y),
                (max_x, max_y),
                (min_x, max_y),
            ]
            best_rectangle = [rotate_point(item, -angle) for item in corners]
    if best_rectangle is None:
        raise RuntimeError("Unable to derive minimum-area rectangle")
    return best_rectangle


def point_on_segment(point: Point, a: Point, b: Point, tolerance: float = 1e-7) -> bool:
    px, py = point
    ax, ay = a
    bx, by = b
    cross = (px - ax) * (by - ay) - (py - ay) * (bx - ax)
    if abs(cross) > tolerance:
        return False
    dot = (px - ax) * (px - bx) + (py - ay) * (py - by)
    return dot <= tolerance


def point_in_polygon(point: Point, polygon: list[Point]) -> bool:
    inside = False
    x, y = point
    for index, first in enumerate(polygon):
        second = polygon[(index + 1) % len(polygon)]
        if point_on_segment(point, first, second):
            return True
        xi, yi = first
        xj, yj = second
        if (yi > y) != (yj > y):
            x_at_y = (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi
            if x < x_at_y:
                inside = not inside
    return inside


def orientation(a: Point, b: Point, c: Point) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def segments_intersect(
    a: Point,
    b: Point,
    c: Point,
    d: Point,
    tolerance: float = 1e-7,
) -> bool:
    first = orientation(a, b, c)
    second = orientation(a, b, d)
    third = orientation(c, d, a)
    fourth = orientation(c, d, b)
    if abs(first) <= tolerance and point_on_segment(c, a, b, tolerance):
        return True
    if abs(second) <= tolerance and point_on_segment(d, a, b, tolerance):
        return True
    if abs(third) <= tolerance and point_on_segment(a, c, d, tolerance):
        return True
    if abs(fourth) <= tolerance and point_on_segment(b, c, d, tolerance):
        return True
    return (first > 0) != (second > 0) and (third > 0) != (fourth > 0)


def segment_intersects_polygon(a: Point, b: Point, polygon: list[Point]) -> bool:
    if point_in_polygon(a, polygon) or point_in_polygon(b, polygon):
        return True
    for index, point in enumerate(polygon):
        next_point = polygon[(index + 1) % len(polygon)]
        if segments_intersect(a, b, point, next_point):
            return True
    return False
