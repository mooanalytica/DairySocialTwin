from __future__ import annotations

import argparse
import json
import mimetypes
import re
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse
from xml.sax.saxutils import escape

try:
    from shapely.geometry import Polygon as ShapelyPolygon
except Exception:  # pragma: no cover - optional dependency
    ShapelyPolygon = None


ROOT = Path(__file__).resolve().parent
STATIC_DIR = ROOT / "static"
OUTPUT_ROOT = ROOT / "output15"

SOURCE_OUTPUT_ROOT = Path("/home/hyw/FloorPlanAnnoEN/output15")
SOURCE_MANIFEST_FILE = SOURCE_OUTPUT_ROOT / "floorplan_groups.json"

FRAME_NAME = "reference_frame.png"
FRAME_META_NAME = "reference_frame_meta.json"
ZONES_NAME = "sna_zones.json"
PREVIEW_NAME = "sna_zones_preview.svg"
CONVENTION_NAME = "SNA_ZONES_CONVENTION.md"

SCHEMA_VERSION = "sna_zones.v1"
COORDINATE_SYSTEM = "reference_frame_pixel"
GROUP_ID_RE = re.compile(r"^farm_ID_(?P<farm>\d+)_camera_ID_(?P<camera>\d+)$")
SLUG_RE = re.compile(r"^[a-z][a-z0-9_]*$")
COLOR_RE = re.compile(r"^#[0-9a-fA-F]{6}$")
DEFAULT_COLORS = [
    "#0b7285",
    "#5f3dc4",
    "#c92a2a",
    "#2f9e44",
    "#e67700",
    "#1864ab",
    "#862e9c",
    "#087f5b",
]
GEOM_EPS = 1.0e-9


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
    path.write_text(content, encoding="utf-8")


def image_size_from_png(path: Path) -> tuple[int | None, int | None]:
    if not path.exists():
        return None, None
    with path.open("rb") as handle:
        header = handle.read(24)
    if len(header) < 24 or header[:8] != b"\x89PNG\r\n\x1a\n":
        return None, None
    return int.from_bytes(header[16:20], "big"), int.from_bytes(header[20:24], "big")


def source_path_text(path: Path) -> str:
    return str(path).replace("\\", "/")


def rel_to_root(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def slugify(value: Any, fallback: str) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    if not text:
        text = fallback
    if not text[0].isalpha():
        text = f"{fallback}_{text}"
    return text


def expected_group_records() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for farm_id in range(1, 4):
        for camera_id in range(1, 6):
            group_id = f"farm_ID_{farm_id}_camera_ID_{camera_id}"
            records.append(
                {
                    "id": group_id,
                    "farmId": farm_id,
                    "cameraId": camera_id,
                    "label": f"Farm {farm_id} / Camera {camera_id}",
                }
            )
    return records


def validate_group_id(group_id: str) -> str:
    if not GROUP_ID_RE.fullmatch(group_id):
        raise ValueError(f"invalid group id: {group_id!r}")
    return group_id


class GroupPaths:
    def __init__(self, group_id: str) -> None:
        self.group_id = validate_group_id(group_id)
        self.local_output_dir = OUTPUT_ROOT / self.group_id
        self.source_output_dir = SOURCE_OUTPUT_ROOT / self.group_id
        self.frame = self.source_output_dir / FRAME_NAME
        self.frame_meta = self.source_output_dir / FRAME_META_NAME
        self.zones = self.local_output_dir / ZONES_NAME
        self.preview = self.local_output_dir / PREVIEW_NAME
        self.convention = self.local_output_dir / CONVENTION_NAME

    @property
    def local_output_rel(self) -> str:
        return rel_to_root(self.local_output_dir)


def read_manifest_groups() -> list[dict[str, Any]]:
    manifest = read_json(SOURCE_MANIFEST_FILE, {})
    groups = manifest.get("groups") if isinstance(manifest, dict) else None
    if not isinstance(groups, list):
        return []
    cleaned: list[dict[str, Any]] = []
    for group in groups:
        if not isinstance(group, dict):
            continue
        group_id = str(group.get("id") or "")
        if GROUP_ID_RE.fullmatch(group_id):
            cleaned.append(group)
    return cleaned


def group_sort_key(group: dict[str, Any]) -> tuple[int, int, str]:
    match = GROUP_ID_RE.fullmatch(str(group.get("id", "")))
    if not match:
        return 999, 999, str(group.get("id", ""))
    return int(match.group("farm")), int(match.group("camera")), str(group["id"])


def all_group_records() -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {group["id"]: dict(group) for group in expected_group_records()}
    for group in read_manifest_groups():
        merged = by_id.get(group["id"], {})
        merged.update(group)
        by_id[group["id"]] = merged

    if SOURCE_OUTPUT_ROOT.exists():
        for group_dir in sorted(SOURCE_OUTPUT_ROOT.glob("farm_ID_*_camera_ID_*")):
            if group_dir.is_dir() and GROUP_ID_RE.fullmatch(group_dir.name):
                by_id.setdefault(group_dir.name, {"id": group_dir.name})

    records = sorted(by_id.values(), key=group_sort_key)
    for record in records:
        paths = GroupPaths(record["id"])
        match = GROUP_ID_RE.fullmatch(record["id"])
        if match:
            record.setdefault("farmId", int(match.group("farm")))
            record.setdefault("cameraId", int(match.group("camera")))
        record.setdefault("label", f"Farm {record.get('farmId')} / Camera {record.get('cameraId')}")
        record["sourceOutputDir"] = source_path_text(paths.source_output_dir)
        record["localOutputDir"] = paths.local_output_rel
        record["frameExists"] = paths.frame.exists()
        record["zonesExists"] = paths.zones.exists()
        record["zoneCount"] = count_saved_zones(paths.zones)
    return records


def count_saved_zones(path: Path) -> int:
    data = read_json(path, {})
    zones = data.get("zones") if isinstance(data, dict) else None
    return len(zones) if isinstance(zones, list) else 0


def resolve_group_id(requested_group_id: str | None = None) -> str:
    if requested_group_id:
        return validate_group_id(requested_group_id)
    records = all_group_records()
    for record in records:
        if record.get("frameExists"):
            return str(record["id"])
    return str(records[0]["id"]) if records else "farm_ID_1_camera_ID_1"


def record_for_group(group_id: str) -> dict[str, Any]:
    for record in all_group_records():
        if record["id"] == group_id:
            return record
    match = GROUP_ID_RE.fullmatch(group_id)
    farm_id = int(match.group("farm")) if match else None
    camera_id = int(match.group("camera")) if match else None
    paths = GroupPaths(group_id)
    return {
        "id": group_id,
        "farmId": farm_id,
        "cameraId": camera_id,
        "label": f"Farm {farm_id} / Camera {camera_id}",
        "sourceOutputDir": source_path_text(paths.source_output_dir),
        "localOutputDir": paths.local_output_rel,
        "frameExists": paths.frame.exists(),
        "zonesExists": paths.zones.exists(),
        "zoneCount": count_saved_zones(paths.zones),
    }


def coordinate_system(paths: GroupPaths, frame_meta: dict[str, Any] | None = None) -> dict[str, Any]:
    frame_meta = frame_meta or read_json(paths.frame_meta, {})
    return {
        "name": COORDINATE_SYSTEM,
        "origin": f"top-left corner of {source_path_text(paths.frame)}",
        "xAxis": "right",
        "yAxis": "down",
        "unit": "pixel",
        "perspectiveCorrected": False,
        "imageToPlaneTransform": "identity",
        "rotationApplied": bool(frame_meta.get("rotationApplied", False)),
        "scaleApplied": bool(frame_meta.get("scaleApplied", False)),
        "notes": [
            "SNA zones are drawn on the clean reference_frame.png.",
            "No resize, crop, perspective correction, or homography is applied by this tool.",
            "Downstream cattle positions must be mapped into this same reference-frame pixel coordinate system.",
        ],
    }


def base_zone_doc(paths: GroupPaths, created_at: str | None = None) -> dict[str, Any]:
    width, height = image_size_from_png(paths.frame)
    frame_meta = read_json(paths.frame_meta, {})
    group = record_for_group(paths.group_id)
    farm_id = int(group.get("farmId") or 0)
    camera_id = int(group.get("cameraId") or 0)
    return {
        "schemaVersion": SCHEMA_VERSION,
        "createdAt": created_at or utc_now(),
        "updatedAt": utc_now(),
        "groupId": paths.group_id,
        "farm": str(farm_id),
        "camera": f"Gopro{camera_id}",
        "farmId": farm_id,
        "cameraId": camera_id,
        "coordinate_system": COORDINATE_SYSTEM,
        "coordinateSystem": coordinate_system(paths, frame_meta),
        "image": {
            "file": source_path_text(paths.frame),
            "width": width,
            "height": height,
            "sourceOutputRoot": source_path_text(SOURCE_OUTPUT_ROOT),
            "sourceVideo": frame_meta.get("sourceVideo"),
            "sourceTimestamp": frame_meta.get("timestamp"),
            "frameExtraction": frame_meta,
        },
        "zones": [],
    }


def load_zones(paths: GroupPaths) -> dict[str, Any]:
    data = read_json(paths.zones, None)
    if data is None:
        return base_zone_doc(paths)
    if not isinstance(data, dict):
        return base_zone_doc(paths)
    base = base_zone_doc(paths, str(data.get("createdAt") or utc_now()))
    zones = data.get("zones") if isinstance(data.get("zones"), list) else []
    base["updatedAt"] = str(data.get("updatedAt") or base["updatedAt"])
    base["zones"] = zones
    return base


def polygon_area(points: list[list[float]]) -> float:
    area = 0.0
    for index, point in enumerate(points):
        next_point = points[(index + 1) % len(points)]
        area += point[0] * next_point[1] - next_point[0] * point[1]
    return area / 2.0


def bbox(points: list[list[float]]) -> tuple[float, float, float, float]:
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return min(xs), min(ys), max(xs), max(ys)


def bboxes_overlap(a: list[list[float]], b: list[list[float]]) -> bool:
    ax1, ay1, ax2, ay2 = bbox(a)
    bx1, by1, bx2, by2 = bbox(b)
    return ax1 < bx2 - GEOM_EPS and bx1 < ax2 - GEOM_EPS and ay1 < by2 - GEOM_EPS and by1 < ay2 - GEOM_EPS


def orientation(a: list[float], b: list[float], c: list[float]) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def point_on_segment(point: list[float], a: list[float], b: list[float]) -> bool:
    if abs(orientation(a, b, point)) > GEOM_EPS:
        return False
    return (
        min(a[0], b[0]) - GEOM_EPS <= point[0] <= max(a[0], b[0]) + GEOM_EPS
        and min(a[1], b[1]) - GEOM_EPS <= point[1] <= max(a[1], b[1]) + GEOM_EPS
    )


def proper_segment_intersection(a: list[float], b: list[float], c: list[float], d: list[float]) -> bool:
    o1 = orientation(a, b, c)
    o2 = orientation(a, b, d)
    o3 = orientation(c, d, a)
    o4 = orientation(c, d, b)
    return o1 * o2 < -GEOM_EPS and o3 * o4 < -GEOM_EPS


def point_in_polygon_strict(point: list[float], polygon: list[list[float]]) -> bool:
    for index, current in enumerate(polygon):
        previous = polygon[index - 1]
        if point_on_segment(point, previous, current):
            return False
    inside = False
    x, y = point
    for index, current in enumerate(polygon):
        previous = polygon[index - 1]
        yi = current[1]
        yj = previous[1]
        if (yi > y) != (yj > y):
            x_intersect = (previous[0] - current[0]) * (y - yi) / (yj - yi + GEOM_EPS) + current[0]
            if x < x_intersect:
                inside = not inside
    return inside


def point_in_or_on_polygon(point: list[float], polygon: list[list[float]]) -> bool:
    if point_in_polygon_strict(point, polygon):
        return True
    return any(point_on_segment(point, polygon[index - 1], current) for index, current in enumerate(polygon))


def polygons_overlap_positive(a: list[list[float]], b: list[list[float]]) -> bool:
    if not bboxes_overlap(a, b):
        return False

    if ShapelyPolygon is not None:
        try:
            poly_a = ShapelyPolygon(a)
            poly_b = ShapelyPolygon(b)
            if not poly_a.is_valid:
                poly_a = poly_a.buffer(0)
            if not poly_b.is_valid:
                poly_b = poly_b.buffer(0)
            return poly_a.intersection(poly_b).area > 1.0e-6
        except Exception:
            pass

    for index_a, point_a in enumerate(a):
        next_a = a[(index_a + 1) % len(a)]
        for index_b, point_b in enumerate(b):
            next_b = b[(index_b + 1) % len(b)]
            if proper_segment_intersection(point_a, next_a, point_b, next_b):
                return True

    if any(point_in_polygon_strict(point, b) for point in a):
        return True
    if any(point_in_polygon_strict(point, a) for point in b):
        return True
    if all(point_in_or_on_polygon(point, b) for point in a) and abs(polygon_area(a)) > GEOM_EPS:
        return True
    if all(point_in_or_on_polygon(point, a) for point in b) and abs(polygon_area(b)) > GEOM_EPS:
        return True
    return False


def validate_zones(data: dict[str, Any], paths: GroupPaths) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise ValueError("zone annotation must be a JSON object")
    zones = data.get("zones")
    if not isinstance(zones, list):
        raise ValueError("zones must be a list")

    width, height = image_size_from_png(paths.frame)
    used_ids: set[str] = set()
    cleaned_zones: list[dict[str, Any]] = []

    for index, zone in enumerate(zones):
        if not isinstance(zone, dict):
            raise ValueError(f"zone {index} must be an object")
        zone_type = slugify(zone.get("zone_type") or zone.get("type") or zone.get("label"), "zone")
        if not SLUG_RE.fullmatch(zone_type):
            raise ValueError(f"zone {index} has invalid zone_type: {zone_type!r}")

        zone_id = slugify(zone.get("zone_id") or f"{zone_type}_{index + 1}", zone_type)
        original_zone_id = zone_id
        suffix = 2
        while zone_id in used_ids:
            zone_id = f"{original_zone_id}_{suffix}"
            suffix += 1
        used_ids.add(zone_id)

        label = str(zone.get("label") or zone_type)
        color = str(zone.get("color") or DEFAULT_COLORS[index % len(DEFAULT_COLORS)])
        if not COLOR_RE.fullmatch(color):
            raise ValueError(f"zone {index} has invalid color: {color!r}")
        color = color.lower()

        polygon = zone.get("polygon", zone.get("points"))
        if not isinstance(polygon, list) or len(polygon) < 3:
            raise ValueError(f"zone {index} must contain at least 3 polygon points")
        cleaned_points: list[list[float]] = []
        for point_index, point in enumerate(polygon):
            if (
                not isinstance(point, list | tuple)
                or len(point) != 2
                or not isinstance(point[0], int | float)
                or not isinstance(point[1], int | float)
            ):
                raise ValueError(f"zone {index} point {point_index} is invalid")
            x = round(float(point[0]), 2)
            y = round(float(point[1]), 2)
            if width is not None and not -1 <= x <= width + 1:
                raise ValueError(f"zone {index} point {point_index} x is outside the frame")
            if height is not None and not -1 <= y <= height + 1:
                raise ValueError(f"zone {index} point {point_index} y is outside the frame")
            cleaned_points.append([x, y])

        if abs(polygon_area(cleaned_points)) < 1.0:
            raise ValueError(f"zone {index} polygon area is too small")

        cleaned_zones.append(
            {
                "zone_id": zone_id,
                "zone_type": zone_type,
                "label": label,
                "color": color,
                "polygon": cleaned_points,
            }
        )

    for left_index, left_zone in enumerate(cleaned_zones):
        for right_index in range(left_index + 1, len(cleaned_zones)):
            right_zone = cleaned_zones[right_index]
            if polygons_overlap_positive(left_zone["polygon"], right_zone["polygon"]):
                raise ValueError(
                    f"zones overlap with positive area: {left_zone['zone_id']} and {right_zone['zone_id']}"
                )

    saved = base_zone_doc(paths, str(data.get("createdAt") or utc_now()))
    saved["updatedAt"] = utc_now()
    saved["zones"] = cleaned_zones
    return saved


def write_preview_svg(annotation: dict[str, Any], paths: GroupPaths) -> None:
    image = annotation.get("image", {})
    width = image.get("width") or 0
    height = image.get("height") or 0
    if not width or not height:
        return
    try:
        frame_href = paths.frame.resolve().as_uri()
    except ValueError:
        frame_href = source_path_text(paths.frame)
    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<image href="{escape(frame_href)}" x="0" y="0" width="100%" height="100%" preserveAspectRatio="none"/>',
    ]
    for zone in annotation.get("zones", []):
        points = " ".join(f'{point[0]},{point[1]}' for point in zone["polygon"])
        color = zone.get("color") or "#0b7285"
        title = f"{zone.get('label') or zone.get('zone_type')} ({zone.get('zone_id')})"
        parts.append(f"<title>{escape(title)}</title>")
        parts.append(
            f'<polygon points="{points}" fill="{color}" fill-opacity="0.24" '
            f'stroke="{color}" stroke-opacity="0.95" stroke-width="5" />'
        )
    parts.append("</svg>")
    paths.preview.parent.mkdir(parents=True, exist_ok=True)
    paths.preview.write_text("\n".join(parts) + "\n", encoding="utf-8")


def write_convention_doc(annotation: dict[str, Any], paths: GroupPaths) -> None:
    image = annotation.get("image", {})
    zones = annotation.get("zones", [])
    zone_types = sorted({str(zone.get("zone_type")) for zone in zones})
    lines = [
        "# SNA Zone Coordinate Convention",
        "",
        "This file is generated by the SNA zone annotation UI when zones are saved.",
        "",
        "## Coordinate system",
        "",
        f"- Schema: `{SCHEMA_VERSION}`",
        f"- Group: `{paths.group_id}`",
        f"- Zone file: `{rel_to_root(paths.zones)}`",
        f"- Reference frame: `{source_path_text(paths.frame)}`",
        f"- Reference frame size: `{image.get('width')}` x `{image.get('height')}` pixels",
        "- Coordinate system: `reference_frame_pixel`",
        "- Origin: top-left corner of the reference frame",
        "- X axis: positive to the right",
        "- Y axis: positive downward",
        "- Unit: reference-frame pixel",
        "- Perspective correction: none",
        "- Image-to-plane transform: identity",
        "",
        "## SNA zones",
        "",
        "- Zones are independent from floorplan annotations and Stage2 floorplan generation.",
        "- `zone_id` and `zone_type` are stable ASCII slugs.",
        "- `label` may use any language.",
        "- Positive-area overlap between zones is not allowed. Shared borders are allowed.",
        "- Points outside all zones should be assigned to `other` by the SNA pipeline.",
        "- Missing cattle coordinates should be assigned to `unknown` by the SNA pipeline.",
        "",
        "## Current zone summary",
        "",
        f"- Zone count: `{len(zones)}`",
        f"- Zone types: `{', '.join(zone_types) if zone_types else 'none'}`",
        "",
    ]
    paths.convention.parent.mkdir(parents=True, exist_ok=True)
    paths.convention.write_text("\n".join(lines), encoding="utf-8")


class SnaZoneHandler(BaseHTTPRequestHandler):
    server_version = "SNAZoneAnno/1.0"

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/":
                self.send_file(STATIC_DIR / "index.html", "text/html; charset=utf-8")
            elif parsed.path == "/api/groups":
                self.send_json({"groups": all_group_records()})
            elif parsed.path == "/api/config":
                self.send_json(self.config_payload(parsed))
            elif parsed.path == "/api/health":
                group_id = self.group_id_from_request(parsed)
                paths = GroupPaths(group_id)
                self.send_json(
                    {
                        "ok": True,
                        "groupId": group_id,
                        "frameExists": paths.frame.exists(),
                        "zonesExists": paths.zones.exists(),
                    }
                )
            elif parsed.path == "/frame":
                group_id = self.group_id_from_request(parsed)
                self.send_file(GroupPaths(group_id).frame, "image/png")
            elif parsed.path.startswith("/static/"):
                static_path = (STATIC_DIR / parsed.path.removeprefix("/static/")).resolve()
                if STATIC_DIR.resolve() not in static_path.parents and static_path != STATIC_DIR.resolve():
                    self.send_error(HTTPStatus.FORBIDDEN)
                    return
                self.send_file(static_path)
            else:
                self.send_error(HTTPStatus.NOT_FOUND)
        except ValueError as exc:
            self.send_json({"ok": False, "error": str(exc)}, status=HTTPStatus.BAD_REQUEST)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/api/save":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length > 20_000_000:
                raise ValueError("request body is too large")
            raw = self.rfile.read(length)
            payload = json.loads(raw.decode("utf-8"))
            requested_group_id = self.query_group_id(parsed) or payload.get("groupId")
            group_id = resolve_group_id(str(requested_group_id) if requested_group_id else None)
            paths = GroupPaths(group_id)
            annotation = validate_zones(payload, paths)
            write_json(paths.zones, annotation)
            write_preview_svg(annotation, paths)
            write_convention_doc(annotation, paths)
            self.send_json(
                {
                    "ok": True,
                    "groupId": group_id,
                    "zonesPath": str(paths.zones),
                    "previewPath": str(paths.preview),
                    "conventionPath": str(paths.convention),
                    "updatedAt": annotation["updatedAt"],
                }
            )
        except Exception as exc:
            self.send_json({"ok": False, "error": str(exc)}, status=HTTPStatus.BAD_REQUEST)

    def query_group_id(self, parsed: Any) -> str | None:
        values = parse_qs(parsed.query).get("group")
        if not values:
            return None
        return values[0]

    def group_id_from_request(self, parsed: Any) -> str:
        requested = self.query_group_id(parsed)
        if requested:
            return resolve_group_id(requested)
        default_group_id = getattr(self.server, "default_group_id", None)
        return resolve_group_id(default_group_id)

    def config_payload(self, parsed: Any) -> dict[str, Any]:
        group_id = self.group_id_from_request(parsed)
        paths = GroupPaths(group_id)
        zones = load_zones(paths)
        frame_mtime = int(paths.frame.stat().st_mtime) if paths.frame.exists() else 0
        group_record = record_for_group(group_id)
        return {
            "schemaVersion": SCHEMA_VERSION,
            "coordinateSystem": COORDINATE_SYSTEM,
            "groups": all_group_records(),
            "activeGroupId": group_id,
            "activeGroup": group_record,
            "annotation": zones,
            "defaultColors": DEFAULT_COLORS,
            "frame": {
                "exists": paths.frame.exists(),
                "url": f"/frame?group={group_id}&mtime={frame_mtime}",
                "path": str(paths.frame),
                "metadataPath": str(paths.frame_meta),
                "metadata": read_json(paths.frame_meta, {}),
            },
            "paths": {
                "sourceOutputRoot": str(SOURCE_OUTPUT_ROOT),
                "sourceOutputDir": str(paths.source_output_dir),
                "localOutputRoot": str(OUTPUT_ROOT),
                "localOutputDir": str(paths.local_output_dir),
                "zones": str(paths.zones),
                "preview": str(paths.preview),
                "convention": str(paths.convention),
            },
        }

    def send_file(self, path: Path, content_type: str | None = None) -> None:
        if not path.exists() or not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        if content_type is None:
            content_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        content = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(content)

    def send_json(self, data: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        content = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(content)

    def log_message(self, format: str, *args: Any) -> None:
        print(f"{self.log_date_time_string()} {self.address_string()} {format % args}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the SNA zone annotation UI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--group",
        default=None,
        help="Default group id, such as farm_ID_1_camera_ID_1. The UI can switch groups.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    default_group_id = resolve_group_id(args.group)
    address = (args.host, args.port)
    httpd = ThreadingHTTPServer(address, SnaZoneHandler)
    httpd.default_group_id = default_group_id  # type: ignore[attr-defined]
    print(f"SNA Zone Annotation UI: http://{args.host}:{args.port}/?group={default_group_id}")
    if not SOURCE_OUTPUT_ROOT.exists():
        print(f"Clean frame source is missing: {SOURCE_OUTPUT_ROOT}")
    elif not GroupPaths(default_group_id).frame.exists():
        print(f"Reference frame is missing for {default_group_id}: {GroupPaths(default_group_id).frame}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping SNA Zone Annotation UI.")


if __name__ == "__main__":
    main()
