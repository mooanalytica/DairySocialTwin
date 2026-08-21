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


ROOT = Path(__file__).resolve().parent
STATIC_DIR = ROOT / "static"
OUTPUT_ROOT = ROOT / "output15"
MANIFEST_FILE = OUTPUT_ROOT / "floorplan_groups.json"

FRAME_NAME = "reference_frame.png"
FRAME_META_NAME = "reference_frame_meta.json"
ANNOTATION_NAME = "floorplan_annotation.json"
PREVIEW_NAME = "floorplan_annotation_preview.svg"
CONVENTION_NAME = "COORDINATE_CONVENTION.md"

SOURCE_VIDEO_ROOT = Path(r"F:\FULLDATA\Dairy Farm Videos")
PREVIEW_FRAME_ROOT = Path(r"F:\DROPBOX_FRAME")

GROUP_ID_RE = re.compile(r"^farm_ID_(?P<farm>\d+)_camera_ID_(?P<camera>\d+)$")

CLASSES = [
    {
        "id": "walkable_ground",
        "label": "possible dairy cattle ground",
        "uiLabel": "Ground",
        "color": "#1e63ff",
    },
    {
        "id": "obstacle",
        "label": "obstacle / not traversable",
        "uiLabel": "Obstacle",
        "color": "#e03131",
    },
    {
        "id": "resource",
        "label": "resource",
        "uiLabel": "Resource",
        "color": "#2f9e44",
    },
]
CLASS_IDS = {item["id"] for item in CLASSES}
OVERLAP_PRIORITY_CLASS_IDS = ["resource", "obstacle", "walkable_ground"]
RECTANGLE_FIT_METHOD = "minimum_area_rotated_bounding_rectangle_from_polygon_points"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def rel_path(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


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
        self.output_dir = OUTPUT_ROOT / self.group_id
        self.frame = self.output_dir / FRAME_NAME
        self.frame_meta = self.output_dir / FRAME_META_NAME
        self.annotation = self.output_dir / ANNOTATION_NAME
        self.preview = self.output_dir / PREVIEW_NAME
        self.convention = self.output_dir / CONVENTION_NAME

    @property
    def frame_rel(self) -> str:
        return rel_path(self.frame)

    @property
    def annotation_rel(self) -> str:
        return rel_path(self.annotation)

    @property
    def preview_rel(self) -> str:
        return rel_path(self.preview)

    @property
    def convention_rel(self) -> str:
        return rel_path(self.convention)


def read_manifest_groups() -> list[dict[str, Any]]:
    manifest = read_json(MANIFEST_FILE, {})
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

    if OUTPUT_ROOT.exists():
        for group_dir in sorted(OUTPUT_ROOT.glob("farm_ID_*_camera_ID_*")):
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
        record["outputDir"] = rel_path(paths.output_dir)
        record["frameExists"] = paths.frame.exists()
        record["annotationExists"] = paths.annotation.exists()
    return records


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
    paths = GroupPaths(group_id)
    match = GROUP_ID_RE.fullmatch(group_id)
    farm_id = int(match.group("farm")) if match else None
    camera_id = int(match.group("camera")) if match else None
    return {
        "id": group_id,
        "farmId": farm_id,
        "cameraId": camera_id,
        "label": f"Farm {farm_id} / Camera {camera_id}",
        "outputDir": rel_path(paths.output_dir),
        "frameExists": paths.frame.exists(),
        "annotationExists": paths.annotation.exists(),
    }


def coordinate_system(paths: GroupPaths, frame_meta: dict[str, Any] | None = None) -> dict[str, Any]:
    frame_meta = frame_meta or read_json(paths.frame_meta, {})
    return {
        "name": "reference_frame_pixel",
        "origin": f"top-left corner of {paths.frame_rel}",
        "xAxis": "right",
        "yAxis": "down",
        "unit": "pixel",
        "perspectiveCorrected": False,
        "imageToPlaneTransform": "identity",
        "rotationApplied": bool(frame_meta.get("rotationApplied", False)),
        "scaleApplied": bool(frame_meta.get("scaleApplied", False)),
        "notes": [
            "Coordinates are measured on the extracted PNG frame.",
            "No top-down homography is applied by this tool.",
            "Use walkable_ground polygons as the valid region for cattle points.",
        ],
    }


def plane_geometry(force_2d_rectangle: bool) -> dict[str, Any]:
    if force_2d_rectangle:
        return {
            "type": "rectangle",
            "enforcedIn2D": True,
            "rectangleFitMethod": RECTANGLE_FIT_METHOD,
        }
    return {
        "type": "polygon",
        "enforcedIn2D": False,
    }


def empty_annotation(paths: GroupPaths) -> dict[str, Any]:
    width, height = image_size_from_png(paths.frame)
    frame_meta = read_json(paths.frame_meta, {})
    return {
        "schemaVersion": "floorplananno.v1",
        "createdAt": utc_now(),
        "updatedAt": utc_now(),
        "groupId": paths.group_id,
        "image": {
            "file": paths.frame_rel,
            "width": width,
            "height": height,
            "sourceVideo": frame_meta.get("sourceVideo"),
            "sourceTimestamp": frame_meta.get("timestamp"),
            "frameExtraction": frame_meta,
        },
        "coordinateSystem": coordinate_system(paths, frame_meta),
        "classes": CLASSES,
        "defaultUnannotatedClassId": None,
        "overlapPriorityClassIds": OVERLAP_PRIORITY_CLASS_IDS,
        "polygons": [],
    }


def load_annotation(paths: GroupPaths) -> dict[str, Any]:
    data = read_json(paths.annotation, None)
    if data is None:
        return empty_annotation(paths)
    data.setdefault("schemaVersion", "floorplananno.v1")
    data["groupId"] = paths.group_id
    data.setdefault("classes", CLASSES)
    data.setdefault("coordinateSystem", coordinate_system(paths))
    data.setdefault("defaultUnannotatedClassId", None)
    data.setdefault("overlapPriorityClassIds", OVERLAP_PRIORITY_CLASS_IDS)
    data.setdefault("polygons", [])
    return data


def validate_annotation(data: dict[str, Any], paths: GroupPaths) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise ValueError("annotation must be a JSON object")
    polygons = data.get("polygons")
    if not isinstance(polygons, list):
        raise ValueError("polygons must be a list")

    width, height = image_size_from_png(paths.frame)
    default_unannotated_class_id = data.get("defaultUnannotatedClassId")
    if default_unannotated_class_id is not None and default_unannotated_class_id not in CLASS_IDS:
        raise ValueError(
            f"defaultUnannotatedClassId must be one of {sorted(CLASS_IDS)} or null"
        )
    overlap_priority_class_ids = data.get("overlapPriorityClassIds", OVERLAP_PRIORITY_CLASS_IDS)
    if overlap_priority_class_ids != OVERLAP_PRIORITY_CLASS_IDS:
        raise ValueError(
            "overlapPriorityClassIds must be ['resource', 'obstacle', 'walkable_ground']"
        )

    cleaned_polygons: list[dict[str, Any]] = []
    for index, polygon in enumerate(polygons):
        if not isinstance(polygon, dict):
            raise ValueError(f"polygon {index} must be an object")
        class_id = polygon.get("classId")
        if class_id not in CLASS_IDS:
            raise ValueError(f"polygon {index} has unknown classId: {class_id!r}")
        points = polygon.get("points")
        if not isinstance(points, list) or len(points) < 3:
            raise ValueError(f"polygon {index} must contain at least 3 points")
        plane_shape = polygon.get("planeGeometry")
        force_2d_rectangle = bool(polygon.get("force2DRectangle", False))
        if isinstance(plane_shape, dict):
            if plane_shape.get("type") == "rectangle":
                force_2d_rectangle = True
            elif plane_shape.get("type") == "polygon":
                force_2d_rectangle = False

        cleaned_points: list[list[float]] = []
        for point_index, point in enumerate(points):
            if (
                not isinstance(point, list | tuple)
                or len(point) != 2
                or not isinstance(point[0], int | float)
                or not isinstance(point[1], int | float)
            ):
                raise ValueError(f"polygon {index} point {point_index} is invalid")
            x = round(float(point[0]), 2)
            y = round(float(point[1]), 2)
            if width is not None and not -1 <= x <= width + 1:
                raise ValueError(f"polygon {index} point {point_index} x is outside the frame")
            if height is not None and not -1 <= y <= height + 1:
                raise ValueError(f"polygon {index} point {point_index} y is outside the frame")
            cleaned_points.append([x, y])

        cleaned_polygons.append(
            {
                "id": str(polygon.get("id") or f"poly-{index + 1}"),
                "classId": class_id,
                "label": str(polygon.get("label") or class_id),
                "color": class_color(class_id),
                "points": cleaned_points,
                "closed": True,
                "force2DRectangle": force_2d_rectangle,
                "planeGeometry": plane_geometry(force_2d_rectangle),
            }
        )

    frame_meta = read_json(paths.frame_meta, {})
    return {
        "schemaVersion": "floorplananno.v1",
        "createdAt": str(data.get("createdAt") or utc_now()),
        "updatedAt": utc_now(),
        "groupId": paths.group_id,
        "image": {
            "file": paths.frame_rel,
            "width": width,
            "height": height,
            "sourceVideo": frame_meta.get("sourceVideo"),
            "sourceTimestamp": frame_meta.get("timestamp"),
            "frameExtraction": frame_meta,
        },
        "coordinateSystem": coordinate_system(paths, frame_meta),
        "classes": CLASSES,
        "defaultUnannotatedClassId": default_unannotated_class_id,
        "overlapPriorityClassIds": OVERLAP_PRIORITY_CLASS_IDS,
        "polygons": cleaned_polygons,
    }


def class_color(class_id: str) -> str:
    for item in CLASSES:
        if item["id"] == class_id:
            return item["color"]
    return "#868e96"


def class_label(class_id: str) -> str:
    for item in CLASSES:
        if item["id"] == class_id:
            return item["label"]
    return class_id


def write_preview_svg(annotation: dict[str, Any], paths: GroupPaths) -> None:
    width = annotation.get("image", {}).get("width") or 0
    height = annotation.get("image", {}).get("height") or 0
    if not width or not height:
        return
    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<image href="{FRAME_NAME}" x="0" y="0" width="100%" height="100%" preserveAspectRatio="none"/>',
    ]
    default_class_id = annotation.get("defaultUnannotatedClassId")
    if default_class_id:
        default_color = class_color(default_class_id)
        parts.append(
            f'<rect x="0" y="0" width="{width}" height="{height}" '
            f'fill="{default_color}" fill-opacity="0.14" />'
        )
    for polygon in annotation.get("polygons", []):
        points = " ".join(f'{point[0]},{point[1]}' for point in polygon["points"])
        color = polygon.get("color") or class_color(polygon["classId"])
        label = class_label(polygon["classId"])
        if polygon.get("force2DRectangle"):
            label = f"{label} / 2D rectangle"
        parts.append(f"<title>{label}</title>")
        stroke_dash = ' stroke-dasharray="18 10"' if polygon.get("force2DRectangle") else ""
        parts.append(
            f'<polygon points="{points}" fill="{color}" fill-opacity="0.24" '
            f'stroke="{color}" stroke-opacity="0.95" stroke-width="5"{stroke_dash} />'
        )
    parts.append("</svg>")
    paths.preview.write_text("\n".join(parts) + "\n", encoding="utf-8")


def write_convention_doc(annotation: dict[str, Any], paths: GroupPaths) -> None:
    image = annotation.get("image", {})
    polygons = annotation.get("polygons", [])
    default_class_id = annotation.get("defaultUnannotatedClassId")
    rectangle_count = sum(1 for polygon in polygons if polygon.get("force2DRectangle"))
    counts = {class_item["id"]: 0 for class_item in CLASSES}
    for polygon in polygons:
        counts[polygon["classId"]] = counts.get(polygon["classId"], 0) + 1

    lines = [
        "# FloorPlanAnno Coordinate Convention",
        "",
        "This file is generated by the annotation UI when annotations are saved.",
        "",
        "## Coordinate system",
        "",
        "- Schema: `floorplananno.v1`",
        f"- Group: `{paths.group_id}`",
        f"- Coordinate file: `{paths.annotation_rel}`",
        f"- Reference frame: `{paths.frame_rel}`",
        f"- Reference frame size: `{image.get('width')}` x `{image.get('height')}` pixels",
        "- Origin: top-left corner of the reference frame",
        "- X axis: positive to the right",
        "- Y axis: positive downward",
        "- Unit: reference-frame pixel",
        "- Perspective correction: none",
        "- Image-to-plane transform: identity",
        "- Rotation after decode: none, unless `frameExtraction.rotationApplied` says otherwise",
        "- Scale after decode: none, unless `frameExtraction.scaleApplied` says otherwise",
        "",
        "## Semantic layers",
        "",
        "- `walkable_ground`: blue polygons. Possible ground region for dairy cattle.",
        "- `obstacle`: red polygons. Obstacles, walls, fences, equipment, or other non-traversable regions.",
        "- `resource`: green polygons. Feed, water, gates, resting places, or other manually marked resources.",
        "- Overlap priority: `resource` > `obstacle` > `walkable_ground`.",
        "- Polygon field `force2DRectangle`: when `true`, the polygon must become a rectangle in the 2D floor-plan interpretation.",
        f"- Rectangle fit method: `{RECTANGLE_FIT_METHOD}`.",
        (
            f"- Default unannotated class: `{default_class_id}`. Pixels not covered by any polygon "
            f"should be treated as this class."
            if default_class_id
            else "- Default unannotated class: none. Pixels not covered by a polygon are unlabeled."
        ),
        "",
        "## How later agents should use it",
        "",
        f"1. Read `{paths.annotation_rel}`.",
        f"2. Interpret every point as `[x, y]` in `{paths.frame_rel}` pixel coordinates.",
        "3. Classify overlaps by `overlapPriorityClassIds`: `resource` first, then `obstacle`, then `walkable_ground`.",
        (
            "4. If `defaultUnannotatedClassId` is `walkable_ground`, start with the whole image as "
            "walkable ground, then let explicit polygons override this default by priority."
            if default_class_id == "walkable_ground"
            else "4. Treat pixels not covered by any polygon as unlabeled."
        ),
        "5. Treat final `obstacle` pixels as excluded/non-traversable regions.",
        "6. Treat final `resource` pixels as labeled spatial context, not as mandatory cattle positions.",
        (
            f"7. For polygons where `force2DRectangle` is `true`, derive the 2D shape with "
            f"`{RECTANGLE_FIT_METHOD}` and use that rectangle instead of the raw polygon boundary."
        ),
        "8. If detection points are later added, keep only detected cattle; do not invent missing cattle.",
        "",
        "## Current polygon counts",
        "",
    ]
    for class_item in CLASSES:
        lines.append(f"- `{class_item['id']}`: {counts.get(class_item['id'], 0)}")
    lines.append(f"- `force2DRectangle`: {rectangle_count}")
    lines.extend(
        [
            "",
            "## Source",
            "",
            f"- Source video: `{image.get('sourceVideo')}`",
            f"- Source timestamp: `{image.get('sourceTimestamp')}`",
            "",
        ]
    )
    paths.convention.write_text("\n".join(lines), encoding="utf-8")


class AnnotatorHandler(BaseHTTPRequestHandler):
    server_version = "FloorPlanAnno/2.0"

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
                self.send_json({"ok": True, "groupId": group_id, "frameExists": paths.frame.exists()})
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
            annotation = validate_annotation(payload, paths)
            write_json(paths.annotation, annotation)
            write_preview_svg(annotation, paths)
            write_convention_doc(annotation, paths)
            self.send_json(
                {
                    "ok": True,
                    "groupId": group_id,
                    "annotationPath": str(paths.annotation),
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
        annotation = load_annotation(paths)
        frame_mtime = int(paths.frame.stat().st_mtime) if paths.frame.exists() else 0
        group_record = record_for_group(group_id)
        return {
            "classes": CLASSES,
            "groups": all_group_records(),
            "activeGroupId": group_id,
            "activeGroup": group_record,
            "annotation": annotation,
            "frame": {
                "exists": paths.frame.exists(),
                "url": f"/frame?group={group_id}&mtime={frame_mtime}",
                "path": str(paths.frame),
                "metadataPath": str(paths.frame_meta),
                "metadata": read_json(paths.frame_meta, {}),
            },
            "paths": {
                "sourceRoot": str(SOURCE_VIDEO_ROOT),
                "previewFrameRoot": str(PREVIEW_FRAME_ROOT),
                "sourceVideoDir": str(group_record.get("sourceVideoDir") or ""),
                "sourceVideo": str(group_record.get("sourceVideo") or ""),
                "outputDir": str(paths.output_dir),
                "annotation": str(paths.annotation),
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
    parser = argparse.ArgumentParser(description="Run the FloorPlanAnno annotation UI.")
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
    httpd = ThreadingHTTPServer(address, AnnotatorHandler)
    httpd.default_group_id = default_group_id  # type: ignore[attr-defined]
    print(f"FloorPlanAnno UI: http://{args.host}:{args.port}/?group={default_group_id}")
    if not GroupPaths(default_group_id).frame.exists():
        print("Reference frame is missing. Run: python tools/setup_output15.py")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping FloorPlanAnno UI.")


if __name__ == "__main__":
    main()
