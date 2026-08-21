from __future__ import annotations

import argparse
import json
import mimetypes
import re
import sys
import threading
import traceback
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

from dashboard_data import DashboardData, DataContractError, ROOT
from dashboard_plotting import FigureRenderer, RenderSupersededError


STATIC_DIR = ROOT / "static"
STATIC_FILES = {
    "/": STATIC_DIR / "index.html",
    "/static/app.css": STATIC_DIR / "app.css",
    "/static/app.js": STATIC_DIR / "app.js",
}


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")


class DashboardServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        address: tuple[str, int],
        data: DashboardData,
        renderer: FigureRenderer,
    ) -> None:
        self.dashboard_data = data
        self.figure_renderer = renderer
        self._request_lock = threading.Lock()
        self._latest_figure_requests: OrderedDict[str, int] = OrderedDict()
        super().__init__(address, DashboardHandler)

    def register_figure_request(self, client_id: str, request_version: int) -> bool:
        with self._request_lock:
            current = self._latest_figure_requests.get(client_id, -1)
            if request_version < current:
                return False
            self._latest_figure_requests[client_id] = request_version
            self._latest_figure_requests.move_to_end(client_id)
            while len(self._latest_figure_requests) > 512:
                self._latest_figure_requests.popitem(last=False)
            return True

    def figure_request_is_current(self, client_id: str, request_version: int) -> bool:
        with self._request_lock:
            return self._latest_figure_requests.get(client_id) == request_version


class DashboardHandler(BaseHTTPRequestHandler):
    server_version = "SnaDashboard/1.0"

    @property
    def dashboard_server(self) -> DashboardServer:
        return self.server  # type: ignore[return-value]

    def log_message(self, format_string: str, *args: object) -> None:
        print(
            f"{self.log_date_time_string()} {self.client_address[0]} "
            f"{format_string % args}",
            flush=True,
        )

    def _common_headers(self) -> None:
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; img-src 'self' blob:; "
            "script-src 'self'; style-src 'self'; base-uri 'none'; "
            "frame-ancestors 'none'; form-action 'none'",
        )

    def _send_bytes(
        self,
        status: int,
        content_type: str,
        body: bytes,
        cache_control: str = "no-store",
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        self.send_response(status)
        self._common_headers()
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", cache_control)
        if extra_headers:
            for name, value in extra_headers.items():
                self.send_header(name, value)
        self.end_headers()
        if self.command != "HEAD":
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                return

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        self._send_bytes(
            status,
            "application/json; charset=utf-8",
            _json_bytes(payload),
        )

    def _send_error_json(self, status: int, code: str, message: str) -> None:
        self._send_json(
            status,
            {
                "ok": False,
                "error": {
                    "code": code,
                    "message": message,
                },
            },
        )

    def _serve_static(self, path: str) -> bool:
        file_path = STATIC_FILES.get(path)
        if file_path is None:
            return False
        if not file_path.is_file():
            raise FileNotFoundError(f"Required static file is missing: {file_path}")
        body = file_path.read_bytes()
        content_type, _ = mimetypes.guess_type(file_path.name)
        if content_type is None:
            content_type = "application/octet-stream"
        if content_type.startswith("text/") or content_type == "application/javascript":
            content_type = f"{content_type}; charset=utf-8"
        self._send_bytes(
            200,
            content_type,
            body,
            cache_control="no-cache",
        )
        return True

    def _parse_selection(self, query: dict[str, list[str]], figure_key: str) -> list[str]:
        data = self.dashboard_server.dashboard_data
        if figure_key in {"03", "10"}:
            values = query.get("cattle", [])
            if any(value.strip() for value in values):
                raise ValueError(f"Figure {figure_key} is not cattle-filtered")
            return []

        raw_values = query.get("cattle", [""])
        if len(raw_values) != 1:
            raise ValueError("The cattle query parameter must appear at most once")
        raw = raw_values[0].strip()
        if not raw:
            return []
        selected = [item.strip() for item in raw.split(",")]
        if any(not item for item in selected):
            raise ValueError("The cattle selection contains an empty identity")
        if len(selected) != len(set(selected)):
            raise ValueError("The cattle selection contains duplicate identities")
        unknown = sorted(set(selected) - set(data.all_cows))
        if unknown:
            raise ValueError(f"Unknown cattle identities: {unknown}")
        return selected

    def _serve_bootstrap(self) -> None:
        payload = self.dashboard_server.dashboard_data.bootstrap_payload()
        self._send_json(200, payload)

    def _serve_health(self) -> None:
        data = self.dashboard_server.dashboard_data
        self._send_json(
            200,
            {
                "ok": True,
                "status": "ready",
                "sampleId": data.data["sample_id"],
                "generationId": data.generation_id,
            },
        )

    def _parse_cow_request(self, query: dict[str, list[str]]) -> str:
        unexpected = sorted(set(query) - {"cow", "generation"})
        if unexpected:
            raise ValueError(f"Unexpected query parameters: {unexpected}")
        data = self.dashboard_server.dashboard_data
        generation_values = query.get("generation", [])
        if (
            len(generation_values) != 1
            or generation_values[0].strip() != data.generation_id
        ):
            raise ValueError(
                "The generation query parameter is required and must match the current result"
            )
        cow_values = query.get("cow", [])
        if len(cow_values) != 1 or not cow_values[0].strip():
            raise ValueError("Exactly one non-empty cow query parameter is required")
        cow_id = cow_values[0].strip()
        if cow_id not in data.all_cows:
            raise ValueError(f"Unknown cattle identity: {cow_id}")
        return cow_id

    def _serve_cow_appearances(self, query: dict[str, list[str]]) -> None:
        cow_id = self._parse_cow_request(query)
        data = self.dashboard_server.dashboard_data
        self._send_json(
            200,
            {
                "ok": True,
                "cowId": cow_id,
                "appearances": data.appearances_for_cow(cow_id),
            },
        )

    def _serve_cow_photo(self, query: dict[str, list[str]]) -> None:
        cow_id = self._parse_cow_request(query)
        photo = self.dashboard_server.dashboard_data.cow_photo_for_cow(cow_id)
        body = photo.path.read_bytes()
        self._send_bytes(
            200,
            photo.mime_type,
            body,
            cache_control="no-store",
            extra_headers={
                "ETag": f'"{photo.sha256}"',
                "X-Cow-ID": cow_id,
                "X-Photo-Clip": photo.clip,
                "X-Photo-Frame": str(photo.local_frame),
            },
        )

    def _parse_figure_request(
        self,
        query: dict[str, list[str]],
    ) -> tuple[str, list[str], str, int]:
        generation_values = query.get("generation", [])
        expected_generation = self.dashboard_server.dashboard_data.generation_id
        if len(generation_values) != 1 or generation_values[0].strip() != expected_generation:
            raise ValueError(
                "The generation query parameter is required and must match the current result"
            )
        client_values = query.get("client", [])
        if (
            len(client_values) != 1
            or re.fullmatch(r"[a-z0-9_-]{8,96}", client_values[0].strip()) is None
        ):
            raise ValueError("A valid client query parameter is required")
        client_id = client_values[0].strip()
        request_values = query.get("request", [])
        if len(request_values) != 1:
            raise ValueError("Exactly one request version is required")
        try:
            request_version = int(request_values[0])
        except ValueError as exc:
            raise ValueError("The request version must be an integer") from exc
        if request_version < 1:
            raise ValueError("The request version must be positive")
        if not self.dashboard_server.register_figure_request(client_id, request_version):
            raise RenderSupersededError("A newer figure request already exists")
        figure_values = query.get("figure", [])
        if len(figure_values) != 1 or not figure_values[0].strip():
            raise ValueError("Exactly one figure query parameter is required")
        figure_key = figure_values[0].strip().upper()
        renderer = self.dashboard_server.figure_renderer
        if figure_key not in renderer.figure_keys:
            raise ValueError(f"Unknown figure: {figure_key}")
        selected = self._parse_selection(query, figure_key)
        return figure_key, selected, client_id, request_version

    def _render_figure_result(
        self,
        query: dict[str, list[str]],
    ) -> tuple[str, list[str], Any]:
        figure_key, selected, client_id, request_version = self._parse_figure_request(
            query
        )
        result = self.dashboard_server.figure_renderer.render_result(
            figure_key,
            selected,
            should_cancel=lambda: not self.dashboard_server.figure_request_is_current(
                client_id,
                request_version,
            ),
        )
        return figure_key, selected, result

    def _serve_figure(self, query: dict[str, list[str]]) -> None:
        figure_key, selected, result = self._render_figure_result(query)
        self._send_bytes(
            200,
            "image/png",
            result.image,
            cache_control="no-store",
            extra_headers={
                "X-Dashboard-Figure": figure_key,
                "X-Selected-Cattle": str(len(selected)),
            },
        )

    def _serve_figure_map(self, query: dict[str, list[str]]) -> None:
        figure_key, _, result = self._render_figure_result(query)
        self._send_json(
            200,
            {
                "ok": True,
                "figure": figure_key,
                "width": result.width,
                "height": result.height,
                "regions": result.regions,
            },
        )

    def _dispatch_get(self) -> None:
        parsed = urlsplit(self.path)
        if parsed.path == "/favicon.ico":
            self._send_bytes(204, "image/x-icon", b"", cache_control="public, max-age=86400")
            return
        if self._serve_static(parsed.path):
            return
        query = parse_qs(parsed.query, keep_blank_values=True)
        if parsed.path == "/api/bootstrap":
            self._serve_bootstrap()
            return
        if parsed.path == "/api/health":
            self._serve_health()
            return
        if parsed.path == "/api/cow-appearances":
            self._serve_cow_appearances(query)
            return
        if parsed.path == "/api/cow-photo":
            self._serve_cow_photo(query)
            return
        if parsed.path == "/api/figure":
            self._serve_figure(query)
            return
        if parsed.path == "/api/figure-map":
            self._serve_figure_map(query)
            return
        self._send_error_json(404, "NOT_FOUND", f"Unknown path: {parsed.path}")

    def do_GET(self) -> None:
        try:
            self._dispatch_get()
        except ValueError as exc:
            self._send_error_json(400, "INVALID_REQUEST", str(exc))
        except RenderSupersededError as exc:
            self._send_error_json(409, "RENDER_SUPERSEDED", str(exc))
        except (DataContractError, FileNotFoundError) as exc:
            traceback.print_exc(file=sys.stderr)
            self._send_error_json(500, "DATA_CONTRACT_ERROR", str(exc))
        except Exception:
            traceback.print_exc(file=sys.stderr)
            self._send_error_json(
                500,
                "INTERNAL_ERROR",
                "The figure could not be rendered. See the server log for details.",
            )

    def do_HEAD(self) -> None:
        if urlsplit(self.path).path in {"/api/figure", "/api/figure-map"}:
            self._send_error_json(
                405,
                "METHOD_NOT_ALLOWED",
                "HEAD does not render figures",
            )
            return
        self.do_GET()

    def do_POST(self) -> None:
        self._send_error_json(405, "METHOD_NOT_ALLOWED", "This Dashboard is read-only")


def create_server(
    host: str,
    port: int,
    data: DashboardData | None = None,
    renderer: FigureRenderer | None = None,
) -> DashboardServer:
    data = DashboardData() if data is None else data
    renderer = FigureRenderer(data) if renderer is None else renderer
    return DashboardServer((host, port), data, renderer)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Serve the standalone SNA results Dashboard",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=2299)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    server = create_server(args.host, args.port)
    host, port = server.server_address[:2]
    print(
        f"SNA Dashboard ready at http://{host}:{port}/ "
        f"(sample={server.dashboard_data.data['sample_id']}, "
        f"generation={server.dashboard_data.generation_id})",
        flush=True,
    )
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        print("Stopping SNA Dashboard.", flush=True)
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
