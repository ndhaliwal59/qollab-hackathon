"""Small standard-library localhost server for the live Signal Rescue app."""

from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
import webbrowser

from qiskit.providers.basic_provider import BasicSimulator

from signal_rescue.live import dispatch

ROOT = Path(__file__).resolve().parent.parent
WEB_ROOT = ROOT / "web"
BACKEND = BasicSimulator()


class Handler(BaseHTTPRequestHandler):
    server_version = "SignalRescueLocal/1.0"

    def _json(self, value: str, status: int = 200):
        raw = value.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        if self.path == "/api/bootstrap":
            self._json(dispatch(None, "bootstrap", "{}"))
            return
        if self.path in ("/", "/index.html"):
            body = (WEB_ROOT / "index.html").read_text(encoding="utf-8")
            page = ("<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
                    "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
                    "<meta name=\"theme-color\" content=\"#07111f\"><title>Signal Rescue — Live Lab</title>"
                    "<link rel=\"stylesheet\" href=\"/web/main.css\"></head><body>" + body +
                    "<script type=\"module\" src=\"/web/local.js\"></script></body></html>")
            raw = page.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
            return
        if self.path.startswith("/web/"):
            target = (WEB_ROOT / self.path.removeprefix("/web/")).resolve()
            if target.parent != WEB_ROOT and WEB_ROOT not in target.parents:
                self.send_error(404)
                return
            if not target.is_file():
                self.send_error(404)
                return
            raw = target.read_bytes()
            self.send_response(200)
            mime = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
            self.send_header("Content-Type", mime + ("; charset=utf-8" if mime in ("text/css", "text/javascript") else ""))
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(raw)
            return
        self.send_error(404)

    def do_POST(self):
        action = {"/api/preview": "preview", "/api/run": "run"}.get(self.path)
        if action is None:
            self.send_error(404)
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if size > 65536:
                self._json(json.dumps({"status": "error", "error": "Request exceeds 64 KB."}), 413)
                return
            raw = self.rfile.read(size).decode("utf-8")
        except (ValueError, UnicodeDecodeError):
            self._json(json.dumps({"status": "error", "error": "Invalid JSON request body."}), 400)
            return
        result = dispatch(BACKEND if action == "run" else None, action, raw)
        self._json(result)

    def log_message(self, fmt, *args):
        print(f"[signal-rescue] {self.address_string()} {fmt % args}")


def main():
    parser = argparse.ArgumentParser(description="Run the Signal Rescue live local laboratory")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (defaults to localhost only)")
    parser.add_argument("--port", default=8000, type=int)
    parser.add_argument("--no-browser", action="store_true", help="Do not open a browser tab")
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{server.server_port}/"
    print(f"Signal Rescue is ready at {url}")
    print("Qiskit BasicSimulator · no external services · press Ctrl-C to stop")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping Signal Rescue")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
