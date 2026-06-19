#!/usr/bin/env python3
"""Serve the OpenOdds26 live-performance dashboard on the LAN.

Standard library ONLY (http.server, subprocess, json, os) -- no pip installs,
runs on a Raspberry Pi.

Behaviour
---------
* Serves the dashboard/ directory (index.html, performance.json, ...) over HTTP
  on 0.0.0.0:8080 so it is reachable from any device on the local network at
  http://<pi-ip>:8080/.
* Adds a refresh trigger route at /api/refresh (POST, and GET for convenience)
  that runs, in order:
      python3 scripts/refresh_match_results.py --allow-baseline-additions
      python3 dashboard/build_performance.py
  with cwd = project root, then returns JSON {"ok": bool, "summary": "..."}.
  The browser polls performance.json every 60s, so a successful refresh shows
  up automatically without a page reload.

This server is READ-ONLY with respect to the model: /api/refresh only refreshes
results and re-scores. It never re-runs the (deliberately frozen) forecast.

Run from the project root:
    python3 dashboard/serve.py
Optional environment overrides:
    DASHBOARD_PORT   (default 8080)
    DASHBOARD_HOST   (default 0.0.0.0)
"""

import json
import os
import subprocess
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

HOST = os.environ.get("DASHBOARD_HOST", "0.0.0.0")
# Port resolution order: CLI arg (python serve.py 8099) > DASHBOARD_PORT env > 8080.
_argv_port = next((a for a in sys.argv[1:] if a.isdigit()), None)
PORT = int(_argv_port or os.environ.get("DASHBOARD_PORT", "8080"))

PYTHON = sys.executable or "python3"
REFRESH_SCRIPT = os.path.join(ROOT, "scripts", "refresh_match_results.py")
LIVE_SCRIPT = os.path.join(HERE, "fetch_live_results.py")
BUILD_SCRIPT = os.path.join(HERE, "build_performance.py")

# Hardened spawn kwargs. On Windows, a long-running server that repeatedly
# spawns children can exhaust its window-station desktop heap, after which new
# processes die at init with 0xC0000142 (STATUS_DLL_INIT_FAILED) before printing
# anything. CREATE_NO_WINDOW avoids allocating a console (conhost) per spawn,
# and stdin=DEVNULL avoids inheriting a stale/broken stdin handle.
_SPAWN_KWARGS = {"stdin": subprocess.DEVNULL}
if os.name == "nt":
    _SPAWN_KWARGS["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)


def run_refresh():
    """Fetch results (martj42 + ESPN live layer) then rebuild. Returns (ok, summary)."""
    steps = [
        ([PYTHON, REFRESH_SCRIPT, "--allow-baseline-additions"], "refresh_match_results"),
        ([PYTHON, LIVE_SCRIPT], "fetch_live_results"),
        ([PYTHON, BUILD_SCRIPT], "build_performance"),
    ]
    summaries = []
    build_ok = False
    for cmd, name in steps:
        critical = name == "build_performance"  # the only step that must succeed
        try:
            proc = subprocess.run(
                cmd, cwd=ROOT, capture_output=True, text=True, timeout=600,
                **_SPAWN_KWARGS,
            )
        except Exception as exc:
            summaries.append("{} FAILED to start: {}".format(name, exc))
            continue
        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout or "").strip()
            summaries.append("{} exited {}: {}".format(name, proc.returncode, err[-200:]))
            continue
        out = (proc.stdout or "").strip().splitlines()
        summaries.append(out[-1] if out else "{} ok".format(name))
        if critical:
            build_ok = True
    # Best-effort fetch (martj42/ESPN may transiently fail); only the rebuild
    # off whatever data is present must succeed.
    return build_ok, " | ".join(summaries)


class Handler(SimpleHTTPRequestHandler):
    # Serve files out of the dashboard/ directory.
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=HERE, **kwargs)

    def _send_json(self, code, obj):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _handle_refresh(self):
        ok, summary = run_refresh()
        self._send_json(200 if ok else 500, {"ok": ok, "summary": summary})

    def do_POST(self):
        if self.path.split("?")[0] == "/api/refresh":
            self._handle_refresh()
        else:
            self._send_json(404, {"ok": False, "summary": "not found"})

    def do_GET(self):
        if self.path.split("?")[0] == "/api/refresh":
            self._handle_refresh()
        else:
            # Never cache performance.json so the 60s poll always sees fresh data.
            super().do_GET()

    def end_headers(self):
        if self.path.split("?")[0].endswith("performance.json"):
            self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, fmt, *args):
        sys.stderr.write("[serve] " + (fmt % args) + "\n")


def main():
    httpd = ThreadingHTTPServer((HOST, PORT), Handler)
    print("OpenOdds26 dashboard serving dir: {}".format(HERE))
    print("Listening on http://{}:{}/  (open http://<pi-ip>:{}/ on the LAN)".format(
        HOST, PORT, PORT))
    print("Refresh trigger: POST or GET http://<pi-ip>:{}/api/refresh".format(PORT))
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        httpd.shutdown()


if __name__ == "__main__":
    main()
