#!/usr/bin/env python3
"""Minimal handover server on local disk (launchd-safe; LaCie exfat blocked)."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

PORT = int(os.environ.get("HANDOVER_PORT", "8080"))
ROOT = Path(__file__).resolve().parents[1]
LOCAL = Path.home() / "Library/Application Support/soccer-coach-handover"
REMOTE_HANDOVER = ROOT / "reports/eval_match3/improve_eng_loop/phase1_handover"


def sync_from_project() -> None:
    LOCAL.mkdir(parents=True, exist_ok=True)
    if not REMOTE_HANDOVER.is_dir():
        return
    for name in (
        "coach_mosaic_pitch_min.mp4",
        "meta.json",
        "emits_render.json",
        "index.html",
        "still_first.jpg",
        "still_mid.jpg",
        "still_last.jpg",
        "handover_info.json",
    ):
        src = REMOTE_HANDOVER / name
        if src.is_file():
            (LOCAL / name).write_bytes(src.read_bytes())
    labels_src = REMOTE_HANDOVER / "labels.json"
    labels_dst = LOCAL / "labels.json"
    if labels_src.is_file():
        labels_dst.write_bytes(labels_src.read_bytes())
    elif not labels_dst.is_file():
        labels_dst.write_text(
            json.dumps({"reviewer": "", "frames": {}, "suggested_events": []}, indent=2),
            encoding="utf-8",
        )


def push_labels_to_project(data: dict) -> None:
    (LOCAL / "labels.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
    if REMOTE_HANDOVER.is_dir():
        try:
            (REMOTE_HANDOVER / "labels.json").write_text(
                json.dumps(data, indent=2), encoding="utf-8"
            )
        except OSError:
            pass


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(LOCAL), **kwargs)

    def log_message(self, fmt, *args):
        print(f"{self.address_string()} {fmt % args}", flush=True)

    def do_GET(self):
        path = urlparse(self.path).path
        if path in ("/", "/phase1-handover", "/phase1-handover/"):
            self.send_response(302)
            self.send_header("Location", "/index.html")
            self.end_headers()
            return
        super().do_GET()

    def do_POST(self):
        path = urlparse(self.path).path
        n = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(n)
        try:
            if path == "/save_phase1_handover_labels":
                data = json.loads(body.decode("utf-8"))
                data["updated_at"] = datetime.now(timezone.utc).isoformat()
                push_labels_to_project(data)
                self._json(200, {"ok": True, "path": str(LOCAL / "labels.json")})
                return
            if path == "/merge_phase1_handover_gold":
                push_labels_to_project(
                    json.loads((LOCAL / "labels.json").read_text(encoding="utf-8"))
                )
                proc = subprocess.run(
                    [sys.executable, str(ROOT / "scripts/gold_set/merge_handover_fuse_gold.py")],
                    cwd=str(ROOT),
                    capture_output=True,
                    text=True,
                )
                if proc.returncode != 0:
                    raise RuntimeError(proc.stderr or proc.stdout or "merge failed")
                clip = (
                    ROOT
                    / "data/processed/gold_sets/match3_events_v2_dribble/clips/real_fuse_15s"
                    / "labels.json"
                )
                coach_n = 0
                if clip.is_file():
                    merged = json.loads(clip.read_text(encoding="utf-8"))
                    coach_n = sum(
                        1 for e in merged.get("events") or [] if e.get("source") == "handover"
                    )
                self._json(200, {"ok": True, "coach_events": coach_n})
                return
            self._json(404, {"ok": False, "error": "unknown path"})
        except Exception as e:
            self._json(500, {"ok": False, "error": str(e)})

    def _json(self, code: int, payload: dict) -> None:
        raw = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "--sync-only":
        sync_from_project()
        print(f"synced → {LOCAL}", flush=True)
        return 0
    sync_from_project()
    if not (LOCAL / "index.html").is_file():
        print(f"ERROR missing {LOCAL}/index.html — run build_phase1_handover_dashboard.py", flush=True)
        return 1
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"Handover local server http://127.0.0.1:{PORT}/phase1-handover", flush=True)
    print(f"Serving {LOCAL}", flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
