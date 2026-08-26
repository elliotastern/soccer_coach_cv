#!/usr/bin/env python3
"""Eng-loop: coach mosaic UX ≥9/10 via Playwright shots + vision rubric notes file.

Offline builds a coach mosaic preview; Playwright captures Streamlit;
score.json is filled by the agent vision pass (this script writes offline + shot paths).
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.review.cam_mosaic import (  # noqa: E402
    VIEW_OPTIONS,
    build_cam_view,
    match3_videos,
    mosaic_quads_coach,
)
from src.state.types import Detection  # noqa: E402

OUT = ROOT / "reports/eval_match3/improve_eng_loop/coach_mosaic_ux"
PASS = 9.0
FRAME = 2397
PW_PY = Path("/Users/elliotstern/.venvs/pitchlab/bin/python3")


def fake_dets(cam: str, frame):
    h, w = frame.shape[:2]
    return [
        Detection(0, 0.9, (int(w * 0.35), int(h * 0.35), 90, 170), "player"),
        Detection(0, 0.85, (int(w * 0.55), int(h * 0.40), 80, 160), "player"),
        Detection(1, 0.95, (int(w * 0.48), int(h * 0.55), 40, 40), "ball"),
    ]


def offline_preview() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    videos = match3_videos(ROOT)
    img = mosaic_quads_coach(
        videos, FRAME, tile_w=640, tile_h=360, detect_fn=fake_dets, apply_defish=True
    )
    path = OUT / "offline_coach_mosaic.jpg"
    cv2.imwrite(str(path), img)
    # Heuristics: no tech jargon burned into pixels
    # Encode as latin-1 so high-byte BGR never crashes utf-8 decode
    raw = img.tobytes()
    textish = ""
    # Check banner region for forbidden words via OCR-less string search on labels
    # We know labels are drawn as pixels — score via size + option name instead
    opt0 = VIEW_OPTIONS[0]
    return {
        "preview": str(path),
        "shape": list(img.shape),
        "view_option": opt0,
        "coach_option_ok": opt0.startswith("Whole pitch"),
        "notes": [],
    }


def run_playwright() -> dict:
    helper = ROOT / "scripts" / "eng_loop_cam_stitch_pw.py"
    # Point helper output into coach_mosaic_ux by env? helper writes cam_stitch_boxes —
    # copy after
    proc = subprocess.run(
        [str(PW_PY), str(helper)],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=300,
    )
    meta_path = ROOT / "reports/eval_match3/improve_eng_loop/cam_stitch_boxes/pw_meta.json"
    if not meta_path.is_file():
        return {"ok": False, "error": proc.stderr[-500:], "shots": {}}
    meta = json.loads(meta_path.read_text())
    shots = {}
    for view, info in (meta.get("shots") or {}).items():
        src = info.get("path")
        if not src:
            shots[view] = info
            continue
        dest = OUT / Path(src).name.replace("pw_", "coach_pw_")
        try:
            import shutil
            shutil.copy2(src, dest)
            shots[view] = {"path": str(dest)}
        except Exception as exc:
            shots[view] = {"error": str(exc)}
    return {"ok": bool(meta.get("ok")), "shots": shots, "stdout": proc.stdout[-200:]}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    off = offline_preview()
    pw = run_playwright()
    report = {
        "score": None,  # filled by vision agent
        "pass": False,
        "gate": PASS,
        "offline": off,
        "playwright": pw,
        "rubric": [
            "No tech jargon visible (defish / 180° / mosaic jargon)",
            "Corner labels plain (Far/Near left/right)",
            "Legend: green=player, orange=ball clear",
            "Layout covers whole pitch and is readable",
            "Boxes look like they sit on people/ball",
        ],
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (OUT / "score.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print("Await vision scoring of screenshots in", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
