#!/usr/bin/env python3
"""Eng-loop: player boxes = full RF-DETR size, 1px stroke (≥9/10)."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "reports/events_testing/coach_player_box_ux"
PASS = 9.0


def _green_strokes(rgb) -> list[float]:
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    mask = cv2.inRange(bgr, np.array([0, 150, 0]), np.array([95, 255, 95]))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cnts:
        if cv2.contourArea(c) < 20:
            continue
        x, y, w, h = cv2.boundingRect(c)
        if h < 14 or w < 3:
            continue
        roi = mask[y : y + h, x : x + w]
        out.append(float(cv2.distanceTransform(roi, cv2.DIST_L2, 3).max()))
    return out


def run_annotate_unit() -> dict:
    from src.review.cam_mosaic import _annotate
    from src.state.types import Detection

    fr = np.zeros((240, 320, 3), dtype=np.uint8)
    fr[:] = (40, 40, 90)
    w_in, h_in = 60, 140
    dets = [Detection(0, 0.9, (80, 40, w_in, h_in), "player")]
    out = _annotate(fr, dets)
    bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    mask = cv2.inRange(bgr, np.array([0, 150, 0]), np.array([95, 255, 95]))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    w_out = h_out = 0
    if cnts:
        _, _, w_out, h_out = cv2.boundingRect(max(cnts, key=cv2.contourArea))
    strokes = _green_strokes(out)
    max_s = max(strokes) if strokes else 99.0
    ok = w_out >= w_in - 2 and h_out >= h_in - 2 and max_s <= 1.6
    return {
        "name": "annotate_full_box_thin_stroke",
        "pass": ok,
        "w_in": w_in,
        "w_out": w_out,
        "h_out": h_out,
        "max_stroke": round(max_s, 2),
    }


def run_dashboard_passthrough() -> dict:
    """Dashboard must not rewrite baked player pixels (GitHub behavior)."""
    from apps.coach_emit_label_dashboard import apply_view_layout

    p = ROOT / "reports/eval_match3/improve_eng_loop/match4_5min/coach_mosaic_first_90s.mp4"
    cap = cv2.VideoCapture(str(p))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 20)
    ok, bgr = cap.read()
    cap.release()
    if not ok:
        return {"name": "dashboard_passthrough", "pass": False, "error": "no frame"}
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    out, _ = apply_view_layout(rgb, "Quad mosaic + pitch", 0, ROOT / "data/output/match_4_5min")
    same = bool(np.array_equal(out, rgb))
    return {"name": "dashboard_passthrough", "pass": same}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    checks = [run_annotate_unit(), run_dashboard_passthrough()]
    passed = sum(1 for c in checks if c.get("pass"))
    total = len(checks)
    payload = {
        "checks": checks,
        "score": round(10.0 * passed / max(total, 1), 2),
        "pass": passed / max(total, 1) * 10 >= PASS,
        "gate": PASS,
        "passed": passed,
        "total": total,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (OUT / "pw_player_box_stroke_score.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))
    return 0 if payload["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
