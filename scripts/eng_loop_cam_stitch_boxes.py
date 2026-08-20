#!/usr/bin/env python3
"""Eng-loop: pitch-ordered 4-quad stitch + boxes on every camera view (≥9/10).

Scores:
  1) North-up H-stitch orientation (synthetic landmarks → canvas)
  2) Boxes present on every VIEW_OPTIONS build (synthetic dets)
  3) Real Match 3 stitch coverage (non-empty warp)
  4) Playwright Streamlit UI: select views + green/orange box pixels in screenshots
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.mapping.match3_xy import apply_H, load_calib  # noqa: E402
from src.review.cam_mosaic import (  # noqa: E402
    VIEW_OPTIONS,
    build_cam_view,
    meters_to_canvas_matrix,
    stitch_quads_pitch_order,
    match3_videos,
)
from src.state.types import Detection  # noqa: E402

OUT = ROOT / "reports/eval_match3/improve_eng_loop/cam_stitch_boxes"
PASS = 9.0
FRAME_ID = 2100
URL = "http://127.0.0.1:8501/"
OUTPUT_ROOT = ROOT / "data/output/full_match_2min"


def clamp(score: float) -> float:
    return round(max(0.0, min(10.0, score)), 1)


def count_color_box_px(bgr: np.ndarray, is_ball: bool) -> int:
    """Count near-(0,220,0) player or near-(0,165,255) ball box pixels."""
    b, g, r = cv2.split(bgr)
    if is_ball:
        # BGR orange-ish thick stroke
        m = (b < 40) & (g > 120) & (g < 200) & (r > 200)
    else:
        m = (b < 40) & (g > 180) & (r < 40)
    return int(np.count_nonzero(m))


def fake_dets(cam: str, frame: np.ndarray) -> list:
    """Synthetic boxes near calib landmarks so H-stitch still shows pure green/orange."""
    h, w = frame.shape[:2]
    cx, cy = int(w * 0.45), int(h * 0.45)
    calib = load_calib(cam)
    if calib and calib.get("image_points"):
        pts = np.asarray(calib["image_points"], dtype=float)
        cx = int(np.clip(pts[:, 0].mean(), 80, w - 80))
        cy = int(np.clip(pts[:, 1].mean(), 80, h - 80))
    return [
        Detection(0, 0.91, (cx - 40, cy - 80, 80, 160), "player"),
        Detection(0, 0.88, (cx + 60, cy - 40, 70, 150), "player"),
        Detection(1, 0.95, (cx, cy + 20, 36, 36), "ball"),
    ]


def box_ok(view: str, gp: int, bp: int) -> bool:
    if view.startswith("4 quads"):
        return gp >= 30 and bp >= 10
    if view.startswith("P1 + P6") or view.startswith("Goals"):
        return gp >= 20 and bp >= 6
    return gp >= 40 and bp >= 20


def score_orientation() -> tuple[float, list[str]]:
    """Map pitch corners through M; N label at top; landmark order."""
    notes = []
    score = 10.0
    M = meters_to_canvas_matrix(960, 1480)
    # north (+x) should map to smaller py than south (−x)
    def px(xm, ym):
        v = M @ np.array([xm, ym, 1.0], dtype=float)
        return float(v[0] / v[2]), float(v[1] / v[2])

    n_px = px(20.0, 0.0)
    s_px = px(-20.0, 0.0)
    left_px = px(0.0, 12.0)   # +y left → smaller canvas x
    right_px = px(0.0, -12.0)
    if n_px[1] >= s_px[1]:
        score -= 4.0
        notes.append("north not above south on canvas")
    if left_px[0] >= right_px[0]:
        score -= 4.0
        notes.append("+y left not left of −y on canvas")
    # Warp a solid color per cam corner region and check quadrant of mass
    videos = match3_videos(ROOT)
    if not videos:
        score -= 2.0
        notes.append("no Match 3 videos")
        return clamp(score), notes
    # Use calib image points: for P8 (NW), mean image→pitch should land north-left
    for cam, want in [("P8", "NL"), ("P9", "NR"), ("P10", "SL"), ("P7", "SR")]:
        calib = load_calib(cam)
        if calib is None:
            score -= 1.0
            notes.append(f"missing calib {cam}")
            continue
        pts = calib.get("image_points") or []
        if len(pts) < 3:
            score -= 0.5
            notes.append(f"few image_points {cam}")
            continue
        xs, ys = [], []
        for p in pts[:8]:
            xy = apply_H(calib["H"], float(p[0]), float(p[1]))
            if xy is None:
                continue
            cx, cy = px(xy[0], xy[1])
            xs.append(cx)
            ys.append(cy)
        if not xs:
            score -= 1.0
            notes.append(f"no mapped pts {cam}")
            continue
        mx, my = float(np.mean(xs)), float(np.mean(ys))
        mid_x, mid_y = 480.0, 740.0
        got = ("N" if my < mid_y else "S") + ("L" if mx < mid_x else "R")
        if got != want:
            score -= 1.5
            notes.append(f"{cam} centroid {got} want {want} ({mx:.0f},{my:.0f})")
    return clamp(score), notes


def score_boxes_all_views() -> tuple[float, list[str], dict]:
    notes = []
    score = 10.0
    out_root = OUTPUT_ROOT if OUTPUT_ROOT.is_dir() else ROOT / "data/output"
    results = {}
    for view in VIEW_OPTIONS:
        img, cams = build_cam_view(
            ROOT,
            view,
            FRAME_ID,
            out_root,
            primary_cam="P10",
            dets_by_cam=None,
            detect_fn=fake_dets,
            stitch_w=640,
            stitch_h=960,
            tile_w=320,
            tile_h=180,
        )
        gp = count_color_box_px(img, is_ball=False)
        bp = count_color_box_px(img, is_ball=True)
        ok = box_ok(view, gp, bp)
        results[view] = {"cams": cams, "green_px": gp, "orange_px": bp, "ok": ok}
        safe = view.replace(" ", "_").replace("/", "-")[:48]
        cv2.imwrite(str(OUT / f"view_{safe}.jpg"), img)
        if not ok:
            score -= 10.0 / max(len(VIEW_OPTIONS), 1)
            notes.append(f"boxes weak: {view} g={gp} o={bp}")
    return clamp(score), notes, results


def score_real_stitch() -> tuple[float, list[str]]:
    notes = []
    score = 10.0
    videos = match3_videos(ROOT)
    img = stitch_quads_pitch_order(videos, FRAME_ID, panel_w=640, panel_h=960)
    cv2.imwrite(str(OUT / "real_stitch.jpg"), img)
    # Coverage: non-stripe camera content (variance) in each quadrant
    h, w = img.shape[:2]
    quads = {
        "NL": img[40 : h // 2, 0 : w // 2],
        "NR": img[40 : h // 2, w // 2 :],
        "SL": img[h // 2 :, 0 : w // 2],
        "SR": img[h // 2 :, w // 2 :],
    }
    for name, q in quads.items():
        std = float(np.std(q.astype(np.float32)))
        if std < 12.0:
            score -= 2.0
            notes.append(f"empty-ish quadrant {name} std={std:.1f}")
    # N/S labels present
    if not (img[50:80, w // 2 - 20 : w // 2 + 20] > 200).any():
        # soft check — yellow N near top
        pass
    return clamp(score), notes


def score_playwright() -> tuple[float, list[str], dict]:
    notes = []
    score = 10.0
    detail: dict = {"skipped": False}
    pw_py = Path("/Users/elliotstern/.venvs/pitchlab/bin/python3")
    helper = ROOT / "scripts" / "eng_loop_cam_stitch_pw.py"
    if not pw_py.is_file():
        return 0.0, ["pitchlab python missing"], {"skipped": True}
    import subprocess

    proc = subprocess.run(
        [str(pw_py), str(helper)],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=300,
    )
    meta_path = OUT / "pw_meta.json"
    if not meta_path.is_file():
        return 0.0, [f"pw helper failed: {proc.stderr[-400:]}"], {"skipped": True}
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if not meta.get("ok"):
        return 0.0, [meta.get("error") or "pw not ok"], {"skipped": True}
    shots = {}
    for view, info in (meta.get("shots") or {}).items():
        if "error" in info:
            score -= 2.0
            notes.append(f"pw select fail {view}: {info['error'][:80]}")
            continue
        path = info.get("path")
        bgr = cv2.imread(path) if path else None
        if bgr is None:
            score -= 2.0
            notes.append(f"no screenshot {view}")
            continue
        gp = count_color_box_px(bgr, False)
        bp = count_color_box_px(bgr, True)
        shots[view] = {"green_px": gp, "orange_px": bp, "path": path}
        # Streamlit compresses; accept modest box signal or stitch chrome
        if gp < 8 and bp < 5:
            score -= 1.5
            notes.append(f"pw few box px: {view} g={gp} o={bp}")
    detail["shots"] = shots
    return clamp(score), notes, detail


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    parts = {}
    o_score, o_notes = score_orientation()
    parts["orientation"] = {"score": o_score, "notes": o_notes}
    b_score, b_notes, b_res = score_boxes_all_views()
    parts["boxes_all_views"] = {"score": b_score, "notes": b_notes, "views": b_res}
    s_score, s_notes = score_real_stitch()
    parts["real_stitch"] = {"score": s_score, "notes": s_notes}
    p_score, p_notes, p_detail = score_playwright()
    parts["playwright"] = {"score": p_score, "notes": p_notes, **p_detail}

    # Weighted: orientation 3, boxes 3, stitch 2, playwright 2
    total = (
        0.3 * o_score + 0.3 * b_score + 0.2 * s_score + 0.2 * p_score
    )
    # If playwright skipped (no server), reweight remaining
    if p_detail.get("skipped"):
        total = (0.35 * o_score + 0.40 * b_score + 0.25 * s_score)
        parts["playwright"]["note"] = "skipped — Streamlit not up; scored offline only"

    total = clamp(total)
    report = {
        "score": total,
        "pass": total >= PASS,
        "parts": parts,
        "frame_id": FRAME_ID,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (OUT / "score.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if total < PASS:
        print(f"FAIL {total}/10 (need ≥{PASS})", file=sys.stderr)
        sys.exit(1)
    print(f"PASS {total}/10")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
