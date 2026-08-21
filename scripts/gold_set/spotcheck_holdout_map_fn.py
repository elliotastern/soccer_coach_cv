#!/usr/bin/env python3
"""Spot-check holdout clear FNs: P10 low_support feet — abort hull if junk."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "gold_set"))

from eval_match2_top_left_multicam_baseline import cache_load  # noqa: E402
from fn_audit_match3_quad import (  # noqa: E402
    CAMS,
    CLEAR_SIDE,
    WH,
    best_raw,
    map_reason,
    product_fuse,
)
from fn_audit_match3_random import clear_focus_cam  # noqa: E402
from multicam_select_policy import MATCH3_THR_BY_CAM, filter_active  # noqa: E402
from src.mapping.match3_xy import (  # noqa: E402
    bbox_foot,
    calib_undistort_params,
    hull_points,
    hull_support,
    load_calib,
    scale_px,
    undistort_px,
)

CACHE_DIR = ROOT / "reports/eval_match3/pitchmap_gallery_holdout/det_cache"
SRC_DIR = ROOT / "reports/eval_match3/pitchmap_gallery_holdout/source"
OUT_DIR = ROOT / "reports/ball_testing/holdout_map_fn_spot"
FOCUS = "P10"


def foot_px(calib, box):
    fx, fy = bbox_foot(box)
    wh = calib.get("image_wh") or WH
    px, py = scale_px(fx, fy, wh, calib.get("image_wh") or wh)
    params = calib_undistort_params(calib)
    if params:
        cw, ch = calib.get("image_wh") or wh
        px, py = undistort_px(px, py, cw, ch, params)
    return float(px), float(py)


def stem_from_cache(path: Path) -> str:
    # det_cache_rand_t00166.0s_thr010.json
    name = path.name
    return name.replace("det_cache_", "").replace("_thr010.json", "")


def grab_still(stem: str, cam: str, frame_i: int, box, out_jpg: Path) -> bool:
    path = SRC_DIR / f"{stem}_{cam}.mp4"
    if not path.is_file():
        return False
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return False
    cap.set(cv2.CAP_PROP_POS_FRAMES, float(frame_i))
    ok, bgr = cap.read()
    cap.release()
    if not ok or bgr is None:
        return False
    x1, y1, x2, y2 = [int(v) for v in box]
    cv2.rectangle(bgr, (x1, y1), (x2, y2), (0, 165, 255), 2)
    fx, fy = bbox_foot(box)
    cv2.circle(bgr, (int(fx), int(fy)), 6, (0, 255, 255), -1)
    out_jpg.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_jpg), bgr)
    return True


def audit_cache(path: Path, calibs: dict) -> list[dict]:
    dets = cache_load(path)
    cams = [c for c in CAMS if c in dets]
    n = len(next(iter(dets.values())))
    rows = []
    prev = None
    gap = 0
    stem = stem_from_cache(path)
    for i in range(n):
        active = filter_active(dets, i, cams, MATCH3_THR_BY_CAM)
        focus = clear_focus_cam(active)
        fused, prev, gap = product_fuse(dets, i, calibs, cams, prev, gap)
        if focus != FOCUS or fused is not None:
            continue
        rows_active = active.get(FOCUS)
        if not rows_active:
            continue
        box, conf, side = rows_active[0]
        if float(side) < CLEAR_SIDE or float(conf) < 0.30:
            continue
        rec = calibs.get(FOCUS)
        if rec is None:
            continue
        hit, why = map_reason(rec, box, float(conf))
        if why != "low_support":
            continue
        px, py = foot_px(rec, box)
        support = hull_support(px, py, hull_points(rec))
        rows.append(
            {
                "cache": path.name,
                "stem": stem,
                "i": i,
                "conf": float(conf),
                "side": float(side),
                "foot_px": [round(px, 1), round(py, 1)],
                "support": round(float(support), 4),
                "box": [float(v) for v in box],
                "in_frame": 0 <= px <= 1920 and 0 <= py <= 1080,
                "lower_half": py >= 540,
            }
        )
    return rows


def decide(rows: list[dict]) -> dict:
    if not rows:
        return {
            "proceed_hull": False,
            "reason": "no P10 low_support clear FNs found",
            "abort_junk": False,
        }
    n = len(rows)
    in_frame = sum(1 for r in rows if r["in_frame"])
    lower = sum(1 for r in rows if r["lower_half"])
    xs = [r["foot_px"][0] for r in rows]
    ys = [r["foot_px"][1] for r in rows]
    # Junk = mostly outside frame or tiny cluster at corner
    junk_frac = 1.0 - (in_frame / n)
    proceed = junk_frac < 0.25 and in_frame >= 5
    return {
        "proceed_hull": proceed,
        "abort_junk": not proceed,
        "reason": (
            "feet in playable FOV outside landmark hull"
            if proceed
            else "feet look junk / off-frame — abort hull"
        ),
        "n": n,
        "in_frame": in_frame,
        "lower_half": lower,
        "foot_x_range": [round(min(xs), 1), round(max(xs), 1)],
        "foot_y_range": [round(min(ys), 1), round(max(ys), 1)],
        "median_support": round(sorted(r["support"] for r in rows)[n // 2], 4),
    }


def main() -> int:
    calibs = {c: v for c, v in ((c, load_calib(c)) for c in CAMS) if v}
    paths = sorted(CACHE_DIR.glob("det_cache_rand_*_thr010.json"))
    all_rows = []
    for p in paths:
        all_rows.extend(audit_cache(p, calibs))
    decision = decide(all_rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # Save up to 12 stills across caches
    saved = 0
    by_cache = {}
    for r in all_rows:
        by_cache.setdefault(r["cache"], []).append(r)
    for cache, group in by_cache.items():
        step = max(1, len(group) // 3)
        for r in group[::step][:3]:
            if saved >= 12:
                break
            jpg = OUT_DIR / f"{r['stem']}_i{r['i']:04d}_P10.jpg"
            if grab_still(r["stem"], FOCUS, r["i"], r["box"], jpg):
                r["still"] = str(jpg.relative_to(ROOT))
                saved += 1
    summary = {
        "focus": FOCUS,
        "n_fn": len(all_rows),
        "decision": decision,
        "samples": all_rows[:40],
        "stills_saved": saved,
        "out_dir": str(OUT_DIR.relative_to(ROOT)),
    }
    out_json = OUT_DIR / "summary.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        f"P10 low_support FNs={len(all_rows)} "
        f"proceed_hull={decision['proceed_hull']} "
        f"({decision['reason']})"
    )
    if decision.get("foot_y_range"):
        print(
            f"  foot x={decision['foot_x_range']} y={decision['foot_y_range']} "
            f"median_support={decision['median_support']}"
        )
    print(f"wrote {out_json} stills={saved}")
    return 0 if decision["proceed_hull"] or len(all_rows) == 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
