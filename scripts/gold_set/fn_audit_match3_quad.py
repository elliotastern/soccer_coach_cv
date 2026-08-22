#!/usr/bin/env python3
"""C1 FN audit: why clear-ball frames on quad strips/caches never emit."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "gold_set"))

from eval_match2_top_left_multicam_baseline import cache_load  # noqa: E402
from multicam_select_policy import MATCH3_THR_BY_CAM, filter_active  # noqa: E402
from score_match3_ball_m1 import HIT_M, infer_cache_stride  # noqa: E402
from src.mapping.match3_xy import (  # noqa: E402
    EMIT_CONF as MAP_EMIT,
    GHOST_CONF,
    MIN_SUPPORT,
    apply_H,
    bbox_foot,
    calib_undistort_params,
    fuse_balls,
    fuse_balls_with_hold,
    hull_points,
    hull_support,
    in_pitch_bounds,
    load_calib,
    scale_px,
    undistort_px,
)

OUT = ROOT / "reports/eval_match3/improve_eng_loop/c1_fn_audit.json"
STRIPS = [
    ROOT / "data/processed/gold_sets/match3_quad_p10_31/labels.json",
    ROOT / "data/processed/gold_sets/match3_quad_p8_87/labels.json",
]
QUAD_CACHE = ROOT / "reports/eval_match3/quad_pitchmap_gallery/det_cache"
CAMS = ["P1", "P6", "P7", "P8", "P9", "P10", "P_Goal1", "P_Goal2"]
WH = (1920, 1080)
CLEAR_SIDE = 25.0
DET_THR = 0.20
FOCUS_GOAL_P = 0.90
FOCUS_GOAL_R = 0.80


def map_reason(calib, box, conf, frame_wh=WH):
    wh = frame_wh or calib.get("image_wh") or WH
    fx, fy = bbox_foot(box)
    px, py = scale_px(fx, fy, wh, calib.get("image_wh") or wh)
    params = calib_undistort_params(calib)
    if params:
        cw, ch = calib.get("image_wh") or wh
        px, py = undistort_px(px, py, cw, ch, params)
    xy = apply_H(calib["H"], px, py)
    if xy is None:
        return None, "h_fail"
    if not in_pitch_bounds(xy[0], xy[1], margin_m=1.0):
        return None, "off_pitch"
    support = hull_support(px, py, hull_points(calib))
    if support < MIN_SUPPORT:
        return None, "low_support"
    return {
        "cam": calib["camera"],
        "xy": xy,
        "conf": float(conf),
        "support": support,
        "weight": float(conf) * support,
    }, "ok"


def best_raw(dets, i, cam):
    rows = dets.get(cam, [[]])[i] if cam in dets else []
    if not rows:
        return None
    box, conf, side = max(rows, key=lambda r: float(r[1]))
    return {
        "conf": float(conf),
        "side": float(side),
        "pass_thr": float(conf) >= DET_THR,
        "pass_emit": float(conf) >= MAP_EMIT,
    }


def mapped_rows(dets, i, calibs, cams):
    active = filter_active(dets, i, cams, MATCH3_THR_BY_CAM)
    out = []
    for cam, pred_rows in active.items():
        rec = calibs.get(cam)
        if rec is None:
            continue
        box, conf, _ = pred_rows[0]
        hit, _ = map_reason(rec, box, float(conf))
        if hit is not None:
            out.append(hit)
    return out, active


def product_fuse(dets, i, calibs, cams, prev, gap):
    rows, _ = mapped_rows(dets, i, calibs, cams)
    kwargs = dict(
        soft_dual_fallback=True,
        solo_max_conf=True,
        ghost_prune=True,
        ghost_conf=GHOST_CONF,
    )
    fresh = fuse_balls(rows, **kwargs)
    if fresh is not None:
        return fresh, fresh, 0
    gap += 1
    held = fuse_balls_with_hold(prev, [], gap, **kwargs)
    return held, prev, gap


def classify_fn(dets, i, calibs, cams, focus):
    fused, rows, active = _classify_inputs(dets, i, calibs, cams)
    if fused is not None:
        return "emit_ok", fused
    fb = best_raw(dets, i, focus)
    if fb is None:
        if not any(best_raw(dets, i, c) for c in cams):
            return "no_det_any", {"focus": focus}
        return "no_det_focus", {"focus": focus}
    rec = calibs.get(focus)
    if rec is None:
        return "no_calib_focus", {"focus": focus}
    all_rows = dets.get(focus, [[]])[i]
    top = max(all_rows, key=lambda r: float(r[1])) if all_rows else None
    if top and float(top[1]) >= DET_THR:
        hit, why = map_reason(rec, top[0], float(top[1]))
        if hit is None:
            return "focus_map_fail", {"reason": why, "conf": float(top[1])}
    if not fb["pass_emit"]:
        if rows:
            return "mapped_conf_below_emit", {
                "focus_conf": fb["conf"],
                "n_mapped": len(rows),
            }
        return "focus_conf_below_emit", {"focus_conf": fb["conf"]}
    if rows and not active.get(focus):
        return "other_cam_only_mapped", {"n_mapped": len(rows)}
    return "fuse_blocked", {"n_mapped": len(rows), "focus_conf": fb["conf"]}


def _classify_inputs(dets, i, calibs, cams):
    rows, active = mapped_rows(dets, i, calibs, cams)
    kwargs = dict(
        soft_dual_fallback=True,
        solo_max_conf=True,
        ghost_prune=True,
        ghost_conf=GHOST_CONF,
    )
    fused = fuse_balls(rows, **kwargs)
    return fused, rows, active


def audit_strip(path: Path) -> dict:
    labels = json.loads(path.read_text(encoding="utf-8"))
    dets = cache_load(ROOT / labels["det_cache"])
    cams = [c for c in CAMS if c in dets]
    calibs = {c: v for c, v in ((c, load_calib(c)) for c in cams) if v}
    focus = labels.get("focus_cam") or "P10"
    stride = infer_cache_stride(dets)
    buckets = {}
    focus_conf_bins = {"ge080": 0, "050_079": 0, "020_049": 0, "lt020": 0, "none": 0}
    prev = None
    gap = 3
    clear = clear_emit = 0
    fn_rows = []
    for fr in labels["frames"]:
        i = int(fr["i"])
        if stride > 1 and i % stride != 0:
            continue
        seed = (fr.get("cams") or {}).get(focus) or {}
        gold = seed.get("gold_xy")
        is_clear = bool(seed.get("clear")) and gold is not None
        if not is_clear:
            continue
        clear += 1
        fused, prev, gap = product_fuse(dets, i, calibs, cams, prev, gap)
        if fused is not None:
            clear_emit += 1
            err = math.hypot(
                float(fused["xy"][0]) - float(gold[0]),
                float(fused["xy"][1]) - float(gold[1]),
            )
            if err > HIT_M:
                tag = "emit_fp"
                buckets[tag] = buckets.get(tag, 0) + 1
            continue
        tag, detail = classify_fn(dets, i, calibs, cams, focus)
        buckets[tag] = buckets.get(tag, 0) + 1
        br = best_raw(dets, i, focus)
        if br is None:
            focus_conf_bins["none"] += 1
        elif br["conf"] >= 0.80:
            focus_conf_bins["ge080"] += 1
        elif br["conf"] >= 0.50:
            focus_conf_bins["050_079"] += 1
        elif br["conf"] >= 0.20:
            focus_conf_bins["020_049"] += 1
        else:
            focus_conf_bins["lt020"] += 1
        if len(fn_rows) < 12:
            fn_rows.append({"i": i, "tag": tag, "detail": detail, "focus": br})
    clear_r = None if clear == 0 else round(clear_emit / clear, 3)
    return {
        "pack": labels.get("pack"),
        "focus_cam": focus,
        "det_cache": labels.get("det_cache"),
        "stride": stride,
        "n_clear": clear,
        "n_clear_emit": clear_emit,
        "clear_ball_R": clear_r,
        "poc_pass_clear_R": bool(clear_r is not None and clear_r >= FOCUS_GOAL_R),
        "fn_buckets": buckets,
        "focus_conf_on_fn": focus_conf_bins,
        "sample_fn": fn_rows,
    }


def audit_quad_cache(path: Path) -> dict:
    dets = cache_load(path)
    cams = [c for c in CAMS if c in dets]
    calibs = {c: v for c, v in ((c, load_calib(c)) for c in cams) if v}
    focus = path.name.split("_")[3]  # det_cache_quad_P8_t...
    buckets = {}
    focus_conf_bins = {"ge080": 0, "050_079": 0, "020_049": 0, "lt020": 0, "none": 0}
    clear = clear_emit = 0
    n = len(next(iter(dets.values())))
    prev = None
    gap = 3
    for i in range(n):
        active = filter_active(dets, i, cams, MATCH3_THR_BY_CAM)
        is_clear = any(
            float(r[0][2]) >= CLEAR_SIDE and float(r[0][1]) >= 0.30
            for r in active.values()
        )
        if not is_clear:
            continue
        clear += 1
        fused, prev, gap = product_fuse(dets, i, calibs, cams, prev, gap)
        if fused is not None:
            clear_emit += 1
            continue
        tag, _ = classify_fn(dets, i, calibs, cams, focus)
        buckets[tag] = buckets.get(tag, 0) + 1
        br = best_raw(dets, i, focus)
        if br is None:
            focus_conf_bins["none"] += 1
        elif br["conf"] >= 0.80:
            focus_conf_bins["ge080"] += 1
        elif br["conf"] >= 0.50:
            focus_conf_bins["050_079"] += 1
        elif br["conf"] >= 0.20:
            focus_conf_bins["020_049"] += 1
        else:
            focus_conf_bins["lt020"] += 1
    clear_r = None if clear == 0 else round(clear_emit / clear, 3)
    return {
        "cache": path.name,
        "focus_cam": focus,
        "n_clear_proxy": clear,
        "n_clear_emit": clear_emit,
        "clear_ball_proxy_R": clear_r,
        "fn_buckets": buckets,
        "focus_conf_on_fn": focus_conf_bins,
    }


def recommend(rows: list[dict]) -> list[str]:
    tips = []
    det_miss = sum(
        r.get("fn_buckets", {}).get(k, 0)
        for r in rows
        for k in ("no_det_any", "no_det_focus", "focus_conf_below_emit")
    )
    map_miss = sum(r.get("fn_buckets", {}).get("focus_map_fail", 0) for r in rows)
    soft = sum(
        r.get("focus_conf_on_fn", {}).get("050_079", 0)
        + r.get("focus_conf_on_fn", {}).get("020_049", 0)
        for r in rows
    )
    if det_miss > map_miss and soft > 0:
        tips.append(
            f"Detection gap: {soft} FN frames have focus conf 0.20–0.79 — "
            "need stronger checkpoint or SAHI on quad cams (do not lower emit 0.80)."
        )
    if map_miss > 0:
        tips.append(
            f"Mapping gap: {map_miss} FN frames fail hull/off-pitch on focus cam — "
            "check L1 landmarks / hull_image_points."
        )
    none = sum(r.get("focus_conf_on_fn", {}).get("none", 0) for r in rows)
    if none > det_miss // 2:
        tips.append(
            f"No focus det on {none} FN frames — ball not detected on quad cam at thr 0.20."
        )
    if not tips:
        tips.append("FN mix unclear — inspect sample_fn rows in output JSON.")
    return tips


def main() -> int:
    strip_rows = [audit_strip(p) for p in STRIPS if p.is_file()]
    cache_rows = (
        [audit_quad_cache(p) for p in sorted(QUAD_CACHE.glob("det_cache_quad_*_thr010.json"))]
        if QUAD_CACHE.is_dir()
        else []
    )
    out = {
        "goals": {
            "P_emit": FOCUS_GOAL_P,
            "clear_ball_R": FOCUS_GOAL_R,
            "emit_conf": MAP_EMIT,
        },
        "strips": strip_rows,
        "quad_caches": cache_rows,
        "recommendations": recommend(strip_rows + cache_rows),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    for row in strip_rows:
        print(
            f"{row['pack']} focus={row['focus_cam']} clear_R={row['clear_ball_R']} "
            f"fn={row['fn_buckets']}"
        )
    for row in cache_rows:
        print(
            f"{row['cache']} proxy_R={row['clear_ball_proxy_R']} fn={row['fn_buckets']}"
        )
    for tip in out["recommendations"]:
        print("→", tip)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
