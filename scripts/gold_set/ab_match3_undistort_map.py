#!/usr/bin/env python3
"""A/B: defish-aware map_ball_box vs raw H on Match 3 strips (emit P >= 0.80)."""
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
    HOLD_MAX_GAP,
    fuse_balls_with_hold,
    load_calib,
    map_ball_box,
)

OUT = ROOT / "reports/eval_match3/improve_eng_loop/undistort_map_ab.json"
STRIPS = [
    ROOT / "data/processed/gold_sets/match3_quad_p10_31/labels.json",
    ROOT / "data/processed/gold_sets/match3_quad_p8_87/labels.json",
]
CAMS = ["P1", "P6", "P7", "P8", "P9", "P10", "P_Goal1", "P_Goal2"]
WH = (1920, 1080)


def mapped_at(dets, i, calibs, cams, apply_undistort):
    active = filter_active(dets, i, cams, MATCH3_THR_BY_CAM)
    rows = []
    for cam, pred_rows in active.items():
        rec = calibs.get(cam)
        if rec is None:
            continue
        box, conf, _ = pred_rows[0]
        hit = map_ball_box(
            rec,
            box,
            float(conf),
            frame_wh=WH,
            apply_undistort=apply_undistort,
        )
        if hit is not None:
            rows.append(hit)
    return rows


def score(labels, dets, calibs, focus, stride, apply_undistort):
    cams = [c for c in CAMS if c in dets]
    tp = fp = emit = clear = clear_emit = 0
    errs = []
    prev = None
    gap = HOLD_MAX_GAP + 1
    for fr in labels["frames"]:
        i = int(fr["i"])
        if stride > 1 and i % stride != 0:
            # still allow F0 hold path via fuse_balls_with_hold
            pass
        seed = (fr.get("cams") or {}).get(focus) or {}
        gold = seed.get("gold_xy")
        is_clear = bool(seed.get("clear")) and gold is not None
        if is_clear:
            clear += 1
        if stride > 1 and i % stride != 0:
            rows = []
        else:
            rows = mapped_at(dets, i, calibs, cams, apply_undistort)
        out = fuse_balls_with_hold(prev, rows, gap)
        if out is None:
            gap += 1
            continue
        if out.get("hold"):
            gap += 1
        else:
            prev = out
            gap = 0
        emit += 1
        if not is_clear or gold is None:
            continue
        clear_emit += 1
        err = math.hypot(out["xy"][0] - gold[0], out["xy"][1] - gold[1])
        errs.append(err)
        if err <= HIT_M:
            tp += 1
        else:
            fp += 1
    p_emit = tp / emit if emit else 0.0
    return {
        "emit": emit,
        "tp": tp,
        "fp": fp,
        "clear": clear,
        "clear_emit": clear_emit,
        "P_emit": round(p_emit, 4),
        "mean_err_m": round(sum(errs) / len(errs), 3) if errs else None,
        "pass": p_emit >= 0.80 if emit else False,
    }


def main() -> int:
    calibs = {c: load_calib(c) for c in CAMS}
    calibs = {c: v for c, v in calibs.items() if v}
    rows = []
    for path in STRIPS:
        if not path.is_file():
            print(f"skip missing {path}")
            continue
        labels = json.loads(path.read_text(encoding="utf-8"))
        focus = labels.get("focus_cam") or "P10"
        cache_rel = labels.get("det_cache")
        if not cache_rel:
            print(f"skip no det_cache in {path}")
            continue
        cache = ROOT / cache_rel
        if not cache.is_file():
            print(f"skip missing cache {cache}")
            continue
        dets = cache_load(cache)
        stride = infer_cache_stride(dets) or 1
        for name, flag in (("product_undistort", True), ("raw_no_undistort", False)):
            s = score(labels, dets, calibs, focus, stride, flag)
            s["strip"] = path.parent.name
            s["variant"] = name
            s["det_cache"] = cache_rel
            s["det_cache_stride"] = stride
            rows.append(s)
            print(
                f"{path.parent.name} {name}: P_emit={s['P_emit']} "
                f"emit={s['emit']} mean_err={s['mean_err_m']} pass={s['pass']}"
            )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"rows": rows}, indent=2), encoding="utf-8")
    print(f"wrote {OUT}")
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
