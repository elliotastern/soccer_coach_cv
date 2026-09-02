#!/usr/bin/env python3
"""A/B AGREE_M 4.0 (product) vs 3.0 / 2.5 on holdout + P10/P8 strips.

Product default stays AGREE_M=4.0. Gold hit distance for P_emit stays 4.0 m.
Kill if holdout clear proxy < 0.884 or strip P_emit < 0.80 (when base passes).
Promote only if gates hold and agree_among_emit rises ≥ +0.02.
"""
from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "gold_set"))

from ab_match3_defish_detect import CAMS, CLEAR_SIDE, WH  # noqa: E402
from ab_match3_fuse_post import STRIPS  # noqa: E402
from eval_match2_top_left_multicam_baseline import cache_load  # noqa: E402
from multicam_select_policy import MATCH3_THR_BY_CAM, filter_active  # noqa: E402
from score_match3_ball_m1 import HOLD_MAX_GAP  # noqa: E402
from src.mapping.match3_xy import (  # noqa: E402
    AGREE_M,
    GHOST_CONF,
    fuse_balls,
    fuse_balls_with_hold,
    load_calib,
    map_ball_box,
)

OUT = ROOT / "reports/eval_match3/improve_eng_loop/agree_m_ab.json"
DEAD_ENDS = ROOT / "reports/ball_testing/DEAD_ENDS.md"
HOLDOUT_DIR = ROOT / "reports/eval_match3/pitchmap_gallery_holdout"
HOLDOUT_GATE = 0.884
HIT_M = 4.0  # gold err gate stays product radius
AGREE_LIFT = 0.02
P10_PACK = "match3_quad_p10_31"

BASE_KW = dict(
    soft_dual_fallback=True,
    solo_max_conf=True,
    ghost_prune=True,
    ghost_conf=GHOST_CONF,
    reproj_prune=False,
)


def mapped_at(dets, i, calibs, cams):
    active = filter_active(dets, i, cams, MATCH3_THR_BY_CAM)
    rows = []
    for cam, pred_rows in active.items():
        rec = calibs.get(cam)
        if rec is None:
            continue
        box, conf, _ = pred_rows[0]
        hit = map_ball_box(rec, box, float(conf), frame_wh=WH)
        if hit is not None:
            rows.append(hit)
    return active, rows


def score_strip(labels, dets, calibs, focus, agree_m: float) -> dict:
    cams = [c for c in CAMS if c in dets]
    kw = {**BASE_KW, "agree_m": agree_m}
    tp = fp = emit = clear = clear_emit = agree = 0
    errs = []
    prev = None
    gap = HOLD_MAX_GAP + 1
    for fr in labels["frames"]:
        i = int(fr["i"])
        seed = (fr.get("cams") or {}).get(focus) or {}
        gold = seed.get("gold_xy")
        is_clear = bool(seed.get("clear")) and gold is not None
        if is_clear:
            clear += 1
        _a, rows = mapped_at(dets, i, calibs, cams)
        fresh = fuse_balls(rows, **kw)
        if fresh is not None:
            fused = fresh
            prev = fresh
            gap = 0
        else:
            gap += 1
            fused = fuse_balls_with_hold(
                prev, [], gap, hold_max_gap=HOLD_MAX_GAP, **kw
            )
        if fused is None or gold is None:
            continue
        emit += 1
        if fused.get("agree"):
            agree += 1
        err = math.hypot(
            float(fused["xy"][0]) - float(gold[0]),
            float(fused["xy"][1]) - float(gold[1]),
        )
        errs.append(err)
        if err <= HIT_M:
            tp += 1
        else:
            fp += 1
        if is_clear:
            clear_emit += 1
    p_emit = None if (tp + fp) == 0 else tp / (tp + fp)
    clear_r = None if clear == 0 else clear_emit / clear
    return {
        "agree_m": agree_m,
        "P_emit": None if p_emit is None else round(p_emit, 3),
        "clear_ball_R": None if clear_r is None else round(clear_r, 3),
        "agree_among_emit": None if emit == 0 else round(agree / emit, 3),
        "err_median_m": None
        if not errs
        else round(sorted(errs)[len(errs) // 2], 3),
        "n_emit_scored": emit,
        "n_clear": clear,
    }


def score_holdout(agree_m: float) -> dict:
    kw = {**BASE_KW, "agree_m": agree_m}
    rows = []
    for path in sorted((HOLDOUT_DIR / "det_cache").glob("det_cache_*.json")):
        dets = cache_load(path)
        cams = [c for c in CAMS if c in dets]
        n = len(next(iter(dets.values())))
        calibs = {c: v for c, v in ((c, load_calib(c)) for c in cams) if v}
        clear = clear_emit = emit = agree = 0
        prev = None
        gap = 0
        for i in range(n):
            active, mapped = mapped_at(dets, i, calibs, cams)
            is_clear = any(
                float(rs[0][2]) >= CLEAR_SIDE and float(rs[0][1]) >= 0.30
                for rs in active.values()
            )
            fresh = fuse_balls(mapped, **kw)
            if fresh is not None:
                fused = fresh
                prev = fresh
                gap = 0
            else:
                gap += 1
                fused = fuse_balls_with_hold(
                    prev, [], gap, hold_max_gap=HOLD_MAX_GAP, **kw
                )
            if fused is not None:
                emit += 1
                if fused.get("agree"):
                    agree += 1
            if is_clear:
                clear += 1
                if fused is not None:
                    clear_emit += 1
        rows.append(
            {
                "cache": path.name,
                "clear_frames": clear,
                "clear_emit": clear_emit,
                "emit": emit,
                "agree": agree,
            }
        )
    clear = sum(int(r["clear_frames"]) for r in rows)
    clear_emit = sum(int(r["clear_emit"]) for r in rows)
    emit = sum(int(r["emit"]) for r in rows)
    agree = sum(int(r["agree"]) for r in rows)
    return {
        "agree_m": agree_m,
        "clear_ball_proxy_R": None if clear == 0 else round(clear_emit / clear, 3),
        "agree_among_emit": None if emit == 0 else round(agree / emit, 3),
        "emit": emit,
    }


def strip_ok(base: dict, cand: dict) -> bool:
    b_p, c_p = base.get("P_emit"), cand.get("P_emit")
    if b_p is not None and b_p >= 0.80:
        if c_p is None or c_p < 0.80:
            return False
    b_r, c_r = base.get("clear_ball_R"), cand.get("clear_ball_R")
    if b_r is not None and c_r is not None and c_r + 1e-9 < b_r - 0.02:
        return False
    return True


def append_dead_end(row: str) -> None:
    if not DEAD_ENDS.is_file():
        return
    text = DEAD_ENDS.read_text(encoding="utf-8")
    if "AGREE_M shrink" in text:
        return
    if not text.endswith("\n"):
        text += "\n"
    text += row + "\n"
    DEAD_ENDS.write_text(text, encoding="utf-8")


def main() -> int:
    if abs(AGREE_M - 4.0) > 1e-9:
        print(f"product AGREE_M={AGREE_M} expected 4.0", file=sys.stderr)
        return 1
    variants = [4.0, 3.0, 2.5]
    holdout = {m: score_holdout(m) for m in variants}
    strips = {}
    for path in STRIPS:
        if not path.is_file():
            continue
        labels = json.loads(path.read_text(encoding="utf-8"))
        dets = cache_load(ROOT / labels["det_cache"])
        cams = [c for c in CAMS if c in dets]
        calibs = {c: v for c, v in ((c, load_calib(c)) for c in cams) if v}
        focus = labels.get("focus_cam") or "P10"
        pack = path.parent.name
        strips[pack] = {
            str(m): score_strip(labels, dets, calibs, focus, m) for m in variants
        }

    base_h = holdout[4.0]
    promote = {}
    for m in (3.0, 2.5):
        h = holdout[m]
        hold_ok = (
            h.get("clear_ball_proxy_R") is not None
            and h["clear_ball_proxy_R"] >= HOLDOUT_GATE
            and h["clear_ball_proxy_R"] + 1e-9
            >= (base_h.get("clear_ball_proxy_R") or 0) - 0.01
        )
        strips_ok = True
        for pack, by_m in strips.items():
            if not strip_ok(by_m["4.0"], by_m[str(m)]):
                strips_ok = False
                break
        agree_b = base_h.get("agree_among_emit") or 0.0
        agree_c = h.get("agree_among_emit") or 0.0
        agree_lift = agree_c - agree_b >= AGREE_LIFT - 1e-9
        promote[str(m)] = bool(hold_ok and strips_ok and agree_lift)

    payload = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "product_agree_m": AGREE_M,
        "hit_m_gold": HIT_M,
        "holdout_gate": HOLDOUT_GATE,
        "holdout": {str(k): v for k, v in holdout.items()},
        "strips": strips,
        "promote": promote,
        "any_promote": any(promote.values()),
        "note": "Product default stays AGREE_M=4.0 unless promote[*]=true",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({
        "holdout": {str(k): v.get("clear_ball_proxy_R") for k, v in holdout.items()},
        "agree": {str(k): v.get("agree_among_emit") for k, v in holdout.items()},
        "promote": promote,
        "wrote": str(OUT),
    }, indent=2))
    if not any(promote.values()):
        append_dead_end(
            "| **AGREE_M shrink** 4→3.0/2.5 | holdout/strips fail or agree lift "
            f"< {AGREE_LIFT} | **Skipped** — product stays 4.0 | "
            "`agree_m_ab.json`"
        )
    return 0 if any(promote.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
