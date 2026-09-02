#!/usr/bin/env python3
"""A/B ball contact point (bbox foot modes) on M1 strips + holdout proxy.

Product fuse stays F1+F2+F0+F3. Only ``map_ball_box(..., foot_mode=)`` changes.
Kill if strip P_emit < 0.80 or holdout clear proxy < 0.884.
Promote if gates hold and agree_among_emit rises (≥ +0.02) or clear_R / err improves.
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
from score_match3_ball_m1 import HIT_M, HOLD_MAX_GAP, PRODUCT_FUSE_KW, infer_cache_stride  # noqa: E402
from src.mapping.match3_xy import (  # noqa: E402
    BALL_FOOT_BOTTOM,
    BALL_FOOT_MODES,
    fuse_balls,
    fuse_balls_with_hold,
    load_calib,
    map_ball_box,
)

OUT = ROOT / "reports/eval_match3/improve_eng_loop/ball_foot_ab.json"
DEAD_ENDS = ROOT / "reports/ball_testing/DEAD_ENDS.md"
HOLDOUT_DIR = ROOT / "reports/eval_match3/pitchmap_gallery_holdout"
HOLDOUT_GATE = 0.884
AGREE_LIFT = 0.02


def mapped_at(dets, i, calibs, cams, foot_mode: str):
    active = filter_active(dets, i, cams, MATCH3_THR_BY_CAM)
    rows = []
    for cam, pred_rows in active.items():
        rec = calibs.get(cam)
        if rec is None:
            continue
        box, conf, _ = pred_rows[0]
        hit = map_ball_box(
            rec, box, float(conf), frame_wh=WH, foot_mode=foot_mode
        )
        if hit is not None:
            rows.append(hit)
    return active, rows


def score_strip(labels, dets, calibs, focus, stride, foot_mode: str) -> dict:
    cams = [c for c in CAMS if c in dets]
    tp = fp = emit = clear = clear_emit = agree = 0
    errs = []
    pair_errs = []
    prev = None
    gap = HOLD_MAX_GAP + 1
    for fr in labels["frames"]:
        i = int(fr["i"])
        seed = (fr.get("cams") or {}).get(focus) or {}
        gold = seed.get("gold_xy")
        is_clear = bool(seed.get("clear")) and gold is not None
        if is_clear:
            clear += 1
        _active, rows = mapped_at(dets, i, calibs, cams, foot_mode)
        if len(rows) >= 2:
            xs = [float(r["xy"][0]) for r in rows]
            ys = [float(r["xy"][1]) for r in rows]
            pair_errs.append(
                math.hypot(max(xs) - min(xs), max(ys) - min(ys))
            )
        fresh = fuse_balls(rows, **PRODUCT_FUSE_KW)
        if fresh is not None:
            fused = fresh
            prev = fresh
            gap = 0
        else:
            gap += 1
            fused = fuse_balls_with_hold(
                prev, [], gap, hold_max_gap=HOLD_MAX_GAP, **PRODUCT_FUSE_KW
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
        "n_clear": clear,
        "n_emit_scored": emit,
        "tp": tp,
        "fp": fp,
        "P_emit": None if p_emit is None else round(p_emit, 3),
        "clear_ball_R": None if clear_r is None else round(clear_r, 3),
        "agree_among_emit": None if emit == 0 else round(agree / emit, 3),
        "err_median_m": None
        if not errs
        else round(sorted(errs)[len(errs) // 2], 3),
        "pair_span_median_m": None
        if not pair_errs
        else round(sorted(pair_errs)[len(pair_errs) // 2], 3),
        "poc_pass_P_emit": bool(p_emit is not None and p_emit >= 0.80),
        "poc_pass_clear_R": bool(clear_r is not None and clear_r >= 0.80),
        "det_cache_stride": stride,
    }


def score_holdout(foot_mode: str) -> dict:
    det_dir = HOLDOUT_DIR / "det_cache"
    rows = []
    for path in sorted(det_dir.glob("det_cache_*.json")):
        dets = cache_load(path)
        cams = [c for c in CAMS if c in dets]
        n = len(next(iter(dets.values())))
        calibs = {c: v for c, v in ((c, load_calib(c)) for c in cams) if v}
        clear = clear_emit = emit = agree = 0
        prev = None
        gap = 0
        for i in range(n):
            active, mapped = mapped_at(dets, i, calibs, cams, foot_mode)
            is_clear = any(
                float(rows[0][2]) >= CLEAR_SIDE and float(rows[0][1]) >= 0.30
                for rows in active.values()
            )
            fresh = fuse_balls(mapped, **PRODUCT_FUSE_KW)
            if fresh is not None:
                fused = fresh
                prev = fresh
                gap = 0
            else:
                gap += 1
                fused = fuse_balls_with_hold(
                    prev, [], gap, hold_max_gap=HOLD_MAX_GAP, **PRODUCT_FUSE_KW
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
                "clear_ball_proxy_R": None
                if clear == 0
                else round(clear_emit / clear, 3),
                "agree_among_emit": None if emit == 0 else round(agree / emit, 3),
            }
        )
    clear = sum(int(r["clear_frames"]) for r in rows)
    clear_emit = sum(int(r["clear_emit"]) for r in rows)
    emit = sum(int(r["emit"]) for r in rows)
    agree = sum(int(r["agree"]) for r in rows)
    return {
        "caches": rows,
        "clear_ball_proxy_R": None if clear == 0 else round(clear_emit / clear, 3),
        "agree_among_emit": None if emit == 0 else round(agree / emit, 3),
        "emit": emit,
    }


def gates_ok(base_strip: dict, cand_strip: dict, base_h: dict, cand_h: dict) -> bool:
    if not cand_strip.get("poc_pass_P_emit") or not cand_strip.get("poc_pass_clear_R"):
        return False
    b_err = base_strip.get("err_median_m")
    c_err = cand_strip.get("err_median_m")
    if b_err is not None and c_err is not None and c_err > b_err + 0.2:
        return False
    cand_r = float(cand_h.get("clear_ball_proxy_R") or 0)
    base_r = float(base_h.get("clear_ball_proxy_R") or 0)
    if cand_r < HOLDOUT_GATE:
        return False
    if cand_r < base_r - 0.01:
        return False
    return True


def is_win(base_strip: dict, cand_strip: dict, base_h: dict, cand_h: dict) -> bool:
    b_agree = float(base_h.get("agree_among_emit") or 0)
    c_agree = float(cand_h.get("agree_among_emit") or 0)
    if c_agree >= b_agree + AGREE_LIFT:
        return True
    b_clear = float(base_strip.get("clear_ball_R") or 0)
    c_clear = float(cand_strip.get("clear_ball_R") or 0)
    if c_clear >= b_clear + 0.01:
        return True
    b_err = base_strip.get("err_median_m")
    c_err = cand_strip.get("err_median_m")
    if b_err is not None and c_err is not None and c_err + 0.05 < b_err:
        return True
    b_span = base_strip.get("pair_span_median_m")
    c_span = cand_strip.get("pair_span_median_m")
    if b_span is not None and c_span is not None and c_span + 0.3 < b_span:
        return True
    return False


def main() -> int:
    strip_packs = {}
    for path in STRIPS:
        if not path.is_file():
            continue
        labels = json.loads(path.read_text(encoding="utf-8"))
        dets = cache_load(ROOT / labels["det_cache"])
        cams = [c for c in CAMS if c in dets]
        calibs = {c: v for c, v in ((c, load_calib(c)) for c in cams) if v}
        focus = labels.get("focus_cam") or "P10"
        stride = infer_cache_stride(dets)
        modes = {}
        for mode in BALL_FOOT_MODES:
            modes[mode] = score_strip(labels, dets, calibs, focus, stride, mode)
        strip_packs[path.parent.name] = {
            "path": str(path.relative_to(ROOT)),
            "modes": modes,
        }

    holdout = {mode: score_holdout(mode) for mode in BALL_FOOT_MODES}
    base_h = holdout[BALL_FOOT_BOTTOM]
    winners = []
    for mode in BALL_FOOT_MODES:
        if mode == BALL_FOOT_BOTTOM:
            continue
        cand_h = holdout[mode]
        ok_all = True
        win_any = False
        for pack, block in strip_packs.items():
            base_s = block["modes"][BALL_FOOT_BOTTOM]
            cand_s = block["modes"][mode]
            if not gates_ok(base_s, cand_s, base_h, cand_h):
                ok_all = False
                break
            if is_win(base_s, cand_s, base_h, cand_h):
                win_any = True
        if ok_all and win_any:
            winners.append(mode)

    promote = bool(winners)
    report = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "hit_m": HIT_M,
        "holdout_gate": HOLDOUT_GATE,
        "agree_lift": AGREE_LIFT,
        "modes": list(BALL_FOOT_MODES),
        "strips": strip_packs,
        "holdout": {
            m: {
                "clear_ball_proxy_R": holdout[m].get("clear_ball_proxy_R"),
                "agree_among_emit": holdout[m].get("agree_among_emit"),
                "emit": holdout[m].get("emit"),
            }
            for m in BALL_FOOT_MODES
        },
        "promote_foot_modes": winners,
        "promote": promote,
        "note": "Default product foot stays bottom unless a mode is promoted.",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("holdout clear_R / agree:")
    for mode in BALL_FOOT_MODES:
        h = holdout[mode]
        print(
            f"  {mode:8s}  R={h.get('clear_ball_proxy_R')}  "
            f"agree={h.get('agree_among_emit')}  emit={h.get('emit')}"
        )
    for name, block in strip_packs.items():
        print(name)
        for mode in BALL_FOOT_MODES:
            s = block["modes"][mode]
            print(
                f"  {mode:8s}  P={s.get('P_emit')}  clear_R={s.get('clear_ball_R')}  "
                f"agree={s.get('agree_among_emit')}  err={s.get('err_median_m')}  "
                f"span={s.get('pair_span_median_m')}"
            )
    print(f"promote={promote} winners={winners}")
    print(f"wrote {OUT}")

    if not promote:
        row = (
            f"| **Ball foot-point modes** (inset/center/radius) | "
            f"holdout/strips A/B | **Skipped** — no gate-safe win vs bottom | "
            f"See `{OUT.relative_to(ROOT)}` |"
        )
        text = DEAD_ENDS.read_text(encoding="utf-8") if DEAD_ENDS.is_file() else ""
        if "Ball foot-point modes" not in text:
            lines = text.splitlines()
            insert = next(
                (i for i, ln in enumerate(lines) if ln.startswith("## Worked")),
                len(lines),
            )
            lines.insert(insert, row)
            DEAD_ENDS.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0 if promote else 1


if __name__ == "__main__":
    raise SystemExit(main())
