#!/usr/bin/env python3
"""A/B soft / gated F4 vs product F0–F3 (never replace product default).

Variants:
  base       — F1+F2+F0+F3
  soft48/72  — reproj_agree_gate (demote agree→solo; never drop maps)
  n3_48/72   — hard reproj prune only when ≥3 mapped cams

Gates (relative): holdout proxy ≥ 0.884 and not worse than base;
P10 strip P_emit/clear_R held; P8 compared only as delta vs base.
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
from score_match3_ball_m1 import HIT_M, HOLD_MAX_GAP, infer_cache_stride  # noqa: E402
from src.mapping.match3_xy import (  # noqa: E402
    GHOST_CONF,
    fuse_balls,
    fuse_balls_with_hold,
    load_calib,
    map_ball_box,
)

OUT = ROOT / "reports/eval_match3/improve_eng_loop/soft_f4_ab.json"
DEAD_ENDS = ROOT / "reports/ball_testing/DEAD_ENDS.md"
HOLDOUT_DIR = ROOT / "reports/eval_match3/pitchmap_gallery_holdout"
HOLDOUT_GATE = 0.884
P10_PACK = "match3_quad_p10_31"

BASE_KW = dict(
    soft_dual_fallback=True,
    solo_max_conf=True,
    ghost_prune=True,
    ghost_conf=GHOST_CONF,
    reproj_prune=False,
    reproj_agree_gate=False,
    reproj_min_n=2,
    reproj_max_px=48.0,
)
VARIANTS = {
    "base": {**BASE_KW},
    "soft48": {**BASE_KW, "reproj_agree_gate": True, "reproj_max_px": 48.0},
    "soft72": {**BASE_KW, "reproj_agree_gate": True, "reproj_max_px": 72.0},
    "n3_48": {
        **BASE_KW,
        "reproj_prune": True,
        "reproj_min_n": 3,
        "reproj_max_px": 48.0,
    },
    "n3_72": {
        **BASE_KW,
        "reproj_prune": True,
        "reproj_min_n": 3,
        "reproj_max_px": 72.0,
    },
}


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


def score_strip(labels, dets, calibs, focus, kw: dict) -> dict:
    cams = [c for c in CAMS if c in dets]
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
        "P_emit": None if p_emit is None else round(p_emit, 3),
        "clear_ball_R": None if clear_r is None else round(clear_r, 3),
        "agree_among_emit": None if emit == 0 else round(agree / emit, 3),
        "err_median_m": None
        if not errs
        else round(sorted(errs)[len(errs) // 2], 3),
        "n_emit_scored": emit,
        "n_clear": clear,
    }


def score_holdout(kw: dict) -> dict:
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
        "clear_ball_proxy_R": None if clear == 0 else round(clear_emit / clear, 3),
        "agree_among_emit": None if emit == 0 else round(agree / emit, 3),
        "emit": emit,
    }


def relative_ok(base_s: dict, cand_s: dict) -> bool:
    """Cand must not regress strip metrics vs base (P8 may already fail absolute)."""
    b_p = base_s.get("P_emit")
    c_p = cand_s.get("P_emit")
    if b_p is not None and c_p is not None and c_p + 1e-9 < b_p - 0.02:
        return False
    b_r = float(base_s.get("clear_ball_R") or 0)
    c_r = float(cand_s.get("clear_ball_R") or 0)
    if c_r + 1e-9 < b_r - 0.02:
        return False
    b_err = base_s.get("err_median_m")
    c_err = cand_s.get("err_median_m")
    if b_err is not None and c_err is not None and c_err > b_err + 0.2:
        return False
    return True


def is_win(base_s: dict, cand_s: dict, base_h: dict, cand_h: dict) -> bool:
    if float(cand_h.get("clear_ball_proxy_R") or 0) >= float(
        base_h.get("clear_ball_proxy_R") or 0
    ) + 0.01:
        return True
    b_err = base_s.get("err_median_m")
    c_err = cand_s.get("err_median_m")
    if b_err is not None and c_err is not None and c_err + 0.05 < b_err:
        return True
    b_p = base_s.get("P_emit")
    c_p = cand_s.get("P_emit")
    if b_p is not None and c_p is not None and c_p >= b_p + 0.02:
        return True
    return False


def main() -> int:
    strips = {}
    for path in STRIPS:
        if not path.is_file():
            continue
        labels = json.loads(path.read_text(encoding="utf-8"))
        dets = cache_load(ROOT / labels["det_cache"])
        cams = [c for c in CAMS if c in dets]
        calibs = {c: v for c, v in ((c, load_calib(c)) for c in cams) if v}
        focus = labels.get("focus_cam") or "P10"
        modes = {
            name: score_strip(labels, dets, calibs, focus, kw)
            for name, kw in VARIANTS.items()
        }
        strips[path.parent.name] = modes

    holdout = {name: score_holdout(kw) for name, kw in VARIANTS.items()}
    base_h = holdout["base"]
    winners = []
    for name in VARIANTS:
        if name == "base":
            continue
        cand_h = holdout[name]
        if float(cand_h.get("clear_ball_proxy_R") or 0) < HOLDOUT_GATE:
            continue
        if float(cand_h.get("clear_ball_proxy_R") or 0) < float(
            base_h.get("clear_ball_proxy_R") or 0
        ) - 0.01:
            continue
        p10 = strips.get(P10_PACK) or {}
        if not p10:
            continue
        if not relative_ok(p10["base"], p10[name]):
            continue
        if float(p10[name].get("P_emit") or 0) < 0.80:
            continue
        if float(p10[name].get("clear_ball_R") or 0) < 0.80:
            continue
        ok_others = True
        for pack, modes in strips.items():
            if not relative_ok(modes["base"], modes[name]):
                ok_others = False
                break
        if not ok_others:
            continue
        if is_win(p10["base"], p10[name], base_h, cand_h):
            winners.append(name)

    promote = bool(winners)
    report = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "holdout_gate": HOLDOUT_GATE,
        "hit_m": HIT_M,
        "variants": list(VARIANTS),
        "strips": strips,
        "holdout": holdout,
        "promote_soft_f4": winners,
        "promote": promote,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("holdout:")
    for name, h in holdout.items():
        print(
            f"  {name:8s} R={h.get('clear_ball_proxy_R')} "
            f"agree={h.get('agree_among_emit')} emit={h.get('emit')}"
        )
    for pack, modes in strips.items():
        print(pack)
        for name, s in modes.items():
            print(
                f"  {name:8s} P={s.get('P_emit')} clear_R={s.get('clear_ball_R')} "
                f"agree={s.get('agree_among_emit')} err={s.get('err_median_m')}"
            )
    print(f"promote={promote} winners={winners}")
    print(f"wrote {OUT}")
    if not promote:
        row = (
            f"| **Soft / gated F4** (agree-gate + n≥3 prune) | "
            f"holdout/strips A/B | **Skipped** — no gate-safe win vs F0–F3 | "
            f"See `{OUT.relative_to(ROOT)}` |"
        )
        text = DEAD_ENDS.read_text(encoding="utf-8") if DEAD_ENDS.is_file() else ""
        if "Soft / gated F4" not in text:
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
