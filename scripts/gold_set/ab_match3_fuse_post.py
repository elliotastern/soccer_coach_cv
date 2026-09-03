#!/usr/bin/env python3
"""A/B Match 3 fuse post (F0–F3) on labeled strips. Gate: P_emit >= 0.80."""
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
    GHOST_CONF,
    HOLD_MAX_GAP,
    fuse_balls,
    fuse_balls_with_hold,
    load_calib,
    map_ball_box,
)

OUT = ROOT / "reports/eval_match3/improve_eng_loop/f_post_ab.json"
STRIPS = [
    ROOT / "data/processed/gold_sets/match3_quad_p10_31/labels.json",
    ROOT / "data/processed/gold_sets/match3_quad_p8_87/labels.json",
]
CAMS = ["P1", "P6", "P7", "P8", "P9", "P10", "P_Goal1", "P_Goal2"]
WH = (1920, 1080)
# name, f1, f2, hold, ghost_prune, reproj_prune
VARIANTS = (
    ("baseline", False, False, False, False, False),
    ("F1", True, False, False, False, False),
    ("F2", False, True, False, False, False),
    ("F3", False, False, False, True, False),
    ("F1+F2", True, True, False, False, False),
    ("F1+F2+F3", True, True, False, True, False),
    ("F1+F2+F0", True, True, True, False, False),
    ("F1+F2+F0+F3", True, True, True, True, False),
    ("F1+F2+F0+F3+F4", True, True, True, True, True),
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
    return rows


def score_variant(labels, dets, calibs, focus, stride, f1, f2, use_hold, ghost, reproj):
    cams = [c for c in CAMS if c in dets]
    tp = fp = emit = clear = clear_emit = 0
    errs = []
    prev = None
    gap = HOLD_MAX_GAP + 1
    frames = labels["frames"]
    for fr in frames:
        i = int(fr["i"])
        if stride > 1 and i % stride != 0 and not use_hold:
            continue
        seed = (fr.get("cams") or {}).get(focus) or {}
        gold = seed.get("gold_xy")
        is_clear = bool(seed.get("clear")) and gold is not None
        if is_clear:
            clear += 1
        rows = mapped_at(dets, i, calibs, cams)
        kwargs = dict(
            soft_dual_fallback=f1,
            solo_max_conf=f2,
            ghost_prune=ghost,
            ghost_conf=GHOST_CONF,
            reproj_prune=reproj,
        )
        if use_hold:
            fresh = fuse_balls(rows, **kwargs)
            if fresh is not None:
                fused = fresh
                prev = fresh
                gap = 0
            else:
                gap += 1
                fused = fuse_balls_with_hold(prev, [], gap, **kwargs)
        else:
            fused = fuse_balls(rows, **kwargs)
        if fused is None or gold is None:
            continue
        emit += 1
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
        "err_median_m": None
        if not errs
        else round(sorted(errs)[len(errs) // 2], 3),
        "poc_pass_P_emit": bool(p_emit is not None and p_emit >= 0.80),
        "poc_pass_clear_R": bool(clear_r is not None and clear_r >= 0.80),
    }


def score_strip_all_variants(path: Path) -> dict:
    labels = json.loads(path.read_text(encoding="utf-8"))
    dets = cache_load(ROOT / labels["det_cache"])
    cams = [c for c in CAMS if c in dets]
    calibs = {c: v for c, v in ((c, load_calib(c)) for c in cams) if v}
    focus = labels.get("focus_cam") or "P10"
    stride = infer_cache_stride(dets)
    rows = {}
    for name, f1, f2, hold, ghost, reproj in VARIANTS:
        rows[name] = score_variant(
            labels, dets, calibs, focus, stride, f1, f2, hold, ghost, reproj
        )
        rows[name]["flags"] = {
            "soft_dual_fallback": f1,
            "solo_max_conf": f2,
            "hold": hold,
            "ghost_prune": ghost,
            "reproj_prune": reproj,
        }
    return {
        "pack": labels.get("pack"),
        "path": str(path.relative_to(ROOT)),
        "det_cache_stride": stride,
        "variants": rows,
    }


PRODUCT_LOCKED = "F1+F2+F0+F3"


def pick_winner(by_strip: dict) -> str | None:
    """Survivors must pass P_emit on every available strip; rank by mean clear_R."""
    names = [v[0] for v in VARIANTS]
    survivors = []
    for name in names:
        ok = True
        rs = []
        for pack, block in by_strip.items():
            row = (block.get("variants") or {}).get(name) or {}
            if not row.get("poc_pass_P_emit"):
                ok = False
                break
            rs.append(float(row.get("clear_ball_R") or 0.0))
        if ok and rs:
            survivors.append((name, sum(rs) / len(rs), min(rs)))
    if not survivors:
        return None
    survivors.sort(key=lambda t: (-t[1], -t[2], -len(t[0]), t[0]))
    return survivors[0][0]


def product_passes(by_strip: dict, name: str = PRODUCT_LOCKED) -> bool:
    for block in by_strip.values():
        row = (block.get("variants") or {}).get(name) or {}
        if not row.get("poc_pass_P_emit"):
            return False
    return bool(by_strip)


def main() -> int:
    by_strip = {}
    for path in STRIPS:
        if not path.is_file():
            print(f"skip missing strip {path}")
            continue
        block = score_strip_all_variants(path)
        by_strip[block["pack"] or path.parent.name] = block
        print(block["pack"], {k: block["variants"][k]["clear_ball_R"] for k in block["variants"]})
    if not by_strip:
        print("no strips")
        return 1
    ranked = pick_winner(by_strip)
    # Product fuse stays F1+F2+F0+F3 whenever it clears P_emit on all strips.
    winner = PRODUCT_LOCKED if product_passes(by_strip) else ranked
    out = {
        "hit_m": HIT_M,
        "ghost_conf": GHOST_CONF,
        "strips": by_strip,
        "winner": winner,
        "winner_ab_rank": ranked,
        "winner_note": (
            f"Product locked at {PRODUCT_LOCKED}; A/B mean-clear_R pick was {ranked}."
            if ranked != winner
            else f"A/B and product agree on {winner}."
        ),
        "gate": "P_emit >= 0.80 on all strips; product prefers F1+F2+F0+F3",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"winner={winner}")
    print(f"wrote {OUT}")
    return 0 if winner else 1


if __name__ == "__main__":
    raise SystemExit(main())
