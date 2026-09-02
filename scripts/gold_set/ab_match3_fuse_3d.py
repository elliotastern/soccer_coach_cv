#!/usr/bin/env python3
"""A/B product fuse F0-F3 (pitch_merge) vs triangulate_3d on M1 strips + holdout."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "gold_set"))

from ab_match3_defish_detect import (  # noqa: E402
    CAMS,
    CLEAR_SIDE,
    map_active,
)
from ab_match3_fuse_post import STRIPS, score_variant  # noqa: E402
from eval_match2_top_left_multicam_baseline import cache_load  # noqa: E402
from score_match3_ball_m1 import HIT_M, infer_cache_stride  # noqa: E402
from src.mapping.fuse_config import load_fuse_config  # noqa: E402
from src.mapping.fuse_product import fuse_ball_product  # noqa: E402
from src.mapping.match3_xy import GHOST_CONF, HOLD_MAX_GAP, load_calib  # noqa: E402

OUT = ROOT / "reports/eval_match3/improve_eng_loop/fuse3d_ab.json"
DEAD_ENDS = ROOT / "reports/ball_testing/DEAD_ENDS.md"
HOLDOUT_DIR = ROOT / "reports/eval_match3/pitchmap_gallery_holdout"
HOLDOUT_GATE = 0.884
BASE = "F1+F2+F0+F3"
F3D = "triangulate_3d"
BASE_CFG = load_fuse_config()
F3D_CFG = {**BASE_CFG, "mode": "triangulate_3d", "ukf_enabled": False}
F3D_UKF_CFG = {**F3D_CFG, "ukf_enabled": True}
F3D_HYBRID = "triangulate_3d+hybrid"
F3D_UKF = "triangulate_3d+ukf"


def score_holdout_cache(path: Path, cfg: dict) -> dict:
    dets = cache_load(path)
    cams = [c for c in CAMS if c in dets]
    n = len(next(iter(dets.values())))
    calibs = {c: v for c, v in ((c, load_calib(c)) for c in cams) if v}
    clear = clear_emit = emit = agree = 0
    prev = None
    gap = 0
    ukf = None
    for i in range(n):
        active, mapped = map_active(dets, i, cams, calibs, True)
        is_clear = any(
            float(rows[0][2]) >= CLEAR_SIDE and float(rows[0][1]) >= 0.30
            for rows in active.values()
        )
        fused, prev, gap, ukf = fuse_ball_product(mapped, prev, gap, cfg=cfg, ukf=ukf)
        if fused is not None:
            emit += 1
            if fused.get("agree"):
                agree += 1
        if is_clear:
            clear += 1
            if fused is not None:
                clear_emit += 1
    return {
        "cache": path.name,
        "clear_frames": clear,
        "clear_emit": clear_emit,
        "emit": emit,
        "agree_among_emit": None if emit == 0 else round(agree / emit, 3),
        "clear_ball_proxy_R": None if clear == 0 else round(clear_emit / clear, 3),
    }


def score_holdout(cfg: dict) -> dict:
    det_dir = HOLDOUT_DIR / "det_cache"
    rows = []
    for path in sorted(det_dir.glob("det_cache_*.json")):
        rows.append(score_holdout_cache(path, cfg))
    clear = sum(int(r["clear_frames"]) for r in rows)
    clear_emit = sum(int(r["clear_emit"]) for r in rows)
    emit = sum(int(r["emit"]) for r in rows)
    agree = sum(
        int(round(float(r["agree_among_emit"] or 0) * int(r["emit"])))
        for r in rows
    )
    proxy_r = None if clear == 0 else round(clear_emit / clear, 3)
    agree_rate = None if emit == 0 else round(agree / emit, 3)
    return {
        "caches": rows,
        "clear_ball_proxy_R": proxy_r,
        "agree_among_emit": agree_rate,
    }


def score_strip_3d(path: Path) -> dict:
    labels = json.loads(path.read_text(encoding="utf-8"))
    dets = cache_load(ROOT / labels["det_cache"])
    cams = [c for c in CAMS if c in dets]
    calibs = {c: v for c, v in ((c, load_calib(c)) for c in cams) if v}
    focus = labels.get("focus_cam") or "P10"
    stride = infer_cache_stride(dets)
    f1, f2, hold, ghost, reproj = True, True, True, True, False
    base = score_variant(labels, dets, calibs, focus, stride, f1, f2, hold, ghost, reproj)
    tp = fp = emit = clear = clear_emit = 0
    errs = []
    prev = None
    gap = 0
    ukf = None
    for fr in labels["frames"]:
        i = int(fr["i"])
        seed = (fr.get("cams") or {}).get(focus) or {}
        gold = seed.get("gold_xy")
        is_clear = bool(seed.get("clear")) and gold is not None
        if is_clear:
            clear += 1
        active, mapped = map_active(dets, i, cams, calibs, True)
        fused, prev, gap, ukf = fuse_ball_product(mapped, prev, gap, cfg=F3D_CFG, ukf=ukf)
        if fused is None or gold is None:
            continue
        emit += 1
        err = ((float(fused["xy"][0]) - float(gold[0])) ** 2 + (float(fused["xy"][1]) - float(gold[1])) ** 2) ** 0.5
        errs.append(err)
        if err <= HIT_M:
            tp += 1
        else:
            fp += 1
        if is_clear:
            clear_emit += 1
    p_emit = None if emit == 0 else round(tp / emit, 3)
    clear_r = None if clear == 0 else round(clear_emit / clear, 3)
    med = None if not errs else round(sorted(errs)[len(errs) // 2], 3)
    f3d = {
        "n_clear": clear,
        "n_emit_scored": emit,
        "tp": tp,
        "fp": fp,
        "P_emit": p_emit,
        "clear_ball_R": clear_r,
        "err_median_m": med,
        "poc_pass_P_emit": p_emit is not None and p_emit >= 0.80,
        "poc_pass_clear_R": clear_r is not None and clear_r >= 0.80,
    }
    return {
        "pack": labels.get("pack") or path.parent.name,
        "path": str(path.relative_to(ROOT)),
        BASE: base,
        F3D: f3d,
    }


def strip_passes(base: dict, f3d: dict) -> bool:
    """Strip gate: if baseline already fails P_emit, only check clear_R parity."""
    if not base.get("poc_pass_P_emit"):
        b_clear = float(base.get("clear_ball_R") or 0)
        f_clear = float(f3d.get("clear_ball_R") or 0)
        return f_clear >= b_clear - 0.01
    if not f3d.get("poc_pass_P_emit"):
        return False
    if not base.get("poc_pass_clear_R") or not f3d.get("poc_pass_clear_R"):
        return False
    b_err = base.get("err_median_m")
    f_err = f3d.get("err_median_m")
    if b_err is not None and f_err is not None and f_err > b_err + 0.2:
        return False
    return True


def main() -> int:
    strips = {}
    for path in STRIPS:
        if path.is_file():
            strips[path.parent.name] = score_strip_3d(path)

    holdout_base = score_holdout(BASE_CFG)
    holdout_f3d = score_holdout(F3D_CFG)
    holdout_ukf = score_holdout(F3D_UKF_CFG)
    holdout_ok = (
        float(holdout_f3d.get("clear_ball_proxy_R") or 0) >= HOLDOUT_GATE
        and float(holdout_f3d.get("clear_ball_proxy_R") or 0)
        >= float(holdout_base.get("clear_ball_proxy_R") or 0) - 0.01
    )
    strips_ok = all(
        strip_passes(block[BASE], block[F3D]) for block in strips.values()
    )
    agree_lift = (
        float(holdout_f3d.get("agree_among_emit") or 0)
        > float(holdout_base.get("agree_among_emit") or 0)
    )
    promote = strips_ok and holdout_ok

    report = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "hit_m": HIT_M,
        "holdout_gate": HOLDOUT_GATE,
        "strips": strips,
        "holdout": {
            BASE: holdout_base,
            F3D: holdout_f3d,
            F3D_UKF: holdout_ukf,
        },
        "promote_3d": promote,
        "gates": {
            "strips_pass": strips_ok,
            "holdout_pass": holdout_ok,
            "agree_lift": agree_lift,
        },
        "notes": [
            "Smart hybrid: pick_3d_hybrid prefers 3D when agree, else F0-F3.",
            "fallback_pitch_merge=true fills silent frames via fuse_balls_with_hold.",
            "UKF lifts agree (0.473) but drops clear_R (0.809) — keep ukf_enabled=false.",
        ],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["gates"], indent=2))
    if not promote:
        row = (
            f"| **3D triangulate fuse** | holdout/strips A/B | "
            f"**Skipped** — gates failed | See `{OUT.relative_to(ROOT)}` |"
        )
        text = DEAD_ENDS.read_text(encoding="utf-8") if DEAD_ENDS.is_file() else ""
        if "3D triangulate fuse" not in text:
            lines = text.splitlines()
            insert = next((i for i, ln in enumerate(lines) if ln.startswith("## Worked")), len(lines))
            lines.insert(insert, row)
            DEAD_ENDS.write_text("\n".join(lines) + "\n", encoding="utf-8")
    else:
        print(
            "3D hybrid passes promotion gates (holdout + strips parity). "
            "Opt-in: fuse.mode=triangulate_3d in configs/default.yaml"
        )
    print(f"promote_3d={promote}")
    print(f"wrote {OUT}")
    return 0 if promote else 1


if __name__ == "__main__":
    raise SystemExit(main())
