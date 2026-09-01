#!/usr/bin/env python3
"""A/B product fuse F3 vs F3+F4 (reprojection prune) on M1 strips + holdout proxy."""
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
    QUAD_CAMS,
    WH,
    map_active,
    score_cache,
)
from ab_match3_fuse_post import STRIPS, score_variant  # noqa: E402
from eval_match2_top_left_multicam_baseline import cache_load  # noqa: E402
from score_match3_ball_m1 import HIT_M, infer_cache_stride  # noqa: E402
from src.mapping.match3_xy import (  # noqa: E402
    GHOST_CONF,
    HOLD_MAX_GAP,
    fuse_balls,
    fuse_balls_with_hold,
    load_calib,
)

OUT = ROOT / "reports/eval_match3/improve_eng_loop/f4_reproj_ab.json"
DEAD_ENDS = ROOT / "reports/ball_testing/DEAD_ENDS.md"
HOLDOUT_DIR = ROOT / "reports/eval_match3/pitchmap_gallery_holdout"
HOLDOUT_GATE = 0.884
BASE = "F1+F2+F0+F3"
F4 = "F1+F2+F0+F3+F4"
PRODUCT_KW = dict(
    soft_dual_fallback=True,
    solo_max_conf=True,
    ghost_prune=True,
    ghost_conf=GHOST_CONF,
    reproj_prune=False,
)
PRODUCT_F4_KW = {**PRODUCT_KW, "reproj_prune": True}


def score_strip_pair(path: Path) -> dict:
    labels = json.loads(path.read_text(encoding="utf-8"))
    dets = cache_load(ROOT / labels["det_cache"])
    cams = [c for c in CAMS if c in dets]
    calibs = {c: v for c, v in ((c, load_calib(c)) for c in cams) if v}
    focus = labels.get("focus_cam") or "P10"
    stride = infer_cache_stride(dets)
    f1, f2, hold, ghost = True, True, True, True
    base = score_variant(labels, dets, calibs, focus, stride, f1, f2, hold, ghost, False)
    f4 = score_variant(labels, dets, calibs, focus, stride, f1, f2, hold, ghost, True)
    return {
        "pack": labels.get("pack") or path.parent.name,
        "path": str(path.relative_to(ROOT)),
        BASE: base,
        F4: f4,
    }


def score_holdout_cache(path: Path, reproj: bool) -> dict:
    dets = cache_load(path)
    cams = [c for c in CAMS if c in dets]
    n = len(next(iter(dets.values())))
    calibs = {c: v for c, v in ((c, load_calib(c)) for c in cams) if v}
    kw = PRODUCT_F4_KW if reproj else PRODUCT_KW
    clear = clear_emit = emit = 0
    prev = None
    gap = 0
    for i in range(n):
        active, mapped = map_active(dets, i, cams, calibs, True)
        is_clear = any(
            float(rows[0][2]) >= CLEAR_SIDE and float(rows[0][1]) >= 0.30
            for rows in active.values()
        )
        fresh = fuse_balls(mapped, **kw)
        if fresh is not None:
            fused = fresh
            prev = fresh
            gap = 0
        else:
            gap += 1
            fused = fuse_balls_with_hold(prev, [], gap, hold_max_gap=HOLD_MAX_GAP, **kw)
        if fused is not None:
            emit += 1
        if is_clear:
            clear += 1
            if fused is not None:
                clear_emit += 1
    return {
        "cache": path.name,
        "clear_frames": clear,
        "clear_emit": clear_emit,
        "emit": emit,
        "clear_ball_proxy_R": None if clear == 0 else round(clear_emit / clear, 3),
    }


def score_holdout(reproj: bool) -> dict:
    det_dir = HOLDOUT_DIR / "det_cache"
    rows = []
    for path in sorted(det_dir.glob("det_cache_*.json")):
        rows.append(score_holdout_cache(path, reproj))
    clear = sum(int(r["clear_frames"]) for r in rows)
    clear_emit = sum(int(r["clear_emit"]) for r in rows)
    proxy_r = None if clear == 0 else round(clear_emit / clear, 3)
    return {"caches": rows, "clear_ball_proxy_R": proxy_r}


def strip_passes(base: dict, f4: dict) -> bool:
    if not base.get("poc_pass_P_emit") or not f4.get("poc_pass_P_emit"):
        return False
    if not base.get("poc_pass_clear_R") or not f4.get("poc_pass_clear_R"):
        return False
    b_err = base.get("err_median_m")
    f_err = f4.get("err_median_m")
    if b_err is not None and f_err is not None and f_err > b_err + 0.2:
        return False
    return True


def main() -> int:
    strips = {}
    for path in STRIPS:
        if not path.is_file():
            continue
        strips[path.parent.name] = score_strip_pair(path)

    holdout_base = score_holdout(False)
    holdout_f4 = score_holdout(True)
    holdout_ok = (
        float(holdout_f4.get("clear_ball_proxy_R") or 0) >= HOLDOUT_GATE
        and float(holdout_f4.get("clear_ball_proxy_R") or 0)
        >= float(holdout_base.get("clear_ball_proxy_R") or 0) - 0.01
    )
    strips_ok = all(
        strip_passes(block[BASE], block[F4]) for block in strips.values()
    )
    promote = strips_ok and holdout_ok

    report = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "hit_m": HIT_M,
        "holdout_gate": HOLDOUT_GATE,
        "strips": strips,
        "holdout": {BASE: holdout_base, F4: holdout_f4},
        "promote_F4": promote,
        "gates": {
            "strips_pass": strips_ok,
            "holdout_pass": holdout_ok,
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["gates"], indent=2))
    print(f"holdout {BASE}={holdout_base.get('clear_ball_proxy_R')} {F4}={holdout_f4.get('clear_ball_proxy_R')}")
    for name, block in strips.items():
        print(
            name,
            f"P_emit {block[BASE]['P_emit']}->{block[F4]['P_emit']}",
            f"clear_R {block[BASE]['clear_ball_R']}->{block[F4]['clear_ball_R']}",
        )
    if not promote:
        row = (
            f"| **F4 reprojection prune** | holdout/strips A/B | "
            f"**Skipped** — gates failed | See `{OUT.relative_to(ROOT)}` |"
        )
        text = DEAD_ENDS.read_text(encoding="utf-8") if DEAD_ENDS.is_file() else ""
        if "F4 reprojection prune" not in text:
            lines = text.splitlines()
            insert = next((i for i, ln in enumerate(lines) if ln.startswith("## Worked")), len(lines))
            lines.insert(insert, row)
            DEAD_ENDS.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"promote_F4={promote}")
    print(f"wrote {OUT}")
    return 0 if promote else 1


if __name__ == "__main__":
    raise SystemExit(main())
