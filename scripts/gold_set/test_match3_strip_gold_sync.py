#!/usr/bin/env python3
"""Fail if strip gold_xy drifts from current map_ball_box(gt_balls).

Catches calib/H updates that invalidate H-seeded pitch gold (P8 Sep-2021 case).
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.mapping.match3_xy import load_calib, map_ball_box  # noqa: E402

WH = (1920, 1080)
# Median remap error above this ⇒ gold stale vs current H.
MAX_MED_M = 1.0
STRIPS = [
    ROOT / "data/processed/gold_sets/match3_quad_p8_87/labels.json",
    ROOT / "data/processed/gold_sets/match3_quad_p10_31/labels.json",
]


def strip_gold_sync(path: Path) -> dict:
    lab = json.loads(path.read_text(encoding="utf-8"))
    focus = lab.get("focus_cam") or "P10"
    rec = load_calib(focus)
    if rec is None:
        raise AssertionError(f"missing calib {focus}")
    errs = []
    for fr in lab.get("frames") or []:
        seed = (fr.get("cams") or {}).get(focus) or {}
        gold = seed.get("gold_xy")
        gt = (seed.get("gt_balls") or [None])[0]
        if gold is None or gt is None:
            continue
        hit = map_ball_box(
            rec, [gt["x"], gt["y"], gt["w"], gt["h"]], 0.99, frame_wh=WH
        )
        if hit is None:
            continue
        errs.append(
            math.hypot(hit["xy"][0] - gold[0], hit["xy"][1] - gold[1])
        )
    if not errs:
        raise AssertionError(f"{path.parent.name}: no gold/gt pairs")
    med = sorted(errs)[len(errs) // 2]
    return {
        "pack": path.parent.name,
        "focus": focus,
        "n": len(errs),
        "med_m": round(med, 3),
        "max_m": round(max(errs), 3),
        "ok": med <= MAX_MED_M,
    }


def main() -> int:
    rows = []
    for path in STRIPS:
        if not path.is_file():
            print(f"skip missing {path}")
            continue
        rows.append(strip_gold_sync(path))
    for row in rows:
        print(row)
    bad = [r for r in rows if not r["ok"]]
    if bad:
        raise SystemExit(
            f"stale strip gold vs current H (med>{MAX_MED_M}m): {bad}"
        )
    print("ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
