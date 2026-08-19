#!/usr/bin/env python3
"""Funnel: Match 2 thr vs Match 3 thr on existing Match 3 det caches."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "gold_set"))

from eval_match2_top_left_multicam_baseline import cache_load  # noqa: E402
from multicam_select_policy import (  # noqa: E402
    MATCH3_THR_BY_CAM,
    TOP_LEFT_THR_BY_CAM,
    filter_active,
)
from src.mapping.match3_xy import fuse_balls, load_calib, map_ball_box  # noqa: E402

CACHE_DIR = ROOT / "reports/eval_match3/pitchmap_gallery/det_cache"
OUT = ROOT / "reports/eval_match3/improve_eng_loop/t1_thr_funnel.json"
CAMS = ["P1", "P6", "P7", "P8", "P9", "P10", "P_Goal1", "P_Goal2"]
WH = (1920, 1080)


def score_cache(path: Path, thr_by_cam: dict) -> dict:
    dets = cache_load(path)
    cams = [c for c in CAMS if c in dets]
    n = len(next(iter(dets.values())))
    calibs = {c: load_calib(c) for c in cams}
    calibs = {c: v for c, v in calibs.items() if v is not None}
    active_ge2 = 0
    mapped_ge2 = 0
    agree = 0
    emit = 0
    p7_active = 0
    for i in range(n):
        active = filter_active(dets, i, cams, thr_by_cam)
        if "P7" in active:
            p7_active += 1
        if len(active) >= 2:
            active_ge2 += 1
        mapped = []
        for cam, rows in active.items():
            rec = calibs.get(cam)
            if rec is None:
                continue
            box, conf, _side = rows[0]
            hit = map_ball_box(rec, box, float(conf), frame_wh=WH)
            if hit is None:
                continue
            mapped.append(hit)
        if len(mapped) >= 2:
            mapped_ge2 += 1
        fused = fuse_balls(mapped)
        if fused is None:
            continue
        emit += 1
        if fused.get("agree"):
            agree += 1
    return {
        "n": n,
        "active_ge2": active_ge2,
        "mapped_ge2": mapped_ge2,
        "agree": agree,
        "emit": emit,
        "p7_active": p7_active,
    }


def main() -> int:
    caches = sorted(CACHE_DIR.glob("det_cache_*_thr010.json"))
    if not caches:
        print(f"no caches in {CACHE_DIR}", file=sys.stderr)
        return 1
    totals = {
        "match2": {"n": 0, "active_ge2": 0, "mapped_ge2": 0, "agree": 0, "emit": 0, "p7_active": 0},
        "match3": {"n": 0, "active_ge2": 0, "mapped_ge2": 0, "agree": 0, "emit": 0, "p7_active": 0},
    }
    per = []
    for path in caches:
        row = {"cache": path.name}
        for name, thr in (("match2", TOP_LEFT_THR_BY_CAM), ("match3", MATCH3_THR_BY_CAM)):
            s = score_cache(path, thr)
            row[name] = s
            for k in totals[name]:
                totals[name][k] += s[k]
        per.append(row)
    out = {
        "step": "T1",
        "thr_match2": dict(TOP_LEFT_THR_BY_CAM),
        "thr_match3": dict(MATCH3_THR_BY_CAM),
        "totals": totals,
        "per_cache": per,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(totals, indent=2))
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
