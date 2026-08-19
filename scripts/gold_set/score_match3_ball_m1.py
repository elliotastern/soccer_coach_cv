#!/usr/bin/env python3
"""M1 provisional Match 3 ball metrics (until labeled gold strip exists).

Reports:
  - emit / agree coverage on random + quad galleries
  - clear-ball proxy: frames with ≥1 cam det side≥25px & conf≥0.30 → emit rate
  - note: true P_emit needs human labels (not measured here)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "gold_set"))

from eval_match2_top_left_multicam_baseline import cache_load  # noqa: E402
from multicam_select_policy import MATCH3_THR_BY_CAM, filter_active  # noqa: E402
from src.mapping.match3_xy import (  # noqa: E402
    EMIT_CONF,
    MIN_SUPPORT,
    fuse_balls,
    load_calib,
    map_ball_box,
)

OUT = ROOT / "reports/eval_match3/improve_eng_loop/m1_provisional.json"
CAMS = ["P1", "P6", "P7", "P8", "P9", "P10", "P_Goal1", "P_Goal2"]
WH = (1920, 1080)
CLEAR_SIDE = 25.0


def score_cache(path: Path) -> dict:
    dets = cache_load(path)
    cams = [c for c in CAMS if c in dets]
    n = len(next(iter(dets.values())))
    calibs = {c: v for c, v in ((c, load_calib(c)) for c in cams) if v}
    clear = 0
    clear_emit = 0
    emit = 0
    agree = 0
    mapped_ge2 = 0
    for i in range(n):
        active = filter_active(dets, i, cams, MATCH3_THR_BY_CAM)
        is_clear = False
        for cam, rows in active.items():
            box, conf, side = rows[0]
            if float(side) >= CLEAR_SIDE and float(conf) >= 0.30:
                is_clear = True
                break
        mapped = []
        for cam, rows in active.items():
            rec = calibs.get(cam)
            if rec is None:
                continue
            box, conf, _side = rows[0]
            hit = map_ball_box(rec, box, float(conf), frame_wh=WH)
            if hit is not None:
                mapped.append(hit)
        if len(mapped) >= 2:
            mapped_ge2 += 1
        fused = fuse_balls(mapped)
        did_emit = fused is not None
        if did_emit:
            emit += 1
            if fused.get("agree"):
                agree += 1
        if is_clear:
            clear += 1
            if did_emit:
                clear_emit += 1
    return {
        "cache": path.name,
        "n": n,
        "emit": emit,
        "agree": agree,
        "mapped_ge2": mapped_ge2,
        "clear_frames": clear,
        "clear_emit": clear_emit,
        "clear_ball_proxy_R": None if clear == 0 else round(clear_emit / clear, 3),
        "agree_among_emit": None if emit == 0 else round(agree / emit, 3),
    }


def main() -> int:
    packs = {
        "random": ROOT / "reports/eval_match3/pitchmap_gallery/det_cache",
        "quad": ROOT / "reports/eval_match3/quad_pitchmap_gallery/det_cache",
    }
    out = {
        "goals": {"P_emit": 0.80, "clear_ball_R": 0.80, "emit_conf": EMIT_CONF, "min_support": MIN_SUPPORT},
        "note": "Proxy only — true P_emit needs labeled Match 3 gold. clear_ball_proxy_R = emit | clear-size det.",
        "packs": {},
    }
    for name, folder in packs.items():
        if not folder.is_dir():
            continue
        rows = [score_cache(p) for p in sorted(folder.glob("det_cache_*_thr010.json"))]
        tot = {k: 0 for k in ["n", "emit", "agree", "mapped_ge2", "clear_frames", "clear_emit"]}
        for r in rows:
            for k in tot:
                tot[k] += r[k]
        out["packs"][name] = {
            "totals": tot,
            "clear_ball_proxy_R": None
            if tot["clear_frames"] == 0
            else round(tot["clear_emit"] / tot["clear_frames"], 3),
            "agree_among_emit": None if tot["emit"] == 0 else round(tot["agree"] / tot["emit"], 3),
            "per_cache": rows,
        }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({k: v.get("totals") if isinstance(v, dict) else v for k, v in out["packs"].items()}, indent=2))
    for name, pack in out["packs"].items():
        print(
            f"{name}: clear_ball_proxy_R={pack['clear_ball_proxy_R']} "
            f"agree_among_emit={pack['agree_among_emit']} "
            f"emit={pack['totals']['emit']}/{pack['totals']['n']}"
        )
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
