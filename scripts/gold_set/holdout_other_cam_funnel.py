#!/usr/bin/env python3
"""Holdout clear FNs: count non-focus cams mapped at conf ≥ 0.80 while fuse silent."""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "gold_set"))

from eval_match2_top_left_multicam_baseline import cache_load  # noqa: E402
from fn_audit_match3_quad import CAMS, CLEAR_SIDE, map_reason, product_fuse  # noqa: E402
from fn_audit_match3_random import clear_focus_cam  # noqa: E402
from multicam_select_policy import MATCH3_THR_BY_CAM, filter_active  # noqa: E402
from src.mapping.match3_xy import EMIT_CONF, load_calib  # noqa: E402

CACHE_DIR = ROOT / "reports/eval_match3/pitchmap_gallery_holdout/det_cache"
OUT = ROOT / "reports/eval_match3/improve_eng_loop/holdout_other_cam_funnel.json"


def main() -> int:
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--cache-dir",
        type=Path,
        default=CACHE_DIR,
        help="holdout det_cache folder (v12 or det_cache_v13)",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output JSON (default: holdout_other_cam_funnel.json)",
    )
    args = p.parse_args()
    cache_dir = args.cache_dir
    if not cache_dir.is_absolute():
        cache_dir = ROOT / cache_dir
    out_path = args.out
    if out_path is None:
        out_path = OUT
        if "v13" in str(cache_dir):
            out_path = OUT.with_name("holdout_other_cam_funnel_v13.json")
    elif not out_path.is_absolute():
        out_path = ROOT / out_path
    paths = sorted(cache_dir.glob("det_cache_rand_*_thr010.json"))
    if not paths:
        raise SystemExit(f"no caches in {cache_dir}")
    calibs = {c: v for c, v in ((c, load_calib(c)) for c in CAMS) if v}
    tot_clear_fn = 0
    other_ge080 = 0
    by_cam = defaultdict(int)
    samples = []
    for path in paths:
        dets = cache_load(path)
        cams = [c for c in CAMS if c in dets]
        n = len(next(iter(dets.values())))
        prev = None
        gap = 0
        for i in range(n):
            active = filter_active(dets, i, cams, MATCH3_THR_BY_CAM)
            focus = clear_focus_cam(active)
            fused, prev, gap = product_fuse(dets, i, calibs, cams, prev, gap)
            if focus is None or fused is not None:
                continue
            # clear FN
            tot_clear_fn += 1
            strong = []
            for cam, rows in active.items():
                if cam == focus:
                    continue
                rec = calibs.get(cam)
                if rec is None:
                    continue
                box, conf, _ = rows[0]
                if float(conf) < EMIT_CONF:
                    continue
                hit, _ = map_reason(rec, box, float(conf))
                if hit is None:
                    continue
                strong.append({"cam": cam, "conf": float(conf), "xy": hit["xy"]})
                by_cam[cam] += 1
            if strong:
                other_ge080 += 1
                if len(samples) < 20:
                    samples.append(
                        {
                            "cache": path.name,
                            "i": i,
                            "focus": focus,
                            "other": strong,
                        }
                    )
    frac = None if tot_clear_fn == 0 else round(other_ge080 / tot_clear_fn, 3)
    out = {
        "note": (
            "Clear FNs where ≥1 non-focus cam maps at conf≥0.80. "
            "High frac → fuse/wiring issue; low → real map/det residual."
        ),
        "cache_dir": str(cache_dir.relative_to(ROOT))
        if cache_dir.is_relative_to(ROOT)
        else str(cache_dir),
        "n_clear_fn": tot_clear_fn,
        "n_with_other_ge080_mapped": other_ge080,
        "frac": frac,
        "by_other_cam": dict(sorted(by_cam.items())),
        "samples": samples,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(
        f"clear_fn={tot_clear_fn} other_ge080={other_ge080} frac={frac} "
        f"by_cam={dict(by_cam)}"
    )
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
