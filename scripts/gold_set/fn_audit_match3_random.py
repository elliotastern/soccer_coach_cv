#!/usr/bin/env python3
"""R1: FN audit on Match 3 random-gallery proxy-clear frames (diagnosis only)."""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "gold_set"))

from eval_match2_top_left_multicam_baseline import cache_load  # noqa: E402
from fn_audit_match3_quad import (  # noqa: E402
    CAMS,
    CLEAR_SIDE,
    best_raw,
    classify_fn,
    product_fuse,
)
from multicam_select_policy import MATCH3_THR_BY_CAM, filter_active  # noqa: E402
from src.mapping.match3_xy import load_calib  # noqa: E402

OUT = ROOT / "reports/eval_match3/improve_eng_loop/r1_random_fn_audit.json"
RANDOM_CACHE = ROOT / "reports/eval_match3/pitchmap_gallery_v12_hard/det_cache"
BASELINE_PRE_F0_R = 0.525


def clear_focus_cam(active: dict) -> str | None:
    """Cam that defines proxy-clear: largest side among side≥25 and conf≥0.30."""
    best = None
    best_side = -1.0
    for cam, rows in active.items():
        conf = float(rows[0][1])
        side = float(rows[0][2])
        if side < CLEAR_SIDE or conf < 0.30:
            continue
        if side > best_side:
            best_side = side
            best = cam
    return best


def audit_random_cache(path: Path) -> dict:
    dets = cache_load(path)
    cams = [c for c in CAMS if c in dets]
    calibs = {c: v for c, v in ((c, load_calib(c)) for c in cams) if v}
    n = len(next(iter(dets.values())))
    buckets = {}
    by_cam_reason = defaultdict(int)
    conf_bins = {"ge080": 0, "050_079": 0, "020_049": 0, "lt020": 0, "none": 0}
    clear = clear_emit = 0
    samples = []
    prev = None
    gap = 0
    for i in range(n):
        active = filter_active(dets, i, cams, MATCH3_THR_BY_CAM)
        focus = clear_focus_cam(active)
        fused, prev, gap = product_fuse(dets, i, calibs, cams, prev, gap)
        if focus is None:
            continue
        clear += 1
        if fused is not None:
            clear_emit += 1
            continue
        tag, detail = classify_fn(dets, i, calibs, cams, focus)
        buckets[tag] = buckets.get(tag, 0) + 1
        reason = None
        if tag == "focus_map_fail":
            reason = (detail or {}).get("reason")
        key = f"{tag}|{focus}|{reason or '-'}"
        by_cam_reason[key] += 1
        br = best_raw(dets, i, focus)
        if br is None:
            conf_bins["none"] += 1
        elif br["conf"] >= 0.80:
            conf_bins["ge080"] += 1
        elif br["conf"] >= 0.50:
            conf_bins["050_079"] += 1
        elif br["conf"] >= 0.20:
            conf_bins["020_049"] += 1
        else:
            conf_bins["lt020"] += 1
        if len(samples) < 16:
            samples.append(
                {
                    "i": i,
                    "focus": focus,
                    "tag": tag,
                    "detail": detail,
                    "focus_raw": br,
                }
            )
    clear_r = None if clear == 0 else round(clear_emit / clear, 3)
    return {
        "cache": path.name,
        "n": n,
        "n_clear_proxy": clear,
        "n_clear_emit": clear_emit,
        "clear_ball_proxy_R": clear_r,
        "fn_buckets": buckets,
        "fn_by_tag_cam_reason": dict(sorted(by_cam_reason.items())),
        "focus_conf_on_fn": conf_bins,
        "sample_fn": samples,
    }


def classify_systemic(per_cache: list[dict]) -> dict:
    """Systemic = same tag|cam|reason on ≥2 caches; else clip-specific."""
    seed_hits = defaultdict(set)
    for row in per_cache:
        cache = row["cache"]
        for key, n in (row.get("fn_by_tag_cam_reason") or {}).items():
            if n > 0:
                seed_hits[key].add(cache)
    systemic = []
    clip_specific = []
    for key, caches in sorted(seed_hits.items(), key=lambda x: (-len(x[1]), x[0])):
        tag, cam, reason = key.split("|", 2)
        row = {
            "key": key,
            "tag": tag,
            "cam": cam,
            "reason": None if reason == "-" else reason,
            "n_caches": len(caches),
            "caches": sorted(caches),
        }
        if len(caches) >= 2:
            systemic.append(row)
        else:
            clip_specific.append(row)
    return {"systemic": systemic, "clip_specific": clip_specific}


def main() -> int:
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--cache-dir",
        type=Path,
        default=RANDOM_CACHE,
        help="det_cache folder (tune or holdout)",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output JSON path",
    )
    p.add_argument(
        "--glob",
        default="det_cache_rand_*_thr010.json",
        help="cache glob under cache-dir",
    )
    args = p.parse_args()
    cache_dir = args.cache_dir
    if not cache_dir.is_absolute():
        cache_dir = ROOT / cache_dir
    out_path = args.out
    if out_path is None:
        tag = "holdout" if "holdout" in str(cache_dir) else "tune"
        out_path = ROOT / f"reports/eval_match3/improve_eng_loop/r1_random_fn_audit_{tag}.json"
    elif not out_path.is_absolute():
        out_path = ROOT / out_path
    if not cache_dir.is_dir():
        raise SystemExit(f"missing {cache_dir}")
    paths = sorted(cache_dir.glob(args.glob))
    if not paths:
        raise SystemExit(f"no caches matching {args.glob} in {cache_dir}")
    per_cache = [audit_random_cache(p) for p in paths]
    tot_clear = sum(r["n_clear_proxy"] for r in per_cache)
    tot_emit = sum(r["n_clear_emit"] for r in per_cache)
    tot_buckets = defaultdict(int)
    for r in per_cache:
        for k, v in (r.get("fn_buckets") or {}).items():
            tot_buckets[k] += v
    classed = classify_systemic(per_cache)
    out = {
        "note": (
            "FN diagnosis only — no hull/checkpoint changes. "
            "Proxy clear = any cam side≥25 and conf≥0.30. "
            "Fuse = F1+F2+F0+F3 product path."
        ),
        "baseline_pre_f0_proxy_R": BASELINE_PRE_F0_R,
        "cache_dir": str(cache_dir.relative_to(ROOT)),
        "totals": {
            "n_clear_proxy": tot_clear,
            "n_clear_emit": tot_emit,
            "clear_ball_proxy_R": None
            if tot_clear == 0
            else round(tot_emit / tot_clear, 3),
            "fn_buckets": dict(sorted(tot_buckets.items())),
        },
        "systemic_vs_clip": classed,
        "per_cache": per_cache,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(
        f"proxy_R={out['totals']['clear_ball_proxy_R']} "
        f"clear={tot_emit}/{tot_clear} fn={dict(tot_buckets)}"
    )
    print(f"systemic keys={len(classed['systemic'])} clip_specific={len(classed['clip_specific'])}")
    for row in classed["systemic"][:8]:
        print(f"  systemic {row['key']} caches={row['n_caches']}")
    for row in classed["clip_specific"][:8]:
        print(f"  clip     {row['key']} → {row['caches'][0]}")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
