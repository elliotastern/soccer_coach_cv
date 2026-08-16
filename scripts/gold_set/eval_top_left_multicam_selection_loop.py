#!/usr/bin/env python3
"""Loop selection fixes on Top Left multicam cache (no re-detect, no labeling).

Tries cheap pick/thr variants vs dual P7+P10 gold. Goal R≥0.8 P≥0.9 on covered
frames. Writes reports/eval_match2_v10/top_left_multicam_selection_loop/.
Never trains.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "gold_set"))

from eval_match2_top_left_multicam_baseline import (  # noqa: E402
    GOAL_P,
    GOAL_R,
    GOLD_CAMS,
    N_FRAMES,
    P_CAMS,
    cache_load,
    filter_rows,
    load_top_left_gt,
    merge_proxy_scores,
    score_proxy,
)
from eval_match2_v10_video_system import pick_selected  # noqa: E402

OUT = ROOT / "reports/eval_match2_v10/top_left_multicam_selection_loop"
CACHE = ROOT / "reports/eval_match2_v10/top_left_multicam_baseline/det_cache_thr010.json"
GOLD_P10 = ROOT / "data/processed/gold_sets/match2_4quad_top_left/gold/annotations.xml"
GOLD_P7 = ROOT / "data/processed/gold_sets/match2_4quad_top_left_p7/gold/annotations.xml"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", type=Path, default=CACHE)
    p.add_argument("--out", type=Path, default=OUT)
    return p.parse_args()


def pick_max_conf(active: dict):
    return pick_selected(active, "max_conf")


def pick_size_weighted(active: dict):
    return pick_selected(active, "size_weighted")


def pick_prefer_p10(active: dict):
    if "P10" in active and active["P10"]:
        return "P10", active["P10"][0]
    return pick_max_conf(active)


def pick_prefer_p10_if_close(active: dict, margin: float = 0.10):
    """P7 wins only if conf beats P10 by margin; else prefer P10 when present."""
    if "P10" in active and active["P10"] and "P7" in active and active["P7"]:
        c10 = active["P10"][0][1]
        c7 = active["P7"][0][1]
        if c7 > c10 + margin:
            return "P7", active["P7"][0]
        return "P10", active["P10"][0]
    if "P10" in active and active["P10"]:
        return "P10", active["P10"][0]
    return pick_max_conf(active)


def pick_gold_cams_only(active: dict):
    gold = {c: r for c, r in active.items() if c in GOLD_CAMS}
    if not gold:
        return None, None
    return pick_max_conf(gold)


def pick_prefer_gold_then_max(active: dict):
    gold = {c: r for c, r in active.items() if c in GOLD_CAMS}
    if gold:
        return pick_max_conf(gold)
    return pick_max_conf(active)


def filter_cam_rows(dets, i: int, thr_by_cam: dict) -> dict:
    out = {}
    for cam in P_CAMS:
        thr = thr_by_cam.get(cam, thr_by_cam.get("_default", 0.30))
        rows = filter_rows(dets[cam][i], thr)
        if rows:
            out[cam] = rows
    return out


def score_variant(dets, gt_by_cam: dict, thr_by_cam: dict, picker, min_cams: int = 1):
    by_frames = {cam: [] for cam in gt_by_cam}
    by_preds = {cam: [] for cam in gt_by_cam}
    select_counts = {}
    n_selected = 0
    for i in range(N_FRAMES):
        cam_rows = filter_cam_rows(dets, i, thr_by_cam)
        if len(cam_rows) < min_cams:
            select_counts["none"] = select_counts.get("none", 0) + 1
            continue
        cam, pred = picker(cam_rows)
        if cam is None or pred is None:
            select_counts["none"] = select_counts.get("none", 0) + 1
            continue
        n_selected += 1
        select_counts[cam] = select_counts.get(cam, 0) + 1
        if cam in gt_by_cam:
            by_frames[cam].append(i)
            by_preds[cam].append([(list(pred[0]), float(pred[1]))])
    proxy = {
        cam: score_proxy(
            gt_by_cam[cam],
            by_frames[cam],
            by_preds[cam],
            note=f"selected={cam}",
        )
        for cam in gt_by_cam
    }
    combined = merge_proxy_scores(proxy["P7"], proxy["P10"])
    return {
        "n_selected": n_selected,
        "selection_share": {k: v / N_FRAMES for k, v in sorted(select_counts.items())},
        "proxy_p7": proxy["P7"],
        "proxy_p10": proxy["P10"],
        "system": combined,
        "hit_goal": combined["hit_goal"],
    }


def variant_specs():
    d = lambda t: {"_default": t}
    specs = [
        ("max_conf_030", d(0.30), pick_max_conf, 1, "baseline max_conf @0.30"),
        ("max_conf_040", d(0.40), pick_max_conf, 1, "max_conf @0.40"),
        ("max_conf_050", d(0.50), pick_max_conf, 1, "max_conf @0.50"),
        ("max_conf_060", d(0.60), pick_max_conf, 1, "max_conf @0.60"),
        ("max_conf_070", d(0.70), pick_max_conf, 1, "max_conf @0.70"),
        (
            "p7_thr050_others030",
            {"_default": 0.30, "P7": 0.50},
            pick_max_conf,
            1,
            "P7 needs ≥0.50; others @0.30",
        ),
        (
            "p7_thr060_others030",
            {"_default": 0.30, "P7": 0.60},
            pick_max_conf,
            1,
            "P7 needs ≥0.60; others @0.30",
        ),
        (
            "p7_thr070_others030",
            {"_default": 0.30, "P7": 0.70},
            pick_max_conf,
            1,
            "P7 needs ≥0.70; others @0.30",
        ),
        ("prefer_p10", d(0.30), pick_prefer_p10, 1, "always prefer P10 if it has a det"),
        (
            "prefer_p10_margin10",
            d(0.30),
            lambda a: pick_prefer_p10_if_close(a, 0.10),
            1,
            "P7 only if conf > P10+0.10",
        ),
        (
            "prefer_p10_margin05",
            d(0.30),
            lambda a: pick_prefer_p10_if_close(a, 0.05),
            1,
            "P7 only if conf > P10+0.05",
        ),
        (
            "gold_cams_only_030",
            d(0.30),
            pick_gold_cams_only,
            1,
            "select only among P7/P10",
        ),
        (
            "prefer_gold_then_max",
            d(0.30),
            pick_prefer_gold_then_max,
            1,
            "prefer P7/P10 pool then max_conf",
        ),
        ("size_weighted_030", d(0.30), pick_size_weighted, 1, "size×conf pick @0.30"),
        (
            "soft_min2_015",
            d(0.15),
            pick_max_conf,
            2,
            "soft ≥2 cams @0.15 then max_conf",
        ),
        (
            "p7_060_prefer_p10_margin10",
            {"_default": 0.30, "P7": 0.60},
            lambda a: pick_prefer_p10_if_close(a, 0.10),
            1,
            "P7≥0.60 + P10 margin 0.10",
        ),
        (
            "p7_050_prefer_p10",
            {"_default": 0.30, "P7": 0.50},
            pick_prefer_p10,
            1,
            "P7≥0.50 + always prefer P10 if present",
        ),
    ]
    return specs


def main() -> int:
    args = parse_args()
    if not args.cache.is_file():
        raise FileNotFoundError(args.cache)
    dets = cache_load(args.cache)
    gt_by_cam = {
        "P10": load_top_left_gt(GOLD_P10),
        "P7": load_top_left_gt(GOLD_P7),
    }
    rows = []
    for vid, thr_map, picker, min_cams, why in variant_specs():
        scored = score_variant(dets, gt_by_cam, thr_map, picker, min_cams)
        s = scored["system"]
        rows.append(
            {
                "id": vid,
                "why": why,
                "precision": s["precision"],
                "recall": s["recall"],
                "f1": s["f1"],
                "tp": s["tp"],
                "fp": s["fp"],
                "fn": s["fn"],
                "n_covered": s["n_frames_scored"],
                "hit_goal": s["hit_goal"],
                "hit_r": s["hit_goal_r"],
                "hit_p": s["hit_goal_p"],
                "p7_p": scored["proxy_p7"]["precision"],
                "p7_r": scored["proxy_p7"]["recall"],
                "p10_p": scored["proxy_p10"]["precision"],
                "p10_r": scored["proxy_p10"]["recall"],
                "n_selected": scored["n_selected"],
                "selection_share": scored["selection_share"],
            }
        )
        print(
            f"{vid}: P={s['precision']:.3f} R={s['recall']:.3f} "
            f"goal={'HIT' if s['hit_goal'] else 'MISS'} "
            f"P7={scored['proxy_p7']['precision']:.3f}/{scored['proxy_p7']['recall']:.3f}",
            flush=True,
        )

    rows.sort(key=lambda r: (-r["hit_goal"], -r["f1"], -r["precision"], -r["recall"]))
    best = rows[0]
    args.out.mkdir(parents=True, exist_ok=True)
    payload = {
        "goal_r": GOAL_R,
        "goal_p": GOAL_P,
        "n_variants": len(rows),
        "best_id": best["id"],
        "best_hit_goal": best["hit_goal"],
        "ranking": rows,
    }
    (args.out / "ranking.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Top Left multicam — selection fix loop (dual gold)",
        "",
        f"Cache re-score only (no re-detect). Goal **R≥{GOAL_R} P≥{GOAL_P}** on P7∪P10-selected.",
        "",
        f"**Best:** `{best['id']}` → P={best['precision']:.3f} R={best['recall']:.3f} "
        f"**{'HIT' if best['hit_goal'] else 'MISS'}**",
        "",
        "| Rank | id | P | R | F1 | P7 P/R | P10 P/R | covered | goal |",
        "|---:|---|---:|---:|---:|---|---|---:|---|",
    ]
    for i, r in enumerate(rows, 1):
        lines.append(
            f"| {i} | `{r['id']}` | {r['precision']:.3f} | {r['recall']:.3f} | "
            f"{r['f1']:.3f} | {r['p7_p']:.3f}/{r['p7_r']:.3f} | "
            f"{r['p10_p']:.3f}/{r['p10_r']:.3f} | {r['n_covered']} | "
            f"{'HIT' if r['hit_goal'] else 'MISS'} |"
        )
    lines += [
        "",
        "## Read",
        "",
        (
            "HIT → lock that pick rule for live path, then 5090 latency."
            if best["hit_goal"]
            else "No thr/pick tweak hits 80/90 on this clip. Next: P7 detect quality "
            "(not more selection knobs) or accept emit-gate tradeoff / more cams."
        ),
        "",
    ]
    md = args.out / "ranking.md"
    md.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {md}", flush=True)
    print(
        f"BEST {best['id']} P={best['precision']:.3f} R={best['recall']:.3f} "
        f"{'HIT' if best['hit_goal'] else 'MISS'}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
