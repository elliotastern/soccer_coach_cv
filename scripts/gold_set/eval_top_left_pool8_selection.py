#!/usr/bin/env python3
"""Score Top Left 8-cam pool selection vs dual P7+P10 gold (cache only).

Locks software #1: Cam4+/Cam5+/P-cams, P7 thr floor, largest_ball pick.
Writes reports/eval_match2_v10/top_left_pool8_selection/.
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
    cache_load,
    load_top_left_gt,
    merge_proxy_scores,
    score_proxy,
)
from eval_match2_v10_video_system import pick_selected  # noqa: E402
from multicam_select_policy import (  # noqa: E402
    GOAL_P,
    GOAL_R,
    P_CAMS,
    SURVEY_CAMS,
    TOP_LEFT_PCAM_ONLY_POLICY_ID,
    TOP_LEFT_POLICY_ID,
    TOP_LEFT_THR_BY_CAM,
    filter_active,
    locked_top_left_spec,
)

OUT = ROOT / "reports/eval_match2_v10/top_left_pool8_selection"
CACHE = (
    ROOT / "reports/eval_match2_v10/4quad_multicam_survey/det_cache_top_left_thr010.json"
)
GOLD_P10 = ROOT / "data/processed/gold_sets/match2_4quad_top_left/gold/annotations.xml"
GOLD_P7 = ROOT / "data/processed/gold_sets/match2_4quad_top_left_p7/gold/annotations.xml"
GOLD_CAM4 = (
    ROOT
    / "data/processed/gold_sets/match2_4quad_top_left_cam4plus/gold/annotations.xml"
)


def load_gt_by_cam():
    """Load validated Top Left gold. Whole gold XML is GT (not only source=manual)."""
    gt = {
        "P10": load_top_left_gt(GOLD_P10),
        "P7": load_top_left_gt(GOLD_P7),
    }
    if GOLD_CAM4.is_file():
        gt["Cam4plus"] = load_top_left_gt(GOLD_CAM4)
    return gt


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", type=Path, default=CACHE)
    p.add_argument("--out", type=Path, default=OUT)
    return p.parse_args()


def score_run(dets, gt_by_cam, cams, thr_by_cam, pick_mode):
    n_frames = min(len(next(iter(dets.values()))), 299)
    by_frames = {cam: [] for cam in gt_by_cam}
    by_preds = {cam: [] for cam in gt_by_cam}
    select_counts = {}
    n_selected = 0
    for i in range(n_frames):
        active = filter_active(dets, i, cams, thr_by_cam)
        if not active:
            select_counts["none"] = select_counts.get("none", 0) + 1
            continue
        cam, pred = pick_selected(active, pick_mode)
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
    parts = [proxy[c] for c in gt_by_cam]
    combined = merge_proxy_scores(*parts)
    unscored = n_selected - combined["n_frames_scored"]
    return {
        "n_frames": n_frames,
        "n_selected": n_selected,
        "n_gold_covered": combined["n_frames_scored"],
        "n_unscored_selected": unscored,
        "selection_share": {
            k: v / n_frames for k, v in sorted(select_counts.items())
        },
        "selection_counts": select_counts,
        "proxy": proxy,
        "system": combined,
        "hit_goal": combined["hit_goal"],
    }


def variant_specs():
    thr = dict(TOP_LEFT_THR_BY_CAM)
    thr030 = {"_default": 0.30}
    return [
        (
            "pcam_max_conf_030",
            P_CAMS,
            thr030,
            "max_conf",
            "old baseline: 6 P-cams max_conf @0.30",
        ),
        (
            TOP_LEFT_PCAM_ONLY_POLICY_ID,
            P_CAMS,
            thr,
            "max_conf",
            "prior lock: 6 P-cams, P7≥0.60, max_conf",
        ),
        (
            "pool8_max_conf_p7_thr060",
            SURVEY_CAMS,
            thr,
            "max_conf",
            "8-cam pool, P7≥0.60, max_conf (not size)",
        ),
        (
            TOP_LEFT_POLICY_ID,
            SURVEY_CAMS,
            thr,
            "largest_ball",
            "LOCKED: 8-cam pool, P7≥0.60, largest_ball",
        ),
        (
            "pool8_largest_ball_030",
            SURVEY_CAMS,
            thr030,
            "largest_ball",
            "8-cam largest_ball @0.30 all cams",
        ),
    ]


def fmt_share(share: dict) -> str:
    parts = [f"{k} {v*100:.0f}%" for k, v in share.items() if k != "none"]
    return ", ".join(parts[:6]) if parts else "—"


def main() -> int:
    args = parse_args()
    if not args.cache.is_file():
        raise FileNotFoundError(args.cache)
    dets = cache_load(args.cache)
    gt_by_cam = load_gt_by_cam()
    gold_ids = sorted(gt_by_cam.keys())
    print(f"gold cams scored: {gold_ids}", flush=True)
    rows = []
    for vid, cams, thr_map, mode, why in variant_specs():
        scored = score_run(dets, gt_by_cam, cams, thr_map, mode)
        s = scored["system"]
        proxy = scored["proxy"]
        row = {
            "id": vid,
            "why": why,
            "precision": s["precision"],
            "recall": s["recall"],
            "f1": s["f1"],
            "n_covered": s["n_frames_scored"],
            "n_selected": scored["n_selected"],
            "n_unscored": scored["n_unscored_selected"],
            "hit_goal": s["hit_goal"],
            "selection_share": scored["selection_share"],
            "selection_counts": scored["selection_counts"],
            "proxy": {c: {"p": proxy[c]["precision"], "r": proxy[c]["recall"]} for c in proxy},
        }
        rows.append(row)
        print(
            f"{vid}: P={s['precision']:.3f} R={s['recall']:.3f} "
            f"covered={s['n_frames_scored']} unscored={scored['n_unscored_selected']} "
            f"goal={'HIT' if s['hit_goal'] else 'MISS'} | {fmt_share(scored['selection_share'])}",
            flush=True,
        )

    locked = next(r for r in rows if r["id"] == TOP_LEFT_POLICY_ID)
    args.out.mkdir(parents=True, exist_ok=True)
    payload = {
        "locked": locked_top_left_spec(),
        "goal_r": GOAL_R,
        "goal_p": GOAL_P,
        "gold_cams": gold_ids,
        "cam4_gold": "Cam4plus" in gt_by_cam,
        "ranking": rows,
        "locked_result": locked,
    }
    (args.out / "ranking.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Top Left — 8-cam pool + largest_ball lock",
        "",
        f"Goal **R≥{GOAL_R} P≥{GOAL_P}** on frames where selected cam has gold.",
        f"Gold cams: `{', '.join(gold_ids)}`. Locked: `{TOP_LEFT_POLICY_ID}`.",
        "",
        "| id | P | R | covered | unscored | goal | who wins |",
        "|---|---:|---:|---:|---:|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| `{r['id']}` | {r['precision']:.3f} | {r['recall']:.3f} | "
            f"{r['n_covered']} | {r['n_unscored']} | "
            f"{'HIT' if r['hit_goal'] else 'MISS'} | {fmt_share(r['selection_share'])} |"
        )
    pcam = next(r for r in rows if r["id"] == TOP_LEFT_PCAM_ONLY_POLICY_ID)
    pool8_mc = next(r for r in rows if r["id"] == "pool8_max_conf_p7_thr060")
    cam4_note = (
        "Cam4plus gold included in score."
        if "Cam4plus" in gt_by_cam
        else "Cam4plus gold missing."
    )
    lines += [
        "",
        "## Locked result",
        "",
        f"- Gold-covered under lock: **{locked['n_covered']}** frames "
        f"(P={locked['precision']:.3f} R={locked['recall']:.3f}).",
        f"- Unscored selected: **{locked['n_unscored']}** / 299. {cam4_note}",
        f"- Selection: {fmt_share(locked['selection_share'])}",
        "",
        "## Vs goal (honest)",
        "",
        f"| Slice | P | R | vs R≥{GOAL_R} P≥{GOAL_P} |",
        "|---|---:|---:|---|",
        (
            f"| P-cam-only prior lock | {pcam['precision']:.3f} | "
            f"{pcam['recall']:.3f} | HIT ({pcam['n_covered']} fr) |"
        ),
        (
            f"| 8-cam max_conf | {pool8_mc['precision']:.3f} | "
            f"{pool8_mc['recall']:.3f} | covered {pool8_mc['n_covered']}; "
            f"unscored {pool8_mc['n_unscored']} |"
        ),
        (
            f"| LOCKED largest_ball 8-cam | {locked['precision']:.3f} | "
            f"{locked['recall']:.3f} | covered {locked['n_covered']}; "
            f"unscored {locked['n_unscored']} |"
        ),
        "",
        f"**Closeness:** see chat — pack at `/4quad-cvat/top_left_cam4plus`.",
    ]
    (args.out / "ranking.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.out / 'ranking.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
