#!/usr/bin/env python3
"""Apply locked 8-cam largest_ball policy across all 4quad regions (cache only).

No labeling / no re-detect. Flags which slots look OK vs need gold later.
Writes reports/eval_match2_v10/4quad_locked_policy_survey/.
Never trains.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "gold_set"))

from eval_match2_top_left_multicam_baseline import cache_load  # noqa: E402
from eval_match2_v10_video_system import pick_selected  # noqa: E402
from multicam_select_policy import (  # noqa: E402
    GOAL_P,
    GOAL_R,
    QUAD_SLOTS,
    SURVEY_CAMS,
    TOP_LEFT_PICK_MODE,
    TOP_LEFT_POLICY_ID,
    TOP_LEFT_THR_BY_CAM,
    filter_active,
    locked_top_left_spec,
)

CACHE_DIR = ROOT / "reports/eval_match2_v10/4quad_multicam_survey"
OUT = ROOT / "reports/eval_match2_v10/4quad_locked_policy_survey"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    p.add_argument("--out", type=Path, default=OUT)
    return p.parse_args()


def score_slot(dets: dict, n_frames: int, thr_by_cam: dict, pick_mode: str) -> dict:
    counts = Counter()
    sides = []
    confs = []
    side_by_cam = {c: [] for c in SURVEY_CAMS}
    for i in range(n_frames):
        active = filter_active(dets, i, SURVEY_CAMS, thr_by_cam)
        if not active:
            counts["none"] += 1
            continue
        cam, pred = pick_selected(active, pick_mode)
        if cam is None or pred is None:
            counts["none"] += 1
            continue
        counts[cam] += 1
        side = float(pred[2])
        conf = float(pred[1])
        sides.append(side)
        confs.append(conf)
        side_by_cam[cam].append(side)
    n_sel = sum(v for k, v in counts.items() if k != "none")
    top = sorted(
        ((c, v) for c, v in counts.items() if c != "none"),
        key=lambda x: -x[1],
    )
    return {
        "n_frames": n_frames,
        "n_selected": n_sel,
        "n_none": counts.get("none", 0),
        "selection_share": {k: v / n_frames for k, v in sorted(counts.items())},
        "selection_counts": dict(counts),
        "top_winners": [{"cam": c, "share": v / n_frames, "n": v} for c, v in top[:4]],
        "selected_side_px": _stats(sides),
        "selected_conf": _stats(confs),
        "side_px_by_cam": {c: _stats(v) for c, v in side_by_cam.items() if v},
        "pick_mode": pick_mode,
        "thr_by_cam": thr_by_cam,
    }


def _stats(vals: list) -> dict | None:
    if not vals:
        return None
    vals = sorted(vals)
    return {
        "n": len(vals),
        "mean": round(statistics.mean(vals), 2),
        "median": round(statistics.median(vals), 2),
        "p10": round(vals[max(0, int(0.1 * (len(vals) - 1)))], 2),
        "p90": round(vals[min(len(vals) - 1, int(0.9 * (len(vals) - 1)))], 2),
        "frac_ge_20px": round(sum(1 for v in vals if v >= 20.0) / len(vals), 3),
    }


def flag_slot(row: dict) -> dict:
    """Heuristic: need_label if winner is unknown-gold cam with weak size, etc."""
    locked = row["locked"]
    top = locked["top_winners"]
    winner = top[0]["cam"] if top else "none"
    side = locked["selected_side_px"] or {}
    med = side.get("median") or 0.0
    ge20 = side.get("frac_ge_20px") or 0.0
    none_share = locked["selection_share"].get("none", 0.0)
    # Top Left already has Cam4/P7/P10 gold
    gold_ok = row["slot"] == "top_left"
    need = False
    reasons = []
    if none_share > 0.05:
        need = True
        reasons.append(f"none={none_share:.0%}")
    if med < 15.0:
        need = True
        reasons.append(f"median_side={med:.1f}px")
    if ge20 < 0.5:
        need = True
        reasons.append(f"frac≥20px={ge20:.0%}")
    if winner in ("Cam4plus", "Cam5plus") and not gold_ok:
        reasons.append(f"winner={winner} unscored (no gold this slot)")
        # not automatic need_label — wait for size flags
    if winner.startswith("P") and not gold_ok and med < 18:
        need = True
        reasons.append(f"P-cam winner {winner} small ball")
    return {
        "winner": winner,
        "need_label": need,
        "reasons": reasons,
        "priority": "high" if need else ("watch" if reasons else "ok"),
    }


def fmt_top(top: list) -> str:
    return ", ".join(f"{t['cam']} {t['share']*100:.0f}%" for t in top[:3]) or "—"


def main() -> int:
    args = parse_args()
    lock = locked_top_left_spec()
    thr = dict(TOP_LEFT_THR_BY_CAM)
    rows = []
    for spec in QUAD_SLOTS:
        cache = args.cache_dir / f"det_cache_{spec['slot']}_thr010.json"
        if not cache.is_file():
            raise FileNotFoundError(cache)
        dets = cache_load(cache)
        n = min(spec["n_frames"], len(next(iter(dets.values()))))
        locked = score_slot(dets, n, thr, TOP_LEFT_PICK_MODE)
        baseline = score_slot(dets, n, {"_default": 0.30}, "max_conf")
        row = {
            "slot": spec["slot"],
            "label": spec["label"],
            "stem": spec["stem"],
            "n_frames": n,
            "locked": locked,
            "baseline_max_conf_030": baseline,
        }
        row["flags"] = flag_slot(row)
        rows.append(row)
        fl = row["flags"]
        side = locked["selected_side_px"] or {}
        print(
            f"{spec['label']}: locked top[{fmt_top(locked['top_winners'])}] "
            f"med_side={side.get('median')} ≥20px={side.get('frac_ge_20px')} "
            f"priority={fl['priority']} {fl['reasons']}",
            flush=True,
        )

    args.out.mkdir(parents=True, exist_ok=True)
    payload = {
        "title": "4quad_locked_policy_survey",
        "locked_policy": lock,
        "goal_r": GOAL_R,
        "goal_p": GOAL_P,
        "slots": rows,
    }
    (args.out / "survey.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# 4quad locked-policy survey (no new labels)",
        "",
        f"Policy: `{TOP_LEFT_POLICY_ID}` — pool Cam4+/Cam5+/P-cams, "
        f"P7≥0.60 others≥0.30, `{TOP_LEFT_PICK_MODE}`.",
        "Caches only (no re-detect). Top Left has gold; other slots = **who wins + ball size**.",
        "",
        "| Region | Locked winners | med px | ≥20px | none | Priority | Notes |",
        "|---|---|---:|---:|---:|---|---|",
    ]
    for r in rows:
        L = r["locked"]
        fl = r["flags"]
        side = L["selected_side_px"] or {}
        lines.append(
            f"| {r['label']} | {fmt_top(L['top_winners'])} | "
            f"{side.get('median', '—')} | {side.get('frac_ge_20px', '—')} | "
            f"{L['selection_share'].get('none', 0)*100:.0f}% | "
            f"**{fl['priority']}** | {'; '.join(fl['reasons']) or '—'} |"
        )
    need = [r for r in rows if r["flags"]["need_label"]]
    watch = [r for r in rows if r["flags"]["priority"] == "watch"]
    lines += [
        "",
        "## Read (minimize labeling)",
        "",
        f"- **Need label now:** "
        + (", ".join(r["label"] for r in need) if need else "**none** — software survey OK"),
        f"- **Watch (winner unscored / soft flags):** "
        + (", ".join(r["label"] for r in watch) if watch else "none"),
        "- Next product step if priorities are ok/watch: **wire lock into live path**, then 5090 latency.",
        "- Only label a non–Top Left Cam4/Cam5 window if a slot stays **high** after live wiring.",
        "",
        "## Baseline max_conf @0.30 (comparison)",
        "",
        "| Region | Baseline winners | med px |",
        "|---|---|---:|",
    ]
    for r in rows:
        B = r["baseline_max_conf_030"]
        side = B["selected_side_px"] or {}
        lines.append(
            f"| {r['label']} | {fmt_top(B['top_winners'])} | {side.get('median', '—')} |"
        )
    (args.out / "survey.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.out / 'survey.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
