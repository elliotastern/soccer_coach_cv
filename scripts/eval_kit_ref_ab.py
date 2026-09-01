#!/usr/bin/env python3
"""A/B team ID: production strategy with vs without pre-labeled kit centroids."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.eval_team_id_strategy_grid import (  # noqa: E402
    CAMS,
    STICKY_M,
    analyze_bidir,
    bag_from_cache_row,
    cache_path,
    score_run,
)
from src.perception.team_strategy import STRATEGIES  # noqa: E402
from src.review.cam_mosaic import match3_videos  # noqa: E402
from src.review.multicam_fuse import fuse_live_dets_for_pitch  # noqa: E402
from src.review.team_live import TeamSession  # noqa: E402

OUT = ROOT / "reports/eval_match3/improve_eng_loop/kit_ref_ab"
DEFAULT_KIT = ROOT / "data/output/match_4_5min/P10-match4/team_centroids.json"
DEFAULT_CACHE = cache_path("plain")
STRAT = STRATEGIES["auto_traj_no_gray"]


def run_labeled(cache: dict, vids: dict, centroids_path: Path | None) -> tuple[list[dict], dict]:
    frames = cache["frames"]
    sess = TeamSession(strategy=STRAT)
    if centroids_path is not None:
        if not sess.load_centroids_file(centroids_path):
            raise FileNotFoundError(centroids_path)
    rows = []
    prev_players = None
    sticky_keep: list[int] = []
    sticky_total = 0
    share_hist = []

    for fr in frames:
        bag = bag_from_cache_row(cache["data"][str(fr)], vids, fr)
        live = fuse_live_dets_for_pitch(
            bag, apply_undistort=False, team_session=sess, fuse_stats=True,
        )
        players = live["players"]
        teams = [int(p[2]) for p in players]
        n0, n1 = teams.count(0), teams.count(1)
        ng = sum(1 for t in teams if t < 0)
        cons = live.get("consensus") or {}
        rows.append({"fr": fr, "n0": n0, "n1": n1, "gray": ng, "n": len(players), **cons})

        if prev_players is not None:
            for p in players:
                if int(p[2]) < 0:
                    continue
                near_t = None
                best_d = STICKY_M
                for q in prev_players:
                    d = ((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2) ** 0.5
                    if d <= best_d:
                        best_d, near_t = d, int(q[2])
                if near_t is None or near_t < 0:
                    continue
                sticky_total += 1
                sticky_keep.append(1 if int(p[2]) == near_t else 0)
        prev_players = [(float(p[0]), float(p[1]), int(p[2])) for p in players]
        tot = n0 + n1
        share_hist.append(n0 / tot if tot > 0 else 0.5)

    extra = {"_sticky_keep": sticky_keep, "_sticky_total": sticky_total, "_shares": share_hist}
    return rows, extra


def summarize(label: str, rows: list[dict], extra: dict) -> dict:
    metrics = analyze_bidir(rows)
    scores = score_run(metrics, extra)
    flip_events = int(sum(1 for k in extra.get("_sticky_keep", []) if k == 0))
    return {
        "label": label,
        "metrics": metrics,
        "scores": scores,
        "flip_events": flip_events,
        "sticky_total": int(extra.get("_sticky_total", 0)),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--kit-ref", type=Path, default=DEFAULT_KIT)
    p.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if not args.cache.is_file():
        print(f"missing cache {args.cache}", file=sys.stderr)
        return 1
    if not args.kit_ref.is_file():
        print(f"missing kit ref {args.kit_ref}", file=sys.stderr)
        return 1

    cache = json.loads(args.cache.read_text(encoding="utf-8"))
    vids = match3_videos(ROOT)
    baseline_rows, baseline_extra = run_labeled(cache, vids, None)
    kit_rows, kit_extra = run_labeled(cache, vids, args.kit_ref)

    baseline = summarize("baseline_online_fit", baseline_rows, baseline_extra)
    kit_ref = summarize("kit_ref_seeded", kit_rows, kit_extra)
    kit_meta = json.loads(args.kit_ref.read_text(encoding="utf-8"))

    payload = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "kit_ref": str(args.kit_ref),
        "kit_meta": {
            k: kit_meta.get(k)
            for k in ("team_names", "n_samples", "kit_mode", "radius")
        },
        "cache": str(args.cache),
        "strategy": STRAT.name,
        "n_frames": len(cache["frames"]),
        "baseline": baseline,
        "kit_ref_run": kit_ref,
        "delta": {
            "mean_blue_share": kit_ref["metrics"]["mean_blue_share"]
            - baseline["metrics"]["mean_blue_share"],
            "collapse_frac": kit_ref["metrics"]["collapse_frac"]
            - baseline["metrics"]["collapse_frac"],
            "balanced_frac": kit_ref["metrics"]["balanced_frac"]
            - baseline["metrics"]["balanced_frac"],
            "both3_frac": kit_ref["metrics"]["both3_frac"]
            - baseline["metrics"]["both3_frac"],
            "flickering_score": kit_ref["scores"]["flickering"]
            - baseline["scores"]["flickering"],
            "sticky_keep": kit_ref["scores"]["sticky_keep"]
            - baseline["scores"]["sticky_keep"],
            "swing_rate": kit_ref["scores"]["swing_rate"]
            - baseline["scores"]["swing_rate"],
            "flip_events": kit_ref["flip_events"] - baseline["flip_events"],
        },
    }
    out_json = OUT / "kit_ref_ab.json"
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def fmt(r: dict) -> str:
        m, s = r["metrics"], r["scores"]
        return (
            f"{r['label']}: blue_share={m['mean_blue_share']:.3f} "
            f"collapse={m['collapse_frac']:.3f} balanced={m['balanced_frac']:.3f} "
            f"both3={m['both3_frac']:.3f} flicker={s['flickering']:.2f} "
            f"sticky_keep={s['sticky_keep']:.3f} swing={s['swing_rate']:.3f} "
            f"flips={r['flip_events']}"
        )

    print(fmt(baseline))
    print(fmt(kit_ref))
    print("delta:", json.dumps(payload["delta"], indent=2))
    print(f"wrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
