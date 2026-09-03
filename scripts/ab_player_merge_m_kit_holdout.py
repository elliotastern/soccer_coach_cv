#!/usr/bin/env python3
"""Match-4 holdout-gated A/B for PLAYER_MERGE_M_LIVE → kit consensus.

Tune = first half of cache frames; hold = second half.
Promote only if hold consensus rises without collapse/mean_players kill.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.eval_kit_ref_ab import DEFAULT_CACHE, DEFAULT_KIT  # noqa: E402
from scripts.eval_team_id_strategy_grid import analyze_bidir, bag_from_cache_row, score_run  # noqa: E402
from src.perception.team_strategy import STRATEGIES  # noqa: E402
from src.review.cam_mosaic import match3_videos  # noqa: E402
from src.review.multicam_fuse import (  # noqa: E402
    PLAYER_MERGE_M_LIVE,
    fuse_live_dets_for_pitch,
)
from src.review.team_live import TeamSession  # noqa: E402

OUT = ROOT / "reports/eval_match3/improve_eng_loop/kit_ref_ab"
STRAT = STRATEGIES["auto_traj_no_gray"]
MERGES = (2.2, 2.5, 2.8, 3.2)
STICKY_M = 1.5


def summarize(rows: list[dict], extra: dict) -> dict:
    metrics = analyze_bidir(rows)
    scores = score_run(metrics, extra)
    return {"metrics": metrics, "scores": scores}


def run_merge(
    cache: dict,
    vids: dict,
    centroids_path: Path,
    frames: list[int],
    merge_m: float,
) -> tuple[list[dict], dict]:
    sess = TeamSession(strategy=STRAT)
    if not sess.load_centroids_file(centroids_path):
        raise FileNotFoundError(centroids_path)
    rows = []
    prev = None
    sticky_keep: list[int] = []
    sticky_total = 0
    shares = []
    for fr in frames:
        bag = bag_from_cache_row(cache["data"][str(fr)], vids, fr)
        live = fuse_live_dets_for_pitch(
            bag,
            apply_undistort=False,
            team_session=sess,
            fuse_stats=True,
            merge_m=float(merge_m),
        )
        players = live["players"]
        teams = [int(p[2]) for p in players]
        n0, n1 = teams.count(0), teams.count(1)
        ng = sum(1 for t in teams if t < 0)
        cons = live.get("consensus") or {}
        rows.append({"fr": fr, "n0": n0, "n1": n1, "gray": ng, "n": len(players), **cons})
        if prev is not None:
            for p in players:
                if int(p[2]) < 0:
                    continue
                near_t = None
                best_d = STICKY_M
                for q in prev:
                    d = ((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2) ** 0.5
                    if d <= best_d:
                        best_d, near_t = d, int(q[2])
                if near_t is None or near_t < 0:
                    continue
                sticky_total += 1
                sticky_keep.append(1 if int(p[2]) == near_t else 0)
        prev = [(float(p[0]), float(p[1]), int(p[2])) for p in players]
        tot = n0 + n1
        shares.append(n0 / tot if tot else 0.5)
    extra = {"_sticky_keep": sticky_keep, "_sticky_total": sticky_total, "_shares": shares}
    return rows, extra


def main() -> int:
    cache = json.loads(Path(DEFAULT_CACHE).read_text())
    frames = [int(f) for f in cache["frames"]]
    mid = len(frames) // 2
    tune_fr, hold_fr = frames[:mid], frames[mid:]
    vids = match3_videos(ROOT)
    kit = Path(DEFAULT_KIT)
    if not kit.is_file():
        kit = ROOT / "data/output/match_4_5min/team_centroids.json"

    results = []
    for m in MERGES:
        print(f"merge_m={m} tune={len(tune_fr)} hold={len(hold_fr)}", flush=True)
        tune_rows, tune_ex = run_merge(cache, vids, kit, tune_fr, m)
        hold_rows, hold_ex = run_merge(cache, vids, kit, hold_fr, m)
        tune_sc = summarize(tune_rows, tune_ex)
        hold_sc = summarize(hold_rows, hold_ex)
        results.append(
            {
                "merge_m": m,
                "tune": tune_sc,
                "hold": hold_sc,
            }
        )
        print(
            f"  tune consensus={tune_sc['scores']['consensus']:.2f} "
            f"multcam={tune_sc['metrics']['multcam_frac']:.3f} "
            f"| hold consensus={hold_sc['scores']['consensus']:.2f} "
            f"multcam={hold_sc['metrics']['multcam_frac']:.3f}",
            flush=True,
        )

    base = next(r for r in results if abs(r["merge_m"] - PLAYER_MERGE_M_LIVE) < 1e-6)
    base_hold_c = float(base["hold"]["scores"]["consensus"])
    cand = None
    for r in results:
        if abs(r["merge_m"] - PLAYER_MERGE_M_LIVE) < 1e-6:
            continue
        hc = float(r["hold"]["scores"]["consensus"])
        hm = float(r["hold"]["metrics"]["multcam_frac"])
        bm = float(base["hold"]["metrics"]["multcam_frac"])
        collapse = float(r["hold"]["metrics"].get("collapse_frac") or 0)
        base_collapse = float(base["hold"]["metrics"].get("collapse_frac") or 0)
        mean_p = float(r["hold"]["metrics"].get("mean_players") or 0)
        base_p = float(base["hold"]["metrics"].get("mean_players") or 0)
        ok = (
            hc >= base_hold_c + 0.3
            and hm >= bm - 1e-6
            and collapse <= base_collapse + 0.02
            and mean_p >= base_p - 1.0
        )
        r["promote_hold"] = ok
        if ok and (cand is None or hc > float(cand["hold"]["scores"]["consensus"])):
            cand = r

    payload = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "cache": str(DEFAULT_CACHE),
        "kit_ref": str(kit),
        "baseline_merge_m": PLAYER_MERGE_M_LIVE,
        "tune_n": len(tune_fr),
        "hold_n": len(hold_fr),
        "results": results,
        "promote_candidate": cand["merge_m"] if cand else None,
        "decision": "promote" if cand else "no_promote",
        "note": "Holdout-gated merge sweep for kit consensus. Smoke Match3 frames not used.",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "ab_player_merge_m_kit_holdout.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"decision={payload['decision']} cand={payload['promote_candidate']} wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
