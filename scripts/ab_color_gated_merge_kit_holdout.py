#!/usr/bin/env python3
"""Holdout-gated A/B: color-gated soft merge for kit consensus.

Hard merge at base_m (default 2.2). Soft merge up to soft_m only when both
points already share the same non-gray team label. Promote only if hold
consensus rises ≥ +0.3 without collapse spike.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.ab_player_merge_m_kit_holdout import (  # noqa: E402
    DEFAULT_CACHE,
    DEFAULT_KIT,
    STICKY_M,
    summarize,
)
from scripts.eval_team_id_strategy_grid import bag_from_cache_row  # noqa: E402
from src.perception.team_strategy import STRATEGIES  # noqa: E402
from src.review.cam_mosaic import match3_videos  # noqa: E402
from src.review import multicam_fuse as mf  # noqa: E402
from src.review.team_live import TeamSession  # noqa: E402

OUT = ROOT / "reports/eval_match3/improve_eng_loop/kit_ref_ab"
STRAT = STRATEGIES["auto_traj_no_gray"]
BASE_M = 2.2
SOFT_MS = (2.5, 2.8, 3.2)


def _dist(a, b) -> float:
    return float(((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5)


def cluster_color_gated(
    player_pts: list[dict],
    *,
    base_m: float,
    soft_m: float,
    base_cluster,
) -> list[list[dict]]:
    """Cluster at base_m, then absorb same-team cross-cam solos up to soft_m."""
    clusters = base_cluster(player_pts, merge_m=float(base_m))
    changed = True
    while changed:
        changed = False
        solos = [i for i, cl in enumerate(clusters) if len(cl) == 1]
        for i in solos:
            p = clusters[i][0]
            tid = int(p.get("team", -1))
            if tid < 0:
                continue
            best_j, best_d = -1, 1e9
            for j, cl in enumerate(clusters):
                if j == i or not cl:
                    continue
                q = max(cl, key=lambda c: c["conf"])
                d = _dist(p["xy"], q["xy"])
                if d <= float(base_m) or d > float(soft_m):
                    continue
                same = any(int(c.get("team", -1)) == tid for c in cl)
                if not same and int(q.get("team", -1)) != tid:
                    continue
                if p.get("cam") and any(c.get("cam") == p.get("cam") for c in cl):
                    continue
                if d < best_d:
                    best_d, best_j = d, j
            if best_j >= 0:
                clusters[best_j].append(p)
                clusters[i] = []
                changed = True
        clusters = [cl for cl in clusters if cl]
    return clusters


def run_variant(
    cache: dict,
    vids: dict,
    centroids_path: Path,
    frames: list[int],
    *,
    soft_m: float | None,
) -> tuple[list[dict], dict]:
    sess = TeamSession(strategy=STRAT)
    if not sess.load_centroids_file(centroids_path):
        raise FileNotFoundError(centroids_path)

    orig = mf._cluster_players

    def _wrapped(player_pts, merge_m=mf.PLAYER_MERGE_M):
        if soft_m is None:
            return orig(player_pts, merge_m=merge_m)
        return cluster_color_gated(
            player_pts, base_m=BASE_M, soft_m=float(soft_m), base_cluster=orig
        )

    mf._cluster_players = _wrapped  # type: ignore
    try:
        rows = []
        prev = None
        sticky_keep: list[int] = []
        sticky_total = 0
        shares = []
        for fr in frames:
            bag = bag_from_cache_row(cache["data"][str(fr)], vids, fr)
            live = mf.fuse_live_dets_for_pitch(
                bag,
                apply_undistort=False,
                team_session=sess,
                fuse_stats=True,
                merge_m=float(BASE_M),
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
    finally:
        mf._cluster_players = orig  # type: ignore


def main() -> int:
    cache = json.loads(Path(DEFAULT_CACHE).read_text(encoding="utf-8"))
    frames = [int(f) for f in cache["frames"]]
    mid = len(frames) // 2
    tune_fr, hold_fr = frames[:mid], frames[mid:]
    vids = match3_videos(ROOT)
    kit = Path(DEFAULT_KIT)
    if not kit.is_file():
        kit = ROOT / "data/output/match_4_5min/team_centroids.json"

    variants: list[tuple[str, float | None]] = [("baseline_2.2", None)]
    variants += [(f"color_soft_{m}", m) for m in SOFT_MS]

    rows = []
    for name, soft in variants:
        tr, te = run_variant(cache, vids, kit, tune_fr, soft_m=soft)
        hr, he = run_variant(cache, vids, kit, hold_fr, soft_m=soft)
        tune = summarize(tr, te)
        hold = summarize(hr, he)
        row = {
            "variant": name,
            "soft_m": soft,
            "tune_consensus": tune["scores"]["consensus"],
            "hold_consensus": hold["scores"]["consensus"],
            "tune_multcam": tune["metrics"]["multcam_frac"],
            "hold_multcam": hold["metrics"]["multcam_frac"],
            "hold_agree": hold["metrics"]["agree_frac"],
            "hold_collapse": hold["metrics"]["collapse_frac"],
            "hold_composite": hold["scores"]["composite"],
            "hold_mean_players": hold["metrics"]["mean_players"],
        }
        rows.append(row)
        print(json.dumps(row), flush=True)

    base = next(r for r in rows if r["variant"] == "baseline_2.2")
    cands = [r for r in rows if r["variant"] != "baseline_2.2"]
    best = max(cands, key=lambda r: r["hold_consensus"])
    lift = best["hold_consensus"] - base["hold_consensus"]
    promote = (
        lift >= 0.3
        and best["hold_collapse"] <= base["hold_collapse"] + 0.03
        and best["hold_mean_players"] >= base["hold_mean_players"] - 1.5
    )
    out = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "base_m": BASE_M,
        "near_miss_why": "kit_ref_ab/color_gate_near_miss_why.json",
        "rows": rows,
        "best": best,
        "hold_lift": round(lift, 3),
        "decision": "promote" if promote else "no_promote",
        "promote_rule": "hold consensus +≥0.3 vs baseline_2.2; collapse/mean_players health",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "ab_color_gated_merge_kit_holdout.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("decision", out["decision"], "lift", out["hold_lift"])
    print("wrote", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
