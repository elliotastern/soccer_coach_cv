#!/usr/bin/env python3
"""Holdout A/B: mutual-nearest color soft merge (anti-teammate) soft_m sweep."""
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
SOFT_MS = (3.2, 3.6, 4.0, 4.5)


def _dist(a, b) -> float:
    return float(((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5)


def cluster_mutual_soft(player_pts, *, base_m: float, soft_m: float, base_cluster):
    clusters = base_cluster(player_pts, merge_m=float(base_m))
    # solos only
    solo_idx = [i for i, cl in enumerate(clusters) if len(cl) == 1]
    solos = [(i, clusters[i][0]) for i in solo_idx]
    # nearest eligible cross-cam same-team within (base, soft]
    nearest = {}
    for i, p in solos:
        tid = int(p.get("team", -1))
        if tid < 0:
            continue
        best_j, best_d = None, 1e9
        for j, q in solos:
            if j == i:
                continue
            if p.get("cam") and q.get("cam") == p.get("cam"):
                continue
            if int(q.get("team", -1)) != tid:
                continue
            d = _dist(p["xy"], q["xy"])
            if d <= float(base_m) or d > float(soft_m):
                continue
            if d < best_d:
                best_d, best_j = d, j
        if best_j is not None:
            nearest[i] = (best_j, best_d)
    # mutual pairs
    used = set()
    pairs = []
    for i, (j, d) in nearest.items():
        if i in used or j in used:
            continue
        if nearest.get(j, (None,))[0] == i:
            pairs.append((i, j, d))
            used.add(i)
            used.add(j)
    for i, j, _ in pairs:
        clusters[i].append(clusters[j][0])
        clusters[j] = []
    return [cl for cl in clusters if cl]


def run_variant(cache, vids, kit, frames, *, soft_m: float, mutual: bool):
    sess = TeamSession(strategy=STRAT)
    if not sess.load_centroids_file(kit):
        raise FileNotFoundError(kit)
    orig = mf._cluster_players
    soft_m_cfg = float(soft_m)

    def _wrapped(player_pts, merge_m=mf.PLAYER_MERGE_M, soft_m=None, color_gate=False):
        # Ignore caller soft kwargs; A/B controls behavior via closure.
        if not mutual:
            return orig(
                player_pts,
                merge_m=BASE_M,
                soft_m=soft_m_cfg,
                color_gate=True,
            )

        def hard_only(pts, merge_m=BASE_M, soft_m=None, color_gate=False):
            return orig(pts, merge_m=merge_m, soft_m=None, color_gate=False)

        return cluster_mutual_soft(
            player_pts, base_m=BASE_M, soft_m=soft_m_cfg, base_cluster=hard_only
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
                merge_m=BASE_M,
                soft_m=soft_m,
                color_gate_soft=True,
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
        return rows, {"_sticky_keep": sticky_keep, "_sticky_total": sticky_total, "_shares": shares}
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

    variants = [
        ("product_soft_3.2", 3.2, False),  # current product absorb-into-cluster
        ("mutual_3.2", 3.2, True),
        ("mutual_3.6", 3.6, True),
        ("mutual_4.0", 4.0, True),
        ("mutual_4.5", 4.5, True),
    ]
    rows = []
    for name, soft, mutual in variants:
        tr, te = run_variant(cache, vids, kit, tune_fr, soft_m=soft, mutual=mutual)
        hr, he = run_variant(cache, vids, kit, hold_fr, soft_m=soft, mutual=mutual)
        tune, hold = summarize(tr, te), summarize(hr, he)
        row = {
            "variant": name,
            "soft_m": soft,
            "mutual": mutual,
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

    base = next(r for r in rows if r["variant"] == "product_soft_3.2")
    cands = [r for r in rows if r["variant"] != "product_soft_3.2"]
    best = max(cands, key=lambda r: (r["hold_consensus"], r["hold_multcam"]))
    lift = best["hold_consensus"] - base["hold_consensus"]
    promote = (
        lift >= 0.3
        and best["hold_collapse"] <= base["hold_collapse"] + 0.03
        and best["hold_mean_players"] >= base["hold_mean_players"] - 1.5
        and best["hold_multcam"] + 1e-9 >= base["hold_multcam"]
    )
    out = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "rows": rows,
        "base": base,
        "best": best,
        "hold_lift": round(lift, 3),
        "decision": "promote" if promote else "no_promote",
        "promote_rule": "hold consensus +≥0.3 vs product_soft_3.2; mfc not down; health",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "ab_mutual_nearest_soft_extend.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("decision", out["decision"], "lift", out["hold_lift"])
    print("wrote", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
