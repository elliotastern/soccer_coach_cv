#!/usr/bin/env python3
"""N10 A/B: ball-proximal majority lock for possession carrier vs baseline fuse.

Gate: composite ≥ baseline+0.3 AND off50 worse ≤ 1.0 pp AND sticky_keep not down >0.02.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import src.review.multicam_fuse as mf  # noqa: E402
from scripts.eval_kit_ref_ab import DEFAULT_CACHE, DEFAULT_KIT  # noqa: E402
from scripts.eval_team_id_strategy_grid import (  # noqa: E402
    STICKY_M,
    analyze_bidir,
    bag_from_cache_row,
    score_run,
)
from src.perception.team_strategy import STRATEGIES  # noqa: E402
from src.review.cam_mosaic import match3_videos  # noqa: E402
from src.review.multicam_fuse import (  # noqa: E402
    fuse_live_dets_for_pitch,
    lock_carrier_team_ball_majority,
)
from src.review.team_live import TeamSession  # noqa: E402

OUT_JSON = ROOT / "reports/eval_match3/improve_eng_loop/kit_n10_ball_prox_ab.json"
OUT_MD = ROOT / "reports/eval_match3/improve_eng_loop/kit_n10_ball_prox_ab.md"
STRAT = STRATEGIES["auto_traj_no_gray"]
GATE_COMPOSITE_DELTA = 0.3
GATE_OFF50_WORSE = 1.0
GATE_STICKY_DROP = 0.02
MAX_FRAMES = 120
FRAME_STRIDE = 3


def _carrier_agree(players, ball_xy) -> bool | None:
    """Whether nearest carrier already matches ball-prox majority (no lock needed)."""
    _, meta = lock_carrier_team_ball_majority(players, ball_xy)
    if meta.get("carrier_i", -1) < 0 or meta.get("maj", -1) < 0:
        return None
    if meta.get("n_prox", 0) < 2:
        return None
    if meta.get("agree_before") is not None:
        return bool(meta["agree_before"])
    # After lock path: recompute on current
    ci = int(meta["carrier_i"])
    return int(players[ci][2]) == int(meta["maj"])


def _run_variant(cache: dict, vids: dict, kit: Path, frames: list[int], *, lock: bool) -> dict:
    mf.USE_BALL_PROX_TEAM_LOCK = bool(lock)
    sess = TeamSession(strategy=STRAT)
    if not sess.load_centroids_file(kit):
        raise FileNotFoundError(kit)
    rows = []
    prev_players = None
    sticky_keep: list[int] = []
    sticky_total = 0
    share_hist = []
    agree_flags: list[int] = []
    n_lock = 0
    n_ball = 0
    off_frames = []

    for fr in frames:
        bag = bag_from_cache_row(cache["data"][str(fr)], vids, fr)
        live = fuse_live_dets_for_pitch(
            bag,
            apply_undistort=False,
            team_session=sess,
            fuse_stats=True,
        )
        players = live["players"]
        ball_xy = live.get("ball_xy")
        if ball_xy is not None:
            n_ball += 1
        bp = live.get("ball_prox") or {}
        if bp.get("locked"):
            n_lock += 1
        if ball_xy is not None and players:
            ag = _carrier_agree(players, ball_xy)
            if ag is not None:
                agree_flags.append(1 if ag else 0)

        teams = [int(p[2]) for p in players]
        n0, n1 = teams.count(0), teams.count(1)
        ng = sum(1 for t in teams if t < 0)
        cons = live.get("consensus") or {}
        rows.append({"fr": fr, "n0": n0, "n1": n1, "gray": ng, "n": len(players), **cons})
        tot = n0 + n1
        share_hist.append(n0 / tot if tot > 0 else 0.5)
        if tot >= 6:
            off_frames.append(abs(100.0 * n0 / tot - 50.0))

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

    metrics = analyze_bidir(rows)
    extra = {
        "_sticky_keep": sticky_keep,
        "_sticky_total": sticky_total,
        "_shares": share_hist,
    }
    scores = score_run(metrics, extra)
    tot = sum(r["n0"] + r["n1"] for r in rows)
    n0 = sum(r["n0"] for r in rows)
    p0 = 100.0 * n0 / max(tot, 1)
    off50 = float(np.mean(off_frames)) if off_frames else abs(p0 - 50.0)
    carrier_agree = float(np.mean(agree_flags)) if agree_flags else 0.0
    return {
        "lock": lock,
        "n_frames": len(frames),
        "n_ball_frames": n_ball,
        "n_locks": n_lock,
        "n_carrier_scored": len(agree_flags),
        "carrier_agree": round(carrier_agree, 4),
        "pct0": round(p0, 2),
        "pct1": round(100.0 - p0, 2) if tot else 0.0,
        "off50": round(off50, 2),
        "sticky_keep": scores["sticky_keep"],
        "scores": scores,
        "metrics": {k: round(float(v), 4) if isinstance(v, float) else v for k, v in metrics.items()},
    }


def main() -> None:
    cache_path = Path(DEFAULT_CACHE)
    cache = json.loads(cache_path.read_text())
    frames = [int(f) for f in cache["frames"]][::FRAME_STRIDE][:MAX_FRAMES]
    vids = match3_videos(ROOT)
    kit = Path(DEFAULT_KIT)
    if not kit.is_file():
        kit = ROOT / "data/output/match_4_5min/team_centroids.json"
    print(f"frames={len(frames)} kit={kit}", flush=True)
    try:
        baseline = _run_variant(cache, vids, kit, frames, lock=False)
        print("baseline", json.dumps({"composite": baseline["scores"]["composite"],
                                      "carrier_agree": baseline["carrier_agree"],
                                      "off50": baseline["off50"]}), flush=True)
        n10 = _run_variant(cache, vids, kit, frames, lock=True)
        print("n10", json.dumps({"composite": n10["scores"]["composite"],
                                 "carrier_agree": n10["carrier_agree"],
                                 "off50": n10["off50"],
                                 "n_locks": n10["n_locks"]}), flush=True)
    finally:
        mf.USE_BALL_PROX_TEAM_LOCK = False

    comp_delta = float(n10["scores"]["composite"] - baseline["scores"]["composite"])
    off_delta = float(n10["off50"] - baseline["off50"])
    sticky_drop = float(baseline["sticky_keep"] - n10["sticky_keep"])
    agree_delta = float(n10["carrier_agree"] - baseline["carrier_agree"])
    worked = (
        comp_delta >= GATE_COMPOSITE_DELTA
        and off_delta <= GATE_OFF50_WORSE
        and sticky_drop <= GATE_STICKY_DROP
    )
    payload = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "cache": str(cache_path),
        "kit": str(kit),
        "gate": {
            "composite_delta_min": GATE_COMPOSITE_DELTA,
            "off50_worse_max_pp": GATE_OFF50_WORSE,
            "sticky_drop_max": GATE_STICKY_DROP,
        },
        "baseline": baseline,
        "N10_ball_prox_lock": n10,
        "composite_delta": round(comp_delta, 3),
        "off50_delta": round(off_delta, 3),
        "sticky_drop": round(sticky_drop, 4),
        "carrier_agree_delta": round(agree_delta, 4),
        "worked": worked,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2) + "\n")
    md = [
        "# N10 ball-proximal carrier lock A/B",
        "",
        f"**Worked: `{worked}`**",
        "",
        f"Gate: composite ≥ baseline+{GATE_COMPOSITE_DELTA}, "
        f"off50 worse ≤ {GATE_OFF50_WORSE} pp, sticky drop ≤ {GATE_STICKY_DROP}.",
        "",
        f"- composite: {baseline['scores']['composite']:.2f} → "
        f"{n10['scores']['composite']:.2f} (Δ {comp_delta:+.2f})",
        f"- carrier_agree: {baseline['carrier_agree']:.3f} → "
        f"{n10['carrier_agree']:.3f} (Δ {agree_delta:+.3f}) · locks={n10['n_locks']}",
        f"- off50: {baseline['off50']:.1f} → {n10['off50']:.1f} (Δ {off_delta:+.2f})",
        f"- sticky_keep: {baseline['sticky_keep']:.3f} → {n10['sticky_keep']:.3f} "
        f"(drop {sticky_drop:+.3f})",
        f"- share: {baseline['pct0']:.1f}/{baseline['pct1']:.1f} → "
        f"{n10['pct0']:.1f}/{n10['pct1']:.1f}",
        f"- ball frames: {n10['n_ball_frames']}/{n10['n_frames']}",
        "",
    ]
    OUT_MD.write_text("\n".join(md) + "\n")
    print(
        json.dumps(
            {
                "worked": worked,
                "composite_delta": comp_delta,
                "carrier_agree_delta": agree_delta,
                "off50_delta": off_delta,
                "sticky_drop": sticky_drop,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
