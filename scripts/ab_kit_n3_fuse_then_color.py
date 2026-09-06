#!/usr/bin/env python3
"""N3 A/B: fuse-then-color (median cluster feats) vs vote-on-labels.

Gate: multi-cam non-gray rate ≥ baseline + 0.05 AND off50 not worse by >1.0 pp.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.eval_kit_ref_ab import DEFAULT_CACHE, DEFAULT_KIT  # noqa: E402
from scripts.eval_team_id_strategy_grid import bag_from_cache_row  # noqa: E402
from src.mapping.match3_xy import load_calib, map_player_box  # noqa: E402
from src.perception.team_strategy import STRATEGIES  # noqa: E402
from src.review.cam_mosaic import match3_videos  # noqa: E402
from src.review import multicam_fuse as mf  # noqa: E402
from src.review.team_live import TeamSession, label_player_pts  # noqa: E402

OUT_JSON = ROOT / "reports/eval_match3/improve_eng_loop/kit_n3_fuse_then_color_ab.json"
OUT_MD = ROOT / "reports/eval_match3/improve_eng_loop/kit_n3_fuse_then_color_ab.md"
STRAT = STRATEGIES["auto_traj_no_gray"]
GATE_AGREE_DELTA = 0.05
GATE_OFF50_WORSE = 1.0
MAX_FRAMES = 90
FRAME_STRIDE = 2


def _bag_player_pts(bag: dict) -> tuple[list[dict], dict]:
    player_pts = []
    frames_by_cam = {}
    for cam, dets in list(bag.items()):
        if not isinstance(cam, str) or cam.endswith("__wh") or cam.endswith("__bgr"):
            continue
        fr_bgr = bag.get(f"{cam}__bgr")
        if fr_bgr is not None:
            frames_by_cam[cam] = fr_bgr
        wh = bag.get(f"{cam}__wh")
        calib = load_calib(cam)
        if calib is None:
            continue
        for d in dets or []:
            name = str(getattr(d, "class_name", "") or "").lower()
            if name == "ball":
                continue
            mapped = map_player_box(
                calib, d.bbox, float(d.confidence), frame_wh=wh, apply_undistort=False
            )
            if mapped is None:
                continue
            player_pts.append(
                {
                    "xy": mapped["xy"],
                    "team": -1,
                    "pid": -1,
                    "conf": float(mapped["conf"]),
                    "cam": cam,
                    "bbox": tuple(float(v) for v in d.bbox),
                    "frame_wh": wh,
                }
            )
    return player_pts, frames_by_cam


def _stats_from_players(players: list) -> dict:
    n0 = sum(1 for p in players if int(p[2]) == 0)
    n1 = sum(1 for p in players if int(p[2]) == 1)
    ng = sum(1 for p in players if int(p[2]) < 0)
    tot = n0 + n1
    p0 = 100.0 * n0 / max(tot, 1)
    return {
        "n0": n0,
        "n1": n1,
        "gray": ng,
        "pct0": p0,
        "off50": abs(p0 - 50.0),
        "tot": tot,
    }


def _multi_nongray(clusters, cents, rad, fuse_then_color: bool) -> tuple[int, int]:
    n_multi = n_ok = 0
    for cl in clusters:
        cams = {c.get("cam") for c in cl if c.get("cam")}
        if len(cl) < 2 or len(cams) < 2:
            continue
        n_multi += 1
        tid = mf._cluster_team(
            cl, centroids=cents, radius=rad, fuse_then_color=fuse_then_color
        )
        if tid >= 0:
            n_ok += 1
    return n_ok, n_multi


def _run_variant(
    cache: dict, vids: dict, kit: Path, frames: list[int], fuse_then_color: bool
) -> dict:
    sess = TeamSession(strategy=STRAT)
    if not sess.load_centroids_file(kit):
        raise FileNotFoundError(kit)
    n0 = n1 = ng = 0
    off_frames = []
    multi_ok = multi_n = 0
    for fr in frames:
        bag = bag_from_cache_row(cache["data"][str(fr)], vids, fr)
        player_pts, frames_by_cam = _bag_player_pts(bag)
        if not player_pts or not frames_by_cam:
            continue
        label_player_pts(player_pts, frames_by_cam, team_session=sess)
        clusters = mf._cluster_players(player_pts, merge_m=mf.PLAYER_MERGE_M_LIVE)
        ok, n = _multi_nongray(clusters, sess.centroids, sess.radius, fuse_then_color)
        multi_ok += ok
        multi_n += n
        fused = mf._fuse_player_clusters(
            clusters,
            merge_m=mf.PLAYER_MERGE_M_LIVE,
            solo_conf=mf.PLAYER_LIVE_SOLO_CONF,
            ghost_conf=mf.PLAYER_LIVE_GHOST_CONF,
            max_players=mf.FUSE_MAX_PLAYERS,
            solo_team_conf=mf.SOLO_TEAM_CONF,
            centroids=sess.centroids,
            radius=sess.radius,
            fuse_then_color=fuse_then_color,
        )
        fused = sess.stabilize_fused(fused)
        st = _stats_from_players(fused)
        n0 += st["n0"]
        n1 += st["n1"]
        ng += st["gray"]
        if st["tot"] >= 6:
            off_frames.append(st["off50"])
    tot = n0 + n1
    p0 = 100.0 * n0 / max(tot, 1)
    off50 = float(sum(off_frames) / len(off_frames)) if off_frames else abs(p0 - 50.0)
    return {
        "fuse_then_color": fuse_then_color,
        "n_frames": len(frames),
        "n0": n0,
        "n1": n1,
        "gray": ng,
        "pct0": round(p0, 2),
        "pct1": round(100.0 - p0, 2) if tot else 0.0,
        "off50": round(off50, 2),
        "n_multi_cam_clusters": multi_n,
        "n_multi_nongray": multi_ok,
        "nongray_multi_rate": round(multi_ok / max(multi_n, 1), 4),
        "coverage": round(tot / max(tot + ng, 1), 4),
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
    baseline = _run_variant(cache, vids, kit, frames, fuse_then_color=False)
    print("baseline", json.dumps(baseline), flush=True)
    n3 = _run_variant(cache, vids, kit, frames, fuse_then_color=True)
    print("n3", json.dumps(n3), flush=True)
    agree_delta = float(n3["nongray_multi_rate"] - baseline["nongray_multi_rate"])
    off_delta = float(n3["off50"] - baseline["off50"])
    worked = agree_delta >= GATE_AGREE_DELTA and off_delta <= GATE_OFF50_WORSE
    payload = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "cache": str(cache_path),
        "kit": str(kit),
        "gate": {
            "nongray_multi_delta_min": GATE_AGREE_DELTA,
            "off50_worse_max_pp": GATE_OFF50_WORSE,
        },
        "baseline_vote": baseline,
        "N3_fuse_then_color": n3,
        "nongray_multi_delta": round(agree_delta, 4),
        "off50_delta": round(off_delta, 3),
        "worked": worked,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2) + "\n")
    md = [
        "# N3 fuse-then-color A/B",
        "",
        f"**Worked: `{worked}`**",
        "",
        f"Gate: nongray_multi ≥ baseline+{GATE_AGREE_DELTA}, off50 worse ≤ {GATE_OFF50_WORSE} pp.",
        "",
        f"- nongray_multi: {baseline['nongray_multi_rate']:.3f} → {n3['nongray_multi_rate']:.3f} "
        f"(Δ {agree_delta:+.3f})",
        f"- off50: {baseline['off50']:.1f} → {n3['off50']:.1f} (Δ {off_delta:+.2f})",
        f"- share: {baseline['pct0']:.1f}/{baseline['pct1']:.1f} → "
        f"{n3['pct0']:.1f}/{n3['pct1']:.1f}",
        f"- gray fused: {baseline['gray']} → {n3['gray']}",
        f"- multi-cam clusters: {baseline['n_multi_cam_clusters']}",
        "",
    ]
    OUT_MD.write_text("\n".join(md) + "\n")
    print(
        json.dumps(
            {"worked": worked, "nongray_multi_delta": agree_delta, "off50_delta": off_delta},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
