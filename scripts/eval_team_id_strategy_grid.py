#!/usr/bin/env python3
"""Top-10 team ID strategy grid on Match 4 90s (det cache + in-memory sweep)."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.perception.rfdetr_local import LocalRFDETRDetector  # noqa: E402
from src.perception.team_core import (  # noqa: E402
    TEAM_MIN_CROPS,
    assign_feature,
    fit_match_centroids,
    jersey_feature,
    torso_crop,
)
from src.perception.team_strategy import STRATEGIES, TeamStrategy, list_strategies  # noqa: E402
from src.perception.team_tracklet import TrackletAccumulator, TrackletTeamModel  # noqa: E402
from src.perception.team import extract_jersey_color  # noqa: E402
from src.review.cam_mosaic import _ensure_cam_dets, match3_videos  # noqa: E402
from src.review.frame_sync import keep_top1_ball  # noqa: E402
from src.review.multicam_fuse import fuse_live_dets_for_pitch  # noqa: E402
from src.review.team_live import STICKY_M, TeamSession, label_player_pts  # noqa: E402

OUT = ROOT / "reports/eval_match3/team_id_strategy_grid"
CAMS = ["P10", "P9", "P7", "P8"]
SPOT_FR = 750
SHARE_SWING = 0.30
MEAN_BLUE_LO, MEAN_BLUE_HI = 0.35, 0.65


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--match-sec", type=float, default=90.0)
    p.add_argument("--stride", type=int, default=15)
    p.add_argument("--src-fps", type=float, default=60.0)
    p.add_argument("--sahi", type=str, default="plain,sahi")
    p.add_argument("--strategies", type=str, default="all")
    p.add_argument("--build-cache-only", action="store_true")
    p.add_argument("--skip-cache-build", action="store_true")
    return p.parse_args()


def frame_ids(start: int, match_sec: float, stride: int, src_fps: float) -> list[int]:
    n = int(match_sec * src_fps)
    return list(range(start, start + n, stride))


def det_to_dict(d) -> dict:
    return {
        "bbox": [float(v) for v in d.bbox],
        "confidence": float(d.confidence),
        "class_id": int(getattr(d, "class_id", 0)),
        "class_name": str(getattr(d, "class_name", "player")),
    }


def dict_to_det(row: dict):
    return SimpleNamespace(
        bbox=tuple(row["bbox"]),
        confidence=float(row["confidence"]),
        class_id=int(row["class_id"]),
        class_name=str(row.get("class_name", "player")),
    )


def cache_path(sahi_key: str) -> Path:
    return OUT / f"m4_90s_det_{sahi_key}.json"


def build_cache(
    frames: list[int],
    use_sahi: bool,
    vids: dict,
    det: LocalRFDETRDetector,
) -> dict:
    def detect_fn(cam: str, frame_bgr):
        return keep_top1_ball(det.detect(frame_bgr))

    data: dict[str, dict] = {}
    for i, fr in enumerate(frames):
        bag: dict = {}
        for cam in CAMS:
            _ensure_cam_dets(vids, cam, fr, bag, detect_fn, True)
        row: dict[str, list] = {}
        for cam in CAMS:
            dets = bag.get(cam) or []
            row[cam] = [det_to_dict(d) for d in dets]
            if f"{cam}__bgr" in bag:
                row[f"{cam}__bgr_shape"] = list(bag[f"{cam}__bgr"].shape)
        data[str(fr)] = {"dets": row, "bgr": {}}
        for cam in CAMS:
            frb = bag.get(f"{cam}__bgr")
            if frb is not None:
                ok, enc = cv2.imencode(".jpg", frb, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                if ok:
                    data[str(fr)]["bgr"][cam] = enc.tobytes().hex()
        if (i + 1) % 20 == 0:
            print(f"cache fr {fr} ({i+1}/{len(frames)})", flush=True)
    return {
        "frames": frames,
        "cams": CAMS,
        "use_sahi": use_sahi,
        "data": data,
    }


def bag_from_cache_row(row: dict, vids: dict, fr: int) -> dict:
    bag: dict = {}
    for cam in CAMS:
        bag[cam] = [dict_to_det(d) for d in row["dets"].get(cam, [])]
        hex_bgr = row.get("bgr", {}).get(cam)
        if hex_bgr:
            buf = bytes.fromhex(hex_bgr)
            arr = np.frombuffer(buf, dtype=np.uint8)
            bag[f"{cam}__bgr"] = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if bag[f"{cam}__bgr"] is not None:
                bag[f"{cam}__wh"] = (bag[f"{cam}__bgr"].shape[1], bag[f"{cam}__bgr"].shape[0])
    return bag


def legacy_label_pts(player_pts: list[dict], frames_by_cam: dict) -> None:
    from sklearn.cluster import KMeans

    colors, idxs = [], []
    for i, p in enumerate(player_pts):
        cam = p.get("cam")
        fr = frames_by_cam.get(cam)
        bbox = p.get("bbox")
        if fr is None or bbox is None:
            continue
        colors.append(extract_jersey_color(fr, bbox))
        idxs.append(i)
    if len(colors) < 2:
        return
    labels = KMeans(n_clusters=2, random_state=42, n_init=10).fit_predict(np.array(colors))
    score = KMeans(n_clusters=2, random_state=42, n_init=10).fit(np.array(colors)).cluster_centers_
    order = np.argsort(-score[:, 2])
    remap = {int(old): int(new) for new, old in enumerate(order)}
    for j, i in enumerate(idxs):
        player_pts[i]["team"] = remap[int(labels[j])]
        player_pts[i]["team_conf"] = 0.7


def per_frame_label(player_pts: list[dict], frames_by_cam: dict, strat: TeamStrategy) -> None:
    feats, idxs = [], []
    for i, p in enumerate(player_pts):
        cam, bbox = p.get("cam"), p.get("bbox")
        fr = frames_by_cam.get(cam) if cam else None
        wh = (fr.shape[1], fr.shape[0]) if fr is not None else None
        if fr is None or bbox is None:
            continue
        crop = torso_crop(fr, bbox, cam=cam, frame_wh=wh)
        feat = jersey_feature(crop) if crop is not None else None
        if feat is None:
            continue
        feats.append(feat)
        idxs.append(i)
    if len(feats) < TEAM_MIN_CROPS:
        return
    fit = fit_match_centroids(feats, min_crops=TEAM_MIN_CROPS, kit_mode=strat.kit_mode)
    if fit is None:
        return
    cents, radius = fit
    for j, i in enumerate(idxs):
        xy = player_pts[i].get("xy")
        pos = (float(xy[0]), float(xy[1])) if xy is not None else None
        tid, conf = assign_feature(feats[j], cents, radius, pos, strat.kit_mode, strat)
        player_pts[i]["team"] = int(tid)
        player_pts[i]["team_conf"] = float(conf)


def legacy_fuse_frame(bag: dict, frames_by_cam: dict) -> tuple[list, dict]:
    from src.mapping.match3_xy import load_calib, map_player_box
    from src.review.multicam_fuse import (
        FUSE_MAX_PLAYERS,
        PLAYER_LIVE_GHOST_CONF,
        PLAYER_LIVE_SOLO_CONF,
        PLAYER_MERGE_M_LIVE,
        SOLO_TEAM_CONF,
        _cluster_players,
        _fuse_player_clusters,
        player_det_ok,
    )

    pts = []
    for cam in CAMS:
        calib = load_calib(cam)
        if calib is None:
            continue
        wh = bag.get(f"{cam}__wh")
        for d in bag.get(cam) or []:
            if str(getattr(d, "class_name", "")).lower() == "ball":
                continue
            if not player_det_ok(d):
                continue
            mapped = map_player_box(
                calib, d.bbox, float(d.confidence), frame_wh=wh, apply_undistort=False,
            )
            if mapped is None:
                continue
            pts.append({
                "xy": mapped["xy"], "team": -1, "pid": -1, "conf": mapped["conf"],
                "cam": cam, "bbox": d.bbox, "frame_wh": wh,
            })
    legacy_label_pts(pts, frames_by_cam)
    clusters = _cluster_players(pts, merge_m=PLAYER_MERGE_M_LIVE)
    return _fuse_player_clusters(
        clusters, merge_m=PLAYER_MERGE_M_LIVE, solo_conf=PLAYER_LIVE_SOLO_CONF,
        ghost_conf=PLAYER_LIVE_GHOST_CONF, max_players=FUSE_MAX_PLAYERS,
        solo_team_conf=SOLO_TEAM_CONF, return_stats=True,
    )


def _cell_key(xy) -> tuple[int, int]:
    return (int(round(float(xy[0]) / 2.0)), int(round(float(xy[1]) / 2.0)))


class TrackletGoldenSession:
    """S10: golden-batch tracklet fit on first N frames."""

    def __init__(self, strat: TeamStrategy):
        self.strat = strat
        self._acc = TrackletAccumulator()
        self._model = TrackletTeamModel()
        self._cell_tid: dict[tuple[int, int], int] = {}
        self._cell_lab: dict[tuple[int, int], tuple[int, float]] = {}
        self._fit = False
        self._n = 0
        self._next_tid = 1

    def label(self, player_pts: list[dict], frames_by_cam: dict) -> list[dict]:
        self._n += 1
        for p in player_pts:
            p["team"] = -1
            p["team_conf"] = 0.0
            xy = p.get("xy") or (0.0, 0.0)
            key = _cell_key(xy)
            if self._fit and key in self._cell_lab:
                tid, conf = self._cell_lab[key]
                p["team"], p["team_conf"] = int(tid), float(conf)
                continue
            cam, bbox = p.get("cam"), p.get("bbox")
            fr = frames_by_cam.get(cam) if cam else None
            wh = (fr.shape[1], fr.shape[0]) if fr is not None else None
            if self._fit or fr is None or bbox is None:
                continue
            tid = self._cell_tid.setdefault(key, self._next_tid)
            if tid == self._next_tid:
                self._next_tid += 1
            self._acc.add(tid, fr, bbox, (float(xy[0]), float(xy[1])), cam, wh)
        if not self._fit and self._n >= self.strat.golden_frames:
            if self._model.fit_from_accumulator(self._acc):
                self._fit = True
                for key, tid in self._cell_tid.items():
                    if tid in self._model.track_labels:
                        self._cell_lab[key] = self._model.track_labels[tid]
        return player_pts

    def stabilize_fused(self, players):
        return players


def run_strategy(
    strat: TeamStrategy,
    cache: dict,
    vids: dict,
) -> tuple[list[dict], dict]:
    frames = cache["frames"]
    sess = TeamSession(strategy=strat) if strat.use_session else None
    golden = TrackletGoldenSession(strat) if strat.use_tracklet_golden else None
    rows = []
    prev_players = None
    sticky_keep: list[int] = []
    sticky_total = 0
    share_hist = []

    for fr in frames:
        bag = bag_from_cache_row(cache["data"][str(fr)], vids, fr)
        frames_by_cam = {c: bag[f"{c}__bgr"] for c in CAMS if f"{c}__bgr" in bag}

        if strat.legacy_rgb_top10:
            players, consensus = legacy_fuse_frame(bag, frames_by_cam)
            live = {"players": players, "consensus": consensus}
        elif strat.per_frame_only:
            sess_pf = TeamSession(strategy=strat)
            live = fuse_live_dets_for_pitch(
                bag, apply_undistort=False, team_session=sess_pf, fuse_stats=True,
            )
            sess_pf.reset()
        elif golden is not None:
            live = fuse_live_dets_for_pitch(
                bag, apply_undistort=False, team_session=golden, fuse_stats=True,
            )
        else:
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


def analyze_bidir(stats: list[dict]) -> dict:
    rows = [s for s in stats if "fr" in s]
    if not rows:
        return {}
    blue_shares, collapse = [], 0
    both3 = balanced = 0
    multi_sum = agree_sum = labeled_sum = conflict_sum = cluster_sum = 0
    for s in rows:
        n0, n1 = int(s["n0"]), int(s["n1"])
        n = n0 + n1
        if n > 0:
            blue_shares.append(n0 / n)
            if 0.25 <= n0 / n <= 0.75:
                balanced += 1
        if (n1 <= 1 and n0 >= 5) or (n0 <= 1 and n1 >= 5):
            collapse += 1
        if n0 >= 3 and n1 >= 3:
            both3 += 1
        cluster_sum += int(s.get("n_clusters", 0))
        multi_sum += int(s.get("n_multi_cam", 0))
        agree_sum += int(s.get("n_agree", 0))
        labeled_sum += int(s.get("n_labeled_multi", 0))
        conflict_sum += int(s.get("n_conflict_gray", 0))
    nf = len(rows)
    return {
        "n_frames": nf,
        "mean_blue_share": float(np.mean(blue_shares)) if blue_shares else 0.0,
        "collapse_frac": collapse / nf,
        "both3_frac": both3 / nf,
        "balanced_frac": balanced / nf,
        "mean_players": float(np.mean([s["n"] for s in rows])),
        "multcam_frac": multi_sum / max(cluster_sum, 1),
        "agree_frac": agree_sum / max(labeled_sum, 1),
        "conflict_gray_frac": conflict_sum / max(labeled_sum, 1),
    }


def score_run(metrics: dict, extra: dict) -> dict:
    shares = extra.get("_shares") or []
    swings = sum(
        1 for i in range(1, len(shares))
        if abs(shares[i] - shares[i - 1]) >= SHARE_SWING
    )
    swing_rate = swings / max(len(shares) - 1, 1)
    sk = extra.get("_sticky_keep") or []
    sticky_keep = float(np.mean(sk)) if sk else 0.0

    mb = metrics.get("mean_blue_share", 0.5)
    bal_blue = 10.0 if MEAN_BLUE_LO <= mb <= MEAN_BLUE_HI else max(0.0, 10.0 - abs(mb - 0.5) * 20)
    collapse = metrics.get("collapse_frac", 1.0)
    bal_col = 10.0 if collapse <= 0.15 else max(0.0, 10.0 * (1.0 - collapse) / 0.85)
    both3 = metrics.get("both3_frac", 0.0)
    bal_b3 = min(10.0, 10.0 * both3 / 0.50)
    balanced = 0.35 * bal_blue + 0.35 * bal_col + 0.30 * bal_b3

    mfc = metrics.get("multcam_frac", 0.0)
    agr = metrics.get("agree_frac", 0.0)
    cgf = metrics.get("conflict_gray_frac", 0.0)
    consensus = 0.5 * mfc * 10 + 0.5 * (agr * 10 + (1.0 - cgf) * 5)

    flicker = sticky_keep * 5.0 + (1.0 - swing_rate) * 5.0
    composite = 0.30 * consensus + 0.35 * balanced + 0.35 * flicker
    return {
        "consensus": round(consensus, 2),
        "balanced": round(balanced, 2),
        "flickering": round(flicker, 2),
        "composite": round(composite, 2),
        "sticky_keep": round(sticky_keep, 3),
        "swing_rate": round(swing_rate, 3),
    }


def spot_row(stats: list[dict], fr: int = SPOT_FR) -> dict | None:
    for s in stats:
        if s.get("fr") == fr:
            return s
    return None


def write_ranking(results: list[dict]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    results.sort(key=lambda r: r["scores"]["composite"], reverse=True)
    (OUT / "ranking.json").write_text(json.dumps({"ts": datetime.now(timezone.utc).isoformat(), "results": results}, indent=2), encoding="utf-8")
    lines = ["# Team ID strategy grid (Match 4 90s)\n", "| Rank | ID | Strategy | SAHI | Composite | Consensus | Balanced | Flicker | fr750 n0/n1/gray |\n", "|---:|---|---|---|---:|---:|---:|---:|---|\n"]
    for i, r in enumerate(results, 1):
        sp = r.get("spot_750") or {}
        lines.append(
            f"| {i} | {r['strategy_id']} | {r['strategy_name']} | {r['sahi']} | "
            f"{r['scores']['composite']:.1f} | {r['scores']['consensus']:.1f} | "
            f"{r['scores']['balanced']:.1f} | {r['scores']['flickering']:.1f} | "
            f"{sp.get('n0','?')}/{sp.get('n1','?')}/{sp.get('gray','?')} |\n"
        )
    (OUT / "ranking.md").write_text("".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    frames = frame_ids(args.start, args.match_sec, args.stride, args.src_fps)
    vids = match3_videos(ROOT)
    sahi_keys = [s.strip() for s in args.sahi.split(",") if s.strip()]
    strategies = list_strategies(args.strategies)
    if not strategies and not args.build_cache_only:
        print("no strategies", file=sys.stderr)
        return 1

    for sahi_key in sahi_keys:
        use_sahi = sahi_key == "sahi"
        cp = cache_path(sahi_key)
        if cp.is_file() and args.skip_cache_build:
            cache = json.loads(cp.read_text(encoding="utf-8"))
            print(f"loaded cache {cp}", flush=True)
        else:
            det = LocalRFDETRDetector(
                player_checkpoint=str(ROOT / "models/people_after_100_epochs.pth"),
                ball_checkpoint=str(ROOT / "models/v12_hard_snaps/post_train/checkpoint.pth"),
                confidence_threshold=0.15,
                enhance_ball=False,
                use_sahi=use_sahi,
                use_kalman=False,
                player_nms_iou=0.30,
                ball_nms_iou=0.4,
            )
            print(f"building cache {sahi_key} n={len(frames)}", flush=True)
            cache = build_cache(frames, use_sahi, vids, det)
            cp.write_text(json.dumps(cache), encoding="utf-8")
        if args.build_cache_only:
            continue

    if args.build_cache_only:
        return 0

    all_results: list[dict] = []
    for sahi_key in sahi_keys:
        cache = json.loads(cache_path(sahi_key).read_text(encoding="utf-8"))
        for strat in strategies:
            print(f"run {strat.id} {strat.name} sahi={sahi_key}", flush=True)
            raw, extra = run_strategy(strat, cache, vids)
            stats = raw
            metrics = analyze_bidir(stats)
            scores = score_run(metrics, extra)
            all_results.append({
                "strategy_id": strat.id,
                "strategy_name": strat.name,
                "sahi": sahi_key,
                "metrics": metrics,
                "scores": scores,
                "spot_750": spot_row(stats),
            })

    write_ranking(all_results)
    winner = max(all_results, key=lambda r: r["scores"]["composite"])
    print(json.dumps(winner, indent=2))
    print(f"Wrote {OUT / 'ranking.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
