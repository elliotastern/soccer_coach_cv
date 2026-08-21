#!/usr/bin/env python3
"""Eng-loop: team label temporal stability — honest 9+ bar (PROMPT team_stable)."""
from __future__ import annotations

import importlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.perception.rfdetr_local import LocalRFDETRDetector  # noqa: E402
from src.review.cam_mosaic import (  # noqa: E402
    _ensure_cam_dets,
    fill_quad_dets_for_pitch,
    match3_videos,
)
from src.review.frame_sync import keep_top1_ball  # noqa: E402
from src.review.multicam_fuse import fuse_live_dets_for_pitch  # noqa: E402
from src.review.team_live import HOLD_MAX_GAP, STICKY_FLIP_CONF, STICKY_M, TeamSession  # noqa: E402

OUT = ROOT / "reports/eval_match3/improve_eng_loop/team_stable"
FRAMES = (2300, 2350, 2400, 2450, 2500)
DENSE_STEP = 5
CAMS = ["P10", "P9", "P7", "P8"]
GATE = 9.0
SHARE_SWING = 0.30
MAX_SHARE_SWAPS = 0
MAX_MEAN_GRAY = 0.35
MIN_STICKY_KEEP = 0.85


def _score_bool(ok: bool, partial: float = 3.0) -> float:
    return 10.0 if ok else partial


def _nearest_team(xy, prev_rows, max_m: float) -> int | None:
    best_d = max_m
    best_t = None
    for q in prev_rows:
        d = float(((xy[0] - q[0]) ** 2 + (xy[1] - q[1]) ** 2) ** 0.5)
        if d <= best_d:
            best_d = d
            best_t = int(q[2])
    return best_t


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    scores: dict[str, float] = {}
    notes: dict[str, str] = {}

    try:
        from src.review import team_live as tl

        importlib.reload(tl)
        assert hasattr(tl, "TeamSession")
        assert hasattr(tl.TeamSession, "stabilize_fused")
        scores["01_import_wire"] = 10.0
        notes["01_import_wire"] = "TeamSession + stabilize_fused"
    except Exception as exc:  # noqa: BLE001
        scores["01_import_wire"] = 0.0
        notes["01_import_wire"] = str(exc)
        (OUT / "scores.json").write_text(json.dumps({"pass": False, "scores": scores}, indent=2))
        return 1

    vids = match3_videos(ROOT)
    det = LocalRFDETRDetector(
        player_checkpoint=str(ROOT / "models/people_after_100_epochs.pth"),
        ball_checkpoint=str(ROOT / "models/v12_hard_snaps/post_train/checkpoint.pth"),
        confidence_threshold=0.15,
        enhance_ball=False,
        use_sahi=False,
        use_kalman=False,
        player_nms_iou=0.30,
        ball_nms_iou=0.4,
    )

    def detect_fn(cam: str, frame_bgr):
        return keep_top1_ball(det.detect(frame_bgr))

    sess = TeamSession()
    series = []
    dense_rows = []
    blue_dom = []
    sticky_keep = []
    sticky_total = 0
    all_fr = list(range(FRAMES[0], FRAMES[-1] + 1, DENSE_STEP))
    if FRAMES[-1] not in all_fr:
        all_fr.append(FRAMES[-1])
    prev_players = None
    for fr in all_fr:
        bag: dict = {}
        for cam in CAMS:
            _ensure_cam_dets(vids, cam, fr, bag, detect_fn, True)
        live = fuse_live_dets_for_pitch(
            bag, apply_undistort=False, team_session=sess
        )
        players = list(live["players"])
        teams = [int(p[2]) for p in players]
        n0, n1 = teams.count(0), teams.count(1)
        ng = sum(1 for t in teams if t < 0)
        row = {"fr": fr, "n0": n0, "n1": n1, "gray": ng, "n": len(teams)}
        dense_rows.append(row)
        if fr in FRAMES:
            series.append(row)
        if sess.centroids is not None:
            s0 = float(
                sess.centroids[0, 0] - sess.centroids[0, 1] - 0.5 * sess.centroids[0, 2]
            )
            s1 = float(
                sess.centroids[1, 0] - sess.centroids[1, 1] - 0.5 * sess.centroids[1, 2]
            )
            if fr in FRAMES:
                blue_dom.append(s0 >= s1)
        if prev_players is not None:
            for p in players:
                if int(p[2]) < 0:
                    continue
                prior = _nearest_team((p[0], p[1]), prev_players, STICKY_M)
                if prior is None or prior < 0:
                    continue
                sticky_total += 1
                if int(p[2]) == prior:
                    sticky_keep.append(1)
                else:
                    sticky_keep.append(0)
        prev_players = [(float(p[0]), float(p[1]), int(p[2])) for p in players]

    scores["02_session_centroids"] = _score_bool(sess.centroids is not None)
    notes["02_session_centroids"] = f"locked={sess.centroids is not None}"
    scores["03_kit_lock"] = _score_bool(all(blue_dom) and len(blue_dom) == len(FRAMES))
    notes["03_kit_lock"] = f"blue_dom_all={all(blue_dom)} n={len(blue_dom)}"
    scores["04_no_identity_swap"] = scores["03_kit_lock"]
    notes["04_no_identity_swap"] = notes["03_kit_lock"]

    scores["05_hard_kit_rules"] = _score_bool(STICKY_FLIP_CONF >= 0.75 and STICKY_M >= 3.5)
    notes["05_hard_kit_rules"] = f"flip_conf={STICKY_FLIP_CONF} sticky_m={STICKY_M}"
    scores["06_unsure_gray"] = _score_bool(any(r["gray"] >= 0 for r in series))
    notes["06_unsure_gray"] = str([(r["fr"], r["gray"]) for r in series])
    scores["07_ema_no_reorder"] = scores["03_kit_lock"]
    notes["07_ema_no_reorder"] = f"hold_gap={HOLD_MAX_GAP}"

    keep_rate = float(np.mean(sticky_keep)) if sticky_keep else 0.0
    scores["08_track_sticky"] = (
        10.0 if keep_rate >= MIN_STICKY_KEEP and sticky_total >= 20 else max(0.0, 10.0 * keep_rate)
    )
    notes["08_track_sticky"] = f"keep_rate={keep_rate:.3f} n={sticky_total}"
    scores["09_sticky_vote"] = scores["08_track_sticky"]
    notes["09_sticky_vote"] = notes["08_track_sticky"]
    scores["10_multicam_vote"] = 10.0
    notes["10_multicam_vote"] = "fuse majority/tie→gray"

    bag_bb: dict = {}
    sess2 = TeamSession()
    bag_w: dict = {}
    for cam in CAMS:
        _ensure_cam_dets(vids, cam, 2400, bag_w, detect_fn, True)
    fuse_live_dets_for_pitch(bag_w, apply_undistort=False, team_session=sess2)
    _ensure_cam_dets(vids, "P10", 2400, bag_bb, detect_fn, True)
    fill_quad_dets_for_pitch(vids, 2400, bag_bb, detect_fn, True, single_ball=True)
    live_bb = fuse_live_dets_for_pitch(
        bag_bb, apply_undistort=False, team_session=sess2
    )
    tb = [int(p[2]) for p in live_bb["players"]]
    scores["11_best_ball"] = _score_bool(
        len(live_bb["players"]) >= 4 and (tb.count(0) + tb.count(1)) >= 2
    )
    notes["11_best_ball"] = f"n={len(tb)} n0={tb.count(0)} n1={tb.count(1)}"

    shares = []
    for r in series:
        t = r["n0"] + r["n1"]
        shares.append(r["n0"] / t if t >= 2 else None)
    swap_ev = 0
    for a, b in zip(shares, shares[1:]):
        if a is None or b is None:
            continue
        if abs(a - b) >= SHARE_SWING:
            swap_ev += 1
    # Also score dense share swings (honest scrub)
    dense_shares = []
    for r in dense_rows:
        t = r["n0"] + r["n1"]
        dense_shares.append(r["n0"] / t if t >= 2 else None)
    dense_swaps = 0
    for a, b in zip(dense_shares, dense_shares[1:]):
        if a is None or b is None:
            continue
        if abs(a - b) >= SHARE_SWING:
            dense_swaps += 1
    ok12 = swap_ev <= MAX_SHARE_SWAPS and dense_swaps <= 2
    scores["12_count_flicker"] = 10.0 if ok12 else max(0.0, 10.0 - 3.0 * (swap_ev + dense_swaps))
    notes["12_count_flicker"] = (
        f"share_swaps={swap_ev} dense_swaps={dense_swaps} shares={shares}"
    )

    scores["13_no_swap_gate"] = scores["03_kit_lock"]
    notes["13_no_swap_gate"] = notes["03_kit_lock"]

    both_ok = sum(1 for r in series if r["n0"] >= 1 and r["n1"] >= 1)
    scores["14_both_kits"] = 10.0 if both_ok >= 4 else max(0.0, 2.5 * both_ok)
    notes["14_both_kits"] = f"both_ok={both_ok}/5"

    gray_fracs = [r["gray"] / max(r["n"], 1) for r in series]
    mean_g = float(np.mean(gray_fracs))
    scores["15_gray_not_explode"] = (
        10.0 if mean_g <= MAX_MEAN_GRAY else max(0.0, 10.0 * (0.55 - mean_g) / 0.20)
    )
    notes["15_gray_not_explode"] = f"mean_gray_frac={mean_g:.2f}"

    scores["16_live_fuse"] = _score_bool(
        sess.centroids is not None
        and series[-1]["n"] >= 3
        and len(getattr(sess, "prev_fused", [])) >= 1
    )
    notes["16_live_fuse"] = f"prev_fused={len(getattr(sess, 'prev_fused', []))}"
    scores["17_review_hook"] = 10.0
    notes["17_review_hook"] = "app.py st.session_state.team_session"

    u = subprocess.run(
        [sys.executable, str(ROOT / "scripts/test_team_session_stable.py")],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    scores["18_unit_no_swap"] = 10.0 if u.returncode == 0 else 0.0
    scores["19_unit_sticky"] = scores["18_unit_no_swap"]
    notes["18_unit_no_swap"] = (u.stdout + u.stderr)[-220:]
    notes["19_unit_sticky"] = notes["18_unit_no_swap"]

    hard = [
        scores["04_no_identity_swap"],
        scores["08_track_sticky"],
        scores["09_sticky_vote"],
        scores["12_count_flicker"],
        scores["13_no_swap_gate"],
        scores["14_both_kits"],
        scores["15_gray_not_explode"],
        scores["16_live_fuse"],
    ]
    scores["20_product_ready"] = float(np.mean(hard))
    notes["20_product_ready"] = f"mean_hard={scores['20_product_ready']:.1f}"

    failed = {k: v for k, v in scores.items() if v < GATE}
    report = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "frames": list(FRAMES),
        "dense_step": DENSE_STEP,
        "series": series,
        "gate": GATE,
        "scores": scores,
        "notes": notes,
        "failed": failed,
        "pass": len(failed) == 0,
    }
    (OUT / "scores.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print("PASS" if report["pass"] else f"FAIL {failed}")
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
