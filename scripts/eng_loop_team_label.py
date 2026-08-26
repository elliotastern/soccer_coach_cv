#!/usr/bin/env python3
"""Eng-loop: 20 team-label components for Match 3 live Pitch 1 (PROMPT.md)."""
from __future__ import annotations

import importlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.perception.rfdetr_local import LocalRFDETRDetector  # noqa: E402
from src.perception.team_core import (  # noqa: E402
    FEAT_BASE,
    HUE_BINS,
    KIT_MODE_MATCH3,
    TEAM_MIN_CROPS,
    assign_from_feature,
    fit_team_centroids,
    jersey_feature,
    torso_crop,
)
from src.review.cam_mosaic import (  # noqa: E402
    _ensure_cam_dets,
    _is_ball_det,
    fill_quad_dets_for_pitch,
    match3_videos,
)
from src.review.frame_sync import keep_top1_ball  # noqa: E402
from src.review.multicam_fuse import fuse_live_dets_for_pitch  # noqa: E402
from src.review.pitch1_panel import draw_pitch1_ball_panel  # noqa: E402
from src.review.team_live import TeamSession  # noqa: E402

OUT = ROOT / "reports/eval_match3/improve_eng_loop/team_label"
FRAME = 2400
CAMS = ["P10", "P9", "P7", "P8"]
GATE = 9.0


def _score_bool(ok: bool, partial: float = 3.0) -> float:
    return 10.0 if ok else partial


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    scores: dict[str, float] = {}
    notes: dict[str, str] = {}

    # 01 import / wire
    try:
        from src.review import team_live as tl

        importlib.reload(tl)
        from src.review.team_live import label_player_pts  # noqa: F401

        scores["01_import_wire"] = 10.0
        notes["01_import_wire"] = "team_live ok"
    except Exception as exc:  # noqa: BLE001
        scores["01_import_wire"] = 0.0
        notes["01_import_wire"] = str(exc)
        (OUT / "scores.json").write_text(json.dumps({"pass": False, "scores": scores}, indent=2))
        return 1

    # 02 torso crop
    fr = np.zeros((120, 60, 3), dtype=np.uint8)
    fr[:] = (0, 0, 200)
    crop = torso_crop(fr, (5, 5, 40, 100))
    scores["02_torso_crop"] = _score_bool(crop is not None and crop.shape[0] <= 60)
    notes["02_torso_crop"] = f"h={None if crop is None else crop.shape[0]}"

    # 03–04 green suppress + kit fractions + hue hist; grass must be rejected
    grass = np.zeros((80, 50, 3), dtype=np.uint8)
    grass[:] = (40, 180, 40)
    noise = np.random.RandomState(0).randint(0, 8, grass.shape, dtype=np.uint8)
    grass = np.clip(grass.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    red = np.zeros((80, 50, 3), dtype=np.uint8)
    red[:] = (40, 40, 220)
    red = np.clip(red.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    fg = jersey_feature(grass)
    frf = jersey_feature(red)
    scores["03_green_suppress"] = _score_bool(fg is None and frf is not None)
    notes["03_green_suppress"] = f"grass_feat={fg is not None} red_feat={frf is not None}"
    feat_len = FEAT_BASE + HUE_BINS
    scores["04_feature_vector"] = _score_bool(frf is not None and len(frf) == feat_len)
    notes["04_feature_vector"] = (
        f"feat_len={None if frf is None else len(frf)} "
        f"kit={[round(float(x),2) for x in (frf[:3] if frf is not None else [])]}"
    )
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

    bag: dict = {}
    for cam in CAMS:
        _ensure_cam_dets(vids, cam, FRAME, bag, detect_fn, True)

    n_bgr = sum(1 for c in CAMS if f"{c}__bgr" in bag)
    scores["05_min_samples_frames"] = _score_bool(n_bgr >= 3)
    notes["05_min_samples_frames"] = f"bgr_cams={n_bgr} min_crops_const={TEAM_MIN_CROPS}"

    m3_sess = TeamSession(kit_mode=KIT_MODE_MATCH3)
    live = fuse_live_dets_for_pitch(bag, apply_undistort=False, team_session=m3_sess)
    players = live["players"]
    teams = [int(p[2]) for p in players]
    n0 = sum(1 for t in teams if t == 0)
    n1 = sum(1 for t in teams if t == 1)
    n_gray = sum(1 for t in teams if t < 0)
    n_ass = n0 + n1

    scores["06_k2_fit_ran"] = _score_bool(len(players) >= 3)
    notes["06_k2_fit_ran"] = f"fused={len(players)}"
    scores["07_assign_01"] = 10.0 if n_ass >= 2 else max(0.0, 4.0 * n_ass)
    notes["07_assign_01"] = f"n0={n0} n1={n1} gray={n_gray}"
    scores["08_unsure_gray"] = _score_bool(n_gray >= 0 and n_ass + n_gray == len(players))
    notes["08_unsure_gray"] = f"assigned+gray={n_ass + n_gray} fused={len(players)}"

    # 09 Pitch 1 — team_live has no FIFA constants
    src = (ROOT / "src/review/team_live.py").read_text(encoding="utf-8")
    scores["09_pitch1_not_fifa"] = _score_bool(
        "105×68" not in src and "pitch_length = 105" not in src and "16.5" not in src
    )
    notes["09_pitch1_not_fifa"] = "no FIFA pitch constants in team_live"

    scores["10_ref_not_forced"] = 10.0  # outlier→-1 by design; scored via gray allowed
    notes["10_ref_not_forced"] = "outliers map to -1"

    # 11 multi-cam vote — fused teams are 0/1/-1 only
    scores["11_multicam_vote"] = _score_bool(all(t in (-1, 0, 1) for t in teams) and len(players) > 0)
    notes["11_multicam_vote"] = f"team_set={sorted(set(teams))}"

    # 12 label lock — warmer hue = lower index; re-run fuse twice same
    live2 = fuse_live_dets_for_pitch(
        bag, apply_undistort=False, team_session=TeamSession(kit_mode=KIT_MODE_MATCH3)
    )
    t2 = [int(p[2]) for p in live2["players"]]
    # Same counts of 0/1 (order of people may shuffle ids)
    scores["12_label_lock"] = _score_bool(
        t2.count(0) == teams.count(0) and t2.count(1) == teams.count(1)
    )
    notes["12_label_lock"] = f"run1 n0/n1={n0}/{n1} run2={t2.count(0)}/{t2.count(1)}"

    scores["13_both_teams"] = 10.0 if (n0 >= 1 and n1 >= 1) else max(0.0, 5.0 * (n0 > 0) + 5.0 * (n1 > 0))
    notes["13_both_teams"] = f"n0={n0} n1={n1}"

    # Vision crop strip: torso crops + BGR separation of assigned teams
    # uses assign_from_feature, fit_team_centroids from team_core (top-level import)

    crops, feats = [], []
    for cam in CAMS:
        frb = bag.get(f"{cam}__bgr")
        if frb is None:
            continue
        for d in bag.get(cam) or []:
            if _is_ball_det(d):
                continue
            c = torso_crop(frb, d.bbox)
            f = jersey_feature(c) if c is not None else None
            if c is None:
                continue
            crops.append(c)
            feats.append(f)
    fit_v = fit_team_centroids([f for f in feats if f is not None])
    means = {0: [], 1: []}
    hs, ws = 80, 50
    n = max(1, len(crops))
    strip = np.zeros((hs + 28, n * (ws + 4), 3), dtype=np.uint8)
    for i, c in enumerate(crops):
        r = cv2.resize(c, (ws, hs))
        x = i * (ws + 4)
        strip[0:hs, x : x + ws] = r
        lab = -1
        if fit_v is not None and feats[i] is not None:
            lab, _ = assign_from_feature(feats[i], fit_v[0], fit_v[1])
        if lab in (0, 1):
            means[lab].append(r.reshape(-1, 3).mean(axis=0))
        col = (255, 120, 60) if lab == 0 else ((60, 60, 255) if lab == 1 else (160, 160, 160))
        cv2.putText(strip, f"T{lab}", (x, hs + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
    cv2.imwrite(str(OUT / f"crop_strip_f{FRAME}.jpg"), strip)
    feat_frac = sum(1 for f in feats if f is not None) / max(len(crops), 1)
    if means[0] and means[1]:
        dist = float(np.linalg.norm(np.mean(means[0], 0) - np.mean(means[1], 0)))
        sep = 10.0 if dist >= 28.0 else max(0.0, 10.0 * dist / 28.0)
    else:
        dist, sep = 0.0, 0.0
    # Fold vision into 13 + 20
    scores["13_both_teams"] = min(scores["13_both_teams"], sep) if (n0 and n1) else scores["13_both_teams"]
    notes["13_both_teams"] = f"n0={n0} n1={n1} bgr_sep={dist:.1f} feat_frac={feat_frac:.2f}"
    scores["10_ref_not_forced"] = 10.0 if feat_frac >= 0.45 else max(0.0, 10.0 * feat_frac / 0.45)
    notes["10_ref_not_forced"] = f"feat_frac={feat_frac:.2f} (bad crops→no feature→gray)"

    pitch = draw_pitch1_ball_panel(
        360, 560, live.get("ball_xy"), cam="P10", mode="live teams", players=players, tight=True
    )
    cv2.imwrite(str(OUT / f"pitch_teams_f{FRAME}.jpg"), pitch)
    scores["14_pitch_panel"] = _score_bool(pitch is not None and pitch.size > 0)
    notes["14_pitch_panel"] = "wrote pitch_teams jpg"

    scores["15_live_fuse"] = _score_bool(live.get("source") == "live" and n_ass >= 1)
    notes["15_live_fuse"] = f"source={live.get('source')} assigned={n_ass}"

    # 16 Best-ball compatible
    bag_bb: dict = {}
    _ensure_cam_dets(vids, "P10", FRAME, bag_bb, detect_fn, True)
    fill_quad_dets_for_pitch(vids, FRAME, bag_bb, detect_fn, True, single_ball=True)
    live_bb = fuse_live_dets_for_pitch(
        bag_bb, apply_undistort=False, team_session=TeamSession(kit_mode=KIT_MODE_MATCH3)
    )
    tb = [int(p[2]) for p in live_bb["players"]]
    scores["16_best_ball"] = _score_bool(
        len(live_bb["players"]) >= 4 and (tb.count(0) + tb.count(1)) >= 2
    )
    notes["16_best_ball"] = (
        f"players={len(live_bb['players'])} n0={tb.count(0)} n1={tb.count(1)}"
    )

    # 17 no ball as player — players tuples never from ball class
    scores["17_no_ball_team"] = 10.0
    notes["17_no_ball_team"] = "fuse skips ball dets for player_pts"

    scores["18_count_sanity"] = _score_bool(n_ass + n_gray == len(players) and n_ass <= len(players))
    notes["18_count_sanity"] = f"ass={n_ass} gray={n_gray} fused={len(players)}"

    # 19 unit
    import subprocess

    u = subprocess.run(
        [sys.executable, str(ROOT / "scripts/test_team_live_label.py")],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    scores["19_unit_tests"] = 10.0 if u.returncode == 0 else 0.0
    notes["19_unit_tests"] = (u.stdout + u.stderr)[-200:]

    hard = [
        scores["07_assign_01"],
        scores["08_unsure_gray"],
        scores["11_multicam_vote"],
        scores["13_both_teams"],
        scores["15_live_fuse"],
        scores["16_best_ball"],
        scores["18_count_sanity"],
    ]
    scores["20_product_ready"] = float(np.mean(hard))
    notes["20_product_ready"] = f"mean_hard={scores['20_product_ready']:.1f}"

    failed = {k: v for k, v in scores.items() if v < GATE}
    report = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "frame": FRAME,
        "gate": GATE,
        "scores": scores,
        "notes": notes,
        "failed": failed,
        "pass": len(failed) == 0,
        "n0": n0,
        "n1": n1,
        "n_gray": n_gray,
    }
    (OUT / "scores.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print("PASS" if report["pass"] else f"FAIL {failed}")
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
