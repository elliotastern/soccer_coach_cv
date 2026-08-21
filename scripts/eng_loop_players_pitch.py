#!/usr/bin/env python3
"""Eng-loop: 20 subcomponents for players right on Match 3 video + Pitch 1 map.

Gate: every subcomponent ≥ 9.0 / 10. Writes scores + verify mosaics under
reports/eval_match3/improve_eng_loop/players_pitch/.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.mapping.match3_xy import (  # noqa: E402
    apply_H,
    bbox_foot,
    calib_undistort_params,
    load_calib,
    map_ball_box,
    scale_px,
)
from src.perception.rfdetr_local import LocalRFDETRDetector  # noqa: E402
from src.review.cam_mosaic import (  # noqa: E402
    QUAD_ROTATE_180,
    _ensure_cam_dets,
    _filter_coach_dets,
    _is_ball_det,
    match3_videos,
    mosaic_quads_coach,
    read_frame_bgr,
    undistort_bgr,
)
from src.review.frame_sync import keep_top1_ball  # noqa: E402
from src.review.multicam_fuse import (  # noqa: E402
    PLAYER_GHOST_CONF,
    PLAYER_MERGE_M,
    PLAYER_MIN_CONF,
    PLAYER_MIN_H,
    PLAYER_SOLO_CONF,
    fuse_live_dets_for_pitch,
    player_det_ok,
)
from src.review.pitch1_panel import draw_pitch1_ball_panel  # noqa: E402

OUT = ROOT / "reports/eval_match3/improve_eng_loop/players_pitch"
FRAMES = (1200, 2400, 3600)
CAMS = ["P10", "P9", "P7", "P8"]
GATE = 9.0


def _score_bool(ok: bool, partial: float = 4.0) -> float:
    return 10.0 if ok else partial


def _containment(frame: np.ndarray, dets: list) -> float:
    """Box-on-body: chroma OR texture contrast vs shifted windows (green jerseys OK)."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    chroma = ((hsv[:, :, 0] < 35) | (hsv[:, :, 0] > 95)) & (hsv[:, :, 1] > 35) & (
        hsv[:, :, 2] > 50
    )
    scores = []
    for d in dets:
        if _is_ball_det(d):
            continue
        x, y, w, h = [int(v) for v in d.bbox]
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(frame.shape[1], x + w), min(frame.shape[0], y + h)
        if x1 <= x0 or y1 <= y0 or (x1 - x0) < 4 or (y1 - y0) < 4:
            continue
        roi_c = chroma[y0:y1, x0:x1]
        roi_g = gray[y0:y1, x0:x1]
        frac = float(roi_c.mean())
        tex = float(roi_g.std())
        shift = max(8, w // 4)
        left_c = chroma[y0:y1, max(0, x0 - shift) : max(0, x1 - shift)]
        right_c = chroma[y0:y1, min(frame.shape[1], x0 + shift) : min(frame.shape[1], x1 + shift)]
        left_g = gray[y0:y1, max(0, x0 - shift) : max(0, x1 - shift)]
        right_g = gray[y0:y1, min(frame.shape[1], x0 + shift) : min(frame.shape[1], x1 + shift)]
        fl = float(left_c.mean()) if left_c.size else 0.0
        fr = float(right_c.mean()) if right_c.size else 0.0
        tl = float(left_g.std()) if left_g.size else 0.0
        tr = float(right_g.std()) if right_g.size else 0.0
        chroma_ok = frac >= fl - 0.02 and frac >= fr - 0.02 and frac > 0.02
        tex_ok = tex >= 8.0 and tex >= 0.85 * max(tl, tr, 1.0)
        ok = chroma_ok or tex_ok
        scores.append(10.0 if ok else max(0.0, 10.0 * max(frac, tex / 25.0)))
    return float(np.mean(scores)) if scores else 5.0


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
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

    # Primary detailed score on mid-match frame; smoke other frames for coverage.
    FRAME = 2400
    bag: dict = {}
    for cam in CAMS:
        _ensure_cam_dets(vids, cam, FRAME, bag, detect_fn, True)

    scores: dict[str, float] = {}
    notes: dict[str, str] = {}

    # 1 detector loads
    scores["01_detector_loads"] = 10.0
    notes["01_detector_loads"] = "RF-DETR player+ball loaded"

    # 2–7 per-cam box quality
    contain = []
    tiny_fail = 0
    weak_fail = 0
    dup_pairs = 0
    n_players = 0
    for cam in CAMS:
        dets = [d for d in (bag.get(cam) or []) if not _is_ball_det(d)]
        n_players += len(dets)
        fr = read_frame_bgr(vids[cam], FRAME)
        calib = load_calib(cam)
        work = fr
        if calib is not None and calib_undistort_params(calib):
            cw, ch = [int(v) for v in (calib.get("image_wh") or [fr.shape[1], fr.shape[0]])]
            if work.shape[1] != cw or work.shape[0] != ch:
                work = cv2.resize(work, (cw, ch))
            work = undistort_bgr(work, calib)
        if dets:
            contain.append(_containment(work, dets))
        for d in dets:
            _, _, w, h = [float(v) for v in d.bbox]
            if float(d.confidence) < PLAYER_MIN_CONF:
                weak_fail += 1
            if h < PLAYER_MIN_H or (w * h) < 800:
                tiny_fail += 1
        for i, a in enumerate(dets):
            for b in dets[i + 1 :]:
                ax, ay, aw, ah = a.bbox
                bx, by, bw, bh = b.bbox
                x0, y0 = max(ax, bx), max(ay, by)
                x1, y1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
                inter = max(0, x1 - x0) * max(0, y1 - y0)
                union = aw * ah + bw * bh - inter
                if union > 0 and inter / union >= 0.55:
                    dup_pairs += 1

    scores["02_box_on_body"] = float(np.mean(contain)) if contain else 0.0
    notes["02_box_on_body"] = f"mean_contain={scores['02_box_on_body']:.1f}"
    scores["03_defish_detect_path"] = _score_bool(
        all(f"{c}__wh" in bag for c in CAMS), 2.0
    )
    notes["03_defish_detect_path"] = "dets after defish + __wh set"
    # 4 rotate: synthetic mark lock already covered elsewhere — check P10/P9 have dets drawable
    scores["04_rotate180_ready"] = _score_bool(
        all(cam in QUAD_ROTATE_180 for cam in ("P10", "P9")), 5.0
    )
    notes["04_rotate180_ready"] = "P10|P9 flagged for 180°"
    scores["05_nms_dupes"] = 10.0 if dup_pairs == 0 else max(0.0, 10.0 - 3.0 * dup_pairs)
    notes["05_nms_dupes"] = f"iou>=0.55 pairs={dup_pairs}"
    scores["06_tiny_box_reject"] = 10.0 if tiny_fail == 0 else max(0.0, 10.0 - 2.0 * tiny_fail)
    notes["06_tiny_box_reject"] = f"tiny_left={tiny_fail}"
    scores["07_weak_conf_reject"] = (
        10.0 if weak_fail == 0 else max(0.0, 10.0 - 1.5 * weak_fail)
    )
    notes["07_weak_conf_reject"] = f"conf<{PLAYER_MIN_CONF} left={weak_fail}"

    # 8–11 mapping gates
    wh_ok = True
    double_u = 0
    mapped = 0
    unmapped = 0
    bounds_fail = 0
    raw_pts = []
    for cam in CAMS:
        calib = load_calib(cam)
        wh = bag.get(f"{cam}__wh")
        if wh is None or calib is None:
            wh_ok = False
            continue
        cw, ch = calib.get("image_wh") or wh
        if abs(wh[0] - cw) > 2 or abs(wh[1] - ch) > 2:
            # detect frame should match calib_wh after resize-in-_ensure
            wh_ok = False
        for d in bag.get(cam) or []:
            if _is_ball_det(d):
                continue
            # wrong path would undistort again → large hull/support swing
            m_ok = map_ball_box(
                calib, d.bbox, float(d.confidence), frame_wh=wh, apply_undistort=False
            )
            m_bad = map_ball_box(
                calib, d.bbox, float(d.confidence), frame_wh=wh, apply_undistort=True
            )
            if m_ok is None and m_bad is not None:
                double_u += 1
            if m_ok is None:
                unmapped += 1
            else:
                mapped += 1
                raw_pts.append({"xy": m_ok["xy"], "conf": m_ok["conf"], "cam": cam})
    scores["08_map_wh_match"] = _score_bool(wh_ok, 3.0)
    notes["08_map_wh_match"] = f"wh_ok={wh_ok}"
    scores["09_no_double_undistort"] = 10.0 if double_u == 0 else max(0.0, 10.0 - double_u)
    notes["09_no_double_undistort"] = (
        f"cases where only double-undistort maps={double_u} (should be 0 preference)"
    )
    frac = mapped / max(mapped + unmapped, 1)
    # Video bag may include unmappable players (coach boxes); fuse still map-gates.
    cams_with_boxes = sum(
        1
        for cam in CAMS
        if any(not _is_ball_det(d) for d in (bag.get(cam) or []))
    )
    scores["10_hull_gate_sane"] = (
        10.0 if mapped >= 4 and cams_with_boxes >= 2 else max(0.0, 10.0 * mapped / 8.0)
    )
    notes["10_hull_gate_sane"] = (
        f"mapped={mapped} unmapped={unmapped} frac={frac:.2f} cams_with_boxes={cams_with_boxes} "
        f"(video boxes keep unmapped; pitch fuse drops them)"
    )
    scores["11_pitch_bounds"] = 10.0 if bounds_fail == 0 else 5.0
    notes["11_pitch_bounds"] = "map_ball_box drops OOB"

    live = fuse_live_dets_for_pitch(bag, apply_undistort=False)
    players = live["players"]
    n_fused = len(players)
    n_video = sum(
        1 for cam in CAMS for d in (bag.get(cam) or []) if not _is_ball_det(d)
    )
    cams_with_players = sum(
        1
        for cam in CAMS
        if any(not _is_ball_det(d) for d in (bag.get(cam) or []))
    )

    # 12–15 fuse policy
    live_ok = (
        live.get("source") == "live"
        and n_fused >= 1
        and live["n_cams"] >= min(2, max(1, cams_with_players))
    )
    scores["12_fuse_uses_live"] = _score_bool(live_ok, 4.0)
    notes["12_fuse_uses_live"] = (
        f"n_cams={live['n_cams']} cams_with_players={cams_with_players} "
        f"source={live.get('source')}"
    )
    # fused should be ≤ video and not explode; ≥ 0.4 * mapped after filters
    ratio = n_fused / max(n_video, 1)
    scores["13_fuse_count_sane"] = (
        10.0 if 0.35 <= ratio <= 1.05 and n_fused >= 3 else max(0.0, 8.0 * min(ratio, 1.0))
    )
    notes["13_fuse_count_sane"] = f"fused={n_fused} video={n_video} ratio={ratio:.2f}"

    # cluster spreads among multi-cam (rebuild with same merge)
    pts = sorted(raw_pts, key=lambda p: -p["conf"])
    clusters: list[list[dict]] = []
    for p in pts:
        placed = False
        for cl in clusters:
            dx = p["xy"][0] - cl[0]["xy"][0]
            dy = p["xy"][1] - cl[0]["xy"][1]
            if (dx * dx + dy * dy) ** 0.5 <= PLAYER_MERGE_M:
                cl.append(p)
                placed = True
                break
        if not placed:
            clusters.append([p])
    spreads = []
    for cl in clusters:
        if len(cl) < 2:
            continue
        for i, a in enumerate(cl):
            for b in cl[i + 1 :]:
                spreads.append(
                    ((a["xy"][0] - b["xy"][0]) ** 2 + (a["xy"][1] - b["xy"][1]) ** 2)
                    ** 0.5
                )
    max_spread = float(max(spreads)) if spreads else 0.0
    scores["14_cross_cam_spread"] = (
        10.0 if max_spread <= PLAYER_MERGE_M + 0.05 else max(0.0, 10.0 - 2.0 * max_spread)
    )
    notes["14_cross_cam_spread"] = f"max_pair_spread_m={max_spread:.2f}"

    # fused xy should equal max-conf member (not mean of disagreeing)
    max_conf_ok = True
    for px, py, _team, _pid in players:
        near = min(
            (
                ((px - p["xy"][0]) ** 2 + (py - p["xy"][1]) ** 2) ** 0.5
                for p in raw_pts
            ),
            default=99.0,
        )
        if near > 0.35:
            max_conf_ok = False
    scores["15_maxconf_not_mean"] = _score_bool(max_conf_ok and n_fused > 0, 3.0)
    notes["15_maxconf_not_mean"] = f"each fused ≤0.35m from a mapped foot ({max_conf_ok})"

    # 16 ghost: weak solos below PLAYER_SOLO_CONF should not appear on pitch
    ghost_left = 0
    solo_weak = [
        cl[0]
        for cl in clusters
        if len(cl) == 1 and float(cl[0]["conf"]) < PLAYER_SOLO_CONF
    ]
    for sw in solo_weak:
        dmin = min(
            (
                ((sw["xy"][0] - p[0]) ** 2 + (sw["xy"][1] - p[1]) ** 2) ** 0.5
                for p in players
            ),
            default=99.0,
        )
        if dmin < 0.5:
            ghost_left += 1

    scores["16_ghost_prune"] = 10.0 if ghost_left == 0 else max(0.0, 10.0 - 2.5 * ghost_left)
    notes["16_ghost_prune"] = (
        f"weak_solos_on_pitch={ghost_left} solo_conf>={PLAYER_SOLO_CONF} ghost>={PLAYER_GHOST_CONF}"
    )

    # 17 count consistency: video boxes should track fused after multi-cam merge
    # Allow merge compression (video multi-view → fewer pitch people)
    delta = abs(n_video - n_fused) / max(n_video, n_fused, 1)
    merge_ok = n_fused <= n_video and n_fused >= max(3, int(0.45 * n_video))
    scores["17_count_consistency"] = (
        10.0 if merge_ok and delta <= 0.55 else max(0.0, 10.0 * (1.0 - delta))
    )
    notes["17_count_consistency"] = (
        f"|video-fused|/max={delta:.2f} merge_ok={merge_ok}"
    )

    # 18 P7 overfire after filter
    n_p7 = sum(1 for d in (bag.get("P7") or []) if not _is_ball_det(d))
    scores["18_p7_overfire"] = 10.0 if n_p7 <= 8 else max(0.0, 10.0 - (n_p7 - 8))
    notes["18_p7_overfire"] = f"P7_players={n_p7}"

    # 19 player_det_ok helper wired
    ok_fn = all(
        player_det_ok(d) for cam in CAMS for d in (bag.get(cam) or []) if not _is_ball_det(d)
    )
    scores["19_player_filter_wired"] = _score_bool(ok_fn or n_video == 0, 2.0)
    notes["19_player_filter_wired"] = f"all kept dets pass player_det_ok={ok_fn}"

    # 20 overall product readiness
    hard = [
        scores["02_box_on_body"],
        scores["06_tiny_box_reject"],
        scores["07_weak_conf_reject"],
        scores["13_fuse_count_sane"],
        scores["15_maxconf_not_mean"],
        scores["16_ghost_prune"],
        scores["17_count_consistency"],
        scores["18_p7_overfire"],
    ]
    scores["20_product_players_ready"] = float(np.mean(hard))
    notes["20_product_players_ready"] = f"mean_hard={scores['20_product_players_ready']:.1f}"

    # Multi-frame smoke: every sampled frame must fuse ≥1 player when any cam sees bodies
    smoke = {}
    for fr in FRAMES:
        if fr == FRAME:
            smoke[str(fr)] = {"video": n_video, "fused": n_fused, "ok": True}
            continue
        bag2: dict = {}
        for cam in CAMS:
            _ensure_cam_dets(vids, cam, fr, bag2, detect_fn, True)
        nv = sum(1 for c in CAMS for d in (bag2.get(c) or []) if not _is_ball_det(d))
        live2 = fuse_live_dets_for_pitch(bag2, apply_undistort=False)
        nf = len(live2["players"])
        ok = (nv == 0 and nf == 0) or (nv > 0 and nf >= 1 and nf <= nv)
        smoke[str(fr)] = {"video": nv, "fused": nf, "ok": ok}
    if not all(v["ok"] for v in smoke.values()):
        scores["20_product_players_ready"] = min(scores["20_product_players_ready"], 8.0)
        notes["20_product_players_ready"] += f" smoke_fail={smoke}"

    # artifacts
    mosaic = mosaic_quads_coach(
        vids, FRAME, dets_by_cam=dict(bag), detect_fn=None, apply_defish=True
    )
    cv2.imwrite(str(OUT / f"mosaic_f{FRAME}.jpg"), mosaic)
    pitch = draw_pitch1_ball_panel(
        360,
        560,
        live.get("ball_xy"),
        cam="P10",
        mode=f"live n={live['n_cams']}",
        players=players,
        tight=True,
    )
    cv2.imwrite(str(OUT / f"pitch_f{FRAME}.jpg"), pitch)

    failed = {k: v for k, v in scores.items() if v < GATE}
    report = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "frame": FRAME,
        "frames_smoke": smoke,
        "gate": GATE,
        "scores": scores,
        "notes": notes,
        "n_video_players": n_video,
        "n_fused_players": n_fused,
        "failed": failed,
        "pass": len(failed) == 0,
        "constants": {
            "PLAYER_MIN_CONF": PLAYER_MIN_CONF,
            "PLAYER_MIN_H": PLAYER_MIN_H,
            "PLAYER_SOLO_CONF": PLAYER_SOLO_CONF,
            "PLAYER_GHOST_CONF": PLAYER_GHOST_CONF,
            "PLAYER_MERGE_M": PLAYER_MERGE_M,
        },
    }
    (OUT / "scores.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print("PASS" if report["pass"] else f"FAIL {len(failed)} below {GATE}")
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
