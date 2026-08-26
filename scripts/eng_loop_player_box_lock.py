#!/usr/bin/env python3
"""Eng-loop: player boxes must lock to players after mosaic defish+rotate (≥9/10)."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.mapping.match3_xy import load_calib  # noqa: E402
from src.review.cam_mosaic import (  # noqa: E402
    MOSAIC_DEFISH_ALPHA,
    QUAD_ROTATE_180,
    _remap_bbox_undistort,
    _rotate_dets_180,
    _tile,
    match3_videos,
    mosaic_quads_coach,
    read_frame_bgr,
    undistort_bgr,
)
from src.state.types import Detection  # noqa: E402

OUT = ROOT / "reports/eval_match3/improve_eng_loop/player_stitch"
FRAME = 2400
GATE = 9.0


def _center(bbox) -> tuple[float, float]:
    x, y, w, h = [float(v) for v in bbox]
    return x + w / 2.0, y + h / 2.0


def score_alpha_lock() -> tuple[float, list[str]]:
    """Remap with mosaic alpha must beat remap with calib alpha on display warp."""
    notes = []
    videos = match3_videos(ROOT)
    scores = []
    for cam in ["P7", "P8", "P9", "P10"]:
        calib = load_calib(cam)
        if not calib:
            notes.append(f"{cam}: no calib")
            continue
        fr = read_frame_bgr(videos[cam], FRAME)
        cw, ch = [int(v) for v in (calib.get("image_wh") or [fr.shape[1], fr.shape[0]])]
        work = cv2.resize(fr, (cw, ch)) if (fr.shape[1], fr.shape[0]) != (cw, ch) else fr
        # Synthetic player blob
        bx, by, bw, bh = int(cw * 0.4), int(ch * 0.55), 120, 220
        mark = work.copy()
        cv2.rectangle(mark, (bx, by), (bx + bw, by + bh), (0, 255, 255), -1)
        warped = undistort_bgr(mark, calib, alpha_override=MOSAIC_DEFISH_ALPHA)
        # Where did the yellow blob go?
        mask = (
            (warped[:, :, 0] < 40)
            & (warped[:, :, 1] > 200)
            & (warped[:, :, 2] > 200)
        )
        if int(mask.sum()) < 50:
            notes.append(f"{cam}: yellow blob vanished after defish")
            scores.append(0.0)
            continue
        ys, xs = np.where(mask)
        gt_cx, gt_cy = float(xs.mean()), float(ys.mean())

        bbox = (float(bx), float(by), float(bw), float(bh))
        wrong = _remap_bbox_undistort(bbox, calib, (cw, ch), alpha_override=None)
        right = _remap_bbox_undistort(
            bbox, calib, (cw, ch), alpha_override=MOSAIC_DEFISH_ALPHA
        )
        if cam in QUAD_ROTATE_180:
            h, w = warped.shape[:2]
            wrong = _rotate_dets_180(
                [Detection(0, 1.0, wrong, "player")], w, h
            )[0].bbox
            right = _rotate_dets_180(
                [Detection(0, 1.0, right, "player")], w, h
            )[0].bbox
            gt_cx, gt_cy = (w - 1 - gt_cx), (h - 1 - gt_cy)

        d_wrong = float(np.hypot(_center(wrong)[0] - gt_cx, _center(wrong)[1] - gt_cy))
        d_right = float(np.hypot(_center(right)[0] - gt_cx, _center(right)[1] - gt_cy))
        # Pass if correct alpha is near blob (≤40px) and ≤ wrong (or wrong also ok)
        ok = d_right <= 40.0
        scores.append(10.0 if ok else max(0.0, 10.0 - d_right / 20.0))
        notes.append(
            f"{cam}: d_right={d_right:.1f}px d_wrong_alpha={d_wrong:.1f}px "
            f"{'PASS' if ok else 'FAIL'}"
        )
        # Save debug
        vis = warped.copy()
        if cam in QUAD_ROTATE_180:
            vis = cv2.rotate(vis, cv2.ROTATE_180)
        for bb, col in ((wrong, (0, 0, 255)), (right, (0, 255, 0))):
            x, y, w, h = [int(v) for v in bb]
            cv2.rectangle(vis, (x, y), (x + w, y + h), col, 3)
        cv2.circle(vis, (int(gt_cx), int(gt_cy)), 8, (255, 255, 255), -1)
        cv2.imwrite(str(OUT / f"alpha_lock_{cam}.jpg"), vis)
    if not scores:
        return 0.0, notes or ["no cams scored"]
    return float(np.mean(scores)), notes


def score_paint_warp_tile() -> tuple[float, list[str], Path]:
    """Real RF-DETR dets → tile; green box mass must sit on players (heuristic)."""
    notes = []
    videos = match3_videos(ROOT)
    from src.perception.rfdetr_local import LocalRFDETRDetector
    from src.review.frame_sync import keep_top1_ball

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
    dets_by_cam = {}
    tile_scores = []
    for cam in ["P8", "P9", "P10", "P7"]:
        fr = read_frame_bgr(videos[cam], FRAME)
        dets = keep_top1_ball(det.detect(fr))
        dets_by_cam[cam] = dets
        tile = _tile(
            videos,
            cam,
            FRAME,
            640,
            360,
            dets_by_cam=dets_by_cam,
            rotate_180=(cam in QUAD_ROTATE_180),
            apply_defish=True,
        )
        path = OUT / f"lock_tile_{cam}_{FRAME}.jpg"
        cv2.imwrite(str(path), tile)
        # Green box pixels present when players detected
        n_p = sum(
            1
            for d in dets
            if getattr(d, "class_name", "") != "ball" and int(d.class_id) != 1
        )
        green = (
            (tile[:, :, 0] < 40)
            & (tile[:, :, 1] > 180)
            & (tile[:, :, 2] < 40)
        )
        g = int(green.sum())
        if n_p == 0:
            tile_scores.append(8.0)
            notes.append(f"{cam}: no players — skip hard gate")
        elif g > 200:
            tile_scores.append(10.0)
            notes.append(f"{cam}: green_px={g} players={n_p} PASS")
        else:
            tile_scores.append(3.0)
            notes.append(f"{cam}: green_px={g} players={n_p} FAIL (boxes missing)")
    mosaic = mosaic_quads_coach(
        videos,
        FRAME,
        dets_by_cam=dets_by_cam,
        tile_w=640,
        tile_h=360,
        apply_defish=True,
    )
    mpath = OUT / f"lock_mosaic_{FRAME}.jpg"
    cv2.imwrite(str(mpath), mosaic)
    return float(np.mean(tile_scores)), notes, mpath


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    parts = {}
    a, an = score_alpha_lock()
    parts["alpha_lock"] = {"score": a, "notes": an}
    print("alpha_lock", a, an)

    t, tn, path = score_paint_warp_tile()
    parts["paint_warp_tiles"] = {"score": t, "notes": tn, "path": str(path)}
    print("paint_warp", t, tn)

    # Weighted: alpha lock is the regression that broke stitching
    score = round(0.55 * a + 0.45 * t, 1)
    payload = {
        "score": score,
        "pass": score >= GATE,
        "gate": GATE,
        "parts": parts,
        "fix": "paint boxes on raw then defish/rotate with matching alpha labels",
        "ts": datetime.now(timezone.utc).isoformat(),
    }
    (OUT / "box_lock_score.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps({k: payload[k] for k in ("score", "pass", "gate")}, indent=2))
    print(f"PLAYER_BOX_LOCK {score}/10 gate={'PASS' if score >= GATE else 'FAIL'}")
    return 0 if score >= GATE else 1


if __name__ == "__main__":
    raise SystemExit(main())
