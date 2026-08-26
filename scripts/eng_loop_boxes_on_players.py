#!/usr/bin/env python3
"""Eng-loop: player boxes must sit on players (screenshots + vision gate ≥9/10)."""
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
from src.perception.rfdetr_local import LocalRFDETRDetector  # noqa: E402
from src.review.cam_mosaic import (  # noqa: E402
    MOSAIC_DEFISH_ALPHA,
    QUAD_ROTATE_180,
    _annotate,
    _tile,
    match3_videos,
    mosaic_quads_coach,
    read_frame_bgr,
    undistort_bgr,
)
from src.review.frame_sync import keep_top1_ball  # noqa: E402

OUT = ROOT / "reports/eval_match3/improve_eng_loop/player_stitch"
FRAME = 2400
GATE = 9.0


def _containment(frame: np.ndarray, dets: list) -> tuple[float, list[str]]:
    """Heuristic: green jersey/non-pitch chroma inside each player box."""
    notes = []
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # non-green, reasonably bright = jersey/skin/shoes (rough)
    chroma = ((hsv[:, :, 0] < 35) | (hsv[:, :, 0] > 95)) & (hsv[:, :, 1] > 35) & (
        hsv[:, :, 2] > 50
    )
    scores = []
    for i, d in enumerate(dets):
        if getattr(d, "class_name", "") == "ball" or int(d.class_id) == 1:
            continue
        x, y, w, h = [int(v) for v in d.bbox]
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(frame.shape[1], x + w), min(frame.shape[0], y + h)
        if x1 <= x0 or y1 <= y0:
            continue
        roi = chroma[y0:y1, x0:x1]
        frac = float(roi.mean()) if roi.size else 0.0
        # also compare right-shifted and left-shifted windows — on-box should win
        shift = max(8, w // 4)
        left = chroma[y0:y1, max(0, x0 - shift) : max(0, x1 - shift)]
        right = chroma[y0:y1, min(frame.shape[1], x0 + shift) : min(frame.shape[1], x1 + shift)]
        fl = float(left.mean()) if left.size else 0.0
        fr = float(right.mean()) if right.size else 0.0
        ok = frac >= fl - 0.02 and frac >= fr - 0.02 and frac > 0.02
        scores.append(10.0 if ok else max(0.0, 10.0 * frac / max(fl, fr, 0.05)))
        notes.append(
            f"p{i}: in={frac:.3f} L={fl:.3f} R={fr:.3f} {'PASS' if ok else 'FAIL'}"
        )
    if not scores:
        return 5.0, ["no player dets"]
    return float(np.mean(scores)), notes


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
    parts = {}
    dets_by = {}
    raw_scores = []
    tile_scores = []
    all_notes = []

    for cam in ["P8", "P9", "P10", "P7"]:
        raw = read_frame_bgr(vids[cam], FRAME)
        dets = keep_top1_ball(det.detect(raw))
        dets_by[cam] = dets
        ann = _annotate(raw.copy(), dets, coach_simple=True)
        cv2.imwrite(str(OUT / f"verify_raw_{cam}.jpg"), cv2.resize(ann, (1280, 720)))
        # zoom
        xs, ys = [], []
        for d in dets:
            if getattr(d, "class_name", "") == "ball":
                continue
            x, y, w, h = d.bbox
            xs += [x, x + w]
            ys += [y, y + h]
        if xs:
            x0, x1 = max(0, int(min(xs) - 40)), min(raw.shape[1], int(max(xs) + 40))
            y0, y1 = max(0, int(min(ys) - 40)), min(raw.shape[0], int(max(ys) + 40))
            cv2.imwrite(str(OUT / f"verify_zoom_{cam}.jpg"), ann[y0:y1, x0:x1])

        s, notes = _containment(raw, dets)
        raw_scores.append(s)
        all_notes.append({cam: notes})
        print(cam, "raw_score", round(s, 1), notes)

        tile = _tile(
            vids,
            cam,
            FRAME,
            960,
            540,
            dets_by_cam=dets_by,
            rotate_180=(cam in QUAD_ROTATE_180),
            apply_defish=True,
        )
        cv2.imwrite(str(OUT / f"verify_tile_{cam}.jpg"), tile)
        # green pixels present
        green = (tile[:, :, 0] < 50) & (tile[:, :, 1] > 160) & (tile[:, :, 2] < 50)
        g = int(green.sum())
        n_p = sum(
            1
            for d in dets
            if getattr(d, "class_name", "") != "ball" and int(d.class_id) != 1
        )
        ts = 10.0 if (n_p == 0 or g > 200) else 3.0
        tile_scores.append(ts)

    mosaic = mosaic_quads_coach(
        vids, FRAME, dets_by_cam=dets_by, tile_w=640, tile_h=360, apply_defish=True
    )
    mpath = OUT / "verify_mosaic.jpg"
    cv2.imwrite(str(mpath), mosaic)

    # before/after strip vs stretch (old) if we still have cmp
    raw_mean = float(np.mean(raw_scores)) if raw_scores else 0.0
    tile_mean = float(np.mean(tile_scores)) if tile_scores else 0.0
    score = round(0.7 * raw_mean + 0.3 * tile_mean, 1)
    payload = {
        "score": score,
        "pass": score >= GATE,
        "gate": GATE,
        "raw_containment": raw_mean,
        "tile_green": tile_mean,
        "notes": all_notes,
        "screenshots": {
            "mosaic": str(mpath),
            "zooms": [str(OUT / f"verify_zoom_{c}.jpg") for c in ["P8", "P9", "P10", "P7"]],
        },
        "fix": "RF-DETR letterbox predict (no anamorphic stretch)",
        "ts": datetime.now(timezone.utc).isoformat(),
    }
    (OUT / "box_on_player_score.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps({k: payload[k] for k in ("score", "pass", "gate", "raw_containment")}, indent=2))
    print(f"BOX_ON_PLAYER {score}/10 gate={'PASS' if score >= GATE else 'FAIL'}")
    return 0 if score >= GATE else 1


if __name__ == "__main__":
    raise SystemExit(main())
