#!/usr/bin/env python3
"""L2 P8: refit H_player with ball-foot lower-FOV anchor (H unchanged)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts/gold_set"))

from match3_landmarks import CALIB_DIR, STILL_DIR, draw_overlay  # noqa: E402
from pitch1 import pitch1_landmarks  # noqa: E402
from src.mapping.match3_xy import (  # noqa: E402
    apply_H,
    bbox_foot,
    load_calib,
    map_ball_box,
    map_player_box,
    scale_px,
)
from src.mapping.pitch_bounds import in_pitch_bounds  # noqa: E402
from src.perception.rfdetr_local import LocalRFDETRDetector  # noqa: E402
from src.review.cam_mosaic import (  # noqa: E402
    _ensure_cam_dets,
    _is_ball_det,
    match3_videos,
)
from src.review.frame_sync import keep_top1_ball  # noqa: E402
from src.review.multicam_fuse import player_det_ok  # noqa: E402

OUT = ROOT / "reports/eval_match3/improve_eng_loop/player_map"
CAM = "P8"
BALL_BOX = [1389.6, 943.8, 30.4, 28.8]
BALL_CONF = 0.84
P8_Y_BOTTOM = 550
LM_WT = 8.0
BALL_WT = 4.0


def fit_weighted_h(src: list, dst: list, weights: list) -> np.ndarray | None:
    A = []
    for (x, y), (X, Y), wt in zip(src, dst, weights):
        w = float(wt) ** 0.5
        A.append([-w * x, -w * y, -w, 0, 0, 0, w * X * x, w * X * y, w * X])
        A.append([0, 0, 0, -w * x, -w * y, -w, w * Y * x, w * Y * y, w * Y])
    _, _, vt = np.linalg.svd(np.asarray(A, float))
    h = vt[-1]
    if abs(h[8]) < 1e-12:
        return None
    return (h / h[8]).reshape(3, 3)


def player_landmark_rt(H: np.ndarray, names: list, imgs: list) -> float:
    LM = pitch1_landmarks()
    errs = []
    for n, p in zip(names, imgs):
        xy = apply_H(H, p[0], p[1])
        tx, ty = LM[n]["xy"]
        errs.append(((xy[0] - tx) ** 2 + (xy[1] - ty) ** 2) ** 0.5)
    return float(max(errs))


def bottom_mapped_frac(vids, det, frames: list[int], calib: dict) -> tuple[float, int, int]:
    def detect_fn(cam, frame_bgr):
        return keep_top1_ball(det.detect(frame_bgr))

    ok, total = 0, 0
    for fr in frames:
        bag = {}
        _ensure_cam_dets(vids, CAM, fr, bag, detect_fn, True)
        wh = bag.get(f"{CAM}__wh")
        cwh = calib.get("image_wh") or wh
        for d in bag.get(CAM) or []:
            if _is_ball_det(d) or not player_det_ok(d):
                continue
            fx, fy = bbox_foot(d.bbox)
            _, py = scale_px(fx, fy, wh, cwh)
            if py < P8_Y_BOTTOM:
                continue
            total += 1
            if map_player_box(
                calib, d.bbox, float(d.confidence), frame_wh=wh, apply_undistort=False
            ):
                ok += 1
    return ok / max(total, 1), ok, total


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    rec = json.loads((CALIB_DIR / f"{CAM}_manual.json").read_text())
    names = list(rec["player_landmark_names"])
    imgs = [list(p) for p in rec["player_image_points"]]
    LM = pitch1_landmarks()
    dst = [LM[n]["xy"] for n in names]

    calib = load_calib(CAM)
    ball_map = map_ball_box(
        calib,
        BALL_BOX,
        BALL_CONF,
        frame_wh=tuple(rec["image_wh"]),
        apply_undistort=False,
    )
    if ball_map is None:
        print("ball kill-switch failed before refit")
        return 1
    fx, fy = bbox_foot(BALL_BOX)
    ball_px = scale_px(fx, fy, rec["image_wh"], rec["image_wh"])
    ball_pitch = ball_map["xy"]

    src = imgs + [list(ball_px)]
    dst_all = dst + [ball_pitch]
    wts = [LM_WT] * len(names) + [BALL_WT]

    Hp = fit_weighted_h(src, dst_all, wts)
    if Hp is None:
        print("H_player fit failed")
        return 1

    rmax = player_landmark_rt(Hp, names, imgs)
    ball_after = map_ball_box(
        calib,
        BALL_BOX,
        BALL_CONF,
        frame_wh=tuple(rec["image_wh"]),
        apply_undistort=False,
    )
    ball_ok = ball_after is not None and abs(ball_after["xy"][0] - ball_pitch[0]) < 5.0

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
    vids = match3_videos(ROOT)
    calib_before = load_calib(CAM)
    d3_path = OUT / "d3_p8_quadrant.json"
    if d3_path.is_file():
        fr_list = [int(r["frame_id"]) for r in json.loads(d3_path.read_text())["frames"]]
    else:
        fr_list = [3022, 3078]

    frac_before, ok_b, tot_b = bottom_mapped_frac(vids, det, fr_list, calib_before)
    calib_try = {**calib_before, "H_player": Hp.tolist()}
    frac_after, ok_a, tot_a = bottom_mapped_frac(vids, det, fr_list, calib_try)

    promote = (
        ball_ok
        and rmax <= 0.15
        and frac_after >= max(0.5, frac_before + 0.05)
    )

    meta = {
        "status": "candidate",
        "l2": "H_player_ball_foot_anchor",
        "ball_anchor_px": [round(ball_px[0], 2), round(ball_px[1], 2)],
        "ball_anchor_pitch": [round(ball_pitch[0], 4), round(ball_pitch[1], 4)],
        "player_landmark_rt_max_m": round(rmax, 4),
        "ball_ok_on_H": ball_ok,
        "bottom_mapped_frac": {
            "before": round(frac_before, 4),
            "after": round(frac_after, 4),
            "ok_before": ok_b,
            "ok_after": ok_a,
            "total_bottom": tot_b,
        },
        "weights": {"landmarks": LM_WT, "ball_anchor": BALL_WT},
        "promote": promote,
    }
    print(json.dumps(meta, indent=2), flush=True)

    if promote and not args.dry_run:
        path = CALIB_DIR / f"{CAM}_manual.json"
        rec2 = json.loads(path.read_text(encoding="utf-8"))
        rec2["H_player"] = Hp.tolist()
        rec2["player_h_note"] = (
            "L2: 5 player landmarks + ball-foot anchor (fr3022 on H); ball keeps H"
        )
        rec2["player_roundtrip_max_m"] = rmax
        rec2["player_ball_anchor_px"] = [round(ball_px[0], 2), round(ball_px[1], 2)]
        path.write_text(json.dumps(rec2, indent=2), encoding="utf-8")
        still = cv2.imread(str(STILL_DIR / f"{CAM}.jpg"))
        ov = draw_overlay(still, Hp, imgs, names)
        cv2.imwrite(str(CALIB_DIR / f"{CAM}_manual_player_overlay.jpg"), ov)
        meta["status"] = "promoted_H_player_bottom_anchor"
    elif not promote:
        meta["status"] = "rejected_l2"
        meta["reason"] = (
            "ball" if not ball_ok else "rt" if rmax > 0.15 else "bottom_frac"
        )

    (OUT / "l1_p8_refit.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("WROTE", OUT / "l1_p8_refit.json", "promote=", promote)
    return 0 if promote or args.dry_run else 1


if __name__ == "__main__":
    raise SystemExit(main())
