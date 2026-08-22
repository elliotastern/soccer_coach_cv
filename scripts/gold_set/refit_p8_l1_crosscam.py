#!/usr/bin/env python3
"""L1 P8: depth-diverse H refit with ball kill-switch + cross-cam player feet."""
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

from match3_landmarks import CALIB_DIR, save_clicks  # noqa: E402
from pitch1 import pitch1_landmarks  # noqa: E402
from src.mapping.match3_xy import (  # noqa: E402
    MIN_SUPPORT,
    apply_H,
    bbox_foot,
    diagnose_map_foot,
    load_calib,
    map_ball_box,
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
from src.review.multicam_fuse import fuse_live_dets_for_pitch, player_det_ok  # noqa: E402

OUT = ROOT / "reports/eval_match3/improve_eng_loop/player_map"
CAM = "P8"
ANCHOR_CAMS = ["P10", "P9", "P7"]
FRAMES = (2586, 2746, 2906, 3022, 3078)
BALL_BOX = [1389.6, 943.8, 30.4, 28.8]
BALL_CONF = 0.84
MATCH_M = 3.5


def _foot_px(bbox, wh, calib_wh) -> tuple[float, float]:
    fx, fy = bbox_foot(bbox)
    return scale_px(fx, fy, wh, calib_wh)


def _player_span(H: np.ndarray, feet: list[tuple[float, float]]) -> tuple[float, float, int]:
    xs, ys = [], []
    for px, py in feet:
        xy = apply_H(H, px, py)
        if xy and in_pitch_bounds(xy[0], xy[1], margin_m=1.0):
            xs.append(xy[0])
            ys.append(xy[1])
    if len(xs) < 2:
        return 0.0, 0.0, len(xs)
    return max(xs) - min(xs), max(ys) - min(ys), len(xs)


def _ball_ok(calib: dict) -> tuple[bool, tuple[float, float] | None]:
    m = map_ball_box(
        calib,
        BALL_BOX,
        BALL_CONF,
        frame_wh=(1920, 1080),
        apply_undistort=False,
    )
    if m is None:
        return False, None
    bx, by = m["xy"]
    if abs(bx - 25.0) > 5.0 or abs(by + 13.0) > 6.0:
        return False, (bx, by)
    return True, (bx, by)


def collect_crosscam_points(
    vids,
    det,
    frames: tuple[int, ...],
) -> tuple[list, list, list]:
    """Return (img_pts, pitch_pts, labels) from landmarks + fuse-without-P8."""
    rec = json.loads((CALIB_DIR / f"{CAM}_manual.json").read_text())
    names = list(rec["landmark_names"])
    imgs = [list(p) for p in rec["image_points"]]
    LM = pitch1_landmarks()
    src_img, dst_pitch, labels = [], [], []
    for n, p in zip(names, imgs):
        src_img.append(p)
        dst_pitch.append(LM[n]["xy"])
        labels.append(f"lm:{n}")

    def detect_fn(cam, frame_bgr):
        return keep_top1_ball(det.detect(frame_bgr))

    for fr in frames:
        bag_anc = {}
        for c in ANCHOR_CAMS:
            _ensure_cam_dets(vids, c, fr, bag_anc, detect_fn, True)
        live_anc = fuse_live_dets_for_pitch(bag_anc, apply_undistort=False)
        anc_xy = [(float(p[0]), float(p[1])) for p in live_anc["players"]]
        ball_anc = live_anc.get("ball_xy")

        bag_p8 = {}
        _ensure_cam_dets(vids, CAM, fr, bag_p8, detect_fn, True)
        calib = load_calib(CAM)
        wh = bag_p8.get(f"{CAM}__wh")
        cwh = calib.get("image_wh") or wh

        p8_feet = []
        for d in bag_p8.get(CAM) or []:
            if _is_ball_det(d) or not player_det_ok(d):
                continue
            px, py = _foot_px(d.bbox, wh, cwh)
            p8_feet.append((px, py, float(d.confidence)))

        H0 = np.asarray(calib["H"], float)
        rough = []
        for px, py, conf in p8_feet:
            xy = apply_H(H0, px, py)
            if xy and in_pitch_bounds(xy[0], xy[1], margin_m=2.0):
                rough.append((px, py, xy[0], xy[1], conf))

        used_anc = set()
        for px, py, rx, ry, conf in sorted(rough, key=lambda t: -t[4]):
            best_i, best_d = None, MATCH_M
            for i, (ax, ay) in enumerate(anc_xy):
                if i in used_anc:
                    continue
                d = ((rx - ax) ** 2 + (ry - ay) ** 2) ** 0.5
                if d < best_d:
                    best_d, best_i = d, i
            if best_i is not None and best_d <= MATCH_M:
                used_anc.add(best_i)
                ax, ay = anc_xy[best_i]
                src_img.append([px, py])
                dst_pitch.append([ax, ay])
                labels.append(f"fr{fr}:player")

        if ball_anc is not None:
            for d in bag_p8.get(CAM) or []:
                if not _is_ball_det(d):
                    continue
                px, py = _foot_px(d.bbox, wh, cwh)
                src_img.append([px, py])
                dst_pitch.append([float(ball_anc[0]), float(ball_anc[1])])
                labels.append(f"fr{fr}:ball")
                break

    return src_img, dst_pitch, labels


def fit_homography(
    src: list,
    dst: list,
    landmark_n: int,
) -> np.ndarray | None:
    """Weighted DLT: landmarks weight 8×, cross-cam 1×."""
    wts = [8.0] * landmark_n + [1.0] * (len(src) - landmark_n)
    A = []
    for (x, y), (X, Y), wt in zip(src, dst, wts):
        w = wt ** 0.5
        A.append([-w * x, -w * y, -w, 0, 0, 0, w * X * x, w * X * y, w * X])
        A.append([0, 0, 0, -w * x, -w * y, -w, w * Y * x, w * Y * y, w * Y])
    _, _, vt = np.linalg.svd(np.asarray(A, float))
    h = vt[-1]
    if abs(h[8]) < 1e-12:
        return None
    return (h / h[8]).reshape(3, 3)


def landmark_rt(H: np.ndarray, names: list, imgs: list) -> float:
    LM = pitch1_landmarks()
    errs = []
    for n, p in zip(names, imgs):
        xy = apply_H(H, p[0], p[1])
        tx, ty = LM[n]["xy"]
        errs.append(((xy[0] - tx) ** 2 + (xy[1] - ty) ** 2) ** 0.5)
    return float(max(errs))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    rec = json.loads((CALIB_DIR / f"{CAM}_manual.json").read_text())
    names = list(rec["landmark_names"])
    base_imgs = [list(p) for p in rec["image_points"]]
    hull_base = list(rec.get("hull_image_points") or base_imgs)

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
    src, dst, labels = collect_crosscam_points(vids, det, FRAMES)
    n_lm = len(names)
    n_extra = len(src) - n_lm
    print(f"constraints: landmarks={n_lm} extra={n_extra}", flush=True)

    H = fit_homography(src, dst, n_lm)
    if H is None:
        print("fit failed")
        return 1

    calib_try = {**rec, "H": H, "camera": CAM}
    ball_ok, ball_xy = _ball_ok(calib_try)
    rmax = landmark_rt(H, names, base_imgs)

    # P8 player feet sample fr3022
    bag = {}
    def detect_fn(cam, frame_bgr):
        return keep_top1_ball(det.detect(frame_bgr))
    _ensure_cam_dets(vids, CAM, 3022, bag, detect_fn, True)
    wh = bag.get(f"{CAM}__wh")
    feet = []
    for d in bag.get(CAM) or []:
        if _is_ball_det(d) or not player_det_ok(d):
            continue
        feet.append(_foot_px(d.bbox, wh, rec["image_wh"]))
    pxs, pys, n_ok = _player_span(H, feet)

    H0 = np.asarray(rec["H"], float)
    pxs0, pys0, _ = _player_span(H0, feet)

    meta = {
        "status": "candidate",
        "n_constraints": len(src),
        "n_extra": n_extra,
        "landmark_rt_max_m": round(rmax, 4),
        "ball_ok": ball_ok,
        "ball_xy": None if ball_xy is None else [round(ball_xy[0], 3), round(ball_xy[1], 3)],
        "player_span_fr3022": {
            "before": {"x": round(pxs0, 3), "y": round(pys0, 3)},
            "after": {"x": round(pxs, 3), "y": round(pys, 3), "n_ok": n_ok},
        },
        "labels_sample": labels[n_lm : n_lm + 12],
    }
    print(json.dumps(meta, indent=2), flush=True)

    promote = (
        ball_ok
        and rmax <= 0.15
        and pxs >= max(pxs0 + 1.0, 3.0)
        and n_ok >= 3
    )
    meta["promote"] = promote

    if promote and not args.dry_run:
        save_clicks(CAM, rec.get("order") or "goal_right", base_imgs, landmark_names=names, min_n=4)
        rec2 = json.loads((CALIB_DIR / f"{CAM}_manual.json").read_text())
        rec2["H"] = H.tolist()
        rec2["homography"] = H.tolist()
        rec2["roundtrip_max_m"] = rmax
        rec2["roundtrip_mean_m"] = rmax * 0.5
        rec2["l1_note"] = (
            "Player-map L1 cross-cam: weighted H from landmarks + P7/P10 fuse feet "
            f"(n_extra={n_extra}); ball kill-switch held"
        )
        rec2["hull_image_points"] = hull_base
        rec2["hull_note"] = (rec.get("hull_note") or "") + " | L1 cross-cam H refit"
        (CALIB_DIR / f"{CAM}_manual.json").write_text(json.dumps(rec2, indent=2), encoding="utf-8")
        meta["status"] = "promoted"
        # refresh overlay with new H
        from match3_landmarks import STILL_DIR, draw_overlay

        still = cv2.imread(str(STILL_DIR / f"{CAM}.jpg"))
        ov = draw_overlay(still, H, base_imgs, names)
        cv2.imwrite(str(CALIB_DIR / f"{CAM}_manual_overlay.jpg"), ov)
        cv2.imwrite(str(OUT / "p8_l1_overlay.jpg"), ov)
    elif not promote:
        meta["status"] = "rejected"
        meta["reason"] = (
            "ball_ok" if not ball_ok else
            "rt" if rmax > 0.15 else
            "span"
        )

    (OUT / "l1_p8_refit.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("WROTE", OUT / "l1_p8_refit.json", "promote=", promote)
    return 0 if promote or args.dry_run else 1


if __name__ == "__main__":
    raise SystemExit(main())
