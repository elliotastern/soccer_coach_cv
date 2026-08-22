#!/usr/bin/env python3
"""Per-cam player map check — quad band funnel + endline smoke + fr3022 stills."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.mapping.match3_xy import (  # noqa: E402
    PLAYER_MIN_SUPPORT,
    bbox_foot,
    calib_undistort_params,
    diagnose_map_foot,
    load_calib,
    map_player_box,
    scale_px,
)
from src.perception.rfdetr_local import LocalRFDETRDetector  # noqa: E402
from src.review.cam_mosaic import (  # noqa: E402
    _ensure_cam_dets,
    _is_ball_det,
    match3_videos,
    read_frame_bgr,
    undistort_bgr,
)
from src.review.frame_sync import keep_top1_ball  # noqa: E402
from src.review.multicam_fuse import player_det_ok  # noqa: E402
from src.review.pitch1_panel import draw_pitch1_ball_panel   # noqa: E402

OUT = ROOT / "reports/eval_match3/improve_eng_loop/player_map/cam_check"
QUAD = ["P10", "P9", "P7", "P8"]
ENDLINE = ["P1", "P6", "P_Goal1", "P_Goal2"]
PACK_FRAMES = [2390, 2426, 2466, 2506, 2586, 2666, 2746, 2826, 2906, 2986, 3022, 3078, 3146, 3226, 3286]
STILL_FR = 3022
Y_TOP, Y_BOT = 350, 550


def _band(py: float) -> str:
    if py < Y_TOP:
        return "top"
    if py < Y_BOT:
        return "mid"
    return "bottom"


def _prep_frame(vids, cam: str, frame_id: int) -> tuple[np.ndarray | None, tuple[int, int]]:
    path = vids.get(cam)
    if path is None:
        return None, (1920, 1080)
    frame = read_frame_bgr(path, frame_id)
    if frame is None:
        return None, (1920, 1080)
    calib = load_calib(cam)
    wh = tuple(calib.get("image_wh") or [frame.shape[1], frame.shape[0]])
    if calib and calib_undistort_params(calib):
        if frame.shape[1] != wh[0] or frame.shape[0] != wh[1]:
            frame = cv2.resize(frame, wh)
        frame = undistort_bgr(frame, calib)
    return frame, wh


def cam_reasons(bag: dict, cam: str, *, product: bool = True) -> Counter:
    calib = load_calib(cam)
    wh = bag.get(f"{cam}__wh")
    out = Counter()
    for d in bag.get(cam) or []:
        if _is_ball_det(d) or not player_det_ok(d):
            continue
        if calib is None:
            out["no_calib"] += 1
            continue
        hp_only = (cam == "P8") and not product
        diag = diagnose_map_foot(
            calib,
            d.bbox,
            float(d.confidence),
            frame_wh=wh,
            apply_undistort=False,
            min_support=PLAYER_MIN_SUPPORT,
            h_player_only=hp_only,
        )
        out[diag["reason"]] += 1
    return out


def cam_bands(bag: dict, cam: str) -> dict[str, Counter]:
    calib = load_calib(cam)
    wh = bag.get(f"{cam}__wh")
    bands = {b: Counter() for b in ("top", "mid", "bottom")}
    hp_only = cam == "P8"
    for d in bag.get(cam) or []:
        if _is_ball_det(d) or not player_det_ok(d) or calib is None:
            continue
        _, py = scale_px(*bbox_foot(d.bbox), wh, calib.get("image_wh") or wh)
        diag = diagnose_map_foot(
            calib,
            d.bbox,
            float(d.confidence),
            frame_wh=wh,
            apply_undistort=False,
            min_support=PLAYER_MIN_SUPPORT,
            h_player_only=hp_only,
        )
        bands[_band(py)][diag["reason"]] += 1
    return {k: dict(v) for k, v in bands.items()}


def summarize_cam(rows: list[Counter]) -> dict:
    pack = Counter()
    for row in rows:
        pack.update(row)
    ok = pack.get("ok", 0)
    fail = pack.get("off_pitch", 0) + pack.get("low_support", 0) + pack.get("bad_H", 0)
    return {
        "reasons": dict(pack),
        "mapped_frac": round(ok / max(ok + fail, 1), 4),
        "n_ok": ok,
        "n_fail": fail,
    }


def gate_cam(cam: str, mapped_frac: float, bands: dict | None = None) -> bool:
    if cam == "P7":
        return mapped_frac >= 0.95
    if cam == "P10":
        return mapped_frac >= 0.85
    if cam == "P9":
        return mapped_frac >= 0.75
    if cam == "P8":
        return mapped_frac >= 0.60
    return mapped_frac >= 0.50


def render_still(vids, det, cam: str, frame_id: int, out_path: Path) -> dict:
    frame, wh = _prep_frame(vids, cam, frame_id)
    if frame is None:
        return {"cam": cam, "error": "no_frame"}
    calib = load_calib(cam)
    dets = keep_top1_ball(det.detect(frame))
    img = frame.copy()
    mapped_xy = []
    n_ok = n_fail = 0
    for d in dets:
        if _is_ball_det(d):
            continue
        if not player_det_ok(d):
            continue
        m = None
        if calib:
            m = map_player_box(
                calib, d.bbox, float(d.confidence), frame_wh=wh, apply_undistort=False
            )
        x, y, w, h = [int(v) for v in d.bbox]
        ok = m is not None
        n_ok += int(ok)
        n_fail += int(not ok)
        col = (40, 220, 40) if ok else (40, 40, 220)
        cv2.rectangle(img, (x, y), (x + w, y + h), col, 2)
        fx, fy = bbox_foot(d.bbox)
        cv2.circle(img, (int(fx), int(fy)), 6, col, -1)
        if ok:
            mapped_xy.append((m["xy"][0], m["xy"][1], 0, 0))
    pitch = draw_pitch1_ball_panel(420, 320, None, cam=cam, players=mapped_xy, tight=True)
    h = max(img.shape[0], pitch.shape[0])
    combo = np.zeros((h, img.shape[1] + pitch.shape[1] + 8, 3), dtype=np.uint8)
    combo[:] = (30, 30, 30)
    combo[: img.shape[0], : img.shape[1]] = img
    combo[: pitch.shape[0], img.shape[1] + 8 :] = pitch
    cv2.putText(
        combo,
        f"{cam} fr{frame_id} map {n_ok}/{n_ok + n_fail}",
        (8, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (240, 240, 240),
        2,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), combo)
    return {"cam": cam, "n_ok": n_ok, "n_fail": n_fail, "path": str(out_path.relative_to(ROOT))}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--still-fr", type=int, default=STILL_FR)
    p.add_argument("--skip-stills", action="store_true")
    p.add_argument("--out-dir", type=Path, default=OUT)
    args = p.parse_args()
    out = args.out_dir if args.out_dir.is_absolute() else (ROOT / args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

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

    def detect_fn(cam, frame_bgr):
        return keep_top1_ball(det.detect(frame_bgr))

    quad_rows = {c: [] for c in QUAD}
    quad_bands = {c: [] for c in QUAD}
    end_rows = {c: [] for c in ENDLINE}
    for fr in PACK_FRAMES:
        bag = {}
        for cam in QUAD + ENDLINE:
            _ensure_cam_dets(vids, cam, fr, bag, detect_fn, True)
        for cam in QUAD:
            quad_rows[cam].append(cam_reasons(bag, cam, product=True))
            quad_bands[cam].append(cam_bands(bag, cam))
        for cam in ENDLINE:
            end_rows[cam].append(cam_reasons(bag, cam, product=True))
        print(f"fr={fr} ok", flush=True)

    d4 = {}
    for cam in QUAD:
        summ = summarize_cam(quad_rows[cam])
        band_pack = {b: Counter() for b in ("top", "mid", "bottom")}
        for row in quad_bands[cam]:
            for b, reasons in row.items():
                band_pack[b].update(reasons)
        bands = {b: dict(v) for b, v in band_pack.items()}
        summ["bands_h_player" if cam == "P8" else "bands"] = bands
        summ["pass"] = gate_cam(cam, summ["mapped_frac"], bands)
        d4[cam] = summ
        print(f"{cam} mapped_frac={summ['mapped_frac']} pass={summ['pass']}", flush=True)

    d5 = {}
    for cam in ENDLINE:
        summ = summarize_cam(end_rows[cam])
        n = summ["n_ok"] + summ["n_fail"]
        summ["sparse"] = n < 5
        summ["pass"] = summ["sparse"] or gate_cam(cam, summ["mapped_frac"])
        d5[cam] = summ
        print(
            f"{cam} smoke mapped_frac={summ['mapped_frac']} "
            f"n={n} pass={summ['pass']} sparse={summ['sparse']}",
            flush=True,
        )

    stills = []
    if not args.skip_stills:
        for cam in QUAD + ENDLINE:
            if vids.get(cam) is None:
                continue
            stills.append(
                render_still(vids, det, cam, args.still_fr, out / f"{cam}_fr{args.still_fr}.jpg")
            )

    summary = {
        "pack_frames": PACK_FRAMES,
        "still_frame": args.still_fr,
        "quad": d4,
        "endline": d5,
        "stills": stills,
        "all_quad_pass": all(d4[c]["pass"] for c in QUAD),
        "endline_pass": all(d5[c]["pass"] for c in ENDLINE if d5[c]["n_ok"] + d5[c]["n_fail"] > 0),
    }
    (out / "d4_quad_cam_bands.json").write_text(json.dumps(d4, indent=2), encoding="utf-8")
    (out / "d5_endline_smoke.json").write_text(json.dumps(d5, indent=2), encoding="utf-8")
    (out / "cam_check_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("WROTE", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
