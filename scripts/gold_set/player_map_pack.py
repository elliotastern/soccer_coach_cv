#!/usr/bin/env python3
"""D1+D2: Player map pack — reason matrix + image→pitch collapse score."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.mapping.match3_xy import (  # noqa: E402
    MIN_SUPPORT,
    PLAYER_MIN_SUPPORT,
    bbox_foot,
    diagnose_map_foot,
    load_calib,
    map_ball_box,
    map_player_box,
    scale_px,
)
from src.perception.rfdetr_local import LocalRFDETRDetector  # noqa: E402
from src.review.cam_mosaic import (  # noqa: E402
    _ensure_cam_dets,
    _is_ball_det,
    match3_videos,
)
from src.review.frame_sync import keep_top1_ball  # noqa: E402
from src.review.multicam_fuse import player_det_ok  # noqa: E402

OUT = ROOT / "reports/eval_match3/improve_eng_loop/player_map"
CAMS = ["P10", "P9", "P7", "P8"]
# check_15s start 2390 @60fps, stride ~4 pack + known failure frames
DEFAULT_FRAMES = [
    2390,
    2426,
    2466,
    2506,
    2586,
    2666,
    2746,
    2826,
    2906,
    2986,
    3022,
    3078,
    3146,
    3226,
    3286,
]
P8_Y_TOP = 350
P8_Y_BOTTOM = 550


def _p8_band(py: float) -> str:
    if py < P8_Y_TOP:
        return "top"
    if py < P8_Y_BOTTOM:
        return "mid"
    return "bottom"


def diagnose_p8_quadrants(bag: dict, frame_id: int) -> dict:
    """P8 player drops by image band; ball on H for same tile."""
    calib = load_calib("P8")
    wh = bag.get("P8__wh")
    bands = {b: Counter() for b in ("top", "mid", "bottom")}
    ball_on_h = False
    for d in bag.get("P8") or []:
        if _is_ball_det(d):
            b = map_ball_box(
                calib,
                d.bbox,
                float(d.confidence),
                frame_wh=wh,
                apply_undistort=False,
            )
            ball_on_h = b is not None
            continue
        if not player_det_ok(d):
            continue
        fx, fy = bbox_foot(d.bbox)
        px, py = scale_px(fx, fy, wh, calib.get("image_wh") or wh)
        band = _p8_band(py)
        diag = diagnose_map_foot(
            calib,
            d.bbox,
            float(d.confidence),
            frame_wh=wh,
            apply_undistort=False,
            min_support=PLAYER_MIN_SUPPORT,
            h_player_only=True,
        )
        bands[band][diag["reason"]] += 1
    return {
        "frame_id": frame_id,
        "bands": {k: dict(v) for k, v in bands.items()},
        "ball_on_H": ball_on_h,
    }


def summarize_p8_quadrants(rows: list[dict], product_rows: list[dict]) -> dict:
    pack = {b: Counter() for b in ("top", "mid", "bottom")}
    for row in rows:
        for band, reasons in row["bands"].items():
            for k, v in reasons.items():
                pack[band][k] += v
    bottom = pack["bottom"]
    n_bottom = sum(bottom.values())
    n_bottom_ok_hp = bottom.get("ok", 0)
    prod_ok = sum(r.get("bottom_ok", 0) for r in product_rows)
    prod_total = sum(r.get("bottom_total", 0) for r in product_rows)
    return {
        "pack_bands_h_player": {b: dict(pack[b]) for b in pack},
        "bottom_h_player_mapped_frac": round(n_bottom_ok_hp / max(n_bottom, 1), 4),
        "bottom_n_ok_h_player": n_bottom_ok_hp,
        "bottom_n_total": n_bottom,
        "bottom_off_pitch_h_player": bottom.get("off_pitch", 0),
        "bottom_mapped_frac": round(prod_ok / max(prod_total, 1), 4),
        "bottom_n_ok": prod_ok,
        "bottom_n_total_product": prod_total,
    }


def diagnose_p8_product_bottom(bag: dict) -> dict:
    calib = load_calib("P8")
    wh = bag.get("P8__wh")
    ok, total = 0, 0
    for d in bag.get("P8") or []:
        if _is_ball_det(d) or not player_det_ok(d):
            continue
        fx, fy = bbox_foot(d.bbox)
        _, py = scale_px(fx, fy, wh, calib.get("image_wh") or wh)
        if py < P8_Y_BOTTOM:
            continue
        total += 1
        if map_player_box(
            calib, d.bbox, float(d.confidence), frame_wh=wh, apply_undistort=False
        ):
            ok += 1
    return {"bottom_ok": ok, "bottom_total": total}


def _foot_px(bbox) -> tuple[float, float]:
    x, y, w, h = [float(v) for v in bbox]
    return x + w / 2.0, y + h


def _pair_crush(pts: list[dict]) -> dict:
    """Image pairwise distance vs pitch pairwise distance → crush score."""
    if len(pts) < 2:
        return {
            "n": len(pts),
            "mean_img_px": 0.0,
            "mean_pitch_m": 0.0,
            "crush": 0.0,
            "n_pairs": 0,
        }
    img_d = []
    pitch_d = []
    for i, a in enumerate(pts):
        for b in pts[i + 1 :]:
            dx = a["foot"][0] - b["foot"][0]
            dy = a["foot"][1] - b["foot"][1]
            img_d.append((dx * dx + dy * dy) ** 0.5)
            px = a["xy"][0] - b["xy"][0]
            py = a["xy"][1] - b["xy"][1]
            pitch_d.append((px * px + py * py) ** 0.5)
    mean_img = sum(img_d) / len(img_d)
    mean_pitch = sum(pitch_d) / len(pitch_d)
    # crush: large image spread → tiny pitch spread (m per 100 px)
    m_per_100px = (mean_pitch / max(mean_img, 1.0)) * 100.0
    # flag crush when < ~0.35 m per 100 px among OK Goal2-ish feet
    crush = 1.0 / max(m_per_100px, 1e-3)
    return {
        "n": len(pts),
        "n_pairs": len(img_d),
        "mean_img_px": round(mean_img, 2),
        "mean_pitch_m": round(mean_pitch, 3),
        "m_per_100px": round(m_per_100px, 4),
        "crush": round(crush, 4),
    }


def diagnose_frame(bag: dict, frame_id: int) -> dict:
    reasons = Counter()
    cam_reasons: dict[str, Counter] = {c: Counter() for c in CAMS}
    mapped_by_cam: dict[str, list] = defaultdict(list)
    drops = []
    n_raw = n_ok = 0
    for cam in CAMS:
        calib = load_calib(cam)
        wh = bag.get(f"{cam}__wh")
        for d in bag.get(cam) or []:
            if _is_ball_det(d):
                continue
            n_raw += 1
            if not player_det_ok(d):
                cam_reasons[cam]["det_ok_fail"] += 1
                reasons["det_ok_fail"] += 1
                continue
            n_ok += 1
            if calib is None:
                cam_reasons[cam]["no_calib"] += 1
                reasons["no_calib"] += 1
                continue
            diag = diagnose_map_foot(
                calib,
                d.bbox,
                float(d.confidence),
                frame_wh=wh,
                apply_undistort=False,
                min_support=PLAYER_MIN_SUPPORT,
            )
            reason = diag["reason"]
            reasons[reason] += 1
            cam_reasons[cam][reason] += 1
            foot = _foot_px(d.bbox)
            row = {
                "cam": cam,
                "conf": round(float(d.confidence), 3),
                "foot": [round(foot[0], 1), round(foot[1], 1)],
                "reason": reason,
                "xy": None
                if diag.get("xy") is None
                else [round(float(diag["xy"][0]), 3), round(float(diag["xy"][1]), 3)],
                "support": diag.get("support"),
            }
            if reason == "ok":
                mapped_by_cam[cam].append(row)
            else:
                drops.append(row)
    collapse = {cam: _pair_crush(mapped_by_cam[cam]) for cam in CAMS}
    return {
        "frame_id": frame_id,
        "n_raw": n_raw,
        "n_det_ok": n_ok,
        "reasons": dict(reasons),
        "per_cam_reasons": {c: dict(cam_reasons[c]) for c in CAMS},
        "collapse": collapse,
        "n_mapped": sum(len(mapped_by_cam[c]) for c in CAMS),
        "drops_sample": drops[:12],
    }


def summarize(rows: list[dict]) -> dict:
    pack_reasons = Counter()
    cam_pack = {c: Counter() for c in CAMS}
    crush_flags = []
    for row in rows:
        for k, v in row["reasons"].items():
            pack_reasons[k] += v
        for cam in CAMS:
            for k, v in row["per_cam_reasons"][cam].items():
                cam_pack[cam][k] += v
            col = row["collapse"][cam]
            if col["n"] >= 3 and col.get("m_per_100px", 99) < 0.40:
                crush_flags.append(
                    {"frame_id": row["frame_id"], "cam": cam, **col}
                )
    n_fail = (
        pack_reasons.get("low_support", 0)
        + pack_reasons.get("off_pitch", 0)
        + pack_reasons.get("bad_H", 0)
    )
    n_ok = pack_reasons.get("ok", 0)
    residual = {
        "low_support": pack_reasons.get("low_support", 0),
        "off_pitch": pack_reasons.get("off_pitch", 0),
        "bad_H": pack_reasons.get("bad_H", 0),
    }
    majority = max(residual, key=residual.get) if n_fail else "ok"
    return {
        "pack_reasons": dict(pack_reasons),
        "per_cam_reasons": {c: dict(cam_pack[c]) for c in CAMS},
        "mapped_frac": round(n_ok / max(n_ok + n_fail, 1), 4),
        "n_ok": n_ok,
        "n_map_fail": n_fail,
        "residual_majority": majority,
        "crush_flags": crush_flags,
        "n_crush_flags": len(crush_flags),
        "recommend": (
            "L1_reclick"
            if majority in ("off_pitch", "bad_H") or len(crush_flags) >= 3
            else ("H_p_hull" if majority == "low_support" else "hold")
        ),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--frames", type=int, nargs="+", default=DEFAULT_FRAMES)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=OUT,
    )
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

    rows = []
    d3_rows = []
    d3_product = []
    for fr in args.frames:
        bag = {}
        for cam in CAMS:
            _ensure_cam_dets(vids, cam, fr, bag, detect_fn, True)
        row = diagnose_frame(bag, fr)
        rows.append(row)
        d3_rows.append(diagnose_p8_quadrants(bag, fr))
        d3_product.append(diagnose_p8_product_bottom(bag))
        col8 = row["collapse"]["P8"]
        print(
            f"fr={fr} mapped={row['n_mapped']}/{row['n_det_ok']} "
            f"reasons={row['reasons']} P8_m_per_100px={col8.get('m_per_100px')}",
            flush=True,
        )

    summary = summarize(rows)
    d3_summary = summarize_p8_quadrants(d3_rows, d3_product)
    d1 = {
        "frames": [{"frame_id": r["frame_id"], "reasons": r["reasons"],
                    "per_cam_reasons": r["per_cam_reasons"],
                    "n_mapped": r["n_mapped"], "n_det_ok": r["n_det_ok"]}
                   for r in rows],
        "summary": {
            "pack_reasons": summary["pack_reasons"],
            "per_cam_reasons": summary["per_cam_reasons"],
            "mapped_frac": summary["mapped_frac"],
            "n_ok": summary["n_ok"],
            "n_map_fail": summary["n_map_fail"],
            "residual_majority": summary["residual_majority"],
            "recommend": summary["recommend"],
        },
        "player_min_support": PLAYER_MIN_SUPPORT,
    }
    d2 = {
        "frames": [{"frame_id": r["frame_id"], "collapse": r["collapse"]} for r in rows],
        "crush_flags": summary["crush_flags"],
        "n_crush_flags": summary["n_crush_flags"],
        "note": "crush when n>=3 OK feet and m_per_100px < 0.40",
        "recommend": summary["recommend"],
    }
    (out / "d1_pack_reasons.json").write_text(json.dumps(d1, indent=2), encoding="utf-8")
    (out / "d2_collapse.json").write_text(json.dumps(d2, indent=2), encoding="utf-8")
    d3 = {
        "frames": d3_rows,
        "summary": d3_summary,
        "bands_y": {"top": f"y<{P8_Y_TOP}", "mid": f"{P8_Y_TOP}<=y<{P8_Y_BOTTOM}", "bottom": f"y>={P8_Y_BOTTOM}"},
        "note": "bands=H_player-only drops; bottom_mapped_frac=product map_player_box",
    }
    (out / "d3_p8_quadrant.json").write_text(json.dumps(d3, indent=2), encoding="utf-8")
    print("SUMMARY", json.dumps(summary, indent=2))
    print("D3", json.dumps(d3_summary, indent=2))
    print("WROTE", out / "d1_pack_reasons.json")
    print("WROTE", out / "d2_collapse.json")
    print("WROTE", out / "d3_p8_quadrant.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
