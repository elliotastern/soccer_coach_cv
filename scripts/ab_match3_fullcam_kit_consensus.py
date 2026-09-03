#!/usr/bin/env python3
"""Match3 full-cam vs quad kit consensus A/B (local P1/P6 available)."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.review.frame_sync import keep_top1_ball  # noqa: E402
from scripts.eval_team_id_strategy_grid import (  # noqa: E402
    STICKY_M,
    analyze_bidir,
    det_to_dict,
    score_run,
)
from src.perception.rfdetr_local import LocalRFDETRDetector  # noqa: E402
from src.perception.team_strategy import STRATEGIES  # noqa: E402
from src.review.cam_mosaic import match3_videos  # noqa: E402
from src.review.multicam_fuse import fuse_live_dets_for_pitch  # noqa: E402
from src.review.team_live import TeamSession  # noqa: E402

OUT = ROOT / "reports/eval_match3/improve_eng_loop/kit_ref_ab"
CACHE = OUT / "m3_60f_fullcam_det_plain.json"
QUAD = ["P10", "P9", "P7", "P8"]
FULL = ["P1", "P6", "P7", "P8", "P9", "P10"]
STRAT = STRATEGIES["auto_traj_no_gray"]
# ~60 frames @ stride 45 from t≈120s (actiony mid window without huge runtime)
START_FR = 3600  # 60s @ 60fps
STRIDE = 45
N_FRAMES = 60


def dict_to_det(row: dict):
    return SimpleNamespace(
        bbox=tuple(row["bbox"]),
        confidence=float(row["confidence"]),
        class_id=int(row["class_id"]),
        class_name=str(row.get("class_name", "player")),
    )


def _ensure_cam_dets(vids, cam, fr, bag, detect_fn):
    path = vids.get(cam)
    if path is None:
        return
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, float(fr))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return
    bag[f"{cam}__bgr"] = frame
    bag[f"{cam}__wh"] = (frame.shape[1], frame.shape[0])
    bag[cam] = detect_fn(cam, frame)


def build_cache(vids: dict, cams: list[str], frames: list[int]) -> dict:
    det = LocalRFDETRDetector(
        player_checkpoint=str(ROOT / "models/people_after_100_epochs.pth"),
        ball_checkpoint=str(ROOT / "models/v14_residual_snaps/post_train/checkpoint.pth"),
        confidence_threshold=0.15,
        enhance_ball=False,
        use_sahi=False,
        use_kalman=False,
        player_nms_iou=0.30,
        ball_nms_iou=0.4,
    )

    def detect_fn(cam: str, frame_bgr):
        return keep_top1_ball(det.detect(frame_bgr))

    data = {}
    for i, fr in enumerate(frames):
        bag = {}
        for cam in cams:
            _ensure_cam_dets(vids, cam, fr, bag, detect_fn)
        row = {}
        for cam in cams:
            row[cam] = [det_to_dict(d) for d in (bag.get(cam) or [])]
        bgr = {}
        for cam in cams:
            frb = bag.get(f"{cam}__bgr")
            if frb is not None:
                ok, enc = cv2.imencode(".jpg", frb, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                if ok:
                    bgr[cam] = enc.tobytes().hex()
        data[str(fr)] = {"dets": row, "bgr": bgr}
        print(f"cache fr {fr} ({i+1}/{len(frames)})", flush=True)
    return {"frames": frames, "cams": cams, "use_sahi": False, "data": data}


def bag_from_row(row: dict, cams: list[str]) -> dict:
    bag = {}
    for cam in cams:
        bag[cam] = [dict_to_det(d) for d in row["dets"].get(cam, [])]
        hex_bgr = row.get("bgr", {}).get(cam)
        if not hex_bgr:
            continue
        arr = np.frombuffer(bytes.fromhex(hex_bgr), dtype=np.uint8)
        im = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if im is None:
            continue
        bag[f"{cam}__bgr"] = im
        bag[f"{cam}__wh"] = (im.shape[1], im.shape[0])
    return bag


def run_cams(cache: dict, cams: list[str], frames: list[int]) -> tuple[list[dict], dict]:
    sess = TeamSession(strategy=STRAT)
    rows = []
    prev = None
    sticky_keep: list[int] = []
    sticky_total = 0
    shares = []
    for fr in frames:
        bag = bag_from_row(cache["data"][str(fr)], cams)
        live = fuse_live_dets_for_pitch(
            bag, apply_undistort=False, team_session=sess, fuse_stats=True
        )
        players = live["players"]
        teams = [int(p[2]) for p in players]
        n0, n1 = teams.count(0), teams.count(1)
        ng = sum(1 for t in teams if t < 0)
        cons = live.get("consensus") or {}
        rows.append({"fr": fr, "n0": n0, "n1": n1, "gray": ng, "n": len(players), **cons})
        if prev is not None:
            for p in players:
                if int(p[2]) < 0:
                    continue
                near_t = None
                best_d = STICKY_M
                for q in prev:
                    d = ((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2) ** 0.5
                    if d <= best_d:
                        best_d, near_t = d, int(q[2])
                if near_t is None or near_t < 0:
                    continue
                sticky_total += 1
                sticky_keep.append(1 if int(p[2]) == near_t else 0)
        prev = [(float(p[0]), float(p[1]), int(p[2])) for p in players]
        tot = n0 + n1
        shares.append(n0 / tot if tot else 0.5)
    return rows, {"_sticky_keep": sticky_keep, "_sticky_total": sticky_total, "_shares": shares}


def summarize(rows, extra):
    metrics = analyze_bidir(rows)
    scores = score_run(metrics, extra)
    return {"metrics": metrics, "scores": scores}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    vids = match3_videos(ROOT)
    frames = [START_FR + i * STRIDE for i in range(N_FRAMES)]
    if CACHE.is_file():
        cache = json.loads(CACHE.read_text(encoding="utf-8"))
        print(f"loaded {CACHE}", flush=True)
    else:
        print(f"building {CACHE} cams={FULL} n={len(frames)}", flush=True)
        cache = build_cache(vids, FULL, frames)
        CACHE.write_text(json.dumps(cache), encoding="utf-8")

    mid = len(frames) // 2
    tune_fr, hold_fr = frames[:mid], frames[mid:]
    results = {}
    for name, cams in (("quad", QUAD), ("full6", FULL)):
        tr, te = run_cams(cache, cams, tune_fr)
        hr, he = run_cams(cache, cams, hold_fr)
        fr, fe = run_cams(cache, cams, frames)
        results[name] = {
            "cams": cams,
            "tune": summarize(tr, te),
            "hold": summarize(hr, he),
            "full": summarize(fr, fe),
        }
        h = results[name]["hold"]
        print(
            f"{name}: hold_cons={h['scores']['consensus']} "
            f"mfc={h['metrics']['multcam_frac']:.3f} "
            f"agree={h['metrics']['agree_frac']:.3f} "
            f"comp={h['scores']['composite']}",
            flush=True,
        )

    hold_full = results["full6"]["hold"]
    hold_quad = results["quad"]["hold"]
    ok = (
        hold_full["scores"]["consensus"] >= 9.0
        or (
            hold_full["metrics"]["multcam_frac"] >= 0.30
            and hold_full["scores"]["consensus"] >= 8.5
        )
    )
    out = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "cache": str(CACHE.relative_to(ROOT)),
        "frames": {"start": START_FR, "stride": STRIDE, "n": N_FRAMES},
        "results": {
            k: {
                "cams": v["cams"],
                "hold_consensus": v["hold"]["scores"]["consensus"],
                "hold_multcam": v["hold"]["metrics"]["multcam_frac"],
                "hold_agree": v["hold"]["metrics"]["agree_frac"],
                "hold_composite": v["hold"]["scores"]["composite"],
                "full_consensus": v["full"]["scores"]["consensus"],
                "full_multcam": v["full"]["metrics"]["multcam_frac"],
                "full_composite": v["full"]["scores"]["composite"],
            }
            for k, v in results.items()
        },
        "lift_hold_consensus": round(
            hold_full["scores"]["consensus"] - hold_quad["scores"]["consensus"], 3
        ),
        "lift_hold_mfc": round(
            hold_full["metrics"]["multcam_frac"] - hold_quad["metrics"]["multcam_frac"], 3
        ),
        "gate_pass": ok,
        "decision": "path_confirmed" if ok else "need_more_overlap_or_longer_pack",
        "note": "Match4 product still needs Catch for P1/P6; this is Match3 local proof.",
    }
    path = OUT / "ab_match3_fullcam_kit_consensus.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("decision", out["decision"], "gate_pass", ok)
    print("wrote", path)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
