#!/usr/bin/env python3
"""Build OSD-synced multi-cam frame pack + prelabels for P/R labeling."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/processed/multicam_20s_match1"
CLIPS = OUT / "clips_osd"
PREDS = OUT / "clips_osd_ball"
PACK = OUT / "eval_pack"
CAMS = ["cam8", "cam9", "cam11", "cam13"]
FPS_SAMPLE = 2.0
DUR_S = 20.0
IOU_DISPLAY_THR = 0.30


def sample_times():
    n = int(round(DUR_S * FPS_SAMPLE))
    return [i / FPS_SAMPLE for i in range(n)]


def extract_frames(times: list[float]) -> None:
    img_dir = PACK / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    for cam in CAMS:
        clip = CLIPS / f"{cam}_20s.mp4"
        cam_dir = img_dir / cam
        cam_dir.mkdir(exist_ok=True)
        for i, t in enumerate(times):
            out = cam_dir / f"t{i:03d}_{t:05.2f}s.jpg"
            subprocess.run([
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-ss", f"{t:.3f}", "-i", str(clip),
                "-frames:v", "1", "-q:v", "2", str(out),
            ], check=True)
        print(f"{cam}: {len(times)} frames")


def load_cam_preds(cam: str) -> list[dict]:
    path = PREDS / f"{cam}_preds.json"
    if not path.is_file():
        return []
    return json.loads(path.read_text())


def nearest_pred(preds: list[dict], t: float) -> list[dict]:
    if not preds:
        return []
    best = min(preds, key=lambda p: abs(float(p["t"]) - t))
    if abs(float(best["t"]) - t) > 0.2:
        return []
    return [
        b for b in best.get("balls", [])
        if float(b.get("confidence", 0)) >= IOU_DISPLAY_THR
    ]


def write_prelabels(times: list[float]) -> None:
    by_cam = {cam: load_cam_preds(cam) for cam in CAMS}
    items = []
    for i, t in enumerate(times):
        cams = {}
        for cam in CAMS:
            balls = nearest_pred(by_cam[cam], t)
            cams[cam] = {
                "image": f"images/{cam}/t{i:03d}_{t:05.2f}s.jpg",
                "prelabel_balls": balls,
                "gt_balls": None,   # null = unlabeled
                "empty": None,
            }
        items.append({"i": i, "t": t, "cams": cams})
    payload = {
        "window_osd": "2026-07-30 18:52:00",
        "duration_s": DUR_S,
        "fps_sample": FPS_SAMPLE,
        "checkpoint": "models/v6_snaps/epoch_110/checkpoint_best_regular.pth",
        "iou_thr": 0.5,
        "note": "Set gt_balls (list of xywh) or empty=true per cam. Then run eval_multicam_ball.py",
        "timestamps": items,
    }
    (PACK / "labels.json").write_text(json.dumps(payload, indent=2))
    print(f"labels template: {PACK / 'labels.json'}")


def main():
    PACK.mkdir(parents=True, exist_ok=True)
    times = sample_times()
    (PACK / "timestamps.json").write_text(json.dumps(times, indent=2))
    extract_frames(times)
    write_prelabels(times)
    # copy labeler next to pack
    src = ROOT / "scripts/gold_set/multicam_labeler.html"
    if src.is_file():
        (PACK / "labeler.html").write_text(src.read_text())
    print(f"pack: {PACK}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
