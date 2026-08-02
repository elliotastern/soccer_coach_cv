#!/usr/bin/env python3
"""Overlay v6 ball dets on OSD-synced 20s multi-cam clips and rebuild quad."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "data/processed/multicam_20s_match1"
CLIPS = OUT / "clips_osd"
ANN = OUT / "clips_osd_ball"
CKPT = ROOT / "models/v6_snaps/epoch_110/checkpoint_best_regular.pth"
CAMS = ["cam8", "cam9", "cam11", "cam13"]
# Detect every Nth source frame; hold box between detects for smooth playback
DETECT_STRIDE = 3
MIN_THR = 0.30


def load_model():
    from src.perception.rfdetr_local import load_ball_model
    return load_ball_model(str(CKPT))


def detect_balls(model, frame_bgr, thr: float):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    raw = model.predict(Image.fromarray(rgb), threshold=thr)
    balls = []
    if not hasattr(raw, "class_id"):
        return balls
    h, w = frame_bgr.shape[:2]
    for i in range(len(raw.class_id)):
        x1, y1, x2, y2 = map(float, raw.xyxy[i])
        bw, bh = x2 - x1, y2 - y1
        if bw <= 0 or bh <= 0:
            continue
        side = min(bw, bh)
        # geometry gate scaled for 4K
        if side < 8 or side > 240:
            continue
        aspect = bw / bh if bh else 999.0
        if aspect < 0.35 or aspect > 2.8:
            continue
        balls.append({
            "bbox": [x1, y1, bw, bh],
            "confidence": float(raw.confidence[i]),
            "side": side,
        })
    balls.sort(key=lambda b: -b["confidence"])
    return balls[:2]


def conf_color(c: float):
    if c >= 0.8:
        return (0, 255, 0)      # green = pass Phase-1 bar
    if c >= 0.5:
        return (0, 220, 255)    # yellow
    return (0, 140, 255)        # orange


def draw_balls(frame, balls, cam: str, frame_i: int):
    out = frame.copy()
    top = balls[0]["confidence"] if balls else 0.0
    label = f"{cam}  balls={len(balls)}  top={top:.2f}"
    cv2.rectangle(out, (8, 8), (520, 52), (0, 0, 0), -1)
    cv2.putText(
        out, label, (16, 40), cv2.FONT_HERSHEY_SIMPLEX,
        1.0, (0, 255, 255), 2, cv2.LINE_AA,
    )
    for b in balls:
        x, y, w, h = b["bbox"]
        c = b["confidence"]
        color = conf_color(c)
        p1 = (int(x), int(y))
        p2 = (int(x + w), int(y + h))
        cv2.rectangle(out, p1, p2, color, 3)
        tag = f"ball {c:.2f}"
        cv2.putText(
            out, tag, (p1[0], max(20, p1[1] - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA,
        )
    return out


def annotate_cam(model, cam: str) -> Path:
    src = CLIPS / f"{cam}_20s.mp4"
    dst = ANN / f"{cam}_20s_ball.mp4"
    ANN.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise RuntimeError(f"open failed: {src}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    writer = cv2.VideoWriter(
        str(dst),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )
    preds = []
    last_balls = []
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % DETECT_STRIDE == 0:
            last_balls = detect_balls(model, frame, MIN_THR)
            preds.append({
                "frame": i,
                "t": i / fps,
                "balls": last_balls,
            })
            print(f"{cam} f={i}/{n} n={len(last_balls)} "
                  f"top={last_balls[0]['confidence']:.3f}" if last_balls
                  else f"{cam} f={i}/{n} n=0")
        writer.write(draw_balls(frame, last_balls, cam, i))
        i += 1
    cap.release()
    writer.release()
    # remux to h264 for browser/quicktime
    h264 = ANN / f"{cam}_20s_ball_h264.mp4"
    subprocess.run([
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(dst), "-c:v", "libx264", "-crf", "18",
        "-preset", "veryfast", "-an", str(h264),
    ], check=True)
    (ANN / f"{cam}_preds.json").write_text(json.dumps(preds, indent=2))
    return h264


def build_quad(paths: list[Path]) -> Path:
    quad = OUT / "quad_20s_osd_ball.mp4"
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(paths[0]), "-i", str(paths[1]),
        "-i", str(paths[2]), "-i", str(paths[3]),
        "-filter_complex",
        "[0:v]fps=30,scale=960:-2[v0];[1:v]fps=30,scale=960:-2[v1];"
        "[2:v]fps=30,scale=960:-2[v2];[3:v]fps=30,scale=960:-2[v3];"
        "[v0][v1]hstack=inputs=2[top];[v2][v3]hstack=inputs=2[bot];"
        "[top][bot]vstack=inputs=2[out]",
        "-map", "[out]", "-c:v", "libx264", "-crf", "20",
        "-preset", "veryfast", "-an", str(quad),
    ]
    subprocess.run(cmd, check=True)
    return quad


def summarize():
    rows = []
    for cam in CAMS:
        preds = json.loads((ANN / f"{cam}_preds.json").read_text())
        for thr in (0.3, 0.5, 0.8):
            hits = sum(1 for p in preds if any(b["confidence"] >= thr for b in p["balls"]))
            rows.append((cam, thr, hits, len(preds)))
    print("\n=== presence by cam / thr (detect frames) ===")
    for cam, thr, hits, n in rows:
        print(f"{cam} conf>={thr}: {hits}/{n} ({hits/n if n else 0:.2f})")


def main():
    print(f"checkpoint: {CKPT}")
    model = load_model()
    paths = [annotate_cam(model, cam) for cam in CAMS]
    quad = build_quad(paths)
    summarize()
    # point watch page at annotated mosaic
    watch = OUT / "watch_ball.html"
    src = (OUT / "watch.html").read_text()
    src = src.replace("quad_20s_osd.mp4", "quad_20s_osd_ball.mp4")
    src = src.replace("clips_osd/", "clips_osd_ball/")
    src = src.replace("_20s.mp4", "_20s_ball_h264.mp4")
    src = src.replace(
        "OSD-synced to 18:52:00",
        "OSD-synced 18:52:00 + v6_epoch_110 ball boxes "
        "(green≥0.8, yellow≥0.5, orange≥0.3)",
    )
    watch.write_text(src)
    print(f"\nquad: {quad}")
    print(f"viewer: {watch}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
