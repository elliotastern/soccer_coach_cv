#!/usr/bin/env python3
"""Out-of-sample demo: locked pick on Top Right 4quad (not Top Left gold).

Uses existing det cache + source clips (no re-detect). Writes overlay MP4,
contact sheet, and HTML under reports/eval_match2_v10/locked_oos_demo/.
Never trains.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "gold_set"))

from eval_match2_top_left_multicam_baseline import cache_load  # noqa: E402
from multicam_select_policy import (  # noqa: E402
    PRODUCT_POLICY_ID,
    SURVEY_CAMS,
    TOP_LEFT_THR_BY_CAM,
    filter_active,
    pick_product,
)

SLOT = "top_right"
STEM = "quad_top_right_t00125.0s"
LABEL = "Top Right (out-of-sample vs Top Left gold)"
CLOCK = "2:05–2:10"
CACHE = ROOT / "reports/eval_match2_v10/4quad_multicam_survey/det_cache_top_right_thr010.json"
SRC = ROOT / "reports/eval_match2_v10/4quad_test/source"
OUT = ROOT / "reports/eval_match2_v10/locked_oos_demo"
DETECT_W = 1920


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", type=Path, default=CACHE)
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--stride", type=int, default=2, help="write every Nth frame")
    p.add_argument("--max-frames", type=int, default=150)
    return p.parse_args()


def resize_w(frame, width=DETECT_W):
    h, w = frame.shape[:2]
    if w == width:
        return frame
    return cv2.resize(frame, (width, int(round(h * width / w))), interpolation=cv2.INTER_AREA)


def paint(frame, cam, pred, overlay_w=1280):
    vis = resize_w(frame, overlay_w)
    scale = overlay_w / float(frame.shape[1])
    h, w = vis.shape[:2]
    if pred is None:
        tag = f"{cam or 'none'}  LOCKED  no ball"
        cv2.rectangle(vis, (8, 8), (min(w - 8, 720), 56), (0, 0, 0), -1)
        cv2.putText(vis, tag, (16, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 140, 255), 2)
        return vis
    box, conf, side = pred
    x, y, bw, bh = [int(round(v * scale)) for v in box]
    color = (0, 255, 0) if conf >= 0.80 else (0, 220, 255)
    cv2.rectangle(vis, (x, y), (x + max(bw, 1), y + max(bh, 1)), color, 3)
    cx, cy = x + max(bw, 1) // 2, y + max(bh, 1) // 2
    cv2.circle(vis, (cx, cy), max(14, int(0.025 * min(h, w))), color, 2)
    tag = f"{cam}  LOCKED  {conf:.2f}  {side:.0f}px"
    cv2.rectangle(vis, (8, 8), (min(w - 8, 780), 56), (0, 0, 0), -1)
    cv2.putText(vis, tag, (16, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2)
    return vis


def open_caps(stem: str):
    caps = {}
    for cam in SURVEY_CAMS:
        path = SRC / f"{stem}_{cam}.mp4"
        if not path.is_file():
            raise FileNotFoundError(path)
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"open failed {path}")
        caps[cam] = cap
    return caps


def encode_h264(path: Path):
    tmp = path.with_name(path.stem + "_h264.mp4")
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(path), "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-preset", "veryfast", "-crf", "20", "-movflags", "+faststart", "-an", str(tmp),
    ]
    subprocess.run(cmd, check=True)
    tmp.replace(path)


def contact_sheet(frames, cols=5, cell=320):
    n = len(frames)
    rows = (n + cols - 1) // cols
    sheet = np.zeros((rows * cell, cols * cell, 3), dtype=np.uint8)
    for i, fr in enumerate(frames):
        r, c = divmod(i, cols)
        thumb = cv2.resize(fr, (cell, cell))
        sheet[r * cell:(r + 1) * cell, c * cell:(c + 1) * cell] = thumb
    return sheet


def main() -> int:
    args = parse_args()
    if not args.cache.is_file():
        raise FileNotFoundError(args.cache)
    dets = cache_load(args.cache)
    n_cache = len(next(iter(dets.values())))
    n = min(args.max_frames * args.stride, n_cache)
    args.out.mkdir(parents=True, exist_ok=True)
    caps = open_caps(STEM)
    fps = float(caps[SURVEY_CAMS[0]].get(cv2.CAP_PROP_FPS) or 30.0) / args.stride

    writer = None
    wins = Counter()
    sides = []
    thumbs = []
    written = 0
    raw_path = args.out / f"{STEM}_locked_overlay.mp4"

    for i in range(n):
        frames = {}
        ok_all = True
        for cam, cap in caps.items():
            ok, fr = cap.read()
            if not ok:
                ok_all = False
                break
            frames[cam] = resize_w(fr, DETECT_W)
        if not ok_all:
            break
        if i % args.stride != 0:
            continue
        active = filter_active(dets, i, SURVEY_CAMS, TOP_LEFT_THR_BY_CAM)
        cam, pred = pick_product(active) if active else (None, None)
        wins[cam or "none"] += 1
        if pred is not None:
            sides.append(float(pred[2]))
        base = frames[cam] if cam in frames else frames[SURVEY_CAMS[0]]
        vis = paint(base, cam, pred)
        if writer is None:
            h, w = vis.shape[:2]
            writer = cv2.VideoWriter(
                str(raw_path), cv2.VideoWriter_fourcc(*"mp4v"), max(fps, 1.0), (w, h)
            )
        writer.write(vis)
        if written % 10 == 0 and len(thumbs) < 12:
            thumbs.append(vis)
        written += 1
        if written >= args.max_frames:
            break

    for cap in caps.values():
        cap.release()
    if writer is not None:
        writer.release()
        encode_h264(raw_path)

    sheet = contact_sheet(thumbs)
    sheet_path = args.out / f"{STEM}_locked_contact.jpg"
    cv2.imwrite(str(sheet_path), sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 88])

    med = sorted(sides)[len(sides) // 2] if sides else None
    ge20 = (sum(1 for s in sides if s >= 20) / len(sides)) if sides else 0.0
    top = [{"cam": c, "n": v, "share": v / written} for c, v in wins.most_common()]
    stats = {
        "slot": SLOT,
        "label": LABEL,
        "clock": CLOCK,
        "policy": PRODUCT_POLICY_ID,
        "n_frames_written": written,
        "stride": args.stride,
        "winners": top,
        "median_side_px": med,
        "frac_ge_20px": round(ge20, 3),
        "overlay": str(raw_path.relative_to(ROOT)),
        "contact": str(sheet_path.relative_to(ROOT)),
        "note": "Out-of-sample vs Top Left gold/labels. Picks from survey det cache + locked policy.",
    }
    (args.out / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    ov_url = f"/reports/eval_match2_v10/locked_oos_demo/{raw_path.name}"
    sheet_url = f"/reports/eval_match2_v10/locked_oos_demo/{sheet_path.name}"
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Locked OOS — Top Right</title>
<style>
body{{font-family:ui-sans-serif,system-ui;margin:24px;background:#111;color:#eee}}
h1{{font-size:1.4rem}} .meta{{opacity:.85;margin-bottom:16px}}
video{{width:min(100%,1100px);background:#000}}
img{{width:min(100%,1100px);margin-top:16px}}
code{{background:#222;padding:2px 6px;border-radius:4px}}
</style></head><body>
<h1>Locked pick — out-of-sample</h1>
<div class="meta">
  <div><b>{LABEL}</b> · clock {CLOCK}</div>
  <div>Policy <code>{PRODUCT_POLICY_ID}</code> · frames {written} · median ball {med}px · ≥20px {ge20:.0%}</div>
  <div>Winners: {', '.join(f"{t['cam']} {t['share']*100:.0f}%" for t in top[:4])}</div>
  <div>Not Top Left gold window — this is a holdout clip.</div>
</div>
<video controls autoplay muted loop src="{ov_url}"></video>
<div><img src="{sheet_url}" alt="contact"></div>
</body></html>
"""
    (args.out / "index.html").write_text(html, encoding="utf-8")
    (args.out / "readme.md").write_text(
        f"# Locked OOS demo\n\n{LABEL}\n\n"
        f"- Overlay: `{raw_path.name}`\n"
        f"- Contact: `{sheet_path.name}`\n"
        f"- Open: http://127.0.0.1:8080/reports/eval_match2_v10/locked_oos_demo/\n",
        encoding="utf-8",
    )
    print(json.dumps(stats, indent=2), flush=True)
    print(f"wrote {args.out / 'index.html'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
