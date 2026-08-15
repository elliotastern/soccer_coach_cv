#!/usr/bin/env python3
"""Build a visual dashboard for the ~0.95 proxy P/R (P10-selected frames).

Uses det_cache + Top Left gold. Green=GT, cyan=pred (TP), red=pred (FP).
Writes reports/eval_match2_v10/top_left_multicam_baseline/proxy_gallery/
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "gold_set"))

from eval_match2_top_left_multicam_baseline import (  # noqa: E402
    GOLD_CAM,
    N_FRAMES,
    P_CAMS,
    cache_load,
    filter_rows,
    load_top_left_gt,
    select_frame,
)
from eval_poc_ball_metrics import iou_xywh  # noqa: E402

OUT = ROOT / "reports/eval_match2_v10/top_left_multicam_baseline/proxy_gallery"
CACHE = ROOT / "reports/eval_match2_v10/top_left_multicam_baseline/det_cache_thr010.json"
GOLD_XML = ROOT / "data/processed/gold_sets/match2_4quad_top_left/gold/annotations.xml"
FRAMES = ROOT / "data/processed/gold_sets/match2_4quad_top_left/review/frames"
THR = 0.30
IOU = 0.5
MAX_SHOW = 60  # keep page light


def draw_box(im, box, color, label: str):
    x, y, w, h = [int(round(v)) for v in box]
    cv2.rectangle(im, (x, y), (x + w, y + h), color, 2)
    cv2.putText(
        im,
        label,
        (x, max(16, y - 6)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        1,
        cv2.LINE_AA,
    )


def classify(gt_boxes, pred):
    box, conf, _ = pred
    best = 0.0
    for g in gt_boxes:
        best = max(best, iou_xywh(box, g))
    return "TP" if best >= IOU else "FP", best


def main() -> int:
    if not CACHE.is_file():
        raise FileNotFoundError(f"missing {CACHE} — run multicam baseline first")
    dets = cache_load(CACHE)
    gt = load_top_left_gt(GOLD_XML)
    out_img = OUT / "frames"
    out_img.mkdir(parents=True, exist_ok=True)

    rows = []
    for i in range(N_FRAMES):
        cam_rows = {cam: filter_rows(dets[cam][i], THR) for cam in P_CAMS}
        cam, pred = select_frame(cam_rows, min_cams=1)
        if cam != GOLD_CAM or pred is None:
            continue
        g = gt.get(i, [])
        kind, iou = classify(g, pred)
        frame_path = FRAMES / f"{i:03d}.jpg"
        im = cv2.imread(str(frame_path))
        if im is None:
            continue
        for gb in g:
            draw_box(im, gb, (0, 220, 0), "GT")
        color = (255, 200, 0) if kind == "TP" else (0, 0, 255)
        draw_box(im, pred[0], color, f"{kind} {pred[1]:.2f}")
        rel = f"frames/{i:03d}_{kind}.jpg"
        cv2.imwrite(str(OUT / rel), im, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        rows.append(
            {
                "frame": i,
                "kind": kind,
                "conf": round(float(pred[1]), 3),
                "iou": round(float(iou), 3),
                "n_gt": len(g),
                "img": rel,
            }
        )

    tp = sum(1 for r in rows if r["kind"] == "TP")
    fp = sum(1 for r in rows if r["kind"] == "FP")
    # gallery: all FP + sample of TP
    fps = [r for r in rows if r["kind"] == "FP"]
    tps = [r for r in rows if r["kind"] == "TP"]
    step = max(1, len(tps) // max(1, MAX_SHOW - len(fps)))
    show = fps + tps[::step]
    show = show[:MAX_SHOW]
    show.sort(key=lambda r: r["frame"])

    p = tp / (tp + fp) if (tp + fp) else 0.0
    # recall vs GT on these proxy frames only
    n_gt = sum(len(gt.get(r["frame"], [])) for r in rows)
    # approx: each TP matches one GT on these frames
    r_proxy = tp / n_gt if n_gt else 0.0

    cards = []
    for r in show:
        cards.append(
            f'<figure class="{r["kind"].lower()}">'
            f'<img src="{r["img"]}" loading="lazy" alt="f{r["frame"]}"/>'
            f"<figcaption>f{r['frame']:03d} · {r['kind']} · "
            f"conf {r['conf']} · IoU {r['iou']}</figcaption></figure>"
        )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>Proxy P/R gallery — P10-selected @0.30</title>
<style>
body {{ font-family: ui-sans-serif, system-ui, sans-serif; margin: 0; background: #111; color: #eee; }}
header {{ padding: 1.25rem 1.5rem; border-bottom: 1px solid #333; position: sticky; top: 0; background: #111; }}
h1 {{ margin: 0 0 .35rem; font-size: 1.25rem; }}
.meta {{ color: #aaa; font-size: .9rem; line-height: 1.45; max-width: 52rem; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 12px; padding: 1rem; }}
figure {{ margin: 0; background: #1a1a1a; border: 1px solid #333; }}
figure.fp {{ border-color: #c44; }}
figure.tp {{ border-color: #2a6; }}
img {{ width: 100%; display: block; aspect-ratio: 16/9; object-fit: cover; }}
figcaption {{ padding: .4rem .6rem; font-size: .8rem; color: #ccc; }}
code {{ color: #8cf; }}
</style>
</head>
<body>
<header>
  <h1>Claimed ~0.95 proxy P/R — what it looks like</h1>
  <div class="meta">
    Only frames where <b>max_conf picks P10</b> (~{len(rows)} / {N_FRAMES}).
    Green = gold · Cyan/yellow = TP pred · Red = FP pred.<br/>
    This gallery subset: <b>P≈{p:.3f}</b> (tp={tp} fp={fp}) ·
    GT-on-proxy-frames R≈{r_proxy:.3f} (n_gt={n_gt}).<br/>
    <b>Not</b> full 6-cam system score — P7 wins many other frames (no gold there).<br/>
    Stack: v10 · thr 0.30 · size · topk2 · no SAHI · 6 P-cams max_conf.
  </div>
</header>
<div class="grid">
{"".join(cards)}
</div>
</body>
</html>
"""
    (OUT / "index.html").write_text(html, encoding="utf-8")
    (OUT / "manifest.json").write_text(
        json.dumps(
            {
                "n_proxy_frames": len(rows),
                "tp": tp,
                "fp": fp,
                "precision": p,
                "n_shown": len(show),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {OUT / 'index.html'} n_proxy={len(rows)} tp={tp} fp={fp}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
