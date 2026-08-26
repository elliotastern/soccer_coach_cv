#!/usr/bin/env python3
"""Build static HTML with video-label stills + pitch for partial review."""
from __future__ import annotations

import base64
import json
from collections import Counter
from pathlib import Path

import cv2
import pandas as pd

from src.review.frame_sync import (
    draw_labels_on_frame,
    guess_video_for_run,
    load_H_inv,
    read_video_frame,
    rows_for_frame,
)

ROOT = Path(__file__).resolve().parents[1]
SNAP = ROOT / "data/output/full_match_2min_partial/P10-002"
OUT = ROOT / "reports/eval_match3/improve_eng_loop/review_partial.html"
HALF_L, HALF_W = 53.9 / 2, 34.84 / 2


def pitch_svg(rows: pd.DataFrame, scale: float = 10.0) -> str:
    w, h = int(53.9 * scale), int(34.84 * scale)

    def px(x, y):
        return (HALF_L + float(x)) * scale, (HALF_W - float(y)) * scale

    dots = [
        f'<rect x="0" y="0" width="{w}" height="{h}" fill="none" stroke="#86efac" stroke-width="2"/>'
    ]
    for _, r in rows.iterrows():
        x, y = px(r.Location_X, r.Location_Y)
        if int(r.Player_ID) == -1:
            dots.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="#f59e0b"/>')
        else:
            col = "#2563eb" if int(r.Team_ID) == 0 else "#dc2626"
            dots.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{col}"/>')
    return f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}">{"".join(dots)}</svg>'


def jpg_b64(bgr) -> str:
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
    if not ok:
        return ""
    return base64.b64encode(buf.tobytes()).decode("ascii")


def main():
    fd = SNAP / "frame_data.csv"
    evj = SNAP / "events.json"
    if not fd.is_file():
        raise SystemExit(f"missing {fd}")
    df = pd.read_csv(fd)
    events = json.loads(evj.read_text()).get("events") if evj.is_file() else []
    events = events or []
    counts = Counter(e.get("type") for e in events)
    video = guess_video_for_run("P10-002", ROOT)
    if video is None:
        raise SystemExit("no video for P10-002")
    H_inv, calib = load_H_inv(video)
    calib_wh = (calib or {}).get("image_wh") or [1920, 1080]

    ball_frames = sorted(df.loc[df.Player_ID == -1, "frame_id"].unique().tolist())
    if not ball_frames:
        ball_frames = sorted(df.frame_id.unique().tolist())[::50][:6]
    picks = []
    if ball_frames:
        for i in [0, len(ball_frames) // 4, len(ball_frames) // 2, 3 * len(ball_frames) // 4, -1]:
            f = int(ball_frames[i])
            if f not in picks:
                picks.append(f)
    picks = picks[:5]

    cards = []
    for fid in picks:
        rows = rows_for_frame(df, fid)
        frame, _, _ = read_video_frame(video, fid)
        # downscale for HTML size
        vis = draw_labels_on_frame(frame, rows, H_inv, calib_wh) if H_inv is not None else frame
        small = cv2.resize(vis, (1280, int(1280 * vis.shape[0] / vis.shape[1])))
        b64 = jpg_b64(small)
        cards.append(
            f"""<div class="card">
<h3>Frame {fid}</h3>
<div class="row">
<img src="data:image/jpeg;base64,{b64}" alt="frame {fid}"/>
{pitch_svg(rows)}
</div>
<p>{len(rows)} tracks · ball={int((rows.Player_ID==-1).sum())}</p>
</div>"""
        )

    html = f"""<!doctype html>
<html><head><meta charset="utf-8"/>
<title>Phase 1 review — video + pitch</title>
<style>
body{{font-family:ui-sans-serif,system-ui;margin:24px;background:#0b1220;color:#e5e7eb}}
h1{{margin:0 0 8px}} .meta{{opacity:.8;margin-bottom:16px}}
.card{{background:#111827;border:1px solid #1f2937;border-radius:12px;padding:16px;margin:16px 0}}
.row{{display:grid;grid-template-columns:1.4fr 1fr;gap:12px;align-items:start}}
img{{width:100%;border-radius:8px;background:#000}}
svg{{background:#052e16;border-radius:8px;width:100%;height:auto}}
.pill{{display:inline-block;background:#1f2937;padding:4px 10px;border-radius:999px;margin-right:8px}}
@media (max-width:900px){{.row{{grid-template-columns:1fr}}}}
</style></head><body>
<h1>Phase 1 review — video labels + pitch</h1>
<div class="meta">{video.name} · snapshot {SNAP}</div>
<div class="card">
<span class="pill">frames {df.frame_id.nunique()}</span>
<span class="pill">ball {(df.Player_ID==-1).sum()}</span>
<span class="pill">events {len(events)}</span>
<span class="pill">pass {counts.get('pass',0)}</span>
<span class="pill">dribble {counts.get('dribble',0)}</span>
</div>
<p>Orange = ball · blue/red = teams (projected via Match 3 H⁻¹). Scrub live in Streamlit.</p>
{''.join(cards)}
</body></html>"""
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(html, encoding="utf-8")
    print(OUT, "stills", picks)


if __name__ == "__main__":
    main()
