#!/usr/bin/env python3
"""Build Match 3 M1 gold strip from quad P10-focus clip.

Provisional GT = clear P10 dets (conf≥0.55, side≥25) mapped through P10 H.
Replace boxes in labels.json / review UI before treating P_emit as final.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "gold_set"))

from eval_match2_top_left_multicam_baseline import cache_load  # noqa: E402
from src.mapping.match3_xy import load_calib, map_ball_box  # noqa: E402

STEM = "quad_P10_t00031.0s"
FOCUS = "P10"
SRC = ROOT / "reports/eval_match3/quad_pitchmap_gallery/source"
CACHE = (
    ROOT
    / "reports/eval_match3/quad_pitchmap_gallery/det_cache"
    / f"det_cache_{STEM}_thr010.json"
)
OUT = ROOT / "data/processed/gold_sets/match3_quad_p10_31"
DETECT_W = 1920
ACCEPT_CONF = 0.55
CLEAR_SIDE = 25.0
CAMS = ["P1", "P6", "P7", "P8", "P9", "P10", "P_Goal1", "P_Goal2"]


def resize_w(frame, width=DETECT_W):
    h, w = frame.shape[:2]
    if w == width:
        return frame, 1.0
    scale = width / float(w)
    return cv2.resize(
        frame, (width, int(round(h * scale))), interpolation=cv2.INTER_AREA
    ), scale


def extract_frames(video: Path, dest: Path, n: int) -> list[str]:
    dest.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"open failed {video}")
    names = []
    for i in range(n):
        ok, fr = cap.read()
        if not ok:
            break
        fr, _ = resize_w(fr)
        name = f"{i:04d}.jpg"
        cv2.imwrite(str(dest / name), fr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        names.append(name)
    cap.release()
    return names


def seed_frame(rows, calib) -> dict:
    if not rows:
        return {"gt_balls": [], "empty": True, "clear": False, "gold_xy": None}
    box, conf, side = rows[0]
    clear = float(side) >= CLEAR_SIDE and float(conf) >= ACCEPT_CONF
    if not clear:
        return {
            "gt_balls": [],
            "empty": True,
            "clear": False,
            "gold_xy": None,
            "prelabel": {
                "bbox": list(box),
                "conf": float(conf),
                "side": float(side),
            },
        }
    hit = map_ball_box(calib, box, float(conf), frame_wh=(DETECT_W, 1080))
    return {
        "gt_balls": [
            {
                "x": float(box[0]),
                "y": float(box[1]),
                "w": float(box[2]),
                "h": float(box[3]),
            }
        ],
        "empty": False,
        "clear": True,
        "gold_xy": None if hit is None else [hit["xy"][0], hit["xy"][1]],
        "gold_support": None if hit is None else hit["support"],
        "seed_conf": float(conf),
        "seed_side": float(side),
    }


def write_review_html(out: Path, n: int) -> None:
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"/><title>Match3 M1 strip</title>
<style>
body{{margin:0;font:14px/1.4 system-ui;background:#111;color:#eee}}
main{{display:grid;grid-template-columns:1fr 280px;gap:12px;padding:12px}}
img{{max-width:100%;background:#000}}
.box{{position:absolute;border:2px solid #0f0}}
.stage{{position:relative;display:inline-block}}
button,select{{font:inherit;margin:4px 0}}
.muted{{color:#aaa}}
code{{color:#8cf}}
</style></head><body>
<main>
  <div>
    <div class="stage"><img id="im" alt="frame"/><div id="box" class="box"></div></div>
  </div>
  <div>
    <h1>Match3 M1 · P10 strip</h1>
    <p class="muted">Provisional seed from clear P10 dets. Correct before final P_emit.</p>
    <label>Frame <input id="idx" type="number" min="0" max="{n - 1}" value="0"/></label>
    <div id="meta"></div>
    <p><button id="prev">Prev</button> <button id="next">Next</button></p>
    <p><a href="labels.json">labels.json</a></p>
  </div>
</main>
<script>
const n={n};
let labels=null;
async function load(){{
  labels=await (await fetch('labels.json')).json();
  show(0);
}}
function show(i){{
  i=Math.max(0,Math.min(n-1,i|0));
  document.getElementById('idx').value=i;
  const fr=labels.frames[i];
  const im=document.getElementById('im');
  im.onload=()=>{{
    const b=document.getElementById('box');
    const g=(fr.cams.P10.gt_balls||[])[0];
    if(!g){{b.style.display='none';return}}
    const sx=im.clientWidth/1920, sy=im.clientHeight/1080;
    b.style.display='block';
    b.style.left=(g.x*sx)+'px'; b.style.top=(g.y*sy)+'px';
    b.style.width=(g.w*sx)+'px'; b.style.height=(g.h*sy)+'px';
  }};
  im.src='review/frames/'+fr.file;
  const xy=fr.cams.P10.gold_xy;
  document.getElementById('meta').innerHTML=
    `<div>clear=<b>${{fr.cams.P10.clear}}</b></div>`+
    `<div>gold_xy=<code>${{xy?xy.map(v=>v.toFixed(2)).join(', '):'null'}}</code></div>`+
    `<div class="muted">${{fr.file}}</div>`;
}}
document.getElementById('idx').onchange=e=>show(+e.target.value);
document.getElementById('prev').onclick=()=>show(+document.getElementById('idx').value-1);
document.getElementById('next').onclick=()=>show(+document.getElementById('idx').value+1);
load();
</script></body></html>
"""
    (out / "review" / "index.html").write_text(html, encoding="utf-8")


def main() -> int:
    if not CACHE.is_file():
        raise FileNotFoundError(CACHE)
    video = SRC / f"{STEM}_{FOCUS}.mp4"
    if not video.is_file():
        raise FileNotFoundError(video)
    dets = cache_load(CACHE)
    n = min(len(dets[FOCUS]), 300)
    calib = load_calib(FOCUS)
    if calib is None:
        raise RuntimeError(f"missing calib {FOCUS}")

    frames_dir = OUT / "review" / "frames"
    names = extract_frames(video, frames_dir, n)
    n = min(n, len(names))

    frames = []
    n_clear = 0
    n_gold = 0
    for i in range(n):
        cam_payload = {}
        for cam in CAMS:
            rows = (dets.get(cam) or [None] * n)[i] or []
            if cam == FOCUS:
                seed = seed_frame(rows, calib)
                if seed["clear"]:
                    n_clear += 1
                if seed.get("gold_xy"):
                    n_gold += 1
                cam_payload[cam] = seed
            else:
                top = None
                if rows:
                    box, conf, side = rows[0]
                    top = {
                        "bbox": list(box),
                        "conf": float(conf),
                        "side": float(side),
                    }
                cam_payload[cam] = {"prelabel": top}
        frames.append({"i": i, "file": names[i], "cams": cam_payload})

    payload = {
        "pack": "match3_quad_p10_31",
        "focus_cam": FOCUS,
        "stem": STEM,
        "clock": "0:31-0:36",
        "source": str(video.relative_to(ROOT)).replace("\\", "/"),
        "det_cache": str(CACHE.relative_to(ROOT)).replace("\\", "/"),
        "detect_wh": [DETECT_W, 1080],
        "n_frames": n,
        "n_clear": n_clear,
        "n_gold_xy": n_gold,
        "seed": {
            "accept_conf": ACCEPT_CONF,
            "clear_side": CLEAR_SIDE,
            "note": "PROVISIONAL — P10 clear dets mapped via P10 H; human-correct before final M1",
        },
        "frames": frames,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "labels.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (OUT / "manifest.json").write_text(
        json.dumps(
            {
                "pack": payload["pack"],
                "focus_cam": FOCUS,
                "clock": payload["clock"],
                "n_frames": n,
                "n_clear": n_clear,
                "n_gold_xy": n_gold,
                "labels": "labels.json",
                "review": "review/index.html",
                "source": payload["source"],
                "provisional": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (OUT / "README.md").write_text(
        "\n".join(
            [
                "# Match 3 M1 strip — quad P10 @ 0:31",
                "",
                "Synced 5 s clip (`quad_P10_t00031.0s`) with provisional clear-ball GT on **P10**.",
                "",
                "- `labels.json` — per-frame P10 `gt_balls` + `gold_xy` (Pitch 1 m)",
                "- `review/frames/` — local JPGs (not for git)",
                "- Score: `python3 scripts/gold_set/score_match3_ball_m1.py`",
                "",
                "Seed is detector-based. Correct boxes before claiming final P_emit.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    write_review_html(OUT, n)
    print(
        f"wrote {OUT}: frames={n} clear={n_clear} gold_xy={n_gold} "
        f"review={OUT / 'review' / 'index.html'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
