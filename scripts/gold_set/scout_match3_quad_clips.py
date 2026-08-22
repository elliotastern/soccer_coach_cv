#!/usr/bin/env python3
"""C1: Scout Match 3 for clips where P8/P9/P10 have thr-pass ball dets.

Samples sparse frames on quad cams, picks ≥1 window per cam, extracts synced
5s clips, detects, renders pitchmap under reports/eval_match3/quad_pitchmap_gallery/.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "gold_set"))

from build_match3_pitchmap_gallery import (  # noqa: E402
    CAM_IDS,
    MATCH3_CAMS,
    detect_clip,
    extract_synced,
    fmt_clock,
    render_one,
    video_duration_sec,
)
from multicam_select_policy import MATCH3_THR_BY_CAM, thr_for_cam  # noqa: E402
from raw_cam_id import load_match_raw  # noqa: E402
from src.perception.ball_prelabel import BallPrelabelConfig, BallPrelabeler  # noqa: E402
from src.perception.rfdetr_local import load_ball_model  # noqa: E402

MATCH3_RAW = ROOT / "data/raw/Match 3"
QUAD_CAMS = ["P8", "P9", "P10"]
GALLERY = ROOT / "reports/eval_match3/quad_pitchmap_gallery"
SRC_DIR = GALLERY / "source"
CACHE_DIR = GALLERY / "det_cache"
CKPT = ROOT / "models/v12_hard_snaps/post_train/checkpoint.pth"
CLIP_SEC = 5.0
SAMPLE_EVERY_SEC = 3.0
DETECT_W, DETECT_H = 1920, 1080
SIZE = dict(use_size_filter=True, min_side=4, max_side=240, use_kalman=False)


def resize_bgr(frame):
    if frame.shape[1] == DETECT_W and frame.shape[0] == DETECT_H:
        return frame
    return cv2.resize(frame, (DETECT_W, DETECT_H), interpolation=cv2.INTER_AREA)


def scout_cam(pre, path: Path, every_sec: float, thr: float) -> list[dict]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"open failed {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, int(round(every_sec * fps)))
    hits = []
    i = 0
    while i < n:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, fr = cap.read()
        if not ok:
            break
        fr = resize_bgr(fr)
        dets = pre.detect_bgr(fr)
        best = 0.0
        for d in dets:
            conf = float(getattr(d, "confidence", getattr(d, "conf", 0.0)))
            best = max(best, conf)
        if best >= thr:
            t = i / max(fps, 1e-6)
            hits.append({"t": round(t, 1), "conf": round(best, 3), "frame": i})
        i += step
        if len(hits) >= 40:
            break
    cap.release()
    return hits


def pick_starts(hits_by_cam: dict, clip_sec: float, dur: float) -> list[dict]:
    """≥1 start per quad cam; prefer high conf; keep starts ≥25s apart."""
    picks = []
    used_t = []

    def ok_t(t):
        if t < 20 or t > dur - clip_sec - 2:
            return False
        return all(abs(t - u) >= 25.0 for u in used_t)

    for cam in QUAD_CAMS:
        ranked = sorted(hits_by_cam.get(cam) or [], key=lambda h: -h["conf"])
        chosen = None
        for h in ranked:
            t = max(0.0, h["t"] - 1.0)
            if ok_t(t):
                chosen = {"cam": cam, "start_sec": round(t, 1), "hit_conf": h["conf"], "hit_t": h["t"]}
                break
        if chosen is None and ranked:
            t = max(20.0, min(dur - clip_sec - 2, ranked[0]["t"] - 1.0))
            chosen = {"cam": cam, "start_sec": round(t, 1), "hit_conf": ranked[0]["conf"], "hit_t": ranked[0]["t"]}
        if chosen is None:
            raise RuntimeError(f"no thr-pass hits for {cam}")
        picks.append(chosen)
        used_t.append(chosen["start_sec"])
    return picks


def write_gallery(entries: list[dict]) -> Path:
    GALLERY.mkdir(parents=True, exist_ok=True)
    (GALLERY / "manifest.json").write_text(json.dumps(entries, indent=2), encoding="utf-8")
    options = "\n".join(
        f'<option value="{e["id"]}" data-src="{e["url"]}">{e["label"]} · {e["clock"]}</option>'
        for e in entries
    )
    html = f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Match 3 — quad FOV pitch clips</title>
<style>
:root {{ --bg:#121612; --panel:#1c241c; --text:#eef2ee; --muted:#9aab9a; --accent:#e8c547; }}
body {{ margin:0; font-family:"IBM Plex Sans",Segoe UI,sans-serif; background:var(--bg); color:var(--text); }}
main {{ max-width: 1280px; margin: 0 auto; padding: 1.2rem 1.4rem 2rem; }}
h1 {{ font-family:"IBM Plex Serif",Georgia,serif; font-size:1.35rem; margin:0 0 .35rem; }}
.sub {{ color:var(--muted); font-size:.9rem; line-height:1.45; margin-bottom:1rem; max-width:52rem; }}
select {{ background:var(--panel); color:var(--text); border:1px solid #3a4a3a; padding:.55rem .7rem; border-radius:6px; min-width:min(100%, 420px); }}
video {{ width:100%; background:#000; border-radius:8px; display:block; }}
.meta {{ margin-top:.65rem; color:var(--muted); font-size:.82rem; }}
code {{ color:var(--accent); }}
</style></head><body><main>
<h1>Match 3 — quad FOV (P8 / P9 / P10)</h1>
<p class="sub">C1 clips scouted for thr-pass ball on quad cams (≥0.30). Fuse still emit ≥0.80 · hull 0.25 · Pitch 1 meters.</p>
<div><label for="clip">Clip </label><select id="clip">{options}</select></div>
<video id="player" controls autoplay muted loop></video>
<p class="meta" id="meta"></p>
<script>
const entries = {json.dumps(entries)};
const byId = Object.fromEntries(entries.map(e => [e.id, e]));
const sel = document.getElementById('clip');
const player = document.getElementById('player');
const meta = document.getElementById('meta');
function show(id) {{
  const e = byId[id]; if (!e) return;
  player.src = e.url; player.play().catch(() => {{}});
  const emit = (e.n_emit == null) ? "" : ` · emit ${{e.n_emit}}/${{e.n_frames}} · agree ${{e.n_agree}}`;
  meta.innerHTML = `<b>${{e.label}}</b> · ${{e.clock}} · focus <code>${{e.focus_cam}}</code>${{emit}}`;
  history.replaceState(null, '', '#' + id);
}}
sel.addEventListener('change', () => show(sel.value));
const initial = (location.hash || '').slice(1) || entries[0].id;
if (byId[initial]) sel.value = initial;
show(sel.value);
</script>
</main></body></html>
"""
    path = GALLERY / "index.html"
    path.write_text(html, encoding="utf-8")
    return path


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--every-sec", type=float, default=SAMPLE_EVERY_SEC)
    p.add_argument("--clip-sec", type=float, default=CLIP_SEC)
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--max-frames", type=int, default=150)
    p.add_argument("--skip-scout", action="store_true")
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    raw = load_match_raw(MATCH3_RAW)
    for cam in QUAD_CAMS:
        if cam not in raw:
            raise FileNotFoundError(cam)
    dur = min(video_duration_sec(raw[c]) for c in QUAD_CAMS)
    scout_path = GALLERY / "scout_hits.json"
    GALLERY.mkdir(parents=True, exist_ok=True)
    SRC_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if args.skip_scout and scout_path.is_file():
        scout = json.loads(scout_path.read_text(encoding="utf-8"))
        hits_by_cam = scout["hits_by_cam"]
        picks = scout["picks"]
        print(f"reuse scout {scout_path}", flush=True)
    else:
        print(f"load model {CKPT}", flush=True)
        model = load_ball_model(str(CKPT))
        pre = BallPrelabeler(
            model,
            BallPrelabelConfig(threshold=0.10, use_sahi=False, topk=5, **SIZE),
            class_id=1,
        )
        hits_by_cam = {}
        for cam in QUAD_CAMS:
            thr = thr_for_cam(MATCH3_THR_BY_CAM, cam)
            print(f"scout {cam} thr={thr} every={args.every_sec}s", flush=True)
            hits_by_cam[cam] = scout_cam(pre, raw[cam], args.every_sec, thr)
            print(f"  hits={len(hits_by_cam[cam])} top={hits_by_cam[cam][:3]}", flush=True)
        picks = pick_starts(hits_by_cam, args.clip_sec, dur)
        scout = {"dur": dur, "hits_by_cam": hits_by_cam, "picks": picks}
        scout_path.write_text(json.dumps(scout, indent=2), encoding="utf-8")
        print(f"wrote {scout_path}", flush=True)
        # keep model for detect
        model_for_detect = model
    if args.skip_scout:
        model_for_detect = load_ball_model(str(CKPT))

    # Monkeypatch build helpers' paths by local extract/detect with our dirs
    import build_match3_pitchmap_gallery as b

    b.SRC_DIR = SRC_DIR
    b.CACHE_DIR = CACHE_DIR
    b.GALLERY = GALLERY

    clips = []
    for p in picks:
        t = p["start_sec"]
        stem = f"quad_{p['cam']}_t{t:07.1f}s"
        cid = f"quad_{p['cam']}_{int(t)}"
        clips.append(
            {
                "id": cid,
                "label": f"{p['cam']} focus {fmt_clock(t)}",
                "clock": f"{fmt_clock(t)}–{fmt_clock(t + args.clip_sec)}",
                "stem": stem,
                "start_sec": t,
                "source_dir": SRC_DIR,
                "cache": CACHE_DIR / f"det_cache_{stem}_thr010.json",
                "out": GALLERY / "clips" / cid,
                "cams": CAM_IDS,
                "focus_cam": p["cam"],
                "hit_conf": p["hit_conf"],
            }
        )

    for clip in clips:
        extract_synced(clip["stem"], clip["start_sec"], args.clip_sec)

    for clip in clips:
        clip["cache"] = detect_clip(model_for_detect, clip["stem"], args.stride)

    entries = []
    for clip in clips:
        e = render_one(clip, args.max_frames, args.stride, force=args.force)
        e["focus_cam"] = clip["focus_cam"]
        e["hit_conf"] = clip["hit_conf"]
        entries.append(e)
    write_gallery(entries)
    summary = {
        "gallery": str(GALLERY.relative_to(ROOT)),
        "picks": picks,
        "entries": [
            {"id": e["id"], "focus_cam": e.get("focus_cam"), "n_emit": e.get("n_emit"), "n_agree": e.get("n_agree")}
            for e in entries
        ],
    }
    (GALLERY / "c1_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
