#!/usr/bin/env python3
"""Build pitchmap gallery: existing OOS clips + 5 random Match 2 windows.

Cuts synced multicam sources, detects (stride frames only), renders
video|pitch dual-pane, writes dropdown index under
reports/eval_match2_v10/locked_oos_pitchmap_gallery/.
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "gold_set"))

from demo_locked_oos_pitchmap import main as render_pitchmap_main  # noqa: E402
from eval_match2_4quad_multicam_survey import cache_dump_n  # noqa: E402
from multicam_select_policy import SURVEY_CAMS  # noqa: E402
from src.perception.rfdetr_local import load_ball_model  # noqa: E402

MATCH2_CAMS = [
    ("P1", ROOT / "data/raw/Match 2/Cam 3-P1.mp4"),
    ("P6", ROOT / "data/raw/Match 2/Cam 6-P6-002.mp4"),
    ("P7", ROOT / "data/raw/Match 2/Cam 11-P7-003.mp4"),
    ("P8", ROOT / "data/raw/Match 2/Cam 14-P8-001.mp4"),
    ("P10", ROOT / "data/raw/Match 2/Cam 8-P10-003.mp4"),
    ("P12", ROOT / "data/raw/Match 2/Cam 10-P12-001.mp4"),
    ("Cam4plus", ROOT / "data/raw/Match 2/Cam 4+-002.mp4"),
    ("Cam5plus", ROOT / "data/raw/Match 2/Cam 5+-004.mp4"),
]

EXISTING = [
    {
        "id": "bottom_right",
        "label": "Bottom Right (6:52)",
        "clock": "6:52–6:58",
        "stem": "quad_bottom_right_t00412.0s",
        "start_sec": 412.0,
        "source_dir": ROOT / "reports/eval_match2_v10/4quad_test/source",
        "cache": ROOT
        / "reports/eval_match2_v10/4quad_multicam_survey/det_cache_bottom_right_thr010.json",
        "out": ROOT / "reports/eval_match2_v10/locked_oos_demo_bottom_right_pitchmap",
    },
    {
        "id": "top_right",
        "label": "Top Right (2:05)",
        "clock": "2:05–2:10",
        "stem": "quad_top_right_t00125.0s",
        "start_sec": 125.0,
        "source_dir": ROOT / "reports/eval_match2_v10/4quad_test/source",
        "cache": ROOT
        / "reports/eval_match2_v10/4quad_multicam_survey/det_cache_top_right_thr010.json",
        "out": ROOT / "reports/eval_match2_v10/locked_oos_demo_top_right_pitchmap",
    },
]

AVOID_STARTS = [8.0, 26.0, 125.0, 412.0]
GALLERY = ROOT / "reports/eval_match2_v10/locked_oos_pitchmap_gallery"
SRC_DIR = GALLERY / "source"
CACHE_DIR = GALLERY / "det_cache"
CLIP_SEC = 6.0
DETECT_STRIDE = 2
SEED = 20260816
CKPT = ROOT / "models/v10_snaps/post_train/checkpoint.pth"


def fmt_clock(sec: float) -> str:
    total = int(round(float(sec)))
    return f"{total // 60}:{total % 60:02d}"


def video_duration_sec(path: Path) -> float:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(path)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    n = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
    cap.release()
    return n / fps


def pick_random_starts(n: int, dur: float, clip_sec: float, seed: int) -> list[float]:
    rng = random.Random(seed)
    lo, hi = 30.0, max(31.0, dur - clip_sec - 5.0)
    picks = []
    tries = 0
    while len(picks) < n and tries < 5000:
        tries += 1
        t = rng.uniform(lo, hi)
        if any(abs(t - a) < 20.0 for a in AVOID_STARTS):
            continue
        if any(abs(t - p) < 25.0 for p in picks):
            continue
        picks.append(round(t, 1))
    if len(picks) < n:
        raise RuntimeError(f"only picked {len(picks)}/{n} random starts")
    return sorted(picks)


def extract_synced(stem: str, start_sec: float, clip_sec: float) -> None:
    SRC_DIR.mkdir(parents=True, exist_ok=True)
    for cam, path in MATCH2_CAMS:
        dest = SRC_DIR / f"{stem}_{cam}.mp4"
        if dest.is_file() and dest.stat().st_size > 1000:
            continue
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-ss", f"{start_sec:.3f}", "-i", str(path),
            "-t", f"{clip_sec:.3f}",
            "-c:v", "libx264", "-crf", "18", "-preset", "veryfast", "-an",
            str(dest),
        ]
        print(f"extract {dest.name}", flush=True)
        subprocess.run(cmd, check=True)


def detect_clip(model, stem: str, stride: int) -> Path:
    from eval_match2_4quad_multicam_survey import (
        DETECT_H,
        DETECT_W,
        SIZE,
        dets_to_rows,
        read_resized,
        source_path,
    )
    from src.perception.ball_prelabel import BallPrelabelConfig, BallPrelabeler

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"det_cache_{stem}_thr010.json"
    if cache_path.is_file():
        print(f"reuse cache {cache_path.name}", flush=True)
        return cache_path
    cap0 = cv2.VideoCapture(str(SRC_DIR / f"{stem}_{SURVEY_CAMS[0]}.mp4"))
    n = int(cap0.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap0.release()
    print(f"detect {stem} n={n} stride={stride} (infer every {stride} frames)", flush=True)
    pre = BallPrelabeler(
        model,
        BallPrelabelConfig(threshold=0.10, use_sahi=False, topk=5, **SIZE),
        class_id=1,
    )
    caps = {}
    for cam in SURVEY_CAMS:
        path = source_path(SRC_DIR, stem, cam)
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"open failed {path}")
        caps[cam] = cap
    out = {cam: [] for cam in SURVEY_CAMS}
    try:
        for i in range(n):
            for cam in SURVEY_CAMS:
                frame = read_resized(caps[cam])
                if frame is None:
                    for c in SURVEY_CAMS:
                        out[c] = out[c][:i]
                    n = i
                    break
                if frame.shape[1] != DETECT_W or frame.shape[0] != DETECT_H:
                    frame = cv2.resize(frame, (DETECT_W, DETECT_H), interpolation=cv2.INTER_AREA)
                if i % stride == 0:
                    out[cam].append(dets_to_rows(pre.detect_bgr(frame)))
                else:
                    out[cam].append([])
            else:
                if i % 30 == 0:
                    print(f"  {stem} detect {i}/{n - 1}", flush=True)
                continue
            break
    finally:
        for cap in caps.values():
            cap.release()
    cache_dump_n(cache_path, out, len(out[SURVEY_CAMS[0]]))
    print(f"wrote {cache_path}", flush=True)
    return cache_path


def render_one(clip: dict, max_frames: int, stride: int, force: bool = False) -> dict:
    out = clip["out"]
    out.mkdir(parents=True, exist_ok=True)
    mp4 = out / f"{clip['stem']}_video_pitch.mp4"
    if (
        not force
        and mp4.is_file()
        and mp4.stat().st_size > 1000
        and (out / "index.html").is_file()
    ):
        print(f"reuse render {clip['id']}", flush=True)
        return {
            "id": clip["id"],
            "label": clip["label"],
            "clock": clip["clock"],
            "url": f"/{out.relative_to(ROOT).as_posix()}/{mp4.name}",
            "page": f"/{out.relative_to(ROOT).as_posix()}/",
        }
    if force and mp4.is_file():
        mp4.unlink()
    # call pitchmap via argv
    argv = [
        "demo_locked_oos_pitchmap.py",
        "--stem", clip["stem"],
        "--label", clip["label"],
        "--clock", clip["clock"],
        "--source-dir", str(clip["source_dir"]),
        "--cache", str(clip["cache"]),
        "--out", str(out),
        "--max-frames", str(max_frames),
        "--stride", str(stride),
    ]
    old = sys.argv
    try:
        sys.argv = argv
        rc = render_pitchmap_main()
    finally:
        sys.argv = old
    if rc != 0:
        raise RuntimeError(f"pitchmap failed for {clip['id']} rc={rc}")
    return {
        "id": clip["id"],
        "label": clip["label"],
        "clock": clip["clock"],
        "url": f"/{out.relative_to(ROOT).as_posix()}/{mp4.name}",
        "page": f"/{out.relative_to(ROOT).as_posix()}/",
    }


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
<title>Match 2 — video + pitch (x,y)</title>
<style>
:root {{ --bg:#121612; --panel:#1c241c; --text:#eef2ee; --muted:#9aab9a; --accent:#e8c547; }}
* {{ box-sizing: border-box; }}
body {{ margin:0; font-family:"IBM Plex Sans",Segoe UI,sans-serif; background:var(--bg); color:var(--text); }}
main {{ max-width: 1280px; margin: 0 auto; padding: 1.2rem 1.4rem 2rem; }}
h1 {{ font-family:"IBM Plex Serif",Georgia,serif; font-size:1.45rem; font-weight:600; margin:0 0 .35rem; }}
.sub {{ color:var(--muted); font-size:.9rem; line-height:1.45; margin-bottom:1rem; max-width:52rem; }}
.bar {{ display:flex; flex-wrap:wrap; gap:.75rem; align-items:center; margin-bottom:.9rem; }}
label {{ color:var(--muted); font-size:.85rem; }}
select {{
  background:var(--panel); color:var(--text); border:1px solid #3a4a3a;
  padding:.55rem .7rem; border-radius:6px; font-size:.95rem; min-width:min(100%, 420px);
}}
select:focus {{ outline:1px solid var(--accent); }}
video {{ width:100%; background:#000; border-radius:8px; display:block; }}
.meta {{ margin-top:.65rem; color:var(--muted); font-size:.82rem; }}
code {{ color:var(--accent); }}
</style>
</head><body><main>
<h1>Match 2 — locked pick + pitch (x,y)</h1>
<p class="sub">
  Dropdown: Bottom Right + Top Right OOS, plus 5 random Match 2 windows.
  Left panel = selected cam / ball; right = 2D pitch meters (sticky cam K=5 + emit N=3; FOV-approx until manual H).
</p>
<div class="bar">
  <label for="clip">Clip</label>
  <select id="clip">{options}</select>
</div>
<video id="player" controls autoplay muted loop></video>
<p class="meta" id="meta"></p>
<script>
const entries = {json.dumps(entries)};
const byId = Object.fromEntries(entries.map(e => [e.id, e]));
const sel = document.getElementById('clip');
const player = document.getElementById('player');
const meta = document.getElementById('meta');
function show(id) {{
  const e = byId[id];
  if (!e) return;
  player.src = e.url;
  player.play().catch(() => {{}});
  meta.innerHTML = `<b>${{e.label}}</b> · ${{e.clock}} · <code>${{e.id}}</code>`;
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
    print(f"wrote {path}", flush=True)
    return path


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-random", type=int, default=5)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--clip-sec", type=float, default=CLIP_SEC)
    p.add_argument("--stride", type=int, default=DETECT_STRIDE)
    p.add_argument("--max-frames", type=int, default=90)
    p.add_argument("--skip-detect", action="store_true")
    p.add_argument("--force", action="store_true", help="re-render pitchmap mp4s even if present")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    # 1) existing renders
    clips = []
    for row in EXISTING:
        clips.append(dict(row))

    # 2) random windows
    dur = video_duration_sec(MATCH2_CAMS[0][1])
    starts = pick_random_starts(args.n_random, dur, args.clip_sec, args.seed)
    print(f"match_dur={dur:.1f}s random_starts={starts}", flush=True)
    random_clips = []
    for t in starts:
        stem = f"rand_t{t:07.1f}s"
        cid = f"rand_{int(t)}"
        random_clips.append(
            {
                "id": cid,
                "label": f"Random {fmt_clock(t)}",
                "clock": f"{fmt_clock(t)}–{fmt_clock(t + args.clip_sec)}",
                "stem": stem,
                "start_sec": t,
                "source_dir": SRC_DIR,
                "cache": CACHE_DIR / f"det_cache_{stem}_thr010.json",
                "out": GALLERY / "clips" / cid,
            }
        )

    for clip in random_clips:
        extract_synced(clip["stem"], clip["start_sec"], args.clip_sec)

    if not args.skip_detect:
        model = load_ball_model(str(CKPT))
        for clip in random_clips:
            clip["cache"] = detect_clip(model, clip["stem"], args.stride)

    clips.extend(random_clips)

    # monkey: ensure pitchmap supports custom args — render all
    entries = []
    for clip in clips:
        entries.append(render_one(clip, args.max_frames, args.stride, force=args.force))

    write_gallery(entries)
    print(json.dumps({"gallery": str(GALLERY.relative_to(ROOT)), "n": len(entries)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
