#!/usr/bin/env python3
"""Match 3: 5 random 5-second clips, locked ball pick + pitch (x, y).

Cuts synced multicam sources from data/raw/Match 3, detects ball,
renders video|pitch dual-pane under reports/eval_match3/pitchmap_gallery/.
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
from raw_cam_id import load_match_raw  # noqa: E402
from src.perception.rfdetr_local import load_ball_model  # noqa: E402

MATCH3_RAW = ROOT / "data/raw/Match 3"
_RAW = load_match_raw(MATCH3_RAW)
_CAM_ORDER = ["P1", "P6", "P7", "P8", "P9", "P10", "P_Goal1", "P_Goal2"]
MATCH3_CAMS = [(cid, _RAW[cid]) for cid in _CAM_ORDER]
CAM_IDS = [c for c, _ in MATCH3_CAMS]

GALLERY = ROOT / "reports/eval_match3/pitchmap_gallery"
SRC_DIR = GALLERY / "source"
CACHE_DIR = GALLERY / "det_cache"
CLIP_SEC = 5.0
N_CLIPS = 5
DETECT_STRIDE = 2
SEED = 20260817
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
    return n / max(fps, 1e-6)


def pick_random_starts(n: int, dur: float, clip_sec: float, seed: int) -> list[float]:
    rng = random.Random(seed)
    lo, hi = 30.0, max(31.0, dur - clip_sec - 5.0)
    picks = []
    tries = 0
    while len(picks) < n and tries < 5000:
        tries += 1
        t = rng.uniform(lo, hi)
        if any(abs(t - p) < 25.0 for p in picks):
            continue
        picks.append(round(t, 1))
    if len(picks) < n:
        raise RuntimeError(f"only picked {len(picks)}/{n} random starts")
    return sorted(picks)


def extract_synced(stem: str, start_sec: float, clip_sec: float) -> None:
    SRC_DIR.mkdir(parents=True, exist_ok=True)
    for cam, path in MATCH3_CAMS:
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
    first = source_path(SRC_DIR, stem, CAM_IDS[0])
    cap0 = cv2.VideoCapture(str(first))
    n = int(cap0.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap0.release()
    print(f"detect {stem} n={n} stride={stride}", flush=True)
    pre = BallPrelabeler(
        model,
        BallPrelabelConfig(threshold=0.10, use_sahi=False, topk=5, **SIZE),
        class_id=1,
    )
    caps = {}
    for cam in CAM_IDS:
        path = source_path(SRC_DIR, stem, cam)
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"open failed {path}")
        caps[cam] = cap
    out = {cam: [] for cam in CAM_IDS}
    try:
        for i in range(n):
            for cam in CAM_IDS:
                frame = read_resized(caps[cam])
                if frame is None:
                    for c in CAM_IDS:
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
    cache_dump_n(cache_path, out, len(out[CAM_IDS[0]]))
    print(f"wrote {cache_path}", flush=True)
    return cache_path


def gallery_entry(clip: dict, out: Path, mp4: Path) -> dict:
    stats_path = out / "stats.json"
    extra = {}
    if stats_path.is_file():
        extra = json.loads(stats_path.read_text(encoding="utf-8"))
    out_abs = out if out.is_absolute() else (ROOT / out)
    mp4_abs = mp4 if mp4.is_absolute() else (out_abs / mp4.name)
    rel = out_abs.resolve().relative_to(ROOT.resolve())
    return {
        "id": clip["id"],
        "label": clip["label"],
        "clock": clip["clock"],
        "url": f"/{rel.as_posix()}/{mp4_abs.name}?v=fpost",
        "page": f"/{rel.as_posix()}/",
        "n_frames": extra.get("n_frames"),
        "n_emit": extra.get("n_emit"),
        "n_agree": extra.get("n_agree"),
        "policy": extra.get("policy"),
    }


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
        return gallery_entry(clip, out, mp4)
    if force and mp4.is_file():
        mp4.unlink()
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
        "--cams", *CAM_IDS,
    ]
    old = sys.argv
    try:
        sys.argv = argv
        rc = render_pitchmap_main()
    finally:
        sys.argv = old
    if rc != 0:
        raise RuntimeError(f"pitchmap failed for {clip['id']} rc={rc}")
    return gallery_entry(clip, out, mp4)


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
<title>Match 3 — video + pitch (x,y)</title>
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
<h1>Match 3 — fused pitch (x,y)</h1>
<p class="sub">
  Five random 5-second windows from <code>data/raw/Match 3</code>
  (P1, P6, P7, P8, P9, P10, P_Goal1, P_Goal2).
  Left = selected cam / ball; right = Pitch 1 meters from 4-click H
  (bbox foot, 4 m fuse, emit ≥ 0.80, F0 detect-tick hold).
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
  const emit = (e.n_emit == null) ? "" : ` · emit ${{e.n_emit}}/${{e.n_frames}} · agree ${{e.n_agree}}`;
  meta.innerHTML = `<b>${{e.label}}</b> · ${{e.clock}} · <code>${{e.id}}</code>${{emit}}`;
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
    p.add_argument("--n-random", type=int, default=N_CLIPS)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--clip-sec", type=float, default=CLIP_SEC)
    p.add_argument("--stride", type=int, default=DETECT_STRIDE)
    p.add_argument("--max-frames", type=int, default=150)
    p.add_argument("--skip-detect", action="store_true")
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    for cam, path in MATCH3_CAMS:
        if not path.is_file():
            raise FileNotFoundError(f"Match 3 missing {cam}: {path}")
    dur = min(video_duration_sec(p) for _, p in MATCH3_CAMS)
    starts = pick_random_starts(args.n_random, dur, args.clip_sec, args.seed)
    print(f"match3_dur={dur:.1f}s random_starts={starts}", flush=True)
    clips = []
    for t in starts:
        stem = f"rand_t{t:07.1f}s"
        cid = f"rand_{int(t)}"
        clips.append(
            {
                "id": cid,
                "label": f"Random {fmt_clock(t)}",
                "clock": f"{fmt_clock(t)}–{fmt_clock(t + args.clip_sec)}",
                "stem": stem,
                "start_sec": t,
                "source_dir": SRC_DIR,
                "cache": CACHE_DIR / f"det_cache_{stem}_thr010.json",
                "out": GALLERY / "clips" / cid,
                "cams": CAM_IDS,
            }
        )
    for clip in clips:
        extract_synced(clip["stem"], clip["start_sec"], args.clip_sec)
    if args.skip_detect:
        missing = [str(c["cache"]) for c in clips if not Path(c["cache"]).is_file()]
        if missing:
            raise FileNotFoundError(f"--skip-detect but no cache: {missing}")
    else:
        model = load_ball_model(str(CKPT))
        for clip in clips:
            clip["cache"] = detect_clip(model, clip["stem"], args.stride)
    entries = []
    for clip in clips:
        entries.append(render_one(clip, args.max_frames, args.stride, force=args.force))
    write_gallery(entries)
    print(json.dumps({"gallery": str(GALLERY.relative_to(ROOT)), "n": len(entries), "starts": starts}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
