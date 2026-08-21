#!/usr/bin/env python3
"""Match 3 holdout random pack: 5×5s clips, times disjoint from tune gallery.

Tune seeds (report-only): t00627.9, t00931.2, t01162.3, t01237.3, t01714.0
Holdout: seed 20260821, ≥25 s from every tune start, v12_hard dets.

Writes reports/eval_match3/pitchmap_gallery_holdout/ (source + det_cache + meta).
Does not change product fuse/hull/checkpoint.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "gold_set"))

import build_match3_pitchmap_gallery as b  # noqa: E402
from src.perception.rfdetr_local import load_ball_model  # noqa: E402

GALLERY = ROOT / "reports/eval_match3/pitchmap_gallery_holdout"
CKPT = ROOT / "models/v12_hard_snaps/post_train/checkpoint.pth"
SEED = 20260821
N_CLIPS = 5
CLIP_SEC = 5.0
STRIDE = 2
TUNE_STARTS = [627.9, 931.2, 1162.3, 1237.3, 1714.0]
MIN_GAP_S = 25.0


def pick_holdout_starts(n: int, dur: float, clip_sec: float, seed: int) -> list[float]:
    rng = random.Random(seed)
    lo, hi = 30.0, max(31.0, dur - clip_sec - 5.0)
    picks = []
    tries = 0
    while len(picks) < n and tries < 20000:
        tries += 1
        t = round(rng.uniform(lo, hi), 1)
        if any(abs(t - p) < MIN_GAP_S for p in picks):
            continue
        if any(abs(t - p) < MIN_GAP_S for p in TUNE_STARTS):
            continue
        picks.append(t)
    if len(picks) < n:
        raise RuntimeError(f"only picked {len(picks)}/{n} holdout starts")
    return sorted(picks)


def wire_gallery_paths() -> None:
    b.GALLERY = GALLERY
    b.SRC_DIR = GALLERY / "source"
    b.CACHE_DIR = GALLERY / "det_cache"


def write_meta(starts: list[float], caches: list[str]) -> None:
    GALLERY.mkdir(parents=True, exist_ok=True)
    entries = []
    for t in starts:
        stem = f"rand_t{t:07.1f}s"
        entries.append(
            {
                "id": f"rand_{int(t)}",
                "label": f"Holdout {b.fmt_clock(t)}",
                "clock": f"{b.fmt_clock(t)}–{b.fmt_clock(t + CLIP_SEC)}",
                "stem": stem,
                "start_sec": t,
                "role": "holdout",
            }
        )
    man = GALLERY / "manifest.json"
    man.write_text(json.dumps(entries, indent=2), encoding="utf-8")
    meta = {
        "gallery": str(GALLERY.relative_to(ROOT)),
        "tag": "v12_hard_holdout",
        "role": "holdout",
        "seed": SEED,
        "tune_starts_excluded": TUNE_STARTS,
        "min_gap_s": MIN_GAP_S,
        "checkpoint": str(CKPT.relative_to(ROOT)),
        "n": len(starts),
        "starts": starts,
        "det_caches": caches,
        "note": (
            "Held-out clear-ball baseline. Do not tune hull/checkpoint on these "
            "times in the same pass as building them."
        ),
    }
    (GALLERY / "ckpt_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"wrote {man}", flush=True)
    print(f"wrote {GALLERY / 'ckpt_meta.json'}", flush=True)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--skip-detect", action="store_true")
    p.add_argument("--n-random", type=int, default=N_CLIPS)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--stride", type=int, default=STRIDE)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not CKPT.is_file():
        raise FileNotFoundError(CKPT)
    for cam, path in b.MATCH3_CAMS:
        if not path.is_file():
            raise FileNotFoundError(f"Match 3 missing {cam}: {path}")
    wire_gallery_paths()
    b.SRC_DIR.mkdir(parents=True, exist_ok=True)
    b.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    dur = min(b.video_duration_sec(p) for _, p in b.MATCH3_CAMS)
    starts = pick_holdout_starts(args.n_random, dur, CLIP_SEC, args.seed)
    print(f"holdout_dur={dur:.1f}s starts={starts}", flush=True)
    for t in starts:
        stem = f"rand_t{t:07.1f}s"
        b.extract_synced(stem, t, CLIP_SEC)
    caches = []
    if args.skip_detect:
        for t in starts:
            stem = f"rand_t{t:07.1f}s"
            path = b.CACHE_DIR / f"det_cache_{stem}_thr010.json"
            if not path.is_file():
                raise FileNotFoundError(path)
            caches.append(str(path.relative_to(ROOT)))
    else:
        model = load_ball_model(str(CKPT))
        for t in starts:
            stem = f"rand_t{t:07.1f}s"
            path = b.detect_clip(model, stem, args.stride)
            caches.append(str(path.relative_to(ROOT)))
    write_meta(starts, caches)
    print(
        json.dumps(
            {
                "gallery": str(GALLERY.relative_to(ROOT)),
                "starts": starts,
                "n_caches": len(caches),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
