#!/usr/bin/env python3
"""Harvest best Match 2 frames for train labeling (+ CVAT pack).

Prefers master cams (Cam5plus / Cam4plus / P10), ranks by conf×size×cam bonus,
excludes Match 2 gold + prior harvests, writes editor + CVAT annotations.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.gold_set.harvest_match2_large_balls import (
    load_exclude_keys,
    save_crops,
    scan_cam,
    write_pack,
    write_progress,
)
from scripts.gold_set.save_accepted_harvest import write_cvat_images_xml
from src.perception.ball_prelabel import BallPrelabelConfig, BallPrelabeler
from src.perception.rfdetr_local import load_ball_model

OUT = ROOT / "data/processed/gold_sets/match2_train_label100"
SRC_EDITOR = ROOT / "data/processed/gold_sets/match1_1_100/review/editor.html"

# Best cams for PoC results (skip weaker P8 — faster + better quality)
BEST_CAMS = [
    ("Cam5plus", ROOT / "data/raw/Match 2/Cam 5+-004.mp4"),
    ("Cam4plus", ROOT / "data/raw/Match 2/Cam 4+-002.mp4"),
    ("P10", ROOT / "data/raw/Match 2/Cam 8-P10-003.mp4"),
]
CAM_BONUS = {"Cam5plus": 1.20, "Cam4plus": 1.15, "P10": 1.10, "P8": 1.00}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument(
        "--ball-checkpoint",
        default=str(ROOT / "models/v8_snaps/post_train/checkpoint.pth"),
    )
    p.add_argument("--target", type=int, default=100)
    p.add_argument("--pool", type=int, default=140, help="Candidates to collect before ranking")
    p.add_argument("--stride", type=int, default=45)
    p.add_argument("--min-side", type=float, default=32.0)
    p.add_argument("--min-conf", type=float, default=0.45)
    p.add_argument("--min-gap-sec", type=float, default=2.5)
    p.add_argument("--static-radius", type=float, default=90.0)
    p.add_argument("--max-per-spot", type=int, default=3)
    p.add_argument("--max-per-cam", type=int, default=40, help="Cap after ranking")
    p.add_argument("--per-cam-cap", type=int, default=55, help="Stop scanning a cam after N hits")
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def quality_score(row: dict) -> float:
    conf = float(row["max_ball_conf"])
    side = float(row["side"])
    bonus = CAM_BONUS.get(row["camera"], 1.0)
    # Prefer confident + large balls on master cams
    return conf * (side / 35.0) * bonus


def diversify_top(pool: list, target: int, max_per_cam: int, min_gap_sec: float) -> list:
    ranked = sorted(pool, key=quality_score, reverse=True)
    picked = []
    per_cam: dict[str, int] = {}
    last_t: dict[str, float] = {}
    for row in ranked:
        cam = row["camera"]
        if per_cam.get(cam, 0) >= max_per_cam:
            continue
        t = float(row["t_sec"])
        if cam in last_t and abs(t - last_t[cam]) < min_gap_sec:
            # allow if much better? skip for spacing
            continue
        # also avoid near-dupe vs already picked same cam
        too_close = False
        for prev in picked:
            if prev["camera"] != cam:
                continue
            if abs(float(prev["t_sec"]) - t) < min_gap_sec:
                too_close = True
                break
        if too_close:
            continue
        picked.append(row)
        per_cam[cam] = per_cam.get(cam, 0) + 1
        last_t[cam] = t
        if len(picked) >= target:
            break
    # stable order by camera priority then time
    cam_order = {n: i for i, (n, _) in enumerate(BEST_CAMS)}
    picked.sort(key=lambda r: (cam_order.get(r["camera"], 99), r["t_sec"]))
    return picked


def build_editor(out: Path, n: int):
    html = SRC_EDITOR.read_text(encoding="utf-8")
    html = html.replace("match1_1_100", "match2_train_label100")
    names = [f"{i:03d}.jpg" for i in range(n)]
    m = re.search(r"const GOLD100 = \{.*?maxFrame:\s*\d+\s*,\s*\};", html, re.S)
    if not m:
        raise RuntimeError("GOLD100 block missing")
    files_js = ",\n                ".join(json.dumps(x) for x in names)
    last = n - 1
    block = f"""const GOLD100 = {{
            files: [
                {files_js}
            ],
            base: '/data/processed/gold_sets/match2_train_label100/review/frames/',
            width: 1920,
            height: 1080,
            maxFrame: {last},
        }};"""
    html = html[: m.start()] + block + html[m.end() :]
    html = html.replace('max="99"', f'max="{last}"')
    html = html.replace("seekToFrame(99)", f"seekToFrame({last})")
    html = html.replace("if (currentFrame < 99)", f"if (currentFrame < {last})")
    html = html.replace(" / 99", f" / {last}")
    cm = re.search(r"\.container\s*\{[^}]*height:\s*([^;]+);", html)
    if not cm or "100vh" not in cm.group(1):
        raise RuntimeError(f"CSS corrupted: {cm.group(1) if cm else None}")
    (out / "review" / "editor.html").write_text(html, encoding="utf-8")


def write_cvat_full(out: Path, kept: list):
    """Copy review-sized images into cvat/images + annotations.xml (1920 boxes)."""
    import cv2

    cvat_img = out / "cvat" / "images"
    cvat_img.mkdir(parents=True, exist_ok=True)
    rows = []
    preds = []
    man = json.loads((out / "manifest.json").read_text())
    pred_list = json.loads((out / "review" / "preds.json").read_text())
    for i, row in enumerate(man["frames"]):
        src = out / "images" / row["image"]
        # Prefer full images; also write 1920 review copy for CVAT import ease
        rev = out / "review" / "frames" / f"{i:03d}.jpg"
        name = f"{i:03d}_{row['camera']}_f{row['frame_idx']:06d}.jpg"
        if rev.is_file():
            shutil.copy2(rev, cvat_img / name)
        elif src.is_file():
            im = cv2.imread(str(src))
            if im is None:
                raise RuntimeError(f"read fail {src}")
            im = cv2.resize(im, (1920, 1080), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(cvat_img / name), im, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        row_out = {**row, "image": name}
        rows.append(row_out)
        preds.append(pred_list[i])
    write_cvat_images_xml(out, rows, preds)
    # also copy track-style xml into gold for local editor save path
    gold = out / "gold"
    gold.mkdir(exist_ok=True)
    shutil.copy2(out / "prelabels" / "annotations.xml", gold / "annotations.xml")


def main() -> int:
    args = parse_args()
    if args.smoke:
        args.target = 8
        args.pool = 20
        args.start_sec = 33.0
        args.max_frames_per_cam = int(33 * 60 + 60 * 20)
        args.stride = 30

    # namespace expected by scan_cam
    if not hasattr(args, "start_sec"):
        args.start_sec = 0.0
    if not hasattr(args, "max_frames_per_cam"):
        args.max_frames_per_cam = 0
    if not hasattr(args, "use_vlm"):
        args.use_vlm = False
    args.start_sec = getattr(args, "start_sec", 0.0)
    args.max_frames_per_cam = getattr(args, "max_frames_per_cam", 0)
    args.use_vlm = False

    exclude = load_exclude_keys(
        [
            ROOT / "data/processed/gold_sets/match2_large_ball_harvest/keep.json",
        ],
        [
            ROOT / "data/processed/gold_sets/match2_gold_frames/manifest.json",
            ROOT / "data/processed/gold_sets/match2_large_ball_accepted50/manifest.json",
            ROOT / "data/processed/gold_sets/match2_large_ball_harvest/manifest.json",
            ROOT / "data/processed/gold_sets/match2_large_ball_harvest_batch2/manifest.json",
        ],
    )
    print(f"Excluding {len(exclude)} prior (camera, frame_idx) keys")
    write_progress({"status": "starting_best_train100", "kept": 0})

    model = load_ball_model(args.ball_checkpoint)
    pre = BallPrelabeler(
        model,
        BallPrelabelConfig(
            threshold=0.30,
            use_sahi=False,
            use_size_filter=True,
            topk=1,
            use_kalman=False,
            min_side=14,
        ),
    )

    # Temporarily raise harvest early-stop by scanning with high local target
    class _A:
        pass

    scan_args = argparse.Namespace(
        out=args.out,
        ball_checkpoint=args.ball_checkpoint,
        target=args.pool,
        stride=args.stride,
        min_side=args.min_side,
        min_conf=args.min_conf,
        min_gap_sec=args.min_gap_sec,
        static_radius=args.static_radius,
        max_per_spot=args.max_per_spot,
        max_frames_per_cam=args.max_frames_per_cam if args.smoke else 0,
        start_sec=args.start_sec if args.smoke else 0.0,
        use_vlm=False,
        smoke=args.smoke,
        pool_cap=args.per_cam_cap,
    )

    pool = []
    cams = BEST_CAMS[:1] if args.smoke else BEST_CAMS
    for name, path in cams:
        if not path.is_file():
            raise FileNotFoundError(path)
        pre.reset()
        hits = scan_cam(pre, name, path, scan_args, exclude)
        for h in hits:
            h["quality"] = quality_score(h)
        pool.extend(hits)
        print(f"  pool after {name}: {len(pool)}")
        if len(pool) >= args.pool and not args.smoke:
            break

    if not pool:
        raise RuntimeError("no candidates for best-train harvest")

    kept = diversify_top(pool, args.target, args.max_per_cam, args.min_gap_sec)
    if len(kept) < args.target:
        print(f"WARN: only {len(kept)}/{args.target} after diversify; filling from pool")
        have = {(r["camera"], r["frame_idx"]) for r in kept}
        for row in sorted(pool, key=quality_score, reverse=True):
            key = (row["camera"], row["frame_idx"])
            if key in have:
                continue
            kept.append(row)
            have.add(key)
            if len(kept) >= args.target:
                break

    if args.out.exists():
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True, exist_ok=True)
    # write_pack expects args with checkpoint fields
    pack_args = scan_args
    pack_args.out = args.out
    save_crops(args.out, kept)
    write_pack(args.out, kept, pack_args)
    write_cvat_full(args.out, kept)
    build_editor(args.out, len(kept))

    from collections import Counter

    cams_c = Counter(r["camera"] for r in kept)
    scores = [quality_score(r) for r in kept]
    man = json.loads((args.out / "manifest.json").read_text())
    man["name"] = "match2_train_label100"
    man["role"] = "train_label_candidates"
    man["display_name"] = "Match 2 train label 100 (best)"
    man["selection"] = {
        "method": "conf * (side/35) * cam_bonus",
        "cam_bonus": CAM_BONUS,
        "min_side": args.min_side,
        "min_conf": args.min_conf,
        "pool": len(pool),
        "cams": dict(cams_c),
        "mean_quality": float(np.mean(scores)),
        "mean_conf": float(np.mean([r["max_ball_conf"] for r in kept])),
        "mean_side": float(np.mean([r["side"] for r in kept])),
    }
    man["cvat"] = {
        "images_dir": "cvat/images",
        "annotations": "cvat/annotations.xml",
        "note": "Import into CVAT or use local editor; labels go to gold/annotations.xml",
    }
    man["short_url"] = "http://127.0.0.1:8080/match2-train100"
    (args.out / "manifest.json").write_text(json.dumps(man, indent=2))
    (args.out / "README.md").write_text(
        f"""# Match 2 train label 100 (best)

Best-ranked large-ball proposals for **train** labeling (eval gold stays separate).

| | |
|--|--|
| Frames | {len(kept)} |
| Cams | {dict(cams_c)} |
| Mean conf | {man['selection']['mean_conf']:.3f} |
| Mean side | {man['selection']['mean_side']:.0f} px |

## Label (local editor — same as accepted50)

http://127.0.0.1:8080/match2-train100

Ball → **N** → draw · **Save**

## CVAT

Import `cvat/images/` + `cvat/annotations.xml` (CVAT for images 1.1).
"""
    )
    write_progress({
        "status": "done_best_train100",
        "kept": len(kept),
        "out": str(args.out),
        "cams": dict(cams_c),
        "editor": "http://127.0.0.1:8080/match2-train100",
    })
    print(f"\nSelected {len(kept)} → {args.out}")
    print(f"cams={dict(cams_c)} mean_conf={man['selection']['mean_conf']:.3f} mean_side={man['selection']['mean_side']:.0f}")
    print("Open: http://127.0.0.1:8080/match2-train100")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
