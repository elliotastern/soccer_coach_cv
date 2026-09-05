#!/usr/bin/env python3
"""Build a Match 3 labeled strip from a quad pitchmap clip (seeded clear focus cam)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "gold_set"))

from eval_match2_top_left_multicam_baseline import cache_load  # noqa: E402
from src.mapping.match3_xy import load_calib, map_ball_box  # noqa: E402

DETECT_W = 1920
ACCEPT_CONF = 0.55
CLEAR_SIDE = 25.0
CAMS = ["P1", "P6", "P7", "P8", "P9", "P10", "P_Goal1", "P_Goal2"]
QUAD_SRC = ROOT / "reports/eval_match3/quad_pitchmap_gallery/source"
QUAD_CACHE = ROOT / "reports/eval_match3/quad_pitchmap_gallery/det_cache"


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


def seed_frame(rows, calib, accept_conf=ACCEPT_CONF, soft_seed_conf=None, soft_side=None) -> dict:
    if not rows:
        return {"gt_balls": [], "empty": True, "clear": False, "gold_xy": None}
    box, conf, side = rows[0]
    conf_f = float(conf)
    side_f = float(side)
    clear = side_f >= CLEAR_SIDE and conf_f >= float(accept_conf)
    soft_min_side = CLEAR_SIDE if soft_side is None else float(soft_side)
    soft = (
        soft_seed_conf is not None
        and (not clear)
        and side_f >= soft_min_side
        and conf_f >= float(soft_seed_conf)
    )
    if not clear and not soft:
        return {
            "gt_balls": [],
            "empty": True,
            "clear": False,
            "gold_xy": None,
            "prelabel": {
                "bbox": list(box),
                "conf": conf_f,
                "side": side_f,
            },
        }
    hit = map_ball_box(calib, box, conf_f, frame_wh=(DETECT_W, 1080))
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
        "clear": bool(clear),
        "gold_xy": None if hit is None else [hit["xy"][0], hit["xy"][1]],
        "gold_support": None if hit is None else hit["support"],
        "seed_conf": conf_f,
        "seed_side": side_f,
        "soft_seed": bool(soft),
        "prelabel": {
            "bbox": list(box),
            "conf": conf_f,
            "side": side_f,
        },
    }


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stem", required=True, help="e.g. quad_P8_t00087.0s")
    p.add_argument("--focus", required=True, help="focus cam id")
    p.add_argument("--pack", required=True, help="pack folder name under gold_sets")
    p.add_argument("--clock", required=True, help="e.g. 1:27-1:32")
    p.add_argument("--max-frames", type=int, default=300)
    p.add_argument(
        "--source-dir",
        type=Path,
        default=QUAD_SRC,
        help="dir with {stem}_{cam}.mp4",
    )
    p.add_argument(
        "--cache-dir",
        type=Path,
        default=QUAD_CACHE,
        help="dir with det_cache_{stem}_thr010.json",
    )
    p.add_argument("--accept-conf", type=float, default=ACCEPT_CONF)
    p.add_argument(
        "--soft-seed-conf",
        type=float,
        default=None,
        help="also seed gt boxes at this conf (clear=false) for blur/soft review",
    )
    p.add_argument(
        "--soft-side",
        type=float,
        default=None,
        help="min side px for soft seeds (default = clear side 25)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cache = args.cache_dir / f"det_cache_{args.stem}_thr010.json"
    video = args.source_dir / f"{args.stem}_{args.focus}.mp4"
    out = ROOT / "data/processed/gold_sets" / args.pack
    if not cache.is_file():
        raise FileNotFoundError(cache)
    if not video.is_file():
        raise FileNotFoundError(video)
    dets = cache_load(cache)
    n = min(len(dets[args.focus]), args.max_frames)
    calib = load_calib(args.focus)
    if calib is None:
        raise RuntimeError(f"missing calib {args.focus}")

    frames_dir = out / "review" / "frames"
    names = extract_frames(video, frames_dir, n)
    n = min(n, len(names))

    frames = []
    n_clear = 0
    n_gold = 0
    n_soft = 0
    for i in range(n):
        cam_payload = {}
        for cam in CAMS:
            rows = (dets.get(cam) or [None] * n)[i] or []
            if cam == args.focus:
                seed = seed_frame(
                    rows,
                    calib,
                    accept_conf=args.accept_conf,
                    soft_seed_conf=args.soft_seed_conf,
                    soft_side=args.soft_side,
                )
                if seed["clear"]:
                    n_clear += 1
                if seed.get("soft_seed"):
                    n_soft += 1
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
        "pack": args.pack,
        "focus_cam": args.focus,
        "stem": args.stem,
        "clock": args.clock,
        "source": str(video.resolve().relative_to(ROOT.resolve())).replace("\\", "/"),
        "det_cache": str(cache.resolve().relative_to(ROOT.resolve())).replace("\\", "/"),
        "detect_wh": [DETECT_W, 1080],
        "n_frames": n,
        "n_clear": n_clear,
        "n_gold_xy": n_gold,
        "n_soft_seed": n_soft,
        "seed": {
            "accept_conf": float(args.accept_conf),
            "soft_seed_conf": args.soft_seed_conf,
            "clear_side": CLEAR_SIDE,
            "note": (
                f"PROVISIONAL — {args.focus} clear dets mapped via {args.focus} H; "
                "human-correct before final"
            ),
        },
        "frames": frames,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "labels.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (out / "manifest.json").write_text(
        json.dumps(
            {
                "pack": args.pack,
                "focus_cam": args.focus,
                "clock": args.clock,
                "n_frames": n,
                "n_clear": n_clear,
                "n_gold_xy": n_gold,
                "n_soft_seed": n_soft,
                "labels": "labels.json",
                "review": "review/index.html",
                "source": payload["source"],
                "provisional": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (out / "README.md").write_text(
        "\n".join(
            [
                f"# Match 3 strip — {args.pack}",
                "",
                f"Focus **{args.focus}** · `{args.stem}` · {args.clock}",
                "",
                "Seeded clear-ball GT from focus-cam dets. Correct before final P_emit.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(
        f"wrote {out}: frames={n} clear={n_clear} soft_seed={n_soft} gold_xy={n_gold}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
