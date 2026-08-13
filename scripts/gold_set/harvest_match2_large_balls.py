#!/usr/bin/env python3
"""Harvest Match 2 large-ball frames for low-effort labeling.

Scans Cam5plus / Cam4plus, keeps RF-DETR proposals with min side >= 40 px,
optional VLM yes/no on a crop, writes a Gold100-style review pack.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from annotation.cvat_xml_generator import create_cvat_xml
from scripts.gold_set.build_match_gold100 import to_ann_tracked, write_coco
from src.perception.ball_prelabel import BallPrelabelConfig, BallPrelabeler
from src.perception.rfdetr_local import load_ball_model
from src.state.types import Detection

CAMS = [
    ("P10", ROOT / "data/raw/Match 2/Cam 8-P10-003.mp4"),
    ("Cam4plus", ROOT / "data/raw/Match 2/Cam 4+-002.mp4"),
    ("P8", ROOT / "data/raw/Match 2/Cam 14-P8-001.mp4"),
    ("Cam5plus", ROOT / "data/raw/Match 2/Cam 5+-004.mp4"),
]
DEFAULT_OUT = ROOT / "data/processed/gold_sets/match2_large_ball_harvest"
PROGRESS_PATH = ROOT / "reports/match2_large_ball_harvest/progress.json"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument(
        "--ball-checkpoint",
        default=str(ROOT / "models/v8_snaps/post_train/checkpoint.pth"),
    )
    p.add_argument("--target", type=int, default=150)
    p.add_argument("--stride", type=int, default=60, help="Detect every Nth frame")
    p.add_argument("--min-side", type=float, default=28.0)
    p.add_argument("--min-conf", type=float, default=0.40)
    p.add_argument("--min-gap-sec", type=float, default=2.0)
    p.add_argument("--static-radius", type=float, default=80.0)
    p.add_argument("--max-per-spot", type=int, default=2)
    p.add_argument("--max-frames-per-cam", type=int, default=0, help="0 = all")
    p.add_argument("--start-sec", type=float, default=0.0)
    p.add_argument("--use-vlm", action="store_true")
    p.add_argument("--smoke", action="store_true", help="Short Cam5plus slice only")
    p.add_argument(
        "--exclude-keep",
        type=Path,
        action="append",
        default=None,
        help="keep.json path(s) — exclude accepted+rejected (camera, frame_idx)",
    )
    p.add_argument(
        "--exclude-manifest",
        type=Path,
        action="append",
        default=None,
        help="Prior harvest manifest.json to exclude all its frames",
    )
    return p.parse_args()


def load_exclude_keys(keep_paths: list[Path], manifest_paths: list[Path]) -> set[tuple[str, int]]:
    keys: set[tuple[str, int]] = set()
    for keep_path in keep_paths:
        if not keep_path.is_file():
            continue
        keep = json.loads(keep_path.read_text())
        pack = Path(keep.get("pack", keep_path.parent))
        if not pack.is_absolute():
            pack = ROOT / pack
        man_path = pack / "manifest.json"
        if not man_path.is_file():
            continue
        frames = {int(r["strip_frame"]): r for r in json.loads(man_path.read_text())["frames"]}
        for idx in list(keep.get("accepted", [])) + list(keep.get("rejected", [])):
            row = frames.get(int(idx))
            if row:
                keys.add((row["camera"], int(row["frame_idx"])))
    for man_path in manifest_paths:
        if not man_path.is_file():
            continue
        for row in json.loads(man_path.read_text())["frames"]:
            keys.add((row["camera"], int(row["frame_idx"])))
    return keys


def write_progress(payload: dict):
    PROGRESS_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload["updated_unix"] = time.time()
    PROGRESS_PATH.write_text(json.dumps(payload, indent=2))


def ball_center(det: Detection):
    x, y, w, h = det.bbox
    return x + w / 2, y + h / 2


def ball_side(det: Detection) -> float:
    return min(det.bbox[2], det.bbox[3])


def too_static(cx: float, cy: float, kept: list, radius: float, max_per_spot: int) -> bool:
    n = 0
    for row in kept:
        det = row["detections"][0]
        px, py = ball_center(det)
        if (cx - px) ** 2 + (cy - py) ** 2 <= radius * radius:
            n += 1
            if n >= max_per_spot:
                return True
    return False


def on_pitch(frame, det: Detection, min_green: float = 0.35) -> bool:
    h, w = frame.shape[:2]
    x, y, bw, bh = det.bbox
    pad = int(max(bw, bh) * 1.5)
    x1 = int(max(0, x - pad))
    y1 = int(max(0, y - pad))
    x2 = int(min(w, x + bw + pad))
    y2 = int(min(h, y + bh + pad))
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return False
    b, g, r = cv2.split(crop)
    green = (g > r) & (g > b) & (g > 40)
    return float(np.mean(green)) >= min_green


def pick_large_ball(dets, min_side: float, min_conf: float):
    balls = [
        d for d in dets
        if d.class_name == "ball"
        and d.confidence >= min_conf
        and ball_side(d) >= min_side
    ]
    if not balls:
        return None
    return max(balls, key=lambda d: (d.confidence, ball_side(d)))


def crop_ball(frame, det: Detection, pad: float = 2.5):
    h, w = frame.shape[:2]
    x, y, bw, bh = det.bbox
    cx, cy = x + bw / 2, y + bh / 2
    side = max(bw, bh) * pad
    x1 = int(max(0, cx - side / 2))
    y1 = int(max(0, cy - side / 2))
    x2 = int(min(w, cx + side / 2))
    y2 = int(min(h, cy + side / 2))
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]


def vlm_is_ball(crop_bgr) -> tuple[bool, str]:
    key = os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
    if not key:
        return True, "no_vlm_key"
    ok, buf = cv2.imencode(".jpg", crop_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    if not ok:
        return True, "encode_fail"
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    prompt = (
        "Is the main object in this crop a soccer ball? "
        "Answer YES or NO only."
    )
    if os.getenv("OPENAI_API_KEY"):
        return _openai_yes_no(b64, prompt)
    return _anthropic_yes_no(b64, prompt)


def _openai_yes_no(b64: str, prompt: str) -> tuple[bool, str]:
    body = json.dumps({
        "model": "gpt-4o-mini",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                },
            ],
        }],
        "max_tokens": 8,
    }).encode()
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=body,
        headers={
            "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
        text = data["choices"][0]["message"]["content"].strip().upper()
        return text.startswith("YES"), text
    except (urllib.error.URLError, KeyError, TimeoutError) as exc:
        return True, f"vlm_error:{exc}"


def _anthropic_yes_no(b64: str, prompt: str) -> tuple[bool, str]:
    body = json.dumps({
        "model": "claude-3-5-haiku-latest",
        "max_tokens": 8,
        "messages": [{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": b64,
                    },
                },
                {"type": "text", "text": prompt},
            ],
        }],
    }).encode()
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=body,
        headers={
            "x-api-key": os.environ["ANTHROPIC_API_KEY"],
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
        text = data["content"][0]["text"].strip().upper()
        return text.startswith("YES"), text
    except (urllib.error.URLError, KeyError, TimeoutError) as exc:
        return True, f"vlm_error:{exc}"


def scan_cam(pre, name: str, path: Path, args, exclude_keys: set) -> list:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"open failed: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 60.0)
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    min_gap = int(round(args.min_gap_sec * fps))
    last_keep = -10**9
    kept = []
    scanned = 0
    i = 0
    if args.start_sec > 0:
        i = int(round(args.start_sec * fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(i))
    print(f"\n=== {name} {path.name} fps={fps:.1f} n={n_total} ===")
    while True:
        if args.max_frames_per_cam and i >= args.max_frames_per_cam:
            break
        ok, frame = cap.read()
        if not ok:
            break
        if i % args.stride != 0:
            i += 1
            continue
        if (name, i) in exclude_keys:
            i += 1
            continue
        scanned += 1
        dets = pre.detect_bgr(frame)
        det = pick_large_ball(dets, args.min_side, args.min_conf)
        if det is None or (i - last_keep) < min_gap:
            if scanned % 25 == 0:
                write_progress({
                    "status": "scanning",
                    "cam": name,
                    "frame": i,
                    "scanned": scanned,
                    "kept_this_cam": len(kept),
                })
            i += 1
            continue
        cx, cy = ball_center(det)
        if too_static(cx, cy, kept, args.static_radius, args.max_per_spot):
            i += 1
            continue
        if not on_pitch(frame, det):
            i += 1
            continue
        crop = crop_ball(frame, det)
        vlm_ok, vlm_note = True, "skipped"
        if args.use_vlm and crop is not None:
            vlm_ok, vlm_note = vlm_is_ball(crop)
        if not vlm_ok:
            print(f"  skip f={i} vlm={vlm_note}")
            i += 1
            continue
        last_keep = i
        kept.append({
            "camera": name,
            "video_path": str(path),
            "video_rel": path.name,
            "frame_idx": i,
            "t_sec": i / fps,
            "fps": fps,
            "width": frame.shape[1],
            "height": frame.shape[0],
            "stratum": "large_ball",
            "n_ball": 1,
            "n_player": 0,
            "max_ball_conf": float(det.confidence),
            "side": float(ball_side(det)),
            "vlm": vlm_note,
            "detections": [det],
            "_frame": frame.copy(),
            "_crop": crop,
        })
        print(
            f"  keep {len(kept)} f={i} t={i/fps:.1f}s "
            f"conf={det.confidence:.3f} side={ball_side(det):.0f} vlm={vlm_note}"
        )
        if getattr(args, "pool_cap", 0) and len(kept) >= int(args.pool_cap):
            i += 1
            break
        i += 1
    cap.release()
    print(f"  {name}: scanned={scanned} kept={len(kept)}")
    return kept


def save_crops(out_dir: Path, rows: list):
    crop_dir = out_dir / "crops"
    crop_dir.mkdir(parents=True, exist_ok=True)
    thumbs = []
    for i, row in enumerate(rows):
        crop = row.pop("_crop", None)
        if crop is None:
            continue
        path = crop_dir / f"{i:03d}_{row['camera']}_s{row['side']:.0f}_c{row['max_ball_conf']:.2f}.jpg"
        cv2.imwrite(str(path), crop, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        thumbs.append(cv2.resize(crop, (160, 160)))
    if not thumbs:
        return
    cols = 10
    rows_n = (len(thumbs) + cols - 1) // cols
    sheet = np.zeros((rows_n * 160, cols * 160, 3), dtype=np.uint8)
    for i, thumb in enumerate(thumbs):
        r, c = divmod(i, cols)
        sheet[r * 160:(r + 1) * 160, c * 160:(c + 1) * 160] = thumb
    cv2.imwrite(str(out_dir / "crops_contact.jpg"), sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 85])


def write_pack(out_dir: Path, rows: list, args):
    from scripts.gold_set.harden_review_pack import (
        clamp_xml_boxes,
        validate_pack,
        write_review_frames,
        write_strip_from_frames,
    )

    images_dir = out_dir / "images"
    prelabels_dir = out_dir / "prelabels"
    review_dir = out_dir / "review"
    for d in (images_dir, prelabels_dir, review_dir):
        d.mkdir(parents=True, exist_ok=True)

    image_names = []
    tracked_by_frame = {}
    manifest_rows = []
    review_preds = []
    next_track_id = 1
    for strip_i, row in enumerate(rows):
        frame = row.pop("_frame")
        name = f"f{row['frame_idx']:06d}_{row['camera']}.jpg"
        cv2.imwrite(str(images_dir / name), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        image_names.append(name)
        strip_h = 1080
        strip_w = 1920
        sx, sy = strip_w / row["width"], strip_h / row["height"]
        scaled = []
        balls = []
        for det in row["detections"]:
            x, y, bw, bh = det.bbox
            box = (x * sx, y * sy, bw * sx, bh * sy)
            scaled.append(Detection(
                class_id=det.class_id,
                confidence=det.confidence,
                bbox=box,
                class_name=det.class_name,
            ))
            balls.append({
                "bbox": [box[0], box[1], box[2], box[3]],
                "confidence": float(det.confidence),
                "side": float(min(box[2], box[3])),
                "side_fullres": float(row["side"]),
            })
        tracked = to_ann_tracked(scaled, next_track_id)
        next_track_id += max(len(tracked), 1)
        tracked_by_frame[strip_i] = tracked
        review_preds.append({"frame": strip_i, "balls": balls})
        manifest_rows.append({
            "strip_frame": strip_i,
            "camera": row["camera"],
            "video_rel": row["video_rel"],
            "video_path": row["video_path"],
            "frame_idx": row["frame_idx"],
            "t_sec": row["t_sec"],
            "image": name,
            "width": row["width"],
            "height": row["height"],
            "stratum": "large_ball",
            "n_ball": 1,
            "n_player": 0,
            "max_ball_conf": row["max_ball_conf"],
            "side": row["side"],
            "vlm": row["vlm"],
        })

    write_coco(rows, image_names, prelabels_dir / "annotations.coco.json")
    xml = create_cvat_xml(
        video_path=str(review_dir / "strip_100.mp4"),
        tracked_objects_by_frame=tracked_by_frame,
        events=[],
        video_metadata={
            "width": 1920,
            "height": 1080,
            "fps": 10.0,
            "frame_count": len(rows),
        },
    )
    (prelabels_dir / "annotations.xml").write_text(xml, encoding="utf-8")
    manifest = {
        "name": "match2_large_ball_harvest",
        "n_frames": len(rows),
        "min_side": args.min_side,
        "min_conf": args.min_conf,
        "stride": args.stride,
        "checkpoint": args.ball_checkpoint,
        "use_vlm": args.use_vlm,
        "frames": manifest_rows,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (review_dir / "preds.json").write_text(json.dumps(review_preds, indent=2))
    names, rw, rh = write_review_frames(out_dir.resolve())
    clamp_xml_boxes(out_dir.resolve(), rw, rh)
    write_strip_from_frames(out_dir.resolve(), len(names), fps=10)
    validate_pack(out_dir.resolve())
    (out_dir / "README.md").write_text(
        f"""# Match 2 large-ball harvest ({len(rows)} frames)

RF-DETR large-ball proposals (min side >= {args.min_side} px).
Accept with A, reject with D, save keep list.

```bash
python3 serve_viewer.py
```

Open: http://127.0.0.1:8080/match2-harvest
"""
    )


def main():
    args = parse_args()
    if args.smoke:
        args.target = 8
        args.start_sec = 33.0
        args.max_frames_per_cam = int(33 * 60 + 60 * 12)
        args.stride = 30
        args.min_gap_sec = 1.0
        args.min_side = 28.0
        args.min_conf = 0.40
    write_progress({"status": "starting", "kept": 0})
    exclude_keys = load_exclude_keys(args.exclude_keep or [], args.exclude_manifest or [])
    if exclude_keys:
        print(f"Excluding {len(exclude_keys)} prior (camera, frame_idx) keys")
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
            max_side=120,
        ),
    )
    cams = CAMS[:1] if args.smoke else CAMS
    kept = []
    for name, path in cams:
        if not path.is_file():
            raise FileNotFoundError(path)
        pre.reset()
        hits = scan_cam(pre, name, path, args, exclude_keys)
        kept.extend(hits)
        if len(kept) >= args.target and not args.smoke:
            kept = kept[: args.target]
            break
    if not kept:
        write_progress({"status": "empty", "kept": 0})
        raise RuntimeError("no large-ball frames harvested")
    args.out.mkdir(parents=True, exist_ok=True)
    save_crops(args.out, kept)
    write_pack(args.out, kept, args)
    rel = args.out if not args.out.is_absolute() else args.out.relative_to(ROOT)
    rel_s = rel.as_posix().lstrip("./")
    editor = f"http://127.0.0.1:8080/match2-harvest?pack=/{rel_s}"
    write_progress({
        "status": "done",
        "kept": len(kept),
        "out": str(args.out),
        "mean_side": float(np.mean([r["side"] for r in kept])),
        "mean_conf": float(np.mean([r["max_ball_conf"] for r in kept])),
        "editor": editor,
    })
    print(f"\nHarvested {len(kept)} frames → {args.out}")
    print(f"Review: python3 serve_viewer.py → {editor}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
