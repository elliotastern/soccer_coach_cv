#!/usr/bin/env python3
"""Build a fresh 100-frame Match pack for CVAT correction (excludes Gold100).

Random sample + RF-DETR prelabels (people + enhance_ball stack).
Output: data/processed/gold_sets/math_1_training/
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

from annotation.cvat_xml_generator import create_cvat_xml  # noqa: E402
from scripts.gold_set.build_match_gold100 import (  # noqa: E402
    DEFAULT_MATCH_DIR,
    discover_videos,
    read_frame,
    require_dir,
    require_file,
    resize_for_strip,
    sample_match_candidates,
    to_ann_tracked,
    write_coco,
    write_strip_video,
)
from src.perception.rfdetr_local import LocalRFDETRDetector  # noqa: E402
from src.state.types import Detection  # noqa: E402

DEFAULT_OUT = ROOT / "data/processed/gold_sets/math_1_training"
DEFAULT_EXCLUDE = ROOT / "data/processed/gold_sets/match1_1_100/manifest.json"


def load_exclude_keys(manifest_path: Path) -> set[tuple[str, int]]:
    if not manifest_path.is_file():
        return set()
    data = json.loads(manifest_path.read_text())
    keys = set()
    for row in data.get("frames", []):
        rel = row.get("video_rel") or ""
        keys.add((rel, int(row["frame_idx"])))
        keys.add((str(row.get("video_path", "")), int(row["frame_idx"])))
    return keys


def pick_random(
    scored: list[dict],
    rng: random.Random,
    n: int,
    exclude: set[tuple[str, int]],
) -> list[dict]:
    pool = []
    for row in scored:
        key_rel = (row["video_rel"], int(row["frame_idx"]))
        key_path = (row["video_path"], int(row["frame_idx"]))
        if key_rel in exclude or key_path in exclude:
            continue
        pool.append(row)
    rng.shuffle(pool)
    selected = pool[:n]
    if len(selected) < n:
        raise RuntimeError(
            f"Only {len(selected)} frames after exclude; need {n}. "
            "Increase --candidates."
        )
    selected.sort(key=lambda r: (r["camera"], r["video_rel"], r["frame_idx"]))
    for row in selected:
        row["stratum"] = "random"
    return selected


def write_readme(out_dir: Path, match_dir: Path) -> None:
    rel = out_dir.resolve().relative_to(ROOT.resolve())
    text = f"""# Match 1 CVAT train pack (100 random frames)

Fresh sample for ball/player correction. **Not** the Gold100 eval set.
Source: `{match_dir}`

## Review / correct

```bash
python serve_viewer.py
```

Open: http://localhost:8080/gold100

Initial detector draft may land under `prelabels/`. **After correction, promote to
`gold/`** (XML + COCO), point the editor at `gold/annotations.xml`, and delete
`prelabels/` so training cannot use stale boxes.

See `docs/ball_detection/TRAIN_LABEL_SOURCE_OF_TRUTH.md`.

## Export (from gold/)

```bash
python scripts/gold_set/export_gold_coco.py \\
  --gold-dir {rel} \\
  --xml {rel}/gold/annotations.xml \\
  --output {rel}/gold/annotations.coco.json
```
"""
    (out_dir / "README.md").write_text(text)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--match-dir", type=Path, default=DEFAULT_MATCH_DIR)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--exclude-manifest", type=Path, default=DEFAULT_EXCLUDE)
    p.add_argument(
        "--player-checkpoint",
        type=Path,
        default=ROOT / "models/people_after_100_epochs.pth",
    )
    p.add_argument(
        "--ball-checkpoint",
        type=Path,
        default=ROOT / "models/ball_89.pth",
    )
    p.add_argument("--seed", type=int, default=20260731)
    p.add_argument("--candidates", type=int, default=160)
    p.add_argument("--n-frames", type=int, default=100)
    p.add_argument("--detect-threshold", type=float, default=0.3)
    return p.parse_args()


def main():
    args = parse_args()
    rng = random.Random(args.seed)
    match_dir = require_dir(args.match_dir, "Match directory")
    player_ckpt = require_file(args.player_checkpoint, "Player checkpoint")
    ball_ckpt = require_file(args.ball_checkpoint, "Ball checkpoint")
    exclude = load_exclude_keys(args.exclude_manifest)
    print(f"Excluding {len(exclude)} keys from {args.exclude_manifest}")

    videos = discover_videos(match_dir)
    print(f"Found {len(videos)} videos")
    out_dir = args.output_dir
    images_dir = out_dir / "images"
    prelabels_dir = out_dir / "prelabels"
    review_dir = out_dir / "review"
    for d in (images_dir, prelabels_dir, review_dir):
        d.mkdir(parents=True, exist_ok=True)

    candidates = sample_match_candidates(videos, args.candidates, rng)
    print(f"Candidates: {len(candidates)}")

    detector = LocalRFDETRDetector(
        player_checkpoint=str(player_ckpt),
        ball_checkpoint=str(ball_ckpt),
        confidence_threshold=args.detect_threshold,
        player_class_id=0,
        ball_class_id=1,
        enhance_ball=True,
    )

    scored = []
    for i, row in enumerate(candidates):
        frame = read_frame(row["video_path"], row["frame_idx"], fps=row["fps"])
        detections = detector.detect(frame)
        balls = [d for d in detections if d.class_name == "ball"]
        players = [d for d in detections if d.class_name == "player"]
        scored.append({
            **row,
            "n_ball": len(balls),
            "n_player": len(players),
            "max_ball_conf": max((d.confidence for d in balls), default=0.0),
            "detections": detections,
        })
        if (i + 1) % 10 == 0 or i + 1 == len(candidates):
            print(f"  Detected {i + 1}/{len(candidates)}")

    selected = pick_random(scored, rng, args.n_frames, exclude)
    print(
        f"Selected {len(selected)}: "
        f"with_ball={sum(1 for r in selected if r['n_ball'] > 0)}, "
        f"players_avg={sum(r['n_player'] for r in selected) / len(selected):.1f}"
    )

    image_names = []
    strip_frames = []
    tracked_by_frame = {}
    manifest_rows = []
    next_track_id = 1
    scale_notes = []

    for strip_i, row in enumerate(selected):
        frame = read_frame(row["video_path"], row["frame_idx"], fps=row["fps"])
        h, w = frame.shape[:2]
        row["width"], row["height"] = w, h
        cam_slug = row["camera"].replace(" ", "")
        name = f"{cam_slug}_f{row['frame_idx']:06d}.jpg"
        cv2.imwrite(str(images_dir / name), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        image_names.append(name)
        strip_frames.append(frame)

        strip = resize_for_strip(frame)
        sh, sw = strip.shape[:2]
        sx, sy = sw / w, sh / h
        scaled_dets = []
        for det in row["detections"]:
            x, y, bw, bh = det.bbox
            scaled_dets.append(Detection(
                class_id=det.class_id,
                confidence=det.confidence,
                bbox=(x * sx, y * sy, bw * sx, bh * sy),
                class_name=det.class_name,
            ))
        tracked = to_ann_tracked(scaled_dets, next_track_id)
        next_track_id += max(len(tracked), 1)
        tracked_by_frame[strip_i] = tracked
        scale_notes.append({"strip_frame": strip_i, "sx": sx, "sy": sy})
        manifest_rows.append({
            "strip_frame": strip_i,
            "camera": row["camera"],
            "video_rel": row["video_rel"],
            "video_path": row["video_path"],
            "frame_idx": row["frame_idx"],
            "image": name,
            "width": w,
            "height": h,
            "stratum": row["stratum"],
            "n_ball": row["n_ball"],
            "n_player": row["n_player"],
            "max_ball_conf": row["max_ball_conf"],
        })

    write_coco(selected, image_names, prelabels_dir / "annotations.coco.json")
    strip_path = review_dir / "strip_100.mp4"
    strip_w, strip_h = write_strip_video(strip_frames, strip_path, fps=60)
    xml = create_cvat_xml(
        video_path=str(strip_path),
        tracked_objects_by_frame=tracked_by_frame,
        events=[],
        video_metadata={
            "width": strip_w,
            "height": strip_h,
            "fps": 60.0,
            "frame_count": len(selected),
        },
    )
    (prelabels_dir / "annotations.xml").write_text(xml, encoding="utf-8")

    manifest = {
        "match_dir": str(match_dir),
        "seed": args.seed,
        "detect_threshold": args.detect_threshold,
        "enhance_ball": True,
        "n_frames": len(selected),
        "strip_size": [strip_w, strip_h],
        "sampling": "random_exclude_gold100",
        "exclude_manifest": str(args.exclude_manifest),
        "camera_counts": {
            cam: sum(1 for r in manifest_rows if r["camera"] == cam)
            for cam in sorted({r["camera"] for r in manifest_rows})
        },
        "ball_frame_count": sum(1 for r in manifest_rows if r["n_ball"] > 0),
        "scale_to_strip": scale_notes,
        "frames": manifest_rows,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    write_readme(out_dir, match_dir)

    from scripts.gold_set.harden_review_pack import (
        build_image_editor,
        clamp_xml_boxes,
        validate_pack,
        write_review_frames,
        write_strip_from_frames,
    )

    names, rw, rh = write_review_frames(out_dir.resolve())
    clamp_xml_boxes(out_dir.resolve(), rw, rh)
    write_strip_from_frames(out_dir.resolve(), len(names))
    build_image_editor(out_dir.resolve(), names, rw, rh)
    manifest["strip_size"] = [rw, rh]
    manifest["review_mode"] = "image_sequence"
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    validate_pack(out_dir.resolve())

    print("Done.")
    print(f"Output: {out_dir}")
    print(f"Cameras: {manifest['camera_counts']}")
    print(f"Frames with ball prelabel: {manifest['ball_frame_count']}")
    print("Review: python serve_viewer.py → http://localhost:8080/gold100")


if __name__ == "__main__":
    main()
