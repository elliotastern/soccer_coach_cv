#!/usr/bin/env python3
"""Sample N frames from a video, RF-DETR prelabel, write review/CVAT pack."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from annotation.cvat_xml_generator import create_cvat_xml  # noqa: E402
from scripts.gold_set.build_match_gold100 import (  # noqa: E402
    read_frame,
    require_file,
    resize_for_strip,
    to_ann_tracked,
    write_coco,
    write_strip_video,
)
from src.perception.rfdetr_local import LocalRFDETRDetector  # noqa: E402
from src.state.types import Detection  # noqa: E402


def sample_indices(n_frames: int, n: int) -> list[int]:
    if n_frames <= 0:
        raise ValueError("video has no frames")
    if n >= n_frames:
        return list(range(n_frames))
    if n == 1:
        return [n_frames // 2]
    # Even spacing, stay off the very last frame (seek reliability)
    last = max(0, n_frames - 2)
    return [int(round(i * last / (n - 1))) for i in range(n)]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--n-frames", type=int, default=20)
    p.add_argument("--detect-threshold", type=float, default=0.3)
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
    return p.parse_args()


def main():
    args = parse_args()
    video = require_file(args.video, "Video")
    player_ckpt = require_file(args.player_checkpoint, "Player checkpoint")
    ball_ckpt = require_file(args.ball_checkpoint, "Ball checkpoint")

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    print(f"Video: {video.name} {width}x{height} @ {fps:.2f}fps, frames={n_total}")

    idxs = sample_indices(n_total, args.n_frames)
    print(f"Sampling {len(idxs)} frames: {idxs[:5]}...{idxs[-3:]}")

    out_dir = args.output_dir
    images_dir = out_dir / "images"
    prelabels_dir = out_dir / "prelabels"
    review_dir = out_dir / "review"
    for d in (images_dir, prelabels_dir, review_dir):
        d.mkdir(parents=True, exist_ok=True)

    detector = LocalRFDETRDetector(
        player_checkpoint=str(player_ckpt),
        ball_checkpoint=str(ball_ckpt),
        confidence_threshold=args.detect_threshold,
        player_class_id=0,
        ball_class_id=1,
        enhance_ball=True,
    )

    selected = []
    for i, frame_idx in enumerate(idxs):
        frame = read_frame(str(video), frame_idx, fps=fps)
        h, w = frame.shape[:2]
        detections = detector.detect(frame)
        balls = [d for d in detections if d.class_name == "ball"]
        players = [d for d in detections if d.class_name == "player"]
        selected.append({
            "video_path": str(video),
            "video_rel": video.name,
            "camera": "clip",
            "frame_idx": frame_idx,
            "width": w,
            "height": h,
            "fps": fps,
            "stratum": "uniform",
            "n_ball": len(balls),
            "n_player": len(players),
            "max_ball_conf": max((d.confidence for d in balls), default=0.0),
            "detections": detections,
            "_frame": frame,
        })
        print(
            f"  [{i+1}/{len(idxs)}] f={frame_idx} "
            f"players={len(players)} balls={len(balls)} "
            f"max_ball={max((d.confidence for d in balls), default=0):.2f}"
        )

    image_names = []
    strip_frames = []
    tracked_by_frame = {}
    manifest_rows = []
    scale_notes = []
    next_track_id = 1

    for strip_i, row in enumerate(selected):
        frame = row.pop("_frame")
        name = f"f{row['frame_idx']:06d}.jpg"
        cv2.imwrite(str(images_dir / name), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        image_names.append(name)
        strip_frames.append(frame)

        strip = resize_for_strip(frame)
        sh, sw = strip.shape[:2]
        sx, sy = sw / row["width"], sh / row["height"]
        scaled = []
        for det in row["detections"]:
            x, y, bw, bh = det.bbox
            scaled.append(Detection(
                class_id=det.class_id,
                confidence=det.confidence,
                bbox=(x * sx, y * sy, bw * sx, bh * sy),
                class_name=det.class_name,
            ))
        tracked = to_ann_tracked(scaled, next_track_id)
        next_track_id += max(len(tracked), 1)
        tracked_by_frame[strip_i] = tracked
        scale_notes.append({"strip_frame": strip_i, "sx": sx, "sy": sy})
        manifest_rows.append({
            "strip_frame": strip_i,
            "camera": "clip",
            "video_rel": row["video_rel"],
            "video_path": row["video_path"],
            "frame_idx": row["frame_idx"],
            "image": name,
            "width": row["width"],
            "height": row["height"],
            "stratum": "uniform",
            "n_ball": row["n_ball"],
            "n_player": row["n_player"],
            "max_ball_conf": row["max_ball_conf"],
        })

    write_coco(selected, image_names, prelabels_dir / "annotations.coco.json")
    strip_path = review_dir / "strip_100.mp4"
    strip_w, strip_h = write_strip_video(strip_frames, strip_path, fps=10)
    xml = create_cvat_xml(
        video_path=str(strip_path),
        tracked_objects_by_frame=tracked_by_frame,
        events=[],
        video_metadata={
            "width": strip_w,
            "height": strip_h,
            "fps": 10.0,
            "frame_count": len(selected),
        },
    )
    (prelabels_dir / "annotations.xml").write_text(xml, encoding="utf-8")

    n = len(selected)
    last = n - 1
    manifest = {
        "video": str(video),
        "n_frames": n,
        "sample_indices": idxs,
        "detect_threshold": args.detect_threshold,
        "enhance_ball": True,
        "strip_size": [strip_w, strip_h],
        "ball_frame_count": sum(1 for r in manifest_rows if r["n_ball"] > 0),
        "scale_to_strip": scale_notes,
        "frames": manifest_rows,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (out_dir / "README.md").write_text(
        f"""# Prelabel pack ({n} frames)

Source: `{video}`

```bash
python serve_viewer.py
```

Open: http://localhost:8080/gold100
"""
    )

    from scripts.gold_set.harden_review_pack import (
        build_image_editor,
        clamp_xml_boxes,
        validate_pack,
        write_review_frames,
        write_strip_from_frames,
    )

    # Patch slider max for packs != 100
    names, rw, rh = write_review_frames(out_dir.resolve())
    clamp_xml_boxes(out_dir.resolve(), rw, rh)
    write_strip_from_frames(out_dir.resolve(), len(names), fps=10)
    build_image_editor(out_dir.resolve(), names, rw, rh)

    manifest["strip_size"] = [rw, rh]
    manifest["review_mode"] = "image_sequence"
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    validate_pack(out_dir.resolve())

    print("Done.")
    print(f"Output: {out_dir}")
    print(f"Ball frames: {manifest['ball_frame_count']}/{n}")
    print("Review: python serve_viewer.py → http://localhost:8080/gold100")


if __name__ == "__main__":
    main()
