#!/usr/bin/env python3
"""Build a stratified 100-frame match gold-set pack with LocalRFDETR prelabels.

Default source: Match 1 multi-camera clips under data/raw/Match 1/Match 1 -1.
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from annotation.cvat_xml_generator import create_cvat_xml
from annotation.types import Detection as AnnDetection
from annotation.types import TrackedObject
from src.perception.rfdetr_local import LocalRFDETRDetector
from src.state.types import Detection

DEFAULT_MATCH_DIR = ROOT / "data/raw/Match 1/Match 1 -1"
DEFAULT_OUT = ROOT / "data/processed/gold_sets/match1_1_100"
STRATA_TARGETS = {
    "camera": 40,
    "uniform": 20,
    "high_ball": 15,
    "uncertain_ball": 15,
    "negative": 10,
}
EDITOR_TEMPLATE = ROOT / "annotation/view_annotations_editor.html"
STRIP_MAX_WIDTH = 1920


def require_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def require_dir(path: Path, label: str) -> Path:
    if not path.is_dir():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def video_meta(path: Path) -> Dict:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        # Some GoPro files report 0; estimate from duration
        duration = float(cap.get(cv2.CAP_PROP_POS_MSEC))
        frame_count = max(1, int(duration / 1000.0 * fps)) if duration > 0 else 1
    meta = {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": fps,
        "frame_count": frame_count,
        "duration_sec": frame_count / fps,
    }
    cap.release()
    # Prefer ffprobe duration when OpenCV frame_count is unreliable
    try:
        probe = subprocess.run(
            [
                "ffprobe", "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=duration,r_frame_rate,width,height",
                "-of", "json", str(path),
            ],
            capture_output=True, text=True, check=True,
        )
        stream = json.loads(probe.stdout)["streams"][0]
        if "duration" in stream:
            dur = float(stream["duration"])
            num, den = stream.get("r_frame_rate", "30/1").split("/")
            fps = float(num) / float(den)
            meta["fps"] = fps
            meta["duration_sec"] = dur
            meta["frame_count"] = max(1, int(round(dur * fps)))
        if "width" in stream:
            meta["width"] = int(stream["width"])
            meta["height"] = int(stream["height"])
    except Exception:
        pass
    return meta


def discover_videos(match_dir: Path) -> List[Dict]:
    videos = []
    for path in sorted(match_dir.rglob("*.mp4")):
        if path.name.startswith("._"):
            continue
        cam = path.parent.name
        meta = video_meta(path)
        if meta["duration_sec"] < 5:
            continue
        videos.append({
            "path": path,
            "camera": cam,
            "rel": str(path.relative_to(match_dir)),
            **meta,
        })
    if not videos:
        raise RuntimeError(f"No usable .mp4 files under {match_dir}")
    return videos


def sample_match_candidates(
    videos: List[Dict],
    n_total: int,
    rng: random.Random,
) -> List[Dict]:
    total_dur = sum(v["duration_sec"] for v in videos)
    rows = []
    for video in videos:
        share = video["duration_sec"] / total_dur
        n = max(8, int(round(n_total * share)))
        # Stay inside reliable seek range (GoPro/OpenCV frame_count can overshoot)
        n_frames = max(1, int(video["frame_count"] * 0.9))
        for i in range(n):
            start = int(i * n_frames / n)
            end = int((i + 1) * n_frames / n) - 1
            end = max(start, end)
            frame_idx = rng.randint(start, end)
            rows.append({
                "video_path": str(video["path"]),
                "camera": video["camera"],
                "video_rel": video["rel"],
                "frame_idx": frame_idx,
                "width": video["width"],
                "height": video["height"],
                "fps": video["fps"],
                "source": "uniform",
                "event_label": None,
            })
    rng.shuffle(rows)
    seen = set()
    out = []
    for row in rows:
        key = (row["video_path"], row["frame_idx"])
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out[:n_total]


def read_frame(video_path: str, frame_idx: int, fps: float = 30.0) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    def try_read(idx: int) -> Optional[np.ndarray]:
        if idx < 0:
            return None
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if ok and frame is not None:
            return frame
        # Time-based seek fallback
        cap.set(cv2.CAP_PROP_POS_MSEC, (idx / max(fps, 1e-3)) * 1000.0)
        ok, frame = cap.read()
        if ok and frame is not None:
            return frame
        return None

    frame = try_read(frame_idx)
    if frame is None:
        for delta in range(1, 60):
            frame = try_read(frame_idx - delta)
            if frame is not None:
                break
            frame = try_read(frame_idx + delta)
            if frame is not None:
                break
    cap.release()
    if frame is None:
        raise RuntimeError(f"Failed to read {video_path} frame {frame_idx}")
    return frame


def det_summary(detections: List[Detection]) -> Dict:
    balls = [d for d in detections if d.class_name == "ball"]
    players = [d for d in detections if d.class_name == "player"]
    max_ball = max((d.confidence for d in balls), default=0.0)
    return {
        "n_ball": len(balls),
        "n_player": len(players),
        "max_ball_conf": max_ball,
        "detections": detections,
    }


def assign_detector_labels(scored: List[Dict]) -> None:
    if not scored:
        return
    n_neg = min(STRATA_TARGETS["negative"] * 3, max(1, len(scored) // 10))
    by_conf = sorted(scored, key=lambda r: r["max_ball_conf"])
    neg_ids = {(r["video_path"], r["frame_idx"]) for r in by_conf[:n_neg]}
    for row in scored:
        key = (row["video_path"], row["frame_idx"])
        if row["n_ball"] == 0 or key in neg_ids:
            row["detector_stratum"] = "negative"
        elif row["max_ball_conf"] >= 0.8:
            row["detector_stratum"] = "high_ball"
        else:
            row["detector_stratum"] = "uncertain_ball"


def pick_strata(scored: List[Dict], rng: random.Random) -> List[Dict]:
    assign_detector_labels(scored)
    selected = []
    used = set()

    def key_of(row: Dict) -> Tuple[str, int]:
        return (row["video_path"], row["frame_idx"])

    def take_rows(rows: List[Dict], n: int, stratum: str) -> None:
        candidates = [r for r in rows if key_of(r) not in used]
        rng.shuffle(candidates)
        for row in candidates[:n]:
            item = dict(row)
            item["stratum"] = stratum
            selected.append(item)
            used.add(key_of(row))

    for stratum in ("high_ball", "uncertain_ball", "negative"):
        pool = [r for r in scored if r["detector_stratum"] == stratum]
        take_rows(pool, STRATA_TARGETS[stratum], stratum)

    # Camera-balanced stratum
    by_cam = defaultdict(list)
    for row in scored:
        if key_of(row) not in used:
            by_cam[row["camera"]].append(row)
    cam_target = STRATA_TARGETS["camera"]
    cams = list(by_cam.keys())
    rng.shuffle(cams)
    per_cam = max(1, cam_target // max(len(cams), 1))
    cam_picked = []
    for cam in cams:
        bucket = by_cam[cam]
        rng.shuffle(bucket)
        cam_picked.extend(bucket[:per_cam])
    take_rows(cam_picked, cam_target, "camera")

    leftovers = [r for r in scored if key_of(r) not in used]
    take_rows(leftovers, STRATA_TARGETS["uniform"], "uniform")

    if len(selected) < 100:
        leftovers = [r for r in scored if key_of(r) not in used]
        rng.shuffle(leftovers)
        for row in leftovers:
            if len(selected) >= 100:
                break
            item = dict(row)
            item["stratum"] = item.get("detector_stratum", "uniform")
            selected.append(item)
            used.add(key_of(row))

    selected = selected[:100]
    selected.sort(key=lambda r: (r["camera"], r["video_rel"], r["frame_idx"]))
    return selected


def to_ann_tracked(detections: List[Detection], start_id: int) -> List[TrackedObject]:
    tracked = []
    for i, det in enumerate(detections):
        ann_det = AnnDetection(
            class_id=det.class_id,
            confidence=det.confidence,
            bbox=det.bbox,
            class_name=det.class_name,
        )
        tracked.append(TrackedObject(object_id=start_id + i, detection=ann_det))
    return tracked


def write_coco(
    selected: List[Dict],
    image_names: List[str],
    out_path: Path,
) -> None:
    categories = [
        {"id": 1, "name": "player", "supercategory": "person"},
        {"id": 2, "name": "ball", "supercategory": "sports"},
    ]
    images = []
    annotations = []
    ann_id = 1
    for image_id, (row, name) in enumerate(zip(selected, image_names), start=1):
        images.append({
            "id": image_id,
            "file_name": name,
            "width": row["width"],
            "height": row["height"],
            "camera": row["camera"],
            "video_rel": row["video_rel"],
            "frame_idx": row["frame_idx"],
            "stratum": row["stratum"],
        })
        for det in row["detections"]:
            x, y, w, h = det.bbox
            cat_id = 2 if det.class_name == "ball" else 1
            annotations.append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": cat_id,
                "bbox": [float(x), float(y), float(w), float(h)],
                "area": float(w * h),
                "iscrowd": 0,
                "score": float(det.confidence),
            })
            ann_id += 1
    payload = {
        "info": {
            "description": "Match 1 / Match 1 -1 gold100 prelabels",
            "version": "1.0",
        },
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }
    out_path.write_text(json.dumps(payload, indent=2))


def resize_for_strip(frame: np.ndarray, max_width: int = STRIP_MAX_WIDTH) -> np.ndarray:
    h, w = frame.shape[:2]
    if w <= max_width:
        return frame
    scale = max_width / w
    return cv2.resize(frame, (max_width, int(round(h * scale))), interpolation=cv2.INTER_AREA)


def write_strip_video(frames: List[np.ndarray], out_path: Path, fps: int = 60) -> Tuple[int, int]:
    if not frames:
        raise ValueError("No frames for strip video")
    frames = [resize_for_strip(f) for f in frames]
    height, width = frames[0].shape[:2]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        raw_path = Path(tmp) / "strip_raw.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(raw_path), fourcc, fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"Could not open VideoWriter for {raw_path}")
        for frame in frames:
            writer.write(frame)
        writer.release()
        # All-intra H.264 so browser frame seeks match strip index exactly.
        cmd = [
            "ffmpeg", "-y", "-i", str(raw_path),
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-g", "1", "-keyint_min", "1", "-bf", "0",
            "-x264-params", "keyint=1:min-keyint=1:scenecut=0",
            "-movflags", "+faststart", "-an", str(out_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg H.264 encode failed: {result.stderr[-500:]}")
    return width, height


def write_review_editor(out_dir: Path, n_frames: int = 100) -> Path:
    template = EDITOR_TEMPLATE.read_text(encoding="utf-8")
    out_dir = out_dir.resolve()
    rel = out_dir.relative_to(ROOT.resolve()).as_posix()
    video_url = f"/{rel}/review/strip_100.mp4"
    xml_url = f"/{rel}/prelabels/annotations.xml"
    xml_save = f"{rel}/prelabels/annotations.xml"
    last = n_frames - 1
    html = template
    html = html.replace(
        "data/raw/real_data/F9D97C58-4877-4905-9A9F-6590FCC758FF.mp4",
        video_url,
    )
    html = html.replace(
        "data/raw/real_data/37CAE053-841F-4851-956E-CBF17A51C506_annotations.xml",
        xml_url,
    )
    html = html.replace(
        "data/raw/real_data/F9D97C58-4877-4905-9A9F-6590FCC758FF_annotations.xml",
        xml_save,
    )
    html = html.replace('max="796"', f'max="{last}"')
    html = html.replace(" / 796", f" / {last}")
    html = html.replace("seekToFrame(796)", f"seekToFrame({last})")
    html = html.replace(
        "Math.min(parseInt(frameNum), 796)",
        f"Math.min(parseInt(frameNum), {last})",
    )
    html = html.replace("if (currentFrame < 796)", f"if (currentFrame < {last})")
    out_path = out_dir / "review" / "editor.html"
    out_path.write_text(html, encoding="utf-8")
    (ROOT / "annotation" / "gold100_editor.html").write_text(html, encoding="utf-8")
    return out_path


def write_readme(out_dir: Path, match_dir: Path) -> None:
    rel = out_dir.resolve().relative_to(ROOT.resolve())
    text = f"""# Match 1 gold set (100 frames)

Source: `{match_dir}`

## Correct prelabels

```bash
python serve_viewer.py
```

Open: http://localhost:8080/gold100

- Scrub frames 0–99 (strip @ 60fps, downscaled for the editor).
- Full-res JPGs are in `images/`.
- Fix boxes; prioritize missed / spurious balls.
- Save writes `{rel}/prelabels/annotations.xml`.

## Export / eval

```bash
python scripts/gold_set/export_gold_coco.py --gold-dir {rel}
python scripts/gold_set/eval_on_gold100.py --gold-dir {rel}
```
"""
    (out_dir / "README.md").write_text(text)


def parse_args():
    parser = argparse.ArgumentParser(description="Build stratified 100-frame match gold set")
    parser.add_argument("--match-dir", type=Path, default=DEFAULT_MATCH_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--player-checkpoint", type=Path, default=ROOT / "models/people_after_100_epochs.pth")
    parser.add_argument("--ball-checkpoint", type=Path, default=ROOT / "models/ball_89.pth")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--candidates", type=int, default=400)
    parser.add_argument("--detect-threshold", type=float, default=0.3)
    return parser.parse_args()


def main():
    args = parse_args()
    rng = random.Random(args.seed)

    match_dir = require_dir(args.match_dir, "Match directory")
    player_ckpt = require_file(args.player_checkpoint, "Player checkpoint")
    ball_ckpt = require_file(args.ball_checkpoint, "Ball checkpoint")

    videos = discover_videos(match_dir)
    print(f"Found {len(videos)} videos under {match_dir}")
    for v in videos:
        print(
            f"  {v['camera']}/{Path(v['rel']).name}: "
            f"{v['width']}x{v['height']} @ {v['fps']:.1f}fps, "
            f"{v['duration_sec']:.1f}s ({v['frame_count']} frames)"
        )

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
    )

    scored = []
    cache_rows = []
    for i, row in enumerate(candidates):
        frame = read_frame(row["video_path"], row["frame_idx"], fps=row["fps"])
        detections = detector.detect(frame)
        summary = det_summary(detections)
        item = {
            **row,
            "n_ball": summary["n_ball"],
            "n_player": summary["n_player"],
            "max_ball_conf": summary["max_ball_conf"],
            "detections": summary["detections"],
        }
        scored.append(item)
        cache_rows.append({
            "video_path": row["video_path"],
            "video_rel": row["video_rel"],
            "camera": row["camera"],
            "frame_idx": row["frame_idx"],
            "n_ball": summary["n_ball"],
            "n_player": summary["n_player"],
            "max_ball_conf": summary["max_ball_conf"],
        })
        if (i + 1) % 10 == 0 or i + 1 == len(candidates):
            print(f"  Detected {i + 1}/{len(candidates)}")

    (out_dir / "candidate_scores.json").write_text(json.dumps(cache_rows, indent=2))

    selected = pick_strata(scored, rng)
    if len(selected) < 100:
        raise RuntimeError(f"Only selected {len(selected)} frames; need 100")

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

        # Scale prelabel boxes if strip is downscaled (XML is on strip coords)
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

    # COCO uses full-res image coordinates (unscaled detections)
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
        "n_frames": len(selected),
        "strip_size": [strip_w, strip_h],
        "strata_targets": STRATA_TARGETS,
        "strata_counts": {
            k: sum(1 for r in manifest_rows if r["stratum"] == k)
            for k in STRATA_TARGETS
        },
        "camera_counts": {
            cam: sum(1 for r in manifest_rows if r["camera"] == cam)
            for cam in sorted({r["camera"] for r in manifest_rows})
        },
        "scale_to_strip": scale_notes,
        "frames": manifest_rows,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    write_readme(out_dir, match_dir)

    # Image-sequence editor + all-intra strip (avoids browser video seek drift)
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
    print(f"Strata: {manifest['strata_counts']}")
    print(f"Cameras: {manifest['camera_counts']}")
    print("Review: python serve_viewer.py → http://localhost:8080/gold100")


if __name__ == "__main__":
    main()
