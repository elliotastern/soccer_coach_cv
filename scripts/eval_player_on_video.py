#!/usr/bin/env python3
"""Judge player detection on real match video (no box GT).

Samples frames, runs people_after_100_epochs.pth, writes overlays + summary
stats for human review. Use when SoccerSynth is not the desired judge set.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.perception.rfdetr_local import load_people_model

DEFAULT_VIDEO = Path(
    "/Volumes/LaCie/Projects/Soccer project data/"
    "soccer_track/videos/117093_panorama_1st_half-017.mp4"
)
DEFAULT_CKPT = ROOT / "models" / "people_after_100_epochs.pth"
# Soft expectation for a full-pitch / half-pitch broadcast-style view
EXPECTED_PLAYERS_MIN = 8
EXPECTED_PLAYERS_MAX = 25


def draw_boxes(frame, boxes, scores):
    out = frame.copy()
    for (x1, y1, x2, y2), score in zip(boxes, scores):
        p1 = (int(x1), int(y1))
        p2 = (int(x2), int(y2))
        cv2.rectangle(out, p1, p2, (0, 255, 0), 2)
        cv2.putText(
            out,
            f"{score:.2f}",
            (p1[0], max(20, p1[1] - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
    return out


def predict(model, frame_bgr, threshold: float):
    pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    raw = model.predict(pil, threshold=threshold)
    if not hasattr(raw, "xyxy") or len(raw.xyxy) == 0:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    return np.asarray(raw.xyxy, dtype=np.float32), np.asarray(raw.confidence, dtype=np.float32)


def sample_frame_ids(frame_count: int, max_frames: int, stride: int) -> list:
    ids = list(range(0, frame_count, stride))
    if len(ids) > max_frames:
        ids = ids[:max_frames]
    return ids


def count_in_range(n: int) -> bool:
    return EXPECTED_PLAYERS_MIN <= n <= EXPECTED_PLAYERS_MAX


def main():
    parser = argparse.ArgumentParser(description="Eval player detector on real video")
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max-frames", type=int, default=40)
    parser.add_argument("--stride", type=int, default=25)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "reports" / "player_detection_eval_soccertrack",
    )
    args = parser.parse_args()

    if not args.video.is_file():
        raise FileNotFoundError(f"Video not found: {args.video}")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    out_dir = args.output_dir
    overlay_dir = out_dir / "overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {args.video}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    frame_ids = sample_frame_ids(frame_count, args.max_frames, args.stride)
    print(f"Video: {args.video}")
    print(f"Frames to score: {len(frame_ids)} / {frame_count} @ stride={args.stride}")
    model = load_people_model(str(args.checkpoint))

    per_frame = []
    all_scores = []
    for frame_id in frame_ids:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ok, frame = cap.read()
        if not ok:
            continue
        boxes, scores = predict(model, frame, args.threshold)
        n = len(boxes)
        mean_score = float(scores.mean()) if n else 0.0
        all_scores.extend(scores.tolist())
        row = {
            "frame_id": frame_id,
            "num_detections": n,
            "mean_confidence": mean_score,
            "max_confidence": float(scores.max()) if n else 0.0,
            "in_expected_count_range": count_in_range(n),
        }
        per_frame.append(row)
        overlay = draw_boxes(frame, boxes, scores)
        # Downscale wide panoramas for easier browsing
        scale = min(1.0, 1600 / max(width, 1))
        if scale < 1.0:
            overlay = cv2.resize(overlay, None, fx=scale, fy=scale)
        name = f"frame_{frame_id:06d}_n{n}_m{mean_score:.2f}.jpg"
        cv2.imwrite(str(overlay_dir / name), overlay)
        print(f"frame {frame_id}: dets={n} mean_conf={mean_score:.3f}")

    cap.release()

    counts = [r["num_detections"] for r in per_frame]
    in_range = sum(1 for r in per_frame if r["in_expected_count_range"])
    summary = {
        "video": str(args.video),
        "checkpoint": str(args.checkpoint),
        "dataset": "soccer_track panorama (real video; no player-box GT)",
        "note": (
            "Cannot compute P/R/mAP without boxes. "
            "Judge from overlays + whether detection counts look plausible "
            f"({EXPECTED_PLAYERS_MIN}-{EXPECTED_PLAYERS_MAX} per frame)."
        ),
        "threshold": args.threshold,
        "metadata": {
            "fps": fps,
            "frame_count": frame_count,
            "width": width,
            "height": height,
        },
        "frames_scored": len(per_frame),
        "detections_per_frame_mean": float(np.mean(counts)) if counts else 0.0,
        "detections_per_frame_median": float(np.median(counts)) if counts else 0.0,
        "detections_per_frame_min": int(min(counts)) if counts else 0,
        "detections_per_frame_max": int(max(counts)) if counts else 0,
        "fraction_frames_in_expected_count_range": (
            in_range / len(per_frame) if per_frame else 0.0
        ),
        "score_mean": float(np.mean(all_scores)) if all_scores else 0.0,
        "score_p50": float(np.percentile(all_scores, 50)) if all_scores else 0.0,
        "score_p90": float(np.percentile(all_scores, 90)) if all_scores else 0.0,
        "per_frame": per_frame,
        "overlays_dir": str(overlay_dir),
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in summary if k != "per_frame"}, indent=2))
    print(f"Wrote {summary_path}")
    print(f"Overlays: {overlay_dir}")
    # Open overlays folder for visual check
    try:
        import subprocess
        subprocess.run(["open", str(overlay_dir)], check=False)
    except Exception:
        pass


if __name__ == "__main__":
    main()
