#!/usr/bin/env python3
"""Smoke-test file ingest + gameplay / scene-cut gate (Phase 1 Step 1)."""

import argparse
import json
import sys
from pathlib import Path

import cv2
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ingest.video import open_video_file
from src.perception.camera import (
    compute_green_ratio,
    detect_scene_cut,
    is_gameplay_view,
)

MAX_DUMPS = 10
KEEP_RATE_PASS = 0.80


def load_camera_config(config_path: str) -> dict:
    with open(config_path) as f:
        cfg = yaml.safe_load(f) or {}
    camera = cfg.get("camera") or {}
    return {
        "green_threshold": float(camera.get("green_threshold", 0.5)),
        "scene_cut_threshold": float(camera.get("scene_cut_threshold", 0.7)),
    }


def ensure_dirs(output_dir: Path) -> dict:
    paths = {
        "root": output_dir,
        "keeps": output_dir / "keeps",
        "skips": output_dir / "skips",
        "scene_cuts": output_dir / "scene_cuts",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def dump_sample(dir_path: Path, frame_id: int, ratio: float, frame, count: int) -> int:
    if count >= MAX_DUMPS:
        return count
    name = f"frame_{frame_id:06d}_green_{ratio:.3f}.jpg"
    cv2.imwrite(str(dir_path / name), frame)
    return count + 1


def video_metadata(cap) -> dict:
    return {
        "fps": float(cap.get(cv2.CAP_PROP_FPS) or 0.0),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
    }


def evaluate_pass(summary: dict) -> tuple:
    if summary["sampled_frames"] == 0:
        return False, "no frames sampled"
    keep_rate = summary["keep_count"] / summary["sampled_frames"]
    cuts = summary["scene_cut_count"]
    cut_rate = cuts / summary["sampled_frames"]
    if keep_rate < KEEP_RATE_PASS:
        return False, f"keep_rate={keep_rate:.3f} < {KEEP_RATE_PASS}"
    if cut_rate > 0.10:
        return False, f"scene_cut_rate={cut_rate:.3f} too high (>0.10)"
    return True, f"keep_rate={keep_rate:.3f} scene_cuts={cuts}"


def run_smoke(video: str, output_dir: Path, max_frames: int, stride: int,
              green_threshold: float, scene_cut_threshold: float) -> dict:
    paths = ensure_dirs(output_dir)
    cap = open_video_file(video)
    meta = video_metadata(cap)

    ratios = []
    keep_count = 0
    skip_count = 0
    scene_cut_count = 0
    keep_dumps = 0
    skip_dumps = 0
    cut_dumps = 0
    prev_frame = None
    sampled = 0
    frame_id = 0

    while True:
        if max_frames is not None and frame_id >= max_frames:
            break
        ok, frame = cap.read()
        if not ok:
            break
        if frame_id % stride != 0:
            frame_id += 1
            continue

        ratio = compute_green_ratio(frame)
        ratios.append(ratio)
        sampled += 1
        is_keep = is_gameplay_view(frame, green_threshold=green_threshold)
        is_cut = detect_scene_cut(
            frame, prev_frame, threshold=scene_cut_threshold
        )

        if is_keep:
            keep_count += 1
            keep_dumps = dump_sample(paths["keeps"], frame_id, ratio, frame, keep_dumps)
        else:
            skip_count += 1
            skip_dumps = dump_sample(paths["skips"], frame_id, ratio, frame, skip_dumps)

        if is_cut:
            scene_cut_count += 1
            cut_dumps = dump_sample(
                paths["scene_cuts"], frame_id, ratio, frame, cut_dumps
            )

        prev_frame = frame
        frame_id += 1

    cap.release()

    summary = {
        "video": str(video),
        "green_threshold": green_threshold,
        "scene_cut_threshold": scene_cut_threshold,
        "max_frames": max_frames,
        "stride": stride,
        "metadata": meta,
        "sampled_frames": sampled,
        "keep_count": keep_count,
        "skip_count": skip_count,
        "scene_cut_count": scene_cut_count,
        "green_ratio_min": min(ratios) if ratios else None,
        "green_ratio_mean": (sum(ratios) / len(ratios)) if ratios else None,
        "green_ratio_max": max(ratios) if ratios else None,
        "keep_dumps": keep_dumps,
        "skip_dumps": skip_dumps,
        "scene_cut_dumps": cut_dumps,
    }
    passed, reason = evaluate_pass(summary)
    summary["passed"] = passed
    summary["pass_reason"] = reason
    return summary


def main():
    parser = argparse.ArgumentParser(description="Smoke-test ingest + gameplay gate")
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--max-frames", type=int, default=300)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--output-dir", default="reports/step1_ingest_gate")
    parser.add_argument("--green-threshold", type=float, default=None)
    parser.add_argument("--scene-cut-threshold", type=float, default=None)
    args = parser.parse_args()

    camera = load_camera_config(args.config)
    green_threshold = (
        args.green_threshold
        if args.green_threshold is not None
        else camera["green_threshold"]
    )
    scene_cut_threshold = (
        args.scene_cut_threshold
        if args.scene_cut_threshold is not None
        else camera["scene_cut_threshold"]
    )

    output_dir = Path(args.output_dir)
    summary = run_smoke(
        video=args.video,
        output_dir=output_dir,
        max_frames=args.max_frames,
        stride=args.stride,
        green_threshold=green_threshold,
        scene_cut_threshold=scene_cut_threshold,
    )

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    status = "PASS" if summary["passed"] else "FAIL"
    gmin = summary["green_ratio_min"]
    gmean = summary["green_ratio_mean"]
    gmax = summary["green_ratio_max"]
    green_txt = (
        f"{gmin:.3f}/{gmean:.3f}/{gmax:.3f}"
        if gmean is not None
        else "n/a"
    )
    print(
        f"{status}: {summary['pass_reason']} | "
        f"sampled={summary['sampled_frames']} "
        f"keep={summary['keep_count']} skip={summary['skip_count']} "
        f"cuts={summary['scene_cut_count']} | "
        f"green min/mean/max={green_txt} | "
        f"wrote {summary_path}"
    )
    sys.exit(0 if summary["passed"] else 1)


if __name__ == "__main__":
    main()
