#!/usr/bin/env python3
"""Build Match 3 team-ID benchmark sample (500 player-instances).

Outputs crop thumbnails + JSON for manual label review.
Labels default to -2 (unset); use eval script with --bootstrap for high-conf auto labels.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
_GS = ROOT / "scripts" / "gold_set"
sys.path.insert(0, str(_GS))

from raw_cam_id import load_match_raw  # noqa: E402

from src.perception.rfdetr_local import LocalRFDETRDetector  # noqa: E402
from src.perception.team_core import jersey_feature, torso_crop  # noqa: E402
from src.review.multicam_fuse import player_det_ok  # noqa: E402

OUT_JSON = ROOT / "data/gold/team_id_match3_500.json"
OUT_DIR = ROOT / "data/gold/team_id_match3_500_crops"
RAW = ROOT / "data/raw/Match 3"
N_BUCKETS = 10
FRAMES_PER_BUCKET = 5
BOXES_PER_FRAME = 10
CAM_ORDER = ["P1", "P6", "P7", "P8", "P9", "P10", "P_Goal1", "P_Goal2"]


def _load_cfg() -> dict:
    import yaml

    return yaml.safe_load((ROOT / "configs/default.yaml").read_text(encoding="utf-8"))


def _frame_count(path: Path) -> int:
    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    return n


def _sample_frames(total: int, rng: random.Random) -> list[int]:
    if total <= 0:
        return []
    bucket = max(1, total // N_BUCKETS)
    frames: list[int] = []
    for b in range(N_BUCKETS):
        lo = b * bucket
        hi = min(total - 1, (b + 1) * bucket - 1)
        if lo > hi:
            continue
        picks = [rng.randint(lo, hi) for _ in range(FRAMES_PER_BUCKET)]
        frames.extend(picks)
    return sorted(set(frames))


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(42)
    cfg = _load_cfg()
    detector = LocalRFDETRDetector(cfg)
    videos = load_match_raw(RAW)
    instances: list[dict] = []
    inst_id = 0

    for cam in CAM_ORDER:
        path = videos.get(cam)
        if path is None:
            continue
        total = _frame_count(path)
        frames = _sample_frames(total, rng)
        cap = cv2.VideoCapture(str(path))
        for fr in frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fr)
            ok, frame = cap.read()
            if not ok:
                continue
            wh = (frame.shape[1], frame.shape[0])
            dets = [d for d in detector.detect(frame) if player_det_ok(d)]
            dets.sort(key=lambda d: float(d.confidence), reverse=True)
            per_cam = [d for d in dets][:BOXES_PER_FRAME]
            if len(per_cam) < 2:
                continue
            for d in per_cam[: max(1, BOXES_PER_FRAME // len(CAM_ORDER))]:
                crop = torso_crop(frame, d.bbox, cam=cam, frame_wh=wh)
                if crop is None or jersey_feature(crop) is None:
                    continue
                fname = f"{inst_id:04d}_{cam}_f{fr}.jpg"
                cv2.imwrite(str(OUT_DIR / fname), crop)
                instances.append(
                    {
                        "id": inst_id,
                        "cam": cam,
                        "frame": int(fr),
                        "bbox": [float(v) for v in d.bbox],
                        "crop": fname,
                        "label": -2,
                        "label_name": "unset",
                        "notes": "",
                    }
                )
                inst_id += 1
                if inst_id >= 500:
                    break
            if inst_id >= 500:
                break
        cap.release()
        if inst_id >= 500:
            break

    payload = {
        "match": "Match 3",
        "n_instances": len(instances),
        "label_values": {
            "0": "team_0_blue",
            "1": "team_1_white_yellow",
            "-1": "unsure_gray",
            "ref": "referee",
            "gk": "goalkeeper",
            "invalid": "invalid_crop",
            "-2": "unset",
        },
        "instances": instances,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {len(instances)} instances → {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
