#!/usr/bin/env python3
"""Apply recommended ball prelabel stack and print per-frame ball boxes (debug)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.perception.ball_prelabel import BallPrelabelConfig, BallPrelabeler
from src.perception.rfdetr_local import load_ball_model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gold-dir", type=Path, default=ROOT / "data/processed/gold_sets/match1_1_100")
    p.add_argument("--max-frame", type=int, default=20)
    p.add_argument("--ball-checkpoint", type=Path, default=ROOT / "models/ball_89.pth")
    return p.parse_args()


def main():
    args = parse_args()
    model = load_ball_model(str(args.ball_checkpoint))
    pre = BallPrelabeler(
        model,
        BallPrelabelConfig(
            threshold=0.30,
            use_sahi=False,
            use_size_filter=True,
            topk=2,
            use_kalman=False,
            min_side=4,
            max_side=120,
        ),
    )
    for f in range(0, args.max_frame + 1):
        path = args.gold_dir / "review" / "frames" / f"{f:03d}.jpg"
        img = cv2.imread(str(path))
        dets = pre.detect_bgr(img)
        print(f"frame {f:02d}: {len(dets)} balls " +
              ", ".join(f"{d.confidence:.2f}@({d.bbox[0]:.0f},{d.bbox[1]:.0f})" for d in dets))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
