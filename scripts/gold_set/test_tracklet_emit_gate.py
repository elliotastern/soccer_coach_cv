#!/usr/bin/env python3
"""Unit smoke: ball tracklets need EMA/instant >= emit_thresh to publish."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.perception.tracker import Tracker
from src.state.types import Detection


def ball(conf: float) -> Detection:
    return Detection(class_id=1, confidence=conf, bbox=(100, 100, 20, 20), class_name="ball")


def main():
    tr = Tracker(track_thresh=0.1, emit_thresh=0.80, ema_alpha=0.5, apply_emit_gate=True)
    # Low conf should associate but not emit
    out = tr.update([ball(0.3)])
    assert out == [], f"expected no emit, got {out}"
    # Burst of rising conf should eventually emit via EMA or instant
    emitted = False
    for c in (0.4, 0.6, 0.75, 0.85):
        out = tr.update([ball(c)])
        if out:
            emitted = True
            assert out[0].detection.class_name == "ball"
            break
    assert emitted, "expected emit once conf/EMA crossed 0.8"
    print("OK tracklet emit gate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
