#!/usr/bin/env python3
"""v11 mix rules: clamp boxes, Match3/quad train vs holdout cuts."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "gold_set"))

from build_ball_finetune_v11 import (  # noqa: E402
    M3_STRIDE,
    M3_TRAIN_MAX,
    QUAD_STRIDE,
    QUAD_TRAIN_MAX,
    clamp_box,
)


def test_clamp() -> None:
    ok = clamp_box([-20, 100, 38, 26], 1920, 1080)
    if ok is None or ok[0] != 0 or abs(ok[2] - 18) > 1e-6:
        raise AssertionError(ok)
    if clamp_box([10, 10, 2, 2], 1920, 1080) is not None:
        raise AssertionError("tiny box should drop")


def test_cuts() -> None:
    if QUAD_TRAIN_MAX != 239 or QUAD_STRIDE != 5:
        raise AssertionError("quad cut")
    if M3_TRAIN_MAX != 119 or M3_STRIDE != 2:
        raise AssertionError("match3 cut")
    if 120 <= M3_TRAIN_MAX:
        raise AssertionError("M1 tail would leak")


def main() -> int:
    test_clamp()
    test_cuts()
    print("ok v11 mix rules")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
