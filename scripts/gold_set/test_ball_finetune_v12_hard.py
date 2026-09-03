#!/usr/bin/env python3
"""v12 hard mix: small filter + tiny sides."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "gold_set"))

from build_ball_finetune_v12_hard import (  # noqa: E402
    SMALL_MAX_SIDE,
    TINY_SIDES,
    is_small,
    max_side,
)


def main() -> int:
    if SMALL_MAX_SIDE != 28.0:
        raise AssertionError(SMALL_MAX_SIDE)
    if min(TINY_SIDES) < 8 or max(TINY_SIDES) > 18:
        raise AssertionError(TINY_SIDES)
    big = {"boxes": [[0, 0, 40, 40]]}
    small = {"boxes": [[0, 0, 20, 18]]}
    if is_small(big) or not is_small(small):
        raise AssertionError("is_small")
    if max_side(small["boxes"]) != 20.0:
        raise AssertionError(max_side(small["boxes"]))
    print("ok v12 hard mix rules")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
