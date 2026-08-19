#!/usr/bin/env python3
"""Smoke test Match 3 M1 strip scorer helpers."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "gold_set"))

from score_match3_ball_m1 import HIT_M, score_strip  # noqa: E402

STRIP = ROOT / "data/processed/gold_sets/match3_quad_p10_31/labels.json"


def main() -> int:
    if not STRIP.is_file():
        print("skip: strip not built yet")
        return 0
    row = score_strip(STRIP)
    if row["n_frames"] < 50:
        raise AssertionError(f"too few frames {row['n_frames']}")
    if row["P_emit"] is None and row["n_emit_scored"] > 0:
        raise AssertionError("P_emit missing with emits")
    if HIT_M != 4.0:
        raise AssertionError(f"HIT_M {HIT_M}")
    print("ok", json.dumps({k: row[k] for k in ["P_emit", "clear_ball_R", "n_clear", "n_emit_scored"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
