#!/usr/bin/env python3
"""L2: P1/P6 have ≥5 overlapping marks (shared center) with RT ≤ 0.15 m."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CALIB = ROOT / "reports/eval_match3/match3_pitch_calib"
MAX_RT = 0.15


def main() -> int:
    for cam in ("P1", "P6"):
        rec = json.loads((CALIB / f"{cam}_manual.json").read_text(encoding="utf-8"))
        names = rec.get("landmark_names") or []
        if len(names) < 5:
            raise AssertionError(f"{cam} has {len(names)} landmarks, need ≥5")
        if "center" not in names:
            raise AssertionError(f"{cam} missing shared center overlap mark")
        rt = float(rec.get("roundtrip_max_m", 99.0))
        if rt > MAX_RT:
            raise AssertionError(f"{cam} round-trip max {rt:.4f} > {MAX_RT}")
        print(f"ok {cam} n={len(names)} rt_max={rt:.4f} names={names}")
    print("ok match3 L2")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
