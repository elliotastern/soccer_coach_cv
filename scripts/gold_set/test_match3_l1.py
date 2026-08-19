#!/usr/bin/env python3
"""L1: P8 / P9 / P_Goal1 are ≥4-click DLT with round-trip ≤ 0.15 m."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CALIB = ROOT / "reports/eval_match3/match3_pitch_calib"
CAMS = ("P8", "P9", "P_Goal1")
MAX_RT = 0.15


def roundtrip_m(rec: dict) -> tuple[float, float]:
    H = np.asarray(rec["H"], float)
    errs = []
    for (u, v), (X, Y) in zip(rec["image_points"], rec["pitch_points"]):
        p = H @ np.array([u, v, 1.0], dtype=float)
        xy = (p[0] / p[2], p[1] / p[2])
        errs.append(float(np.hypot(xy[0] - X, xy[1] - Y)))
    return float(np.mean(errs)), float(np.max(errs))


def main() -> int:
    for cam in CAMS:
        path = CALIB / f"{cam}_manual.json"
        rec = json.loads(path.read_text(encoding="utf-8"))
        n = len(rec.get("landmark_names") or [])
        if n < 4:
            raise AssertionError(f"{cam} has {n} landmarks, need ≥4")
        if rec.get("version") != "manual_clicks":
            raise AssertionError(f"{cam} version={rec.get('version')} want manual_clicks")
        # no FIFA invent
        bad = [n for n in rec["landmark_names"] if "penalty" in n]
        if bad:
            raise AssertionError(f"{cam} still has penalty landmarks {bad}")
        mean_e, max_e = roundtrip_m(rec)
        if max_e > MAX_RT:
            raise AssertionError(f"{cam} round-trip max {max_e:.4f} > {MAX_RT}")
        print(f"ok {cam} n={n} rt_max={max_e:.6f}m names={rec['landmark_names']}")
    print("ok match3 L1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
