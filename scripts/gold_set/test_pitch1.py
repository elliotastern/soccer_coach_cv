#!/usr/bin/env python3
"""Pitch 1 FIFA marks → exact landmark meters."""
from __future__ import annotations

from pitch1 import load_pitch1, pitch1_landmarks


def main() -> int:
    rec = load_pitch1()
    if rec["length_m"] != 105.0 or rec["width_m"] != 68.0:
        print("pitch1 must be 105×68")
        return 1
    lms = pitch1_landmarks(rec)
    want = {
        "left_box_18_near": (-36.0, 20.16),
        "right_box_18_far": (36.0, -20.16),
        "right_6_box_near": (47.0, 9.16),
        "left_penalty_spot": (-41.5, 0.0),
        "right_post_near": (52.5, 3.66),
    }
    for name, (x, y) in want.items():
        got = tuple(lms[name]["xy"])
        if abs(got[0] - x) > 1e-6 or abs(got[1] - y) > 1e-6:
            print(name, "got", got, "want", (x, y))
            return 1
    if "16.5" not in lms["right_box_18_near"]["spec"]:
        print("18-yard spec missing 16.5 m")
        return 1
    print("pitch1 landmarks ok", len(lms))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
