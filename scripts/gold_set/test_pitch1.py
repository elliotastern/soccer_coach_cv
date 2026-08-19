#!/usr/bin/env python3
"""Pitch 1 measured marks → exact landmark meters."""
from __future__ import annotations

from pitch1 import load_pitch1, pitch1_landmarks


def main() -> int:
    rec = load_pitch1()
    if abs(rec["length_m"] - 53.9) > 1e-6 or abs(rec["width_m"] - 34.84) > 1e-6:
        print("pitch1 must be 53.90×34.84")
        return 1
    lms = pitch1_landmarks(rec)
    want = {
        "left_box_18_near": (-21.0, 3.6),
        "right_box_18_far": (21.0, -5.415),
        "left_post_near": (-26.95, 1.45),
        "right_post_far": (26.95, -3.265),
        "circle_far": (0.0, -3.5),
        "left_near_corner": (-26.95, 17.42),
        "right_far_corner": (26.95, -17.405),
    }
    for name, (x, y) in want.items():
        got = tuple(lms[name]["xy"])
        if abs(got[0] - x) > 1e-6 or abs(got[1] - y) > 1e-6:
            print(name, "got", got, "want", (x, y))
            return 1
    if "5.95" not in lms["right_box_18_near"]["spec"]:
        print("box spec missing 5.95 m")
        return 1
    if "left_penalty_spot" in lms or "left_6_box_near" in lms:
        print("FIFA-only marks must not be on Pitch 1")
        return 1
    print("pitch1 landmarks ok", len(lms))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
