#!/usr/bin/env python3
"""Both goal boxes on the pitch; P9 still is the lengthwise camera."""
from __future__ import annotations

from match3_landmarks import (
    CAMS,
    DISPLAY,
    EXTRA_XY,
    ORDERS,
    RAW_FILE,
    all_landmarks,
    families,
    on_pitch_names,
)

SOUTH_BOX = {
    "left_box_goal_near", "left_box_goal_far", "left_box_18_near",
    "left_box_18_far", "left_penalty_spot", "left_post_near", "left_post_far",
}
NORTH_BOX = {
    "right_box_goal_near", "right_box_goal_far", "right_box_18_near",
    "right_box_18_far", "right_penalty_spot", "right_post_near", "right_post_far",
}


def main() -> int:
    catalog = all_landmarks()
    missing_labels = [n for n in catalog if n not in DISPLAY]
    if missing_labels:
        print("DISPLAY missing", missing_labels)
        return 1
    on_pitch = on_pitch_names()
    for n in SOUTH_BOX | NORTH_BOX:
        if n not in on_pitch:
            print("box point missing from pitch", n)
            return 1
    if RAW_FILE["P9"].name != "P1-006.mp4":
        print("P9 must use P1-006.mp4")
        return 1
    if RAW_FILE["P1"].name != "P9-004.mp4":
        print("P1 must use P9-004.mp4")
        return 1
    fam = families()
    for key, names in fam.items():
        if names != on_pitch:
            print(f"{key} family is not full on-pitch set")
            return 1
    for n in EXTRA_XY:
        if n not in catalog:
            print("extra missing", n)
            return 1
    defaults = {c["id"]: c["order"] for c in CAMS}
    for cam, key in defaults.items():
        ys = [xy[1] for _, xy in ORDERS[key]]
        if max(ys) < 1 or min(ys) > -1:
            print(f"{cam} {key} does not span both sidelines", ys)
            return 1
    print("two boxes ok", len(on_pitch), "points")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
