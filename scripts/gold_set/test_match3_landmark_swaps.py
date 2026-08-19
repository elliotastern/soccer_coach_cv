#!/usr/bin/env python3
"""Both goal boxes on the pitch; camera id = video filename P-code."""
from __future__ import annotations

from match3_landmarks import (
    CAMS,
    DISPLAY,
    ORDERS,
    P9_VISIBLE,
    RAW_FILE,
    all_landmarks,
    families,
    nearby_unused,
    on_pitch_names,
    resolve_landmark_names,
    save_clicks,
    swap_groups,
)
from raw_cam_id import cam_id_from_raw_name

SOUTH_BOX = {
    "left_box_goal_near", "left_box_goal_far", "left_box_18_near",
    "left_box_18_far", "left_post_near", "left_post_far",
}
NORTH_BOX = {
    "right_box_goal_near", "right_box_goal_far", "right_box_18_near",
    "right_box_18_far", "right_post_near", "right_post_far",
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
    for cam, path in RAW_FILE.items():
        got = cam_id_from_raw_name(path.name)
        if got != cam:
            print(f"{path.name} parsed as {got}, mapped as {cam}")
            return 1
    if RAW_FILE["P1"].name != "P1-006.mp4":
        print("P1 must be P1-006.mp4 (title = camera)")
        return 1
    if RAW_FILE["P9"].name != "P9-004.mp4":
        print("P9 must be P9-004.mp4 (title = camera)")
        return 1
    if cam_id_from_raw_name("P1-006.mp4") != "P1":
        print("P1-006.mp4 must parse as P1")
        return 1
    fam = families()
    for key, names in fam.items():
        if names != on_pitch:
            print(f"{key} family is not full on-pitch set")
            return 1
    defaults = {c["id"]: c["order"] for c in CAMS}
    for cam, key in defaults.items():
        ys = [xy[1] for _, xy in ORDERS[key]]
        if max(ys) < 1 or min(ys) > -1:
            print(f"{cam} {key} does not span both sidelines", ys)
            return 1
    for name, (x, y) in catalog.items():
        label = DISPLAY[name]
        if "Left" in label and y <= 0:
            print(f"{label} must have +y (P1 left), got {y}")
            return 1
        if "Right" in label and y >= 0:
            print(f"{label} must have -y (P1 right), got {y}")
            return 1
        if "North" in label and x <= 2:
            print(f"{label} must have +x (north), got {x}")
            return 1
        if "South" in label and x >= -2:
            print(f"{label} must have -x (south), got {x}")
            return 1
    if defaults["P9"] != "both_sides_north":
        print("P9 default order must be both_sides_north (near corner = North Right)")
        return 1
    grouped = []
    for g in swap_groups():
        grouped.extend(g["names"])
    if sorted(grouped) != sorted(on_pitch):
        print("swap_groups must list each on-pitch mark once")
        return 1
    used = {
        "right_box_goal_near", "right_box_goal_far",
        "right_box_18_near", "right_box_18_far",
    }
    near = nearby_unused("right_box_goal_near", used, 5)
    if "right_post_near" not in near:
        print("P8 unknown swap should offer north left post", near)
        return 1
    if set(near) & used:
        print("nearby_unused leaked live marks", near)
        return 1
    try:
        nearby_unused("not_a_mark", set())
        print("unknown landmark should raise")
        return 1
    except ValueError as err:
        if "unknown landmark" not in str(err):
            print(err)
            return 1
    if "right_box_18_far" not in catalog:
        print("catalog missing right_box_18_far")
        return 1
    names, pts = resolve_landmark_names("both_sides_north", P9_VISIBLE)
    if names != P9_VISIBLE or len(pts) != 4:
        print("P9 visible set failed", names)
        return 1
    dry = save_clicks("P9", "both_sides_north", [], landmark_names=P9_VISIBLE, dry_run=True)
    if not dry.get("ok") or "right_box_18_far" not in dry.get("landmarks", []):
        print("dry save rejected P9 visible set", dry)
        return 1
    try:
        resolve_landmark_names("both_sides_north", ["not_a_mark"] + P9_VISIBLE[:3])
        print("unknown landmark should raise")
        return 1
    except ValueError as err:
        if "unknown landmarks" not in str(err):
            print(err)
            return 1
    print("two boxes ok", len(on_pitch), "points")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
