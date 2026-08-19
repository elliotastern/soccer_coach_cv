"""Pitch 1 (FIFA) marks from docs — exact meters for each landmark."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PITCH1_JSON = ROOT / "docs/product/PITCH1_DIMENSIONS.json"


def load_pitch1() -> dict:
    return json.loads(PITCH1_JSON.read_text(encoding="utf-8"))


def pitch1_landmarks(rec: dict | None = None) -> dict:
    rec = rec or load_pitch1()
    L, W = float(rec["length_m"]), float(rec["width_m"])
    m = rec["marks"]
    hx, hy = L / 2, W / 2
    pd = float(m["penalty_area_depth_m"])
    ph = float(m["penalty_area_half_width_m"])
    gd = float(m["goal_area_depth_m"])
    gh = float(m["goal_area_half_width_m"])
    ps = float(m["penalty_spot_m"])
    cr = float(m["center_circle_radius_m"])
    gp = float(m["goal_post_half_m"])
    pts = {
        "halfway_near_touch": ("Halfway Left Sideline", (0.0, hy),
            f"Halfway × left touch. y=+{hy:.2f} m (P1 left)."),
        "halfway_far_touch": ("Halfway Right Sideline", (0.0, -hy),
            f"Halfway × right touch. y=-{hy:.2f} m (P1 right)."),
        "left_near_corner": ("South Left Corner", (-hx, hy),
            f"South-left flag. {L:.0f}×{W:.0f} corner, x=-{hx:.2f} y=+{hy:.2f}."),
        "left_far_corner": ("South Right Corner", (-hx, -hy),
            f"South-right flag. x=-{hx:.2f} y=-{hy:.2f}."),
        "right_near_corner": ("North Left Corner", (hx, hy),
            f"North-left flag — P1’s left (P8 near). x=+{hx:.2f} y=+{hy:.2f}."),
        "right_far_corner": ("North Right Corner", (hx, -hy),
            f"North-right flag — P1’s right (P9 near). x=+{hx:.2f} y=-{hy:.2f}."),
        "center": ("Center Spot", (0.0, 0.0),
            "Kickoff mark. Origin x=+0.00 y=+0.00."),
        "circle_near": ("Center Circle Left", (0.0, cr),
            f"Halfway × center circle, left. r={cr:.2f} m. x=+0.00 y=+{cr:.2f}."),
        "circle_far": ("Center Circle Right", (0.0, -cr),
            f"Halfway × center circle, right. r={cr:.2f} m. x=+0.00 y=-{cr:.2f}."),
        "left_box_goal_near": ("South Left Goal-Line Corner", (-hx, ph),
            f"18-yard × south goal line, left. {pd} m box, {ph} m left of center."),
        "left_box_goal_far": ("South Right Goal-Line Corner", (-hx, -ph),
            f"18-yard × south goal line, right. {pd} m box, {ph} m right of center."),
        "left_box_18_near": ("South Left 18-Yard Corner", (-hx + pd, ph),
            f"Outer 18-yard corner, south left. {pd} m from goal line, {ph} m left."),
        "left_box_18_far": ("South Right 18-Yard Corner", (-hx + pd, -ph),
            f"Outer 18-yard corner, south right. {pd} m from goal line, {ph} m right."),
        "left_6_goal_near": ("South Left 6-Yard Goal-Line Corner", (-hx, gh),
            f"6-yard × south goal line, left. {gd} m box, {gh} m left of center."),
        "left_6_goal_far": ("South Right 6-Yard Goal-Line Corner", (-hx, -gh),
            f"6-yard × south goal line, right. {gd} m box, {gh} m right of center."),
        "left_6_box_near": ("South Left 6-Yard Corner", (-hx + gd, gh),
            f"Outer 6-yard corner, south left. {gd} m from goal line, {gh} m left."),
        "left_6_box_far": ("South Right 6-Yard Corner", (-hx + gd, -gh),
            f"Outer 6-yard corner, south right. {gd} m from goal line, {gh} m right."),
        "left_post_near": ("South Left Goal Post", (-hx, gp),
            f"South left post. Goal {m['goal_width_m']} m wide, post {gp} m left of center."),
        "left_post_far": ("South Right Goal Post", (-hx, -gp),
            f"South right post. {gp} m right of center."),
        "left_penalty_spot": ("South Penalty Spot", (-hx + ps, 0.0),
            f"South penalty mark. {ps} m from goal line, on center."),
        "right_box_goal_near": ("North Left Goal-Line Corner", (hx, ph),
            f"18-yard × north goal line, left. {pd} m box, {ph} m left of center."),
        "right_box_goal_far": ("North Right Goal-Line Corner", (hx, -ph),
            f"18-yard × north goal line, right. {pd} m box, {ph} m right of center."),
        "right_box_18_near": ("North Left 18-Yard Corner", (hx - pd, ph),
            f"Outer 18-yard corner, north left. {pd} m from goal line, {ph} m left."),
        "right_box_18_far": ("North Right 18-Yard Corner", (hx - pd, -ph),
            f"Outer 18-yard corner, north right. {pd} m from goal line, {ph} m right."),
        "right_6_goal_near": ("North Left 6-Yard Goal-Line Corner", (hx, gh),
            f"6-yard × north goal line, left. {gd} m box, {gh} m left of center."),
        "right_6_goal_far": ("North Right 6-Yard Goal-Line Corner", (hx, -gh),
            f"6-yard × north goal line, right. {gd} m box, {gh} m right of center."),
        "right_6_box_near": ("North Left 6-Yard Corner", (hx - gd, gh),
            f"Outer 6-yard corner, north left. {gd} m from goal line, {gh} m left."),
        "right_6_box_far": ("North Right 6-Yard Corner", (hx - gd, -gh),
            f"Outer 6-yard corner, north right. {gd} m from goal line, {gh} m right."),
        "right_post_near": ("North Left Goal Post", (hx, gp),
            f"North left post. Goal {m['goal_width_m']} m wide, post {gp} m left of center."),
        "right_post_far": ("North Right Goal Post", (hx, -gp),
            f"North right post. {gp} m right of center."),
        "right_penalty_spot": ("North Penalty Spot", (hx - ps, 0.0),
            f"North penalty mark. {ps} m from goal line, on center."),
    }
    out = {}
    for name, (label, xy, spec) in pts.items():
        out[name] = {
            "label": label,
            "xy": [round(xy[0], 4), round(xy[1], 4)],
            "spec": spec,
        }
    return out
