"""Pitch 1 marks from the measured plan — exact meters for each landmark."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PITCH1_JSON = ROOT / "docs/product/PITCH1_DIMENSIONS.json"


def load_pitch1() -> dict:
    return json.loads(PITCH1_JSON.read_text(encoding="utf-8"))


def _end_ys(width_m: float, left_to_post: float, goal_w: float, box_w: float) -> dict:
    hy = width_m / 2.0
    post_l = hy - left_to_post
    post_r = post_l - goal_w
    mid = 0.5 * (post_l + post_r)
    half = box_w / 2.0
    return {
        "hy": hy,
        "post_l": post_l,
        "post_r": post_r,
        "box_l": mid + half,
        "box_r": mid - half,
    }


def _pt(label: str, xy: tuple[float, float], spec: str) -> dict:
    return {"label": label, "xy": [round(xy[0], 4), round(xy[1], 4)], "spec": spec}


def pitch1_landmarks(rec: dict | None = None) -> dict:
    rec = rec or load_pitch1()
    L = float(rec["length_m"])
    m = rec["marks"]
    hx = L / 2.0
    pd = float(m["penalty_area_depth_m"])
    bw = float(m["penalty_area_width_m"])
    cr = float(m["center_circle_radius_m"])
    gw = float(m["goal_width_m"])
    south = _end_ys(m["south"]["width_m"], m["south"]["left_corner_to_post_m"], gw, bw)
    north = _end_ys(m["north"]["width_m"], m["north"]["left_corner_to_post_m"], gw, bw)
    hy_mid = 0.5 * (south["hy"] + north["hy"])
    return {
        "halfway_near_touch": _pt(
            "Halfway Left Sideline", (0.0, hy_mid),
            f"Halfway × left touch. y=+{hy_mid:.2f} m (P1 left)."),
        "halfway_far_touch": _pt(
            "Halfway Right Sideline", (0.0, -hy_mid),
            f"Halfway × right touch. y=-{hy_mid:.2f} m (P1 right)."),
        "left_near_corner": _pt(
            "South Left Corner", (-hx, south["hy"]),
            f"South-left flag (Corner 2). {L:.2f}×{south['hy'] * 2:.2f} m."),
        "left_far_corner": _pt(
            "South Right Corner", (-hx, -south["hy"]),
            f"South-right flag (Corner 1). x={-hx:.2f} y={-south['hy']:.2f}."),
        "right_near_corner": _pt(
            "North Left Corner", (hx, north["hy"]),
            f"North-left flag (Corner 4 / P8 near). x=+{hx:.2f} y=+{north['hy']:.2f}."),
        "right_far_corner": _pt(
            "North Right Corner", (hx, -north["hy"]),
            f"North-right flag (Corner 3 / P9 near). x=+{hx:.2f} y={-north['hy']:.2f}."),
        "center": _pt("Center Spot", (0.0, 0.0), "Kickoff mark. Origin x=+0.00 y=+0.00."),
        "circle_near": _pt(
            "Center Circle Left", (0.0, cr),
            f"Halfway × centre circle, left. r={cr:.2f} m."),
        "circle_far": _pt(
            "Center Circle Right", (0.0, -cr),
            f"Halfway × centre circle, right. r={cr:.2f} m."),
        "left_box_goal_near": _pt(
            "South Left Box Goal-Line Corner", (-hx, south["box_l"]),
            f"South box × goal line, left. {pd} m box, y=+{south['box_l']:.2f}."),
        "left_box_goal_far": _pt(
            "South Right Box Goal-Line Corner", (-hx, south["box_r"]),
            f"South box × goal line, right. y={south['box_r']:.2f}."),
        "left_box_18_near": _pt(
            "South Left Box Corner", (-hx + pd, south["box_l"]),
            f"Outer south-left box. {pd} m from goal line."),
        "left_box_18_far": _pt(
            "South Right Box Corner", (-hx + pd, south["box_r"]),
            f"Outer south-right box. {pd} m from goal line."),
        "left_post_near": _pt(
            "South Left Goal Post", (-hx, south["post_l"]),
            f"South left post. Goal {gw} m, {m['south']['left_corner_to_post_m']} m from Corner 2."),
        "left_post_far": _pt(
            "South Right Goal Post", (-hx, south["post_r"]),
            f"South right post. {m['south']['right_corner_to_post_m']} m from Corner 1."),
        "right_box_goal_near": _pt(
            "North Left Box Goal-Line Corner", (hx, north["box_l"]),
            f"North box × goal line, left. {pd} m box, y=+{north['box_l']:.2f}."),
        "right_box_goal_far": _pt(
            "North Right Box Goal-Line Corner", (hx, north["box_r"]),
            f"North box × goal line, right. y={north['box_r']:.2f}."),
        "right_box_18_near": _pt(
            "North Left Box Corner", (hx - pd, north["box_l"]),
            f"Outer north-left box. {pd} m from goal line."),
        "right_box_18_far": _pt(
            "North Right Box Corner", (hx - pd, north["box_r"]),
            f"Outer north-right box. {pd} m from goal line."),
        "right_post_near": _pt(
            "North Left Goal Post", (hx, north["post_l"]),
            f"North left post. Goal {gw} m, {m['north']['left_corner_to_post_m']} m from Corner 4."),
        "right_post_far": _pt(
            "North Right Goal Post", (hx, north["post_r"]),
            f"North right post. {m['north']['right_corner_to_post_m']} m from Corner 3."),
    }
