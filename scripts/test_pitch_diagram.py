#!/usr/bin/env python3
"""
Generate a reference HTML page with a top-down pitch diagram using full FIFA terminology.
Separate from the mark UI so we can revert to the old diagram if needed.
Run from project root. Output: data/output/2dmap_manual_mark/pitch_diagram_reference.html
View: http://localhost:8080/data/output/2dmap_manual_mark/pitch_diagram_reference.html
"""
from pathlib import Path
import math

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "data" / "output" / "2dmap_manual_mark"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUT_DIR / "pitch_diagram_reference.html"

# FIFA dimensions in meters; scale 10 px/m for SVG
SCALE = 10
LENGTH = 105   # touchlines
WIDTH = 68     # goal lines
CENTER_CIRCLE_R = 9.15  # 10 yards
GOAL_WIDTH = 7.32
GOAL_DEPTH = 2.5   # goal box depth (net) for diagram visibility
CORNER_ARC_R = 1  # 1 yard
FLAG_POLE_LEN = 14   # corner flag pole length (px), outward from corner
FLAG_W = 10          # flag width (px)
FLAG_H = 6           # flag height (px)

def _px(m):
    return m * SCALE

def _build_svg():
    w = _px(LENGTH)
    h = _px(WIDTH)
    cx = w / 2
    cy = h / 2
    goal_half = _px(GOAL_WIDTH / 2)
    goal_left_y1 = cy - goal_half
    goal_left_y2 = cy + goal_half
    goal_depth_px = _px(GOAL_DEPTH)
    cr = _px(CORNER_ARC_R)
    # 45° diagonal for corner flags (outward from each corner)
    r2 = math.sqrt(2)
    flag_d = FLAG_POLE_LEN / r2
    flag_fw = FLAG_W / r2
    flag_h2 = FLAG_H / (2 * r2)
    flag_diag_extent = (FLAG_POLE_LEN + FLAG_W) / r2
    # viewBox padding: enough for goals and diagonal flags so nothing is cut off
    pad_left = max(120, goal_depth_px + 30, flag_diag_extent + 15)
    pad_top = max(80, flag_diag_extent + 15)
    # Right side: content = goal + label "Goal + net" + flags. We draw at sx() = x+pad_left; viewBox right = w+pad_right. Need pad_right >= pad_left + extent.
    right_extent = max(goal_depth_px, flag_d + flag_fw, 85)  # 85 = room for "Goal + net" label
    pad_right = max(120, goal_depth_px + 30, flag_diag_extent + 15, pad_left + right_extent + 15)
    pad_bottom = max(120, flag_diag_extent + 15)
    vb_x = -pad_left
    vb_y = -pad_top
    vb_w = w + pad_left + pad_right
    vb_h = h + pad_top + pad_bottom
    # Shift all coords by (pad_left, pad_top)
    def s(x, y):
        return (x + pad_left, y + pad_top)
    def sx(x):
        return x + pad_left
    def sy(y):
        return y + pad_top

    svg_pitch = f'''<svg viewBox="{vb_x} {vb_y} {vb_w} {vb_h}" xmlns="http://www.w3.org/2000/svg" class="pitch-svg" style="overflow: visible;">
  <defs>
    <style>
      .pitch-rect {{ fill: #2d5a27; stroke: #fff; stroke-width: 2; }}
      .line {{ stroke: #fff; stroke-width: 2; fill: none; }}
      .line-thin {{ stroke: #fff; stroke-width: 1; fill: none; }}
      .label {{ fill: #fff; font-size: 11px; font-family: sans-serif; text-anchor: middle; }}
      .label-small {{ fill: #ddd; font-size: 9px; }}
      .spot {{ fill: #fff; stroke: #fff; }}
      .goal-frame {{ fill: none; stroke: #fff; stroke-width: 2.5; }}
      .goal-net {{ fill: url(#netPattern); stroke: #ccc; stroke-width: 0.5; opacity: 0.9; }}
      .flag-pole {{ stroke: #fff; stroke-width: 2; fill: none; }}
      .flag {{ fill: #ffcc00; stroke: #b8860b; stroke-width: 0.8; }}
    </style>
    <pattern id="netPattern" patternUnits="userSpaceOnUse" width="6" height="6">
      <path d="M 0 0 L 6 6 M 6 0 L 0 6" stroke="#e0e0e0" stroke-width="0.6" fill="none"/>
    </pattern>
  </defs>
  <!-- Boundaries: touchlines (long) + goal lines (short) -->
  <rect x="{sx(0)}" y="{sy(0)}" width="{w}" height="{h}" class="pitch-rect"/>
  <text x="{sx(cx)}" y="{sy(-50)}" class="label">Touchlines (sidelines): longer, length of field</text>
  <text x="{sx(-70)}" y="{sy(cy)}" class="label" transform="rotate(-90 {sx(-70)} {sy(cy)})">Goal lines (endlines): shorter, at each end</text>

  <!-- Halfway line -->
  <line x1="{sx(cx)}" y1="{sy(0)}" x2="{sx(cx)}" y2="{sy(h)}" class="line"/>
  <text x="{sx(cx)}" y="{sy(-25)}" class="label-small">Halfway line (touchline to touchline)</text>

  <!-- Center mark + center circle -->
  <circle cx="{sx(cx)}" cy="{sy(cy)}" r="4" class="spot"/>
  <circle cx="{sx(cx)}" cy="{sy(cy)}" r="{_px(CENTER_CIRCLE_R)}" class="line" fill="none"/>
  <text x="{sx(cx)}" y="{sy(cy - _px(CENTER_CIRCLE_R) - 12)}" class="label-small">Center circle (9.15m / 10 yards)</text>
  <text x="{sx(cx)}" y="{sy(cy + _px(CENTER_CIRCLE_R) + 20)}" class="label-small">Center mark</text>

  <!-- Goals: box with frame + net extending OFF pitch (left goal = left of x=0) -->
  <rect x="{sx(-goal_depth_px)}" y="{sy(goal_left_y1)}" width="{goal_depth_px}" height="{goal_left_y2 - goal_left_y1}" class="goal-frame"/>
  <rect x="{sx(-goal_depth_px)}" y="{sy(goal_left_y1)}" width="{goal_depth_px}" height="{goal_left_y2 - goal_left_y1}" class="goal-net"/>
  <!-- Right goal = right of x=w -->
  <rect x="{sx(w)}" y="{sy(goal_left_y1)}" width="{goal_depth_px}" height="{goal_left_y2 - goal_left_y1}" class="goal-frame"/>
  <rect x="{sx(w)}" y="{sy(goal_left_y1)}" width="{goal_depth_px}" height="{goal_left_y2 - goal_left_y1}" class="goal-net"/>
  <text x="{sx(-goal_depth_px/2 - 20)}" y="{sy(cy)}" class="label-small">Goal + net</text>
  <text x="{sx(w + goal_depth_px/2 + 20)}" y="{sy(cy)}" class="label-small">Goal + net</text>

  <!-- Corner arcs (1m radius) -->
  <path d="M {sx(cr)} {sy(0)} A {cr} {cr} 0 0 1 {sx(0)} {sy(cr)}" class="line-thin" fill="none"/>
  <path d="M {sx(w - cr)} {sy(0)} A {cr} {cr} 0 0 0 {sx(w)} {sy(cr)}" class="line-thin" fill="none"/>
  <path d="M {sx(w - cr)} {sy(h)} A {cr} {cr} 0 0 0 {sx(w)} {sy(h - cr)}" class="line-thin" fill="none"/>
  <path d="M {sx(cr)} {sy(h)} A {cr} {cr} 0 0 1 {sx(0)} {sy(h - cr)}" class="line-thin" fill="none"/>

  <!-- Corner flags: pole at 45° outward from corner + triangular flag at end (TL, TR, BR, BL) -->
  <line x1="{sx(0)}" y1="{sy(0)}" x2="{sx(-flag_d)}" y2="{sy(-flag_d)}" class="flag-pole"/>
  <polygon points="{sx(-flag_d-flag_fw)},{sy(-flag_d-flag_fw)} {sx(-flag_d+flag_h2)},{sy(-flag_d-flag_h2)} {sx(-flag_d-flag_h2)},{sy(-flag_d+flag_h2)}" class="flag"/>
  <line x1="{sx(w)}" y1="{sy(0)}" x2="{sx(w+flag_d)}" y2="{sy(-flag_d)}" class="flag-pole"/>
  <polygon points="{sx(w+flag_d+flag_fw)},{sy(-flag_d-flag_fw)} {sx(w+flag_d-flag_h2)},{sy(-flag_d+flag_h2)} {sx(w+flag_d+flag_h2)},{sy(-flag_d-flag_h2)}" class="flag"/>
  <line x1="{sx(w)}" y1="{sy(h)}" x2="{sx(w+flag_d)}" y2="{sy(h+flag_d)}" class="flag-pole"/>
  <polygon points="{sx(w+flag_d+flag_fw)},{sy(h+flag_d+flag_fw)} {sx(w+flag_d+flag_h2)},{sy(h+flag_d-flag_h2)} {sx(w+flag_d-flag_h2)},{sy(h+flag_d+flag_h2)}" class="flag"/>
  <line x1="{sx(0)}" y1="{sy(h)}" x2="{sx(-flag_d)}" y2="{sy(h+flag_d)}" class="flag-pole"/>
  <polygon points="{sx(-flag_d-flag_fw)},{sy(h+flag_d+flag_fw)} {sx(-flag_d+flag_h2)},{sy(h+flag_d+flag_h2)} {sx(-flag_d-flag_h2)},{sy(h+flag_d-flag_h2)}" class="flag"/>

  <text x="{sx(w/2)}" y="{sy(h + 45)}" class="label-small">Corner arcs (1m); corner flags at each corner</text>
</svg>'''

    return svg_pitch


def main():
    svg = _build_svg()
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Pitch diagram reference – boundaries &amp; midfield</title>
  <style>
    body {{ font-family: sans-serif; margin: 20px; background: #1a1a1a; color: #eee; }}
    h1 {{ color: #4CAF50; }}
    .info {{ background: #2d2d2d; padding: 14px; border-radius: 8px; margin: 16px 0; font-size: 14px; line-height: 1.5; }}
    .pitch-wrap {{ margin: 20px auto; max-width: 900px; background: #1a1a1a; border-radius: 8px; padding: 16px; overflow: visible; }}
    .pitch-svg {{ display: block; width: 100%; height: auto; overflow: visible; }}
  </style>
</head>
<body>
  <h1>Pitch diagram reference (top-down, 105m × 68m)</h1>
  <div class="info">
    <strong>Boundaries &amp; midfield:</strong> Touchlines (sidelines); goal lines (endlines); halfway line; center mark; center circle (9.15m / 10 yards).<br>
    <strong>Goal &amp; corners:</strong> Goal (posts + crossbar) at center of each goal line; corner arcs (1m); corner flags at each corner.
  </div>
  <div class="pitch-wrap">
    {svg}
  </div>
</body>
</html>"""
    OUT_FILE.write_text(html, encoding="utf-8")
    print(f"Wrote {OUT_FILE}")
    print(f"View: http://localhost:8080/data/output/2dmap_manual_mark/pitch_diagram_reference.html")


if __name__ == "__main__":
    main()
