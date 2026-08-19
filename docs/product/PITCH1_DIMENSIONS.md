# Pitch 1 (Field 1) dimensions

**Official field** for this product. Canonical **Pitch 1 / Field 1** is the measured pitch in [pitch 1 dimensions.webp](../pitch%201%20dimensions.webp) — **not** FIFA/IFAB 105 × 68 m.

Source file: [PITCH1_DIMENSIONS.json](PITCH1_DIMENSIONS.json). Code: `scripts/gold_set/pitch1.py`. Used by Match 3 landmark clicks (`match3_landmarks.py`), `src/mapping/pitch_bounds.py`, and `configs/default.yaml`.

## Field

| Mark | Meters |
|------|--------|
| Length (touchline, south→north) | **53.90** |
| Width south / Goal 1 (P1) | **34.84** (15.97 + 4.65 + 14.22) |
| Width north / Goal 2 (P6) | **34.81** (16.02 + 4.65 + 14.14) |
| Goal clear width | 4.65 |
| Goal post height | 2.24 |
| Goal box depth | 5.95 |
| Goal box width | 8.95 (centred on each goal, not on the pitch midline) |
| Centre circle diameter / radius | 7.00 / 3.50 |

Goals are **not** centred on the end lines. Origin is pitch **center**. **+x = north (P6)**, **−x = south (P1)**. **+y = left**, **−y = right** (from P1 looking north).

Diagram corners: Corner 2 = south-left, Corner 1 = south-right (P1), Corner 4 = north-left, Corner 3 = north-right (P6).

## Landmarks (benchmarks)

| Id | Label | x (m) | y (m) | Exact size |
|----|-------|------:|------:|------------|
| `halfway_near_touch` | Halfway Left Sideline | 0.00 | 17.41 | Halfway × left touch. |
| `halfway_far_touch` | Halfway Right Sideline | 0.00 | -17.41 | Halfway × right touch. |
| `left_near_corner` | South Left Corner | -26.95 | 17.42 | Corner 2. |
| `left_far_corner` | South Right Corner | -26.95 | -17.42 | Corner 1 (P1). |
| `right_near_corner` | North Left Corner | 26.95 | 17.41 | Corner 4 (P8 near). |
| `right_far_corner` | North Right Corner | 26.95 | -17.41 | Corner 3 (P9 near). |
| `center` | Center Spot | 0.00 | 0.00 | Kickoff mark. |
| `circle_near` | Center Circle Left | 0.00 | 3.50 | r=3.50 m. |
| `circle_far` | Center Circle Right | 0.00 | -3.50 | r=3.50 m. |
| `left_box_goal_near` | South Left Box Goal-Line Corner | -26.95 | 3.60 | 5.95 m box, left of south goal. |
| `left_box_goal_far` | South Right Box Goal-Line Corner | -26.95 | -5.35 | 5.95 m box, right of south goal. |
| `left_box_18_near` | South Left Box Corner | -21.00 | 3.60 | Outer south-left box. |
| `left_box_18_far` | South Right Box Corner | -21.00 | -5.35 | Outer south-right box. |
| `left_post_near` | South Left Goal Post | -26.95 | 1.45 | 15.97 m from Corner 2. |
| `left_post_far` | South Right Goal Post | -26.95 | -3.20 | 14.22 m from Corner 1. |
| `right_box_goal_near` | North Left Box Goal-Line Corner | 26.95 | 3.54 | 5.95 m box, left of north goal. |
| `right_box_goal_far` | North Right Box Goal-Line Corner | 26.95 | -5.42 | 5.95 m box, right of north goal. |
| `right_box_18_near` | North Left Box Corner | 21.00 | 3.54 | Outer north-left box. |
| `right_box_18_far` | North Right Box Corner | 21.00 | -5.42 | Outer north-right box. |
| `right_post_near` | North Left Goal Post | 26.95 | 1.39 | 16.02 m from Corner 4. |
| `right_post_far` | North Right Goal Post | 26.95 | -3.27 | 14.14 m from Corner 3. |

No FIFA 6-yard box or penalty spot on this plan. Keep `*_box_18_*` ids for the 5.95 m box so existing clicks still resolve.

P8’s tight north-goal FOV: use the **north box** rows (plus north-left corner / posts). Halfway and far-sideline points are not in that still.
