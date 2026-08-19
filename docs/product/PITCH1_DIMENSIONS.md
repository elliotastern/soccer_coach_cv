# Pitch 1 dimensions

Canonical **Pitch 1** is FIFA / IFAB standard. Source file: [PITCH1_DIMENSIONS.json](PITCH1_DIMENSIONS.json). Code: `scripts/gold_set/pitch1.py`.

Used by Match 3 landmark clicks (`match3_landmarks.py`) and `configs/default.yaml` (`105 × 68` m).

## Field

| Mark | Meters |
|------|--------|
| Length (touchline, south→north) | **105.0** |
| Width (goal line, left→right from P1) | **68.0** |
| Goal width | 7.32 |
| Goal post from center | 3.66 |
| Penalty area (18-yard) depth | 16.5 |
| Penalty area width | 40.32 (20.16 either side of center) |
| Goal area (6-yard) depth | 5.5 |
| Goal area width | 18.32 (9.16 either side of center) |
| Penalty spot from goal line | 11.0 |
| Center circle radius | 9.15 |
| Corner arc radius | 1.0 |

Origin is pitch **center**. **+x = north (P6)**, **−x = south (P1)**. **+y = left**, **−y = right** (from P1 looking north).

## Landmarks (benchmarks)

| Id | Label | x (m) | y (m) | Exact size |
|----|-------|------:|------:|------------|
| `halfway_near_touch` | Halfway Left Sideline | 0.00 | 34.00 | Halfway × left touch. y=+34.00 m (P1 left). |
| `halfway_far_touch` | Halfway Right Sideline | 0.00 | -34.00 | Halfway × right touch. y=-34.00 m (P1 right). |
| `left_near_corner` | South Left Corner | -52.50 | 34.00 | South-left flag. 105×68 corner. |
| `left_far_corner` | South Right Corner | -52.50 | -34.00 | South-right flag. |
| `right_near_corner` | North Left Corner | 52.50 | 34.00 | North-left flag — P1’s left (P8 near). |
| `right_far_corner` | North Right Corner | 52.50 | -34.00 | North-right flag — P1’s right (P9 near). |
| `center` | Center Spot | 0.00 | 0.00 | Kickoff mark. Origin (0, 0). |
| `circle_near` | Center Circle Left | 0.00 | 9.15 | Halfway × circle, left. r=9.15 m. |
| `circle_far` | Center Circle Right | 0.00 | -9.15 | Halfway × circle, right. r=9.15 m. |
| `left_box_goal_near` | South Left Goal-Line Corner | -52.50 | 20.16 | 18-yard × south goal line, left. 16.5 m box. |
| `left_box_goal_far` | South Right Goal-Line Corner | -52.50 | -20.16 | 18-yard × south goal line, right. |
| `left_box_18_near` | South Left 18-Yard Corner | -36.00 | 20.16 | 16.5 m from south goal line, 20.16 m left. |
| `left_box_18_far` | South Right 18-Yard Corner | -36.00 | -20.16 | 16.5 m from south goal line, 20.16 m right. |
| `left_6_goal_near` | South Left 6-Yard Goal-Line Corner | -52.50 | 9.16 | 6-yard × south goal line, left. 5.5 m box. |
| `left_6_goal_far` | South Right 6-Yard Goal-Line Corner | -52.50 | -9.16 | 6-yard × south goal line, right. |
| `left_6_box_near` | South Left 6-Yard Corner | -47.00 | 9.16 | 5.5 m from south goal line, 9.16 m left. |
| `left_6_box_far` | South Right 6-Yard Corner | -47.00 | -9.16 | 5.5 m from south goal line, 9.16 m right. |
| `left_post_near` | South Left Goal Post | -52.50 | 3.66 | Goal 7.32 m wide. |
| `left_post_far` | South Right Goal Post | -52.50 | -3.66 | Goal 7.32 m wide. |
| `left_penalty_spot` | South Penalty Spot | -41.50 | 0.00 | 11 m from south goal line. |
| `right_box_goal_near` | North Left Goal-Line Corner | 52.50 | 20.16 | 18-yard × north goal line, left. |
| `right_box_goal_far` | North Right Goal-Line Corner | 52.50 | -20.16 | 18-yard × north goal line, right. |
| `right_box_18_near` | North Left 18-Yard Corner | 36.00 | 20.16 | 16.5 m from north goal line, 20.16 m left. |
| `right_box_18_far` | North Right 18-Yard Corner | 36.00 | -20.16 | 16.5 m from north goal line, 20.16 m right. |
| `right_6_goal_near` | North Left 6-Yard Goal-Line Corner | 52.50 | 9.16 | 6-yard × north goal line, left. |
| `right_6_goal_far` | North Right 6-Yard Goal-Line Corner | 52.50 | -9.16 | 6-yard × north goal line, right. |
| `right_6_box_near` | North Left 6-Yard Corner | 47.00 | 9.16 | 5.5 m from north goal line, 9.16 m left. |
| `right_6_box_far` | North Right 6-Yard Corner | 47.00 | -9.16 | 5.5 m from north goal line, 9.16 m right. |
| `right_post_near` | North Left Goal Post | 52.50 | 3.66 | Goal 7.32 m wide. |
| `right_post_far` | North Right Goal Post | 52.50 | -3.66 | Goal 7.32 m wide. |
| `right_penalty_spot` | North Penalty Spot | 41.50 | 0.00 | 11 m from north goal line. |

P8’s tight north-goal FOV: use the **north 18-yard** and **north 6-yard** rows (plus north-left corner / posts / penalty spot). Halfway and far-sideline points are not in that still.
