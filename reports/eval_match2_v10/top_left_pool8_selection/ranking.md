# Top Left — 8-cam pool + largest_ball lock

Goal **R≥0.8 P≥0.9** on frames where selected cam has gold.
Gold cams: `Cam4plus, P10, P7`. Locked: `pool8_largest_ball_p7_thr060`.

| id | P | R | covered | unscored | goal | who wins |
|---|---:|---:|---:|---:|---|---|
| `pcam_max_conf_030` | 0.749 | 0.784 | 267 | 27 | MISS | P1 1%, P10 38%, P12 4%, P6 2%, P7 51%, P8 2% |
| `p7_thr060_others030` | 0.914 | 0.914 | 187 | 62 | HIT | P1 2%, P10 41%, P12 8%, P6 4%, P7 22%, P8 7% |
| `pool8_max_conf_p7_thr060` | 0.997 | 0.974 | 299 | 0 | HIT | Cam4plus 79%, P10 21% |
| `pool8_largest_ball_p7_thr060` | 0.993 | 0.948 | 277 | 22 | HIT | Cam4plus 93%, Cam5plus 6%, P8 1% |
| `pool8_largest_ball_030` | 0.993 | 0.948 | 277 | 22 | HIT | Cam4plus 93%, Cam5plus 6%, P8 1% |

## Locked result

- Gold-covered under lock: **277** frames (P=0.993 R=0.948).
- Unscored selected: **22** / 299. Cam4plus gold included in score.
- Selection: Cam4plus 93%, Cam5plus 6%, P8 1%

## Vs goal (honest)

| Slice | P | R | vs R≥0.8 P≥0.9 |
|---|---:|---:|---|
| P-cam-only prior lock | 0.914 | 0.914 | HIT (187 fr) |
| 8-cam max_conf | 0.997 | 0.974 | covered 299; unscored 0 |
| LOCKED largest_ball 8-cam | 0.993 | 0.948 | covered 277; unscored 22 |

**Closeness:** see chat — pack at `/4quad-cvat/top_left_cam4plus`.
