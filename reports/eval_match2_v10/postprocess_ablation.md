# v10 inference post-process ranking

Frozen `models/v10_snaps/post_train/checkpoint.pth`. Pick on **train 87 @ conf ≥ 0.3** (precision ≥ 0.90 and FP ≤ baseline). Then score held-out **gold 50**. Kalman/ByteTrack not re-run ([track_tune.md](track_tune.md)).

**Locked pick:** `baseline_thr30_size_topk2` (thr 0.3 + geometry filter + NMS + topk=2).

No tested recover method beat baseline on train without adding FPs.

## Train 87 (124 GT balls)

| method | R@0.3 | P@0.3 | FP@0.3 | P_emit@0.8 | n@0.8 |
|---|---:|---:|---:|---:|---:|
| **baseline topk=2** | **0.952** | **0.975** | **3** | **1.00** | 44 |
| topk=3 | 0.952 | 0.975 | 3 | 1.00 | 44 |
| SAHI fallback-only | 0.952 | 0.975 | 3 | 1.00 | 44 |
| multiscale 1.5× | 0.960 | 0.967 | 4 | 1.00 | 44 |
| HFlip TTA + NMS | 0.968 | 0.952 | 6 | 1.00 | 57 |
| SAHI recover-always | 0.968 | 0.938 | 8 | 1.00 | 46 |
| topk=1 | 0.685 | 0.977 | 2 | 1.00 | 44 |

Multiscale / TTA / SAHI-always raised train recall only by adding FPs, so they were ineligible.

## Held-out gold 50 (62 GT balls)

| method | R@0.3 | P@0.3 | FP@0.3 | P_emit@0.8 | n@0.8 | FP@0.8 |
|---|---:|---:|---:|---:|---:|---:|
| **baseline topk=2** | **0.919** | **0.966** | **2** | **1.00** | 28 | 0 |
| topk=3 | 0.935 | 0.967 | 2 | 1.00 | 28 | 0 |
| SAHI fallback-only | 0.919 | 0.966 | 2 | 1.00 | 28 | 0 |
| multiscale 1.5× | 0.919 | 0.966 | 2 | 1.00 | 28 | 0 |
| HFlip TTA + NMS | 0.919 | 0.966 | 2 | 1.00 | 32 | 0 |
| SAHI recover-always | 0.919 | 0.905 | 6 | 0.97 | 29 | 1 |
| topk=1 | 0.774 | 0.960 | 2 | 1.00 | 28 | 0 |

Gold peek (not used for pick): topk=3 found one extra ball at the same 2 FPs; TTA published 4 more boxes at 0.8 with still 0 FPs. Train did not show those gains without extra FPs, so they stay off the locked stack.

## Final rating

| Rank | Method | Verdict |
|---|---|---|
| 1 | **thr 0.3 + size filter + NMS + topk=2** | **Use.** Best train score under the FP cap. Gold R 92% P 97%; P_emit 1.0 / 28. |
| 2 | topk=3 | Same train as baseline; gold +1 TP, no extra FP. Not locked (train-blind). |
| 3 | HFlip TTA | Train +FPs. Gold @0.3 unchanged; @0.8 more emits still P=1.0. Extra cost. |
| — | SAHI fallback-only | Identical to baseline (frames already have a det). |
| — | Multiscale 1.5× | Train +1 TP +1 FP; gold no gain. |
| Avoid | SAHI recover-always | Train/gold extra FPs; gold P_emit 0.97 with 1 FP @0.8. |
| Avoid | topk=1 | Drops second balls (train R 69%, gold R 77%). |
| Avoid | Kalman | Gold R 0.71 P 0.88 (prior tune). |
| Avoid | ByteTrack | Drops boxes; emit 0.80 R 0.40 (prior tune). |
| Publish-only | conf ≥ 0.80 | P_emit 1.0, not more balls. |

JSON: `postprocess_ablation.json`.
