# 4quad locked-policy survey (no new labels)

Policy: `pool8_largest_ball_p7_thr060` — pool Cam4+/Cam5+/P-cams, P7≥0.60 others≥0.30, `largest_ball`.
Caches only (no re-detect). Top Left has gold; other slots = **who wins + ball size**.

| Region | Locked winners | med px | ≥20px | none | Priority | Notes |
|---|---|---:|---:|---:|---|---|
| Top Left | Cam4plus 93%, Cam5plus 6%, P8 1% | 25.38 | 1.0 | 0% | **ok** | — |
| Top Right | Cam5plus 50%, Cam4plus 30%, P1 20% | 30.84 | 1.0 | 0% | **watch** | winner=Cam5plus unscored (no gold this slot) |
| Center Start | Cam4plus 47%, Cam5plus 46%, P1 7% | 25.68 | 1.0 | 0% | **watch** | winner=Cam4plus unscored (no gold this slot) |
| Bottom Right | Cam5plus 63%, Cam4plus 28%, P8 9% | 27.34 | 1.0 | 0% | **watch** | winner=Cam5plus unscored (no gold this slot) |

## Read (minimize labeling)

- **Need label now:** **none** — software survey OK
- **Watch (winner unscored / soft flags):** Top Right, Center Start, Bottom Right
- Next product step if priorities are ok/watch: **wire lock into live path**, then 5090 latency.
- Only label a non–Top Left Cam4/Cam5 window if a slot stays **high** after live wiring.

## Baseline max_conf @0.30 (comparison)

| Region | Baseline winners | med px |
|---|---|---:|
| Top Left | Cam4plus 79%, P10 21% | 24.5 |
| Top Right | Cam4plus 47%, P7 29%, P1 23% | 24.2 |
| Center Start | Cam4plus 82%, P8 12%, P1 5% | 24.16 |
| Bottom Right | Cam4plus 58%, Cam5plus 27%, P8 15% | 23.94 |
