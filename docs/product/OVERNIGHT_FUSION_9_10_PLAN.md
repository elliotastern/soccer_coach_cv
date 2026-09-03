# Pre-overnight → post-v14: camera fusion → 9/10+

**Goal:** improve camera fusion so ball / player / kit evaluation metrics rise. Product gates stay locked (`EMIT_CONF=0.80`, agree ≤4 m, Pitch 1, F1+F2+F0+F3, no inventing landmarks).

## Top 5 levers (current)

| Rank | Lever | Overfit risk | Status |
|-----:|-------|--------------|--------|
| ~~1~~ | Detector soft-conf residual | Low (holdout out of train) | **Done — v14 promoted** (holdout 0.960; strips 0.929/0.908) |
| ~~2~~ | Widen player merge for kit | Medium (smoke/tune) | **Dead end** — hold +0.15 &lt; bar |
| ~~3~~ | Auto-H / snap / drop existing P1/P6 marks | High if cold promote | **Dead end** — cannot hit RT ≤0.15 |
| **1 (now)** | **Human re-click P1/P6 worst existing IDs** | Medium if one-clip | Only path for `l2_overlap`→9 |
| **2** | Expand Match-4 multi-cam kit batch (P1/P6/goals) when Catch up | Low if holdout-gated | Unlocks kit consensus beyond H-span on quad-only |
| 3 | Opt-in 3D when agree | Low | Already ~parity |
| 4 | v15 residual on remaining ~11 soft FNs | Low if holdout excluded | Diminishing; funnel still 0 other-cam≥0.80 |
| 5 | Catch @1288 when Tailscale returns | — | Deferred |

**Do not overnight:** lower emit, widen agree, invent mid-pitch landmarks, hull-for-one-seed, drop F3 because mean-clear_R preferred F0-only.

## Scoreboard (v14 strip caches refreshed)

- `product_goals` **10.0** · `f_post` **10.0** · `product_post` **10.0**
- Only fail: `l2_overlap=6.0` (P1/P6 live fit residual &gt;0.15 m)

## Top-1 now → 10 subgoals (human re-click)

1. Confirm worst residuals from `h_consistency/snap_drop_h_ab.json` / live `landmark_roundtrip_m`
2. Open landmark dashboard stills for P1 then P6 (defished)
3. Nudge only listed existing names (no new IDs)
4. Refit H via `refit_match3_h_from_landmarks.py` (backup on)
5. Write `roundtrip_max_m` from live residual
6. Rematch strip gold if H-seeded (`rematch_match3_m1_gold.py`)
7. Kill-switch: strip P_emit ≥0.80 both packs
8. Kill-switch: holdout clear proxy ≥0.884
9. Re-run `eng_loop_match3_improve.py` — need `l2_overlap` ≥9
10. If any kill-switch trips → restore backup; log DEAD_ENDS

## Loop prompt

```
Goal: P1/P6 landmark fit residual ≤0.15 m on EXISTING names only (Top-1 human re-click).
For THIS subgoal only:
1) Measure why (per-point residual JSON; no guessing).
2) Why note ≥9/10 in improve_eng_loop/.
3) Only then nudge clicks / refit; A/B vs backup calib.
4) Stop when l2_overlap ≥9 and strip/holdout kill-switches pass.
Hard nos: invent landmarks; lower EMIT; widen AGREE; promote cold auto-H.
```

## Completed overnight

- v14 residual promote · strip labels → v14 det caches · `f_post_ab` refresh with **product-locked** winner F1+F2+F0+F3
- Kit merge A/B no promote · Auto-H/snap/drop no promote
- Tracker / status: `reports/eval_match3/improve_eng_loop/OVERNIGHT_STATUS.md`
