# Pre-overnight: camera fusion → 9/10+

**Goal:** improve camera fusion so ball / player / kit evaluation metrics rise. Product gates stay locked (`EMIT_CONF=0.80`, agree ≤4 m, Pitch 1, F1+F2+F0+F3, no inventing landmarks).

## Top 5 levers (ranked)

| Rank | Lever | Overfit risk | Why ranked here |
|-----:|-------|--------------|-----------------|
| **1** | **Detector soft-conf lift** (specialty residual, holdout frames out of train) | **Low** if holdout gallery + Match3 120–194 stay out; promote only on holdout + strip A/B | Clear FNs are 0/32 other-cam≥0.80; majority bucket is `mapped_conf_below_emit` 0.50–0.79. Fuse wiring is not the bottleneck. |
| 2 | Homography / map quality on existing landmarks (re-click / RT, not new marks) | Medium if tuned to one clip | H consistency is the ceiling for multi-cam agree; landmarks FOV already labelled. |
| 3 | Player pitch-map + cross-cam stitch quality | Medium | Kit metrics need stable player feet maps before color centroids help. |
| 4 | Kit-ref sticky assignment / cross-cam team consistency | Low–medium | Improves kit eval once player maps exist; already partially shipped. |
| 5 | Opt-in 3D triangulate when cams agree (keep F0 fallback) | Low if gated | Agree lift only where geometry supports it; default stays pitch_merge. |

**Do not overnight:** lower emit, widen agree, invent mid-pitch landmarks, hull-for-one-seed, Phase-2 RNN as Phase-1 requirement.

## Top-1 decision

**Top 1 does not overfit** under the residual recipe (strip soft/no_det + v12 anchors; holdout excluded; promote on held-out gallery). Proceed with loop/graph engineering on Top 1.

## Top-1 → 10 subgoals

1. Residual FN harvest correctness (v13 caches, no holdout leakage)
2. Train pack integrity (COCO counts, test=valid symlink, resolution 560)
3. Specialty resume continuity (absolute epochs 265→275 from v13)
4. Strip P10 clear_R / P_emit after train (no regress vs v13)
5. Strip P8 clear_R / P_emit after train (no regress vs v13)
6. Holdout gallery clear_ball_proxy_R ≥ gate (~0.884)
7. Holdout other-cam≥0.80 funnel stays ~0 (not fuse wiring regression)
8. Systemic FN bucket shrink (`mapped_conf_below_emit` P10/P6/P8)
9. Catch @1288 path ready when Tailscale returns (sync scripts)
10. Product pointer promote only if A/B passes; else DEAD_ENDS + keep v13

## Loop prompt (Top 1 / each subgoal)

```
Goal: detector soft-conf specialty (Top 1) until fusion-fed clear-ball metrics are 9/10+.
For THIS subgoal only:
1) Measure why it fails (script + JSON evidence; no guessing).
2) Write a 9/10+ “why” note in reports/eval_match3/improve_eng_loop/.
3) Only then change code/weights; A/B vs locked baseline.
4) Stop subgoal when score ≥9/10; else loop with a sharper prompt.
Hard nos: train on holdout gallery; lower EMIT; invent landmarks; hull for one clip.
```

## Current overnight status

- Product ball ckpt: `models/v13_residual_snaps/post_train/checkpoint.pth` (promoted)
- v14 Mac MPS train running: epochs 265→275 · log `/tmp/v14_residual_mac_train.log`
- Subgoals **1–3, 9** measured ≥9/10; **4–8, 10** have ≥9/10 *why* prompts; fix gated on train finish
- Auto finish: `scripts/overnight_v14_finish.sh` (waits for train → A/B → holdout → promote or DEAD_ENDS)
- Tracker: `reports/eval_match3/improve_eng_loop/overnight_top1_subgoal_tracker.json`
