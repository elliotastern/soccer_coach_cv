# Catch: Match-4 kit consensus ≥9 (productize full-cam proof)

**Status:** Match3 proof **locked** (two disjoint windows, hold consensus **9.11** / **9.31** with P1+P6+quad + tune-fit freeze). Match4 quad-only stays ~**7.73** by multi-cam fraction ceiling — do **not** widen merge further for this metric.

**Blocked until:** Catch Tailscale SSH is up (`ssh catch-soccer` or `catch@100.113.134.41`).

## Why this is the lever

| Setup | Hold consensus | Multi-cam fraction |
|-------|---------------:|-------------------:|
| Match4 / Match3 **quad only** | ~6.5–8.2 | ~0.11–0.22 |
| Match3 **P1+P6+quad** + kit freeze | **≥9.1** | **≥0.43** |

Kit consensus 9 needs end-line cams in the fuse bag, plus frozen kit centroids (kit-ref or tune-fit→freeze). Soft merge alone cannot beat the quad multi-cam fraction ceiling (~8.6 even at perfect agree).

## Steps on Catch (when reachable)

1. Mac: `git push` (if code changed). Catch: `cd ~/soccer_coach_cv && git pull`
2. `source ~/.venvs/soccer-rfdetr312/bin/activate`
3. Kit-ref: label on 8503 or reuse `data/output/match_4_5min/team_centroids.json` if lighting unchanged. Seed match root + every cam (see [KIT_REF.md](KIT_REF.md)).
4. Batch with end cams (not quad-only default):

```bash
cd ~/soccer_coach_cv
CAMS="P10-match4 P9-match4 P7-match4 P8-match4 P1-match4 P6-match4" \
  bash scripts/run_batch_match4_5min.sh
```

Optional goals later: add `P_Goal1-match4 P_Goal2-match4` if raw + calibs exist.

5. Build a Match4 player det cache spanning those cams (mirror `scripts/ab_match3_fullcam_kit_consensus.py` pattern, or reuse batch track JSON if already multi-cam).
6. Score hold consensus with **frozen** kit-ref centroids (do not online-refit on the hold half).
7. **Pass:** hold consensus ≥ **9.0** and multi-cam fraction ≥ **0.30**.
8. Kill-switch: camera fusion eng-loop still all ≥9; ball emit ≥0.80; no merge soft_m change unless a separate holdout A/B passes.

## Mac after

```bash
bash scripts/pull_from_catch.sh
# or rsync from ~/soccer_exchange/from_catch/
```

Update `reports/eval_match3/improve_eng_loop/OVERNIGHT_STATUS.md` with Match4 hold numbers.

## Hard nos

- Do not chase kit consensus 9 via `PLAYER_MERGE_SOFT_M_LIVE` &gt; 4.5 on quad-only.
- Do not invent landmarks; do not lower emit / widen agree.
- Do not put developer GitHub auth on Catch.
