# Kit-ref (pre-labeled team centroids)

**Default for Match 4 / new matches:** label kits once before batch, then seed `team_centroids.json` so Golden Batch clustering is skipped.

Cursor rule: `.cursor/rules/kit_ref.mdc`

## Label before batch

```bash
bash scripts/run_kit_label_dashboard.sh   # http://127.0.0.1:8503
```

1. Pick a clear pre-game / early frame (P10 works well).
2. Detect players → assign Team 0 / Team 1 (clear torsos only).
3. Save → **merges** with `kit_samples_bank.json` (does not wipe prior labels),
   writes cam + match-root `team_centroids.json`, timestamp-backs up the previous
   centroids. Replace-all is opt-in only.

## Resolve order (batch)

1. `team_assignment.kit_centroids_path` in YAML (if set and exists)
2. `{output_root}/team_centroids.json` (match-level)
3. `{run_dir}/team_centroids.json`

Match 4 scripts also seed before the cam loop:

- `$KIT_REF` env, else `$OUT/team_centroids.json`, else `$OUT/P10-match4/team_centroids.json`
- `cp -f` into each `$OUT/$cam/team_centroids.json` (kit-ref wins over Golden Batch junk)

## Match 4 continue

Canonical labeled file (reuse until lighting/kits change):

`data/output/match_4_5min/P10-match4/team_centroids.json`  
→ also keep a copy at `data/output/match_4_5min/team_centroids.json`.

```bash
# optional explicit override
KIT_REF=data/output/match_4_5min/team_centroids.json bash scripts/run_batch_match4_5min.sh
```

## Crop quality

- Features = color fractions + hue hist (pitch green stripped); no extra cleaning.
- Need clear torsos, correct team, **3–5 samples per side** (Match 4 currently 9 / 3 — more white-kit samples help).
- Skip blur, grass-heavy crops, and GK/ref if kits differ.

## Do / don’t

- **Do** re-label when lighting or kits change for a new match.
- **Do** keep kit meta (`source`, `n_samples`, `team_names`) on save — batch preserves it.
- **Don’t** reopen F4 reprojection / 3D fuse as part of kit work (separate digressions; F4 not promoted).
- **Don’t** hardcode a Match-4-only absolute path as the permanent default in `configs/default.yaml` (breaks Match 3); empty + script seeding is preferred.
