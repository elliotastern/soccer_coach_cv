# Fuse vs single-cam batch (2026-08-25)

## Finding

Product **fuse mosaic** (prior 5-min render `emits_render.json`) and **single-cam batch** disagree hard on Match 4:

| Source | Emits (conf≥~0.8) | Types |
|--------|------------------:|-------|
| Fuse mosaic 5-min (stride 15, events bar) | **54** | shot 2 · movement 19 · recovery 7 · dribble 14 · **pass 12** |
| Batch P10 5-min | **16** | **pass 16** only |
| Batch P10 full (~215 s processed) | **16** | pass only · last emit @83 s · **~187 s silence** |

Same match time windows: fuse often emits **movement/dribble** where batch emits **pass** (see `real_fuse_eval_20s`, `real_fuse_eval_49s`).

## Implication for improvement

1. **Do not** retune batch emit thresholds to “match” fuse volume — different inputs (fused xy vs one cam).
2. **Product truth = fuse** for coach mosaic / Phase 1 multicam UI.
3. Coach scrub should rate **fuse mosaic events bar** once new `team_core` mosaic finishes (meta.json regenerates).
4. Single-cam batch export is still valid for delivery CSV, but event-type precision claims need **fuse gold**, not P10-only.

## Artifacts

- [FUSE_VS_BATCH_COMPARE_20260825.json](FUSE_VS_BATCH_COMPARE_20260825.json)
- [COACH_SCRUB_HUB.html](COACH_SCRUB_HUB.html)
- Prior mosaic emits: `reports/eval_match3/improve_eng_loop/match4_5min/emits_render.json` (pre–team_core re-render)

## Next

- Wait for mosaic5 `RENDER_DONE` → new `emits_render.json` + staged MP4
- Coach-rate fuse emits on finished mosaic
- Lock `real_fuse_eval_49s` / `69s` labels from fuse MP4s
