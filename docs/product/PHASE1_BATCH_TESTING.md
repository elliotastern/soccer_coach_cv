# Phase 1 batch testing — default workflow

**Locked preference:** short chunked batch runs with **live Streamlit review** while output is being created. Not full-match overnight batch for testing or handover.

Cursor rule: `.cursor/rules/phase1_batch_testing.mdc`

---

## Why

| Approach | Wall time (5090, Match 4) | Review |
|----------|---------------------------|--------|
| Full 8-cam batch (`run_batch_match4.sh`) | ~26–28 hours | Only after each cam finishes |
| **5-min quad chunked (default)** | ~45–90 min | **After every ~30 s chunk** |

Chunked runs merge cumulative `frame_data.csv` / `events.json` into the run dir after each chunk so the dashboard shows growing coverage mid-run.

**Mac developer:** Do not run long batch/mosaic locally if Catch is online — SSH + tmux on the 5090 ([CATCH_REMOTE_COMPUTE.md](CATCH_REMOTE_COMPUTE.md)).

---

## Match 4 (Catch / RTX 5090)

```bash
# Stop a long full batch if still running
pkill -f batch_pipeline || true

tmux new -s match4_5min
source ~/.venvs/soccer-rfdetr312/bin/activate
cd ~/soccer_coach_cv && git pull
bash scripts/run_batch_match4_5min.sh
# Ctrl+B then D to detach
```

| Parameter | Default | Override |
|-----------|---------|----------|
| Output root | `data/output/match_4_5min` | 1st arg |
| Frames per cam | `18000` (5 min @ 60 fps) | 2nd arg |
| Chunk size | `1800` (~30 s video) | 3rd arg |
| Cameras | P10, P9, P7, P8 (quad) | `CAMS=P10-match4` … |

**Config:** merges `configs/default.yaml` + `configs/batch_rtx5090.yaml` (no `enhance_ball` / `use_kalman`; 600-frame event checkpoints).

**Kit-ref before batch:** label on http://127.0.0.1:8503 (`bash scripts/run_kit_label_dashboard.sh`), then ensure `data/output/match_4_5min/team_centroids.json` (or `$KIT_REF`) exists — scripts seed every cam. See [KIT_REF.md](KIT_REF.md).

**Review while running:** http://127.0.0.1:8501 → **Expert mode** → output root `data/output/match_4_5min`. Refresh after each chunk.

**Fastest smoke (one cam):**

```bash
CAMS=P10-match4 bash scripts/run_batch_match4_5min.sh
```

**Progress:**

```bash
tail -f reports/eval_match3/improve_eng_loop/batch_match4_5min_*.log
watch -n 30 'wc -l data/output/match_4_5min/*/frame_data.csv 2>/dev/null'
```

---

## Match 3 / Mac smoke

Same chunk + cumulative promote pattern:

```bash
bash scripts/run_phase1_2min_sample.sh
# → data/output/full_match_2min/ (2 min default; uses batch_mac_stable.yaml)
```

---

## Full-match batch (explicit only)

Use only when the user asks for **delivery** or a **complete match** archive:

```bash
bash scripts/run_batch_match4.sh   # → data/output/match_4/ (8 cams, no --max-frames)
```

Run overnight in tmux. Not the default for testing or client handover sessions.

---

## Related docs

- [MATCH_REVIEW_HANDOVER.md](MATCH_REVIEW_HANDOVER.md) — dashboard + output layout
- [CATCH_MACHINE_CURSOR_CONTEXT.md](CATCH_MACHINE_CURSOR_CONTEXT.md) — Catch machine setup
- [PHASE1_SCOPE.md](PHASE1_SCOPE.md) — Phase 1 delivery scope
