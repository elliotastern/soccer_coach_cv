# Catch remote compute (developer Mac → client GPU)

Use **Catch’s Ubuntu PC** (RTX 5090) to speed up Phase 1 work. The developer Mac is for Cursor, git, and review; **GPU-heavy jobs run on Catch** over Tailscale + SSH.

Related: [CATCH_MAC_FILE_SYNC.md](CATCH_MAC_FILE_SYNC.md) · [CATCH_MACHINE_CURSOR_CONTEXT.md](CATCH_MACHINE_CURSOR_CONTEXT.md) · [PHASE1_BATCH_TESTING.md](PHASE1_BATCH_TESTING.md)

---

## Split of machines

| Machine | Role |
|---------|------|
| **Developer Mac** | Cursor, `git push`, eng-loops on saved JSON, pull videos/CSV, Streamlit review of pulled output |
| **Catch PC** | `git pull`, batch pipeline, mosaic renders, fuse timeline builds (RF-DETR), Streamlit @ `127.0.0.1:8501` during Catch sessions |

Match 4 videos are on **Catch** (`~/soccer_coach_cv/data/raw/Match 3` → Match 4 folder). Running batch or mosaic on the Mac duplicates transfer time and is much slower (MPS vs 5090).

---

## What to run on Catch

| Job | Typical wall time (5090) | Script / entry |
|-----|--------------------------|----------------|
| 5-min quad batch | ~45–90 min | `bash scripts/run_batch_match4_5min.sh` |
| Full match batch (one cam) | ~4–6 h | `bash scripts/run_batch_match4_full_chunked.sh` |
| 5-min mosaic render | ~1–2 h | `render_phase1_check_mosaic.py` (stride 15, 300 s) |
| 15 s handover mosaic | ~15 min | `render_phase1_check_mosaic.py` (stride 4) |
| Fuse timeline (15 s, stride 4) | ~15 min | `build_fuse_15s_timeline.py` |

Use **tmux** so jobs keep running after SSH disconnect.

---

## SSH from Mac (one-time setup)

See [CATCH_MAC_FILE_SYNC.md](CATCH_MAC_FILE_SYNC.md). Example `~/.ssh/config`:

```text
Host catch-soccer
    HostName 100.113.134.41
    User catch
    IdentityFile ~/.ssh/id_ed25519_soccer_catch
    IdentitiesOnly yes
```

If SSH **times out**, Tailscale is usually stopped on the Mac — not a bad key. Run `bash scripts/diagnose_catch_tailscale.sh` or open Tailscale → Connect. Details: [CATCH_MAC_FILE_SYNC.md](CATCH_MAC_FILE_SYNC.md#tailscale-down-2026-09-05-incident).

```bash
ssh catch-soccer
```

---

## Standard remote session

```bash
# On Catch (after ssh)
source ~/.venvs/soccer-rfdetr312/bin/activate
cd ~/soccer_coach_cv && git pull

# Long job — always tmux
tmux new -s match4_5min
bash scripts/run_batch_match4_5min.sh
# Ctrl+B, D to detach

# Mosaic re-render (team_core stack)
tmux new -s mosaic5
python3 scripts/gold_set/render_phase1_check_mosaic.py \
  --start 0 --match-sec 300 --stride 15 --out-fps 4 \
  --out-dir reports/eval_match3/improve_eng_loop/match4_5min \
  --out-file coach_mosaic_pitch_5min.mp4
```

**Monitor from Mac:**

```bash
ssh catch-soccer 'tmux capture-pane -t mosaic5 -p | tail -5'
```

---

## Get results back on Mac

```bash
# On Catch — stage large files
cp reports/eval_match3/improve_eng_loop/match4_5min/coach_mosaic_pitch_5min.mp4 \
  ~/soccer_exchange/from_catch/

# On Mac
bash scripts/pull_from_catch.sh
# → ~/Downloads/soccer_catch_sync/
```

Or use `scripts/catch_stage_mosaic.sh` on Catch (auto-copy when render finishes).

---

## Cursor agents

Rule: `.cursor/rules/catch_remote_compute.mdc`

When estimating **>15 minutes** of RF-DETR or batch on the Mac, **SSH to Catch and start tmux** instead of blocking the developer machine. Push code first; pull artifacts after.

**Do not** put developer GitHub credentials on Catch — public clone + `git pull` only ([catch_client_credentials.mdc](../cursorrules/catch_client_credentials.mdc)).

---

## Active tmux sessions (examples)

| Session | Purpose |
|---------|---------|
| `match4_5min` | 5-min chunked quad batch |
| `match4_full` | Full-match P10 + P1 delivery batch |
| `mosaic5` | 5-min coach mosaic re-render |
| `mosaic_watch` | Auto-stage mosaic to `soccer_exchange/` |

List: `tmux ls` on Catch.
