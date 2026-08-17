# Match 2 — cut pitch noise / hit P≥0.9 R≥0.8 (loop plan)

**Goal:** Keep the main in-play ball; kill small random noise around the pitch so multicam emitted outputs approach **P ≥ 0.9, R ≥ 0.8** (PoC gate: precision of emitted ≥ ~0.8).

**Evidence (gallery pitch trails):** Most >8m teleports are **camera switches** under FOV-approx mapping, not ByteTrack gaps. Main ball is already found well — noise is brief wrong winners + off-pitch / tiny dets.

## Do not do (unless gold proves otherwise)

- Super-low ByteTrack as primary fix (sticky FPs; weak on cam flips).
- Re-label gold for this pass.
- Dense SAHI on live path.

## Minimize user work

1. Ship **software gates first** (hysteresis + N-frame emit) — no clicks.
2. Manual H only when ready (unblocks true `in_pitch_bounds`).
3. Validate on existing gallery + Top Left gold / locked OOS — no new harvest unless stuck.

## Loop steps

| Step | Do | Done when |
|---|---|---|
| **1** | **Cam hysteresis** in product pick: keep current cam unless challenger is clearly better for `K` frames (larger side and/or conf margin) | Unit tests; fewer Cam5↔P1 flips on Bottom Right track |
| **2** | **N-frame emit stickiness**: only emit / plot when same cam (or same track) wins `N` consecutive frames (start N=3) | Unit tests; gallery trail less speckled |
| **3** | Wire 1–2 into `pick_product` / pitchmap render; re-render gallery (reuse det caches) | Dropdown gallery updated; stats: switch rate + big-jump count down |
| **4** | Score Top Left locked policy + gold (Cam4+/P10/P7 as available): report ΔP/ΔR vs baseline | Note in this file; stop if P drops >2pp without R gain |
| **5** | When `Cam4plus_manual.json` + `Cam5plus_manual.json` exist → `in_pitch_bounds` + wire AND with color gate | Sideline FPs rejected on Top Right OOS |
| **6** | Re-demo gallery with real H (drop `fov_approx` label for those cams) | Visual: noise on brown track gone |

## Defaults to try

- `K = 5` frames hysteresis  
- `N = 3` consecutive wins to emit  
- Emit conf floor stays **0.80** at product gate  
- Color pitch filter stays **on**

## Stop / handoff

- If 1–4 land and gold P/R already hit → stop; document.
- If noise remains mostly off-pitch → handoff for **manual calib** (step 5); idle loop until JSONs appear.
- If hysteresis hurts R on Top Left gold → dial `K`/`N` down and re-score once.

## Status

- **2026-08-16:** Plan created. Gallery live (`locked_oos_pitchmap_gallery/`). Pitch-mapping plan still blocked on manual H. Loop starts at step **1**.
