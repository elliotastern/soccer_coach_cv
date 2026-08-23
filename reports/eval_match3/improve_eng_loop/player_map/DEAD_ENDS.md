# Player map — dead ends

| Tried | Result | Why not again |
|-------|--------|----------------|
| Full P8 L1 refit replacing H | Ball foot → off_pitch | Use **H_player** dual H; ball keeps H |
| Dual H_player @ depth landmarks | Player Goal2 spread ↑; ball `(~25,-13)` on H | Keep for P8 until midfield visible on still |
| Hull-for-fr-3022-only | Forbidden | One-seed overfit |
| Lower ball MIN_SUPPORT for player dots | Forbidden | Clear-ball precision |
| Fuse/ghost soften as primary | Wrong lever while H collapses depth | See players_pitch DEAD_ENDS |

## Worked (keep)

| Tried | Result |
|-------|--------|
| D1/D2 pack | P8 residual `off_pitch`; pack majority was `low_support` before H_p |
| Restore git HEAD P8 after failed L1 | Ball map `(~25, -13)` restored |
| H_p P7 `hull_image_points` | 6/6 in-FOV `low_support` feet → support 1.0; H unchanged |
| H_p P8 hull add 3 low_support feet | Lifted; ball kill switch held |

## Residual

- P8 upper/mid `off_pitch` on **H_player** still needs midfield-visible landmarks before another H refit
- Do not re-run full depth refit without a ball lower-FOV gate in the search objective

## L2 bottom (2026-08)

| Tried | Result |
|-------|--------|
| 6pt weighted **H_player** + ball-foot anchor | **Rejected** — landmark RT 58 m (anchor vs goal-box landmarks incompatible in one H) |
| P8 lower-zone fallback (`y≥350`, retry **H**) | **Promoted** — mid+lower when H_player fails; ball H unchanged |
| Step1 top-FOV hull + cross-cam probe | **14 hull pts**; P8 pack **100%** mapped; H_player RT probe 0.70m — manual midfield still next |

D3 baseline (H_player-only bands): bottom-third mostly `off_pitch`; ball maps on **H** at same y.
