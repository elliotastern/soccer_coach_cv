# Match 2 gold 100 frames in a row test

Window: Match 2 t=33.0s, 100 consecutive frames. Checkpoint: `models/v8_snaps/post_train/checkpoint.pth`.

Detect floor 0.30. Green boxes = conf ≥ 0.8. Clear-ball size proxy = min side ≥ 25 px on full-res.

**Dashboard:** run `python3 serve_viewer.py` then open [http://127.0.0.1:8080/match2-100row](http://127.0.0.1:8080/match2-100row)

See also [PATH_TO_80.md](PATH_TO_80.md).

## Match 2 (new settings, 8 cams)

| cam | n | @0.5 | @0.8 | mean side | median side | p10 side | clear≥25 @0.5 | max conf |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| P1 | 100 | 0 (0%) | 0 (0%) | 39.6 | 37.2 | 34.7 | 0 (0%) | 0.388 |
| P6 | 100 | 0 (0%) | 0 (0%) | 27.4 | 26.1 | 19.2 | 0 (0%) | 0.424 |
| P7 | 100 | 0 (0%) | 0 (0%) | 27.1 | 27.0 | 23.6 | 0 (0%) | 0.484 |
| P8 | 100 | 2 (2%) | 0 (0%) | 29.9 | 30.2 | 24.8 | 2 (2%) | 0.548 |
| P10 | 100 | 10 (10%) | 0 (0%) | 32.5 | 32.6 | 27.9 | 10 (10%) | 0.637 |
| P12 | 100 | 0 (0%) | 0 (0%) | — | — | — | 0 (0%) | — |
| Cam4plus | 100 | 25 (25%) | 0 (0%) | 32.5 | 32.2 | 27.1 | 25 (25%) | 0.703 |
| Cam5plus | 100 | 44 (44%) | 0 (0%) | 61.1 | 71.7 | 19.0 | 44 (44%) | 0.664 |

## Match 1 baseline (old settings, multicam_20s)

| cam | n | @0.5 | @0.8 | mean side | median side | p10 side | clear≥25 @0.5 | max conf |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| match1_cam8 | 100 | 0 (0%) | 0 (0%) | — | — | — | 0 (0%) | — |
| match1_cam9 | 100 | 2 (2%) | 0 (0%) | 33.9 | 25.5 | 22.2 | 2 (2%) | 0.505 |
| match1_cam11 | 100 | 0 (0%) | 0 (0%) | — | — | — | 0 (0%) | — |
| match1_cam13 | 100 | 0 (0%) | 0 (0%) | 23.4 | 23.1 | 22.7 | 0 (0%) | 0.349 |

## Artifacts

- Per-cam: `<cam>/contact_10x10.jpg`, `<cam>/overlay.mp4`
- Mosaic: `mosaic_contact.jpg`
- Review pack: `data/processed/match2_gold_100_in_a_row/`
- Raw: `summary.json`

## Claim limit

Visual size + raw detect/emit rates only. Not stratified Match Gold100; not true P_emit.
