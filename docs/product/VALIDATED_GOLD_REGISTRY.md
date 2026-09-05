# Validated gold bounding-box registry

**Purpose:** Single checklist of every human-validated ball (and related) label pack so we do not lose corrected boxes.  
**Rule:** Canonical labels live in `gold/annotations.xml` only. Never treat `prelabels/` as GT.  
**Last updated:** 2026-08-19

After any editor **Save**, re-run checksums (below) and push `gold/annotations.xml` to GitHub.

## Match 2 — Top Left window (0:26–0:31, synced)

| Pack | Camera | Status | Ball boxes | Frames w/ ball | Review UI | SHA256 (gold XML) |
|---|---|---|---:|---:|---|---|
| `match2_4quad_top_left` | P10 | **Validated** | 265 | 265 | `/4quad-cvat/top_left` | `ac5cfc0fd877a4efc0ed10f5080c930382dd280de946854abf970146280d2dcc` |
| `match2_4quad_top_left_p7` | P7 | **Validated** | 282 | 280 | `/4quad-cvat/top_left_p7` | `a23ad0e36b2c02af1d42b2682d240b0ea2d36c992a7a7b3d7f6f4e04a3f5eb0c` |
| `match2_4quad_top_left_cam4plus` | Cam4plus | **Validated** (2026-08-16) | 314 | 299 | `/4quad-cvat/top_left_cam4plus` | `8e0504e91e9b6256cb198510f4ae320a22b8a736986e028241352667aaca912e` |

Notes:
- Label rule: **match ball in play only**; delete sideline / spare balls.
- Editor often keeps `source="auto"` on accepted prelabels and sets `source="manual"` only on newly drawn boxes. **The whole `gold/annotations.xml` is the validated artifact**, not only `manual` rows.
- Cam4plus pack vs its prelabel draft: ~296 frames unchanged, ~3 edited (2 frames with explicit `manual` tags). Saved 2026-08-16 16:48.

Spec detail: [MATCH2_4QUAD_TOP_LEFT_300.md](MATCH2_4QUAD_TOP_LEFT_300.md).

## Match 2 — other validated / train packs

| Pack | Role | Status | SHA256 (gold XML) |
|---|---|---|---|
| `match2_large_ball_accepted50` | Eval large-ball 50 | **Validated** | `4d8ddea428e9cbbc5e4fd3cc55c224478c69b3ad97ceda245784f9ac6fadbdec` |
| `match2_gold_frames` | Alias / same labels as accepted50 | **Validated** | same as accepted50 |
| `match2_train_label100` | Train | **Validated** | `ae35d4977846bcc9eeabf1dfe1029ab4bbd2488792ec37cce19975dde82186a7` |

## Match 1 / math_1 (must stay on GitHub)

| Pack | Role | Status | SHA256 (gold XML) |
|---|---|---|---|
| `match1_1_100` | Eval Gold100 | **Validated** | `04d8f6102fc39b21340bce41196f5da2079519422a36399e27ea257d57f6ee5a` |
| `math_1_training` | Train | **Validated** | `0cf70efb31f5780af7c0ecf9d1eadc714cd4c871e7138425280299bde51c0baf` |
| `math_1_training_batch3` | Train batch | **Validated** | `6b0b54df1539d81ac2b543a4af4b6366abe7f8cd06299eda035dce65dc36078d` |

## Match 3 — M1 strip (P10 0:31–0:36)

| Pack | Camera | Status | Notes |
|---|---|---|---|
| `match3_quad_p10_31` | P10 | **Human-reviewed** frames 0–194 (`labels.json`) | Product M1 eval. Train mix v11 uses **≤119 stride 2 only**; **120–194 held out**. |

## Match 3 — human blur / soft session (2026-09-04)

| Pack | Status | Notes |
|---|---|---|
| `match3_human_blur_gold` | **Human-only export** | `gold/human_labels.json` — only `human_conf` boxes (**176** as of export). Seed/prelabel excluded. Frames stay local. |
| `match3_quad_p9_655` | Partial human | P9 @ 10:55; 32 human boxes in export |
| `match3_blur_p1_soft1500` | Partial human | soft-harvest P1; 101 human boxes in export |
| `match3_blur_p1_250` | Partial human | soft_t00250 P1; added to human export |

## Continuous → TrackNet sequence pack

Dense Top Left 300 (P10/P7/Cam4plus) + Match3 P10 M1 are exported as consecutive `(prev, mid, next)` triplets for a TrackNet/VballNet side test:

- Builder: `scripts/gold_set/build_tracknet_seq_pack.py`
- Pack: `/Volumes/LaCie/Projects/Soccer project data/ball_tracknet_seq_v1/`
- Doc: [TRAIN_MIX_TRACKNET_SEQ.md](../ball_detection/TRAIN_MIX_TRACKNET_SEQ.md)

Same temporal holdouts as v11 (4quad 240+, Match3 120–194). Does **not** include `match3_quad_p8_87`.

## Not validated (do not treat as GT)

| Pack | Why |
|---|---|
| `match2_4quad_center_start_cam4plus` | Dense / auto prelabels only; sideline-ball labeling deprioritized |
| `match2_4quad_label` | Draft multi-quad pack; mostly auto |
| `math_1_training_batch2` | Auto-only in `gold/` (no manual review signal) |

## Integrity

Checksum file (same hashes): [`gold_annotations.sha256`](gold_annotations.sha256)

```bash
# Verify after pull / before big evals
shasum -a 256 -c docs/product/gold_annotations.sha256
```

## What GitHub must track

For each **Validated** pack above, at minimum:

- `gold/annotations.xml` (canonical)
- `manifest.json` / `README.md` when present
- `prelabels/annotations.xml` optional (draft only; never GT)

Review JPGs stay local (too large). XML is the restore key.

## After you label

1. Save in the viewer (`gold/annotations.xml`).
2. Update this registry + `gold_annotations.sha256` if the SHA changed.
3. `git add` the XML + docs and **push to GitHub**.
