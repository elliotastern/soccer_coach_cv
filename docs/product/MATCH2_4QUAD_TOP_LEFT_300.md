# Match2 Top Left 300-frame gold labels

**Canonical name:** Match2 Top Left 300-frame gold labels  
**Alias / pack id:** `match2_4quad_top_left`

Human-corrected ball boxes for one continuous Match 2 window from the 4-quad ball test (Top Left only). When someone says “the 300 Top Left labels” / “Match2 Top Left 300,” this is the pack.

| | |
|---|---|
| **Pack path** | `data/processed/gold_sets/match2_4quad_top_left/` |
| **Canonical labels** | `gold/annotations.xml` (git-tracked; `prelabels/` is detector draft only) |
| **Camera / clock** | **P10**, Match 2 clock **0:26–0:31** |
| **Size** | **300** frames, stride 1, **60 fps** (full 5 s clip — not the old stride-12 25-frame pack) |
| **Source clip** | `reports/eval_match2_v10/4quad_test/source/quad_top_left_t00026.0s_P10.mp4` |
| **Review UI** | http://127.0.0.1:8080/4quad-cvat/top_left |
| **Build** | `scripts/gold_set/build_4quad_cvat_pack.py` |

## What it is for

- Dense temporal gold on a hard Top Left / P10 ball sequence (4-quad test slot).
- Compare detector / SAHI-fallback prelabels to human boxes on every frame of that window.

It is **not** Match Gold100 (`match1_1_100`), **not** `match2_large_ball_accepted50`, and **not** `match2_train_label100`. Do not train from `prelabels/`; see [TRAIN_LABEL_SOURCE_OF_TRUTH.md](../ball_detection/TRAIN_LABEL_SOURCE_OF_TRUTH.md).

## Pack layout

```text
data/processed/gold_sets/match2_4quad_top_left/
├── gold/annotations.xml      # human labels (source of truth)
├── prelabels/annotations.xml # model draft for the editor
├── review/frames/000.jpg…    # local only (not in git)
├── manifest.json
└── README.md
```

Frames stay local (~153 MB JPGs). XML + manifest are what GitHub carries.

## Post-process ranking (v10 checkpoint)

All 30 stacks from the postproc / SAHI-hurt / SAHI-next galleries were rescored on this gold (IoU≥0.5, rank by F1 @ conf≥0.3):

`reports/eval_match2_v10/top_left_300_postproc_rank/ranking.md`

**Winner on this clip:** `sahi_dense_tiles` (640px / 40% overlap + topk=3) — F1 **0.847**, R **0.796**, P **0.906**.  
Runner-up: `D7_adaptive_asahi` (F1 0.821). Plain `sahi_fallback` is #11 (F1 0.795); baseline topk=2 is #22 (F1 0.681). Kalman / emit-0.80 stacks rank near the bottom for recall.

## 6 P-cam system baseline + soft consensus

Target: system ball **R ≥ 0.80**, **P ≥ 0.90** across Match 2 **P1, P6, P7, P8, P10, P12** (≤125 ms ball path on RTX 5090). True epipolar needs calib (none on disk yet).

Script: `scripts/gold_set/eval_match2_top_left_multicam_baseline.py`  
Reports:
- [`reports/eval_match2_v10/top_left_multicam_baseline/`](../../reports/eval_match2_v10/top_left_multicam_baseline/) — max_conf @0.30 / emit @0.80
- [`reports/eval_match2_v10/top_left_multicam_consensus/`](../../reports/eval_match2_v10/top_left_multicam_consensus/) — thr 0.15 + ≥2-cam co-occurrence

Proxy R/P uses **P10-selected frames only** (this gold is P10-only). On the first run, proxy hit ~**P/R 0.95** when P10 wins, but P10 is selected only ~**38%** of frames (**P7 ~51%** wins max_conf). Soft consensus (thr 0.15, ≥2 cams) did not lift proxy P/R. Next: 5090 latency; **P7 Top Left gold** (below) for honest system R/P when P7 is selected; calib before epipolar.

## P7 companion pack (labeling)

| | |
|---|---|
| **Pack path** | `data/processed/gold_sets/match2_4quad_top_left_p7/` |
| **Camera / clock** | **P7**, same Match 2 clock **0:26–0:31** (synced with P10 pack) |
| **Size** | **300** frames, stride 1 |
| **Source clip** | `reports/eval_match2_v10/4quad_test/source/quad_top_left_t00026.0s_P7.mp4` |
| **Review UI** | http://127.0.0.1:8080/4quad-cvat/top_left_p7 |
| **Prelabel** | Offline `sahi_dense_tiles` (640 / 0.4 overlap, topk=3) — accuracy for labeling, **not** the live path |
| **Build** | `python3 scripts/gold_set/build_4quad_cvat_pack.py --standalone --camera P7 --stem quad_top_left_t00026.0s --slot top_left --label "Top Left P7" --prelabel dense --stride 1 --out data/processed/gold_sets/match2_4quad_top_left_p7` |

Git tracks XML + manifest only (frames local). **Human-corrected** (`gold/annotations.xml` ≠ prelabels; ~282 GT boxes / 280 frames). Next: rescore multicam baseline with P7 GT on P7-selected frames (plus existing P10 proxy).


