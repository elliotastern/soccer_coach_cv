# Match2 Top Left 300 — post-process ranking

Gold: `/Volumes/LaCie/Projects/Soccer Coach CV/data/processed/gold_sets/match2_4quad_top_left/gold/annotations.xml` · 300 frames · 265 GT boxes · IoU≥0.5

Ranked by **F1 @ conf≥0.3** (then recall, then precision). Also report product emit @ conf≥0.8.

| Rank | id | family | F1@0.3 | R@0.3 | P@0.3 | R@0.8 | P_emit@0.8 | tp/fp/fn@0.3 |
|---:|---|---|---:|---:|---:|---:|---:|---|
| 1 | `sahi_dense_tiles` | sahi_hurt | 0.847 | 0.796 | 0.906 | 0.242 | 0.970 | 211/22/54 |
| 2 | `D7_adaptive_asahi` | sahi_next | 0.821 | 0.789 | 0.857 | 0.253 | 1.000 | 209/35/56 |
| 3 | `sahi_always_multiscale` | sahi_hurt | 0.805 | 0.702 | 0.944 | 0.204 | 1.000 | 186/11/79 |
| 4 | `sahi_always_topk3` | sahi_hurt | 0.805 | 0.702 | 0.944 | 0.200 | 1.000 | 186/11/79 |
| 5 | `sahi_always_topk5` | sahi_hurt | 0.805 | 0.702 | 0.944 | 0.200 | 1.000 | 186/11/79 |
| 6 | `sahi_recover_always` | sahi_hurt | 0.805 | 0.702 | 0.944 | 0.200 | 1.000 | 186/11/79 |
| 7 | `sahi_always_nosize` | sahi_hurt | 0.803 | 0.702 | 0.939 | 0.200 | 1.000 | 186/12/79 |
| 8 | `sahi_always_tta` | sahi_hurt | 0.801 | 0.743 | 0.868 | 0.238 | 1.000 | 197/30/68 |
| 9 | `D2_wbf_merge` | sahi_next | 0.797 | 0.691 | 0.943 | 0.200 | 1.000 | 183/11/82 |
| 10 | `D3_diou_merge` | sahi_next | 0.797 | 0.691 | 0.943 | 0.200 | 1.000 | 183/11/82 |
| 11 | `sahi_fallback` | postproc | 0.795 | 0.687 | 0.943 | 0.200 | 1.000 | 182/11/83 |
| 12 | `D4_temporal_ema` | sahi_next | 0.780 | 0.664 | 0.946 | 0.200 | 1.000 | 176/10/89 |
| 13 | `D9_dotd_tracker` | sahi_next | 0.779 | 0.664 | 0.941 | 0.200 | 1.000 | 176/11/89 |
| 14 | `sahi_always_thr20` | sahi_hurt | 0.779 | 0.664 | 0.941 | 0.200 | 1.000 | 176/11/89 |
| 15 | `D1_sparse_logit` | sahi_next | 0.778 | 0.660 | 0.946 | 0.200 | 1.000 | 175/10/90 |
| 16 | `D6_player_context` | sahi_next | 0.777 | 0.672 | 0.922 | 0.200 | 1.000 | 178/15/87 |
| 17 | `D5_soft_edge` | sahi_next | 0.771 | 0.653 | 0.940 | 0.189 | 1.000 | 173/11/92 |
| 18 | `D10_entropy_crop` | sahi_next | 0.750 | 0.623 | 0.943 | 0.200 | 1.000 | 165/10/100 |
| 19 | `D8_sr_conditional` | sahi_next | 0.748 | 0.623 | 0.938 | 0.200 | 1.000 | 165/11/100 |
| 20 | `sahi_always_bt_sticky` | sahi_hurt | 0.694 | 0.543 | 0.960 | 0.192 | 1.000 | 144/6/121 |
| 21 | `hflip_tta` | postproc | 0.686 | 0.574 | 0.854 | 0.215 | 1.000 | 152/26/113 |
| 22 | `baseline_topk2` | postproc | 0.681 | 0.536 | 0.934 | 0.185 | 1.000 | 142/10/123 |
| 23 | `multiscale_1p5` | postproc | 0.681 | 0.536 | 0.934 | 0.189 | 1.000 | 142/10/123 |
| 24 | `topk3` | postproc | 0.681 | 0.536 | 0.934 | 0.185 | 1.000 | 142/10/123 |
| 25 | `thr50_topk2` | postproc | 0.604 | 0.438 | 0.975 | 0.185 | 1.000 | 116/3/149 |
| 26 | `bytetrack_iou08` | postproc | 0.571 | 0.408 | 0.956 | 0.177 | 1.000 | 108/5/157 |
| 27 | `bytetrack_emit80` | postproc | 0.349 | 0.211 | 1.000 | 0.211 | 1.000 | 56/0/209 |
| 28 | `kalman_detect` | postproc | 0.316 | 0.279 | 0.363 | 0.143 | 0.776 | 74/130/191 |
| 29 | `emit80_pass` | postproc | 0.312 | 0.185 | 1.000 | 0.185 | 1.000 | 49/0/216 |
| 30 | `sahi_always_kalman` | sahi_hurt | 0.267 | 0.253 | 0.283 | 0.151 | 0.755 | 67/170/198 |

## Top 5

1. **sahi_dense_tiles** — 10. Dense SAHI tiles (640 / 40% overlap) + topk=3  
   More/smaller tiles → more false peaks; keep three of them.  
   F1@0.3=0.847 R=0.796 P=0.906
2. **D7_adaptive_asahi** — 7. D7 Adaptive ASAHI grid  
   Far pitch (top): 640px tiles; near (bottom): 1024px — then WBF.  
   F1@0.3=0.821 R=0.789 P=0.857
3. **sahi_always_multiscale** — 6. SAHI always + multiscale 1.5×  
   Two recover paths at once — train already showed multiscale +FPs.  
   F1@0.3=0.805 R=0.702 P=0.944
4. **sahi_always_topk3** — 2. SAHI always + topk=3  
   Always tiles then keep 3 boxes — more room for tile junk.  
   F1@0.3=0.805 R=0.702 P=0.944
5. **sahi_always_topk5** — 3. SAHI always + topk=5  
   Even looser topk — second/third FPs more likely to survive.  
   F1@0.3=0.805 R=0.702 P=0.944
