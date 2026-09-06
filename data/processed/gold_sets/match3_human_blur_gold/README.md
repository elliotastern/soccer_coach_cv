# match3_human_blur_gold

Human-only (`human_conf`) Match 3 ball boxes for soft/blur work.

- **n=404** · streaky=53 · blurry=161 · clear=196
- by pack: `{'match3_quad_p9_655': 32, 'match3_blur_p1_soft1500': 101, 'match3_blur_p1_250': 96, 'match3_blur_p1_1100': 43, 'match3_blur_p1_1300': 132}`

Canonical JSON: `gold/human_labels.json` (in git).
`frames/` copies stay local (gitignored).

See `docs/product/MATCH3_HUMAN_BLUR_GOLD.md`.
Refresh: `PYTHONPATH=. python3 scripts/gold_set/export_match3_human_blur_gold.py`
