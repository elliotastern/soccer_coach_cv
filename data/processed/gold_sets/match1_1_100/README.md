# Match 1 gold set (100 frames)

Source: `/Volumes/LaCie/Projects/Soccer Coach CV/data/raw/Match 1/Match 1 -1`

## Correct prelabels

```bash
python serve_viewer.py
```

Open: http://localhost:8080/gold100

- Scrub frames 0–99 via **image sequence** (not video seek) so boxes stay aligned.
- Review frames: `review/frames/000.jpg`…; full-res gold JPGs: `images/`.
- Fix boxes; prioritize missed / spurious balls.
- Save writes `data/processed/gold_sets/match1_1_100/prelabels/annotations.xml`.

## Export / eval

```bash
python scripts/gold_set/export_gold_coco.py --gold-dir data/processed/gold_sets/match1_1_100
python scripts/gold_set/eval_on_gold100.py --gold-dir data/processed/gold_sets/match1_1_100
```
