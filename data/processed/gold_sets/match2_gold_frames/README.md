# Match 2 gold frames

50 Match 2 frames: harvest accepts + manual ball labels (including second balls).

| | |
|--|--|
| Frames | 50 |
| Ball boxes | 62 |
| Multi-ball frames | 12 |
| Source accepts | `match2_large_ball_harvest` keep.json |
| Labels | `match2_large_ball_accepted50/gold/annotations.xml` |

## Open

```bash
python3 serve_viewer.py --port 8080
# http://127.0.0.1:8080/match2-gold
```

Ball → **N** → draw · **Save** writes `gold/annotations.xml`.
