# Match 2 accepted large balls (50)

Saved from harvest keep.json on accept/reject pass.

## Draw / add a second ball (local editor)

```bash
python3 serve_viewer.py --port 8080
# open http://127.0.0.1:8080/accepted50
```

1. Label dropdown → **Ball**
2. Press **N** (mode shows `draw (N)`)
3. Click-drag a box around the second ball
4. **Save**

Rebuild editor if needed: `python3 scripts/gold_set/build_accepted50_editor.py`

## CVAT (optional; needs Docker)

See `docs/product/MATCH2_CVAT_SECOND_BALL.md` — port 8090.

## Next harvest batch

http://127.0.0.1:8080/match2-harvest?pack=/data/processed/gold_sets/match2_large_ball_harvest_batch2
