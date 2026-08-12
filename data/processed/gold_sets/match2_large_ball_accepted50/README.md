# Match 2 accepted large balls (50)

Saved from harvest keep.json on accept/reject pass.

## Add a second ball (CVAT)

```bash
cd annotation && docker compose -f docker-compose.cvat.yml up -d
# open http://127.0.0.1:8090  (admin / admin)
```

1. Create project → labels: `ball`, `player`
2. Create task → upload all files from `cvat/images/`
3. Actions → Upload annotations → `cvat/annotations.xml` (CVAT for images 1.1)
4. Draw extra `ball` boxes for sideline balls; Export when done

## Local accept/reject for next batch

http://127.0.0.1:8080/match2-harvest?pack=/data/processed/gold_sets/match2_large_ball_harvest_batch2
