# Match 2 — add a second ball in CVAT

Accepted pack: `data/processed/gold_sets/match2_large_ball_accepted50/`

CVAT runs on **8090** so it does not collide with `serve_viewer` on 8080.

```bash
cd annotation
docker compose -f docker-compose.cvat.yml up -d
# open http://127.0.0.1:8090  (default admin / admin on first setup)
```

## Import the accepted 50

1. Create a project with labels: `ball`, `player`.
2. Create a task → **Connected file share** (or upload) from  
   `match2_accepted50/images` (mounted at `/home/django/share/match2_accepted50/images`)  
   or upload files from `data/processed/gold_sets/match2_large_ball_accepted50/cvat/images/`.
3. **Actions → Upload annotations** → `cvat/annotations.xml` (format: **CVAT for images 1.1**).
4. For frames with a sideline / second ball: draw another `ball` box (keep the first).
5. Export when done (CVAT for images 1.1) back into the pack if you want it versioned.

Do **not** reject a frame in the harvest editor only because a second ball lacks a box — Accept the clear on-pitch ball, then add the extra box here.
