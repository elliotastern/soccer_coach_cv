# Match 2 — match2_4quad_top_left_p7

Cameras: Top Left P7=P7.
Frames: **300** (stride 1 from 4quad source clips).
Prelabel: thr0.3 + size + NMS + topk=3 + dense SAHI tiles (offline).

## Local editor (no Docker)

http://127.0.0.1:8080/4quad-cvat/top_left_p7

Ball → N → draw · Save → `gold/annotations.xml`

## CVAT (needs Docker on port 8090)

```bash
cd annotation && docker compose -f docker-compose.cvat.yml up -d
```

Import `cvat/images/` + `cvat/annotations.xml` (CVAT for images 1.1).

**Status:** Human-corrected gold — `gold/annotations.xml` is source of truth (~282 ball boxes / 280 frames; prelabels stay the dense-tile draft).
