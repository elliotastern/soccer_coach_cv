# Match 2 — match2_4quad_center_start_cam4plus

Cameras: Center Start Cam4plus=Cam4plus.
Frames: **300** (stride 1 from 4quad source clips).
Prelabel: thr0.3 + size + NMS + topk=3 + dense SAHI tiles (offline).

## Local editor (no Docker)

http://127.0.0.1:8080/4quad-cvat/center_start_cam4plus

Ball → N → draw · Save → `gold/annotations.xml`

## CVAT (needs Docker on port 8090)

```bash
cd annotation && docker compose -f docker-compose.cvat.yml up -d
```

Import `cvat/images/` + `cvat/annotations.xml` (CVAT for images 1.1).

**Status:** Dense prelabel draft — correct in editor and Save before using as GT.
