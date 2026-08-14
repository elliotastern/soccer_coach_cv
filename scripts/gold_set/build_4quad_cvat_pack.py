#!/usr/bin/env python3
"""Build Match 2 4-quad labeling pack (CVAT XML + local editor).

Uses Cam4plus (bestcam top camera on all 4 windows). Frames subsampled from
`reports/eval_match2_v10/4quad_test/source/`. Prelabels: thr0.3 + size + NMS
+ topk=2 + SAHI fallback. Docker CVAT optional; local editor always works.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.perception.ball_prelabel import BallPrelabelConfig, BallPrelabeler  # noqa: E402
from src.perception.rfdetr_local import load_ball_model  # noqa: E402

CKPT = ROOT / "models/v10_snaps/post_train/checkpoint.pth"
SRC_DIR = ROOT / "reports/eval_match2_v10/4quad_test/source"
OUT_DEFAULT = ROOT / "data/processed/gold_sets/match2_4quad_label"
EDITOR_TMPL = (
    ROOT / "data/processed/gold_sets/match2_large_ball_accepted50/review/editor.html"
)
REVIEW_W = 1920
CAMERA = "Cam4plus"
STRIDE_DEFAULT = 12  # ~5 fps @ 60Hz → ~100 frames across 4 windows
TMPL_PACK = "match2_large_ball_accepted50"
TMPL_MAX = 49

CLIPS = [
    # Cameras = emit winners on each 4quad window (not blanket Cam4plus).
    {"slot": "center_start", "label": "Center Start", "stem": "quad_center_start_t00008.0s", "camera": "Cam4plus"},
    {"slot": "bottom_right", "label": "Bottom Right", "stem": "quad_bottom_right_t00412.0s", "camera": "Cam5plus"},
    {"slot": "top_left", "label": "Top Left", "stem": "quad_top_left_t00026.0s", "camera": "P10"},
    {"slot": "top_right", "label": "Top Right", "stem": "quad_top_right_t00125.0s", "camera": "P7"},
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=OUT_DEFAULT)
    p.add_argument("--stride", type=int, default=STRIDE_DEFAULT)
    p.add_argument("--ckpt", type=Path, default=CKPT)
    return p.parse_args()


def source_path(stem: str, camera: str) -> Path:
    path = SRC_DIR / f"{stem}_{camera}.mp4"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def resize_review(frame):
    h, w = frame.shape[:2]
    if w == REVIEW_W:
        return frame, 1.0
    scale = REVIEW_W / float(w)
    out = cv2.resize(frame, (REVIEW_W, int(round(h * scale))), interpolation=cv2.INTER_AREA)
    return out, scale


def make_prelabler(ckpt: Path) -> BallPrelabeler:
    if not ckpt.is_file():
        raise FileNotFoundError(ckpt)
    model = load_ball_model(str(ckpt))
    cfg = BallPrelabelConfig(
        threshold=0.30,
        use_sahi=True,
        sahi_fallback_only=True,
        topk=2,
        use_kalman=False,
        use_size_filter=True,
        min_side=4,
        max_side=240,
    )
    return BallPrelabeler(model, cfg)


def extract_clip(pre, clip: dict, stride: int, start_idx: int, dirs: dict):
    path = source_path(clip["stem"], clip["camera"])
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"open failed: {path}")
    rows = []
    preds = []
    idx = 0
    kept = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % stride != 0:
            idx += 1
            continue
        review, scale = resize_review(frame)
        rh, rw = review.shape[:2]
        name = f"{start_idx + kept:03d}.jpg"
        cv2.imwrite(str(dirs["review"] / name), review, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        shutil.copy2(dirs["review"] / name, dirs["cvat"] / name)
        dets = pre.detect_bgr(review)
        balls = []
        for d in dets:
            if d.class_name != "ball":
                continue
            x, y, w, h = d.bbox
            balls.append({
                "bbox": [float(x), float(y), float(w), float(h)],
                "confidence": float(d.confidence),
            })
        rows.append({
            "image": name,
            "slot": clip["slot"],
            "label": clip["label"],
            "camera": clip["camera"],
            "source": str(path.relative_to(ROOT)),
            "frame_idx": idx,
            "width": rw,
            "height": rh,
        })
        preds.append({"image": name, "balls": balls, "players": []})
        kept += 1
        idx += 1
    cap.release()
    print(f"  {clip['label']}: {kept} frames from {path.name}")
    return rows, preds


def write_track_xml(pack_name: str, rows: list, preds: list) -> str:
    """CVAT video-style tracks — required by local gold editor parseAnnotations()."""
    root = ET.Element("annotations")
    ET.SubElement(root, "version").text = "1.1"
    meta = ET.SubElement(root, "meta")
    task = ET.SubElement(meta, "task")
    ET.SubElement(task, "name").text = pack_name
    ET.SubElement(task, "size").text = str(len(rows))
    ET.SubElement(task, "mode").text = "annotation"
    ET.SubElement(task, "start_frame").text = "0"
    ET.SubElement(task, "stop_frame").text = str(max(0, len(rows) - 1))
    labels = ET.SubElement(task, "labels")
    for name, color in (("ball", "#00ff00"), ("player", "#ff0000")):
        label = ET.SubElement(labels, "label")
        ET.SubElement(label, "name").text = name
        ET.SubElement(label, "color").text = color
    tid = 1
    for frame_i, pred in enumerate(preds):
        for b in pred.get("balls", []):
            x, y, w, h = b["bbox"]
            track = ET.SubElement(root, "track", {
                "id": str(tid),
                "label": "ball",
                "source": "auto",
            })
            box = ET.SubElement(track, "box", {
                "frame": str(frame_i),
                "xtl": f"{x:.2f}",
                "ytl": f"{y:.2f}",
                "xbr": f"{x + w:.2f}",
                "ybr": f"{y + h:.2f}",
                "occluded": "0",
                "outside": "0",
                "keyframe": "1",
            })
            attr = ET.SubElement(box, "attribute", {"name": "confidence"})
            attr.text = f"{float(b['confidence']):.3f}"
            tid += 1
    xml = ET.tostring(root, encoding="unicode")
    return '<?xml version="1.0" encoding="utf-8"?>\n' + xml



def write_cvat_xml(out: Path, rows: list, preds: list):
    root = ET.Element("annotations")
    ET.SubElement(root, "version").text = "1.1"
    meta = ET.SubElement(root, "meta")
    task = ET.SubElement(meta, "task")
    ET.SubElement(task, "name").text = out.name
    ET.SubElement(task, "size").text = str(len(rows))
    labels = ET.SubElement(task, "labels")
    for name, color in (("ball", "#00ff00"), ("player", "#ff0000")):
        label = ET.SubElement(labels, "label")
        ET.SubElement(label, "name").text = name
        ET.SubElement(label, "color").text = color
    for i, (row, pred) in enumerate(zip(rows, preds)):
        img = ET.SubElement(root, "image", {
            "id": str(i),
            "name": row["image"],
            "width": str(row["width"]),
            "height": str(row["height"]),
        })
        for b in pred["balls"]:
            x, y, w, h = b["bbox"]
            box = ET.SubElement(img, "box", {
                "label": "ball",
                "occluded": "0",
                "source": "auto",
                "xtl": f"{x:.2f}",
                "ytl": f"{y:.2f}",
                "xbr": f"{x + w:.2f}",
                "ybr": f"{y + h:.2f}",
            })
            attr = ET.SubElement(box, "attribute", {"name": "confidence"})
            attr.text = f"{float(b['confidence']):.3f}"
    xml = ET.tostring(root, encoding="unicode")
    text = '<?xml version="1.0" encoding="utf-8"?>\n' + xml
    (out / "cvat" / "annotations.xml").write_text(text, encoding="utf-8")
    # Local editor expects <track> boxes (video 1.1), not <image> children.
    track_xml = write_track_xml(out.name, rows, preds)
    (out / "prelabels" / "annotations.xml").write_text(track_xml, encoding="utf-8")
    (out / "gold" / "annotations.xml").write_text(track_xml, encoding="utf-8")


def write_editor(out: Path, names: list):
    """Build editor from the working accepted50 template (has loadReviewFrame)."""
    if not EDITOR_TMPL.is_file():
        raise FileNotFoundError(EDITOR_TMPL)
    pack = out.name
    html = EDITOR_TMPL.read_text(encoding="utf-8")
    if "loadReviewFrame" not in html:
        raise RuntimeError(f"template missing loadReviewFrame: {EDITOR_TMPL}")
    html = html.replace(TMPL_PACK, pack)
    m = re.search(r"const GOLD100 = \{.*?maxFrame:\s*\d+\s*,\s*\};", html, re.S)
    if not m:
        raise RuntimeError("GOLD100 block missing in editor template")
    files_js = ",\n                ".join(json.dumps(n) for n in names)
    last = len(names) - 1
    block = f"""const GOLD100 = {{
            files: [
                {files_js}
            ],
            base: '/data/processed/gold_sets/{pack}/review/frames/',
            width: 1920,
            height: 1080,
            maxFrame: {last},
        }};"""
    html = html[: m.start()] + block + html[m.end() :]
    # accepted50 hardcodes 49 in a few nav spots
    html = html.replace(f'max="{TMPL_MAX}"', f'max="{last}"')
    html = html.replace(f"seekToFrame({TMPL_MAX})", f"seekToFrame({last})")
    html = html.replace(
        f"if (currentFrame < {TMPL_MAX})", f"if (currentFrame < {last})"
    )
    html = html.replace(f" / {TMPL_MAX}", f" / {last}")
    if "loadReviewFrame" not in html:
        raise RuntimeError("editor write dropped loadReviewFrame")
    if f"gold_sets/{pack}/gold/annotations.xml" not in html:
        raise RuntimeError("editor save path not remapped to pack")
    out_path = out / "review" / "editor.html"
    out_path.write_text(html, encoding="utf-8")
    if out_path.stat().st_size < 50000:
        raise RuntimeError(f"editor too small ({out_path.stat().st_size} bytes) — truncated?")



def write_readme(out: Path, n: int, stride: int):
    text = f"""# Match 2 — 4 quad label pack

Cameras: Center Start=Cam4plus · Bottom Right=Cam5plus · Top Left=P10 · Top Right=P7.
Frames: **{n}** (stride {stride} from 4quad source clips).
Prelabel: thr0.3 + size + NMS + topk=2 + SAHI fallback.

## Local editor (no Docker)

http://127.0.0.1:8080/4quad-cvat

Ball → N → draw · Save → `gold/annotations.xml`

## CVAT (needs Docker on port 8090)

```bash
cd annotation && docker compose -f docker-compose.cvat.yml up -d
```

Import `cvat/images/` + `cvat/annotations.xml` (CVAT for images 1.1).
"""
    (out / "README.md").write_text(text, encoding="utf-8")


def main() -> int:
    args = parse_args()
    out = args.out
    dirs = {
        "review": out / "review" / "frames",
        "cvat": out / "cvat" / "images",
    }
    for d in (*dirs.values(), out / "prelabels", out / "gold", out / "videos"):
        d.mkdir(parents=True, exist_ok=True)

    print(f"loading {args.ckpt}")
    pre = make_prelabler(args.ckpt)
    rows = []
    preds = []
    for clip in CLIPS:
        src = source_path(clip["stem"], clip["camera"])
        shutil.copy2(src, out / "videos" / src.name)
        part_rows, part_preds = extract_clip(pre, clip, args.stride, len(rows), dirs)
        rows.extend(part_rows)
        preds.extend(part_preds)

    write_cvat_xml(out, rows, preds)
    names = [r["image"] for r in rows]
    write_editor(out, names)
    (out / "review" / "preds.json").write_text(json.dumps(preds, indent=2), encoding="utf-8")
    (out / "manifest.json").write_text(
        json.dumps(
            {
                "pack": out.name,
                "camera": clip["camera"],
                "stride": args.stride,
                "n_frames": len(rows),
                "clips": CLIPS,
                "checkpoint": str(args.ckpt.relative_to(ROOT)),
                "frames": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_readme(out, len(rows), args.stride)
    print(f"wrote {len(rows)} frames → {out}")
    print(f"editor: http://127.0.0.1:8080/4quad-cvat")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
