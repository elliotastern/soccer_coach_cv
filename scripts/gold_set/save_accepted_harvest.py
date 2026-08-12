#!/usr/bin/env python3
"""Save accepted harvest frames into a durable accepted pack + CVAT folder."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from annotation.cvat_xml_generator import create_cvat_xml
from scripts.gold_set.build_match_gold100 import to_ann_tracked, write_coco
from src.state.types import Detection


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--src",
        type=Path,
        default=ROOT / "data/processed/gold_sets/match2_large_ball_harvest",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=ROOT / "data/processed/gold_sets/match2_large_ball_accepted50",
    )
    p.add_argument(
        "--keep",
        type=Path,
        default=None,
        help="keep.json path (default: <src>/keep.json)",
    )
    return p.parse_args()


def load_keep(path: Path) -> list[int]:
    data = json.loads(path.read_text())
    accepted = sorted(set(int(i) for i in data.get("accepted", [])))
    if not accepted:
        raise RuntimeError(f"no accepted frames in {path}")
    return accepted


def box_from_pred(pred_ball: dict) -> Detection:
    x, y, w, h = pred_ball["bbox"]
    return Detection(
        class_id=1,
        confidence=float(pred_ball["confidence"]),
        bbox=(float(x), float(y), float(w), float(h)),
        class_name="ball",
    )


def write_cvat_images_xml(out_dir: Path, rows: list, preds: list):
    """CVAT for images 1.1 — one <image> per frame with box children."""
    root = ET.Element("annotations")
    ET.SubElement(root, "version").text = "1.1"
    meta = ET.SubElement(root, "meta")
    task = ET.SubElement(meta, "task")
    ET.SubElement(task, "name").text = out_dir.name
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
            "width": str(1920),
            "height": str(1080),
        })
        for b in pred.get("balls", []):
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
    (out_dir / "cvat" / "annotations.xml").write_text(
        '<?xml version="1.0" encoding="utf-8"?>\n' + xml, encoding="utf-8"
    )


def main():
    args = parse_args()
    src = args.src
    keep_path = args.keep or (src / "keep.json")
    accepted = load_keep(keep_path)
    manifest = json.loads((src / "manifest.json").read_text())
    preds_all = json.loads((src / "review" / "preds.json").read_text())
    src_frames = {int(r["strip_frame"]): r for r in manifest["frames"]}

    out = args.out
    for d in (
        out / "images",
        out / "review" / "frames",
        out / "prelabels",
        out / "cvat" / "images",
        out / "gold",
    ):
        d.mkdir(parents=True, exist_ok=True)

    rows = []
    preds = []
    tracked_by_frame = {}
    selected_for_coco = []
    next_track = 1
    for new_i, old_i in enumerate(accepted):
        if old_i not in src_frames:
            raise RuntimeError(f"accepted index {old_i} missing from manifest")
        row = dict(src_frames[old_i])
        old_name = row["image"]
        src_img = src / "images" / old_name
        src_rev = src / "review" / "frames" / f"{old_i:03d}.jpg"
        if not src_img.is_file():
            raise FileNotFoundError(src_img)
        new_name = f"{new_i:03d}_{row['camera']}_f{row['frame_idx']:06d}.jpg"
        shutil.copy2(src_img, out / "images" / new_name)
        shutil.copy2(src_rev, out / "review" / "frames" / f"{new_i:03d}.jpg")
        shutil.copy2(src_img, out / "cvat" / "images" / new_name)

        pred = preds_all[old_i]
        balls = [box_from_pred(b) for b in pred.get("balls", [])]
        tracked = to_ann_tracked(balls, next_track)
        next_track += max(len(tracked), 1)
        tracked_by_frame[new_i] = tracked
        row_out = {
            **row,
            "strip_frame": new_i,
            "image": new_name,
            "source_strip_frame": old_i,
        }
        rows.append(row_out)
        preds.append({"frame": new_i, "balls": pred.get("balls", []), "source_strip_frame": old_i})
        selected_for_coco.append({
            "width": row["width"],
            "height": row["height"],
            "camera": row["camera"],
            "video_rel": row["video_rel"],
            "frame_idx": row["frame_idx"],
            "stratum": "accepted_large_ball",
            "detections": [
                Detection(
                    class_id=1,
                    confidence=float(b["confidence"]),
                    bbox=(
                        float(b["bbox"][0]) * row["width"] / 1920.0,
                        float(b["bbox"][1]) * row["height"] / 1080.0,
                        float(b["bbox"][2]) * row["width"] / 1920.0,
                        float(b["bbox"][3]) * row["height"] / 1080.0,
                    ),
                    class_name="ball",
                )
                for b in pred.get("balls", [])
            ],
        })

    write_coco(
        selected_for_coco,
        [r["image"] for r in rows],
        out / "prelabels" / "annotations.coco.json",
    )
    xml = create_cvat_xml(
        video_path=str(out / "review" / "strip_100.mp4"),
        tracked_objects_by_frame=tracked_by_frame,
        events=[],
        video_metadata={
            "width": 1920,
            "height": 1080,
            "fps": 10.0,
            "frame_count": len(rows),
        },
    )
    (out / "prelabels" / "annotations.xml").write_text(xml, encoding="utf-8")
    shutil.copy2(out / "prelabels" / "annotations.xml", out / "gold" / "annotations.xml")
    write_cvat_images_xml(out, rows, preds)

    keep_snap = json.loads(keep_path.read_text())
    keep_snap["saved_as"] = str(out)
    keep_snap["n_accepted_saved"] = len(accepted)
    (out / "keep_source.json").write_text(json.dumps(keep_snap, indent=2))
    (src / "keep.json").write_text(json.dumps(keep_snap, indent=2))

    out_manifest = {
        "name": "match2_large_ball_accepted50",
        "n_frames": len(rows),
        "source_pack": str(src),
        "accepted_source_indices": accepted,
        "frames": rows,
        "cvat": {
            "images_dir": "cvat/images",
            "annotations": "cvat/annotations.xml",
            "url": "http://127.0.0.1:8090",
            "note": "Import images then annotations (CVAT for images 1.1). Add extra ball boxes there.",
        },
    }
    (out / "manifest.json").write_text(json.dumps(out_manifest, indent=2))
    (out / "review" / "preds.json").write_text(json.dumps(preds, indent=2))
    (out / "README.md").write_text(
        f"""# Match 2 accepted large balls ({len(rows)})

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
"""
    )
    print(f"Saved {len(rows)} accepted frames → {out}")
    print(f"CVAT images: {out / 'cvat' / 'images'}")
    print(f"CVAT XML:    {out / 'cvat' / 'annotations.xml'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
