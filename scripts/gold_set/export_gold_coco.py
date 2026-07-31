#!/usr/bin/env python3
"""Export corrected CVAT XML for the gold100 pack to COCO JSON."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from annotation.utils.cvat_to_coco import (
    cvat_bbox_to_coco,
    extract_frame_annotations,
    label_to_category_id,
    parse_cvat_xml,
)

DEFAULT_GOLD_DIR = ROOT / "data/processed/gold_sets/match1_1_100"


def parse_args():
    parser = argparse.ArgumentParser(description="Export gold100 CVAT XML → COCO")
    parser.add_argument("--gold-dir", type=Path, default=DEFAULT_GOLD_DIR)
    parser.add_argument(
        "--xml",
        type=Path,
        default=None,
        help="Corrected XML (default: <gold-dir>/prelabels/annotations.xml)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output COCO path (default: <gold-dir>/gold/annotations.coco.json)",
    )
    return parser.parse_args()


def load_manifest(gold_dir: Path) -> dict:
    path = gold_dir / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"manifest.json not found: {path}")
    return json.loads(path.read_text())


def image_size(gold_dir: Path, file_name: str) -> tuple:
    import cv2

    path = gold_dir / "images" / file_name
    if not path.is_file():
        raise FileNotFoundError(f"Image missing: {path}")
    img = cv2.imread(str(path))
    if img is None:
        raise RuntimeError(f"Could not read image: {path}")
    height, width = img.shape[:2]
    return width, height


def build_coco(gold_dir: Path, xml_path: Path) -> dict:
    manifest = load_manifest(gold_dir)
    tree = parse_cvat_xml(str(xml_path))
    categories = [
        {"id": 1, "name": "player", "supercategory": "person"},
        {"id": 2, "name": "ball", "supercategory": "sports"},
    ]
    images = []
    annotations = []
    ann_id = 1
    for row in manifest["frames"]:
        strip_frame = int(row["strip_frame"])
        file_name = row["image"]
        width, height = image_size(gold_dir, file_name)
        image_id = strip_frame + 1
        images.append({
            "id": image_id,
            "file_name": file_name,
            "width": width,
            "height": height,
            "frame_idx": row["frame_idx"],
            "stratum": row["stratum"],
        })
        for ann in extract_frame_annotations(tree, strip_frame):
            xtl, ytl, xbr, ybr = ann["bbox"]
            x, y, w, h = cvat_bbox_to_coco(xtl, ytl, xbr, ybr)
            if w <= 0 or h <= 0:
                continue
            annotations.append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": label_to_category_id(ann["label"]),
                "bbox": [x, y, w, h],
                "area": w * h,
                "iscrowd": 0,
            })
            ann_id += 1
    return {
        "info": {
            "description": "Match 117093 half-2 gold100 (corrected)",
            "version": "1.0",
        },
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }


def main():
    args = parse_args()
    gold_dir = args.gold_dir
    xml_path = args.xml or (gold_dir / "prelabels" / "annotations.xml")
    out_path = args.output or (gold_dir / "gold" / "annotations.coco.json")
    if not xml_path.is_file():
        raise FileNotFoundError(f"XML not found: {xml_path}")
    coco = build_coco(gold_dir, xml_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(coco, indent=2))
    n_ball = sum(1 for a in coco["annotations"] if a["category_id"] == 2)
    n_player = sum(1 for a in coco["annotations"] if a["category_id"] == 1)
    print(f"Wrote {out_path}")
    print(
        f"images={len(coco['images'])} anns={len(coco['annotations'])} "
        f"player={n_player} ball={n_ball}"
    )


if __name__ == "__main__":
    main()
