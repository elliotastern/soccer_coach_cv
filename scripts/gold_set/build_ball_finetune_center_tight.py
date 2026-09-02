#!/usr/bin/env python3
"""Build RF-DETR center-tight ball boxes for finetune (YOLOopt-style, RF-DETR stack)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "data/processed/gold_sets/ball_center_tight_manifest.json"
SIDE_PX = 24
V12_DATA = Path("/Volumes/LaCie/Projects/Soccer project data/ball_finetune_v12_hard")


def center_box(cx: float, cy: float, side: float = SIDE_PX) -> list[float]:
    h = side / 2.0
    return [cx - h, cy - h, side, side]


def items_from_coco(coco_path: Path, split: str) -> list[dict]:
    if not coco_path.is_file():
        return []
    coco = json.loads(coco_path.read_text(encoding="utf-8"))
    images = {int(im["id"]): im for im in coco.get("images") or []}
    rows = []
    for ann in coco.get("annotations") or []:
        if int(ann.get("category_id", -1)) != 0:
            continue
        im = images.get(int(ann["image_id"]))
        if im is None:
            continue
        x, y, w, h = [float(v) for v in ann["bbox"]]
        cx, cy = x + w / 2.0, y + h / 2.0
        file_name = str(im.get("file_name") or "")
        rows.append(
            {
                "split": split,
                "file_name": file_name,
                "image_id": int(ann["image_id"]),
                "width": int(im.get("width") or 0),
                "height": int(im.get("height") or 0),
                "bbox": [x, y, w, h],
                "bbox_center": [cx, cy],
                "bbox_tight": center_box(cx, cy),
            }
        )
    return rows


def v12_coco_roots() -> list[tuple[Path, str]]:
    roots = [V12_DATA]
    local = ROOT / "data/processed/gold_sets/ball_finetune_v12_hard"
    if local.is_dir():
        roots.append(local)
    out = []
    for root in roots:
        for split in ("train", "valid", "test"):
            p = root / split / "_annotations.coco.json"
            if p.is_file():
                out.append((p, split))
    return out


def main() -> int:
    rows: list[dict] = []
    sources: list[str] = []
    for coco_path, split in v12_coco_roots():
        key = str(coco_path)
        if key in sources:
            continue
        sources.append(key)
        rows.extend(items_from_coco(coco_path, split))
    payload = {
        "recipe": "rfdetr_center_tight",
        "source_coco": sources,
        "side_px": SIDE_PX,
        "n": len(rows),
        "items": rows,
        "note": "Use bbox_tight for RF-DETR finetune; compare vs v12_hard on quad funnel + fuse3d agree",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {OUT} n={len(rows)}")
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
