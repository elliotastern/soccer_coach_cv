#!/usr/bin/env python3
"""Match 2 train/test split from gold XML (never prelabels).

train/valid  = match2_train_label100 gold labels
test         = match2_gold_frames (held-out eval; do not train)
"""
from __future__ import annotations

import json
import shutil
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.gold_set.export_gold_coco import build_coco, resolve_xml

TRAIN_PACK = ROOT / "data/processed/gold_sets/match2_train_label100"
TEST_PACK = ROOT / "data/processed/gold_sets/match2_gold_frames"
OUT = ROOT / "data/processed/gold_sets/match2_train_test"
VALID_N = 10


def key_of(row: dict) -> tuple[str, int]:
    return (row["camera"], int(row["frame_idx"]))


def ball_anns_by_image(coco: dict) -> dict[int, list]:
    out: dict[int, list] = {}
    for ann in coco["annotations"]:
        if ann["category_id"] != 2:
            continue
        out.setdefault(ann["image_id"], []).append(ann)
    return out


def ball_only_coco(images: list, anns: list, description: str) -> dict:
    remapped = []
    for i, ann in enumerate(anns, start=1):
        remapped.append({**ann, "id": i, "category_id": 1})
    return {
        "info": {"description": description, "version": "1.0"},
        "licenses": [],
        "images": images,
        "annotations": remapped,
        "categories": [{"id": 1, "name": "ball", "supercategory": "sports"}],
    }


def write_split_dir(split_dir: Path, pack: Path, images: list, anns: list, desc: str):
    if split_dir.exists():
        shutil.rmtree(split_dir)
    split_dir.mkdir(parents=True)
    for img in images:
        src = pack / "images" / img["file_name"]
        if not src.is_file():
            raise FileNotFoundError(src)
        shutil.copy2(src, split_dir / img["file_name"])
    coco = ball_only_coco(images, anns, desc)
    (split_dir / "_annotations.coco.json").write_text(json.dumps(coco, indent=2))
    return coco


def export_pack_gold(pack: Path) -> dict:
    xml = resolve_xml(pack, None)
    gold_xml = pack / "gold" / "annotations.xml"
    if xml.resolve() != gold_xml.resolve():
        raise RuntimeError(f"refusing to train from non-gold XML: {xml}")
    coco = build_coco(pack, xml)
    out = pack / "gold" / "annotations.coco.json"
    out.write_text(json.dumps(coco, indent=2))
    n_ball = sum(1 for a in coco["annotations"] if a["category_id"] == 2)
    print(f"exported {out} images={len(coco['images'])} ball={n_ball}")
    return coco


def main() -> int:
    train_coco = export_pack_gold(TRAIN_PACK)
    test_coco = export_pack_gold(TEST_PACK)
    train_man = json.loads((TRAIN_PACK / "manifest.json").read_text())
    test_man = json.loads((TEST_PACK / "manifest.json").read_text())
    test_keys = {key_of(r) for r in test_man["frames"]}
    overlap = [key_of(r) for r in train_man["frames"] if key_of(r) in test_keys]
    if overlap:
        raise RuntimeError(f"train/test overlap {len(overlap)}: {overlap[:5]}")

    train_by_id = ball_anns_by_image(train_coco)
    labeled = []
    skipped = []
    for img in train_coco["images"]:
        anns = train_by_id.get(img["id"], [])
        row = next(r for r in train_man["frames"] if int(r["strip_frame"]) + 1 == img["id"])
        if not anns:
            skipped.append({"camera": row["camera"], "frame_idx": row["frame_idx"], "strip_frame": row["strip_frame"]})
            continue
        labeled.append((img, anns, row))

    labeled.sort(key=lambda t: (t[2]["camera"], t[2]["t_sec"]))
    if len(labeled) <= VALID_N:
        raise RuntimeError(f"not enough labeled train frames: {len(labeled)}")
    valid_rows = labeled[-VALID_N:]
    train_rows = labeled[:-VALID_N]

    if OUT.exists():
        shutil.rmtree(OUT)
    train_coco_out = write_split_dir(
        OUT / "train",
        TRAIN_PACK,
        [t[0] for t in train_rows],
        [a for t in train_rows for a in t[1]],
        "Match2 train (train_label100 gold, ball-only)",
    )
    valid_coco_out = write_split_dir(
        OUT / "valid",
        TRAIN_PACK,
        [t[0] for t in valid_rows],
        [a for t in valid_rows for a in t[1]],
        "Match2 valid (held from train_label100, not gold eval)",
    )
    test_by_id = ball_anns_by_image(test_coco)
    test_images = [img for img in test_coco["images"] if test_by_id.get(img["id"])]
    test_anns = [a for img in test_images for a in test_by_id[img["id"]]]
    test_coco_out = write_split_dir(
        OUT / "test",
        TEST_PACK,
        test_images,
        test_anns,
        "Match2 test = gold frames (eval only)",
    )

    split = {
        "name": "match2_train_test",
        "source_of_truth": "gold/annotations.xml",
        "train_pack": str(TRAIN_PACK.relative_to(ROOT)),
        "test_pack": str(TEST_PACK.relative_to(ROOT)),
        "n_train": len(train_rows),
        "n_valid": len(valid_rows),
        "n_test": len(test_images),
        "n_train_skipped_empty": len(skipped),
        "skipped_empty": skipped,
        "train_cams": dict(Counter(t[2]["camera"] for t in train_rows)),
        "valid_cams": dict(Counter(t[2]["camera"] for t in valid_rows)),
        "test_cams": dict(Counter(r["camera"] for r in test_man["frames"])),
        "n_ball_train": len(train_coco_out["annotations"]),
        "n_ball_valid": len(valid_coco_out["annotations"]),
        "n_ball_test": len(test_coco_out["annotations"]),
        "overlap_train_test": 0,
        "paths": {
            "train": "data/processed/gold_sets/match2_train_test/train",
            "valid": "data/processed/gold_sets/match2_train_test/valid",
            "test": "data/processed/gold_sets/match2_train_test/test",
        },
        "rule": "Do not train on match2_gold_frames. Test is held-out eval.",
    }
    (OUT / "split.json").write_text(json.dumps(split, indent=2))
    (OUT / "README.md").write_text(
        f"""# Match 2 train / test

Source of truth: `gold/annotations.xml` (not `prelabels/`).

| Split | Pack | Frames | Ball boxes |
|-------|------|--------|------------|
| train | `match2_train_label100` | {len(train_rows)} | {split['n_ball_train']} |
| valid | last {VALID_N} of labeled train100 | {len(valid_rows)} | {split['n_ball_valid']} |
| test | `match2_gold_frames` (held out) | {len(test_images)} | {split['n_ball_test']} |

Skipped unlabeled train frames: {len(skipped)}.

RF-DETR layout:

```
data/processed/gold_sets/match2_train_test/{{train,valid,test}}/_annotations.coco.json
```
"""
    )
    print(json.dumps({k: split[k] for k in (
        "n_train", "n_valid", "n_test", "n_ball_train", "n_ball_valid", "n_ball_test",
        "n_train_skipped_empty", "train_cams", "test_cams",
    )}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
