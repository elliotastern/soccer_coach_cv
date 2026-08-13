#!/usr/bin/env python3
"""v9 mix: Match1 gold + Match2 train100, Gold100+Match2 gold as test.

Never trains on match2_gold_frames or Gold100 strip 0-49.
"""
from __future__ import annotations

import importlib.util
import json
import random
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MIX_PY = Path("/Volumes/LaCie/Projects/Soccer project data/scripts/build_ball_finetune_match_mix.py")
OUT = Path("/Volumes/LaCie/Projects/Soccer project data/ball_finetune_match2_v9")
MATCH2_TRAIN = ROOT / "data/processed/gold_sets/match2_train_label100"
MATCH2_GOLD = ROOT / "data/processed/gold_sets/match2_gold_frames"
SPLIT = ROOT / "data/processed/gold_sets/match2_train_test/split.json"
SEED = 42
N_OFFICIAL = 100
N_KJOYY = 200
N_VALID_OFFICIAL = 30
N_VALID_KJOYY = 50
MATCH_AUG_FACTOR = 3


def load_mix():
    if not MIX_PY.is_file():
        raise FileNotFoundError(MIX_PY)
    spec = importlib.util.spec_from_file_location("mix", MIX_PY)
    mix = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mix)
    return mix


def names_in_coco(path: Path) -> set[str]:
    coco = json.loads(path.read_text())
    return {im["file_name"] for im in coco["images"]}


def keys_from_pack(pack: Path) -> set[tuple]:
    man = json.loads((pack / "manifest.json").read_text())
    return {(r["camera"], int(r["frame_idx"])) for r in man["frames"]}


def match2_test_items(mix) -> list[dict]:
    return mix.match_train_ball_items_from(MATCH2_GOLD, "match2_gold_test")


def gold100_test_items(mix) -> list[dict]:
    """Gold100 strip 0-49 from gold/ only. Prelabels may still exist; ignore them."""
    gold = ROOT / "data/processed/gold_sets/match1_1_100"
    coco_path = gold / "gold" / "annotations.coco.json"
    if not coco_path.is_file():
        raise FileNotFoundError(coco_path)
    coco = json.loads(coco_path.read_text())
    cats = {c["id"]: c["name"] for c in coco["categories"]}
    id2im = {im["id"]: im for im in coco["images"]}
    by_img = {}
    for a in coco["annotations"]:
        if cats.get(a["category_id"]) != "ball":
            continue
        by_img.setdefault(a["image_id"], []).append(a["bbox"])
    items = []
    for image_id, boxes in sorted(by_img.items()):
        im = id2im[image_id]
        strip_frame = int(im["id"]) - 1
        if strip_frame < 0 or strip_frame > 49:
            continue
        src = gold / "images" / im["file_name"]
        if not src.is_file():
            raise FileNotFoundError(src)
        items.append({
            "src": src,
            "width": im["width"],
            "height": im["height"],
            "boxes": boxes,
            "source": "gold100_test_0_49",
            "file_name": im["file_name"],
            "strip_frame": strip_frame,
        })
    if not items:
        raise RuntimeError("no Gold100 ball frames in strip 0-49")
    return items


def main() -> int:
    mix = load_mix()
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    mix.OUT = OUT
    rng = random.Random(SEED)

    train_names = names_in_coco(ROOT / "data/processed/gold_sets/match2_train_test/train/_annotations.coco.json")
    valid_names = names_in_coco(ROOT / "data/processed/gold_sets/match2_train_test/valid/_annotations.coco.json")
    gold_keys = keys_from_pack(MATCH2_GOLD)

    match1_raw = mix.match_train_ball_items()
    match2_all = mix.match_train_ball_items_from(MATCH2_TRAIN, "match2_train100")
    match2_man = json.loads((MATCH2_TRAIN / "manifest.json").read_text())
    match2_by_name = {r["image"]: r for r in match2_man["frames"]}
    match2_train = []
    match2_valid = []
    for item in match2_all:
        row = match2_by_name[item["file_name"]]
        key = (row["camera"], int(row["frame_idx"]))
        if key in gold_keys:
            raise RuntimeError(f"Match2 gold leak into train source: {key}")
        if item["file_name"] in valid_names:
            match2_valid.append(item)
        elif item["file_name"] in train_names:
            match2_train.append(item)
        else:
            print(f"skip unlabeled/empty Match2 {item['file_name']}")

    if len(match2_train) != 87 or len(match2_valid) != 10:
        raise RuntimeError(
            f"expected 87/10 Match2 train/valid, got {len(match2_train)}/{len(match2_valid)}"
        )

    match_raw = match1_raw + match2_train
    match_items = mix.expand_match_with_augs(match_raw, MATCH_AUG_FACTOR, rng)
    print(f"Match1={len(match1_raw)} Match2_train={len(match2_train)} with_aug={len(match_items)}")

    official_all = (
        mix.official_ball_items("train")
        + mix.official_ball_items("valid")
        + mix.official_ball_items("test")
    )
    rng.shuffle(official_all)
    official_train = official_all[:N_OFFICIAL]
    official_valid = official_all[N_OFFICIAL : N_OFFICIAL + N_VALID_OFFICIAL]

    kjoyy_all = (
        mix.kjoyy_ball_items("train")
        + mix.kjoyy_ball_items("valid")
        + mix.kjoyy_ball_items("test")
    )
    rng.shuffle(kjoyy_all)
    kjoyy_train = kjoyy_all[:N_KJOYY]
    kjoyy_valid = kjoyy_all[N_KJOYY : N_KJOYY + N_VALID_KJOYY]

    gold100_test = gold100_test_items(mix)
    match2_test = match2_test_items(mix)
    gold_man = json.loads((MATCH2_GOLD / "manifest.json").read_text())
    gold_by_name = {r["image"]: r for r in gold_man["frames"]}
    for item in match2_test:
        item["strip_frame"] = int(gold_by_name[item["file_name"]]["strip_frame"])

    train_file_names = {m.get("file_name") for m in match_raw if m.get("file_name")}
    leak = train_file_names & {t["file_name"] for t in gold100_test}
    if leak:
        raise RuntimeError(f"Gold100 leak: {sorted(leak)[:5]}")
    leak2 = train_file_names & {t["file_name"] for t in match2_test}
    if leak2:
        raise RuntimeError(f"Match2 gold filename leak: {sorted(leak2)[:5]}")

    train_items = match_items + official_train + kjoyy_train
    valid_items = official_valid + kjoyy_valid + match2_valid
    rng.shuffle(train_items)
    rng.shuffle(valid_items)
    test_items = gold100_test + sorted(match2_test, key=lambda x: x["strip_frame"])

    train_stats = mix.write_split(OUT / "train", train_items, "train")
    valid_stats = mix.write_split(OUT / "valid", valid_items, "valid")
    test_stats = mix.write_split(OUT / "test", test_items, "test")
    split_meta = json.loads(SPLIT.read_text()) if SPLIT.is_file() else {}
    manifest = {
        "seed": SEED,
        "recipe": "match2_v9",
        "train": train_stats,
        "valid": valid_stats,
        "test": test_stats,
        "held_out": {
            "match2_gold_frames": str(MATCH2_GOLD.relative_to(ROOT)),
            "gold100_strip_0_49": "data/processed/gold_sets/match1_1_100",
            "match2_valid_from_train100": len(match2_valid),
        },
        "match2_split": {
            "n_train": len(match2_train),
            "n_valid": len(match2_valid),
            "n_test": len(match2_test),
            "skipped_empty": split_meta.get("n_train_skipped_empty"),
        },
        "notes": [
            "v9: v7 Match1 gold + Match2 train100 gold, x3 aug, light OFFICIAL/kjoyy.",
            "match2_gold_frames and Gold100 0-49 are test only — never train.",
            "Resume from v8 post_train; rank on Match2 gold P_emit.",
        ],
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    local = ROOT / "data/processed/gold_sets/ball_finetune_match2_v9_manifest.json"
    local.write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))
    print(f"Wrote pack {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
