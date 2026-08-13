#!/usr/bin/env python3
"""Assert Match 2 train/test split is disjoint and gold-sourced."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPLIT = ROOT / "data/processed/gold_sets/match2_train_test"
TRAIN_PACK = ROOT / "data/processed/gold_sets/match2_train_label100"
TEST_PACK = ROOT / "data/processed/gold_sets/match2_gold_frames"


def load_coco(path: Path) -> dict:
    return json.loads(path.read_text())


def keys_from_pack(pack: Path, file_names: set[str]) -> set[tuple]:
    man = json.loads((pack / "manifest.json").read_text())
    out = set()
    for row in man["frames"]:
        if row["image"] in file_names:
            out.add((row["camera"], int(row["frame_idx"])))
    return out


def main() -> int:
    split = json.loads((SPLIT / "split.json").read_text())
    train = load_coco(SPLIT / "train" / "_annotations.coco.json")
    valid = load_coco(SPLIT / "valid" / "_annotations.coco.json")
    test = load_coco(SPLIT / "test" / "_annotations.coco.json")
    assert split["n_train"] == len(train["images"])
    assert split["n_valid"] == len(valid["images"])
    assert split["n_test"] == len(test["images"])
    assert split["n_ball_train"] == len(train["annotations"])
    assert split["n_ball_test"] == len(test["annotations"])
    assert train["categories"][0]["name"] == "ball"
    gold_xml = TRAIN_PACK / "gold" / "annotations.xml"
    pre_xml = TRAIN_PACK / "prelabels" / "annotations.xml"
    assert gold_xml.stat().st_mtime > pre_xml.stat().st_mtime
    train_names = {i["file_name"] for i in train["images"]}
    valid_names = {i["file_name"] for i in valid["images"]}
    test_names = {i["file_name"] for i in test["images"]}
    assert not (train_names & valid_names)
    train_keys = keys_from_pack(TRAIN_PACK, train_names | valid_names)
    test_keys = keys_from_pack(TEST_PACK, test_names)
    assert not (train_keys & test_keys)
    for split_dir in ("train", "valid", "test"):
        coco = load_coco(SPLIT / split_dir / "_annotations.coco.json")
        for img in coco["images"]:
            assert (SPLIT / split_dir / img["file_name"]).is_file()
    print(
        f"ok train={split['n_train']} valid={split['n_valid']} "
        f"test={split['n_test']} skipped={split['n_train_skipped_empty']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
