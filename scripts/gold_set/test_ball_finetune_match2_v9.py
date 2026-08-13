#!/usr/bin/env python3
"""v9 pack: Match2 gold and Gold100 must not appear in train."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PACK = Path("/Volumes/LaCie/Projects/Soccer project data/ball_finetune_match2_v9")
GOLD = ROOT / "data/processed/gold_sets/match2_gold_frames"
G100 = ROOT / "data/processed/gold_sets/match1_1_100"


def coco_names(split: str) -> set[str]:
    return {
        im["file_name"]
        for im in json.loads((PACK / split / "_annotations.coco.json").read_text())["images"]
    }


def pack_image_stems(pack: Path) -> set[str]:
    man = json.loads((pack / "manifest.json").read_text())
    return {Path(r["image"]).stem for r in man["frames"]}


def main() -> int:
    man = json.loads((PACK / "manifest.json").read_text())
    assert man["recipe"] == "match2_v9"
    train = coco_names("train")
    test = coco_names("test")
    assert man["match2_split"]["n_train"] == 87
    assert man["match2_split"]["n_valid"] == 10
    assert man["match2_split"]["n_test"] == 50
    gold_stems = pack_image_stems(GOLD)
    g100_stems = pack_image_stems(G100)
    train_stems = {Path(n).stem for n in train}
    for stem in gold_stems:
        assert stem not in train_stems, f"gold frame in train: {stem}"
    # Gold100 originals keep file_name; mix renames to train_#####_source.ext
    sources = man["train"]["by_source"]
    assert "match2_train100" in sources or any("match2" in k for k in sources)
    assert "gold100_test_0_49" in man["test"]["by_source"]
    assert "match2_gold_test" in man["test"]["by_source"]
    assert not (train & test)
    print(
        "ok",
        "train", man["train"]["images"],
        "valid", man["valid"]["images"],
        "test", man["test"]["images"],
        "sources", sources,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
