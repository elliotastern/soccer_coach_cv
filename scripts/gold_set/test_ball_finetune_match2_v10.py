#!/usr/bin/env python3
"""v10 pack: no gold leak; Match2 heavier than Match1; stills stay for other fields."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PACK = Path("/Volumes/LaCie/Projects/Soccer project data/ball_finetune_match2_v10")
GOLD = ROOT / "data/processed/gold_sets/match2_gold_frames"


def coco_names(split: str) -> set[str]:
    return {
        im["file_name"]
        for im in json.loads((PACK / split / "_annotations.coco.json").read_text())["images"]
    }


def main() -> int:
    man = json.loads((PACK / "manifest.json").read_text())
    assert man["recipe"] == "match2_v10"
    assert man["aug"]["match2"] == 5
    assert man["aug"]["match1"] == 2
    train = coco_names("train")
    test = coco_names("test")
    gold_stems = {
        Path(r["image"]).stem
        for r in json.loads((GOLD / "manifest.json").read_text())["frames"]
    }
    for stem in gold_stems:
        assert stem not in {Path(n).stem for n in train}, f"gold in train: {stem}"
    src = man["train"]["by_source"]
    match2_n = sum(v for k, v in src.items() if k.startswith("match2_"))
    match1_n = sum(v for k, v in src.items() if k.startswith("match1_"))
    still_n = sum(v for k, v in src.items() if k.startswith("official_") or k.startswith("kjoyy_"))
    assert match2_n > match1_n, f"Match2 {match2_n} should exceed Match1 {match1_n}"
    assert still_n >= 80, f"need light stills for other fields, got {still_n}"
    assert "gold100_test_0_49" in man["test"]["by_source"]
    assert "match2_gold_test" in man["test"]["by_source"]
    assert not (train & test)
    print("ok", "train", man["train"]["images"], "m2", match2_n, "m1", match1_n, "stills", still_n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
