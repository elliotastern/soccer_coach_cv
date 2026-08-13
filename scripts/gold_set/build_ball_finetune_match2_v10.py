#!/usr/bin/env python3
"""v10 mix: Match2-heavy, keep Match1 + light stills so other fields don't vanish.

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
V9_PY = ROOT / "scripts/gold_set/build_ball_finetune_match2_v9.py"
MIX_PY = Path("/Volumes/LaCie/Projects/Soccer project data/scripts/build_ball_finetune_match_mix.py")
OUT = Path("/Volumes/LaCie/Projects/Soccer project data/ball_finetune_match2_v10")
MATCH2_TRAIN = ROOT / "data/processed/gold_sets/match2_train_label100"
MATCH2_GOLD = ROOT / "data/processed/gold_sets/match2_gold_frames"
SPLIT = ROOT / "data/processed/gold_sets/match2_train_test/split.json"
SEED = 42
MATCH2_AUG = 5
MATCH1_AUG = 2
N_OFFICIAL = 40
N_KJOYY = 60
N_VALID_OFFICIAL = 15
N_VALID_KJOYY = 20


def load_py(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def split_match2(mix, v9) -> tuple[list, list]:
    train_names = v9.names_in_coco(
        ROOT / "data/processed/gold_sets/match2_train_test/train/_annotations.coco.json"
    )
    valid_names = v9.names_in_coco(
        ROOT / "data/processed/gold_sets/match2_train_test/valid/_annotations.coco.json"
    )
    gold_keys = v9.keys_from_pack(MATCH2_GOLD)
    match2_all = mix.match_train_ball_items_from(MATCH2_TRAIN, "match2_train100")
    match2_man = json.loads((MATCH2_TRAIN / "manifest.json").read_text())
    by_name = {r["image"]: r for r in match2_man["frames"]}
    train, valid = [], []
    for item in match2_all:
        row = by_name[item["file_name"]]
        key = (row["camera"], int(row["frame_idx"]))
        if key in gold_keys:
            raise RuntimeError(f"Match2 gold leak into train source: {key}")
        if item["file_name"] in valid_names:
            valid.append(item)
        elif item["file_name"] in train_names:
            train.append(item)
    if len(train) != 87 or len(valid) != 10:
        raise RuntimeError(f"expected 87/10 Match2, got {len(train)}/{len(valid)}")
    return train, valid


def stills(mix, rng) -> tuple[list, list]:
    official = (
        mix.official_ball_items("train")
        + mix.official_ball_items("valid")
        + mix.official_ball_items("test")
    )
    rng.shuffle(official)
    kjoyy = (
        mix.kjoyy_ball_items("train")
        + mix.kjoyy_ball_items("valid")
        + mix.kjoyy_ball_items("test")
    )
    rng.shuffle(kjoyy)
    train = official[:N_OFFICIAL] + kjoyy[:N_KJOYY]
    valid = (
        official[N_OFFICIAL : N_OFFICIAL + N_VALID_OFFICIAL]
        + kjoyy[N_KJOYY : N_KJOYY + N_VALID_KJOYY]
    )
    return train, valid


def main() -> int:
    mix = load_py(MIX_PY, "mix")
    v9 = load_py(V9_PY, "v9")
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    mix.OUT = OUT
    rng = random.Random(SEED)

    match1_raw = mix.match_train_ball_items()
    match2_train, match2_valid = split_match2(mix, v9)
    match2_items = mix.expand_match_with_augs(match2_train, MATCH2_AUG, rng)
    match1_items = mix.expand_match_with_augs(match1_raw, MATCH1_AUG, rng)
    print(
        f"Match1={len(match1_raw)} x{MATCH1_AUG}={len(match1_items)} "
        f"Match2={len(match2_train)} x{MATCH2_AUG}={len(match2_items)}"
    )

    still_train, still_valid = stills(mix, rng)
    gold100_test = v9.gold100_test_items(mix)
    match2_test = v9.match2_test_items(mix)
    gold_man = json.loads((MATCH2_GOLD / "manifest.json").read_text())
    gold_by_name = {r["image"]: r for r in gold_man["frames"]}
    for item in match2_test:
        item["strip_frame"] = int(gold_by_name[item["file_name"]]["strip_frame"])

    raw_names = {m["file_name"] for m in match1_raw + match2_train}
    leak = raw_names & {t["file_name"] for t in gold100_test + match2_test}
    if leak:
        raise RuntimeError(f"held-out leak: {sorted(leak)[:5]}")

    train_items = match2_items + match1_items + still_train
    valid_items = still_valid + match2_valid
    rng.shuffle(train_items)
    rng.shuffle(valid_items)
    test_items = gold100_test + sorted(match2_test, key=lambda x: x["strip_frame"])

    train_stats = mix.write_split(OUT / "train", train_items, "train")
    valid_stats = mix.write_split(OUT / "valid", valid_items, "valid")
    test_stats = mix.write_split(OUT / "test", test_items, "test")
    split_meta = json.loads(SPLIT.read_text()) if SPLIT.is_file() else {}
    manifest = {
        "seed": SEED,
        "recipe": "match2_v10",
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
        "aug": {"match2": MATCH2_AUG, "match1": MATCH1_AUG},
        "notes": [
            "v10: Match2 x5 (majority), Match1 x2 (other fields), light stills.",
            "Shorter resume from v9 to limit Match2 overfitting.",
            "match2_gold_frames and Gold100 0-49 are test only.",
        ],
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    local = ROOT / "data/processed/gold_sets/ball_finetune_match2_v10_manifest.json"
    local.write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))
    print(f"Wrote pack {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
