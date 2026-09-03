#!/usr/bin/env python3
"""v12 hard-ball mix: small + blur specialty on top of v11 validated sources.

Resume target: v11 snap. Same holdouts as v11 (Gold100, Match2 gold,
Match3 120–194, 4quad 240–299). Overweights already-small boxes and adds
in-place blur/JPEG crush + tiny paste FN augs.
"""
from __future__ import annotations

import json
import random
import shutil
import sys
from pathlib import Path

from PIL import Image, ImageFilter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "gold_set"))

from build_ball_finetune_v11 import (  # noqa: E402
    MATCH1_AUG,
    MIX_PY,
    V9_PY,
    items_from_match3,
    items_from_quad,
    load_py,
    split_match2,
    stills,
    QUAD_PACKS,
)

OUT = Path("/Volumes/LaCie/Projects/Soccer project data/ball_finetune_v12_hard")
SEED = 42
TINY_SIDES = (8, 10, 12, 14, 16, 18)
SMALL_MAX_SIDE = 28.0
HARD_COPIES = 3
TINY_PASTES = 2
ANCHOR_DENSE_AUG = 2
N_OFFICIAL = 40
N_VALID_OFFICIAL = 15
N_KJOYY = 60
N_VALID_KJOYY = 20


def max_side(boxes: list) -> float:
    return max(max(float(b[2]), float(b[3])) for b in boxes)


def is_small(item: dict) -> bool:
    return bool(item.get("boxes")) and max_side(item["boxes"]) <= SMALL_MAX_SIDE


def blur_image(img: Image.Image, rng: random.Random) -> Image.Image:
    if rng.random() < 0.55:
        # Approximate motion: stronger Gaussian + optional second pass
        r = rng.uniform(1.2, 3.2)
        out = img.filter(ImageFilter.GaussianBlur(radius=r))
        if rng.random() < 0.4:
            out = out.filter(ImageFilter.BoxBlur(radius=rng.uniform(0.5, 1.5)))
        return out
    return img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.8, 2.4)))


def hard_blur_copies(items: list[dict], rng: random.Random) -> list[dict]:
    """In-place blur/JPEG on already-small balls (keep real location)."""
    aug_dir = OUT / "_hard_blur"
    if aug_dir.exists():
        shutil.rmtree(aug_dir)
    aug_dir.mkdir(parents=True)
    out = []
    small = [it for it in items if is_small(it)]
    for idx, item in enumerate(small):
        with Image.open(item["src"]) as im0:
            base = im0.convert("RGB")
        w, h = item["width"], item["height"]
        for k in range(HARD_COPIES):
            img = blur_image(base.copy(), rng)
            if rng.random() < 0.35:
                # Mild whole-frame shrink→expand (extra softness)
                scale = rng.uniform(0.55, 0.85)
                sw, sh = max(64, int(w * scale)), max(64, int(h * scale))
                img = img.resize((sw, sh), Image.BILINEAR).resize((w, h), Image.BILINEAR)
            path = aug_dir / f"hard{k}_{idx:04d}.jpg"
            img.save(path, format="JPEG", quality=int(rng.choice([55, 65, 75, 85])))
            out.append(
                {
                    "src": path,
                    "width": w,
                    "height": h,
                    "boxes": [list(b) for b in item["boxes"]],
                    "source": f"hard_blur_{k}",
                    "file_name": path.name,
                }
            )
    return out


def erase_ball_patch(img: Image.Image, boxes: list, w: int, h: int) -> None:
    for x, y, bw, bh in boxes:
        x0, y0 = max(0, int(x)), max(0, int(y))
        x1, y1 = min(w, int(x + bw)), min(h, int(y + bh))
        if x1 <= x0 or y1 <= y0:
            continue
        patch = img.crop((x0, y0, x1, y1)).resize((1, 1)).resize((x1 - x0, y1 - y0))
        img.paste(patch, (x0, y0))


def tiny_paste_items(items: list[dict], rng: random.Random) -> list[dict]:
    """FN-style tiny paste (8–18 px) with blur, like v8 but smaller."""
    aug_dir = OUT / "_tiny_paste"
    if aug_dir.exists():
        shutil.rmtree(aug_dir)
    aug_dir.mkdir(parents=True)
    out = []
    for idx, item in enumerate(items):
        if not item["boxes"]:
            continue
        with Image.open(item["src"]) as im0:
            base = im0.convert("RGB")
        w, h = item["width"], item["height"]
        bx, by, bw, bh = item["boxes"][0]
        crop = base.crop((int(bx), int(by), int(bx + bw), int(by + bh)))
        for k in range(TINY_PASTES):
            img = base.copy()
            erase_ball_patch(img, item["boxes"], w, h)
            side = rng.choice(TINY_SIDES)
            ball = crop.resize((side, side), Image.BILINEAR)
            ball = ball.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.6, 2.2)))
            px = rng.randint(side, max(side + 1, w - side - 1))
            py = rng.randint(side, max(side + 1, h - side - 1))
            img.paste(ball, (px, py))
            path = aug_dir / f"tiny{k}_{idx:04d}.jpg"
            img.save(path, format="JPEG", quality=int(rng.choice([60, 70, 80])))
            out.append(
                {
                    "src": path,
                    "width": w,
                    "height": h,
                    "boxes": [[float(px), float(py), float(side), float(side)]],
                    "source": f"tiny_paste_{k}",
                    "file_name": path.name,
                }
            )
    return out


def collect_raw(mix, v9):
    match1_raw = [
        it for it in mix.match_train_ball_items()
        if it["source"] != "match1_train_batch2"
    ]
    match2_train, match2_valid = split_match2(mix, v9)
    quad_train = []
    for pack, tag in QUAD_PACKS:
        tr, _ = items_from_quad(pack, tag, 1920, 1080)
        quad_train.extend(tr)
    m3_train, _ = items_from_match3()
    dense_raw = quad_train + m3_train
    return match1_raw, match2_train, dense_raw, match2_valid


def main() -> int:
    mix = load_py(MIX_PY, "mix")
    v9 = load_py(V9_PY, "v9")
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    mix.OUT = OUT
    rng = random.Random(SEED)

    match1_raw, match2_train, dense_raw, match2_valid = collect_raw(mix, v9)
    all_raw = match1_raw + match2_train + dense_raw
    n_small = sum(1 for it in all_raw if is_small(it))

    # Anchor: keep validated diversity so medium balls don't collapse
    match1_items = mix.expand_match_with_augs(match1_raw, MATCH1_AUG, rng)
    match2_items = mix.expand_match_with_augs(match2_train, ANCHOR_DENSE_AUG, rng)
    dense_items = mix.expand_match_with_augs(dense_raw, ANCHOR_DENSE_AUG, rng)
    still_train, still_valid = stills(mix, rng)

    hard = hard_blur_copies(all_raw, rng)
    tiny = tiny_paste_items(all_raw, rng)

    gold100_test = v9.gold100_test_items(mix)
    match2_test = v9.match2_test_items(mix)

    train_items = match1_items + match2_items + dense_items + still_train + hard + tiny
    valid_items = still_valid + match2_valid
    rng.shuffle(train_items)
    rng.shuffle(valid_items)
    test_items = gold100_test + match2_test

    train_stats = mix.write_split(OUT / "train", train_items, "train")
    valid_stats = mix.write_split(OUT / "valid", valid_items, "valid")
    test_stats = mix.write_split(OUT / "test", test_items, "test")
    manifest = {
        "seed": SEED,
        "recipe": "v12_hard_small_blur",
        "resume_from": "models/v11_snaps/post_train/checkpoint.pth",
        "train": train_stats,
        "valid": valid_stats,
        "test": test_stats,
        "raw_counts": {
            "match1": len(match1_raw),
            "match2": len(match2_train),
            "dense": len(dense_raw),
            "small_raw": n_small,
            "hard_blur": len(hard),
            "tiny_paste": len(tiny),
        },
        "hard": {
            "small_max_side_px": SMALL_MAX_SIDE,
            "hard_copies": HARD_COPIES,
            "tiny_sides": list(TINY_SIDES),
            "tiny_pastes_per_frame": TINY_PASTES,
        },
        "held_out": "same as v11 (Gold100 0-49, match2_gold, M3 120-194, quad 240-299)",
        "notes": [
            "Specialty finetune for tiny + blur balls after v11.",
            "In-place hard blur keeps real ball coords; tiny paste adds FN-style 8-18px.",
            "Anchor augs keep Match1/Match2/dense so AP50 does not collapse.",
        ],
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    local = ROOT / "data/processed/gold_sets/ball_finetune_v12_hard_manifest.json"
    local.write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))
    print(f"Wrote pack {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
