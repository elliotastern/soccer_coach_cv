#!/usr/bin/env python3
"""v15 residual specialty: leftover soft/no_det under v14 strip caches.

Resume-from-v14. Prefer v14_residual_det_cache when scoring residual frames.
Extra hard-blur copies vs v14. Match3 120–194 held out. No holdout gallery.
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
    MIX_PY,
    M3_TRAIN_MAX,
    clamp_box,
    frame_jpg,
    load_py,
)
from build_ball_finetune_v12_hard import (  # noqa: E402
    blur_image,
    erase_ball_patch,
)
from eval_match2_top_left_multicam_baseline import cache_load  # noqa: E402

OUT = Path("/Volumes/LaCie/Projects/Soccer project data/ball_finetune_v15_residual")
V14_CACHE_DIR = ROOT / "reports/eval_match3/improve_eng_loop/v14_residual_det_cache"
V13_CACHE_DIR = ROOT / "reports/eval_match3/improve_eng_loop/v13_residual_det_cache"
V12_CANDIDATES = [
    Path("/Volumes/LaCie/Projects/Soccer project data/ball_finetune_v12_hard"),
    ROOT / "data/processed/gold_sets/ball_finetune_v12_hard",
]
SEED = 42
EMIT_CONF = 0.80
SOFT_PREF_LO = 0.50
SOFT_PREF_HI = 0.79
SOFT_WIDE_LO = 0.20
HARD_COPIES = 3
TINY_PASTES = 3
TINY_SIDES = (8, 10, 12, 14, 16, 18)
N_V12_ANCHOR_TRAIN = 200
N_V12_ANCHOR_VALID = 40
SIDE_TIGHT = 24
WH = (1920, 1080)
M3_HOLDOUT_LO = 120
M3_HOLDOUT_HI = 194

STRIP_SPECS = [
    {
        "pack": "match3_quad_p10_31",
        "focus": "P10",
        "v12_cache_names": [
            "det_cache_quad_P10_t00031.0s_thr010.json",
        ],
    },
    {
        "pack": "match3_quad_p8_87",
        "focus": "P8",
        "v12_cache_names": [
            "det_cache_quad_P8_t00087.0s_stride1_thr010.json",
            "det_cache_quad_P8_t00087.0s_thr010.json",
        ],
    },
    {
        "pack": "match3_quad_p9",
        "focus": "P9",
        "v12_cache_names": [],
        "optional": True,
    },
]


def require(path: Path, what: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{what} missing: {path}")
    return path


def center_tight(box: list[float], side: float = SIDE_TIGHT) -> list[float]:
    x, y, w, h = [float(v) for v in box]
    cx, cy = x + w / 2.0, y + h / 2.0
    half = side / 2.0
    return [cx - half, cy - half, side, side]


def resolve_v12_root() -> Path:
    for root in V12_CANDIDATES:
        coco = root / "train" / "_annotations.coco.json"
        if coco.is_file():
            return root
    raise FileNotFoundError(
        "v12_hard pack not found. Tried:\n  "
        + "\n  ".join(str(p) for p in V12_CANDIDATES)
    )


def resolve_det_cache(pack_dir: Path, labels: dict, spec: dict) -> Path:
    """Prefer v14 strip caches, then v13, then labels.json, then gallery."""
    stem = labels.get("stem") or ""
    if stem:
        v14 = V14_CACHE_DIR / f"det_cache_{stem}_v14_thr010.json"
        if v14.is_file():
            return v14
        v13 = V13_CACHE_DIR / f"det_cache_{stem}_v13_thr010.json"
        if v13.is_file():
            return v13
    rel = labels.get("det_cache")
    if rel:
        p = ROOT / rel
        if p.is_file():
            return p
    names = list(spec.get("v12_cache_names") or [])
    if stem:
        names.extend(
            [
                f"det_cache_{stem}_stride1_thr010.json",
                f"det_cache_{stem}_thr010.json",
            ]
        )
    search_dirs = [
        V14_CACHE_DIR,
        V13_CACHE_DIR,
        ROOT / "reports/eval_match3/quad_pitchmap_gallery/det_cache",
        ROOT / "reports/eval_match3/quad_pitchmap_gallery_v12_hard/det_cache",
        ROOT / "reports/eval_match3/quad_pitchmap_gallery/det_cache_defish_detect",
    ]
    seen = set()
    for d in search_dirs:
        for name in names:
            p = d / name
            key = str(p)
            if key in seen:
                continue
            seen.add(key)
            if p.is_file():
                return p
    raise FileNotFoundError(
        f"No strip det_cache for {pack_dir.name}. Checked labels.det_cache + gallery fallbacks"
    )


def focus_conf(dets: dict, focus: str, i: int) -> float | None:
    frames = dets.get(focus) or []
    if i < 0 or i >= len(frames):
        return None
    rows = frames[i]
    if not rows:
        return None
    return float(max(rows, key=lambda r: float(r[1]))[1])


def conf_band(conf: float | None) -> str:
    if conf is None:
        return "no_det"
    if conf >= EMIT_CONF:
        return "ge080"
    if conf >= SOFT_PREF_LO:
        return "050_079"
    if conf >= SOFT_WIDE_LO:
        return "020_049"
    return "lt020"


def is_residual(conf: float | None, clear: bool, focus: str) -> bool:
    """FN residual: soft conf / low / no_det. Clear preferred; P8 no_det+gt ok."""
    if conf is not None and conf >= EMIT_CONF:
        return False
    if clear:
        return True
    # Extra P8 miss coverage: no_det with gt even if not marked clear
    return focus == "P8" and conf is None


def residual_items_from_strip(spec: dict) -> tuple[list[dict], dict]:
    pack = spec["pack"]
    focus = spec["focus"]
    pack_dir = ROOT / "data/processed/gold_sets" / pack
    if not pack_dir.is_dir():
        if spec.get("optional"):
            return [], {"pack": pack, "skipped": "missing_pack"}
        raise FileNotFoundError(f"Required strip pack missing: {pack_dir}")
    labels_path = require(pack_dir / "labels.json", f"{pack} labels")
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    frames_dir = require(pack_dir / "review" / "frames", f"{pack} frames")
    cache_path = resolve_det_cache(pack_dir, labels, spec)
    dets = cache_load(cache_path)
    w, h = WH
    out = []
    held = []
    bands = {"050_079": 0, "020_049": 0, "lt020": 0, "no_det": 0, "ge080": 0}
    for fr in labels["frames"]:
        i = int(fr["i"])
        seed = (fr.get("cams") or {}).get(focus) or {}
        balls = seed.get("gt_balls") or []
        if not balls:
            continue
        conf = focus_conf(dets, focus, i)
        band = conf_band(conf)
        bands[band] = bands.get(band, 0) + 1
        clear = bool(seed.get("clear"))
        if not is_residual(conf, clear, focus):
            continue
        src = frames_dir / fr["file"] if fr.get("file") else None
        if src is None or not src.is_file():
            src = frame_jpg(frames_dir, i)
        if src is None or not src.is_file():
            raise FileNotFoundError(f"{pack} missing frame {i}")
        kept = [
            b
            for b in (
                clamp_box([bb["x"], bb["y"], bb["w"], bb["h"]], w, h) for bb in balls
            )
            if b
        ]
        if not kept:
            continue
        item = {
            "src": src,
            "width": w,
            "height": h,
            "boxes": kept,
            "boxes_tight": [center_tight(b) for b in kept],
            "source": f"residual_{focus.lower()}",
            "file_name": f"{pack}_{src.name}",
            "frame": i,
            "pack": pack,
            "focus": focus,
            "focus_conf": conf,
            "conf_band": band,
            "clear": clear,
        }
        if M3_HOLDOUT_LO <= i <= M3_HOLDOUT_HI:
            held.append(item)
            continue
        if i > M3_TRAIN_MAX:
            continue
        out.append(item)
    meta = {
        "pack": pack,
        "focus": focus,
        "det_cache": str(cache_path.relative_to(ROOT)),
        "n_residual_train": len(out),
        "n_holdout_120_194": len(held),
        "bands_all_gt": bands,
        "bands_train": _band_counts(out),
    }
    return out, meta


def _band_counts(items: list[dict]) -> dict:
    out = {"050_079": 0, "020_049": 0, "lt020": 0, "no_det": 0}
    for it in items:
        b = it.get("conf_band")
        if b in out:
            out[b] += 1
    return out


def items_from_v12_coco(split_dir: Path, n: int, rng: random.Random, tag: str) -> list[dict]:
    coco_path = require(split_dir / "_annotations.coco.json", f"v12 {tag} coco")
    coco = json.loads(coco_path.read_text(encoding="utf-8"))
    images = {int(im["id"]): im for im in coco.get("images") or []}
    by_img: dict[int, list] = {}
    for ann in coco.get("annotations") or []:
        if int(ann.get("category_id", -1)) != 0:
            continue
        iid = int(ann["image_id"])
        box = [float(v) for v in ann["bbox"]]
        by_img.setdefault(iid, []).append(box)
    ids = [iid for iid, boxes in by_img.items() if boxes]
    if not ids:
        raise RuntimeError(f"No ball annotations in {coco_path}")
    rng.shuffle(ids)
    picked = ids[: min(n, len(ids))]
    items = []
    for iid in picked:
        im = images[iid]
        fname = str(im["file_name"])
        src = split_dir / fname
        if not src.is_file():
            raise FileNotFoundError(f"v12 image missing: {src}")
        w = int(im.get("width") or 0)
        h = int(im.get("height") or 0)
        boxes = []
        for b in by_img[iid]:
            c = clamp_box(b, w, h)
            if c:
                boxes.append(c)
        if not boxes:
            continue
        items.append(
            {
                "src": src,
                "width": w,
                "height": h,
                "boxes": boxes,
                "boxes_tight": [center_tight(b) for b in boxes],
                "source": f"v12_anchor_{tag}",
                "file_name": fname,
            }
        )
    return items


def hard_blur_copies(items: list[dict], rng: random.Random) -> list[dict]:
    aug_dir = OUT / "_hard_blur"
    if aug_dir.exists():
        shutil.rmtree(aug_dir)
    aug_dir.mkdir(parents=True)
    out = []
    for idx, item in enumerate(items):
        with Image.open(item["src"]) as im0:
            base = im0.convert("RGB")
        w, h = item["width"], item["height"]
        for k in range(HARD_COPIES):
            img = blur_image(base.copy(), rng)
            if rng.random() < 0.30:
                scale = rng.uniform(0.60, 0.88)
                sw, sh = max(64, int(w * scale)), max(64, int(h * scale))
                img = img.resize((sw, sh), Image.BILINEAR).resize((w, h), Image.BILINEAR)
            path = aug_dir / f"hard{k}_{idx:04d}.jpg"
            img.save(path, format="JPEG", quality=int(rng.choice([60, 70, 80])))
            boxes = [list(b) for b in item["boxes"]]
            out.append(
                {
                    "src": path,
                    "width": w,
                    "height": h,
                    "boxes": boxes,
                    "boxes_tight": [center_tight(b) for b in boxes],
                    "source": f"hard_blur_{k}",
                    "file_name": path.name,
                }
            )
    return out


def tiny_paste_items(items: list[dict], rng: random.Random) -> list[dict]:
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
            ball = ball.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.6, 2.0)))
            px = rng.randint(side, max(side + 1, w - side - 1))
            py = rng.randint(side, max(side + 1, h - side - 1))
            img.paste(ball, (px, py))
            path = aug_dir / f"tiny{k}_{idx:04d}.jpg"
            img.save(path, format="JPEG", quality=int(rng.choice([65, 75, 85])))
            boxes = [[float(px), float(py), float(side), float(side)]]
            out.append(
                {
                    "src": path,
                    "width": w,
                    "height": h,
                    "boxes": boxes,
                    "boxes_tight": [center_tight(b) for b in boxes],
                    "source": f"tiny_paste_{k}",
                    "file_name": path.name,
                }
            )
    return out


def tight_manifest_rows(items: list[dict], split: str) -> list[dict]:
    rows = []
    for it in items:
        for box, tight in zip(it["boxes"], it.get("boxes_tight") or []):
            x, y, w, h = box
            rows.append(
                {
                    "split": split,
                    "file_name": it.get("file_name"),
                    "source": it.get("source"),
                    "pack": it.get("pack"),
                    "frame": it.get("frame"),
                    "focus_conf": it.get("focus_conf"),
                    "bbox": [float(x), float(y), float(w), float(h)],
                    "bbox_tight": [float(v) for v in tight],
                    "side_px": SIDE_TIGHT,
                }
            )
    return rows


def main() -> int:
    require(MIX_PY, "match mix helper")
    mix = load_py(MIX_PY, "mix")
    v12_root = resolve_v12_root()
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    mix.OUT = OUT
    rng = random.Random(SEED)

    residual = []
    strip_meta = []
    for spec in STRIP_SPECS:
        items, meta = residual_items_from_strip(spec)
        strip_meta.append(meta)
        residual.extend(items)

    if not residual:
        raise RuntimeError("No residual FN frames selected from strip packs")

    # Prefer soft-conf band ordering in manifest notes; keep all residuals
    residual.sort(
        key=lambda it: (
            0 if it.get("conf_band") == "050_079" else 1 if it.get("conf_band") == "020_049" else 2,
            it.get("pack") or "",
            int(it.get("frame") or 0),
        )
    )

    n_p8 = sum(1 for it in residual if it.get("focus") == "P8")
    n_p10 = sum(1 for it in residual if it.get("focus") == "P10")
    n_p9 = sum(1 for it in residual if it.get("focus") == "P9")

    hard = hard_blur_copies(residual, rng)
    tiny = tiny_paste_items(residual, rng)
    anchor_train = items_from_v12_coco(
        v12_root / "train", N_V12_ANCHOR_TRAIN, rng, "train"
    )
    anchor_valid = items_from_v12_coco(
        v12_root / "valid", N_V12_ANCHOR_VALID, rng, "valid"
    )

    train_items = residual + hard + tiny + anchor_train
    valid_items = list(anchor_valid)
    rng.shuffle(train_items)
    rng.shuffle(valid_items)

    # Holdout leak check: no 120–194 M3 frames in train
    for it in train_items:
        fr = it.get("frame")
        if fr is None:
            continue
        if M3_HOLDOUT_LO <= int(fr) <= M3_HOLDOUT_HI:
            raise RuntimeError(f"Match3 holdout leak into train: frame {fr}")

    train_stats = mix.write_split(OUT / "train", train_items, "train")
    valid_stats = mix.write_split(OUT / "valid", valid_items, "valid")

    manifest = {
        "seed": SEED,
        "recipe": "v15_residual_fn_conf_after_v14",
        "resume_from": "models/v14_residual_snaps/post_train/checkpoint.pth",
        "train": train_stats,
        "valid": valid_stats,
        "raw_counts": {
            "residual_fn": len(residual),
            "residual_p8": n_p8,
            "residual_p10": n_p10,
            "residual_p9": n_p9,
            "hard_blur": len(hard),
            "tiny_paste": len(tiny),
            "v12_anchor_train": len(anchor_train),
            "v12_anchor_valid": len(anchor_valid),
        },
        "strips": strip_meta,
        "hard": {
            "hard_copies": HARD_COPIES,
            "tiny_pastes_per_frame": TINY_PASTES,
            "tiny_sides": list(TINY_SIDES),
            "applied_to": "residual_items_only",
        },
        "conf_policy": {
            "emit_conf": EMIT_CONF,
            "soft_pref": [SOFT_PREF_LO, SOFT_PREF_HI],
            "soft_wide": [SOFT_WIDE_LO, EMIT_CONF],
            "include_no_det": True,
            "m3_train_max": M3_TRAIN_MAX,
            "held_out_frames": f"{M3_HOLDOUT_LO}–{M3_HOLDOUT_HI}",
        },
        "v12_anchor_root": str(v12_root),
        "bbox_tight": {
            "side_px": SIDE_TIGHT,
            "note": "Primary COCO boxes stay gold xywh; bbox_tight is center square for optional experiments",
            "n_train_rows": len(tight_manifest_rows(train_items, "train")),
            "n_valid_rows": len(tight_manifest_rows(valid_items, "valid")),
            "items_sample": tight_manifest_rows(residual[:20], "residual_raw"),
        },
        "held_out": "Match3 M1 frames 120–194 (same as v11); not written to train",
        "notes": [
            "Thin residual specialty after v14 promote (leftover soft conf + no_det under v14 caches).",
            "Resume from v14_residual; short epoch budget (see configs/finetune_v15_residual_catch.yaml).",
            "Hard blur / tiny paste only on residual strip items (3 copies each).",
            "v12_hard train subset anchors medium balls so specialty does not collapse AP.",
            "Do not include holdout gallery frames.",
        ],
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    local = ROOT / "data/processed/gold_sets/ball_finetune_v15_residual_manifest.json"
    local.parent.mkdir(parents=True, exist_ok=True)
    local.write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))
    print(f"Wrote pack {OUT}")
    print(
        f"residual FN: total={len(residual)} P8={n_p8} P10={n_p10} P9={n_p9} "
        f"train_images={train_stats.get('images')} valid_images={valid_stats.get('images')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
