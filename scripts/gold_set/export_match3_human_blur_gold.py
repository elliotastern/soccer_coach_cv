#!/usr/bin/env python3
"""Rebuild match3_human_blur_gold from pack labels.json (human_conf only)."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GOLD = ROOT / "data/processed/gold_sets"
OUT = GOLD / "match3_human_blur_gold"
PACKS = [
    "match3_quad_p9_655",
    "match3_blur_p1_soft1500",
    "match3_blur_p1_250",
    "match3_blur_p1_1100",
    "match3_blur_p1_1300",
]


def main() -> int:
    frames_out = OUT / "frames"
    gold_out = OUT / "gold"
    frames_out.mkdir(parents=True, exist_ok=True)
    gold_out.mkdir(parents=True, exist_ok=True)
    items = []
    by_pack: dict[str, int] = {}
    n_streaky = n_blurry = n_clear = 0
    for pname in PACKS:
        pack = GOLD / pname
        lab_path = pack / "labels.json"
        if not lab_path.is_file():
            continue
        lab = json.loads(lab_path.read_text(encoding="utf-8"))
        focus = lab["focus_cam"]
        for fr in lab["frames"]:
            seed = ((fr.get("cams") or {}).get(focus)) or {}
            if seed.get("human_conf") is None:
                continue
            balls = seed.get("gt_balls") or []
            if not balls:
                continue
            src = pack / "review" / "frames" / fr["file"]
            if not src.is_file():
                src = pack / "frames" / fr["file"]
            if not src.is_file():
                raise FileNotFoundError(src)
            dest_name = f"{pname}__{fr['file']}"
            dest = frames_out / dest_name
            if not dest.is_file():
                shutil.copy2(src, dest)
            b = balls[0]
            w, h = float(b["w"]), float(b["h"])
            asp = float(seed.get("aspect") or (max(w, h) / max(min(w, h), 1e-6)))
            streaky = bool(seed.get("streaky") or asp >= 1.35)
            lap = seed.get("lap_var")
            blurry = bool(seed.get("blurry") or (isinstance(lap, (int, float)) and lap < 45))
            if streaky:
                n_streaky += 1
            if blurry:
                n_blurry += 1
            if seed.get("clear"):
                n_clear += 1
            items.append(
                {
                    "pack": pname,
                    "focus_cam": focus,
                    "frame_i": fr["i"],
                    "file": dest_name,
                    "src_file": fr["file"],
                    "gt_balls": balls,
                    "human_conf": seed.get("human_conf"),
                    "gold_xy": seed.get("gold_xy"),
                    "aspect": round(asp, 3),
                    "streaky": streaky,
                    "blurry": blurry,
                    "lap_var": lap,
                    "soft_seed": seed.get("soft_seed"),
                    "clear": seed.get("clear"),
                }
            )
            by_pack[pname] = by_pack.get(pname, 0) + 1
    payload = {
        "role": "human_only_blur_session",
        "rule": "only frames with human_conf (Confirm box or drag)",
        "n": len(items),
        "n_streaky": n_streaky,
        "n_blurry": n_blurry,
        "n_clear": n_clear,
        "by_pack": by_pack,
        "items": items,
    }
    (gold_out / "human_labels.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (OUT / "labels.json").write_text(
        json.dumps(
            {
                "n": len(items),
                "n_streaky": n_streaky,
                "n_blurry": n_blurry,
                "by_pack": by_pack,
                "items": [
                    {**{k: v for k, v in it.items() if k != "gt_balls"}, "bbox": it["gt_balls"][0]}
                    for it in items
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (OUT / "manifest.json").write_text(
        json.dumps(
            {
                "pack": "match3_human_blur_gold",
                "n_human": len(items),
                "n_streaky": n_streaky,
                "n_blurry": n_blurry,
                "n_clear": n_clear,
                "by_pack": by_pack,
                "frames_local": True,
                "note": "JSON tracked in git; frames/ local only",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (OUT / "README.md").write_text(
        "# match3_human_blur_gold\n\n"
        "Human-only (`human_conf`) Match 3 ball boxes for soft/blur work.\n\n"
        f"- **n={len(items)}** · streaky={n_streaky} · blurry={n_blurry} · clear={n_clear}\n"
        f"- by pack: `{by_pack}`\n\n"
        "Canonical JSON: `gold/human_labels.json` (in git).\n"
        "`frames/` copies stay local (gitignored).\n\n"
        "See `docs/product/MATCH3_HUMAN_BLUR_GOLD.md`.\n"
        "Refresh: `PYTHONPATH=. python3 scripts/gold_set/export_match3_human_blur_gold.py`\n",
        encoding="utf-8",
    )
    print(f"wrote {OUT}: n={len(items)} by_pack={by_pack}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
