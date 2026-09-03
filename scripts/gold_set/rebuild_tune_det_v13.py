#!/usr/bin/env python3
"""Rebuild tune (pitchmap_gallery_v12_hard) det caches with product ball ckpt.

Writes det_cache_v13/ next to existing det_cache/. Scores clear_ball_proxy_R.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "gold_set"))

from rebuild_holdout_det_v13 import (  # noqa: E402
    CAMS,
    detect_stem,
    product_ckpt,
)
from score_match3_ball_m1 import score_cache  # noqa: E402
from src.perception.rfdetr_local import load_ball_model  # noqa: E402

TUNE = ROOT / "reports/eval_match3/pitchmap_gallery_v12_hard"
OUT_JSON = ROOT / "reports/eval_match3/improve_eng_loop/ab_v13_tune_score.json"
STRIDE = 2


def pack_score(cache_dir: Path) -> dict:
    paths = sorted(cache_dir.glob("det_cache_*_thr010.json"))
    rows = [score_cache(p) for p in paths]
    clear = sum(int(r.get("clear_frames") or 0) for r in rows)
    emit = sum(int(r.get("clear_emit") or 0) for r in rows)
    return {
        "n_caches": len(rows),
        "clear_frames": clear,
        "clear_emit": emit,
        "clear_ball_proxy_R": None if clear == 0 else round(emit / clear, 3),
        "rows": rows,
    }


def stems_from_caches(cache_dir: Path) -> list[str]:
    out = []
    for path in sorted(cache_dir.glob("det_cache_rand_*_thr010.json")):
        # det_cache_rand_t00627.9s_thr010.json
        name = path.name[len("det_cache_") :].rsplit("_thr", 1)[0]
        out.append(name)
    return out


def main() -> int:
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--skip-detect", action="store_true")
    p.add_argument("--stride", type=int, default=STRIDE)
    args = p.parse_args()
    ckpt = product_ckpt()
    if not ckpt.is_file():
        raise SystemExit(f"missing {ckpt}")
    base_dir = TUNE / "det_cache"
    v13_dir = TUNE / "det_cache_v13"
    src = TUNE / "source"
    stems = stems_from_caches(base_dir)
    if not stems:
        raise SystemExit(f"no baseline caches in {base_dir}")
    built = []
    if not args.skip_detect:
        model = load_ball_model(str(ckpt))
        for stem in stems:
            out_cache = v13_dir / f"det_cache_{stem}_thr010.json"
            print(f"detect {stem} → {out_cache.name}", flush=True)
            built.append(detect_stem(model, src, stem, out_cache, args.stride))
    baseline = pack_score(base_dir)
    cand = pack_score(v13_dir)
    payload = {
        "checkpoint": str(ckpt.relative_to(ROOT)),
        "gallery": str(TUNE.relative_to(ROOT)),
        "stride": args.stride,
        "baseline": baseline,
        "v13": cand,
        "built": built,
        "d_clear_R": None
        if baseline.get("clear_ball_proxy_R") is None or cand.get("clear_ball_proxy_R") is None
        else round(
            float(cand["clear_ball_proxy_R"]) - float(baseline["clear_ball_proxy_R"]),
            3,
        ),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        f"tune baseline_R={baseline.get('clear_ball_proxy_R')} "
        f"v13_R={cand.get('clear_ball_proxy_R')} d={payload['d_clear_R']}"
    )
    print(f"wrote {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
