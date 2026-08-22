#!/usr/bin/env python3
"""C2: quad det funnel — v10 vs v12_hard (+ optional SAHI) on Match 3 quad caches."""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "gold_set"))

import build_match3_pitchmap_gallery as b  # noqa: E402
from eval_match2_4quad_multicam_survey import (  # noqa: E402
    DETECT_H,
    DETECT_W,
    SIZE,
    cache_dump_n,
    dets_to_rows,
    read_resized,
    source_path,
)
from eval_match2_top_left_multicam_baseline import cache_load  # noqa: E402
from fn_audit_match3_quad import audit_quad_cache  # noqa: E402
from score_match3_ball_m1 import score_cache  # noqa: E402
from src.perception.ball_prelabel import BallPrelabelConfig, BallPrelabeler  # noqa: E402
from src.perception.rfdetr_local import load_ball_model  # noqa: E402

OUT = ROOT / "reports/eval_match3/improve_eng_loop/c2_quad_det_funnel.json"
QUAD_SRC = ROOT / "reports/eval_match3/quad_pitchmap_gallery/source"
QUAD_CACHE = ROOT / "reports/eval_match3/quad_pitchmap_gallery/det_cache"
CKPT_V10 = ROOT / "models/v10_snaps/post_train/checkpoint.pth"
CKPT_V12 = ROOT / "models/v12_hard_snaps/post_train/checkpoint.pth"
STEMS = [
    "quad_P9_t00559.2s",
    "quad_P9_t00655.3s",
    "quad_P8_t00087.0s",
    "quad_P10_t00031.0s",
]
STRIDE = 2


def variant_cfgs() -> list[tuple[str, Path, BallPrelabelConfig]]:
    plain = BallPrelabelConfig(threshold=0.10, use_sahi=False, topk=5, **SIZE)
    sahi_fb = BallPrelabelConfig(
        threshold=0.10,
        use_sahi=True,
        sahi_fallback_only=True,
        sahi_recover_only=True,
        topk=5,
        **SIZE,
    )
    return [
        ("v10_plain", CKPT_V10, plain),
        ("v12_plain", CKPT_V12, plain),
        ("v12_sahi_fallback", CKPT_V12, sahi_fb),
    ]


def detect_stem(stem: str, model, cfg: BallPrelabelConfig) -> dict:
    first = source_path(QUAD_SRC, stem, b.CAM_IDS[0])
    cap0 = cv2.VideoCapture(str(first))
    n = int(cap0.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap0.release()
    pre = BallPrelabeler(model, cfg, class_id=1)
    caps = {}
    for cam in b.CAM_IDS:
        path = source_path(QUAD_SRC, stem, cam)
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"open failed {path}")
        caps[cam] = cap
    out = {cam: [] for cam in b.CAM_IDS}
    try:
        for i in range(n):
            for cam in b.CAM_IDS:
                frame = read_resized(caps[cam])
                if frame is None:
                    for c in b.CAM_IDS:
                        out[c] = out[c][:i]
                    n = i
                    break
                if frame.shape[1] != DETECT_W or frame.shape[0] != DETECT_H:
                    frame = cv2.resize(
                        frame, (DETECT_W, DETECT_H), interpolation=cv2.INTER_AREA
                    )
                if i % STRIDE == 0:
                    out[cam].append(dets_to_rows(pre.detect_bgr(frame)))
                else:
                    out[cam].append([])
            else:
                continue
            break
    finally:
        for cap in caps.values():
            cap.release()
    return out


def score_stem_cache(path: Path) -> dict:
    row = score_cache(path)
    audit = audit_quad_cache(path)
    return {
        "emit": row["emit"],
        "clear_ball_proxy_R": row["clear_ball_proxy_R"],
        "clear_frames": row["clear_frames"],
        "clear_emit": row["clear_emit"],
        "agree": row["agree"],
        "product_clear_R": audit.get("clear_ball_proxy_R"),
    }


def run_variant(name: str, ckpt: Path, cfg: BallPrelabelConfig) -> dict:
    if not ckpt.is_file():
        raise FileNotFoundError(ckpt)
    model = load_ball_model(str(ckpt))
    per_stem = {}
    tmp_dir = OUT.parent / f"_c2_tmp_{name}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    for stem in STEMS:
        print(f"  detect {name} {stem}", flush=True)
        payload = detect_stem(stem, model, cfg)
        cache_path = tmp_dir / f"det_cache_{stem}_thr010.json"
        cache_dump_n(cache_path, payload, len(payload[b.CAM_IDS[0]]))
        per_stem[stem] = score_stem_cache(cache_path)
    totals = {"clear_frames": 0, "clear_emit": 0, "emit": 0}
    for row in per_stem.values():
        for k in totals:
            totals[k] += row[k]
    totals["clear_ball_proxy_R"] = (
        None
        if totals["clear_frames"] == 0
        else round(totals["clear_emit"] / totals["clear_frames"], 3)
    )
    return {"variant": name, "checkpoint": str(ckpt.name), "per_stem": per_stem, "totals": totals}


def pick_winner(rows: list[dict]) -> str | None:
    ok = [r for r in rows if float(r["totals"].get("clear_ball_proxy_R") or 0) >= 0.0]
    if not ok:
        return None
    ok.sort(
        key=lambda r: (
            -float(r["totals"].get("clear_ball_proxy_R") or 0),
            -int(r["totals"].get("clear_emit") or 0),
            r["variant"],
        )
    )
    return ok[0]["variant"]


def promote_winner(name: str) -> None:
    tmp_dir = OUT.parent / f"_c2_tmp_{name}"
    if not tmp_dir.is_dir():
        raise FileNotFoundError(tmp_dir)
    QUAD_CACHE.mkdir(parents=True, exist_ok=True)
    for stem in STEMS:
        src = tmp_dir / f"det_cache_{stem}_thr010.json"
        dst = QUAD_CACHE / src.name
        bak = dst.with_suffix(".json.v10bak")
        if dst.is_file() and not bak.is_file():
            shutil.copy2(dst, bak)
        shutil.copy2(src, dst)
        print(f"promoted {dst.name}", flush=True)


def main() -> int:
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--promote", type=str, default="", help="promote variant tmp caches to quad det_cache")
    args = p.parse_args()
    if args.promote:
        promote_winner(args.promote)
        return 0

    baseline = {}
    for stem in STEMS:
        path = QUAD_CACHE / f"det_cache_{stem}_thr010.json"
        if path.is_file():
            baseline[stem] = score_stem_cache(path)

    rows = []
    for name, ckpt, cfg in variant_cfgs():
        print(f"variant {name}", flush=True)
        rows.append(run_variant(name, ckpt, cfg))

    winner = pick_winner(rows)
    out = {
        "stems": STEMS,
        "stride": STRIDE,
        "baseline_v10": baseline,
        "variants": rows,
        "winner": winner,
        "gate": "max quad clear_ball_proxy_R (product fuse in score_cache)",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    for r in rows:
        t = r["totals"]
        print(
            f"{r['variant']}: clear_R={t['clear_ball_proxy_R']} "
            f"emit={t['emit']} clear={t['clear_frames']}"
        )
    print(f"winner={winner}")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
