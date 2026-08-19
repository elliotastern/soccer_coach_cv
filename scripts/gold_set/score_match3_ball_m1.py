#!/usr/bin/env python3
"""M1 Match 3 ball metrics.

When `data/processed/gold_sets/match3_quad_p10_31/labels.json` exists:
  - P_emit: among fuse emits on frames with gold_xy, fraction within HIT_M of gold
  - clear_ball_R: emit rate on frames marked clear

Also writes proxy gallery metrics (until human-corrected strip is final).
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "gold_set"))

from eval_match2_top_left_multicam_baseline import cache_load  # noqa: E402
from multicam_select_policy import MATCH3_THR_BY_CAM, filter_active  # noqa: E402
from src.mapping.match3_xy import (  # noqa: E402
    AGREE_M,
    EMIT_CONF,
    MIN_SUPPORT,
    fuse_balls,
    load_calib,
    map_ball_box,
)

OUT = ROOT / "reports/eval_match3/improve_eng_loop/m1_provisional.json"
STRIP = ROOT / "data/processed/gold_sets/match3_quad_p10_31/labels.json"
CAMS = ["P1", "P6", "P7", "P8", "P9", "P10", "P_Goal1", "P_Goal2"]
WH = (1920, 1080)
CLEAR_SIDE = 25.0
HIT_M = AGREE_M


def score_cache(path: Path) -> dict:
    dets = cache_load(path)
    cams = [c for c in CAMS if c in dets]
    n = len(next(iter(dets.values())))
    calibs = {c: v for c, v in ((c, load_calib(c)) for c in cams) if v}
    clear = clear_emit = emit = agree = mapped_ge2 = 0
    for i in range(n):
        active = filter_active(dets, i, cams, MATCH3_THR_BY_CAM)
        is_clear = any(
            float(rows[0][2]) >= CLEAR_SIDE and float(rows[0][1]) >= 0.30
            for rows in active.values()
        )
        mapped = []
        for cam, rows in active.items():
            rec = calibs.get(cam)
            if rec is None:
                continue
            box, conf, _side = rows[0]
            hit = map_ball_box(rec, box, float(conf), frame_wh=WH)
            if hit is not None:
                mapped.append(hit)
        if len(mapped) >= 2:
            mapped_ge2 += 1
        fused = fuse_balls(mapped)
        did_emit = fused is not None
        if did_emit:
            emit += 1
            if fused.get("agree"):
                agree += 1
        if is_clear:
            clear += 1
            if did_emit:
                clear_emit += 1
    return {
        "cache": path.name,
        "n": n,
        "emit": emit,
        "agree": agree,
        "mapped_ge2": mapped_ge2,
        "clear_frames": clear,
        "clear_emit": clear_emit,
        "clear_ball_proxy_R": None if clear == 0 else round(clear_emit / clear, 3),
        "agree_among_emit": None if emit == 0 else round(agree / emit, 3),
    }


def infer_cache_stride(dets: dict) -> int:
    """Gallery caches often detect every 2nd frame (demo --stride 2)."""
    for frames in dets.values():
        nonempty = [i for i, rows in enumerate(frames) if rows]
        if len(nonempty) < 10:
            continue
        even = sum(i % 2 == 0 for i in nonempty)
        odd = len(nonempty) - even
        if even > 0 and odd == 0:
            return 2
        if odd > 0 and even == 0:
            return 2
    return 1


def fuse_frame(dets, i, calibs):
    cams = [c for c in CAMS if c in dets]
    active = filter_active(dets, i, cams, MATCH3_THR_BY_CAM)
    mapped = []
    for cam, rows in active.items():
        rec = calibs.get(cam)
        if rec is None:
            continue
        box, conf, _ = rows[0]
        hit = map_ball_box(rec, box, float(conf), frame_wh=WH)
        if hit is not None:
            mapped.append(hit)
    return fuse_balls(mapped), len(mapped)


def fuse_frame_carry(dets, i, calibs, stride: int):
    fused, n_map = fuse_frame(dets, i, calibs)
    if fused is not None or stride <= 1:
        return fused, n_map
    for j in (i - 1, i + 1, i - 2, i + 2):
        if j < 0:
            continue
        n = len(next(iter(dets.values())))
        if j >= n:
            continue
        fused, n_map = fuse_frame(dets, j, calibs)
        if fused is not None:
            return fused, n_map
    return None, 0


def _score_strip_mode(labels, dets, calibs, focus: str, mode: str, stride: int) -> dict:
    tp = fp = emit = clear = clear_emit = 0
    errs = []
    for fr in labels["frames"]:
        i = int(fr["i"])
        if mode == "detect_ticks" and stride > 1 and i % stride != 0:
            continue
        seed = (fr.get("cams") or {}).get(focus) or {}
        gold = seed.get("gold_xy")
        is_clear = bool(seed.get("clear")) and gold is not None
        if is_clear:
            clear += 1
        if mode == "carry":
            fused, _ = fuse_frame_carry(dets, i, calibs, stride)
        else:
            fused, _ = fuse_frame(dets, i, calibs)
        if fused is None or gold is None:
            continue
        emit += 1
        err = math.hypot(
            float(fused["xy"][0]) - float(gold[0]),
            float(fused["xy"][1]) - float(gold[1]),
        )
        errs.append(err)
        if err <= HIT_M:
            tp += 1
        else:
            fp += 1
        if is_clear:
            clear_emit += 1
    p_emit = None if (tp + fp) == 0 else tp / (tp + fp)
    clear_r = None if clear == 0 else clear_emit / clear
    return {
        "n_clear": clear,
        "n_emit_scored": emit,
        "tp": tp,
        "fp": fp,
        "P_emit": None if p_emit is None else round(p_emit, 3),
        "clear_ball_R": None if clear_r is None else round(clear_r, 3),
        "err_median_m": None
        if not errs
        else round(sorted(errs)[len(errs) // 2], 3),
        "poc_pass_P_emit": bool(p_emit is not None and p_emit >= 0.80),
        "poc_pass_clear_R": bool(clear_r is not None and clear_r >= 0.80),
    }


def score_strip(labels_path: Path) -> dict:
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    cache_rel = labels.get("det_cache")
    if not cache_rel:
        raise ValueError("labels missing det_cache")
    dets = cache_load(ROOT / cache_rel)
    cams = [c for c in CAMS if c in dets]
    calibs = {c: v for c, v in ((c, load_calib(c)) for c in cams) if v}
    focus = labels.get("focus_cam") or "P10"
    stride = infer_cache_stride(dets)
    raw = _score_strip_mode(labels, dets, calibs, focus, "raw", stride)
    ticks = _score_strip_mode(labels, dets, calibs, focus, "detect_ticks", stride)
    carry = _score_strip_mode(labels, dets, calibs, focus, "carry", stride)
    # Primary = detect ticks (fair vs stride-2 cache). Carry ≈ hold last emit.
    primary = ticks if stride > 1 else raw
    return {
        "pack": labels.get("pack"),
        "provisional": True,
        "hit_m": HIT_M,
        "n_frames": len(labels["frames"]),
        "det_cache_stride": stride,
        "P_emit": primary["P_emit"],
        "clear_ball_R": primary["clear_ball_R"],
        "n_clear": primary["n_clear"],
        "n_emit_scored": primary["n_emit_scored"],
        "tp": primary["tp"],
        "fp": primary["fp"],
        "err_median_m": primary["err_median_m"],
        "poc_pass_P_emit": primary["poc_pass_P_emit"],
        "poc_pass_clear_R": primary["poc_pass_clear_R"],
        "modes": {
            "raw_all_label_frames": raw,
            "detect_ticks_only": ticks,
            "carry_neighbor_tick": carry,
        },
        "note": (
            f"det cache stride={stride}: raw clear_R under-counts when odd frames were never detected. "
            "Primary metrics use detect ticks; carry_neighbor approximates 60fps hold from 30Hz detect."
        ),
        "seed_note": (labels.get("seed") or {}).get("note"),
    }


def score_proxy_packs() -> dict:
    packs = {
        "random": ROOT / "reports/eval_match3/pitchmap_gallery/det_cache",
        "quad": ROOT / "reports/eval_match3/quad_pitchmap_gallery/det_cache",
    }
    out = {}
    for name, folder in packs.items():
        if not folder.is_dir():
            continue
        rows = [score_cache(p) for p in sorted(folder.glob("det_cache_*_thr010.json"))]
        tot = {k: 0 for k in ["n", "emit", "agree", "mapped_ge2", "clear_frames", "clear_emit"]}
        for r in rows:
            for k in tot:
                tot[k] += r[k]
        out[name] = {
            "totals": tot,
            "clear_ball_proxy_R": None
            if tot["clear_frames"] == 0
            else round(tot["clear_emit"] / tot["clear_frames"], 3),
            "agree_among_emit": None if tot["emit"] == 0 else round(tot["agree"] / tot["emit"], 3),
            "per_cache": rows,
        }
    return out


def main() -> int:
    out = {
        "goals": {
            "P_emit": 0.80,
            "clear_ball_R": 0.80,
            "emit_conf": EMIT_CONF,
            "min_support": MIN_SUPPORT,
            "hit_m": HIT_M,
        },
        "packs": score_proxy_packs(),
    }
    if STRIP.is_file():
        out["strip"] = score_strip(STRIP)
        out["note"] = (
            "strip P_emit/clear_ball_R are provisional until human-corrected labels; "
            "proxy packs remain coverage-only."
        )
    else:
        out["note"] = (
            "Proxy only — build strip via scripts/gold_set/build_match3_m1_strip.py"
        )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    if "strip" in out:
        s = out["strip"]
        print(
            f"strip: P_emit={s['P_emit']} clear_ball_R={s['clear_ball_R']} "
            f"(stride={s.get('det_cache_stride')}) "
            f"emit={s['n_emit_scored']} clear={s['n_clear']} "
            f"err_med={s['err_median_m']}m pass_P={s['poc_pass_P_emit']} "
            f"pass_R={s['poc_pass_clear_R']}"
        )
        modes = s.get("modes") or {}
        if modes:
            raw = modes.get("raw_all_label_frames") or {}
            carry = modes.get("carry_neighbor_tick") or {}
            print(
                f"  raw_R={raw.get('clear_ball_R')} "
                f"carry_R={carry.get('clear_ball_R')} "
                f"note={s.get('note')}"
            )
        out["note"] = s.get("note") or out.get("note")

    for name, pack in out["packs"].items():
        print(
            f"{name}: clear_ball_proxy_R={pack['clear_ball_proxy_R']} "
            f"agree_among_emit={pack['agree_among_emit']} "
            f"emit={pack['totals']['emit']}/{pack['totals']['n']}"
        )
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
