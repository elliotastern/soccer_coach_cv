#!/usr/bin/env python3
"""Survey 4quad windows: who wins max_conf under baseline vs Top Left lock.

Detect once per slot (P-cams + Cam4plus/Cam5plus), no gold required. Answers:
does the Top Left P7≥0.60 rule change selection on other pitch regions?
Never trains. Writes reports/eval_match2_v10/4quad_multicam_survey/.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "gold_set"))

from eval_match2_top_left_multicam_baseline import (  # noqa: E402
    DETECT_H,
    DETECT_W,
    cache_load,
    dets_to_rows,
    filter_rows,
    read_resized,
)
from eval_match2_v10_video_system import pick_selected  # noqa: E402
from multicam_select_policy import (  # noqa: E402
    BASELINE_THR,
    QUAD_SLOTS,
    SURVEY_CAMS,
    TOP_LEFT_POLICY_ID,
    TOP_LEFT_THR_BY_CAM,
    thr_for_cam,
)
from src.perception.ball_prelabel import BallPrelabelConfig, BallPrelabeler  # noqa: E402
from src.perception.rfdetr_local import load_ball_model  # noqa: E402

CKPT = ROOT / "models/v10_snaps/post_train/checkpoint.pth"
SOURCE_DIR = ROOT / "reports/eval_match2_v10/4quad_test/source"
OUT = ROOT / "reports/eval_match2_v10/4quad_multicam_survey"
SIZE = dict(use_size_filter=True, min_side=4, max_side=240, use_kalman=False)
TOP_LEFT_CACHE = (
    ROOT / "reports/eval_match2_v10/top_left_multicam_baseline/det_cache_thr010.json"
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ball-checkpoint", type=Path, default=CKPT)
    p.add_argument("--source-dir", type=Path, default=SOURCE_DIR)
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--detect-thr", type=float, default=0.10)
    p.add_argument("--skip-detect", action="store_true")
    p.add_argument(
        "--slots",
        nargs="*",
        default=[s["slot"] for s in QUAD_SLOTS],
        help="Subset of slots to run",
    )
    return p.parse_args()


def source_path(source_dir: Path, stem: str, cam: str) -> Path:
    return source_dir / f"{stem}_{cam}.mp4"


def cache_dump_n(path: Path, dets: dict, n_frames: int):
    serial = {
        cam: [[[list(box), conf, side] for box, conf, side in rows] for rows in frames]
        for cam, frames in dets.items()
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"n_frames": n_frames, "cams": serial}), encoding="utf-8"
    )


def run_detect_cams(
    model, source_dir: Path, stem: str, n_frames: int, thr: float, cams: list
) -> dict:
    pre = BallPrelabeler(
        model,
        BallPrelabelConfig(threshold=thr, use_sahi=False, topk=5, **SIZE),
        class_id=1,
    )
    caps = {}
    for cam in cams:
        path = source_path(source_dir, stem, cam)
        if not path.is_file():
            raise FileNotFoundError(path)
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"open failed {path}")
        caps[cam] = cap
    out = {cam: [] for cam in cams}
    try:
        for i in range(n_frames):
            for cam in cams:
                frame = read_resized(caps[cam])
                if frame is None:
                    # Synced clips can differ by 1 frame; stop at shortest.
                    if i == 0:
                        raise RuntimeError(f"{cam} empty ({stem})")
                    print(
                        f"  {stem} short cam={cam} at frame {i}; trimming to {i}",
                        flush=True,
                    )
                    for c in cams:
                        out[c] = out[c][:i]
                    return out
                if frame.shape[1] != DETECT_W or frame.shape[0] != DETECT_H:
                    frame = cv2.resize(
                        frame, (DETECT_W, DETECT_H), interpolation=cv2.INTER_AREA
                    )
                out[cam].append(dets_to_rows(pre.detect_bgr(frame)))
            if i % 50 == 0:
                print(f"  {stem} detect {i}/{n_frames - 1} cams={cams}", flush=True)
    finally:
        for cap in caps.values():
            cap.release()
    return out


def pad_missing_cams(dets: dict, n_frames: int) -> dict:
    out = dict(dets)
    for cam in SURVEY_CAMS:
        if cam not in out:
            out[cam] = [[] for _ in range(n_frames)]
        else:
            out[cam] = out[cam][:n_frames]
            if len(out[cam]) < n_frames:
                out[cam] = out[cam] + [[] for _ in range(n_frames - len(out[cam]))]
    return out


def select_share(dets: dict, n_frames: int, thr_by_cam: dict) -> dict:
    counts = Counter()
    n_selected = 0
    for i in range(n_frames):
        active = {}
        for cam in SURVEY_CAMS:
            rows = filter_rows(dets[cam][i], thr_for_cam(thr_by_cam, cam))
            if rows:
                active[cam] = rows
        if not active:
            counts["none"] += 1
            continue
        cam, _pred = pick_selected(active, "max_conf")
        if cam is None:
            counts["none"] += 1
            continue
        n_selected += 1
        counts[cam] += 1
    return {
        "n_frames": n_frames,
        "n_selected": n_selected,
        "selection_share": {k: v / n_frames for k, v in sorted(counts.items())},
        "selection_counts": dict(counts),
        "thr_by_cam": thr_by_cam,
    }


def policies_for_slot() -> list[tuple[str, dict]]:
    return [
        ("max_conf_030", {"_default": 0.30}),
        ("max_conf_060", {"_default": 0.60}),
        (TOP_LEFT_POLICY_ID, dict(TOP_LEFT_THR_BY_CAM)),
        ("cam4_thr060_others030", {"_default": 0.30, "Cam4plus": 0.60}),
        ("cam5_thr060_others030", {"_default": 0.30, "Cam5plus": 0.60}),
    ]


def load_or_detect_slot(args, model, spec: dict):
    slot, stem, n = spec["slot"], spec["stem"], spec["n_frames"]
    cache = args.out / f"det_cache_{slot}_thr010.json"
    if args.skip_detect:
        if not cache.is_file():
            raise FileNotFoundError(cache)
        return pad_missing_cams(cache_load(cache), n), model

    if cache.is_file():
        print(f"loaded existing {cache}", flush=True)
        return pad_missing_cams(cache_load(cache), n), model

    if model is None:
        model = load_ball_model(str(args.ball_checkpoint))

    if slot == "top_left" and TOP_LEFT_CACHE.is_file():
        dets = pad_missing_cams(cache_load(TOP_LEFT_CACHE), n)
        print(f"seeded P-cams from {TOP_LEFT_CACHE}", flush=True)
        need = [c for c in ("Cam4plus", "Cam5plus") if not any(dets[c])]
        if need:
            print(f"detecting only {need}…", flush=True)
            fill = run_detect_cams(
                model, args.source_dir, stem, n, args.detect_thr, need
            )
            for cam in need:
                dets[cam] = fill[cam]
    else:
        dets = run_detect_cams(
            model, args.source_dir, stem, n, args.detect_thr, SURVEY_CAMS
        )
    cache_dump_n(cache, {c: dets[c] for c in SURVEY_CAMS}, n)
    print(f"wrote {cache}", flush=True)
    return dets, model


def main() -> int:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    slots = [s for s in QUAD_SLOTS if s["slot"] in args.slots]
    if not slots:
        raise SystemExit("no slots selected")

    model = None
    results = []
    for spec in slots:
        print(f"=== {spec['label']} ({spec['slot']}) ===", flush=True)
        dets, model = load_or_detect_slot(args, model, spec)
        n = spec["n_frames"]
        slot_row = {
            "slot": spec["slot"],
            "label": spec["label"],
            "stem": spec["stem"],
            "policies": {},
        }
        for pid, thr_map in policies_for_slot():
            scored = select_share(dets, n, thr_map)
            slot_row["policies"][pid] = scored
            top = sorted(
                ((c, v) for c, v in scored["selection_share"].items() if c != "none"),
                key=lambda x: -x[1],
            )[:3]
            top_s = ", ".join(f"{c}={p:.0%}" for c, p in top) or "none"
            print(
                f"  {pid}: selected={scored['n_selected']}/{n} top[{top_s}]",
                flush=True,
            )
        results.append(slot_row)

    payload = {
        "title": "4quad_multicam_survey",
        "cams": SURVEY_CAMS,
        "top_left_lock": TOP_LEFT_POLICY_ID,
        "top_left_thr": TOP_LEFT_THR_BY_CAM,
        "slots": results,
    }
    (args.out / "survey.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# 4quad multicam survey — selection by pitch region",
        "",
        f"Cams: `{', '.join(SURVEY_CAMS)}`. No gold R/P here — **who wins** under thr rules.",
        f"Top Left lock under test: `{TOP_LEFT_POLICY_ID}` ({TOP_LEFT_THR_BY_CAM}).",
        "",
    ]
    for slot_row in results:
        lines += [f"## {slot_row['label']} (`{slot_row['slot']}`)", ""]
        lines += [
            "| policy | n_selected | top cams (share) |",
            "|---|---:|---|",
        ]
        for pid, scored in slot_row["policies"].items():
            top = sorted(
                ((c, v) for c, v in scored["selection_share"].items() if c != "none"),
                key=lambda x: -x[1],
            )[:4]
            top_s = ", ".join(f"{c} {p:.0%}" for c, p in top) or "—"
            lines.append(
                f"| `{pid}` | {scored['n_selected']}/{scored['n_frames']} | {top_s} |"
            )
        lines.append("")
    lines += [
        "## Read",
        "",
        "- If another slot’s winner is **Cam4plus/Cam5plus**, Top Left’s P7 floor does not transfer.",
        "- Next gold pack should target the **#1 selected cam** on that slot (min labels).",
        "- Only claim match-wide 80/90 after ≥2 regions have dual-cam gold + thr lock.",
        "",
    ]
    md = args.out / "survey.md"
    md.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {md}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
