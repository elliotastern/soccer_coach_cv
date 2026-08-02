#!/usr/bin/env python3
"""Build Match train labeling pack (batch3) from Gold FN audit.

FN audit (Gold 0-49, v6_epoch_100 @0.5): R=0.095; median FN ball ~13px;
33/35 FN frames had no pred >=0.5. Gold 0-49 is TEST — never train on it.

This pack excludes Gold100 + math_1_training + batch2, and oversamples
tiny-ball / low-conf candidates for the next human gold labels.
Prelabels only — promote to gold/ after human correction.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from annotation.cvat_xml_generator import create_cvat_xml  # noqa: E402
from scripts.gold_set.build_fresh_cvat100 import load_exclude_keys  # noqa: E402
from scripts.gold_set.build_match_gold100 import (  # noqa: E402
    DEFAULT_MATCH_DIR,
    discover_videos,
    read_frame,
    require_dir,
    require_file,
    resize_for_strip,
    sample_match_candidates,
    to_ann_tracked,
    write_coco,
    write_strip_video,
)
from src.perception.rfdetr_local import LocalRFDETRDetector  # noqa: E402
from src.state.types import Detection  # noqa: E402

DEFAULT_OUT = ROOT / "data/processed/gold_sets/math_1_training_batch3"

def run_with_timeout(fn, args=(), timeout_s=45):
    import multiprocessing as mp
    q = mp.Queue()
    def _wrap(q, args):
        try:
            q.put(("ok", fn(*args)))
        except Exception as e:
            q.put(("err", repr(e)))
    proc = mp.Process(target=_wrap, args=(q, args))
    proc.start()
    proc.join(timeout_s)
    if proc.is_alive():
        proc.terminate()
        proc.join(5)
        raise TimeoutError(f"timeout after {timeout_s}s")
    if q.empty():
        raise RuntimeError("worker exited without result")
    status, payload = q.get()
    if status == "err":
        raise RuntimeError(payload)
    return payload

DEFAULT_EXCLUDES = [
    ROOT / "data/processed/gold_sets/match1_1_100/manifest.json",
    ROOT / "data/processed/gold_sets/math_1_training/manifest.json",
    ROOT / "data/processed/gold_sets/math_1_training_batch2/manifest.json",
]


def merge_excludes(paths: list[Path]) -> set[tuple[str, int]]:
    keys: set[tuple[str, int]] = set()
    for p in paths:
        keys |= load_exclude_keys(p)
        print(f"exclude {p.name}: total_keys={len(keys)}")
    return keys


def ball_area(det: Detection) -> float:
    _, _, w, h = det.bbox
    return float(max(w, 0.0) * max(h, 0.0))


def pick_hard(
    scored: list[dict],
    rng: random.Random,
    n: int,
    exclude: set[tuple[str, int]],
) -> list[dict]:
    pool = []
    for row in scored:
        key_rel = (row["video_rel"], int(row["frame_idx"]))
        key_path = (row["video_path"], int(row["frame_idx"]))
        if key_rel in exclude or key_path in exclude:
            continue
        pool.append(row)
    if len(pool) < n:
        raise RuntimeError(
            f"Only {len(pool)} frames after exclude; need {n}. Increase --candidates."
        )

    uncertain = []
    tiny = []
    miss_like = []
    other = []
    for row in pool:
        conf = float(row["max_ball_conf"])
        area = float(row["min_ball_area"])
        # Prefer tiny first (FN audit: median miss ~13px)
        if row["n_ball"] > 0 and area > 0 and area < 400:
            tiny.append(row)
        elif row["n_ball"] == 0:
            miss_like.append(row)
        elif 0.15 <= conf < 0.80:
            uncertain.append(row)
        else:
            other.append(row)

    rng.shuffle(uncertain)
    rng.shuffle(tiny)
    rng.shuffle(miss_like)
    rng.shuffle(other)

    # FN audit: most misses are tiny (<20px). Oversample tiny + uncertain.
    quotas = [
        (tiny, 50, "tiny_ball"),
        (uncertain, 30, "uncertain_ball"),
        (miss_like, 15, "miss_like"),
        (other, 5, "fill"),
    ]
    selected = []
    used = set()
    for bucket, quota, tag in quotas:
        take = 0
        for row in bucket:
            key = (row["video_rel"], int(row["frame_idx"]))
            if key in used:
                continue
            row = dict(row)
            row["stratum"] = tag
            selected.append(row)
            used.add(key)
            take += 1
            if take >= quota or len(selected) >= n:
                break
        if len(selected) >= n:
            break

    if len(selected) < n:
        for row in pool:
            key = (row["video_rel"], int(row["frame_idx"]))
            if key in used:
                continue
            row = dict(row)
            row["stratum"] = "fill"
            selected.append(row)
            used.add(key)
            if len(selected) >= n:
                break

    selected = selected[:n]
    selected.sort(key=lambda r: (r["camera"], r["video_rel"], r["frame_idx"]))
    return selected


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--match-dir", type=Path, default=DEFAULT_MATCH_DIR)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument(
        "--exclude-manifest",
        type=Path,
        action="append",
        default=None,
        help="Repeatable. Defaults to Gold100 + math_1_training manifests.",
    )
    p.add_argument(
        "--player-checkpoint",
        type=Path,
        default=ROOT / "models/people_after_100_epochs.pth",
    )
    p.add_argument(
        "--ball-checkpoint",
        type=Path,
        default=ROOT / "models/ball_89.pth",
    )
    p.add_argument("--seed", type=int, default=20260801)
    p.add_argument("--candidates", type=int, default=360)
    p.add_argument("--n-frames", type=int, default=100)
    p.add_argument("--detect-threshold", type=float, default=0.2)
    p.add_argument(
        "--pack-from-scores",
        action="store_true",
        help="Skip scoring; build pack from existing candidate_scores.json only.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    rng = random.Random(args.seed)
    match_dir = require_dir(args.match_dir, "Match directory")
    player_ckpt = require_file(args.player_checkpoint, "Player checkpoint")
    ball_ckpt = require_file(args.ball_checkpoint, "Ball checkpoint")
    excl_paths = args.exclude_manifest or DEFAULT_EXCLUDES
    exclude = merge_excludes(excl_paths)

    videos = discover_videos(match_dir)
    print(f"Found {len(videos)} videos")
    out_dir = args.output_dir
    images_dir = out_dir / "images"
    prelabels_dir = out_dir / "prelabels"
    review_dir = out_dir / "review"
    for d in (images_dir, prelabels_dir, review_dir):
        d.mkdir(parents=True, exist_ok=True)

    score_path = out_dir / "candidate_scores.json"
    scored = []
    if score_path.exists():
        try:
            scored = json.loads(score_path.read_text())
            print(f"Loaded scores: {len(scored)}", flush=True)
        except Exception as e:
            print(f"Could not load scores: {e}", flush=True)
            scored = []

    if not args.pack_from_scores:
        candidates = sample_match_candidates(videos, args.candidates, rng)
        print(f"Candidates: {len(candidates)}", flush=True)

        # Ball-only scoring (lighter on MPS). Full dual detect only on selected.
        from src.perception.rfdetr_local import load_ball_model

        print("Loading ball model for scoring (CPU)...", flush=True)
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        import torch
        # Disable MPS so rfdetr/predict stays on CPU (avoids MPS/CPU mismatch + hangs)
        try:
            torch.backends.mps.is_available = lambda: False  # type: ignore
        except Exception:
            pass
        ball_model = load_ball_model(str(ball_ckpt))
        try:
            if hasattr(ball_model, "model") and hasattr(ball_model.model, "model"):
                ball_model.model.model.to("cpu")
            elif hasattr(ball_model, "model"):
                ball_model.model.to("cpu")
            print("Ball scoring model on CPU", flush=True)
        except Exception as e:
            print(f"CPU move skipped: {e}", flush=True)
        done_keys = {(r["video_rel"], int(r["frame_idx"])) for r in scored}
        for i, row in enumerate(candidates):
            key = (row["video_rel"], int(row["frame_idx"]))
            if key in done_keys:
                continue
            print(f"  score {len(scored)+1}: {key}", flush=True)
            try:
                import signal
                def _timeout_handler(signum, frame):
                    raise TimeoutError("frame timeout")
                signal.signal(signal.SIGALRM, _timeout_handler)
                signal.alarm(60)
                frame = read_frame(row["video_path"], row["frame_idx"], fps=row["fps"])
                # Half-res scoring for speed/memory; scale areas back to full-res
                h, w = frame.shape[:2]
                small = cv2.resize(frame, (max(1, w // 2), max(1, h // 2)))
                rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                from PIL import Image

                dets = ball_model.predict(
                    Image.fromarray(rgb), threshold=args.detect_threshold
                )
                balls = []
                if dets is not None and len(dets) > 0:
                    for j in range(len(dets)):
                        x1, y1, x2, y2 = map(float, dets.xyxy[j])
                        # scale box back to full-res pixels
                        bw, bh = (x2 - x1) * 2.0, (y2 - y1) * 2.0
                        balls.append(
                            type("B", (), {"confidence": float(dets.confidence[j]),
                                           "bbox": (0.0, 0.0, bw, bh)})()
                        )
                scored.append({
                    **{k: row[k] for k in row if k != "detections"},
                    "n_ball": len(balls),
                    "n_player": 0,
                    "max_ball_conf": max((b.confidence for b in balls), default=0.0),
                    "min_ball_area": min((ball_area(b) for b in balls), default=0.0),
                })
                signal.alarm(0)
            except Exception as e:
                try:
                    import signal as _signal
                    _signal.alarm(0)
                except Exception:
                    pass
                print(f"  SKIP {key}: {e}", flush=True)
                scored.append({
                    **{k: row[k] for k in row if k != "detections"},
                    "n_ball": 0,
                    "n_player": 0,
                    "max_ball_conf": 0.0,
                    "min_ball_area": 0.0,
                    "error": str(e),
                })
            if len(scored) % 5 == 0 or len(scored) == len(candidates):
                score_path.write_text(json.dumps(scored))
                print(f"  Detected {len(scored)}/{len(candidates)}", flush=True)

        if len(scored) < len(candidates):
            score_path.write_text(json.dumps(scored))
            raise SystemExit(
                f"PARTIAL_SCORES {len(scored)}/{len(candidates)} — rerun to resume"
            )
    else:
        if len(scored) < 20:
            raise SystemExit(f"Need more scores for --pack-from-scores; have {len(scored)}")
        print(f"Packing from partial scores only ({len(scored)})", flush=True)

    n_frames = min(args.n_frames, max(20, len(scored) - 1))
    selected = pick_hard(scored, rng, n_frames, exclude)
    strata = {}
    for r in selected:
        strata[r["stratum"]] = strata.get(r["stratum"], 0) + 1
    print(
        f"Selected {len(selected)} strata={strata} "
        f"with_ball={sum(1 for r in selected if r['n_ball'] > 0)}",
        flush=True,
    )

    # Pack-from-scores: skip dual MPS detect (crashes). Empty prelabels — human gold.
    # Full scoring path still can dual-detect when not --pack-from-scores.
    use_empty_prelabels = bool(args.pack_from_scores)
    detector = None
    if not use_empty_prelabels:
        print("Loading dual detector for selected-frame prelabels...", flush=True)
        detector = LocalRFDETRDetector(
            player_checkpoint=str(player_ckpt),
            ball_checkpoint=str(ball_ckpt),
            confidence_threshold=args.detect_threshold,
            player_class_id=0,
            ball_class_id=1,
            enhance_ball=True,
        )
    else:
        print("Empty prelabels mode (label in editor; precision-first)", flush=True)

    image_names = []
    strip_frames = []
    tracked_by_frame = {}
    manifest_rows = []
    next_track_id = 1
    scale_notes = []

    packed_rows = []
    for row in selected:
        idx = len(packed_rows) + 1
        print(
            f"  pack {idx}/{len(selected)} {row['video_rel']}@{row['frame_idx']}",
            flush=True,
        )
        cam_slug = row["camera"].replace(" ", "")
        name = f"{cam_slug}_f{row['frame_idx']:06d}.jpg"
        img_path = images_dir / name
        try:
            if img_path.exists():
                frame = cv2.imread(str(img_path))
                if frame is None:
                    raise RuntimeError(f"bad image {img_path}")
            else:
                frame = read_frame(row["video_path"], row["frame_idx"], fps=row["fps"])
                cv2.imwrite(str(img_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            if detector is not None:
                detections = detector.detect(frame)
            else:
                detections = []
        except Exception as e:
            print(f"  SKIP pack frame: {e}", flush=True)
            continue
        row = dict(row)
        row["detections"] = detections
        h, w = frame.shape[:2]
        row["width"], row["height"] = w, h
        strip_i = len(packed_rows)
        image_names.append(name)
        strip_frames.append(frame)

        strip = resize_for_strip(frame)
        sh, sw = strip.shape[:2]
        sx, sy = sw / w, sh / h
        scaled_dets = []
        for det in detections:
            x, y, bw, bh = det.bbox
            scaled_dets.append(Detection(
                class_id=det.class_id,
                confidence=det.confidence,
                bbox=(x * sx, y * sy, bw * sx, bh * sy),
                class_name=det.class_name,
            ))
        tracked = to_ann_tracked(scaled_dets, next_track_id)
        next_track_id += max(len(tracked), 1)
        tracked_by_frame[strip_i] = tracked
        scale_notes.append({"strip_frame": strip_i, "sx": sx, "sy": sy})
        manifest_rows.append({
            "strip_frame": strip_i,
            "camera": row["camera"],
            "video_rel": row["video_rel"],
            "video_path": row["video_path"],
            "frame_idx": row["frame_idx"],
            "image": name,
            "width": w,
            "height": h,
            "stratum": row["stratum"],
            "n_ball": row["n_ball"],
            "n_player": row["n_player"],
            "max_ball_conf": row["max_ball_conf"],
            "min_ball_area": row["min_ball_area"],
        })
        packed_rows.append(row)

    if len(packed_rows) < 20:
        raise SystemExit(f"Only packed {len(packed_rows)} frames; need >=20")
    selected = packed_rows
    strata = {}
    for r in selected:
        strata[r["stratum"]] = strata.get(r["stratum"], 0) + 1

    # write_coco expects detections on rows; empty list is fine
    write_coco(selected, image_names, prelabels_dir / "annotations.coco.json")
    strip_path = review_dir / "strip_100.mp4"
    strip_w, strip_h = write_strip_video(strip_frames, strip_path, fps=60)
    xml = create_cvat_xml(
        video_path=str(strip_path),
        tracked_objects_by_frame=tracked_by_frame,
        events=[],
        video_metadata={
            "width": strip_w,
            "height": strip_h,
            "fps": 60.0,
            "frame_count": len(selected),
        },
    )
    (prelabels_dir / "annotations.xml").write_text(xml, encoding="utf-8")

    manifest = {
        "match_dir": str(match_dir),
        "seed": args.seed,
        "detect_threshold": args.detect_threshold,
        "enhance_ball": True,
        "n_frames": len(selected),
        "strip_size": [strip_w, strip_h],
        "sampling": "fn_audit_partial_scores_pack_from_scores",
        "fn_audit_ref": "reports/fn_audit_gold049_v6_epoch100.txt",
        "exclude_manifests": [str(p) for p in excl_paths],
        "strata": strata,
        "camera_counts": {
            cam: sum(1 for r in manifest_rows if r["camera"] == cam)
            for cam in sorted({r["camera"] for r in manifest_rows})
        },
        "ball_frame_count": sum(1 for r in manifest_rows if r["n_ball"] > 0),
        "scale_to_strip": scale_notes,
        "frames": manifest_rows,
        "note": "Prelabels only. Correct then promote to gold/ before training.",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (out_dir / "README.md").write_text(
        "# math_1_training_batch3\n\n"
        "Tiny-ball-heavy Match frames from Gold 0-49 FN audit patterns.\n"
        "Excludes Gold100 + math_1_training + batch2. "
        "**Never train on Gold 0-49 test frames.**\n\n"
        "**Do not train from prelabels/**. Correct in the editor, promote to "
        "`gold/annotations.xml`, then `export_gold_coco.py`.\n\n"
        "See `docs/ball_detection/TRAIN_LABEL_SOURCE_OF_TRUTH.md`.\n"
    )

    from scripts.gold_set.harden_review_pack import (
        build_image_editor,
        clamp_xml_boxes,
        validate_pack,
        write_review_frames,
        write_strip_from_frames,
    )

    names, rw, rh = write_review_frames(out_dir.resolve())
    clamp_xml_boxes(out_dir.resolve(), rw, rh)
    write_strip_from_frames(out_dir.resolve(), len(names))
    build_image_editor(out_dir.resolve(), names, rw, rh)
    manifest["strip_size"] = [rw, rh]
    manifest["review_mode"] = "image_sequence"
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    validate_pack(out_dir.resolve())

    print("Done.")
    print(f"Output: {out_dir}")
    print(f"Cameras: {manifest['camera_counts']}")
    print(f"Strata: {strata}")
    print(f"Frames with ball prelabel: {manifest['ball_frame_count']}")


if __name__ == "__main__":
    main()
