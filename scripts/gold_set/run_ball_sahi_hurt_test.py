#!/usr/bin/env python3
"""P10 Top Left gallery: 10 SAHI combos most likely to hurt precision/recall.

Same window as ball_postprocessing_test (0:26–0:31, P10). Ordered by how
likely they are to add FPs or stick junk (recover-always, loose thr, no size
filter, dense tiles, TTA/multiscale/Kalman/ByteTrack piled on). Not a gold
rescore. Writes reports/eval_match2_v10/ball_sahi_hurt_test/.
Never trains.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "gold_set"))

from run_ball_postprocessing_test import (  # noqa: E402
    CKPT,
    END_CLOCK,
    OVERLAY_WIDTH,
    SIZE,
    START_CLOCK,
    ensure_source,
    parse_clock,
    run_variant,
    write_html,
    write_summary,
)
from src.perception.ball_prelabel import BallPrelabelConfig  # noqa: E402
from src.perception.rfdetr_local import load_ball_model  # noqa: E402

OUT_DEFAULT = ROOT / "reports/eval_match2_v10/ball_sahi_hurt_test"
NOSIZE = dict(use_size_filter=False, min_side=4, max_side=240)


def sahi_always(**extra) -> BallPrelabelConfig:
    base = dict(
        threshold=0.30,
        use_sahi=True,
        sahi_fallback_only=False,
        sahi_recover_only=True,
        topk=2,
        use_kalman=False,
        **SIZE,
    )
    base.update(extra)
    return BallPrelabelConfig(**base)


def variant_specs() -> list[dict]:
    """Most likely to hurt first (measured avoid + aggressive SAHI piles)."""
    return [
        {
            "id": "sahi_recover_always",
            "title": "1. SAHI recover-always",
            "why": "Measured hurt on train/gold — extra FPs; gold P_emit dropped.",
            "mode": "detect",
            "cfg": sahi_always(),
        },
        {
            "id": "sahi_always_topk3",
            "title": "2. SAHI always + topk=3",
            "why": "Always tiles then keep 3 boxes — more room for tile junk.",
            "mode": "detect",
            "cfg": sahi_always(topk=3),
        },
        {
            "id": "sahi_always_topk5",
            "title": "3. SAHI always + topk=5",
            "why": "Even looser topk — second/third FPs more likely to survive.",
            "mode": "detect",
            "cfg": sahi_always(topk=5),
        },
        {
            "id": "sahi_always_thr20",
            "title": "4. SAHI always + thr 0.20",
            "why": "Lower detect floor on fullframe and tiles — weak false balls.",
            "mode": "detect",
            "cfg": sahi_always(threshold=0.20),
        },
        {
            "id": "sahi_always_nosize",
            "title": "5. SAHI always, no size filter",
            "why": "Drops geometry gate — huge/tiny junk from tiles can pass.",
            "mode": "detect",
            "cfg": BallPrelabelConfig(
                threshold=0.30,
                use_sahi=True,
                sahi_fallback_only=False,
                sahi_recover_only=True,
                topk=2,
                use_kalman=False,
                **NOSIZE,
            ),
        },
        {
            "id": "sahi_always_multiscale",
            "title": "6. SAHI always + multiscale 1.5×",
            "why": "Two recover paths at once — train already showed multiscale +FPs.",
            "mode": "detect",
            "cfg": sahi_always(use_multiscale=True),
        },
        {
            "id": "sahi_always_tta",
            "title": "7. SAHI always + HFlip TTA",
            "why": "TTA + always-tile: most aggressive recover; FP risk stacks.",
            "mode": "tta",
            "cfg": sahi_always(topk=99),
        },
        {
            "id": "sahi_always_kalman",
            "title": "8. SAHI always + Kalman",
            "why": "Kalman coasts wrong tile dets across frames (gold R/P both hurt).",
            "mode": "detect",
            "cfg": sahi_always(use_kalman=True),
        },
        {
            "id": "sahi_always_bt_sticky",
            "title": "9. SAHI always + ByteTrack (no emit gate)",
            "why": "Sticky tracks on tile FPs — junk can linger without 0.80 gate.",
            "mode": "bytetrack",
            "emit_gate": False,
            "match_thresh": 0.8,
            "cfg": sahi_always(),
        },
        {
            "id": "sahi_dense_tiles",
            "title": "10. Dense SAHI tiles (640 / 40% overlap) + topk=3",
            "why": "More/smaller tiles → more false peaks; keep three of them.",
            "mode": "detect",
            "cfg": sahi_always(slice_size=640, overlap=0.4, topk=3),
        },
    ]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ball-checkpoint", type=Path, default=CKPT)
    p.add_argument("--out", type=Path, default=OUT_DEFAULT)
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--overlay-width", type=int, default=OVERLAY_WIDTH)
    p.add_argument("--skip-extract", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "overlay").mkdir(parents=True, exist_ok=True)
    src = ensure_source(out, args.skip_extract)
    print(f"source {src}", flush=True)
    if not args.ball_checkpoint.is_file():
        raise FileNotFoundError(f"missing checkpoint {args.ball_checkpoint}")
    model = load_ball_model(str(args.ball_checkpoint))
    specs = variant_specs()
    if len(specs) != 10:
        raise RuntimeError(f"expected 10 variants, got {len(specs)}")
    variants = []
    for spec in specs:
        ov = out / "overlay" / f"{spec['id']}.mp4"
        print(f"variant {spec['id']} → {ov}", flush=True)
        variants.append(
            run_variant(model, src, ov, spec, args.stride, args.overlay_width)
        )
    payload = {
        "title": "ball_sahi_hurt_test",
        "clock": f"{START_CLOCK}–{END_CLOCK}",
        "start_sec": parse_clock(START_CLOCK),
        "end_sec": parse_clock(END_CLOCK),
        "camera": "P10",
        "source": str(src),
        "checkpoint": str(args.ball_checkpoint),
        "stride": args.stride,
        "ranking_note": "SAHI combos most likely to hurt (theory + measured recover-always)",
        "page_note": (
            "These are the SAHI stacks most likely to hurt — recover-always, loose thr, "
            "no size filter, dense tiles, and SAHI piled with TTA/multiscale/Kalman/ByteTrack. "
            "Same P10 Top Left clip as ball_postprocessing_test. Not a gold rescore."
        ),
        "variants": variants,
    }
    summary = write_summary(out, payload)
    write_html(out, payload)
    print(f"wrote {summary}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
