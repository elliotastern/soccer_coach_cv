#!/usr/bin/env python3
"""Compare player fuse with vs without F4 reprojection prune (default off)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.gold_set.player_map_funnel import CAMS, DEFAULT_FRAMES, funnel_frame  # noqa: E402
from src.perception.rfdetr_local import LocalRFDETRDetector  # noqa: E402
from src.review.cam_mosaic import _ensure_cam_dets, match3_videos  # noqa: E402
from src.review.frame_sync import keep_top1_ball  # noqa: E402
from src.review.multicam_fuse import fuse_live_dets_for_pitch  # noqa: E402

OUT = ROOT / "reports/eval_match3/improve_eng_loop/players_pitch/player_reproj_ab.json"


def main() -> int:
    vids = match3_videos(ROOT)
    det = LocalRFDETRDetector(
        player_checkpoint=str(ROOT / "models/people_after_100_epochs.pth"),
        ball_checkpoint=str(ROOT / "models/v12_hard_snaps/post_train/checkpoint.pth"),
        confidence_threshold=0.15,
        enhance_ball=False,
        use_sahi=False,
        use_kalman=False,
        player_nms_iou=0.30,
        ball_nms_iou=0.4,
    )

    def detect_fn(cam, frame_bgr):
        return keep_top1_ball(det.detect(frame_bgr))

    rows = []
    for fr in DEFAULT_FRAMES:
        bag = {}
        for cam in CAMS:
            _ensure_cam_dets(vids, cam, fr, bag, detect_fn, True)
        off = fuse_live_dets_for_pitch(bag, apply_undistort=False, reproj_prune_players=False)
        on = fuse_live_dets_for_pitch(bag, apply_undistort=False, reproj_prune_players=True)
        rows.append(
            {
                "frame_id": fr,
                "n_off": len(off["players"]),
                "n_on": len(on["players"]),
                "consensus_off": off.get("consensus"),
                "consensus_on": on.get("consensus"),
            }
        )
        print(f"fr {fr}: players off={len(off['players'])} on={len(on['players'])}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"frames": rows}, indent=2), encoding="utf-8")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
