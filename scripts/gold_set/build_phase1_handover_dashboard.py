#!/usr/bin/env python3
"""Sync Phase 1 handover clip + meta into phase1_handover/ for coach marking UI."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ENG = ROOT / "reports/eval_match3/improve_eng_loop"
SRC = ENG / "player_map/check_15s_s4"
OUT = ENG / "phase1_handover"

COPY_FILES = (
    "coach_mosaic_pitch_min.mp4",
    "meta.json",
    "emits_render.json",
    "still_first.jpg",
    "still_mid.jpg",
    "still_last.jpg",
)


def h264_ok(mp4: Path) -> bool:
    probe = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=codec_name", "-of", "csv=p=0", str(mp4),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return probe.stdout.strip() == "h264"


def ensure_h264(mp4: Path) -> None:
    if h264_ok(mp4):
        return
    tmp = mp4.with_suffix(".h264.mp4")
    subprocess.run(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(mp4), "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", str(tmp),
        ],
        check=True,
    )
    tmp.replace(mp4)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for name in COPY_FILES:
        src = SRC / name
        if not src.is_file():
            raise FileNotFoundError(f"missing {src}")
        shutil.copy2(src, OUT / name)
        print(f"COPY {name}", flush=True)
    mp4 = OUT / "coach_mosaic_pitch_min.mp4"
    ensure_h264(mp4)
    labels = OUT / "labels.json"
    if not labels.is_file():
        labels.write_text(
            json.dumps(
                {
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                    "reviewer": "",
                    "video": "coach_mosaic_pitch_min.mp4",
                    "source_dir": str(SRC.relative_to(ROOT)),
                    "frames": {},
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print("INIT labels.json", flush=True)
    sys.path.insert(0, str(ROOT))
    from scripts.gold_set.merge_handover_fuse_gold import seed_handover_suggestions  # noqa: E402

    seed_handover_suggestions(OUT)
    from scripts.gold_set.handover_dashboard_html import write_index  # noqa: E402

    html_path = write_index(OUT)
    print("WROTE", html_path, flush=True)
    info = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "video": "coach_mosaic_pitch_min.mp4",
        "url": "/phase1-handover",
        "source": str(SRC.relative_to(ROOT)),
    }
    if (OUT / "meta.json").is_file():
        meta = json.loads((OUT / "meta.json").read_text(encoding="utf-8"))
        info["duration_s"] = meta.get("duration_s")
        info["stride"] = meta.get("stride")
        info["n_frames"] = meta.get("n_out_frames")
    (OUT / "handover_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print("WROTE", OUT / "handover_info.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
