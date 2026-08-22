#!/usr/bin/env python3
"""Build phase1_proof video pack: copy existing pillar clips + render ball section."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ENG = ROOT / "reports/eval_match3/improve_eng_loop"
OUT_ROOT = ENG / "phase1_proof"
RENDER = ROOT / "scripts/gold_set/render_phase1_check_mosaic.py"

COPY_PACKS = {
    "integrated": {
        "pillar": "Full product demo (ball + mapping + players + team + events)",
        "source_dir": ENG / "phase1_check",
        "files": [
            "coach_mosaic_pitch_min.mp4",
            "meta.json",
            "still_mid.jpg",
            "emits_render.json",
        ],
    },
    "mapping": {
        "pillar": "Pitch 1 player/ball mapping (cam labels on dots)",
        "source_dir": ENG / "player_map/check_15s_s4",
        "files": ["coach_mosaic_pitch_min.mp4", "meta.json"],
    },
    "players_team": {
        "pillar": "Multi-cam player boxes + team A/B colors on pitch",
        "source_dir": ENG / "players_pitch/check_15s_s4",
        "files": ["coach_mosaic_pitch_min.mp4", "meta.json"],
    },
    "events": {
        "pillar": "Heuristic pass/shot/recovery events bar",
        "source_dir": ENG / "phase1_check_smooth",
        "files": [
            "coach_mosaic_pitch_min.mp4",
            "meta.json",
            "emits_render.json",
        ],
        "extra": [
            (
                ENG / "heuristic_events/v1_check25_event_labels.jpg",
                "v1_check25_event_labels.jpg",
            ),
        ],
    },
}

GAPS = [
    {
        "pillar": "Batch / checkpoints / CSV/JSON export",
        "note": "No MP4; run apps/batch_pipeline.py for data outputs",
    },
    {
        "pillar": "Review app (Streamlit)",
        "note": "Live UI; screen recording not included",
    },
    {
        "pillar": "2-match delivery / handover",
        "note": "Process milestone, not a single demo clip",
    },
]


def h264_encode(src: Path) -> None:
    """QuickTime/Cursor preview need yuv420p H.264."""
    tmp = src.with_suffix(".h264.mp4")
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(src), "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(tmp),
    ]
    subprocess.run(cmd, check=True)
    tmp.replace(src)


def ensure_h264(mp4: Path) -> None:
    probe = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=codec_name", "-of", "csv=p=0", str(mp4),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    if probe.stdout.strip() != "h264":
        print(f"H264 encode {mp4}", flush=True)
        h264_encode(mp4)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--force-ball", action="store_true", help="Re-render ball/ clip")
    p.add_argument("--skip-ball-render", action="store_true", help="Copy only; no render")
    return p.parse_args()


def copy_file(src: Path, dst: Path) -> None:
    if not src.is_file():
        raise FileNotFoundError(f"missing source {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def copy_pack(name: str, spec: dict) -> list[str]:
    copied = []
    src_dir = spec["source_dir"]
    dst_dir = OUT_ROOT / name
    dst_dir.mkdir(parents=True, exist_ok=True)
    for fname in spec["files"]:
        copy_file(src_dir / fname, dst_dir / fname)
        copied.append(str((dst_dir / fname).relative_to(ROOT)))
    for src, fname in spec.get("extra", []):
        copy_file(src, dst_dir / fname)
        copied.append(str((dst_dir / fname).relative_to(ROOT)))
    return copied


def needs_ball_render(force: bool) -> bool:
    mp4 = OUT_ROOT / "ball/coach_mosaic_pitch_min.mp4"
    return force or not mp4.is_file()


def render_ball(force: bool) -> None:
    out_dir = OUT_ROOT / "ball"
    out_dir.mkdir(parents=True, exist_ok=True)
    if not needs_ball_render(force):
        print("SKIP ball render (exists)", out_dir / "coach_mosaic_pitch_min.mp4")
        return
    cmd = [
        sys.executable,
        str(RENDER),
        "--start",
        "2390",
        "--match-sec",
        "25",
        "--stride",
        "2",
        "--out-fps",
        "30",
        "--no-events-bar",
        "--out-dir",
        str(out_dir),
    ]
    print("RUN", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=str(ROOT))


def read_meta(folder: str) -> dict:
    meta_path = OUT_ROOT / folder / "meta.json"
    if not meta_path.is_file():
        return {}
    return json.loads(meta_path.read_text(encoding="utf-8"))


def build_manifest(copied: dict[str, list[str]]) -> dict:
    entries = []
    for name, spec in COPY_PACKS.items():
        meta = read_meta(name)
        entries.append(
            {
                "folder": name,
                "pillar": spec["pillar"],
                "video": f"phase1_proof/{name}/coach_mosaic_pitch_min.mp4",
                "source_dir": str(spec["source_dir"].relative_to(ROOT)),
                "duration_s": meta.get("duration_s"),
                "ball_frame_frac": meta.get("ball_frame_frac"),
                "n_emits": meta.get("n_emits", 0),
                "events_bar": meta.get("events_bar"),
                "copied_files": copied.get(name, []),
            }
        )
    ball_meta = read_meta("ball")
    entries.append(
        {
            "folder": "ball",
            "pillar": "Ball detect + fuse on mosaic (no events bar)",
            "video": "phase1_proof/ball/coach_mosaic_pitch_min.mp4",
            "source_dir": "rendered",
            "duration_s": ball_meta.get("duration_s"),
            "ball_frame_frac": ball_meta.get("ball_frame_frac"),
            "n_emits": ball_meta.get("n_emits", 0),
            "events_bar": ball_meta.get("events_bar"),
            "copied_files": [],
        }
    )
    order = ["integrated", "ball", "mapping", "players_team", "events"]
    entries.sort(key=lambda e: order.index(e["folder"]))
    return {
        "ts": datetime.now(timezone.utc).isoformat(),
        "root": str(OUT_ROOT.relative_to(ROOT)),
        "clips": entries,
        "gaps": GAPS,
    }


def main() -> int:
    args = parse_args()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    copied: dict[str, list[str]] = {}
    for name, spec in COPY_PACKS.items():
        copied[name] = copy_pack(name, spec)
        print(f"COPIED {name}: {len(copied[name])} files")
    if not args.skip_ball_render:
        render_ball(args.force_ball)
    for name in list(COPY_PACKS) + ["ball"]:
        mp4 = OUT_ROOT / name / "coach_mosaic_pitch_min.mp4"
        if mp4.is_file():
            ensure_h264(mp4)
    manifest = build_manifest(copied)
    manifest_path = OUT_ROOT / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("WROTE", manifest_path)
    for clip in manifest["clips"]:
        print(
            f"  {clip['folder']}: dur={clip['duration_s']}s "
            f"ball_frac={clip['ball_frame_frac']} emits={clip['n_emits']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
