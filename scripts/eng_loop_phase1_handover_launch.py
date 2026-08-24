#!/usr/bin/env python3
"""Eng-loop: supervised Phase 1 handover viewer launch stability."""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "reports/eval_match3/improve_eng_loop/phase1_handover_launch"
HANDOVER = ROOT / "reports/eval_match3/improve_eng_loop/phase1_handover"
PORT = 8080
PASS = 9.0
HANDOVER_URL = f"http://127.0.0.1:{PORT}/phase1-handover"
INDEX_URL = (
    f"http://127.0.0.1:{PORT}/reports/eval_match3/improve_eng_loop/phase1_handover/index.html"
)


def _score(ok: bool, partial: float = 5.0) -> float:
    return 10.0 if ok else partial


def handover_healthy() -> bool:
    try:
        with urlopen(HANDOVER_URL, timeout=5) as r:
            if r.status != 200:
                return False
        for url in (
            f"http://127.0.0.1:{PORT}/index.html",
            INDEX_URL,
        ):
            try:
                with urlopen(url, timeout=5) as r:
                    body = r.read(8000).decode("utf-8", errors="replace")
                    if "Events to validate" in body:
                        return True
            except (URLError, OSError, TimeoutError):
                continue
        return False
    except (URLError, OSError, TimeoutError):
        return False


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["/bin/bash", str(ROOT / "scripts/start_phase1_handover.sh"), "stop"],
        cwd=str(ROOT),
        capture_output=True,
    )
    build = subprocess.run(
        [sys.executable, str(ROOT / "scripts/gold_set/build_phase1_handover_dashboard.py")],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    index_ok = (HANDOVER / "index.html").is_file()
    clip_ok = (HANDOVER / "coach_mosaic_pitch_min.mp4").is_file()
    index_txt = (HANDOVER / "index.html").read_text(encoding="utf-8") if index_ok else ""

    start = subprocess.run(
        ["/bin/bash", str(ROOT / "scripts/start_phase1_handover.sh"), "start-bg"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    start_ok = start.returncode == 0
    t0_healthy = handover_healthy() if start_ok else False

    time.sleep(15)
    t15_healthy = handover_healthy()

    status = subprocess.run(
        ["/bin/bash", str(ROOT / "scripts/start_phase1_handover.sh"), "status"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )

    comps = {
        "01_prompt_eval": _score(
            (OUT / "PROMPT.md").is_file() and (OUT / "PROMPT_EVAL.md").is_file()
        ),
        "02_dashboard_build": _score(build.returncode == 0 and index_ok and clip_ok),
        "03_start_healthy": _score(start_ok and t0_healthy),
        "04_stable_15s": _score(t15_healthy),
        "05_events_panel": _score("Events to validate" in index_txt),
        "06_status_cmd": _score(status.returncode == 0 and t15_healthy),
    }
    failed = {k: v for k, v in comps.items() if v < PASS}
    payload = {
        "components": comps,
        "failed": failed,
        "pass": len(failed) == 0,
        "handover_url": HANDOVER_URL,
        "start_stdout": (start.stdout or "").strip()[-200:],
        "status_stdout": (status.stdout or "").strip(),
    }
    (OUT / "scores.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if not payload["pass"]:
        return 1
    print("all phase1_handover_launch >= 9/10")
    print(f"Coach URL: {HANDOVER_URL}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
