#!/usr/bin/env bash
# Ali handover 15s (start=2390) as best_ball + product P7 ghost, ckpt A/B.
set -euo pipefail
ROOT="/Volumes/LaCie/Projects/Soccer Coach CV"
cd "$ROOT"
export PYTHONPATH=. PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl-soccer}"

OUT=reports/eval_match3/improve_eng_loop/ali15s_best_ball_ckpt_ab
mkdir -p "$OUT"

START=2390
MATCH_SEC=15
STRIDE=4
OUT_FPS=15

render_one() {
  local tag="$1"
  local ckpt="$2"
  local out_file="coach_best_ball_ali15s_${tag}.mp4"
  echo "=== render $tag ckpt=$ckpt ==="
  test -f "$ckpt"
  python3 scripts/gold_set/render_phase1_check_mosaic.py \
    --start "$START" --match-sec "$MATCH_SEC" --stride "$STRIDE" --out-fps "$OUT_FPS" \
    --layout best_ball \
    --out-dir "$OUT" \
    --out-file "$out_file" \
    --ball-checkpoint "$ckpt" \
    2>&1 | tee "$OUT/render_${tag}.log"
  cp -f "$OUT/meta.json" "$OUT/meta_${tag}.json"
}

render_one v16 models/v16_residual_snaps/post_train/checkpoint.pth
render_one v26 models/v26_streak_holdout_snaps/post_train/checkpoint.pth
render_one v28 models/v28_m4_streak_holdout_snaps/post_train/checkpoint.pth
render_one v29 models/v29_m4_of_streak_holdout_snaps/post_train/checkpoint.pth

python3 - <<'PY'
import json
from pathlib import Path
out = Path("reports/eval_match3/improve_eng_loop/ali15s_best_ball_ckpt_ab")
rows = []
for tag in ("v16", "v26", "v28", "v29"):
    meta = out / f"meta_{tag}.json"
    if not meta.is_file():
        continue
    d = json.loads(meta.read_text())
    rows.append({
        "tag": tag,
        "ball_frame_frac": d.get("ball_frame_frac"),
        "n_emits": d.get("n_emits"),
        "path": d.get("path"),
        "duration_s": d.get("duration_s"),
        "layout": d.get("layout"),
    })
summary = {
    "clip": "ali_handover_start2390_15s_stride4_best_ball_ghost_p7",
    "note": "Same span as phase1_handover/check_15s_s4; layout best_ball; product static ghost P7-only",
    "rows": rows,
}
(out / "ckpt_ab_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
PY
echo "=== ali15s best_ball ckpt A/B done ==="
