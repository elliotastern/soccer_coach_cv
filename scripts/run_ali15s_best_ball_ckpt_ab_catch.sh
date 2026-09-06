#!/usr/bin/env bash
# Catch: Ali handover 15s as best_ball + P7 ghost, ckpt A/B (shorts start @0).
set -euo pipefail
cd "${ROOT:-$HOME/soccer_coach_cv}"
source ~/.venvs/soccer-rfdetr312/bin/activate
export PYTHONPATH=. PYTHONUNBUFFERED=1

OUT=reports/eval_match3/improve_eng_loop/ali15s_best_ball_ckpt_ab
RAW=reports/eval_match3/improve_eng_loop/ali15s_match3_shorts
STAGE=~/soccer_exchange/from_catch/ali15s_best_ball_ckpt_ab
mkdir -p "$OUT" "$STAGE"

# Shorts begin at former frame 2390 → seek start=0
START=0
MATCH_SEC=15
STRIDE=4
OUT_FPS=15

render_one() {
  local tag="$1"
  local ckpt="$2"
  local out_file="coach_best_ball_ali15s_${tag}.mp4"
  echo "=== render $tag ckpt=$ckpt ==="
  test -f "$ckpt"
  test -d "$RAW"
  python3 scripts/gold_set/render_phase1_check_mosaic.py \
    --start "$START" --match-sec "$MATCH_SEC" --stride "$STRIDE" --out-fps "$OUT_FPS" \
    --layout best_ball \
    --raw-dir "$RAW" \
    --out-dir "$OUT" \
    --out-file "$out_file" \
    --ball-checkpoint "$ckpt" \
    2>&1 | tee "$OUT/render_${tag}.log"
  cp -f "$OUT/meta.json" "$OUT/meta_${tag}.json"
  cp -f "$OUT/$out_file" "$STAGE/"
  cp -f "$OUT/meta_${tag}.json" "$STAGE/"
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
    })
summary = {
    "clip": "ali_handover_ex_start2390_as_start0_15s_stride4_best_ball_ghost_p7",
    "raw_dir": "reports/eval_match3/improve_eng_loop/ali15s_match3_shorts",
    "rows": rows,
}
(out / "ckpt_ab_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
Path.home().joinpath("soccer_exchange/from_catch/ali15s_best_ball_ckpt_ab/ckpt_ab_summary.json").write_text(
    json.dumps(summary, indent=2) + "\n"
)
print(json.dumps(summary, indent=2))
PY
echo "=== ali15s best_ball ckpt A/B done ==="
