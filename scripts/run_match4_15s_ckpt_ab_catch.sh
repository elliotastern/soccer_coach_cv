#!/usr/bin/env bash
# Catch: same 15s best_ball clip (t087 / start=5220) for v16 + specialty ckpts.
set -euo pipefail
cd "${ROOT:-$HOME/soccer_coach_cv}"
source ~/.venvs/soccer-rfdetr312/bin/activate
export PYTHONPATH=. PYTHONUNBUFFERED=1

OUT=reports/eval_match3/improve_eng_loop/match4_15s_ckpt_ab
mkdir -p "$OUT" ~/soccer_exchange/from_catch/match4_15s_ckpt_ab

START=5220
MATCH_SEC=15
STRIDE=4
OUT_FPS=4
KIT=data/output/match_4_5min/team_centroids.json

render_one() {
  local tag="$1"
  local ckpt="$2"
  local out_file="coach_best_ball_15s_t087_${tag}.mp4"
  echo "=== render $tag ckpt=$ckpt ==="
  test -f "$ckpt"
  python3 scripts/gold_set/render_phase1_check_mosaic.py \
    --start "$START" --match-sec "$MATCH_SEC" --stride "$STRIDE" --out-fps "$OUT_FPS" \
    --layout best_ball \
    --out-dir "$OUT" \
    --out-file "$out_file" \
    --kit-centroids "$KIT" \
    --ball-checkpoint "$ckpt" \
    2>&1 | tee "$OUT/render_${tag}.log"
  cp -f "$OUT/$out_file" ~/soccer_exchange/from_catch/match4_15s_ckpt_ab/ 2>/dev/null || true
  cp -f "$OUT/meta.json" ~/soccer_exchange/from_catch/match4_15s_ckpt_ab/meta_${tag}.json 2>/dev/null || true
  # snapshot meta under tag name (render overwrites meta.json)
  cp -f "$OUT/meta.json" "$OUT/meta_${tag}.json" 2>/dev/null || true
}

render_one v16 models/v16_residual_snaps/post_train/checkpoint.pth
render_one v26 models/v26_streak_holdout_snaps/post_train/checkpoint.pth
render_one v28 models/v28_m4_streak_holdout_snaps/post_train/checkpoint.pth
render_one v29 models/v29_m4_of_streak_holdout_snaps/post_train/checkpoint.pth

echo "=== summarize ball_frac ==="
python3 - <<'PY'
import json
from pathlib import Path
out = Path("reports/eval_match3/improve_eng_loop/match4_15s_ckpt_ab")
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
    })
summary = {"clip": "t087_start5220_15s_stride4_best_ball", "rows": rows}
(out / "ckpt_ab_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
Path.home().joinpath("soccer_exchange/from_catch/match4_15s_ckpt_ab/ckpt_ab_summary.json").write_text(
    json.dumps(summary, indent=2) + "\n"
)
PY
echo "=== 15s ckpt A/B done ==="
