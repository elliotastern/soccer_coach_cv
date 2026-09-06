#!/usr/bin/env bash
# Catch: Ali 15s best_ball with anti-flicker kit labeling.
set -euo pipefail
cd "${ROOT:-$HOME/soccer_coach_cv}"
source ~/.venvs/soccer-rfdetr312/bin/activate
export PYTHONPATH=. PYTHONUNBUFFERED=1

OUT=reports/eval_match3/improve_eng_loop/ali15s_best_ball_ckpt_ab
RAW=reports/eval_match3/improve_eng_loop/ali15s_match3_shorts
STAGE=~/soccer_exchange/from_catch/ali15s_kit_stable
OUT_FILE=coach_best_ball_ali15s_v16_kit_stable.mp4
mkdir -p "$OUT" "$STAGE"

python3 scripts/gold_set/render_phase1_check_mosaic.py \
  --start 0 --match-sec 15 --stride 4 --out-fps 15 \
  --layout best_ball \
  --raw-dir "$RAW" \
  --out-dir "$OUT" \
  --out-file "$OUT_FILE" \
  --ball-checkpoint models/v16_residual_snaps/post_train/checkpoint.pth \
  2>&1 | tee "$OUT/render_v16_kit_stable.log"

cp -f "$OUT/meta.json" "$OUT/meta_v16_kit_stable.json"
cp -f "$OUT/$OUT_FILE" "$STAGE/"
cp -f "$OUT/meta_v16_kit_stable.json" "$STAGE/"

python3 - <<'PY'
import json
from pathlib import Path
out = Path("reports/eval_match3/improve_eng_loop/ali15s_best_ball_ckpt_ab")
d = json.loads((out / "meta_v16_kit_stable.json").read_text())
stats = d.get("stats") or []
n0 = sum(int(s.get("n0", 0)) for s in stats)
n1 = sum(int(s.get("n1", 0)) for s in stats)
# frame-to-frame team share volatility
shares = []
flips_proxy = 0
prev = None
for s in stats:
    a, b = int(s.get("n0", 0)), int(s.get("n1", 0))
    n = a + b
    shares.append(a / n if n else 0.5)
    if prev is not None and n and (prev[0] + prev[1]):
        # large swing in counts as flicker proxy
        if abs(a - prev[0]) + abs(b - prev[1]) >= 4:
            flips_proxy += 1
    prev = (a, b)
import statistics as st
print({
    "n0": n0, "n1": n1,
    "share": (round(100 * n0 / max(n0 + n1, 1), 1), round(100 * n1 / max(n0 + n1, 1), 1)),
    "share_std": round(st.pstdev(shares), 4) if len(shares) > 1 else 0,
    "big_swings": flips_proxy,
    "emits": d.get("n_emits"),
    "ball_frac": d.get("ball_frame_frac"),
})
PY
ls -lh "$STAGE"
