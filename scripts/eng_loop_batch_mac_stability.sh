#!/usr/bin/env bash
# Eng-loop: N isolated batch chunks → stability score /10. Gate ≥ 9.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="." PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl-soccer}"
mkdir -p "$MPLCONFIGDIR" reports/eval_match3/improve_eng_loop

N="${1:-10}"
CHUNK="${2:-150}"
VIDEO="${3:-data/raw/Match 3/P10-002.mp4}"
OUT="data/output/_stab_eng_loop"
LOG="reports/eval_match3/improve_eng_loop/batch_mac_stability.log"
SCORE_JSON="reports/eval_match3/improve_eng_loop/batch_mac_stability_score.json"
CFG="reports/eval_match3/improve_eng_loop/_batch_mac_stable_merged.yaml"
NOTES_FILE="reports/eval_match3/improve_eng_loop/_stab_notes.txt"

python3 - <<'PY'
import yaml
from pathlib import Path
base = yaml.safe_load(Path("configs/default.yaml").read_text())
over = yaml.safe_load(Path("configs/batch_mac_stable.yaml").read_text())
for k, v in over.items():
    if isinstance(v, dict) and isinstance(base.get(k), dict):
        base[k].update(v)
    else:
        base[k] = v
Path("reports/eval_match3/improve_eng_loop/_batch_mac_stable_merged.yaml").write_text(
    yaml.dump(base)
)
print("merged config ready")
PY

rm -rf "$OUT"
mkdir -p "$OUT"
: > "$LOG"
: > "$NOTES_FILE"
ok=0

echo "stability eng-loop N=$N chunk=$CHUNK" | tee -a "$LOG"
for i in $(seq 0 $((N - 1))); do
  start=$((i * CHUNK))
  echo "[$(date -u +%H:%M:%S)] trial $((i+1))/$N start=$start" | tee -a "$LOG"
  set +e
  python3 apps/batch_pipeline.py \
    --video "$VIDEO" \
    --config "$CFG" \
    --output "$OUT" \
    --start-frame "$start" \
    --max-frames "$CHUNK" \
    >>"$LOG" 2>&1
  ec=$?
  set -e
  if (( ec == 0 )) && tail -30 "$LOG" | grep -q "Processing complete!"; then
    ok=$((ok + 1))
    echo "[$(date -u +%H:%M:%S)] PASS trial $((i+1)) exit=$ec" | tee -a "$LOG"
  else
    echo "trial_$((i+1))_exit_$ec" >> "$NOTES_FILE"
    echo "[$(date -u +%H:%M:%S)] FAIL trial $((i+1)) exit=$ec" | tee -a "$LOG"
    sleep 10
  fi
  sleep 3
done

python3 - <<PY
import json
from pathlib import Path
ok, n = $ok, $N
notes = [l.strip() for l in Path("$NOTES_FILE").read_text().splitlines() if l.strip()]
score = round(10.0 * ok / max(1, n), 1)
payload = {
    "score": score,
    "ok": ok,
    "fail": n - ok,
    "n": n,
    "chunk": $CHUNK,
    "gate": 9.0,
    "pass": score >= 9.0,
    "notes": notes,
    "log": "$LOG",
    "config": "$CFG",
}
Path("$SCORE_JSON").write_text(json.dumps(payload, indent=2))
print(json.dumps(payload, indent=2))
print(f"STABILITY_SCORE {score}/10 gate={'PASS' if score >= 9 else 'FAIL'}")
PY
