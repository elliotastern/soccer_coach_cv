#!/usr/bin/env bash
# After v14 Mac train finishes: snapshot → strip A/B → holdout rebuild → FN/funnel → promote or DEAD_ENDS.
set -euo pipefail
ROOT="${ROOT:-/Volumes/LaCie/Projects/Soccer Coach CV}"
cd "$ROOT"
export PYTHONUNBUFFERED=1 PYTORCH_ENABLE_MPS_FALLBACK=1 PYTHONPATH=.
LOG=/tmp/v14_overnight_finish.log
exec >>"$LOG" 2>&1

echo "=== $(date -u) overnight finish start ==="
source /Users/elliotstern/.venvs/soccer-rfdetr312/bin/activate

# Wait until train pid is gone (or already gone)
PID_FILE=/tmp/v14_train.pid
if [[ -f "$PID_FILE" ]]; then
  pid=$(cat "$PID_FILE")
  while kill -0 "$pid" 2>/dev/null; do
    echo "$(date -u) waiting for train pid=$pid"
    sleep 60
  done
fi

mkdir -p models/v14_residual_snaps/post_train reports/eval_match3/improve_eng_loop
if [[ -f models/checkpoint.pth ]]; then
  cp -f models/checkpoint.pth models/v14_residual_snaps/post_train/checkpoint.pth
  cp -f models/checkpoint.pth models/v14_residual_snaps/post_train/checkpoint_epoch275.pth 2>/dev/null || true
fi
[[ -f models/checkpoint_best_regular.pth ]] && \
  cp -f models/checkpoint_best_regular.pth models/v14_residual_snaps/post_train/checkpoint_best_regular.pth

echo "=== strip A/B ==="
python scripts/gold_set/ab_v14_residual_vs_v13.py \
  --ckpt models/v14_residual_snaps/post_train/checkpoint.pth \
  --baseline-json reports/eval_match3/improve_eng_loop/ab_v13_residual_vs_v12.json

echo "=== holdout rebuild + score ==="
python scripts/gold_set/rebuild_holdout_det_v13.py \
  --ckpt "$ROOT/models/v14_residual_snaps/post_train/checkpoint.pth"

# Point rebuild out is ab_v13_holdout_score — also write v14-tagged copy via python below
python - <<'PY'
import json, shutil, yaml
from pathlib import Path
ROOT = Path("/Volumes/LaCie/Projects/Soccer Coach CV")
src = ROOT / "reports/eval_match3/improve_eng_loop/ab_v13_holdout_score.json"
dst = ROOT / "reports/eval_match3/improve_eng_loop/ab_v14_holdout_score.json"
if src.is_file():
    shutil.copy(src, dst)
ab = json.loads((ROOT / "reports/eval_match3/improve_eng_loop/ab_v14_residual_vs_v13.json").read_text())
hold = json.loads(dst.read_text()) if dst.is_file() else {}
promote = bool(ab.get("promote_candidate"))
hold_r = (hold.get("v13_caches") or hold.get("v14_caches") or {}).get("clear_ball_proxy_R")
# rebuild_holdout writes under key v13_caches even when ckpt is v14 — use cand R
if hold_r is None:
    hold_r = (hold.get("v13_caches") or {}).get("clear_ball_proxy_R")
gate_ok = hold_r is not None and float(hold_r) >= 0.884 - 0.001
decision = {
    "promote": promote and gate_ok,
    "strip_promote_candidate": promote,
    "holdout_R": hold_r,
    "holdout_gate_ok": gate_ok,
    "ab_path": "reports/eval_match3/improve_eng_loop/ab_v14_residual_vs_v13.json",
    "holdout_path": "reports/eval_match3/improve_eng_loop/ab_v14_holdout_score.json",
}
(ROOT / "reports/eval_match3/improve_eng_loop/overnight_subgoal_10_promote.json").write_text(
    json.dumps(decision, indent=2)
)
print(decision)
if decision["promote"]:
    cfg_path = ROOT / "configs/default.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())
    cfg["detection"]["ball_checkpoint"] = "models/v14_residual_snaps/post_train/checkpoint.pth"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    # patch review pointers
    for rel in ("apps/kit_label_dashboard.py", "src/review/app.py", "docs/product/MATCH_REVIEW_HANDOVER.md"):
        p = ROOT / rel
        if not p.is_file():
            continue
        t = p.read_text()
        t2 = t.replace(
            "models/v13_residual_snaps/post_train/checkpoint.pth",
            "models/v14_residual_snaps/post_train/checkpoint.pth",
        )
        if t2 != t:
            p.write_text(t2)
            print("updated", rel)
    print("PROMOTED v14")
else:
    dead = ROOT / "reports/ball_testing/DEAD_ENDS.md"
    note = (
        "\n| **v14 residual 10-epoch** (Mac MPS @560; no holdout train) | "
        f"strip_promote={promote} holdout_R={hold_r} | **No promote** — see overnight_subgoal_10_promote.json |\n"
    )
    text = dead.read_text() if dead.is_file() else ""
    if "v14 residual 10-epoch" not in text:
        # append under Residual if present
        dead.write_text(text + note)
    print("NO_PROMOTE kept v13")
PY

echo "=== FN audit + funnel on holdout det_cache_v13 (post v14 detect) ==="
python scripts/gold_set/fn_audit_match3_random.py \
  --cache-dir reports/eval_match3/pitchmap_gallery_holdout/det_cache_v13 \
  --out reports/eval_match3/improve_eng_loop/r1_random_fn_audit_holdout_v14.json || true
python scripts/gold_set/holdout_other_cam_funnel.py \
  --cache-dir reports/eval_match3/pitchmap_gallery_holdout/det_cache_v13 \
  --out reports/eval_match3/improve_eng_loop/holdout_other_cam_funnel_v14.json || true

echo "=== eng_loop ==="
python scripts/gold_set/eng_loop_match3_improve.py || true

echo "=== $(date -u) overnight finish done ==="
