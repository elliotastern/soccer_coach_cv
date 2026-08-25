#!/usr/bin/env bash
# Catch: after mosaic5, build fuse eval timelines + coach MP4s for holdout windows.
set -euo pipefail
KEY="${CATCH_SSH_KEY:-${HOME}/.ssh/id_ed25519_soccer_catch}"
HOST="${CATCH_SSH_TARGET:-catch@100.113.134.41}"
SSH="ssh -i ${KEY} -o IdentitiesOnly=yes -o BatchMode=yes ${HOST}"

run_eval() {
  local start="$1" clip="$2" file="$3"
  cat <<REMOTE
if tmux has-session -t mosaic5 2>/dev/null; then sleep 30; else break; fi
source ~/.venvs/soccer-rfdetr312/bin/activate
cd ~/soccer_coach_cv
export PYTHONPATH=. PYTHONUNBUFFERED=1
python3 scripts/gold_set/build_fuse_eval_window.py --start ${start} --clip-id ${clip} --stride 4 \
  --labels-json data/processed/gold_sets/match3_events_v2_dribble/clips/${clip}/labels.json \
  2>&1 | tee reports/eval_match3/improve_eng_loop/${clip}.log
python3 scripts/gold_set/render_phase1_check_mosaic.py \
  --start ${start} --match-sec 15 --stride 4 --out-fps 4 \
  --out-dir reports/eval_match3/improve_eng_loop/${clip} \
  --out-file ${file} \
  2>&1 | tee -a reports/eval_match3/improve_eng_loop/${clip}_render.log
cp -f reports/eval_match3/improve_eng_loop/${clip}/${file} ~/soccer_exchange/from_catch/${file}
echo DONE_${clip}
REMOTE
}

${SSH} bash -s <<'REMOTE'
set -euo pipefail
if tmux has-session -t fuse_eval_queue 2>/dev/null; then
  echo "fuse_eval_queue already running"
  exit 0
fi
tmux new-session -d -s fuse_eval_queue \
  'while tmux has-session -t mosaic5 2>/dev/null; do sleep 30; done; \
   source ~/.venvs/soccer-rfdetr312/bin/activate; cd ~/soccer_coach_cv; \
   export PYTHONPATH=. PYTHONUNBUFFERED=1; \
   for spec in "2940:real_fuse_eval_49s:coach_mosaic_eval_49s_stride4.mp4" \
                "4226:real_fuse_eval_69s:coach_mosaic_eval_69s_stride4.mp4"; do \
     start=${spec%%:*}; rest=${spec#*:}; clip=${rest%%:*}; file=${rest#*:}; \
     labels=data/processed/gold_sets/match3_events_v2_dribble/clips/${clip}/labels.json; \
     python3 scripts/gold_set/build_fuse_eval_window.py --start $start --clip-id $clip --stride 4 \
       --labels-json $labels 2>&1 | tee reports/eval_match3/improve_eng_loop/${clip}.log; \
     python3 scripts/gold_set/render_phase1_check_mosaic.py --start $start --match-sec 15 --stride 4 --out-fps 4 \
       --out-dir reports/eval_match3/improve_eng_loop/${clip} --out-file $file \
       2>&1 | tee reports/eval_match3/improve_eng_loop/${clip}_render.log; \
     cp -f reports/eval_match3/improve_eng_loop/${clip}/$file ~/soccer_exchange/from_catch/$file; \
     echo DONE_$clip; \
   done; tmux kill-session -t eval49_wait 2>/dev/null || true'
echo "Queued fuse_eval_queue (49s + 69s after mosaic5)"
REMOTE
