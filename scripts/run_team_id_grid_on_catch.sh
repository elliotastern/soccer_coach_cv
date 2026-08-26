#!/usr/bin/env bash
# Run team ID strategy grid on Catch (Match 4 90s, plain+SAHI).
set -euo pipefail
REPO="${1:-$HOME/soccer_coach_cv}"
cd "$REPO"
source ~/.venvs/soccer-rfdetr312/bin/activate
SESSION=team_id_grid
tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" \
  "python3 scripts/eval_team_id_strategy_grid.py --start 0 --match-sec 90 --stride 15 --sahi plain,sahi --strategies all 2>&1 | tee reports/eval_match3/team_id_strategy_grid/grid_run.log"
echo "Started tmux session $SESSION"
echo "Monitor: tmux capture-pane -t $SESSION -p | tail -10"
