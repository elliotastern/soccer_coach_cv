#!/usr/bin/env bash
# Install macOS LaunchAgent — dashboard survives Cursor/Terminal close.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LABEL="com.soccercoach.review-dashboard"
AGENT_DIR="$HOME/Library/LaunchAgents"
PLIST_DST="$AGENT_DIR/${LABEL}.plist"
PY="${REVIEW_PYTHON:-$HOME/.venvs/soccer-rfdetr312/bin/python3}"
LOGDIR="$HOME/Library/Logs/soccer-coach-review"
SUP="$ROOT/scripts/review_dashboard_supervisor.sh"

chmod +x "$SUP"
mkdir -p "$AGENT_DIR" "$LOGDIR"

if [[ ! -x "$PY" ]]; then
  echo "Python not found: $PY" >&2
  exit 1
fi

sed \
  -e "s|REVIEW_SUPERVISOR_SH|$SUP|g" \
  -e "s|REVIEW_ROOT|$ROOT|g" \
  -e "s|REVIEW_LOGDIR|$LOGDIR|g" \
  -e "s|REVIEW_PYTHON_BIN|$PY|g" \
  "$ROOT/scripts/com.soccercoach.review-dashboard.plist" >"$PLIST_DST"

launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$PLIST_DST"
launchctl enable "gui/$(id -u)/$LABEL"
launchctl kickstart -k "gui/$(id -u)/$LABEL"

echo "Installed LaunchAgent → $PLIST_DST"
echo "Start/stop: bash scripts/start_review_dashboard.sh {start|stop|status}"
echo "URL: http://127.0.0.1:${REVIEW_PORT:-8501}/"
