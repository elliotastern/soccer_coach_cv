#!/usr/bin/env bash
# Install macOS LaunchAgent — local-disk handover server (LaCie exfat blocked for launchd).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LABEL="com.soccercoach.phase1-handover"
AGENT_DIR="$HOME/Library/LaunchAgents"
PLIST_DST="$AGENT_DIR/${LABEL}.plist"
LOGDIR="$HOME/Library/Logs/soccer-coach-handover"
PY="$(command -v python3)"

chmod +x "$ROOT/scripts/start_phase1_handover.sh" 2>/dev/null || true
mkdir -p "$AGENT_DIR" "$LOGDIR"
python3 "$ROOT/scripts/gold_set/build_phase1_handover_dashboard.py"
python3 "$ROOT/scripts/handover_local_server.py" --sync-only

sed \
  -e "s|HANDOVER_PYTHON|$PY|g" \
  -e "s|HANDOVER_SERVER_PY|$ROOT/scripts/handover_local_server.py|g" \
  -e "s|HANDOVER_ROOT|$ROOT|g" \
  -e "s|HANDOVER_LOGDIR|$LOGDIR|g" \
  "$ROOT/scripts/com.soccercoach.phase1-handover.plist" >"$PLIST_DST"

launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$PLIST_DST"
launchctl enable "gui/$(id -u)/$LABEL"
launchctl kickstart -k "gui/$(id -u)/$LABEL"

echo "Installed LaunchAgent → $PLIST_DST"
echo "Local files: ~/Library/Application Support/soccer-coach-handover/"
echo "URL: http://127.0.0.1:8080/phase1-handover"
echo "Logs: $LOGDIR/server_stderr.log"
