#!/usr/bin/env bash
# Open review dashboard in macOS Terminal (survives Cursor/agent shell exit).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUN="$ROOT/scripts/run_review_dashboard_foreground.sh"
chmod +x "$RUN"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "Use: bash $RUN"
  exit 0
fi

# Escape for AppleScript
ESCAPED=$(printf '%s' "$RUN" | sed "s/'/'\\\\''/g")
osascript <<EOF
tell application "Terminal"
  activate
  do script "bash '$ESCAPED'"
end tell
EOF
echo "Opened Terminal — dashboard starting at http://127.0.0.1:${REVIEW_PORT:-8501}/"
echo "If browser still fails, wait ~10s then open that URL in Safari or Chrome."
