#!/usr/bin/env bash
exec "$(dirname "$0")/scripts/gold_set/run_viewer_stable.sh" "${1:-8765}"
