#!/usr/bin/env bash
# Match-4 5-min batch with end cams (P1+P6) for kit consensus ≥9.
# Run on Catch when Tailscale is up. See docs/product/CATCH_MATCH4_KIT_CONSENSUS_9.md
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export CAMS="${CAMS:-P10-match4 P9-match4 P7-match4 P8-match4 P1-match4 P6-match4}"
echo "Match4 fullcam kit batch CAMS=$CAMS"
exec bash scripts/run_batch_match4_5min.sh "$@"
