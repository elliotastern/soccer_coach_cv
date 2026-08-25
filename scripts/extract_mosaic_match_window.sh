#!/usr/bin/env bash
# Extract a match-time window from mosaic MP4.
# Default: stride 15, src 60fps, out 4fps → video_t == match_t.
set -euo pipefail
MP4="${1:?mosaic mp4}"
START_S="${2:?start sec}"
DUR_S="${3:-15}"
OUT="${4:?output mp4}"
SRC_FPS="${SRC_FPS:-60}"
STRIDE="${STRIDE:-15}"
OUT_FPS="${OUT_FPS:-4}"
# video_t = match_t * src_fps / (stride * out_fps)
START_V=$(python3 -c "print(${START_S} * ${SRC_FPS} / (${STRIDE} * ${OUT_FPS}))")
DUR_V=$(python3 -c "print(${DUR_S} * ${SRC_FPS} / (${STRIDE} * ${OUT_FPS}))")
ffmpeg -y -hide_banner -loglevel error \
  -ss "${START_V}" -t "${DUR_V}" -i "${MP4}" \
  -c:v libx264 -pix_fmt yuv420p -movflags +faststart "${OUT}"
echo "WROTE ${OUT} (match ${START_S}s +${DUR_S}s → video ss=${START_V}s dur=${DUR_V}s)"
