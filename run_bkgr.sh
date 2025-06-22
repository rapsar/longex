#!/usr/bin/env bash
# chmod +x run_bkgr.sh
# ./run_bkgr.sh
set -euo pipefail

# ─── EDIT THESE ──────────────────────────────────────────────────────────────
VIDEO="/Volumes/Untitled/PRIVATE/M4ROOT/CLIP/20250604_RS0053.MP4"
METHOD="minim"       # unique | median | mean | minim | pmode | expo
FRAMES=300            # max frames to use
INDEX=0               # only used if METHOD=unique
ALPHA=0.05            # only used if METHOD=expo
OUTPUT="res/minim_bkgr.png"
SHOW=true             # true or false
# ─────────────────────────────────────────────────────────────────────────────

# Build args
args=(--method "$METHOD" --frames "$FRAMES" --index "$INDEX" --alpha "$ALPHA" --output "$OUTPUT")
if [ "$SHOW" = true ]; then
  args+=(--show)
fi

# Run
python3 "/Users/rss367/Documents/projects/long_exp/bkgr.py" "$VIDEO" "${args[@]}"