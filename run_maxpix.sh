#!/usr/bin/env bash
# chmod +x run_maxpix.sh
# ./run_maxpix.sh
set -euo pipefail

# ─── EDIT THESE ──────────────────────────────────────────────────────────────
VIDEO="/Users/rss367/Desktop/ff/20250621_RS0075.MP4"
N_FRAMES=""               # put "" is None, else some number
CHANNEL="G"               # B, G, or R
OUTPUT_DIR="res"
GENERATE_GIF=false        # true or false
GIF_PATH=""               # optional, leave empty to auto-generate
SAMPLE_EVERY=30
DURATION=0.05
# ─────────────────────────────────────────────────────────────────────────────

# Build the command
cmd=(python "maxpix.py" -v "$VIDEO" -c "$CHANNEL" -o "$OUTPUT_DIR")

# Include frame limit if set
if [ -n "$N_FRAMES" ]; then
  cmd+=("-n" "$N_FRAMES")
fi

if [ "$GENERATE_GIF" = true ]; then
  cmd+=(--gif)
  if [ -n "$GIF_PATH" ]; then
    cmd+=(--gif_path "$GIF_PATH")
  fi
  cmd+=(--sample_every "$SAMPLE_EVERY" --duration "$DURATION")
fi

# Run it
echo "Running: ${cmd[*]}"
"${cmd[@]}"