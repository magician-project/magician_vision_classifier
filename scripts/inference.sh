#!/bin/bash

# Runs the standalone classifier against a recorded dataset folder instead of
# a live camera: starts the SharedMemoryVideoBuffers server, streams the
# dataset into shared memory stream "stream1" (looping), then runs
# mvc/inference/live_torch.py against that stream.
#
# Usage:
#   scripts/inference.sh /path/to/dataset [streamer args...] [-- live_torch.py args...]
#
# Args before "--" are passed through to folder_shared_memory_streamer.py
# (e.g. --delay, --fps, --label); args after "--" are passed to live_torch.py.
#
# Example:
#   scripts/inference.sh /home/ammar/Documents/Programming/magician_grabber/FORTH_NEGA_650
#   scripts/inference.sh /path/to/dataset --delay 500 -- --config low_false_alarm --no-visualization

set -e

THISDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "$THISDIR/.." && pwd )"
cd "$REPO_ROOT"

if (( $# < 1 )); then
  echo "Usage: $0 /path/to/dataset [streamer args...] [-- live_torch.py args...]"
  exit 1
fi

DATASET="$1"
shift

STREAMER_ARGS=()
while (( $# > 0 )) && [ "$1" != "--" ]; do
  STREAMER_ARGS+=("$1")
  shift
done
if [ "$1" = "--" ]; then
  shift
fi
LIVE_TORCH_ARGS=("$@")

if [ ! -d "$DATASET" ]; then
  echo "Dataset directory not found: $DATASET"
  exit 1
fi

# ---- Python environment ----
if [ -d venv/ ]; then
  echo "Activating venv/"
  source venv/bin/activate
elif [ -d venvNoROS/ ]; then
  echo "Activating venvNoROS/"
  source venvNoROS/bin/activate
else
  echo "No venv/ or venvNoROS/ found -- set one up first (see README.md)."
  exit 1
fi

# ---- Shared memory library ----
if [ ! -f libSharedMemoryVideoBuffers.so ]; then
  echo "libSharedMemoryVideoBuffers.so missing, building it..."
  bash scripts/updateSharedMemoryMechanism.sh
fi

SERVER_LOG="/tmp/mvc_inference_server.log"
STREAMER_LOG="/tmp/mvc_inference_streamer.log"
STARTED_SERVER=0
SERVER_PID=""
STREAMER_PID=""

cleanup() {
  if [ -n "$STREAMER_PID" ]; then
    kill "$STREAMER_PID" 2>/dev/null
    wait "$STREAMER_PID" 2>/dev/null
  fi
  if [ "$STARTED_SERVER" = "1" ] && [ -n "$SERVER_PID" ]; then
    kill "$SERVER_PID" 2>/dev/null
    wait "$SERVER_PID" 2>/dev/null
  fi
}
trap cleanup EXIT INT TERM

# ---- Shared memory server ----
if pgrep -x server >/dev/null; then
  echo "Shared memory server already running, reusing it."
else
  echo "Starting shared memory server (log: $SERVER_LOG)..."
  SharedMemoryVideoBuffers/server --nokb >"$SERVER_LOG" 2>&1 &
  SERVER_PID=$!
  STARTED_SERVER=1
  sleep 0.5
fi

# ---- Stream the dataset into shared memory ----
echo "Streaming $DATASET into shared memory stream 'stream1' (log: $STREAMER_LOG)..."
python3 -u -m mvc.inference.folder_shared_memory_streamer "$DATASET" --stream stream1 \
  "${STREAMER_ARGS[@]}" >"$STREAMER_LOG" 2>&1 &
STREAMER_PID=$!
sleep 1

if ! kill -0 "$STREAMER_PID" 2>/dev/null; then
  echo "Streamer failed to start, see $STREAMER_LOG"
  cat "$STREAMER_LOG"
  exit 1
fi

# ---- Run the classifier ----
python3 -m mvc.inference.live_torch --stream stream1 "${LIVE_TORCH_ARGS[@]}"
