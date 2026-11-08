#!/bin/bash
# Package a rig's pattern-based extrinsics map + camera calibration
# (analysis/extrinsics_from_pattern.py) into a timestamped zip and push it to the same
# server the model checkpoints live on (scripts/uploadModels.sh), in a SIBLING
# directory (CameraV2Extrinsics/, not CameraV2Models/) so it never appears in the
# annotator's model "Download & Use" dropdown -- these two files are not a model.
#
# No allowlist here, unlike uploadModels.sh: a rig's calibration is a single
# deliberate artifact you built and are choosing to publish, not one of 170 research
# runs that needs curating before an operator sees it.
#
# Usage:
#   uploadExtrinsics.sh                    # rig "default", map/intrinsics at the
#                                           # default local paths
#   uploadExtrinsics.sh myrig              # a named rig (multiple cars/cameras)
#   uploadExtrinsics.sh myrig /path/to/map.npz /path/to/intrinsics.json
#   DRY_RUN=1 uploadExtrinsics.sh          # show what would be pushed, transfer nothing
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"
RIG="${1:-default}"
MAP_SRC="${2:-$REPO_DIR/experiments/extrinsics/${RIG}_map.npz}"
INTRINSICS_SRC="${3:-$REPO_DIR/experiments/extrinsics/${RIG}_intrinsics.json}"
STORE="${STORE:-$REPO_DIR/models}"
SSH_PORT="${SSH_PORT:-2222}"
SERVER="${SERVER:-ammar@ammar.gr}"
SERVER_DIR="${SERVER_DIR:-/home/ammar/public_html/magician/models/CameraV2Extrinsics/}"
DRY_RUN="${DRY_RUN:-0}"

if [ ! -f "$MAP_SRC" ]; then
    echo "missing $MAP_SRC -- build it first: python analysis/extrinsics_from_pattern.py build ..." >&2
    exit 1
fi
if [ ! -f "$INTRINSICS_SRC" ]; then
    echo "missing $INTRINSICS_SRC -- see analysis/extrinsics_from_markers.py's calibrate mode" >&2
    exit 1
fi

# Sanity check both files before packaging -- a corrupt map or unparsable calibration
# should fail HERE, not silently ship and fail on a deployed box instead.
python3 -c "
import json, sys
import numpy as np
d = np.load('$MAP_SRC', allow_pickle=False)
for k in ('points', 'descriptors', 'reference_marker'):
    assert k in d, f'{k} missing from map.npz'
print(f'map OK: {len(d[\"points\"]):,} points, reference_marker={int(d[\"reference_marker\"])}')
intr = json.load(open('$INTRINSICS_SRC'))
for k in ('camera_matrix', 'dist_coeffs'):
    assert k in intr, f'{k} missing from intrinsics.json'
print('intrinsics OK')
"

mkdir -p "$STORE"
ts="$(date +%Y%m%d_%H%M%S)"
zip_name="${RIG}_extrinsics_${ts}.zip"
zip_path="$STORE/$zip_name"

workdir="$(mktemp -d)"
trap 'rm -rf "$workdir"' EXIT
cp "$MAP_SRC" "$workdir/${RIG}_map.npz"
cp "$INTRINSICS_SRC" "$workdir/${RIG}_intrinsics.json"
( cd "$workdir" && zip -j "$zip_path" "${RIG}_map.npz" "${RIG}_intrinsics.json" )

echo "packaged: $zip_path"
if [ "$DRY_RUN" = "1" ]; then
    echo "DRY_RUN=1 -- nothing transferred."
    exit 0
fi

echo "rsync $zip_name -> $SERVER:$SERVER_DIR"
rsync -av --progress --partial -e "ssh -p $SSH_PORT" \
      "$zip_path" "$SERVER:$SERVER_DIR"

echo "Done. Fetch on any box with: python3 -m mvc.inference.extrinsics_download $RIG"
