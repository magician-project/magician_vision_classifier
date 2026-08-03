#!/bin/bash
# Package trained model(s) into timestamped zips in the local model store and push
# them to the server directory the wxAnnotator "Download & Use" button reads
# (ModelDownload.py BASE_URL = http://ammar.gr/magician/models/CameraV2Models/).
#
# The store is <repo>/models/ -- resolved from THIS SCRIPT's own location, not from the
# caller's working directory, so it works from anywhere and on any machine. That is
# also exactly where the trainer drops its archives (`os.makedirs("models/")` +
# `zip -r models/{name}_{ts}.zip`), so a plain `uploadModels.sh` with no arguments
# pushes whatever training has produced.
#
# Usage:
#   uploadModels.sh                   # push everything in <repo>/models/
#   uploadModels.sh name [name ...]   # also package each {name}.pth+.json first, then push
#   SRC=/path uploadModels.sh name    # override where the .pth/.json live
#   STORE=/path SERVER_DIR=... uploadModels.sh ...
#
# Note: the CameraV2Models path below is the one the annotator download actually
# lists. The `.../ckpts2` path some older prints mention is a DIFFERENT directory
# and is NOT read by "Download & Use" -- don't push there expecting it to appear.
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"             # the classifier repo, via this script
SRC="${SRC:-$REPO_DIR}"                                # where {name}.pth / .json live
STORE="${STORE:-$REPO_DIR/models}"                     # ../models relative to this script
SSH_PORT="${SSH_PORT:-2222}"
SERVER="${SERVER:-ammar@ammar.gr}"
SERVER_DIR="${SERVER_DIR:-/home/ammar/public_html/magician/models/CameraV2Models/}"

mkdir -p "$STORE"
ts="$(date +%Y%m%d_%H%M%S)"

# --- 1. package the named models into the store as {name}_{ts}.zip ------------
for name in "$@"; do
    pth="$SRC/$name.pth"; json="$SRC/$name.json"
    if [ ! -f "$pth" ] || [ ! -f "$json" ]; then
        echo "SKIP $name: missing $name.pth or $name.json in $SRC" >&2
        continue
    fi
    zip_path="$STORE/${name}_${ts}.zip"
    # optional side-cars (confusion json + plots), only if present
    extras=()
    for e in "$SRC/${name}_confusion.json" "$SRC/${name}"*.png; do
        [ -e "$e" ] && extras+=("$e")
    done
    echo "packaging $name -> $zip_path"
    # -j flatten paths; zip follows symlinks by default so allclass_* aliases
    # store their real .pth/.json content under the alias basename.
    zip -j "$zip_path" "$pth" "$json" "${extras[@]}" >/dev/null
done

# --- 2. push every store zip to the server (rsync skips ones already there) ----
shopt -s nullglob
zips=("$STORE"/*.zip)
if [ ${#zips[@]} -eq 0 ]; then
    echo "No .zip archives in $STORE to upload." >&2
    exit 1
fi
echo "rsync ${#zips[@]} zip(s): $STORE -> $SERVER:$SERVER_DIR"
rsync -av --progress --partial -e "ssh -p $SSH_PORT" \
      "$STORE"/*.zip "$SERVER:$SERVER_DIR"

echo "Done. New models now appear in wxAnnotator's Online 'Download & Use' list."
