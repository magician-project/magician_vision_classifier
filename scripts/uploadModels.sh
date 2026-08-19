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
# A named model is looked up in TWO layouts, the same pair ClassifierPnm.model_locate()
# reads: flat in $SRC (where the trainer first drops it), and filed under
# $SRC/experiments/<campaign>/<run>/ (where tidy_experiments.py moves it once the run is
# finished). Without the second one, uploading a model was a race against tidying --
# `uploadModels.sh anc_convnext_pico` worked right after training and answered
# "SKIP: missing" a day later. Side-cars are taken from wherever the model was found.
#
# Usage:
#   uploadModels.sh                   # push everything in <repo>/models/
#   uploadModels.sh name [name ...]   # also package each {name}.pth+.json first, then push
#   SRC=/path uploadModels.sh name    # override where the .pth/.json live (both layouts)
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

# Print the directory holding {name}.pth + {name}.json, or fail. Flat $SRC wins over a
# filed run, matching ClassifierPnm.model_locate().
locate_model_dir() {
    local name="$1" pth hits=()
    if [ -f "$SRC/$name.pth" ] && [ -f "$SRC/$name.json" ]; then
        echo "$SRC"
        return 0
    fi
    for pth in "$SRC"/experiments/*/*/"$name.pth"; do
        if [ -f "$pth" ] && [ -f "${pth%.pth}.json" ]; then
            hits+=("$(dirname "$pth")")
        fi
    done
    if [ ${#hits[@]} -eq 0 ]; then
        return 1
    fi
    if [ ${#hits[@]} -gt 1 ]; then
        echo "WARNING $name: ${#hits[@]} filed copies, packaging ${hits[0]}" >&2
    fi
    echo "${hits[0]}"
}

# --- 1. package the named models into the store as {name}_{ts}.zip ------------
for name in "$@"; do
    if ! dir="$(locate_model_dir "$name")"; then
        echo "SKIP $name: no $name.pth + $name.json in $SRC or $SRC/experiments/*/*/" >&2
        continue
    fi
    pth="$dir/$name.pth"; json="$dir/$name.json"
    zip_path="$STORE/${name}_${ts}.zip"
    # optional side-cars (confusion json + plots), only if present
    extras=()
    for e in "$dir/${name}_confusion.json" "$dir/${name}"*.png; do
        [ -e "$e" ] && extras+=("$e")
    done
    echo "packaging $name from $dir -> $zip_path"
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
