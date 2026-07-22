#!/bin/bash
# Pull model archives from the server into the local store, and optionally extract
# the newest .pth/.json of each into the directory the annotator/classifier loads
# from. Mirror of uploadModels.sh; same CameraV2Models path the annotator lists.
#
# Usage:
#   downloadModels.sh                       # mirror ALL server zips -> store
#   downloadModels.sh name [name ...]       # only those models -> store
#   downloadModels.sh --extract [name ...]  # ...and unzip newest of each into DEST
#   STORE=/path DEST=/path downloadModels.sh --extract customwide
#
# --extract puts {name}.pth + {name}.json where ClassifierPnm.model_scan() finds
# them (DEST, default = classifier repo root), so wxAnnotator lists them locally
# without going through the in-app download. (For a single model you can also just
# run:  python3 ModelDownload.py <name> --dest DEST )
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
DEST="${DEST:-$( cd "$SCRIPT_DIR/.." && pwd )}"        # default: classifier repo root
STORE="${STORE:-/media/ammar/games2/Datasets/Models}"
SSH_PORT="${SSH_PORT:-2222}"
SERVER="${SERVER:-ammar@ammar.gr}"
SERVER_DIR="${SERVER_DIR:-/home/ammar/public_html/magician/models/CameraV2Models/}"

extract=0
if [ "${1:-}" = "--extract" ]; then extract=1; shift; fi
names=("$@")

mkdir -p "$STORE"

# --- 1. pull zips from the server into the store -----------------------------
# rsync only transfers what's missing/newer. Filter to the requested base names
# via include/exclude rules; with no names, mirror everything.
if [ ${#names[@]} -eq 0 ]; then
    echo "mirroring all server zips: $SERVER:$SERVER_DIR -> $STORE"
    rsync -av --progress --partial -e "ssh -p $SSH_PORT" \
          "$SERVER:$SERVER_DIR"'*.zip' "$STORE/"
else
    filters=()
    for n in "${names[@]}"; do filters+=(--include="${n}_"'[0-9]*.zip'); done
    echo "pulling ${names[*]} -> $STORE"
    rsync -av --progress --partial -e "ssh -p $SSH_PORT" \
          "${filters[@]}" --include='*/' --exclude='*' \
          "$SERVER:$SERVER_DIR" "$STORE/"
fi

# --- 2. optionally extract the newest archive of each model into DEST ---------
if [ "$extract" -eq 1 ]; then
    mkdir -p "$DEST"
    shopt -s nullglob
    # which base names to extract: the requested ones, else every base in the store
    bases=("${names[@]}")
    if [ ${#bases[@]} -eq 0 ]; then
        declare -A seen
        for z in "$STORE"/*_[0-9]*.zip; do
            b="$(basename "$z")"; b="${b%_[0-9]*.zip}"
            seen["$b"]=1
        done
        bases=("${!seen[@]}")
    fi
    for name in "${bases[@]}"; do
        newest="$(ls -1 "$STORE/${name}_"[0-9]*.zip 2>/dev/null | sort | tail -1)"
        if [ -z "$newest" ]; then
            echo "SKIP $name: no archive in $STORE" >&2; continue
        fi
        echo "extracting $(basename "$newest") -> $DEST"
        unzip -o -j "$newest" '*.pth' '*.json' -d "$DEST" >/dev/null
    done
fi

echo "Done."
