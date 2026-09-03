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
# PACKAGING IS NOT DONE HERE. It calls export_models.py, which is the single definition
# of what an archive contains and what makes it valid. This script used to run its own
# `zip -j pth json confusion *.png`, which quietly shipped a WORSE archive than the
# trainer's: no _threshold_curve.json and no _coverage.json -- the KPI curve and the
# coverage table, both written after the trainer exits -- and no verification at all,
# where export_models refuses to succeed unless the weights torch.load, the config parses
# and the finished zip passes a CRC check with the expected members inside.
#
# WHAT GETS PUBLISHED IS ALLOWLISTED. scripts/upload_allowlist.txt names the models that
# may reach the operator-facing "Download & Use" list. The store holds 170 archives, and
# almost all of them are research runs -- the zoo sweep, seed replicates, smoke tests,
# and failures like fzmnas05 at 40.36 miss@FA5 against an incumbent at 9.24. An operator
# has no way to tell those apart in a dropdown. Publishing is also effectively permanent
# (public_html: cached and indexable before withdrawal), so the default is to ship
# nothing unless a human has vouched for it.
#
# Usage:
#   uploadModels.sh                   # push allowlisted archives already in <repo>/models/
#   uploadModels.sh name [name ...]   # (re-)export each name via export_models.py, then push
#   ALLOW_ALL=1 uploadModels.sh       # deliberately publish everything -- prints what it will do
#   DRY_RUN=1 uploadModels.sh         # show what would be pushed, transfer nothing
#   SRC=/path uploadModels.sh name    # override where the .pth/.json live (both layouts)
#   STORE=/path SERVER_DIR=... ALLOWLIST=/path uploadModels.sh ...
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
#SERVER="${SERVER:-ammar@ammar.gr}"
#SERVER_DIR="${SERVER_DIR:-/home/ammar/public_html/magician/models/CameraV2Models/}"
ALLOWLIST="${ALLOWLIST:-$SCRIPT_DIR/upload_allowlist.txt}"
DRY_RUN="${DRY_RUN:-0}"


SERVER="${SERVER:-ammar@anoiksi.ammar.gr}"
SERVER_DIR="${SERVER_DIR:-/media/ammar/games2/Datasets/Models/}"



#Hardcoded to allow all uploads..
#ALLOW_ALL="${ALLOW_ALL:-0}"
ALLOW_ALL=1

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

# --- 1. package the named models, via THE packager ---------------------------
for name in "$@"; do
    echo "exporting $name via export_models.py"
    if ! ( cd "$REPO_DIR" && python -m mvc.export --apply --force --run "$name" ); then
        echo "SKIP $name: export_models.py could not package it" >&2
        continue
    fi
done

# --- 2. publish, gated by the allowlist --------------------------------------
shopt -s nullglob
zips=("$STORE"/*.zip)
if [ ${#zips[@]} -eq 0 ]; then
    echo "No .zip archives in $STORE to upload." >&2
    exit 1
fi

# Archive name is {run}_{YYYYmmdd}_{HHMMSS}.zip; strip the stamp to recover the run name.
run_of() { basename "$1" .zip | sed -E 's/_[0-9]{8}_[0-9]{6}$//'; }

send=()
if [ "$ALLOW_ALL" = "1" ]; then
    echo "!!! ALLOW_ALL=1: publishing ALL ${#zips[@]} archive(s), allowlist ignored."
    echo "!!! Every one becomes selectable by an operator in the annotator dropdown."
    send=("${zips[@]}")
else
    if [ ! -f "$ALLOWLIST" ]; then
        echo "No allowlist at $ALLOWLIST -- refusing to publish." >&2
        echo "Create it, or run with ALLOW_ALL=1 to publish everything." >&2
        exit 1
    fi
    allowed="$(grep -vE '^[[:space:]]*(#|$)' "$ALLOWLIST" | tr -d '\r' | awk '{print $1}')"
    if [ -z "$allowed" ]; then
        echo "Allowlist $ALLOWLIST has no entries -- nothing to publish." >&2
        exit 1
    fi
    # NEWEST archive per run only. Re-exporting leaves several stamps for one model
    # (50 runs currently have more than one), and publishing them all puts the same model
    # in the operator's dropdown several times with no way to tell which is current.
    # The stamp is YYYYmmdd_HHMMSS, so a lexical sort is chronological.
    skipped=0
    for want in $allowed; do
        newest=""
        for z in "${zips[@]}"; do
            [ "$(run_of "$z")" = "$want" ] || continue
            if [ -z "$newest" ] || [[ "$z" > "$newest" ]]; then newest="$z"; fi
        done
        [ -n "$newest" ] && send+=("$newest")
    done
    for z in "${zips[@]}"; do
        printf '%s\n' "${send[@]}" | grep -qxF "$z" || skipped=$((skipped + 1))
    done
    echo "allowlist $ALLOWLIST: ${#send[@]} archive(s) to publish, $skipped held back"
    # Name any allowlisted model that has no archive, rather than silently shipping less
    # than was asked for.
    while read -r want; do
        [ -n "$want" ] || continue
        found=0
        for z in "${zips[@]}"; do
            if [ "$(run_of "$z")" = "$want" ]; then found=1; break; fi
        done
        [ "$found" = "1" ] || echo "  NOTE allowlisted but no archive in $STORE: $want" >&2
    done <<< "$allowed"
fi

if [ ${#send[@]} -eq 0 ]; then
    echo "Nothing to publish." >&2
    exit 1
fi
for z in "${send[@]}"; do echo "  publish: $(basename "$z")"; done

if [ "$DRY_RUN" = "1" ]; then
    echo "DRY_RUN=1 -- nothing transferred."
    exit 0
fi

echo "rsync ${#send[@]} zip(s): $STORE -> $SERVER:$SERVER_DIR"
rsync -av --progress --partial -e "ssh -p $SSH_PORT" \
      "${send[@]}" "$SERVER:$SERVER_DIR"

echo "Done. Published models now appear in wxAnnotator's Online 'Download & Use' list."
