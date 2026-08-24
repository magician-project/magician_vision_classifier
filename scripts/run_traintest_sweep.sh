#!/usr/bin/env bash
# The 47-model in-distribution train/test split campaign (analysis/sweeps/traintest_sweep.py).
#
# Every other campaign's validation set is deliberately recording-disjoint. This one is the
# opposite on purpose: a random TILE-level split of the training data
# (dataloader.frozen_tile_split, see build_tile_split_val.py), which leaks sibling tiles by
# design. It answers an in-distribution fit-ceiling question, not a generalization one --
# do not compare its miss@FA5 against the factory/coverage numbers in 21-8/24-8-report.md as
# if it were the same kind of instrument.
#
# Per run: train -> score_checkpoints (aggregate miss@FA5, every epoch) ->
# eval_traintest_split (per-class macro detect@FA5 at the monitored-best checkpoint, the
# number actually comparable to the TIER_A coverage macro in 21-8/24-8-report.md -- the raw
# aggregate miss@FA5 is incidence-weighted and is NOT comparable, see eval_traintest_split.py).
# NO eval_coverage.py call -- this campaign has no carve-out/coverage concept, just the one
# (leaky, in-distribution) validation set.
#
# RESUMABLE. A run whose highest-epoch threshold_curve.json AND traintest_detect.json already
# exist is skipped. Uses
# artifact_paths.exists() (root OR experiments/, once tidied), same fix already applied to
# run_full_zoo_sweep.sh / run_seed_replicates.sh -- a root-only check would silently retrain
# every finished model on a resume once tidy_experiments.py has filed them away.
set -u
GPU="${GPU:-2}"
LOGDIR=/storage/ammarkov/logs
cd "$(dirname "$0")/.." || exit 1
mkdir -p "$LOGDIR"
MANIFEST=experiments/traintest_sweep_manifest.tsv
EPOCHS=2                # matches analysis/sweeps/traintest_sweep.py's EPOCHS
LAST_EPOCH=$((EPOCHS - 1))

# Same protocol-fingerprint mechanism as run_full_zoo_sweep.sh / run_seed_replicates.sh,
# computed PER RUN so a mid-queue code change is caught rather than silently spanning runs.
PROTOCOL_FILES="mvc/core/model_zoo.py mvc/train.py mvc/core/datasets.py mvc/core/lit_classifier.py analysis/eval/score_checkpoints.py"
require_protocol_files() {
    local missing=""
    for f in $PROTOCOL_FILES; do
        [ -f "$f" ] || missing="$missing $f"
    done
    [ -z "$missing" ] && return 0
    echo "PROTOCOL FILE(S) NOT FOUND:$missing" >&2
    exit 1
}
code_fingerprint() {
    # shellcheck disable=SC2086
    md5sum $PROTOCOL_FILES | md5sum | cut -c1-12
}
git_rev() { git rev-parse --short HEAD 2>/dev/null || echo nogit; }
require_protocol_files
if [ ! -f "$MANIFEST" ]; then
    printf 'run\tmodel\tfingerprint\tgit\tstarted\n' > "$MANIFEST"
fi

. "$(dirname "$0")/gpu_lock.sh"
gpu_wait

mapfile -t CFGS < <(python3 -m analysis.sweeps.traintest_sweep --dry-run 2>/dev/null \
    | awk 'NR>1 && NF==3 {print $3}')
echo "=== $(date -Is) queue: ${#CFGS[@]} models"
[ "${#CFGS[@]}" -gt 0 ] || { echo "!!! queue is empty -- generator failed, aborting"; exit 1; }

scored_exists() {
    local run="$1" model="$2"
    python3 -c "
from mvc.core.artifact_paths import exists
import sys
run, ep, model = sys.argv[1], sys.argv[2], sys.argv[3]
ok = exists(f'{run}_ep{ep}_{model}_threshold_curve.json') and exists(f'{run}_{model}_traintest_detect.json')
sys.exit(0 if ok else 1)
" "$run" "$LAST_EPOCH" "$model"
}

for CFG in "${CFGS[@]}"; do
    run="${CFG%.json}"
    model=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['model'].replace('/', '_'))" "$CFG" 2>/dev/null || true)
    [ -f "$CFG" ] || { echo "!!! missing $CFG, skipping"; continue; }
    if [ -n "$model" ] && scored_exists "$run" "$model"; then
        echo "=== $(date -Is) $run already complete, skipping"
        continue
    fi

    ckdir=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['checkpoint_dir'])" "$CFG" 2>/dev/null || true)
    if [ -n "$ckdir" ] && [ -d "$ckdir" ]; then
        echo "=== $(date -Is) [$run] clearing partial checkpoints from an interrupted run: $ckdir"
        rm -rf "${ckdir:?}"
    fi

    FP="$(code_fingerprint)"
    REV="$(git_rev)"
    printf '%s\t%s\t%s\t%s\t%s\n' "$run" "${CFG%.json}" "$FP" "$REV" "$(date -Is)" >> "$MANIFEST"

    log="$LOGDIR/${run}_$(date +%Y%m%d_%H%M).log"
    echo "=== $(date -Is) [$run] train -> $log (fingerprint $FP, git $REV)"
    CUDA_VISIBLE_DEVICES="$GPU" python -u -m mvc.train "$CFG" > "$log" 2>&1
    rc=$?
    echo "=== $(date -Is) [$run] train rc=$rc"
    [ $rc -eq 0 ] || { echo "!!! $run FAILED, continuing to next model"; continue; }

    echo "=== $(date -Is) [$run] score_checkpoints"
    CUDA_VISIBLE_DEVICES="$GPU" python -u -m analysis.eval.score_checkpoints "$CFG" >> "$log" 2>&1
    echo "=== $(date -Is) [$run] score rc=$?"

    echo "=== $(date -Is) [$run] eval_traintest_split"
    CUDA_VISIBLE_DEVICES="$GPU" python -u -m analysis.eval.eval_traintest_split "$CFG" >> "$log" 2>&1
    echo "=== $(date -Is) [$run] traintest_detect rc=$?"

    echo "--- [$run] ---"
    grep -aA 8 'epoch   val_loss' "$log" | tail -10
done

echo "############ $(date -Is) TRAIN/TEST SPLIT SWEEP COMPLETE -- run: python -m analysis.sweeps.traintest_sweep_report"
