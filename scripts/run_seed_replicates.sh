#!/usr/bin/env bash
# Seed sweep (Experiment B): the 4 top-5 challengers the step curve (Experiment A, run
# 2026-08-21) did not rule out, seeds 1337 and 7. See
# EXPERIMENT-seed-sweep-and-step-curve.md Sec.5 for the full argument.
#
# convnext_pico (the incumbent) already has n=3 (anc, anc1337, anc7) and is not re-run here.
#
#   convnext_tiny      +3.51 coverage -- the recommendation itself
#   efficientnet_b0    +1.22
#   regnet_y_800mf     +0.80 -- marginal by design, the arm this sweep exists to settle
#   convnext_femto     -1.21 -- the density-arm control
#
# ORDER IS SEED-MAJOR, ON PURPOSE (same reasoning as the original anc/stride-2 sweep this
# script replaces): all four seed-1337 replicates run before any seed-7 replicate, so the
# queue is at every point a COMPLETE, EQUAL-N set across all four models. Stopping partway
# leaves n=2 for everyone rather than n=3 for some and n=1 for others.
#
#   1  fzcnxtiny1337   2  fzeffb01337   3  fzregy8001337   4  msfemto1337   (~1st pass)
#   5  fzcnxtiny7      6  fzeffb07      7  fzregy8007      8  msfemto7      (~2nd pass)
#                 ~40 h total (4 models x 2 seeds x ~5 h)
#
# Each run is train -> score_checkpoints (factory KPI, every epoch) -> eval_coverage
# (per-class detection at the monitored-best checkpoint). Restartable: a run whose
# coverage table already exists (root OR under experiments/, once tidied) is skipped.
set -u
GPU="${GPU:-2}"                       # GPU 2 is the reserved one on this box
LOGDIR=/storage/ammarkov/logs
cd "$(dirname "$0")/.." || exit 1
mkdir -p "$LOGDIR"
MANIFEST=experiments/seed_sweep_manifest.tsv

CFGS=(
    fzcnxtiny1337_convnext_tiny.json
    fzeffb01337_efficientnet_b0.json
    fzregy8001337_regnet_y_800mf.json
    msfemto1337_convnext_femto.json
    fzcnxtiny7_convnext_tiny.json
    fzeffb07_efficientnet_b0.json
    fzregy8007_regnet_y_800mf.json
    msfemto7_convnext_femto.json
)

# Files whose contents define the training protocol. Computed PER RUN (not once at driver
# launch) so each manifest line certifies the code state that run actually trained under --
# a mid-queue `git pull` changes the fingerprint for every run after it, and a fixed
# once-at-launch value would hide that instead of flagging it.
PROTOCOL_FILES="mvc/core/model_zoo.py mvc/train.py mvc/core/datasets.py mvc/core/lit_classifier.py analysis/eval/score_checkpoints.py analysis/eval/eval_coverage.py"
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

# A run is complete iff a coverage table exists, in the cwd OR under experiments/ (a
# finished run gets tidied away by tidy_experiments.py). find_artifact/exists() also
# already handles the `timm/`-in-model-name spelling, though none of these 8 models need
# that here.
coverage_exists() {
    local run="$1"
    python3 -c "
from mvc.core.artifact_paths import exists
import sys
sys.exit(0 if exists(sys.argv[1] + '_coverage.json') else 1)
" "$run"
}

# Don't contend with a trainer already on the box -- the ~5h/run estimate assumes the GPU
# to itself.
running_trainers() {
    ps -eo pid,cmd --no-headers | awk '$2=="python" && $3=="-u" && $4 ~ /^train.*MagicianVisionClassifierTorch\.py$/ {print $1}'
}
while [ -n "$(running_trainers)" ]; do
    echo "=== $(date -Is) waiting, trainer alive: $(running_trainers | tr '\n' ' ')"
    sleep 120
done

for CFG in "${CFGS[@]}"; do
    run="${CFG%.json}"
    [ -f "$CFG" ] || { echo "!!! missing $CFG, skipping"; continue; }
    if coverage_exists "$run"; then
        echo "=== $(date -Is) $run already complete, skipping"
        continue
    fi

    FP="$(code_fingerprint)"
    REV="$(git_rev)"
    printf '%s\t%s\t%s\t%s\t%s\n' "$run" "${CFG%.json}" "$FP" "$REV" "$(date -Is)" >> "$MANIFEST"

    log="$LOGDIR/${run}_$(date +%Y%m%d_%H%M).log"
    echo "=== $(date -Is) [$run] train -> $log (fingerprint $FP, git $REV)"
    CUDA_VISIBLE_DEVICES="$GPU" python -u -m mvc.train "$CFG" > "$log" 2>&1
    rc=$?
    echo "=== $(date -Is) [$run] train rc=$rc"
    [ $rc -eq 0 ] || { echo "!!! $run failed, continuing to next run"; continue; }

    # Per-epoch, because the epoch choice has to be made on the KPI itself.
    echo "=== $(date -Is) [$run] score_checkpoints"
    CUDA_VISIBLE_DEVICES="$GPU" python -u -m analysis.eval.score_checkpoints "$CFG" >> "$log" 2>&1
    echo "=== $(date -Is) [$run] score rc=$?"

    echo "=== $(date -Is) [$run] eval_coverage (monitored-best checkpoint)"
    CUDA_VISIBLE_DEVICES="$GPU" python -u -m analysis.eval.eval_coverage "$CFG" >> "$log" 2>&1
    echo "=== $(date -Is) [$run] coverage rc=$?"

    echo "--- [$run] factory, per epoch ---"
    grep -aA 8 'epoch   val_loss' "$log" | tail -10
    echo "--- [$run] coverage ---"
    grep -aA 4 'TIER_A (honest' "$log" | tail -5
done

echo "############ $(date -Is) SEED SWEEP COMPLETE -- run: python -m analysis.sweeps.seed_replicates_report"
