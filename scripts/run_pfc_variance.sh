#!/usr/bin/env bash
# penalize_false_clean = 4, three seeds, full budget. A VARIANCE experiment.
#
# See pfc_variance.py for the argument. Short version: the dev box measures seed sd 0.10 at
# pfc=4 (twice, on two unrelated experiments); I measure sd 1.29 at pfc=0.5 on the same
# model and dataset, and that variance just cost me a headline result when stride 4->2
# reversed sign between seeds. If pfc=4 genuinely collapses seed variance it fixes the
# measurement problem that makes every n=1 comparison in this campaign uninterpretable.
#
# THIS RUNS BEFORE THE BACKBONE SWEEP ON PURPOSE. The sweep is ~84 h at pfc=0.5 and needs a
# ~60 h stage 2 at 3 seeds to be readable. If variance collapses at pfc=4, single-seed
# comparisons become interpretable, stage 2 is largely unnecessary, and the sweep should
# arguably run at pfc=4 in the first place. ~13 h here to decide how to spend ~144 h there.
#
# Interest is the WITHIN-ARM sd, not the mean. The comparison is against the anchor's own
# three seeds at pfc=0.5, which the seed-replicate queue is producing.
set -u
GPU="${GPU:-2}"
LOGDIR=/storage/ammarkov/logs
cd "$(dirname "$0")/.." || exit 1
mkdir -p "$LOGDIR"

RUNS=(pfc4s42 pfc4s1337 pfc4s7)
MODEL=convnext_pico

# Serialise: wait out the seed-replicate queue (started before the lock existed, so it
# holds no flock), then take the lock for the whole script. The model sweep waits on this
# script by name, which makes the order replicates -> pfc -> sweep deterministic rather
# than a race for the mutex.
. "$(dirname "$0")/gpu_lock.sh"
gpu_wait_legacy run_seed_replicates.sh
gpu_wait

for run in "${RUNS[@]}"; do
    CFG="${run}_${MODEL}.json"
    [ -f "$CFG" ] || { echo "!!! missing $CFG, skipping"; continue; }
    if [ -f "${run}_${MODEL}_coverage.json" ]; then
        echo "=== $(date -Is) $run already complete, skipping"
        continue
    fi

    log="$LOGDIR/${run}_$(date +%Y%m%d_%H%M).log"
    echo "=== $(date -Is) [$run] train -> $log"
    CUDA_VISIBLE_DEVICES="$GPU" python -u -m mvc.train "$CFG" > "$log" 2>&1
    rc=$?
    echo "=== $(date -Is) [$run] train rc=$rc"
    [ $rc -eq 0 ] || { echo "!!! $run FAILED, continuing"; continue; }

    echo "=== $(date -Is) [$run] score_checkpoints"
    CUDA_VISIBLE_DEVICES="$GPU" python -u -m analysis.eval.score_checkpoints "$CFG" >> "$log" 2>&1
    echo "=== $(date -Is) [$run] score rc=$?"

    echo "=== $(date -Is) [$run] eval_coverage"
    CUDA_VISIBLE_DEVICES="$GPU" python -u -m analysis.eval.eval_coverage "$CFG" >> "$log" 2>&1
    echo "=== $(date -Is) [$run] coverage rc=$?"

    echo "--- [$run] factory (pfc=0.5 anchor: 9.24 s42 / 7.41 s1337, sd 1.29) ---"
    grep -aA 8 'epoch   val_loss' "$log" | tail -10
done

echo "############ $(date -Is) PFC VARIANCE TEST COMPLETE -- run: python -m analysis.sweeps.pfc_variance_report"
