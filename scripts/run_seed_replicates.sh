#!/usr/bin/env bash
# Full-budget seed replicates: anchor and stride-2, seeds 1337 and 7.
#
# Purpose (see seed_replicates.py for the full argument): every full-budget number in this
# campaign is n=1, including the stride-2 result of 9.24 -> 7.15 miss@FA5. The 30k screen
# cannot arbitrate it -- sd 1.83 on Aug26. This queue buys three things at once: the
# stride-2 verdict at n=3 PAIRED, the full-budget noise floor, and the per-class coverage
# sd that gates the ship rule on stride-2's -1.37 on PositiveDentClassB.
#
# ORDER IS SEED-MAJOR, ON PURPOSE. Each seed's anchor is followed immediately by its
# stride-2 twin, so the queue is at every point a COMPLETE set of pairs. Stopping after
# ~14 h leaves a clean n=2 paired comparison; stopping halfway through an arm-major order
# would leave three anchors and one stride-2 and answer nothing.
#
#   1  anc1337    ~4.2 h    2 epochs @ ~2h07
#   2  ancs21337  ~9.5 h    2 epochs @ ~4h45
#   3  anc7       ~4.2 h
#   4  ancs27     ~9.5 h
#                 ~27.4 h total
#
# Each run is train -> score_checkpoints (factory KPI, every epoch) -> eval_coverage
# (per-class detection at the monitored-best checkpoint). Restartable: a run whose
# _coverage.json already exists is skipped, so re-invoking after an interruption resumes.
set -u
GPU="${GPU:-2}"                       # GPU 2 is the reserved one on this box
LOGDIR=/storage/ammarkov/logs
cd "$(dirname "$0")/.." || exit 1
mkdir -p "$LOGDIR"

RUNS=(anc1337 ancs21337 anc7 ancs27)

# Don't contend with a trainer already on the box -- the anchor's timings assume the GPU
# to itself, and a shared GPU would silently make every duration in PLAN.md wrong.
running_trainers() {
    ps -eo pid,cmd --no-headers | awk '$2=="python" && $3=="-u" && $4 ~ /^train.*MagicianVisionClassifierTorch\.py$/ {print $1}'
}
while [ -n "$(running_trainers)" ]; do
    echo "=== $(date -Is) waiting, trainer alive: $(running_trainers | tr '\n' ' ')"
    sleep 120
done

for run in "${RUNS[@]}"; do
    CFG="${run}_convnext_pico.json"
    [ -f "$CFG" ] || { echo "!!! missing $CFG, skipping"; continue; }
    if [ -f "${run}_convnext_pico_coverage.json" ]; then
        echo "=== $(date -Is) $run already complete, skipping"
        continue
    fi

    log="$LOGDIR/${run}_$(date +%Y%m%d_%H%M).log"
    echo "=== $(date -Is) [$run] train -> $log"
    CUDA_VISIBLE_DEVICES="$GPU" python -u trainMagicianVisionClassifierTorch.py "$CFG" > "$log" 2>&1
    rc=$?
    echo "=== $(date -Is) [$run] train rc=$rc"
    [ $rc -eq 0 ] || { echo "!!! $run failed, continuing to next run"; continue; }

    # Per-epoch, because the epoch choice has to be made on the KPI itself: on the anchor,
    # val_detect_auroc picked the true best epoch while val_loss would have cost +1.67.
    echo "=== $(date -Is) [$run] score_checkpoints"
    CUDA_VISIBLE_DEVICES="$GPU" python -u score_checkpoints.py "$CFG" >> "$log" 2>&1
    echo "=== $(date -Is) [$run] score rc=$?"

    echo "=== $(date -Is) [$run] eval_coverage (monitored-best checkpoint)"
    CUDA_VISIBLE_DEVICES="$GPU" python -u eval_coverage.py "$CFG" >> "$log" 2>&1
    echo "=== $(date -Is) [$run] coverage rc=$?"

    echo "--- [$run] factory, per epoch (seed-42 reference: anc 9.24 / ancs2 7.15 miss@FA5) ---"
    grep -aA 8 'epoch   val_loss' "$log" | tail -10
    echo "--- [$run] coverage (seed-42 reference: anc 73.59 / ancs2 76.62 TIER_A det@FA5) ---"
    grep -aA 4 'TIER_A (honest' "$log" | tail -5
done

echo "############ $(date -Is) SEED REPLICATES COMPLETE -- run: python seed_replicates_report.py"
