#!/usr/bin/env bash
# Aug26 modifier sweep — 4 arms x 3 seeds, 30k screens, on the CORRECT 10-class label space.
#
# Re-asks the modifier questions under the campaign standard: recording-disjoint factory
# val, corrected 10-class scheme, coverage set carved OUT of train. The legacy mx* sweep
# ran on a frame-disjoint split with recording leakage, and the dev box's honest-split
# ablation reversed DoLP's point estimate -- but theirs ran on the broken 18-class default,
# so neither box has a clean answer yet.
#
#   base     4ch raw                 the reference (NOT 4ch+DoLP -- see aug26_sweep.py)
#   dolp     4ch + DoLP              the open cross-box disagreement
#   mono     4ch, all = their mean   what the polarization camera is worth, on the correct
#                                    label space, on the model that would ship (mean
#                                    replicated x4: same shape, zero polarimetric signal)
#   stride2  patchify stride 4 -> 2  the largest accuracy finding in the campaign and the
#                                    only one never tested on an honest split
#
# Each run is scored on BOTH validations immediately after it trains, so a stop at any
# point leaves a complete picture of the arms that finished rather than 12 unscored
# checkpoints. ~10 h training + ~2 h scoring.
set -u
GPU="${GPU:-2}"
LOGDIR=/storage/ammarkov/logs
cd "$(dirname "$0")/.." || exit 1
mkdir -p "$LOGDIR"

running_trainers() {
    ps -eo pid,cmd --no-headers | awk '$2=="python" && $3=="-u" && $4 ~ /^train.*MagicianVisionClassifierTorch\.py$/ {print $1}'
}
wait_for_gpu() {
    while [ -n "$(running_trainers)" ]; do
        echo "=== $(date -Is) waiting, trainer alive: $(running_trainers | tr '\n' ' ')"
        sleep 120
    done
}

python aug26_sweep.py || exit 1

# Seed-major: every arm gets a second seed before any gets a third, so an early stop still
# leaves paired comparisons rather than one arm at 3 seeds and the rest at 1. stride2 last
# within each seed because it costs 3.2x.
for seed in 42 1337 7; do
  for arm in base dolp mono stride2; do
    name="s26${arm}${seed}"
    cfg="${name}_convnext_pico.json"
    [ -f "$cfg" ] || { echo "=== $(date -Is) MISSING $cfg, skipping"; continue; }
    if [ -f "${name}_convnext_pico_coverage.json" ]; then
        echo "=== $(date -Is) [${arm}/${seed}] already complete, skipping"
        continue
    fi

    wait_for_gpu
    log="$LOGDIR/${name}_$(date +%Y%m%d_%H%M).log"
    echo "=== $(date -Is) [${arm}/${seed}] training -> $log"
    CUDA_VISIBLE_DEVICES="$GPU" python -u trainMagicianVisionClassifierTorch.py "$cfg" > "$log" 2>&1
    rc=$?
    echo "=== $(date -Is) [${arm}/${seed}] train exited rc=$rc"
    [ $rc -eq 0 ] || { echo "=== training failed, not scoring"; continue; }

    # Coverage only. The trainer already writes the factory threshold curve for its own
    # checkpoint, and these screens keep save_top_k=1, so there is no epoch sweep to do.
    echo "=== $(date -Is) [${arm}/${seed}] coverage"
    CUDA_VISIBLE_DEVICES="$GPU" python -u eval_coverage.py "$cfg" >> "$log" 2>&1
    echo "=== $(date -Is) [${arm}/${seed}] coverage exited rc=$?"
  done
  echo "############ after seed $seed:"
  python aug26_sweep_report.py 2>&1 | tail -40
done

echo "############ $(date -Is) AUG26 SWEEP COMPLETE"
python aug26_sweep_report.py 2>&1 | tee "$LOGDIR/aug26_sweep_$(date +%Y%m%d_%H%M).txt"
