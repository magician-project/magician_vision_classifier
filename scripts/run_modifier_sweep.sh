#!/usr/bin/env bash
# convnext_pico modifier sweep — 7 arms x 3 seeds, 30k screens, frozen val split.
#
# Design note: 3 seeds per arm is not optional here. A single 30k screen has sd 0.43
# miss@FA5, so one run per modifier cannot resolve anything below ~0.85 — and every
# modifier effect measured on a pretrained backbone so far has been smaller than that.
# Single-run modifier screens are what produced the Phase 1a and Phase 3 claims that had
# to be withdrawn. modifier_sweep.py does a PAIRED comparison (same seeds in both arms)
# so part of the run-to-run variance cancels.
#
#   base     DoLP only                    the reference arm
#   nodolp   no derived channel           re-opens the Phase 1a question properly
#   aolp     + AoLP                       never tested on a PRETRAINED backbone; it was
#                                         the worst modifier on the from-scratch net
#                                         (13.87 vs 13.52 baseline), so this asks whether
#                                         that was the channel or the small network
#   mmr      + Max/Min/Range              never tested at all; unlike AoLP these are
#                                         continuous and well-defined everywhere
#   unpol    + Unpolarized                expected to do little (a conv can form the mean)
#   mono     MONOCHROME                   the deliverable headline: what the polarization
#                                         camera is worth ON THE DEPLOYMENT MODEL. The
#                                         existing +0.0194 balanced-accuracy figure is
#                                         from convnext_tiny, a different architecture
#   stride2  patchify stride 4 -> 2       convnext_pico collapses 48px to 12x12 and then
#                                         to a 1x1 final map, so its last stage does no
#                                         spatial reasoning at all. Stride 2 keeps 24x24
#                                         (stages 24/12/6/3) with the pretrained kernel
#                                         copied verbatim. NOTE this arm costs 3.2x
#                                         (22.3 -> 7.1 Hz) and CANNOT ship at step 16/18
#                                         whatever it scores — it is here to tell us
#                                         whether the collapse is costing accuracy.
#
# ~19 new runs (2 seed-42 arms reuse existing identical runs). ~38 min each, except the
# stride2 arm at ~90 min. Expect ~14-15 h.
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

# Cheapest arms first so an early stop still leaves the most decision-relevant answers,
# and seed-major within each arm so every arm gets a second seed before any gets a third.
for seed in 42 1337 7; do
  for arm in base nodolp aolp mmr unpol mono stride2; do
    cfg="mx${arm}${seed}_convnext_pico.json"
    [ -f "$cfg" ] || { echo "=== skip ${arm}/${seed} (reusing an existing identical run)"; continue; }
    wait_for_gpu
    log="$LOGDIR/mx${arm}${seed}_$(date +%Y%m%d_%H%M).log"
    echo "=== $(date -Is) [${arm}/${seed}] launching $cfg -> $log"
    CUDA_VISIBLE_DEVICES="$GPU" python -u trainMagicianVisionClassifierTorch.py "$cfg" > "$log" 2>&1
    echo "=== $(date -Is) [${arm}/${seed}] exited rc=$?"
  done
  echo "############ after seed $seed:"
  python modifier_sweep.py 2>&1 | tail -22
done

echo "############ $(date -Is) MODIFIER SWEEP COMPLETE"
python modifier_sweep.py 2>&1 | tee "$LOGDIR/modifier_sweep_$(date +%Y%m%d_%H%M).txt"
