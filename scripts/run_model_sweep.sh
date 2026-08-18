#!/usr/bin/env bash
# STAGE 1 of the backbone sweep: 11 candidates, seed 42, full budget (2 epochs), Aug26_78K.
#
# See model_sweep.py for the design. In short: the existing tz bench's THROUGHPUT transfers
# but its ACCURACY does not (different dataset, leaky frame-disjoint split, 12 classes,
# 1 epoch), so every candidate is re-measured here; and because full-budget seed noise is
# ~1.8 miss@FA5, stage 1 can only ELIMINATE (a 5+ point gap is 3+ sd), never rank. Ranking
# the survivors is stage 2, at 3 seeds.
#
# ORDER IS BY PRIOR, STRONGEST FIRST, so that stopping the queue early stops on the
# candidates least likely to matter. Rough per-run estimates, scaling the incumbent's
# 4h24m by inverse relative speed on the TRAINING graph (reparam models train on the slow
# multi-branch graph, so mobileone is far more expensive to train than to deploy):
#
#   msfemto    ~3.6 h     msghost    ~7.9 h     mslcnet    ~2.3 h
#   msrepvgg   ~5.5 h     msedgex    ~6.1 h     msfastvit  ~9.0 h
#   msnano     ~6.3 h     msmone1   ~12.9 h     msmone0   ~19.8 h
#   msatto     ~6.0 h
#   msmnv4     ~4.5 h                           TOTAL     ~84 h
#
# 84 h is the honest cost of asking the question at a budget where the answer is readable.
# The first four runs (~21 h) cover every candidate with both a competitive prior AND
# throughput headroom over the incumbent; everything after that is due diligence on
# candidates the legacy bench already ranked 3-13 points worse.
#
# Restartable: a run whose _coverage.json exists is skipped. Safe to stop between runs.
set -u
GPU="${GPU:-2}"
LOGDIR=/storage/ammarkov/logs
cd "$(dirname "$0")/.." || exit 1
mkdir -p "$LOGDIR"

# tag:model, in queue order
RUNS=(
  msfemto:convnext_femto
  msrepvgg:repvgg_a0
  msnano:convnext_nano
  msatto:convnext_atto
  msmnv4:mobilenetv4_conv_small
  msghost:ghostnet_100
  msedgex:edgenext_xx_small
  msmone1:mobileone_s1
  mslcnet:lcnet_050
  msfastvit:fastvit_t8
  msmone0:mobileone_s0
)

# Serialise against the other queues. The first version of this guard watched only for a
# live trainer process, which let this sweep start a run while the seed-replicate queue was
# between runs in score_checkpoints.py / eval_coverage.py -- two trainers on GPU 2 at once
# on 2026-08-12. A process-name heuristic can only enumerate the states someone remembered;
# the state that broke it was the gap between them. Now: a real flock, plus an explicit
# wait for the seed-replicate queue, which started before the lock existed and so holds no
# lock of its own.
#
# Order is replicates -> pfc variance test -> this sweep. The pfc test gates how this
# sweep's results should be read (if pfc=4 collapses seed variance, stage 2's ~60 h is
# largely unnecessary and the sweep arguably belongs at pfc=4), so waiting on it by name
# makes the ordering deterministic instead of a race for the mutex.
. "$(dirname "$0")/gpu_lock.sh"
gpu_wait_legacy run_seed_replicates.sh run_pfc_variance.sh
gpu_wait

for entry in "${RUNS[@]}"; do
    run="${entry%%:*}"; model="${entry##*:}"
    CFG="${run}_${model}.json"
    [ -f "$CFG" ] || { echo "!!! missing $CFG, skipping"; continue; }
    if [ -f "${run}_${model}_coverage.json" ]; then
        echo "=== $(date -Is) $run already complete, skipping"
        continue
    fi

    log="$LOGDIR/${run}_$(date +%Y%m%d_%H%M).log"
    echo "=== $(date -Is) [$run] $model train -> $log"
    CUDA_VISIBLE_DEVICES="$GPU" python -u trainMagicianVisionClassifierTorch.py "$CFG" > "$log" 2>&1
    rc=$?
    echo "=== $(date -Is) [$run] train rc=$rc"
    # One bad backbone must not take the queue down with it.
    [ $rc -eq 0 ] || { echo "!!! $run FAILED, continuing to next candidate"; continue; }

    echo "=== $(date -Is) [$run] score_checkpoints"
    CUDA_VISIBLE_DEVICES="$GPU" python -u score_checkpoints.py "$CFG" >> "$log" 2>&1
    echo "=== $(date -Is) [$run] score rc=$?"

    echo "=== $(date -Is) [$run] eval_coverage"
    CUDA_VISIBLE_DEVICES="$GPU" python -u eval_coverage.py "$CFG" >> "$log" 2>&1
    echo "=== $(date -Is) [$run] coverage rc=$?"

    echo "--- [$run] factory (incumbent convnext_pico: 9.24 s42 / 7.41 s1337 miss@FA5) ---"
    grep -aA 8 'epoch   val_loss' "$log" | tail -10
    echo "--- [$run] coverage (incumbent TIER_A det@FA5: 73.59 s42 / 72.83 s1337) ---"
    grep -aA 4 'TIER_A (honest' "$log" | tail -5
done

echo "############ $(date -Is) MODEL SWEEP STAGE 1 COMPLETE -- run: python model_sweep_report.py"
