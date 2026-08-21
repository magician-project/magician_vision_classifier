#!/usr/bin/env bash
# Follow-up queue after Phases 1-4. Every run pins validation via
# dataloader.frozen_val_frames -> val_frames_frozen.json, which reproduces the exact split
# all recorded results used (verified bit-identical) and keeps these numbers comparable
# once the annotation effort grows the training set.
#
#   A. sv{1337,7,2024}_convnext_atto  30k screens differing ONLY in hparams.seed (~35 min ea)
#      Establishes the NOISE FLOOR the campaign has never had. Several live conclusions rest
#      on sub-point gaps -- convnext_atto 8.92 vs convnext_femto 8.93, and Phase 1a's 0.17
#      DoLP effect on atto -- and without a spread those are uninterpretable. With the val
#      split frozen this isolates training randomness (init, sampler order, augmentation,
#      cudnn nondeterminism); tz_convnext_atto (seed 42, 8.92) is the 4th sample.
#
#   B. ft_efficientnet_b0             full train, 4 epochs, top_k=4  (~11 h)
#      Phase 3's best accuracy-per-parameter: 6.98 at 4.02M, beating convnext_pico's 7.51
#      at 8.54M on a matched 30k screen. ~2.6 h/epoch (measured: 44 min for its 30k screen).
#
#   C. ft_convnext_pico_long          full train, 6 epochs, top_k=6  (~15 h)
#      p2 hit 4.37 with val_loss, AUROC and the KPI ALL still improving at its last epoch,
#      so 4.37 is an upper bound. Plain constant-LR AdamW (no scheduler), so extending the
#      epoch count is a clean extension of the same recipe, not a different one.
#
# Every full train is followed by score_checkpoints.py over all retained epochs, because
# the monitor picks one checkpoint and AUROC saturates.
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
run_cfg() {   # run_cfg <config.json> <tag>
    local cfg="$1" tag="$2"
    [ -f "$cfg" ] || { echo "!! missing $cfg, skipping"; return 0; }
    wait_for_gpu
    local log="$LOGDIR/${tag}_$(date +%Y%m%d_%H%M).log"
    echo "=== $(date -Is) [$tag] launching $cfg -> $log"
    CUDA_VISIBLE_DEVICES="$GPU" python -u -m mvc.train "$cfg" > "$log" 2>&1
    echo "=== $(date -Is) [$tag] $cfg exited rc=$?"
}
score() {     # score <config.json> <tag>
    wait_for_gpu
    echo "=== $(date -Is) [$2] scoring every retained epoch"
    CUDA_VISIBLE_DEVICES="$GPU" python -u -m analysis.eval.score_checkpoints "$1" \
        > "$LOGDIR/${2}_score_$(date +%Y%m%d_%H%M).log" 2>&1
    echo "=== $(date -Is) [$2] scoring exited rc=$?"
}

echo "############ A. SEED VARIANCE — the noise floor (4 samples incl. tz seed 42 = 8.92)"
for s in 1337 7 2024; do run_cfg "sv${s}_convnext_atto.json" "sv${s}"; done
python -m analysis.sweeps.seed_variance 2>&1 | tee "$LOGDIR/seed_variance_$(date +%Y%m%d_%H%M).txt"

echo "############ B. efficientnet_b0 FULL TRAIN (4 epochs)"
run_cfg ft_efficientnet_b0.json ft_effnet
score   ft_efficientnet_b0.json ft_effnet

echo "############ C. convnext_pico FULL TRAIN, 6 epochs"
run_cfg ft_convnext_pico_long.json ftl_pico
score   ft_convnext_pico_long.json ftl_pico

echo "############ $(date -Is) FOLLOW-UPS COMPLETE"
python -m analysis.datasets.phase2_select --dry-run 2>&1 | tail -30
