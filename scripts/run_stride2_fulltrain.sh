#!/usr/bin/env bash
# stride2 FULL TRAIN on Aug26_78K — the one arm the 30k sweep earned a full budget for.
#
# The 30k screen on Aug26 has sd 1.83 miss@FA5 and resolves nothing (PLAN.md). What it DID
# show, by rank rather than by mean, is that stride 4->2 occupies val_loss ranks 1-3 of all
# 12 runs (p = 0.45%) while being invisible at the FA5 operating point. This run asks the
# question on the KPI, at the budget that matters.
#
# Configured to differ from the anchor (`anc`, 9.24 miss@FA5) in the STEM STRIDE ALONE:
# same seed 42, same 4ch+DoLP input, same 10-class scheme, same coverage carve-out. The
# anchor is a 5-channel run -- DoLP is on -- so this one keeps DoLP on too. Changing both
# would make the result uninterpretable.
#
# 3 epochs, save_top_k=3: the anchor peaked at epoch 1 and fell monotonically after, so
# three epochs covers the peak with a margin. At ~3.2x the anchor's per-step cost that is
# roughly 6-7 h/epoch, ~20 h total. Safe to stop after any epoch -- checkpoints are scored
# independently, and score_checkpoints.py ranks whatever exists.
set -u
GPU="${GPU:-2}"
LOGDIR=/storage/ammarkov/logs
cd "$(dirname "$0")/.." || exit 1
mkdir -p "$LOGDIR"

CFG=ancs2_convnext_pico.json
[ -f "$CFG" ] || { echo "missing $CFG"; exit 1; }

running_trainers() {
    ps -eo pid,cmd --no-headers | awk '$2=="python" && $3=="-u" && $4 ~ /^train.*MagicianVisionClassifierTorch\.py$/ {print $1}'
}
while [ -n "$(running_trainers)" ]; do
    echo "=== $(date -Is) waiting, trainer alive: $(running_trainers | tr '\n' ' ')"
    sleep 120
done

log="$LOGDIR/ancs2_$(date +%Y%m%d_%H%M).log"
echo "=== $(date -Is) stride2 full train -> $log"
CUDA_VISIBLE_DEVICES="$GPU" python -u trainMagicianVisionClassifierTorch.py "$CFG" > "$log" 2>&1
rc=$?
echo "=== $(date -Is) train exited rc=$rc"
[ $rc -eq 0 ] || exit $rc

# Per-epoch scoring: val_detect_auroc picked the true best epoch on the anchor while
# val_loss would have cost +1.67, so the epoch choice has to be made on the KPI itself.
echo "=== $(date -Is) scoring all checkpoints on the factory val"
CUDA_VISIBLE_DEVICES="$GPU" python -u score_checkpoints.py "$CFG" >> "$log" 2>&1
echo "=== $(date -Is) score exited rc=$?"

echo "=== $(date -Is) coverage (picks the monitored-best checkpoint)"
CUDA_VISIBLE_DEVICES="$GPU" python -u eval_coverage.py "$CFG" >> "$log" 2>&1
echo "=== $(date -Is) coverage exited rc=$?"

echo "############ $(date -Is) STRIDE2 FULL TRAIN COMPLETE"
echo "--- factory, per epoch (anchor reference: 9.24 miss@FA5 / 4.93 miss@FA10) ---"
grep -aA 12 'epoch   val_loss' "$log" | tail -14
echo "--- coverage (anchor reference: TIER_A det@FA5 73.59%) ---"
grep -aA 4 'TIER_A (honest' "$log" | tail -5
