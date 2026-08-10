#!/bin/bash
# In-distribution backbone sweep on FORTH+Altinay (granular defect+severity, keep
# clean = detection + KPI). One run per backbone, SEQUENTIAL (one GPU job at a
# time, so heat is bounded and you can stop between models with nothing lost:
# each model independently writes mix_<bb>.pth / mix_<bb>_confusion.json /
# mix_<bb>_threshold_curve.json + the optimal thresholds embedded in mix_<bb>.json).
#
# Data: [train_nonaltinay_v2 + val_altinay_granular], frame_disjoint_split 0.1/seed42.
# Recipe FIXED across backbones (v2rot: pfc0.5, gain1.0, flip, chan0.4, rot);
# only `model` varies. Selection metric = val_detect_auroc. 4 epochs each.
# ~108 min/epoch (clean-driven balanced-sampler length) => ~7h/backbone, ~50h total.
#
# Usage:  scripts/run_mix_campaign.sh                # all 7, in order
#         scripts/run_mix_campaign.sh resnet18 custom   # only these
set -uo pipefail

REPO="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
PY=/home/ammar/Documents/Programming/Magician/src/python/classifier/venv/bin/python3
cd "$REPO"
mkdir -p logs

# Fast backbones first so the ranking signal lands early; heaviest (custom) last.
BACKBONES=("shufflenet_v2_x1_0" "resnet18" "mobilenet_v3_large" "efficientnet_b0" \
           "regnet_y_800mf" "convnext_tiny" "custom")
[ "$#" -gt 0 ] && BACKBONES=("$@")

echo "=== mix campaign: ${#BACKBONES[@]} backbone(s) -> ${BACKBONES[*]} ==="
echo "python: $PY"
for bb in "${BACKBONES[@]}"; do
    cfg="mix_${bb}.json"
    log="logs/mix_${bb}.log"
    if [ ! -f "$cfg" ]; then echo "SKIP $bb: missing $cfg"; continue; fi
    mkdir -p "/home/ammar/Documents/Programming/magician_datasets/mix_ckpts/${bb}"
    echo ""
    echo ">>> [$(date '+%F %T')] TRAIN $bb  (cfg=$cfg log=$log)"
    "$PY" trainMagicianVisionClassifierTorch.py "$cfg" > "$log" 2>&1
    rc=$?
    if [ $rc -eq 0 ]; then
        echo "<<< [$(date '+%F %T')] DONE $bb (rc=0)  ->  mix_${bb}.pth / mix_${bb}_confusion.json"
    else
        echo "!!! [$(date '+%F %T')] FAILED $bb (rc=$rc) -- see $log; continuing with next backbone"
    fi
done
echo ""
echo "=== campaign finished: $(date '+%F %T') ==="
