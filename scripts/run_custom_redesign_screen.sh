#!/bin/bash
# 1-epoch SCREEN of the custom-net redesign candidates (R0/R1/R2). The current
# `custom` wastes 79% of its params on a bloated FC head (fc1: 64->2304) that
# expands a 64-dim global-avg-pooled vector; these variants reallocate that
# budget into the conv backbone (custom_channels + custom_res_blocks) and cut
# final_dense_layer. Same data/recipe/seed as the mix campaign, so the resulting
# mix_r*_custom_threshold_curve.json miss@FA is directly comparable to custom 12.7.
#
#   R0 2.37M  same-budget rebalance   chan [96,128,160,160]  res [1,1,1,1]
#   R1 4.27M  moderate capacity       chan [96,128,192,192]  res [1,1,2,2]
#   R2 6.88M  larger (still <<convnext 27.8M, plain-CNN backbone)
#
# Sequential (one GPU job at a time). ~2h/run => ~6-7h total. After this, pick
# the winner and full-train it at 4 epochs.
set -uo pipefail
REPO="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
PY=/home/ammar/Documents/Programming/Magician/src/python/classifier/venv/bin/python3
cd "$REPO"
mkdir -p logs

for cfg in mix_r0_custom mix_r1_custom mix_r2_custom; do
    echo ""
    echo ">>> [$(date '+%F %T')] SCREEN $cfg  (log=logs/$cfg.log)"
    "$PY" trainMagicianVisionClassifierTorch.py "$cfg.json" > "logs/$cfg.log" 2>&1
    rc=$?
    if [ $rc -eq 0 ]; then
        echo "<<< [$(date '+%F %T')] DONE $cfg (rc=0) -> ${cfg}.pth / ${cfg}_threshold_curve.json"
    else
        echo "!!! [$(date '+%F %T')] FAILED $cfg (rc=$rc) -- see logs/$cfg.log; continuing"
    fi
done
echo ""
echo "=== screen finished: $(date '+%F %T') ==="
