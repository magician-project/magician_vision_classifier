#!/bin/bash
# Post-R1 screen queue on the R0 base. Runs the TWO-HEAD experiment FIRST (user's
# priority: type+severity heads, expected to help all backbones incl. convnext), then
# the wavelet / deeper / DoLP modifier screens. All are 30k-bounded 1-epoch runs WITH
# full eval, so each yields a real 30k miss@FA comparable to R0-base's 30k baseline 14.2.
# The two-head run uses the SEPARATE fork train2HeadMagicianVisionClassifierTorch.py;
# the modifiers use the original trainer. Waits for the R1 full-train (PID $1) first.
#
# Usage: run_r0_experiments_screen.sh <R1_PID>
set -uo pipefail
REPO="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
PY=/home/ammar/Documents/Programming/Magician/src/python/classifier/venv/bin/python3
cd "$REPO"; mkdir -p logs

R1PID="${1:-}"
if [ -n "$R1PID" ]; then
    echo "=== waiting for R1 full-train (PID $R1PID) to release the GPU ==="
    while kill -0 "$R1PID" 2>/dev/null; do sleep 120; done
    echo "=== R1 finished at $(date '+%F %T'); starting R0 experiment screens ==="
    sleep 30
fi

# cfg:script pairs — two-head uses the fork, modifiers use the original trainer
run_one () {
    local cfg="$1" script="$2"
    echo ""
    echo ">>> [$(date '+%F %T')] SCREEN $cfg  (via $script, log=logs/$cfg.log)"
    "$PY" "$script" "$cfg.json" > "logs/$cfg.log" 2>&1
    local rc=$?
    if [ $rc -eq 0 ]; then
        echo "<<< [$(date '+%F %T')] DONE $cfg (rc=0) -> ${cfg}_threshold_curve.json"
    else
        echo "!!! [$(date '+%F %T')] FAILED $cfg (rc=$rc) -- see logs/$cfg.log; continuing"
    fi
}

run_one mix_r0_2head_custom  train2HeadMagicianVisionClassifierTorch.py
run_one mix_r0w_custom       trainMagicianVisionClassifierTorch.py
run_one mix_r0d_custom       trainMagicianVisionClassifierTorch.py
run_one mix_r0dolp_custom    trainMagicianVisionClassifierTorch.py

echo ""
echo "=== R0 experiment screens finished: $(date '+%F %T') ==="
