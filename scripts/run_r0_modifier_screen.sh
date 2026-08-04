#!/bin/bash
# Modifier screen on the R0 base (chan[96,128,160,192] res[0,0,1,1] dense160 early2,
# the full-trained winner at miss@FA5 8.4). Three SINGLE-modifier deltas, each a
# 30k-bounded 1-epoch run WITH the full eval (so we get each variant's 30k miss@FA to
# compare against R0-base's 30k baseline 14.2 -- val_detect_auroc saturates and can't
# rank these). Waits for the R1 full-train (PID passed as $1) to release the GPU first.
#
#   mix_r0w    + custom_wavelet_pools=[1]   (stage-1 lossless Haar pool; preserve fine detail)
#   mix_r0d    + custom_res_blocks=[0,1,2,3] (deeper, low-res stages; ~4.27M)
#   mix_r0dolp + DoLP=true                   (extra DoLP-derived input channel)
#
# Usage: run_r0_modifier_screen.sh <R1_PID>
set -uo pipefail
REPO="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
PY=/home/ammar/Documents/Programming/Magician/src/python/classifier/venv/bin/python3
cd "$REPO"; mkdir -p logs

R1PID="${1:-}"
if [ -n "$R1PID" ]; then
    echo "=== waiting for R1 full-train (PID $R1PID) to finish before screening ==="
    while kill -0 "$R1PID" 2>/dev/null; do sleep 120; done
    echo "=== R1 finished at $(date '+%F %T'); GPU free -> starting modifier screens ==="
    sleep 30
fi

for cfg in mix_r0w_custom mix_r0d_custom mix_r0dolp_custom; do
    echo ""
    echo ">>> [$(date '+%F %T')] SCREEN $cfg  (log=logs/$cfg.log)"
    "$PY" trainMagicianVisionClassifierTorch.py "$cfg.json" > "logs/$cfg.log" 2>&1
    rc=$?
    if [ $rc -eq 0 ]; then
        echo "<<< [$(date '+%F %T')] DONE $cfg (rc=0) -> ${cfg}_threshold_curve.json"
    else
        echo "!!! [$(date '+%F %T')] FAILED $cfg (rc=$rc) -- see logs/$cfg.log; continuing"
    fi
done
echo ""
echo "=== modifier screen finished: $(date '+%F %T') ==="
