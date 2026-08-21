#!/bin/bash
# Backbone sweep on the WINNING cross-site recipe (2026-07-13):
#   v2 dataset + pfc0.5 + gain_jitter 1.0 + polar_flip + channel_jitter 0.4 + polar_rot
# Backbone is a CLI arg to one config; outputs are crossvalv2rot_<backbone>_*.
# Ordered cheapest-first so early results land before the heavy models finish.
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"; cd ..
source venv/bin/activate

CFG=configs/crossval_v2_rot.json
# 2026-07-13/14 verdict: resnet18 dominates all backbones on the v2rot recipe;
# custom stems fail (fresh-stem vs epoch-0-best confound). Kept list = what ran.
for MODEL in mobilenet_v3_large regnet_y_800mf convnext_tiny; do  # efficientnet_v2_s/densenet121/swin_v2_t cut (heavy, resnet18 already wins)
    # skip if already trained (resumable)
    if [ -f "crossvalv2rot_${MODEL}_confusion.json" ]; then
        echo "=== SKIP ${MODEL} (already done) ==="
        continue
    fi
    echo "=== ${MODEL} start $(date) ==="
    python3 -m mvc.train "$CFG" "$MODEL" || echo "${MODEL} FAILED exit $?"
    echo "=== ${MODEL} done $(date) ==="
done
echo "=== BACKBONE SWEEP COMPLETE $(date) ==="
