#!/usr/bin/env bash
# Sequential 30k modifier screens on one GPU, compared against the in-situ
# R0 baseline (miss@FA5 13.52 / FA10 9.46, mix_r0_s30k_custom).
#
# r0d (res_blocks[0,1,2,3], 4.27M) is deliberately NOT in this list: R1 showed
# added capacity has poor returns at matched size, so pure-scaling variants are
# deprioritised. Add it back if that conclusion changes.
set -u

GPU="${GPU:-2}"
LOGDIR=/storage/ammarkov/logs
cd "$(dirname "$0")/.." || exit 1

for cfg in mix_r0w_custom.json mix_r0dolp_custom.json; do
    name="${cfg%_custom.json}"
    log="$LOGDIR/${name}_$(TZ=UTC date +%Y%m%d_%H%M).log"
    echo "=== $(TZ=UTC date -Is) launching $cfg on GPU $GPU -> $log"
    CUDA_VISIBLE_DEVICES="$GPU" python -u -m mvc.train "$cfg" > "$log" 2>&1
    rc=$?
    echo "=== $(TZ=UTC date -Is) $cfg exited rc=$rc"
    if [ $rc -ne 0 ]; then
        echo "!! $cfg FAILED (rc=$rc) -- continuing to the next screen anyway"
    fi
done
echo "=== $(TZ=UTC date -Is) all modifier screens done"
