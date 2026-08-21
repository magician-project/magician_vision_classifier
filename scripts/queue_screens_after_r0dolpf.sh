#!/usr/bin/env bash
# Run the queued 30k screens on GPU 2 once the r0dolpf full-train finishes.
#   mix_r0da  = DoLP + AoLP          -> vs r0dolp's 11.30 (does AoLP add on top?)
#   mix_r0dws = DoLP + wavelet stem  -> vs r0dolp's 11.30 (does the 3.7x speedup cost accuracy?)
#
# Two lessons are baked in here:
#  1. Wait on the TRAINER's pid, not a grep of the command line. A `ps|grep <config>`
#     also matches the launching bash wrapper (whose eval string contains the config
#     name) and this script itself -- waiting on the wrapper fires the queue while
#     training is still running.
#  2. Independently of (1), refuse to start while ANY trainer is alive. Two runs on
#     one GPU would also mean 2x73GB of RAM cache.
set -u
TRAIN_PID=5571          # verified: python -u -m mvc.train mix_r0dolpf_custom.json

running_trainers() {
    ps -eo pid,cmd --no-headers | awk '$2=="python" && $3=="-u" && $4 ~ /^train.*MagicianVisionClassifierTorch\.py$/ {print $1}'
}

while kill -0 "$TRAIN_PID" 2>/dev/null; do sleep 120; done
echo "=== $(TZ=UTC date -Is) r0dolpf (pid $TRAIN_PID) finished"

cd /home/user/workspace/magician_vision_classifier || exit 1
for cfg in mix_r0da_custom.json mix_r0dws_custom.json; do
    # safety net: never run two trainers at once
    while [ -n "$(running_trainers)" ]; do
        echo "=== $(TZ=UTC date -Is) waiting, trainer(s) still alive: $(running_trainers | tr '\n' ' ')"
        sleep 120
    done
    name="${cfg%_custom.json}"
    log=/storage/ammarkov/logs/${name}_$(TZ=UTC date +%Y%m%d_%H%M).log
    echo "=== $(TZ=UTC date -Is) launching $cfg -> $log"
    CUDA_VISIBLE_DEVICES=2 python -u -m mvc.train "$cfg" > "$log" 2>&1
    echo "=== $(TZ=UTC date -Is) $cfg exited rc=$?"
done
echo "=== $(TZ=UTC date -Is) all queued screens done"
