# Shared GPU serialisation for the queue runners.  source this, then call gpu_wait.
#
# WHY THIS EXISTS: on 2026-08-12 the model sweep started a training run while the
# seed-replicate queue was still going, and both landed on GPU 2 at once. The guard each
# runner carried looked only for a live `trainMagicianVisionClassifierTorch.py`, so during
# the ~5 minutes a queue spends in `score_checkpoints.py` and `eval_coverage.py` BETWEEN
# runs it reported the GPU idle and the next queue jumped in. Two trainers then shared the
# card, which corrupts every timing in PLAN.md and risks OOM while both preload 7.1M
# samples into RAM.
#
# The fix is a real mutex rather than a better process pattern: `flock` on a shared file.
# A process-name heuristic can only ever enumerate the states someone remembered to list,
# and the state that broke it was the one between the listed states.
#
# Usage in a runner:
#     . "$(dirname "$0")/gpu_lock.sh"
#     gpu_wait                      # blocks until this runner owns the GPU
#     ... run everything ...        # lock is held for the whole script, released on exit

GPU_LOCKFILE="${GPU_LOCKFILE:-/tmp/magician_gpu${GPU:-2}.lock}"

gpu_wait() {
    exec 9>"$GPU_LOCKFILE" || { echo "cannot open $GPU_LOCKFILE"; exit 1; }
    if ! flock -n 9; then
        echo "=== $(date -Is) GPU ${GPU:-2} held by another queue; waiting for the lock"
        flock 9
    fi
    echo "=== $(date -Is) acquired GPU ${GPU:-2} lock ($GPU_LOCKFILE), pid $$"
    # fd 9 stays open for the life of the script; the kernel drops the lock when the
    # process dies, so a killed runner cannot leave the queue wedged.
}

# Belt-and-braces for queues started BEFORE the lock existed: they hold no flock, so wait
# for their driver script to exit as well. Pass the script basenames to wait on.
#
# The pattern is ANCHORED to the start of the command line. `pgrep -f` matches anywhere in
# the full command line, so an unanchored name would also match any wrapper shell whose
# `-c` string happens to mention the script -- including the ones that launch these queues.
# A false match here does not fail loudly; it waits forever.
gpu_wait_legacy() {
    local pat
    for pat in "$@"; do
        while pgrep -f "^bash scripts/${pat}" > /dev/null 2>&1; do
            echo "=== $(date -Is) waiting for pre-lock queue ${pat} to finish"
            sleep 300
        done
    done
}
