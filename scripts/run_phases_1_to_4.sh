#!/usr/bin/env bash
# Phases 1-4 of the post-timm-sweep experiment plan, queued behind the running sweep.
#
#   Phase 1a  p1n_*   convnext_atto/pico WITHOUT DoLP, 30k
#             Every timm screen so far included DoLP, but the -2.22 that justified it
#             was measured on the from-scratch custom net. If a pretrained backbone
#             does not need it (or ImageNet stem statistics fight it) the whole sweep
#             is carrying a handicap and the next campaign's default is wrong.
#   Phase 1b  p1b_*   convnext_nano AND convnext_pico at 60k
#             A fixed 30k budget structurally favours fast-converging models. pico is
#             the control: if both improve, that is just "more steps", not evidence
#             about nano.
#   Phase 2   p2_*    full train of the Phase-1 winner, chosen by phase2_select.py's
#             pre-registered rule, then score_checkpoints.py over all 3 epochs.
#   Phase 3   p3_*    efficientnet_b0 / regnet_y_800mf / convnext_tiny re-screened at
#             30k with the pretrained stem SEEDED (ModelZoo._stem_like). Their existing
#             full-train numbers (5.9 / 6.2 / 4.00) were obtained with RANDOM stems while
#             timm adapts its own, so no timm-vs-torchvision claim is currently honest.
#   Phase 4   clean inference benchmark on an idle GPU + the real tiles/frame geometry.
#
# Phase 2 is the long pole (~8-11 h); everything else is ~30-60 min per run.
set -u
GPU="${GPU:-2}"
LOGDIR=/storage/ammarkov/logs
cd "$(dirname "$0")/.." || exit 1
mkdir -p "$LOGDIR"

running_trainers() {
    ps -eo pid,cmd --no-headers | awk '$2=="python" && $3=="-u" && $4 ~ /^train.*MagicianVisionClassifierTorch\.py$/ {print $1}'
}

# Independent of who launched them: never start while ANY trainer is alive, so this
# script can be started right now and will simply wait out the rest of the timm sweep.
wait_for_gpu() {
    while [ -n "$(running_trainers)" ]; do
        echo "=== $(date -Is) waiting, trainer alive: $(running_trainers | tr '\n' ' ')"
        sleep 120
    done
}

run_cfg() {   # run_cfg <config.json> <phase-tag>
    local cfg="$1" tag="$2"
    if [ ! -f "$cfg" ]; then echo "!! missing $cfg, skipping"; return 1; fi
    wait_for_gpu
    local log="$LOGDIR/${tag}_$(date +%Y%m%d_%H%M).log"
    echo "=== $(date -Is) [$tag] launching $cfg -> $log"
    CUDA_VISIBLE_DEVICES="$GPU" python -u -m mvc.train "$cfg" > "$log" 2>&1
    local rc=$?
    echo "=== $(date -Is) [$tag] $cfg exited rc=$rc"
    [ $rc -ne 0 ] && echo "!! $cfg FAILED (rc=$rc) -- continuing"
    return 0
}

echo "############ PHASE 1a: is DoLP still a win on a PRETRAINED backbone?"
run_cfg p1n_convnext_atto.json p1n_atto
run_cfg p1n_convnext_pico.json p1n_pico

echo "############ PHASE 1b: is the 30k ranking a step-budget artifact?"
run_cfg p1b_convnext_nano.json p1b_nano
run_cfg p1b_convnext_pico.json p1b_pico

echo "############ PHASE 1 results"
python -m analysis.datasets.phase2_select --dry-run 2>&1 | tee "$LOGDIR/phase1_summary_$(date +%Y%m%d_%H%M).txt"

echo "############ PHASE 2: full train of the Phase-1 winner"
wait_for_gpu
python -m analysis.datasets.phase2_select | tee "$LOGDIR/phase2_pick_$(date +%Y%m%d_%H%M).txt"
P2_CFG="$(ls -1t p2_*.json 2>/dev/null | head -1)"
if [ -z "${P2_CFG:-}" ]; then
    echo "!! phase2_select.py wrote no config -- skipping Phase 2"
else
    run_cfg "$P2_CFG" p2_fulltrain
    # AUROC picked the right epoch last time by a +0.00 margin and val_loss would have
    # picked one 0.75 miss@FA5 worse, so rank all 3 epochs on the KPI directly.
    echo "=== $(date -Is) [p2] scoring every retained epoch"
    wait_for_gpu
    CUDA_VISIBLE_DEVICES="$GPU" python -u -m analysis.eval.score_checkpoints "$P2_CFG" \
        > "$LOGDIR/p2_score_checkpoints_$(date +%Y%m%d_%H%M).log" 2>&1
    echo "=== $(date -Is) [p2] scoring exited rc=$?"
fi

echo "############ PHASE 3: torchvision re-screens with SEEDED pretrained stems"
run_cfg p3_efficientnet_b0.json p3_effnet
run_cfg p3_regnet_y_800mf.json  p3_regnet
run_cfg p3_convnext_tiny.json   p3_convnext_tiny

echo "############ PHASE 4: clean inference benchmark (idle GPU, real geometry)"
wait_for_gpu
CUDA_VISIBLE_DEVICES="$GPU" python -u -m analysis.sweeps.bench_inference \
    2>&1 | tee "$LOGDIR/phase4_bench_$(date +%Y%m%d_%H%M).log"

echo "############ $(date -Is) PHASES 1-4 COMPLETE"
echo "screens:"
python -m analysis.datasets.phase2_select --dry-run 2>&1 | tail -30
