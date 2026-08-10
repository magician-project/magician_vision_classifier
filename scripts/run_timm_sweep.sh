#!/usr/bin/env bash
# 30k screens of the timm small/edge backbones, DoLP input, ImageNet-pretrained.
#
# AoLP is deliberately NOT used: it measured worse than having no derived channel at
# all on the custom net (13.87 vs 13.52 baseline, vs 11.30 for DoLP alone) because it
# is an angle that wraps at +-pi/2 and is undefined wherever DoLP ~ 0.
#
# Ordered most-promising first so stopping early still leaves the best candidates
# measured. Compare against: custom+DoLP screen 11.30 / full-train 7.02;
# effnet_b0 5.9 and regnet_y_800mf 6.2 (full trains, pretrained, but with RANDOM stems
# -- these timm runs adapt the pretrained stem via in_chans, so they have a fair edge
# until the torchvision stem-seeding fix lands).
set -u
GPU="${GPU:-2}"
LOGDIR=/storage/ammarkov/logs
cd "$(dirname "$0")/.." || exit 1

MODELS=(convnext_atto convnext_femto convnext_pico convnext_nano
        mobilenetv4_conv_small edgenext_xx_small lcnet_050 ghostnet_100
        repvgg_a0 fastvit_t8 mobileone_s1 mobileone_s0)

running_trainers() {
    ps -eo pid,cmd --no-headers | awk '$2=="python" && $3=="-u" && $4 ~ /^train.*MagicianVisionClassifierTorch\.py$/ {print $1}'
}

for m in "${MODELS[@]}"; do
    cfg="tz_${m}.json"
    [ -f "$cfg" ] || { echo "!! missing $cfg, skipping"; continue; }
    while [ -n "$(running_trainers)" ]; do
        echo "=== $(TZ=UTC date -Is) waiting, trainer already alive: $(running_trainers | tr '\n' ' ')"
        sleep 120
    done
    log="$LOGDIR/tz_${m}_$(TZ=UTC date +%Y%m%d_%H%M).log"
    echo "=== $(TZ=UTC date -Is) launching $m -> $log"
    CUDA_VISIBLE_DEVICES="$GPU" python -u trainMagicianVisionClassifierTorch.py "$cfg" > "$log" 2>&1
    rc=$?
    echo "=== $(TZ=UTC date -Is) $m exited rc=$rc"
    [ $rc -ne 0 ] && echo "!! $m FAILED (rc=$rc) -- continuing with the rest"
done
echo "=== $(TZ=UTC date -Is) timm sweep complete"
