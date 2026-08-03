#!/usr/bin/env python3
"""Phase 4: clean inference benchmark for the deployment candidates.

Two things this fixes about the numbers recorded so far.

1. Every Hz in ModelZoo's table was measured on a CONTENDED A6000 (a training job was
   running on the same card), so they understate throughput by an unknown amount.
   Run this with the GPU otherwise idle.

2. Those Hz assumed ~2175 tiles/frame. That number came from a note of mine, not from
   the deployment code. The real geometry is fixed by classifierPnm/liveClassifierTorch:
   a 1024x1224 4-channel frame (2048x2448 DoFP sensor, one sample per polarization
   angle per 2x2 macro-pixel), tile_size 48, step 16 -- which is 4588 tiles, over 2x the
   assumption. So this reports Hz at the ACTUAL geometry for each candidate step, and
   the legacy 2175 column only so the older table can be lined up against it.

Reparameterizable backbones (repvgg, mobileone) are timed BOTH ways: training runs the
multi-branch graph, deployment must call timm.utils.reparameterize_model, and the gap is
4-8x. The fused number is the one that matters for the deployment decision; the KPI is
unaffected because the fused model is mathematically equivalent.

Usage:
    CUDA_VISIBLE_DEVICES=2 python scripts/bench_inference.py
    CUDA_VISIBLE_DEVICES=2 python scripts/bench_inference.py --models convnext_atto lcnet_050
"""

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from ModelZoo import build_backbone, REPARAM_BACKBONES   # noqa: E402

# Deployment geometry, from classifierPnm.tile_and_cast_data_torch / liveClassifierTorch.
FRAME_H, FRAME_W = 1024, 1224
TILE = 48
STEPS = (16, 18, 24, 32)          # 16 = liveClassifierTorch default, 18 = recommended_configuration
LEGACY_TILES_PER_FRAME = 2175     # the unsourced assumption in the old table, for alignment
TARGET_HZ = 23.0


def tiles_per_frame(step, tile=TILE, h=FRAME_H, w=FRAME_W):
    """Exactly what classifierPnm counts: unfold(1,tile,step).unfold(2,tile,step)."""
    return ((h - tile) // step + 1) * ((w - tile) // step + 1)


def bench(model, in_ch, batch, device, iters=30, warmup=10, tile=TILE):
    model = model.to(device).eval().half()
    x = torch.randn(batch, in_ch, tile, tile, device=device, dtype=torch.half)
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            model(x)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
    ms = dt / iters * 1000.0
    return ms, batch / (dt / iters)      # ms/batch, tiles/s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', nargs='+', default=None, help='subset to benchmark')
    ap.add_argument('--batch', type=int, default=504)
    ap.add_argument('--in-channels', type=int, default=5, help='4 polarization + DoLP')
    ap.add_argument('--num-classes', type=int, default=12)
    ap.add_argument('--iters', type=int, default=30)
    ap.add_argument('--out', default='phase4_inference_bench.json')
    args = ap.parse_args()

    if not torch.cuda.is_available():
        sys.exit('no CUDA device visible')
    device = torch.device('cuda')
    print(f'device: {torch.cuda.get_device_name(0)}   batch={args.batch} '
          f'in_chans={args.in_channels} fp16\n')

    free, total = torch.cuda.mem_get_info()
    used = (total - free) / 1e9
    if used > 2.0:
        print(f'!! WARNING: {used:.1f} GB already in use on this GPU -- another job is '
              f'running and these numbers will be contended, exactly the problem this '
              f'benchmark exists to fix.\n')

    candidates = args.models or [
        'convnext_atto', 'convnext_femto', 'convnext_pico', 'convnext_nano',
        'lcnet_050', 'mobilenetv4_conv_small', 'edgenext_xx_small', 'ghostnet_100',
        'repvgg_a0', 'mobileone_s0', 'mobileone_s1', 'fastvit_t8',
        'efficientnet_b0', 'regnet_y_800mf', 'convnext_tiny', 'custom',
    ]

    geom = {s: tiles_per_frame(s) for s in STEPS}
    print('deployment geometry: %dx%d, tile %d' % (FRAME_H, FRAME_W, TILE))
    for s, n in geom.items():
        print(f'    step {s:2d} -> {n:6,d} tiles/frame'
              + ('   <- liveClassifierTorch default' if s == 16 else '')
              + ('   <- recommended_configuration' if s == 18 else ''))
    print(f'    (old table assumed {LEGACY_TILES_PER_FRAME:,} -- '
          f'{geom[16]/LEGACY_TILES_PER_FRAME:.2f}x optimistic vs step 16)\n')

    rows = []
    for name in candidates:
        try:
            model = build_backbone(name, in_channels=args.in_channels,
                                   num_classes=args.num_classes, pretrained=False,
                                   tile_size=TILE)
        except Exception as e:                      # noqa: BLE001 - report and continue
            print(f'  {name}: SKIP ({type(e).__name__}: {e})')
            continue
        params = sum(p.numel() for p in model.parameters()) / 1e6

        variants = [('unfused' if name in REPARAM_BACKBONES else 'plain', model)]
        if name in REPARAM_BACKBONES:
            import timm.utils
            variants.append(('FUSED', timm.utils.reparameterize_model(model)))

        for tag, m in variants:
            try:
                ms, tps = bench(m, args.in_channels, args.batch, device, iters=args.iters)
            except Exception as e:                  # noqa: BLE001
                print(f'  {name} [{tag}]: SKIP ({type(e).__name__}: {e})')
                continue
            rows.append({
                'model': name, 'variant': tag, 'params_M': round(params, 2),
                'ms_per_batch': round(ms, 2), 'tiles_per_s': round(tps),
                'hz_legacy_2175': round(tps / LEGACY_TILES_PER_FRAME, 1),
                **{f'hz_step{s}': round(tps / n, 1) for s, n in geom.items()},
            })
            del m
        del model
        torch.cuda.empty_cache()

    hdr = (f"\n{'model':24s} {'variant':8s} {'params':>7s} {'ms/bat':>7s} {'tiles/s':>9s} "
           f"{'Hz@2175':>8s} " + ' '.join(f'{"Hz@s"+str(s):>8s}' for s in STEPS))
    print(hdr)
    print('-' * len(hdr))
    for r in sorted(rows, key=lambda r: -r[f'hz_step{STEPS[0]}']):
        flag = '  <- meets 23 Hz' if r['hz_step16'] >= TARGET_HZ else ''
        print(f"{r['model']:24s} {r['variant']:8s} {r['params_M']:7.2f} "
              f"{r['ms_per_batch']:7.2f} {r['tiles_per_s']:9,d} {r['hz_legacy_2175']:8.1f} "
              + ' '.join(f"{r[f'hz_step{s}']:8.1f}" for s in STEPS) + flag)

    json.dump({'device': torch.cuda.get_device_name(0), 'batch': args.batch,
               'in_channels': args.in_channels, 'dtype': 'fp16',
               'frame': [FRAME_H, FRAME_W], 'tile': TILE,
               'tiles_per_frame': geom, 'rows': rows},
              open(args.out, 'w'), indent=2)
    print(f'\nwrote {args.out}')


if __name__ == '__main__':
    main()
