#!/usr/bin/env python3
"""Detection/false-alarm AFTER the live node's spatial vote, not just the raw gate.

WHY THIS EXISTS
---------------
Every threshold curve this repo has ever produced (`score_checkpoints.py`,
`eval_coverage.py`, `eval_step_curve.py`, `eval_traintest_split.py`) scores tiles in
isolation. But `mvc/inference/classifier_pnm.py:process_predictions_erode()` -- which the
live node applies on every frame by default (`recommended_configuration.json`'s runtime
block: `erosion_kernel: 1, min_votes: 2`) -- keeps an activated tile only if at least
`min_votes` activated tiles (including itself) exist in its `(2*erosion_kernel+1)^2`
neighbourhood. Nothing has ever measured what that buys or costs: it should suppress
isolated false-alarm tiles, but it can just as easily erase a real defect that only lights
up 1-2 tiles -- exactly the PositiveDentClassB/C failure mode already visible in every
per-class table in this repo (see e.g. `knowledge/25-8-mono-vs-pol-report.md` §3).

This runs ONE deployment-step frame pass per frame (reusing `eval_step_curve.py`'s
frame-reading, labelling and tiling machinery exactly, so a mislabel here is not a second
place to get the annotator's conventions subtly wrong), caches each frame's mass grid, and
then sweeps GATE THRESHOLD x EROSION_KERNEL x MIN_VOTES in memory -- one inference pass per
frame, not one per combination.

KERNEL IS SWEPT TOO, NOT FIXED AT THE DEPLOYED DEFAULT
--------------------------------------------------------
`min_votes` only means something relative to its neighbourhood size: `min_votes=2` in a 3x3
window (9 cells, kernel=1) is a very different bar than `min_votes=2` in a 5x5 window (25
cells, kernel=2). Fixing kernel at the deployed default and only sweeping threshold x votes
cannot tell you whether that default is actually a good choice, or whether a different
neighbourhood size gives a strictly better tradeoff at the same or fewer required votes. The
per-(frame, threshold, kernel) neighbour count is an integral image, O(cells) regardless of
kernel size, and it is the once-per-frame inference pass that dominates cost either way -- so
this is cheap to add. `kernel=0` (a 1x1 "neighbourhood" -- only the tile itself) makes
`votes<=1` a no-op and `votes>=2` unsatisfiable by construction (a tile only ever counts
itself); that is not a bug, it is kernel=0 correctly reporting "no spatial context available".

WHY IT ONLY APPLIES TO CONFIGS WITH A REAL COVERAGE CARVE-OUT
---------------------------------------------------------------
This scores held-out FRAMES (`val_coverage_frames.json`, recording-disjoint), same as
`eval_coverage.py`. A config that removed `dataloader.exclude_frames` -- e.g. every run in
`analysis/sweeps/traintest_sweep.py` (`EXPERIMENT-traintest-split-sweep.md`) -- trained on
those frames, so scoring it here would be leaking, not measuring. `main()` refuses to run
against such a config rather than silently producing a leaky number.

THE MEASUREMENT
----------------
For a fixed step (default: the model's ships-at step, else 16), voted_mask = raw activation
AND (neighbour-count >= min_votes) -- exactly `process_predictions_erode`'s rule, vectorised
via an integral image per (frame, threshold) rather than the live path's per-tile Python
loop. A point is detected if ANY tile containing it survives the vote; false-alarm rate is
the fraction of the clean pool (same containment/quarantine exclusions as
`eval_step_curve.py`) that survives it. `min_votes <= 1` is a no-op by construction (an
activated tile always counts itself), matching the ROS service's own "0/1 disables voting"
contract -- so the votes=0 or 1 row of the output is exactly the existing no-voting curve.

Usage:
    python -m analysis.eval.eval_vote_curve <config.json> [checkpoint.ckpt] \\
        [--step 16] [--stride N] [--limit N] \\
        [--kernels 0,1,2] [--votes 0,1,2,3,4] [--thresholds 0.30:0.995:0.01]
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

from analysis.eval.eval_step_curve import (DEV_ROOT, QUARANTINE_PX, RAW_ROOT, TILE,
                                            frame_points, grid_origins, local_path)
from mvc.core.artifact_paths import find_artifact, out_path
from mvc.core.config import load_hyperparameters
from mvc.core.datasets import clean_class_index
from mvc.core.lit_classifier import Classifier
from mvc.core.read_data import readPolarPNMToRGBA
from mvc.inference.classifier_pnm import tile_and_cast_data_torch


def neighbor_counts(mask, kernel):
    """Per-cell count of True neighbours (including itself) in a (2k+1)^2 window.

    Integral image, not the live path's per-tile Python loop -- this runs once per
    (frame, threshold) across the whole sweep, so it has to be array-fast.
    """
    h, w = mask.shape
    m = mask.astype(np.int32)
    padded = np.zeros((h + 2 * kernel, w + 2 * kernel), dtype=np.int32)
    padded[kernel:kernel + h, kernel:kernel + w] = m
    ii = np.zeros((padded.shape[0] + 1, padded.shape[1] + 1), dtype=np.int64)
    ii[1:, 1:] = np.cumsum(np.cumsum(padded, axis=0), axis=1)
    k = 2 * kernel + 1
    # window covering padded[i:i+k, j:j+k] for original cell (i, j), i in [0,h), j in [0,w)
    total = ii[k:k + h, k:k + w] - ii[0:h, k:k + w] - ii[k:k + h, 0:w] + ii[0:h, 0:w]
    return total


def cache_frame(model, dev, rgba, points, step, clean_id):
    """One inference pass at `step`; returns everything the vote sweep needs, or None."""
    h, w = rgba.shape[:2]
    xs, ys = grid_origins(w, step), grid_origins(h, step)
    if not len(xs) or not len(ys):
        return None
    tiles = tile_and_cast_data_torch(rgba, tile_size=TILE, step=step)
    tiles = tiles.permute(0, 3, 1, 2).contiguous().to(dev)
    mass = np.empty(len(tiles), np.float32)
    CH = 2048
    with torch.no_grad():
        for s in range(0, len(tiles), CH):
            chunk = tiles[s:s + CH]
            with torch.amp.autocast(device_type=dev.type, dtype=torch.float16,
                                    enabled=(dev.type == 'cuda')):
                logits = model(chunk)
            prob = torch.softmax(logits.float(), dim=1)
            mass[s:s + len(chunk)] = (1.0 - prob[:, clean_id]).cpu().numpy()
    del tiles
    mass2d = mass.reshape(len(ys), len(xs))

    keep = np.ones((len(ys), len(xs)), bool)
    point_cells = []   # (iy_array, ix_array, cls) per annotated point
    for (px, py, cls) in points:
        cx = (xs <= px) & (px < xs + TILE)
        cy = (ys <= py) & (py < ys + TILE)
        contains = np.outer(cy, cx)
        keep &= ~contains
        dx = np.maximum(np.maximum(xs - px, 0), px - (xs + TILE - 1))
        dy = np.maximum(np.maximum(ys - py, 0), py - (ys + TILE - 1))
        d2 = dy[:, None] ** 2 + dx[None, :] ** 2
        keep &= ~((d2 > 0) & (d2 < QUARANTINE_PX ** 2))
        if cls is None:
            continue
        iy, ix = np.where(contains)
        point_cells.append((iy, ix, cls) if len(iy) else (None, None, cls))
    return {'mass2d': mass2d, 'keep': keep, 'points': point_cells,
            'tiles_per_frame': mass.size}


def parse_thresholds(spec):
    lo, hi, step = (float(x) for x in spec.split(':'))
    return np.round(np.arange(lo, hi, step), 4)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('config')
    ap.add_argument('checkpoint', nargs='?')
    ap.add_argument('--step', type=int, default=None,
                    help='deployment tiling step (default: ships-at step from the bench, else 16)')
    ap.add_argument('--stride', type=int, default=1, help='score every Nth frame')
    ap.add_argument('--limit', type=int, default=0, help='stop after N frames (0 = all)')
    ap.add_argument('--kernels', default='0,1,2',
                    help='erosion_kernel (neighbourhood radius) values to sweep; 1 is the '
                         'current deployed default (recommended_configuration.json)')
    ap.add_argument('--votes', default='0,1,2,3,4', help='min_votes values to sweep')
    ap.add_argument('--thresholds', default='0.30:0.995:0.01', help='lo:hi:step')
    ap.add_argument('--min-points', type=int, default=25)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    cfg = load_hyperparameters(args.config)

    if cfg['dataloader'].get('frozen_tile_split'):
        sys.exit(
            f"{args.config}: dataloader.frozen_tile_split is set -- this config removed "
            f"the coverage carve-out and trained on val_coverage_frames.json's frames "
            f"(see EXPERIMENT-traintest-split-sweep.md). Scoring the vote curve against "
            f"those frames here would be leakage, not measurement. This tool only applies "
            f"to configs with a real, held-out coverage carve-out (dataloader.exclude_frames).")
    if not cfg['dataloader'].get('exclude_frames'):
        sys.exit(f"{args.config}: no dataloader.exclude_frames -- no coverage carve-out to "
                 f"score against.")

    classes = cfg.get('classes')
    if not classes:
        sys.exit(f'{args.config} has no "classes" list -- score the run first')
    clean_id = clean_class_index(classes)
    if clean_id is None:
        sys.exit(f'no clean class among {classes}')
    merges = cfg.get('class_merges') or {}
    drops = set(cfg.get('drop_classes') or [])

    step = args.step
    if step is None:
        step = 16
        for fname in ('phase4_inference_bench.json', 'zoo_inference_bench.json'):
            p = find_artifact(fname)
            if not p:
                continue
            for r in json.load(open(p))['rows']:
                if r['model'] == cfg['model'] and r.get('hz_step16', 0) * 1.6 < 23.0:
                    step = 18   # matches the one non-16 deployment step used in this repo
            break
    votes_list = [int(v) for v in args.votes.split(',')]
    kernels = [int(k) for k in args.kernels.split(',')]
    thresholds = parse_thresholds(args.thresholds)

    p = find_artifact('val_coverage_frames.json')
    if not p:
        sys.exit('val_coverage_frames.json not found')
    wanted = sorted(set(json.load(open(p))['frames']))[::args.stride]
    if args.limit:
        wanted = wanted[:args.limit]
    print(f'[vote-curve] step {step}, kernels {kernels}, votes {votes_list}, '
          f'{len(thresholds)} thresholds, {len(wanted):,} coverage frames requested')

    import glob
    import re
    ckpt = args.checkpoint
    if ckpt is None:
        cks = sorted(glob.glob(os.path.join(cfg['checkpoint_dir'], '*.ckpt')))
        if not cks:
            sys.exit(f'no checkpoints in {cfg["checkpoint_dir"]}')
        mon = cfg.get('checkpoint_monitor', 'val_detect_auroc')
        best = 'max' if cfg.get('checkpoint_mode', 'max') == 'max' else 'min'
        scored = [(float(m.group(1)), c) for c in cks
                  if (m := re.search(rf'{re.escape(mon)}=([0-9]+\.[0-9]+)', c))]
        ckpt = (max if best == 'max' else min)(scored)[1] if scored else cks[-1]
    print(f'[vote-curve] {os.path.basename(ckpt)}')

    model = Classifier.from_config(cfg, num_classes=len(classes), clean_class=clean_id)
    model.load_state_dict(torch.load(ckpt, weights_only=False, map_location='cpu')['state_dict'])
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(dev).eval()

    frames = []
    missing = skipped = 0
    for i, fj in enumerate(wanted):
        img_path, json_path = local_path(fj)
        if img_path is None:
            missing += 1
            continue
        pts, no_light = frame_points(json_path, merges, drops, True)
        if pts is None or no_light:
            skipped += 1
            continue
        rgba = readPolarPNMToRGBA(img_path)
        if rgba is None or rgba.ndim != 3 or rgba.shape[2] != 4:
            missing += 1
            continue
        c = cache_frame(model, dev, rgba, pts, step, clean_id)
        if c is not None:
            frames.append(c)
        if i % 25 == 0:
            print(f'  {i:,}/{len(wanted):,} frames cached', end='\r')
    print()
    if missing:
        print(f'[vote-curve] WARNING {missing:,} frames unreadable or absent under {RAW_ROOT}')
    if skipped:
        print(f'[vote-curve] {skipped:,} frames skipped (no-light / unreadable json)')
    if not frames:
        sys.exit('no frames scored')

    from collections import defaultdict
    counts = defaultdict(int)
    for f in frames:
        for (_iy, _ix, cls) in f['points']:
            counts[cls] += 1
    macro_set = sorted(c for c, n in counts.items() if n >= args.min_points)
    tier_of = {}
    cov = find_artifact('val_coverage_frames.json')
    if cov:
        tier_of = {k: v['tier'] for k, v in json.load(open(cov))['classes'].items()}
    tier_a_set = [c for c in macro_set if tier_of.get(c) == 'TIER_A']
    tpf = float(np.mean([f['tiles_per_frame'] for f in frames]))
    print(f'[vote-curve] {len(frames):,} frames cached, ~{tpf:,.0f} tiles/frame, '
          f'macro over {len(macro_set)} classes ({len(tier_a_set)} TIER_A)')

    # --- the sweep ----------------------------------------------------------------------
    # masks depend only on threshold, not kernel -- computed once per threshold and shared
    # across every kernel in the loop below, so the kernel axis adds neighbour-count cost
    # only (integral image, O(cells)), not a second inference or mask pass.
    rows = []
    for t in thresholds:
        masks = [f['mass2d'] >= t for f in frames]
        for k in kernels:
            counts_per_frame = [neighbor_counts(m, k) for m in masks]
            for v in votes_list:
                voted = [m & (c >= v) for m, c in zip(masks, counts_per_frame)]
                fa_num = fa_den = 0
                per_class_hit = defaultdict(int)
                per_class_n = defaultdict(int)
                for f, vm in zip(frames, voted):
                    keep = f['keep']
                    fa_num += int(vm[keep].sum())
                    fa_den += int(keep.sum())
                    for (iy, ix, cls) in f['points']:
                        if cls not in macro_set:
                            continue
                        per_class_n[cls] += 1
                        if iy is not None and vm[iy, ix].any():
                            per_class_hit[cls] += 1
                fa = fa_num / max(fa_den, 1)
                per_class_det = {c: per_class_hit[c] / max(per_class_n[c], 1) for c in macro_set}
                macro = float(np.mean([per_class_det[c] for c in macro_set])) if macro_set else float('nan')
                macro_a = float(np.mean([per_class_det[c] for c in tier_a_set])) if tier_a_set else float('nan')
                rows.append({'threshold': round(float(t), 4), 'kernel': k, 'min_votes': v,
                             'false_alarm': fa, 'macro_detect': macro,
                             'macro_detect_tier_a': macro_a})

    # --- pick an operating point per (kernel, votes) pair: highest macro_detect_tier_a
    # inside a false-alarm budget matched to what the raw curve needs for ~1% FA, so the
    # picks are comparable to what score_checkpoints.py already reports.
    picks = {}
    for k in kernels:
        for v in votes_list:
            vrows = [r for r in rows if r['kernel'] == k and r['min_votes'] == v]
            in_budget = [r for r in vrows if r['false_alarm'] <= 0.01]
            picks[(k, v)] = max(in_budget, key=lambda r: r['macro_detect_tier_a']) if in_budget \
                else min(vrows, key=lambda r: r['false_alarm'])

    print(f"\n{'kernel':>6s} {'min_votes':>9s} {'threshold':>9s} {'FA%':>7s} {'macro%':>7s} "
          f"{'TIER_A%':>8s}   (best inside a 1% FA budget)")
    for k in kernels:
        for v in votes_list:
            p = picks[(k, v)]
            print(f"{k:6d} {v:9d} {p['threshold']:9.3f} {p['false_alarm']*100:7.3f} "
                  f"{p['macro_detect']*100:7.2f} {p['macro_detect_tier_a']*100:8.2f}")

    best = max(picks.values(), key=lambda r: r['macro_detect_tier_a'])
    print(f"\nbest overall inside the 1% FA budget: kernel={best['kernel']} "
          f"min_votes={best['min_votes']} threshold={best['threshold']:.3f} "
          f"-> TIER_A {best['macro_detect_tier_a']*100:.2f}%")

    deployed_key = (1, 2)   # recommended_configuration.json's current default
    no_vote_key = (1, 1)
    if deployed_key in picks and no_vote_key in picks:
        d, nv = picks[deployed_key], picks[no_vote_key]
        print(f"\ndeployed default (kernel=1, min_votes=2) vs no voting (kernel=1, "
              f"min_votes<=1), both at their own best-in-budget threshold:"
              f"\n  ΔFA     = {(d['false_alarm']-nv['false_alarm'])*100:+.3f} pts"
              f"\n  ΔTIER_A = {(d['macro_detect_tier_a']-nv['macro_detect_tier_a'])*100:+.2f} pts")

    out = args.out or out_path(f"{cfg['name']}_{cfg['model']}", '_vote_curve.json')
    json.dump({'config': os.path.basename(args.config), 'model': cfg['model'],
               'checkpoint': os.path.basename(ckpt), 'step': step, 'kernels': kernels,
               'frames_scored': len(frames), 'tiles_per_frame': tpf,
               'macro_classes': macro_set, 'macro_classes_tier_a': tier_a_set,
               'rows': rows,
               'picks': {f'{k}_{v}': picks[(k, v)] for k in kernels for v in votes_list}},
              open(out, 'w'), indent=1)
    print(f'\nwrote {out}')


if __name__ == '__main__':
    main()
