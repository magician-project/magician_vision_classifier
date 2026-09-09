#!/usr/bin/env python3
"""Cascade ensemble (eval_cascade_ensemble.py) x deployment tiling STEP -- the axis that
tool never touched.

WHY THIS EXISTS
----------------
`eval_cascade_ensemble.py` measures the screen-then-recheck cascade
(`mvc.inference.ensemble_classifier.EnsembleClassifierPnm`) entirely from the coverage
campaign's pre-tiled held-out TILES -- which were extracted at whatever single step the
dataset dump used. That silently assumes stage 1 (the cheap screen) and stage 2 (the
expensive recheck) should tile at the SAME step. There is no reason they should: a screen
only has to notice a defect is SOMEWHERE nearby, a coarser grid costs it little; the
recheck stage only pays for the promoted subset, so it is the one place a FINER grid (more
chances to catch a small/weak defect, `eval_step_curve.py`'s whole finding) is nearly free.
Independently varying (step1, step2) is the natural next question and untested until now.

This can't be done from the pre-tiled h5 (`eval_step_curve.py`'s own finding: defect tiles
in the dump sit at ~arbitrary offsets, not a step-S grid -- only 0.5% land on one by chance).
So, like `eval_vote_curve.py` and `eval_step_curve.py`, this re-tiles RAW FRAMES itself, at
whatever step is asked for.

MECHANISM
---------
For each candidate step, one inference pass per model produces a full per-frame mass GRID
(`eval_vote_curve.cache_frame`, reused verbatim) -- cached to disk once per (run, step) and
shared across every pairing/threshold combo that needs it, exactly like every other cache in
this repo.

Combining two DIFFERENT grids (step1 != step2 -> different tile counts/positions over the
same frame) needs a geometric mapping, computed ONCE per (step1, step2) pair (pure geometry,
independent of frame content or threshold) via a containment matrix per axis: which step1
tile's box contains the CENTER of each step2 tile. Because the tile grid is separable (square
tiles, uniform steps), the 2-D eligibility mask is two matrix multiplies:

    eligible2[y2,x2] = OR over (y1,x1) promoted1[y1,x1] AND y1-contains-y2 AND x1-contains-x2
                      = (y_contain.T @ promoted1 @ x_contain) > 0

Then activated2 = eligible2 & (mass2 >= t2) is exactly the cascade's decision at (step1,
step2, t1, t2) -- no raw-pixel rasterising needed, and it is cheap enough to sweep t1 x t2 x
step1 x step2 entirely in memory once the per-(run,step) caches exist.

METRIC -- point detection, not the h5's tile-level TP/TN/FP/FN
----------------------------------------------------------------
This inherits `eval_vote_curve.py`'s convention, not `eval_ensemble.py`'s: a point is
detected if ANY step-2 tile containing it is activated; false-alarm is the fraction of the
(quarantine/annotation-excluded) clean grid cells that activate. These are POINTS and TILES
respectively -- do not average them against eval_ensemble.py's coverage-h5 tile-level
TP/TN/FP/FN, which is a different sampling universe (independent held-out tiles, not a frame
grid) and cannot be step-varied at all (see eval_step_curve.py's docstring for why).

Usage:
    python -m analysis.eval.eval_cascade_step_sweep <stage1_run> <stage2_run> [<run2> ...] \\
        [--steps1 16,18,24,32] [--steps2 16,18,24,32] [--stride N] [--limit N] [--cache-only]
"""

import argparse
import json
import os
import pickle
import sys
from collections import defaultdict

import numpy as np
import torch

from analysis.eval.eval_ensemble import best_ckpt, full_coverage_pool
from analysis.eval.eval_step_curve import (DEV_ROOT, QUARANTINE_PX, RAW_ROOT, TILE,
                                            frame_points, grid_origins, local_path)
from analysis.eval.eval_vote_curve import cache_frame
from mvc.core.artifact_paths import find_artifact, find_config_with_classes, out_path
from mvc.core.config import load_hyperparameters
from mvc.core.datasets import clean_class_index
from mvc.core.lit_classifier import Classifier
from mvc.core.read_data import readPolarPNMToRGBA

CACHE_DIR = 'experiments/cascade_step_cache'
FRAME_H, FRAME_W = 1024, 1224          # deployment geometry, see bench_inference.py
STEPS = (16, 18, 24, 32)               # the only steps the deployment tiler actually offers
TARGET_HZ = 23.0

# Screening t1 stays permissive (a screen's job is to not miss defects, not to be precise);
# recheck t2 covers the same "final decision" range eval_cascade_ensemble.py used.
T1_GRID = np.round(np.arange(0.05, 0.65, 0.05), 3)
T2_GRID = np.round(np.arange(0.30, 0.99, 0.03), 3)


def bench_hz_table():
    """model -> {step: 5090-estimated Hz}, straight from the bench JSONs' hz_stepN fields --
    all four steps come from ONE benched tiles/s number (compute-bound: tiles/s does not
    depend on step, only how many tiles a frame needs does), so this costs no extra GPU time."""
    table = {}
    for fname in ('phase4_inference_bench.json', 'zoo_inference_bench.json'):
        p = find_artifact(fname)
        if not p:
            continue
        for r in json.load(open(p))['rows']:
            if r['model'] not in table or r.get('variant', '').lower() == 'fused':
                table[r['model']] = {s: r.get(f'hz_step{s}', 0.0) * 1.6 for s in STEPS}
    return table


def load_model(run, model_name):
    sfx = model_name.replace('/', '_')
    cfg_path = find_config_with_classes(f'{run}_{sfx}.json')
    if not cfg_path:
        sys.exit(f'{run}_{sfx}: no config found')
    cfg = load_hyperparameters(cfg_path)
    if cfg['dataloader'].get('frozen_tile_split'):
        sys.exit(f'{run}: trained on the leaky in-distribution split (frozen_tile_split) -- '
                 f'scoring it against held-out coverage frames would be leakage.')
    if not cfg['dataloader'].get('exclude_frames'):
        sys.exit(f'{run}: no coverage carve-out (dataloader.exclude_frames)')
    classes = cfg.get('classes')
    if not classes:
        sys.exit(f'{cfg_path}: no "classes" -- score the run first')
    clean_id = clean_class_index(classes)
    merges = cfg.get('class_merges') or {}
    drops = set(cfg.get('drop_classes') or [])
    ckpt = best_ckpt(cfg)
    if ckpt is None:
        sys.exit(f'{run}_{sfx}: no checkpoint')
    net = Classifier.from_config(cfg, num_classes=len(classes), clean_class=clean_id)
    net.load_state_dict(torch.load(ckpt, weights_only=False, map_location='cpu')['state_dict'])
    return net, classes, clean_id, merges, drops


def wanted_frames(stride, limit):
    p = find_artifact('val_coverage_frames.json')
    if not p:
        sys.exit('val_coverage_frames.json not found')
    wanted = sorted(set(json.load(open(p))['frames']))[::stride]
    return wanted[:limit] if limit else wanted


def cache_path(run, model_name, step, n_wanted):
    # n_wanted (not stride/limit) in the filename: what actually determines the cached
    # content is WHICH/HOW MANY frames were scored, and two different --stride/--limit
    # combinations that happen to select the same frame count would otherwise silently
    # collide -- caught live when a 15-frame smoke-test cache was reused for a 200-frame run.
    return os.path.join(CACHE_DIR,
                         f'{run}_{model_name.replace("/", "_")}_step{step}_n{n_wanted}.pkl')


def cache_run_step(run, model_name, step, wanted, dev):
    out = cache_path(run, model_name, step, len(wanted))
    if os.path.exists(out):
        return out
    print(f'--- caching {run} ({model_name}) @ step {step} ---')
    net, classes, clean_id, merges, drops = load_model(run, model_name)
    net.to(dev).eval()
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
        if rgba.shape[:2] != (FRAME_H, FRAME_W):
            sys.exit(f'{img_path}: frame shape {rgba.shape[:2]} != assumed {(FRAME_H, FRAME_W)} '
                     f'-- the containment-matrix geometry below assumes a fixed frame size, '
                     f'fix that assumption before trusting any cross-step combination.')
        c = cache_frame(net, dev, rgba, pts, step, clean_id)
        if c is not None:
            c['mass2d'] = c['mass2d'].astype(np.float16)   # cache is disk-bound, not accuracy-bound
            frames.append(c)
        if i % 50 == 0:
            print(f'  {i:,}/{len(wanted):,} frames cached', end='\r')
    print()
    if missing or skipped:
        print(f'  ({missing:,} missing/unreadable, {skipped:,} no-light/unreadable json)')
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(out, 'wb') as fh:
        pickle.dump({'classes': classes, 'clean_id': clean_id, 'frames': frames}, fh, protocol=4)
    del net
    torch.cuda.empty_cache()
    return out


def load_cache(run, model_name, step, n_wanted):
    with open(cache_path(run, model_name, step, n_wanted), 'rb') as fh:
        return pickle.load(fh)


def macro_classes(frames, min_points=25):
    counts = defaultdict(int)
    for f in frames:
        for (_iy, _ix, cls) in f['points']:
            counts[cls] += 1
    macro_set = sorted(c for c, n in counts.items() if n >= min_points)
    tier_of = {}
    cov = find_artifact('val_coverage_frames.json')
    if cov:
        tier_of = {k: v['tier'] for k, v in json.load(open(cov))['classes'].items()}
    tier_a_set = [c for c in macro_set if tier_of.get(c) == 'TIER_A']
    return macro_set, tier_a_set


def containment_matrix(step_from, step_to, size):
    """(n_from, n_to) int8: does tile `f` of step_from contain the CENTER of tile `t` of
    step_to? Geometry only -- independent of frame content and threshold, computed once per
    (step_from, step_to) pair and reused for every frame/threshold combo that needs it."""
    xs_from = grid_origins(size, step_from)
    xs_to = grid_origins(size, step_to)
    centers_to = xs_to + TILE // 2
    return ((xs_from[:, None] <= centers_to[None, :]) &
            (centers_to[None, :] < xs_from[:, None] + TILE)).astype(np.int8)


def eligible_mask(mass2d1, t1, y_contain, x_contain):
    promoted = (mass2d1.astype(np.float32) >= t1).astype(np.int32)
    return ((y_contain.T.astype(np.int32) @ promoted) @ x_contain.astype(np.int32)) > 0


def sweep(frames1, frames2, step1, step2, macro_set, tier_a_set, t1_grid, t2_grid):
    assert len(frames1) == len(frames2), \
        f'{len(frames1)} vs {len(frames2)} frames cached -- rerun caching with identical ' \
        f'--stride/--limit for both runs'
    y_c = containment_matrix(step1, step2, FRAME_H)
    x_c = containment_matrix(step1, step2, FRAME_W)
    results = []
    for t1 in t1_grid:
        elig_list = [eligible_mask(f1['mass2d'], t1, y_c, x_c) for f1 in frames1]
        promoted_frac = float(np.mean([(f1['mass2d'].astype(np.float32) >= t1).mean()
                                        for f1 in frames1]))
        for t2 in t2_grid:
            fa_num = fa_den = 0
            hit, n = defaultdict(int), defaultdict(int)
            for f2, elig in zip(frames2, elig_list):
                activated = elig & (f2['mass2d'].astype(np.float32) >= t2)
                keep = f2['keep']
                fa_num += int(activated[keep].sum())
                fa_den += int(keep.sum())
                for (iy, ix, cls) in f2['points']:
                    if cls not in macro_set:
                        continue
                    n[cls] += 1
                    if iy is not None and activated[iy, ix].any():
                        hit[cls] += 1
            fa = fa_num / max(fa_den, 1)
            per_class = {c: hit[c] / max(n[c], 1) for c in macro_set}
            macro = float(np.mean([per_class[c] for c in tier_a_set])) if tier_a_set else float('nan')
            tp_points = sum(hit[c] for c in macro_set)
            fn_points = sum(n[c] for c in macro_set) - tp_points
            results.append({'t1': float(t1), 't2': float(t2), 'promoted_frac': promoted_frac,
                             'false_alarm': fa, 'macro_tier_a': macro,
                             'tp_points': tp_points, 'fn_points': fn_points,
                             'fa_tiles': fa_num, 'clean_tiles': fa_den - fa_num})
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('stage1_run')
    ap.add_argument('stage2_runs', nargs='+')
    ap.add_argument('--steps1', default=','.join(str(s) for s in STEPS))
    ap.add_argument('--steps2', default=','.join(str(s) for s in STEPS))
    ap.add_argument('--stride', type=int, default=1)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--cache-only', action='store_true')
    args = ap.parse_args()

    steps1 = [int(s) for s in args.steps1.split(',')]
    steps2 = [int(s) for s in args.steps2.split(',')]
    pool = {r: m for r, m, _ in full_coverage_pool()}
    for r in [args.stage1_run] + args.stage2_runs:
        if r not in pool:
            sys.exit(f'{r}: not in full_coverage_pool() -- see eval_ensemble.py')

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    wanted = wanted_frames(args.stride, args.limit)
    print(f'[cascade-step] {len(wanted):,} coverage frames requested, steps1={steps1}, '
          f'steps2={steps2}\n')

    for s in steps1:
        cache_run_step(args.stage1_run, pool[args.stage1_run], s, wanted, dev)
    for run in args.stage2_runs:
        for s in steps2:
            cache_run_step(run, pool[run], s, wanted, dev)
    if args.cache_only:
        return

    hz = bench_hz_table()
    hz1_by_step = hz.get(pool[args.stage1_run], {})

    for s2_run in args.stage2_runs:
        hz2_by_step = hz.get(pool[s2_run], {})
        print(f'\n{"="*100}\n=== stage1={args.stage1_run} ({pool[args.stage1_run]}) -> '
              f'stage2={s2_run} ({pool[s2_run]}) ===')

        frames2_by_step = {s: load_cache(s2_run, pool[s2_run], s, len(wanted)) for s in steps2}
        macro_set, tier_a_set = macro_classes(frames2_by_step[steps2[0]]['frames'])
        print(f'    macro over {len(macro_set)} classes ({len(tier_a_set)} TIER_A)')

        all_rows = []
        for step1 in steps1:
            hz1 = hz1_by_step.get(step1)
            frames1 = load_cache(args.stage1_run, pool[args.stage1_run], step1, len(wanted))['frames']
            if hz1 is None:
                print(f'  !! no benched Hz for {pool[args.stage1_run]} @ step {step1}, skipping')
                continue
            for step2 in steps2:
                hz2 = hz2_by_step.get(step2)
                if hz2 is None:
                    print(f'  !! no benched Hz for {pool[s2_run]} @ step {step2}, skipping')
                    continue
                res = sweep(frames1, frames2_by_step[step2]['frames'], step1, step2,
                            macro_set, tier_a_set, T1_GRID, T2_GRID)
                for r in res:
                    r['step1'], r['step2'] = step1, step2
                    r['hz'] = 1.0 / (1.0 / hz1 + r['promoted_frac'] / hz2)
                all_rows.extend(res)

        deployable = [r for r in all_rows if r['hz'] >= TARGET_HZ]
        if deployable:
            b = max(deployable, key=lambda r: r['macro_tier_a'])
            print(f"  BEST DEPLOYABLE (hz>={TARGET_HZ:.0f}): step1={b['step1']} step2={b['step2']} "
                  f"t1={b['t1']:.2f} t2={b['t2']:.2f} promoted={b['promoted_frac']*100:5.1f}% "
                  f"hz={b['hz']:5.1f} TIER_A={b['macro_tier_a']*100:6.2f}% "
                  f"(points TP={b['tp_points']} FN={b['fn_points']}, "
                  f"tiles FA={b['fa_tiles']} clean={b['clean_tiles']})")
        else:
            best_hz = max(all_rows, key=lambda r: r['hz']) if all_rows else None
            print(f"  NO deployable point in this grid" +
                  (f" -- best Hz was {best_hz['hz']:.2f} (step1={best_hz['step1']}, "
                   f"step2={best_hz['step2']})" if best_hz else ""))

        b_all = max(all_rows, key=lambda r: r['macro_tier_a']) if all_rows else None
        if b_all:
            print(f"  BEST OVERALL (ignoring Hz):      step1={b_all['step1']} step2={b_all['step2']} "
                  f"t1={b_all['t1']:.2f} t2={b_all['t2']:.2f} promoted={b_all['promoted_frac']*100:5.1f}% "
                  f"hz={b_all['hz']:5.1f} TIER_A={b_all['macro_tier_a']*100:6.2f}%")

        # Per-(step1,step2) best deployable, so the STEP axis's own effect is visible rather
        # than buried inside the single global best.
        print(f"\n  {'step1':>5s} {'step2':>5s}  {'best deployable (hz>=23)':<55s}  "
              f"{'best overall':<30s}")
        for step1 in steps1:
            for step2 in steps2:
                cell = [r for r in all_rows if r['step1'] == step1 and r['step2'] == step2]
                if not cell:
                    continue
                dep = [r for r in cell if r['hz'] >= TARGET_HZ]
                d_str = (f"t1={max(dep, key=lambda r: r['macro_tier_a'])['t1']:.2f} "
                         f"t2={max(dep, key=lambda r: r['macro_tier_a'])['t2']:.2f} "
                         f"hz={max(dep, key=lambda r: r['macro_tier_a'])['hz']:5.1f} "
                         f"TIER_A={max(dep, key=lambda r: r['macro_tier_a'])['macro_tier_a']*100:6.2f}%"
                         if dep else "-- not deployable --")
                b = max(cell, key=lambda r: r['macro_tier_a'])
                o_str = f"hz={b['hz']:5.1f} TIER_A={b['macro_tier_a']*100:6.2f}%"
                print(f"  {step1:5d} {step2:5d}  {d_str:<55s}  {o_str:<30s}")

        out = out_path(f"{args.stage1_run}_{s2_run}", '_cascade_step_sweep.json')
        json.dump({'stage1': args.stage1_run, 'stage2': s2_run,
                   'stage1_model': pool[args.stage1_run], 'stage2_model': pool[s2_run],
                   'steps1': steps1, 'steps2': steps2, 'rows': all_rows}, open(out, 'w'), indent=1)
        print(f'\n  wrote {out}')


if __name__ == '__main__':
    main()
