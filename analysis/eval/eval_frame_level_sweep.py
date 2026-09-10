#!/usr/bin/env python3
"""Frame-level detection: solo models, cascades, and votes -- at the unit an operator
actually acts on (a whole camera frame), not a tile or an annotated point.

WHY THIS EXISTS
---------------
Nothing in this repo currently computes a genuine two-sided frame-level metric. The two
things previously called "frame-level" are NOT this:
  - evaluate_detection.py groups H5 tiles by source frame and checks "did any true-defect
    tile fire" -- but never computes a frame-level false-alarm rate (one-sided). See
    knowledge/PLAN.md's "Frame-level does NOT rescue the low-FA regime" section, whose own
    caveat is that "frame" there is a tile-dump grouping proxy, not a real camera frame.
  - eval_vote_curve.py / eval_cascade_step_sweep.py measure per-POINT detection against a
    per-TILE false-alarm rate, both on raw frames -- frame is just the cache's container.

METRIC
------
Ground truth: a frame is GT-defect if it has ANY annotated non-clean point, re-derived
directly from each frame's own sidecar JSON with EMPTY merges/drops (frame_points' no_light/
missing-file skip logic does not depend on a model's class scheme, only on the frame itself
-- so this one ground-truth list aligns positionally with every model's cascade_step_cache
pickle, all of which were built from the identical, unstrided, unlimited 3,054-frame
val_coverage_frames.json list; verified zero skips in every existing cache before writing
this).

DECISION -- NOT plain "any tile fires". First attempt was: a frame is flagged if
mass2d.max() >= threshold. Measured broken before trusting it further: with ~4,588 tiles
per frame, EVERY clean frame in this corpus has at least one tile scoring >=0.98 (98.9% score
>=0.99) purely from softmax overconfidence on an isolated tile -- a multiple-comparisons
problem, not signal. This is exactly why the live deployed path never acts on a single raw
tile either (`mvc/inference/classifier_pnm.py:process_predictions_erode`, default
erosion_kernel=1/min_votes=2): a tile only counts if enough of its neighbours also fire.
So the frame decision here reuses that same neighbourhood-vote primitive
(`eval_vote_curve.neighbor_counts`, batched across frames): a frame is FLAGGED if any cell's
spatial-neighbour count of activated cells (kernel, min_votes swept alongside threshold)
survives the vote. Frame-level miss/FA are computed directly from that boolean per frame
(same style as eval_cascade_step_sweep.py's point-level sweep, just reduced to one flag per
frame instead of counting per-point hits).

DATA -- reuses experiments/cascade_step_cache/*.pkl (eval_cascade_step_sweep.py's cache),
zero new GPU inference. 5 models x 4 steps (16/18/24/32), 3,054 coverage frames each.

Usage:
    python -m analysis.eval.eval_frame_level_sweep [--solo] [--cascade] [--vote]
        (no flags = all three)
"""

import argparse
import itertools
import json
import os
import pickle
import sys

import numpy as np

from analysis.eval.eval_cascade_step_sweep import (CACHE_DIR, STEPS, T1_GRID, T2_GRID,
                                                    cache_path, containment_matrix,
                                                    eligible_mask, wanted_frames)
from analysis.eval.eval_step_curve import frame_points, local_path

FRAME_H, FRAME_W = 1024, 1224
FA_BUDGETS = (0.05, 0.10)
MODELS = {
    'anc':        'convnext_pico',
    'fzcnxtiny':  'convnext_tiny',
    'fzsqueeze':  'squeezenet1_1',
    'fzswin':     'swin_v2_t',
    'msfemto':    'convnext_femto',
}
GT_CACHE = os.path.join(CACHE_DIR, '_frame_gt_n3054.json')
# (kernel, min_votes) settings to sweep -- kernel=0/min_votes<=1 is the "no voting" baseline
# (a cell only ever counts itself, matching the ROS service's own 0/1-disables-voting
# contract), kernel=1/min_votes=2 is the shipped default, kernel=2/min_votes=3 a stronger
# denoise. Not a full kernel x min_votes grid -- these three points already answer "does
# voting fix the max-of-4588-tiles blowup", a full grid is a follow-up if these disagree.
VOTE_SETTINGS = [(0, 1), (1, 2), (2, 3)]
# Coarser than eval_cascade_step_sweep's own T1_GRID -- this first pass is about which
# (model pair, step pair) combos matter, not fine-tuning t1 to the last decimal; a screen
# only has to notice something's nearby, so 4 points spanning its permissive range is
# enough to see whether the axis matters at all. Widen later if a promising cell needs it.
T1_GRID_COARSE = np.array([0.05, 0.20, 0.40, 0.60])
# Curated (screen, recheck) pairs -- the practical cascade shape (cheap screen, expensive
# precise recheck), not all 20 permutations of the 5 cached models. squeezenet1_1 and
# convnext_femto are the two cheapest/most-permissive-screen candidates from the earlier
# step-sweep campaign (knowledge/7-9-report.md SS8.4-8.5); convnext_tiny and swin_v2_t are
# the two recheck targets that campaign found real step-revival effects for.
CASCADE_PAIRS = [
    ('fzsqueeze', 'fzcnxtiny'), ('fzsqueeze', 'fzswin'),
    ('msfemto', 'fzcnxtiny'), ('msfemto', 'fzswin'),
]
# Same-step diagonal plus the two "coarser screen, finer recheck" cells the earlier
# step-sweep campaign found revived a screen -- not the full 4x4=16 grid (this pass is CPU
# time, not GPU, but still worth keeping cheap for a first look).
CASCADE_STEP_PAIRS = [(16, 16), (18, 18), (24, 24), (32, 32), (32, 16), (24, 16)]


def batched_neighbor_counts(mask3d, kernel):
    """neighbor_counts, vectorised over a stack of frames: mask3d is (n, h, w) bool ->
    (n, h, w) int32 count of True cells (including itself) in each cell's (2k+1)^2
    neighbourhood. Same integral-image construction as eval_vote_curve.neighbor_counts,
    just with the frame axis carried through unreduced."""
    if kernel == 0:
        return mask3d.astype(np.int32)
    n, h, w = mask3d.shape
    m = mask3d.astype(np.int32)
    padded = np.zeros((n, h + 2 * kernel, w + 2 * kernel), dtype=np.int32)
    padded[:, kernel:kernel + h, kernel:kernel + w] = m
    ii = np.zeros((n, padded.shape[1] + 1, padded.shape[2] + 1), dtype=np.int64)
    ii[:, 1:, 1:] = np.cumsum(np.cumsum(padded, axis=1), axis=2)
    k = 2 * kernel + 1
    total = (ii[:, k:k + h, k:k + w] - ii[:, 0:h, k:k + w]
             - ii[:, k:k + h, 0:w] + ii[:, 0:h, 0:w])
    return total


def frame_ground_truth():
    """Model-independent per-frame GT: True = frame has >=1 annotated non-clean point.
    Cached once (pure CPU/disk, no GPU) -- re-derivable any time from the raw JSONs."""
    if os.path.exists(GT_CACHE):
        return np.array(json.load(open(GT_CACHE))['is_defect'], dtype=bool)
    wanted = wanted_frames(1, 0)
    assert len(wanted) == 3054, f'expected 3054 frames, got {len(wanted)} -- caches assume this'
    is_defect = []
    for fj in wanted:
        img_path, json_path = local_path(fj)
        if img_path is None:
            continue  # matches cache_run_step's skip -- verified zero skips in every cache
        pts, no_light = frame_points(json_path, {}, set(), True)
        if pts is None or no_light:
            continue
        is_defect.append(len(pts) > 0)
    assert len(is_defect) == 3054, \
        f'{len(is_defect)} survived GT scan, but every cache has 3054 frames -- a skip ' \
        f'criterion here disagrees with cache_run_step\'s; fix before trusting alignment'
    os.makedirs(CACHE_DIR, exist_ok=True)
    json.dump({'is_defect': [bool(x) for x in is_defect]}, open(GT_CACHE, 'w'))
    return np.array(is_defect, dtype=bool)


def load(run, step):
    with open(cache_path(run, MODELS[run], step, 3054), 'rb') as fh:
        return pickle.load(fh)['frames']


def stacked_mass(frames):
    return np.stack([f['mass2d'] for f in frames]).astype(np.float32)


def fine_grid(mass3d, floor=0.9):
    """Sorted unique values of mass3d at/above `floor`, ASCENDING -- the meaningful
    breakpoints. This data is float16-quantized on disk (~1e-3 resolution near 1.0, mostly
    saturated well above 0.9 -- see the module docstring), so a fixed linear grid either
    misses the region entirely or wastes evaluations on levels that collapse to duplicates;
    the actual unique values are both correct and far fewer."""
    u = np.unique(mass3d)
    u = u[u >= floor]
    return u if len(u) else np.array([floor], dtype=mass3d.dtype)


def _search_best(threshold_grid, kernel, min_votes, mass3d, is_defect, fa_budget):
    """Smallest t in threshold_grid (ascending) with frame FA<=fa_budget, via binary
    search -- valid because raising t can only shrink the activated set, so both FA(t) and
    detect(t) are monotonically non-increasing in t for a fixed (kernel, min_votes). That
    means the smallest t clearing the FA budget is also the one with the highest detect%
    among everything that clears it -- no need to scan the whole grid."""
    def eval_at(i):
        t = threshold_grid[i]
        counts = batched_neighbor_counts(mass3d >= t, kernel)
        flagged = (counts >= min_votes).any(axis=(1, 2))
        fa = float(flagged[~is_defect].mean())
        det = float(flagged[is_defect].mean()) if is_defect.any() else float('nan')
        return t, fa, det

    n = len(threshold_grid)
    t_hi, fa_hi, det_hi = eval_at(n - 1)
    if fa_hi > fa_budget:
        return None  # even the strictest threshold in the grid does not hold the budget
    lo, hi = 0, n - 1
    best = (float(t_hi), det_hi, fa_hi)
    while lo < hi:
        mid = (lo + hi) // 2
        t, fa, det = eval_at(mid)
        if fa <= fa_budget:
            best = (float(t), det, fa)
            hi = mid
        else:
            lo = mid + 1
    return best


def best_at_fa_budget(mass3d, is_defect, fa_budget, vote_settings=VOTE_SETTINGS, floor=0.9):
    """Best (threshold, kernel, min_votes) by frame-detect%, among combos holding frame
    FA <= fa_budget. Returns None if nothing in the grid holds the budget for ANY setting."""
    grid = fine_grid(mass3d, floor)
    best = None
    for kernel, min_votes in vote_settings:
        r = _search_best(grid, kernel, min_votes, mass3d, is_defect, fa_budget)
        if r is not None and (best is None or r[1] > best[3]):
            t, det, fa = r
            best = (t, kernel, min_votes, det, fa)
    return best


# --------------------------------------------------------------------------------- solo
def solo_report(is_defect):
    print(f'\n{"="*90}\nSOLO MODELS -- frame-level detect% at frame FA<=5%/10%, vote-filtered '
          f'(kernel,min_votes swept over {VOTE_SETTINGS})\n{"="*90}')
    print(f'{"model":18s} {"step":>5s} {"det@FA5":>8s} {"(k,mv,t)":>10s} '
          f'{"det@FA10":>9s} {"(k,mv,t)":>10s}')
    print('-' * 76)
    rows = []
    for run, model in MODELS.items():
        for step in STEPS:
            mass3d = stacked_mass(load(run, step))
            b5 = best_at_fa_budget(mass3d, is_defect, 0.05)
            b10 = best_at_fa_budget(mass3d, is_defect, 0.10)
            d5 = f'{b5[3]*100:8.2f}' if b5 else '     --'
            k5 = f'({b5[1]},{b5[2]},{b5[0]:.2f})' if b5 else '  --'
            d10 = f'{b10[3]*100:9.2f}' if b10 else '      --'
            k10 = f'({b10[1]},{b10[2]},{b10[0]:.2f})' if b10 else '  --'
            print(f'{model:18s} {step:5d} {d5} {k5:>10s} {d10} {k10:>10s}')
            rows.append({'run': run, 'model': model, 'step': step,
                         'best_at_fa5': b5 and {'t': b5[0], 'kernel': b5[1], 'min_votes': b5[2],
                                                'detect_pct': b5[3] * 100, 'fa_pct': b5[4] * 100},
                         'best_at_fa10': b10 and {'t': b10[0], 'kernel': b10[1], 'min_votes': b10[2],
                                                  'detect_pct': b10[3] * 100, 'fa_pct': b10[4] * 100}})
    return rows


# ------------------------------------------------------------------------------ cascade
def _search_best_cascade(t2_grid, elig, kernel, min_votes, mass2, is_defect, fa_budget):
    """Same monotonicity argument as _search_best, one level up: for FIXED t1 (hence fixed
    `elig`) and fixed (kernel, min_votes), raising t2 only shrinks the recheck stage's
    activated set, so FA(t2)/detect(t2) are both monotonically non-increasing -- binary
    search for the smallest t2 clearing the budget."""
    def eval_at(i):
        t2 = t2_grid[i]
        counts = batched_neighbor_counts(mass2 >= t2, kernel)
        activated = elig & (counts >= min_votes)
        flagged = activated.any(axis=(1, 2))
        fa = float(flagged[~is_defect].mean())
        det = float(flagged[is_defect].mean()) if is_defect.any() else float('nan')
        return t2, fa, det

    n = len(t2_grid)
    t_hi, fa_hi, det_hi = eval_at(n - 1)
    if fa_hi > fa_budget:
        return None
    lo, hi = 0, n - 1
    best = (float(t_hi), det_hi, fa_hi)
    while lo < hi:
        mid = (lo + hi) // 2
        t2, fa, det = eval_at(mid)
        if fa <= fa_budget:
            best = (float(t2), det, fa)
            hi = mid
        else:
            lo = mid + 1
    return best


def cascade_sweep_pair(run1, run2, is_defect, step_pairs=CASCADE_STEP_PAIRS,
                       t1_grid=T1_GRID_COARSE, fa_budget=0.05):
    """(step1, step2) x t1 x (t2, kernel, min_votes) combos for one ordered (screen,
    recheck) pairing, over the given step_pairs only (not a full steps x steps grid).
    Voting applies to the RECHECK stage only -- the screen's job is to not miss anything
    nearby, not to be spatially precise (same reasoning eval_cascade_step_sweep.py already
    uses for why t1 stays permissive). t2 is binary-searched per (t1, kernel, min_votes)
    -- see _search_best_cascade."""
    best_by_step = {}
    frames1_by_step = {}
    mass2_by_step = {}
    for step1, step2 in step_pairs:
        if step1 not in frames1_by_step:
            frames1_by_step[step1] = load(run1, step1)
        if step2 not in mass2_by_step:
            mass2_by_step[step2] = stacked_mass(load(run2, step2))
        frames1, mass2 = frames1_by_step[step1], mass2_by_step[step2]
        t2_grid = fine_grid(mass2)
        y_c = containment_matrix(step1, step2, FRAME_H)
        x_c = containment_matrix(step1, step2, FRAME_W)
        best = None  # (t1, t2, kernel, min_votes) -> best detect at FA<=fa_budget
        for t1 in t1_grid:
            elig = np.stack([eligible_mask(f1['mass2d'], t1, y_c, x_c) for f1 in frames1])
            for kernel, min_votes in VOTE_SETTINGS:
                r = _search_best_cascade(t2_grid, elig, kernel, min_votes, mass2,
                                         is_defect, fa_budget)
                if r is not None and (best is None or r[1] > best[4]):
                    t2, det, fa = r
                    best = (float(t1), t2, kernel, min_votes, det, fa)
        best_by_step[(step1, step2)] = best
    return best_by_step


def cascade_report(is_defect):
    print(f'\n{"="*90}\nCASCADE (screen -> recheck) -- best frame detect% at frame FA<=5%, '
          f'per (step1, step2), recheck vote-filtered\n'
          f'curated pairs={CASCADE_PAIRS}, step pairs={CASCADE_STEP_PAIRS}\n{"="*90}')
    rows = []
    for run1, run2 in CASCADE_PAIRS:
        print(f'\n--- screen={run1} ({MODELS[run1]}) -> recheck={run2} ({MODELS[run2]}) ---')
        print(f'{"step1":>6s} {"step2":>6s} {"t1":>6s} {"t2":>6s} {"(k,mv)":>7s} '
              f'{"detect%":>8s} {"FA%":>6s}')
        best_by_step = cascade_sweep_pair(run1, run2, is_defect)
        for (s1, s2), b in best_by_step.items():
            if b is None:
                print(f'{s1:6d} {s2:6d}   -- no combo in grid holds frame FA<=5% --')
                continue
            t1, t2, kernel, min_votes, det, fa = b
            print(f'{s1:6d} {s2:6d} {t1:6.2f} {t2:6.2f} ({kernel},{min_votes}) '
                  f'{det*100:8.2f} {fa*100:6.2f}')
            rows.append({'screen': run1, 'recheck': run2, 'step1': s1, 'step2': s2,
                         't1': t1, 't2': t2, 'kernel': kernel, 'min_votes': min_votes,
                         'frame_detect_pct': det * 100, 'frame_fa_pct': fa * 100})
    return rows


def _search_best_vote(t_grid, masses, subset, min_votes, is_defect, fa_budget):
    """Same monotonicity as _search_best: for fixed min_votes, raising the shared
    threshold t only shrinks every member's activated set, so the cross-model vote count
    can only shrink too -- FA(t)/detect(t) are non-increasing, binary search applies."""
    def eval_at(i):
        t = t_grid[i]
        votes = sum((masses[run] >= t).astype(np.int32) for run in subset)
        flagged = (votes >= min_votes).any(axis=(1, 2))
        fa = float(flagged[~is_defect].mean())
        det = float(flagged[is_defect].mean()) if is_defect.any() else float('nan')
        return t, fa, det

    n = len(t_grid)
    t_hi, fa_hi, det_hi = eval_at(n - 1)
    if fa_hi > fa_budget:
        return None
    lo, hi = 0, n - 1
    best = (float(t_hi), det_hi, fa_hi)
    while lo < hi:
        mid = (lo + hi) // 2
        t, fa, det = eval_at(mid)
        if fa <= fa_budget:
            best = (float(t), det, fa)
            hi = mid
        else:
            lo = mid + 1
    return best


# --------------------------------------------------------------------------------- vote
def vote_report(is_defect):
    print(f'\n{"="*90}\nVOTE (majority-agreement across models, SAME shared step) -- best frame '
          f'detect% at frame FA<=5%\n{"="*90}')
    runs = list(MODELS)
    rows = []
    for step in STEPS:
        cached = {run: load(run, step) for run in runs}
        masses = {run: np.stack([f['mass2d'] for f in cached[run]]).astype(np.float32)
                 for run in runs}  # (n_frames, ny, nx) -- same grid shape, all models @ this step
        t_grid = fine_grid(np.stack(list(masses.values())))
        for k in (2, 3, 4, 5):
            for subset in itertools.combinations(runs, k):
                best = None
                for min_votes in range(1, k + 1):
                    r = _search_best_vote(t_grid, masses, subset, min_votes, is_defect, 0.05)
                    if r is not None and (best is None or r[1] > best[2]):
                        t, det, fa = r
                        best = (t, min_votes, det, fa)
                if best:
                    t, mv, det, fa = best
                    label = '+'.join(subset)
                    print(f'step={step:3d} {label:40s} t={t:.4f} votes>={mv}/{k}  '
                          f'detect%={det*100:6.2f}  FA%={fa*100:5.2f}')
                    rows.append({'step': step, 'models': list(subset), 'k': k,
                                 'threshold': t, 'min_votes': mv,
                                 'frame_detect_pct': det * 100, 'frame_fa_pct': fa * 100})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--solo', action='store_true')
    ap.add_argument('--cascade', action='store_true')
    ap.add_argument('--vote', action='store_true')
    args = ap.parse_args()
    run_all = not (args.solo or args.cascade or args.vote)

    is_defect = frame_ground_truth()
    print(f'[frame-level] {len(is_defect):,} frames, {int(is_defect.sum()):,} GT-defect '
          f'({100*is_defect.mean():.1f}%), {int((~is_defect).sum()):,} GT-clean')

    out = {}
    if args.solo or run_all:
        out['solo'] = solo_report(is_defect)
    if args.cascade or run_all:
        out['cascade'] = cascade_report(is_defect)
    if args.vote or run_all:
        out['vote'] = vote_report(is_defect)

    path = os.path.join(CACHE_DIR, 'frame_level_sweep_results.json')
    json.dump(out, open(path, 'w'), indent=1)
    print(f'\nwrote {path}')


if __name__ == '__main__':
    main()
