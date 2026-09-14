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
        (no flags among these three = all three)
    python -m analysis.eval.eval_frame_level_sweep --html-report
        writes a self-contained "confusion sweep" visualisation -- a confusion matrix
        (ground-truth class x flagged/not-flagged) where every cell sparklines its rate
        across the threshold sweep instead of showing one number at one threshold --
        for the current best solo config at frame FA<=5%. Independent of --solo/
        --cascade/--vote (those are not run unless also requested).
"""

import argparse
import datetime
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
from mvc.core.artifact_paths import find_config_with_classes
from mvc.core.config import load_hyperparameters

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
CLASS_GT_CACHE = os.path.join(CACHE_DIR, '_frame_class_gt_n3054.json')
# (kernel, min_votes) settings to sweep -- kernel=0/min_votes<=1 is the "no voting" baseline
# (a cell only ever counts itself, matching the ROS service's own 0/1-disables-voting
# contract), kernel=1/min_votes=2 is the shipped default, kernel=2/min_votes=3 a stronger
# denoise. Not a full kernel x min_votes grid -- these three points already answer "does
# voting fix the max-of-4588-tiles blowup", a full grid is a follow-up if these disagree.
VOTE_SETTINGS = [(0, 1), (1, 2), (2, 3)]
# REFINED 2026-09-11, superseding the original 4-point coarse grid: that first pass found
# every winning config sitting at t1=0.05 (the grid's own floor), across every pairing and
# step combo -- meaning the screen was a no-op (nearly everything gets promoted at that
# threshold) and the sweep never got to test whether a REAL screen adds value. Checked why
# before widening blindly: the screen models' own per-cell score distribution has the same
# near-1.0 saturation the recheck stage did (only 21 of 8,680+ unique values sit above 0.99
# for a representative model/step) -- so the interesting range for t1 is not the original
# grid's 0.05-0.60 span, it is close to 1.0, same as t2. This grid keeps light coverage of
# the traditional "permissive screen" range and adds dense coverage near 1.0 to actually
# test that hypothesis, while staying small enough to sweep in a few minutes (a full
# unique-value grid there is 1,800+ points -- checked and rejected as too slow for this
# pass).
T1_GRID_REFINED = np.array([
    0.05, 0.15, 0.30, 0.45, 0.60,                                   # original range, kept
    0.75, 0.85, 0.90, 0.95, 0.97, 0.99, 0.995, 0.999, 0.9995, 0.9999, 0.99995,  # near-1.0
])
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


def canonical_class_scheme():
    """merges/drops shared by all 5 cached models -- verified identical merge_classes/
    drop_classes print output across every eval_failure_conditions.py run this session
    (all 5 configs are Aug26_78K campaign runs under the same canonical 10-class scheme).
    Read from one config rather than hardcoded, so a future scheme change can't silently
    desync this from what the models actually trained on."""
    cfg = load_hyperparameters(find_config_with_classes('anc_convnext_pico.json'))
    return cfg.get('class_merges') or {}, set(cfg.get('drop_classes') or [])


def class_ground_truth():
    """Model-independent per-frame, PER-CLASS GT: for each of the campaign's real defect
    classes, True = frame has >=1 point mapped to that class under the canonical scheme.
    Same iteration/skip logic and positional-alignment guarantee as frame_ground_truth()
    (just also keeping which class(es) instead of collapsing to one bool) -- a frame with
    points from multiple classes counts toward each of them, same as any other per-class
    recall table in this repo (e.g. eval_failure_conditions.class_breakdown())."""
    if os.path.exists(CLASS_GT_CACHE):
        d = json.load(open(CLASS_GT_CACHE))
        return {k: np.array(v, dtype=bool) for k, v in d.items()}
    merges, drops = canonical_class_scheme()
    wanted = wanted_frames(1, 0)
    assert len(wanted) == 3054, f'expected 3054 frames, got {len(wanted)} -- caches assume this'
    present_sets = []
    for fj in wanted:
        img_path, json_path = local_path(fj)
        if img_path is None:
            continue
        pts, no_light = frame_points(json_path, merges, drops, True)
        if pts is None or no_light:
            continue
        present_sets.append({cls for (_x, _y, cls) in pts
                             if cls is not None and cls != 'class_clean'})
    assert len(present_sets) == 3054, \
        f'{len(present_sets)} survived GT scan, but every cache has 3054 frames'
    all_classes = sorted({c for s in present_sets for c in s})
    per_class = {c: [c in s for s in present_sets] for c in all_classes}
    os.makedirs(CACHE_DIR, exist_ok=True)
    json.dump(per_class, open(CLASS_GT_CACHE, 'w'))
    return {k: np.array(v, dtype=bool) for k, v in per_class.items()}


def per_class_report(flagged, class_gt, indent='    '):
    """detect% among GT-defect frames of each class, at an ALREADY-CHOSEN operating
    point (the pooled-optimal one found by the search functions above) -- this is the
    threshold-impact-per-class study: does the config that maximises the POOLED number
    actually serve every class, or does it favour the easy ones? Not a per-class
    re-optimisation -- one fixed flagged array, sliced by class."""
    print(f'{indent}{"class":26s} {"n":>6s} {"detect%":>8s}')
    for cname in sorted(class_gt):
        gt = class_gt[cname]
        n = int(gt.sum())
        if n == 0:
            continue
        det = 100.0 * float(flagged[gt].mean())
        print(f'{indent}{cname:26s} {n:6,d} {det:8.2f}')


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
def solo_report(is_defect, class_gt):
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
            if b5:
                t, kernel, min_votes = b5[0], b5[1], b5[2]
                flagged = (batched_neighbor_counts(mass3d >= t, kernel) >= min_votes).any(axis=(1, 2))
                print(f'  per-class detect% at this FA5 operating point (pooled={b5[3]*100:.2f}%):')
                per_class_report(flagged, class_gt)
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
                       t1_grid=T1_GRID_REFINED, fa_budget=0.05):
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


def cascade_report(is_defect, class_gt):
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
            frames1 = load(run1, s1)
            mass2 = stacked_mass(load(run2, s2))
            y_c = containment_matrix(s1, s2, FRAME_H)
            x_c = containment_matrix(s1, s2, FRAME_W)
            elig = np.stack([eligible_mask(f1['mass2d'], t1, y_c, x_c) for f1 in frames1])
            counts = batched_neighbor_counts(mass2 >= t2, kernel)
            flagged = (elig & (counts >= min_votes)).any(axis=(1, 2))
            print(f'  per-class detect% at this operating point (pooled={det*100:.2f}%):')
            per_class_report(flagged, class_gt)
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
def vote_report(is_defect, class_gt, top_n_class_breakdown=5):
    print(f'\n{"="*90}\nVOTE (majority-agreement across models, SAME shared step) -- best frame '
          f'detect% at frame FA<=5%\n{"="*90}')
    runs = list(MODELS)
    rows = []
    masses_by_step = {}
    for step in STEPS:
        cached = {run: load(run, step) for run in runs}
        masses = {run: np.stack([f['mass2d'] for f in cached[run]]).astype(np.float32)
                 for run in runs}  # (n_frames, ny, nx) -- same grid shape, all models @ this step
        masses_by_step[step] = masses
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

    # Per-class breakdown only for the top N by pooled detect% -- 104 combos is too many
    # to print a 9-row table for each; this still answers the threshold-impact-per-class
    # question for every config a reader would actually consider deploying.
    top = sorted(rows, key=lambda r: -r['frame_detect_pct'])[:top_n_class_breakdown]
    print(f'\n--- per-class detect%, top {len(top)} vote configs by pooled detect% ---')
    for r in top:
        subset, step, t, mv = r['models'], r['step'], r['threshold'], r['min_votes']
        masses = masses_by_step[step]
        votes = sum((masses[run] >= t).astype(np.int32) for run in subset)
        flagged = (votes >= mv).any(axis=(1, 2))
        print(f"\n  step={step} {'+'.join(subset)} t={t:.4f} votes>={mv}/{len(subset)} "
              f"(pooled detect%={r['frame_detect_pct']:.2f}, FA%={r['frame_fa_pct']:.2f}):")
        per_class_report(flagged, class_gt, indent='    ')
    return rows


# --------------------------------------------------------------------------- html report
def find_best_solo(is_defect, fa_budget=0.05):
    """Best (run, step) by pooled frame-detect% among solo configs holding frame
    FA<=fa_budget -- same search solo_report() prints per-row, kept standalone so
    --html-report works even when --solo wasn't also requested (cheap: 5 models x 4
    steps of cached-pickle CPU work, no GPU)."""
    best = None
    for run in MODELS:
        for step in STEPS:
            mass3d = stacked_mass(load(run, step))
            b = best_at_fa_budget(mass3d, is_defect, fa_budget)
            if b is not None and (best is None or b[3] > best[1][3]):
                best = ((run, step), b)
    if best is None:
        return None
    (run, step), (t, kernel, min_votes, det, fa) = best
    return {'run': run, 'step': step, 't': t, 'kernel': kernel, 'min_votes': min_votes,
            'detect_pct': det * 100, 'fa_pct': fa * 100}


def build_confusion_sweep_data(run, step, kernel, min_votes, op_t, is_defect, class_gt,
                               n_points=60, floor=0.90):
    """Threshold-swept per-class detect%/FA% payload for the --html-report confusion-
    sweep visualisation: for up to n_points thresholds spanning this config's real
    cached score range, record pooled FA% and every class's detect% at each one -- not
    just at the single chosen operating point, so a reader can see how sensitive each
    cell is to exactly where the threshold sits."""
    mass3d = stacked_mass(load(run, step))
    grid_full = fine_grid(mass3d, floor)
    idx = np.unique(np.linspace(0, len(grid_full) - 1, min(n_points, len(grid_full))).astype(int))
    grid = grid_full[idx]
    classes = sorted(class_gt)
    rows = []
    for t in grid:
        counts = batched_neighbor_counts(mass3d >= t, kernel)
        flagged = (counts >= min_votes).any(axis=(1, 2))
        row = {'t': float(t), 'fa_pct': 100.0 * float(flagged[~is_defect].mean())}
        for c in classes:
            row[c] = 100.0 * float(flagged[class_gt[c]].mean())
        rows.append(row)
    return {
        'run': run, 'model': MODELS[run], 'step': step,
        'kernel': kernel, 'min_votes': min_votes, 'operating_point_t': float(op_t),
        'classes': classes,
        'n_by_class': {c: int(class_gt[c].sum()) for c in classes},
        'n_clean': int((~is_defect).sum()), 'n_defect_total': int(is_defect.sum()),
        'sweep': rows,
    }


# Plain (non-f) string constants -- CSS/JS content is interpolated as-is into the f-string
# in render_confusion_sweep_html() below, so their own literal { } characters never need
# escaping. System font stack only (no Google Fonts import): this file gets scp'd/opened
# on boxes with no guaranteed internet access, unlike a hosted artifact.
_CONFUSION_SWEEP_CSS = """
:root {
  --bg: #f2f5f6;
  --surface: #ffffff;
  --surface-2: #eaeef0;
  --text: #182028;
  --text-dim: #5c6773;
  --text-faint: #8a95a0;
  --border: #d7dee2;
  --detect: 31, 138, 112;
  --fa: 193, 68, 60;
  --op: 199, 138, 24;
  --clean-row: rgba(193, 68, 60, 0.05);
  --warn-bg: #fbeceb;
  --warn-text: #9a3f37;
  --font-display: -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
  --font-mono: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace;
}

* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--text);
  font-family: var(--font-display);
  padding: 2.5rem 1.5rem 4rem;
}
.wrap { max-width: 1180px; margin: 0 auto; }

header h1 { font-size: 1.7rem; font-weight: 700; letter-spacing: -0.01em; margin: 0 0 0.3rem; }
header p.sub { margin: 0; color: var(--text-dim); font-size: 0.95rem; max-width: 68ch; line-height: 1.5; }
.config-line {
  margin-top: 0.9rem;
  font-family: var(--font-mono);
  font-size: 0.78rem;
  color: var(--text-faint);
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 0.5rem 0.75rem;
  display: inline-block;
}
.config-line b { color: var(--text-dim); font-weight: 600; }

.legend {
  display: flex; flex-wrap: wrap; gap: 1.4rem; align-items: center;
  margin: 1.6rem 0 1.1rem; font-size: 0.82rem; color: var(--text-dim);
}
.legend .item { display: flex; align-items: center; gap: 0.45rem; }
.swatch { width: 13px; height: 13px; border-radius: 3px; flex: none; }
.swatch.line { width: 18px; height: 2px; border-radius: 0; }

.scroller { overflow-x: auto; border-radius: 10px; border: 1px solid var(--border); }
.matrix {
  display: grid;
  grid-template-columns: 236px minmax(300px, 1fr) minmax(300px, 1fr);
  min-width: 900px;
  background: var(--surface);
}
.cell {
  padding: 0.7rem 0.9rem;
  border-bottom: 1px solid var(--border);
  display: flex; flex-direction: column; justify-content: center;
}
.matrix > .cell:nth-child(3n+1) { border-right: 1px solid var(--border); }
.matrix > .cell:nth-child(3n+2) { border-right: 1px solid var(--border); }

.col-head {
  background: var(--surface-2);
  font-size: 0.68rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.06em;
  color: var(--text-dim); padding-top: 0.9rem; padding-bottom: 0.9rem;
}
.col-head .sub { text-transform: none; letter-spacing: 0; font-weight: 400; font-size: 0.72rem; margin-top: 0.15rem; }

.row-label { justify-content: center; }
.row-label .name { font-size: 0.85rem; font-weight: 600; }
.row-label .meta { font-family: var(--font-mono); font-size: 0.72rem; color: var(--text-faint); margin-top: 0.2rem; }
.row-label .lowN {
  display: inline-block; margin-top: 0.35rem; font-size: 0.66rem; font-weight: 600;
  color: var(--warn-text); background: var(--warn-bg); border-radius: 4px;
  padding: 0.12rem 0.4rem; width: fit-content;
}

.chart-cell { position: relative; padding: 0.5rem 0.9rem 0.4rem; }
.chart-top { display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 0.15rem; }
.chart-top .op-val { font-family: var(--font-mono); font-weight: 600; font-size: 1.05rem; }
.chart-top .op-val .unit { font-size: 0.7rem; font-weight: 400; color: var(--text-faint); }
.chart-top .range-val { font-family: var(--font-mono); font-size: 0.68rem; color: var(--text-faint); }
svg.spark { display: block; width: 100%; height: 56px; overflow: visible; cursor: crosshair; }
svg.spark .baseline { stroke: var(--border); stroke-width: 1; }
svg.spark .opline { stroke: rgb(var(--op)); stroke-width: 1.3; stroke-dasharray: 3 2; }
svg.spark .hoverline { stroke: var(--text-dim); stroke-width: 1; opacity: 0; }
svg.spark .hoverdot { opacity: 0; }

.axis-cell { padding-top: 0.3rem; padding-bottom: 0.9rem; border-bottom: none; }
.axis-cell svg { width: 100%; height: 18px; display: block; }
.axis-cell text { font-family: var(--font-mono); font-size: 9.5px; fill: var(--text-faint); }

.divider { height: 1px; background: var(--border); grid-column: 1 / -1; }

.tooltip {
  position: fixed; pointer-events: none;
  background: var(--text); color: var(--bg);
  font-family: var(--font-mono); font-size: 0.72rem;
  padding: 0.3rem 0.55rem; border-radius: 5px;
  transform: translate(-50%, -130%); white-space: nowrap;
  opacity: 0; transition: opacity 0.08s ease; z-index: 10;
}

footer { margin-top: 1.6rem; font-size: 0.78rem; color: var(--text-faint); line-height: 1.6; max-width: 78ch; }
footer strong { color: var(--text-dim); }
"""

_CONFUSION_SWEEP_JS = """
(function () {
  var DATA = JSON.parse(document.getElementById('data-json').textContent);
  var sweep = DATA.sweep;
  var N = sweep.length;
  var opIdx = 0, best = Infinity;
  sweep.forEach(function (r, i) {
    var d = Math.abs(r.t - DATA.operating_point_t);
    if (d < best) { best = d; opIdx = i; }
  });

  var classes = DATA.classes.slice().sort(function (a, b) {
    return DATA.n_by_class[b] - DATA.n_by_class[a];
  });

  var PLOT_W = 300, PLOT_H = 56, PAD_T = 4, PAD_B = 4;

  function yOf(v) {
    var h = PLOT_H - PAD_T - PAD_B;
    return PAD_T + h - (v / 100) * h;
  }
  function xOf(i) { return (i / (N - 1)) * PLOT_W; }

  function buildPath(values) {
    var d = '';
    for (var i = 0; i < values.length; i++) {
      d += (i === 0 ? 'M' : 'L') + xOf(i).toFixed(1) + ',' + yOf(values[i]).toFixed(1) + ' ';
    }
    return d.trim();
  }
  function buildAreaPath(values) {
    var line = buildPath(values);
    return line + ' L' + xOf(values.length - 1).toFixed(1) + ',' + yOf(0).toFixed(1)
      + ' L' + xOf(0).toFixed(1) + ',' + yOf(0).toFixed(1) + ' Z';
  }

  function svgNS(tag) { return document.createElementNS('http://www.w3.org/2000/svg', tag); }

  function makeSpark(values, rgbVar, muted) {
    var svg = svgNS('svg');
    svg.setAttribute('class', 'spark');
    svg.setAttribute('viewBox', '0 0 ' + PLOT_W + ' ' + PLOT_H);
    svg.setAttribute('preserveAspectRatio', 'none');

    var base = svgNS('line');
    base.setAttribute('class', 'baseline');
    base.setAttribute('x1', 0); base.setAttribute('x2', PLOT_W);
    base.setAttribute('y1', yOf(0)); base.setAttribute('y2', yOf(0));
    svg.appendChild(base);

    var color = muted ? 'var(--text-faint)' : ('rgb(' + rgbVar + ')');
    var fillColor = muted ? 'rgba(140,150,160,0.12)' : ('rgba(' + rgbVar + ',0.16)');

    var area = svgNS('path');
    area.setAttribute('d', buildAreaPath(values));
    area.setAttribute('fill', fillColor);
    area.setAttribute('stroke', 'none');
    svg.appendChild(area);

    var line = svgNS('path');
    line.setAttribute('d', buildPath(values));
    line.setAttribute('fill', 'none');
    line.setAttribute('stroke', color);
    line.setAttribute('stroke-width', muted ? 1.3 : 1.8);
    line.setAttribute('stroke-linejoin', 'round');
    line.setAttribute('stroke-linecap', 'round');
    svg.appendChild(line);

    var opX = xOf(opIdx);
    var opLine = svgNS('line');
    opLine.setAttribute('class', 'opline');
    opLine.setAttribute('x1', opX); opLine.setAttribute('x2', opX);
    opLine.setAttribute('y1', PAD_T); opLine.setAttribute('y2', PLOT_H - PAD_B);
    svg.appendChild(opLine);

    var opDot = svgNS('circle');
    opDot.setAttribute('cx', opX);
    opDot.setAttribute('cy', yOf(values[opIdx]));
    opDot.setAttribute('r', 2.6);
    opDot.setAttribute('fill', muted ? 'var(--text-dim)' : color);
    svg.appendChild(opDot);

    var hoverLine = svgNS('line');
    hoverLine.setAttribute('class', 'hoverline');
    hoverLine.setAttribute('y1', PAD_T); hoverLine.setAttribute('y2', PLOT_H - PAD_B);
    svg.appendChild(hoverLine);

    var hoverDot = svgNS('circle');
    hoverDot.setAttribute('class', 'hoverdot');
    hoverDot.setAttribute('r', 3);
    hoverDot.setAttribute('fill', muted ? 'var(--text-dim)' : color);
    svg.appendChild(hoverDot);

    var tooltip = document.getElementById('tooltip');
    svg.addEventListener('mousemove', function (ev) {
      var rect = svg.getBoundingClientRect();
      var frac = (ev.clientX - rect.left) / rect.width;
      var idx = Math.max(0, Math.min(N - 1, Math.round(frac * (N - 1))));
      var x = xOf(idx), y = yOf(values[idx]);
      hoverLine.setAttribute('x1', x); hoverLine.setAttribute('x2', x);
      hoverLine.style.opacity = 1;
      hoverDot.setAttribute('cx', x); hoverDot.setAttribute('cy', y);
      hoverDot.style.opacity = 1;
      tooltip.style.left = ev.clientX + 'px';
      tooltip.style.top = ev.clientY + 'px';
      tooltip.style.opacity = 1;
      tooltip.textContent = 't=' + sweep[idx].t.toFixed(4) + '  ->  ' + values[idx].toFixed(1) + '%';
    });
    svg.addEventListener('mouseleave', function () {
      hoverLine.style.opacity = 0;
      hoverDot.style.opacity = 0;
      tooltip.style.opacity = 0;
    });
    return svg;
  }

  function chartCell(values, rgbVar, muted) {
    var cell = document.createElement('div');
    cell.className = 'cell chart-cell';
    var top = document.createElement('div');
    top.className = 'chart-top';
    var opVal = document.createElement('div');
    opVal.className = 'op-val';
    opVal.style.color = muted ? 'var(--text-dim)' : ('rgb(' + rgbVar + ')');
    opVal.innerHTML = values[opIdx].toFixed(1) + '<span class="unit">% at op.</span>';
    var range = document.createElement('div');
    range.className = 'range-val';
    var mn = Math.min.apply(null, values), mx = Math.max.apply(null, values);
    range.textContent = 'range ' + mn.toFixed(1) + '-' + mx.toFixed(1) + '%';
    top.appendChild(opVal); top.appendChild(range);
    cell.appendChild(top);
    cell.appendChild(makeSpark(values, rgbVar, muted));
    if (!muted) {
      var alpha = Math.max(0.03, Math.min(0.30, (values[opIdx] / 100) * 0.30));
      cell.style.background = 'rgba(' + rgbVar + ',' + alpha.toFixed(3) + ')';
    }
    return cell;
  }

  function labelCell(name, n, isClean, lowN) {
    var cell = document.createElement('div');
    cell.className = 'cell row-label';
    var nm = document.createElement('div');
    nm.className = 'name';
    nm.textContent = isClean ? 'class_clean (false alarms)' : name;
    if (isClean) { nm.style.color = 'rgb(var(--fa))'; cell.style.background = 'var(--clean-row)'; }
    var meta = document.createElement('div');
    meta.className = 'meta';
    meta.textContent = 'n=' + n.toLocaleString();
    cell.appendChild(nm); cell.appendChild(meta);
    if (lowN) {
      var w = document.createElement('div');
      w.className = 'lowN';
      w.textContent = 'low n - noisy';
      cell.appendChild(w);
    }
    return cell;
  }

  var matrix = document.getElementById('matrix');

  var h1 = document.createElement('div'); h1.className = 'cell col-head';
  h1.innerHTML = 'Ground-truth row';
  var h2 = document.createElement('div'); h2.className = 'cell col-head';
  h2.innerHTML = 'Flagged<div class="sub">detect % (classes) / FA % (clean)</div>';
  var h3 = document.createElement('div'); h3.className = 'cell col-head';
  h3.innerHTML = 'Not flagged<div class="sub">miss % (classes) / correct-reject % (clean)</div>';
  matrix.appendChild(h1); matrix.appendChild(h2); matrix.appendChild(h3);

  classes.forEach(function (cname) {
    var vals = sweep.map(function (r) { return r[cname]; });
    var inv = vals.map(function (v) { return 100 - v; });
    matrix.appendChild(labelCell(cname, DATA.n_by_class[cname], false, DATA.n_by_class[cname] < 30));
    matrix.appendChild(chartCell(vals, 'var(--detect)', false));
    matrix.appendChild(chartCell(inv, 'var(--detect)', true));
  });

  var divider = document.createElement('div');
  divider.className = 'divider';
  matrix.appendChild(divider);

  (function () {
    var vals = sweep.map(function (r) { return r.fa_pct; });
    var inv = vals.map(function (v) { return 100 - v; });
    matrix.appendChild(labelCell('class_clean', DATA.n_clean, true, false));
    matrix.appendChild(chartCell(vals, 'var(--fa)', false));
    var mutedCell = chartCell(inv, 'var(--fa)', true);
    mutedCell.style.background = 'var(--clean-row)';
    matrix.appendChild(mutedCell);
  })();

  var axisEmpty = document.createElement('div');
  axisEmpty.className = 'cell axis-cell';
  matrix.appendChild(axisEmpty);
  [0, 1].forEach(function () {
    var cell = document.createElement('div');
    cell.className = 'cell axis-cell';
    var svg = svgNS('svg');
    svg.setAttribute('viewBox', '0 0 ' + PLOT_W + ' 18');
    svg.setAttribute('preserveAspectRatio', 'none');
    [0, Math.round((N - 1) / 2), N - 1].forEach(function (idx) {
      var t = svgNS('text');
      var x = xOf(idx);
      t.setAttribute('x', x);
      t.setAttribute('y', 9);
      t.setAttribute('text-anchor', idx === 0 ? 'start' : (idx === N - 1 ? 'end' : 'middle'));
      t.textContent = 't=' + sweep[idx].t.toFixed(3);
      svg.appendChild(t);
    });
    var opTick = svgNS('text');
    opTick.setAttribute('x', xOf(opIdx));
    opTick.setAttribute('y', 9);
    opTick.setAttribute('text-anchor', 'middle');
    opTick.setAttribute('fill', 'rgb(var(--op))');
    opTick.setAttribute('font-weight', '600');
    opTick.textContent = '^ op';
    svg.appendChild(opTick);
    cell.appendChild(svg);
    matrix.appendChild(cell);
  });
})();
"""


def render_confusion_sweep_html(data, fa_budget=0.05):
    """Self-contained HTML: a confusion matrix (ground-truth class x flagged/not-
    flagged) where every cell sparklines its rate across data['sweep'] instead of
    showing one number at one threshold. Prototyped and validated interactively
    (hover crosshair, background-tint-by-operating-point, muted complement column)
    before wiring in here -- see knowledge/PLAN.md's confusion-sweep note."""
    generated = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
    cfg = (f"{data['run']} / {data['model']} &middot; step={data['step']} &middot; "
          f"kernel={data['kernel']}, min_votes={data['min_votes']}")
    n_points = len(data['sweep'])
    t_lo, t_hi = data['sweep'][0]['t'], data['sweep'][-1]['t']
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Confusion Sweep -- {data['model']}</title>
<style>{_CONFUSION_SWEEP_CSS}</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>Confusion Sweep</h1>
    <p class="sub">
      A confusion matrix where every cell is a curve, not a number. Rows are ground-truth
      classes; columns are the two outcomes a frame can get (flagged / not flagged). Each
      cell sweeps the raw detection threshold across its real operating range, so you can
      read both the rate at the chosen operating point (bold number, background tint) and
      how sensitive that cell is to exactly where the threshold sits (the sparkline).
    </p>
    <div class="config-line">
      <b>config</b> {cfg}
      &nbsp;|&nbsp; <b>sweep</b> t &isin; [{t_lo:.3f}, {t_hi:.3f}], {n_points} real cached threshold levels
      &nbsp;|&nbsp; <b>operating point</b> t={data['operating_point_t']:.4f} (best solo config at frame FA&le;{fa_budget*100:.0f}%)
      &nbsp;|&nbsp; <b>frames</b> {data['n_defect_total'] + data['n_clean']:,} &middot; generated {generated}
    </div>
  </header>
  <div class="legend">
    <div class="item"><span class="swatch" style="background: rgba(31,138,112,0.85)"></span>defect class, flagged (detect rate)</div>
    <div class="item"><span class="swatch" style="background: rgba(193,68,60,0.85)"></span>clean frames, flagged (false-alarm rate)</div>
    <div class="item"><span class="swatch line" style="background: rgb(199,138,24)"></span>chosen operating threshold</div>
    <div class="item"><span class="swatch" style="background: #eaeef0; border:1px solid #d7dee2"></span>"not flagged" column = 100 &minus; left column, muted</div>
  </div>
  <div class="scroller"><div class="matrix" id="matrix"></div></div>
  <footer>
    <strong>Reading it:</strong> background tint intensity in each chart cell mirrors the
    bold number (the rate at the marked operating point) for a fast per-class scan; the
    curve then shows every other threshold in the swept range, so a steep cell means small
    threshold changes move that class's rate a lot -- worth knowing before nudging a
    threshold in production. <strong>Scope:</strong> one config (the current best solo
    pick at this FA budget), not a re-sweep of every model/step/cascade combo.
  </footer>
</div>
<div class="tooltip" id="tooltip"></div>
<script id="data-json" type="application/json">{json.dumps(data)}</script>
<script>{_CONFUSION_SWEEP_JS}</script>
</body>
</html>
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--solo', action='store_true')
    ap.add_argument('--cascade', action='store_true')
    ap.add_argument('--vote', action='store_true')
    ap.add_argument('--html-report', nargs='?', metavar='PATH',
                    const=os.path.join(CACHE_DIR, 'confusion_sweep.html'), default=None,
                    help='Write the confusion-sweep HTML visualisation for the best solo '
                         'config at frame FA<=5%% (see module docstring). PATH is optional '
                         '-- defaults to confusion_sweep.html next to the pkl cache.')
    args = ap.parse_args()
    run_all = not (args.solo or args.cascade or args.vote or args.html_report)

    is_defect = frame_ground_truth()
    class_gt = class_ground_truth()
    print(f'[frame-level] {len(is_defect):,} frames, {int(is_defect.sum()):,} GT-defect '
          f'({100*is_defect.mean():.1f}%), {int((~is_defect).sum()):,} GT-clean')
    print(f'[frame-level] {len(class_gt)} defect classes: ' +
          ', '.join(f'{c}={int(v.sum())}' for c, v in sorted(class_gt.items())))

    out = {}
    if args.solo or run_all:
        out['solo'] = solo_report(is_defect, class_gt)
    if args.cascade or run_all:
        out['cascade'] = cascade_report(is_defect, class_gt)
    if args.vote or run_all:
        out['vote'] = vote_report(is_defect, class_gt)

    if args.html_report:
        best = find_best_solo(is_defect, fa_budget=0.05)
        if best is None:
            print('[html-report] no solo config holds frame FA<=5% -- skipping.')
        else:
            data = build_confusion_sweep_data(best['run'], best['step'], best['kernel'],
                                              best['min_votes'], best['t'], is_defect, class_gt)
            html = render_confusion_sweep_html(data, fa_budget=0.05)
            with open(args.html_report, 'w') as fh:
                fh.write(html)
            print(f"[html-report] best solo: {best['run']} ({MODELS[best['run']]}) "
                  f"step={best['step']} detect%={best['detect_pct']:.2f} "
                  f"FA%={best['fa_pct']:.2f} kernel={best['kernel']} "
                  f"min_votes={best['min_votes']} -> wrote {args.html_report}")

    if out:
        # Merge into whatever's already on disk rather than clobbering it -- e.g. a
        # `--html-report`-only run (or any other partial subset) must not blow away
        # solo/cascade/vote rows an earlier full run already wrote.
        path = os.path.join(CACHE_DIR, 'frame_level_sweep_results.json')
        existing = {}
        if os.path.exists(path):
            with open(path) as fh:
                existing = json.load(fh)
        existing.update(out)
        json.dump(existing, open(path, 'w'), indent=1)
        print(f'\nwrote {path}')


if __name__ == '__main__':
    main()
