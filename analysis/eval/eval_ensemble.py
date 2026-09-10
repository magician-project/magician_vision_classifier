#!/usr/bin/env python3
"""Ensemble search over the current Aug26_78K zoo, on the COVERAGE validation set, or (with
--mode traintest) on the in-distribution train/test-split campaign's own held-out set.

Soft-averages defect_mass = 1 - P(clean) across a subset of models, then recomputes the
exact same metric the corresponding single-model scorer reports:
  --mode coverage  (default): eval_coverage.py's TIER_A (recording-disjoint, honest) macro.
  --mode traintest           : eval_traintest_split.py's macro over ALL non-clean classes
                                (no tiers -- the whole split is leakage-expected by design).
A solo-model row here reproduces that model's own single-model number exactly (up to
shuffle-order-independent float rounding), so every subset's number is directly comparable
to the existing per-model numbers in 24-8-report.md / 2-9-report.md respectively.

METHODOLOGY, carried over from knowledge/31-6-report.md Sec.10 (the mix_* campaign's own
ensemble analysis) and PLAN.md's ensemble-pool findings, reapplied to the current zoo:
  - Select on DETECTION (macro detect@FA5), never on N-way balanced accuracy -- that
    objective has previously been shown to pick members that raise false alarms (31-6
    Sec.10's balanced-accuracy pick added `custom`, which over-fires on clean).
  - GREEDY FORWARD selection over a candidate pool, not exhaustive search and not just the
    top-N solo performers -- "ensemble value is not single-model value" (PLAN.md): a model
    that loses solo can still improve a subset, and top solo performers from the same
    architecture family can be too correlated to help each other.
  - Stop greedy as soon as no remaining candidate improves the objective. No forced size.
  - Report compute cost alongside accuracy: an N-model soft ensemble costs N times the
    forward-pass compute of the single best member, which competes directly with the 23 Hz
    deployment gate every report in this repo has used.

CACHING: each candidate's defect_mass over the shared held-out tiles is computed ONCE -- the
only GPU-heavy step, one forward pass per model -- and written to
experiments/ensemble_cache/<run>_<model>_mass_<mode>.npy. The search itself (combining
cached arrays, recomputing the FA threshold + macro per subset) is pure numpy, so a
different pool or objective can be re-run for free without re-scoring anything.

Usage:
    python -m analysis.eval.eval_ensemble [--mode coverage|traintest]
    python -m analysis.eval.eval_ensemble --mode traintest --cache-only
    python -m analysis.eval.eval_ensemble --mode traintest --search-only
    python -m analysis.eval.eval_ensemble --pool full   # all 49 coverage-campaign models,
                                                          # not just the 13-model curated set
"""

import glob
import json
import os
import re
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

from mvc.core.artifact_paths import find_config_with_classes
from mvc.core.config import load_hyperparameters
from mvc.core.datasets import build_val_only, clean_class_index
from mvc.core.lit_classifier import Classifier
from mvc.core.metrics import detection_at_fa, fa_threshold
from analysis.eval.eval_coverage import build_coverage_loader

CACHE_DIR = 'experiments/ensemble_cache'

# (run, model, note) -- top individual TIER_A performers from 24-8-report.md's full zoo
# map, chosen for architectural spread (ConvNeXt/V2, ResNet, RepVGG, NAS-derived TinyNet/
# HardcoreNAS, EdgeNeXt hybrid conv/attn, SqueezeNet, MobileNetV4), not purely by solo rank
# -- per PLAN.md, ensembling only the top-N of one family risks correlated errors.
COVERAGE_POOL = [
    ('anc',        'convnext_pico',            'incumbent'),
    ('msfemto',    'convnext_femto',           'closest ConvNeXt sibling'),
    ('fztinyc',    'timm/tinynet_c',           'NAS-derived, different family'),
    ('msedgex',    'edgenext_xx_small',        'hybrid conv/attention'),
    ('fzv2femto',  'timm/convnextv2_femto',    'ConvNeXt V2'),
    ('msatto',     'convnext_atto',            'ConvNeXt'),
    ('msnano',     'convnext_nano',            'ConvNeXt, most params in pool'),
    ('fzv2atto',   'timm/convnextv2_atto',     'ConvNeXt V2'),
    ('fzr18',      'resnet18',                 'ResNet, unrelated family'),
    ('msrepvgg',   'repvgg_a0',                'RepVGG, front model'),
    ('fzhcnas',    'timm/hardcorenas_a',       'HardcoreNAS'),
    ('fzsqueeze',  'squeezenet1_1',            'SqueezeNet, front model, cheapest'),
    ('msmnv4',     'mobilenetv4_conv_small',   'modern mobile baseline'),
]

def full_coverage_pool():
    """Every model the coverage campaign actually scored: the incumbent + all of
    full_zoo_sweep.py's TAGS + all of model_sweep.py's CANDIDATES -- the same 49-model
    union full_zoo_report.py prints ("49 models scored"), not the 13-model curated subset
    above. Built from the same source dicts those two sweeps use, so this can't drift out
    of sync with what actually got trained."""
    from analysis.sweeps.full_zoo_sweep import TAGS as ZOO_TAGS
    from analysis.sweeps.model_sweep import CANDIDATES as MS_CANDIDATES
    pool = [('anc', 'convnext_pico', 'incumbent')]
    pool += [(f'fz{tag}', model, 'full-zoo') for model, tag in ZOO_TAGS.items()]
    pool += [(f'ms{tag}', model, 'model-sweep') for tag, model, *_ in MS_CANDIDATES]
    return pool


# Top individual performers from the in-distribution train/test-split campaign
# (knowledge/2-9-report.md), same architectural-spread reasoning as COVERAGE_POOL. This is
# a DIFFERENT training run per model (dataloader.frozen_tile_split, not exclude_frames) --
# disjoint checkpoints from COVERAGE_POOL even where the architecture name matches.
TRAINTEST_POOL = [
    ('ttcnxtiny',   'convnext_tiny',        'top in-distribution fit'),
    ('tteffv2s',    'efficientnet_v2_s',    'EfficientNetV2'),
    ('tteffb0',     'efficientnet_b0',      'EfficientNet'),
    ('ttfastvit',   'fastvit_t8',           'hybrid conv/attn'),
    ('ttdense121',  'densenet121',          'DenseNet, unrelated family'),
    ('ttrx50',      'resnext50',            'ResNeXt'),
    ('ttcnxpico',   'convnext_pico',        'incumbent'),
    ('ttregy800',   'regnet_y_800mf',       'RegNet'),
    ('ttfemto',     'convnext_femto',       'ConvNeXt, small'),
    ('tthcnas',     'timm/hardcorenas_a',   'HardcoreNAS'),
    ('tttinyc',     'timm/tinynet_c',       'NAS-derived, different family'),
    ('ttr18',       'resnet18',             'ResNet'),
]


def best_ckpt(cfg):
    ck_dir = cfg['checkpoint_dir']
    cks = sorted(glob.glob(os.path.join(ck_dir, '*.ckpt')))
    if not cks:
        return None
    mon = cfg.get('checkpoint_monitor', 'val_detect_auroc')
    mode = 'max' if cfg.get('checkpoint_mode', 'max') == 'max' else 'min'
    scored = [(float(m.group(1)), c) for c in cks
              if (m := re.search(rf'{re.escape(mon)}=([0-9]+\.[0-9]+)', c))]
    return (max if mode == 'max' else min)(scored)[1] if scored else cks[-1]


def cache_one(mode, run, model, subset, class_names, clean_id):
    sfx = model.replace('/', '_')
    out = os.path.join(CACHE_DIR, f'{run}_{sfx}_mass_{mode}.npy')
    if os.path.exists(out):
        return out
    cfg_path = find_config_with_classes(f'{run}_{sfx}.json')
    if not cfg_path:
        print(f'  !! no config found for {run}_{sfx}, skipping')
        return None
    cfg = load_hyperparameters(cfg_path)
    assert cfg['model'] == model, f'{cfg_path}: expected {model}, got {cfg["model"]}'
    assert list(cfg.get('classes') or []) == class_names or not cfg.get('classes'), \
        f'{cfg_path}: class order does not match the shared held-out set -- ' \
        f'combining scores across models requires an identical class scheme'
    ckpt = best_ckpt(cfg)
    if ckpt is None:
        print(f'  !! no checkpoint for {run}_{sfx}, skipping')
        return None
    net = Classifier.from_config(cfg, num_classes=len(class_names), clean_class=clean_id)
    net.load_state_dict(torch.load(ckpt, weights_only=False, map_location='cpu')['state_dict'])
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.to(dev).eval()
    loader = DataLoader(subset, batch_size=cfg['hparams']['batch_size'], shuffle=False,
                         num_workers=8)
    mass = []
    with torch.no_grad():
        for bi, (x, y) in enumerate(loader):
            prob = torch.softmax(net(x.to(dev)).float(), dim=1).cpu().numpy()
            mass.append(1.0 - prob[:, clean_id])
            if bi % 50 == 0:
                print(f'  {run}_{sfx}: batch {bi}/{len(loader)}', end='\r')
    print()
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.save(out, np.concatenate(mass).astype(np.float32))
    del net
    torch.cuda.empty_cache()
    return out


def macro_at_threshold(mass, truth, clean_id, class_names, tiers, thr):
    sel = [i for i, c in enumerate(class_names)
           if i != clean_id and (truth == i).any()
           and (tiers is None or tiers.get(c) == 'TIER_A')]
    if not sel:
        return None
    return float(np.mean([(mass[truth == i] >= thr).mean() * 100.0 for i in sel]))


def confusion_at_threshold(mass, truth, clean_id, thr):
    """Binary defect-vs-clean confusion at a threshold -- TP/FN are defect tiles correctly
    flagged/missed, TN/FP are clean tiles correctly passed/wrongly flagged."""
    is_clean = truth == clean_id
    pred_defect = mass >= thr
    tp = int(np.sum(pred_defect & ~is_clean))
    fn = int(np.sum(~pred_defect & ~is_clean))
    fp = int(np.sum(pred_defect & is_clean))
    tn = int(np.sum(~pred_defect & is_clean))
    return tp, tn, fp, fn


# Threshold is not fixed at 5% FA (every other report in this repo's convention) -- it is
# swept across these FA budgets and the resulting operating points are all reported side by
# side. NOTE: unconstrained "maximize macro detect" is mathematically degenerate here --
# detection is monotonically non-increasing in threshold (a stricter threshold can only ever
# drop tiles, never add them), so an unbounded search always just drives the threshold to
# zero (flag everything, TP and FP both saturate). A meaningful "optimal" threshold only
# exists relative to an false-alarm budget, so this scans several budgets instead of
# pretending one FA-free optimum exists.
FA_TARGETS = (0.01, 0.02, 0.05, 0.10, 0.20)


def threshold_scan(mass, truth, clean_id, class_names, tiers):
    is_clean = truth == clean_id
    rows = []
    for fa in FA_TARGETS:
        thr = fa_threshold(mass, is_clean, fa)
        tp, tn, fp, fn = confusion_at_threshold(mass, truth, clean_id, thr)
        macro = macro_at_threshold(mass, truth, clean_id, class_names, tiers, thr)
        rows.append({'fa_target': fa, 'threshold': thr, 'tp': tp, 'tn': tn, 'fp': fp,
                     'fn': fn, 'macro': macro})
    best = max(rows, key=lambda r: r['macro'] if r['macro'] is not None else -1.0)
    return rows, best


# Reasonable operating range for the per-metric extremes below -- same convention
# eval_vote_curve.py uses (0.30:0.995:0.01). Not the raw [0,1] range: threshold=0 (flag
# literally every tile) or threshold=1 (flag nothing) are mathematically "optimal" for a
# single metric but operationally meaningless, so bounding to a range anyone would actually
# consider deploying is what makes reporting a "best FN" or "best FP" informative rather
# than trivially restating the scan's own endpoints.
EXTREME_RANGE = np.round(np.arange(0.30, 0.995, 0.01), 4)


def metric_extremes(mass, truth, clean_id):
    """TP/FN and FP/TN each move monotonically opposite ways as threshold rises (a
    stricter gate can only ever pass MORE tiles through, never fewer) -- so the best
    achievable value of any ONE of the four, on its own, is always at one end of the
    threshold range searched. That is not a bug in the search; it is what "optimize one
    count in isolation" means. Reporting it plainly (with the real counts, over a bounded,
    operationally sane range) is more honest than picking a single blended number and
    calling it optimal."""
    lo, hi = EXTREME_RANGE.min(), EXTREME_RANGE.max()
    tp_lo, tn_lo, fp_lo, fn_lo = confusion_at_threshold(mass, truth, clean_id, lo)
    tp_hi, tn_hi, fp_hi, fn_hi = confusion_at_threshold(mass, truth, clean_id, hi)
    return {
        'max_TP': (lo, tp_lo), 'min_FN': (lo, fn_lo),   # both at the lowest threshold
        'max_TN': (hi, tn_hi), 'min_FP': (hi, fp_hi),   # both at the highest threshold
    }


def print_extremes(mass, truth, clean_id, label='  '):
    ext = metric_extremes(mass, truth, clean_id)
    print(f'{label}per-metric optima over thr in [{EXTREME_RANGE.min():.2f}, '
          f'{EXTREME_RANGE.max():.2f}] (monotonic -- each sits at one end, see note above):')
    for name, (thr, val) in ext.items():
        print(f'{label}  {name:7s} = {val:8,d}  at thr={thr:.3f}')


def print_scan(rows, label='  '):
    print(f'{label}{"FA target":>9s} {"thr":>6s} {"TP":>8s} {"TN":>8s} {"FP":>8s} '
          f'{"FN":>8s} {"macro%":>7s}')
    for r in rows:
        print(f'{label}{r["fa_target"]*100:8.0f}% {r["threshold"]:6.3f} {r["tp"]:8,d} '
              f'{r["tn"]:8,d} {r["fp"]:8,d} {r["fn"]:8,d} {r["macro"]:7.2f}')


def main():
    mode = 'traintest' if '--mode' in sys.argv and \
        sys.argv[sys.argv.index('--mode') + 1] == 'traintest' else 'coverage'
    cache_only = '--cache-only' in sys.argv
    search_only = '--search-only' in sys.argv
    wide = '--pool' in sys.argv and sys.argv[sys.argv.index('--pool') + 1] == 'full'
    if wide:
        assert mode == 'coverage', '--pool full is only wired for --mode coverage'
        POOL = full_coverage_pool()
    else:
        POOL = TRAINTEST_POOL if mode == 'traintest' else COVERAGE_POOL
    metric_label = ('macro detect@FA5 (in-distribution, leakage expected)' if mode == 'traintest'
                     else 'TIER_A macro detect@FA5')

    # Build the held-out tile set ONCE, from any one candidate's config -- every model in a
    # given mode trains off the same underlying dataset.h5 with the same frozen carve-out/
    # split, so the tile subset (and its ordering) is identical across all of them.
    ref_run, ref_model, _ = POOL[0]
    ref_cfg_path = find_config_with_classes(f'{ref_run}_{ref_model.replace("/", "_")}.json')
    ref_cfg = load_hyperparameters(ref_cfg_path)
    if mode == 'traintest':
        split = build_val_only(ref_cfg)
        class_names, subset, tiers = list(split.classes), split.val, None
    else:
        ds, subset, payload = build_coverage_loader(ref_cfg)
        class_names = list(ds.classes)
        tiers = {k: v['tier'] for k, v in payload['classes'].items()}
    clean_id = clean_class_index(class_names)
    n_sel = sum(1 for c in class_names if c != 'class_clean' and
                (tiers is None or tiers.get(c) == 'TIER_A'))
    print(f'[ensemble:{mode}] {len(subset):,} tiles, {len(class_names)} classes, '
          f'{n_sel} classes in the macro\n')

    if not search_only:
        for run, model, note in POOL:
            print(f'--- caching {run}_{model} ({note}) ---')
            cache_one(mode, run, model, subset, class_names, clean_id)
    if cache_only:
        return

    # Reload truth alongside each cached mass array once, straight off the loader, so the
    # tile order used for `truth` is provably the same order every cache_one() call used
    # (same `subset`, shuffle=False) rather than trusted by construction.
    loader = DataLoader(subset, batch_size=ref_cfg['hparams']['batch_size'], shuffle=False,
                        num_workers=8)
    truth = np.concatenate([y.numpy() for _, y in loader])

    masses, names = {}, []
    for run, model, note in POOL:
        sfx = model.replace('/', '_')
        p = os.path.join(CACHE_DIR, f'{run}_{sfx}_mass_{mode}.npy')
        if not os.path.exists(p):
            print(f'  (skipping {run}_{model}, no cache)')
            continue
        m = np.load(p)
        assert len(m) == len(truth), f'{p}: {len(m)} tiles, expected {len(truth)}'
        masses[run] = m
        names.append(run)

    print(f'\n=== solo models ({metric_label}) -- threshold scanned, not fixed at FA5 ===')
    solo_best = {}
    for run in names:
        model = next(m for r, m, _ in POOL if r == run)
        rows, best_row = threshold_scan(masses[run], truth, clean_id, class_names, tiers)
        solo_best[run] = best_row['macro']
        print(f'\n  {run} ({model}):')
        print_scan(rows, label='    ')
        print(f'    best of the above: {best_row["macro"]:.2f}% at FA target '
              f'{best_row["fa_target"]*100:.0f}% (thr={best_row["threshold"]:.3f})')
        print_extremes(masses[run], truth, clean_id, label='    ')

    def ens_best(subset_runs):
        combined = np.mean([masses[r] for r in subset_runs], axis=0)
        _, best_row = threshold_scan(combined, truth, clean_id, class_names, tiers)
        return best_row

    print(f'\n=== greedy forward (maximize best-of-scan {metric_label}) ===')
    chosen, best = [], -1.0
    remaining = set(names)
    while remaining:
        scored = {r: ens_best(chosen + [r])['macro'] for r in remaining}
        cand = max(scored, key=scored.get)
        if chosen and scored[cand] <= best + 1e-9:
            print(f'  stopping: adding {cand} would not improve ({scored[cand]:.2f}% <= '
                  f'{best:.2f}%)')
            break
        chosen.append(cand)
        remaining.discard(cand)
        best = scored[cand]
        model = next(m for r, m, _ in POOL if r == cand)
        print(f'  + {cand:12s} {model:28s} -> {[*chosen]}  best {best:.2f}%  (N={len(chosen)})')

    print(f'\nBEST ensemble found: {chosen}')
    print(f'  models: {[next(m for r, m, _ in POOL if r == c) for c in chosen]}')
    combined = np.mean([masses[r] for r in chosen], axis=0)
    rows, best_row = threshold_scan(combined, truth, clean_id, class_names, tiers)
    print_scan(rows)
    print(f'  best of the above: {best_row["macro"]:.2f}% at FA target '
          f'{best_row["fa_target"]*100:.0f}% (thr={best_row["threshold"]:.3f})')
    print_extremes(combined, truth, clean_id)

    best_solo_run = max(solo_best, key=solo_best.get)
    best_solo_model = next(m for r, m, _ in POOL if r == best_solo_run)
    print(f'\nBest SOLO model: {best_solo_run} ({best_solo_model}) at '
          f'{solo_best[best_solo_run]:.2f}% (best-of-scan)')
    print(f'Ensemble gain over best solo: {best_row["macro"] - solo_best[best_solo_run]:+.2f} pts')


if __name__ == '__main__':
    main()
