#!/usr/bin/env python3
"""Evaluate the two-stage CASCADE ensemble strategy on the coverage set, reusing already-
cached per-tile scores from eval_ensemble.py (experiments/ensemble_cache/) -- no new
inference. This is the structural alternative flagged but unevaluated in knowledge/
7-9-report.md §8.3: mvc.inference.ensemble_classifier.EnsembleClassifierPnm's live
mechanism, simplified to its cost/accuracy essentials (its exact voting/legend/heatmap
plumbing is ROS-node concerns this eval doesn't need).

THE MECHANISM (confirmed by reading ensemble_classifier.py directly): a STAGE-1 model
classifies every tile; any tile whose defect_mass clears a screening threshold t1 is
"promoted" to a STAGE-2 model (tile_and_cast_selected_tiles_torch -- literally built to
only re-tile the promoted subset). Tiles stage 1 does NOT promote are locked "clean" --
stage 2 never gets a chance to reconsider them, so **stage-1's screening recall at t1 is a
hard ceiling on what the cascade can ever detect**, independent of how good stage 2 is.

COMPUTE MODEL: stage 1 always runs on every tile (cost 1/hz1 per tile); stage 2 only runs
on the promoted fraction (cost promoted_fraction/hz2). Sequential frame Hz:
    hz_cascade = 1 / (1/hz1 + promoted_fraction * 1/hz2)
This is the entire premise of a cascade over the flat soft-ensemble evaluated in
eval_ensemble.py: if promoted_fraction is small (most tiles really are clean), an
expensive stage-2 model that could never ship solo becomes affordable as a selective
re-check.

Usage:
    python -m analysis.eval.eval_cascade_ensemble <stage1_run> <stage2_run> [<stage2_run2> ...]
    e.g. python -m analysis.eval.eval_cascade_ensemble anc fzcnxtiny
"""

import json
import os
import sys

import numpy as np
from torch.utils.data import DataLoader

from mvc.core.artifact_paths import find_artifact, find_config_with_classes
from mvc.core.config import load_hyperparameters
from mvc.core.datasets import clean_class_index
from analysis.eval.eval_coverage import build_coverage_loader
from analysis.eval.eval_ensemble import (CACHE_DIR, confusion_at_threshold,
                                          full_coverage_pool, macro_at_threshold)

TARGET_HZ = 23.0
# Screening should be PERMISSIVE (catch nearly everything, precision doesn't matter yet) --
# a much lower range than eval_ensemble.py's FA-budget-derived thresholds, which are
# already tuned for final-decision precision, not a first-pass screen.
T1_GRID = np.round(np.arange(0.05, 0.65, 0.05), 3)
T2_GRID = np.round(np.arange(0.30, 0.99, 0.03), 3)


def bench_hz():
    hz = {}
    for fname in ('phase4_inference_bench.json', 'zoo_inference_bench.json'):
        p = find_artifact(fname)
        if not p:
            continue
        for r in json.load(open(p))['rows']:
            key = r['model']
            if key not in hz or r.get('variant', '').lower() == 'fused':
                hz[key] = r.get('hz_step16', 0.0) * 1.6
    return hz


def load_mass(run, model):
    sfx = model.replace('/', '_')
    p = os.path.join(CACHE_DIR, f'{run}_{sfx}_mass_coverage.npy')
    if not os.path.exists(p):
        sys.exit(f'no cached mass for {run}_{model} -- run '
                 f'"python -m analysis.eval.eval_ensemble --pool full --cache-only" first')
    return np.load(p)


def main():
    args = sys.argv[1:]
    if len(args) < 2:
        sys.exit(__doc__.strip().splitlines()[-1])
    stage1_run, stage2_runs = args[0], args[1:]

    pool = {r: m for r, m, _ in full_coverage_pool()}
    for r in [stage1_run] + stage2_runs:
        if r not in pool:
            sys.exit(f'{r}: not in full_coverage_pool() -- see eval_ensemble.py for valid run tags')
    hz = bench_hz()

    ref_cfg_path = find_config_with_classes(f'{stage1_run}_{pool[stage1_run].replace("/", "_")}.json')
    ref_cfg = load_hyperparameters(ref_cfg_path)
    ds, subset, payload = build_coverage_loader(ref_cfg)
    class_names = list(ds.classes)
    clean_id = clean_class_index(class_names)
    tiers = {k: v['tier'] for k, v in payload['classes'].items()}
    loader = DataLoader(subset, batch_size=ref_cfg['hparams']['batch_size'], shuffle=False,
                         num_workers=8)
    truth = np.concatenate([y.numpy() for _, y in loader])
    print(f'[cascade] {len(truth):,} coverage tiles\n')

    m1 = load_mass(stage1_run, pool[stage1_run])
    hz1 = hz.get(pool[stage1_run])
    if hz1 is None:
        sys.exit(f'no benched Hz for {pool[stage1_run]}')

    for s2_run in stage2_runs:
        m2 = load_mass(s2_run, pool[s2_run])
        hz2 = hz.get(pool[s2_run])
        if hz2 is None:
            print(f'!! no benched Hz for {pool[s2_run]}, skipping')
            continue

        print(f'=== stage1={stage1_run} ({pool[stage1_run]}, {hz1:.1f} Hz solo) -> '
              f'stage2={s2_run} ({pool[s2_run]}, {hz2:.1f} Hz solo) ===')

        results = []
        for t1 in T1_GRID:
            promoted = m1 >= t1
            promoted_frac = float(promoted.mean())
            hz_cascade = 1.0 / (1.0 / hz1 + promoted_frac / hz2)
            final_mass_base = np.where(promoted, m2, 0.0)
            for t2 in T2_GRID:
                macro = macro_at_threshold(final_mass_base, truth, clean_id, class_names, tiers, t2)
                if macro is None:
                    continue
                tp, tn, fp, fn = confusion_at_threshold(final_mass_base, truth, clean_id, t2)
                results.append(dict(t1=t1, t2=t2, promoted_frac=promoted_frac, hz=hz_cascade,
                                     macro=macro, tp=tp, tn=tn, fp=fp, fn=fn))

        deployable = [r for r in results if r['hz'] >= TARGET_HZ]
        if deployable:
            b = max(deployable, key=lambda r: r['macro'])
            print(f"  BEST DEPLOYABLE (hz>={TARGET_HZ:.0f}): t1={b['t1']:.2f} t2={b['t2']:.2f} "
                  f"promoted={b['promoted_frac']*100:5.1f}% hz={b['hz']:5.1f} macro={b['macro']:6.2f}%  "
                  f"TP={b['tp']:,} TN={b['tn']:,} FP={b['fp']:,} FN={b['fn']:,}")
        else:
            best_hz = max(results, key=lambda r: r['hz'])
            print(f"  NO deployable point in this grid -- best Hz achieved was "
                  f"{best_hz['hz']:.2f} (t1={best_hz['t1']:.2f}, promoted="
                  f"{best_hz['promoted_frac']*100:.1f}%), still under {TARGET_HZ:.0f}")

        b_all = max(results, key=lambda r: r['macro'])
        print(f"  BEST OVERALL (ignoring Hz):      t1={b_all['t1']:.2f} t2={b_all['t2']:.2f} "
              f"promoted={b_all['promoted_frac']*100:5.1f}% hz={b_all['hz']:5.1f} "
              f"macro={b_all['macro']:6.2f}%  TP={b_all['tp']:,} TN={b_all['tn']:,} "
              f"FP={b_all['fp']:,} FN={b_all['fn']:,}")

        # Stage-1's own screening recall at each t1 -- the hard ceiling promised above,
        # shown explicitly so a bad cascade result can be diagnosed (screen missing
        # defects outright) rather than blamed on stage 2.
        is_clean = truth == clean_id
        print(f"  stage-1 screening recall (fraction of REAL defects promoted, before stage2 sees them):")
        for t1 in T1_GRID[::2]:
            promoted = m1 >= t1
            recall = float(promoted[~is_clean].mean())
            print(f"    t1={t1:.2f}  promoted={promoted.mean()*100:5.1f}% of all tiles  "
                  f"defect-recall={recall*100:5.1f}%")
        print()


if __name__ == '__main__':
    main()
