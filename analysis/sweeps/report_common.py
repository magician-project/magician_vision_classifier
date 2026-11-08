#!/usr/bin/env python3
"""Shared primitives for analysis/sweeps/*_report.py.

Extracted 2026-09-10 (knowledge/10-9-plan.md Phase 6) after best_epoch() was independently
reimplemented 5x (full_zoo_report.py, model_sweep_report.py, pfc_variance_report.py,
seed_replicates_report.py, traintest_sweep_report.py -- with small, easy-to-miss
differences: some cap at a max_epoch, some don't; one used glob.glob('*.ckpt') where the
rest used os.listdir), the TIER_A-macro-from-coverage-rows filter was duplicated near
identically in aug26_sweep_report.py/full_zoo_report.py/model_sweep_report.py/
pfc_variance_report.py/seed_replicates_report.py, and the inference-bench loader was
duplicated 2-3x with two different return shapes (a bare hz_step16 scalar per model vs.
the full bench row per model). mono_frontier_report.py already proved this works, by
importing full_zoo_report.score() directly instead of reimplementing it.
"""

import json
import os
import re
from statistics import mean

from mvc.core.artifact_paths import find_artifact


def best_epoch(ckpt_dir, max_epoch=None):
    """Epoch with the highest val_detect_auroc among *ckpt_dir*'s checkpoints.

    Matches `epoch=<N>` and `val_detect_auroc=<F>` in each filename (Lightning's
    ModelCheckpoint naming -- the only files ever written into datasets/mix_ckpts/<run>/).
    max_epoch, if given, excludes any checkpoint past it -- callers scoring a
    fixed-budget replicate (e.g. a 2-epoch screen) use this so a later, differently
    budgeted continuation run filed into the same directory can't get silently picked up.
    Returns None if the directory is missing or nothing matches both patterns.
    """
    if not os.path.isdir(ckpt_dir):
        return None
    best = None
    for b in os.listdir(ckpt_dir):
        m_ep = re.search(r'epoch=(\d+)', b)
        m_auroc = re.search(r'val_detect_auroc=([0-9]+\.[0-9]+)', b)
        if not (m_ep and m_auroc):
            continue
        ep = int(m_ep.group(1))
        if max_epoch is not None and ep > max_epoch:
            continue
        cand = (float(m_auroc.group(1)), ep)
        if best is None or cand[0] > best[0]:
            best = cand
    return best[1] if best else None


def tier_a_macro_from_rows(rows):
    """Mean TIER_A detect@FA5 over an iterable of coverage-json row dicts.

    The recording-disjoint ship-rule column every report reads the same way: TIER_A,
    non-clean, and only where detect_at_fa5 was actually computed. Takes any iterable of
    row dicts -- a plain list (json.load(...)['rows']) or a class-keyed dict's .values()
    (some callers pre-filter/index by class for their own per-class table) both work.
    """
    vals = [r['detect_at_fa5'] for r in rows
            if r.get('tier') == 'TIER_A' and r.get('class') != 'class_clean'
            and r.get('detect_at_fa5') is not None]
    return mean(vals) if vals else None


def tier_a_macro(coverage_path):
    """tier_a_macro_from_rows, loading the coverage json at *coverage_path* first.
    Returns None if coverage_path is falsy (not found)."""
    if not coverage_path:
        return None
    return tier_a_macro_from_rows(json.load(open(coverage_path))['rows'])


def load_bench(filenames=('phase4_inference_bench.json', 'zoo_inference_bench.json')):
    """model -> its full inference-bench row, merged across *filenames* (found via
    find_artifact, so this keeps working after tidy_experiments.py archives them).

    A reparameterizable model deploys FUSED, so that variant's row wins when a model has
    more than one; the loop order also means a later file's row for a model already seen
    only overrides the earlier one when IT is the fused variant, never demotes a fused row
    already recorded back to a non-fused one from a later file.
    """
    out = {}
    for fname in filenames:
        p = find_artifact(fname)
        if not p:
            continue
        for r in json.load(open(p))['rows']:
            key = r['model']
            if key not in out or r.get('variant', '').lower() == 'fused':
                out[key] = r
    return out
