#!/usr/bin/env python3
"""Equivalence test for Metrics.py -- the project's KPI, previously implemented eight times.

  [1] CURVE FORM. Metrics.miss_at_fa reproduces both prior curve implementations
      (phase2_select.py's hand interpolation and score_checkpoints.py's np.interp) on every
      real *_threshold_curve.json in the tree.

  [2] SCORE FORM. Metrics.miss_at_fa_from_scores reproduces the two eval_domain_split.py
      copies and detection_ensemble.py's, and Metrics.fa_threshold reproduces
      evaluateDetection.py's, on random score distributions including degenerate ones.

  [3] THE ESTIMATORS ARE NOT INTERCHANGEABLE. eval_ema_tta.py used a constrained-maximum
      rule rather than a quantile. This measures how far apart they actually are, on score
      distributions with the saturation ties that softmax outputs really have -- because
      "close enough" is a claim that needs a number attached, not an assumption.

Run:  python test_metrics.py
"""

import glob
import json
import sys

import numpy as np

# Runnable both as `python -m tests.test_metrics` and as `python tests/test_metrics.py`.
# Run directly, sys.path[0] is tests/ rather than the repo root, so `mvc` is not importable
# -- and this file has a __main__ block, so direct invocation is a supported entry point.
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mvc.core.metrics import (detection_at_fa, fa_threshold, miss_at_fa, miss_at_fa_from_scores,
                     miss_at_fa_sweep)

failures = []


def check(cond, msg):
    if not cond:
        failures.append(msg)
    return cond


# ------------------------------------------------------------------ [1] curve form
def old_phase2_select(curve_path, targets=(0.05, 0.10)):
    """phase2_select.miss_at_fa, transcribed before removal."""
    d = json.load(open(curve_path))
    pts = (d.get('sweeps') or {}).get('defect_mass') or d['sweep']
    fa = [p['false_alarm'] for p in pts]
    det = [p['detected'] for p in pts]
    order = sorted(range(len(fa)), key=lambda i: fa[i])
    fa = [fa[i] for i in order]
    det = [det[i] for i in order]

    def interp(t):
        if t <= fa[0]:
            return det[0]
        if t >= fa[-1]:
            return det[-1]
        for i in range(1, len(fa)):
            if fa[i] >= t:
                span = fa[i] - fa[i - 1]
                w = 0.0 if span == 0 else (t - fa[i - 1]) / span
                return det[i - 1] + w * (det[i] - det[i - 1])
        return det[-1]

    return {t: (1.0 - interp(t)) * 100.0 for t in targets}


def old_score_checkpoints(curve_path, targets=(0.05, 0.10)):
    """score_checkpoints.miss_at_fa, transcribed before removal."""
    d = json.load(open(curve_path))
    pts = (d.get('sweeps') or {}).get('defect_mass') or d['sweep']
    fa = np.array([p['false_alarm'] for p in pts], float)
    det = np.array([p['detected'] for p in pts], float)
    o = np.argsort(fa)
    fa, det = fa[o], det[o]
    return {t: (1.0 - float(np.interp(t, fa, det))) * 100.0 for t in targets}


def test_curve_form():
    curves = sorted(glob.glob('*_threshold_curve.json') +
                    glob.glob('experiments/**/*_threshold_curve.json', recursive=True))
    worst = {'phase2_select': 0.0, 'score_checkpoints': 0.0}
    n = 0
    for c in curves:
        try:
            ref_p, ref_s, got = old_phase2_select(c), old_score_checkpoints(c), miss_at_fa(c)
        except Exception as exc:
            failures.append(f'{c}: {exc}')
            continue
        n += 1
        for t in (0.05, 0.10):
            worst['phase2_select'] = max(worst['phase2_select'], abs(ref_p[t] - got[t]))
            worst['score_checkpoints'] = max(worst['score_checkpoints'], abs(ref_s[t] - got[t]))
    check(n > 0, 'no threshold curves found to test against')
    for k, v in worst.items():
        check(v == 0.0, f'Metrics.miss_at_fa differs from {k} by up to {v} pp')
    print(f'  [1] curve form: exact match to both prior implementations on {n} real curves')


# ------------------------------------------------------------------ [2] score form
def old_domain_split(s, isdef, mask, fa_percent):
    """eval_domain_split.miss_at_fa, transcribed. Note fa as a PERCENTAGE."""
    clean = mask & ~isdef
    if clean.sum() == 0 or (mask & isdef).sum() == 0:
        return float('nan')
    thr = np.quantile(s[clean], 1 - fa_percent / 100.0)
    return 100.0 * (~(s >= thr)[mask & isdef]).mean()


def old_evaluate_detection_thr(s, isdef, mask, budget):
    """evaluateDetection.thr_for_fp, transcribed."""
    return np.quantile(s[(~isdef) & mask], 1 - budget / 100.0)


def old_eval_coverage(mass, truth, clean_id, n, support, fa):
    """eval_coverage.py's INLINE copy, transcribed before removal.

    This one is here because it was the copy that was NOT transcribed the first time, and
    converting it is what broke a live coverage run -- the tests covered Metrics.py in
    isolation but never compared it against the block it replaced.
    """
    clean_mass = np.sort(mass[truth == clean_id])
    if len(clean_mass) == 0:
        return None
    thr = float(np.quantile(clean_mass, 1.0 - fa))
    return thr, {i: float((mass[truth == i] >= thr).mean() * 100.0)
                 for i in range(n) if support[i] and i != clean_id}


def test_eval_coverage_form():
    """Coverage numbers before and after the refactor must be the same number.

    31 models were scored with the inline block and everything after fzv2nano with
    Metrics. If these disagreed at all, the sweep would stop being one comparable table.
    """
    rng = np.random.default_rng(7)
    n_classes, clean_id = 6, 5
    for label, spread in (('separable', 6.0), ('overlapping', 1.4), ('saturated', 0.0)):
        truth = rng.integers(0, n_classes, 20000)
        mass = np.clip(rng.normal(0.5, 0.2, len(truth))
                       + np.where(truth == clean_id, -spread * 0.05, spread * 0.05), 0, 1)
        if label == 'saturated':
            mass = np.round(mass, 1)
        support = np.array([(truth == i).sum() for i in range(n_classes)])
        for fa in (0.05, 0.10):
            ref = old_eval_coverage(mass, truth, clean_id, n_classes, support, fa)
            is_clean = truth == clean_id
            got_thr = fa_threshold(mass, is_clean, fa)
            got = {i: detection_at_fa(mass, is_clean, truth == i, fa)
                   for i in range(n_classes) if support[i] and i != clean_id}
            check(abs(ref[0] - got_thr) < 1e-12,
                  f'{label} fa={fa}: threshold {got_thr} != inline {ref[0]}')
            for i, v in ref[1].items():
                check(abs(v - got[i]) < 1e-12,
                      f'{label} fa={fa} class {i}: detection {got[i]} != inline {v}')
        print(f'  [4] eval_coverage form: {label:12s} exact match to the inline block')


def test_score_form():
    rng = np.random.default_rng(0)
    cases = {
        'separable': lambda n: (rng.beta(2, 8, n), rng.beta(8, 2, n)),
        'overlapping': lambda n: (rng.beta(3, 4, n), rng.beta(4, 3, n)),
        # softmax outputs really do pile up at the ends; ties are the interesting case
        'saturated': lambda n: (np.round(rng.beta(1, 12, n), 2), np.round(rng.beta(12, 1, n), 2)),
    }
    for label, gen in cases.items():
        clean_s, def_s = gen(4000)
        s = np.concatenate([clean_s, def_s])
        isdef = np.concatenate([np.zeros(len(clean_s), bool), np.ones(len(def_s), bool)])
        mask = np.ones(len(s), bool)
        for fa in (0.05, 0.10):
            ref = old_domain_split(s, isdef, mask, fa * 100)
            got = miss_at_fa_from_scores(s, mask & ~isdef, mask & isdef, fa)
            check(abs(ref - got) < 1e-12,
                  f'{label} fa={fa}: miss_at_fa_from_scores {got} != domain_split {ref}')
            rt = old_evaluate_detection_thr(s, isdef, mask, fa * 100)
            gt = fa_threshold(s, mask & ~isdef, fa)
            check(abs(rt - gt) < 1e-12,
                  f'{label} fa={fa}: fa_threshold {gt} != evaluateDetection {rt}')
            check(abs((100.0 - detection_at_fa(s, mask & ~isdef, mask & isdef, fa)) - got) < 1e-12,
                  f'{label} fa={fa}: detection/miss are not complements')
        print(f'  [2] score form: {label:12s} exact match to domain_split + evaluateDetection')

    # degenerate inputs must return nan, not raise -- one prior copy had no guard
    empty = np.zeros(0, bool)
    check(np.isnan(miss_at_fa_from_scores(np.array([0.5]), empty, empty, 0.05)),
          'empty input should be nan, not an exception')
    print('  [2] score form: empty clean/defect selection returns nan instead of raising')


# ------------------------------------------- [3] the estimators are NOT interchangeable
def test_estimators_differ():
    rng = np.random.default_rng(1)
    rows = []
    # Deliberately OVERLAPPING: on a well-separated problem the miss rate is 0 at both
    # estimators and the comparison is vacuous. The KPI is only interesting where the
    # distributions overlap, so that is where they have to be compared.
    for label, round_to in (('continuous', None), ('2dp ties', 2), ('1dp ties', 1)):
        clean_s = rng.beta(2, 5, 4000)
        def_s = rng.beta(3, 3, 4000)
        if round_to is not None:
            clean_s, def_s = np.round(clean_s, round_to), np.round(def_s, round_to)
        s = np.concatenate([clean_s, def_s])
        isc = np.concatenate([np.ones(len(clean_s), bool), np.zeros(len(def_s), bool)])
        isd = ~isc
        for fa in (0.05, 0.10):
            q = miss_at_fa_from_scores(s, isc, isd, fa)
            w = miss_at_fa_sweep(s, isc, isd, fa)
            # What false alarm does the quantile threshold ACTUALLY produce? With ties it
            # can overshoot the budget the KPI claims to match, which is the whole reason
            # the two estimators disagree -- the sweep rule refuses to exceed `fa`.
            thr = fa_threshold(s, isc, fa)
            realised = float((s[isc] >= thr).mean())
            rows.append((label, fa, q, w, abs(q - w), realised))
    print('  [3] quantile vs constrained-maximum (eval_ema_tta\'s rule):')
    print(f"      {'distribution':14}{'fa':>6}{'quantile':>11}{'sweep':>10}{'gap (pp)':>11}"
          f"{'realised FA':>13}")
    for label, fa, q, w, gap, realised in rows:
        flag = '  <-- over budget' if realised > fa + 1e-9 else ''
        print(f'      {label:14}{fa:6.2f}{q:11.3f}{w:10.3f}{gap:11.3f}{realised:13.4f}{flag}')
    worst = max(r[4] for r in rows)
    # Not an assertion of agreement -- a record of how far apart they are, so that
    # unifying them later is a decision made with the number in hand.
    print(f'      max gap {worst:.3f} pp  (campaign noise floor: factory sd 1.01, '
          f'coverage TIER_A sd 0.43)')


def main():
    print('Metrics.py equivalence test\n')
    test_curve_form()
    test_score_form()
    test_eval_coverage_form()
    test_estimators_differ()
    print()
    if failures:
        print(f'FAILED ({len(failures)}):')
        for f in failures:
            print('  -', f)
        return 1
    print('all checks passed')
    return 0


if __name__ == '__main__':
    sys.exit(main())
