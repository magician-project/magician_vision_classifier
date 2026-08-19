"""THE detection KPI. miss@FA -- miss rate at matched false alarm.

Every ship/no-ship call in PLAN.md turns on this number, and it had six implementations:
phase2_select.py, score_checkpoints.py, eval_domain_split.py (twice), eval_ema_tta.py and
an inline copy in eval_coverage.py. Two reporting tools imported it from phase2_select --
a one-off model-selection script, which therefore could never be moved or deleted without
breaking the reports.

WHAT THE KPI IS
---------------
Score every tile by `defect_mass = 1 - P(clean)`. Pick the threshold at which the CLEAN
tiles produce a false-alarm rate of exactly `fa`. Report the fraction of DEFECT tiles that
score below it -- the misses. Lower is better. Matching the false-alarm rate is what makes
two models comparable: a model can always trade misses for false alarms, so a miss rate
quoted without a matched FA is not a measurement.

TWO ESTIMATORS, AND THEY DO NOT AGREE
------------------------------------
The six copies were not six copies of one thing. They were two different estimators:

  from a CURVE   `miss_at_fa()` reads a saved threshold_curve.json -- (threshold, detected,
                 false_alarm) sampled on a grid -- and interpolates `detected` at the target
                 false alarm. Resolution is limited by the grid the sweep was written on.

  from SCORES    `miss_at_fa_from_scores()` has the per-tile scores, so it takes the exact
                 (1-fa) quantile of the clean scores and counts directly. No interpolation.

The score-based answer is the accurate one; the curve-based answer is what you get when the
scores are gone and only the artifact survives. They are close but NOT equal, so a number
computed one way must not be compared against a number computed the other way. Both live
here, named differently on purpose, so that choice is visible at the call site instead of
being an accident of which file the function was copied into.

A THIRD, SUBTLY DIFFERENT RULE
------------------------------
eval_ema_tta.py swept every unique threshold and took the best detection subject to
`false_alarm <= fa`. That is a CONSTRAINED-MAXIMUM rule, not a quantile: with ties in the
score distribution it lands on a different threshold and reports a different number.
`miss_at_fa_sweep()` preserves it exactly, because silently swapping one estimator for
another under a KPI is the failure this module exists to prevent. Prefer
`miss_at_fa_from_scores` for new work; see test_metrics.py for the measured gap.
"""

import json

import numpy as np

DEFAULT_TARGETS = (0.05, 0.10)


# --------------------------------------------------------------------------- curve
def load_threshold_curve(path):
    """The sweep points from a *_threshold_curve.json, oldest and newest layouts.

    Newer files nest several scores under `sweeps` (the KPI uses `defect_mass`); older
    ones have a single flat `sweep`. Returns a list of dicts with threshold / detected /
    false_alarm.
    """
    with open(path) as fh:
        d = json.load(fh)
    pts = (d.get('sweeps') or {}).get('defect_mass') or d.get('sweep')
    if not pts:
        raise ValueError(f'{path} has no defect_mass sweep and no legacy "sweep" block')
    return pts


def miss_at_fa(curve, targets=DEFAULT_TARGETS):
    """miss rate (%) at each target false alarm, interpolated from a saved sweep.

    `curve` is a path to a threshold_curve.json or an already-loaded list of points.
    Returns {target_fa: miss_percent}.

    Interpolation is in FALSE-ALARM space, not threshold space: the grid is uniform in
    threshold, which makes it dense where the KPI is read (fa near 0.05) only by accident.
    np.interp needs its x ascending, hence the sort -- the curves are written in
    descending false alarm.
    """
    pts = load_threshold_curve(curve) if isinstance(curve, str) else curve
    fa = np.array([p['false_alarm'] for p in pts], float)
    det = np.array([p['detected'] for p in pts], float)
    order = np.argsort(fa)
    fa, det = fa[order], det[order]
    return {t: (1.0 - float(np.interp(t, fa, det))) * 100.0 for t in targets}


# -------------------------------------------------------------------------- scores
def fa_threshold(mass, is_clean, fa):
    """The score at which the clean tiles false-alarm at exactly `fa` (a FRACTION)."""
    clean = np.asarray(mass)[np.asarray(is_clean, bool)]
    if clean.size == 0:
        return float('nan')
    return float(np.quantile(clean, 1.0 - fa))


def detection_at_fa(mass, is_clean, is_defect, fa):
    """Detection rate (%) of `is_defect` tiles at the FA-matched threshold."""
    mass = np.asarray(mass, float)
    is_defect = np.asarray(is_defect, bool)
    thr = fa_threshold(mass, is_clean, fa)
    if not np.isfinite(thr) or is_defect.sum() == 0:
        return float('nan')
    return 100.0 * float((mass[is_defect] >= thr).mean())


def miss_at_fa_from_scores(mass, is_clean, is_defect, fa):
    """miss rate (%) = 100 - detection. `fa` is a FRACTION (0.05), never a percentage.

    Taking `fa` as a fraction everywhere is deliberate: eval_domain_split.py passed 5 and
    10 and divided by 100 internally, while everything else passed 0.05. Two conventions
    for one argument is a silent factor-of-20 waiting to happen, so there is now one.
    """
    d = detection_at_fa(mass, is_clean, is_defect, fa)
    return float('nan') if not np.isfinite(d) else 100.0 - d


def miss_at_fa_sweep(mass, is_clean, is_defect, fa):
    """miss rate (%) by CONSTRAINED MAXIMUM: best detection with false_alarm <= fa.

    Preserved verbatim from eval_ema_tta.py. This is not the same estimator as
    miss_at_fa_from_scores -- it searches thresholds and takes the best feasible one,
    which under ties is not the (1-fa) quantile. Kept so that converting that script is a
    no-op; prefer the quantile form for anything new.
    """
    mass = np.asarray(mass, float)
    is_clean = np.asarray(is_clean, bool)
    is_defect = np.asarray(is_defect, bool)
    best = 0.0
    for t in np.unique(mass):
        if (mass[is_clean] >= t).mean() <= fa:
            det = (mass[is_defect] >= t).mean()
            if det > best:
                best = det
    return 100.0 * (1.0 - best)
