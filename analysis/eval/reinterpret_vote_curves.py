#!/usr/bin/env python3
"""Re-interpret the EXISTING *_vote_curve.json artifacts (from the eval_vote_curve.py
backfill campaign, 2026-09) across the neighborhood-voting dimension (kernel = erosion
radius, min_votes = tiles required in that neighborhood), at the same FA budgets used by
eval_ensemble.py's threshold scan. No inference is re-run -- every number here is pulled
from a threshold x kernel x min_votes sweep that was already computed and cached.

CAVEAT, stated once here rather than repeated per row: the cached rows store RATES
(false_alarm, macro_detect_tier_a as fractions), not raw TP/TN/FP/FN counts, and the JSON
does not carry per-class support denominators either -- eval_vote_curve.py was built to
report a compact curve, not to be replayed into absolute counts later. So this reports
false_alarm% (a false-positive RATE, not a count) and macro_detect_tier_a% (a true-positive
RATE, not a count) side by side, which carries the same information for comparing operating
points, but cannot be converted into a TP/TN/FP/FN table without re-scoring -- which is
exactly the GPU work this script exists to avoid.

For each model and each FA budget, this picks the (kernel, min_votes) setting that gives the
best macro_detect_tier_a among rows whose false_alarm does not exceed the budget -- i.e. the
same "scan thresholds, don't fix at FA5" idea eval_ensemble.py applies to models, applied
here to the voting-neighborhood knobs instead.

Usage:
    python -m analysis.eval.reinterpret_vote_curves
"""

import json

from mvc.core.artifact_paths import find_artifact
from analysis.eval.eval_ensemble import COVERAGE_POOL, FA_TARGETS


def load_rows(run, model):
    sfx = model.replace('/', '_')
    p = find_artifact(f'{run}_{sfx}_vote_curve.json')
    if not p:
        return None
    return json.load(open(p))


def best_at_fa_budget(rows, fa_budget):
    """Among all (threshold, kernel, min_votes) rows at or under the FA budget, the one
    with the highest macro_detect_tier_a -- None if nothing in the sweep meets the budget."""
    under = [r for r in rows if r['false_alarm'] <= fa_budget]
    if not under:
        return None
    return max(under, key=lambda r: r['macro_detect_tier_a'])


def no_voting_baseline(rows, fa_budget):
    """kernel=1, min_votes<=1 is the ROS service's own "0/1 disables voting" contract --
    the reference point voting is measured against, not kernel=0 (which makes min_votes>=2
    structurally unsatisfiable, not a real 'off' state)."""
    base_rows = [r for r in rows if r['kernel'] == 1 and r['min_votes'] <= 1]
    return best_at_fa_budget(base_rows, fa_budget)


def main():
    print(f'{"model":28s} {"FA budget":>9s} {"no-vote det%":>13s} {"best (k,v)":>11s} '
          f'{"best det%":>10s} {"delta":>7s}')
    print('-' * 84)
    for run, model, note in COVERAGE_POOL:
        d = load_rows(run, model)
        if d is None:
            print(f'{run}_{model}: no vote_curve.json cached, skipping')
            continue
        rows = d['rows']
        for fa in FA_TARGETS:
            base = no_voting_baseline(rows, fa)
            best = best_at_fa_budget(rows, fa)
            if base is None or best is None:
                print(f'{model:28s} {fa*100:8.0f}%  (no row under this FA budget)')
                continue
            base_pct = base['macro_detect_tier_a'] * 100
            best_pct = best['macro_detect_tier_a'] * 100
            kv = f"({best['kernel']},{best['min_votes']})"
            print(f'{model:28s} {fa*100:8.0f}% {base_pct:13.2f} {kv:>11s} {best_pct:10.2f} '
                  f'{best_pct - base_pct:+7.2f}')
        print()

    print('\nno-vote baseline = kernel=1, min_votes<=1 (the ROS "0/1 disables voting" '
          'contract), at the best threshold under the same FA budget.')
    print('best (k,v) = the (kernel, min_votes) setting achieving the highest macro '
          'TIER_A detect@FA within the same budget, searched over all 70 thresholds cached.')
    print('delta = best - no-vote, in TIER_A macro detect percentage points. Positive means '
          'the voting neighborhood genuinely buys coverage at that FA budget, not just noise '
          'suppression that was free anyway.')


if __name__ == '__main__':
    main()
