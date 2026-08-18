#!/usr/bin/env python3
"""Analyse the Aug26 modifier sweep on BOTH validations.

The legacy reporter (`modifier_sweep.py`) only knew the factory KPI. Under the two-validation
design that is half an answer: the ship rule is "factory KPI improves AND no TIER_A coverage
class regresses beyond noise", and TIER_A must be read on the DETECTION column, not
exact-class recall (exact-class is dominated by a near-constant NegativeDentClassA<->B
confusion that can hide a real detection change -- see PLAN.md).

Both comparisons are printed per validation:

  unpaired  arm mean vs baseline mean. The honest headline.
  paired    mean of per-seed differences. Baseline and arm share seed values, so sampler
            order and augmentation draws line up and part of the variance cancels.

ESTABLISHED requires |effect| > 2 sd of the paired comparison. Everything else is reported
as unresolved -- NOT as a small effect. Reporting a verdict from n=2 is what produced the
AoLP claim that had to be withdrawn.

Sign convention: positive Δ = the modifier made things WORSE (higher miss, lower detection
cost is reported as its own sign). This matches the dev box's convention. The legacy
`mx*` sweep used the opposite one because its baseline had DoLP on.

Usage:  python aug26_sweep_report.py
"""

import json
import os
from statistics import mean, stdev

from artifact_paths import find_artifact
from phase2_select import miss_at_fa

SEEDS = (42, 1337, 7)
ARMS = ['base', 'dolp', 'mono', 'stride2']
LABEL = {
    'base':    '4ch raw (baseline)',
    'dolp':    '+ DoLP',
    'mono':    'MONOCHROME (no polarimetry)',
    'stride2': 'patchify stride 4 -> 2',
}


def factory(arm, seed):
    # find_artifact: these screens have been filed into experiments/aug26_screens/ by
    # tidy_experiments.py, so bare-name lookup in the cwd no longer finds them.
    p = find_artifact(f's26{arm}{seed}_convnext_pico_threshold_curve.json')
    return miss_at_fa(p)[0.05] if p else None


def coverage(arm, seed):
    """TIER_A macro DETECTION at FA5 -- the ship-rule column."""
    p = find_artifact(f's26{arm}{seed}_convnext_pico_coverage.json')
    if not p:
        return None
    rows = [r for r in json.load(open(p))['rows']
            if r['tier'] == 'TIER_A' and r['class'] != 'class_clean']
    vals = [r['detect_at_fa5'] for r in rows if r.get('detect_at_fa5') is not None]
    return mean(vals) if vals else None


def table(title, getter, lower_is_better, data):
    print(f'\n\n{"=" * 96}\n{title}\n{"=" * 96}')
    hdr = f"{'arm':30s} " + ' '.join(f'{("seed " + str(s)):>9s}' for s in SEEDS) + \
          f" {'mean':>8s} {'sd':>6s}"
    print(hdr)
    print('-' * len(hdr))
    for arm in ARMS:
        v = data.get(arm) or {}
        if not v:
            continue
        cells = ' '.join(f'{v[s]:9.2f}' if s in v else f'{"--":>9s}' for s in SEEDS)
        got = list(v.values())
        sd = stdev(got) if len(got) > 1 else float('nan')
        n = '' if len(got) == len(SEEDS) else f'  (n={len(got)})'
        print(f'{LABEL[arm]:30s} {cells} {mean(got):8.2f} {sd:6.2f}{n}')

    base = data.get('base')
    if not base:
        print('\n(no baseline runs finished yet)')
        return

    print(f'\n{"modifier":30s} {"unpaired Δ":>11s} {"paired Δ":>10s} {"paired sd":>10s} '
          f'{"seeds":>6s} {"verdict":>26s}')
    print('-' * 98)
    for arm in ARMS:
        if arm == 'base' or arm not in data:
            continue
        v = data[arm]
        shared = [s for s in SEEDS if s in v and s in base]
        if not shared:
            continue
        # A modifier is "worse" when it raises miss or lowers detection; normalise so a
        # positive Δ always means WORSE, whichever metric this table is.
        sign = 1.0 if lower_is_better else -1.0
        diffs = [sign * (v[s] - base[s]) for s in shared]
        unp = sign * (mean(list(v.values())) - mean(list(base.values())))
        pd = mean(diffs)
        psd = stdev(diffs) if len(diffs) > 1 else float('nan')

        if len(diffs) < 3:
            verdict = f'n={len(diffs)} — NO VERDICT'
        elif psd == psd and abs(pd) > 2 * psd:
            verdict = 'ESTABLISHED' + (' (worse)' if pd > 0 else ' (BETTER)')
        elif psd == psd and abs(pd) > psd:
            verdict = 'suggestive (1–2 sd)'
        else:
            verdict = 'unresolved'
        print(f'{LABEL[arm]:30s} {unp:11.2f} {pd:10.2f} {psd:10.2f} {len(diffs):6d} '
              f'{verdict:>26s}')


def main():
    fac = {a: {s: v for s in SEEDS if (v := factory(a, s)) is not None} for a in ARMS}
    cov = {a: {s: v for s in SEEDS if (v := coverage(a, s)) is not None} for a in ARMS}

    print('\nconvnext_pico, Aug26_78K, 10-class scheme, coverage carved out, 30k screens')
    print('Anchor reference (full train, epoch 1): factory 9.24 miss@FA5 · '
          'coverage TIER_A det@FA5 73.59%')

    table('FACTORY VALIDATION — miss@FA5 (lower is better).  SHIP/NO-SHIP METRIC.',
          factory, True, fac)
    table('COVERAGE VALIDATION — TIER_A macro detection@FA5 % (higher is better).  '
          'REGRESSION TRIPWIRE.\nPositive Δ = detection got WORSE.',
          coverage, False, cov)

    print('\n\nSHIP RULE: a change ships only if the factory KPI improves AND no TIER_A')
    print('coverage detection regresses beyond noise. A factory gain with a TIER_A')
    print('detection drop is a regression — at 84% welding in the factory val, optimising')
    print('the headline alone would optimise welding and call it progress.')


if __name__ == '__main__':
    main()
