#!/usr/bin/env python3
"""Analyse the convnext_pico modifier sweep with the noise floor taken seriously.

Every arm is run at 3 seeds because a single 30k screen has sd 0.43 miss@FA5 and cannot
resolve anything below ~0.85 (see the noise-floor section of PLAN.md). Reporting one run
per modifier is what produced the Phase 1a and Phase 3 claims that had to be withdrawn.

Two comparisons are printed:

  * unpaired  — arm mean vs baseline mean, with the pooled sd. This is the honest headline.
  * paired    — mean of the per-seed differences. Baseline and arm share seed values, so
                sampler order and augmentation draws line up and part of the run-to-run
                variance cancels. When the paired sd is much smaller than the unpaired one,
                the paired test resolves effects the unpaired test cannot.

A modifier is called ESTABLISHED only when |effect| exceeds 2 sd of the relevant
comparison. Anything else is reported as unresolved, not as a small effect.

Usage:  python modifier_sweep.py
"""

import glob
import json
import os
import sys
from statistics import mean, stdev

from Metrics import miss_at_fa

SEEDS = (42, 1337, 7)
ARMS = ['base', 'nodolp', 'aolp', 'mmr', 'unpol', 'mono', 'stride2']
LABEL = {
    'base':    'DoLP only (baseline)',
    'nodolp':  'no derived channel',
    'aolp':    '+ AoLP',
    'mmr':     '+ Max/Min/Range',
    'unpol':   '+ Unpolarized',
    'mono':    'MONOCHROME (no polarimetry)',
    'stride2': 'patchify stride 4 -> 2',
}
REUSE = {('base', 42): 'tz', ('nodolp', 42): 'p1n'}


def curve_for(arm, seed):
    name = REUSE.get((arm, seed), f'mx{arm}{seed}')
    p = f'{name}_convnext_pico_threshold_curve.json'
    return p if os.path.exists(p) else None


def main():
    data = {}
    for arm in ARMS:
        vals = {}
        for s in SEEDS:
            c = curve_for(arm, s)
            if c:
                vals[s] = miss_at_fa(c)[0.05]
        if vals:
            data[arm] = vals

    if 'base' not in data:
        sys.exit('no baseline runs finished yet')

    print('\nconvnext_pico, 30k screens, frozen val split, 3 seeds per arm\n')
    hdr = f"{'arm':30s} " + ' '.join(f'{("seed "+str(s)):>9s}' for s in SEEDS) + f" {'mean':>8s} {'sd':>6s}"
    print(hdr); print('-' * len(hdr))
    for arm in ARMS:
        if arm not in data:
            continue
        v = data[arm]
        cells = ' '.join(f'{v[s]:9.2f}' if s in v else f'{"--":>9s}' for s in SEEDS)
        got = list(v.values())
        sd = stdev(got) if len(got) > 1 else float('nan')
        n = '' if len(got) == len(SEEDS) else f'  (n={len(got)})'
        print(f'{LABEL[arm]:30s} {cells} {mean(got):8.2f} {sd:6.2f}{n}')

    base = data['base']
    base_vals = list(base.values())
    print(f'\n{"modifier":30s} {"unpaired Δ":>11s} {"paired Δ":>10s} {"paired sd":>10s} {"verdict":>28s}')
    print('-' * 94)
    for arm in ARMS:
        if arm == 'base' or arm not in data:
            continue
        v = data[arm]
        unpaired = mean(list(v.values())) - mean(base_vals)
        shared = [s for s in SEEDS if s in v and s in base]
        if len(shared) > 1:
            diffs = [v[s] - base[s] for s in shared]
            pd_mean, pd_sd = mean(diffs), stdev(diffs)
            if abs(pd_mean) > 2 * pd_sd:
                verdict = 'ESTABLISHED (>2 sd paired)'
            elif abs(pd_mean) > pd_sd:
                verdict = 'suggestive (1-2 sd)'
            else:
                verdict = 'unresolved'
            print(f'{LABEL[arm]:30s} {unpaired:+11.2f} {pd_mean:+10.2f} {pd_sd:10.2f} {verdict:>28s}')
        else:
            print(f'{LABEL[arm]:30s} {unpaired:+11.2f} {"--":>10s} {"--":>10s} {"incomplete":>28s}')

    print('\nNegative Δ = better (lower miss@FA5). "unresolved" means this sweep could not '
          'tell the\nmodifier apart from doing nothing — NOT that the effect is small.')


if __name__ == '__main__':
    main()
