#!/usr/bin/env python3
"""Did `penalize_false_clean=4` collapse the seed variance? Read the sd, not the mean.

The mean is the secondary question here. The dev box already measured pfc=4 as an
inverted-U optimum on the KPI (10.50 -> 10.30, i.e. neutral) with +4 points of coverage.
What this reports is the WITHIN-ARM sd, because that is what determines whether any n=1
comparison in this campaign can be believed -- and the campaign has just retracted a
headline result to seed noise.

HOW TO READ THE RATIO. Two sds from n=3 have 2 and 2 degrees of freedom. An F-test needs
F ≈ 19 for p<0.05 at that size, so even a 10x reduction here is SUGGESTIVE, not proof. Say
so in the write-up. What makes it actionable is corroboration: the dev box has two
independent tight sds at pfc=4 (their pfc sweep and their augmentation LOO, both ~0.10),
and this campaign has two independent loose ones at pfc=0.5 (the anchor arm and the
stride-2 arm, 1.29 and 1.63). A third independent tight sd here would make four
observations pointing the same way, which is worth acting on even without a significant F.

A ratio near 1 kills the idea outright, which is just as valuable: it means the variance is
intrinsic to the dataset, every future comparison has to be paid for at n=3, and the
backbone sweep needs its full stage 2.

Usage:  python pfc_variance_report.py
"""

import json
import os
import re
from statistics import mean, stdev

from mvc.core.artifact_paths import find_artifact
from mvc.core.metrics import miss_at_fa

MODEL = 'convnext_pico'
SEEDS = (42, 1337, 7)
MAX_EPOCH = 1
ARMS = {
    'pfc=0.5 (anchor)': {42: 'anc', 1337: 'anc1337', 7: 'anc7'},
    'pfc=4.0':          {s: f'pfc4s{s}' for s in SEEDS},
}


def best_epoch(run):
    d = f'datasets/mix_ckpts/{run}_{MODEL}'
    if not os.path.isdir(d):
        return None
    best = None
    for b in os.listdir(d):
        m_ep = re.search(r'epoch=(\d+)', b)
        m_mon = re.search(r'val_detect_auroc=([0-9]+\.[0-9]+)', b)
        if not (m_ep and m_mon) or int(m_ep.group(1)) > MAX_EPOCH:
            continue
        cand = (float(m_mon.group(1)), int(m_ep.group(1)))
        if best is None or cand[0] > best[0]:
            best = cand
    return best[1] if best else None


def factory(run, ep):
    p = find_artifact(f'{run}_ep{ep}_{MODEL}_threshold_curve.json')
    return miss_at_fa(p)[0.05] if p else None


def coverage(run, ep):
    p = find_artifact(f'{run}_{MODEL}_coverage.json') or \
        find_artifact(f'epochcov_{run}_ep{ep}.json')
    if not p:
        return None
    rows = json.load(open(p))['rows']
    v = [r['detect_at_fa5'] for r in rows
         if r['tier'] == 'TIER_A' and r['class'] != 'class_clean'
         and r.get('detect_at_fa5') is not None]
    return mean(v) if v else None


def collect(getter):
    out = {}
    for arm, runs in ARMS.items():
        vals = {}
        for seed, run in runs.items():
            ep = best_epoch(run)
            if ep is None:
                continue
            v = getter(run, ep)
            if v is not None:
                vals[seed] = v
        out[arm] = vals
    return out


def table(title, note, data, lower_is_better):
    print(f'\n\n{"=" * 92}\n{title}\n{note}\n{"=" * 92}')
    hdr = (f'{"arm":22s} ' + ' '.join(f'{("seed " + str(s)):>10s}' for s in SEEDS) +
           f' {"mean":>9s} {"sd":>7s}')
    print(hdr + '\n' + '-' * len(hdr))
    sds = {}
    for arm, v in data.items():
        if not v:
            print(f'{arm:22s} {"-- not run yet --":>40s}')
            continue
        cells = ' '.join(f'{v[s]:10.2f}' if s in v else f'{"--":>10s}' for s in SEEDS)
        vals = list(v.values())
        sd = stdev(vals) if len(vals) > 1 else float('nan')
        sds[arm] = sd
        tag = '' if len(vals) == len(SEEDS) else f'  (n={len(vals)})'
        print(f'{arm:22s} {cells} {mean(vals):9.2f} {sd:7.2f}{tag}')

    a, b = sds.get('pfc=0.5 (anchor)'), sds.get('pfc=4.0')
    if a and b and a == a and b == b and b > 0:
        ratio = a / b
        print(f'\nvariance ratio  sd(pfc=0.5) / sd(pfc=4.0) = {a:.2f} / {b:.2f} = '
              f'**{ratio:.1f}x**')
        if ratio >= 19:
            v = 'F-significant at n=3 (p<0.05) — act on it'
        elif ratio >= 4:
            v = ('SUGGESTIVE — not F-significant at n=3, but combines with the dev box\'s '
                 'two\n              independent tight sds at pfc=4 into four observations '
                 'pointing one way')
        elif ratio >= 1.5:
            v = 'weak — could easily be luck at 2 df'
        else:
            v = ('NO VARIANCE COLLAPSE — the noise is intrinsic; every comparison must be '
                 'paid\n              for at n=3 and the backbone sweep needs its full '
                 'stage 2')
        print(f'reading: {v}')


def main():
    print('\nconvnext_pico · Aug26_78K · 10-class · 4ch+DoLP · coverage carved out · '
          '2 epochs · 3 seeds')
    print('The QUESTION IS THE sd, not the mean. Everything else about these arms is '
          'identical.')

    table('FACTORY VALIDATION — miss@FA5. Lower mean is better; '
          'LOWER sd IS THE POINT.',
          'pfc=0.5 reference: the anchor arm. Its sd is what retracted the stride-2 result.',
          collect(factory), True)

    table('COVERAGE VALIDATION — TIER_A macro detection@FA5.',
          'The dev box reports pfc=4 buying ~+4 points here (on exact-class, on the broken\n'
          'label space) — this measures it on detection, on the corrected 10-class scheme.',
          collect(coverage), False)

    print('\n\nIF THE COLLAPSE IS REAL: the 11-backbone sweep should run at pfc=4 and may '
          'not\nneed its ~60 h stage 2 at all. IF IT IS NOT: the sweep needs stage 2, and '
          'every\nfuture n=1 comparison in this campaign is uninterpretable by construction.')


if __name__ == '__main__':
    main()
