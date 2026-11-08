#!/usr/bin/env python3
"""The whole field on one page: every model trained on Aug26, both validations, plus Hz.

Merges the three sources that share a protocol -- the incumbent (`anc`), the 11-backbone
sweep (`ms*`), and the full zoo sweep (`fz*`). All ran seed 42, 2 epochs, 4ch+DoLP,
10-class, coverage carved out, pfc=0.5, scored at the monitored-best checkpoint, so the
numbers are directly comparable and belong in one table rather than three.

SORTED BY COVERAGE, NOT BY THE FACTORY KPI. Two reasons, both measured on this dataset:

  * coverage is the more precise instrument -- seed sd 0.43 against the factory KPI's 1.01;
  * the factory val is 84% welding, so it rewards models that learn welding and abandon
    the weak dents. Every one of the six models that beat convnext_pico on the factory KPI
    in the first sweep did exactly that, at 7-23 sigma of coverage cost.

The `weak` column is the mean of PositiveDent A/B/C detection -- the classes that bind the
KPI and where the whole spread lives. `strong` is everything else. A model can only be
interesting if `weak` holds up; a good factory number with a collapsed `weak` column is the
failure mode this harness exists to catch.

n=1 per model, by request. Seed sd is 1.01 factory / 0.43 coverage, so read gaps of a few
points as noise and gaps of ten as real. That asymmetry is the point of a breadth scan:
it cannot rank neighbours, but it finds outliers, and the outliers here have been large.

Usage:  python full_zoo_report.py [--all]      # --all includes models that cannot ship
"""

import json
import sys
from statistics import mean

from mvc.core.artifact_paths import find_artifact
from mvc.core.metrics import miss_at_fa
from . import report_common as rc

TARGET_HZ, GPU_SCALE = 23.0, 1.6
COV_SD, FAC_SD = 0.43, 1.01          # anchor-arm seed sd, n=3, 2026-08-12


def load_bench():
    # Thin wrapper: this report only ever wants the deploy-variant hz_step16 scalar per
    # model, so unpack report_common's full-row form into that shape at the one call site.
    return {k: v.get('hz_step16', 0.0) for k, v in rc.load_bench().items()}


def runs():
    """(label, run-name, model-key, ckpt-dir-suffix) for everything comparable."""
    out = [('convnext_pico  [INCUMBENT]', 'anc', 'convnext_pico', 'convnext_pico')]
    try:
        from .model_sweep import CANDIDATES
        out += [(m, f'ms{t}', m, m) for t, m, *_ in CANDIDATES]
    except Exception:
        pass
    try:
        from .full_zoo_sweep import TAGS
        out += [(m, f'fz{t}', m, m.replace('/', '_')) for m, t in TAGS.items()]
    except Exception:
        pass
    return out


def score(run, sfx):
    ep = rc.best_epoch(f'datasets/mix_ckpts/{run}_{sfx}')
    if ep is None:
        return None
    curve = find_artifact(f'{run}_ep{ep}_{sfx}_threshold_curve.json')
    cov = find_artifact(f'{run}_{sfx}_coverage.json') or \
        find_artifact(f'epochcov_{run}_ep{ep}.json')
    if not (curve and cov):
        return None
    rows = [r for r in json.load(open(cov))['rows'] if r['class'] != 'class_clean']
    weak = [r['detect_at_fa5'] for r in rows if r['class'].startswith('class_PositiveDent')]
    strong = [r['detect_at_fa5'] for r in rows
              if not r['class'].startswith('class_PositiveDent')]
    return {'ep': ep, 'miss5': miss_at_fa(curve)[0.05],
            'tier_a': rc.tier_a_macro_from_rows(rows),
            'weak': mean(weak) if weak else None,
            'strong': mean(strong) if strong else None}


def main():
    show_all = '--all' in sys.argv
    hz = load_bench()
    got = []
    for label, run, key, sfx in runs():
        s = score(run, sfx)
        if s:
            s.update(label=label, hz=hz.get(key, 0.0))
            got.append(s)
    if not got:
        print('nothing scored yet')
        return

    inc = next((g for g in got if 'INCUMBENT' in g['label']), None)
    got.sort(key=lambda g: -(g['tier_a'] or 0))

    print(f'\nAug26_78K · 4ch+DoLP · 10-class · coverage carved · seed 42 · 2 epochs · '
          f'monitored-best')
    print(f'{len(got)} models scored. SORTED BY COVERAGE (sd 0.43) not factory (sd 1.01).')
    if inc:
        print(f'Incumbent: factory {inc["miss5"]:.2f} · TIER_A {inc["tier_a"]:.2f} · '
              f'weak {inc["weak"]:.2f} · {inc["hz"] * GPU_SCALE:.0f} Hz on 5090')
    print()
    hdr = (f'{"model":38s} {"fac":>6s} {"TIER_A":>7s} {"Δcov":>6s} {"weak":>6s} '
           f'{"Δweak":>6s} {"5090Hz":>7s} {"ship":>5s}')
    print(hdr + '\n' + '-' * len(hdr))
    hidden = 0
    for g in got:
        ships = g['hz'] * GPU_SCALE >= TARGET_HZ
        if not ships and not show_all and 'INCUMBENT' not in g['label']:
            hidden += 1
            continue
        dc = g['tier_a'] - inc['tier_a'] if inc else 0.0
        dw = g['weak'] - inc['weak'] if inc else 0.0
        print(f'{g["label"]:38s} {g["miss5"]:6.2f} {g["tier_a"]:7.2f} {dc:+6.2f} '
              f'{g["weak"]:6.2f} {dw:+6.2f} {g["hz"] * GPU_SCALE:7.1f} '
              f'{"yes" if ships else "NO":>5s}')
    if hidden:
        print(f'\n({hidden} scored models hidden: cannot reach {TARGET_HZ:.0f} Hz on the '
              f'5090 at step 16. --all to show.)')

    if inc:
        better = [g for g in got
                  if g['tier_a'] > inc['tier_a'] + 2 * COV_SD
                  and g['hz'] * GPU_SCALE >= TARGET_HZ]
        print('\n' + ('CANDIDATES BEATING THE INCUMBENT ON COVERAGE BEYOND NOISE:'
                      if better else
                      'NO MODEL BEATS THE INCUMBENT ON COVERAGE BEYOND NOISE (bar: '
                      f'+{2 * COV_SD:.2f}).'))
        for g in better:
            print(f'   {g["label"]}  TIER_A {g["tier_a"]:.2f} (+{g["tier_a"] - inc["tier_a"]:.2f})')


if __name__ == '__main__':
    main()
