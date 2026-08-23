#!/usr/bin/env python3
"""The full-budget noise floor, and the stride-2 verdict at n=3 paired.

This reporter exists because the campaign has been comparing full-budget runs without ever
measuring how much two full-budget runs differ FOR NO REASON. The 30k screen's noise floor
(sd 1.83 miss@FA5 on Aug26) does not transfer: 30k steps is a quarter of one epoch, and a
screen's variance is dominated by where in the schedule it was cut off. So every full-train
comparison in PLAN.md -- including the headline 9.24 -> 7.15 -- has until now been a single
paired difference against an unknown scale.

Three things are printed, in order of what they gate:

  1. FACTORY miss@FA5 per arm per seed, and the WITHIN-ARM sd. That sd is the answer to
     "how big does a full-budget difference have to be before it means anything".
  2. The PAIRED stride-2 effect at n=3. Paired because both arms run the same seed values,
     so the seed-to-seed component cancels; the campaign standard is |effect| > 2 sd of the
     PAIRED differences, and n=3 is the minimum at which any verdict may be stated at all
     (the AoLP reversal: +0.83 sd 0.03 "ESTABLISHED" at n=2, unresolved at n=3).
  3. PER-CLASS coverage detection, with a within-arm sd per class. This is the one that
     gates shipping. Stride-2's only per-class regression was PositiveDentClassB at -1.37
     detection; the ship rule is per-class-on-detection, so -1.37 either trips it or is
     noise, and nothing in the campaign could tell which until this table existed.

EPOCH SELECTION. The replicates were budgeted at 2 epochs, so every run here is scored at
whichever of epoch 0/1 the run's own checkpoint_monitor (val_detect_auroc) preferred --
exactly the rule eval_coverage.py applies. The seed-42 parents are restricted to the same
two epochs even though they trained longer, or the comparison would be
best-of-6 against best-of-2. Both happen to peak at epoch 1 anyway.

Usage:  python seed_replicates_report.py
        Prints one anchor-vs-challenger section per entry in CHALLENGERS below.

GENERALISED 2026-08-23 for Experiment B (the seed sweep over the top-5 challengers, see
EXPERIMENT-seed-sweep-and-step-curve.md Sec.5), which replaced the original anc/ancs2
stride-2 comparison this script was written for. ARMS now maps a run PREFIX to its own
MODEL (challengers differ in architecture, not just stem stride), and the report runs once
per (anchor, challenger) pair -- printing the same anchor/challenger/paired-Δ/per-class
ship table this script always has, just parametrised instead of hardcoded to
anc/ancs2/convnext_pico. Default pairs: the anchor against each of the four challengers
Experiment A's step curve did not rule out.
"""

import argparse
import glob
import json
import os
import re
from statistics import mean, stdev

from mvc.core.artifact_paths import find_artifact
from mvc.core.metrics import miss_at_fa

SEEDS = (42, 1337, 7)
MAX_EPOCH = 1                      # the replicate budget; see the note above

# prefix -> (model, label). 'anc' keeps its legacy coverage spelling for seed 42 (scored
# by epoch_cov.sh before the replicate mechanism existed); every other arm, including all
# four challengers, writes the standard `{name}_{model}_coverage.json` at every seed.
ANCHOR = ('anc', 'convnext_pico', 'convnext_pico (incumbent)')
CHALLENGERS = [
    ('fzcnxtiny',  'convnext_tiny',     'convnext_tiny (+3.51 single-seed)'),
    ('fzeffb0',    'efficientnet_b0',   'efficientnet_b0 (+1.22 single-seed)'),
    ('fzregy800',  'regnet_y_800mf',    'regnet_y_800mf (+0.80 single-seed)'),
    ('msfemto',    'convnext_femto',    'convnext_femto (-1.21 single-seed)'),
]

LEGACY_COV = {('anc', 42): 'epochcov_anc_ep{ep}.json'}


def run_name(arm, seed):
    return arm if seed == 42 else f'{arm}{seed}'


def best_epoch(arm, model, seed):
    """The epoch the run's own checkpoint monitor prefers, within the replicate budget."""
    d = f'datasets/mix_ckpts/{run_name(arm, seed)}_{model}'
    best = None
    for ck in glob.glob(os.path.join(d, '*.ckpt')):
        b = os.path.basename(ck)
        m_ep = re.search(r'epoch=(\d+)', b)
        m_mon = re.search(r'val_detect_auroc=([0-9]+\.[0-9]+)', b)
        if not (m_ep and m_mon) or int(m_ep.group(1)) > MAX_EPOCH:
            continue
        cand = (float(m_mon.group(1)), int(m_ep.group(1)))
        if best is None or cand[0] > best[0]:
            best = cand
    return best[1] if best else None


def factory(arm, model, seed, ep):
    # find_artifact so the report keeps working after tidy_experiments.py files these
    # away into experiments/<campaign>/<run>/. Root wins, so a fresh write always beats
    # an archived copy of the same name.
    p = find_artifact(f'{run_name(arm, seed)}_ep{ep}_{model}_threshold_curve.json')
    return miss_at_fa(p)[0.05] if p else None


def cov_rows(arm, model, seed, ep):
    p = find_artifact(LEGACY_COV.get((arm, seed), '').format(ep=ep) or
                      f'{run_name(arm, seed)}_{model}_coverage.json')
    if not p:
        return None
    d = json.load(open(p))
    return {r['class']: r for r in d['rows'] if r['class'] != 'class_clean'}


def tier_a_macro(rows):
    v = [r['detect_at_fa5'] for r in rows.values()
         if r['tier'] == 'TIER_A' and r.get('detect_at_fa5') is not None]
    return mean(v) if v else None


def spread(vals):
    return (mean(vals), stdev(vals) if len(vals) > 1 else float('nan'))


def arm_table(title, note, data, models, labels, lower_is_better):
    print(f'\n\n{"=" * 100}\n{title}\n{note}\n{"=" * 100}')
    hdr = f'{"arm":34s} ' + ' '.join(f'{("seed " + str(s)):>10s}' for s in SEEDS) + \
          f' {"mean":>9s} {"sd":>7s}'
    print(hdr + '\n' + '-' * len(hdr))
    for arm, label in labels.items():
        v = data.get(arm, {})
        if not v:
            continue
        cells = ' '.join(f'{v[s]:10.2f}' if s in v else f'{"--":>10s}' for s in SEEDS)
        m, sd = spread(list(v.values()))
        tag = '' if len(v) == len(SEEDS) else f'  (n={len(v)})'
        print(f'{label:34s} {cells} {m:9.2f} {sd:7.2f}{tag}')

    a, b = data.get('anc', {}), data.get('chal', {})
    shared = [s for s in SEEDS if s in a and s in b]
    if not shared:
        return
    sign = 1.0 if lower_is_better else -1.0     # positive Δ always = challenger is WORSE
    diffs = [sign * (b[s] - a[s]) for s in shared]
    pd, psd = spread(diffs)
    print(f'\nper-seed paired Δ (challenger − anchor, +ve = worse): '
          f'{"  ".join(f"{s}: {d:+.2f}" for s, d in zip(shared, diffs))}')
    if len(diffs) < 3:
        verdict = f'n={len(diffs)} — NO VERDICT (campaign rule: never from n<3)'
    elif abs(pd) > 2 * psd:
        verdict = 'ESTABLISHED ' + ('(challenger WORSE)' if pd > 0 else '(challenger BETTER)')
    elif abs(pd) > psd:
        verdict = 'suggestive (1–2 sd) — not established'
    else:
        verdict = 'UNRESOLVED — indistinguishable from seed noise'
    print(f'paired mean Δ {pd:+.2f}   paired sd {psd:.2f}   n={len(diffs)}   ->  {verdict}')


def report_pair(anchor, challenger):
    a_prefix, a_model, a_label = anchor
    c_prefix, c_model, c_label = challenger
    ARMS = {'anc': (a_prefix, a_model), 'chal': (c_prefix, c_model)}
    labels = {'anc': a_label, 'chal': c_label}

    print(f'\n\n{"#" * 100}\n# {a_label}  vs  {c_label}\n{"#" * 100}')

    eps = {(k, s): best_epoch(prefix, model, s)
           for k, (prefix, model) in ARMS.items() for s in SEEDS}
    print('FULL BUDGET (2 epochs), scored at the checkpoint-monitor-preferred epoch:')
    for (k, s), e in sorted(eps.items()):
        prefix, _ = ARMS[k]
        print(f'   {run_name(prefix, s):16s} epoch {e if e is not None else "-- (no run yet)"}')

    fac = {k: {s: v for s in SEEDS
               if eps[(k, s)] is not None and
               (v := factory(prefix, model, s, eps[(k, s)])) is not None}
           for k, (prefix, model) in ARMS.items()}
    rows = {(k, s): cov_rows(prefix, model, s, eps[(k, s)])
            for k, (prefix, model) in ARMS.items() for s in SEEDS if eps[(k, s)] is not None}
    rows = {k: v for k, v in rows.items() if v}
    cov = {k: {s: m for s in SEEDS
               if (k, s) in rows and (m := tier_a_macro(rows[(k, s)])) is not None}
           for k in ARMS}

    arm_table('FACTORY VALIDATION — miss@FA5 (lower is better).  SHIP/NO-SHIP METRIC.',
              'The within-arm sd IS the full-budget noise floor.',
              fac, ARMS, labels, lower_is_better=True)

    arm_table('COVERAGE VALIDATION — TIER_A macro detection@FA5 % (higher is better).',
              'Recording-disjoint tripwire. The per-class table below is the one that '
              'gates shipping.',
              cov, ARMS, labels, lower_is_better=False)

    # ---- per class: the ship-rule table -------------------------------------------
    classes = sorted({c for r in rows.values() for c in r})
    print(f'\n\n{"=" * 100}\nPER-CLASS COVERAGE DETECTION@FA5 — THE SHIP-RULE TABLE\n'
          'Within-arm sd per class = how big a per-class move has to be to mean anything.\n'
          'Paired Δ = challenger minus anchor on shared seeds. A regression only counts if '
          '|Δ| > 2 × paired sd.\n' + '=' * 100)
    hdr = (f'{"class":30s} {"tier":7s} ' +
           ' '.join(f'{a:>14s}' for a in ('anchor', 'challenger')) +
           f' {"paired Δ":>10s} {"psd":>6s} {"verdict":>14s}')
    print(hdr + '\n' + '-' * len(hdr))
    for c in classes:
        per = {k: {s: rows[(k, s)][c]['detect_at_fa5']
                   for s in SEEDS if (k, s) in rows and c in rows[(k, s)]}
               for k in ARMS}
        if not per['anc'] or not per['chal']:
            continue
        tier = next(r[c]['tier'] for r in rows.values() if c in r)
        cells = []
        for k in ARMS:
            m, sd = spread(list(per[k].values()))
            cells.append(f'{m:7.2f}±{0 if sd != sd else sd:5.2f}' if sd == sd
                         else f'{m:7.2f}   n=1')
        shared = [s for s in SEEDS if s in per['anc'] and s in per['chal']]
        d = [per['chal'][s] - per['anc'][s] for s in shared]
        pd, psd = spread(d)
        if len(d) < 3:
            v = f'n={len(d)}'
        elif abs(pd) > 2 * psd:
            v = 'REAL ' + ('gain' if pd > 0 else 'REGRESSION')
        else:
            v = 'within noise'
        print(f'{c:30s} {tier:7s} {cells[0]:>14s} {cells[1]:>14s} '
              f'{pd:+10.2f} {psd:6.2f} {v:>14s}')

    print('\nSHIP RULE: a challenger ships only if the factory KPI improves beyond the')
    print('paired noise AND no TIER_A class regresses beyond 2× its own paired sd.')


def main():
    for challenger in CHALLENGERS:
        report_pair(ANCHOR, challenger)


if __name__ == '__main__':
    main()
