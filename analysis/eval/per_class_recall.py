#!/usr/bin/env python3
"""Per-class recall from a run's confusion matrix — the number miss@FA5 hides.

Why this exists. The headline KPI is a DETECTION rate: defect-vs-clean, aggregated over
whatever defects the validation set happens to contain. On the Aug26_78K split that is
84% WeldingClassA, because the integrator's car is welding-heavy. That makes miss@FA5 a
faithful estimate of performance AT THAT SITE and a poor guide to whether the model can
see a positive dent -- a model could improve its headline number while quietly losing
dents, and nothing in the current reporting would show it.

The confusion matrix already contains the answer; it was simply never surfaced. This
prints recall and support per class so a welding gain cannot be mistaken for progress.

Usage:
    python per_class_recall.py p2_convnext_pico_confusion.json [more_confusion.json ...]
    python per_class_recall.py --compare a_confusion.json b_confusion.json
"""

import json
import sys


def per_class(path):
    d = json.load(open(path))
    labels, m = d['labels'], d['matrix']
    out = []
    for i, name in enumerate(labels):
        support = sum(m[i])
        correct = m[i][i]
        out.append((name, support, (correct / support * 100.0) if support else None))
    return out


def show(path):
    rows = per_class(path)
    total = sum(r[1] for r in rows)
    print(f'\n{path}')
    print(f"{'class':34s} {'support':>10s} {'share':>7s} {'recall %':>9s}")
    print('-' * 63)
    for name, support, rec in sorted(rows, key=lambda r: -r[1]):
        share = support / total * 100.0 if total else 0.0
        r = f'{rec:9.2f}' if rec is not None else f'{"n/a":>9s}'
        flag = '   <- NOT MEASURABLE (no val tiles)' if not support else ''
        print(f'{name:34s} {support:10,d} {share:6.1f}% {r}{flag}')
    seen = [r for r in rows if r[1]]
    if seen:
        macro = sum(r[2] for r in seen) / len(seen)
        print(f'\n  macro recall over the {len(seen)} classes that HAVE val tiles: {macro:.2f}%')
        print(f'  ({len(rows) - len(seen)} class(es) have none and cannot be measured here)')
    return dict((r[0], r[2]) for r in rows)


def main():
    args = [a for a in sys.argv[1:] if a != '--compare']
    if not args:
        sys.exit(__doc__)
    if '--compare' in sys.argv and len(args) == 2:
        a, b = show(args[0]), show(args[1])
        print(f"\n{'class':34s} {'A recall':>9s} {'B recall':>9s} {'Δ (B-A)':>9s}")
        print('-' * 63)
        for k in a:
            if a[k] is None or b.get(k) is None:
                continue
            print(f'{k:34s} {a[k]:9.2f} {b[k]:9.2f} {b[k]-a[k]:+9.2f}')
        print('\nA per-class REGRESSION alongside an improved headline is exactly what the '
              'aggregate\nKPI cannot show you on a val set dominated by one class.')
    else:
        for p in args:
            show(p)


if __name__ == '__main__':
    main()
