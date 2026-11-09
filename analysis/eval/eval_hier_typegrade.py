#!/usr/bin/env python3
"""Type accuracy and grade (severity) accuracy, with confusion matrices, derived from
an existing `eval_coverage.py` output.

WHY THIS EXISTS
---------------
`eval_coverage.py` (2026-10-07, see its own comment at the json.dump site) already writes
'labels' (class names, true=rows/predicted=cols order) and 'confusion' (the full flat
argmax confusion matrix over the coverage tiles) into every `<run>_coverage.json` it
produces. That is everything a TYPE accuracy and a GRADE (severity) accuracy need -- no
new inference, no new GPU work. Nothing in the repo turns those two fields into the
type/severity breakdown a hierarchical (Detect-Type-Grade) model's paper claims need. This
script is that missing step, run as pure post-processing on an existing coverage.json.

DEFINITIONS
-----------
Type accuracy: P(predicted class's TYPE == true class's TYPE), marginalising over
severity, computed by collapsing the flat confusion matrix's rows/cols by type (using
`parse_type_severity`, the SAME function `HierConvNeXt` itself uses to factorise classes --
so this cannot disagree with what the model was actually trained to predict).

Grade accuracy: computed PER GRADED TYPE (a type with >1 severity in this class scheme,
e.g. PositiveDent/NegativeDent have A/B/C; Welding/Deformation/MaterialDefect have only
one grade and are excluded -- there is nothing to confuse). Restricted to tiles where the
TYPE was ALSO predicted correctly (i.e. predicted class's type == true class's type), so
this isolates the severity branch's own accuracy from the type branch's errors -- a tile
predicted as the wrong type was never going to get graded against the right type's scale
in the first place, so including it would conflate two different error sources into one
number. Reported overall (pooled across graded types) and per graded type.

This works on ANY model's coverage.json -- a flat classifier's severity-letter mistakes
collapse into the grade confusion the same way a hier model's do, so the same report
can be used as a flat-head baseline's type/grade breakdown for comparison, not only a
hier model's.

Usage:
    python -m analysis.eval.eval_hier_typegrade <run>_coverage.json [<run2>_coverage.json ...]
"""
import argparse
import json
import sys

import numpy as np

from mvc.core.model_zoo import parse_type_severity


def type_confusion(labels, confusion):
    """(type_names, type_confusion_matrix, type_accuracy)."""
    confusion = np.array(confusion, dtype=np.int64)
    pairs = parse_type_severity(labels)
    types = sorted({t for t, _s in pairs})
    t_idx = {t: i for i, t in enumerate(types)}
    tc = np.zeros((len(types), len(types)), dtype=np.int64)
    for i, (ti, _si) in enumerate(pairs):
        for j, (tj, _sj) in enumerate(pairs):
            tc[t_idx[ti], t_idx[tj]] += confusion[i, j]
    acc = float(np.trace(tc)) / float(tc.sum()) if tc.sum() else float('nan')
    return types, tc, acc


def grade_confusion(labels, confusion):
    """dict: graded_type -> (severities, grade_confusion_matrix, grade_accuracy),
    restricted to tiles where the type was also predicted correctly (see module
    docstring). Also returns the pooled (all graded types stacked) accuracy."""
    confusion = np.array(confusion, dtype=np.int64)
    pairs = parse_type_severity(labels)
    by_type = {}
    for t, s in pairs:
        if s is not None:
            by_type.setdefault(t, set()).add(s)
    graded_types = {t: sorted(sevs) for t, sevs in by_type.items() if len(sevs) > 1}

    out = {}
    pooled_correct = pooled_total = 0
    for t, sevs in graded_types.items():
        idx = [i for i, (ti, si) in enumerate(pairs) if ti == t and si in sevs]
        s_idx = {s: k for k, s in enumerate(sevs)}
        gc = np.zeros((len(sevs), len(sevs)), dtype=np.int64)
        for i in idx:
            si = pairs[i][1]
            for j in idx:  # restricted to type==t on BOTH sides -- predicted type correct
                sj = pairs[j][1]
                gc[s_idx[si], s_idx[sj]] += confusion[i, j]
        acc = float(np.trace(gc)) / float(gc.sum()) if gc.sum() else float('nan')
        out[t] = (sevs, gc, acc)
        pooled_correct += int(np.trace(gc))
        pooled_total += int(gc.sum())
    pooled_acc = pooled_correct / pooled_total if pooled_total else float('nan')
    return out, pooled_acc


def print_matrix(names, mat, title):
    print(f'\n{title}')
    w = max(len(n) for n in names) + 2
    print(' ' * w + ''.join(f'{n:>{w}s}' for n in names))
    for i, n in enumerate(names):
        print(f'{n:<{w}s}' + ''.join(f'{mat[i, j]:>{w}d}' for j in range(len(names))))


def report_one(path):
    d = json.load(open(path))
    labels, confusion = d['labels'], d['confusion']
    print(f'\n{"="*70}\n{path}\n{"="*70}')

    types, tc, t_acc = type_confusion(labels, confusion)
    print_matrix(types, tc, 'TYPE confusion (rows=true, cols=predicted)')
    print(f'\nType accuracy (all classes, marginalised over severity): {t_acc*100:.2f}%')

    by_type, pooled_acc = grade_confusion(labels, confusion)
    for t, (sevs, gc, acc) in sorted(by_type.items()):
        print_matrix(sevs, gc, f'GRADE confusion for {t} '
                               f'(restricted to type predicted correctly)')
        print(f'  {t} grade accuracy: {acc*100:.2f}% (n={int(gc.sum()):,})')
    print(f'\nPooled grade accuracy (all graded types, type-correct subset): '
         f'{pooled_acc*100:.2f}%')
    return {'path': path, 'type_names': types, 'type_confusion': tc.tolist(),
           'type_accuracy': t_acc,
           'grade': {t: {'severities': sevs, 'confusion': gc.tolist(), 'accuracy': acc}
                     for t, (sevs, gc, acc) in by_type.items()},
           'pooled_grade_accuracy': pooled_acc}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('coverage_json', nargs='+')
    ap.add_argument('--out', default=None, help='write a combined JSON report here')
    args = ap.parse_args()

    results = [report_one(p) for p in args.coverage_json]
    if args.out:
        json.dump(results, open(args.out, 'w'), indent=1)
        print(f'\nwrote {args.out}')


if __name__ == '__main__':
    main()
