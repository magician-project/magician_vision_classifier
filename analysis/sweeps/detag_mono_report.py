#!/usr/bin/env python3
"""Polarized vs monochrome on the DeTaG head (hier_convnext_pico, hybrid, no distillation).

The sixth row of `knowledge/25-8-mono-vs-pol-report.md`'s table. That campaign asked
whether polarization still pays on the CURRENT protocol across the deployment front, and
found mono costs 4.3-7.1 TIER_A points on all 5 flat-headed architectures. DeTaG is the
architecture that came after it: a hierarchical detect/type/grade head whose grading
branch reads the same pooled vector the flat head does. If polarization's contribution
were an artefact of the flat classifier, this is where it would show up.

    pol  = hiernodist1ep   (hparams.monochrome=False)
    mono = mondetagnodist  (hparams.monochrome=True)

The two configs are byte-identical apart from `monochrome`, `name`, `checkpoint_dir` and
`checkpoint_save_top_k` (6 vs 1 -- inert at training_epochs=1, which writes exactly one
end-of-epoch checkpoint). Same seed 42, same coverage carve-out, same checkpoint_monitor,
same box, same dataloader. Only the polarization channel moves -- `mono_frontier_sweep.py`'s
methodology, applied to a pair where the mono arm happened to be trained first.

WHY NOT full_zoo_report.score()
-------------------------------
`mono_frontier_report.py` reuses it unmodified, but it resolves the threshold curve as
`{run}_ep{N}_{sfx}_threshold_curve.json` -- the multi-epoch scoring path's name. These two
runs are single-epoch and their curves are written as `{run}_{sfx}_threshold_curve.json`,
so `score()` returns None for both. Rather than widen a function seven other reports read,
this resolves both spellings itself and computes the same two quantities with the same
shared helpers (`mvc.core.metrics.miss_at_fa`, `report_common.tier_a_macro_from_rows`), so
the numbers remain comparable to every other table in the campaign.

Usage:  python -m analysis.sweeps.detag_mono_report
"""

import json

from mvc.core.artifact_paths import find_artifact
from mvc.core.metrics import miss_at_fa

from . import report_common as rc

SFX = 'hier_convnext_pico'
POL = 'hiernodist1ep'
MONO = 'mondetagnodist'


def _curve_path(run):
    """The run's threshold curve, under either the single- or multi-epoch spelling."""
    hit = find_artifact(f'{run}_{SFX}_threshold_curve.json')
    if hit:
        return hit
    ep = rc.best_epoch(f'datasets/mix_ckpts/{run}_{SFX}')
    return None if ep is None else find_artifact(
        f'{run}_ep{ep}_{SFX}_threshold_curve.json')


def arm(run):
    """{miss5, tier_a, rows} for one arm, or None if it has not been scored yet."""
    curve, cov = _curve_path(run), find_artifact(f'{run}_{SFX}_coverage.json')
    if not (curve and cov):
        return None
    rows = [r for r in json.load(open(cov))['rows'] if r['class'] != 'class_clean']
    return {'miss5': miss_at_fa(curve)[0.05],
            'tier_a': rc.tier_a_macro_from_rows(rows),
            'rows': {r['class']: r for r in rows}}


def main():
    pol, mono = arm(POL), arm(MONO)
    print('\nPolarized vs monochrome, DeTaG (hier_convnext_pico, hybrid, no distillation),')
    print('Aug26_78K, 4ch+DoLP, 10-class, coverage carved out, seed 42, 1 epoch.')
    print(f'pol = {POL} (monochrome=False) · mono = {MONO} (monochrome=True), '
          'identical otherwise.\n')

    for label, a in (('pol', pol), ('mono', mono)):
        if a is None:
            print(f'!!! {label} arm not scored yet '
                  f'({POL if label == "pol" else MONO}) -- '
                  'needs score_checkpoints + eval_coverage.')
    if pol is None or mono is None:
        return

    hdr = (f'{"arm":6s} {"fac miss@FA5":>13s} {"TIER_A cov":>11s}')
    print(hdr + '\n' + '-' * len(hdr))
    print(f'{"pol":6s} {pol["miss5"]:13.2f} {pol["tier_a"]:11.2f}')
    print(f'{"mono":6s} {mono["miss5"]:13.2f} {mono["tier_a"]:11.2f}')
    print(f'{"Δ":6s} {mono["miss5"] - pol["miss5"]:+13.2f} '
          f'{mono["tier_a"] - pol["tier_a"]:+11.2f}')

    print('\nPer-class TIER_A detect@FA5 (mono − pol), the §3 breakdown of '
          '25-8-mono-vs-pol-report.md:')
    hdr = f'{"class":28s} {"pol":>7s} {"mono":>7s} {"Δ":>7s}'
    print(hdr + '\n' + '-' * len(hdr))
    for cls, r in sorted(pol['rows'].items()):
        m = mono['rows'].get(cls)
        if r.get('tier') != 'TIER_A' or m is None:
            continue
        if r.get('detect_at_fa5') is None or m.get('detect_at_fa5') is None:
            continue
        print(f'{cls:28s} {r["detect_at_fa5"]:7.2f} {m["detect_at_fa5"]:7.2f} '
              f'{m["detect_at_fa5"] - r["detect_at_fa5"]:+7.2f}')

    print('\nNegative Δ = polarization helps (mono is worse without it). The 5 flat-headed '
          'front models\nin 25-8-mono-vs-pol-report.md sit at −4.28 to −7.11 TIER_A, '
          'mean −5.72.')
    print('n=1 per arm. Coverage seed sd is 0.43 (24-8-report.md §10), so read the '
          'direction and\nmagnitude, not sub-point differences. fac miss@FA5 is '
          'incidence-weighted and was the\nunreliable column in that campaign — TIER_A '
          'coverage is the number to read.')


if __name__ == '__main__':
    main()
