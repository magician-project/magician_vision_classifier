#!/usr/bin/env python3
"""pol vs mono on the 5 front models (analysis/sweeps/mono_frontier_sweep.py).

Reuses full_zoo_report.score() unmodified for both sides -- it already reads
{run}_{sfx}_coverage.json + the threshold curve at the monitored-best checkpoint, which is
exactly what both the existing pol runs and the new mono runs produce under this protocol.

Usage:  python mono_frontier_report.py
"""

from .full_zoo_report import score
from .mono_frontier_sweep import FRONT


def main():
    print('\nPolarized vs monochrome, deployment front, Aug26_78K, 4ch+DoLP, 10-class, '
          'coverage carved out.')
    print('mono = hparams.monochrome=True, identical config otherwise (same seed/epochs/'
          'checkpoint_monitor as its pol twin).\n')

    hdr = (f'{"model":18s} {"pol fac":>8s} {"mono fac":>9s} {"Δ fac":>7s} '
           f'{"pol TIER_A":>11s} {"mono TIER_A":>12s} {"Δ TIER_A":>9s}')
    print(hdr + '\n' + '-' * len(hdr))

    rows = []
    for pol_run, model, mono_tag in FRONT:
        pol = score(pol_run, model)
        mono = score(mono_tag, model)
        rows.append((model, pol, mono))
        if pol is None:
            print(f'{model:18s}  pol not scored -- run_mono_frontier_sweep.sh has not '
                  f'trained/scored the baseline?')
            continue
        if mono is None:
            print(f'{model:18s} {pol["miss5"]:8.2f} {"--":>9s} {"--":>7s} '
                  f'{(pol["tier_a"] or float("nan")):11.2f} {"--":>12s} {"--":>9s}  '
                  f'(mono not run yet)')
            continue
        dfac = mono['miss5'] - pol['miss5']
        dcov = (mono['tier_a'] - pol['tier_a']) if (mono['tier_a'] is not None
                                                      and pol['tier_a'] is not None) else None
        cov_p = pol['tier_a'] if pol['tier_a'] is not None else float('nan')
        cov_m = mono['tier_a'] if mono['tier_a'] is not None else float('nan')
        dcov_s = f'{dcov:+9.2f}' if dcov is not None else f'{"--":>9s}'
        print(f'{model:18s} {pol["miss5"]:8.2f} {mono["miss5"]:9.2f} {dfac:+7.2f} '
              f'{cov_p:11.2f} {cov_m:12.2f} {dcov_s}')

    done = [(m, p, mo) for m, p, mo in rows if p is not None and mo is not None]
    if done:
        mean_dcov = sum(mo['tier_a'] - p['tier_a'] for _, p, mo in done
                         if p['tier_a'] is not None and mo['tier_a'] is not None) / len(done)
        print(f'\nmean Δ TIER_A (mono - pol) over {len(done)} scored models: {mean_dcov:+.2f}')
        print('Negative = polarization helps (mono is worse without it), matching the '
              "PLAN.md 2026-07-28 direction on convnext_tiny.")
    print('\nfac = factory-style aggregate miss@FA5 (this table\'s own protocol, not the '
          'factory KPI). TIER_A = recording-disjoint coverage macro, same instrument as '
          '21-8/24-8-report.md.')


if __name__ == '__main__':
    main()
