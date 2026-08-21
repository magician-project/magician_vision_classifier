#!/usr/bin/env python3
"""Report the run-to-run noise floor from the repeated-seed screens.

The campaign has ranked models, accepted and rejected input channels, and chosen what to
full-train, all on single runs — with no idea how much an identical experiment varies.
Several conclusions on record sit inside a point: convnext_atto 8.92 vs convnext_femto
8.93, and Phase 1a's 0.17 DoLP effect on atto. If the spread here is ~0.3, both are noise.

The runs differ ONLY in hparams.seed, with the validation split pinned by
val_frames_frozen.json, so what is measured is training randomness — weight init, sampler
order, augmentation draws, cudnn nondeterminism — against a fixed benchmark.

Usage:  python seed_variance.py [--model convnext_atto]
"""

import glob
import json
import os
import sys

from mvc.core.metrics import miss_at_fa
from mvc.core.artifact_paths import find_artifact


def main():
    model = 'convnext_atto'
    if '--model' in sys.argv:
        model = sys.argv[sys.argv.index('--model') + 1]

    rows = []
    for cfg_path in sorted(glob.glob(f'sv*_{model}.json')) + [f'tz_{model}.json']:
        base = cfg_path[:-len('.json')]
        if base.endswith(('_confusion', '_threshold_curve')) or not os.path.exists(cfg_path):
            continue
        cfg = json.load(open(cfg_path))
        curve_name = f"{cfg['name']}_{cfg['model']}_threshold_curve.json"
        curve = find_artifact(curve_name)
        if curve is None:
            print(f'  (skip {cfg_path}: no {curve_name} yet)')
            continue
        r = miss_at_fa(curve)
        rows.append((cfg['hparams']['seed'], cfg['name'], r[0.05], r[0.10]))

    if len(rows) < 2:
        sys.exit(f'need at least 2 completed runs for {model}, have {len(rows)}')

    print(f"\n{model}, 30k screen, frozen val split — identical except hparams.seed\n")
    print(f"{'seed':>6s} {'run':>8s} {'miss@FA5':>10s} {'miss@FA10':>11s}")
    for seed, name, m5, m10 in sorted(rows):
        print(f'{seed:6d} {name:>8s} {m5:10.2f} {m10:11.2f}')

    for label, i in (('miss@FA5', 2), ('miss@FA10', 3)):
        vals = [r[i] for r in rows]
        n = len(vals)
        mean = sum(vals) / n
        sd = (sum((v - mean) ** 2 for v in vals) / (n - 1)) ** 0.5
        spread = max(vals) - min(vals)
        print(f'\n{label}:  n={n}  mean={mean:.2f}  sd={sd:.2f}  '
              f'range={min(vals):.2f}..{max(vals):.2f}  spread={spread:.2f}')
        if label == 'miss@FA5':
            print(f'  -> Treat differences below ~{2*sd:.2f} (2 sd) between single runs as '
                  f'NOT established.')
            for gap, what in ((0.01, 'convnext_atto 8.92 vs convnext_femto 8.93'),
                              (0.17, 'Phase 1a DoLP effect on convnext_atto'),
                              (0.59, 'Phase 1a DoLP effect on convnext_pico'),
                              (0.53, 'efficientnet_b0 6.98 vs convnext_pico 7.51 (Phase 3)'),
                              (0.99, 'convnext_pico vs convnext_nano (Phase 1b)')):
                verdict = 'NOISE' if gap < 2 * sd else ('marginal' if gap < 3 * sd else 'real')
                print(f'     {gap:5.2f}  {what:52s} -> {verdict}')


if __name__ == '__main__':
    main()
