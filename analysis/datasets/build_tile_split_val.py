#!/usr/bin/env python3
"""Freeze a RANDOM, TILE-level train/test split of the training data.

This is the deliberate opposite of every other split in this codebase. `frame_disjoint_split`
and the coverage carve-out (`build_coverage_val.py`) both exist because a naive split leaks --
tiles from the same frame, or frames from the same recording, are near-duplicates, so a
tile-level random split measures memorisation more than generalisation (see 21-8-report.md
Sec.2.2/2.5).

This script builds that leaky split ON PURPOSE, for a different question: an IN-DISTRIBUTION
CEILING / FIT CHECK -- how well can an architecture fit this problem at all, independent of
generalisation -- to compare against the honest coverage numbers and see how much of the gap
is generalisation vs underfitting. Not a generalisation measurement. Do not read its miss@FA5
against the factory/coverage numbers as if it were.

WHY FROZEN TO A FILE, NOT RE-DERIVED BY SEED AT SCORE TIME. `mvc/core/datasets.py` already
has a tile-level `random_split(..., generator=torch.Generator().manual_seed(seed))` fallback,
but `build_val_only` (used by every scorer) explicitly refuses to rebuild it -- untested for
post-hoc reproducibility, and no current config uses it. Rather than resurrect and trust that
path, this follows the same pattern as `val_coverage_frames.json` / `frozen_val_frames`:
freeze the split once, consume it by lookup everywhere. A tile has no identity independent of
its row position (unlike a frame, which has a name), so what gets frozen is plain positional
row indices into the training h5 -- which only works because the dataset is frozen on disk
(21-8-report.md Sec.2: "the dataset is frozen on disk"). `tile_split_indices()` in
`mvc/core/datasets.py` refuses outright if the row count has drifted since this was built.

Usage:
    python build_tile_split_val.py anc_convnext_pico.json [-o OUT] [--val-fraction 0.1] [--seed 42]
"""

import argparse
import datetime
import json
import os

import numpy as np

from mvc.core.config import load_hyperparameters
from mvc.core.datasets import load_training_dataset

DEFAULT_OUT = 'experiments/configs_frozen/traintest_tile_split_frozen.json'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('config', help='a config naming the training_dataset to split '
                                   '(only training_dataset/classes are used from it)')
    ap.add_argument('-o', '--out', default=DEFAULT_OUT)
    ap.add_argument('--val-fraction', type=float, default=0.1,
                    help='matches the dormant dataloader.validation_split default (0.1) '
                         'already sitting in every config')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    cfg = load_hyperparameters(args.config)
    dataset = load_training_dataset(cfg, verbose=True)
    n_total = len(dataset)

    rng = np.random.default_rng(args.seed)
    n_val = int(round(n_total * args.val_fraction))
    val_indices = rng.choice(n_total, size=n_val, replace=False)
    val_indices = sorted(int(i) for i in val_indices)

    payload = {
        'created': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'frozen_from_config': os.path.basename(args.config),
        'seed': args.seed,
        'val_fraction': args.val_fraction,
        'training_dataset': cfg['training_dataset'],
        'n_total': n_total,
        'n_val': len(val_indices),
        '_README': (
            'RANDOM TILE-LEVEL split, sibling-tile leakage EXPECTED and accepted -- this is '
            'an in-distribution fit-ceiling check, not a generalisation measurement. Do not '
            'compare its miss@FA5 against the factory or coverage numbers as if it were the '
            'same kind of instrument. val_indices are plain positional row indices into '
            'training_dataset\'s dataset.h5, valid only as long as that h5 is unchanged -- '
            'tile_split_indices() in mvc/core/datasets.py refuses to use this file if the '
            'dataset\'s current length does not match n_total above. Do not regenerate this '
            'file to "refresh" it once runs exist against it -- that silently breaks '
            'comparability with every prior result, the same reason frozen_val_frames exists.'),
        'val_indices': val_indices,
    }
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w') as fh:
        json.dump(payload, fh, indent=1)

    print(f'\nwrote {args.out}')
    print(f'  {n_total:,} total tiles -> {len(val_indices):,} val / '
          f'{n_total - len(val_indices):,} train ({args.val_fraction:.1%})')
    print(f'\nAdd to every config in this campaign:\n  "dataloader": {{ ..., '
          f'"frozen_tile_split": "{args.out}" }}\n'
          f'and remove "validation_dataset" / "dataloader.exclude_frames" from it.')


if __name__ == '__main__':
    main()
