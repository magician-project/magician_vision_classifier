#!/usr/bin/env python3
"""Freeze the current validation split by FRAME NAME so it survives dataset growth.

The problem this closes. `frame_disjoint_split` picks validation frames with
`rng.choice` over DENSE frame ids produced by `np.unique`. Those ids are a function of
the frame SET, so adding newly-annotated frames renumbers everything and selects a
completely different validation set — frames that were held out become training data and
vice versa. Every number in PLAN.md (convnext_pico 4.37, the 30k screen table, the
custom-net series) was measured on one particular split; the moment the annotation
effort adds frames, a rerun measures something else and the comparison is gone. It
cannot be recovered afterwards, because reconstructing the old split requires the old
frame set.

Freezing writes the validation frames out by name. `frame_disjoint_split(...,
frozen_val_frames=...)` then reproduces exactly that validation set and sends everything
else — including frames that did not exist at freeze time — to training. So the training
set grows with the annotation effort while the yardstick stays fixed.

Run this BEFORE the next dataset dump, then set in every config:

    "dataloader": { "frozen_val_frames": "val_frames_frozen.json", ... }

Usage:
    python freeze_val_split.py p2_convnext_pico.json [-o val_frames_frozen.json]
"""

import argparse
import datetime
import json
import os
import sys

import numpy as np

from mvc.core.class_scheme import apply_class_scheme, align_dataset_to_classes
from mvc.core.config import checkIfFileExists, load_hyperparameters
from mvc.core.datasets import CombinedDataset, _dataset_source_frames, frame_disjoint_split


def build_dataset(config_json):
    """Mirror the trainer's dataset assembly (no RAM cache: only metadata is needed)."""
    directory = config_json['training_dataset']
    directories = directory if isinstance(directory, list) else [directory]

    def _load_one(d):
        h5 = '%s/dataset.h5' % d
        assert checkIfFileExists(h5), f'expected an h5 dataset at {h5}'
        from mvc.core.dataset_converter import HDF5Dataset
        ds = HDF5Dataset(h5)
        ds.metadata = None
        return apply_class_scheme(ds, config_json,
                                  label=os.path.basename(str(d).rstrip('/')))

    subs = [_load_one(d) for d in directories]
    if len(subs) == 1:
        return subs[0]
    canon = config_json.get('canonical_classes') or list(subs[0].classes)
    for ds in subs:
        align_dataset_to_classes(ds, canon)
    return CombinedDataset(subs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('config', help='a config whose dataset/seed/val_split define the split to freeze')
    ap.add_argument('-o', '--out', default='val_frames_frozen.json')
    args = ap.parse_args()

    cfg = load_hyperparameters(args.config)
    dl = cfg['dataloader']
    if not dl.get('frame_disjoint_split'):
        sys.exit('this config does not use a frame-disjoint split; nothing meaningful to freeze')
    if dl.get('frozen_val_frames'):
        sys.exit(f"this config ALREADY uses a frozen list ({dl['frozen_val_frames']}); "
                 f"freezing from it would just copy that file")

    seed, val_split = dl['seed'], dl['validation_split']
    dataset = build_dataset(cfg)

    srcs = np.array(_dataset_source_frames(dataset))
    tr_idx, va_idx = frame_disjoint_split(dataset, val_split, seed)
    val_frames = sorted(set(srcs[va_idx].tolist()))
    train_frames = set(srcs[tr_idx].tolist())

    # A frame landing in both halves would mean the split is not frame-disjoint after
    # all, and the frozen list would silently pull training frames into validation.
    overlap = set(val_frames) & train_frames
    assert not overlap, f'{len(overlap)} frames straddle the split -- refusing to freeze'

    payload = {
        'created': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'frozen_from_config': os.path.basename(args.config),
        'seed': seed,
        'validation_split': val_split,
        'training_dataset': cfg['training_dataset'],
        'n_val_frames': len(val_frames),
        'n_train_frames': len(train_frames),
        'n_val_tiles': len(va_idx),
        'n_train_tiles': len(tr_idx),
        '_README': (
            'Validation frames pinned by name. Pass via dataloader.frozen_val_frames so '
            'validation stays fixed while the training set grows with new annotations. '
            'seed/validation_split above are what PRODUCED this list; they are ignored '
            'when it is used. Do not regenerate this file to "refresh" it -- that defeats '
            'the entire purpose and silently breaks comparability with every prior result.'),
        'val_frames': val_frames,
    }
    with open(args.out, 'w') as fh:
        json.dump(payload, fh, indent=1)

    print(f'\nwrote {args.out}')
    print(f'  {len(val_frames):,} val frames / {len(train_frames):,} train frames')
    print(f'  {len(va_idx):,} val tiles / {len(tr_idx):,} train tiles')
    print(f'\nAdd to every future config:\n  "dataloader": {{ ..., '
          f'"frozen_val_frames": "{args.out}" }}')


if __name__ == '__main__':
    main()
