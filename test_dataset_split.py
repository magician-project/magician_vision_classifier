#!/usr/bin/env python3
"""Equivalence test for Datasets.build_train_val / build_val_only.

The validation split used to be rebuilt by hand in the trainer and mirrored in
score_checkpoints.py, eval_coverage.py and eval_ema_tta.py, coupled only by comments
naming line numbers in another file. This asserts the shared implementation reproduces
what those copies produced, on REAL datasets -- not on synthetic fixtures, because the
failure being guarded against is a class ORDER difference between two real dumps.

Checked per config, against verbatim transcriptions of the old blocks:

  [A] build_val_only  == score_checkpoints.build_val_loader   (the four scorers' path)
  [B] build_train_val == the trainer's dataset block           (the training path)

Equality means: same class list IN ORDER, same length, same per-tile source frames and
same per-tile labels -- i.e. the same tiles in the same order carrying the same targets.
Comparing lengths alone would pass on two splits that disagree tile for tile.

Configs are deduplicated by the fields the chain actually reads, so this covers every
distinct dataset path rather than re-testing 114 near-identical configs.

Run:  python test_dataset_split.py [--all]
"""

import glob
import json
import os
import sys

import numpy as np

from Config import load_hyperparameters

failures = []


def check(cond, msg):
    if not cond:
        failures.append(msg)
    return cond


def identity(ds):
    """(source frame, label) per tile -- the thing that must not differ.

    Both sides hand back Subsets (the exclude_frames carve-out, the frame-disjoint split),
    and _dataset_source_frames only understands the underlying HDF5/Combined dataset. So
    unwrap the Subset chain first, composing the index maps as we go: with
    Subset(Subset(base, i1), i2) the tile at position k is base[i1[i2[k]]].
    """
    from torch.utils.data import Subset

    from Datasets import _dataset_source_frames

    idx = None
    while isinstance(ds, Subset):
        inner = list(ds.indices)
        idx = inner if idx is None else [inner[i] for i in idx]
        ds = ds.dataset

    frames = np.asarray(_dataset_source_frames(ds))
    if getattr(ds, 'labels', None) is not None:
        labels = np.asarray(ds.labels)
    elif getattr(ds, 'targets', None) is not None:
        labels = np.asarray(ds.targets)
    else:
        labels = np.asarray([ds[i][1] for i in range(len(ds))])

    if idx is not None:
        sel = np.asarray(idx, dtype=np.int64)
        frames, labels = frames[sel], labels[sel]
    return frames, labels


def same(a, b, label):
    fa, la = identity(a)
    fb, lb = identity(b)
    if not check(len(a) == len(b), f'{label}: length {len(a)} != {len(b)}'):
        return False
    ok = fa.shape == fb.shape and bool((fa == fb).all())
    check(ok, f'{label}: source frames differ')
    ok2 = la.shape == lb.shape and bool((la == lb).all())
    check(ok2, f'{label}: labels differ')
    return ok and ok2


# -------------------------------------------------------- [A] scorer transcription
def old_build_val_loader(config_json):
    """score_checkpoints.build_val_loader as it was BEFORE the refactor, transcribed.

    Transcribed rather than imported on purpose: score_checkpoints now calls the shared
    chain, so importing it would compare the new implementation against itself and pass
    unconditionally.
    """
    import os as _os

    from ClassScheme import apply_class_scheme, align_dataset_to_classes
    from Config import checkIfFileExists
    from Datasets import CombinedDataset, frame_disjoint_split
    from torch.utils.data import DataLoader, Subset
    import pytorch_lightning as pl
    import random as _random
    import torch as _torch

    def _load_one_dir(d, cfg, label=None):
        h5 = '%s/dataset.h5' % d
        assert checkIfFileExists(h5), h5
        from DatasetConverter import HDF5Dataset
        ds = HDF5Dataset(h5)
        ds.metadata = None
        return apply_class_scheme(ds, cfg,
                                  label=label or _os.path.basename(str(d).rstrip('/')))

    directory = config_json['training_dataset']
    directories = directory if isinstance(directory, list) else [directory]

    val_dir = config_json.get('validation_dataset')
    if val_dir:
        val_dir = val_dir[0] if isinstance(val_dir, list) else val_dir
        train_ds = _load_one_dir(directories[0], config_json, label='train(classes only)')
        val_ds = _load_one_dir(val_dir, config_json, label='validation')
        if list(val_ds.classes) != list(train_ds.classes):
            align_dataset_to_classes(val_ds, train_ds.classes)
        if list(val_ds.classes) != list(train_ds.classes):
            raise ValueError('train/val class mismatch')
        bs = config_json['hparams']['batch_size']
        nw = config_json['dataloader']['num_workers']
        kw = {'pin_memory': True, 'persistent_workers': True} if nw > 0 else {}
        return val_ds, DataLoader(val_ds, batch_size=bs, shuffle=False,
                                  num_workers=nw, drop_last=False, **kw)

    subs = [_load_one_dir(d, config_json) for d in directories]
    if len(subs) == 1:
        dataset = subs[0]
    else:
        canon = config_json.get('canonical_classes') or list(subs[0].classes)
        for ds in subs:
            align_dataset_to_classes(ds, canon)
        dataset = CombinedDataset(subs)

    seed = config_json['hparams'].get('seed', config_json['dataloader']['seed'])
    val_split = config_json['dataloader']['validation_split']
    _torch.manual_seed(seed); np.random.seed(seed); _random.seed(seed)
    pl.seed_everything(seed, workers=True)
    assert config_json['dataloader'].get('frame_disjoint_split')
    _, va_idx = frame_disjoint_split(
        dataset, val_split, seed,
        frozen_val_frames=config_json['dataloader'].get('frozen_val_frames') or None)
    val_dataset = Subset(dataset, va_idx)
    nw = config_json['dataloader']['num_workers']
    lk = {}
    if nw > 0:
        lk = {"pin_memory": True, "persistent_workers": True,
              "prefetch_factor": int(config_json['dataloader'].get('prefetch_factor', 4))}
    return dataset, DataLoader(val_dataset, batch_size=config_json['hparams']['batch_size'],
                               shuffle=False, num_workers=nw, drop_last=False, **lk)


# ------------------------------------------------------------ [B] trainer transcription
def trainer_block(config_json):
    """The trainer's dataset chain, transcribed verbatim as the reference.

    Kept deliberately as a copy: the point of the test is that the shared implementation
    agrees with what the trainer USED to do, so paraphrasing it here would test nothing.
    """
    import os as _os

    from ClassScheme import apply_class_scheme, align_dataset_to_classes
    from Config import checkIfFileExists
    from Datasets import CombinedDataset, exclude_frames_indices, frame_disjoint_split
    from torch.utils.data import Subset

    directory = config_json['training_dataset']
    directories = directory if isinstance(directory, list) else [directory]

    def _load_one(d):
        h5 = '%s/dataset.h5' % d
        assert checkIfFileExists(h5), h5
        from DatasetConverter import HDF5Dataset
        ds = HDF5Dataset(h5)
        ds.metadata = None
        return apply_class_scheme(ds, config_json,
                                  label=_os.path.basename(str(d).rstrip('/')))

    subs = [_load_one(d) for d in directories]
    if len(subs) == 1:
        dataset = subs[0]
    else:
        canon = config_json.get('canonical_classes') or list(subs[0].classes)
        for ds in subs:
            align_dataset_to_classes(ds, canon)
        dataset = CombinedDataset(subs)

    val_directory = config_json.get('validation_dataset') or None
    if val_directory:
        H5PYValFilename = f"{val_directory}/dataset.h5"
        from DatasetConverter import HDF5Dataset
        val_dataset = HDF5Dataset(H5PYValFilename)
        val_dataset.metadata = None
        apply_class_scheme(val_dataset, config_json, label="validation")
        if val_dataset.classes != dataset.classes:
            align_dataset_to_classes(val_dataset, dataset.classes)
        if dataset.classes != val_dataset.classes:
            raise ValueError('class mismatch')
        if dataset.class_to_idx != val_dataset.class_to_idx:
            raise ValueError('class_to_idx mismatch')
        train_dataset = dataset
        _excl = config_json['dataloader'].get('exclude_frames') or None
        if _excl:
            train_dataset = Subset(dataset, exclude_frames_indices(dataset, _excl))
        return dataset, train_dataset, val_dataset

    seed = config_json['hparams']['seed']
    val_split = config_json['dataloader']['validation_split']
    if config_json['dataloader'].get('frame_disjoint_split'):
        frozen_val = config_json['dataloader'].get('frozen_val_frames') or None
        tr_idx, va_idx = frame_disjoint_split(dataset, val_split, seed,
                                              frozen_val_frames=frozen_val)
        return dataset, Subset(dataset, tr_idx), Subset(dataset, va_idx)
    raise RuntimeError('tile-level split: not covered')


# ------------------------------------------------------------------------- config picking
CHAIN_KEYS = ('training_dataset', 'validation_dataset', 'canonical_classes',
              'keep_classes', 'merge_classes', 'drop_classes', 'strip_severity',
              'class_scheme')
LOADER_KEYS = ('exclude_frames', 'frame_disjoint_split', 'frozen_val_frames',
               'validation_split')


def signature(cfg):
    d = cfg.get('dataloader', {})
    return json.dumps([[cfg.get(k) for k in CHAIN_KEYS],
                       [d.get(k) for k in LOADER_KEYS],
                       cfg.get('hparams', {}).get('seed')], sort_keys=True, default=str)


def pick_configs(all_of_them):
    seen, out = {}, []
    for p in sorted(set(glob.glob('*.json') + glob.glob('configs/*.json') +
                        glob.glob('experiments/configs_*/*.json') +
                        # run configs are filed beside their weights now; without this the
                        # corpus silently shrank from 167 to 135 when they moved.
                        glob.glob('experiments/**/*.json', recursive=True))):
        try:
            cfg = json.load(open(p))
        except Exception:
            continue
        if not isinstance(cfg, dict) or 'dataloader' not in cfg or 'hparams' not in cfg:
            continue
        tr = cfg.get('training_dataset')
        tr = tr[0] if isinstance(tr, list) else tr
        if not tr or not os.path.exists(f'{tr}/dataset.h5'):
            continue
        sig = signature(cfg)
        if not all_of_them and sig in seen:
            continue
        seen[sig] = p
        out.append(p)
    return out


def test_ram_preload_is_order_preserving():
    """The one thing preload_ram=False above leaves unchecked, checked directly.

    If RAMPreloadedDataset could reorder or drop samples, the split taken after it would
    hold different tiles than the split taken before it -- so this is the assumption the
    [B] comparison rests on, tested on a small synthetic dataset rather than on 7.1M tiles.
    """
    import torch
    from torch.utils.data import Dataset

    from Datasets import RAMPreloadedDataset

    class Tiny(Dataset):
        classes = ['a', 'b']
        class_to_idx = {'a': 0, 'b': 1}

        def __init__(self, n=64):
            self.targets = [i % 2 for i in range(n)]

        def __len__(self):
            return len(self.targets)

        def __getitem__(self, i):
            return torch.full((1, 2, 2), float(i)), self.targets[i]

    base = Tiny()
    wrapped = RAMPreloadedDataset(base, show_progress=False)
    check(len(wrapped) == len(base), 'RAMPreloadedDataset changed the dataset length')
    check(list(wrapped.targets) == list(base.targets),
          'RAMPreloadedDataset reordered the targets')
    mismatched = [i for i in range(len(base))
                  if not torch.equal(wrapped[i][0], base[i][0]) or wrapped[i][1] != base[i][1]]
    check(not mismatched,
          f'RAMPreloadedDataset returned different samples at positions {mismatched[:5]}')
    print('  [D] RAMPreloadedDataset preserves order, length and content')


def main():
    all_of_them = '--all' in sys.argv
    configs = pick_configs(all_of_them)
    print(f'{len(configs)} distinct dataset chain(s) to check'
          f'{"" if all_of_them else " (deduplicated; --all for every config)"}\n')

    for path in configs:
        cfg = load_hyperparameters(path)
        name = os.path.basename(path)
        from Datasets import build_train_val, build_val_only

        # [A] the scorers' path
        _, loader_old = old_build_val_loader(cfg)
        sp = build_val_only(cfg, verbose=False)
        check(sp.train is None, f'{name}: build_val_only returned a train split')
        ok_a = same(loader_old.dataset, sp.val, f'{name} [A] val_only')

        # [B] the trainer's path.
        # preload_ram=False: the reference transcription does not RAM-preload either, so
        # enabling it here would compare a cached dataset against an uncached one AND cost
        # ten minutes per chain caching 7.1M tiles. RAMPreloadedDataset caches dataset[i]
        # for i in 0..n-1 in order, so it cannot reorder or drop tiles -- test_ram_preload_
        # is_order_preserving below asserts exactly that, cheaply, instead.
        ds_t, train_t, val_t = trainer_block(cfg)
        sp2 = build_train_val(cfg, verbose=False, preload_ram=False)
        ok_b = (check(list(ds_t.classes) == sp2.classes,
                      f'{name} [B] class order differs') &
                same(train_t, sp2.train, f'{name} [B] train') &
                same(val_t, sp2.val, f'{name} [B] val'))

        # the two paths must agree with each other on val, too
        ok_c = same(sp.val, sp2.val, f'{name} [C] val_only vs train_val')

        print(f'  {name:44s} val={len(sp.val):>8,d} train={len(sp2.train):>8,d} '
              f'A={"ok" if ok_a else "FAIL"} B={"ok" if ok_b else "FAIL"} '
              f'C={"ok" if ok_c else "FAIL"}')

    test_ram_preload_is_order_preserving()

    print()
    if failures:
        print(f'FAILED ({len(failures)}):')
        for f in failures:
            print('  -', f)
        return 1
    print('all checks passed -- shared chain reproduces both hand-written blocks')
    return 0


if __name__ == '__main__':
    sys.exit(main())
