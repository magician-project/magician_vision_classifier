#!/usr/bin/env python3
"""Score a checkpoint on the COVERAGE validation set — per-class recall, with tiers.

The factory val (Aug26_78K/val) is the ship/no-ship metric and stays untouched. It cannot,
however, see five defect classes at all, and those are the model's weakest ones, while
WeldingClassA -- its strongest -- is 84% of its defect tiles. This scores the complementary
set built by build_coverage_val.py from frames CARVED OUT of train.

Read the tiers, not just the numbers:
    TIER_A  recording-disjoint -- honest, comparable in kind to the factory val
    TIER_B  frame-disjoint     -- optimistic; its recordings also appear in train, so this
                                  is a regression tripwire, NOT a generalization estimate
    TIER_C  not evaluated      -- vestigial classes (annotation slips, <=500 tiles)

Usage:
    python eval_coverage.py <config.json> [checkpoint.ckpt]
"""

import json
import os
import sys

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Subset

from ClassScheme import apply_class_scheme
from Config import checkIfFileExists, load_hyperparameters
from Datasets import _dataset_source_frames
from LitClassifier import Classifier

COVERAGE = 'val_coverage_frames.json'


def build_coverage_loader(config_json, coverage_path=COVERAGE):
    """The coverage frames, taken from the TRAINING dump with the run's own class scheme."""
    train_dir = config_json['training_dataset']
    train_dir = train_dir[0] if isinstance(train_dir, list) else train_dir
    h5 = f'{train_dir}/dataset.h5'
    assert checkIfFileExists(h5), f'expected an h5 dataset at {h5}'
    from DatasetConverter import HDF5Dataset
    ds = HDF5Dataset(h5)
    ds.metadata = None
    apply_class_scheme(ds, config_json, label='coverage')

    payload = json.load(open(coverage_path))
    frames = set(payload['frames'])
    srcs = np.array(_dataset_source_frames(ds))
    idx = np.where(np.isin(srcs, list(frames)))[0].tolist()
    if not idx:
        sys.exit(f'{coverage_path} matched no frames in {h5}')
    print(f'[coverage] {len(idx):,} tiles from {len(frames):,} held-out frames')
    return ds, Subset(ds, idx), payload


def main():
    cfg_path = sys.argv[1]
    cfg = load_hyperparameters(cfg_path)
    h = cfg['hparams']

    if not cfg['dataloader'].get('exclude_frames'):
        print('!! WARNING: this config has no dataloader.exclude_frames, so the coverage '
              'frames were IN the training set. Recall below measures memorisation, not '
              'detection, and must not be used as a regression signal.\n')

    ds, subset, payload = build_coverage_loader(cfg)
    class_names = list(ds.classes)
    cleanID = next((i for i, c in enumerate(class_names) if 'clean' in c.lower()), None)

    ckpt = sys.argv[2] if len(sys.argv) > 2 else None
    if ckpt is None:
        import glob
        cks = sorted(glob.glob(os.path.join(cfg['checkpoint_dir'], '*.ckpt')))
        if not cks:
            sys.exit(f'no checkpoints in {cfg["checkpoint_dir"]}')
        ckpt = cks[-1]
    print(f'[coverage] scoring {os.path.basename(ckpt)}')

    model = Classifier(
        model=cfg['model'], num_classes=len(class_names), tile_size=h['tile_size'],
        dropout_rate=h['dropout_rate'], base_channels=h.get('base_channels', 32),
        final_dense_layer=h.get('final_dense_layer', 512), clean_class=cleanID,
        penalize_false_clean=float(cfg.get('penalize_false_clean', 0.0)),
        AoLP=h.get('AoLP', False), DoLP=h.get('DoLP', False),
        Unpolarized=bool(h.get('Unpolarized', h.get('unpolarized', False))),
        MaxPolarization=h.get('MaxPolarization', False),
        MinPolarization=h.get('MinPolarization', False),
        RangePolarization=h.get('RangePolarization', False),
        monochrome=h.get('monochrome', False),
        pretrained=bool(h.get('pretrained', True)),
        seed_pretrained_stem=bool(h.get('seed_pretrained_stem', True)),
        timm_stem_stride=h.get('timm_stem_stride', None),
    )
    model.load_state_dict(torch.load(ckpt, weights_only=False, map_location='cpu')['state_dict'])
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(dev).eval()

    loader = DataLoader(subset, batch_size=h['batch_size'], shuffle=False, num_workers=8)
    correct = np.zeros(len(class_names), np.int64)
    support = np.zeros(len(class_names), np.int64)
    with torch.no_grad():
        for bi, (x, y) in enumerate(loader):
            pred = model(x.to(dev)).argmax(1).cpu().numpy()
            y = y.numpy()
            for t, p in zip(y, pred):
                support[t] += 1
                correct[t] += (t == p)
            if bi % 50 == 0:
                print(f'  batch {bi}/{len(loader)}', end='\r')
    print()

    tiers = {k: v['tier'] for k, v in payload['classes'].items()}
    print(f"\n{'class':30s} {'tier':7s} {'support':>9s} {'recall %':>9s}")
    print('-' * 60)
    rows = []
    for i, name in enumerate(class_names):
        if not support[i]:
            continue
        t = tiers.get(name, '-')
        rec = correct[i] / support[i] * 100.0
        rows.append((name, t, int(support[i]), rec))
        print(f'{name:30s} {t:7s} {support[i]:9,d} {rec:9.2f}')

    a = [r for r in rows if r[1] == 'TIER_A' and r[0] != 'class_clean']
    b = [r for r in rows if r[1] == 'TIER_B']
    if a:
        print(f'\n  TIER_A macro recall (honest, recording-disjoint): '
              f'{sum(r[3] for r in a)/len(a):6.2f}%  over {len(a)} classes')
    if b:
        print(f'  TIER_B macro recall (OPTIMISTIC, frame-disjoint) : '
              f'{sum(r[3] for r in b)/len(b):6.2f}%  over {len(b)} classes')
    print('\nCompare these ACROSS MODELS, not against the factory KPI — different data,'
          '\ndifferent difficulty. A factory-KPI gain with a TIER_A drop is a regression.')

    out = f"{cfg['name']}_{cfg['model']}_coverage.json"
    json.dump({'checkpoint': os.path.basename(ckpt),
               'rows': [{'class': n, 'tier': t, 'support': s, 'recall': r} for n, t, s, r in rows]},
              open(out, 'w'), indent=1)
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
