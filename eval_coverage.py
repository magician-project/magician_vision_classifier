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
    # Exact match -- see the note in score_checkpoints.py. Harmless here (cleanID only
    # feeds the model rebuild), but the same idiom is fatal wherever defect_mass is
    # computed, so it is fixed identically in both places.
    cleanID = next((i for i, c in enumerate(class_names)
                    if c.lower() in ('class_clean', 'clean')), None)
    if cleanID is None:
        sys.exit(f'no clean class among {class_names}')

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
    n = len(class_names)
    correct = np.zeros(n, np.int64)
    support = np.zeros(n, np.int64)
    confuse = np.zeros((n, n), np.int64)   # where the misses actually go
    mass, truth = [], []                   # defect_mass = 1 - P(clean), per tile
    with torch.no_grad():
        for bi, (x, y) in enumerate(loader):
            logits = model(x.to(dev))
            prob = torch.softmax(logits.float(), dim=1).cpu().numpy()
            pred = prob.argmax(1)
            y = y.numpy()
            mass.append(1.0 - prob[:, cleanID])
            truth.append(y)
            for t, p in zip(y, pred):
                support[t] += 1
                correct[t] += (t == p)
                confuse[t, p] += 1
            if bi % 50 == 0:
                print(f'  batch {bi}/{len(loader)}', end='\r')
    print()
    mass = np.concatenate(mass)
    truth = np.concatenate(truth)

    # DETECTION, on the same score the factory KPI uses. Exact-class recall answers "did
    # it name the defect right"; the deployment question is "did it fire at all". A tile
    # called NegativeDentClassC when it is NegativeDentClassA is a detection, not a miss,
    # and the two numbers can differ enormously on classes that look alike. The threshold
    # is calibrated on the coverage set's OWN clean tiles so the false-alarm rate is
    # matched here, exactly as miss@FA5 matches it on the factory val.
    clean_mass = np.sort(mass[truth == cleanID])
    det_at = {}
    for fa in (0.05, 0.10):
        if len(clean_mass) == 0:
            det_at[fa] = None
            continue
        thr = float(np.quantile(clean_mass, 1.0 - fa))
        det_at[fa] = (thr, {i: float((mass[truth == i] >= thr).mean() * 100.0)
                            for i in range(n) if support[i] and i != cleanID})
    print(f'[coverage] FA-matched thresholds from {len(clean_mass):,} coverage clean tiles: '
          + ', '.join(f'FA{int(f*100)} -> {det_at[f][0]:.4f}' for f in (0.05, 0.10)))

    tiers = {k: v['tier'] for k, v in payload['classes'].items()}
    print(f"\n{'class':30s} {'tier':7s} {'support':>9s} {'det@FA5':>8s} {'det@FA10':>9s} "
          f"{'exact %':>8s}  {'misses mostly ->':s}")
    print('-' * 104)
    rows = []
    for i, name in enumerate(class_names):
        if not support[i]:
            continue
        t = tiers.get(name, '-')
        rec = correct[i] / support[i] * 100.0
        d5 = det_at[0.05][1].get(i) if det_at[0.05] else None
        d10 = det_at[0.10][1].get(i) if det_at[0.10] else None
        # dominant wrong prediction, to separate "named it another defect" from "missed it"
        off = confuse[i].copy()
        off[i] = 0
        top = int(off.argmax())
        top_s = f'{class_names[top]} {off[top]/support[i]*100:.0f}%' if off[top] else '-'
        rows.append((name, t, int(support[i]), rec, d5, d10, top_s))
        print(f'{name:30s} {t:7s} {support[i]:9,d} '
              f'{"-" if d5 is None else f"{d5:8.2f}"} {"-" if d10 is None else f"{d10:9.2f}"} '
              f'{rec:8.2f}  {top_s}')

    a = [r for r in rows if r[1] == 'TIER_A' and r[0] != 'class_clean']
    b = [r for r in rows if r[1] == 'TIER_B']
    for label, sel in (('TIER_A (honest, recording-disjoint)', a),
                       ('TIER_B (OPTIMISTIC, frame-disjoint) ', b)):
        if not sel:
            continue
        print(f'\n  {label} macro over {len(sel)} classes:')
        print(f'    detection@FA5  {sum(r[4] for r in sel)/len(sel):6.2f}%   '
              f'detection@FA10 {sum(r[5] for r in sel)/len(sel):6.2f}%   '
              f'exact-class {sum(r[3] for r in sel)/len(sel):6.2f}%')

    print('\nDETECTION is the deployment-relevant column and the one to compare against the'
          '\nfactory KPI (100 - det@FA5 is a miss rate on the same footing as miss@FA5).'
          '\nEXACT-CLASS is stricter: it counts a defect named as the wrong defect as a'
          '\nfailure. A big det/exact gap means the taxonomy is confusable, not that the'
          '\nmodel is blind. Compare ACROSS MODELS; a factory gain with a TIER_A detection'
          '\ndrop is a regression.')

    out = f"{cfg['name']}_{cfg['model']}_coverage.json"
    json.dump({'checkpoint': os.path.basename(ckpt),
               'fa_thresholds': {f'FA{int(f*100)}': (det_at[f][0] if det_at[f] else None)
                                 for f in (0.05, 0.10)},
               'rows': [{'class': n, 'tier': t, 'support': s, 'recall': r,
                         'detect_at_fa5': d5, 'detect_at_fa10': d10, 'top_confusion': tc}
                        for n, t, s, r, d5, d10, tc in rows]},
              open(out, 'w'), indent=1)
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
