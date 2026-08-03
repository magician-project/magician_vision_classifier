#!/usr/bin/env python3
"""Score saved epoch checkpoints on the detection KPI (miss @ matched false-alarm).

Motivation: the trainer selects ONE checkpoint via `checkpoint_monitor` (default
val_detect_auroc) and only that one ever gets a threshold sweep. But AUROC saturates
-- across a whole r0dolpf run it moved 0.9813..0.9840 while the KPI moves by points --
so the monitor may not be picking the epoch that is actually best on miss@FA. With
`checkpoint_save_top_k` > 1 every epoch survives, and the eval is ~94 s on GPU, so the
question is answerable directly.

Rebuilds the SAME validation split the training run used (same config, same seed, same
frame-disjoint split) and runs the same sweep from Evaluation.py, so the numbers are
comparable to the run's own reported result. As a control it re-scores the epoch the
trainer already evaluated; that number must reproduce, or the split was rebuilt wrong.

Usage:
    python score_checkpoints.py <config.json> [ckpt_dir]
"""

import glob
import os
import random
import re
import sys

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Subset

from ClassScheme import apply_class_scheme, align_dataset_to_classes
from Config import checkIfFileExists, load_hyperparameters
from Datasets import CombinedDataset, frame_disjoint_split
from Evaluation import run_confusion_and_threshold_sweep
from LitClassifier import Classifier


def build_val_loader(config_json):
    """Rebuild the run's validation split. Mirrors the trainer's dataset path."""
    directory = config_json['training_dataset']
    directories = directory if isinstance(directory, list) else [directory]

    def _load_one(d):
        h5 = '%s/dataset.h5' % d
        assert checkIfFileExists(h5), f'expected an h5 dataset at {h5}'
        from DatasetConverter import HDF5Dataset
        ds = HDF5Dataset(h5)
        ds.metadata = None
        return apply_class_scheme(ds, config_json,
                                  label=os.path.basename(str(d).rstrip('/')))

    subs = [_load_one(d) for d in directories]
    if len(subs) == 1:
        dataset = subs[0]
    else:
        canon = config_json.get('canonical_classes') or list(subs[0].classes)
        for ds in subs:
            align_dataset_to_classes(ds, canon)
        dataset = CombinedDataset(subs)

    seed = config_json['dataloader']['seed']
    val_split = config_json['dataloader']['validation_split']
    batch_size = config_json['hparams']['batch_size']
    num_workers = config_json['dataloader']['num_workers']

    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    pl.seed_everything(seed, workers=True)

    assert config_json['dataloader'].get('frame_disjoint_split'), \
        'this run did not use a frame-disjoint split; rebuild logic would differ'
    _, va_idx = frame_disjoint_split(dataset, val_split, seed)
    val_dataset = Subset(dataset, va_idx)

    loader_kwargs = {}
    if num_workers > 0:
        loader_kwargs = {"pin_memory": True, "persistent_workers": True,
                         "prefetch_factor": int(config_json['dataloader'].get('prefetch_factor', 4))}
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, drop_last=False, **loader_kwargs)
    return dataset, val_loader


def miss_at_fa(curve_path, targets=(0.05, 0.10)):
    import json
    d = json.load(open(curve_path))
    pts = (d.get('sweeps') or {}).get('defect_mass') or d['sweep']
    fa = np.array([p['false_alarm'] for p in pts], float)
    det = np.array([p['detected'] for p in pts], float)
    o = np.argsort(fa); fa, det = fa[o], det[o]
    return {t: (1.0 - float(np.interp(t, fa, det))) * 100.0 for t in targets}


def main():
    cfg_path = sys.argv[1]
    config_json = load_hyperparameters(cfg_path)
    ckpt_dir = sys.argv[2] if len(sys.argv) > 2 else config_json['checkpoint_dir']

    ckpts = sorted(glob.glob(os.path.join(ckpt_dir, '*.ckpt')))
    assert ckpts, f'no checkpoints in {ckpt_dir}'
    print(f'Scoring {len(ckpts)} checkpoints from {ckpt_dir}\n')

    dataset, val_loader = build_val_loader(config_json)
    print(f'\nval tiles: {len(val_loader.dataset)}\n')

    class_names = list(dataset.classes)
    cleanClassID = next((i for i, c in enumerate(class_names) if 'clean' in c.lower()), None)
    h = config_json['hparams']

    trainer = pl.Trainer(accelerator=config_json.get('accelerator', 'auto'),
                         devices=config_json.get('devices', 1), logger=False,
                         enable_checkpointing=False, enable_progress_bar=True)

    results = []
    for ck in ckpts:
        tag = os.path.basename(ck)
        epoch = int(re.search(r'epoch=(\d+)', tag).group(1))
        print(f'\n=== scoring {tag}')
        # Mirror the trainer's Classifier construction. Only the args that change the
        # ARCHITECTURE or the input features matter for scoring (augmentation and loss
        # terms are inert in eval), but keeping the list aligned avoids surprises.
        model = Classifier(
            model=config_json['model'], num_classes=len(class_names),
            tile_size=h['tile_size'], dropout_rate=h['dropout_rate'],
            base_channels=h['base_channels'], final_dense_layer=h['final_dense_layer'],
            clean_class=cleanClassID,
            penalize_false_clean=float(config_json.get('penalize_false_clean', 0.0)),
            AoLP=h.get('AoLP', False), DoLP=h.get('DoLP', False),
            Unpolarized=bool(h.get('Unpolarized', h.get('unpolarized', False))),
            MaxPolarization=h.get('MaxPolarization', False),
            MinPolarization=h.get('MinPolarization', False),
            RangePolarization=h.get('RangePolarization', False),
            monochrome=h.get('monochrome', False),
            custom_early_convs=h.get('custom_early_convs', 0),
            custom_channels=h.get('custom_channels'),
            custom_res_blocks=h.get('custom_res_blocks'),
            custom_wavelet_pools=h.get('custom_wavelet_pools'),
            custom_wavelet_stem=h.get('custom_wavelet_stem', 0),
            pretrained=bool(h.get('pretrained', True)),
        )
        state = torch.load(ck, weights_only=False, map_location='cpu')
        model.load_state_dict(state['state_dict'])
        model.eval()

        # write artifacts under a per-epoch name so nothing clobbers the run's own
        name = f"{config_json['name']}_ep{epoch}"
        cfg_copy = dict(config_json)
        run_confusion_and_threshold_sweep(model, trainer, val_loader, dataset,
                                          cfg_copy, f"{name}_{config_json['model']}",
                                          cleanClassID,
                                          tile_size=h['tile_size'],
                                          epochs=h.get('training_epochs'))
        curve = f"{name}_{config_json['model']}_threshold_curve.json"
        r = miss_at_fa(curve)
        vl = re.search(r'val_loss=([0-9]+\.[0-9]+)', tag)
        va = re.search(r'val_detect_auroc=([0-9]+\.[0-9]+)', tag)
        results.append((epoch, float(vl.group(1)) if vl else None,
                        float(va.group(1)) if va else None, r[0.05], r[0.10]))

    print('\n' + '=' * 74)
    print('%6s %10s %18s %10s %11s' % ('epoch', 'val_loss', 'val_detect_auroc', 'miss@FA5', 'miss@FA10'))
    for e, vl, va, m5, m10 in results:
        print('%6d %10.4f %18.4f %10.2f %11.2f' % (e, vl, va, m5, m10))
    best_kpi = min(results, key=lambda r: r[3])
    best_auroc = max(results, key=lambda r: r[2])
    best_loss = min(results, key=lambda r: r[1])
    print()
    print(f'best by miss@FA5        : epoch {best_kpi[0]}  ({best_kpi[3]:.2f})')
    print(f'best by val_detect_auroc: epoch {best_auroc[0]}  -> miss@FA5 {best_auroc[3]:.2f}   <- what the trainer selects')
    print(f'best by val_loss        : epoch {best_loss[0]}  -> miss@FA5 {best_loss[3]:.2f}')
    cost = best_auroc[3] - best_kpi[3]
    print(f'\nmonitor cost: {cost:+.2f} miss@FA5 vs picking the true best epoch')


if __name__ == '__main__':
    main()
