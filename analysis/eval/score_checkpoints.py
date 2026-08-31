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
import re
import sys

import numpy as np
import pytorch_lightning as pl
import torch

from mvc.core.config import load_hyperparameters
from mvc.core.datasets import build_val_only, val_dataloader
from mvc.core.metrics import miss_at_fa
from mvc.core.artifact_paths import find_artifact, sanitise
from mvc.core.evaluation import run_confusion_and_threshold_sweep
from mvc.core.lit_classifier import Classifier


def build_val_loader(config_json):
    """Rebuild the run's validation split -- now by IMPORT, not by mirroring.

    This function used to be a hand-copy of the trainer's dataset block, kept in step by
    a comment naming line numbers in another file. Those line numbers went stale, which
    is what a comment-as-contract always does. The chain lives in Datasets.build_val_only
    now, so the trainer and the scorers cannot disagree about what the validation set is.

    test_dataset_split.py asserts the shared chain reproduces this function's previous
    output tile for tile -- same frames, same labels, same order -- on every distinct
    dataset chain in the config corpus.
    """
    split = build_val_only(config_json)
    return split.dataset, val_dataloader(config_json, split)


# The KPI lives in Metrics.py -- this file used to carry its own np.interp copy.


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
    # Exact match, not substring: 'class_WeldingClassAClean' contains "clean" and sorts
    # BEFORE 'class_clean' (ASCII: 'W' < 'c'), so a substring test silently picks a
    # 2-tile vestigial class as the clean class -- and defect_mass = 1 - P(that) makes
    # every tile score as a defect. Audited: no run in this campaign had both classes
    # present, so no recorded number is affected. Reported by the dev box, who hit it.
    cleanClassID = next((i for i, c in enumerate(class_names)
                         if c.lower() in ('class_clean', 'clean')), None)
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
        # Single source of truth (LitClassifier.Classifier.from_config). This block used
        # to omit custom_channels/custom_res_blocks, so a CustomCNN was rebuilt at the
        # default width. The extra kwargs from_config supplies are the training-only
        # augmentation knobs (noise/gain_jitter/polar_flip/polar_rot/channel_jitter),
        # every one of which is gated on self.training -- inert here, where the model is
        # in eval(). Equivalence on all 167 real configs is asserted by
        # test_classifier_from_config.py, which is what makes this conversion a no-op.
        model = Classifier.from_config(config_json, num_classes=len(class_names),
                                      clean_class=cleanClassID)
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
        # Resolve rather than assume the cwd: Evaluation.py now writes into
        # experiments/<campaign>/<run>/ via artifact_paths.out_path, so building the path
        # by hand here would look for the curve where it is no longer written. This is the
        # same fault that broke eval_coverage -- a reader left pointing at the old layout
        # after the writer moved.
        #
        # sanitise(), not the raw model name: out_path() (what the write above actually
        # used, via run_confusion_and_threshold_sweep) sanitises `timm/x` -> `timm_x`
        # before writing. A `timm/*` model name here has a bare slash, which
        # find_artifact() then treats as an unwanted directory separator and never
        # finds -- this crashed on the FIRST checkpoint for every timm/* model in the
        # traintest-split sweep (2026-08-26 to -31), silently dropping every later
        # checkpoint's score for 12 of 14 timm models (confirmed via the sweep logs).
        # eval_traintest_split.py's macro detect@FA5 was unaffected -- it picks its
        # checkpoint by its own glob, not by reading this file -- so only the aggregate
        # miss@FA5 column was ever missing.
        curve_name = f"{name}_{sanitise(config_json['model'])}_threshold_curve.json"
        curve = find_artifact(curve_name)
        if curve is None:
            raise FileNotFoundError(
                f'{curve_name} was not written where expected (checked the repo root and '
                f'experiments/); run_confusion_and_threshold_sweep may have failed')
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
