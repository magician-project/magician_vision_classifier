#!/usr/bin/env python3
"""Per-class detection@FA on the in-distribution train/test split (traintest_sweep.py).

WHY THIS EXISTS, SEPARATELY FROM score_checkpoints.py's miss@FA5. score_checkpoints.py's
miss@FA5 is an AGGREGATE, incidence-weighted number -- dominated by whichever class has the
most tiles in the split. That incidence differs a lot between validation sets (87.6% welding
on the factory val, 53.4% welding in this random split's training-distribution mix), so a
smoke-test comparison of the two raw miss@FA5 numbers showed almost no gap even though the
split IS leaky -- the class-mix difference was swamping the leakage effect on that number.
The coverage number this campaign wants to compare against (TIER_A macro in 21-8/24-8-report.md)
is a MACRO over per-class detection@FA5, not incidence-weighted, so the comparable number for
this campaign has to be computed the same way -- this script does that, mirroring
eval_coverage.py's per-class threshold-matching exactly, on this campaign's own held-out
Subset instead of the recording-disjoint coverage carve-out.

No tiers here (unlike eval_coverage.py): every class in this split is equally "leaky" by
design, there is no recording-disjoint/frame-disjoint distinction to draw within a random
tile-level split. Macro is over every non-clean class with support.

Usage:
    python eval_traintest_split.py <config.json> [checkpoint.ckpt]
"""

import glob
import json
import os
import re
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

from mvc.core.config import load_hyperparameters
from mvc.core.datasets import build_val_only, clean_class_index
from mvc.core.metrics import detection_at_fa, fa_threshold
from mvc.core.artifact_paths import out_path
from mvc.core.lit_classifier import Classifier


def pick_best_checkpoint(cfg):
    """Monitored-best checkpoint -- same rule eval_coverage.py applies, so the two
    campaigns' per-class numbers are scored at the same kind of checkpoint."""
    cks = sorted(glob.glob(os.path.join(cfg['checkpoint_dir'], '*.ckpt')))
    if not cks:
        sys.exit(f'no checkpoints in {cfg["checkpoint_dir"]}')
    mon = cfg.get('checkpoint_monitor', 'val_detect_auroc')
    best = 'max' if cfg.get('checkpoint_mode', 'max') == 'max' else 'min'
    scored = [(float(m.group(1)), c) for c in cks
              if (m := re.search(rf'{re.escape(mon)}=([0-9]+\.[0-9]+)', c))]
    if scored:
        ckpt = (max if best == 'max' else min)(scored)[1]
        print(f'[traintest] {len(cks)} checkpoints; picking best {mon} '
              f'({best}) -> {os.path.basename(ckpt)}')
        return ckpt
    ckpt = cks[-1]
    print(f'[traintest] WARNING: no "{mon}=" in checkpoint names; falling back to the last '
          f'one ({os.path.basename(ckpt)}), which may not be the best.')
    return ckpt


def main():
    cfg_path = sys.argv[1]
    cfg = load_hyperparameters(cfg_path)
    h = cfg['hparams']

    if not cfg['dataloader'].get('frozen_tile_split'):
        sys.exit('this config has no dataloader.frozen_tile_split -- wrong campaign for this '
                 'script; use eval_coverage.py or score_checkpoints.py instead')

    split = build_val_only(cfg)
    class_names = list(split.classes)
    cleanID = clean_class_index(class_names)
    if cleanID is None:
        sys.exit(f'no clean class among {class_names}')

    ckpt = sys.argv[2] if len(sys.argv) > 2 else pick_best_checkpoint(cfg)
    print(f'[traintest] scoring {os.path.basename(ckpt)}')

    model = Classifier.from_config(cfg, num_classes=len(class_names), clean_class=cleanID)
    model.load_state_dict(torch.load(ckpt, weights_only=False, map_location='cpu')['state_dict'])
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(dev).eval()

    loader = DataLoader(split.val, batch_size=h['batch_size'], shuffle=False, num_workers=8)
    n = len(class_names)
    support = np.zeros(n, np.int64)
    mass, truth = [], []
    with torch.no_grad():
        for bi, (x, y) in enumerate(loader):
            logits = model(x.to(dev))
            prob = torch.softmax(logits.float(), dim=1).cpu().numpy()
            y = y.numpy()
            mass.append(1.0 - prob[:, cleanID])
            truth.append(y)
            for t in y:
                support[t] += 1
            if bi % 50 == 0:
                print(f'  batch {bi}/{len(loader)}', end='\r')
    print()
    mass = np.concatenate(mass)
    truth = np.concatenate(truth)

    is_clean = truth == cleanID
    det_at = {}
    for fa in (0.05, 0.10):
        thr = fa_threshold(mass, is_clean, fa)
        det_at[fa] = (thr, {i: detection_at_fa(mass, is_clean, truth == i, fa)
                            for i in range(n) if support[i] and i != cleanID})
    print(f'[traintest] FA-matched thresholds from {int(is_clean.sum()):,} clean tiles: '
          + ', '.join(f'FA{int(f*100)} -> {det_at[f][0]:.4f}' for f in (0.05, 0.10)))

    print(f"\n{'class':30s} {'support':>9s} {'det@FA5':>8s} {'det@FA10':>9s}")
    print('-' * 60)
    rows = []
    for i, name in enumerate(class_names):
        if not support[i] or i == cleanID:
            continue
        d5 = det_at[0.05][1].get(i)
        d10 = det_at[0.10][1].get(i)
        rows.append((name, int(support[i]), d5, d10))
        print(f'{name:30s} {support[i]:9,d} '
              f'{"-" if d5 is None else f"{d5:8.2f}"} {"-" if d10 is None else f"{d10:9.2f}"}')

    macro5 = sum(r[2] for r in rows) / len(rows) if rows else None
    macro10 = sum(r[3] for r in rows) / len(rows) if rows else None
    print(f'\n  macro over {len(rows)} non-clean classes:')
    print(f'    detection@FA5  {macro5:6.2f}%   detection@FA10 {macro10:6.2f}%')
    print('\n  ⚠️  This split is random and tile-level -- leakage is expected. Compare this '
          'number against\n  the TIER_A coverage macro in 21-8/24-8-report.md as an '
          'in-distribution CEILING, not as an\n  apples-to-apples generalization estimate.')

    out = out_path(f"{cfg['name']}_{cfg['model']}", '_traintest_detect.json')
    json.dump({'checkpoint': os.path.basename(ckpt),
               'fa_thresholds': {f'FA{int(f*100)}': det_at[f][0] for f in (0.05, 0.10)},
               'macro_detect_at_fa5': macro5, 'macro_detect_at_fa10': macro10,
               'rows': [{'class': n, 'support': s, 'detect_at_fa5': d5, 'detect_at_fa10': d10}
                        for n, s, d5, d10 in rows]},
              open(out, 'w'), indent=1)
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
