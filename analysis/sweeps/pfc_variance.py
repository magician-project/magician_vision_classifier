#!/usr/bin/env python3
"""Test whether `penalize_false_clean=4` collapses SEED VARIANCE, not whether it helps.

This is a variance experiment wearing the clothes of an ablation, and the distinction is
the whole point. The dev box swept `penalize_false_clean` and reported pfc=4 as an
inverted-U optimum: factory KPI held (10.50 -> 10.30), coverage +4 points, and seed
**sd 1.19 -> 0.10**. I told them to hold the sd claim loosely -- a 12x variance reduction
from n=3 (2 df) is mostly luck.

I now think that was the wrong call, for a reason that only became visible on 2026-08-12:

  * their augmentation leave-one-out, a completely different experiment, independently
    reports base sd **0.10 at n=3** on pfc=4;
  * my full-budget anchor at pfc=0.5 has sd **1.29**, and the stride-2 arm 1.63;
  * that reversal cost me a headline result -- stride 4->2 flipped sign between seeds.

Two independent tight sds at pfc=4 against two independent catastrophic ones at pfc=0.5 is
much better evidence than either alone. **If pfc=4 really collapses seed variance, it fixes
the measurement problem that makes every n=1 comparison in this campaign uninterpretable**
-- and it costs one config line.

WHAT THIS RUNS
--------------
Three seeds at pfc=4, full budget, everything else identical to the anchor. Compared
against the anchor's own three seeds at pfc=0.5. The quantity of interest is the WITHIN-ARM
sd, not the mean:

    pfc=0.5   sd 1.29 (n=2 so far, 3rd seed running)
    pfc=4.0   sd ?

An F-test on two sds from n=3 each (2 and 2 df) needs a ratio of ~19x for p<0.05, so even a
clean result here will be suggestive rather than conclusive -- but a ratio near 12x would
combine with the dev box's two independent observations into something worth acting on, and
a ratio near 1x kills the idea outright and saves everyone from chasing it.

WHY THE ORDER MATTERS: this gates how the 11-backbone sweep should be read. At pfc=0.5 the
sweep needs stage 1 (n=1, eliminate only) plus stage 2 (n=3 on survivors), ~84 h + ~60 h.
If pfc=4 collapses variance, single-seed comparisons become interpretable and stage 2 is
largely unnecessary. ~13 h spent here can save ~60 h there -- or confirm it must be spent.

Usage:  python pfc_variance.py [--dry-run]
"""

import copy
import json

from mvc.core.artifact_paths import find_artifact
import sys

BASE_CONFIG = 'anc_convnext_pico.json'
SEEDS = (42, 1337, 7)          # the same seeds as the anchor, so the pairing is exact
PFC = 4.0
EPOCHS = 2

RESULT_KEYS = ('confusion_matrix', 'classes', 'classes_int', 'gate', 'model_md5',
               'best_threshold_balanced', 'best_threshold_kpi', 'best_threshold_deployment')


def main():
    dry = '--dry-run' in sys.argv
    with open(find_artifact(BASE_CONFIG) or BASE_CONFIG) as fh:
        base = json.load(fh)

    assert base['dataloader'].get('exclude_frames'), 'anchor lost its coverage carve-out'
    assert len(base['classes']) == 10, 'anchor is not on the 10-class scheme'
    assert base['penalize_false_clean'] == 0.5, \
        f'anchor pfc is {base["penalize_false_clean"]}, expected the 0.5 baseline'

    for seed in SEEDS:
        cfg = copy.deepcopy(base)
        for k in RESULT_KEYS:
            cfg.pop(k, None)
        cfg['penalize_false_clean'] = PFC
        cfg['hparams']['seed'] = seed
        cfg['hparams']['training_epochs'] = EPOCHS
        name = f'pfc4s{seed}'
        cfg['name'] = name
        cfg['checkpoint_save_top_k'] = EPOCHS
        cfg['checkpoint_dir'] = f'datasets/mix_ckpts/{name}_{cfg["model"]}'

        out = f'{name}_{cfg["model"]}.json'
        print(f'{out:32s} pfc={PFC} seed={seed} epochs={EPOCHS}')
        if not dry:
            with open(out, 'w') as fh:
                json.dump(cfg, fh, indent=2)

    print(f'\n{len(SEEDS)} configs {"planned" if dry else "written"} — '
          f'compare within-arm sd against the anchor at pfc=0.5')


if __name__ == '__main__':
    main()
