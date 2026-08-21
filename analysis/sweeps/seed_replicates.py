#!/usr/bin/env python3
"""Generate the full-budget seed replicates for the anchor and the stride-2 arm.

WHY THIS RUNS BEFORE ANY FURTHER ARCHITECTURE WORK
--------------------------------------------------
Every full-budget number in this campaign is n=1. The anchor is one seed, the stride-2 run
is one seed, and the 9.24 -> 7.15 miss@FA5 gap between them is a single paired difference.
The 30k screen cannot arbitrate it: its Aug26 noise floor is sd 1.83 miss@FA5 (pooled
within-arm, 4 arms x 3 seeds, PLAN.md), against a legacy 0.43 -- a dead instrument at that
budget. So the only way to know whether -2.09 is a result or a coin flip is to pay for the
seeds at full budget.

Three answers come out of this one queue:

  1. STRIDE-2, ESTABLISHED OR WITHDRAWN, at n=3 paired. Same seeds on both arms, so the
     comparison is paired and the seed-to-seed component cancels.
  2. THE FULL-BUDGET NOISE FLOOR. Currently unknown at any budget except 30k. Every claim
     of the form "X beats Y at full budget" in this campaign is unfalsifiable without it.
  3. THE PER-CLASS COVERAGE NOISE, which is what actually gates shipping. Stride-2 posted
     -1.37 detection on PositiveDentClassB -- its sole per-class regression -- and the ship
     rule is per-class-on-detection. Without a per-class sd at full budget there is no way
     to say whether -1.37 trips it or is indistinguishable from zero.

BUDGET: 2 EPOCHS, NOT THE FULL 6/3
----------------------------------
Both existing full trains peak at epoch 1 on every metric and fall hard at epoch 2 (anchor
9.24 -> 11.53, stride-2 7.15 -> 14.43). Two epochs therefore covers the operating point that
both n=1 runs actually selected, with checkpoint_monitor=val_detect_auroc choosing between
epoch 0 and epoch 1 exactly as it did for seed 42. Paying for epochs 2-5 would buy only
worse checkpoints at ~14 h of extra GPU time. This does mean the comparison is
best-of-2-epochs, and if a replicate seed were to peak at epoch 2 it would be missed -- but
both existing runs collapse there rather than peak, in the same direction, so the risk is
small and the same for both arms.

Everything except `seed`, the epoch budget and the output names is copied verbatim from the
two parent configs, so the stride-2 arm still differs from the anchor in the stem stride
ALONE (DoLP stays on in both -- the anchor is a 5-channel run).

Usage:  python seed_replicates.py [--dry-run]
"""

import copy
import json

from mvc.core.artifact_paths import find_artifact
import sys

# (parent config, name prefix) -- the two arms being replicated.
PARENTS = (
    ('anc_convnext_pico.json',   'anc'),
    ('ancs2_convnext_pico.json', 'ancs2'),
)
SEEDS = (1337, 7)        # seed 42 already exists at full budget for both arms
EPOCHS = 2               # see the budget note above
SAVE_TOP_K = 2           # keep both epochs so per-epoch scoring stays possible

# Result keys the trainer writes back into the config after a run. Copying them into a
# fresh config would leave a run advertising the parent's confusion matrix until it
# finishes, and eval_coverage.py reads `classes` from the config.
RESULT_KEYS = ('confusion_matrix', 'classes', 'classes_int', 'gate', 'model_md5',
               'best_threshold_balanced', 'best_threshold_kpi', 'best_threshold_deployment')


def main():
    dry = '--dry-run' in sys.argv
    written = []

    for parent_path, prefix in PARENTS:
        with open(find_artifact(parent_path) or parent_path) as fh:
            parent = json.load(fh)

        # The carve-out is what makes the coverage validation disjoint from train. A
        # replicate that lost it would be silently training on its own tripwire.
        assert parent['dataloader'].get('exclude_frames'), \
            f'{parent_path}: no exclude_frames -- coverage set is not carved out'
        assert len(parent['classes']) == 10, \
            f'{parent_path}: {len(parent["classes"])} classes, expected the 10-class scheme'

        for seed in SEEDS:
            cfg = copy.deepcopy(parent)
            for k in RESULT_KEYS:
                cfg.pop(k, None)

            cfg['hparams']['seed'] = seed
            cfg['hparams']['training_epochs'] = EPOCHS
            # dataloader.seed is deliberately NOT varied: the factory val is an explicit
            # directory and the coverage set is a fixed frame list, so this seed only
            # affects the unused validation_split. Holding it fixed keeps the data
            # identical across replicates and confines the variation to training
            # stochasticity -- init, sampler order, augmentation.

            name = f'{prefix}{seed}'
            cfg['name'] = name
            cfg['checkpoint_save_top_k'] = SAVE_TOP_K
            cfg['checkpoint_dir'] = f'datasets/mix_ckpts/{name}_{cfg["model"]}'

            out = f'{name}_{cfg["model"]}.json'
            stride = cfg['hparams'].get('timm_stem_stride', 'default(4)')
            print(f'{out:34s} seed={seed:<5d} stride={stride} epochs={EPOCHS} '
                  f'DoLP={cfg["hparams"]["DoLP"]}')
            if not dry:
                with open(out, 'w') as fh:
                    json.dump(cfg, fh, indent=2)
            written.append(out)

    print(f'\n{len(written)} configs {"planned" if dry else "written"}')


if __name__ == '__main__':
    main()
