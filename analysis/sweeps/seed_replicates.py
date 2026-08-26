#!/usr/bin/env python3
"""Generate the seed replicates for the seed-sweep experiment (Experiment B).

See EXPERIMENT-seed-sweep-and-step-curve.md Sec.5 for the full context. Every one of the
47-model campaign's coverage deltas is n=1, and the campaign's base rate for single-seed
findings surviving replication is 0-for-4. This generates seed 1337 and seed 7 replicates
(seed 42 already exists) for the four challengers the step curve (Experiment A, run
2026-08-21) did not rule out, so each gets n=3 before the campaign calls a winner:

    convnext_tiny      +3.51 coverage -- the recommendation itself
    efficientnet_b0    +1.22
    regnet_y_800mf     +0.80 -- marginal by design, the arm this sweep exists to settle
    convnext_femto     -1.21 -- the density-arm control

The incumbent `convnext_pico` already has n=3 (anc, anc1337, anc7 -- the source of the
campaign's coverage sd 0.43) and is not regenerated here.

BUDGET: 2 EPOCHS
-----------------
Matches the whole 47-model campaign, and matches what all four parent configs already ran
at (seed 42, epochs=2) -- see the "Config paths matter" table in the handoff for why these
specific parent paths and not same-named copies elsewhere in the repo.

Everything except `seed` and the output/checkpoint names is copied verbatim from each
parent config, so a replicate differs from its own seed-42 run in the seed alone.

Usage:  python seed_replicates.py [--dry-run]
"""

import copy
import json

from mvc.core.artifact_paths import find_artifact
import sys

# (parent config, name prefix) -- the four challengers being replicated. Full relative
# paths, not bare basenames: `tiny`/`effb0`/`regy800` all have a same-named but WRONG copy
# under experiments/configs_runs/ (classes=0, no `classes` list) elsewhere in the repo, and
# find_artifact resolves an exact existing path immediately without touching that ambiguous
# basename index -- so the exact path is what keeps this from silently grabbing the wrong
# file. Verified 2026-08-21: all four resolve, all carry classes=10, seed=42, epochs=2.
PARENTS = (
    ('experiments/zoo_sweep_full/fzcnxtiny/fzcnxtiny_convnext_tiny.json', 'fzcnxtiny'),
    ('experiments/zoo_sweep_full/fzeffb0/fzeffb0_efficientnet_b0.json',   'fzeffb0'),
    ('experiments/zoo_sweep_full/fzregy800/fzregy800_regnet_y_800mf.json', 'fzregy800'),
    ('configs/msfemto_convnext_femto.json',                                'msfemto'),
)
SEEDS = (1337, 7)        # seed 42 already exists at full budget for all four
EPOCHS = 2               # matches what all four parents already ran at
SAVE_TOP_K = 2           # keep both epochs so per-epoch scoring stays possible

# Result keys the trainer writes back into the config after a run. Copying them into a
# fresh config would leave a run advertising the parent's confusion matrix until it
# finishes, and eval_coverage.py reads `classes` from the config.
RESULT_KEYS = ('confusion_matrix', 'classes', 'classes_int', 'gate', 'model_md5',
               'best_threshold_balanced', 'best_threshold_kpi', 'best_threshold_deployment',
               'training_started', 'training_finished', 'training_seconds', 'fit_seconds')


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
