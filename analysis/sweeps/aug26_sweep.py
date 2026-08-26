#!/usr/bin/env python3
"""Generate the Aug26 modifier sweep configs from the anchor config.

Why this sweep exists: the legacy modifier sweep (`mx*`, PLAN.md) ran on a frame-disjoint
split with recording leakage, and the dev box's recording-disjoint ablation reversed DoLP's
point estimate. Neither box has yet measured these modifiers on the CORRECT Aug26 label
space -- the dev box's arms were run on the broken 18-class default. So the question is
genuinely open and has to be re-asked under the campaign standard.

Two deliberate departures from the legacy sweep, both fixing flaws it had:

  1. THE BASELINE IS 4-CHANNEL RAW, not 4ch+DoLP. The legacy baseline had DoLP on, so its
     deltas were the cost of REMOVING a modifier while the dev box reports the cost of
     ADDING one -- opposite signs, which is exactly what made the two boxes' DoLP numbers
     look like they agreed when they disagreed. Same convention on both boxes from here.

  2. THE MONO ARM IS PURE MONOCHROME. The legacy `mono` arm kept DoLP on, so it measured
     "mono + DoLP vs pol + DoLP" rather than the polarization camera's actual worth.

Arms (4), chosen for decision-relevance, not coverage:

  base     4ch raw                 the reference
  dolp     4ch + DoLP              the open disagreement between the two boxes
  mono     4ch, all = their mean   the deliverable justification for the polarization
                                   camera; the only modifier that has replicated across
                                   both splits, so re-measuring it on the correct label
                                   space is what makes it quotable. NOTE this is the mean
                                   replicated x4 -- same tensor shape and parameter count,
                                   zero polarimetric signal -- not a 1-channel model. The
                                   dev box's mono arm IS 1ch, so the two are not identical
                                   arms; this one isolates the polarimetry from the input
                                   width, theirs does not.
  stride2  patchify stride 4 -> 2  the campaign's largest accuracy finding (-1.49 on the
                                   leaky split) and the ONLY one never tested on an honest
                                   split. Costs 3.2x and cannot ship at step 16/18, so this
                                   asks whether the 1x1 collapse costs accuracy, not
                                   whether to deploy it.

3 seeds per arm because the honest split is NOISIER than the legacy one -- the dev box saw
per-arm sd from 0.49 to 3.23 there, against the legacy noise floor of 0.43. Three seeds may
still not resolve DoLP; that is a fact about the split, not a reason to run one seed.

Usage:  python aug26_sweep.py [--dry-run]
"""

import copy
import json
import os
import sys

BASE_CONFIG = 'anc_convnext_pico.json'
SEEDS = (42, 1337, 7)
LIMIT_TRAIN_BATCHES = 30000   # ~31 min/run; the anchor did 121,322 steps in ~2h07

# Result keys the trainer writes back into the config. Copying them into a fresh config
# would leave a run advertising another run's confusion matrix until it finishes.
RESULT_KEYS = ('confusion_matrix', 'classes', 'classes_int', 'gate', 'model_md5',
               'best_threshold_balanced', 'best_threshold_kpi', 'best_threshold_deployment',
               'training_started', 'training_finished', 'training_seconds', 'fit_seconds')

ARMS = {
    'base':    {},
    'dolp':    {'DoLP': True},
    'mono':    {'monochrome': True},
    'stride2': {'timm_stem_stride': 2},
}


def build(arm, seed, base):
    cfg = copy.deepcopy(base)
    for k in RESULT_KEYS:
        cfg.pop(k, None)

    name = f's26{arm}{seed}'
    cfg['name'] = name
    cfg['limit_train_batches'] = LIMIT_TRAIN_BATCHES
    cfg['checkpoint_dir'] = f'datasets/mix_ckpts/{name}_convnext_pico'
    cfg['checkpoint_save_top_k'] = 1

    h = cfg['hparams']
    h['seed'] = seed
    h['training_epochs'] = 1
    # Every arm starts from 4ch raw and turns on exactly one thing, so an arm can never
    # inherit a modifier from the anchor config by accident.
    h['DoLP'] = False
    h['AoLP'] = False
    h['unpolarized'] = False
    h['monochrome'] = False
    h.pop('timm_stem_stride', None)
    h.update(ARMS[arm])

    return name, cfg


def main():
    dry = '--dry-run' in sys.argv
    if not os.path.exists(BASE_CONFIG):
        sys.exit(f'{BASE_CONFIG} not found -- the anchor must exist first')
    base = json.load(open(BASE_CONFIG))

    assert base['dataloader'].get('exclude_frames'), \
        'the anchor config has no exclude_frames; the coverage set would be IN training'

    written = []
    for arm in ARMS:
        for seed in SEEDS:
            name, cfg = build(arm, seed, base)
            path = f'{name}_convnext_pico.json'
            if not dry:
                json.dump(cfg, open(path, 'w'), indent=1)
            written.append((path, arm, seed, cfg['hparams']))

    print(f"{'config':34s} {'arm':9s} {'seed':>5s}  modifiers")
    print('-' * 78)
    for path, arm, seed, h in written:
        mods = {k: v for k, v in h.items()
                if k in ('DoLP', 'AoLP', 'unpolarized', 'monochrome', 'timm_stem_stride') and v}
        print(f'{path:34s} {arm:9s} {seed:5d}  {mods or "4ch raw"}')
    print(f'\n{len(written)} configs {"(dry run, nothing written)" if dry else "written"}; '
          f'{LIMIT_TRAIN_BATCHES:,} steps each, 3 arms at ~31 min + stride2 at ~100 min '
          f'=> ~10 h training.')


if __name__ == '__main__':
    main()
