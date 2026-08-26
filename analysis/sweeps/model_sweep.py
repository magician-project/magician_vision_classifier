#!/usr/bin/env python3
"""Generate the full-budget backbone sweep on Aug26_78K.

WHY THE EXISTING BACKBONE BENCH CANNOT ANSWER THIS
--------------------------------------------------
There is already a 12-backbone bench in `experiments/bench_backbones_tile48/` (the `tz*`
runs). Its THROUGHPUT half transfers -- it benched the architectures at tile 48, in_chans 5,
fp16, which is the deployment shape, and architecture speed does not depend on the dataset.
Its ACCURACY half does not transfer at all:

  * it ran on `train_nonaltinay_v2` + `val_altinay_granular` + `yongatek`, not Aug26_78K;
  * with `frame_disjoint_split: true` -- the split with recording leakage, which is exactly
    what reversed DoLP's point estimate when the dev box re-ran it honestly;
  * on a 12-class label space, not the corrected 10-class one;
  * for `training_epochs: 1`.

So the `tz` ranking (pico 7.51, nano 8.50, atto 8.92, femto 8.93 ... mobileone_s0 21.24) is
a weak prior on a different problem, and using it to eliminate candidates would import the
leakage it was measured under. Every candidate is re-measured here. The prior is used only
to ORDER the queue so that stopping early stops on the least promising runs.

ELIGIBILITY IS BY THROUGHPUT, AND IT EXCLUDES ALMOST NOTHING
-----------------------------------------------------------
Target is 23 Hz on an RTX 5090. The bench is an A6000; taking a 5090 at a conservative 1.6x
means ~14.4 Hz on the A6000 bench is the bar, and step size is an accepted lever (step 16 ->
4,588 tiles/frame, step 18 -> 3,630, step 24 -> 2,050). At step 16 every candidate except
`ghostnet_100` and `fastvit_t8` already clears it, and those two clear it at step 18. So
throughput does not narrow the field -- the field has to be narrowed on Aug26 accuracy,
which is what this sweep exists to measure.

THE STATISTICS DICTATE A TWO-STAGE DESIGN
-----------------------------------------
The full-budget seed noise measured on 2026-08-11 is ~1.8 miss@FA5 between two anchor seeds.
At that scale a one-seed ranking of 12 backbones is mostly noise: adjacent models would swap
places at random. But ELIMINATION is cheap -- a backbone 5+ points worse than the incumbent
is 3+ sd away and one seed settles it.

  STAGE 1 (this file): every candidate at seed 42, full budget. Screens out the clearly
                       worse. Does NOT rank the survivors -- an ordering among runs within
                       ~2 points of each other means nothing at n=1 and must not be quoted.
  STAGE 2 (later):     seeds 1337 and 7 on the survivors only, giving the same n=3 paired
                       comparison the stride-2 question is getting.

`convnext_pico` is not regenerated: the running seed-replicate queue is already producing it
at three seeds, and it is the incumbent every candidate is compared against.

Everything except `model` and the output names is copied from `anc_convnext_pico.json`, so
each candidate differs from the anchor in the BACKBONE ALONE -- same 4ch+DoLP input, same
10-class scheme, same coverage carve-out, same 2-epoch budget, same `pfc=0.5`.

Usage:  python model_sweep.py [--dry-run]
"""

import copy
import json

from mvc.core.artifact_paths import find_artifact
import sys

BASE_CONFIG = 'anc_convnext_pico.json'
SEED = 42
EPOCHS = 2

# (short tag, model, A6000 Hz at step 16 as deployed, legacy tz miss@FA5, note)
# The tag keeps the artifact names readable: the writers emit `{name}_{model}`, so a name of
# `msconvnext_femto` would produce `msconvnext_femto_convnext_femto_confusion.json`.
# Ordered most- to least-promising: the queue runs in this order so that stopping it early
# stops on the candidates with the weakest prior, not the strongest.
CANDIDATES = [
    ('femto',  'convnext_femto',         27.8,  8.93, 'closest sibling to the incumbent, +23% headroom'),
    ('repvgg', 'repvgg_a0',              48.3, 10.20, 'FUSED at deploy: 2.1x the incumbent, the only '
                                                      'candidate with room for stride-2'),
    ('nano',   'convnext_nano',          15.8,  8.50, 'best legacy accuracy of the field, least headroom'),
    ('atto',   'convnext_atto',          16.5,  8.92, 'smallest ConvNeXt above the bar'),
    ('mnv4',   'mobilenetv4_conv_small', 21.9, 12.10, 'modern mobile baseline'),
    ('ghost',  'ghostnet_100',           12.5,  9.42, 'needs step 18+ to clear the bar'),
    ('edgex',  'edgenext_xx_small',      16.3, 11.27, '1.16M params, hybrid conv/attn'),
    ('mone1',  'mobileone_s1',           26.8, 11.34, 'FUSED at deploy; trains on the slow multi-branch graph'),
    ('lcnet',  'lcnet_050',              43.2, 14.55, 'throughput champion, 0.62M params; legacy accuracy poor'),
    ('fastvit','fastvit_t8',             11.0, 15.83, 'slowest candidate and weak legacy prior'),
    ('mone0',  'mobileone_s0',           38.2, 21.24, 'weakest legacy prior in the field; ~20 h to train unfused'),
]

# Reparameterizable backbones train as a multi-branch graph and MUST be fused before the
# throughput above is realised. The KPI is unaffected (the fused model is equivalent), but a
# deployment that forgets to fuse gets 5.0 Hz instead of 38.2.
REPARAM = {'repvgg_a0', 'mobileone_s0', 'mobileone_s1'}

RESULT_KEYS = ('confusion_matrix', 'classes', 'classes_int', 'gate', 'model_md5',
               'best_threshold_balanced', 'best_threshold_kpi', 'best_threshold_deployment',
               'training_started', 'training_finished', 'training_seconds', 'fit_seconds')


def main():
    dry = '--dry-run' in sys.argv
    with open(find_artifact(BASE_CONFIG) or BASE_CONFIG) as fh:
        base = json.load(fh)

    assert base['dataloader'].get('exclude_frames'), 'anchor lost its coverage carve-out'
    assert len(base['classes']) == 10, 'anchor is not on the 10-class scheme'

    print(f'{"model":24s} {"Hz@16":>7s} {"tz prior":>9s}  config')
    for tag, model, hz, prior, _note in CANDIDATES:
        cfg = copy.deepcopy(base)
        for k in RESULT_KEYS:
            cfg.pop(k, None)

        cfg['model'] = model
        cfg['hparams']['seed'] = SEED
        cfg['hparams']['training_epochs'] = EPOCHS
        cfg['hparams'].pop('timm_stem_stride', None)   # stride is a separate axis
        name = f'ms{tag}'
        cfg['name'] = name
        cfg['checkpoint_save_top_k'] = EPOCHS
        cfg['checkpoint_dir'] = f'datasets/mix_ckpts/{name}_{model}'

        out = f'{name}_{model}.json'
        flag = ' [REPARAM: fuse at deploy]' if model in REPARAM else ''
        print(f'{model:24s} {hz:7.1f} {prior:9.2f}  {out}{flag}')
        if not dry:
            with open(out, 'w') as fh:
                json.dump(cfg, fh, indent=2)

    print(f'\n{len(CANDIDATES)} configs {"planned" if dry else "written"}, seed {SEED}, '
          f'{EPOCHS} epochs each')
    print('incumbent convnext_pico is NOT regenerated -- the seed-replicate queue is '
          'already producing it at 3 seeds')


if __name__ == '__main__':
    main()
