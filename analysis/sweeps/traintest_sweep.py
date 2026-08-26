#!/usr/bin/env python3
"""Generate the 47-model queue for the in-distribution train/test split campaign.

Same 47 architectures that appear in `21-8-report.md`/`24-8-report.md`'s appendix: the 35
completed `full_zoo_sweep.TAGS` architectures (imported directly, not re-typed, so the lists
cannot drift -- `swin_v2_t` and `resnet18_fullres` are excluded, matching the appendix: they
never finished the original sweep and are below the throughput gate regardless) + the 11
`model_sweep.CANDIDATES` architectures + `convnext_pico` itself.

`convnext_pico` needs a FRESH run here, unlike the other two campaigns -- there is no
existing checkpoint trained without the factory val / coverage carve-out, so the incumbent
baseline for this campaign is new, not reused from `anc_convnext_pico`.

Everything except `model`, the validation mechanism, and the output names is copied from
`anc_convnext_pico.json` verbatim, so every run in this queue differs from the anchor in the
backbone alone -- same 4ch+DoLP input, same 10-class scheme, same 2-epoch budget, same seed.
The validation mechanism is swapped: `validation_dataset` and `dataloader.exclude_frames`
(factory val + coverage carve-out) are removed, and `dataloader.frozen_tile_split` (the
random, leaky, in-distribution split from `build_tile_split_val.py`) is set instead.

Usage:  python traintest_sweep.py [--dry-run]
"""

import copy
import json
import sys

from mvc.core.artifact_paths import find_artifact

BASE_CONFIG = 'anc_convnext_pico.json'
SEED = 42
EPOCHS = 2
TILE_SPLIT = 'experiments/configs_frozen/traintest_tile_split_frozen.json'

# Architectures that never finished the original full_zoo_sweep -- excluded here too, same
# as they're excluded from the 47-row appendix in 21-8-report.md/24-8-report.md.
EXCLUDE_UNFINISHED = {'swin_v2_t', 'resnet18_fullres'}

RESULT_KEYS = ('confusion_matrix', 'classes', 'classes_int', 'gate', 'model_md5',
               'best_threshold_balanced', 'best_threshold_kpi', 'best_threshold_deployment',
               'training_started', 'training_finished', 'training_seconds', 'fit_seconds')


def model_list():
    from analysis.sweeps.full_zoo_sweep import TAGS as ZOO_TAGS
    from analysis.sweeps.model_sweep import CANDIDATES as MS_CANDIDATES

    models = [('cnxpico', 'convnext_pico')]
    models += [(tag, model) for model, tag in ZOO_TAGS.items()
               if model not in EXCLUDE_UNFINISHED]
    models += [(tag, model) for tag, model, *_ in MS_CANDIDATES]
    return models


def main():
    dry = '--dry-run' in sys.argv
    with open(find_artifact(BASE_CONFIG) or BASE_CONFIG) as fh:
        base = json.load(fh)

    assert base['dataloader'].get('exclude_frames'), 'anchor lost its coverage carve-out'
    assert len(base['classes']) == 10, 'anchor is not on the 10-class scheme'
    assert find_artifact(TILE_SPLIT), \
        f'{TILE_SPLIT} not found -- run build_tile_split_val.py first'

    models = model_list()
    written = []
    print(f'{"model":40s} {"tag":10s} {"config":50s}')
    for tag, model in models:
        cfg = copy.deepcopy(base)
        for k in RESULT_KEYS:
            cfg.pop(k, None)

        cfg['model'] = model
        cfg.pop('validation_dataset', None)
        cfg['dataloader'].pop('exclude_frames', None)
        cfg['dataloader']['frozen_tile_split'] = TILE_SPLIT
        cfg['hparams']['seed'] = SEED
        cfg['hparams']['training_epochs'] = EPOCHS
        cfg['hparams'].pop('timm_stem_stride', None)

        name = f'tt{tag}'
        cfg['name'] = name
        cfg['checkpoint_save_top_k'] = EPOCHS
        # `timm/x` is not a legal path component.
        cfg['checkpoint_dir'] = f'datasets/mix_ckpts/{name}_{model.replace("/", "_")}'

        out = f'{name}_{model.replace("/", "_")}.json'
        print(f'{model:40s} {tag:10s} {out:50s}')
        if not dry:
            with open(out, 'w') as fh:
                json.dump(cfg, fh, indent=2)
        written.append(out)

    print(f'\n{len(written)} configs {"planned" if dry else "written"} · seed {SEED} · '
          f'{EPOCHS} epochs · validation: {TILE_SPLIT}')


if __name__ == '__main__':
    main()
