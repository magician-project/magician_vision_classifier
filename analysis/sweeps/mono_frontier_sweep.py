#!/usr/bin/env python3
"""Polarized vs monochrome, on the 5 front models from knowledge/24-8-report.md §8.

PLAN.md's 2026-07-28 result (convnext_tiny, an older dataset/class scheme) found
polarization strictly dominates monochrome on every class and every operating point
(+1.94 balanced accuracy). That result is one architecture, one dataset generation ago.
This asks the same question on the CURRENT protocol (Aug26_78K, 4ch+DoLP, 10-class,
coverage carve-out) across the whole deployment front, not just one model, since a
modifier effect measured on one backbone does not necessarily transfer to another
(exactly the caution modifier_sweep.py's `mono` arm docstring raises).

FRONT = the 5 models in 24-8-report.md §8 ("front models"): convnext_pico (incumbent),
convnext_tiny (challenger), convnext_femto, repvgg_a0, squeezenet1_1.

SHORT BY DESIGN: reuses each model's existing polarized (pol) run as the baseline --
does not retrain it. Every front model already has a fully scored coverage table under
this exact protocol (seed 42, coverage carve-out, monitored-best checkpoint). Only the
monochrome twin is new per model: an EXACT copy of that model's pol config with
`hparams.monochrome=True` and nothing else changed -- same seed, same epoch budget, same
checkpoint_monitor -- so the only axis that moves is the polarization channel, same
methodology as PLAN.md's pol/mono A/B. 5 new training runs, not 10.

Usage:  python mono_frontier_sweep.py [--dry-run]
"""

import copy
import json
import sys

from mvc.core.artifact_paths import find_config_with_classes

RESULT_KEYS = ('confusion_matrix', 'classes', 'classes_int', 'gate', 'model_md5',
               'best_threshold_balanced', 'best_threshold_kpi', 'best_threshold_deployment',
               'training_started', 'training_finished', 'training_seconds', 'fit_seconds')

# (pol run name, model, new mono run name)
FRONT = [
    ('anc',       'convnext_pico',    'moncnxpico'),
    ('fzcnxtiny', 'convnext_tiny',    'moncnxtiny'),
    ('msfemto',   'convnext_femto',   'monfemto'),
    ('msrepvgg',  'repvgg_a0',        'monrepvgg'),
    ('fzsqueeze', 'squeezenet1_1',    'monsqueeze'),
]


def main():
    dry = '--dry-run' in sys.argv
    print(f'{"model":20s} {"pol run":12s} {"mono run":12s} {"config":40s}')
    for pol_run, model, mono_tag in FRONT:
        sfx = model.replace('/', '_')
        # find_config_with_classes(), not find_artifact(): the root copy of a tidied run's
        # config is the pre-training template (no `classes`), the enriched copy lives only
        # under experiments/ -- see the same fix in eval_vote_curve.py (2026-09-03).
        src = find_config_with_classes(f'{pol_run}_{sfx}.json') or f'{pol_run}_{sfx}.json'
        with open(src) as fh:
            cfg = json.load(fh)

        assert cfg['model'] == model, f'{src}: expected model {model}, got {cfg["model"]}'
        assert cfg['dataloader'].get('exclude_frames'), f'{src}: lost its coverage carve-out'
        assert len(cfg['classes']) == 10, f'{src}: not on the 10-class scheme'
        assert cfg['hparams'].get('monochrome') is False, \
            f'{src}: pol baseline already has monochrome set -- not a clean pol source'

        cfg = copy.deepcopy(cfg)
        for k in RESULT_KEYS:
            cfg.pop(k, None)
        cfg['hparams']['monochrome'] = True
        cfg['name'] = mono_tag
        cfg['checkpoint_dir'] = f'datasets/mix_ckpts/{mono_tag}_{sfx}'

        out = f'{mono_tag}_{sfx}.json'
        print(f'{model:20s} {pol_run:12s} {mono_tag:12s} {out:40s}')
        if not dry:
            with open(out, 'w') as fh:
                json.dump(cfg, fh, indent=2)

    print(f'\n{len(FRONT)} configs {"planned" if dry else "written"} -- monochrome twins '
          f'of the existing pol runs, everything else byte-identical to each pol config.')


if __name__ == '__main__':
    main()
