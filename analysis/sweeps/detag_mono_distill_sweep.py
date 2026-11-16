#!/usr/bin/env python3
"""The monochrome twin of the DEPLOYED DeTaG, teacher included.

`kbachhybpenscjensemaavg1ep` -- what `recommended_configuration.json` ships and
`scripts/downloadKBach.sh` fetches -- is not a run of its own. Its config carries
`weights: "ema"`, `ema_source_run: "kbachhybpenscjensema1ep"` and
`finalized_from: .../epoch=0-step=121322-averaged.ckpt`: it is the EMA shadow weights of a
single `kbachhybpenscjensema1ep` run, written by `LitClassifier.save_ema_checkpoint` at
epoch end into `<checkpoint_dir>_ema/`. So its twin is one student run, not an ensemble.

The student distills from `convnext_tiny` teachers (`distill_weight` 1.0, T=2.0). Those
teachers are POLARIZED. A mono student taught by pol teachers still receives
polarization-derived knowledge through the soft labels, which would UNDERSTATE what the
camera is worth -- so this flips `monochrome` everywhere, teacher included.

    pol  (reference, trained on the kbach box, scored from its shipped artifacts)
        kbachcovtiny1ep + s7 + s1337   ->  kbachhybpenscjensema1ep  ->  EMA  = deployed
    mono (this campaign)
        moncnxtiny (epoch 0)           ->  monkbachhybpenscjensema1ep -> EMA

THE ONE ASYMMETRY, STATED UP FRONT
----------------------------------
The pol student distills from THREE teachers (seeds 42/7/1337), whose softened
distributions are averaged into one target. The mono side has one, because one is what
exists: `moncnxtiny` is the only monochrome `convnext_tiny` in the campaign (the mono
frontier sweep, 2026-08-25, built mono twins at seed 42 only).

That is not free. `lit_classifier.py`'s own distillation note records the 3-seed average
beating every single tiny it was built from -- 5.02 vs 6.53 miss@FA5, 81.37 vs 78.54
coverage. So the mono arm carries a weaker teacher for a reason unrelated to polarization,
and the measured pol-vs-mono gap is an UPPER BOUND on what the camera contributes. Closing
it means training `moncnxtiny` at seeds 7 and 1337 (2 x convnext_tiny) and re-running the
student with all three; `--teachers` writes those two configs for exactly that.

`moncnxtiny` is otherwise a clean analogue: monochrome `convnext_tiny`, seed 42, same
Aug26_78K data, same coverage carve-out, same 10-class scheme, same tile/batch -- and its
epoch-0 checkpoint is `step=121322`, the identical step count to the 1-epoch kbach pol
teachers. It is also the monitored-best of its two checkpoints (val_detect_auroc 0.9792 at
epoch 0 vs 0.9720 at epoch 1), so the standard rule picks the 1-epoch one on its own.

Dataloader settings stay as the kbach config has them (`cacheAllDataToRAM` False,
`num_workers` 6) rather than this box's faster cached path: the pol reference was trained
that way, and a 7.1M-tile RAM preload is ~66 GB, which this box does not reliably have
free. The comparison is cross-box regardless (different machine, different augmentation
RNG realisation) -- a seed-level difference, stated in the report rather than papered over.

Usage:
    python -m analysis.sweeps.detag_mono_distill_sweep --student  [--dry-run]
    python -m analysis.sweeps.detag_mono_distill_sweep --teachers [--dry-run]   # seeds 7/1337
"""

import copy
import glob
import json
import os
import re
import sys

# The enrichment a scored run's config picks up; a fresh training config must not carry it.
from .mono_frontier_sweep import RESULT_KEYS

LOCAL_TRAIN = '/storage/ammarkov/magician_datasets/Aug26_78K/train/'
LOCAL_VAL = '/storage/ammarkov/magician_datasets/Aug26_78K/val/'

# The mono teacher(s) the student distills from, as run dirs under datasets/mix_ckpts/.
TEACHER_RUNS = ['moncnxtiny_convnext_tiny']

# Optional extra mono teacher seeds, to match the pol side's 3-teacher ensemble.
# (source config in configs/, new mono run name, model) -- seeds live in hparams.seed.
EXTRA_TEACHERS = [
    ('configs/kbachcovtiny1eps7.json',    'moncnxtiny7',    'convnext_tiny'),
    ('configs/kbachcovtiny1eps1337.json', 'moncnxtiny1337', 'convnext_tiny'),
]

STUDENT_SRC = 'configs/kbachhybpenscjensema1ep.json'
STUDENT = 'monkbachhybpenscjensema1ep'
STUDENT_MODEL = 'hier_convnext_pico'


def localise(cfg, name, model):
    """Strip run results, point at this box's dataset and checkpoint dir, go monochrome."""
    cfg = copy.deepcopy(cfg)
    for k in RESULT_KEYS:
        cfg.pop(k, None)
    assert cfg['model'] == model, f'expected model {model}, got {cfg["model"]}'
    assert cfg['dataloader'].get('exclude_frames'), 'lost its coverage carve-out'
    assert cfg['hparams'].get('monochrome') is False, \
        'source is already monochrome -- not a clean pol source'

    cfg['hparams']['monochrome'] = True
    cfg['training_dataset'] = LOCAL_TRAIN
    cfg['validation_dataset'] = LOCAL_VAL
    cfg['name'] = name
    cfg['wandb']['name'] = name
    cfg['checkpoint_dir'] = f'datasets/mix_ckpts/{name}_{model}'
    return cfg


def best_checkpoint(ckpt_dir, monitor='val_detect_auroc'):
    """The monitored-best checkpoint in *ckpt_dir* -- eval_coverage.py's rule, not the last
    one, which is routinely the worst."""
    scored = [(float(m.group(1)), c)
              for c in sorted(glob.glob(os.path.join(ckpt_dir, '*.ckpt')))
              if (m := re.search(rf'{re.escape(monitor)}=([0-9]+\.[0-9]+)', c))]
    return max(scored)[1] if scored else None


def write(path, cfg, dry):
    print(f'  {"would write" if dry else "wrote"} {path}')
    if not dry:
        with open(path, 'w') as fh:
            json.dump(cfg, fh, indent=2)


def do_teachers(dry):
    print('Extra monochrome convnext_tiny teachers, to match the pol side\'s 3-seed '
          'ensemble:')
    for src, name, model in EXTRA_TEACHERS:
        with open(src) as fh:
            cfg = localise(json.load(fh), name, model)
        assert not cfg.get('distill_teacher'), f'{src}: a teacher must not itself distill'
        write(f'{name}_{model}.json', cfg, dry)
    print('\n  After training these, add them to TEACHER_RUNS and re-run --student.')


def do_student(dry):
    teachers, missing = [], []
    for run in TEACHER_RUNS:
        ck = best_checkpoint(f'datasets/mix_ckpts/{run}')
        (teachers.append(ck) if ck else missing.append(run))
    if missing:
        sys.exit(f'mono teacher(s) not found: {", ".join(missing)}')

    with open(STUDENT_SRC) as fh:
        cfg = localise(json.load(fh), STUDENT, STUDENT_MODEL)
    assert cfg.get('ema_decay'), f'{STUDENT_SRC}: lost ema_decay -- the deployed model IS ' \
                                 f'the EMA checkpoint of this run'
    assert cfg.get('distill_teacher'), f'{STUDENT_SRC}: lost distill_teacher'
    n_pol = len(cfg['distill_teacher'])
    cfg['distill_teacher'] = teachers

    print(f'mono student {STUDENT}, distilling from {len(teachers)} mono teacher(s):')
    for t in teachers:
        print(f'    {t}')
    if len(teachers) != n_pol:
        print(f'  !! the pol student it is compared against used {n_pol} teachers, not '
              f'{len(teachers)} -- the measured pol-vs-mono gap is an UPPER BOUND; see the '
              f'module docstring.')
    write(f'{STUDENT}_{STUDENT_MODEL}.json', cfg, dry)
    print(f'\n  ema_decay {cfg["ema_decay"]} -> score '
          f'datasets/mix_ckpts/{STUDENT}_{STUDENT_MODEL}_ema/, NOT the plain checkpoint '
          f'dir: the pol model it is compared against is an EMA checkpoint.')


def main():
    dry = '--dry-run' in sys.argv
    if '--teachers' in sys.argv:
        do_teachers(dry)
    elif '--student' in sys.argv:
        do_student(dry)
    else:
        sys.exit('usage: --student | --teachers  [--dry-run]')


if __name__ == '__main__':
    main()
