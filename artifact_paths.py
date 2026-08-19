#!/usr/bin/env python3
"""Locate a run artifact by name, whether it is still in the repo root or filed away.

The writers (`trainMagicianVisionClassifierTorch.py`, `score_checkpoints.py`,
`eval_coverage.py`) all emit `{name}_{model}_{kind}.{ext}` into the cwd, and the readers
all opened those by bare name. That coupling is why the root grew to 1,337 loose files:
tidying it would have broken every reporter.

`tidy_experiments.py` moves finished artifacts to `experiments/<campaign>/<run>/`. This
resolver is the other half -- readers ask for a NAME and get a path, so an artifact that
has been filed away is still found. Root is searched first, so a freshly written artifact
always wins over an older archived copy of the same name.

Nothing here creates or writes. Writers keep emitting to the cwd; the tidy script files
them afterwards, under an age guard so a running job's outputs are never moved mid-flight.
"""

import glob
import os
from functools import lru_cache

ROOT = os.path.dirname(os.path.abspath(__file__))
ARCHIVE = os.path.join(ROOT, 'experiments')


@lru_cache(maxsize=1)
def _index():
    """name -> path, for everything under experiments/. Built once per process."""
    idx = {}
    for p in glob.glob(os.path.join(ARCHIVE, '**', '*'), recursive=True):
        if os.path.isfile(p):
            idx.setdefault(os.path.basename(p), p)
    return idx


def _timm_slash_variants(name):
    """Artifact names for models configured as `timm/<x>`.

    The writers compose filenames as `{config.name}_{config.model}_{kind}.json`. When the
    model is `timm/tinynet_e` that embeds a SLASH, so `fztinye_timm/tinynet_e_..._.json`
    lands as a file `tinynet_e_..._.json` inside a directory `fztinye_timm/`. The runs are
    unaffected -- the trainer, scorer and coverage eval all wrote correctly, the path just
    has a directory separator in the middle of what was meant to be one filename.

    Callers ask for the sanitised form (`fztinye_timm_tinynet_e_...`), so map that back by
    turning the `_timm_` separator into `_timm/`. Cheaper and safer than renaming ~10
    models' artifacts or re-running them.
    """
    i = name.find('_timm_')
    if i == -1:
        return []
    return [name[:i] + '_timm/' + name[i + len('_timm_'):]]


def find_artifact(name):
    """Path to `name` in the cwd/root, else under experiments/, else None."""
    for cand in (name, *_timm_slash_variants(name)):
        if os.path.exists(cand):
            return cand
        p = os.path.join(ROOT, cand)
        if os.path.exists(p):
            return p
    return _index().get(os.path.basename(name))


def exists(name):
    return find_artifact(name) is not None


# ---------------------------------------------------------------------------------
# WRITER side: where a run's artifacts should be created in the first place.
# ---------------------------------------------------------------------------------
# find_artifact above is the reader half -- it exists because the writers all emitted
# into the cwd and the root grew to 1,017 entries. tidy_experiments.py then filed them
# away afterwards. This is the writers' half: emit into the run's directory to begin
# with, so there is nothing to tidy later.
#
# The campaign mapping lives HERE, not in tidy_experiments.py, so the script that files
# artifacts away and the writers that create them cannot disagree about where a run
# belongs. tidy_experiments imports these.

import re as _re

CAMPAIGNS = (
    (_re.compile(r'^s26'),                   'aug26_screens'),
    (_re.compile(r'^(ancs2|anc|a26)'),       'aug26_fulltrain'),
    (_re.compile(r'^mx'),                    'legacy_modifier_sweep'),
    (_re.compile(r'^tz'),                    'bench_backbones_tile48'),
    (_re.compile(r'^(p1b|p1n|p2|p3|phase)'), 'legacy_phase_sweeps'),
    (_re.compile(r'^ftl?$'),                 'legacy_finetune'),
    (_re.compile(r'^(matrix|wavelet|ensemble|smoke|perf|tile)'), 'legacy_misc'),
    (_re.compile(r'^(crossval|allclass|binary|mix|sv|merged)'), 'legacy_forth_altinay'),
    (_re.compile(r'^fz'),                    'zoo_sweep_full'),
    (_re.compile(r'^pfc'),                   'pfc_variance'),
    (_re.compile(r'^ms'),                    'model_sweep'),
)


def campaign(run):
    for pat, camp in CAMPAIGNS:
        if pat.match(run):
            return camp
    return 'unsorted'


def sanitise(model_name):
    """`timm/tinynet_e` -> `timm_tinynet_e`. Same rule as export_models.sanitise.

    Applied to every emitted filename, which is what stops the writers from creating
    `fztinye_timm/tinynet_e_coverage.json` -- a directory separator in the middle of what
    was meant to be one name. That bug put 48 stray directories in the root and made two
    runs sharing a backbone collide in the reader index.
    """
    return _re.sub(r'[^A-Za-z0-9._+-]', '_', model_name)


def run_output_dir(model_name, create=True):
    """The directory a run's artifacts belong in: experiments/<campaign>/<run>/."""
    run = sanitise(model_name).split('_')[0]
    d = os.path.join(ARCHIVE, campaign(run), run)
    if create:
        os.makedirs(d, exist_ok=True)
    return d


def out_path(model_name, suffix='', create=True):
    """Full path to write `{sanitised model_name}{suffix}` for this run.

    out_path('fztinye_timm/tinynet_e', '_coverage.json')
        -> experiments/zoo_sweep_full/fztinye/fztinye_timm_tinynet_e_coverage.json
    """
    return os.path.join(run_output_dir(model_name, create),
                        sanitise(model_name) + suffix)
