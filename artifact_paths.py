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
