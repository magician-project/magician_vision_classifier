#!/usr/bin/env python3
"""File per-run artifacts out of the repo root into experiments/<campaign>/<run>/.

The root had accumulated 1,337 loose files -- 737 PNGs, 462 JSONs, 68 .pth totalling
5.4 GB -- because the trainer, `score_checkpoints.py` and `eval_coverage.py` all write
`{name}_{model}_{kind}.{ext}` into the cwd. None of it is tracked, so it never showed up as
a diff; it just buried the 51 source files and the handful of live configs.

WHAT THIS MOVES, AND WHAT IT REFUSES TO
---------------------------------------
Artifacts are grouped by RUN NAME, so a run's curve, confusion matrix, plots, coverage
table and weights all land in one directory and stay legible as a unit.

Four hard exclusions, in priority order:

  1. ANYTHING GIT TRACKS IS LEFT WHERE IT IS. The dev box has just deleted the tracked
     root `.json` files on its side; moving the same paths here would turn a clean delete
     into a rename/delete conflict on the next merge. Tracked cleanup is the user's, and
     it is already in flight.
  2. LIVE FILES, by explicit name: the six Aug26 configs (`anc`, `ancs2` and their seed
     replicates), the coverage carve-out `val_coverage_frames.json`, and
     `recommended_configuration.json`.
  3. ANYTHING YOUNGER THAN --min-age-hours (default 6). A ~30 h seed-replicate queue is
     running as this is written; each run writes a confusion/curve pair per epoch and a
     coverage table at the end, and `seed_replicates_report.py` reads them back by bare
     name from the cwd. The age guard means an in-flight run's outputs can never be moved
     out from under it -- and makes the script safe to re-run after the queue to file the
     new outputs away.
  4. SOURCE AND DOCS: *.py, *.md, and the build/package files.

The readers were made path-agnostic at the same time (`artifact_paths.find_artifact`), so
a moved artifact is still found by name wherever it ended up. That is what makes moving
these safe rather than a rename-everything-and-fix-the-fallout exercise.

Usage:
    python tidy_experiments.py --dry-run          # default: show, move nothing
    python tidy_experiments.py --apply
    python tidy_experiments.py --apply --min-age-hours 0    # after the queue is done
"""

import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections import defaultdict

from mvc.paths import repo_root

ROOT = repo_root()
DEST = 'experiments'

# Files that are inputs to something running or shipping. Never moved at any age.
LIVE = {
    'val_coverage_frames.json',        # the coverage carve-out -- a run that lost this
                                       # would silently train on its own tripwire
    'recommended_configuration.json',
    'anc_convnext_pico.json', 'ancs2_convnext_pico.json',
    'anc1337_convnext_pico.json', 'ancs21337_convnext_pico.json',
    'anc7_convnext_pico.json', 'ancs27_convnext_pico.json',
    'coverage_extract_cache.npz',      # rebuilt at cost; eval_coverage.py reads it
}
KEEP_EXT = {'.py', '.md', '.txt', '.xml', '.yaml', '.yml', '.sh', '.cfg', '.toml'}

# Only these are ever considered artifacts. Anything not matching is left alone -- the
# script moves what it recognises rather than everything it fails to recognise.
ARTIFACT_EXT = {'.json', '.png', '.pth', '.bak', '.npz', '.csv'}

# run-name prefix -> campaign directory. First match wins, so order matters.
CAMPAIGNS = (
    (re.compile(r'^s26'),                   'aug26_screens'),
    (re.compile(r'^(ancs2|anc|a26)'),       'aug26_fulltrain'),
    (re.compile(r'^mx'),                    'legacy_modifier_sweep'),
    (re.compile(r'^tz'),                    'bench_backbones_tile48'),
    (re.compile(r'^(p1b|p1n|p2|p3|phase)'), 'legacy_phase_sweeps'),
    (re.compile(r'^ftl?$'),                 'legacy_finetune'),
    (re.compile(r'^(matrix|wavelet|ensemble|smoke|perf|tile)'), 'legacy_misc'),
    (re.compile(r'^(crossval|allclass|binary|mix|sv|merged)'), 'legacy_forth_altinay'),
    # The live campaigns, so the current work is not filed under 'unsorted'.
    (re.compile(r'^fz'),                    'zoo_sweep_full'),
    (re.compile(r'^pfc'),                   'pfc_variance'),
    (re.compile(r'^ms'),                    'model_sweep'),
)


def is_run_config(path):
    """True for a training config -- an INPUT, never an artifact.

    Configs and artifacts are both bare `.json` in the root, and telling them apart by
    name does not work: `fzv2nano_timm_convnextv2_nano.json` is a config while
    `fzv2nano_timm_convnextv2_nano_coverage.json` is an artifact. So look inside.

    Getting this wrong is expensive rather than untidy. `export_models.discover()` globs
    the root for configs, and the sweep's restart skip-check decides what NOT to re-train
    by looking for them -- filing the configs away would have made a resumed sweep
    re-train every finished model.
    """
    try:
        with open(path) as fh:
            d = json.load(fh)
    except Exception:
        return False
    return isinstance(d, dict) and 'hparams' in d and 'model' in d


def timm_slash_dirs():
    """The `fz*_timm/` directories the slash bug created, and what is inside them.

    A model configured as `timm/convnextv2_nano` makes the writers compose
    `fzv2nano_timm/convnextv2_nano_coverage.json` -- a directory separator in the middle
    of what was meant to be one filename. Moving those files as-is would be worse than
    leaving them: find_artifact indexes experiments/ by BASENAME, first-wins, so two runs
    sharing a backbone would collide and one would become unreachable. So they are
    renamed to the sanitised form on the way out, which is also the name the archives in
    models/ already use.
    """
    out = []
    for d in sorted(glob.glob('*_timm')):
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            src = os.path.join(d, f)
            if os.path.isfile(src):
                out.append((src, f'{d}_{f}'))     # fzv2nano_timm/ + x -> fzv2nano_timm_x
    return out


def backbone_to_run():
    """backbone name -> run prefix, for the plots the slash bug left unprefixed.

    Some plots for a `timm/<backbone>` run were written as `convnextv2_atto_*.png` with no
    run prefix at all, so grouping them by leading token files them under a BACKBONE --
    `experiments/unsorted/convnextv2/` -- which is not a run and tells you nothing.

    The configs recover the mapping: a config whose model is `timm/convnextv2_atto` and
    whose name is `fzv2atto` says those plots belong to `fzv2atto`. Only unambiguous
    backbones are mapped; if two runs share one, the plots stay put rather than being
    filed under a guess.
    """
    owners = defaultdict(set)
    for c in glob.glob('*.json'):
        if not is_run_config(c):
            continue
        try:
            with open(c) as fh:
                cfg = json.load(fh)
        except Exception:
            continue
        model = str(cfg.get('model', ''))
        if model.startswith('timm/') and cfg.get('name'):
            owners[model.split('/', 1)[1]].add(cfg['name'])
    return {bb: next(iter(runs)) for bb, runs in owners.items() if len(runs) == 1}


def run_name(fname, backbones=None):
    """Recover the run FAMILY from an artifact filename.

    The leading token, rather than a suffix-and-model-stripping scheme. The first attempt
    stripped a list of known model names off the end, which split `tz_convnext_pico` from
    `tz_convnext_femto` purely because the second model was not on the list -- one bench
    campaign scattered across ten directories, and silently wrong for any model added
    later. The leading token needs no such list and cannot go stale.
    """
    stem = os.path.splitext(fname)[0]
    if stem.endswith('.json'):            # *.json.bak
        stem = stem[:-5]
    if stem.startswith('epochcov_'):      # per-epoch coverage, written by epoch_cov.sh
        stem = stem[len('epochcov_'):]
    # Unprefixed timm plots: longest backbone match wins, so convnextv2_atto is not
    # shadowed by a hypothetical convnextv2.
    for bb in sorted(backbones or (), key=len, reverse=True):
        if stem.startswith(bb):
            return backbones[bb]
    return stem.split('_')[0]


def campaign(run):
    for pat, camp in CAMPAIGNS:
        if pat.match(run):
            return camp
    return 'unsorted'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--apply', action='store_true', help='actually move files')
    ap.add_argument('--dry-run', action='store_true', default=True)
    ap.add_argument('--min-age-hours', type=float, default=6.0,
                    help='never move a file younger than this (protects a running queue)')
    ap.add_argument('--run', action='append',
                    help='only file artifacts whose run family matches (repeatable). Use '
                         'with --min-age-hours 0 to tidy one finished run without '
                         'touching anything else.')
    args = ap.parse_args()
    apply = args.apply

    os.chdir(ROOT)
    tracked = set(subprocess.run(['git', 'ls-files'], capture_output=True, text=True,
                                 check=True).stdout.split('\n'))
    cutoff = time.time() - args.min_age_hours * 3600
    backbones = backbone_to_run()

    plan = defaultdict(list)
    skipped = defaultdict(int)
    for f in sorted(os.listdir('.')):
        if os.path.isdir(f) or f.startswith('.'):
            continue
        ext = os.path.splitext(f)[1]
        if f in LIVE:
            skipped['live (never moved)'] += 1
            continue
        if f in tracked:
            skipped['git-tracked (user is cleaning these upstream)'] += 1
            continue
        if ext in KEEP_EXT or ext not in ARTIFACT_EXT:
            skipped['source/docs/unrecognised'] += 1
            continue
        if ext == '.json' and is_run_config(f):
            skipped['run config (an INPUT -- export + restart-skip read these)'] += 1
            continue
        if os.path.getmtime(f) > cutoff:
            skipped[f'younger than {args.min_age_hours} h (in-flight run)'] += 1
            continue
        run = run_name(f, backbones)
        if args.run and run not in args.run:
            skipped['other runs (--run filter)'] += 1
            continue
        plan[os.path.join(DEST, campaign(run), run)].append(f)

    # The timm-slash directories, renamed to the sanitised form on the way out.
    renames = {}
    for src, dst in timm_slash_dirs():
        if os.path.getmtime(src) > cutoff:
            skipped[f'younger than {args.min_age_hours} h (in-flight run)'] += 1
            continue
        if dst.endswith('.json') and is_run_config(src):
            skipped['run config (an INPUT -- export + restart-skip read these)'] += 1
            continue
        run = run_name(dst, backbones)
        if args.run and run not in args.run:
            skipped['other runs (--run filter)'] += 1
            continue
        dest = os.path.join(DEST, campaign(run), run)
        plan[dest].append(src)
        renames[src] = dst

    total = sum(len(v) for v in plan.values())
    nbytes = sum(os.path.getsize(f) for v in plan.values() for f in v)
    for dest in sorted(plan):
        files = plan[dest]
        size = sum(os.path.getsize(f) for f in files) / 1e6
        print(f'{dest:66s} {len(files):4d} files  {size:8.1f} MB')
        if apply:
            os.makedirs(dest, exist_ok=True)
            for f in files:
                shutil.move(f, os.path.join(dest, renames.get(f, os.path.basename(f))))

    print(f'\n{"MOVED" if apply else "WOULD MOVE"} {total} files, {nbytes / 1e9:.2f} GB '
          f'into {len(plan)} directories')
    for reason, n in sorted(skipped.items(), key=lambda kv: -kv[1]):
        print(f'  left in root: {n:5d}  {reason}')
    if not apply:
        print('\ndry run -- nothing moved. re-run with --apply')
    return 0


if __name__ == '__main__':
    sys.exit(main())
