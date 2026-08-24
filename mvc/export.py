#!/usr/bin/env python3
"""THE single source of truth for packaging a trained run into models/{run}_{ts}.zip.

The trainer calls `export_run()` at the end of training; the CLI below does the same thing
for runs that were trained earlier, so there is exactly one implementation of "what a model
archive contains and what makes it valid".

WHY IT EXISTS
-------------
The trainer used to build its own archive inline:

    model_name = f"{config['name']}_{config['model']}"
    subprocess.run(["zip", "-r", f"models/{model_name}_{ts}.zip", ...], check=False)

Two things were wrong with that, and they compounded.

  1. NO SANITISATION. A model configured as `timm/tinynet_e` makes `model_name`
     `fztinye_timm/tinynet_e`, so the archive path became
     `models/fztinye_timm/tinynet_e_<ts>.zip` -- a directory that does not exist. The slash
     also landed in the member names, where `tinynet_e.pth` would collide across any two
     runs sharing a backbone.
  2. `check=False`. zip failed with "zip I/O error: No such file or directory", the return
     code was discarded, and the trainer printed its usual success message. **13 fully
     trained models ended up with no archive at all and nothing reported it.**

So this module sanitises names, and it VERIFIES rather than hopes: it refuses to declare
success unless the weights load, the config parses, and the finished zip passes a CRC check
with the expected members inside. A packaging failure now raises.

ARCHIVE CONTENTS
----------------
    {run}.json                  the run config, including the calibrated `gate`
    {run}_confusion.json        confusion matrix
    {run}_threshold_curve.json  the miss/FA curve the KPI is read off
    {run}_coverage.json         per-class coverage detection, when the run has one
    {run}*.png                  confusion and threshold plots
    tensorboard/{run}/...       training curves, paths preserved

Sidecars are included when present and reported when absent -- a model without a
`gate` or a threshold curve can still be packaged, but the caller is told.

TIMING: THE TRAINER CANNOT PACKAGE A COMPLETE ARCHIVE ON ITS OWN
---------------------------------------------------------------
`_threshold_curve.json` is written by `score_checkpoints.py` and `_coverage.json` by
`eval_coverage.py` -- both of which run AFTER the trainer exits. So the archive the trainer
builds necessarily lacks them, and reports them as missing sidecars. That is not a bug in
either place; it is the order of the pipeline.

To get archives with the full sidecar set, re-export once the pipeline has finished:

    python3 -m mvc.export --apply --force

Re-exporting is cheap (the weights are stored uncompressed, not recompressed) and this is
the single implementation either way, so the contents stay consistent.

Usage:
    python3 -m mvc.export                       # list runs missing an archive
    python3 -m mvc.export --apply
    python3 -m mvc.export --apply --run fztinyc_timm_tinynet_c
    python3 -m mvc.export --apply --force       # rebuild even if an archive exists
    python3 -m mvc.export --apply --no-verify-weights   # skip the torch.load check
"""

import argparse
import datetime
import glob
import json
import os
import re
import sys
import zipfile

STORE = 'models'
# Only catches an obviously-empty or truncated file. Deliberately NOT sized to "a real
# model": the smallest backbone in the zoo (shufflenet_v2_x0_5, 0.35M params) is ~1.4 MB
# fp32 and would be under a 1 MB bar in fp16, so a plausible-looking threshold would start
# rejecting valid models. The torch.load check below is the real validator.
MIN_PTH_BYTES = 4096

# Sidecars, in archive order. (suffix, required)
SIDECARS = (
    ('.json', True),
    ('_confusion.json', False),
    ('_threshold_curve.json', False),
    ('_coverage.json', False),
)


# --------------------------------------------------------------------------- naming
def model_name_of(cfg):
    """The prefix the trainer's artifacts actually carry -- slash and all."""
    name = cfg.get('name')
    return f"{name}_{cfg['model']}" if name else cfg['model']


def sanitise(model_name):
    """A filesystem-safe, collision-free run name.

    `timm/tinynet_e` -> `timm_tinynet_e`. Applied to BOTH the archive filename and the
    member names: flattening to the basename alone would make two runs of the same
    backbone indistinguishable inside their archives.
    """
    return re.sub(r'[^A-Za-z0-9._+-]', '_', model_name)


# --------------------------------------------------------------------------- checks
def _check_config(path, problems, warnings_):
    try:
        with open(path) as fh:
            cfg = json.load(fh)
    except Exception as exc:
        problems.append(f'config {path} does not parse: {exc}')
        return None
    for key in ('model', 'classes'):
        if not cfg.get(key):
            problems.append(f'config {path} has no "{key}"')
    if not cfg.get('gate'):
        warnings_.append('no calibrated `gate` in the config -- deployment needs one')
    return cfg


def _check_weights(path, problems, warnings_, deep=True):
    size = os.path.getsize(path)
    if size < MIN_PTH_BYTES:
        problems.append(f'{path} is only {size} bytes -- truncated?')
        return
    if not deep:
        return
    try:
        import torch
        # weights_only=False: torch 2.6+ defaults it to True, which refuses the
        # AttributeDict save_hyperparameters() stores. Refusing to unpickle would make
        # this validator reject exactly the archives it is meant to bless. Safe here --
        # the file was written by our own trainer moments ago, on this box.
        obj = torch.load(path, map_location='cpu', weights_only=False)
    except Exception as exc:
        problems.append(f'{path} does not load as a torch checkpoint: {exc}')
        return
    sd = obj.get('state_dict', obj) if isinstance(obj, dict) else None
    if not isinstance(sd, dict) or not sd:
        problems.append(f'{path} contains no state_dict')
        return
    n_tensors = sum(1 for v in sd.values() if hasattr(v, 'shape'))
    if n_tensors == 0:
        problems.append(f'{path} state_dict holds no tensors')


def _verify_archive(zip_path, expected, problems):
    """Reopen the finished archive: CRC-check it and confirm the members are really there."""
    try:
        with zipfile.ZipFile(zip_path) as zf:
            bad = zf.testzip()
            if bad is not None:
                problems.append(f'{zip_path}: CRC failure in {bad}')
                return
            inside = {i.filename: i.file_size for i in zf.infolist()}
    except Exception as exc:
        problems.append(f'{zip_path} is not readable as a zip: {exc}')
        return
    for arc, src in expected:
        if arc not in inside:
            problems.append(f'{zip_path}: member {arc} missing')
        elif inside[arc] != os.path.getsize(src):
            problems.append(f'{zip_path}: member {arc} size mismatch')


# --------------------------------------------------------------------------- collect
def _locate(model_name, suffix):
    """Where a run's `{model_name}{suffix}` actually is.

    Two layouts coexist: runs finished before the writers were made path-aware left their
    artifacts loose in the repo root, and everything after emits into
    experiments/<campaign>/<run>/. The run directory is checked FIRST so a re-exported run
    picks up its filed copy, with the root as the legacy fallback.
    """
    from mvc.core.artifact_paths import find_artifact, out_path
    filed = out_path(model_name, suffix, create=False)
    if os.path.exists(filed):
        return filed
    legacy = f'{model_name}{suffix}'
    if os.path.exists(legacy):
        return legacy
    # Last resort, by NAME: the frozen configs live at experiments/configs_frozen/, which
    # is neither the run directory nor the root. Without this the incumbent could not be
    # exported at all.
    return find_artifact(os.path.basename(f'{model_name}{suffix}'))


def _find_plots(model_name):
    """Every `{model_name}*.png` belonging to this run, wherever it lives.

    Unlike the SIDECARS (one fixed suffix each, so _locate() can check root vs. filed
    exactly), a run can have several plots (`_confusion_raw.png`,
    `_threshold_curve_curve.png`, ...) under a prefix that only the trainer knows in
    full, so this is a prefix glob rather than a single-file lookup. It has to search
    both layouts _locate() does: loose in the cwd -- where the trainer's own
    export_run() call finds them, before tidy_experiments.py has run -- and anywhere
    under experiments/, since a tidied run is not necessarily at the one path
    out_path() would compute for it (e.g. `experiments/all/old/`, filed by hand before
    the campaign scheme existed). Searching the whole tree, not one guessed directory,
    is what makes this reliable after tidying.
    """
    found = {}
    for stem in (os.path.basename(model_name), sanitise(model_name)):
        for p in (glob.glob(f'{stem}*.png') +
                  glob.glob(os.path.join('experiments', '**', f'{stem}*.png'), recursive=True)):
            found.setdefault(os.path.basename(p), p)
    return sorted(found.values())


def collect_members(model_name, include_tensorboard=True):
    """[(arcname, srcpath)] for everything that belongs in this run's archive."""
    run = sanitise(model_name)
    out, missing = [], []
    pth = _locate(model_name, '.pth') or f'{model_name}.pth'
    if os.path.exists(pth):
        out.append((f'{run}.pth', pth))
    for suffix, required in SIDECARS:
        p = _locate(model_name, suffix) or f'{model_name}{suffix}'
        if os.path.exists(p):
            out.append((f'{run}{suffix}', p))
        elif required:
            missing.append(p)
        else:
            missing.append(p)
    plots = _find_plots(model_name)
    if not plots:
        missing.append(f'{model_name}*.png (no plots found under . or experiments/)')
    for p in plots:
        stem = os.path.basename(p)
        for pref in (os.path.basename(model_name), sanitise(model_name)):
            if stem.startswith(pref):
                out.append((run + stem[len(pref):], p))
                break
    if include_tensorboard:
        for p in sorted(glob.glob(f'tensorboard/{model_name}/**/*', recursive=True)):
            if os.path.isfile(p):
                out.append((f'tensorboard/{run}/' + os.path.relpath(p, f'tensorboard/{model_name}'), p))
    return out, missing


# --------------------------------------------------------------------------- export
def export_run(cfg, store=STORE, include_tensorboard=True, verify_weights=True,
               quiet=False):
    """Package one run. Returns the archive path. Raises RuntimeError on any hard failure.

    `cfg` is a config dict or a path to one. Deliberately raises rather than returning a
    status: the whole reason this module exists is that the previous implementation
    discarded its exit code.
    """
    if isinstance(cfg, str):
        with open(cfg) as fh:
            cfg = json.load(fh)
    model_name = model_name_of(cfg)
    run = sanitise(model_name)

    problems, warns = [], []
    pth = _locate(model_name, '.pth')
    if pth is None:
        problems.append(f'no weights at {model_name}.pth (root or run dir)')
    else:
        _check_weights(pth, problems, warns, deep=verify_weights)
    cfg_path = _locate(model_name, '.json')
    if cfg_path and os.path.exists(cfg_path):
        _check_config(cfg_path, problems, warns)
    else:
        problems.append(f'no config at {cfg_path}')

    members, missing = collect_members(model_name, include_tensorboard)
    for m in missing:
        warns.append(f'sidecar not found: {m}')
    if problems:
        raise RuntimeError(f'cannot export {run}:\n  ' + '\n  '.join(problems))

    os.makedirs(store, exist_ok=True)
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    zip_path = os.path.join(store, f'{run}_{ts}.zip')

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for arc, src in members:
            # .pth is already a compressed container; recompressing costs time and gains
            # nothing, so store it verbatim.
            ctype = zipfile.ZIP_STORED if src.endswith('.pth') else zipfile.ZIP_DEFLATED
            zf.write(src, arcname=arc, compress_type=ctype)

    verify_problems = []
    _verify_archive(zip_path, members, verify_problems)
    if verify_problems:
        os.remove(zip_path)
        raise RuntimeError(f'archive for {run} failed verification:\n  ' +
                           '\n  '.join(verify_problems))

    if not quiet:
        size = os.path.getsize(zip_path) / 1e6
        print(f'[export] {zip_path}  ({len(members)} members, {size:.1f} MB)')
        for w in warns:
            print(f'[export] note: {w}')
    return zip_path


# --------------------------------------------------------------------------- CLI
def already_exported():
    out = {}
    for z in glob.glob(os.path.join(STORE, '*.zip')):
        out.setdefault(re.sub(r'_\d{8}_\d{6}\.zip$', '', os.path.basename(z)), z)
    return out


def discover():
    runs = []
    # Root for live configs, plus every config filed anywhere under experiments/.
    # This was `experiments/*/*/*.json` -- exactly three levels -- which missed the frozen
    # configs at experiments/configs_frozen/<name>.json, two levels down. The incumbent
    # `anc_convnext_pico` lives there, so a --force re-export silently skipped the one
    # model that is actually deployed, leaving it published without its threshold curve
    # or coverage table.
    for cfg_path in sorted(glob.glob('*.json')) + sorted(
            glob.glob(os.path.join('experiments', '**', '*.json'), recursive=True)):
        try:
            with open(cfg_path) as fh:
                cfg = json.load(fh)
        except Exception:
            continue
        if not isinstance(cfg, dict) or 'hparams' not in cfg or 'model' not in cfg:
            continue
        ckdir = cfg.get('checkpoint_dir', '')
        if ckdir and not glob.glob(os.path.join(ckdir, '*.ckpt')):
            continue
        runs.append((sanitise(model_name_of(cfg)), cfg))
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--apply', action='store_true')
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--run', action='append')
    ap.add_argument('--no-verify-weights', action='store_true',
                    help='skip the torch.load check (faster, less safe)')
    args = ap.parse_args()

    have = already_exported()
    todo = [(r, c) for r, c in discover()
            if (not args.run or r in args.run) and (args.force or r not in have)]
    if not todo:
        print('nothing to export -- every trained run already has an archive')
        return 0

    print(f'{len(todo)} run(s) without an archive:')
    for run, _ in todo:
        print(f'  {run}')
    if not args.apply:
        print('\ndry run -- nothing written. re-run with --apply')
        return 0

    failed = 0
    for run, cfg in todo:
        try:
            export_run(cfg, verify_weights=not args.no_verify_weights)
        except RuntimeError as exc:
            failed += 1
            print(f'!! {exc}')
    print(f'\nbuilt {len(todo) - failed} archive(s), {failed} failed')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
