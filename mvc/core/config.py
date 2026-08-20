"""Config loading and small path helpers.

load_hyperparameters() merges configs/common.json (shared defaults, sitting next to
the trainer) with the per-run config; the per-run config wins. Note that the run
config doubles as the RESULTS file -- the trainer writes classes, the confusion
matrix and the tuned thresholds back into it -- so reruns with the same `name`
overwrite previous results.
"""

import json
import os
import sys

from mvc.paths import repo_root

def checkIfPathExists(filename):
    """Return True if the given path exists (file or directory)."""
    return os.path.exists(filename)

def checkIfPathIsDirectory(filename):
    """Return True if the given path exists and is a directory."""
    return os.path.isdir(filename)

def checkIfFileExists(filename):
    """Return True if the given path exists and is a file."""
    return os.path.isfile(filename)

def resolve_config(config_file):
    """Find a config by name, wherever it has been filed.

    Campaign configs live in `configs/`, and a finished run's config is filed beside its
    weights in `experiments/<campaign>/<run>/`. Callers -- the queue scripts, the report
    tools, a person on the command line -- pass a bare name like
    `anc_convnext_pico.json`, and every one of them would have to be updated in step with
    any move. This is the same fix already applied to artifacts (artifact_paths.
    find_artifact): the reader resolves a NAME rather than trusting a path.

    An existing path always wins, so nothing that works today changes behaviour.
    """
    if os.path.exists(config_file):
        return config_file
    here = repo_root()
    base = os.path.basename(config_file)
    for cand in (os.path.join(here, 'configs', base),
                 os.path.join(here, base)):
        if os.path.exists(cand):
            return cand
    import glob as _glob
    hits = sorted(_glob.glob(os.path.join(here, 'experiments', '**', base), recursive=True))
    if not hits:
        return config_file
    if len(hits) > 1:
        # A run's config exists twice: the INPUT staged under experiments/configs_runs/,
        # and the copy the trainer filed beside the weights with `gate`, `classes` and
        # `model_md5` written back into it. They are not the same file, and picking the
        # wrong one hands a caller a config with no calibrated gate. Prefer the filed
        # copy, and say so rather than resolving silently.
        filed = [h for h in hits if os.sep + 'configs_runs' + os.sep not in h]
        if filed and len(filed) < len(hits):
            print(f'[config] {base}: {len(hits)} copies; using the run-dir copy '
                  f'({os.path.relpath(filed[0], here)}), not the staged input')
            hits = filed
        elif len(filed) > 1:
            print(f'[config] WARNING {base}: {len(hits)} ambiguous copies, using '
                  f'{os.path.relpath(hits[0], here)}')
    return hits[0]


def load_hyperparameters(config_file):
    """
    Load and parse a JSON configuration file containing model hyperparameters,
    dataloader settings, optimizer config, and training options.

    Args:
        config_file: Path to a .json configuration file.

    Returns:
        Parsed dict with all configuration values.

    Exits:
        sys.exit(1) if the file does not exist.
    """
    config_file = resolve_config(config_file)
    if not checkIfFileExists(config_file):
        print("Config file not found")
        sys.exit(1)
    with open(config_file) as json_file:
        data = json.load(json_file)
    # Inherit shared defaults from configs/common.json (next to the trainer), so
    # cross-cutting settings (e.g. class_merges, the discard toggles) live in ONE
    # place for every config regardless of where the config file itself sits. The
    # specific config deep-overrides the common one.
    common_path = os.path.join(repo_root(), "configs", "common.json")
    if checkIfFileExists(common_path):
        with open(common_path) as cf:
            common = json.load(cf)
        def _deep_merge(base, override):
            out = dict(base)
            for k, v in override.items():
                out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
            return out
        data = _deep_merge(common, data)
        print("Merged shared defaults from", common_path)
    return data
