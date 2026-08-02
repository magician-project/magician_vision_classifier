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

def checkIfPathExists(filename):
    """Return True if the given path exists (file or directory)."""
    return os.path.exists(filename)

def checkIfPathIsDirectory(filename):
    """Return True if the given path exists and is a directory."""
    return os.path.isdir(filename)

def checkIfFileExists(filename):
    """Return True if the given path exists and is a file."""
    return os.path.isfile(filename)

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
    if not checkIfFileExists(config_file):
        print("Config file not found")
        sys.exit(1)
    with open(config_file) as json_file:
        data = json.load(json_file)
    # Inherit shared defaults from configs/common.json (next to the trainer), so
    # cross-cutting settings (e.g. class_merges, the discard toggles) live in ONE
    # place for every config regardless of where the config file itself sits. The
    # specific config deep-overrides the common one.
    common_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", "common.json")
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
