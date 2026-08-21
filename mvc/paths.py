"""Repo-root resolution.

Every module that needs a root-anchored path (configs/, models/, experiments/,
recommended_configuration.json) goes through repo_root() instead of chaining
os.path.dirname() on __file__ -- so a future layout move cannot break a dozen
files at once.
"""

import os


def repo_root():
    """Absolute path of the magician_vision_classifier root (two levels above this file)."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
