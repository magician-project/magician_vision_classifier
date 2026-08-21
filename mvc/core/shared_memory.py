"""Shared memory bindings -- a thin wrapper over the upstream repository.

The real code lives in the SharedMemoryVideoBuffers clone at the repo root
(see scripts/updateSharedMemoryMechanism.sh): this module loads
SharedMemoryVideoBuffers/src/python/SharedMemoryManager.py from there and
re-exports it, so a `git pull` + `make` in the clone is the only sync step
and there is no local copy to drift.

The single local change is `loadLibrary` below: upstream's version looks in
the CWD and, when the library is missing, runs a bare `make` (there is no
Makefile at the repo root) and then sys.exit(0) -- a silent "success". Ours
resolves bare names against the repo root and fails loudly instead.
"""
import ctypes
import importlib.util
import os

from mvc.paths import repo_root

_UPSTREAM_DIR = os.path.join(repo_root(), "SharedMemoryVideoBuffers", "src", "python")


def load_upstream_module(module_name):
    """Import a module from the SharedMemoryVideoBuffers clone by file path.

    Loaded under a private name (not registered in sys.modules) so it cannot
    collide with the root-level SharedMemoryManager.py shim that external
    repos import by its old name.
    """
    path = os.path.join(_UPSTREAM_DIR, module_name + ".py")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found -- run scripts/updateSharedMemoryMechanism.sh "
            "to clone and build the SharedMemoryVideoBuffers library")
    spec = importlib.util.spec_from_file_location(f"smvb_{module_name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def loadLibrary(filename, relativePath="", forceUpdate=False):
    """Replacement for upstream's loadLibrary -- the only local change.

    Upstream's looks in the CWD, tries `make` when the file is missing, and
    sys.exit(0)s if make failed. This resolves bare names against the repo
    root -- where the runtime keeps the library -- and raises instead.
    """
    if not os.path.isabs(filename) and not os.path.exists(filename):
        at_root = os.path.join(repo_root(), os.path.basename(filename))
        if os.path.exists(at_root):
            filename = at_root
    if not os.path.exists(filename):
        raise FileNotFoundError(
            f"{filename} not found -- build it with scripts/updateSharedMemoryMechanism.sh")
    return ctypes.CDLL(filename, mode=ctypes.RTLD_GLOBAL)


_upstream = load_upstream_module("SharedMemoryManager")
_upstream.loadLibrary = loadLibrary   # the class __init__ calls the module global
SharedMemoryManager = _upstream.SharedMemoryManager
