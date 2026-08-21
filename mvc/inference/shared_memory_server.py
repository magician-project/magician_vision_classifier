"""Standalone shared memory server for testing -- re-exported from upstream.

The class comes from the SharedMemoryVideoBuffers clone (src/python/SharedMemoryServer.py)
via mvc.core.shared_memory; its load_library resolves the .so next to its own file, and
the clone carries a symlink src/python/libSharedMemoryVideoBuffers.so pointing at the
clone's built library, so it works once the clone is built
(scripts/updateSharedMemoryMechanism.sh creates the symlink too).
"""
from mvc.core.shared_memory import load_upstream_module

_upstream = load_upstream_module("SharedMemoryServer")
SharedMemoryServer = _upstream.SharedMemoryServer

if __name__ == "__main__":
    server = SharedMemoryServer()
    server.run()
