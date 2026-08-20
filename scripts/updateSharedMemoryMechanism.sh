#!/bin/bash
THISDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$THISDIR"
cd ..


if [ -f SharedMemoryVideoBuffers/README.md ]
then
  echo "Shared Memory Repo seems to already been git cloned once.."
  cd SharedMemoryVideoBuffers
  git pull
  make
  cd ..
else
  echo "Cloning a fresh Shared Memory Repo.."
  git clone https://github.com/AmmarkoV/SharedMemoryVideoBuffers
  cd SharedMemoryVideoBuffers
  make
  cd ..
fi



if [ -f SharedMemoryVideoBuffers/README.md ]
then
  # The Python bindings are imported from the clone itself (mvc/core/shared_memory.py)
  # -- nothing to copy. Just make the built library reachable from both resolution
  # points: the repo root (loadLibrary) and next to the upstream Python (Server).
  ln -sfn SharedMemoryVideoBuffers/libSharedMemoryVideoBuffers.so
  ln -sfn ../../libSharedMemoryVideoBuffers.so SharedMemoryVideoBuffers/src/python/libSharedMemoryVideoBuffers.so
else
  echo "Failed updating shared memory code"
fi

exit 0
