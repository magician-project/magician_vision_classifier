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
  cp SharedMemoryVideoBuffers/src/python/SharedMemoryServer.py ./
  cp SharedMemoryVideoBuffers/src/python/SharedMemoryManager.py ./
  ln -s SharedMemoryVideoBuffers/libSharedMemoryVideoBuffers.so
else
  echo "Failed updating shared memory code"
fi

exit 0
