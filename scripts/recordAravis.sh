#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

cd ../../../dependencies/aravis-c-examples/build
#./server&


#16666.66 microseconds = 60Hz / 10000 microseconds = 100 Hz / 8000 microseconds = 125 Hz  / 4000 microseconds = 250Hz / 3000 microseconds = 333.33Hz 
EXPOSURE_MICROSECONDS=6500 
FRAMES_PER_SECOND=10
FRAME_LIMIT=400

sleep 1
ARV_DEBUG=stream:2 ./06-grabber --exposure $EXPOSURE_MICROSECONDS --size 2448 2048 --fps $FRAMES_PER_SECOND --maxFrames $FRAME_LIMIT --buffers 40 -o $1

killall server

exit 0
