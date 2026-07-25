#!/bin/bash

WATTS="200"

echo "Setting GPU to $WATTS W" 

sudo nvidia-smi -pl $WATTS

exit 0
