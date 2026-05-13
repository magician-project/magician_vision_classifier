#!/bin/bash

#This script should be put in the root directory of the ROS workspace

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

if [ -z "$1" ]; then
    echo "Usage: $0 <model_name>"
    echo "Example: $0 allclass_resnet18"
    exit 1
fi

#Let's load a sane virtual environment
source install/setup.bash

ros2 service call /magician_vision_classifier/set_model magician_vision_classifier/srv/SetString "{data: '$1'}"

exit 0
