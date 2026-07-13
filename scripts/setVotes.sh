#!/bin/bash

#This script should be put in the root directory of the ROS workspace

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

#Let's load a sane virtual environment
source install/setup.bash 

ros2 service call /magician_vision_classifier/set_min_votes magician_vision_classifier/srv/SetFloat64 "{data: $@}"

exit 0
