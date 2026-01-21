#!/bin/bash

#This script should be put in the root directory of the ROS workspace

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

#Let's load a sane virtual environment
source ~/.bashrc
source src/magician_vision_classifier/venv/bin/activate
source install/setup.bash 

ros2 topic list
ros2 topic echo /detections


exit 0
