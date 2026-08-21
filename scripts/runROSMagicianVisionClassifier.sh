#!/bin/bash

#This script should be put in the root directory of the ROS workspace

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

#Let's load a sane virtual environment
source ~/.bashrc
source src/magician_vision_classifier/venv/bin/activate
source install/setup.bash 

cd src/magician_vision_classifier
python3 -m mvc.inference.live_torch_ros 


exit 0
