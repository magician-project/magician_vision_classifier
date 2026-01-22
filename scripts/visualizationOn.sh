#!/bin/bash

#This script should be put in the root directory of the ROS workspace

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

#Let's load a sane virtual environment
source install/setup.bash 

ros2 service call /set_visualization std_srvs/srv/SetBool "{data: true}"

exit 0
