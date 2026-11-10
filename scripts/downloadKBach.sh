#!/bin/bash

#Downloads the kbach hier_convnext_pico model used by the kbach_pico preset of
#recommended_configuration.json into the root of this repository

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR/.."

#Let's load a sane virtual environment
if [ -f venv/bin/activate ]; then
    source venv/bin/activate
fi

python3 -m mvc.inference.model_download kbachhybpenscjensemaavg1ep_hier_convnext_pico_20261107_121606.zip

exit $?
