#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"
cd ..
 

source venv/bin/activate

python3 trainClassifierTorch.py configs/stage1.json
python3 trainClassifierTorch.py configs/smallmodel.json 
python3 trainClassifierTorch.py configs/bigmodel.json 

