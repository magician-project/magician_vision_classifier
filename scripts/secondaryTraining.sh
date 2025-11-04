#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"
cd ..
 

source venv/bin/activate

python3 trainClassifierTorch.py configs/stage1.json resnet18
python3 trainClassifierTorch.py configs/stage1.json resnext50

python3 trainClassifierTorch.py configs/bigmodel.json resnext50
python3 trainClassifierTorch.py configs/bigmodel.json convnext_tiny 

python3 trainClassifierTorch.py configs/bigmodel.json efficientnet_v2_s

