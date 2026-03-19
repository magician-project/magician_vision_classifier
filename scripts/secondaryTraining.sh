#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"
cd ..

source venv/bin/activate

# ---------------------------------------------------------------
# bigmodel.json  — heavy pretrained backbones
#   resnext50      ~25M params
#   convnext_tiny  ~28M params
#   efficientnet_v2_s ~21M params
#   swin_v2_t      ~28M params
#   densenet121    ~8M params (memory-heavy due to dense connections)
# ---------------------------------------------------------------
python3 trainClassifierTorch.py configs/bigmodel.json resnext50
python3 trainClassifierTorch.py configs/bigmodel.json convnext_tiny
python3 trainClassifierTorch.py configs/bigmodel.json efficientnet_v2_s
python3 trainClassifierTorch.py configs/bigmodel.json swin_v2_t
python3 trainClassifierTorch.py configs/bigmodel.json densenet121

# ---------------------------------------------------------------
# smallmodel.json — medium pretrained backbones + medium custom CNN
#   resnet18           ~11M params
#   regnet_y_800mf     ~6M params
#   regnet_y_400mf     ~4M params
#   mobilenet_v3_large ~5M params
#   efficientnet_b0    ~5M params
#   mnasnet1_0         ~4M params
#   shufflenet_v2_x1_0 ~2.3M params
#   custom             medium custom CNN (base_channels=48, final_dense_layer=512)
# ---------------------------------------------------------------
python3 trainClassifierTorch.py configs/smallmodel.json resnet18
python3 trainClassifierTorch.py configs/smallmodel.json regnet_y_800mf
python3 trainClassifierTorch.py configs/smallmodel.json regnet_y_400mf
python3 trainClassifierTorch.py configs/smallmodel.json mobilenet_v3_large
python3 trainClassifierTorch.py configs/smallmodel.json efficientnet_b0
python3 trainClassifierTorch.py configs/smallmodel.json mnasnet1_0
python3 trainClassifierTorch.py configs/smallmodel.json shufflenet_v2_x1_0
python3 trainClassifierTorch.py configs/smallmodel.json custom

# ---------------------------------------------------------------
# verysmallmodel.json — lightweight backbones + tiny custom CNN
#   mobilenet_v3_small  ~2.5M params
#   shufflenet_v2_x0_5  ~1.4M params
#   squeezenet1_1       ~1.2M params  (fire modules, highest throughput)
#   mnasnet0_5          ~2.2M params
#   verysmall_cnn       tiny custom CNN (base_channels=30, final_dense_layer=256)
# ---------------------------------------------------------------
python3 trainClassifierTorch.py configs/verysmallmodel.json mobilenet_v3_small
python3 trainClassifierTorch.py configs/verysmallmodel.json shufflenet_v2_x0_5
python3 trainClassifierTorch.py configs/verysmallmodel.json squeezenet1_1
python3 trainClassifierTorch.py configs/verysmallmodel.json mnasnet0_5
python3 trainClassifierTorch.py configs/verysmallmodel.json verysmall_cnn


