#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"
cd ..
 

source venv/bin/activate

python3 -m mvc.train configs/stage1.json
python3 -m mvc.train configs/smallmodel.json 
python3 -m mvc.train configs/bigmodel.json 

