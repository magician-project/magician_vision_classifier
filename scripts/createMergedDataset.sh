#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"
cd ..

rm -rf keras_dataset/

rm -rf keras_dataset_negative/
python3 legacy/dumpKerasDataset.py magician/20-0.02-lights-negativedents/*.pnm
mv keras_dataset/ keras_dataset_negative/

rm -rf keras_dataset_positive/
python3 legacy/dumpKerasDataset.py magician/21-0.02-lights-positivedents/*.pnm
mv keras_dataset/ keras_dataset_positive/


rm -rf keras_dataset_merged/
mkdir -p keras_dataset_merged/

./mergeCleanData.sh keras_dataset_positive/class_clean/ keras_dataset_negative/class_clean/

cd keras_dataset_merged
ln -s ../keras_dataset_negative/class_NegativeDent
ln -s ../keras_dataset_positive/class_PositiveDent

#python3 trainClassifierKeras.py keras_dataset_merged/
#python3 evaluate.py tile_classifier.keras tile_classifier.classes magician/20-0.02-lights-negativedents/colorFrame_0_00364.pnm
#mv heatmap.png heatmap364.png


exit 0
