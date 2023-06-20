#!/bin/bash

# Run bash script to train models and predict compound mechanism of action
# For each type of features
#     1. Using CellProfiler features
#     2. Using DINO features
#     3. Using CNN features

cp_data_dir="../../model_data/"

model_pred_dir_cellpro="../../prediction/cellprofiler/"
file_indicator_cellpro="_cellprofiler_final"

model_pred_dir_dino="../../prediction/dino/"
file_indicator_dino="_dino_final"

model_pred_dir_cnn="../../prediction/cnn/"
file_indicator_cnn="_CNN_final"

cd resnet_models_moa_prediction

python cp_resnet_train_pred.py --data_dir $cp_data_dir --feat_type "cellprofiler" --model_pred_dir $model_pred_dir_cellpro --file_indicator $file_indicator_cellpro

python cp_resnet_train_pred.py --data_dir $cp_data_dir --feat_type "cellprofiler" --model_pred_dir $model_pred_dir_cellpro --shuffle --file_indicator $file_indicator_cellpro

python cp_resnet_train_pred.py --data_dir $cp_data_dir --feat_type "dino" --model_pred_dir $model_pred_dir_dino --file_indicator $file_indicator_dino

python cp_resnet_train_pred.py --data_dir $cp_data_dir --feat_type "dino" --model_pred_dir $model_pred_dir_dino --shuffle --file_indicator $file_indicator_dino

python cp_resnet_train_pred.py --data_dir $cp_data_dir --feat_type "cnn" --model_pred_dir $model_pred_dir_cnn --file_indicator $file_indicator_cnn

python cp_resnet_train_pred.py --data_dir $cp_data_dir --feat_type "cnn" --model_pred_dir $model_pred_dir_cnn --shuffle --file_indicator $file_indicator_cnn
