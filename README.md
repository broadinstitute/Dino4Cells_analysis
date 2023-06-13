# DINO4Cells_analysis

This repo will contain the code for reproducing the results and figures of the paper

## Installation

`pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116`

`pip install -r requirements.py`

## Classification

To train classifiers on the features extracted using DINO, please use the following pipeline

### Train / test division

First, divide the data into train and test. There are two separate train / test divisions, based on whether the classification task was protein localization or cell type prediction.

#### For single cell classification:

`python divide_train_valid.py --config config_HPA_single_cells.yaml`

`python divide_train_valid.py --config config_HPA_single_cells.yaml --cells` 

#### For HPA FOV classification:

`python divide_train_valid.py --config config_HPA_FOV.yaml`

`python divide_train_valid.py --config config_HPA_FOV.yaml --cells` 

### Train classififier

#### For HPA FOV protein localization classification

`CUDA_VISIBLE_DEVICES=0,1,2,3 python  -m torch.distributed.launch --nproc_per_node=4 run_end_to_end.py --config config_HPA_FOV.yaml --epochs 10 --balance True --num_classes 28 --train_cell_type False --train_protein True --train True --test True --classifier_state_dict results/HPA_FOV_classification/classifier_protein.pth`

#### For HPA FOV cell line classification

`CUDA_VISIBLE_DEVICES=0,1,2,3 python  -m torch.distributed.launch --nproc_per_node=4 run_end_to_end.py --config config_HPA_FOV.yaml --epochs 10 --balance True --num_classes 35 --train_cell_type True --train_protein False --train True --test True --classifier_state_dict results/HPA_FOV_classification/classifier_cells.pth`

#### For HPA single cells protein localization classification

`CUDA_VISIBLE_DEVICES=0,1,2,3 python  -m torch.distributed.launch --nproc_per_node=4 run_end_to_end.py --config config_HPA_single_cells.yaml --epochs 10 --balance True --num_classes 19 --train_cell_type False --train_protein True --train True --test True --classifier_state_dict results/HPA_single_cells_classification/classifier_protein.pth`

#### For HPA single cells cell line classification

`CUDA_VISIBLE_DEVICES=0,1,2,3 python  -m torch.distributed.launch --nproc_per_node=4 run_end_to_end.py --config config_HPA_single_cells.yaml --epochs 10 --balance True --num_classes 29 --train_cell_type True --train_protein False --train True --test True --classifier_state_dict results/HPA_single_cells_classification/classifier_cells.pth`
