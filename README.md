# DINO4Cells_analysis

This repo will contain the code for reproducing the results and figures of the paper

## Installation

`pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116`

`pip install -r requirements.py`

## Classification

To train classifiers on the features extracted using DINO, please use the following pipeline for the different datasets and tasks:

### HPA whole images

#### Train / test division

`python divide_train_valid.py --config config_HPA_FOV.yaml`

`python divide_train_valid.py --config config_HPA_FOV.yaml --cells` 

#### Protein localization classification

`CUDA_VISIBLE_DEVICES=0,1,2,3 python  -m torch.distributed.launch --nproc_per_node=4 run_end_to_end.py --config config_HPA_FOV.yaml --epochs 10 --balance True --num_classes 28 --train_cell_type False --train_protein True --train True --test True --classifier_state_dict results/HPA_FOV_classification/classifier_protein.pth`

#### Cell line classification

`CUDA_VISIBLE_DEVICES=0,1,2,3 python  -m torch.distributed.launch --nproc_per_node=4 run_end_to_end.py --config config_HPA_FOV.yaml --epochs 10 --balance True --num_classes 35 --train_cell_type True --train_protein False --train True --test True --classifier_state_dict results/HPA_FOV_classification/classifier_cells.pth`

### HPA single cells

#### Train / test division

`python divide_train_valid.py --config config_HPA_single_cells.yaml`

`python divide_train_valid.py --config config_HPA_single_cells.yaml --cells` 

#### Protein localization classification

`CUDA_VISIBLE_DEVICES=0,1,2,3 python  -m torch.distributed.launch --nproc_per_node=4 run_end_to_end.py --config config_HPA_single_cells.yaml --epochs 10 --balance True --num_classes 19 --train_cell_type False --train_protein True --train True --test True --classifier_state_dict results/HPA_single_cells_classification/classifier_protein.pth`

#### Cell line classification

`CUDA_VISIBLE_DEVICES=0,1,2,3 python  -m torch.distributed.launch --nproc_per_node=4 run_end_to_end.py --config config_HPA_single_cells.yaml --epochs 10 --balance True --num_classes 29 --train_cell_type True --train_protein False --train True --test True --classifier_state_dict results/HPA_single_cells_classification/classifier_cells.pth`

### WTC11 cell stage classification

Run `notebooks/WTC11_classifiers.ipynb`

### Cell Painting 

#### To preprocess and sphere the LINCS data

Run `notebooks/02-lincs-well-aggregation-sphering.ipynb`

#### To classify the MOA

[TBD]



## In addition to the code above, we release the following models, features, embeddings, and auxiliery files to facilitate the reproduction of the results:

## HPA FOV

### Model checkpoints

HPA_FOV_data/DINO_FOV_checkpoint.pth

### Baselines

HPA_FOV_data/Pretrained_base8_DINO_checkpoint.pth

### features

#### DINO features of HPA FOV
HPA_FOV_data/DINO_features_for_HPA_FOV.pth

#### Bestfitting features of HPA FOV
HPA_FOV_data/bestfitting_features_for_HPA_FOV.pth

### Embeddings

HPA_FOV_data/DINO_FOV_harmonized_embeddings.csv
HPA_FOV_data/DINO_FOV_embeddings.csv


### Misc

#### HPA FOV metadata
HPA_FOV_data/whole_images.csv

#### train / test divisions for protein localizations and cell line classification
HPA_FOV_data/cells_train_IDs.pth
HPA_FOV_data/cells_valid_IDs.pth
HPA_FOV_data/train_IDs.pth
HPA_FOV_data/valid_IDs.pth

#### HPA single cell kaggle protein localization competition, download from https://www.kaggle.com/competitions/human-protein-atlas-image-classification/leaderboard
HPA_FOV_data/human-protein-atlas-image-classification-publicleaderboard.csv

#### HPA cell line RNASeq data
HPA_FOV_data/rna_cellline.tsv

#### HPA FOV color visualization
HPA_FOV_data/whole_image_cell_color_indices.pth
HPA_FOV_data/whole_image_protein_color_indices.pth

### Classifiers

HPA_FOV_data/classifier_cells.pth
HPA_FOV_data/classifier_proteins.pth

## HPA single cells


### Model checkpoints

#### DINO model checkpoint
HPA_single_cells_data/DINO_single_cell_checkpoint.pth
HPA_single_cells_data/HPA_single_cell_model_checkpoint.pth

### Baselines

#### Pretrained DINO model checkpoint
HPA_single_cells_data/Pretrained_base16_DINO_checkpoint.pth

### features

#### DINO features of HPA single cells
HPA_single_cells_data/DINO_features_for_HPA_single_cells.pth

#### Dualhead features of HPA single cells
HPA_single_cells_data/dualhead_features_for_HPA_single_cells.pth

#### Pretrained DINO features of HPA single cells
HPA_single_cells_data/pretrained_DINO_features_for_HPA_single_cells.pth

### Embeddings

#### UMAP of HPA single cells features aggregated by FOV
HPA_single_cells_data/DINO_embedding_average_umap.csv

#### UMAP of HPA single cells features aggregated by FOV, harmonized
HPA_single_cells_data/DINO_harmonized_embedding_average_umap.csv

### Misc

#### HPA single cell kaggle protein localization competition, download from https://www.kaggle.com/competitions/hpa-single-cell-image-classification/leaderboard
HPA_single_cells_data/hpa-single-cell-image-classification-publicleaderboard.csv

#### HPA XML data
HPA_single_cells_data/XML_HPA.csv

#### UNIPROT interaction dataset 
HPA_single_cells_data/uniport_interactions.tsv

#### HPA single cells Metadata
HPA_single_cells_data/fixed_size_masked_single_cells_for_sc.csv

#### HPA gene heterogeneity annotated by experts
HPA_single_cells_data/gene_heterogeneity.tsv

#### single cell metadata with genetic information
HPA_single_cells_data/Master_scKaggle.csv

#### HPA single cell color visualization
HPA_single_cells_data/cell_color_indices.pth
HPA_single_cells_data/protein_color_indices.pth

### Classifiers

HPA_single_cells_data/classifier_cells.pth
HPA_single_cells_data/classifier_proteins.pth

## WTC11


### Model checkpoints

#### DINO model for WTC11 data
WTC11_data/model_3_channels_checkpoint.pth

### Baselines

#### Pretrained model for WTC11 data
WTC11_data/Pretrained_base16_DINO_checkpoint.pth

### features

#### DINO features for WTC11 data
WTC11_data/DINO_features_and_df.pth

#### Engineered features for WTC11 data
WTC11_data/engineered_features.pth

#### DINO Pretrained features for WTC11 data
WTC11_data/pretrained_features_and_df.pth
WTC11_data/pretrained_b_features.pth
WTC11_data/pretrained_g_features.pth
WTC11_data/pretrained_y_features.pth

### Embeddings

#### UMAP of WTC11 features aggregated by FOV
WTC11_data/Allen_3_channel_trained_embedding.pth

#### UMAP of WTC11 features aggregated by FOV, harmonized
WTC11_data/Allen_3_channel_trained_embedding_harmonized.pth

### Classifiers

#### Cell stage predictions for pretrained DINO model
WTC11_data/predictions_for_WTC11_pretrained_model.pth

#### Cell stage predictions for trained DINO model
WTC11_data/predictions_for_WTC11_trained_model.pth

### Misc

#### Cell stage predictions for XGBoost
WTC11_data/predictions_for_WTC11_xgb.pth

#### WTC11 metadata
WTC11_data/normalized_cell_df.csv

#### Train / test indices
WTC11_data/train_indices.pth
WTC11_data/test_indices.pth


## Cell Painting


### Model checkpoints
### Baselines
### features
### Embeddings
### Classifiers
### Misc
 
