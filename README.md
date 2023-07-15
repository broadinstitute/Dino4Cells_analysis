# DINO4Cells_analysis 

This repo will contain the code for reproducing the results and figures of the paper [Unbiased single-cell morphology with self-supervised vision transformers
](https://www.biorxiv.org/content/10.1101/2023.06.16.545359v1).

For the code to train the model, go to https://github.com/broadinstitute/Dino4Cells_analysis.

## Data

For the WTC11 data, access [https://zenodo.org/record/8061424](https://zenodo.org/record/8061424)
For the Cell Painting data, access [https://zenodo.org/record/8061428](https://zenodo.org/record/8061428)
For the HPA single cell data, access [https://zenodo.org/record/8061426](https://zenodo.org/record/8061426)
For the HPA FOV data, access [https://zenodo.org/record/8061392](https://zenodo.org/record/8061392)

## Installation

`pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116`

`pip install -r requirements.py`

Typical installation time: 10 minutes

## Classification

To train classifiers on the features extracted using DINO, please use the following pipeline for the different datasets and tasks:

For each dataset, we want to publish:

DINO model checkpoints
metadata
features
embeddings
classifiers
misc (data partition indices, preprocessed data, etc.)

Typical running time: 1 hour
The expected output of the code is shown inside the jupyter notebooks

# HPA FOV

## Model checkpoints

HPA_FOV_data/DINO_FOV_checkpoint.pth
HPA_FOV_data/densenet_model_batch.onnx
HPA_FOV_data/densenet_model.onnx

## metadata

HPA_FOV_data/whole_images.csv

## features

HPA_FOV_data/DINO_features_for_HPA_FOV.pth
HPA_FOV_data/bestfitting_features_for_HPA_FOV.pth
HPA_FOV_data/pretrained_DINO_features_for_HPA_FOV.pth

## Embeddings

HPA_FOV_data/DINO_FOV_harmonized_embeddings.csv
HPA_FOV_data/DINO_FOV_embeddings.csv

## Classifiers

HPA_FOV_data/classifier_cells.pth
HPA_FOV_data/classifier_proteins.pth

## Misc

### train / test divisions for protein localizations and cell line classification
HPA_FOV_data/cells_train_IDs.pth
HPA_FOV_data/cells_valid_IDs.pth
HPA_FOV_data/train_IDs.pth
HPA_FOV_data/valid_IDs.pth

### HPA single cell kaggle protein localization competition, download from https://www.kaggle.com/competitions/human-protein-atlas-image-classification/leaderboard
HPA_FOV_data/human-protein-atlas-image-classification-publicleaderboard.csv

### HPA cell line RNASeq data
HPA_FOV_data/rna_cellline.tsv

### HPA FOV color visualization
HPA_FOV_data/whole_image_cell_color_indices.pth
HPA_FOV_data/whole_image_protein_color_indices.pth

# HPA single cells

## Model checkpoints

HPA_single_cells_data/DINO_single_cell_checkpoint.pth
HPA_single_cells_data/HPA_single_cell_model_checkpoint.pth
HPA_single_cells_data/dualhead_config.json
HPA_single_cells_data/dualhead_matched_state.pth

## metadata

HPA_single_cells_data/fixed_size_masked_single_cells_for_sc.csv

## features

HPA_single_cells_data/DINO_features_for_HPA_single_cells.pth
HPA_single_cells_data/dualhead_features_for_HPA_single_cells.pth
HPA_single_cells_data/pretrained_DINO_features_for_HPA_single_cells.pth

## Embeddings

HPA_single_cells_data/DINO_embedding_average_umap.csv
HPA_single_cells_data/DINO_harmonized_embedding_average_umap.csv

## Classifiers

HPA_single_cells_data/classifier_cells.pth
HPA_single_cells_data/classifier_proteins.pth
## Misc

### HPA single cell kaggle protein localization competition, download from https://www.kaggle.com/competitions/hpa-single-cell-image-classification/leaderboard
HPA_single_cells_data/hpa-single-cell-image-classification-publicleaderboard.csv

### HPA XML data
HPA_single_cells_data/XML_HPA.csv

### UNIPROT interaction dataset
HPA_single_cells_data/uniport_interactions.tsv

### HPA gene heterogeneity annotated by experts
HPA_single_cells_data/gene_heterogeneity.tsv

### single cell metadata with genetic information
HPA_single_cells_data/Master_scKaggle.csv

### HPA single cell color visualization
HPA_single_cells_data/cell_color_indices.pth
HPA_single_cells_data/protein_color_indices.pth

# WTC11

## Model checkpoints

WTC11_data/DINO_checkpoint.pth

## metadata

WTC11_data/normalized_cell_df.csv

## features

WTC11_data/DINO_features_and_df.pth
WTC11_data/engineered_features.pth
WTC11_data/pretrained_features_and_df.pth

## Embeddings

WTC11_data/DINO_trained_embedding.pth
WTC11_data/DINO_trained_harmonized_embedding.pth
WTC11_data/pretrained_embedding.pth
WTC11_data/pretrained_harmonized_embedding.pth
WTC11_data/engineered_embedding.pth

## Classifiers

WTC11_data/predictions_for_WTC11_trained_model.pth
WTC11_data/predictions_for_WTC11_pretrained_model.pth
WTC11_data/predictions_for_WTC11_xgb.pth

## Misc

WTC11_data/train_indices.pth
WTC11_data/test_indices.pth

# Cell Painting

## DINO model checkpoints

Cell_Painting_data/DINO_cell_painting_base_checkpoint.pth
Cell_Painting_data/DINO_cell_painting_small_checkpoint.pth

## metadata and embeddings

Cell_Painting_data/LINCS_ViT_Small_Compressed_df_and_UMAP.csv
Cell_Painting_data/Combined_CP_df_and_UMAP.csv

## features

Code to calculate the PUMA results is in:
https://github.com/CaicedoLab/2023_Moshkov_NatComm


## misc (data partition indices, preprocessed data, etc.)

Cell_Painting_data/scaffold_median_python_dino.csv
