import pandas as pd
import torch
import numpy as np
from utils.label_dict import (
    protein_to_num_full,
    protein_to_num_single_cells,
    cell_to_num_full,
    cell_to_num_single_cells,
)
from tqdm import tqdm
import argparse
import oyaml as yaml
import os
from pathlib import Path

parser = argparse.ArgumentParser("Get embeddings from model")
parser.add_argument("--config", type=str, default=".", help="path to config file")
parser.add_argument(
    "--output_prefix", type=str, default=None, help="path to config file"
)
parser.add_argument(
    "--cells",
    action="store_true",
)
args = parser.parse_args()
config = yaml.safe_load(open(args.config, "r"))

if config["classification"]["whole_images"]:
    labels = sorted(protein_to_num_full.keys())
else:
    labels = sorted(protein_to_num_single_cells.keys())

train_path = config["classification"]["averaged_train_path"]
valid_path = config["classification"]["averaged_valid_path"]
Path(train_path).parents[0].mkdir(exist_ok=True)

if args.output_prefix is None:
    feature_dir = config["embedding"]["output_path"]
else:
    feature_dir = args.output_prefix

all_features, all_proteins, all_cell_lines, all_IDs, df = torch.load(feature_dir)
if args.cells:
    train_IDs = torch.load("HPA_FOV_data/cells_train_IDs.pth")
    valid_IDs = torch.load("HPA_FOV_data/cells_valid_IDs.pth")
else:
    train_IDs = torch.load("HPA_FOV_data/train_IDs.pth")
    valid_IDs = torch.load("HPA_FOV_data/valid_IDs.pth")

if config["classification"]["whole_images"]:
    protein_matrix = df[labels].values.astype(int)
    averaged_features = []
    new_IDs = np.array(pd.DataFrame(all_IDs)[0].unique())
    all_IDs = np.array(all_IDs)
    protein_localizations = []
    cell_lines = []
    ID_to_indices_dict = pd.DataFrame(all_IDs, columns=["ID"]).groupby("ID").groups
    for ID in tqdm(list(ID_to_indices_dict.keys())):
        indices = ID_to_indices_dict[ID]
        averaged_features.append(all_features[indices, :].mean(axis=0))
        cell_lines.append(all_cell_lines[indices[0]])
        protein_localizations.append(protein_matrix[indices[0]])
    averaged_features = torch.stack(averaged_features)
    IDs = new_IDs
    protein_localizations = np.stack(protein_localizations)
else:
    averaged_features = all_features
    IDs = np.array(all_IDs)
    cell_lines = np.array(all_cell_lines)
    sorted_indices = np.argsort(IDs)

    averaged_features = averaged_features[sorted_indices, :]
    IDs = IDs[sorted_indices]
    cell_lines = cell_lines[sorted_indices]
    sc_indices = np.where(pd.DataFrame(IDs, columns=["IDs"]).IDs.isin(df.ID))[0]

    averaged_features = averaged_features[sc_indices, :]
    IDs = IDs[sc_indices]
    cell_lines = cell_lines[sc_indices]

    df = df.sort_values(by="ID").reset_index()

    protein_matrix = df[labels].values.astype(int)
    protein_localizations = protein_matrix

for g, path in zip([train_IDs, valid_IDs], [train_path, valid_path]):
    indices_averaged = np.where(pd.DataFrame(IDs, columns=["ID"]).ID.isin(g))[0]
    torch.save(
        (
            averaged_features[indices_averaged],
            protein_localizations[indices_averaged],
            np.array(cell_lines)[indices_averaged],
            np.array(IDs)[indices_averaged],
        ),
        path,
    )
