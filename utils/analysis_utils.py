import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import cm
import cuml
import numpy as np
import torch
from pathlib import Path
import mantel
from sklearn.cross_decomposition import CCA
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist, cdist
from scipy.spatial.distance import squareform
import pandas as pd
from sklearn import decomposition
from sklearn.metrics import average_precision_score
import matplotlib.ticker as mtick
import matplotlib.patches as patches
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances
from tqdm import tqdm
from utils.label_dict import protein_to_num_full
from utils.label_dict import hierarchical_organization_single_cell_low_level, hierarchical_organization_single_cell_high_level
from utils.label_dict import hierarchical_organization_whole_image_low_level, hierarchical_organization_whole_image_high_level

cmap = cm.nipy_spectral

def get_embeddings(features_to_fit, features_to_transform, labels=None):
    scaled_features_to_fit = features_to_fit
    scaled_features_to_transform = features_to_transform
    umap_unique = cuml.UMAP(init='spectral', metric='euclidean', min_dist=0.1, n_neighbors=15, n_components=2, spread=4, output_type="numpy", verbose=False, n_epochs=2600, transform_queue_size=20, random_state=42)
    umap_unique = umap_unique.fit(scaled_features_to_fit.numpy(), y=labels)
    transformed_features = umap_unique.transform(scaled_features_to_transform.numpy())
    return umap_unique, 0, 0, transformed_features, scaled_features_to_transform

def plot_UMAP(df, labels, embedding, title, filename, color_indices):
    mat, labels = get_col_matrix(df, labels)
    plt.figure(figsize=(10,10), facecolor='white', dpi=300)
    plt.axis(False)
    plt.scatter(embedding[:, 0],
                embedding[:, 1],
                s=0.01,
                label='All data',
                color='grey'
               )
    for i, ind in enumerate(np.argsort(mat.sum(axis=0))[::-1]):
        indices = np.where((mat[:, ind] == 1) & (mat.sum(axis=1) == 1))[0]
        plt.scatter(embedding[indices, 0],
                    embedding[indices, 1],
                    s=0.01,
                    label=labels[ind],
                    color=cmap(color_indices[ind] / mat.shape[1])
                   )

    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.title(title, fontsize=15)
    lgnd = plt.legend(bbox_to_anchor=(1,1), frameon=False)
    for h in lgnd.legendHandles:
        h._sizes = [30]
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(filename, format='pdf')

def get_col_matrix(df, labels):
    if len(labels) == 1:
        values = df[labels[0]]
        unique_values = sorted(np.unique(values))
        mat = np.zeros((len(df), len(unique_values)))
        for ind, value in enumerate(unique_values):
            mat[np.where(values == value)[0], ind] = 1
        columns = unique_values
    else:
        mat = df[sorted(labels)].values.astype(int)
        columns = sorted(labels)
    return mat, columns

def get_averaged_features(df, features, labels, sort=True):
    mat, columns = get_col_matrix(df, labels)
    averaged_features = []
    for key in range(len(columns)):
        indices = np.where((mat[:,key] == 1) & (mat.sum(axis=1) == 1))
        averaged_features.append(features[indices].mean(axis=0))
    averaged_features = torch.stack(averaged_features)
    return averaged_features, columns

def scale(features, features_mean=None, features_std=None):
    if features_mean is None:
        features_mean = features.mean(axis=0)
    if features_std is None:
        features_std = features.std(axis=0)
    transformed_features = (features - features_mean) / (features_std + 0.00001)
    return transformed_features, features_mean, features_std



def get_col_matrix(df, labels):
    if len(labels) == 1:
        values = df[labels[0]]
        unique_values = sorted(np.unique(values))
        mat = np.zeros((len(df), len(unique_values)))
        for ind, value in enumerate(unique_values):
            mat[np.where(values == value)[0], ind] = 1
        columns = unique_values
    else:
        mat = df[sorted(labels)].values.astype(int)
        columns = sorted(labels)
    return mat, columns

def get_averaged_features(df, features, labels, sort=True):
    mat, columns = get_col_matrix(df, labels)
    averaged_features = []
    for key in range(len(columns)):
        indices = np.where((mat[:,key] == 1) & (mat.sum(axis=1) == 1))
        averaged_features.append(features[indices].mean(axis=0))
    averaged_features = torch.stack(averaged_features)
    return averaged_features, columns

def create_cell_comparison(features, protein_localizations, cell_lines, IDs, df, rna, savedir):
    averaged_features, cell_columns = get_averaged_features(df, features, ['cell_type'], sort=True)
    tpms = []
    for cell_line in cell_columns:
        df = rna[rna["Cell line"] == cell_line]
        tpms.append(df.TPM.to_numpy())
    tpms = np.array(tpms)
    scaled_tpms, features_mean, features_std = scale(tpms)
    scaled_averaged_features, features_mean, features_std = scale(averaged_features)

    pca = decomposition.PCA(n_components=10)
    scaled_tpms = pca.fit_transform(scaled_tpms)
    scaled_averaged_features = pca.fit_transform(scaled_averaged_features)
    print(sum(pca.explained_variance_ratio_))

    reduced_rna_matrix = 1 - squareform(pdist(scaled_tpms, metric='cosine'))
    reduced_averaged_features = 1 - squareform(pdist(scaled_averaged_features, metric='cosine'))
    plt.figure()
    ground_truth_Z = linkage(reduced_rna_matrix)
    dn = dendrogram(ground_truth_Z, labels = cell_columns)
    reduced_rna_matrix = reduced_rna_matrix[dn['leaves'], :][:, dn['leaves']]
    reduced_averaged_features = reduced_averaged_features[dn['leaves'], :][:, dn['leaves']]

    columns = dn['ivl']
    plt.figure(figsize=(5,5), dpi=300)
    plt.imshow(reduced_rna_matrix, cmap='Blues')
    plt.title('RNASeq distance matrix')
    plt.xticks(range(len(columns)), columns, rotation=90)
    plt.yticks(range(len(columns)), columns)
    print('')
    plt.savefig(f'{savedir}/HPA_FOV_RNA_similarity.pdf', format='pdf')

    plt.figure(figsize=(5,5), dpi=300)
    plt.imshow(reduced_averaged_features, cmap='Blues')
    plt.title('DINO similarity matrix')
    plt.xticks(range(len(columns)), columns, rotation=90)
    plt.yticks(range(len(columns)), columns)
    print('')
    plt.savefig(f'{savedir}/HPA_FOV_DINO_cell_similarity.pdf', format='pdf')

    for i in range(len(reduced_averaged_features)):
        reduced_averaged_features[i,i] = 0
        reduced_rna_matrix[i,i] = 0

    mantel_test = mantel.test(squareform(reduced_averaged_features), squareform(reduced_rna_matrix))
    print(f"cell line / RNASeq mantel test: {mantel_test}")
    rnaseq_mantel_test = mantel_test[0]

    ca = CCA()
    ca.fit(scaled_tpms, scaled_averaged_features)
    X_c, Y_c = ca.transform(scaled_tpms, scaled_averaged_features)
    cc_res = pd.DataFrame({"CCX_1":X_c[:, 0],
                           "CCY_1":Y_c[:, 0],
                           "CCX_2":X_c[:, 1],
                           "CCY_2":Y_c[:, 1],})

    cc_distance_matrix = cdist(X_c, Y_c, metric='euclidean')

    k = 1
    top_1_accuracy = np.mean((np.tile(np.arange(cc_distance_matrix.shape[0])[:,np.newaxis], k) == np.argsort(cc_distance_matrix, axis=0)[0:k,:].T).sum(axis=1))
    print(f"protein CC knn, top-1: {top_1_accuracy}")
    k = 1

    cc_results = (np.tile(np.arange(cc_distance_matrix.shape[0])[:,np.newaxis], k) == np.argsort(cc_distance_matrix, axis=0)[0:k,:].T).sum(axis=1)
    print(cc_results)

    cc_res = pd.DataFrame({"CCX_1":X_c[:, 0],
                           "CCY_1":Y_c[:, 0],
                           "CCX_2":X_c[:, 1],
                           "CCY_2":Y_c[:, 1],})

    fig, ax = plt.subplots(figsize=(5,5), dpi=300)
    c = [cmap(i / len(cell_columns)) for i in range(len(cell_columns))]

    for i in range(len(X_c)):
        linecolor = 'lightgrey'
        linewidth = 3
        plt.plot([X_c[i,0], Y_c[i, 0]],
                 [X_c[i,1], Y_c[i, 1]],
                 c=linecolor,
                 linewidth=linewidth,
                zorder=1)
        plt.scatter(X_c[i,0], X_c[i, 1], color='red', marker='o', s=15, label='', zorder=2)
        plt.scatter(Y_c[i,0], Y_c[i, 1], color='blue', marker='o', s=15, zorder=2)

    plt.xlabel('CC 1')
    plt.ylabel('CC 2')

    ax2 = ax.twinx()
    ax2.scatter(np.NaN, np.NaN, color='red', label='RNASeq', marker = 'o')
    ax2.scatter(np.NaN, np.NaN, color='blue', label='DINO features', marker = 'o')
    ax2.get_yaxis().set_visible(False)
    plt.savefig(f'{savedir}/HPA_FOV_CC.pdf', format='pdf')
    return (rnaseq_mantel_test, top_1_accuracy)

def create_protein_hierarchy(features, protein_localizations, cell_lines, IDs, df, savedir):
    labels = protein_to_num_full
    low_level_labels = hierarchical_organization_whole_image_low_level
    high_level_labels = hierarchical_organization_whole_image_high_level

    scaled_features, features_mean, features_std = scale(features)
    protein_averaged_features, protein_columns = get_averaged_features(df, torch.Tensor(scaled_features), labels)
    protein_distance_matrix = 1 - squareform(pdist(protein_averaged_features, metric='cosine'))
    ground_truth_distance_matrix = np.zeros(protein_distance_matrix.shape)
    for g in high_level_labels:
        inner_indices = []
        for l in g:
            index = np.where(np.array(protein_columns) == l)[0]
            if len(index) > 0:
                inner_indices.append(index[0])
        for i in inner_indices:
            for j in inner_indices:
                ground_truth_distance_matrix[i][j] = 0.5
    for g in low_level_labels:
        inner_indices = []
        for l in g:
            index = np.where(np.array(protein_columns) == l)[0]
            if len(index) > 0:
                inner_indices.append(index[0])
        for i in inner_indices:
            for j in inner_indices:
                ground_truth_distance_matrix[i][j] = 1
    ground_truth_Z = linkage(ground_truth_distance_matrix)

    plt.figure()
    dn = dendrogram(ground_truth_Z, labels = protein_columns)
    ground_truth_distance_matrix = ground_truth_distance_matrix[dn['leaves'], :][:, dn['leaves']]
    protein_distance_matrix = protein_distance_matrix[dn['leaves'], :][:, dn['leaves']]
    ground_truth_Z = linkage(ground_truth_distance_matrix)
    high_level_clusters = fcluster(ground_truth_Z, criterion='distance', t=2)
    low_level_clusters = fcluster(ground_truth_Z, criterion='distance', t=0.5)
    high_level_groups = pd.DataFrame(high_level_clusters).groupby(0).groups
    low_level_groups = pd.DataFrame(low_level_clusters).groupby(0).groups
    high_level_groups = {k : [high_level_groups[k].min(), high_level_groups[k].max()] for k in high_level_groups.keys()}
    low_level_groups = {k : [low_level_groups[k].min(), low_level_groups[k].max()] for k in low_level_groups.keys()}

    columns = dn['ivl']
    plt.figure(figsize=(5,5), dpi=300)
    plt.imshow(ground_truth_distance_matrix, cmap='Blues')
    plt.title('Protein ground truth matrix')
    plt.xticks(range(len(columns)), columns, rotation=90)
    plt.yticks(range(len(columns)), columns)
    print('')
    plt.savefig(f'{savedir}/HPA_FOV_Protein_ground_truth.pdf', format='pdf')

    plt.figure()
    fig, ax = plt.subplots(figsize=(5,5), dpi=300)
    plt.imshow(protein_distance_matrix, cmap='Blues')
    plt.title('DINO similarity matrix')
    plt.xticks(range(len(columns)), columns, rotation=90)
    plt.yticks(range(len(columns)), columns)
    for r in high_level_groups.keys():
        min_index, max_index = high_level_groups[r]
        rect = patches.Rectangle((min_index - 0.5, min_index - 0.5),
                                 max_index - min_index + 1,
                                 max_index - min_index + 1,
                                 linewidth=4,
                                 edgecolor='#00ff00',
                                 facecolor='none')
        ax.add_patch(rect)
    for r in low_level_groups.keys():
        min_index, max_index = low_level_groups[r]
        rect = patches.Rectangle((min_index - 0.5, min_index - 0.5),
                                 max_index - min_index + 1,
                                 max_index - min_index + 1,
                                 linewidth=4,
                                 edgecolor='#ff0000',
                                 facecolor='none')
        ax.add_patch(rect)
    print('')
    plt.savefig(f'{savedir}/HPA_FOV_DINO_protein_similarity.pdf', format='pdf')

    for i in range(len(protein_distance_matrix)):
        protein_distance_matrix[i,i] = 0
        ground_truth_distance_matrix[i,i] = 0

    mantel_test = mantel.test(squareform(protein_distance_matrix), squareform(ground_truth_distance_matrix))
    print(f"protein hierarchy mantel test: {mantel_test}")
    protein_hierarchy_mantel_test = mantel_test[0]
    return protein_hierarchy_mantel_test

import matplotlib
font = {'size'   : 5}
matplotlib.rc('font', **font)

def get_gene_heterogeneity_enrichement(feature_files, df_master_file, gene_heterogeneity_file, heterogenous_type='HPA_variable', condition=None, use_embedding=False):
    enrichments_conditioned_on_gene = []
    enrichments_conditioned_on_image = []
    genes_sorted_conditioned_on_image = []
    genes_sorted_conditioned_on_gene = []
    merged_dfs = []
    sorted_metrics_conditioned_on_image = []
    sorted_metrics_conditioned_on_gene = []
    images_sorted_conditioned_on_image = []
    average_precisions_condition_on_gene = []
    average_precisions_condition_on_image = []
    for feature_file in feature_files:
        enrichment, merged_df, genes_sorted, _, sorted_metric_conditioned_on_gene, _ = find_gene_enrichment(feature_file, df_master_file, gene_heterogeneity_file, heterogenous_type = heterogenous_type, condition=condition, use_embedding=use_embedding, granularity='gene')
        sorted_metrics_conditioned_on_gene.append(sorted_metric_conditioned_on_gene)
        genes_sorted_conditioned_on_gene.append(genes_sorted)
        enrichments_conditioned_on_gene.append(enrichment)
        gene_ground_truth = pd.merge(pd.DataFrame(genes_sorted, columns=['Gene']), merged_df[['Gene','HPA_variable']].drop_duplicates(), on='Gene')['HPA_variable'].apply(lambda x:pd.isnull(x) == False).astype(int).values
        gene_predictions = np.where(np.isnan(sorted_metric_conditioned_on_gene), 0, sorted_metric_conditioned_on_gene)
        average_precisions_condition_on_gene.append(average_precision_score(gene_ground_truth, gene_predictions))

        enrichment, merged_df, genes_sorted, images_sorted, sorted_metric_conditioned_on_gene, sorted_metric_conditioned_on_image = find_gene_enrichment(feature_file, df_master_file, gene_heterogeneity_file, heterogenous_type = heterogenous_type, condition=condition, use_embedding=use_embedding, granularity='image')
        sorted_metrics_conditioned_on_image.append(sorted_metric_conditioned_on_gene)
        genes_sorted_conditioned_on_image.append(genes_sorted)
        enrichments_conditioned_on_image.append(enrichment)
        gene_ground_truth = pd.merge(pd.DataFrame(genes_sorted, columns=['Gene']), merged_df[['Gene','HPA_variable']].drop_duplicates(), on='Gene')['HPA_variable'].apply(lambda x:pd.isnull(x) == False).astype(int).values
        gene_predictions = np.where(np.isnan(sorted_metric_conditioned_on_gene), 0, sorted_metric_conditioned_on_gene)
        average_precisions_condition_on_image.append(average_precision_score(gene_ground_truth, gene_predictions))

        merged_dfs.append(merged_df)
        images_sorted_conditioned_on_image.append(images_sorted)
    return (enrichments_conditioned_on_gene, enrichments_conditioned_on_image, merged_dfs, genes_sorted_conditioned_on_gene, genes_sorted_conditioned_on_image, sorted_metrics_conditioned_on_gene, sorted_metrics_conditioned_on_image, images_sorted_conditioned_on_image, average_precisions_condition_on_gene, average_precisions_condition_on_image)

def plot_gene_heterogeneity_enrichement(labels, enrichments, enrichments_image, heterogenous_type, gene_filename, image_filename):
    colors = ['#529de5','#d32c23','#55c87b',]
    if len(enrichments) > 0:
        fig, ax = plt.subplots(1,1,figsize=(3.5,2.5), facecolor='white', dpi=300)
        for ind, label in enumerate(labels):
#             plt.plot(enrichments[ind], label=label)
            plt.plot((np.array(enrichments[ind][10:510]) * 100).astype(int), label=label, color=colors[ind])
        ax.yaxis.set_major_formatter('{x:.0f}')
        plt.legend(frameon=False, fontsize=7.5)
        ax.set_xlabel('Number of genes\nsorted by feature variance', fontsize=10)
        ax.set_ylabel('% Heterogeneous Genes', fontsize=10)
        # ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        plt.xticks(range(0, 501, 100), range(10, 511, 100), fontsize=10)
        plt.yticks(fontsize=10)
        print('')
        plt.title(f'Heterogeneous gene ranking ({heterogenous_type})')
        plt.savefig(gene_filename, format='pdf')
    if len(enrichments_image) > 0:
        fig, ax = plt.subplots(1,1,figsize=(3.5,2.5), facecolor='white', dpi=300)
        for ind, label in enumerate(labels):
#             plt.plot(enrichments_image[ind], label=label)
            plt.plot((np.array(enrichments_image[ind][10:510]) * 100).astype(int), label=label, color=colors[ind])
        ax.yaxis.set_major_formatter('{x:.0f}')
        plt.legend(frameon=False, fontsize=7.5)
        ax.set_xlabel('Number of genes\nsorted by image heterogeneity', fontsize=10)
        ax.set_ylabel('% Heterogeneous Genes', fontsize=10)
        plt.xticks(range(0, 501, 100), range(10, 511, 100), fontsize=10)
        plt.yticks(fontsize=10)

        print('')
        plt.title(f'Heterogeneous gene ranking ({heterogenous_type})')
        plt.savefig(image_filename, format='pdf')



def get_heirarchical_clustering(df, features, labels, zero_diagonal=True, metric='similarity'):
    averaged_features, columns = get_averaged_features(df, features, labels)
    if metric == 'similarity':
        distance_matrix = cosine_similarity(averaged_features, averaged_features)
    else:
        distance_matrix = cosine_distances(averaged_features, averaged_features)

    Z = linkage(distance_matrix)
    plt.figure(figsize=(20,5))
    dn = dendrogram(Z, labels = columns)
    plt.xticks(rotation=45)
    distance_matrix = distance_matrix[dn['leaves'], :][:, dn['leaves']]
    columns = dn['ivl']
    plt.figure(figsize=(10,10))
    if zero_diagonal:
        for i in range(len(distance_matrix)):
            distance_matrix[i,i] = 0
    plt.imshow(distance_matrix)
    plt.xticks(range(len(columns)), columns, rotation=90)
    plt.yticks(range(len(columns)), columns)
    return distance_matrix, columns, Z

def get_heterogeneousity_per_whole_image(df, features, indices, metric='distance', verbose=False):
    cells_per_ID = df.iloc[indices].groupby('ID').groups
    mean_distances_per_ID = []
    for ID in tqdm(sorted(cells_per_ID.keys()), disable=verbose == False):
        if metric == 'distance':
            distance_matrix = cosine_distances(features[cells_per_ID[ID]],
                                                features[cells_per_ID[ID]])
            distances = np.triu(distance_matrix)[np.triu_indices(distance_matrix.shape[0], k=1)]
            if len(distances) == 0: continue
            mean_distances_per_ID.append(distances.mean())
        elif metric == 'std':
            mean_distances_per_ID.append(features[cells_per_ID[ID]].std(axis=0).mean().item())
    return mean_distances_per_ID, sorted(cells_per_ID.keys())

def get_heterogeneousity_per_gene(merged_df, features, indices, metric='distance', verbose=False):
    cells_per_gene = merged_df.iloc[indices].groupby(['Gene']).groups
    keys = sorted(cells_per_gene.keys())
    mean_distances_per_ID = []
    for gene in tqdm(keys):
        if metric == 'distance':
            distance_matrix = cosine_distances(features[cells_per_gene[gene]],
                                                features[cells_per_gene[gene]])
            distances = np.triu(distance_matrix)[np.triu_indices(distance_matrix.shape[0], k=1)]
            if len(distances) == 0: continue
            mean_distances_per_ID.append(distances.mean())
        elif metric == 'std':
            mean_distances_per_ID.append(features[cells_per_gene[gene]].std(axis=0).mean().item())
    return mean_distances_per_ID, keys

def get_heterogeneity_df(df, df_master_file, gene_heterogeneity_file):
    df_master = pd.read_csv(df_master_file)
    df = pd.merge(df, df_master[['ID','gene']], on='ID')
    df['Gene'] = df['gene']
    proto_heterogeneity_df = pd.read_csv(gene_heterogeneity_file, delimiter='\t')
    proto_heterogeneity_df['v_spatial'] = proto_heterogeneity_df['Single-cell variation spatial']
    proto_heterogeneity_df['v_intensity'] = proto_heterogeneity_df['Single-cell variation intensity']
    proto_heterogeneity_df['HPA_variable'] = np.nan
    proto_heterogeneity_df.loc[(pd.isnull(proto_heterogeneity_df['v_intensity']) == False),'v_intensity'] = True
    proto_heterogeneity_df.loc[(pd.isnull(proto_heterogeneity_df['v_intensity'])),'v_intensity'] = False
    proto_heterogeneity_df.loc[(pd.isnull(proto_heterogeneity_df['v_spatial']) == False),'v_spatial'] = True
    proto_heterogeneity_df.loc[(pd.isnull(proto_heterogeneity_df['v_spatial'])),'v_spatial'] = False
    proto_heterogeneity_df.loc[(proto_heterogeneity_df['v_spatial']) | (proto_heterogeneity_df['v_intensity']), 'HPA_variable'] = True
    heterogeneity_df = pd.merge(proto_heterogeneity_df, df, on='Gene')[['Gene','ID','v_spatial','v_intensity','HPA_variable', 'cell_type']]
    return heterogeneity_df

def find_gene_enrichment(feature_file, df_master_file, gene_heterogeneity_file, heterogenous_type = 'HPA_variable', granularity='gene', condition = None, use_embedding=False):
    features, protein_localizations, cell_lines, IDs, df = torch.load(feature_file)
    heterogeneity_df = get_heterogeneity_df(df, df_master_file, gene_heterogeneity_file)
    merged_df = pd.merge(heterogeneity_df, df, on='ID', how='right').drop_duplicates()
    merged_df.loc[pd.isnull(merged_df['Gene']),'Gene'] = ''
    merged_df = merged_df.reset_index()
    merged_df['original_index'] = merged_df.index.values
    merged_df['cell_type'] = merged_df.cell_type_x
    original_indices = merged_df.original_index.values
    gene_to_ind = merged_df[['Gene','original_index']].groupby('Gene').groups
    gene_to_ind = {k : original_indices[gene_to_ind[k]] for k in gene_to_ind.keys()}
    merged_df[pd.isnull(merged_df.Gene)].Gene = ''

    if condition == 'U-2 OS':
        features = features[np.where(merged_df.cell_type == 'U-2 OS')[0]]
        merged_df = merged_df[merged_df.cell_type == 'U-2 OS'].reset_index()
    else:
        merged_df = merged_df.reset_index()

    if use_embedding:
        umap_reducer, features_mean, features_std, embedding, scaled_features = get_embeddings(torch.Tensor(features), torch.Tensor(features))
        scaled_features = embedding
    else:
        scaled_features, features_mean, features_std = scale(features)
        scaled_features = scaled_features.numpy()
    indices = merged_df.index.values
    merged_df['original_index'] = merged_df.index.values

    if granularity == 'gene':
        mean_stds, sorted_genes = get_heterogeneousity_per_gene(merged_df, scaled_features, indices, metric='std', verbose=False)
    elif granularity == 'image':
        mean_stds, sorted_images = get_heterogeneousity_per_whole_image(merged_df, scaled_features, indices, metric='std', verbose=False)

    metric = mean_stds
    sorted_distances = np.array(metric)[np.argsort(metric)[::-1]]
    if granularity == 'gene':
        genes_sorted = np.array(sorted_genes)[np.argsort(metric)[::-1]]
        enrichment_raw = pd.DataFrame(genes_sorted)[0].isin(merged_df[merged_df[heterogenous_type] == True].Gene.unique()).values
        enrichment = []
        for ind in tqdm(range(1, len(genes_sorted), 1)):
            enrichment.append(enrichment_raw[:ind].mean())
        images_sorted = None
        sorted_distances_per_image = None
        sorted_distances_per_gene = sorted_distances
    elif granularity == 'image':
        images_sorted = np.array(sorted_images)[np.argsort(metric)[::-1]]
        merged_df['image_rank'] = merged_df.ID.map(dict(zip(images_sorted, range(len(images_sorted)))))
        merged_df['image_score'] = merged_df.ID.map(dict(zip(images_sorted, sorted_distances)))
        genes_and_image_ranks = merged_df.groupby('Gene')['image_rank'].min()
        genes_and_image_scores = merged_df.groupby('Gene')['image_score'].max()
        genes_ranked_by_images_scores = genes_and_image_scores.values
        genes_ranked_by_images_rank = genes_and_image_ranks.values
        genes_ranked_by_images_names = genes_and_image_ranks.index.values
        genes_sorted = genes_ranked_by_images_names[np.argsort(genes_ranked_by_images_rank)]

        enrichment_raw = pd.DataFrame(genes_sorted)[0].isin(merged_df[merged_df[heterogenous_type] == True].Gene.unique()).values
        enrichment = []
        for ind in tqdm(range(1, len(genes_sorted), 1)):
            enrichment.append(enrichment_raw[:ind].mean())
        sorted_distances_per_image = sorted_distances
        sorted_distances_per_gene = np.array(genes_ranked_by_images_scores)[np.argsort(genes_ranked_by_images_scores)[::-1]]

    return enrichment, merged_df, genes_sorted, images_sorted, sorted_distances_per_gene, sorted_distances_per_image

