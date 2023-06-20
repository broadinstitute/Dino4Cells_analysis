import numpy as np
import pandas as pd
import sys
import os
import random
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers, regularizers, Sequential, Model, backend, optimizers, metrics, losses
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import tensorflow as tf
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.metrics import precision_recall_curve,log_loss
from sklearn.metrics import average_precision_score,roc_auc_score

import warnings
warnings.filterwarnings('ignore')

def split_data(df_train, df_test, metadata_cols, target_cols):
    """
    Split train and test data into two parts: 
    1st part(x): comprises only the numeric phenotypic/morphologic features in the data
    2nd part(y): comprises only the MOA target labels
    """
    df_train_y = df_train[target_cols].copy()
    df_train_x = df_train.drop(target_cols, axis = 1).copy()
    df_test_y = df_test[target_cols].copy()
    df_test_x = df_test.drop(target_cols, axis = 1).copy()
    df_train_x.drop(metadata_cols, axis = 1, inplace = True)
    df_test_x.drop(metadata_cols, axis = 1, inplace = True)
    
    return df_train_x, df_train_y, df_test_x, df_test_y

def check_if_shuffle_data(shuffle, model_file_name=None, model_dir_name=None, trn_pred_name=None, tst_pred_name=None):
    """Rename directories if you are training the model with Shuffle data"""
    dir_name_list = [model_file_name, model_dir_name, trn_pred_name, tst_pred_name]
    for idx, x in enumerate(dir_name_list):
        if shuffle:
            dir_name_list[idx] = f"{x}_shuffle"
    return dir_name_list

def drug_stratification(df, nfold, target_cols,col_name,cpd_freq_num=20):
    """
    This function performs multi-label stratification on the compounds/drugs found
    in the train dataset. Here, because the distribution of drugs is highly imbalanced
    i.e. some drugs appear a lot more frequently than others, we divide the drugs/compounds
    into two categories based on how frequent they appear in the train data using the 
    'cpd_freq_num' argument.
    
    Individual drugs that are said to be less frequent i.e. below the cpd_freq_num are all assigned
    individually to a specific fold, whereas drugs that are said to be frequent i.e. above the 
    cpd_freq_num are evenly distributed among all folds.
    
    The intuition behind this approach is that drugs that appear very frequently are also expected 
    to be frequent in the test dataset so they are not assigned to their own fold while drugs that 
    are rare belong to the same fold. 
    For more info: https://www.kaggle.com/c/lish-moa/discussion/195195
    
    Args:
            df: train data - pandas dataframe containing all drugs and features.
            
            nfold: Number of K-fold to be used for multi-label stratification
            
            target_cols: A list of all target MOA (Mechanism of actions) labels that will predicted.
            
            col_name: A string that indicates the replicate ids/replicate name.
            
            cpd_freq_num: A number that is used to divide drugs/compounds into two categories i.e.
            first category consist of highly frequent drugs in the train data and the second one
            consist of rarely seen/less frequent drugs in the train data
    
    Returns:
            df: train data - pandas dataframe with a new column called 'fold', which wil be used for cross-validation
            during model training
    """
    drug_value_ct = df['pert_iname'].value_counts()
    drug_vc1 = drug_value_ct.loc[drug_value_ct <= cpd_freq_num].index.sort_values()
    drug_vc2 = drug_value_ct.loc[drug_value_ct > cpd_freq_num].index.sort_values()
    dct1 = {}; dct2 = {}
    skf = MultilabelStratifiedKFold(n_splits=nfold, shuffle=True, random_state=33)
    df_drug_vc1 = df.groupby('pert_iname')[target_cols].mean().loc[drug_vc1]

    # STRATIFY DRUGS X OR LESS based on each specific drug/compound
    for fold,(idxT,idxV) in enumerate(skf.split(df_drug_vc1,df_drug_vc1[target_cols])):
        drugs_fold = {drugs:fold for drugs in df_drug_vc1.index[idxV].values}
        dct1.update(drugs_fold)
        
    # STRATIFY DRUGS X OR MORE based on the drug's replicates
    skf = MultilabelStratifiedKFold(n_splits=nfold, shuffle=True)
    df_drug_vc2 = df.loc[df.pert_iname.isin(drug_vc2)].reset_index(drop=True)
    if df_drug_vc2.shape[0]>0:
        for fold,(idxT,idxV) in enumerate(skf.split(df_drug_vc2,df_drug_vc2[target_cols])):
            drugs_fold = {drugs:fold for drugs in df_drug_vc2[col_name][idxV].values}
            dct2.update(drugs_fold)

    ##fold column
    df['fold'] = df.pert_iname.map(dct1)
    df.loc[df.fold.isna(),'fold'] = df.loc[df.fold.isna(),col_name].map(dct2)
    df['fold'] = df['fold'].astype(int)
    
    return df
    
def normalize(trn, val, test):
    """
    Performs z-score/standard normalization on the train, test and validation data. The StandardScaler
    is fitted on the train data, and transformed on test and validation data.
    
    Args:
            trn: train data - pandas dataframe.
            val: validation data - pandas dataframe.
            test: test data - pandas dataframe.
    
    Returns:
            trn_norm: normalized train data - pandas dataframe.
            val_norm: normalized validation - pandas dataframe.
            test_norm: normalized test data - pandas dataframe.
    """
    norm_model = StandardScaler()
    trn_norm = pd.DataFrame(norm_model.fit_transform(trn),index = trn.index,columns = trn.columns)
    val_norm = pd.DataFrame(norm_model.transform(val),index = val.index,columns = val.columns)
    tst_norm = pd.DataFrame(norm_model.transform(test),index = test.index,columns = test.columns)
    return trn_norm, val_norm, tst_norm
    
def pca_features(train,validation,test,no_of_comp):
    """
    This function performs PCA (Principal Component Analysis) transformation on the train, 
    test and validation data. The PCA is fitted on the train data, and transformed on test 
    and validation data.
    
    Args:
            train: train data - pandas dataframe.
            validation: validation data - pandas dataframe.
            test: test data - pandas dataframe.
            no_of_components: Number of principal components (PCs) to extract from PCA.
    
    Returns:
            train_pca: train data - pandas dataframe with only PCs.
            validation_pca: validation data - pandas dataframe with only PCs.
            test_pca: test data - pandas dataframe with only PCs.
    """
    pca = PCA(n_components=no_of_comp, random_state=42)
    feat_new = ['pca'+ str(i) for i in range(no_of_comp)]
    train_pca = pd.DataFrame(pca.fit_transform(train),columns=feat_new)
    validation_pca = pd.DataFrame(pca.transform(validation),columns=feat_new)
    test_pca = pd.DataFrame(pca.transform(test),columns=feat_new)
    return(train_pca, validation_pca, test_pca)
    
def preprocess(fold, df_train, df_train_x, df_train_y, df_test_x, no_of_comp):
    """
    This function split the train data into a K-fold subset, performs normalization on
    them and engineer new features including PCA on the train, test and validation data and 
    finally concatenate new PCs with the existing dataframes.
    
    Args:
            fold: fold value.
            df_train: train data - pandas dataframe.
            df_train_x: train data - pandas dataframe with only phenotypic/morphological features.
            df_train_y: train data - pandas dataframe with only the Mechanism of actions (MOAs) target labels.
            df_test_x: test data - pandas dataframe with only phenotypic/morphological features.
            no_of_comp: Number of principal components (PCs) to extract from PCA.
    
    Returns:
            x_fold_train: K-fold train data with only phenotypic/morphological features and PCs - numpy array.
            y_fold_train: K-fold train data with only the Mechanism of actions (MOAs) target labels - numpy array.
            x_fold_val: K-fold validation data with only phenotypic/morphological features and PCs - numpy array.
            y_fold_val: K-fold validation data with only the Mechanism of actions (MOAs) target labels - numpy array.
            df_test_x: test data - pandas dataframe with only phenotypic/morphological features and PCs.
            val_idx: A list of the K-fold validation indices from the train data
            x_fold_train.shape[1]: Number of phenotypic/morphological features and PCs 
    """
    trn_idx = df_train[df_train['fold'] != fold].index
    val_idx = df_train[df_train['fold'] == fold].index
    
    x_fold_train = df_train_x.loc[trn_idx].reset_index(drop=True).copy()
    y_fold_train = df_train_y.loc[trn_idx].reset_index(drop=True).copy()
    x_fold_val = df_train_x.loc[val_idx].reset_index(drop=True).copy()
    y_fold_val = df_train_y.loc[val_idx].reset_index(drop=True).copy()
    df_test_x_copy = df_test_x.copy()
    
    x_trn_mean = pd.DataFrame(x_fold_train.mean(axis=1), columns = ['mean_of_features'])
    x_val_mean = pd.DataFrame(x_fold_val.mean(axis=1), columns = ['mean_of_features'])
    x_tst_mean = pd.DataFrame(df_test_x_copy.mean(axis=1), columns = ['mean_of_features'])
    
    x_fold_train = pd.concat([x_fold_train, x_trn_mean], axis = 1)
    x_fold_val = pd.concat([x_fold_val, x_val_mean], axis = 1)
    df_test_x_copy = pd.concat([df_test_x_copy, x_tst_mean], axis = 1)
    
    ### -- normalize using standardization ----
    x_fold_train, x_fold_val, df_test_x_copy = normalize(x_fold_train, x_fold_val, df_test_x_copy)
    
    ### --- add pca components to the original features ----
    trn_fold_pca,val_fold_pca,test_pca = pca_features(x_fold_train,x_fold_val,df_test_x_copy,no_of_comp)
    x_fold_train = pd.concat([x_fold_train,trn_fold_pca],axis = 1)
    x_fold_val = pd.concat([x_fold_val,val_fold_pca],axis = 1)
    df_test_x_copy  = pd.concat([df_test_x_copy,test_pca],axis = 1)
    
    return x_fold_train.values,y_fold_train.values, x_fold_val.values, y_fold_val.values, df_test_x_copy.values, \
val_idx, x_fold_train.shape[1]

def logloss(y_true, y_preds):
    label_smoothing_alpha = 0.0005
    P_MIN = label_smoothing_alpha
    P_MAX = 1 - P_MIN
    y_preds = tf.clip_by_value(y_preds, P_MIN, P_MAX)
    return -backend.mean(y_true * backend.log(y_preds) + (1 - y_true) * backend.log(1 - y_preds))

def mean_logloss(y_preds, y_true):
    logloss = (1 - y_true) * np.log(1 - y_preds + 1e-15) + y_true * np.log(y_preds + 1e-15)
    return np.mean(-logloss)

def model_eval_results(df_trn_y, oofs, df_tst_y, df_preds, target_cols):
    """
    This function prints out the model evaluation results from the train and test predictions.
    The evaluation metrics used in assessing the performance of the models are: ROC AUC score,
    log loss and Precision-Recall AUC score
    """
    eval_metrics = ['log loss', 'ROC AUC score', 'PR-AUC/Average_precision_score',]
    print('\n','-' * 10, 'Train data prediction results', '-' * 10)
    print(f'{eval_metrics[0]}:', log_loss(np.ravel(df_trn_y), np.ravel(oofs)))
    print(f'{eval_metrics[1]}:', roc_auc_score(df_trn_y.values,oofs, average='micro'))
    print(f'{eval_metrics[2]}:', average_precision_score(df_trn_y,oofs, average="micro"))
    
    ###test prediction results
    print('\n','-' * 10, 'Test data prediction results', '-' * 10)
    print(f'{eval_metrics[0]}:', log_loss(np.ravel(df_tst_y), np.ravel(df_preds)))
#     print(f'{eval_metrics[1]}:', roc_auc_score(df_tst_y.values,df_preds.values, average='macro'))
    print(f'{eval_metrics[2]}:', average_precision_score(df_tst_y.values, df_preds.values, average="micro"))
    
def save_to_csv(df, path, file_name, compress=None):
    """saves dataframes to csv"""
    
    if not os.path.exists(path):
        os.mkdir(path)
    
    df.to_csv(os.path.join(path, file_name), index=False, compression=compress)
