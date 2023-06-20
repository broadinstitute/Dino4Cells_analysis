import os
import argparse
import pandas as pd
import numpy as np
import re
from os import walk
from collections import Counter
import random
import shutil

def sort_moas(cpds_moa):
    """
    Sort MOAs based on the number of compounds that are attributed to them in ASCENDING order.
    This is HIGHLY Required before performing the compounds split into train & test.
    """
    cpds_moa_split = {cpd:cpds_moa[cpd].split('|') for cpd in cpds_moa}
    moa_listts = [moa for moa_lt in cpds_moa_split.values() for moa in moa_lt]
    moa_count_dict = {ky:val for ky,val in sorted(Counter(moa_listts).items(),key=lambda item: item[1])}
    moa_lists = list(moa_count_dict.keys())
    return moa_lists

def create_cpd_moa_df(cpds_moa):
    """
    Create a dataframe that comprises of compounds with their corresponding MOAs, including three additional 
    columns: "test", "train" & "marked" which are needed for the compounds split.
    """
    cpds_moa_split = {cpd:cpds_moa[cpd].split('|') for cpd in cpds_moa}
    df_pert_cpds_moas = pd.DataFrame([(key, moa) for key,moa_list in cpds_moa_split.items() for moa in moa_list], 
                                     columns = ['pert_iname', 'moa'])
    df_pert_cpds_moas['train'] = False
    df_pert_cpds_moas['test'] = False
    df_pert_cpds_moas['marked'] = df_pert_cpds_moas['train'] | df_pert_cpds_moas['test']
    return df_pert_cpds_moas

def split_cpds_moas(cpd_moas_dict, train_ratio=0.8, test_ratio=0.2):
    """
    This function splits compounds into test & train data based on the number of MOAs that are attributed to them,
    i.e. if the MOAs are present in just one compound, the compounds for those specific MOAs are given to only the 
    train data, but if present in more than one compound, the compounds for that MOA are divided into Train/Test 
    split based on the test/train ratio.
    
    - This function was extracted from https://rpubs.com/shantanu/lincs_split_moa
    and then refactored to Python
    
    Args:
         cpd_moas_dict: Dictionary comprises of compounds as the keys and their respective MOAs (Mechanism of actions)
         as the values
         train_ratio: A decimal value that represent what percent of the data should be given to the train set
         test_ratio: A decimal value that represent what percent of the data should be given to the test set
    
    Returns:
            df: pandas dataframe containing compounds, MOAs and three new boolean columns (Train, Test, Marked)
            indicating whether a compound is in Train or Test dataset.
    """
    ##preliminary funcs
    moa_list = sort_moas(cpd_moas_dict)
    df = create_cpd_moa_df(cpd_moas_dict)
    
    random.seed(333)
    for moa in moa_list:
        df_moa = df[df['moa'] == moa].reset_index(drop=True)
        no_cpd = df_moa.shape[0]
        
        if no_cpd == 1:
            n_trn, n_tst = 1, 0
        else:
            n_trn, n_tst = np.floor(no_cpd*train_ratio), np.ceil(no_cpd*test_ratio),
            
        n_tst_mk = sum(df_moa.test)
        n_trn_mk = sum(df_moa.train)
        moa_mk = df_moa[df_moa['marked']].copy()
        moa_not_mk = df_moa[~df_moa['marked']].copy()
        trn_needed = int(n_trn - n_trn_mk)
        tst_needed = int(n_tst - n_tst_mk)
        n_cpds_needed = trn_needed + tst_needed
        ##print(moa, df_moa.shape[0], moa_not_mk.shape[0], n_cpds_needed, trn_needed, tst_needed)
    
        trn_needed = max(trn_needed, 0)
        tst_needed = max(tst_needed, 0)
        trn_flg = list(np.concatenate((np.tile(True, trn_needed), np.tile(False, tst_needed))))
        trn_flg = random.sample(trn_flg, n_cpds_needed)
        tst_flg = [not boolean for boolean in trn_flg]
        moa_not_mk.train = trn_flg
        moa_not_mk.test = tst_flg
        if moa_not_mk.shape[0] > 0:
            moa_not_mk.marked = True
        df_moa = pd.concat([moa_not_mk, moa_mk], axis=0, ignore_index=True)
        df_other_moa = df[df['moa'] != moa].reset_index(drop=True)
        df_otrs_mk = df_other_moa[df_other_moa['marked']].reset_index(drop=True)
        df_otrs_not_mk= df_other_moa[~df_other_moa['marked']].reset_index(drop=True)
        df_otrs_not_mk = df_otrs_not_mk[['pert_iname', 'moa']].merge(moa_not_mk.drop(['moa'], axis=1),
                                                                     on=['pert_iname'], how='left').fillna(False)
        
        df = pd.concat([df_moa, df_otrs_mk, df_otrs_not_mk], axis=0, ignore_index=True)
        df[['train', 'test']] = df[['train', 'test']].apply(lambda x: x.astype(bool))
        
    return df