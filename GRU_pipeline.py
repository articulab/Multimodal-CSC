import json
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence

## -- paths -- ##
paths = {
    'path_to_embeds'            : "data/final/text/embeds",
    'openface_spk_1_path'       : "data/final/openface/openface_speaker_1_05_by_05",
    'openface_spk_2_path'       : "data/final/openface/openface_speaker_2_05_by_05",
    'openface_spk_1_dict_path'  : "data/final/openface/openface_speaker_1_dict_utt_to_frame.txt",
    'openface_spk_2_dict_path'  : "data/final/openface/openface_speaker_2_dict_utt_to_frame.txt",
    'opensmile_spk_1_path'      : "data/final/opensmile/opensmile_speaker_1_05_by_05",
    'opensmile_spk_2_path'      : "data/final/opensmile/opensmile_speaker_2_05_by_05",
    'opensmile_spk_1_dict_path' : "data/final/opensmile/opensmile_speaker_1_dict_utt_to_frame.txt",
    'opensmile_spk_2_dict_path' : "data/final/opensmile/opensmile_speaker_2_dict_utt_to_frame.txt"
}

def filter_dict(dic:dict, max_len=30)->dict:
    """Filters on 0<len(value)<max_len"""
    return {key:value for (key,value) in dic.items() if len(value) in range(1,max_len)}

def retrieve_frames(df, row):
    """Enables the creation of a dict similar to the ones used latter"""
    d = row.Dyad
    s = row.Session
    b = row.begin_time
    e = row.end_time
    return list(df.query("Dyad==@d & Session==@s & timestamp >= @b & timestamp < @e").index)

class DataHolder():
    def __init__(
        self,
        path_to_embeds,
        openface_spk_1_path,
        openface_spk_2_path,
        openface_spk_1_dict_path,
        openface_spk_2_dict_path,
        opensmile_spk_1_path,
        opensmile_spk_2_path,
        opensmile_spk_1_dict_path,
        opensmile_spk_2_dict_path
    ):
        #import embeds
        self.embeds = pd.read_feather(path_to_embeds)

        #import openface data
        self.openface_spk_1 = pd.read_feather(openface_spk_1_path)
        self.openface_spk_2 = pd.read_feather(openface_spk_2_path)
        
        #import opensmile data
        self.opensmile_spk_1 = pd.read_feather(opensmile_spk_1_path)
        self.opensmile_spk_2 = pd.read_feather(opensmile_spk_2_path)

        #import dicts and filter them
        with open(openface_spk_1_dict_path, "r") as fp:
            openface_spk_1_dict = json.load(fp)
        with open(openface_spk_2_dict_path, "r") as fp:
            openface_spk_2_dict = json.load(fp)
        with open(opensmile_spk_1_dict_path, "r") as fp:
            opensmile_spk_1_dict = json.load(fp)
        with open(opensmile_spk_2_dict_path, "r") as fp:
            opensmile_spk_2_dict = json.load(fp)

        self.openface_spk_1_dict = filter_dict(openface_spk_1_dict)
        self.openface_spk_2_dict = filter_dict(openface_spk_2_dict)
        self.opensmile_spk_1_dict = filter_dict(opensmile_spk_1_dict)
        self.opensmile_spk_2_dict = filter_dict(opensmile_spk_2_dict)

        self._find_unused_index()

        #create useful params
        self.target_col = ['SD', 'QE', 'SV', 'PR', 'HD']
        self.openface_col = self.openface_spk_1.iloc[:,3:].columns
        self.opensmile_col = self.opensmile_spk_1.iloc[:,3:].columns
        self.class_weights = self.embeds[self.target_col].sum() / self.embeds.shape[0]

    def _find_unused_index(self):
        """We need to remove the index of embeds that are not present in all dicts"""
        a = set( self.openface_spk_1_dict.keys() )
        b = set( self.openface_spk_2_dict.keys() )
        c = set( self.opensmile_spk_1_dict.keys() )
        d = set( self.opensmile_spk_2_dict.keys() )
        self._unused_index = pd.Index(set(self.embeds.index).difference(set(int(i) for i in (a.union(b)).intersection(c.union((d))))))
        return None

    def _filter_indexes_before_stratify(self, df):
        #make all combinations of feature
        combinations = df.astype(str).agg('-'.join, axis=1)
        #count them
        counts = combinations.value_counts()
        #filter them
        to_avoid = counts.loc[counts < 3].index
        return combinations[combinations.isin(to_avoid)].index

    def stratify_embeds(self, speaker=1, test_size=.3,val_size=None, none_count=300):
        embeds_col = self.embeds[self.target_col]
        if speaker:
            embeds_col = embeds_col.loc[self.embeds.speaker==str(speaker)]
        #useful indexes to stratify
        target_index = embeds_col.loc[embeds_col.sum(axis=1)>0].index
        #none indexes
        none_index = embeds_col.index.difference(target_index)
        selected_none_index = embeds_col.loc[none_index.difference(self._unused_index)].sample(none_count).index
        #find unique annoying rare combinations of features - they will be put in test
        annoying_indexes = self._filter_indexes_before_stratify(embeds_col.loc[target_index])
        #constitute df to stratify
        df = embeds_col.loc[target_index.union(selected_none_index).difference(annoying_indexes)]
        class_weights = df.sum()/df.shape[0]
        train, test = train_test_split(df, test_size=test_size, stratify=df)
        if val_size :
            train, val = train_test_split(train, test_size=val_size, stratify=train)
            return {
                'train':train.index,
                'valid':val.index,
                'test':test.index.union(annoying_indexes),
                'class_weights':class_weights}
        return {
            'train':train.index,
            'test':test.index.union(annoying_indexes),
            'class_weights':class_weights}

#create dataset from dict, features_data, and targets_data
class dicDataset(Dataset):
    """
    dict has {index in targets_data : related indexes in features_data}
    features_data is a tensor of openface or opensmile
    targets_data is the tensor of targets (from embeds[feature_col])
    """
    def __init__(self, dic, features_data, targets_data):
        self.dic = dic
        self.f = features_data
        self.t = targets_data
        self.keys = [int(i) for i in dic.keys()]
    
    def __len__(self):
        return len(self.dic)

    def __getitem__(self, idx):
        features_indexes = self.dic[str(self.keys[idx])]
        features = self.f[features_indexes, :]
        targets = self.t[self.keys[idx],:]
        return features, targets, len(features_indexes)

def pad_collate(batch):
    """Pads and packs a batch of items from dicDataset"""
    f,t,l = [*zip(*batch)]
    output = {
        'features':pack_padded_sequence(pad_sequence(f,batch_first=True), l, batch_first=True, enforce_sorted=False),
        'targets':torch.stack(t)
    }
    return output