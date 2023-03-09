import json
import random
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence, pad_packed_sequence

## -- paths -- ##
paths = {
    'path_to_embeds'            : "data/recurrent_data/embeds",
    'openface_spk_1_path'       : "data/recurrent_data/openface_speaker_1_05_by_05",
    'openface_spk_2_path'       : "data/recurrent_data/openface_speaker_2_05_by_05",
    'openface_spk_1_dict_path'  : "data/recurrent_data/openface_speaker_1_dict_utt_to_frame.txt",
    'openface_spk_2_dict_path'  : "data/recurrent_data/openface_speaker_2_dict_utt_to_frame.txt",
    'opensmile_spk_1_path'      : "data/recurrent_data/opensmile_speaker_1_05_by_05",
    'opensmile_spk_2_path'      : "data/recurrent_data/opensmile_speaker_2_05_by_05",
    'opensmile_spk_1_dict_path' : "data/recurrent_data/opensmile_speaker_1_dict_utt_to_frame.txt",
    'opensmile_spk_2_dict_path' : "data/recurrent_data/opensmile_speaker_2_dict_utt_to_frame.txt"
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
        self.openface={}
        self.openface[1] = pd.read_feather(openface_spk_1_path)
        self.openface[2] = pd.read_feather(openface_spk_2_path)
        
        #import opensmile data
        self.opensmile={}
        self.opensmile[1] = pd.read_feather(opensmile_spk_1_path)
        self.opensmile[2] = pd.read_feather(opensmile_spk_2_path)

        #import dicts and filter them
        with open(openface_spk_1_dict_path, "r") as fp:
            openface_spk_1_dict = json.load(fp)
        with open(openface_spk_2_dict_path, "r") as fp:
            openface_spk_2_dict = json.load(fp)
        with open(opensmile_spk_1_dict_path, "r") as fp:
            opensmile_spk_1_dict = json.load(fp)
        with open(opensmile_spk_2_dict_path, "r") as fp:
            opensmile_spk_2_dict = json.load(fp)
        
        self.openface_dict={}
        self.opensmile_dict={}
        self.openface_dict[1] = filter_dict(openface_spk_1_dict)
        self.openface_dict[2] = filter_dict(openface_spk_2_dict)
        self.opensmile_dict[1] = filter_dict(opensmile_spk_1_dict)
        self.opensmile_dict[2] = filter_dict(opensmile_spk_2_dict)

        self._find_unused_index()

        #create useful params
        self.target_col = ['SD', 'QE', 'SV', 'PR', 'HD']
        self.openface_col = self.openface[1].iloc[:,3:].columns
        self.opensmile_col = self.opensmile[1].iloc[:,3:].columns
        self.class_weights = self.embeds[self.target_col].sum() / self.embeds.shape[0]
        self.openface_tensor = {
            1:torch.from_numpy(self.openface[1][self.openface_col].values).to(torch.float32),
            2:torch.from_numpy(self.openface[2][self.openface_col].values).to(torch.float32)
        }
        self.opensmile_tensor = {
            1:torch.from_numpy(self.opensmile[1][self.opensmile_col].values).to(torch.float32),
            2:torch.from_numpy(self.opensmile[2][self.opensmile_col].values).to(torch.float32)
        }
        self.target_tensor = torch.from_numpy(self.embeds[self.target_col].values).to(torch.float32)

        self._make_index()
        self._filter_indexes_before_stratify()

    def _find_unused_index(self):
        """We need to remove the index of embeds that are not present in all dicts"""
        of1 = set( self.openface_dict[1].keys() )
        of2 = set( self.openface_dict[2].keys() )
        os1 = set( self.opensmile_dict[1].keys() )
        os2 = set( self.opensmile_dict[2].keys() )
        self._unused_index = pd.Index(set(self.embeds.index).difference(set(int(i) for i in (of1.union(of2)).intersection(os1.union((os2))))))
        return None

    def _make_index(self):
        #practicity and remove unused_index
        df = self.embeds[self.target_col].loc[self.embeds.index.difference(self._unused_index)]

        self.indexes={}
        for cat in self.target_col:
            self.indexes[cat] = df.loc[df[cat]==1].index
        self.indexes['target'] = df.loc[df.sum(axis=1)>0].index
        self.indexes['none'] = df.index.difference(self.indexes['target'])

        names = ['SD', 'QE', 'SV', 'PR', 'HD', 'target', 'none']

        self.indexes[1]={}
        self.indexes[2]={}
        indexes1=self.embeds.loc[self.embeds.speaker=='1'].index
        indexes2=self.embeds.loc[self.embeds.speaker=='2'].index
        for n in names:
            self.indexes[1][n] = self.indexes[n].intersection(indexes1)
            self.indexes[2][n] = self.indexes[n].intersection(indexes2)
        return None

    def _filter_indexes_before_stratify(self):
        #make all combinations of feature
        combinations = self.embeds[self.target_col].astype(str).agg('-'.join, axis=1)
        #count them
        counts = combinations.value_counts()
        #filter them
        to_avoid = counts.loc[counts < 3].index
        self._to_avoid = combinations[combinations.isin(to_avoid)].index
        return None

    def stratified_train_test_split(self, feature = 'openface', speaker=1, test_size = .3,val_size=None, none_count=300):
        output={
            'targets':self.target_tensor
        }

        if not feature in ('openface', 'opensmile'):
            print('Choose feature in ("openface","opensmile")')
            raise

        if feature=='openface' :
            dic = self.openface_dict[speaker]
            output['features'] = self.openface_tensor[speaker]
        else :
            dic = self.opensmile_dict[speaker]
            output['features'] = self.opensmile_tensor[speaker]

        #indexes of categories to stratify
        target_index = self.indexes[speaker]['target']
        #none indexes
        none_index = pd.Index(random.sample(list(self.indexes[speaker]['none']),none_count))
        #constitute df to stratify
        df = self.embeds[self.target_col].loc[target_index.union(none_index).difference(self._to_avoid)]
        class_weights = torch.Tensor((df.sum()/df.shape[0]).values)
        train, test = train_test_split(df, test_size=test_size, stratify=df)
        output['test_dic']  = {k:v for (k,v) in dic.items() if int(k) in test.index.union(self._to_avoid)}

        if val_size :
            train, valid = train_test_split(train, test_size = val_size, stratify=train)
            
            output['train_dic'] = {k:v for (k,v) in dic.items() if int(k) in train.index}
            output['valid_dic'] = {k:v for (k,v) in dic.items() if int(k) in valid.index}
            return {'data' : output, 'class_weights':class_weights}

        
        output['train_dic'] = {k:v for (k,v) in dic.items() if int(k) in train.index}
        return {'data' : output, 'class_weights':class_weights}

#create dataset from dict, features_data, and targets_data
class dicDataset(Dataset):
    """
    dict has {index in targets_data : related indexes in features_data}
    features_data is a tensor of openface or opensmile
    targets_data is the tensor of targets (from embeds[feature_col])
    """
    def __init__(self, train_dic, features, targets, test_dic, valid_dic=None):
        self.train_dic = train_dic
        self.test_dic = test_dic
        self.valid_dic = valid_dic
        self.f = features
        self.t = targets
        self.keys = [int(i) for i in train_dic.keys()]
    
    def __len__(self):
        return len(self.train_dic)

    def __getitem__(self, idx):
        features_indexes = self.train_dic[str(self.keys[idx])]
        features = self.f[features_indexes, :]
        targets = self.t[self.keys[idx],:]
        return features, targets, len(features_indexes)

    def _get_test_item(self, idx):
        features_indexes = self.test_dic[str(idx)]
        features = self.f[features_indexes, :]
        return features, self.t[int(idx),:], len(features_indexes)

    def get_test(self):
        return pad_collate([self._get_test_item(idx) for idx in self.test_dic.keys()])

    def get_train(self):
        return pad_collate([self.__getitem__(idx) for idx in range(self.__len__())])

    def _get_valid_item(self, idx):
        features_indexes = self.valid_dic[str(idx)]
        features = self.f[features_indexes, :]
        return features, self.t[int(idx),:], len(features_indexes)

    def get_valid(self):
        if not self.valid_dic:
            print('No valid data')
            raise
        return pad_collate([self._get_valid_item(idx) for idx in self.valid_dic.keys()])

def pad_collate(batch):
    """Pads and packs a batch of items from dicDataset"""
    f,t,l = [*zip(*batch)]
    output = {
        'features':pack_padded_sequence(pad_sequence(f,batch_first=True), l, batch_first=True, enforce_sorted=False),
        'targets':torch.stack(t)
    }
    return output

class GRUModel(nn.Module):
    def __init__(self, input_dim=17, hidden_dim=8, layer_dim=3, output_dim=5, dropout_prob=.1):
        super(GRUModel, self).__init__()

        self.gru = nn.GRU(
            input_dim, hidden_dim, layer_dim, dropout=dropout_prob, batch_first = True
        )

        self.fc = nn.Linear(hidden_dim, output_dim)
        self.s = nn.Sigmoid()

    def forward(self, x_packed):

        x, hidden = self.gru(x_packed)

        x,l = pad_packed_sequence(x, batch_first=True)

        out = torch.stack([x[i][l[i]-1] for i in range(x.shape[0])])

        out = self.fc(out)
        out = self.s(out)

        return out