import json
import random
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence, pad_packed_sequence
import torch.nn.functional as F
from tqdm import tqdm

SEED = 42

random.seed(SEED)

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
        opensmile_spk_2_dict_path,
        none_as_class = False
    ):
        #import embeds
        random.seed(SEED)
        self.embeds = pd.read_feather(path_to_embeds)
        self.none_as_class = none_as_class
        if none_as_class:
            self.target_col = ['SD', 'QE', 'SV', 'PR', 'HD', "N"]
            self.embeds["N"] = (1 - self.embeds[['SD', 'QE', 'SV', 'PR', 'HD']].sum(axis = 1)).replace(-1, 0)
            self.embeds_col = self.embeds.iloc[:, 13:].columns
        else:
            self.target_col = ['SD', 'QE', 'SV', 'PR', 'HD']
            self.embeds_col = self.embeds.iloc[:, 12:].columns
        
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
        self.openface_col = self.openface[1].iloc[:,3:].columns
        self.opensmile_col = self.opensmile[1].iloc[:,3:].columns
        # self.embeds_col = self.embeds.iloc[:, 13:].columns
        self.class_weights = self.embeds[self.target_col].sum() / self.embeds.shape[0]
        self.openface_tensor = {
            1:torch.from_numpy(self.openface[1][self.openface_col].values).to(torch.float32),
            2:torch.from_numpy(self.openface[2][self.openface_col].values).to(torch.float32)
        }
        self.opensmile_tensor = {
            1:torch.from_numpy(self.opensmile[1][self.opensmile_col].values).to(torch.float32),
            2:torch.from_numpy(self.opensmile[2][self.opensmile_col].values).to(torch.float32)
        }
        self.embeds_tensor = torch.from_numpy(self.embeds[self.embeds_col].values).to(torch.float32)
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
        if self.none_as_class:
            names = ['SD', 'QE', 'SV', 'PR', 'HD', 'N', 'target', 'none']
            self.indexes['none'] = df[df["N"] == 1].index
            self.indexes['target'] = df.index.difference(self.indexes['none'])
        else:
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
            'targets':self.target_tensor,
            'embeds' :self.embeds_tensor
        }

        if not feature in ('openface', 'opensmile', 'multimodal'):
            print('Choose feature in ("openface","opensmile")')
            raise
        
        elif feature=="multimodal":
            dic_openface = self.openface_dict[speaker]
            dic_opensmile = self.opensmile_dict[speaker]
            output["features_opensmile"] = self.opensmile_tensor[speaker]
            output["features_openface"] = self.openface_tensor[speaker]

            #indexes of categories to stratify
            target_index = self.indexes[speaker]['target']
            #none indexes
            none_index = pd.Index(random.sample(list(self.indexes[speaker]['none']), none_count))
            #constitute df to stratify
            df = self.embeds[self.target_col].loc[target_index.union(none_index).difference(self._to_avoid)]
            class_weights = torch.Tensor((df.sum()/df.shape[0]).values)
            train, test = train_test_split(df, test_size=test_size, stratify=df, random_state = SEED)

            output['test_dic_openface']  = {k:v for (k,v) in dic_openface.items() if int(k) in test.index.union(self._to_avoid)}
            output['test_dic_opensmile']  = {k:v for (k,v) in dic_opensmile.items() if int(k) in test.index.union(self._to_avoid)}
            if val_size :
                train, valid = train_test_split(train, test_size = val_size, stratify=train, random_state = SEED)
                
                output['train_dic_openface'] = {k:v for (k,v) in dic_openface.items() if int(k) in train.index}
                output['valid_dic_openface'] = {k:v for (k,v) in dic_openface.items() if int(k) in valid.index}

                output['train_dic_opensmile'] = {k:v for (k,v) in dic_opensmile.items() if int(k) in train.index}
                output['valid_dic_opensmile'] = {k:v for (k,v) in dic_opensmile.items() if int(k) in valid.index}
                return {'data' : output, 'class_weights':class_weights}

            
            output['train_dic_openface'] = {k:v for (k,v) in dic_openface.items() if int(k) in train.index}
            output['train_dic_opensmile'] = {k:v for (k,v) in dic_opensmile.items() if int(k) in train.index}
            return {'data' : output, 'class_weights':class_weights}


        else:
            if feature=='openface' :
                dic = self.openface_dict[speaker]
                output['features'] = self.openface_tensor[speaker]

            elif feature=="opensmile":
                dic = self.opensmile_dict[speaker]
                output['features'] = self.opensmile_tensor[speaker]

            #indexes of categories to stratify
            target_index = self.indexes[speaker]['target']
            #none indexes
            none_index = pd.Index(random.sample(list(self.indexes[speaker]['none']),none_count))
            #constitute df to stratify
            df = self.embeds[self.target_col].loc[target_index.union(none_index).difference(self._to_avoid)]
            class_weights = torch.Tensor((df.sum()/df.shape[0]).values)
            train, test = train_test_split(df, test_size=test_size, stratify=df, random_state = SEED)
            output['test_dic']  = {k:v for (k,v) in dic.items() if int(k) in test.index.union(self._to_avoid)}

            if val_size :
                train, valid = train_test_split(train, test_size = val_size, stratify=train, random_state = SEED)
                
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

#create dataset from dict, features_data, and targets_data
class MultiModalDicDataset(Dataset):
    """
    dict has {index in targets_data : related indexes in features_data}
    features_data is a tensor of openface or opensmile
    targets_data is the tensor of targets (from embeds[feature_col])
    """
    def __init__(self, train_dic_openface, train_dic_opensmile, test_dic_openface, test_dic_opensmile, features_openface, features_opensmile, embeds, targets, valid_dic_openface=None, valid_dic_opensmile=None):
        self.train_dic_openface, self.train_dic_opensmile = train_dic_openface, train_dic_opensmile
        self.test_dic_openface, self.test_dic_opensmile = test_dic_openface, test_dic_opensmile
        self.valid_dic_openface, self.valid_dic_opensmile = valid_dic_openface, valid_dic_opensmile
        self.f_openface = features_openface
        self.f_opensmile = features_opensmile
        self.t = targets
        self.e = embeds
        self.keys_openface = [int(i) for i in train_dic_openface.keys()]
        self.keys_opensmile = [int(i) for i in train_dic_opensmile.keys()]
    
    def __len__(self):
        return len(self.train_dic_openface)

    def __getitem__(self, idx):
        features_indexes_openface = self.train_dic_openface[str(self.keys_openface[idx])]
        features_indexes_opensmile = self.train_dic_opensmile[str(self.keys_opensmile[idx])]

        features_openface = self.f_openface[features_indexes_openface, :]
        features_opensmile = self.f_opensmile[features_indexes_opensmile, :]

        targets = self.t[self.keys_openface[idx],:]
        embeds = self.e[self.keys_openface[idx], :]
        return features_openface, features_opensmile, embeds, targets, len(features_indexes_openface), len(features_indexes_opensmile)
        # return {"features_openface":features_openface, 
        #         "features_opensmile": features_opensmile, 
        #         "targets":targets, 
        #         "len_openface":len(features_indexes_openface),
        #         "len_opensmile":len(features_indexes_opensmile)}

    def _get_test_item(self, idx):
        of_features_indexes = self.test_dic_openface[str(idx)]
        os_features_indexes = self.test_dic_opensmile[str(idx)]
        openface_features = self.f_openface[of_features_indexes, :]
        opensmile_features = self.f_opensmile[os_features_indexes, :]

        return openface_features, opensmile_features, self.e[int(idx)], self.t[int(idx),:], len(of_features_indexes), len(os_features_indexes)
        # return {"openface_features":openface_features,
        #         "opensmile_features":opensmile_features,
        #         "targets":self.t[int(idx),:],
        #         "len_openface_features":len(of_features_indexes),
        #         "len_opensmile_features":len(os_features_indexes)
        #     }

    def get_test(self):
        return pad_collate([self._get_test_item(idx) for idx in self.test_dic.keys()])

    def get_train(self):
        return pad_collate([self.__getitem__(idx) for idx in range(self.__len__())])

    def _get_valid_item(self, idx):
        of_features_indexes = self.valid_dic_openface[str(idx)]
        os_features_indexes = self.valid_dic_opensmile[str(idx)]
        openface_features = self.f_openface[of_features_indexes, :]
        opensmile_features = self.f_opensmile[os_features_indexes, :]

        return openface_features, opensmile_features, self.e[nt(idx)], self.t[int(idx),:], len(of_features_indexes), len(os_features_indexes)

    def get_valid(self):
        if not self.valid_dic:
            print('No valid data')
            raise
        return pad_collate([self._get_valid_item(idx) for idx in self.valid_dic.keys()])

def pad_collate(batch):
    """Pads and packs a batch of items from dicDataset"""
    of_f, os_f, e, t, of_l, os_l = [*zip(*batch)]
    output = {
        'features_of':pack_padded_sequence(pad_sequence(of_f,batch_first=True), of_l, batch_first=True, enforce_sorted=False),
        'features_os':pack_padded_sequence(pad_sequence(os_f,batch_first=True), os_l, batch_first=True, enforce_sorted=False),
        'embeds':torch.stack(e),
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

        x, l = pad_packed_sequence(x, batch_first=True)

        out = torch.stack([x[i][l[i]-1] for i in range(x.shape[0])])

        out = self.fc(out)
        out = self.s(out)

        return out

def eval_on_val(model, val_loader_1, val_loader_2, criterion, modality = None):
    if model.__class__.__name__ == "GRUMultiModal":
        model.eval()
        tot_loss=0.0
        for i, batch in enumerate(val_loader_1):
            features_of, features_os, embeds, targets = batch['features_of'], batch['features_os'], batch['embeds'], batch['targets']
            with torch.no_grad():
                pred = model(embeds, features_os, features_of)
                loss = criterion(pred, targets)
                tot_loss += loss / pred.shape[0]
        for i, batch in enumerate(val_loader_2):
            features_of, features_os, embeds, targets = batch['features_of'], batch['features_os'], batch['embeds'], batch['targets']
            with torch.no_grad():
                pred = model(embeds, features_os, features_of)
                loss = criterion(pred, targets)
                tot_loss += loss / pred.shape[0]
        return (tot_loss / (len(val_loader_1) + len(val_loader_2)))
    elif model.__class__.__name__ == "BertClassif":
        model.eval()
        tot_loss=0.0
        for i, batch in enumerate(val_loader_1):
            embeds, targets = batch['embeds'], batch['targets']
            with torch.no_grad():
                pred = model(embeds)
                loss = criterion(pred, targets)
                tot_loss += loss / pred.shape[0]
        for i, batch in enumerate(val_loader_2):
            embeds, targets = batch['embeds'], batch['targets']
            with torch.no_grad():
                pred = model(embeds)
                loss = criterion(pred, targets)
                tot_loss += loss / pred.shape[0]
        return (tot_loss / (len(val_loader_1) + len(val_loader_2)))
    elif model.__class__.__name__ == "GRUBiModal":
        tot_loss=0.0
        if modality == "audio":
            model.eval()
            for i, batch in enumerate(val_loader_1):
                _, features_os, embeds, targets = batch['features_of'], batch['features_os'], batch['embeds'], batch['targets']
                with torch.no_grad():
                    pred = model(embeds, features_os)
                    loss = criterion(pred, targets)
                    tot_loss += loss / pred.shape[0]
            for i, batch in enumerate(val_loader_2):
                _, features_os, embeds, targets = batch['features_of'], batch['features_os'], batch['embeds'], batch['targets']
                with torch.no_grad():
                    pred = model(embeds, features_os)
                    loss = criterion(pred, targets)
                    tot_loss += loss / pred.shape[0]
            return (tot_loss / (len(val_loader_1) + len(val_loader_2)))
        elif modality == "video":
            model.eval()
            for i, batch in enumerate(val_loader_1):
                features_of, _, embeds, targets = batch['features_of'], batch['features_os'], batch['embeds'], batch['targets']
                with torch.no_grad():
                    pred = model(embeds, features_of)
                    loss = criterion(pred, targets)
                    tot_loss += loss / pred.shape[0]
            for i, batch in enumerate(val_loader_2):
                features_of, _, embeds, targets = batch['features_of'], batch['features_os'], batch['embeds'], batch['targets']
                with torch.no_grad():
                    pred = model(embeds, features_of)
                    loss = criterion(pred, targets)
                    tot_loss += loss / pred.shape[0]
            return (tot_loss / (len(val_loader_1) + len(val_loader_2)))


def train_one_epoch(epoch, model, criterion, dataloader_1, dataloader_2, val_loader_1, val_loader_2, hist_train_loss, hist_val_loss, stagnation, best_vloss, opt = None, modality = None, save_dir = ""):
    if not (opt is None):
        optimizer = opt
    if model.__class__.__name__ == "GRUMultiModal":
        model.train(True)

        epoch_loss = 0.0
        for i, batch in enumerate(dataloader_1):
            features_of, features_os, embeds, targets = batch['features_of'], batch['features_os'], batch['embeds'], batch['targets']
            pred = model(embeds, features_os, features_of)
            loss = criterion(pred, targets)

            epoch_loss += loss / pred.shape[0]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        for i, batch in enumerate(dataloader_2):
            features_of, features_os, embeds, targets = batch['features_of'], batch['features_os'], batch['embeds'], batch['targets']
            pred = model(embeds, features_os, features_of)
            loss = criterion(pred, targets)

            epoch_loss += loss / pred.shape[0]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        epoch_loss = epoch_loss / (len(dataloader_1) + len(dataloader_2))
        hist_train_loss = hist_train_loss + [epoch_loss]
        model.train(False)
        val_loss = eval_on_val(model, val_loader_1, val_loader_2, criterion)
        hist_val_loss = hist_val_loss + [val_loss]
        if epoch % 30 == 0:
            stagnation += 1
            print("EPOCH {}:".format(epoch + 1))
            tqdm.write(f"================\nTraining epoch {epoch} :\nTrain loss = {1000 * epoch_loss}, Val loss = {1000 * val_loss}\n================")
            if val_loss < best_vloss:
                best_vloss = val_loss
                torch.save(model.state_dict(), save_dir + "MultiModalBert")
                stagnation = 0
        return hist_train_loss, hist_val_loss, stagnation, best_vloss
        
    elif model.__class__.__name__ == "BertClassif":
        model.train(True)

        epoch_loss = 0.0

        for i, batch in enumerate(dataloader_1):
            embeds, targets = batch['embeds'], batch['targets']
            pred = model(embeds)
            loss = criterion(pred, targets)

            epoch_loss += loss / pred.shape[0]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        for i, batch in enumerate(dataloader_2):
            embeds, targets = batch['embeds'], batch['targets']
            pred = model(embeds)
            loss = criterion(pred, targets)

            epoch_loss += loss / pred.shape[0]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        epoch_loss = epoch_loss / (len(dataloader_1) + len(dataloader_2))
        hist_train_loss = hist_train_loss + [epoch_loss]
        model.train(False)
        val_loss = eval_on_val(model, val_loader_1, val_loader_2, criterion)
        hist_val_loss = hist_val_loss + [val_loss]
        if epoch % 30 == 0:
            stagnation += 1
            print("EPOCH {}:".format(epoch + 1))
            tqdm.write(f"================\nTraining epoch {epoch} :\nTrain loss = {1000 * epoch_loss:.4f}, Val loss = {1000 * val_loss:.4f}\n================")
            if val_loss < best_vloss:
                best_vloss = val_loss
                torch.save(model.state_dict(), save_dir + "BertClassif")
                stagnation = 0
        return hist_train_loss, hist_val_loss, stagnation, best_vloss
    elif model.__class__.__name__ == "GRUBiModal":
        model.train(True)
        epoch_loss = 0.0
        if modality == "video":
            for i, batch in enumerate(dataloader_1):
                features_of, _, embeds, targets = batch['features_of'], batch['features_os'], batch['embeds'], batch['targets']
                pred = model(embeds, features_of)
                loss = criterion(pred, targets)

                epoch_loss += loss / pred.shape[0]

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            for i, batch in enumerate(dataloader_2):
                features_of, _, embeds, targets = batch['features_of'], batch['features_os'], batch['embeds'], batch['targets']
                pred = model(embeds, features_of)
                loss = criterion(pred, targets)

                epoch_loss += loss / pred.shape[0]

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        elif modality == "audio":
            for i, batch in enumerate(dataloader_1):
                _, features_os, embeds, targets = batch['features_of'], batch['features_os'], batch['embeds'], batch['targets']
                pred = model(embeds, features_os)
                loss = criterion(pred, targets)

                epoch_loss += loss / pred.shape[0]

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            for i, batch in enumerate(dataloader_2):
                _, features_os, embeds, targets = batch['features_of'], batch['features_os'], batch['embeds'], batch['targets']
                pred = model(embeds, features_os)
                loss = criterion(pred, targets)

                epoch_loss += loss / pred.shape[0]

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        epoch_loss = epoch_loss / (len(dataloader_1) + len(dataloader_2))
        hist_train_loss = hist_train_loss + [epoch_loss]
        model.train(False)
        val_loss = eval_on_val(model, val_loader_1, val_loader_2, criterion, modality)
        hist_val_loss = hist_val_loss + [val_loss]
        if epoch % 30 == 0:
            stagnation += 1
            print("EPOCH {}:".format(epoch + 1))
            tqdm.write(f"================\nTraining epoch {epoch} :\nTrain loss = {1000 * epoch_loss}, Val loss = {1000 * val_loss}\n================")
            if val_loss < best_vloss:
                best_vloss = val_loss
                torch.save(model.state_dict(), save_dir + f"Bert{modality}Bimodal")
                stagnation = 0
        return hist_train_loss, hist_val_loss, stagnation, best_vloss