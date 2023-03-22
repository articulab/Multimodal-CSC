import json
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, precision_score, accuracy_score, f1_score
from sklearn.feature_selection import chi2

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence, pad_packed_sequence

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
    return {key:value for (key,value) in dic.items() if len(value) in range(1,max_len)} # can be changed (key cond etc)

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
        #import embeds and targets
        self.embeds = pd.read_feather(path_to_embeds)
        self.embeds_col = self.embeds.iloc[:,12:].columns
        self.embeds_tensors = torch.from_numpy(self.embeds[self.embeds_col].values).to(torch.float32)
        self.target_col = ['SD', 'QE', 'SV', 'PR', 'HD']
        self.target_tensor = torch.from_numpy(self.embeds[self.target_col].values).to(torch.float32)

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
        self._make_dyad_session_dict()

        #create useful params
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

        self._make_index()
        self._filter_indexes_before_stratify()

    def _make_dyad_session_dict(self):
        d_s = [ (dyad, session, speaker) for speaker in ('1','2') for session in self.embeds.Session.unique() for dyad in self.embeds.Dyad.unique() ]
        def func(tup):
            d = tup[0]
            s = tup[1]
            sp = tup[2]
            return self.embeds.query(f"Dyad==@d & Session==@s & speaker==@sp").index.difference(self._unused_index)
        self.dyad_session_dict = { tup : func(tup) for tup in d_s if len(func(tup))>0}
        return None

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

    def stratified_train_test_split(self, feature = 'openface', speaker=1, test_size=.3,val_size=None, none_count=None, none_prop=.3):
        output={
            'targets':self.target_tensor
        }

        if feature=='openface' :
            dic = self.openface_dict[speaker]
            output['features'] = self.openface_tensor[speaker]
        elif feature=='opensmile' :
            dic = self.opensmile_dict[speaker]
            output['features'] = self.opensmile_tensor[speaker]
        else:
            print('Choose feature in ("openface","opensmile")')
            raise

        #indexes of categories to stratify
        target_index = self.indexes[speaker]['target']
        #none indexes
        if not none_count : none_count = int( (len(target_index) * none_prop) / (1 - none_prop) )
        none_index = pd.Index(random.sample(list(self.indexes[speaker]['none']),none_count))
        #constitute df to stratify
        df = self.embeds[self.target_col].loc[target_index.union(none_index).difference(self._to_avoid)]
        class_weights = torch.Tensor((df.sum()/df.shape[0]).values)
        train, test = train_test_split(df, test_size=test_size, stratify=df)
        output['test_dic']  = {k:v for (k,v) in dic.items() if int(k) in test.index.union(self._to_avoid)}

        if val_size :
            train, valid = train_test_split(train, test_size=val_size, stratify=train)
            
            output['train_dic'] = {k:v for (k,v) in dic.items() if int(k) in train.index}
            output['valid_dic'] = {k:v for (k,v) in dic.items() if int(k) in valid.index}
            return {'data' : output, 'class_weights':class_weights}
        
        output['train_dic'] = {k:v for (k,v) in dic.items() if int(k) in train.index}
        return {'data' : output, 'class_weights':class_weights}

    def make_train_test_datasets(self,test_size=.3,val_size=None, none_count=None, none_prop=.3):

        tts_openface_1  = self.stratified_train_test_split(feature = 'openface', speaker=1, test_size=test_size,val_size=val_size, none_count=none_count, none_prop=none_prop)
        tts_openface_2  = self.stratified_train_test_split(feature = 'openface', speaker=2, test_size=test_size,val_size=val_size, none_count=none_count, none_prop=none_prop)
        tts_opensmile_1 = self.stratified_train_test_split(feature = 'opensmile', speaker=1, test_size=test_size,val_size=val_size, none_count=none_count, none_prop=none_prop)
        tts_opensmile_2 = self.stratified_train_test_split(feature = 'opensmile', speaker=2, test_size=test_size,val_size=val_size, none_count=none_count, none_prop=none_prop)

        openface_1  = dicDataset(**tts_openface_1['data'])
        openface_2  = dicDataset(**tts_openface_2['data'])
        opensmile_1 = dicDataset(**tts_opensmile_1['data'])
        opensmile_2 = dicDataset(**tts_opensmile_2['data'])

        class_weights = torch.stack((tts_openface_1['class_weights'], tts_openface_2['class_weights'],tts_opensmile_1['class_weights'], tts_opensmile_2['class_weights'])).mean(dim=0)

        return {
            'datasets' : {'dyad_session_dict': self.dyad_session_dict, 'openface_1':openface_1, 'openface_2':openface_2, 'opensmile_1':opensmile_1, 'opensmile_2':opensmile_2},
            'class_weights':class_weights
        }
    
    def prepare_hierarchical(self):
        output = {
            'dyad_session_dict' : self.dyad_session_dict,
            'openface_1'  : dicDataset( train_dic=self.openface_dict[1], features=self.openface_tensor[1], targets=self.target_tensor ),
            'openface_2'  : dicDataset( train_dic=self.openface_dict[2], features=self.openface_tensor[2], targets=self.target_tensor ),
            'opensmile_1' : dicDataset( train_dic=self.opensmile_dict[1], features=self.opensmile_tensor[1], targets=self.target_tensor ),
            'opensmile_2' : dicDataset( train_dic=self.opensmile_dict[2], features=self.opensmile_tensor[2], targets=self.target_tensor )
        }


#create dataset from dict, features_data, and targets_data
class dicDataset(Dataset):
    """
    dict has {index in targets_data : related indexes in features_data}
    features_data is a tensor of openface or opensmile
    targets_data is the tensor of targets (from embeds[feature_col])
    """
    def __init__(self, train_dic, features, targets, test_dic=None, valid_dic=None):
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
    
    def to(self, device):
        self.f = self.f.to(device)
        self.t = self.t.to(device)
        return None

    def get_item_with_index(self, idx):
        features_indexes = self.train_dic[str(idx)]
        features = self.f[features_indexes, :]
        targets = self.t[idx,:]
        return features, targets, len(features_indexes)

    def _get_test_item(self, idx):
        features_indexes = self.test_dic[str(idx)]
        features = self.f[features_indexes, :]
        return features, self.t[int(idx),:], len(features_indexes)

    def get_test(self):
        if not self.test_dic:
            print('No valid data')
            raise
        return pad_collate([self._get_test_item(idx) for idx in self.test_dic.keys()])

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

class HierarchicalDataset(Dataset):

    def __init__(self, DH):
        self.DH = DH
        self.all_indexes = list(self.DH.dyad_session_dict.keys())
        self.indexes = self.all_indexes
        self._len=(len(self.indexes))
        self.openface_datasets = {
            '1' : dicDataset( train_dic=DH.openface_dict[1], features=DH.openface_tensor[1], targets=DH.target_tensor ),
            '2' : dicDataset( train_dic=DH.openface_dict[2], features=DH.openface_tensor[2], targets=DH.target_tensor )
        }
        self.opensmile_datasets = {
            '1' : dicDataset( train_dic=DH.opensmile_dict[1], features=DH.opensmile_tensor[1], targets=DH.target_tensor ),
            '2' : dicDataset( train_dic=DH.opensmile_dict[2], features=DH.opensmile_tensor[2], targets=DH.target_tensor )
        }
        self._train_indexes = None
        self._eval_indexes = None
        self._test_indexes = None
        self.train_eval_test_split()
        return None
    
    def sort(self):
        self.all_indexes.sort(key= lambda x : (int(x[0]), int(x[1]), int(x[2])))
        return None

    def train_test_split(self, test_size=.3, shuffle=True):
        train = int(len(self.all_indexes) * (1-test_size))
        self.sort()
        if shuffle : random.shuffle(self.all_indexes)
        self._train_indexes = self.all_indexes[:train]
        self._test_indexes = self.all_indexes[train:]
        self.sort()
        print(f"Train test split:\nThere are {len(self._train_indexes)} training elements and {len(self._test_indexes)} testing elements")
        return None

    def train_eval_test_split(self, test_size=.15, eval_size=.17, shuffle=True):
        train = int( len(self.all_indexes) * (1-test_size - eval_size) )
        test = int( len(self.all_indexes) * test_size )
        self.sort()
        if shuffle : random.shuffle(self.all_indexes)
        self._train_indexes = self.all_indexes[:train]
        self._test_indexes = self.all_indexes[train:train+test+1]
        self._eval_indexes = self.all_indexes[train+test+1:]
        self.sort()
        print(f"Train eval test split:\nThere are {len(self._train_indexes)} training elements, {len(self._eval_indexes)} eval elements, and {len(self._test_indexes)} testing elements")
        return None

    def train(self):
        self.indexes = self._train_indexes
        return None
    def eval(self):
        self.indexes = self._eval_indexes
        return None
    def test(self):
        self.indexes = self._test_indexes
        return None
    def reset(self):
        self.indexes = self.all_indexes
        return None

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, idx):
        speaker = self.indexes[idx][2]
        index = self.DH.dyad_session_dict[self.indexes[idx]]

        of_batch = [ self.openface_datasets[speaker].get_item_with_index(i) for i in index ]
        os_batch = [ self.opensmile_datasets[speaker].get_item_with_index(i) for i in index ]
        return {
            'targets'   : self.DH.target_tensor[index,:],
            'embeds'    : self.DH.embeds_tensors[index,:],
            'openface'  : pad_collate(of_batch),
            'opensmile' : pad_collate(os_batch)
        }
    
    def dataloader(self):
        def collate_fn(batch):
            return batch[0]
        return DataLoader(self, batch_size=1, collate_fn = collate_fn)

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

class Pipeline():
    def __init__(self, model, features_1, features_2, criterion, cats=['SD', 'QE', 'SV', 'PR', 'HD']):
        self.datasets = {1 : features_1, 2 : features_2}
        self.criterion = criterion
        self.model = model
        self.hist_train_loss = None
        self.hist_test_loss = None
        self.hist_val_loss = None
        self.hist_auf1c = None
        self.cats = cats

    def eval_on_batch(self, batch):
        self.model.eval()
        with torch.no_grad():
            pred = self.model(batch['features'])
        return self.criterion(pred, batch['targets']).cpu().detach().numpy()
    
    def to(self, device):
        self.model.to(device)
        self.datasets[1].to(device)
        self.datasets[2].to(device)
        return None

    
    def train(self, batch_size=48, epoch=50, lr=1e-4, gpu=False, early_stop=None):

        if gpu :
            if torch.backends.mps.is_available(): device = 'mps'
            elif torch.cuda.is_available():       device = 'cuda'
            else :                                device = 'cpu'
        else : device = 'cpu'
        self.to(device)
        print("Training on", device)

        if not self.hist_train_loss :
            self.hist_train_loss = []
            self.hist_eval_loss = []
            self.hist_test_loss = []
            self.hist_auf1c = []
        
        test1, eval1 = self.datasets[1].get_test(), self.datasets[1].get_valid()
        test2, eval2 = self.datasets[2].get_test(), self.datasets[2].get_valid()

        dataloader1 = DataLoader(self.datasets[1], batch_size = batch_size, shuffle = True, collate_fn = pad_collate)
        dataloader2 = DataLoader(self.datasets[2], batch_size = batch_size, shuffle = True, collate_fn = pad_collate)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)


        for e in range(epoch):
            epoch_loss = 0.0

            for batch1,batch2 in zip(dataloader1,dataloader2):

                self.model.train()
                pred1 = self.model(batch1['features'])
                loss1 = self.criterion(pred1, batch1['targets'])

                epoch_loss += loss1

                for param in self.model.parameters():
                    param.grad = None
                loss1.backward()
                optimizer.step()

                pred2 = self.model(batch2['features'])
                loss2 = self.criterion(pred2, batch2['targets'])

                epoch_loss += loss2

                for param in self.model.parameters():
                    param.grad = None
                loss2.backward()
                optimizer.step()
            
            if (e+1)%(epoch//5)==0:
                print(f"loss epoch {e}: {epoch_loss:2f}")
        
            self.hist_train_loss.append(( (loss1+loss2) / ( len(self.datasets[1]) + len(self.datasets[2] ) ) ).cpu().detach().numpy())

            if e%3==0:
                test_loss = ( self.eval_on_batch(test1) + self.eval_on_batch(test2) ) / ( len(test1) + len(test2) )
                self.hist_test_loss.append(test_loss)
                eval_loss = ( self.eval_on_batch(eval1) + self.eval_on_batch(eval2) ) / ( len(eval1) + len(eval2) )
                self.hist_eval_loss.append(eval_loss)
                self.hist_auf1c.append(self.eval_model(plot=False))
        
        self.to('cpu')
        return None

    def plot_losses(self):
        fig, ax = plt.subplots(1,4, figsize=(18,4))
        ax[0].plot(self.hist_train_loss)
        ax[0].set_title('Train')
        ax[1].plot(self.hist_eval_loss)
        ax[1].set_title('Eval')
        ax[2].plot(self.hist_test_loss)
        ax[2].set_title('Test')
        pd.DataFrame(self.hist_auf1c, columns=self.cats).plot(ax=ax[3])
        ax[3].set_title('Area under f1 score')
        plt.show()
        return None
    
    def eval_model(self, plot=True):

        self.model.eval()

        test1 = self.datasets[1].get_test()
        pred1 = self.model(test1['features']).cpu().detach().numpy()
        test2 = self.datasets[2].get_test()
        pred2 = self.model(test2['features']).cpu().detach().numpy()

        eval1 = self.datasets[1].get_valid()
        pred3 = self.model(eval1['features']).cpu().detach().numpy()
        eval2 = self.datasets[2].get_valid()
        pred4 = self.model(eval2['features']).cpu().detach().numpy()

        pred = np.concatenate((pred1, pred2, pred3, pred4), axis=0)

        true1 = test1['targets'].cpu().detach().numpy()
        true2 = test2['targets'].cpu().detach().numpy()
        true3 = eval1['targets'].cpu().detach().numpy()
        true4 = eval2['targets'].cpu().detach().numpy()
        true = np.concatenate((true1,true2,true3,true4), axis=0)
        counts = true.sum(axis=0)

        #fig, ax = plt.subplots(len(self.cats),2, figsize=(18,7))
        f1=[]
        for i, cat in enumerate(self.cats):
            rd = simulate_randomness(true[:,i],pred[:,i])
            res = explore_tresh(true[:,i],pred[:,i])

            f1.append((res['F1 score'].values - rd['Rd. F1 score'].values).mean())
            if plot :
                ax = res.plot(color =list(mcolors.TABLEAU_COLORS.values()) )#ax=ax[i][0])
                rd.plot( ax=ax, linestyle='dashed',color =list(mcolors.TABLEAU_COLORS.values()))#ax=ax[i][0])
                plt.title(f"{self.cats[i]}\nSupport:{int(counts[i])}")
                plt.show()
        
        return f1

def simulate_randomness(true, pred):
    tresh = np.linspace(pred.min(),pred.max(),20)
    p=true.sum()/true.shape[0]
    func = lambda t : np.where(pred > t,1,0).mean()
    out = [(
        func(t)*(2*p-1) + 1 - p,
        p,
        func(t),
        (2*p*func(t)) / (p + func(t)) if func(t) > 0 else np.nan
    ) for t in tresh[:-1]]
    out = pd.DataFrame(out, columns=['Rd. accuracy', 'Rd. precision', 'Rd. recall', 'Rd. F1 score'], index=tresh[:-1])
    return out

def explore_tresh(true, pred):
    prop=true.sum()/true.shape[0]
    tresh = np.linspace(pred.min(),pred.max(),20)
    out = [(
        accuracy_score(true, np.where(pred>t,1,0)),
        precision_score(true, np.where(pred>t,1,0), zero_division=1),
        recall_score(true, np.where(pred>t,1,0)),
        f1_score(true, np.where(pred>t,1,0)),
        chi2( np.where(pred>t,1,0).reshape(-1,1) , true )[1][0]
    ) for t in tresh[:-1]]
    out = pd.DataFrame(out, columns=['Accuracy', 'Precision', 'Recall', 'F1 score', '- .5 log p-value'], index=tresh[:-1])
    out['- .5 log p-value'] = - np.log10(out['- .5 log p-value']) / 2
    return out