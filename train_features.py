from time import time
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

import warnings

warnings.filterwarnings("ignore")

from transformers import BertTokenizer, logging

logging.set_verbosity_warning()
logging.set_verbosity_error()

from tqdm import tqdm

from torch.utils.data import DataLoader

from transformers import DistilBertTokenizer, RobertaTokenizer

import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import argparse

import models #models_aus_only, models_roberta_novsn, models_novsn, models_roberta, 

import warnings

warnings.filterwarnings("ignore")

# Setting up the device for GPU usage
try : device = torch.device("mps")
except : device = torch.device("cpu")

SEED = 12061999  # If you think that this is my birth date.... then you are right


class FocalLoss(nn.Module):
    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float = -1.0,
        weight = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.weight = weight
        self.reduction = reduction

    def forward(self, inp: torch.Tensor, targ: torch.Tensor):
        alpha = self.alpha
        gamma = self.gamma
        ce_loss = F.binary_cross_entropy_with_logits(
            inp, targ, weight=self.weight, reduction="none"
        )
        p = torch.sigmoid(inp)
        p_t = p * targ + (1 - p) * (1 - targ)
        loss = ce_loss * ((1 - p_t) ** gamma)

        if alpha >= 0:
            alpha_t = alpha * targ + (1 - alpha) * (1 - targ)
            loss = alpha_t * loss
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        return loss


class PadSequence:
    def __call__(self, batch):
        # Let's assume that each element in "batch" is a tuple (data, label).
        # Sort the batch in the descending order
        sorted_batch = sorted(batch, key=lambda x: x[0].shape[0], reverse=True)
        # Get each sequence and pad it
        sequences = [x[0] for x in sorted_batch]
        sequences_padded = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True)
        # Also need to store the length of each sequence
        # This is later needed in order to unpad the sequences
        lengths = torch.LongTensor([len(x) for x in sequences])
        # Don't forget to grab the labels of the *sorted* batch
        labels = torch.LongTensor(map(lambda x: x[1], sorted_batch))
        return sequences_padded, lengths, labels


class AudioVideoDataset(Dataset):
    def get_duration(self, ts):
        times, ms = ts.split(".")[0].split(":"), int(ts.split(".")[1])
        hours, minutes, seconds = int(times[0]), int(times[1]), int(times[2])
        if hours > 0 or minutes > 0 or seconds > 12:
            return 12
        else:
            return round5(seconds + ms / 1000)

    def __init__(self, df, type = "audio"):
        super(AudioVideoDataset, self).__init__()
        """TextDataset is a dataset containing utterances along with their CS. Used as a baseline for comparison with our MultiModal architecture.

        Args:
            text_df (_type_): pandas DataFrame containing the utterances along with the dyad, Session, Period, Timestamps and Labels informations.
            tokenizer (_type_): tokenizer to use for Bert-based models
            max_len (_type_): maximum length of utterances in terms of tokens
        """
        audio_features_list = ["alphaRatio","hammarbergIndex","slope0-500","slope500-1500","spectralFlux","mfcc1","mfcc2","mfcc3","mfcc4","F0semitoneFrom27.5Hz","jitterLocal","shimmerLocaldB","HNRdBACF","logRelF0-H1-H2","logRelF0-H1-A3","F1frequency","F1bandwidth","F1amplitudeLogRelF0","F2frequency","F2amplitudeLogRelF0","F3frequency","F3amplitudeLogRelF0"]
        video_features_list = ["AU01_r","AU02_r", "AU04_r", "AU05_r", "AU06_r", "AU07_r", "AU09_r", "AU10_r", "AU12_r", "AU14_r", "AU15_r", "AU17_r", "AU20_r", "AU23_r", "AU25_r", "AU26_r", "AU45_r"]

        audio_features, video_features = list(), list()
        
        for i in range(24):
            audio_features += [f_ + str(i) for f_ in audio_features_list]
            video_features += [f_ + str(i) for f_ in video_features_list]

        if type == "audio":
            self.df = df[["Dyad","Session","Begin_time","End_time","Duration","P1","P2","SD","QE","SV","PR","HD"] + audio_features].replace("", pd.NA)
        else:
            self.df = df[["Dyad","Session","Begin_time","End_time","Duration","P1","P2","SD","QE","SV","PR","HD"] + video_features].replace("", pd.NA)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """
        Getitem method

        Returns:
            dictionary: dictionary containing the aforementioned information. It does the following :
            Given an index -> features + information.
        """

        df = self.df
        if torch.is_tensor(idx):
            idx = idx.tolist()

        end_time = df["Duration"].apply(self.get_duration).values
        frame_number = max(1, int(2 * end_time))

        feature_list = list(df.columns)[- (frame_number * 22): - 17 * 24]
        features = df[idx][feature_list].values.reshape(-1, 24)

        y = torch.Tensor(
            self.utterances.iloc[idx][["SD", "QE", "SV", "PR", "HD"]]
            .fillna("")
            .apply(len)
        ).squeeze().type(torch.LongTensor)


        output = {
            "features": torch.tensor(features, dtype = torch.float),
            "sequence_lengths" : torch.tensor(frame_number, dtype = torch.long), 
            "target": y,
        }

        return output


def round5(number):
    """I used to be a very smart mathematics student. I came up with this clever technique to round a number to .5, using all of my neurons and my amazing ability to use python.
    I plan to publish this amazing and disruptive work soon in Nature. Please do not plagiate.

    Args:
        number (number): a number

    Returns:
        number: the same number rounded to the closest .5 value.
    """
    return round(number * 2) / 2


def class_split(tot_dataframe, class_list = None):
    """Using a Dataframe containing utterances and classes, returns a forced split of the Dataset to even out the classes between Train, Test and Validation.

    Args:
        tot_dataframe (_type_): Dataframe containing Utterances and their associated CS. Output of the read_dataset() function.

    Returns:
        _type_: Three datasets containing an amount proportional to their size of each class.
    """
    train_data, test_data, val_data = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    df = tot_dataframe.replace("", pd.NA)

    df["None"] = 0

    if not class_list:
        class_list = ["PR", "SD", "QE", "SV", "HD", "None"]

    df.loc[df[class_list].isna().any(axis=1), "None"] = "x"
    print(df["None"].unique())

    total_length = 0
    for i, class_ in enumerate(class_list):
        label_df = df[df[class_] == "x"].sample(frac = 1)
        if class_ == "None":

            # Here, the idea is to downsample the None class.

            current_length = len(label_df)
            total_length += current_length
            train_data = pd.concat(
                [train_data, label_df.iloc[: int(current_length * 0.15)]]
            )
            test_data = pd.concat(
                [
                    test_data,
                    label_df.iloc[
                        int(current_length * 0.15) : int(current_length * 0.175)
                    ],
                ]
            )
            val_data = pd.concat(
                [
                    val_data,
                    label_df.iloc[
                        int(current_length * 0.175) : int(current_length * 0.2)
                    ],
                ]
            )
        else:
            current_length = len(label_df)
            total_length += current_length
            train_data = pd.concat(
                [train_data, label_df.iloc[:int(current_length * 0.75)]]
            )
            test_data = pd.concat(
                [
                    test_data,
                    label_df.iloc[
                        int(current_length * 0.75):int(current_length * 0.875)
                    ],
                ]
            )
            val_data = pd.concat(
                [val_data, label_df.iloc[int(current_length * 0.875):]]
            )
    for i, class_ in enumerate(class_list):
        curr_test_len_ = len(test_data[test_data[class_] == "x"])
        curr_train_len_ = len(train_data[train_data[class_] == "x"])
        curr_val_len_ = len(val_data[val_data[class_] == "x"])

        print(
            f"{class_} : TEST {curr_test_len_} TRAIN {curr_train_len_} VAL {curr_val_len_}"
        )

    test_index = test_data.index
    train_index = train_data.index
    test_data = test_data[~test_index.isin(train_index)]
    val_data = val_data.iloc[~val_data.index.isin(test_index.append(train_index))]

    train_data = train_data.reset_index()
    test_data = test_data.reset_index()
    val_data = val_data.reset_index()

    return train_data, test_data, val_data


def train_one_epoch(epoch_index, is_text=False, is_au=False):
    """train_one_epoch trains an epoch, as the name cleverly suggests.

    Args:
        epoch_index (_type_): not used here, used to print which epoch is currently running.
        is_text, is_au (bool, optional): Instead of writing several functions, just did a big one that can take those booleans as inputs. As the Dataset to use is different for Text-only, AUs-only or Bi-modal classification . Defaults to False.

    Returns:
        _type_: nothing, just trains the model.
    """
    running_loss = 0.0
    last_loss = 0.0

    if is_text:
        for i, data in tqdm(enumerate(train_loader)):
            # Every data instance is an input + label pair
            input_features, labels = (
                data["features"].to(device, dtype=torch.float),
                data["target"].to(device, dtype=torch.long),
            )

            # labels = 1 - labels
            # Zero your gradients for every batch!

            # Make predictions for this batch
            outputs = model(input_features)

            optimizer.zero_grad()

            # Compute the loss and its gradients
            loss = loss_fn(outputs, labels)
            loss.backward()

            # Adjust learning weights
            optimizer.step()

            # Gather data and report
            running_loss += loss.item()
            if i % 50 == 1:
                last_loss = running_loss / 50  # loss per batch
                print("  batch {} loss: {}".format(i + 1, last_loss))
                running_loss = 0.0
    return last_loss


if __name__ == "__main__":
    import datetime

    now = datetime.datetime.now().strftime(r"%y_%m_%d_%H")
    audio_df = pd.read_csv("audio_features_test.csv")
    model_l = [
        models.AudioGRU(),
    ]
    
    model_names_l = [type(m_).__name__ for m_ in model_l]

    for i, model_ in enumerate(model_l):

        print(f"===========\nTRAINING MODEL {type(model_).__name__} :")
        EPOCHS = 3
        MAX_LEN = 128

        model = model_
        model.to(device)

        train_df, test_df, val_df = class_split(audio_df)

        train_data, test_data, val_data = (
            AudioVideoDataset(train_df, type = "audio"),
            AudioVideoDataset(test_df, type = "audio"),
            AudioVideoDataset(val_df, type = "audio")
        )

        train_loader, test_loader, val_loader = (
            DataLoader(train_data, batch_size=8, shuffle=True, num_workers=8),
            DataLoader(test_data, batch_size=8, shuffle=True, num_workers=8),
            DataLoader(val_data, batch_size=8, shuffle=True, num_workers=8),
        )


        # loss_fn = FocalLoss(gamma=1, alpha=0.6988592276480736).to(device)  # alpha = .9
        # loss_fn_cpu = FocalLoss(gamma=1, alpha=0.6988592276480736).to("cpu")  # alpha = .9
        # loss_fn = torch.nn.BCELoss() #weight=weights
        loss_fn = torch.nn.BCEWithLogitsLoss() #weight=weights.to("cpu")

        params = model.parameters()
        optimizer = torch.optim.Adam(params, lr=1e-5)

        timestamp = pd.datetime.now().strftime("%Y%m%d_%H%M%S")
        epoch_number = 0
        best_vloss = 100

        num_classes = 6

        MODEL_DIR = f"models/audio_nocontext_{num_classes}_{now}/"

        for epoch in range(EPOCHS):
            print("EPOCH {}:".format(epoch_number + 1))

            # Make sure gradient tracking is on, and do a pass over the data
            model.train(True)
            avg_loss = train_one_epoch(
                epoch, is_text=type(model).__name__ == "BertClassif"
            )

            # We don't need gradients on to do reporting
            model.train(False)
            model.eval()
            running_vloss = 0.0

            fin_targets, fin_outputs = list(), list()

            with torch.no_grad():
                for i, vdata in enumerate(val_loader):
                    print(model_(vdata))
                    break
                    vinput_ids, vattention_mask, vlabels = (
                        vdata["ids"].to(device, dtype=torch.long),
                        vdata["mask"].to(device, dtype=torch.long),
                        vdata["target"].to(device, dtype=torch.float),
                    )

                    voutputs = model(vinput_ids, vattention_mask)
                    fin_targets.extend(vlabels.cpu().detach().numpy().tolist())
                    fin_outputs.extend(torch.sigmoid(voutputs).cpu().detach().numpy().tolist())
                    # vlabels = 1 - vlabels
                    vloss = loss_fn(voutputs, vlabels)
                    running_vloss += vloss
                    final_i = i

                avg_vloss = running_vloss / (final_i + 1)
                print("LOSS train {} valid {}".format(avg_loss, avg_vloss))

                # Track best performance, and save the model's state
                if avg_vloss < best_vloss or epoch == 9:
                    best_vloss = avg_vloss
                    model_name = "{}".format(type(model).__name__)
                    if not os.path.exists(MODEL_DIR):
                        os.mkdir(MODEL_DIR)
                    torch.save(
                        model.state_dict(), MODEL_DIR + model_name,
                    )

                epoch_number += 1