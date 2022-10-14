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

from torch import cuda
from torch.utils.data import DataLoader

from transformers import DistilBertTokenizer, RobertaTokenizer

import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import argparse

import models_aus_only, models_roberta_novsn, models_novsn, models_roberta, models

import warnings

warnings.filterwarnings("ignore")

text_path = "./data/2016_important/2016_text_only.csv"
aus_path = "./data/2016_important/2016_aus_med_500ms.csv"

# Setting up the device for GPU usage
device = "cuda:0" if cuda.is_available() else "cpu"

SEED = 12061999  # If you think that this is my birth date.... then you are right


class MultiModalDataset(Dataset):
    def __init__(
        self, text_df: pd.DataFrame(), aus_df: pd.DataFrame(), tokenizer, max_len
    ):
        """MultiModalDataset is a dataset containing utterances, with their 12 seconds * 2 slices per second * 17 features facial informations matrix, flattened for our purpose of passing the AUs through a dense layer.

        Args:
            text_df (_type_): pandas DataFrame containing the utterances along with the dyad, Session, Period, Timestamps and Labels informations.
            aus_df (_type_): pandas DataFrame containing the median values of Facial Action Units over .5s slices, along with the Dyad, Session, Participant and Timestamps information.
            tokenizer (_type_): tokenizer to use for Bert-based models
            max_len (_type_): maximum length of utterances in terms of tokens
        """

        self.utterances, self.action_units = text_df, aus_df
        self.aus_list = [
            "AU01_r",
            "AU02_r",
            "AU04_r",
            "AU05_r",
            "AU06_r",
            "AU07_r",
            "AU09_r",
            "AU10_r",
            "AU12_r",
            "AU14_r",
            "AU15_r",
            "AU17_r",
            "AU20_r",
            "AU23_r",
            "AU25_r",
            "AU26_r",
            "AU45_r",
        ]
        self.aus = self.action_units[self.aus_list]

        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.utterances)

    def __getitem__(self, idx):
        """Getitem method

        Returns:
            dictionary: dictionary containing the aforementioned information. It does the following :
            Given an index -> utterances + its timestamps -> Gets the timestamps for the 12s period that finishes at the end of the sentence -> gathers the facial informations of the participant during this 12s slice.
            Disregards the sentences finishing before the 00:00:12.000 second mark for each dyad. Not much information is lost.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        participant_id = (
            int(not pd.isna(self.utterances.iloc[idx]["P1"]))
            + int(not pd.isna(self.utterances.iloc[idx]["P2"])) * 2
        )

        time_start = round5(
            pd.to_timedelta(self.utterances.iloc[idx]["Time_begin"]).total_seconds()
        )
        time_end = round5(
            pd.to_timedelta(self.utterances.iloc[idx]["Time_end"]).total_seconds()
        )

        duration = self.utterances.iloc[idx]["Duration_s"]

        dyad = self.utterances.iloc[idx]["Dyad"]
        session = self.utterances.iloc[idx]["Session"]
        period = self.utterances.iloc[idx]["Period"]
        period = "Social" * int(period[0] == "S") + "Tutoring" * int(period[0] == "T")

        # role = (
        #     int(str(session)[-1] == str(participant_id)[-1]) * "Tutor"
        #     + int(str(session)[-1] != str(participant_id)[-1]) * "Tutee"
        # )

        aus = self.action_units[
            (self.action_units.dyad == dyad)
            & (self.action_units.session == session)
            & (self.action_units.participant == participant_id)
            & (self.action_units.timestamp >= time_start + duration - 11.9)
            & (self.action_units.timestamp <= time_start + duration)
        ][self.aus_list]

        aus = (
            F.pad(
                torch.tensor(np.array(aus.values), dtype=torch.float),
                (0, 0, 0, 24 - len(aus)),
            )
            .reshape(24, 17)
            .flatten()
        )

        # Depending on the situation, you can extend the list of classes. Choice was made here to rule out VSN, as at the current time, I had no Data that properly distinguishes between
        # off-task talk and VSN, the Conversational Strategy.

        y = torch.Tensor(
            self.utterances.iloc[idx][["PR", "SD", "QE", "None"]]  # ,"VSN",
            .fillna("")
            .apply(len)
        ).squeeze()

        if True:
            current_text = self.utterances.iloc[idx][f"P{participant_id}"]
            context = self.utterances.iloc[idx]["Context"]
            text = (
                " <P> ".join(context)
                + " <start> "
                + current_text
                + " <end>".replace("[", "")
                .replace("]", "")
                .replace("(", "")
                .replace(")", "")
                .replace("laughter", "")
            )

        else:
            text = self.utterances.iloc[idx][f"P{participant_id}"]
            text = (
                "".join(text.split())
                .replace("[", "")
                .replace("]", "")
                .replace("(", "")
                .replace(")", "")
                .replace("laughter", "")
            )

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            return_token_type_ids=True,
            return_tensors="pt",
        )

        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]

        output = {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "aus": aus,
            "target": y,
        }

        try:
            test = torch.reshape(ids, [-1, 128])
        except:
            print(text, ids)

        # output = {
        #     "ids": torch.tensor(ids, dtype=torch.long),
        #     "mask": torch.tensor(mask, dtype=torch.long),
        #     "aus": torch.tensor(aus, dtype=torch.float),
        #     "target": torch.tensor(y, dtype=torch.long),
        # }

        return output


class AUDataset(Dataset):
    def __init__(self, text_df, aus_df, tokenizer, max_len):
        """AUDataset is a dataset containing the action units only, with their 12 seconds * 2 slices per second * 17 features facial informations matrix.

        Args:
            text_df (_type_): pandas DataFrame containing the utterances along with the dyad, Session, Period, Timestamps and Labels informations -> to get the timestamps where the conversational strategy occured.
            aus_df (_type_): pandas DataFrame containing the median values of Facial Action Units over .5s slices, along with the Dyad, Session, Participant and Timestamps information.
            tokenizer (_type_): useless here.
            max_len (_type_): also useless here.
        """
        self.utterances, self.action_units = text_df, aus_df
        self.aus_list = [
            "AU01_r",
            "AU02_r",
            "AU04_r",
            "AU05_r",
            "AU06_r",
            "AU07_r",
            "AU09_r",
            "AU10_r",
            "AU12_r",
            "AU14_r",
            "AU15_r",
            "AU17_r",
            "AU20_r",
            "AU23_r",
            "AU25_r",
            "AU26_r",
            "AU45_r",
        ]
        self.aus = self.action_units[self.aus_list]

        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.utterances)

    def __getitem__(self, idx):
        """Returns the action units of the selected face occuring during the conversational strategy.

        Returns:
            dictionary : dictionary containing the 17*2*12 tensor of action units.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        participant_id = (
            int(not pd.isna(self.utterances.iloc[idx]["P1"]))
            + int(not pd.isna(self.utterances.iloc[idx]["P2"])) * 2
        )

        time_start = round5(
            pd.to_timedelta(self.utterances.iloc[idx]["Time_begin"]).total_seconds()
        )
        time_end = round5(
            pd.to_timedelta(self.utterances.iloc[idx]["Time_end"]).total_seconds()
        )

        duration = self.utterances.iloc[idx]["Duration_s"]

        dyad = self.utterances.iloc[idx]["Dyad"]
        session = self.utterances.iloc[idx]["Session"]
        period = self.utterances.iloc[idx]["Period"]
        period = "Social" * int(period[0] == "S") + "Tutoring" * int(period[0] == "T")

        # role = (
        #     int(str(session)[-1] == str(participant_id)[-1]) * "Tutor"
        #     + int(str(session)[-1] != str(participant_id)[-1]) * "Tutee"
        # )

        aus = self.action_units[
            (self.action_units.dyad == dyad)
            & (self.action_units.session == session)
            & (self.action_units.participant == participant_id)
            & (self.action_units.timestamp >= time_start + duration - 11.9)
            & (self.action_units.timestamp <= time_start + duration)
        ][self.aus_list]

        aus = (
            F.pad(
                torch.tensor(np.array(aus.values), dtype=torch.float),
                (0, 0, 0, 24 - len(aus)),
            )
            .reshape(24, 17)
            .flatten()
        )

        # Depending on the situation, you can extend the list of classes. Choice was made here to rule out VSN, as at the current time, I had no Data that properly distinguishes between
        # off-task talk and VSN, the Conversational Strategy.

        y = torch.Tensor(
            self.utterances.iloc[idx][["VSN", "PR", "SD", "QE", "None"]]  # "VSN",
            .fillna("")
            .apply(len)
        ).squeeze()

        output = {
            "aus": aus,
            "target": y,
        }

        # output = {
        #     "ids": torch.tensor(ids, dtype=torch.long),
        #     "mask": torch.tensor(mask, dtype=torch.long),
        #     "aus": torch.tensor(aus, dtype=torch.float),
        #     "target": torch.tensor(y, dtype=torch.long),
        # }

        return output


class MultiModalDatasetRNN(Dataset):
    def __init__(self, text_df, aus_df, tokenizer, max_len):
        """MultiModalDataset is a dataset containing utterances, with their 12 seconds * 2 slices per second * 17 features facial informations matrix. Formatted as a sequence of length 12 * 2 and with features of size 17 to pass through a RNN-type layer.

        Args:
            text_df (_type_): pandas DataFrame containing the utterances along with the dyad, Session, Period, Timestamps and Labels informations.
            aus_df (_type_): pandas DataFrame containing the median values of Facial Action Units over .5s slices, along with the Dyad, Session, Participant and Timestamps information.
            tokenizer (_type_): tokenizer to use for Bert-based models
            max_len (_type_): maximum length of utterances in terms of tokens
        """
        self.utterances, self.action_units = text_df, aus_df
        self.aus_list = [
            "AU01_r",
            "AU02_r",
            "AU04_r",
            "AU05_r",
            "AU06_r",
            "AU07_r",
            "AU09_r",
            "AU10_r",
            "AU12_r",
            "AU14_r",
            "AU15_r",
            "AU17_r",
            "AU20_r",
            "AU23_r",
            "AU25_r",
            "AU26_r",
            "AU45_r",
        ]
        self.aus = self.action_units[self.aus_list]

        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.utterances)

    def __getitem__(self, idx):
        """Getitem method

        Returns:
            dictionary: dictionary containing the aforementioned information. It does the following :
            Given an index -> utterances + its timestamps -> Gets the timestamps for the 12s period that finishes at the end of the sentence -> gathers the facial informations of the participant during this 12s slice.
            Disregards the sentences finishing before the 00:00:12.000 second mark for each dyad. Not much information is lost.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        participant_id = (
            int(not pd.isna(self.utterances.iloc[idx]["P1"]))
            + int(not pd.isna(self.utterances.iloc[idx]["P2"])) * 2
        )

        time_start = round5(
            pd.to_timedelta(self.utterances.iloc[idx]["Time_begin"]).total_seconds()
        )
        time_end = round5(
            pd.to_timedelta(self.utterances.iloc[idx]["Time_end"]).total_seconds()
        )

        dyad = self.utterances.iloc[idx]["Dyad"]
        session = self.utterances.iloc[idx]["Session"]
        period = self.utterances.iloc[idx]["Period"]
        period = "Social" * int(period[0] == "S") + "Tutoring" * int(period[0] == "T")

        aus = self.action_units[
            (self.action_units.dyad == dyad)
            & (self.action_units.session == session)
            & (self.action_units.participant == participant_id)
            & (self.action_units.timestamp >= max(0, time_start - 1.0))
            & (self.action_units.timestamp <= min(time_start + 10.0, time_end) + 1.0)
        ][self.aus_list]

        aus = F.pad(
            torch.tensor(np.array(aus.values), dtype=torch.float),
            (0, 0, 0, 24 - len(aus)),
        ).reshape(24, 17)

        # Depending on the situation, you can extend the list of classes. Choice was made here to rule out VSN, as at the current time, I had no Data that properly distinguishes between
        # off-task talk and VSN, the Conversational Strategy.

        y = torch.Tensor(
            self.utterances.iloc[idx][["PR", "SD", "QE", "None"]]  # "VSN",
            .fillna("")
            .apply(len)
        ).squeeze()

        if True:
            current_text = self.utterances.iloc[idx][f"P{participant_id}"]
            context = self.utterances.iloc[idx]["Context"]
            text = (
                " <P> ".join(context)
                + " <start> "
                + current_text
                + " <end>".replace("[", "")
                .replace("]", "")
                .replace("(", "")
                .replace(")", "")
                .replace("laughter", "")
            )

        else:
            text = self.utterances.iloc[idx][f"P{participant_id}"]
            text = (
                "".join(text.split())
                .replace("[", "")
                .replace("]", "")
                .replace("(", "")
                .replace(")", "")
                .replace("laughter", "")
            ) + ""

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            return_token_type_ids=True,
            return_tensors="pt",
        )

        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]

        output = {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "aus": aus,
            "target": y,
        }

        # output = {
        #     "ids": torch.tensor(ids, dtype=torch.long),
        #     "mask": torch.tensor(mask, dtype=torch.long),
        #     "aus": torch.tensor(aus, dtype=torch.float),
        #     "target": torch.tensor(y, dtype=torch.long),
        # }

        return output


class TextDataset(Dataset):
    def __init__(self, text_df, tokenizer, max_len):
        super(TextDataset, self).__init__()
        """TextDataset is a dataset containing utterances along with their CS. Used as a baseline for comparison with our MultiModal architecture.

        Args:
            text_df (_type_): pandas DataFrame containing the utterances along with the dyad, Session, Period, Timestamps and Labels informations.
            tokenizer (_type_): tokenizer to use for Bert-based models
            max_len (_type_): maximum length of utterances in terms of tokens
        """
        self.utterances = text_df

        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.utterances)

    def __getitem__(self, idx):
        """Getitem method

        Returns:
            dictionary: dictionary containing the aforementioned information. It does the following :
            Given an index -> utterances + labels.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        participant_id = (
            int(not pd.isna(self.utterances.iloc[idx]["P1"]))
            + int(not pd.isna(self.utterances.iloc[idx]["P2"])) * 2
        )

        session = self.utterances.iloc[idx]["Session"]
        period = self.utterances.iloc[idx]["Period"]
        period = "Social" * int(period[0] == "S") + "Tutoring" * int(period[0] == "T")

        # Depending on the situation, you can extend the list of classes. Choice was made here to rule out VSN, as at the current time, I had no Data that properly distinguishes between
        # off-task talk and VSN, the Conversational Strategy.

        y = torch.Tensor(
            self.utterances.iloc[idx][["PR", "SD", "QE", "None"]]  # "VSN",
            .fillna("")
            .apply(len)
        ).squeeze()

        if True:
            current_text = self.utterances.iloc[idx][f"P{participant_id}"]
            context = self.utterances.iloc[idx]["Context"]
            text = (
                " <P> ".join(context)
                + " <start> "
                + current_text
                + " <end>".replace("[", "")
                .replace("]", "")
                .replace("(", "")
                .replace(")", "")
                .replace("laughter", "")
            )

        else:
            text = self.utterances.iloc[idx][f"P{participant_id}"]
            text = (
                "".join(text.split())
                .replace("[", "")
                .replace("]", "")
                .replace("(", "")
                .replace(")", "")
                .replace("laughter", "")
            ) + ""

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            return_token_type_ids=True,
            return_tensors="pt",
        )

        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]

        output = {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "target": y,
        }

        # output = {
        #     "ids": torch.tensor(ids, dtype=torch.long),
        #     "mask": torch.tensor(mask, dtype=torch.long),
        #     "aus": torch.tensor(aus, dtype=torch.float),
        #     "target": torch.tensor(y, dtype=torch.long),
        # }

        return output


class FocalLoss(nn.Module):
    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float = -1.0,
        weight=None,
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

    if not is_text:
        # Here, we use enumerate(training_loader) instead of
        # iter(training_loader) so that we can track the batch
        # index and do some intra-epoch reporting
        for i, data in tqdm(enumerate(train_loader)):
            # Every data instance is an input + label pair
            input_ids, attention_mask, aus, labels = (
                data["ids"].to(device, dtype=torch.long),
                data["mask"].to(device, dtype=torch.long),
                data["aus"].to(device, dtype=torch.float),
                data["target"].to(device, dtype=torch.float),
            )
            # labels = 1 - labels

            # Zero your gradients for every batch!
            optimizer.zero_grad()

            # Make predictions for this batch
            outputs = model(input_ids, attention_mask, aus)
            # Compute the loss and its gradients
            loss = loss_fn(outputs, labels)
            loss.backward()

            # Adjust learning weights
            optimizer.step()

            # Gather data and report
            running_loss += loss.item()
            if i % 200 == 199:
                last_loss = running_loss / 200  # loss per batch
                print("  batch {} loss: {}".format(i + 1, last_loss))
                running_loss = 0.0
        return last_loss

    elif is_text:
        for i, data in tqdm(enumerate(train_loader)):
            # Every data instance is an input + label pair
            input_ids, attention_mask, labels = (
                data["ids"].to(device, dtype=torch.long),
                data["mask"].to(device, dtype=torch.long),
                data["target"].to(device, dtype=torch.float),
            )

            # labels = 1 - labels
            # Zero your gradients for every batch!
            optimizer.zero_grad()

            # Make predictions for this batch
            outputs = model(input_ids, attention_mask)

            # Compute the loss and its gradients
            loss = loss_fn(outputs, labels)
            loss.backward()

            # Adjust learning weights
            optimizer.step()

            # Gather data and report
            running_loss += loss.item()
            if i % 100 == 99:
                last_loss = running_loss / 200  # loss per batch
                print("  batch {} loss: {}".format(i + 1, last_loss))
                running_loss = 0.0
    return last_loss


def train_one_epoch_3():
    """Trains the model using a fork-loss, linear combination of several losses, word I just invented. Used for experiments, to disregard except if really needed.

    Returns:
        _type_: _description_
    """
    running_loss = 0.0
    last_loss = 0.0

    for i, data in tqdm(enumerate(train_loader)):
        # Every data instance is an input + label pair
        input_ids, attention_mask, aus, labels = (
            data["ids"].to(device, dtype=torch.long),
            data["mask"].to(device, dtype=torch.long),
            data["aus"].to(device, dtype=torch.float),
            data["target"].to(device, dtype=torch.float),
        )

        # labels = 1 - labels
        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs_t, output_au, output_mul = model(input_ids, attention_mask, aus)

        # Compute the loss and its gradients
        loss = (
            loss_fn(outputs_t, labels) / 3
            + loss_fn(output_au, labels) / 3
            + loss_fn(output_mul, labels) / 3
        )
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % 100 == 99:
            last_loss = running_loss / 200  # loss per batch
            print("  batch {} loss: {}".format(i + 1, last_loss))
            running_loss = 0.0
    return last_loss


def read_dataset(text_path, aus_path):
    """Using the path to the text-transcript and extracted AUs, returns properly formatted dataframes for the approach I chose here. This is not a mandatory choice, just mine.

    Args:
        text_path (_type_): path to the utterance dataset containing timestamps and conversational strategies.
        aus_path (_type_): path to the AUs dataset containing timestamps.

    Returns:
        _type_: two dataframes, one containing text utterances and the aligned
    """

    # The following dyads were ruled out based on the analysis dit by Aishik and Ewen on June of 2022 -> alignement problems. Currently the alignement is being worked on.

    df_main = pd.read_csv(text_path)
    df_main = df_main[(df_main["Dyad"] <= 11) & (df_main["Dyad"] >= 3)]
    df_main = df_main[~((df_main["Dyad"] == 8) & (df_main["Session"] == 1))]
    df_main = df_main[~((df_main["Dyad"] == 11) & (df_main["Session"] == 2))]

    # Personal choice : no sentence should be longer than 10.0s. This solved a lot of alignement problems at the time.

    df_main["Duration_s"].apply(lambda val: min(float(val), 10.0))
    df_main["Dialog"] = df_main[["Dyad", "Session"]].apply(
        lambda row: str(row.Dyad) + "_" + str(row.Session), axis=1
    )
    df_main["Time"] = pd.to_timedelta(df_main["Time_begin"]).dt.total_seconds()

    # Get a dictionary of the dialog history for each turn

    input_sentences = []
    # df_main["VSN"] = (
    #     df_main[["SV_Tutor", "SV_Tutee"]]
    #     .apply(lambda row: np.sum(row.astype(str)), axis=1)
    #     .replace("xnan", "x")
    #     .replace("nanx", "x")
    #     .replace("xx", "x")
    #     .replace("nannan", "")
    # )
    df_main["PR"] = (
        df_main[["PR_Tutor", "PR_Tutee"]]
        .apply(lambda row: np.sum(row.astype(str)), axis=1)
        .replace("xnan", "x")
        .replace("nanx", "x")
        .replace("xx", "x")
        .replace("nannan", "")
    )
    df_main["SD"] = (
        df_main[["SD_Tutor", "SD_Tutee"]]
        .apply(lambda row: np.sum(row.astype(str)), axis=1)
        .replace("xnan", "x")
        .replace("nanx", "x")
        .replace("xx", "x")
        .replace("nannan", "")
    )
    df_main["QE"] = (
        df_main[["QE_Tutor", "QE_Tutee"]]
        .apply(lambda row: np.sum(row.astype(str)), axis=1)
        .replace("xnan", "x")
        .replace("nanx", "x")
        .replace("xx", "x")
        .replace("nannan", "")
    )
    df_main["None"] = (
        df_main[
            [
                "PR_Tutor",
                "PR_Tutee",
                "SD_Tutor",
                "SD_Tutee",
                "QE_Tutor",
                "QE_Tutee",
            ]  # "SV_Tutor", "SV_Tutee"
        ]
        .isna()
        .all(axis="columns")
        .replace(True, "x")
        .replace(False, "")
        .replace("nannan", "")
    )  #'SV_Tutor','SV_Tutee',

    df_main = df_main.replace("", np.NaN)

    context_dict = {}

    for i in df_main.Dialog.unique():
        dialog_df = df_main.loc[df_main["Dialog"] == i]
        dialog_df.sort_values(by=["Time"], inplace=True)

        prev = []
        for j, row in enumerate(dialog_df.iterrows()):
            if isinstance(row[1]["P1"], str):
                sentence = row[1]["P1"]
                context_dict[str(row[1]["Dialog"]) + "_" + str(row[1]["Time"])] = prev[
                    -3:
                ]
                prev.append(sentence)
            elif isinstance(row[1]["P2"], str):
                sentence = row[1]["P2"]
                context_dict[str(row[1]["Dialog"]) + "_" + str(row[1]["Time"])] = prev[
                    -3:
                ]
                prev.append(sentence)

    df_main["Marker"] = df_main["Time"] + df_main["Duration_s"]
    df_main = df_main[df_main["Marker"] > 12.0]

    P1_sentences = df_main["P1"].values
    P2_sentences = df_main["P2"].values

    # Make a list of utterances in each turn i.e. it either comes from P1 or P2(not both) and we merge them here

    for i in range(len(P1_sentences)):
        if isinstance(P1_sentences[i], str):
            input_sentence = P1_sentences[i]
        elif isinstance(P2_sentences[i], str):
            input_sentence = P2_sentences[i]
        # Apply model
        input_sentences.append(input_sentence)

    df_main["Dialog_time"] = df_main.apply(
        lambda row: str(row.Dialog) + "_" + str(row.Time), axis=1
    )
    df_main["Context"] = df_main.apply(
        lambda row: context_dict[row.Dialog_time], axis=1
    )

    df_lookup = pd.read_csv(aus_path)

    df_main = df_main[~((~pd.isna(df_main.P1)) & (~pd.isna(df_main.P2)))]
    df_main = df_main.drop(
        columns=[
            "SV_Tutor",
            "SV_Tutee",
            "PR_Tutor",
            "PR_Tutee",
            "SD_Tutor",
            "SD_Tutee",
            "QE_Tutor",
            "QE_Tutee",
        ]
    )

    return df_main, df_lookup


def round5(number):
    """I used to be a very smart mathematics student. I came up with this clever technique to round a number to .5, using all of my neurons and my amazing ability to use python.
    I plan to publish this amazing and disruptive work soon in Nature. Please do not plagiate.

    Args:
        number (number): a number

    Returns:
        number: the same number rounded to the closest .5 value.
    """
    return round(number * 2) / 2


def class_split(tot_dataframe):
    """Using a Dataframe containing utterances and classes, returns a forced split of the Dataset to even out the classes between Train, Test and Validation.

    Args:
        tot_dataframe (_type_): Dataframe containing Utterances and their associated CS. Output of the read_dataset() function.

    Returns:
        _type_: Three datasets containing an amount proportional to their size of each class.
    """
    train_data, test_data, val_data = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    df = tot_dataframe

    class_list = ["PR", "QE", "SD", "None"]  # "VSN"

    # df_none = df[df["None"]=='x']
    # df_pr = df[df["PR"]=='x']
    # df_sd = df[df["SD"]=='x']
    # df_qe = df[df["QE"]=='x']
    # # df_vsn = df[df["VSN"]=='x']

    # df_list = [df_pr,df_qe,df_sd,df_none] # df_vsn

    total_length = 0
    for i, class_ in enumerate(class_list):
        label_df = df[df[class_] == "x"]
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
                [train_data, label_df.iloc[: int(current_length * 0.75)]]
            )
            test_data = pd.concat(
                [
                    test_data,
                    label_df.iloc[
                        int(current_length * 0.75) : int(current_length * 0.875)
                    ],
                ]
            )
            val_data = pd.concat(
                [val_data, label_df.iloc[int(current_length * 0.875) :]]
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


if __name__ == "__main__":
    is_au = False
    import datetime

    now = datetime.datetime.now().strftime(r"%y_%m_%d_%H")

    model_l = [
        # models.BertGRUConcat(),
        # models.BertDenseConcat(),
        models_novsn.BertClassif(dropout1=0.15),
        # models.BertLSTMConcat(),
    ]

    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

    #     models_novsn.BertDenseConcat(),
    # models_novsn.BertGRUConcat(),
    #

    # models_aus_only.BertClassif(),

    model_names_l = [type(m_).__name__ for m_ in model_l]

    text_df, aus_df = read_dataset(text_path, aus_path)

    for i, model_ in enumerate(model_l):
        print(f"===========\nTRAINING MODEL {type(model_).__name__} :")
        EPOCHS = 3
        MAX_LEN = 128

        model = model_
        model.to(device)

        train_df, test_df, val_df = class_split(text_df)

        if type(model).__name__ == "BertClassif":
            train_data, test_data, val_data = (
                TextDataset(train_df, tokenizer, MAX_LEN),
                TextDataset(test_df, tokenizer, MAX_LEN),
                TextDataset(val_df, tokenizer, MAX_LEN),
            )
        elif type(model).__name__ == "BertDenseConcat":
            train_data, test_data, val_data = (
                MultiModalDataset(train_df, aus_df, tokenizer, MAX_LEN),
                MultiModalDataset(test_df, aus_df, tokenizer, MAX_LEN),
                MultiModalDataset(val_df, aus_df, tokenizer, MAX_LEN),
            )
        else:
            train_data, test_data, val_data = (
                MultiModalDatasetRNN(train_df, aus_df, tokenizer, MAX_LEN),
                MultiModalDatasetRNN(test_df, aus_df, tokenizer, MAX_LEN),
                MultiModalDatasetRNN(val_df, aus_df, tokenizer, MAX_LEN),
            )

        train_loader, test_loader, val_loader = (
            DataLoader(train_data, batch_size=8, shuffle=True, num_workers=8),
            DataLoader(test_data, batch_size=8, shuffle=True, num_workers=8),
            DataLoader(val_data, batch_size=8, shuffle=True, num_workers=8),
        )
        # Order of the columns : PR, SD, QE, NONE
        class_proportion_5 = [
            0.11898949156140537,
            0.005944167285850758,
            0.03916781657998089,
            0.008650886317800657,
            0.8272476382549623,
        ]
        class_proportion_4 = [0.0306, 0.24, 0.0451, 0.70]

        weights = torch.tensor(
            [(1 - x) / x for x in class_proportion_4], dtype=torch.float
        ).to(device)

        loss_fn = FocalLoss(gamma=1, alpha=0.6988592276480736)  # alpha = .9
        loss_fn_cpu = FocalLoss(gamma=1, alpha=0.6988592276480736)  # alpha = .9
        # loss_fn = torch.nn.BCEWithLogitsLoss() #weight=weights
        # loss_fn_cpu = torch.nn.BCEWithLogitsLoss() #weight=weights.to("cpu")

        params = model.parameters()
        optimizer = torch.optim.Adam(params, lr=1e-5)

        timestamp = pd.datetime.now().strftime("%Y%m%d_%H%M%S")
        epoch_number = 0
        best_vloss = 100

        embedder = str(type(tokenizer).__name__)[:-9]
        num_classes = (
            int(model_l[0].class_num == 5) * "vsn"
            + int(model_l[0].class_num == 4) * "novsn"
        )

        MODEL_DIR = f"models/{embedder}_nocontext_{num_classes}_{now}/"

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
            if is_au:
                with torch.no_grad():
                    for i, vdata in enumerate(val_loader):
                        vaus, vlabels = (
                            vdata["aus"].to(device, dtype=torch.float),
                            vdata["target"].to(device, dtype=torch.float),
                        )
                        # vlabels = 1 - vlabels

                        voutputs = model(vaus)
                        vloss = loss_fn(voutputs, vlabels)
                        running_vloss += vloss

                    avg_vloss = running_vloss / (i + 1)
                    print("LOSS train {} valid {}".format(avg_loss, avg_vloss))

                    # Track best performance, and save the model's state
                    if avg_vloss < best_vloss:
                        best_vloss = avg_vloss
                        model_name = "{}".format(type(model).__name__)
                        if not os.path.exists(MODEL_DIR):
                            os.mkdir(MODEL_DIR)
                        torch.save(
                            model.state_dict(), MODEL_DIR + model_name,
                        )

                    epoch_number += 1
            elif type(model).__name__ == "BertClassif":
                with torch.no_grad():
                    for i, vdata in enumerate(val_loader):
                        vinput_ids, vattention_mask, vlabels = (
                            vdata["ids"].to(device, dtype=torch.long),
                            vdata["mask"].to(device, dtype=torch.long),
                            vdata["target"].to(device, dtype=torch.float),
                        )

                        voutputs = model(vinput_ids, vattention_mask)
                        # vlabels = 1 - vlabels
                        vloss = loss_fn(voutputs, vlabels)
                        running_vloss += vloss

                    avg_vloss = running_vloss / (i + 1)
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
            elif type(model).__name__ == "GRUClassif3":
                with torch.no_grad():
                    for i, vdata in enumerate(val_loader):
                        vinput_ids, vattention_mask, vaus, vlabels = (
                            vdata["ids"].to(device, dtype=torch.long),
                            vdata["mask"].to(device, dtype=torch.long),
                            vdata["aus"].to(device, dtype=torch.float),
                            vdata["target"].to(device, dtype=torch.float),
                        )

                        # vlabels = 1 - vlabels

                        voutputs_t, voutputs_au, voutputs_mul = model(
                            vinput_ids, vattention_mask, vaus
                        )
                        vloss = (
                            loss_fn(voutputs_t, vlabels) / 3
                            + loss_fn(voutputs_au, vlabels) / 3
                            + loss_fn(voutputs_mul, vlabels) / 3
                        )
                        running_vloss += vloss

                    avg_vloss = running_vloss / (i + 1)
                    print("LOSS train {} valid {}".format(avg_loss, avg_vloss))

                    # Track best performance, and save the model's state
                    if avg_vloss < best_vloss:
                        best_vloss = avg_vloss
                        model_name = "{}".format(type(model).__name__)
                        if not os.path.exists(MODEL_DIR):
                            os.mkdir(MODEL_DIR)
                        torch.save(
                            model.state_dict(), MODEL_DIR + model_name,
                        )

                    epoch_number += 1

            else:
                with torch.no_grad():
                    for i, vdata in enumerate(val_loader):
                        vinput_ids, vattention_mask, vaus, vlabels = (
                            vdata["ids"].to(device, dtype=torch.long),
                            vdata["mask"].to(device, dtype=torch.long),
                            vdata["aus"].to(device, dtype=torch.float),
                            vdata["target"].to(device, dtype=torch.float),
                        )

                        # vlabels = 1 - vlabels

                        voutputs = model(vinput_ids, vattention_mask, vaus)
                        vloss = loss_fn(voutputs, vlabels)
                        running_vloss += vloss

                    avg_vloss = running_vloss / (i + 1)
                    print("LOSS train {} valid {}".format(avg_loss, avg_vloss))

                    # Track best performance, and save the model's state
                    if avg_vloss < best_vloss:
                        best_vloss = avg_vloss
                        model_name = "{}".format(type(model).__name__)
                        if not os.path.exists(MODEL_DIR):
                            os.mkdir(MODEL_DIR)
                        torch.save(
                            model.state_dict(), MODEL_DIR + model_name,
                        )

                    epoch_number += 1
