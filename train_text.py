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

text_path = "merged_df_2016.csv"

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
        # try : period = "Social" * int(period[0] == "S") + "Tutoring" * int(period[0] == "T")
        # except : period = "Preparation"
        # Depending on the situation, you can extend the list of classes. Choice was made here to rule out VSN, as at the current time, I had no Data that properly distinguishes between
        # off-task talk and VSN, the Conversational Strategy.

        y = torch.Tensor(
            self.utterances.iloc[idx][["PR", "SD", "QE", "VSN", "HD", "None"]]  # "VSN",
            .fillna("")
            .apply(len)
        ).squeeze()

        if False:
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
            return_tensors="pt",
        )

        ids = inputs['input_ids']
        mask = inputs['attention_mask']

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


def read_dataset(text_path):
    """Using the path to the text-transcript and extracted AUs, returns properly formatted dataframes for the approach I chose here. This is not a mandatory choice, just mine.

    Args:
        text_path (_type_): path to the utterance dataset containing timestamps and conversational strategies.
        aus_path (_type_): path to the AUs dataset containing timestamps.

    Returns:
        _type_: two dataframes, one containing text utterances and the aligned
    """

    # The following dyads were ruled out based on the analysis dit by Aishik and Ewen on June of 2022 -> alignement problems. Currently the alignement is being worked on.

    df_main = pd.read_csv(text_path)
    df_main = df_main[(df_main["Dyad"].isin([3,4,5,6,7,8,10]))].replace(pd.NA, "").replace("SV1A", "").replace(np.nan, "")
    df_main = df_main.rename(columns = {"SV2_P1" : "SV_Tutor", "SV2_P2" : "SV_Tutee"})

    df_main["Dialog"] = df_main[["Dyad", "Session"]].apply(
        lambda row: str(row.Dyad) + "_" + str(row.Session), axis=1
    )
    df_main["Time"] = pd.to_timedelta(df_main["Begin_time"]).dt.total_seconds()

    # Get a dictionary of the dialog history for each turn

    input_sentences = []

    vsn_unique = pd.unique(df_main[['SV_Tutor', 'SV_Tutee']].values.ravel('K'))
    # print(vsn_unique)
    df_main["VSN"] = (
        df_main[["SV_Tutor", "SV_Tutee"]]
        .apply(lambda row: int(np.sum(row.astype(str)) in ['SV1C', 'SV2A', 'SV3A', 'SV1D', 'SV2B', 'SV3C', 'SV1B', 'SV2C'])  * "x", axis=1)
    )
    pr_unique = pd.unique(df_main[['PR_Tutor', 'PR_Tutee']].values.ravel('K'))
    # print(pr_unique)
    df_main["PR"] = (
        df_main[["PR_Tutor", "PR_Tutee"]]
        .apply(lambda row: int(np.sum(row.astype(str)) in ['UL', 'LP', 'LPP', 'LPA', '0', 'not sure if this is praise', 'LPB']) * "x", axis=1)
    )

    df_main["SD"] = (
        df_main[["SD_Tutor", "SD_Tutee"]]
        .apply(lambda row: int(np.sum(row.astype(str)) == "SD") * "x", axis=1)
    )
    
    df_main["QE"] = (
        df_main[["SD_Tutor", "SD_Tutee"]]
        .apply(lambda row: int(np.sum(row.astype(str)) == "QE") * "x", axis=1)
    )

    hd_unique = pd.unique(df_main[['HD_Tutor', 'HD_Tutee']].values.ravel('K'))
    df_main["HD"] = (
        df_main[["HD_Tutor", "HD_Tutee"]]
        .apply(lambda row: int((np.sum(row.astype(str)) in hd_unique) & (len(np.sum(row.astype(str))) > 1)) * "x", axis=1)
    )

    df_main = df_main.replace("", np.nan)

    df_main["None"] = (
        df_main[
            [
                "PR_Tutor",
                "PR_Tutee",
                "SD_Tutor",
                "SD_Tutee",
                "HD_Tutor",
                "HD_Tutee",
                "SV_Tutor",
                "SV_Tutee"
            ]
        ]
        .isna()
        .all(axis="columns")
        .replace(True, "x")
        .replace(False, "")
    )

    df_main = df_main.replace(to_replace = "", value = np.nan)

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

    df_main = df_main[~((~pd.isna(df_main.P1)) & (~pd.isna(df_main.P2)))]
    df_main = df_main.drop(
        columns=[
            "SV_Tutor",
            "SV_Tutee",
            "PR_Tutor",
            "PR_Tutee",
            "SD_Tutor",
            "SD_Tutee",
            "HD_Tutor",
            "HD_Tutee"
        ]
    )
    df_main.to_csv("firsttry.csv")
    return df_main


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
    df = tot_dataframe

    if not class_list:
        class_list = ["PR", "SD", "QE", "VSN", "HD", "None"]

    # df_none = df[df["None"]=='x']
    # df_pr = df[df["PR"]=='x']
    # df_sd = df[df["SD"]=='x']
    # df_qe = df[df["QE"]=='x']
    # # df_vsn = df[df["VSN"]=='x']

    # df_list = [df_pr,df_qe,df_sd,df_none] # df_vsn

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
            input_ids, attention_mask, labels = (
                data["ids"].to(device, dtype=torch.long),
                data["mask"].to(device, dtype=torch.long),
                data["target"].to(device, dtype=torch.float),
            )

            # labels = 1 - labels
            # Zero your gradients for every batch!

            # Make predictions for this batch
            outputs = model(input_ids, attention_mask)

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

    model_l = [
        models.BertClassif(),
    ]

    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

    #     models_novsn.BertDenseConcat(),
    # models_novsn.BertGRUConcat(),
    #

    # models_aus_only.BertClassif(),

    model_names_l = [type(m_).__name__ for m_ in model_l]

    text_df = read_dataset(text_path)

    for i, model_ in enumerate(model_l):
        print(f"===========\nTRAINING MODEL {type(model_).__name__} :")
        EPOCHS = 3
        MAX_LEN = 128

        model = model_
        model.to(device)

        train_df, test_df, val_df = class_split(text_df)

        train_data, test_data, val_data = (
            TextDataset(train_df, tokenizer, MAX_LEN),
            TextDataset(test_df, tokenizer, MAX_LEN),
            TextDataset(val_df, tokenizer, MAX_LEN),
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

        embedder = str(type(tokenizer).__name__)[:-9]
        num_classes = 6

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

            fin_targets, fin_outputs = list(), list()

            with torch.no_grad():
                for i, vdata in enumerate(val_loader):

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