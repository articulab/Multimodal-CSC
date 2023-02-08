from cgi import test
from email import iterators

# Importing stock ml libraries
import warnings
import json
import ast

import warnings


def f():
    warnings.warn("deprecated", DeprecationWarning)


with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    f()

warnings.simplefilter("ignore")

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn import metrics
import transformers
from torch.nn import functional as F
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, DistilBertTokenizer, DistilBertModel
import logging
import os
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)

from train_bimodal import (
    class_split,
    FocalLoss,
    TextDataset,
    MultiModalDataset,
    MultiModalDatasetRNN,
)
import models_aus_only, models_roberta_novsn, models_novsn, models_roberta, models

from ray.tune import grid_search

from torch import cuda

# from pytorch_lightning.core.lightning import LightningModule
# from pytorch_lightning.loggers import TensorBoardLogger
# from pytorch_lightning.callbacks.early_stopping import EarlyStopping

import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler

import sklearn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

MAX_LEN = 128
TRAIN_BATCH_SIZE = 4
TEST_BATCH_SIZE = 4
VALID_BATCH_SIZE = 1
EPOCHS = 3
LEARNING_RATE = 1e-05

TXT_PATH = r"C:\Users\Alafate\Desktop\Work\Programming\non_verbal\2016_work\data\2016_important\2016_text_only.csv"
AUS_PATH = r"C:\Users\Alafate\Desktop\Work\Programming\non_verbal\2016_work\data\2016_important\2016_aus_med_500ms.csv"


def objective_function(model, data_loader, *args, **kwargs):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.eval()
    preds = np.asarray([])
    targs = np.asarray([])
    with torch.no_grad():
        for batch_idx, inputs in enumerate(data_loader):
            # We set this just for the example to run quickly.
            if batch_idx * len(inputs) > TEST_BATCH_SIZE:
                break

            ids, mask, aus = inputs["ids"], inputs["mask"], inputs["aus"]
            target = np.squeeze(np.asarray(inputs["target"].to("cpu"), dtype=np.uint))
            outputs = model(ids, mask, aus)

            # We gather when the prediction probability is over .5 (arbitrary).
            prediction = np.asarray(torch.sigmoid(outputs).to("cpu")) > 0.4
            try:
                # We then transform those places in 1 and the others in 0 : int(True) = 1.
                prediction = np.asarray([[int(x) for x in y] for y in prediction])
            except:
                prediction = np.asarray([int(x) for x in prediction])

            if batch_idx == 0:
                targs = target
                preds = prediction
            else:
                # targs.extend(target)
                targs = np.concatenate((targs, target), axis=0)
                preds = np.concatenate((preds, prediction), axis=0)

    print(f"targs : {targs} \n Preds : {preds}")
    f1_score = sklearn.metrics.f1_score(targs, preds, average="macro", zero_division=1)

    return f1_score


def test_best_model(best_trial):
    best_trained_model = models_novsn.BertGRUConcat(
        dropout1=best_trial.config["dropout1"], gru_layers=1
    )
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    best_trained_model.to(device)

    checkpoint_path = os.path.join(best_trial.checkpoint.value, "checkpoint")

    model_state, optimizer_state = torch.load(checkpoint_path)
    best_trained_model.load_state_dict(model_state)

    df = pd.read_csv("data/2016_important/2016_text_only.csv")
    aus_df = pd.read_csv("data/2016_important/2016_aus_med_500ms.csv")
    tokenizer = DistilBertTokenizer.from_pretrained("bert-base-uncased")
    _, test_data, _ = class_split(df)

    test_dataset = (MultiModalDatasetRNN(test_data, aus_df, tokenizer, MAX_LEN),)

    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, inputs in enumerate(test_loader):
            # We set this just for the example to run quickly.
            if batch_idx * len(inputs) > TEST_BATCH_SIZE:
                break

            ids, mask, aus = inputs["ids"], inputs["mask"], inputs["aus"]
            target = np.squeeze(np.asarray(inputs["target"].to("cpu"), dtype=np.uint))
            outputs = best_trained_model(ids, mask, aus)

            # We gather when the prediction probability is over .5 (arbitrary).
            prediction = np.asarray(torch.sigmoid(outputs).to("cpu")) > 0.5
            try:
                # We then transform those places in 1 and the others in 0 : int(True) = 1.
                prediction = np.asarray([[int(x) for x in y] for y in prediction])
            except:
                prediction = np.asarray([int(x) for x in prediction])

            if batch_idx == 0:
                targs = target
                preds = prediction
            else:
                # targs.extend(target)
                targs = np.concatenate((targs, target), axis=0)
                preds = np.concatenate((preds, prediction), axis=0)

    print(f"targs : {targs} \n Preds : {preds}")
    f1_score = sklearn.metrics.f1_score(
        targs, preds, average="weighted", zero_division=1
    )

    print("Best trial test set accuracy: {}".format(f1_score))


def read_dataset(text_path, aus_path):
    """_summary_

    Args:
        text_path (_type_): _description_
        aus_path (_type_): _description_

    Returns:
        _type_: _description_
    """
    df_main = pd.read_csv(text_path)
    df_main = df_main[(df_main["Dyad"] <= 11) & (df_main["Dyad"] >= 3)]
    df_main = df_main[~((df_main["Dyad"] == 8) & (df_main["Session"] == 1))]
    df_main = df_main[~((df_main["Dyad"] == 11) & (df_main["Session"] == 2))]

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
            ["PR_Tutor", "PR_Tutee", "SD_Tutor", "SD_Tutee", "QE_Tutor", "QE_Tutee"]
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
    # df_check = df_main[(df_main["PR"] == 'x') | (df_main["SD"] == 'x') | (df_main["QE"] == 'x')]
    # df_check.to_csv("labelled_data.csv")
    return df_main, df_lookup


def train_nvb(config, checkpoint_dir=None):
    tokenizer = DistilBertTokenizer.from_pretrained("bert-base-uncased")

    text_df, aus_df = read_dataset(TXT_PATH, AUS_PATH)
    train_df, test_df, val_df = class_split(text_df)
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

    model = models_novsn.BertGRUConcat(dropout1=config["dropout1"])

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
    model.to(device)

    criterion = FocalLoss(alpha=config["alpha"], gamma=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    for epoch in range(EPOCHS):
        running_loss = 0.0
        epoch_steps = 0
        for i, inputs in enumerate(train_loader):
            bert_id = inputs["ids"]
            bert_mask = inputs["mask"]
            aus = inputs["aus"]

            # get the inputs; data is a list of [inputs, labels]
            y = inputs["target"]

            output = model(input_ids=bert_id, attention_mask=bert_mask, aus=aus)
            loss = criterion(output, y.float())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # if (i+1) % 100 == 0:
            #    print (f'Epoch [{epoch+1}/{EPOCHS}], Step [{i+1}/{len(trainloader)}], Loss: {loss.item():.4f}')
        acc = objective_function(model, val_loader)
        tune.report(score=acc)
        with tune.checkpoint_dir(step=epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((model.state_dict(), optimizer.state_dict()), path)


if __name__ == "__main__":
    tokenizer = DistilBertTokenizer.from_pretrained("bert-base-uncased")

    text_df, aus_df = read_dataset(TXT_PATH, AUS_PATH)
    train_df, test_df, val_df = class_split(text_df)
    train_data, test_data, val_data = (
        MultiModalDatasetRNN(train_df, aus_df, tokenizer, MAX_LEN),
        MultiModalDatasetRNN(test_df, aus_df, tokenizer, MAX_LEN),
        MultiModalDatasetRNN(val_df, aus_df, tokenizer, MAX_LEN),
    )
    train_loader, test_loader, val_loader = (
        DataLoader(train_data, batch_size=8, shuffle=True, num_workers=4),
        DataLoader(test_data, batch_size=8, shuffle=True, num_workers=4),
        DataLoader(val_data, batch_size=8, shuffle=True, num_workers=4),
    )

    config = {
        "dropout1": tune.sample_from(lambda _: np.random.uniform(0.05, 0.5)),
        "alpha": tune.sample_from(lambda _: np.random.uniform(0.2, 1)),
    }

    scheduler = ASHAScheduler(max_t=EPOCHS, grace_period=1, reduction_factor=2)

    result = tune.run(
        train_nvb,
        num_samples = 60,
        config = config,
        metric = "score",
        mode = "max",
        scheduler=scheduler,
        resources_per_trial={"gpu": 0.25, "cpu": 4},
    )

    best_trial = result.get_best_trial("score", "max", "last")
    print("Best trial config: {}".format(best_trial.config))
    print(
        "Best trial final validation score: {}".format(best_trial.last_result["score"])
    )

    if ray.util.client.ray.is_connected():
        # If using Ray Client, we want to make sure checkpoint access
        # happens on the server. So we wrap `test_best_model` in a Ray task.
        # We have to make sure it gets executed on the same node that
        # ``tune.run`` is called on.
        from ray.util.ml_utils.node import force_on_current_node

        remote_fn = force_on_current_node(ray.remote(test_best_model))
        ray.get(remote_fn.remote(best_trial))
    else:
        test_best_model(best_trial)
