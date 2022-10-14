import pandas as pd

import torch
from torch import cuda

import transformers
from transformers import DistilBertTokenizer, DistilBertModel
from transformers import logging

logging.set_verbosity_warning()
logging.set_verbosity_error()
# Setting up the device for GPU usage
device = "cuda" if cuda.is_available() else "cpu"


class AUDense(torch.nn.Module):
    """
    This model is a bi-modal classifier using a bilinear fusion layer for the two modalities.
    Text is embedded through pre-trained BERT, Action Units are embedded through two dense layers.
    """

    def __init__(self):
        super(AUDense, self).__init__()

        # Define the AUs pipeline
        self.au_embedder1 = torch.nn.Linear(17 * 12 * 2, 128)  # 408 * 128 -> 128
        self.au_embedder2 = torch.nn.Linear(128, 128)

        self.dropout = torch.nn.Dropout(0.25)

        # Define the classifier
        self.classifier1 = torch.nn.Linear(128, 80)
        self.classifier2 = torch.nn.Linear(80, 4)

    def forward(self, aus):
        aus = torch.reshape(self.dropout(aus), (-1, 408))

        # Forward pass of the AUs
        output_2 = self.au_embedder1(aus)
        pooler2 = torch.nn.ReLU()(output_2)
        pooler2 = self.dropout(pooler2)

        pooler2 = self.au_embedder2(pooler2)
        pooler2 = torch.nn.ReLU()(pooler2)
        pooler2 = self.dropout(pooler2)

        # Fusion of the modalities
        fusion = self.classifier1(pooler2)
        fusion = torch.nn.ReLU()(fusion)

        # Classification of the fusion
        output = self.classifier2(fusion)

        return output


class AUGRU(torch.nn.Module):
    """
    This model is a bi-modal classifier using a bilinear fusion layer for the two modalities.
    Text is embedded through pre-trained BERT, Action Units are embedded through two dense layers.
    """

    def __init__(self):
        super(AUGRU, self).__init__()

        # Define the AUs pipeline
        self.au_embedder1 = torch.nn.GRU(
            input_size=17, hidden_size=128, bidirectional=True, batch_first=True
        )  # 408 * 128 -> 128
        self.au_embedder2 = torch.nn.Linear(256, 128)
        self.dropout = torch.nn.Dropout(0.25)

        # Define the classifier
        self.classifier1 = torch.nn.Linear(128, 80)
        self.classifier2 = torch.nn.Linear(80, 4)

    def forward(self, aus):

        aus = torch.reshape(self.dropout(aus), (-1, 24, 17))

        # Forward pass of the AUs
        output_2, _ = self.au_embedder1(aus)
        output_2 = output_2[:, -1, :]
        pooler2 = torch.nn.ReLU()(output_2)
        pooler2 = self.dropout(pooler2)

        pooler2 = self.au_embedder2(pooler2)
        pooler2 = torch.nn.ReLU()(pooler2)
        pooler2 = self.dropout(pooler2)

        # Fusion of the modalities
        fusion = self.classifier1(pooler2)
        fusion = torch.nn.ReLU()(fusion)

        # Classification of the fusion
        output = self.classifier2(fusion)

        return output


class AULSTM(torch.nn.Module):

    """
    This model is a bi-modal classifier using a bilinear fusion layer for the two modalities.
    Text is embedded through pre-trained BERT, Action Units are embedded through two dense layers.
    """

    def __init__(self):
        super(AULSTM, self).__init__()

        # Define the AUs pipeline
        self.au_embedder1 = torch.nn.LSTM(
            input_size=17, hidden_size=128, bidirectional=True, batch_first=True
        )  # 408 * 128 -> 128
        self.au_embedder2 = torch.nn.Linear(256, 128)

        self.dropout = torch.nn.Dropout(0.25)

        # Define the classifier
        self.classifier1 = torch.nn.Linear(128, 80)
        self.classifier2 = torch.nn.Linear(80, 4)

    def forward(self, aus):

        aus = torch.reshape(self.dropout(aus), (-1, 24, 17))
        # Forward pass of the AUs
        output_2, _ = self.au_embedder1(aus)
        output_2 = output_2[:, -1, :]
        pooler2 = torch.nn.ReLU()(output_2)
        pooler2 = self.dropout(pooler2)

        pooler2 = self.au_embedder2(pooler2)
        pooler2 = torch.nn.ReLU()(pooler2)
        pooler2 = self.dropout(pooler2)

        fusion = self.classifier1(pooler2)
        fusion = torch.nn.ReLU()(fusion)

        # Classification of the fusion
        output = self.classifier2(fusion)

        return output
