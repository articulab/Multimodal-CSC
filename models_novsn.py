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


class BertDenseConcat(torch.nn.Module):
    """
    This model is a bi-modal classifier using a bilinear fusion layer for the two modalities.
    Text is embedded through pre-trained BERT, Action Units are embedded through two dense layers.
    """

    def __init__(self):
        super(BertDenseConcat, self).__init__()

        # Define the Bert pipeline
        self.bert_layer = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.bert_dense = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(0.25)
        self.class_num = 4
        # Define the AUs pipeline
        self.au_embedder1 = torch.nn.Linear(17 * 12 * 2, 128)  # 408 * 128 -> 128
        self.au_embedder2 = torch.nn.Linear(128, 128)

        # Define the classifier
        self.classifier1 = torch.nn.Linear(896, 512)
        self.classifier2 = torch.nn.Linear(512, 4)

    def forward(self, input_ids, attention_mask, aus):
        input_ids = torch.reshape(input_ids, [-1, 128])
        attention_mask = torch.reshape(attention_mask, [-1, 128])
        aus = torch.reshape(aus, [-1, 17 * 12 * 2])
        # Forward pass of the toplkenized text
        output_1 = self.bert_layer(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = output_1[0]
        pooler1 = hidden_state[:, 0]

        pooler1 = self.bert_dense(pooler1)
        pooler1 = torch.nn.ReLU()(pooler1)
        pooler1 = self.dropout(pooler1)

        # Forward pass of the AUs
        output_2 = self.au_embedder1(aus)
        pooler2 = torch.nn.ReLU()(output_2)
        pooler2 = self.dropout(pooler2)

        pooler2 = self.au_embedder2(pooler2)
        pooler2 = torch.nn.ReLU()(pooler2)
        pooler2 = self.dropout(pooler2)

        # Fusion of the modalities
        concat_inputs = torch.cat([pooler1, pooler2], 1)
        fusion = self.classifier1(concat_inputs)
        fusion = torch.nn.ReLU()(fusion)

        # Classification of the fusion
        output = self.classifier2(fusion)

        return output


class BertGRUConcat(torch.nn.Module):
    """
    This model is a bi-modal classifier using a bilinear fusion layer for the two modalities.
    Text is embedded through pre-trained BERT, Action Units are embedded through two dense layers.
    """

    def __init__(self, dropout1: float = 0.110051):
        super(BertGRUConcat, self).__init__()

        # Define the Bert pipeline
        self.bert_layer = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.bert_dense = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(dropout1)
        self.class_num = 4
        # Define the AUs pipeline
        self.au_embedder1 = torch.nn.GRU(
            input_size=17, hidden_size=128, bidirectional=True, batch_first=True
        )  # 408 * 128 -> 128
        self.au_embedder2 = torch.nn.Linear(256, 128)

        # Define the classifier
        self.classifier1 = torch.nn.Linear(896, 512)
        self.classifier2 = torch.nn.Linear(512, 4)

    def forward(self, input_ids, attention_mask, aus):

        input_ids = torch.reshape(input_ids, [-1, 128])
        attention_mask = torch.reshape(attention_mask, [-1, 128])
        aus = torch.reshape(aus, [-1, 12 * 2, 17])

        # Forward pass of the toplkenized text
        output_1 = self.bert_layer(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = output_1[0]
        pooler1 = hidden_state[:, 0]

        pooler1 = self.bert_dense(pooler1)
        pooler1 = torch.nn.ReLU()(pooler1)
        pooler1 = self.dropout(pooler1)

        # Forward pass of the AUs
        output_2, _ = self.au_embedder1(aus)
        output_2 = output_2[:, -1, :]
        pooler2 = torch.nn.ReLU()(output_2)
        pooler2 = self.dropout(pooler2)

        pooler2 = self.au_embedder2(pooler2)
        pooler2 = torch.nn.ReLU()(pooler2)
        pooler2 = self.dropout(pooler2)

        # Fusion of the modalities
        concat_inputs = torch.cat([pooler1, pooler2], 1)
        fusion = self.classifier1(concat_inputs)
        fusion = torch.nn.ReLU()(fusion)

        # Classification of the fusion
        output = self.classifier2(fusion)

        return output


class BertLSTMConcat(torch.nn.Module):

    """
    This model is a bi-modal classifier using a bilinear fusion layer for the two modalities.
    Text is embedded through pre-trained BERT, Action Units are embedded through two dense layers.
    """

    def __init__(self):
        super(BertLSTMConcat, self).__init__()

        # Define the Bert pipeline
        self.bert_layer = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.bert_dense = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(0.25)
        self.class_num = 4
        # Define the AUs pipeline
        self.au_embedder1 = torch.nn.LSTM(
            input_size=17, hidden_size=128, bidirectional=True, batch_first=True
        )  # 408 * 128 -> 128
        self.au_embedder2 = torch.nn.Linear(256, 128)

        # Define the classifier
        self.classifier1 = torch.nn.Linear(896, 512)
        self.classifier2 = torch.nn.Linear(512, 4)

    def forward(self, input_ids, attention_mask, aus):

        input_ids = torch.reshape(input_ids, [-1, 128])
        attention_mask = torch.reshape(attention_mask, [-1, 128])
        aus = torch.reshape(aus, [-1, 12 * 2, 17])

        # Forward pass of the toplkenized text
        output_1 = self.bert_layer(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = output_1[0]
        pooler1 = hidden_state[:, 0]

        pooler1 = self.bert_dense(pooler1)
        pooler1 = torch.nn.ReLU()(pooler1)
        pooler1 = self.dropout(pooler1)

        # Forward pass of the AUs
        output_2, _ = self.au_embedder1(aus)
        output_2 = output_2[:, -1, :]
        pooler2 = torch.nn.ReLU()(output_2)
        pooler2 = self.dropout(pooler2)

        pooler2 = self.au_embedder2(pooler2)
        pooler2 = torch.nn.ReLU()(pooler2)
        pooler2 = self.dropout(pooler2)

        # Fusion of the modalities
        concat_inputs = torch.cat([pooler1, pooler2], 1)
        fusion = self.classifier1(concat_inputs)
        fusion = torch.nn.ReLU()(fusion)

        # Classification of the fusion
        output = self.classifier2(fusion)

        return output


class BertClassif(torch.nn.Module):
    """
    This model is a bi-modal classifier using a bilinear fusion layer for the two modalities.
    Text is embedded through pre-trained BERT, Action Units are embedded through two dense layers.
    """

    def __init__(self, dropout1=0.1):
        super(BertClassif, self).__init__()

        # Define the Bert pipeline
        self.bert_layer = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.bert_dense = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(dropout1)
        self.class_num = 4
        # Define the classifier
        self.classifier = torch.nn.Linear(768, 4)

    def forward(self, input_ids, attention_mask):

        input_ids = torch.reshape(input_ids, [-1, 128])
        attention_mask = torch.reshape(attention_mask, [-1, 128])

        # Forward pass of the toplkenized text
        output_1 = self.bert_layer(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = output_1[0]
        pooler1 = hidden_state[:, 0]

        pooler1 = self.bert_dense(pooler1)
        pooler1 = torch.nn.ReLU()(pooler1)
        pooler1 = self.dropout(pooler1)

        # Classification of the fusion
        output = self.classifier(pooler1)

        return output
