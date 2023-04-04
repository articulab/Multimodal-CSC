import pandas as pd

import torch
from torch import cuda

import transformers
from transformers import BertModel, DistilBertTokenizer, DistilBertModel
from transformers import logging
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

logging.set_verbosity_warning()
logging.set_verbosity_error()
# Setting up the device for GPU usage
device = "cuda" if cuda.is_available() else "cpu"


class BertDenseBilinearFusion(torch.nn.Module):
    """
    This model is a bi-modal classifier using a bilinear fusion layer for the two modalities.
    Text is embedded through pre-trained BERT, Action Units are embedded through two dense layers.
    """

    def __init__(self):
        super(BertDenseBilinearFusion, self).__init__()

        # Define the Bert pipeline
        self.bert_layer = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.bert_dense = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(0.25)

        # Define the AUs pipeline
        self.au_embedder1 = torch.nn.Linear(17 * 12 * 2, 128)  # 408 * 128 -> 128
        self.au_embedder2 = torch.nn.Linear(128, 128)

        # Define the fusion layer
        self.fusion_block = fusions.Block([768, 128], 64)

        # Define the classifier
        self.classifier = torch.nn.Linear(64, 5)

    def forward(self, input_ids, attention_mask, aus):

        # Forward pass of the toplkenized text
        output_1 = self.bert_layer(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = output_1[0]
        pooler1 = hidden_state[:, 0]

        pooler1 = self.bert_dense(pooler1)
        pooler1 = torch.nn.ReLU()(pooler1)
        pooler1 = self.dropout(pooler1)

        # Forward pass of the AUs
        aus = torch.reshape(aus, [-1, 408])
        output_2 = self.au_embedder1(aus)
        pooler2 = torch.nn.ReLU()(output_2)
        pooler2 = self.dropout(pooler2)

        pooler2 = self.au_embedder2(pooler2)
        pooler2 = torch.nn.ReLU()(pooler2)
        pooler2 = self.dropout(pooler2)

        # Fusion of the modalities
        fusion_inputs = [pooler1, pooler2]
        fusion = self.fusion_block(fusion_inputs)
        fusion = torch.nn.ReLU()(fusion)

        # Classification of the fusion
        output = self.classifier(fusion)

        return output


class BertGRUBilinearFusion(torch.nn.Module):
    """
    This model is a bi-modal classifier using a bilinear fusion layer for the two modalities.
    Text is embedded through pre-trained BERT, Action Units are embedded through two dense layers.
    """

    def __init__(self):
        super(BertGRUBilinearFusion, self).__init__()

        # Define the Bert pipeline
        self.bert_layer = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.bert_dense = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(0.25)

        # Define the AUs pipeline
        self.au_embedder1 = torch.nn.GRU(
            input_size=17, hidden_size=128, bidirectional=True, batch_first=True
        )  # 408 * 128 -> 128
        self.au_embedder2 = torch.nn.Linear(256, 128)

        # Define the fusion layer
        self.fusion_block = fusions.Block([768, 128], 64)

        # Define the classifier
        self.classifier = torch.nn.Linear(64, 5)

    def forward(self, input_ids, attention_mask, aus):

        aus = torch.reshape(aus, (-1, 24, 17))

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
        fusion_inputs = [pooler1, pooler2]
        fusion = self.fusion_block(fusion_inputs)

        # Classification of the fusion
        output = self.classifier(fusion)

        return output


class BertLSTMBilinearFusion(torch.nn.Module):
    """
    This model is a bi-modal classifier using a bilinear fusion layer for the two modalities.
    Text is embedded through pre-trained BERT, Action Units are embedded through two dense layers.
    """

    def __init__(self):
        super(BertLSTMBilinearFusion, self).__init__()

        # Define the Bert pipeline
        self.bert_layer = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.bert_dense = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(0.25)

        # Define the AUs pipeline
        self.au_embedder1 = torch.nn.LSTM(
            input_size=17,
            num_layers=2,
            hidden_size=128,
            bidirectional=True,
            batch_first=True,
        )  # 408 * 128 -> 128
        self.au_embedder2 = torch.nn.Linear(256, 128)

        # Define the fusion layer
        self.fusion_block = fusions.Block([768, 128], 64)

        # Define the classifier
        self.classifier = torch.nn.Linear(64, 5)

    def forward(self, input_ids, attention_mask, aus):

        aus = torch.reshape(aus, (-1, 24, 17))

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
        fusion_inputs = [pooler1, pooler2]
        fusion = self.fusion_block(fusion_inputs)
        fusion = torch.nn.ReLU()(fusion)

        # Classification of the fusion
        output = self.classifier(fusion)

        return output


class BertGRUConcatMono(torch.nn.Module):
    """
    This model is a bi-modal classifier using a bilinear fusion layer for the two modalities.
    Text is embedded through pre-trained BERT, Action Units are embedded through two dense layers.
    """

    def __init__(self):
        super(BertGRUConcatMono, self).__init__()

        # Define the Bert pipeline
        self.bert_layer = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.bert_dense = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(0.25)

        # Define the AUs pipeline
        self.au_embedder1 = torch.nn.GRU(
            input_size=17, hidden_size=128, batch_first=True
        )  # 408 * 128 -> 128
        self.au_embedder2 = torch.nn.Linear(128, 128)

        # Define the classifier
        self.classifier1 = torch.nn.Linear(896, 512)
        self.classifier2 = torch.nn.Linear(512, 5)

    def forward(self, input_ids, attention_mask, aus):

        aus = torch.reshape(aus, (-1, 24, 17))

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


class BertLSTMConcatMono(torch.nn.Module):

    """
    This model is a bi-modal classifier using a bilinear fusion layer for the two modalities.
    Text is embedded through pre-trained BERT, Action Units are embedded through two dense layers.
    """

    def __init__(self):
        super(BertLSTMConcatMono, self).__init__()

        # Define the Bert pipeline
        self.bert_layer = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.bert_dense = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(0.25)

        # Define the AUs pipeline
        self.au_embedder1 = torch.nn.LSTM(
            input_size=17, hidden_size=128, batch_first=True
        )  # 408 * 128 -> 128
        self.au_embedder2 = torch.nn.Linear(128, 128)

        # Define the classifier
        self.classifier1 = torch.nn.Linear(896, 512)
        self.classifier2 = torch.nn.Linear(512, 5)

    def forward(self, input_ids, attention_mask, aus):

        aus = torch.reshape(aus, (-1, 24, 17))

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


class BertGRUBilinearFusionMono(torch.nn.Module):
    """
    This model is a bi-modal classifier using a bilinear fusion layer for the two modalities.
    Text is embedded through pre-trained BERT, Action Units are embedded through two dense layers.
    """

    def __init__(self):
        super(BertGRUBilinearFusionMono, self).__init__()

        # Define the Bert pipeline
        self.bert_layer = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.bert_dense = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(0.25)

        # Define the AUs pipeline
        self.au_embedder1 = torch.nn.GRU(
            input_size=17, hidden_size=128, batch_first=True
        )  # 408 * 128 -> 128
        self.au_embedder2 = torch.nn.Linear(128, 128)

        # Define the fusion layer
        self.fusion_block = fusions.Block([768, 128], 64)

        # Define the classifier
        self.classifier = torch.nn.Linear(64, 5)

    def forward(self, input_ids, attention_mask, aus):

        aus = torch.reshape(aus, (-1, 24, 17))

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
        fusion_inputs = [pooler1, pooler2]
        fusion = self.fusion_block(fusion_inputs)

        # Classification of the fusion
        output = self.classifier(fusion)

        return output


class BertLSTMBilinearFusionMono(torch.nn.Module):
    """
    This model is a bi-modal classifier using a bilinear fusion layer for the two modalities.
    Text is embedded through pre-trained BERT, Action Units are embedded through two dense layers.
    """

    def __init__(self):
        super(BertLSTMBilinearFusionMono, self).__init__()

        # Define the Bert pipeline
        self.bert_layer = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.bert_dense = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(0.25)

        # Define the AUs pipeline
        self.au_embedder1 = torch.nn.LSTM(
            input_size=17, num_layers=2, hidden_size=128, batch_first=True
        )  # 408 * 128 -> 128
        self.au_embedder2 = torch.nn.Linear(128, 128)

        # Define the fusion layer
        self.fusion_block = fusions.Block([768, 128], 64)

        # Define the classifier
        self.classifier = torch.nn.Linear(64, 5)

    def forward(self, input_ids, attention_mask, aus):

        aus = torch.reshape(aus, (-1, 24, 17))

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
        fusion_inputs = [pooler1, pooler2]
        fusion = self.fusion_block(fusion_inputs)
        fusion = torch.nn.ReLU()(fusion)

        # Classification of the fusion
        output = self.classifier(fusion)

        return output


# -----------------------------------------------------------------------


class BertDenseConcat(torch.nn.Module):
    """
    This model is a bi-modal classifier using a bilinear fusion layer for the two modalities.
    Text is embedded through pre-trained BERT, Action Units are embedded through two dense layers.
    """

    def __init__(self):
        super(BertDenseConcat, self).__init__()

        # Define the Bert pipeline
        self.bert_layer = DistilBertModel.from_pretrained("bert-base-uncased")
        self.bert_dense = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(0.25)
        self.class_num = 5

        # Define the AUs pipeline
        self.au_embedder1 = torch.nn.Linear(17 * 12 * 2, 128)  # 408 * 128 -> 128
        self.au_embedder2 = torch.nn.Linear(128, 128)

        # Define the classifier
        self.classifier1 = torch.nn.Linear(896, 512)
        self.classifier2 = torch.nn.Linear(512, 5)

    def forward(self, input_ids, attention_mask, aus):
        aus = torch.reshape(aus, [-1, 408])
        # Forward pass of the toplkenized text

        input_ids = torch.reshape(input_ids, (-1, 128))
        attention_mask = torch.reshape(attention_mask, (-1, 128))
        hidden_state1 = self.bert_layer(
            input_ids=input_ids, attention_mask=attention_mask
        )[0]
        pooler1 = hidden_state1[:, 0]

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

    def __init__(
        self, dropout1: float = 0.110051, gru_layers: int = 1,
    ):
        super(BertGRUConcat, self).__init__()

        # Define the Bert pipeline
        self.bert_layer = DistilBertModel.from_pretrained("bert-base-uncased")
        self.bert_dense = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(dropout1)
        self.class_num = 5
        # Define the AUs pipeline
        self.au_embedder1 = torch.nn.GRU(
            input_size=17,
            hidden_size=128,
            num_layers=gru_layers,
            bidirectional=True,
            batch_first=True,
        )  # 408 * 128 -> 128
        self.au_embedder2 = torch.nn.Linear(256, 128)

        # Define the classifier
        self.classifier1 = torch.nn.Linear(896, 512)
        self.classifier2 = torch.nn.Linear(512, 5)

    def forward(self, input_ids, attention_mask, aus):

        aus = torch.reshape(aus, (-1, 24, 17))
        input_ids = torch.reshape(input_ids, (-1, 128))
        attention_mask = torch.reshape(attention_mask, (-1, 128))

        # Forward pass of the toplkenized text
        hidden_state1 = self.bert_layer(
            input_ids=input_ids, attention_mask=attention_mask
        )[0]
        pooler1 = hidden_state1[:, 0]

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
        self.bert_layer = DistilBertModel.from_pretrained("bert-base-uncased")
        self.bert_dense = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(0.25)
        self.class_num = 5
        # Define the AUs pipeline
        self.au_embedder1 = torch.nn.LSTM(
            input_size=17, hidden_size=128, bidirectional=True, batch_first=True
        )  # 408 * 128 -> 128
        self.au_embedder2 = torch.nn.Linear(256, 128)

        # Define the classifier
        self.classifier1 = torch.nn.Linear(896, 512)
        self.classifier2 = torch.nn.Linear(512, 5)

    def forward(self, input_ids, attention_mask, aus):

        aus = torch.reshape(aus, (-1, 24, 17))
        input_ids = torch.reshape(input_ids, (-1, 128))
        attention_mask = torch.reshape(attention_mask, (-1, 128))

        # Forward pass of the toplkenized text
        hidden_state1 = self.bert_layer(
            input_ids=input_ids, attention_mask=attention_mask
        )[0]
        pooler1 = hidden_state1[:, 0]

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


class GRUClassif3(torch.nn.Module):
    """
    This model is a bi-modal classifier using a bilinear fusion layer for the two modalities.
    Text is embedded through pre-trained BERT, Action Units are embedded through two dense layers.
    """

    def __init__(self):
        super(GRUClassif3, self).__init__()

        # Define the Bert pipeline
        self.bert_layer = DistilBertModel.from_pretrained("bert-base-uncased")
        self.bert_dense = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(0.25)
        self.class_num = 5

        # Define the AUs pipeline
        self.au_embedder1 = torch.nn.GRU(
            input_size=17, hidden_size=128, bidirectional=True, batch_first=True
        )  # 408 * 128 -> 128
        self.au_embedder2 = torch.nn.Linear(256, 768)

        # Define the classifier
        self.classifier1 = torch.nn.Linear(768, 512)
        self.classifier2 = torch.nn.Linear(512, 5)

    def forward(self, input_ids, attention_mask, aus):

        aus = torch.reshape(aus, (-1, 24, 17))
        input_ids = torch.reshape(input_ids, (-1, 128))
        attention_mask = torch.reshape(attention_mask, (-1, 128))

        # Forward pass of the tokenized text
        pooler_txt = self.bert_layer(
            input_ids=input_ids, attention_mask=attention_mask
        ).pooler_output
        pooler_txt = self.bert_dense(pooler_txt)
        pooler_txt = torch.nn.ReLU()(pooler_txt)
        pooler_txt = self.dropout(pooler_txt)

        # Forward pass of the AUs
        pooler_aus, _ = self.au_embedder1(aus)
        pooler_aus = pooler_aus[:, -1, :]
        pooler_aus = torch.nn.ReLU()(pooler_aus)
        pooler_aus = self.dropout(pooler_aus)

        pooler_aus = self.au_embedder2(pooler_aus)
        pooler_aus = torch.nn.ReLU()(pooler_aus)
        pooler_aus = self.dropout(pooler_aus)

        # Fusion of the modalities
        pooler_mul = torch.mul(pooler_txt, pooler_aus)

        # Classification of the fusion
        output_t, output_au, output_mul = (
            self.classifier1(pooler_txt),
            self.classifier1(pooler_aus),
            self.classifier1(pooler_mul),
        )
        output_t, output_au, output_mul = (
            torch.nn.ReLU()(output_t),
            torch.nn.ReLU()(output_au),
            torch.nn.ReLU()(output_mul),
        )
        output_t, output_au, output_mul = (
            self.dropout(output_t),
            self.dropout(output_au),
            self.dropout(output_mul),
        )

        output_t, output_au, output_mul = (
            self.classifier2(output_t),
            self.classifier2(output_au),
            self.classifier2(output_mul),
        )

        return output_t, output_au, output_mul


# ----------- UNIMODAL CLASSIFIERS


class AudioGRU(torch.nn.Module):
    """
    24 OPENSMILE FEATURES
    This model is a bi-modal classifier using a bilinear fusion layer for the two modalities.
    Text is embedded through pre-trained BERT, Action Units are embedded through two dense layers.
    """

    def __init__(
        self, dropout1: float = 0.110051, gru_layers: int = 2,
    ):
        super(AudioGRU, self).__init__()

        self.dropout = torch.nn.Dropout(dropout1)
        self.class_num = 6
        # Define the AUs pipeline
        self.gru = torch.nn.GRU(
            input_size = 23,
            hidden_size = 128,
            num_layers = gru_layers,
            bidirectional = True,
            batch_first = True,
        ) 
        self.ReLU = torch.nn.LeakyReLU(.1)
        self.sigmoid = torch.nn.Sigmoid()


        # Define the dense layer and the classifier
        self.fc1 = torch.nn.Linear(256, 128)
        self.classifier = torch.nn.Linear(128, 5)

    def forward(self, audio_features):

        output, hidden = self.gru(audio_features)

        output = self.ReLU(output)

        output = self.ReLU(self.fc1(output))


        # Classification of the fusion
        result = self.sigmoid(self.classifier(output))

        return result


class VideoGRU(torch.nn.Module):
    """
    24 OPENSMILE FEATURES
    This model is a bi-modal classifier using a bilinear fusion layer for the two modalities.
    Text is embedded through pre-trained BERT, Action Units are embedded through two dense layers.
    """

    def __init__(self, input_dim=17, hidden_dim=8, hidden_dim2=20, layer_dim=3, output_dim=5, dropout_prob=.1):
        super(VideoGRU, self).__init__()

        self.dropout = torch.nn.Dropout(dropout_prob)
        self.class_num = 6
        # Define the AUs pipeline

        # Look at :
        # 1. 
        self.gru = torch.nn.GRU(
            input_size = input_dim,
            hidden_size = hidden_dim,
            num_layers = layer_dim,
            bidirectional = True,
            batch_first = True,
        ) 
        self.ReLU = torch.nn.LeakyReLU(.1)
        self.sigmoid = torch.nn.Sigmoid()


        # Define the dense layer and the classifier
        self.fc1 = torch.nn.Linear(hidden_dim * 2, hidden_dim2)

        # Do not make a class for the "None" class
        self.classifier = torch.nn.Linear(hidden_dim2, output_dim) # 6

    def forward(self, x_packed):
        x, hidden = self.gru(x_packed)

        x,l = pad_packed_sequence(x, batch_first=True)

        out = torch.stack([x[i][l[i]-1] for i in range(x.shape[0])])

        out = self.fc1(out)
        out = self.ReLU(out)

        out = self.classifier(out)

        return out


class GRUMultiModal(torch.nn.Module):
    def __init__(self, embeddings_dim = 768, audio_input_dim = 17, audio_hidden_dim=32, audio_layer_dim=2, video_input_dim=17, video_hidden_dim=32, video_layer_dim=2, output_dim=6, dropout_prob=.1, activation = "sigmoid"):
        super(GRUMultiModal, self).__init__()
        self.dropout = torch.nn.Dropout(dropout_prob)
        self.class_num = 6
        # Define the AUs pipeline

        # Define the Audio part of the multi-modal model
        self.audio_gru = torch.nn.GRU(
            input_size = audio_input_dim,
            hidden_size = audio_hidden_dim,
            num_layers = audio_layer_dim,
            bidirectional = True,
            batch_first = True,
        ) 
        self.audio_fc1 = torch.nn.Linear(audio_hidden_dim * 2, audio_hidden_dim)


        # Define the Video part of the multi-modal model
        self.video_gru = torch.nn.GRU(
            input_size = video_input_dim,
            hidden_size = video_hidden_dim,
            num_layers = video_layer_dim,
            bidirectional = True,
            batch_first = True,
        ) 
        self.video_fc1 = torch.nn.Linear(video_hidden_dim * 2, video_hidden_dim)

        # Define the text embeddings part
        self.embeds_fc1 = torch.nn.Linear(embeddings_dim, embeddings_dim)
        self.embeds_fc2 = torch.nn.Linear(embeddings_dim, embeddings_dim // 2)


        self.ReLU = torch.nn.LeakyReLU(.1)
        if activation == "sigmoid":
            self.activation = torch.nn.Sigmoid()
        elif activation == "softmax":
            self.activation = torch.nn.Softmax()
        # Do not make a class for the "None" class
        self.classifier = torch.nn.Linear(embeddings_dim // 2 + audio_hidden_dim + video_hidden_dim, output_dim) # 6

    def forward(self, embeddings, audio_packed, video_packed):
        # Forward pass of the features
        audio_x, audio_hidden = self.audio_gru(audio_packed)
        video_x, video_hidden = self.video_gru(video_packed)

        audio_x, audio_l = pad_packed_sequence(audio_x, batch_first = True)
        video_x, video_l = pad_packed_sequence(video_x, batch_first = True)

        audio_out = torch.stack([audio_x[i][audio_l[i] - 1] for i in range(audio_x.shape[0])])
        audio_out = self.audio_fc1(self.ReLU(audio_out))

        video_out = torch.stack([video_x[i][video_l[i] - 1] for i in range(video_x.shape[0])])
        video_out = self.video_fc1(self.ReLU(video_out))

        # Forward pass of the embeddings
        embeds_x = self.embeds_fc1(embeddings)
        embeds_x = self.ReLU(embeds_x)

        embeds_x = self.embeds_fc2(embeds_x)
        embeds_out = self.ReLU(embeds_x)

        # Concatenate all the inputs

        cat = torch.cat([embeds_out, video_out, audio_out], dim = 1)

        cat = self.ReLU(cat)
        out = self.classifier(cat)
        out = self.activation(out)

        return out


class GRUBiModal(torch.nn.Module):
    def __init__(self, embeddings_dim = 768, input_dim=17, hidden_dim=32, layer_dim=2, output_dim=6, dropout_prob=.1, activation = "sigmoid"):
        super(GRUBiModal, self).__init__()
        self.dropout = torch.nn.Dropout(dropout_prob)
        self.class_num = 6
        # Define the AUs pipeline

        # Define the Audio part of the multi-modal model
        self.modality_gru = torch.nn.GRU(
            input_size = input_dim,
            hidden_size = hidden_dim,
            num_layers = layer_dim,
            bidirectional = True,
            batch_first = True,
        ) 
        self.modality_fc1 = torch.nn.Linear(hidden_dim * 2, hidden_dim)

        # Define the text embeddings part
        self.embeds_fc1 = torch.nn.Linear(embeddings_dim, embeddings_dim)
        self.embeds_fc2 = torch.nn.Linear(embeddings_dim, embeddings_dim // 2)


        self.ReLU = torch.nn.LeakyReLU(.1)
        if activation == "sigmoid":
            self.activation = torch.nn.Sigmoid()
        elif activation == "softmax":
            self.activation = torch.nn.Softmax()
        # Do not make a class for the "None" class
        self.classifier = torch.nn.Linear(embeddings_dim // 2 + hidden_dim, output_dim) # 6

    def forward(self, embeddings, x_packed):
        # Forward pass of the features
        x, hidden = self.modality_gru(x_packed)

        x, l = pad_packed_sequence(x, batch_first = True)

        out = torch.stack([x[i][l[i] - 1] for i in range(x.shape[0])])
        out = self.modality_fc1(self.ReLU(out))

        # Forward pass of the embeddings
        embeds_x = self.embeds_fc1(embeddings)
        embeds_x = self.ReLU(embeds_x)

        embeds_x = self.embeds_fc2(embeds_x)
        embeds_out = self.ReLU(embeds_x)

        # Concatenate all the inputs

        cat = torch.cat([embeds_out, out], dim = 1)

        cat = self.ReLU(cat)
        out = self.classifier(cat)
        out = self.activation(out)

        return out

class BertClassif(torch.nn.Module):
    """
    This model is a bi-modal classifier using a bilinear fusion layer for the two modalities.
    Text is embedded through pre-trained BERT, Action Units are embedded through two dense layers.
    """

    def __init__(self, embeddings_dim, hidden_dim, output_dim, activation = "sigmoid"):
        super(BertClassif, self).__init__()

        # Define the Bert pipeline
        self.embeds_fc1 = torch.nn.Linear(embeddings_dim, hidden_dim)
        self.embeds_fc2 = torch.nn.Linear(hidden_dim, hidden_dim // 2)
        self.classifier = torch.nn.Linear(hidden_dim // 2, output_dim) # 6

        self.dropout = torch.nn.Dropout(0.25)
        self.class_num = 6
        self.ReLU = torch.nn.LeakyReLU(.1)
        if activation == "sigmoid":
            self.activation = torch.nn.Sigmoid()
        elif activation == "softmax":
            self.activation = torch.nn.Softmax()
        # Define the classifier

    def forward(self, embeddings):

        embeds_x = self.embeds_fc1(embeddings)
        embeds_x = self.ReLU(embeds_x)

        embeds_x = self.embeds_fc2(embeds_x)
        embeds_out = self.ReLU(embeds_x)

        out = self.classifier(embeds_out)
        out = self.activation(out)
        return out