import os

os.environ["TRANSFORMERS_OFFLINE"] = "yes"
import torch
import numpy as np
from torch import cuda

from tqdm import tqdm

from torch.utils.data import DataLoader
import torch.nn.functional as F
from transformers import DistilBertTokenizer, RobertaTokenizer, BertTokenizer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    hamming_loss,
)

from train_text import TextDataset, read_dataset, class_split
from models import BertClassif

# from models_roberta_novsn import BertGRUConcat, BertClassif, BertDenseConcat, BertLSTMConcat
# from models_roberta import BertGRUConcat, BertClassif, BertDenseConcat, BertLSTMConcat
# from models import BertGRUConcat, BertClassif, BertDenseConcat, BertLSTMConcat
# from models import BertDenseBilinearFusion, BertDenseConcat, BertGRUBilinearFusion, BertGRUConcat, BertLSTMBilinearFusion, BertLSTMConcat, BertClassif, BertGRUConcatMono, BertGRUBilinearFusionMono, BertLSTMBilinearFusionMono, BertLSTMConcatMono, GRUClassif3


device = torch.device("mps")

TEXT_PATH = "merged_df_2016.csv"

SEED = 12061999
MAX_LEN = 128


def hamming_score(y_true, y_pred, normalize=True, sample_weight=None):
    acc_list = []
    for i in range(len(y_true)):
        set_true = set(np.where(y_true[i])[0])
        set_pred = set(np.where(y_pred[i])[0])

        tmp_a = None
        if len(set_true) == 0 and len(set_pred) == 0:
            tmp_a = 1
        else:
            tmp_a = len(set_true.intersection(set_pred)) / float(
                len(set_true.union(set_pred))
            )
        acc_list.append(tmp_a)

    return np.mean(acc_list)


def scoring(y, y_hat, label_columns):
    if isinstance(y, torch.Tensor):
        cpu_y = y.cpu().detach().numpy()
    else:
        cpu_y = np.array(y)
    if isinstance(y_hat, torch.Tensor):
        cpu_y_hat = y_hat.cpu().detach().numpy()
    else:
        cpu_y_hat = np.array(y_hat)
    cpu_y = cpu_y.reshape(cpu_y_hat.shape)
    results = {}

    for i, c in enumerate(label_columns):
        label_y = cpu_y[i, :]
        label_pred = cpu_y_hat[i, :]
        acc = accuracy_score(label_y, label_pred)
        bal_acc = balanced_accuracy_score(label_y, label_pred)
        f_score = f1_score(label_y, label_pred, average="weighted")
        auc = roc_auc_score(label_y, label_pred)
        results[c] = {
            "acc": acc,
            "bal_acc": bal_acc,
            "f_score": f_score,
            "roc_auc_score": auc,
        }
    return results


if True:
    if __name__ == "__main__":
        is_aus = False
        tokenizer = DistilBertTokenizer.from_pretrained(
            "distilbert-base-uncased", truncation=True, do_lower_case=True
        )

        # We will evaluate 7 metrics, namely

        """
        Precision
        Recall
        F1
        Sensitivity
        Specificity
        ROC
        AUC
        """

        # Setting up the datasets and predictions

        text_df = read_dataset(TEXT_PATH)
        _, test_df, _ = class_split(text_df)
        model_l = [
            # BertClassif(),
            # BertDenseConcat(),
            # BertLSTMConcat(),
            BertClassif()
        ]

        # print(len(test_df[test_df["VSN"] == "x"]))
        print(len(test_df[test_df["PR"] == "x"]))
        print(len(test_df[test_df["SD"] == "x"]))
        print(len(test_df[test_df["QE"] == "x"]))
        print(len(test_df[test_df["None"] == "x"]))

        model_names_l = [type(m_).__name__ for m_ in model_l]

        test_data_txt = TextDataset(test_df, tokenizer, MAX_LEN)

        dataset_l = [
            # test_data_txt,
            # test_data_aus,
            # test_data_rnn,
            test_data_txt
        ]

        prediction_dict = {}

        models_dict = dict(
            zip(
                model_names_l,
                [
                    {"model": model_, "dataset": dataset_}
                    for model_, dataset_ in zip(model_l, dataset_l)
                ],
            )
        )

        for i, model_ in enumerate(model_l):
            print("========================")
            print("EVALUATING MODEL {}".format(type(model_).__name__))

            test_data = models_dict[type(model_).__name__]["dataset"]
            model_.load_state_dict(
                torch.load(
                    f"models/DistilBert_nocontext_6_23_02_08_15/{type(model_).__name__}"
                )
            )
            model_.to(device)
            model_.eval()
            fin_targets = []
            fin_outputs = []

            data_loader = DataLoader(
                test_data, batch_size=4, shuffle=True, num_workers=8
            )

            with torch.no_grad():
                for j, data in tqdm(enumerate(data_loader)):
                    input_ids, attention_mask, labels = (
                        data["ids"].to(device, dtype=torch.long),
                        data["mask"].to(device, dtype=torch.long),
                        data["target"].to(device, dtype=torch.float),
                    )
                    outputs = model_(input_ids, attention_mask)
                    fin_targets.extend(labels.cpu().detach().numpy().tolist())
                    fin_outputs.extend((np.array(outputs.cpu().detach().numpy()) >= (1 / 3)).tolist())
                    if j == 5:
                        print("outputs: ")
                        print(outputs)
                        print("fin_outputs")
                        print(fin_outputs)
                        print("fin_targets")
                        print(fin_targets)
                prediction_dict[model_] = fin_outputs
                val_hamming_loss = hamming_loss(fin_targets, fin_outputs)
                val_hamming_score = hamming_score(fin_targets, fin_outputs)

            cpu_y = np.where(
                fin_targets == False,
                0,
                np.where(fin_targets == True, 1, np.array(fin_targets)),
            )
            cpu_y_hat = np.where(
                fin_targets == False,
                0,
                np.where(fin_targets == True, 1, np.array(fin_outputs)),
            )
            cpu_y = cpu_y.reshape(cpu_y_hat.shape)
            results = {}
            f_score_tot_none = f1_score(cpu_y, cpu_y_hat, average=None)
            f_score_tot_micro = f1_score(cpu_y, cpu_y_hat, average="micro")
            f_score_tot_macro = f1_score(cpu_y, cpu_y_hat, average="macro")
            f_score_tot_weighted = f1_score(cpu_y, cpu_y_hat, average="weighted")

            class_list = ["PR", "SD", "QE", "VSN", "HD", "None"]

            for j, c in enumerate(class_list):
                print(
                    f"Total label for {c} : {np.sum(cpu_y[:, j])}, number of correct pred {np.dot(cpu_y[:, j], cpu_y_hat[:, j])}"
                )
                label_y = cpu_y[:, j]
                label_pred = cpu_y_hat[:, j]
                rec = recall_score(label_y, label_pred)
                prec = precision_score(label_y, label_pred)
                acc = accuracy_score(label_y, label_pred)

                try:
                    auc = roc_auc_score(label_y, label_pred)
                except ValueError:
                    auc = 1

                results[c] = {
                    "Recall": f"{rec:.4f}",
                    "Precision": f"{prec:.4f}",
                    "Accuracy": f"{acc:.2f}",
                }

            print(f"Hamming Score for {type(model_).__name__} = ", val_hamming_score)
            print(f"Hamming Loss for {type(model_).__name__} = ", val_hamming_loss)
            with open("models/results.txt", "a") as f:
                f.write("=========================\n")
                f.write(f"EVALUATING {type(model_).__name__}\n")
                f.write("=========================\n")
                for k_ in results.keys():
                    f.write(str(k_) + str(results[k_]) + "\n")
                f.write("Total f1 score : {}\n".format(f_score_tot_none))
                f.write("Total f1 score micro : {}\n".format(f_score_tot_micro))
                f.write("Total f1 score macro : {}\n".format(f_score_tot_macro))
                f.write("Total f1 score weighted : {}\n".format(f_score_tot_weighted))

else:
    tokenizer = DistilBertTokenizer.from_pretrained("bert-base-uncased")

    model = BertGRUConcat(dropout1=0.28947651671912833)

    model.load_state_dict(
        torch.load(
            f"models/DistilBert_nocontext_novsn_22_07_07_10/{type(model).__name__}"
        )
    )

    s_l = [
        "Ewen would be an awesome perfect mathematics teacher",
        "I hate mondays",
        "I love mondays",
        "I do not have an opinion on courses on a monday",
        "Great job !",
        "What is your preferred teacher?",
        "You were impressive!",
        "This is a great idea!",
    ]

    for sentence in s_l:
        inputs = tokenizer.encode_plus(
            sentence,
            None,
            add_special_tokens=True,
            max_length=128,
            padding="max_length",
            return_token_type_ids=True,
            return_tensors="pt",
        )

        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        result = torch.sigmoid(model(ids, mask)).detach().numpy()
        print(f"{sentence} classification : {result}")
