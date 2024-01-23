import os
import torch
import pandas as pd
from utils import get_df, get_model, get_dataloader, train_model_a_epoch, evaluation, to_metric, save_model, AnomalyScorer, PromptBERT
from sklearn.metrics import accuracy_score
import shutil
import time
import sys

# get year date time to string
now_date = time.strftime("%Y%m%d%H%M%S", time.localtime())

# get args
train_classifier, optimizer = sys.argv[1], sys.argv[2]

train_data_path, test_data_path = sys.argv[1], sys.argv[2]

bert_name = "bert-base-multilingual-cased"
save_base_path = f"saved_{bert_name}_{now_date}_{train_classifier}_{optimizer}"
bank_base_path = f"faiss_bank_{now_date}_{train_classifier}_{optimizer}"
print(save_base_path, bank_base_path)
# delete folder
shutil.rmtree(save_base_path, ignore_errors=True)
shutil.rmtree(bank_base_path, ignore_errors=True)

tokenizer, _, cls_model = get_model(bert_name=bert_name)
cls_model = PromptBERT(
    cls_model, train_classifier=train_classifier, optimizer=optimizer)

device = torch.device("cuda")

train, test = get_df(train_data_path=train_data_path,
                     test_data_path=test_data_path,
                     reduced=False, deduplicated=True)

test.rename(columns={'generated': 'label'}, inplace=True)

train_loader, test_loader = get_dataloader(train, test, tokenizer)

anomaly_scorer = AnomalyScorer(tokenizer, train, test)


def test_f(train_loader, test_loader):

    bank_path = os.path.join(bank_base_path, f"epoch_{epoch}")
    anomaly_scorer.save_anomaly_bank(
        cls_model, save_path=bank_path)

    accs = []
    train_metric = pd.DataFrame()
    preds_argmax, preds_for_1, labels_t = evaluation(
        cls_model=cls_model, test_loader=train_loader)
    train_anomaly_scores = anomaly_scorer.get_anomaly_score(
        cls_model, load_path=bank_path, target='train')
    for ratio in [0.01, 0.05, 0.1, 0.5]:
        train_metric = to_metric(
            train_metric, preds_argmax, preds_for_1, labels_t, train_anomaly_scores, ratio, epoch)
    accs.append(accuracy_score(labels_t, preds_argmax))

    test_metric = pd.DataFrame()
    preds_argmax, preds_for_1, labels_t = evaluation(
        cls_model=cls_model, test_loader=test_loader)
    test_anomaly_scores = anomaly_scorer.get_anomaly_score(
        cls_model, load_path=bank_path, target='test')
    for ratio in [0.01, 0.05, 0.1, 0.5]:
        test_metric = to_metric(
            test_metric, preds_argmax, preds_for_1, labels_t, test_anomaly_scores, ratio, epoch)

    accs.append(accuracy_score(labels_t, preds_argmax))

    return train_metric, test_metric, pd.DataFrame(accs).T.rename({0: 'train', 1: 'test'}, axis=1)


metric_train = pd.DataFrame()
metric_test = pd.DataFrame()
acc_df = pd.DataFrame()

for epoch in range(1, 100):
    train_model_a_epoch(cls_model=cls_model, train_loader=train_loader)
    # train_model_a_epoch(cls_model=cls_model, train_loader=train_loader, partial_train_model=cls_model.classifier)
    save_path = os.path.join(save_base_path, f"{epoch}")
    save_model(save_path, cls_model, tokenizer)
    # _, _, cls_model = get_model(load_path=save_path)
    cls_model = PromptBERT(get_model(load_path=save_path)[
                           2], load_path=save_path, train_classifier=train_classifier, optimizer=optimizer)

    metric_train_tmp, metric_test_tmp, accuracy = test_f(
        train_loader, test_loader)
    acc_df = pd.concat([acc_df, accuracy])
    metric_train, metric_test = pd.concat([metric_train, metric_train_tmp], axis=0).sort_index(
        axis=0),  pd.concat([metric_test, metric_test_tmp], axis=0).sort_index(axis=0)

    metric_train.to_csv(os.path.join(save_base_path, f"metric_train.csv"))
    metric_test.to_csv(os.path.join(save_base_path, f"metric_test.csv"))
    acc_df.to_csv(
        os.path.join(save_base_path, f"accuracies.csv"))
