from transformers import AdamW
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import os
import re
from collections import defaultdict
from itertools import zip_longest
from tqdm import tqdm
from faiss_u import FaissNN, NearestNeighbourScorer
from sampler_u import ApproximateGreedyCoresetSampler
from detection_u import Detection
from transformers import BertTokenizer, BertModel
from transformers import BertModel, BertConfig
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, roc_auc_score
from transformers.modeling_outputs import (
    SequenceClassifierOutput,
)


def get_df(train_data_path, test_data_path, reduced, deduplicated):
    """
    Parameters
    ----------
    reduced: bool
        True: 100 배 줄임
        False: 원래 데이터 사용
    dedupliaced: bool
        True: test 에서 train 과 중복되는 항목 제거
        False: 원래 데이터 사용
    """
    # train_df, test_df 생성
    # train_df 에는 memory bank 생성을 위한 데이터 포함
    # test_df 에는 prediction score 생성을 위한 데이터 포함

    # LABEL 1 for AI generated, 0 for human
    train = pd.read_csv(train_data_path, sep=',').sample(
        frac=1, random_state=42)
    train = train.drop_duplicates(subset=['text'])
    train.reset_index(drop=True, inplace=True)

    # train, test = Dataset_U.split_df_train_val(train, train_ratio=0.9)
    test = pd.read_csv(test_data_path).drop_duplicates(subset=['text']).reset_index(
        drop=True).sample(frac=1, random_state=42)  # .iloc[:500]  # label 편향  떄문에 섞고 자름
    # [['text', 'generated']].rename({'generated': 'label'}, axis=1)
    test = test.drop_duplicates(subset=['text'])
    test.reset_index(drop=True, inplace=True)

    if deduplicated:
        # test 에서 train 과 중복되는 항목 제거
        merged = pd.merge(train, test, left_on="text",
                          right_on="text", how='outer', indicator=True)  # 왼쪽 df, 오른쪽 df 모두 "text" column 을 기준으로 outer join
        train = merged[merged['_merge'] != 'right_only'][train.keys().tolist()]
        test = merged[merged['_merge'] == 'right_only'][test.keys().tolist()]

    if reduced:
        train_len_ = len(train)//100  # 100 배 줄임
        test_len_ = len(train)//300  # 100 배 줄임
        train = train.sample(frac=1, random_state=42).iloc[:train_len_]
        # times //= 10
        test = test.sample(frac=1, random_state=42).iloc[:test_len_]

    return train.reset_index(drop=True), test.reset_index(drop=True)


def get_model(load_path=None, bert_name="bert-base-multilingual-cased"):

    id2label = {1: "AI", 0: "HUMAN"}
    label2id = {"AI": 1, "HUMAN": 0}

    # training 수행할 model
    if load_path is None:
        load_path = bert_name

    try:
        config = BertConfig.from_pretrained(
            load_path, output_hidden_states=True)
        embed_model = BertModel.from_pretrained(
            load_path, config=config).to("cuda")
    except:
        embed_model = BertModel.from_pretrained(load_path).to("cuda")

    try:
        tokenizer = BertTokenizer.from_pretrained(load_path)
    except:
        tokenizer = None

    cls_model = AutoModelForSequenceClassification.from_pretrained(
        load_path, num_labels=2, id2label=id2label, label2id=label2id, output_hidden_states=True).to("cuda")

    return tokenizer, embed_model, cls_model


def save_model(save_path, tokenizer, model, config=None):
    # model 저장
    tokenizer.save_pretrained(save_directory=save_path)
    model.save_pretrained(save_directory=save_path)
    if config is not None:
        config.save_pretrained(save_directory=save_path)


def get_dataloader(train, test, tokenizer):
    # BERT training and test 를 위한 dataloader 생성
    data = {}
    data["train"] = defaultdict(list)
    data["test"] = defaultdict(list)

    text_col, label_col = "text", "label"
    for df, name in zip([train, test], ["train", "test"]):
        for i in tqdm(range(len(df))):
            data[name]['texts'].append(
                "[CLS] " + re.sub(re.compile("\n+"), ' ', df.iloc[i][text_col].strip()))
            data[name]['labels'].append(int(df.iloc[i][label_col]))

        print('tokenizing...')
        data[name]['texts'] = tokenizer(
            data[name]['texts'], truncation=True, padding=True, return_tensors="pt")

    class TextDataset(torch.utils.data.Dataset):
        def __init__(self, data):
            self.data = data

        def __getitem__(self, idx):
            items = {key: val[idx] for key, val in self.data['texts'].items()}
            items['labels'] = torch.tensor(self.data['labels'][idx])
            return items

        def __len__(self):
            return len(self.data['labels'])

    train_data = TextDataset(data["train"])
    test_data = TextDataset(data["test"])

    train_loader = DataLoader(train_data, batch_size=14,
                              shuffle=True, num_workers=4)
    test_loader = DataLoader(test_data, batch_size=14,
                             shuffle=False, num_workers=0)

    return train_loader, test_loader


def load_anomaly_model(model):
    device = torch.device("cuda")
    coreset_p = 0.1
    layers_to_extract_from = [2, 3, 4, 5, 6, 7, 8, 9]
    n_nearest_neighbours = 1

    # load sampler and algorithm
    sampler = ApproximateGreedyCoresetSampler(coreset_p, device)
    scorer = NearestNeighbourScorer(n_nearest_neighbours=n_nearest_neighbours)
    uniformaly_instance = Detection(
        model, sampler=sampler, scorer=scorer, layers_to_extract_from=layers_to_extract_from)
    return uniformaly_instance


class AnomalyScorer:
    def __init__(self, tokenizer, train_df, test_df):
        dataset = Dataset(tokenizer)
        self.train_encoded_input_list = dataset.build_splited_tokens(
            train_df, target_column_list=["text"])
        self.test_encoded_input_list = dataset.build_splited_tokens(
            test_df, target_column_list=["text"])

    def save_anomaly_bank(self, model, save_path=None):
        os.makedirs(save_path, exist_ok=True)
        model.eval()
        uniformaly_instance = load_anomaly_model(model)
        with torch.no_grad():
            uniformaly_instance.fit(self.train_encoded_input_list)
        uniformaly_instance.anomaly_scorer.save(save_path)

    def get_anomaly_score(self, model, load_path, target):
        model.eval()
        uniformaly_instance = load_anomaly_model(model)
        uniformaly_instance.anomaly_scorer.load(load_path)
        with torch.no_grad():
            # uniformaly_instance.anomaly_scorer.save(faiss_path)
            if target == 'train':
                scores = uniformaly_instance.predict(
                    self.train_encoded_input_list)
            elif target == 'test':
                scores = uniformaly_instance.predict(
                    self.test_encoded_input_list)
            else:
                raise ValueError("target should be 'train' or 'test'")

        return scores


def get_pred_score():
    # pre-trained BERT 및 datalader 를 이용해서 prediction score 생성
    pass


def evaluation(cls_model, test_loader):
    model = cls_model
    model.eval()
    preds = []
    labels_t = []
    loss_t = []
    device = torch.device("cuda")
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(
                input_ids, attention_mask=attention_mask, labels=labels)
            labels_t.append(labels.cpu().detach().numpy())
            loss_t.append(outputs[0].detach().cpu())
            preds.append(torch.nn.functional.softmax(
                outputs[1].cpu().detach(), 1).numpy())

    loss_t = float((sum(loss_t)/len(loss_t)).numpy())
    preds = np.concatenate(preds)
    labels_t = np.concatenate(labels_t)
    preds_argmax = np.argmax(preds, axis=1)
    preds_for_1 = preds[:, 1]

    return preds_argmax, preds_for_1, labels_t


def reproduce_metric(anomaly_scores, ratio, labels_t):
    len_ = int(len(anomaly_scores)*ratio)
    normal_ids, abnomal_ids = anomaly_scores.sort(
    )[1][:len_].numpy(), anomaly_scores.sort()[1][-len_:].numpy()
    index = [
        f'normal_score_max',
        f'normal_num',
        f'normal_label_ratio',
        f'abnormal_score_min',
        f'abnormal_num',
        f'abnormal_label_ratio',
    ]
    index = [f'{val}' for val in index]
    values = [
        [anomaly_scores[normal_ids].max().cpu().numpy()],
        [len(normal_ids)],
        [labels_t[normal_ids].mean()],
        [anomaly_scores[abnomal_ids].min().cpu().numpy()],
        [len(abnomal_ids)],
        [labels_t[abnomal_ids].mean()],
    ]
    return pd.DataFrame(values, columns=[ratio], index=index)


def to_metric(metric, preds_argmax, preds_for_1, labels_t, anomaly_scores, ratio, epoch):
    """
    ratio: ratio of anomaly to normal
        0 < raio <= 0.5"""
    len_ = int(len(anomaly_scores)*ratio)
    normal_ids, abnomal_ids = anomaly_scores.sort(
    )[1][:len_].numpy(), anomaly_scores.sort()[1][-len_:].numpy()

    try:
        index = [f'normal_acc',
                 f'normal_auroc',
                 f'normal_score_max',
                 f'normal_num',
                 f'normal_label_ratio',
                 f'abnormal_acc',
                 f'abnormal_auroc',
                 f'abnormal_score_min',
                 f'abnormal_num',
                 f'abnormal_label_ratio',
                 ]
        epoch = '0' * (4 - len(str(epoch))) + str(epoch)
        index = [f'{val}_{epoch}' for val in index]
        values = [[accuracy_score(labels_t[normal_ids], preds_argmax[normal_ids])],
                  [roc_auc_score(y_true=labels_t[normal_ids],
                                 y_score=preds_for_1[normal_ids]) if np.unique(labels_t[normal_ids]).shape[0] == 2 else np.nan],
                  [anomaly_scores[normal_ids].max().cpu().numpy()],
                  [len(normal_ids)],
                  [labels_t[normal_ids].mean()],
                  [accuracy_score(labels_t[abnomal_ids],
                                  preds_argmax[abnomal_ids])],
                  [roc_auc_score(y_true=labels_t[abnomal_ids],
                                 y_score=preds_for_1[abnomal_ids]) if np.unique(labels_t[abnomal_ids]).shape[0] == 2 else np.nan],
                  [anomaly_scores[abnomal_ids].min().cpu().numpy()],
                  [len(abnomal_ids)],
                  [labels_t[abnomal_ids].mean()],
                  ]
        tmp_df = pd.DataFrame(values, columns=[ratio], index=index)
    except Exception as e:
        print(f"with ratio {ratio}", e)
        tmp_df = pd.DataFrame(np.nan, columns=[ratio], index=index)
    metric = pd.concat([metric, tmp_df], axis=1)
    return metric


class PromptBERT(torch.nn.Module):
    def __init__(self, bert, num_tokens=50, load_path=None, train_classifier=True, optimizer='sgd'):
        super().__init__()
        self.bert = bert
        self.prompt_embeddings = torch.nn.Parameter(
            torch.zeros(1, num_tokens, 768).to("cuda"))
        self.prompt_dropout = torch.nn.Dropout(0.0).to("cuda")
        self.prompt_proj = torch.nn.Linear(768, 768).to("cuda")
        self.num_tokens = num_tokens
        self.train_classifier = train_classifier
        self.optimizer = optimizer

        if load_path is not None:
            self.load_pretrained(load_path)

    def parameters(self):
        if self.train_classifier:
            ret = [self.prompt_embeddings, *self.prompt_proj.parameters(),
                   *self.bert.classifier.parameters()]
        else:
            ret = [self.prompt_embeddings, *self.prompt_proj.parameters()]
        return ret

    def incorporate_prompt(self, x, attention_mask):
        attention_mask = torch.cat(
            [torch.ones(attention_mask.shape[0], self.num_tokens).to(attention_mask.device), attention_mask], dim=1)

        attention_mask = attention_mask[:, :512]  # BERT max length

        # combine prompt embeddings with image-patch embeddings
        B = x.shape[0]
        # after CLS token, all before image patches
        # (batch_size, 1 + n_patches, hidden_dim)
        x = self.bert.bert.embeddings(x)
        x = torch.cat((
            x[:, :1, :],
            self.prompt_dropout(self.prompt_proj(
                self.prompt_embeddings).expand(B, -1, -1)),
            x[:, 1:, :]
        ), dim=1)

        x = x[:, :512, :]  # BERT max length

        # (batch_size, cls_token + n_prompt + n_patches, hidden_dim)

        return x, attention_mask

    def forward(self, input_ids, **kwargs):
        # 1. from /compuworks/anaconda3/envs/py39_ggle/lib/python3.9/site-packages/transformers/models/bert/modeling_bert.py > BertForSequenceClassification > forward()
        # 2. from /compuworks/anaconda3/envs/py39_ggle/lib/python3.9/site-packages/transformers/models/bert/modeling_bert.py > BertModel > forward()

        # self.bert(x, **kwargs)

        # this is the default version:
        embedding_output,  attention_mask = self.incorporate_prompt(
            input_ids, kwargs['attention_mask'])
        attention_mask = self.bert.get_extended_attention_mask(
            attention_mask, attention_mask.shape, attention_mask.device)

        encoder_outputs = self.bert.bert.encoder(
            embedding_output, attention_mask=attention_mask, output_hidden_states=True)
        pooled_outputs = self.bert.bert.pooler(encoder_outputs[0])

        # pooled_output = outputs[1]
        # pooled_output = self.dropout(pooled_output)
        # logits = self.classifier(pooled_output)

        logits = self.bert.classifier(self.bert.dropout(pooled_outputs))

        try:
            loss = torch.nn.CrossEntropyLoss()(logits, kwargs['labels'])
        except:
            loss = None

        ret = SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
        # print(ret)
        return ret

    def train(self):
        self.bert.eval()
        if self.train_classifier:
            self.bert.classifier.train()
        self.prompt_proj.train()
        self.prompt_dropout.train()

    def eval(self):
        self.bert.eval()
        self.prompt_proj.eval()
        self.prompt_dropout.eval()

    def save_pretrained(self, save_directory):
        self.bert.save_pretrained(save_directory)
        torch.save(self.prompt_embeddings, os.path.join(
            save_directory, "prompt_embeddings.pt"))
        torch.save(self.prompt_proj, os.path.join(
            save_directory, "prompt_proj.pt"))
        torch.save(self.num_tokens, os.path.join(
            save_directory, "num_tokens.pt"))

    def load_pretrained(self, load_directory):
        self.prompt_embeddings = torch.load(os.path.join(
            load_directory, "prompt_embeddings.pt"))
        self.prompt_proj = torch.load(
            os.path.join(load_directory, "prompt_proj.pt"))
        self.num_tokens = torch.load(
            os.path.join(load_directory, "num_tokens.pt"))


def train_model_a_epoch(cls_model, train_loader, partial_train_model=None):
    device = torch.device("cuda")
    model = cls_model
    if partial_train_model is not None:
        model.eval()
        partial_train_model.train()
        optim = AdamW(partial_train_model.parameters(), lr=5e-2)
    else:
        model.train()
        if isinstance(model, PromptBERT):
            if model.optimizer == 'adam':
                lr = 3e-1
                optim = torch.optim.Adam(
                    model.parameters(), lr=lr)
            elif model.optimizer == 'sgd':
                lr = 3e-1
                optim = torch.optim.SGD(
                    model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-5)
            else:
                raise ValueError("optimizer should be 'adam' or 'sgd'")
        else:
            lr = 5e-6
            optim = AdamW(model.parameters(), lr=lr)

    print(optim)
    best_loss = 1000
    metric_t = pd.DataFrame()
    for batch in train_loader:
        optim.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(
            input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs[0]
        loss.backward()
        optim.step()
    return

    scores = get_anomaly_score(model, train, test, tokenizer)
    metric, loss = evaluation(
        model, test_loader, scores, epoch)
    metric_t = pd.concat([metric_t, metric], axis=0)
    metric_t.sort_index(axis=0, inplace=True)
    print(f"\n{metric_t}")
    metric_t.to_csv(os.path.join(save_path, f"metric.csv"))
    epoch = '0' * (4 - len(str(epoch))) + str(epoch)
    if metric_t.loc[f'normal_auroc_{epoch}'].max() > 0.97:
        best_loss = loss
        print(
            f"save model, : normal_auroc_{epoch} {metric_t.loc[f'normal_auroc_{epoch}'].max()}")
        model.save_pretrained(
            save_directory=os.path.join(save_path, f"{epoch}"))
        tokenizer.save_pretrained(
            save_directory=os.path.join(save_path, f"{epoch}"))


def df_update(source, target, source_ratio):
    """
    source: pd.DataFrame
    target: pd.DataFrame
    """
    target.loc[source.index, 'generated'] = (
        target.loc[source.index]['generated']*(1-source_ratio) + source['generated']*source_ratio).to_frame()
    return target


def test_roc(source, target):
    backup = target.copy()
    try:
        print(
            f"source roc {roc_auc_score(source['label'], source['generated']), source['generated'].mean()}")
    except:
        pass
    print(
        f"target roc {roc_auc_score(target['label'], target['generated']), target['generated'].mean()}")

    for ratio in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        target = df_update(source, target, ratio)
        print(
            f"ratio {ratio} roc {roc_auc_score(target['label'], target['generated'])}, mean {target['generated'].mean()}")
        target = backup.copy()


class Dataset:
    def __init__(self, tokenizer):
        """
        Parameters
        ----------
        tokenizer: tokenizer
            tokenizer
        """
        self.tokenizer = tokenizer

    def build_splited_tokens(self, df, target_column_list):
        encoded_input_list = []

        for col in target_column_list:
            for i in tqdm(range(len(df))):
                # NOTE 총 train 모드 시, 약 4분, memory 2.2GB 필요.
                # NOTE 총 test 모드 시, train 모드와 차이 없음.
                ok_text = "[CLS] " + re.sub(re.compile("\n+"), ' ',
                                            df.iloc[i][col].strip())
                # encoded_input = self.tokenizer(f"[CLS] {ok_text} [SEP]", return_tensors='pt') # pt = pytorch # NOTE CLS 쓰는 건 나중에 시도
                encoded_input = self.tokenizer(
                    ok_text, truncation=True, padding=True, return_tensors='pt')  # pt = pytorch
                len_tokens = len(encoded_input["input_ids"][0])
                with torch.no_grad():
                    inputs_per_text = []
                    for st, ed in zip_longest(range(0, len_tokens, 512), range(512, len_tokens, 512), fillvalue=len_tokens):
                        tmp_encoded_input = {}
                        for k, v in encoded_input.items():
                            tmp_encoded_input[k] = v[:, st:ed].to("cuda")
                        inputs_per_text.append(tmp_encoded_input)
                    encoded_input_list.append(inputs_per_text)

        return encoded_input_list

    @classmethod
    def split_df_train_val(self, df, train_ratio=0.5):
        """
        df: pd.DataFrame
        train_ratio: float

        EXAMPLE
        -------
        >>> train_df, val_df = split_df_train_val(df, train_ratio=0.5)
        """
        train_df = df.sample(frac=train_ratio, random_state=0)
        val_df = df.drop(train_df.index)
        return train_df, val_df

    def split_train_val(self, OK_encoded_input_list, NG_encoded_input_list, train_ratio=0.5):
        """
        OK_encoded_input_list: list of list of dict from build_splited_tokens()
        NG_encoded_input_list: list of list of dict from build_splited_tokens()

        EXAMPLE
        -------
        >>> ok_encoded_input_list                                       = build_splited_tokens(df, ["ok_text", "ok_text2"])
        >>> ng_encoded_input_list                                       = build_splited_tokens(df, ["ng_text"])
        >>> train_encoded_input_list, val_encoded_input_list, val_label = split_train_val(ok_encoded_input_list, ng_encoded_input_list, train_ratio=0.5)
        """
        len_OK = len(OK_encoded_input_list)
        len_NG = len(NG_encoded_input_list)
        train_encoded_input_list = []
        val_encoded_input_list = []
        val_label = []

        # random choice
        OK_train_idx = np.random.choice(
            len_OK, int(len_OK * train_ratio), replace=False)
        OK_val_idx = np.array(list(set(range(len_OK)) - set(OK_train_idx)))
        # NOTE 만약 len_NG 가 추출 수보다 작으면 에러가 발생할 수 있음
        NG_val_idx = np.random.choice(len_NG, int(
            len_OK * (1-train_ratio)), replace=False)

        # get items from OK_encoded_input_list with OK_train_idx
        for idx in OK_train_idx:
            train_encoded_input_list.append(OK_encoded_input_list[idx])
        for idx in OK_val_idx:
            val_encoded_input_list.append(OK_encoded_input_list[idx])
            val_label.append(0)
        for idx in NG_val_idx:
            val_encoded_input_list.append(NG_encoded_input_list[idx])
            val_label.append(1)

        return train_encoded_input_list, val_encoded_input_list, val_label
