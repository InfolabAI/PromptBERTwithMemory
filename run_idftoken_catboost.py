import pandas as pd
import json
# # Introduction
# In this notebook, I modified the notebook from https://www.kaggle.com/code/siddhvr/llm-daigt-sub.
# # What I modified
# I added a pre-condition to check the run time is in scoring or not. If it's run, training with Catboost added, otherwise, saving sample submission and submission.csv instead.

import sys
import gc

import pandas as pd
from sklearn.model_selection import StratifiedKFold
import numpy as np
from sklearn.metrics import roc_auc_score
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer,
)

from datasets import Dataset
from tqdm.auto import tqdm
from transformers import PreTrainedTokenizerFast

from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import roc_auc_score
import os
from utils import Dataset as Dataset_U
from utils import get_df


def cat_train_test(train, test):
    LOWERCASE = False
    VOCAB_SIZE = 14000000

    # Creating Byte-Pair Encoding tokenizer
    raw_tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    # Adding normalization and pre_tokenizer
    raw_tokenizer.normalizer = normalizers.Sequence(
        [normalizers.NFC()] + [normalizers.Lowercase()] if LOWERCASE else [])
    raw_tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
    # Adding special tokens and creating trainer instance
    special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE, special_tokens=special_tokens)
    # Creating huggingface dataset object
    dataset = Dataset.from_pandas(test[['text']])

    def train_corp_iter():
        for i in range(0, len(dataset), 1000):
            yield dataset[i: i + 1000]["text"]
    raw_tokenizer.train_from_iterator(train_corp_iter(), trainer=trainer)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=raw_tokenizer,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
    )
    tokenized_texts_test = []

    print("tokenizing..")
    for text in tqdm(test['text'].tolist()):
        tokenized_texts_test.append(tokenizer.tokenize(text))

    tokenized_texts_train = []

    for text in tqdm(train['text'].tolist()):
        tokenized_texts_train.append(tokenizer.tokenize(text))

    def dummy(text):
        return text

    vectorizer = TfidfVectorizer(ngram_range=(3, 5), lowercase=False, sublinear_tf=True, analyzer='word',
                                 tokenizer=dummy,
                                 preprocessor=dummy,
                                 token_pattern=None, strip_accents='unicode')

    print("vectorizer.fit(tokenized_texts_test)")
    vectorizer.fit(tokenized_texts_test)

    # Getting vocab
    vocab = vectorizer.vocabulary_

    vectorizer = TfidfVectorizer(ngram_range=(3, 5), lowercase=False, sublinear_tf=True, vocabulary=vocab,
                                 analyzer='word',
                                 tokenizer=dummy,
                                 preprocessor=dummy,
                                 token_pattern=None, strip_accents='unicode'
                                 )

    print("tf_train= vectorizer.fit_transform(tokenized_texts_train)")
    tf_train = vectorizer.fit_transform(tokenized_texts_train)
    print("tf_test = vectorizer.transform(tokenized_texts_test)")
    tf_test = vectorizer.transform(
        tokenized_texts_test)  # document-term matrix

    del vectorizer
    gc.collect()

    y_train = train['label'].values

    def get_model():
        from catboost import CatBoostClassifier

    #     clf2 = MultinomialNB(alpha=0.01)
        clf = MultinomialNB(alpha=0.0225)  # 0.8 초 걸림

    #     clf2 = MultinomialNB(alpha=0.01)
        sgd_model = SGDClassifier(
            max_iter=9000, tol=1e-4, loss="modified_huber", random_state=6743)  # 2.2 초 걸림

        p6 = {'n_iter': 3000,
              'verbose': -1,
              'objective': 'cross_entropy',
              'metric': 'auc',
              'learning_rate': 0.00581909898961407,
              'colsample_bytree': 0.78,
              'colsample_bynode': 0.8,
              #         'lambda_l1': 4.562963348932286,
              # 'lambda_l2': 2.97485, 'min_data_in_leaf': 115, 'max_depth': 23, 'max_bin': 898
              }
        p6["random_state"] = 6743

        lgb = LGBMClassifier(**p6)  # 37.4 분 걸림. 1/100 data 에 대해서는 40초 걸림

        cat = CatBoostClassifier(iterations=3000,
                                 verbose=0,
                                 random_seed=6543,
                                 #                            l2_leaf_reg=6.6591278779517808,
                                 learning_rate=0.005599066836106983,
                                 subsample=0.35,
                                 allow_const_label=True, loss_function='CrossEntropy')  # ,task_type="GPU",devices='0') # gpu 사용 불가 # 206.1 분 걸림. 1/100 data 에 대해서는 10분 걸림
        weights = [0.2, 0.31, 0.31, 0.46]

        ensemble = VotingClassifier(estimators=[('mnb', clf),
                                                ('sgd', sgd_model),
                                                ('lgb', lgb),
                                                ('cat', cat)
                                                ],
                                    weights=weights, voting='soft', n_jobs=-1)  # , verbose=True)

        return ensemble

    model = get_model()
    print(model)

    print("model.fit(tf_train, y_train)")
    model.fit(tf_train, y_train)

    # save the model
    # import pickle
    # with open('catboost_model.pkl', 'wb') as f:
    #    pickle.dump(model, f)

    # load the model
    # with open('catboost_model.pkl', 'rb') as f:
    #    model = pickle.load(f)

    gc.collect()

    print("final_preds = model.predict_proba(tf_test)[:,1]")
    final_preds = model.predict_proba(tf_test)[:, 1]
    test['generated'] = final_preds
    return test

    # calc roc_auc with sklearn
    # 0.9954567540423839
    print(roc_auc_score(test['label'], test['generated']))


if __name__ == "__main__":
    import sys
    train, test = get_df(train_data_path=sys.argv[1],
                         test_data_path=sys.argv[2],
                         reduced=False, deduplicated=True)
    cat_train_test(train, test)
