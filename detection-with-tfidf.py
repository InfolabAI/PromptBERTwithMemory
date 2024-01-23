import warnings
import json
from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import MultinomialNB, CategoricalNB
from sklearn.linear_model import SGDClassifier
from transformers import PreTrainedTokenizerFast
from tqdm.auto import tqdm
from datasets import Dataset
from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer,
)
from catboost import CatBoostClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
import numpy as np
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import gc
from config import args
import wandb


# convert string None in args
for k, v in vars(args).items():
    if v == 'None':
        setattr(args, k, None)

wan = wandb.init(project="ours", config=args)


def warn(*args, **kwargs):
    pass


warnings.warn = warn


TEST = True
LOAD_NPZ = True

if not LOAD_NPZ:
    train = pd.read_csv(
        "/mnt/share_nfs/my_method/lg/ggle/dataset/llm-detect-ai-generated-text/DAIST_dataset/train_v2_drcat_02.csv", sep=',')[['text', 'label']]
    edges = pd.read_csv(
        '/mnt/share_nfs/my_method/lg/ggle/dataset/llm-detect-ai-generated-text/DAIST_dataset/edges.csv')
    # breakpoint()

    # more data
    # train2 = pd.read_csv("/mnt/share_nfs/my_method/lg/ggle/dataset/llm-detect-ai-generated-text/DAIST_dataset/concatenated.csv")[['text', 'generated']].rename({'generated': 'label'}, axis=1)
    # org_train = pd.read_csv('/kaggle/input/llm-detect-ai-generated-text/train_essays.csv')[['text', 'generated']].rename({'generated': 'label'}, axis=1)
    # train = pd.concat([org_train, train, train2]).reset_index(drop=True)

    # post process
    train = train.drop_duplicates(subset=['text'])
    train.reset_index(drop=True, inplace=True)

    # reduced data
    if TEST:
        num_data, test_num = 1000, 500000
        random_state = 20
        train_df = train.sample(frac=1, random_state=random_state)  # shuffle
        train = train_df.iloc[:num_data]

        # test = train_df.iloc[-test_num:]
        train2 = pd.read_csv("/mnt/share_nfs/my_method/lg/ggle/dataset/llm-detect-ai-generated-text/DAIST_dataset/concatenated.csv")[['text', 'generated']].rename(
            {'generated': 'label'}, axis=1).drop_duplicates(subset=['text']).reset_index(drop=True).sample(frac=1, random_state=random_state)
        # test = train2.iloc[-test_num:]
        test = edges[['text', 'label']]

    # In[16]:

    print(len(train), train.keys(), train,
          "\n\n", len(test), test.keys(), test)

    # In[17]:

    LOWERCASE = False
    VOCAB_SIZE = 14000000

    # In[18]:

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

    for text in tqdm(test['text'].tolist()):
        tokenized_texts_test.append(tokenizer.tokenize(text))

    tokenized_texts_train = []

    for text in tqdm(train['text'].tolist()):
        tokenized_texts_train.append(tokenizer.tokenize(text))

    # In[19]:

    extractor = TfidfVectorizer

    # In[20]:

    def dummy(text):
        return text

    n_gram = (3, 5)
    vectorizer = extractor(ngram_range=n_gram, lowercase=False, sublinear_tf=True, analyzer='word',
                           tokenizer=dummy,
                           preprocessor=dummy,
                           token_pattern=None, strip_accents='unicode')

    print("vectorizer.fit(tokenized_texts_test)")
    vectorizer.fit(tokenized_texts_test)

    # Getting vocab
    vocab = vectorizer.vocabulary_

    # print(vocab)

    vectorizer = extractor(ngram_range=n_gram, lowercase=False, sublinear_tf=True, vocabulary=vocab,
                           analyzer='word',
                           tokenizer=dummy,
                           preprocessor=dummy,
                           token_pattern=None, strip_accents='unicode'
                           )

    print("tf_train = vectorizer.fit_transform(tokenized_texts_train)")
    tf_train = vectorizer.fit_transform(tokenized_texts_train)
    print("tf_test = vectorizer.transform(tokenized_texts_test)")
    tf_test = vectorizer.transform(tokenized_texts_test)

    del vectorizer
    gc.collect()

    print(n_gram, len(vocab))

    y_train = train['label'].values

else:
    import scipy.sparse as sp
    tf_train = sp.load_npz(
        "/mnt/share_nfs/my_method/lg/ggle/dataset/llm-detect-ai-generated-text/DAIST_dataset/train.npz")
    y_train = np.load(
        "/mnt/share_nfs/my_method/lg/ggle/dataset/llm-detect-ai-generated-text/DAIST_dataset/train_label.npy")
    tf_test = sp.load_npz(
        "/mnt/share_nfs/my_method/lg/ggle/dataset/llm-detect-ai-generated-text/DAIST_dataset/edges.npz")
    test = pd.read_csv(
        '/mnt/share_nfs/my_method/lg/ggle/dataset/llm-detect-ai-generated-text/DAIST_dataset/edges.csv')

# In[21]:


if TEST:
    iter_u = args.iter_u
else:
    iter_u = 1000


# In[22]:


def get_catboost(verbose):
    model = CatBoostClassifier(iterations=iter_u*3,
                               random_seed=6543,
                               learning_rate=0.005599066836106983,
                               subsample=0.35,
                               allow_const_label=True, loss_function='CrossEntropy',
                               bootstrap_type='Poisson',  # GPU support
                               task_type="GPU",
                               devices="0:1")
    model.fit(tf_train, y_train, verbose=verbose)
    ret = model.predict_proba(tf_test)[:, 1]
    print(ret)
    return ret


def get_ensemble(verbose):
    nb = MultinomialNB(alpha=args.nb_alpha)

    sgd_model = SGDClassifier(max_iter=iter_u*args.sgd_iter, tol=args.sgd_tol,
                              loss="modified_huber", random_state=6743, penalty=args.sgd_penalty,
                              alpha=args.sgd_alpha, learning_rate=args.sgd_learning_rate, eta0=args.sgd_eta0,
                              power_t=args.sgd_power_t, early_stopping=args.sgd_early_stopping,
                              warm_start=args.sgd_warm_start)

    lgb = LGBMClassifier(boosting_type=args.lgbm_boosting_type,
                         num_leaves=args.lgbm_num_leaves,
                         max_depth=args.lgbm_max_depth,
                         learning_rate=args.lgbm_learning_rate,
                         n_estimators=args.lgbm_n_estimators,
                         subsample_for_bin=args.lgbm_subsample_for_bin,
                         colsample_bytree=args.lgbm_colsample_bytree,
                         colsample_bynode=args.lgbm_colsample_bynode,
                         reg_alpha=args.lgbm_reg_alpha,
                         reg_lambda=args.lgbm_reg_lambda,
                         random_state=6743)

    cat = CatBoostClassifier(iterations=iter_u*args.catboost_iter,
                             random_seed=6543,
                             learning_rate=args.catboost_learning_rate,
                             subsample=args.catboost_subsample,
                             allow_const_label=True, loss_function='CrossEntropy',
                             l2_leaf_reg=args.catboost_l2_leaf_reg,
                             random_strength=args.catboost_random_strength,
                             bagging_temperature=args.catboost_bagging_temperature,
                             bootstrap_type='Poisson',  # GPU support
                             task_type="GPU",
                             devices="0:1")

    weights = [args.voting_weight_nb, args.voting_weight_sgd,
               args.voting_weight_lgbm, args.voting_weight_catboost]

    model = VotingClassifier(estimators=[
        ('nb', nb),
        ('sgd', sgd_model),
        ('lgb', lgb),
        ('cat', cat)
    ],
        weights=weights, voting='soft', n_jobs=-1, verbose=verbose)

    model.fit(tf_train, y_train)
    ret = model.predict_proba(tf_test)[:, 1]
    # print(ret)
    return ret


if tf_test.shape[0] <= 5:
    # if not, just sample submission
    # sub.to_csv('submission1.csv', index=False)
    test['generated'] = 0
    test[['id', 'generated']].to_csv('submission.csv', index=False)
else:
    gc.collect()
    final_preds = get_ensemble(True if TEST else False)
    test['generated'] = final_preds


def save_edge_cases(df):
    ng_edges = df[lambda x:x.label == 1][lambda x:x.generated <
                                         df[lambda x:x.label == 0]['generated'].max()]
    ok_edges = df[lambda x:x.label == 0][lambda x:x.generated >
                                         df[lambda x:x.label == 1]['generated'].min()]
    edges = pd.concat([ng_edges, ok_edges])
    edges.to_csv(
        '/mnt/share_nfs/my_method/lg/ggle/dataset/llm-detect-ai-generated-text/DAIST_dataset/edges.csv', index=False)


if TEST:
    roc = roc_auc_score(test['label'], test['generated'])
    print('ROC score', roc)
    # save_edge_cases(test)
    wandb.log({"roc": roc, })

else:
    test[['id', 'generated']].to_csv('submission.csv', index=False)
