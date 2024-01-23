import gc
import pandas as pd
from utils import get_df, get_model, get_dataloader, train_model_a_epoch, evaluation, to_metric, save_model, AnomalyScorer, PromptBERT, reproduce_metric, test_roc, df_update
from run_idftoken_catboost import cat_train_test
from sklearn.metrics import roc_auc_score
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def test_with_bert(
    # model_path="/mnt/share_nfs/my_method/lg/ggle/src/llm-detect-ai-generated-text/saved_bert-base-multilingual-cased_20240121092622_True_sgd/14",
    # faiss_path="/mnt/share_nfs/my_method/lg/ggle/src/llm-detect-ai-generated-text/faiss_bank_20240121092622_True_sgd/epoch_14",
    model_path="/mnt/share_nfs/my_method/lg/ggle/src/llm-detect-ai-generated-text/saved_bert-base-multilingual-cased_20240121092622_True_sgd/16",
    faiss_path="/mnt/share_nfs/my_method/lg/ggle/src/llm-detect-ai-generated-text/faiss_bank_20240121092622_True_sgd/epoch_16",
    train_path="/mnt/share_nfs/my_method/lg/ggle/dataset/llm-detect-ai-generated-text/DAIST_dataset/train_v2_drcat_02.csv",
    # test_path="/mnt/share_nfs/my_method/lg/ggle/dataset/llm-detect-ai-generated-text/DAIST_dataset/concatenated.csv",
    test_path="/mnt/share_nfs/my_method/lg/ggle/dataset/llm-detect-ai-generated-text/test_essays.csv",
    bert_ratio=0.1,
    virtual_test_path=None,
    reduced=False,
    deduplicated=False
):

    tokenizer, _, cls_model = get_model(
        load_path=model_path)
    cls_model = PromptBERT(
        cls_model, load_path=model_path, train_classifier=True, optimizer='sgd')
    if virtual_test_path is not None:
        train, test = get_df(train_data_path=train_path,
                             test_data_path=test_path, reduced=True, deduplicated=False)
        test['generated'] = 1
        _, test2 = get_df(train_data_path=train_path,
                          test_data_path=virtual_test_path, reduced=True, deduplicated=True)
        test = pd.concat(
            [test, test2[['id', 'prompt_id', 'text', 'generated']]])
    else:
        train, test = get_df(train_data_path=train_path,
                             test_data_path=test_path, reduced=reduced, deduplicated=deduplicated)
        if len(test) < 5:
            test['generated'] = 0
            return train, test
        # else:
        #    test['generated'] = 0.5
        #    return train, test

    if 'generated' in list(test.keys()):
        test.rename(columns={'generated': 'label'}, inplace=True)

    # test0 = test.iloc[:100]
    # test1 = test.iloc[100:200]
    # test.set_index('id', inplace=True)
    # test0.set_index('id', inplace=True)
    # test1.set_index('id', inplace=True)
    #
    # test0['generated'] = 0.1
    # test1['generated'] = 0.9
    # test01 = pd.concat([test0, test1])
    # test['generated'] = test01['generated']

    anomaly_scorer = AnomalyScorer(tokenizer, train, test)

    scores = anomaly_scorer.get_anomaly_score(
        cls_model, load_path=faiss_path, target='test')

    sensitivity = scores.sort()[0][int(scores.shape[0]*bert_ratio)].item()
    test['scores'] = scores

    bert_test = test[lambda x: x.scores < sensitivity]
    cat_test = test.copy()  # test[lambda x: x.scores >= sensitivity]

    print(f"\n\n>>>     The length of bert_test {len(bert_test)}, {bert_test}")
    print(f"\n\n>>>     The length of cat_test {len(cat_test)}, {cat_test}")

    if len(bert_test) > 0:
        if "label" not in list(bert_test.keys()):
            bert_test['label'] = 1

        _, bert_test_loader = get_dataloader(bert_test, bert_test, tokenizer)
        preds_argmax, preds_for_1, labels_t = evaluation(
            cls_model=cls_model, test_loader=bert_test_loader)
        bert_test['generated'] = preds_for_1
        bert_test['method'] = 'bert'
        gc.collect()
        # test_roc(bert_test.set_index('id'), pd.read_csv('cat.csv').set_index(
        #    'id'))
        # breakpoint()

    if len(cat_test) > 0:
        cat_test = cat_train_test(train, cat_test)
        cat_test['method'] = 'cat'

    if len(bert_test) > 0 and len(cat_test) > 0:
        bert_test.set_index('id', inplace=True)
        cat_test.set_index('id', inplace=True)
        test.set_index('id', inplace=True)
        # test['generated'] = pd.concat([bert_test, cat_test])['generated']
        cat_test = df_update(bert_test, cat_test, 0.7)
        test['generated'] = cat_test['generated']
        test.reset_index(inplace=True)
    else:
        test = cat_test if len(bert_test) == 0 else bert_test

    try:
        print(roc_auc_score(test['label'].to_numpy(),
                            test['generated'].to_numpy()))
    except:
        pass

    # test.to_csv('cat.csv', index=False)
    return train, test
    # breakpoint()

    # cat_test['generated'] = 1  # NOTE 제거 필요

    # print(reproduce_metric(scores, 0.05, test['label'].to_numpy()))
    # breakpoint()


if __name__ == "__main__":
    import sys
    print(test_with_bert(model_path=sys.argv[1], faiss_path=sys.argv[2], train_path=sys.argv[3], test_path=sys.argv[4], bert_ratio=0.5, reduced=True, deduplicated=False)
          )
