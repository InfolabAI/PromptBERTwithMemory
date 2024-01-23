import pandas as pd
import os
import torch
from faiss_u import FaissNN, NearestNeighbourScorer
from sampler_u import ApproximateGreedyCoresetSampler
from detection_u import Detection
from transformers import BertTokenizer, BertModel
from transformers import BertModel, BertConfig
from utils import Dataset as Dataset_U
from sklearn.metrics import roc_auc_score

data_path = "/home/robert.lim/heechul/ggle/dataset/llm-detect-ai-generated-text"
faiss_path = "faiss_bert-base-multilingual-cased"
os.makedirs(faiss_path, exist_ok=True)

tokenizer = BertTokenizer.from_pretrained('saved_bert-base-multilingual-cased')
config = BertConfig.from_pretrained("saved_bert-base-multilingual-cased", output_hidden_states=True)
model = BertModel.from_pretrained("saved_bert-base-multilingual-cased", config=config).to("cuda")
#tokenizer.save_pretrained(save_directory=save_path)
#config.save_pretrained(save_directory=save_path)
#model.save_pretrained(save_directory=save_path)
device = torch.device("cuda")

train_df = pd.read_csv(os.path.join(data_path, "train_essays.csv"))[["text", "generated"]].rename({"generated": "label"}, axis=1)
#test_df = pd.read_csv(os.path.join(data_path, "test_essays.csv"))[["id", "text"]]

train_df2 = pd.read_csv(os.path.join("/home/robert.lim/heechul/ggle/dataset/llm-detect-ai-generated-text", "DAIST_dataset", "train_v2_drcat_02.csv"), sep=',')[["text", "label"]] # LABEL 1 for AI generated, 0 for human
train_df2 = train_df2.drop_duplicates(subset=['text'])
train_df2.reset_index(drop=True, inplace=True)
train_df = pd.concat([train_df, train_df2])

#tmp0 = train_df[lambda x:x.label == 0]
#tmp1 = train_df[lambda x:x.label == 1]
#times = 100
#tmp0 = tmp0.head(len(tmp0)//times) 
#tmp1 = tmp1.head(len(tmp1)//times) 
#train_df = pd.concat([tmp0, tmp1])

train_df, test_df = Dataset_U.split_df_train_val(train_df, train_ratio=0.9)

coreset_p=0.1
layers_to_extract_from=[2, 3, 4, 5, 6, 7, 8, 9]
n_nearest_neighbours=1

# load sampler and algorithm
sampler = ApproximateGreedyCoresetSampler(coreset_p, device)
scorer = NearestNeighbourScorer(n_nearest_neighbours=n_nearest_neighbours)
dataset = Dataset_U(tokenizer)
train_encoded_input_list = dataset.build_splited_tokens(train_df, target_column_list=["text"])

params = locals()
uniformaly_instance = Detection(model, sampler=sampler, scorer=scorer, layers_to_extract_from=layers_to_extract_from)
uniformaly_instance.fit(train_encoded_input_list)
#ret = uniformaly_instance.predict(train_encoded_input_list)

uniformaly_instance.anomaly_scorer.save(faiss_path)
test_encoded_input_list = dataset.build_splited_tokens(test_df, target_column_list=["text"])
scores = uniformaly_instance.predict(test_encoded_input_list)

# NOTE 여기서 score 를 어떻게 nomalize 하느냐 가 성능에 매우 중요할 것
min_scores = scores.min()
max_scores = scores.max()
scores = (scores - min_scores) / (max_scores - min_scores)

test_df["generated"] = scores.numpy()
#test_df = test_df.drop(["prompt_id", "text"], axis=1)
test_df.to_csv(os.path.join(data_path, "submission.csv"), index=False)
print(f"roc_auc_score: {roc_auc_score(test_df['label'], test_df['generated'])}")


"""
# HPO targets
- model
- sampler
- coreset_p
- layers_to_extract_from
- n_nearest_neighbours
- normalization
- embedding target['all tokens', 'CLS token']
- token length limit 초과를 다루는 방법
# HOw to make valiation dataset?
"""