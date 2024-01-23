
import numpy as np
import pandas as pd
import os
import torch
from itertools import zip_longest
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
from transformers import BertModel, BertConfig
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
config = BertConfig.from_pretrained("bert-base-uncased", output_hidden_states=True)
model = BertModel.from_pretrained("bert-base-uncased", config=config).to("cuda")
sample_text = "Replace me by any text you'd like."
device = torch.device("cuda")


data_path = "/home/robert.lim/heechul/ggle/dataset/llm-detect-ai-generated-text"
train_df = pd.read_csv(os.path.join(data_path, "train_essays.csv"))
test_df = pd.read_csv(os.path.join(data_path, "test_essays.csv"))



# get text embedding
def embed(train_df):
    """
    BERT 를 이용해 train_df 를 embed 하는 function
    """
    embedding_list = []
    for i in tqdm(range(len(train_df))):
        # NOTE 총 train 모드 시, 약 4분, memory 2.2GB 필요.
        # NOTE 총 test 모드 시, train 모드와 차이 없음.
        ok_text = train_df.loc[i, "text"].replace("\n", " ")
        encoded_input = tokenizer(f"[CLS] {ok_text} [SEP]", return_tensors='pt') # pt = pytorch
        last_hidden_state_list = []
        len_tokens = len(encoded_input["input_ids"][0])
        model.eval()
        with torch.no_grad():
            for st, ed  in zip_longest(range(0, len_tokens, 512), range(512, len_tokens, 512), fillvalue=len_tokens):
                tmp_encoded_input = {}
                for k, v in encoded_input.items():
                    tmp_encoded_input[k] = v[:, st:ed].to("cuda")

                output = model(**tmp_encoded_input)
                last_hidden_state_list.append(output['last_hidden_state'].detach().cpu())
        
        output = torch.cat(last_hidden_state_list, dim=1).mean(1) # token 방향으로 모두 합치고 평균
        embedding_list.append(output)

    output = torch.cat(embedding_list, dim=0)
    print(output)