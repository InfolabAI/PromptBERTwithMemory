#from transformers import AutoTokenizer
#import transformers
#import torch
#
##model = "meta-llama/Llama-2-7b-chat-hf"
#model = "TheBloke/Llama-2-7B-Chat-GGUF"
#token = "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"
#
#tokenizer = AutoTokenizer.from_pretrained(model)
#pipeline = transformers.pipeline(
#    "text-generation",
#    model=model,
#    torch_dtype=torch.float16,
#    device_map="auto",
#)
#
#sequences = pipeline(
#    'I liked "Breaking Bad" and "Band of Brothers". Do you have any recommendations of other shows I might like?\n',
#    do_sample=True,
#    top_k=10,
#    num_return_sequences=1,
#    eos_token_id=tokenizer.eos_token_id,
#    max_length=200,
#)
#for seq in sequences:
#    print(f"Result: {seq['generated_text']}")


from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, TextStreamer
import torch
import os
from tqdm import tqdm
import pandas as pd
import re

def cleanhtml(raw_html): # html 요소 제거
  cleanr = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
  cleantext = re.sub(cleanr, '', raw_html)
  return cleantext


#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
cuda_text = "cuda:1"
device = torch.device(cuda_text)
model_name_or_path = "TheBloke/Llama-2-7b-Chat-GPTQ"
# To use a different branch, change revision
# For example: revision="gptq-4bit-64g-actorder_True"
model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                             #device_map="auto",
                                             device_map=cuda_text,
                                             trust_remote_code=False,
                                             revision="main")#.to(device)

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
streamer = TextStreamer(tokenizer) # stdout 에 stream 하는 기본형 from https://huggingface.co/docs/transformers/internal/generation_utils

data_path = "/home/robert.lim/heechul/ggle/dataset/llm-detect-ai-generated-text"
train_df = pd.read_csv(os.path.join(data_path, "train_essays.csv"))
data_list = []
for i in tqdm(range(len(train_df))):
    ok_text = train_df.loc[i, "text"].replace("\n", " ")
    id = train_df.loc[i, "id"]

    prompt = f"[User's essay] {ok_text}"

    prompt_template=f'''[INST] <<SYS>>
    You are a essay writer. And you have to make the essay as similar as given user's essay.
    <</SYS>>
    {prompt}[/INST]

    '''

    input_ids = tokenizer(prompt_template, return_tensors='pt').input_ids.to(device)
    output = model.generate(inputs=input_ids, temperature=0.7, do_sample=True, top_p=0.95, top_k=40, max_new_tokens=2048)#, streamer=streamer)
    generated_text = cleanhtml(tokenizer.decode(output[0]).split("[/INST]")[1].strip()) # output 은 입력 text 도 포함하므로, [/INST] 뒤의 text 만 추출하고, 앞뒤 공백 제거 후, html 요소 제거
    data_list.append((id, generated_text, ok_text))
    pd.DataFrame(data_list, columns=["id", "generated_text", "ok_text"]).to_csv(os.path.join(data_path, "generated_essays.csv"), index=False)

# Inference can also be done using transformers' pipeline

#print("*** Pipeline:")
#pipe = pipeline(
#    "text-generation",
#    model=model,
#    tokenizer=tokenizer,
#    max_new_tokens=512,
#    do_sample=True,
#    temperature=0.7,
#    top_p=0.95,
#    top_k=40,
#    repetition_penalty=1.1
#)
#
#output = pipe(prompt_template)
#print(output[0]['generated_text'])