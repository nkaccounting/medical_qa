import time

import faiss
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel

model_dir = "../sbert-base-chinese-nli"

# Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModel.from_pretrained(model_dir)


# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


# Sentences we want sentence embeddings for
def get_sentence_embedding(sentences: str):
    # Tokenize sentences
    encoded_input = tokenizer(sentences, truncation=True, padding='max_length', max_length=512, return_tensors='pt')

    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)

    # Perform pooling. In this case, max pooling.
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    return sentence_embeddings


def search_one_query(question, index, top_k):
    t1 = time.time()
    query = get_sentence_embedding(question)
    t2 = time.time()
    D, I = index.search(query.numpy(), top_k)
    t3 = time.time()
    print("句子生成向量时间，", t2 - t1)
    print("索引向量时间，", t3 - t2)
    return D, I


t1 = time.time()
index = faiss.read_index("./compression_index.bin")
t2 = time.time()
print("读取向量构建faiss索引所用时间，", t2 - t1)

t1 = time.time()
df = pd.read_csv("../data/questions.csv", encoding="utf-8")
questions = [question for question in df["questions"]]
t2 = time.time()
print("加载索引和question的对应关系所用时间，", t2 - t1)

t1 = time.time()
df = pd.read_csv("../data/answers.csv", encoding="utf-8")
answers = [answer for answer in df["answers"]]
t2 = time.time()
print("加载索引和answer的对应关系所用时间，", t2 - t1)

while True:
    text = input('请输入您想咨询的疾病问题，目前仅支持（儿科，妇产科，男科，内科，外科，肿瘤科）:')
    D, I = (search_one_query(text, index, 5))

    for i, id in enumerate(I[0]):
        if D[0][i] > 150:
            print("对不起，目前我还不会这个问题，或者您的提问不够明确，待我学习后再来吧~")
            break
        print("相似度距离信息", D[0][i])
        print("相似问题{i}:".format(i=i), questions[id])
        print('候选回答{i}:'.format(i=i), answers[id])
