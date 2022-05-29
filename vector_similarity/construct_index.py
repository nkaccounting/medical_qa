import ast
import time

import faiss
import numpy as np
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


def construct_normal_index(d, vectors):
    index = faiss.IndexFlatL2(d)
    index.add(vectors)
    return index


def construct_center_index(d, center_num, vectors):
    quantizer = faiss.IndexFlatL2(d)  # 量化器
    index = faiss.IndexIVFFlat(quantizer, d, center_num, faiss.METRIC_L2)
    index.train(vectors)  # 要对这堆向量算出聚类中心
    index.add(vectors)
    index.nprobe = 5  # 修改查找的聚类中心，默认的时候是先去找nprobe个聚类中心，然后再比较这里面的所有
    return index


# 注意nbits_per_idx必须小于等于8,最后一个参数表示的是子空间聚类中心的个数，8-256,7-128……
# 倒数第二个参数是指把原始的向量空间划分成M等份
def construct_compression_index(d, center_num, compression_per_size, nbits_per_idx, vectors):
    quantizer = faiss.IndexFlatL2(d)  # 量化器
    index = faiss.IndexIVFPQ(quantizer, d, center_num, compression_per_size, nbits_per_idx)
    index.train(vectors)  # 要对这堆向量算出聚类中心
    index.add(vectors)
    index.nprobe = 5  # 修改查找的聚类中心，默认的时候是先去找nprobe个聚类中心，然后再比较这里面的所有
    return index


def search_one_query(question, index, top_k):
    t1 = time.time()
    query = get_sentence_embedding(question)
    t2 = time.time()
    D, I = index.search(query.numpy(), top_k)
    t3 = time.time()
    print("句子生成向量时间，", t2 - t1)
    print("索引向量时间，", t3 - t2)
    return D, I


df = pd.read_csv("../data/question.csv", encoding="utf-8")

questions = [question for question in df["questions"]]
vectors = [ast.literal_eval(vector)[0] for vector in df["vectors"]]

vectors = np.array(vectors, dtype="float32")

# bert向量维度获取
d = len(vectors[0])

# normal_index = construct_normal_index(d, vectors)
#
# center_index = construct_center_index(d, 10, vectors)

# 大的聚类和小的聚类都选256-8；每个768维向量划分成8份
compression_index = construct_compression_index(d, 256, 8, 8, vectors)

# search_one_query("孩童中耳炎流黄水要如何医治", normal_index, 2)
#
# search_one_query("孩童中耳炎流黄水要如何医治", center_index, 2)
#
search_one_query("孩童中耳炎流黄水要如何医治", compression_index, 2)

faiss.write_index(compression_index, "./compression_index.bin")
