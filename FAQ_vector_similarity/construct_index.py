import ast

import faiss
import numpy as np
import pandas as pd


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


df = pd.read_csv("../data/vectors.csv", encoding="utf-8")
vectors = [ast.literal_eval(vector)[0] for vector in df["vectors"]]

vectors = np.array(vectors, dtype="float32")

# bert向量维度获取
d = len(vectors[0])

# normal_index = construct_normal_index(d, vectors)
#
# center_index = construct_center_index(d, 10, vectors)

# 大的聚类和小的聚类都选256-8；每个768维向量划分成8份
compression_index = construct_compression_index(d, 256, 8, 8, vectors)

faiss.write_index(compression_index, "./compression_index.bin")
