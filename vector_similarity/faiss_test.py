# 准备示例数据
import numpy as np

d = 64  # 维度
nb = 100  # 数据库的数据量
nq = 10  # 查询数据的数据量
np.random.seed(1234)  # make reproducible
xb = np.random.random((nb, d)).astype('float32')
xb[:, 0] += np.arange(nb) / 1000.
xq = np.random.random((nq, d)).astype('float32')
xq[:, 0] += np.arange(nq) / 1000.

# 构建索引并添加向量
import faiss  # make faiss available

index = faiss.IndexFlatL2(d)  # 构建索引
# 这里使用了L2范式构建索引，还可以使用IndexFlatIP（内积）等其他类型索引
print(index.is_trained)
index.add(xb)  # 向索引添加向量
print(index.ntotal)

# 查找
k = 4  # 查找最近的top4
D, I = index.search(xb[:5], k)  # 合理性检验
print(xb[:5])
print(I)
print(D)
# 返回的i是index，自上而下，从左到右是k个；D是distance，表示距离的远近

D, I = index.search(xq, k)  # 查找
print(I[:5])  # 展示前五个查询的查询结果
print(I[-5:])  # 展示后五个查询的查询结果
