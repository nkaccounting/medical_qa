import ast
from collections import defaultdict

import jieba
import pandas as pd

df = pd.read_csv("../data/words2id.csv")

word2id = defaultdict(list)

for item in df.itertuples():
    word2id[item[1]] = ast.literal_eval(item[2])

df = pd.read_csv("../data/words_score.csv")

word_scores = [eval(item) for item in df['words_score']]

query = "宝宝咳嗽怎么办？"

query_list = jieba.lcut(query)

index = set()

for word in query_list:
    index.update(word2id[word])

res = []

for i in index:
    score = 0
    for word in query_list:
        score += word_scores[i].get(word, 0)
    res.append((i, score))

print(res)
