import ast
from collections import defaultdict

import pandas as pd

df = pd.read_csv("../data/answer_cut.csv")

length = len(df)

word_IVF = defaultdict(set)

for i, one_text in enumerate(df['answer_cut']):
    if i % 1000 == 0:
        print(i / length)
    one_text = ast.literal_eval(one_text)
    for word in one_text:
        word_IVF[word].add(i)

words = []
ids = []

length = len(word_IVF)
i = 0
for word in word_IVF:
    i += 1
    if i % 1000 == 0:
        print(i / length)
    words.append(word)
    ids.append(list(word_IVF[word]))

dataframe = pd.DataFrame()

dataframe["word"] = pd.Series(words)
dataframe["id"] = pd.Series(ids)

dataframe.to_csv("../data/words2id.csv", index=0, encoding="utf-8")
