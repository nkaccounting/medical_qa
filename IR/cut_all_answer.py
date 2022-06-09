import jieba
import pandas as pd

from utils import utils

doc = []

df = pd.read_csv("../data/all.csv")
df.replace('\s+|\n', '', regex=True, inplace=True)
df = df.dropna()

answers = pd.Series(list(set(df["answer"])))

length = len(answers)

i = 0
for answer in answers:
    i += 1
    if i % 1000 == 0:
        print(i / length)
    words = list(jieba.cut(answer))
    words = utils.filter_stop(words)
    doc.append(words)

out = pd.DataFrame()
out["answer"] = answers
out["answer_cut"] = pd.Series(doc)

out.to_csv("../data/answer_cut.csv", index=0, encoding="utf-8")
