# encoding=utf-8
from collections import defaultdict

import pandas as pd

df = pd.read_csv("../data/all.csv", encoding="utf-8")
# 简单的数据清洗
df['title'] = df['title'].str.rstrip('?|？')
df.replace('\s+|\n', '', regex=True, inplace=True)
df.dropna(inplace=True)

questions = []

ask_nums = [defaultdict(int) for i in range(8)]

# 支持1~8个字的统计
for question in df["title"]:
    for i in range(8):
        ask_nums[i][question[-i - 1:]] += 1

sorted_result = []
for ask_num in ask_nums:
    sorted_result.append(list(reversed(sorted(ask_num.items(), key=lambda d: d[1]))))

# 显示top100
for item in sorted_result[1][:100]:
    print(item)
