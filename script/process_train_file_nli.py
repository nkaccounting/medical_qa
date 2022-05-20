import json
import random

import pandas as pd

df = pd.read_csv('../data/all.csv')

# 去除多余的？
df['title'] = df['title'].str.rstrip('?|？') + '？'

df.replace('\s+|\n', '', regex=True, inplace=True)

df = df.sample(frac=1.0)

df = df.dropna()

n = len(df) - 1

data=[]


for i, item in enumerate(df.itertuples()):
    if i==6:
        break
    entailment_item = {
        'sentence1': item[2],
        'sentence2': item[4],
        'label': 'entailment'
    }
    seed = random.randint(0, n)
    while seed == i:
        seed = random.randint(0, n)
        print('Duplicated sample....')
        print('Resample....')
    contradiction_item = {
        'sentence1': item[2],
        'sentence2': df.iloc[seed, 3],
        'label': 'contradiction'
    }
    data.append(entailment_item)
    data.append(contradiction_item)

with open('QNLI_val_file.json', 'w', encoding='utf-8') as fp:
    json.dump({
        'data': data
    }, fp, ensure_ascii=False, indent=2)
