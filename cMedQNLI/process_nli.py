import json

import pandas as pd

df = pd.read_csv("data/test.csv", header=None, delimiter="\t")

print(len(df))

data = []

for i, item in enumerate(df.itertuples()):
    if item[1] == 0:
        contradiction_item = {
            'sentence1': item[2],
            'sentence2': item[3],
            'label': 'contradiction'
        }
        data.append(contradiction_item)
    else:
        entailment_item = {
            'sentence1': item[2],
            'sentence2': item[3],
            'label': 'entailment'
        }
        data.append(entailment_item)

with open('data/QNLI_eval_file.json', 'w', encoding='utf-8') as fp:
    json.dump({
        'data': data
    }, fp, ensure_ascii=False, indent=2)
