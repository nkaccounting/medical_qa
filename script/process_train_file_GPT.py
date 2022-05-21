import pandas as pd

name = "all"

original_data = '../data/{name}.csv'.format(name=name)

out_put_name = '../{name}.txt'.format(name=name)

dataframe = pd.read_csv(original_data)

# 去除多余的？
dataframe['title'] = dataframe['title'].str.rstrip('?|？')

# 合并
dataframe['text'] = '<QBOS>' + dataframe['title'] + '？<QEOS><ABOS>' + dataframe['answer'] + '<AEOS>'

dataframe.replace('\s+|\n', '', regex=True, inplace=True)

dataframe = dataframe.sample(frac=1.0)

dataframe = dataframe.dropna()

with open(out_put_name, 'w', encoding='utf-8') as f:
    for text in dataframe['text']:
        f.write(text + '\n')
