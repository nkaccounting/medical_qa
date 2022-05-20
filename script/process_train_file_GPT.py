import pandas as pd

df = pd.read_csv('../data/hello.csv')

# 去除多余的？
df['title'] = df['title'].str.rstrip('?|？')

# 合并
df['text'] = '<BOS>'+df['title'] + '？<EOS>' + df['answer']+'<EOS>'

df.replace('\s+|\n', '', regex=True, inplace=True)

df = df.sample(frac=1.0)

df = df.dropna()

with open('../all.txt', 'w', encoding='utf-8') as f:
    for text in df['text']:
        f.write(text+'\n')
