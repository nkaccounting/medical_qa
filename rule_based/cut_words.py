# encoding=utf-8

import jieba
import pandas as pd

# 设置idf文档
# jieba.analyse.set_idf_path("../extra_dict/idf.txt.big")

# 设置停用词文档
# jieba.analyse.set_stop_words("../extra_dict/stop_words.txt")

df = pd.read_csv("../data/all.csv", encoding="utf-8")

questions = []
answers = []

for question in df["title"]:
    question = str(question)
    one_res = jieba.lcut(question)
    questions.append(one_res)

df['question_cut'] = pd.Series(questions)

for answer in df["answer"]:
    answer = str(answer)
    one_res = jieba.lcut(answer)
    answers.append(one_res)

df['question_cut'] = pd.Series(questions)
df['answer_cut'] = pd.Series(answers)

df.to_csv("new.csv", encoding='utf-8', index=0)

# textrank
# jieba.analyse.textrank(sentence, topK=20, withWeight=False, allowPOS=('ns', 'n', 'vn', 'v')) 直接使用，接口相同，注意默认过滤词性。


# git clone https://huggingface.co/csebuetnlp/mT5_m2o_chinese_simplified_crossSum
