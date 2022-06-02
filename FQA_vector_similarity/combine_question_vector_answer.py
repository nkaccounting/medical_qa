from collections import defaultdict

import pandas as pd

dataframe = pd.read_csv("../data/question.csv")

original_dataframe = pd.read_csv("../data/all.csv")
# 去除多余的？
original_dataframe['title'] = original_dataframe['title'].str.rstrip('?|？')

original_dataframe.replace('\s+|\n', '', regex=True, inplace=True)

original_dataframe = original_dataframe.dropna()

question2answer = defaultdict(str)

for item in original_dataframe.itertuples():
    # 以第一个出现的答案作为最终答案
    if not question2answer.get(item[2]):
        question2answer[item[2]] = item[4]

answers = []

for question in dataframe['questions']:
    answers.append(question2answer[question])

dataframe["answers"] = pd.Series(answers)

dataframe.to_csv("../data/question_vector_answer.csv", encoding="utf-8", index=0)
