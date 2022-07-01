import json

import pandas as pd
import random

import requests

questions = pd.read_csv("../data/questions.csv")

answers = pd.read_csv("../data/answers.csv")

max_len = len(questions)
# 采样1000条来作为测试的target
sample_num = 200

sample_index = random.sample(range(0, max_len), sample_num)

ans = []

total_score = 0

for i in sample_index:
    response = requests.get("http://192.168.242.239:2265/qa", params={'text': questions["questions"][i]})
    ans.append(response.text)
    u = json.loads(response.text)
    score = 0
    for j, dit in enumerate(u["information"]):
        if dit['候选回答{j}'.format(j=j)] == answers["answers"][i]:
            score = 1 / (j + 1)
            break
    total_score += score

MRR = total_score / sample_num
print(MRR)
