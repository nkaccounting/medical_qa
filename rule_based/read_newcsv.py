import ast
from collections import Counter

import pandas as pd

df = pd.read_csv("new.csv")

all_question = []
for question in df['question_cut']:
    all_question += ast.literal_eval(question)

res = Counter(all_question)

print(res)
