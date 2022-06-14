from collections import defaultdict

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel

original_data = '../data/all.csv'
index_vector_name = '../data/vectors.csv'
index_question_name = "../data/questions.csv"
index_answer_name = "../data/answers.csv"
model_dir = "../sbert-base-chinese-nli"

# Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModel.from_pretrained(model_dir)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 将模型加到GPU
model = model.to(device)


# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


# Sentences we want sentence embeddings for
def get_sentence_embedding(sentences: str):
    # Tokenize sentences
    encoded_input = tokenizer(sentences, truncation=True, padding='max_length', max_length=512, return_tensors='pt')
    encoded_input = encoded_input.to(device)

    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)

    # Perform pooling. In this case, max pooling.
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    return sentence_embeddings


dataframe = pd.read_csv(original_data)

# 一些预处理，数据清洗操作
dataframe['title'] = dataframe['title'].str.rstrip('?|？')
dataframe.replace('\s+|\n', '', regex=True, inplace=True)
dataframe = dataframe.dropna()

questions = pd.Series(list(set(dataframe["title"])))

question2answer = defaultdict(str)

for item in dataframe.itertuples():
    # 以第一个出现的答案作为最终答案
    if not question2answer.get(item[2]):
        question2answer[item[2]] = item[4]

answers = []

for question in questions:
    answers.append(question2answer[question])

df = pd.DataFrame()

df["questions"] = questions
df["answers"] = pd.Series(answers)

vectors = []
i = 0
for question in questions:
    i += 1
    if i % 1000 == 0:
        print("已完成：", i)
    vector = get_sentence_embedding(question)
    vectors.append(vector.tolist())
df["vectors"] = pd.Series(vectors)

df["vectors"].to_csv(index_vector_name, encoding="utf-8", index=0)
df["questions"].to_csv(index_question_name, encoding="utf-8", index=0)
df["answers"].to_csv(index_answer_name, encoding="utf-8", index=0)
