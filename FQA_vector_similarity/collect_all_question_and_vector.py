import pandas as pd

name = "all"
out_name = "question"

original_data = '../data/{name}.csv'.format(name=name)

out_put_name = '../data/{out_name}.csv'.format(out_name=out_name)

dataframe = pd.read_csv(original_data)

# 去除多余的？
dataframe['title'] = dataframe['title'].str.rstrip('?|？')

dataframe.replace('\s+|\n', '', regex=True, inplace=True)

dataframe = dataframe.dropna()

print(len(dataframe))

questions = pd.Series(list(set(dataframe["title"])))

print(len(questions))

df = pd.DataFrame()

df["questions"] = questions

import torch
from transformers import AutoTokenizer, AutoModel

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


vectors = []

for question in questions:
    vector = get_sentence_embedding(question)
    vectors.append(vector.tolist())

df["vectors"] = pd.Series(vectors)

df.to_csv(out_put_name, encoding="utf-8", index=0)
