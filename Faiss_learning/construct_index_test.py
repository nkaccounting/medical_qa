import torch
from transformers import AutoTokenizer, AutoModel

model_dir = "../sbert-base-chinese-nli"

# Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModel.from_pretrained(model_dir)


# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


# Sentences we want sentence embeddings for
def get_sentence_embedding(sentences: str):
    # Tokenize sentences
    encoded_input = tokenizer(sentences, truncation=True, padding='max_length', max_length=512, return_tensors='pt')

    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)

    # Perform pooling. In this case, max pooling.
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    return sentence_embeddings


# import torch
# similarity = torch.cosine_similarity(sentence_embeddings[0], sentence_embeddings[1], dim=0)
# print('similarity', similarity)


import faiss
import pandas as pd

df = pd.read_csv("../data/question.csv", encoding="utf-8", nrows=20)

questions = list(df['questions'])
sentence_embeddings = get_sentence_embedding(questions)

# bert向量维度获取
d = len(sentence_embeddings[0])
index = faiss.IndexFlatL2(d)
index.add(sentence_embeddings.numpy())
print(index.ntotal)

query = get_sentence_embedding("小儿孝喘怎么治疗")
k = 2
D, I = index.search(query.numpy(), k)

print(D)
print(I)
