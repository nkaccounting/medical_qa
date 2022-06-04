# encoding=utf-8
import time

import torch
from transformers import BertForSequenceClassification, BertTokenizer

qnli_model_dir = './qnli'

qnli_model = BertForSequenceClassification.from_pretrained(qnli_model_dir)
qnli_tokenizers = BertTokenizer.from_pretrained(qnli_model_dir)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

qnli_model = qnli_model.to(device)

while True:
    sentence1 = input("问句：")
    sentence2 = input("回答：")

    t1 = time.time()
    paraphrase = qnli_tokenizers(sentence1, sentence2, return_tensors="pt")
    paraphrase = paraphrase.to(device)
    paraphrase_classification_logits = qnli_model(**paraphrase).logits
    paraphrase_results = torch.softmax(paraphrase_classification_logits, dim=1).tolist()[0]
    print(paraphrase_results)

    paraphrase_results = torch.argmax(paraphrase_classification_logits, dim=1).tolist()[0]
    print(paraphrase_results)

    t2 = time.time()

    print('推理用时', t2 - t1)
