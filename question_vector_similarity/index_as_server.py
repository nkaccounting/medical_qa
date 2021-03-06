import os
import time
from collections import defaultdict

import faiss
import pandas as pd
import torch
from flask import Flask, request
from transformers import AutoTokenizer, AutoModel, QuestionAnsweringPipeline, BertForQuestionAnswering
from transformers import BertForSequenceClassification, BertTokenizer

encode_model_dir = "../sbert-base-chinese-nli"
qnli_model_dir = '../cMedQNLI/qnli'
mrc_model_dir = "../chinese_pretrain_mrc_roberta_wwm_ext_large"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load model from HuggingFace Hub
encode_tokenizer = AutoTokenizer.from_pretrained(encode_model_dir)
encode_model = AutoModel.from_pretrained(encode_model_dir)

qnli_model = BertForSequenceClassification.from_pretrained(qnli_model_dir)
qnli_model = qnli_model.to(device)
qnli_tokenizers = BertTokenizer.from_pretrained(qnli_model_dir)

mrc_tokenizer = AutoTokenizer.from_pretrained(mrc_model_dir)
mrc_model = BertForQuestionAnswering.from_pretrained(mrc_model_dir)
if torch.cuda.is_available():
    mrc_pipeline = QuestionAnsweringPipeline(model=mrc_model, tokenizer=mrc_tokenizer, device=0)
else:
    mrc_pipeline = QuestionAnsweringPipeline(model=mrc_model, tokenizer=mrc_tokenizer)


# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


# Sentences we want sentence embeddings for
def get_sentence_embedding(sentences: str):
    # Tokenize sentences
    encoded_input = encode_tokenizer(sentences, truncation=True, padding='max_length', max_length=512,
                                     return_tensors='pt')

    # Compute token embeddings
    with torch.no_grad():
        model_output = encode_model(**encoded_input)

    # Perform pooling. In this case, max pooling.
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    return sentence_embeddings


def search_one_query(question, index, top_k):
    t1 = time.time()
    query = get_sentence_embedding(question)
    t2 = time.time()
    D, I = index.search(query.numpy(), top_k)
    t3 = time.time()
    print("???????????????????????????", t2 - t1)
    print("?????????????????????", t3 - t2)
    return D, I


def isQApair(question, answer):
    res = []
    scores = []
    t1 = time.time()
    # ?????????batch???qa pair????????????
    batch = list(zip(question, answer))
    paraphrase = qnli_tokenizers(batch, truncation=True, padding='max_length', max_length=512,
                                 return_tensors="pt")
    paraphrase = paraphrase.to(device)
    paraphrase_classification_logits = qnli_model(**paraphrase).logits
    paraphrase_score_results = torch.softmax(paraphrase_classification_logits, dim=1).tolist()
    for item in paraphrase_score_results:
        if item[0] >= item[1]:
            res.append(0)
            scores.append(item[0])
        else:
            res.append(1)
            scores.append(item[1])
    t2 = time.time()
    print("????????????QApair???????????????", (t2 - t1) / len(question))
    print("???????????????QApair?????????", t2 - t1)
    return res, scores


t1 = time.time()
index = faiss.read_index("./compression_index.bin")
t2 = time.time()
print("??????????????????faiss?????????????????????", t2 - t1)

t1 = time.time()
df = pd.read_csv("../data/questions.csv", encoding="utf-8")
questions = [question for question in df["questions"]]
t2 = time.time()
print("???????????????question??????????????????????????????", t2 - t1)

t1 = time.time()
df = pd.read_csv("../data/answers.csv", encoding="utf-8")
answers = [answer for answer in df["answers"]]
t2 = time.time()
print("???????????????answer??????????????????????????????", t2 - t1)

t1 = time.time()
df = pd.read_csv("../data/simple_answers.csv", encoding="utf-8")
simple_answers = [simple_answer for simple_answer in df["simple_answers"]]
t2 = time.time()
print("???????????????answer??????????????????????????????", t2 - t1)

PORT = os.getenv('PORT', 2265)

app = Flask(__name__)
USE_ASCII = False
app.config['JSON_AS_ASCII'] = USE_ASCII


@app.route('/qa', methods=['GET'])
def re():
    text = request.args.get('text', None)
    # === Error Detection ===

    if text is None:
        response = app.response_class(
            response='The "text" argument is a must have.',
            status=500,
            mimetype='text/plain'
        )
        return response

    res, mrc_answer = one_question(text)

    response = {
        'question': text,
        "information": res,
        "mrc_answer": mrc_answer
    }

    return response


def get_sum_answer(mrc_res):
    table = defaultdict(float)
    max_value = 0
    max_key = ""
    for item in mrc_res:
        table[item["answer"]] += item['score']
        if table[item["answer"]] > max_value:
            max_value = table[item["answer"]]
            max_key = item["answer"]
    return max_key


def one_question(text: str, not_use_qnli=False):
    top_k = 5
    result = []
    D, I = (search_one_query(text, index, top_k))

    res = [answers[id] for id in I[0]]

    mrc_res = mrc_pipeline(
        question=[text] * top_k,
        context=res,
    )

    sum_answer = get_sum_answer(mrc_res)

    isqa, scores = isQApair([text] * top_k, res)

    no_answer = 0

    for i, id in enumerate(I[0]):
        if not_use_qnli or isqa[i]:
            one_answer = {
                "????????????": questions[id],
                "?????????????????????": str(D[0][i]),
                '?????????': scores[i],
                '????????????': answers[id],
                # "??????????????????": simple_answers[id]
            }
            result.append(one_answer)
        else:
            no_answer += 1
            if no_answer == top_k:
                result.append({
                    'no_answer': "??????????????????????????????????????????????????????????????????????????????????????????????????????~"
                })
    return result, sum_answer


if __name__ == '__main__':
    app.run('0.0.0.0', port=PORT)
