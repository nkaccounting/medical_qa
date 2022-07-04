FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-devel

COPY question_vector_similarity /workspace

COPY sbert-base-chinese-nli /sbert-base-chinese-nli

COPY cMedQNLI/qnli /cMedQNLI/qnli

COPY chinese_pretrain_mrc_roberta_wwm_ext_large /chinese_pretrain_mrc_roberta_wwm_ext_large

COPY data/answers.csv /data/answers.csv

COPY data/questions.csv /data/questions.csv

COPY requirements.txt /workspace/requirements.txt

WORKDIR /workspace

RUN pip install -r /workspace/requirements.txt

ARG PORT=1997
ARG VERSION

ENV PORT=${PORT}

EXPOSE ${PORT}

ENTRYPOINT python index_as_server.py
