FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-devel

COPY FAQ_vector_similarity /workspace

COPY sbert-base-chinese-nli /sbert-base-chinese-nli

COPY data /data

COPY cMedQNLI /cMedQNLI

COPY requirements.txt /workspace/requirements.txt

WORKDIR /workspace

RUN pip install -r /workspace/requirements.txt

ARG PORT=1997
ARG VERSION

ENV PORT=${PORT}

EXPOSE ${PORT}

ENTRYPOINT python index_as_server.py
