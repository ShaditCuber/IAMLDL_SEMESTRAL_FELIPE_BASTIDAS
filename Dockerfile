FROM ubuntu:22.04

RUN apt-get update && \
    apt-get install -y python3.10 python3-pip

WORKDIR /app


COPY ./original_models ./original_models
COPY Pipfile* ./
COPY index.py ./
COPY ./matrices_de_confusion_original ./matrices_de_confusion_original
COPY dataset.json ./

RUN pip3 install pipenv && \
    pipenv install --system --deploy --ignore-pipfile

