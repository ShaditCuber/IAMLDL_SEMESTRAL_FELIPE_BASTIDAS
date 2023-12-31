FROM python:3.10-slim

WORKDIR /app


COPY models/ ./
COPY Pipfile* ./
COPY index.py ./
COPY matrices_de_consusion/ ./

RUN pip install pipenv && \
    pipenv install --system --deploy --ignore-pipfile

