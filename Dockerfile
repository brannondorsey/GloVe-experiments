FROM python:3.7-slim-stretch

RUN apt-get update && \
    apt-get install -y --no-install-recommends curl unzip git build-essential && \
    git clone https://github.com/brannondorsey/glove-experiments && \
    cd glove-experiments && \
    pip install -r requirements.txt && \
    ./download_data.sh

WORKDIR /glove-experiments
CMD python word_arithmetic.py
