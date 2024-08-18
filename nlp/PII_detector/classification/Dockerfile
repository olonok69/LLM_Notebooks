FROM python:3.11-slim

RUN apt-get update && apt-get install -y  libsm6 libxext6 net-tools  git nano procps libmagic1

# Create and activate virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"


ENV PORT=5000

WORKDIR /classification

COPY requirements.txt requirements.txt

RUN pip install --upgrade pip

RUN pip install --no-cache-dir --prefer-binary -r requirements.txt \
    && pip install --no-cache-dir --prefer-binary https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-3.6.0/en_core_web_lg-3.6.0-py3-none-any.whl \
    && pip install --no-cache-dir --prefer-binary https://github.com/explosion/spacy-models/releases/download/xx_ent_wiki_sm-3.6.0/xx_ent_wiki_sm-3.6.0-py3-none-any.whl \
    && pip install --no-cache-dir "presidio_analyzer==2.2.355" "presidio_anonymizer==2.2.355" \
    && pip install --no-cache-dir uvicorn \
    && pip install --no-cache-dir --upgrade detectaicore

RUN python3 -m nltk.downloader punkt
RUN python3 -m nltk.downloader wordnet
RUN python3 -m nltk.downloader stopwords

COPY . .

RUN rm -rf notebooks archive models out output out_semantic __pycache__ .ipynb_checkpoints requirements.txt *.ipynb 
RUN mkdir output

EXPOSE ${PORT}
#CMD ["uvicorn", "endpoint_classification:endpoint", "--host", "0.0.0.0", "--port",  ${PORT}, "--reload"]
CMD ["sh", "-c", "uvicorn endpoint_classification:endpoint --host 0.0.0.0 --port $PORT --reload"]