FROM python:3.11-slim

RUN apt-get update && apt-get -y upgrade \
    && pip install --upgrade pip \
    && pip --version

RUN apt-get update && apt-get install -y procps \
    && rm -rf /var/lib/apt/lists/*


RUN groupadd mlflow && useradd --create-home -g mlflow olonok
ENV PATH /home/olonok/.local/bin:${PATH}

WORKDIR /home/olonok/mlflow

COPY ./mlflow/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && rm requirements.txt
RUN mkdir /home/olonok/mlflow/mlruns

RUN chown -R olonok ./mlruns 
RUN chmod -R 777 /home/olonok

USER olonok

EXPOSE 5000

CMD mlflow server --backend-store-uri ${BACKEND_STORE_URI} --default-artifact-root ${DEFAULT_ARTIFACT_ROOT} --artifacts-destination ${DEFAULT_ARTIFACTS_DESTINATION} --no-serve-artifacts --host 0.0.0.0 --port 5000

