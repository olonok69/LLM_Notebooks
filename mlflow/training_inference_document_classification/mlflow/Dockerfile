FROM python:3.11-slim

ARG mlflow_user

# Use the build argument to set an environment variable
ENV USER_MLFLOW=${mlflow_user}

# You can now use the USER_MLFLOW environment variable in your Dockerfile
RUN echo "Mlflow user is: ${USER_MLFLOW}"

RUN apt-get update && apt-get -y upgrade \
    && apt-get install -y libsm6 libxext6 git net-tools  python3-magic nano iputils-ping procps \
    && pip install --upgrade pip \
    && pip --version

RUN apt-get update && apt-get install -y procps \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd ${USER_MLFLOW} && useradd --create-home -g ${USER_MLFLOW} ${USER_MLFLOW}
ENV PATH /home/${USER_MLFLOW}/.local/bin:${PATH}

WORKDIR /home/${USER_MLFLOW}/mlflow

COPY ./mlflow/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt \
    && rm requirements.txt
RUN mkdir /home/${USER_MLFLOW}/mlflow/mlruns

RUN chown -R ${USER_MLFLOW}:${USER_MLFLOW} /home/${USER_MLFLOW}/mlflow 
RUN chown -R ${USER_MLFLOW}:${USER_MLFLOW} /home/${USER_MLFLOW}/mlflow/mlruns 
RUN chmod -R 777 /home/${USER_MLFLOW}

USER ${USER_MLFLOW}

EXPOSE 5000

CMD mlflow server --backend-store-uri ${BACKEND_STORE_URI} --default-artifact-root ${DEFAULT_ARTIFACT_ROOT} --artifacts-destination ${DEFAULT_ARTIFACTS_DESTINATION} --no-serve-artifacts --host 0.0.0.0 --port 5000