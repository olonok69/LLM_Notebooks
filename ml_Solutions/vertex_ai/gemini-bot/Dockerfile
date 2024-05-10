FROM ubuntu:22.04

# Set environment variables
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Minsk

# Install necessary packages
RUN apt update && \
    apt install -y software-properties-common wget && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update -y && \
    apt install -y python3.11 python3-pip python3-venv virtualenv libpoppler-dev poppler-utils

WORKDIR /gemini
# Ref:
# * https://github.com/GoogleCloudPlatform/python-runtime/blob/8cdc91a88cd67501ee5190c934c786a7e91e13f1/README.md#kubernetes-engine--other-docker-hosts
# * https://github.com/GoogleCloudPlatform/python-runtime/blob/8cdc91a88cd67501ee5190c934c786a7e91e13f1/scripts/testdata/hello_world_golden/Dockerfile
LABEL python_version=python3
RUN virtualenv  /env -p python3.11

ENV VIRTUAL_ENV /env
ENV PATH /env/bin:$PATH
COPY ./requirements.txt /gemini/requirements.txt


RUN pip install -r /gemini/requirements.txt
COPY . .

ENTRYPOINT [ "streamlit", "run", "pages.py", "--server.port", "8080" ]