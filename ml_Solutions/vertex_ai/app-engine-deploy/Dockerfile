FROM gcr.io/google-appengine/python

# Ref:
# * https://github.com/GoogleCloudPlatform/python-runtime/blob/8cdc91a88cd67501ee5190c934c786a7e91e13f1/README.md#kubernetes-engine--other-docker-hosts
# * https://github.com/GoogleCloudPlatform/python-runtime/blob/8cdc91a88cd67501ee5190c934c786a7e91e13f1/scripts/testdata/hello_world_golden/Dockerfile
LABEL python_version=python3
RUN virtualenv --no-download /env -p python3

ENV VIRTUAL_ENV /env
ENV PATH /env/bin:$PATH

ADD requirements.txt /app/
RUN pip install -r requirements.txt

ADD . /app
ENTRYPOINT [ "streamlit", "run", "main.py", "--server.port", "8080" ]