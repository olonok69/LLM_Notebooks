# https://docs.docker.com/guides/docker-concepts/building-images/writing-a-dockerfile/
FROM python:3.12
WORKDIR /usr/local/app

# Install the application dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy in the source code
COPY . .
EXPOSE 5020

# Setup an app user so the container doesn't run as the root user
RUN useradd app
USER app

CMD ["uvicorn", "app:app", "--host=0.0.0.0", "--port=5020"]