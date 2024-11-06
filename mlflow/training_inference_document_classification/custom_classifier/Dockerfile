# Use an official Python runtime as a parent image
FROM python:3.11-slim

RUN apt-get update && \
    apt-get install -y python3-pip  ffmpeg libsm6 libxext6 git net-tools  python3-magic nano iputils-ping procps && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*
# Set the working directory to /app
WORKDIR /custom_classifier

# Copy the requirements file into the image
COPY ./requirements.txt /custom_classifier/requirements.txt
# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt

# Copy every content from the local folder to the image
COPY . .

EXPOSE 5020
# Run server
CMD ["uvicorn", "endpoint_custom_classifier:endpoint", "--host=0.0.0.0", "--port=5020"]
