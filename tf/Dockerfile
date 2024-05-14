FROM python:3.11-slim


# Install additional packages
RUN apt-get -y update && \
         apt-get -y upgrade && \
         apt-get install -y python3-pip python3-dev  

RUN apt-get install -y git

# Install any python packages you need
COPY requirements.txt requirements.txt

RUN python3 -m pip install --upgrade pip && \
   python3 -m pip install -r requirements.txt
RUN python3 -m pip install tensorflow[and-cuda]

COPY . .
# alias
RUN echo 'alias pip="pip3" ' >> ~/.bashrc

CMD tail -f  /dev/null
