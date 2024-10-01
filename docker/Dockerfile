FROM stanfordnmbl/opensim-python:4.3

# We used to do this locally until 12/2/23...but there isn't a Release file for ubuntu 20.04.
# RUN add-apt-repository ppa:jonathonf/ffmpeg-4 ; apt update
# RUN apt-get install ffmpeg -y --fix-missing

RUN apt update
RUN apt install ffmpeg -y

WORKDIR /workspace/
COPY requirements.txt /workspace/

RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

COPY . /workspace/

ARG GIT_COMMIT_HASH
ENV GIT_COMMIT_HASH=${GIT_COMMIT_HASH}

CMD python3.8 app.py
