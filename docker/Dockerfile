FROM stanfordnmbl/opensim-python:4.3

RUN add-apt-repository ppa:jonathonf/ffmpeg-4 ; apt update
RUN apt-get install ffmpeg -y --fix-missing

WORKDIR /workspace/
COPY requirements.txt /workspace/

RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

COPY . /workspace/

CMD python3.8 app.py
