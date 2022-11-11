FROM stanfordnmbl/openpose-gpu:11.3.1

WORKDIR /workspace_openpose/
COPY requirements.txt /workspace_openpose/requirements.txt

RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

COPY . /workspace_openpose/
CMD python3.6 loop.py
