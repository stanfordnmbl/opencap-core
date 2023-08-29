FROM stanfordnmbl/openpose-gpu:11.3.1

COPY requirements.txt /openpose

RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

COPY loop.py /openpose
COPY ../../defaultOpenCapSettings.json /openpose
CMD python3.6 loop.py

