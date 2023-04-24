FROM stanfordnmbl/opensim-python:4.3

RUN add-apt-repository ppa:jonathonf/ffmpeg-4 ; apt update
RUN apt-get install ffmpeg -y --fix-missing

# cudnn install and environment vars
# still wasn't able to find gpu for tensorflow
# ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:/usr/local/cuda-11.4/targets/x86_64-linux/lib/:${LD_LIBRARY_PATH}
# ENV PATH=/usr/local/cuda-11.4/bin:${PATH}

# ENV OS=ubuntu2004
# ENV cudnn_version=8.2.4.15
# ENV cuda_version=cuda11.4

# RUN wget https://developer.download.nvidia.com/compute/cuda/repos/${OS}/x86_64/cuda-${OS}.pin

# RUN mv cuda-${OS}.pin /etc/apt/preferences.d/cuda-repository-pin-600
# RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/${OS}/x86_64/3bf863cc.pub
# RUN add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/${OS}/x86_64/ /"
# RUN apt-get update

# RUN apt-get install libcudnn8=${cudnn_version}-1+${cuda_version}
# RUN apt-get install libcudnn8-dev=${cudnn_version}-1+${cuda_version}

# install other packages
WORKDIR /workspace/
COPY requirements.txt /workspace/

RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

COPY . /workspace/

CMD python3.8 app.py
