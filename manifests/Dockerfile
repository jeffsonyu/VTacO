FROM nvidia/cuda:11.6.2-base-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

# Optional: change mirror
RUN sed -i 's/http:\/\/archive.ubuntu.com/https:\/\/mirror.sjtu.edu.cn/g' /etc/apt/sources.list

RUN apt-get update && \
    apt-get install -y python3.8 python3-pip git vim htop libsm6 libxext6 libxrender-dev libgl1-mesa-glx libglib2.0-dev && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1 && \
    pip install --upgrade pip

COPY requirements.txt /tmp/requirements.txt

# Optional: change pip mirror
RUN pip config set global.index-url https://mirror.sjtu.edu.cn/pypi/web/simple

# Install required packages
RUN pip install -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt

WORKDIR /opt/vtaco
COPY . /opt/vtaco/

ENV JUPYTER_ALLOW_ROOT=1
ENV SHELL=/bin/bash

CMD ["jupyter-lab",  "--ip=0.0.0.0", "--allow-root" ]