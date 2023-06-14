FROM nvidia/cuda:10.1-base-ubuntu18.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y python3.6 python3-pip git vim htop libsm6 libxext6 libxrender-dev && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1 && \
    pip install --upgrade pip

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt

WORKDIR /opt/vtaco
COPY . /opt/vtaco/
RUN python setup.py build_ext --inplace

ENV JUPYTER_ALLOW_ROOT=1

CMD ["jupyter-lab",  "--ip=0.0.0.0",  "--ip=0.0.0.0", "--allow-root" ]