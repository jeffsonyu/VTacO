# Capturing forceful interaction with arbitrary objects using a deep learning-powered stretchable tactile array
This repository contains the implementation of the paper:

**Capturing forceful interaction with arbitrary objects using a deep learning-powered stretchable tactile array**



## Get-Started

The code is only tested on Ubuntu, we will soon test it on Windows system. 

## With conda and pip

Install [anaconda](https://www.anaconda.com/) or [miniconda](https://docs.conda.io/en/latest/miniconda.html). Supposing that the name `vtaco` is used for conda environment:

```shell
conda create -y -n vitam python=3.8
conda activate vitam
```

Then, install dependencies with `pip install`

```shell
pip install -r requirements.txt
```

## With Docker

Install Docker under the [instructions](https://docs.docker.com/get-started/). Supposing hat the tag `vitam-train` is used for docker image:

```shell
docker build -t vitam-train -f ./manifests/Dockerfile .
```

To start a develop container, run

```shell
docker run --ipc=host --rm -it -p 8888:8888 vitam-train
```

This will launch a jupyterlab server inside the container. The server can be accessed via port `8888`.

If the Docker installation is configured with Nvidia's GPU support, an additional `--gpus all` flag can be passed

```shell
docker run --ipc=host --gpus all --rm -it -p 8888:8888 vitam-train
```

To mount the dataset, add an additional `--volume` mapping. 

```shell
docker run --ipc=host --gpus all --rm -it -p 8888:8888 --volume <path/to/dataset>:/opt/vitam/data vitam-train
```

**Note**: The `<path/to/dataset>` should be replaced by actual path on the host system.


## Training
Example for training ViTaM:

```shell
python train.py configs/vitam.yaml
```

All the results will be saved in `out/` folder, including checkpoints, visualization results and logs for tensorboard.
