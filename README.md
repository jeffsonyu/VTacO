# Visual-Tactile Sensing for In-Hand Object Reconstruction
[**Paper**](https://arxiv.org/pdf/2303.14498.pdf) | [**Project Page**](https://sites.google.com/view/vtaco) <br>

<div style="text-align: center">
<img src="media/VTacO.png" width="1000"/>
</div>

This repository contains the implementation of the paper:

**Visual-Tactile Sensing for In-Hand Object Reconstruction**  
Wenqiang Xu*, Zhenjun Yu*, Han Xue, Ruolin Ye, Siqiong Yao, Cewu Lu (* = Equal contribution)  
**CVPR 2023**  

If you find our code or paper useful, please consider citing
```bibtex
@inproceedings{xu2023visual,
  title={Visual-Tactile Sensing for In-Hand Object Reconstruction},
  author={Xu, Wenqiang and Yu, Zhenjun and Xue, Han and Ye, Ruolin and Yao, Siqiong and Lu, Cewu},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={8803--8812},
  year={2023}
}

```

## Get-Started

The code is only tested on Ubuntu, we will soon test it on Windows system. 

## With conda and pip

Install [anaconda](https://www.anaconda.com/) or [miniconda](https://docs.conda.io/en/latest/miniconda.html). Supposing that the name `vtaco` is used for conda environment:

```shell
conda create -y -n vtaco python=3.6.7
conda activate vtaco
```

Then, install dependencies with `pip install`

```shell
pip install -r requirements.txt
```

Next, compile the extension modules.
You can do this via

```shell
python setup.py build_ext --inplace
```

## With Docker

Install Docker under the [instructions](https://docs.docker.com/get-started/). Supposing hat the tag `vtaco-train` is used for docker image:

```shell
docker build -t vtaco-train -f ./manifests/Dockerfile .
```

To start a develop container, run

```shell
docker run --ipc=host --rm -it -p 8888:8888 vtaco-train
```

This will launch a jupyterlab server inside the container. The server can be accessed via port `8888`.

If the Docker installation is configured with Nvidia's GPU support, an additional `--gpus all` flag can be passed

```shell
docker run --ipc=host --gpus all --rm -it -p 8888:8888 vtaco-train
```

To mount the dataset, add an additional `--volume` mapping. 

```shell
docker run --ipc=host --gpus all --rm -it -p 8888:8888 --volume <path/to/dataset>:/opt/vtaco/data vtaco-train
```

**Note**: The `<path/to/dataset>` should be replaced by actual path on the host system.

## Dataset
We are uploading the dataset, which will be available on https://huggingface.co/datasets/robotflow/vtaco/  
You can follow the instructions to download the dataset for training and testing dataset for VTacO and VTacOH.

## VT-Sim
The VT-Sim has been released [here](https://github.com/jeffsonyu/VT-Sim)

## Training
To train the Depth Estimator $U_I(\cdot)$ and the sensor pose estimator, we provide a config file `configs/tactile/tactile_test.yaml`, you can run the following command to train from scratch:
```
python train_depth.py configs/tactile/tactile_test.yaml
```

With the pretrained model of $U_I(\cdot)$ and the sensor pose estimator, examples for training VTacO or VTacOH are as follows:

```shell
python train.py configs/VTacO/VTacO_YCB.yaml
python train.py configs/VTacOH/VTacOH_YCB.yaml
```

**Note**: you might need to change *path* in *data*, and *model_file* in *encoder_t2d_kwargs* of the config file, to your data path and pretrained model path. This path is `out/tactile/test/model_best.pt` by default.

All the results will be saved in `out/` folder, including checkpoints, visualization results and logs for tensorboard.
