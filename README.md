# Visual-Tactile Sensing for In-Hand Object Reconstruction
[**Paper**] | [**Project Page**](https://sites.google.com/view/vtaco) <br>

<div style="text-align: center">
<img src="media/VTacO.png" width="1000"/>
</div>

This repository contains the implementation of the paper:

**Visual-Tactile Sensing for In-Hand Object Reconstruction**  
Wenqiang Xu*, Zhenjun Yu*, Han Xue, Ruolin Ye, Siqiong Yao, Cewu Lu (* = Equal contribution)  
**CVPR 2023**  

## Installation
First you have to make sure that you have all dependencies in place.
The simplest way to do so, is to use [anaconda](https://www.anaconda.com/). 

You can create an anaconda environment called `vtaco` using
```
conda env create -f environment.yaml
conda activate vtaco
```
**Note**: you might need to install **torch-scatter** mannually following [the official instruction](https://github.com/rusty1s/pytorch_scatter#pytorch-140):
```
pip install torch-scatter==2.0.4 -f https://pytorch-geometric.com/whl/torch-1.4.0+cu101.html
```

Next, compile the extension modules.
You can do this via
```
python setup.py build_ext --inplace
```

## Dataset and VT-Sim
<!-- For downloading the training and testing dataset for VTacO and VTacOH, you can simply run the following command to download our preprocessed dataset:

```
bash scripts/download_data.sh
```

This script should download and unpack the data automatically into the `data/` folder, which should look like:
```
VTacO
├── data
│   ├── VTacO_AKB_class
    │   │   │── 001
    │   │   │   |── $class_name
    │   │   │   |── metadata.yaml
    │   │   │── 002
    │   │   │── ...
    │   │   │── 007
    ├── VTacO_YCB
    │   │   │── YCB
    │   │   │── metadata.yaml
    ├── VTacO_mesh
    │   │   │── mesh
    │   │   │── mesh_obj
    │   │   │── depth_origin.txt
``` -->
We will soon release the dataset and VT-Sim!

## Training
To train the Depth Estimator $U_I(\cdot)$ and the sensor pose estimator, we provide a config file `configs/tactile/tactile_test.yaml`, you can run the following command to train from scratch:
```
python train_depth.py configs/tactile/tactile_test.yaml
```

With the pretrained model of $U_I(\cdot)$ and the sensor pose estimator, examples for training VTacO or VTacOH are as follows: 
```
python train.py configs/VTacO/VTacO_AKB_001.yaml
python train.py configs/VTacOH/VTacOH_AKB_001.yaml
```
**Note**: you might need to change *path* in *data*, and *model_file* in *encoder_t2d_kwargs* of the config file, to your data path and pretrained model path.  

All the results will be saved in `out/` folder, including checkpoints, visualization results and logs for tensorboard.
