from src.encoder import (
    pointnet, voxels, pointnetpp, manolayer
)

from src.layers import (
    Resnet18, Resnet34, Resnet50, UNet
)

from src.TransformerFusion import TransformerFusion

encoder_dict = {
    'pointnet_local_pool': pointnet.LocalPoolPointnet,
    'pointnet_crop_local_pool': pointnet.PatchLocalPoolPointnet,
    'pointnet_plus_plus': pointnetpp.PointNetPlusPlus,
    'voxel_simple_local': voxels.LocalVoxelEncoder,
    'Resnet18': Resnet18,
    'Resnet34': Resnet34,
    'Resnet50': Resnet50,
    'UNet': UNet
}
