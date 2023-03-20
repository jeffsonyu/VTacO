import torch
import torch.distributions as dist
from torch import nn
import os
from src.encoder import encoder_dict
from src.conv_onet import models, training, inferencing
from src.conv_onet import generation
from src.checkpoints import CheckpointIO
from src import data
from src import config
from src.common import decide_total_volume_range, update_reso
from torchvision import transforms
import numpy as np


def get_model(cfg, device=None, dataset=None, **kwargs):
    ''' Return the Occupancy Network model.

    Args:
        cfg (dict): imported yaml config 
        device (device): pytorch device
        dataset (dataset): dataset
    '''
    decoder = cfg['model']['decoder']
    encoder = cfg['model']['encoder']

    encoder_hand = cfg['model']['encoder_hand']
    dim = cfg['data']['dim']
    c_dim = cfg['model']['c_dim']
    decoder_kwargs = cfg['model']['decoder_kwargs']
    encoder_kwargs = cfg['model']['encoder_kwargs']
    encoder_hand_kwargs = cfg['model']['encoder_hand_kwargs']
    
    encoder_img = cfg['model']['encoder_img']
    encoder_img_kwargs = cfg['model']['encoder_img_kwargs']
    
    encoder_t2d = cfg['model']['encoder_t2d']
    encoder_t2d_kwargs = cfg['model']['encoder_t2d_kwargs']
    
    padding = cfg['data']['padding']
    with_img = cfg['model']['with_img']
    with_contact = cfg['model']['with_contact']
    
    # for pointcloud_crop
    try: 
        encoder_kwargs['unit_size'] = cfg['data']['unit_size']
        encoder_hand_kwargs['unit_size'] = cfg['data']['unit_size']
        decoder_kwargs['unit_size'] = cfg['data']['unit_size']
    except:
        pass
    # local positional encoding
    if 'local_coord' in cfg['model'].keys():
        encoder_kwargs['local_coord'] = cfg['model']['local_coord']
        encoder_hand_kwargs['local_coord'] = cfg['model']['local_coord']
        decoder_kwargs['local_coord'] = cfg['model']['local_coord']
    if 'pos_encoding' in cfg['model']:
        encoder_kwargs['pos_encoding'] = cfg['model']['pos_encoding']
        encoder_hand_kwargs['pos_encoding'] = cfg['model']['pos_encoding']
        decoder_kwargs['pos_encoding'] = cfg['model']['pos_encoding']

    # update the feature volume/plane resolution
    if cfg['data']['input_type'] == 'pointcloud_crop':
        fea_type = cfg['model']['encoder_kwargs']['plane_type']
        if (dataset.split == 'train') or (cfg['generation']['sliding_window']):
            recep_field = 2**(cfg['model']['encoder_kwargs']['unet3d_kwargs']['num_levels'] + 2)
            reso = cfg['data']['query_vol_size'] + recep_field - 1
            if 'grid' in fea_type:
                encoder_kwargs['grid_resolution'] = update_reso(reso, dataset.depth)
            if bool(set(fea_type) & set(['xz', 'xy', 'yz'])):
                encoder_kwargs['plane_resolution'] = update_reso(reso, dataset.depth)
        # if dataset.split == 'val': #TODO run validation in room level during training
        else:
            if 'grid' in fea_type:
                encoder_kwargs['grid_resolution'] = dataset.total_reso
            if bool(set(fea_type) & set(['xz', 'xy', 'yz'])):
                encoder_kwargs['plane_resolution'] = dataset.total_reso
    

    if decoder == False:
        decoder = None
    else:
        decoder = models.decoder_dict[decoder](
            dim=dim, c_dim=c_dim, padding=padding, with_contact=with_contact,
            **decoder_kwargs, 
        )

    if encoder == 'idx':
        encoder = nn.Embedding(len(dataset), c_dim)
    elif encoder != False:
        encoder = encoder_dict[encoder](
            dim=dim, c_dim=c_dim, padding=padding,
            **encoder_kwargs
        )
    else:
        encoder = None
        

    if encoder_hand != False:
        encoder_hand = encoder_dict[encoder_hand](
            dim=dim, c_dim=c_dim, padding=padding,
            **encoder_hand_kwargs
        )
    else:
        encoder_hand = None
        
    if with_img and (encoder_img != False):
        encoder_img = encoder_dict[encoder_img](
            **encoder_img_kwargs
        )
    else:
        encoder_img = None
    
    if encoder_t2d != False:
        encoder_img_t2d = encoder_t2d_kwargs['encoder_img']
        encoder_img_t2d_kwargs = encoder_t2d_kwargs['encoder_img_kwargs']
        encoder_img_t2d = encoder_dict[encoder_img_t2d](
            **encoder_img_t2d_kwargs
        )
        
        encoder_d_t2d = encoder_t2d_kwargs['encoder_hand']
        encoder_d_t2d_kwargs = encoder_t2d_kwargs['encoder_hand_kwargs']
        encoder_d_t2d = encoder_dict[encoder_d_t2d](
            dim=dim, padding=padding,
            **encoder_d_t2d_kwargs
        )
        
        encoder_t2d = models.ConvolutionalOccupancyNetwork(
            None, None, encoder_d_t2d, encoder_img_t2d, None, device=device
        )
        
        if encoder_t2d_kwargs['pretrained']:
            checkpoint_io = CheckpointIO(cfg['training']['out_dir'], model=encoder_t2d)
            load_dict = checkpoint_io.load(encoder_t2d_kwargs['model_file'], device=device)
        
    else:
        encoder_t2d = None
        

    model = models.ConvolutionalOccupancyNetwork(
        decoder, encoder, encoder_hand, encoder_img, encoder_t2d, device=device
    )
        
    return model
    

def get_trainer(model, optimizer, cfg, device, **kwargs):
    ''' Returns the trainer object.

    Args:
        model (nn.Module): the Occupancy Network model
        optimizer (optimizer): pytorch optimizer object
        cfg (dict): imported yaml config
        device (device): pytorch device
    '''
    threshold = cfg['test']['threshold']
    out_dir = cfg['training']['out_dir']
    vis_dir = os.path.join(out_dir, 'vis')
    input_type = cfg['data']['input_type']
    with_img = cfg['model']['with_img']
    with_contact = cfg['model']['with_contact']
    train_tactile = cfg['model']['train_tactile']
    encode_t2d = cfg['model']['encoder_t2d']
    try:
        pretrained_t2d = cfg['model']['encoder_t2d_kwargs']['pretrained']
    except:
        pretrained_t2d = False

    trainer = training.Trainer(
        model, optimizer,
        device=device, input_type=input_type,
        vis_dir=vis_dir, threshold=threshold,
        eval_sample=cfg['training']['eval_sample'],
        num_sample=cfg['data']['num_sample'],
        with_img=with_img,
        with_contact=with_contact,
        train_tactile=train_tactile,
        encode_t2d=encode_t2d,
        pretrained_t2d=pretrained_t2d
    )

    return trainer

def get_inferencer(model, optimizer, generator, cfg, device, **kwargs):
    ''' Returns the trainer object.

    Args:
        model (nn.Module): the Occupancy Network model
        optimizer (optimizer): pytorch optimizer object
        cfg (dict): imported yaml config
        device (device): pytorch device
    '''
    threshold = cfg['test']['threshold']
    out_dir = cfg['training']['out_dir']
    vis_dir = os.path.join(out_dir, 'vis')
    input_type = cfg['data']['input_type']
    with_img = cfg['model']['with_img']
    with_contact = cfg['model']['with_contact']
    train_tactile = cfg['model']['train_tactile']
    encode_t2d = cfg['model']['encoder_t2d']

    inferencer = inferencing.Inferencer(
        model, optimizer, generator,
        device=device, input_type=input_type,
        vis_dir=vis_dir, threshold=threshold,
        eval_sample=cfg['training']['eval_sample'],
        num_sample=cfg['data']['num_sample'],
        with_img=with_img,
        with_contact=with_contact,
        train_tactile=train_tactile,
        encode_t2d=encode_t2d
    )

    return inferencer

def get_generator(model, cfg, device, **kwargs):
    ''' Returns the generator object.

    Args:
        model (nn.Module): Occupancy Network model
        cfg (dict): imported yaml config
        device (device): pytorch device
    '''
    
    if cfg['data']['input_type'] == 'pointcloud_crop':
        # calculate the volume boundary
        query_vol_metric = cfg['data']['padding'] + 1
        unit_size = cfg['data']['unit_size']
        recep_field = 2**(cfg['model']['encoder_kwargs']['unet3d_kwargs']['num_levels'] + 2)
        if 'unet' in cfg['model']['encoder_kwargs']:
            depth = cfg['model']['encoder_kwargs']['unet_kwargs']['depth']
        elif 'unet3d' in cfg['model']['encoder_kwargs']:
            depth = cfg['model']['encoder_kwargs']['unet3d_kwargs']['num_levels']
        
        vol_info = decide_total_volume_range(query_vol_metric, recep_field, unit_size, depth)
        
        grid_reso = cfg['data']['query_vol_size'] + recep_field - 1
        grid_reso = update_reso(grid_reso, depth)
        query_vol_size = cfg['data']['query_vol_size'] * unit_size
        input_vol_size = grid_reso * unit_size
        # only for the sliding window case
        vol_bound = None
        if cfg['generation']['sliding_window']:
            vol_bound = {'query_crop_size': query_vol_size,
                         'input_crop_size': input_vol_size,
                         'fea_type': cfg['model']['encoder_kwargs']['plane_type'],
                         'reso': grid_reso}

    else: 
        vol_bound = None
        vol_info = None

    generator = generation.Generator3D(
        model,
        device=device,
        threshold=cfg['test']['threshold'],
        resolution0=cfg['generation']['resolution_0'],
        upsampling_steps=cfg['generation']['upsampling_steps'],
        sample=cfg['generation']['use_sampling'],
        refinement_step=cfg['generation']['refinement_step'],
        simplify_nfaces=cfg['generation']['simplify_nfaces'],
        input_type = cfg['data']['input_type'],
        padding=cfg['data']['padding'],
        vol_info = vol_info,
        vol_bound = vol_bound,
        alpha = cfg['generation']['alpha'],
        with_img = cfg['model']['with_img'],
        encode_t2d = cfg['model']['encoder_t2d']
    )
    return generator


def get_data_fields(mode, cfg):
    ''' Returns the data fields.

    Args:
        mode (str): the mode which is used
        cfg (dict): imported yaml config
    '''
    points_transform = data.SubsamplePoints(cfg['data']['points_subsample'])
    
    input_type = cfg['data']['input_type']
    fields = {}
    if cfg['data']['points_file'] is not None:
        if input_type != 'pointcloud_crop':
            fields['points'] = data.PointsField(
                cfg['data']['points_file'], points_transform,
                unpackbits=cfg['data']['points_unpackbits'],
                multi_files=cfg['data']['multi_files']
            )
        else:
            fields['points'] = data.PatchPointsField(
                cfg['data']['points_file'], 
                transform=points_transform,
                unpackbits=cfg['data']['points_unpackbits'],
                multi_files=cfg['data']['multi_files']
            )

    
    if mode in ('val', 'test', 'vis'):
        points_iou_file = cfg['data']['points_iou_file']
        voxels_file = cfg['data']['voxels_file']
        if points_iou_file is not None:
            if input_type == 'pointcloud_crop':
                fields['points_iou'] = data.PatchPointsField(
                points_iou_file,
                unpackbits=cfg['data']['points_unpackbits'],
                multi_files=cfg['data']['multi_files']
                )
            else:
                fields['points_iou'] = data.PointsField(
                    points_iou_file,
                    unpackbits=cfg['data']['points_unpackbits'],
                    multi_files=cfg['data']['multi_files']
                )
        if voxels_file is not None:
            fields['voxels'] = data.VoxelsField(voxels_file)

    return fields
