import torch
import torch.optim as optim
from tensorboardX import SummaryWriter
import numpy as np
import os
import argparse
import time, datetime
import matplotlib; matplotlib.use('Agg')
from src import config, data
from src.checkpoints import CheckpointIO
from collections import defaultdict
import shutil
from tqdm import tqdm



# Arguments
parser = argparse.ArgumentParser(
    description='Train a 3D reconstruction model.'
)
parser.add_argument('config', type=str, help='Path to config file.')
parser.add_argument('--exit-after', type=int, default=-1,
                    help='Checkpoint and exit after specified number of seconds'
                         'with exit code 2.')

args = parser.parse_args()
cfg = config.load_config(args.config, 'configs/default.yaml')
device = torch.device("cuda:{}".format(cfg['training']['gpu']))
print("Training on gpu %d" % cfg['training']['gpu'])

# Set t0
t0 = time.time()

# Shorthands
out_dir = cfg['training']['out_dir']
batch_size = cfg['training']['batch_size']
backup_every = cfg['training']['backup_every']
# vis_n_outputs = cfg['generation']['vis_n_outputs']
exit_after = args.exit_after
if exit_after > 0:
    print("exit_after: %ds" % exit_after)

model_selection_metric = cfg['training']['model_selection_metric']
if cfg['training']['model_selection_mode'] == 'maximize':
    model_selection_sign = 1
elif cfg['training']['model_selection_mode'] == 'minimize':
    model_selection_sign = -1
else:
    raise ValueError('model_selection_mode must be '
                     'either maximize or minimize.')

# Output directory
if not os.path.exists(out_dir): os.makedirs(out_dir)

shutil.copyfile(args.config, os.path.join(out_dir, 'config.yaml'))



# Dataset
val_dataset = config.get_dataset('val', cfg, return_idx=True)
vis_name_list = val_dataset.models

# For visualizations
vis_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=1, shuffle=False,
    collate_fn=data.collate_remove_none,
    worker_init_fn=data.worker_init_fn)
model_counter = defaultdict(int)
data_vis_list = []

iterator = iter(vis_loader)


# Model
model = config.get_model(cfg, device=device, dataset=val_dataset)

# Generator
generator = config.get_generator(model, cfg, device=device)

# Intialize training
if cfg['training']['opt'] == "Adam":
    optimizer = optim.Adam(model.parameters(), lr=cfg['training']['lr'])
if cfg['training']['opt'] == "SGD":
    optimizer = optim.SGD(model.parameters(), lr=cfg['training']['lr'], momentum=0.9)

inferencer = config.get_inferencer(model, optimizer, generator, cfg, device=device)

checkpoint_io = CheckpointIO(out_dir, model=model, optimizer=optimizer)
try:
    load_dict = checkpoint_io.load(cfg['test']['model_file'], device=device)
except FileExistsError:
    load_dict = dict()

epoch_it = load_dict.get('epoch_it', 0)
it = load_dict.get('it', 0)
metric_val_best = load_dict.get(
    'loss_val_best', -model_selection_sign * np.inf)

if metric_val_best == np.inf or metric_val_best == -np.inf:
    metric_val_best = -model_selection_sign * np.inf
print('Current best validation metric (%s): %.8f'
    % (model_selection_metric, metric_val_best))
# logger = SummaryWriter(os.path.join(out_dir, 'logs'))


# Print model
nparameters = sum(p.numel() for p in model.parameters())
print('Total number of parameters: %d' % nparameters)

print('output path: ', cfg['training']['out_dir'])
    

real = True

if real == False:

    vis_class = 0
    data_vis_class = []
    data_vis_name = []
    for i in range(len(vis_loader)):
        
        data_vis = next(iterator)
        vis_name = vis_name_list[i]['model']

        vis_name_split = vis_name.split("_")
        if len(vis_name_split) < 4:
            vis_class_now = int(vis_name_split[0][-2:])
        else:
            vis_class_now = int(vis_name_split[0])
        vis_angle = int(vis_name_split[-1])
        
        if vis_class_now != vis_class:
            data_vis_list.append(data_vis_class)
            data_vis_class = []
            vis_class = vis_class_now
            data_vis_name.append(vis_name_split[0])
        else:
            data_vis = {'touch_id': int(vis_name_split[-2]), 'data': data_vis, 'name': vis_name}
            
            if len(vis_name_split) < 4:
                if vis_angle in [6, 7]:
                    data_vis_class.append(data_vis)
            else:
                data_vis_class.append(data_vis)
        
    data_vis_list.append(data_vis_class)
    data_vis_list = data_vis_list[1:]

    for class_idx, data_vis_class in enumerate(data_vis_list):
        print("Begin inferencing {}".format(data_vis_name[class_idx]))
        mesh_list_obj, mesh_list_hand = inferencer.inference_step(data_vis_class)

        for mesh_idx, (mesh_obj, mesh_hand) in enumerate(zip(mesh_list_obj, mesh_list_hand)):
            mesh_obj.export(os.path.join(out_dir, 'vis', '{}_{}_{:03d}_obj.off'.format(it, data_vis_name[class_idx], data_vis_class[mesh_idx]['touch_id'])))
            mesh_hand.export(os.path.join(out_dir, 'vis', '{}_{}_{:03d}_hand.off'.format(it, data_vis_name[class_idx], data_vis_class[mesh_idx]['touch_id'])))

    print("Finished inferencing!")

else:

    for i in range(len(vis_loader)):
        
        data_vis = next(iterator)
        vis_name = vis_name_list[i]['model']

        if cfg['generation']['vis_all']:
            idx = data_vis['idx'].item()
            model_dict = val_dataset.get_model_dict(idx)
            category_id = model_dict.get('category', 'n/a')
            category_name = val_dataset.metadata[category_id].get('name', 'n/a')
            category_name = category_name.split(',')[0]
            if category_name == 'n/a':
                category_name = category_id

            c_it = model_counter[category_id]
            # if c_it < vis_n_outputs:
            data_vis_list.append({'category': category_name, 'it': c_it, 'data': data_vis, 'name': vis_name})

            model_counter[category_id] += 1
        else:
            vis_split = cfg['generation']['vis_split']
            if i % vis_split == 0:
                idx = data_vis['idx'].item()
                model_dict = val_dataset.get_model_dict(idx)
                category_id = model_dict.get('category', 'n/a')
                category_name = val_dataset.metadata[category_id].get('name', 'n/a')
                category_name = category_name.split(',')[0]
                if category_name == 'n/a':
                    category_name = category_id

                c_it = model_counter[category_id]
                # if c_it < vis_n_outputs:
                data_vis_list.append({'category': category_name, 'it': c_it, 'data': data_vis, 'name': vis_name})

                model_counter[category_id] += 1
    
    print('Visualizing at iteration: %d' % it)
    for data_vis in tqdm(data_vis_list):
        # if cfg['generation']['sliding_window']:
        #     out = generator.generate_mesh_sliding(data_vis['data'])    
        # else:
        #     out = generator.generate_mesh(data_vis['data'])
        # # Get statistics
        # try:
        #     mesh, stats_dict = out
        # except TypeError:
        #     mesh, stats_dict = out, {}
        
        mesh_hand = generator.generate_hand_mesh(data_vis['data'])
        mesh_obj, _, _ = generator.generate_obj_mesh_wnf(data_vis['data'])
        
        mesh_obj.export(os.path.join(out_dir, 'vis', '{}_{}.off'.format(it, data_vis['name'])))
        mesh_hand.export(os.path.join(out_dir, 'vis', '{}_{}_hand.off'.format(it, data_vis['name'])))