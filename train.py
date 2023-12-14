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
import igl



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



# Shorthands
out_dir = cfg['training']['out_dir']
batch_size = cfg['training']['batch_size']
backup_every = cfg['training']['backup_every']

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
train_dataset = config.get_dataset('train', cfg)
train_name_list = train_dataset.models
val_dataset = config.get_dataset('val', cfg, return_idx=True)
vis_name_list = val_dataset.models

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, num_workers=cfg['training']['n_workers'], shuffle=True,
    collate_fn=data.collate_remove_none,
    worker_init_fn=data.worker_init_fn)

val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=1, num_workers=cfg['training']['n_workers_val'], shuffle=False,
    collate_fn=data.collate_remove_none,
    worker_init_fn=data.worker_init_fn)

# For visualizations
vis_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=1, shuffle=False,
    collate_fn=data.collate_remove_none,
    worker_init_fn=data.worker_init_fn)
model_counter = defaultdict(int)
data_vis_list = []

# Build a data dictionary for visualization
iterator = iter(vis_loader)

for i in range(len(vis_loader)):
    
    data_vis = next(iterator)
    if cfg['data']['dataset'] == "Sequence":
        data_vis = data_vis[3]
        
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

# Model
model = config.get_model(cfg, device=device, dataset=train_dataset)

# Generator
generator = config.get_generator(model, cfg, device=device)

# Intialize training
if cfg['training']['opt'] == "Adam":
    optimizer = optim.Adam(model.parameters(), lr=cfg['training']['lr'])
if cfg['training']['opt'] == "SGD":
    optimizer = optim.SGD(model.parameters(), lr=cfg['training']['lr'], momentum=0.9)
trainer = config.get_trainer(model, optimizer, cfg, device=device)

checkpoint_io = CheckpointIO(out_dir, model=model, optimizer=optimizer)
try:
    load_dict = checkpoint_io.load(cfg['test']['model_file'], device=device)
except FileNotFoundError:
    load_dict = dict()

epoch_it = load_dict.get('epoch_it', 0)
it = load_dict.get('it', 0)
metric_val_best = load_dict.get(
    'loss_val_best', -model_selection_sign * np.inf)

if metric_val_best == np.inf or metric_val_best == -np.inf:
    metric_val_best = -model_selection_sign * np.inf
print('Current best validation metric (%s): %.8f'
      % (model_selection_metric, metric_val_best))
logger = SummaryWriter(os.path.join(out_dir, 'logs'))

# Shorthands
print_every = cfg['training']['print_every']
checkpoint_every = cfg['training']['checkpoint_every']
validate_every = cfg['training']['validate_every']
visualize_every = cfg['training']['visualize_every']

# Print model
nparameters = sum(p.numel() for p in model.parameters())
print('Total number of parameters: %d' % nparameters)

print('output path: ', cfg['training']['out_dir'])

vf_dict = dict()

for model_name in train_name_list:
    obj_name = model_name['model'][:-5]
    if obj_name not in vf_dict.keys():
        mesh_path = os.path.join("./data/VTacO_mesh/mesh_obj", obj_name+".off")
        if not os.path.exists(mesh_path):
            mesh_path = os.path.join("./data/VTacO_mesh/mesh_obj", obj_name+".obj")
        v, f = igl.read_triangle_mesh(mesh_path)
        vf_obj = dict()
        vf_obj['v'] = v.astype(np.float32)
        vf_obj['f'] = f
        vf_dict[obj_name] = vf_obj


# Set t0
t0 = time.time()


while True:
    epoch_it += 1

    for batch in train_loader:

        it += 1
        if cfg['model']['with_contact'] == False:
            loss, loss_mano, loss_pc = trainer.train_step(batch, vf_dict)
        else:
            loss, loss_mano, loss_pc, loss_contact = trainer.train_step(batch, vf_dict)
            logger.add_scalar('train/loss_contact', loss_contact, it)
        
        logger.add_scalar('train/loss', loss, it)
        logger.add_scalar('train/loss_mano', loss_mano, it)

        # Print output
        if print_every > 0 and (it % print_every) == 0:
            t = datetime.datetime.now()
            if cfg['model']['with_contact'] == False:
                print('[Epoch %02d] it=%03d, loss=%.4f, loss_mano=%.4f, loss_pc=%.5f, time: %.2fs, %02d:%02d'
                        % (epoch_it, it, loss-loss_mano, loss_mano, loss_pc, time.time() - t0, t.hour, t.minute))
            else:
                print('[Epoch %02d] it=%03d, loss=%.4f, loss_c=%.4f, loss_mano=%.4f, loss_pc=%.5f, time: %.2fs, %02d:%02d'
                        % (epoch_it, it, loss, loss_contact, loss_mano, loss_pc, time.time() - t0, t.hour, t.minute))

        # Run validation
        if validate_every > 0 and (it % validate_every) == 0:
            print("Validate at iteration: %d" % it)
            eval_dict = trainer.evaluate(val_loader, vf_dict)
            metric_val = eval_dict[model_selection_metric]
            print('Validation metric (%s): %.4f'
                  % (model_selection_metric, metric_val))


            for k, v in eval_dict.items():
                if k != model_selection_metric:
                    print('Validation metric (%s): %.6f' % (k, v))
                logger.add_scalar('val/%s' % k, v, it)

            if model_selection_sign * (metric_val - metric_val_best) > 0:
                metric_val_best = metric_val
                print('New best model (loss %.4f)' % metric_val_best)
                checkpoint_io.save('model_best.pt', epoch_it=epoch_it, it=it,
                                   loss_val_best=metric_val_best)
        
        # Save checkpoint
        if (checkpoint_every > 0 and (it % checkpoint_every) == 0):
            print('Saving checkpoint at iteration: %d' % it)
            checkpoint_io.save('model.pt', epoch_it=epoch_it, it=it,
                               loss_val_best=metric_val_best)

        # Backup if necessary
        if (backup_every > 0 and (it % backup_every) == 0):
            print('Backup checkpoint at iteration: %d' % it)
            checkpoint_io.save('model_%d.pt' % it, epoch_it=epoch_it, it=it,
                               loss_val_best=metric_val_best)  
                
        # Visualize output
        if visualize_every > 0 and (it % visualize_every) == 0:
            emd_total = []
            cd_total = []
            print('Visualizing at iteration: %d' % it)
            for data_vis in tqdm(data_vis_list):
                
                mesh_hand = generator.generate_hand_mesh(data_vis['data'])
                mesh_obj, emd, cd = generator.generate_obj_mesh_wnf(data_vis['data'])
                emd_total.append(emd)
                cd_total.append(cd)

                mesh_hand.export(os.path.join(out_dir, 'vis', '{}_{}_hand.off'.format(it, data_vis['name'])))
                mesh_obj.export(os.path.join(out_dir, 'vis', '{}_{}_obj.off'.format(it, data_vis['name'])))

            print("Finish visualizing!")
            
            print("Metrics EMD: {}".format(np.mean(emd_total)))
            print("Metrics CD: {}".format(np.mean(cd_total)))

        # Exit if necessary
        if exit_after > 0 and (time.time() - t0) >= exit_after:
            print('Time limit reached. Exiting.')
            checkpoint_io.save('model.pt', epoch_it=epoch_it, it=it,
                               loss_val_best=metric_val_best)
            exit(3)
