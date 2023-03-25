import torch
import torch.optim as optim
from torch import autograd
import numpy as np
from tqdm import trange, tqdm
import trimesh
from src.utils import libmcubes
from src.common import (
    make_3d_grid, normalize_coord, add_key, coord2index, 
    RFUniverseCamera, R_from_PYR, norm_pc_1, pc_cam_to_world,
    chamfer_distance, EarthMoverDistance, compute_iou
)
from src.utils.libsimplify import simplify_mesh
from src.utils.libmise import MISE
import time
import math
from skimage import measure
from scipy.spatial.transform import Rotation as R
from scipy.spatial import distance
import time

counter = 0
depth_origin = np.loadtxt("./data/VTacO_mesh/depth_origin.txt")
w = 240
h = 320

class Generator3D(object):
    '''  Generator class for Occupancy Networks.

    It provides functions to generate the final mesh as well refining options.

    Args:
        model (nn.Module): trained Occupancy Network model
        points_batch_size (int): batch size for points evaluation
        threshold (float): threshold value
        refinement_step (int): number of refinement steps
        device (device): pytorch device
        resolution0 (int): start resolution for MISE
        upsampling steps (int): number of upsampling steps
        with_normals (bool): whether normals should be estimated
        padding (float): how much padding should be used for MISE
        sample (bool): whether z should be sampled
        input_type (str): type of input
        vol_info (dict): volume infomation
        vol_bound (dict): volume boundary
        simplify_nfaces (int): number of faces the mesh should be simplified to
    '''

    def __init__(self, model, points_batch_size=100000,
                 threshold=0.5, refinement_step=0, device=None,
                 resolution0=16, upsampling_steps=3,
                 with_normals=False, padding=0.1, sample=False,
                 input_type = None,
                 vol_info = None,
                 vol_bound = None,
                 simplify_nfaces=None,
                 alpha=0.2,
                 with_img=False, encode_t2d=False):
        self.model = model.to(device)
        self.points_batch_size = points_batch_size
        self.refinement_step = refinement_step
        self.threshold = threshold
        self.device = device
        self.resolution0 = resolution0
        self.upsampling_steps = upsampling_steps
        self.with_normals = with_normals
        self.input_type = input_type
        self.padding = padding
        self.sample = sample
        self.simplify_nfaces = simplify_nfaces
        self.alpha = alpha
        self.with_img = with_img
        self.encode_t2d = encode_t2d
        
        # for pointcloud_crop
        self.vol_bound = vol_bound
        if vol_info is not None:
            self.input_vol, _, _ = vol_info
        
    def generate_hand_mesh(self, data):
        
        self.model.eval()
        device = self.device
        inputs = data.get('inputs', torch.empty(1, 0)).to(device)
        pc_ply = data.get('inputs.pc_ply').to(device)
        mano_gt = data.get('points.mano').to(device)
        wrist_rot_euler = data.get('points.wrist').to(device)
        wrist_rot_euler = wrist_rot_euler.squeeze().detach().cpu().numpy()
        
        t0 = time.time()
        with torch.no_grad():
            c_hand = self.model.encode_hand_inputs(inputs)
            
        mano_param, verts = c_hand['mano_param'].squeeze().detach().cpu().numpy(), c_hand['mano_verts'].squeeze().detach().cpu().numpy()
        joints, faces = c_hand['mano_joints'].squeeze().detach().cpu().numpy(), c_hand['mano_faces'].squeeze().detach().cpu().numpy()
        wrist_pos, wrist_rotvec = mano_param[:3], mano_param[3:6]
        wrist_rot = R.from_rotvec(wrist_rotvec)
        wrist_rot_euler = wrist_rot.as_euler('XYZ', degrees=False)
        
        # wrist_pos = mano_gt.squeeze().detach().cpu().numpy()[:3]
        

        
        verts = verts - np.array([0.11, 0.005, 0], dtype=np.float32)
        verts = np.linalg.inv(R_from_PYR(np.array([-np.pi/2, np.pi/2, 0]))) @ verts.T
        verts = np.linalg.inv(R_from_PYR(np.array(wrist_rot_euler))) @ verts
        verts = verts.T + wrist_pos
        
        joints = joints - np.array([0.11, 0.005, 0], dtype=np.float32)
        joints = np.linalg.inv(R_from_PYR(np.array([-np.pi/2, np.pi/2, 0]))) @ joints.T
        joints = np.linalg.inv(R_from_PYR(np.array(wrist_rot_euler))) @ joints
        joints = joints.T + wrist_pos
        
        verts = norm_pc_1(verts, pc_ply.squeeze().detach().cpu().numpy())
        joints = norm_pc_1(joints, pc_ply.squeeze().detach().cpu().numpy())
        
        mesh = trimesh.Trimesh(verts, faces)
        
        return mesh

    def generate_obj_mesh_wnf(self, data):
        
        self.model.eval()
        
        box_size = 1 + self.padding
        nx = self.resolution0 * 4
        
        device = self.device
        inputs = data.get('inputs', torch.empty(1, 0)).to(device)
        inputs = add_key(inputs, data.get('inputs.ind'), 'points', 'index', device=device)
        imgs = data.get('inputs.img').to(device)
        depths = data.get('inputs.depth').to(device)
        touch_success = data.get('inputs.touch_success').to(device)
        pc_ply = data.get('inputs.pc_ply').to(device).detach().cpu().numpy()
        mano_gt = data.get('points.mano').to(device)
        points_obj = data.get('points.points_obj')
        kwargs = {}
        
        wrist_rot_euler = data.get('points.wrist').to(device)
        wrist_rot_euler = wrist_rot_euler.squeeze().detach().cpu().numpy()
        
        wrist_pos = mano_gt.squeeze().detach().cpu().numpy()[:3]
        
        cam_pos = data.get('points.cam_pos').to(device).reshape(1, -1)
        cam_rot = data.get('points.cam_rot').to(device).reshape(1, -1)
        cam_info = torch.cat((cam_pos, cam_rot), dim=1)
        
        cam_pos_d = cam_pos.cpu().detach().numpy().reshape(1, 5, 3)
        cam_rot_d = cam_rot.cpu().detach().numpy().reshape(1, 5, 3)
        
        width = w
        
        near_plane = 0.019
        far_plane = 0.022
        fov = 60
        cam_unity = RFUniverseCamera(width, height, near_plane, far_plane, fov)
        
        t0 = time.time()
        
        pointsf = box_size * make_3d_grid(
                (-0.5,)*3, (0.5,)*3, (nx,)*3
            )
        
        if self.with_img:
            
            if not self.encode_t2d:
                with torch.no_grad():
                    
                    p = pointsf
                    B = 1
                    N, D = p.size()
                    
                    # if hand and object are separated
                    c = self.model.encode_inputs(inputs)
                    c_hand = self.model.encode_hand_inputs(inputs)

                    c_img = self.model.encode_img_inputs(imgs)
                    c_img_all = torch.zeros(B, N, c_img.size()[2]).to(device)

                    mano_param = c_hand['mano_param']
                    mano_pc = c_hand['mano_verts']
                    mano_joints = c_hand['mano_joints']
                    # wrist_pos, wrist_rotvec = mano_param[0, :3].cpu().detach().numpy(), mano_param[0, 3:6].cpu().detach().numpy()
                    
                    tips_idx = [4, 8, 12, 16, 20]
                    tips_pos = mano_joints.cpu().detach().numpy()[0, tips_idx]
                    tips_pos = tips_pos - np.array([0.11, 0.005, 0], dtype=np.float32)
                    tips_pos = np.linalg.inv(R_from_PYR(np.array([-np.pi/2, np.pi/2, 0]))) @ tips_pos.T
                    tips_pos = np.linalg.inv(R_from_PYR(np.array(wrist_rot_euler))) @ tips_pos
                    tips_pos_b = tips_pos.T + wrist_pos
                    
                    tips_pos_b = norm_pc_1(tips_pos_b, pc_ply.squeeze())
                    
                    p_new = p.cpu().detach().numpy()
                    
                    p_new_b = p_new
                    # print(p_new_b.shape, tips_pos_b.shape)
                    dist_p_tips = distance.cdist(p_new_b, tips_pos_b)
                    
                    for finger in range(5):
                        # if touch successful, cat the local feature to the query points
                        if touch_success[0, finger]:
                            tips_points_idx_1 = np.where((np.min(dist_p_tips, 1) < 0.05) & (np.argmin(dist_p_tips, 1) == finger))[0]

                            c_img_all[0, tips_points_idx_1, :] = c_img[0, finger, :]
            
            else:
                
                with torch.no_grad():
                    
                    p = pointsf
                    B = 1
                    N, D = p.size()
                    
                    # if hand and object are separated
                    
                    pred_depth, c_hand = self.model.encode_t2d(inputs, imgs)
                    pred_d_detach = pred_depth.cpu().detach().numpy()
                    
                    digit_param = c_hand['mano_param']
                    cam_p = digit_param[:, :15].reshape(B, 5, 3)
                    cam_r = digit_param[:, 15:].reshape(B, 5, 3)
        
                    c = self.model.encode_inputs(inputs)
                    c_hand = self.model.encode_hand_inputs(inputs)

                    c_img = self.model.encode_img_inputs(imgs)
                    c_img_all = torch.zeros(B, N, c_img.size()[2]).to(device)

                    for t_idx in range(5):
                        # if touch successful, cat the local feature to the query points
                        if touch_success[0, t_idx]:
                            depth = pred_d_detach[0, t_idx].reshape(h, w)
                            depth = depth*0.005 + 0.019
                            depth = depths.squeeze().cpu().numpy()[t_idx].reshape(h, w)
                                    
                            depth_diff = depth.reshape(w * h) - depth_origin
                            idx_points = np.where(abs(depth_diff)>0.0001)
                            if idx_points[0].shape == 0: continue
                            
                            _, pc_depth_all = cam_unity.depth_2_camera_pointcloud(depth)
                            pc_depth_new = pc_depth_all[idx_points]
                            
                            if pc_depth_new.shape[0] > 128:
                                pc_world_indice = np.random.randint(pc_depth_new.shape[0], size=128)
                                pc_depth_new = pc_depth_new[pc_world_indice]
                            
                            pc_world_all = pc_cam_to_world(pc_depth_new, rot=cam_rot_d[0, t_idx]+[-np.pi/2, 0, np.pi/2], trans=cam_pos_d[0, t_idx])
                                    
                            pc_world_all = norm_pc_1(pc_world_all, pc_ply.squeeze())
                            # np.savetxt("/newssd1/home/zhenjun/conv_pcnet/out/inference/akb_grid_img_d_7/test_{}.txt".format(t_idx), pc_world_all)
                            
                            reso_split = 64**3
                            for p_split in range(8):
                                
                                dist_pc_sample = distance.cdist(pc_world_all, p[p_split*reso_split:(p_split+1)*reso_split])
                                idx_img = np.where(dist_pc_sample<0.015)[1]
                                # print(idx_img.shape)
                                if idx_img.shape[0] != 0:
                                    c_img_all[0, idx_img+p_split*reso_split, :] = c_img[0, t_idx, :]

                            # dist_pc_sample = distance.cdist(pc_world_all, p)
                            # idx_img = np.where(dist_pc_sample<0.02)[1]
                            
                            # c_img_all[0, idx_img, :] = c_img[0, t_idx, :]

            values = self.eval_points(pointsf, c, c_img_all, **kwargs).cpu().numpy()
            
            
        else:
            with torch.no_grad():
                c = self.model.encode_inputs(inputs)
            
            values = self.eval_points(pointsf, c, **kwargs).cpu().numpy()

            
        
        value_grid = values.reshape(nx, nx, nx)
        # print(np.max(value_grid), np.min(value_grid), np.mean(value_grid))
        
        vertices, faces, normals, _ = measure.marching_cubes_lewiner(value_grid,
                                                                    #  level = self.threshold,
                                                                     gradient_direction='ascent')
        vertices -= np.array([nx/2, nx/2, nx/2], dtype=np.float32)
        vertices *= 1.1/nx
        mesh = trimesh.Trimesh(vertices, faces)

        np.random.shuffle(vertices)
        vertices = vertices[:2048]
        
        vertices = np.ascontiguousarray(vertices, dtype=np.float32)

        cd = chamfer_distance(points_obj.to(self.device), torch.FloatTensor([vertices]).to(self.device), use_kdtree=False)
        emd = EarthMoverDistance(points_obj[0].numpy(), vertices)
        
        return mesh, emd, cd.item()

    def generate_tactile_pc(self, data):
        
        device = self.device
        p = data.get('points').to(device)
        data_name = data.get('points.name')
        B, N, D = p.size()
        
        inputs = data.get('inputs', torch.empty(p.size(0), 0)).to(device)
        pc_ply = data.get('inputs.pc_ply').to(device)
        imgs = data.get('inputs.img').to(device)
        depths = data.get('inputs.depth').to(device).cpu().detach().numpy() # (5, 60000)
        cam_pos = data.get('points.cam_pos').to(device).reshape(p.size(0), -1) # (1, 15)
        cam_rot = data.get('points.cam_rot').to(device).reshape(p.size(0), -1) # (1, 15)
        cam_info = torch.cat((cam_pos, cam_rot), dim=1)
        
        cam_pos = cam_pos.cpu().detach().numpy().reshape(B, 5, 3)
        cam_rot = cam_rot.cpu().detach().numpy().reshape(B, 5, 3)
        
        # print(depths.shape, cam_pos.shape, cam_rot.shape)
        
        
        # with torch.no_grad():
        #     pred_info = self.model.encode_hand_inputs(inputs)
        #     digit_param = pred_info['mano_param'].cpu().detach().numpy()[0] # (30,)
        #     cam_p = digit_param[:15].reshape(5, 3)
        #     cam_r = digit_param[15:].reshape(5, 3)
            
        #     pred_depth = self.model.encode_img_inputs(imgs).cpu().detach().numpy()[0] # (5, 60000)
            
        #     width = 240
        #     height = 320
        #     near_plane = 0.019
        #     far_plane = 0.022
        #     fov = 60
        #     cam_unity = RFUniverseCamera(width, height, near_plane, far_plane, fov)
            
        #     pc_world_l = np.zeros((pred_depth.shape[0], 76800, 3))
        #     for t_idx in range(pred_depth.shape[0]):
        #         depth = pred_depth[t_idx].reshape(320, 240)
        #         depth = depth*0.005 + 0.019
        #         pc_depth, pc_depth_all = cam_unity.depth_2_camera_pointcloud(depth)
                
        #         # pc_world = pc_cam_to_world(pc_depth, rot=cam_r[t_idx]+[-90, 0, 90], trans=cam_p[t_idx])
        #         pc_world_all = pc_cam_to_world(pc_depth_all, rot=cam_rot[t_idx]+[-np.pi/2, 0, np.pi/2], trans=cam_pos[t_idx])
        #         pc_world_l[t_idx] = norm_pc_1(pc_world_all, pc_ply.squeeze().detach().cpu().numpy())
                
        #         # depth = depths[t_idx].reshape(h, w)
        #         # pc_depth, pc_depth_all = cam_unity.depth_2_camera_pointcloud(depth)
        #         # pc_world_all = pc_cam_to_world(pc_depth_all, rot=cam_rot[t_idx]+[-np.pi/2, 0, np.pi/2], trans=cam_pos[t_idx])
        #         # pc_world_l[t_idx] = norm_pc_1(pc_world_all, pc_ply.squeeze().detach().cpu().numpy())
                
        #     return pc_world_l
        
        with torch.no_grad():
            pred_info = self.model.encode_hand_inputs(inputs)
            pred_depth = self.model.encode_img_inputs(imgs).cpu().detach().numpy() # (B, 5, 60000)
            
            digit_param = pred_info['mano_param'].cpu().detach().numpy() # (B, 30,)
            
            width = 240
            height = 320
            near_plane = 0.019
            far_plane = 0.022
            fov = 60
            cam_unity = RFUniverseCamera(width, height, near_plane, far_plane, fov)
            
            pc_world_l = np.zeros((B, pred_depth.shape[1], 76800, 3))
             
            for batch in range(B):
                cam_p = digit_param[batch, :15].reshape(5, 3)
                cam_r = digit_param[batch, 15:].reshape(5, 3)

               
                for t_idx in range(pred_depth.shape[1]):
                    depth = pred_depth[batch, t_idx].reshape(320, 240)
                    depth = depth*0.005 + 0.019
                    pc_depth, pc_depth_all = cam_unity.depth_2_camera_pointcloud(depth)
                    
                    # pc_world = pc_cam_to_world(pc_depth, rot=cam_r[t_idx]+[-90, 0, 90], trans=cam_p[t_idx])
                    pc_world_all = pc_cam_to_world(pc_depth_all, rot=cam_rot[batch, t_idx, :]+[-np.pi/2, 0, np.pi/2], trans=cam_pos[batch, t_idx, :])
                    pc_world_l[batch, t_idx] = norm_pc_1(pc_world_all, pc_ply[batch].detach().cpu().numpy())
                    
                    # depth = depths[t_idx].reshape(h, w)
                    # pc_depth, pc_depth_all = cam_unity.depth_2_camera_pointcloud(depth)
                    # pc_world_all = pc_cam_to_world(pc_depth_all, rot=cam_rot[t_idx]+[-np.pi/2, 0, np.pi/2], trans=cam_pos[t_idx])
                    # pc_world_l[t_idx] = norm_pc_1(pc_world_all, pc_ply.squeeze().detach().cpu().numpy())
        
            return pc_world_l, data_name


    def generate_mesh(self, data, return_stats=True):
        ''' Generates the output mesh.

        Args:
            data (tensor): data tensor
            return_stats (bool): whether stats should be returned
        '''
        self.model.eval()
        device = self.device
        stats_dict = {}

        inputs = data.get('inputs', torch.empty(1, 0)).to(device)
        kwargs = {}

        t0 = time.time()
        
        # obtain features for all crops
        if self.vol_bound is not None:
            self.get_crop_bound(inputs)
            c = self.encode_crop(inputs, device)
        else: # input the entire volume
            inputs = add_key(inputs, data.get('inputs.ind'), 'points', 'index', device=device)
            t0 = time.time()
            with torch.no_grad():
                c = self.model.encode_inputs(inputs)
        stats_dict['time (encode inputs)'] = time.time() - t0
        
        mesh = self.generate_from_latent(c, stats_dict=stats_dict, **kwargs)

        if return_stats:
            return mesh, stats_dict
        else:
            return mesh
    
    def generate_from_latent(self, c=None, stats_dict={}, **kwargs):
        ''' Generates mesh from latent.
            Works for shapes normalized to a unit cube

        Args:
            c (tensor): latent conditioned code c
            stats_dict (dict): stats dictionary
        '''
        threshold = np.log(self.threshold) - np.log(1. - self.threshold)

        t0 = time.time()
        # Compute bounding box size
        box_size = 1 + self.padding
        
        # Shortcut
        if self.upsampling_steps == 0:
            nx = self.resolution0
            pointsf = box_size * make_3d_grid(
                (-0.5,)*3, (0.5,)*3, (nx,)*3
            )

            values = self.eval_points(pointsf, c, **kwargs).cpu().numpy()
            value_grid = values.reshape(nx, nx, nx)
        else:
            mesh_extractor = MISE(
                self.resolution0, self.upsampling_steps, threshold)

            points = mesh_extractor.query()
            while points.shape[0] != 0:
                # Query points
                pointsf = points / mesh_extractor.resolution
                # Normalize to bounding box
                pointsf = box_size * (pointsf - 0.5)
                pointsf = torch.FloatTensor(pointsf).to(self.device)
                # Evaluate model and update
                values = self.eval_points(pointsf, c, **kwargs).cpu().numpy()
                values = values.astype(np.float64)
                mesh_extractor.update(points, values)
                points = mesh_extractor.query()

            value_grid = mesh_extractor.to_dense()

        # Extract mesh
        stats_dict['time (eval points)'] = time.time() - t0

        mesh = self.extract_mesh(value_grid, c, stats_dict=stats_dict)
        return mesh

    def generate_mesh_sliding(self, data, return_stats=True):
        ''' Generates the output mesh in sliding-window manner.
            Adapt for real-world scale.

        Args:
            data (tensor): data tensor
            return_stats (bool): whether stats should be returned
        '''
        self.model.eval()
        device = self.device
        stats_dict = {}
        
        threshold = np.log(self.threshold) - np.log(1. - self.threshold)

        inputs = data.get('inputs', torch.empty(1, 0)).to(device)
        kwargs = {}

        # acquire the boundary for every crops
        self.get_crop_bound(inputs)

        nx = self.resolution0
        n_crop = self.vol_bound['n_crop']
        n_crop_axis = self.vol_bound['axis_n_crop']

        # occupancy in each direction
        r = nx * 2**self.upsampling_steps
        occ_values = np.array([]).reshape(r,r,0)
        occ_values_y = np.array([]).reshape(r,0,r*n_crop_axis[2])
        occ_values_x = np.array([]).reshape(0,r*n_crop_axis[1],r*n_crop_axis[2])
        for i in trange(n_crop):
            # encode the current crop
            vol_bound = {}
            vol_bound['query_vol'] = self.vol_bound['query_vol'][i]
            vol_bound['input_vol'] = self.vol_bound['input_vol'][i]
            c = self.encode_crop(inputs, device, vol_bound=vol_bound)

            bb_min = self.vol_bound['query_vol'][i][0]
            bb_max = bb_min + self.vol_bound['query_crop_size']

            if self.upsampling_steps == 0:
                t = (bb_max - bb_min)/nx # inteval
                pp = np.mgrid[bb_min[0]:bb_max[0]:t[0], bb_min[1]:bb_max[1]:t[1], bb_min[2]:bb_max[2]:t[2]].reshape(3, -1).T
                pp = torch.from_numpy(pp).to(device)
                values = self.eval_points(pp, c, vol_bound=vol_bound, **kwargs).detach().cpu().numpy()
                values = values.reshape(nx, nx, nx)
            else:
                mesh_extractor = MISE(self.resolution0, self.upsampling_steps, threshold)
                points = mesh_extractor.query()
                while points.shape[0] != 0:
                    pp = points / mesh_extractor.resolution
                    pp = pp * (bb_max - bb_min) + bb_min
                    pp = torch.from_numpy(pp).to(self.device)

                    values = self.eval_points(pp, c, vol_bound=vol_bound, **kwargs).detach().cpu().numpy()
                    values = values.astype(np.float64)
                    mesh_extractor.update(points, values)
                    points = mesh_extractor.query()
                
                values = mesh_extractor.to_dense()
                # MISE consider one more voxel around boundary, remove
                values = values[:-1, :-1, :-1]

            # concatenate occ_value along every axis
            # along z axis
            occ_values = np.concatenate((occ_values, values), axis=2)
            # along y axis
            if (i+1) % n_crop_axis[2] == 0: 
                occ_values_y = np.concatenate((occ_values_y, occ_values), axis=1)
                occ_values = np.array([]).reshape(r, r, 0)
            # along x axis
            if (i+1) % (n_crop_axis[2]*n_crop_axis[1]) == 0:
                occ_values_x = np.concatenate((occ_values_x, occ_values_y), axis=0)
                occ_values_y = np.array([]).reshape(r, 0,r*n_crop_axis[2])
            
        value_grid = occ_values_x    
        mesh = self.extract_mesh(value_grid, c, stats_dict=stats_dict)

        if return_stats:
            return mesh, stats_dict
        else:
            return mesh

    def get_crop_bound(self, inputs):
        ''' Divide a scene into crops, get boundary for each crop

        Args:
            inputs (dict): input point cloud
        '''
        query_crop_size = self.vol_bound['query_crop_size']
        input_crop_size = self.vol_bound['input_crop_size']
        lb_query_list, ub_query_list = [], []
        lb_input_list, ub_input_list = [], []
        
        lb = inputs.min(axis=1).values[0].cpu().numpy() - 0.01
        ub = inputs.max(axis=1).values[0].cpu().numpy() + 0.01
        lb_query = np.mgrid[lb[0]:ub[0]:query_crop_size,\
                    lb[1]:ub[1]:query_crop_size,\
                    lb[2]:ub[2]:query_crop_size].reshape(3, -1).T
        ub_query = lb_query + query_crop_size
        center = (lb_query + ub_query) / 2
        lb_input = center - input_crop_size/2
        ub_input = center + input_crop_size/2
        # number of crops alongside x,y, z axis
        self.vol_bound['axis_n_crop'] = np.ceil((ub - lb)/query_crop_size).astype(int)
        # total number of crops
        num_crop = np.prod(self.vol_bound['axis_n_crop'])
        self.vol_bound['n_crop'] = num_crop
        self.vol_bound['input_vol'] = np.stack([lb_input, ub_input], axis=1)
        self.vol_bound['query_vol'] = np.stack([lb_query, ub_query], axis=1)
        
    def encode_crop(self, inputs, device, vol_bound=None):
        ''' Encode a crop to feature volumes

        Args:
            inputs (dict): input point cloud
            device (device): pytorch device
            vol_bound (dict): volume boundary
        '''
        if vol_bound == None:
            vol_bound = self.vol_bound

        index = {}
        for fea in self.vol_bound['fea_type']:
            # crop the input point cloud
            mask_x = (inputs[:, :, 0] >= vol_bound['input_vol'][0][0]) &\
                    (inputs[:, :, 0] < vol_bound['input_vol'][1][0])
            mask_y = (inputs[:, :, 1] >= vol_bound['input_vol'][0][1]) &\
                    (inputs[:, :, 1] < vol_bound['input_vol'][1][1])
            mask_z = (inputs[:, :, 2] >= vol_bound['input_vol'][0][2]) &\
                    (inputs[:, :, 2] < vol_bound['input_vol'][1][2])
            mask = mask_x & mask_y & mask_z
            
            p_input = inputs[mask]
            if p_input.shape[0] == 0: # no points in the current crop
                p_input = inputs.squeeze()
                ind = coord2index(p_input.clone(), vol_bound['input_vol'], reso=self.vol_bound['reso'], plane=fea)
                if fea == 'grid':
                    ind[~mask] = self.vol_bound['reso']**3
                else:
                    ind[~mask] = self.vol_bound['reso']**2
            else:
                ind = coord2index(p_input.clone(), vol_bound['input_vol'], reso=self.vol_bound['reso'], plane=fea)
            index[fea] = ind.unsqueeze(0)
            input_cur = add_key(p_input.unsqueeze(0), index, 'points', 'index', device=device)
        
        with torch.no_grad():
            c = self.model.encode_inputs(input_cur)
        return c
    
    def predict_crop_occ(self, pi, c, vol_bound=None, **kwargs):
        ''' Predict occupancy values for a crop

        Args:
            pi (dict): query points
            c (tensor): encoded feature volumes
            vol_bound (dict): volume boundary
        '''
        occ_hat = pi.new_empty((pi.shape[0]))
    
        if pi.shape[0] == 0:
            return occ_hat
        pi_in = pi.unsqueeze(0)
        pi_in = {'p': pi_in}
        p_n = {}
        for key in self.vol_bound['fea_type']:
            # projected coordinates normalized to the range of [0, 1]
            p_n[key] = normalize_coord(pi.clone(), vol_bound['input_vol'], plane=key).unsqueeze(0).to(self.device)
        pi_in['p_n'] = p_n
        
        # predict occupancy of the current crop
        with torch.no_grad():
            occ_cur = self.model.decode(pi_in, c, **kwargs).logits
        occ_hat = occ_cur.squeeze(0)
        
        return occ_hat

    def eval_points(self, p, c=None, c_img_all=None, vol_bound=None, **kwargs):
        ''' Evaluates the occupancy values for the points.

        Args:
            p (tensor): points 
            c (tensor): encoded feature volumes
        '''
        p_split = torch.split(p, self.points_batch_size)
        if c_img_all is not None:
            c_img = torch.split(c_img_all.squeeze(), self.points_batch_size)
            
        occ_hats = []
        for idx, pi in enumerate(p_split):
            
            if self.input_type == 'pointcloud_crop':
                if self.vol_bound is not None: # sliding-window manner
                    occ_hat = self.predict_crop_occ(pi, c, vol_bound=vol_bound, **kwargs)
                    occ_hats.append(occ_hat)
                else: # entire scene
                    pi_in = pi.unsqueeze(0).to(self.device)
                    pi_in = {'p': pi_in}
                    p_n = {}
                    for key in c.keys():
                        # normalized to the range of [0, 1]
                        p_n[key] = normalize_coord(pi.clone(), self.input_vol, plane=key).unsqueeze(0).to(self.device)
                    pi_in['p_n'] = p_n
                    with torch.no_grad():
                        occ_hat = self.model.decode(pi_in, c, **kwargs).logits
                    occ_hats.append(occ_hat.squeeze(0).detach().cpu())
            else:
                pi = pi.unsqueeze(0).to(self.device)
                
                if self.with_img:
                    c_img_i = c_img[idx]
                    c_img_i = c_img_i.unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        occ_hat = self.model.decode_img(pi, c, c_img_i, **kwargs).logits
                    
                else:
                    
                    with torch.no_grad():
                        occ_hat = self.model.decode(pi, c, **kwargs).logits
                occ_hats.append(occ_hat.squeeze(0).detach().cpu())
        
        occ_hat = torch.cat(occ_hats, dim=0)
        return occ_hat

    def extract_mesh(self, occ_hat, c=None, stats_dict=dict()):
        ''' Extracts the mesh from the predicted occupancy grid.

        Args:
            occ_hat (tensor): value grid of occupancies
            c (tensor): encoded feature volumes
            stats_dict (dict): stats dictionary
        '''
        # Some short hands
        n_x, n_y, n_z = occ_hat.shape
        box_size = 1 + self.padding
        threshold = np.log(self.threshold) - np.log(1. - self.threshold)
        # Make sure that mesh is watertight
        t0 = time.time()
        occ_hat_padded = np.pad(
            occ_hat, 1, 'constant', constant_values=-1e6)

        vertices, triangles = libmcubes.marching_cubes(
            occ_hat_padded, threshold)
        # vertices, triangles, normals_m, values = measure.marching_cubes_lewiner(occ_hat_padded)
        stats_dict['time (marching cubes)'] = time.time() - t0
        # Strange behaviour in libmcubes: vertices are shifted by 0.5
        vertices -= 0.5
        # # Undo padding
        vertices -= 1
        
        if self.vol_bound is not None:
            # Scale the mesh back to its original metric
            bb_min = self.vol_bound['query_vol'][:, 0].min(axis=0)
            bb_max = self.vol_bound['query_vol'][:, 1].max(axis=0)
            mc_unit = max(bb_max - bb_min) / (self.vol_bound['axis_n_crop'].max() * self.resolution0*2**self.upsampling_steps)
            vertices = vertices * mc_unit + bb_min
        else: 
            # Normalize to bounding box
            vertices /= np.array([n_x-1, n_y-1, n_z-1])
            vertices = box_size * (vertices - 0.5)
        
        # Estimate normals if needed
        if self.with_normals and not vertices.shape[0] == 0:
            t0 = time.time()
            normals = self.estimate_normals(vertices, c)
            stats_dict['time (normals)'] = time.time() - t0

        else:
            normals = None


        # Create mesh
        mesh = trimesh.Trimesh(vertices, triangles,
                               vertex_normals=normals,
                               process=False)
        


        # Directly return if mesh is empty
        if vertices.shape[0] == 0:
            return mesh

        # TODO: normals are lost here
        if self.simplify_nfaces is not None:
            t0 = time.time()
            mesh = simplify_mesh(mesh, self.simplify_nfaces, 5.)
            stats_dict['time (simplify)'] = time.time() - t0

        # Refine mesh
        if self.refinement_step > 0:
            t0 = time.time()
            self.refine_mesh(mesh, occ_hat, c)
            stats_dict['time (refine)'] = time.time() - t0

        return mesh

    def estimate_normals(self, vertices, c=None):
        ''' Estimates the normals by computing the gradient of the objective.

        Args:
            vertices (numpy array): vertices of the mesh
            c (tensor): encoded feature volumes
        '''
        device = self.device
        vertices = torch.FloatTensor(vertices)
        vertices_split = torch.split(vertices, self.points_batch_size)

        normals = []
        c = c.unsqueeze(0)
        for vi in vertices_split:
            vi = vi.unsqueeze(0).to(device)
            vi.requires_grad_()
            occ_hat = self.model.decode(vi, c).logits
            out = occ_hat.sum()
            out.backward()
            ni = -vi.grad
            ni = ni / torch.norm(ni, dim=-1, keepdim=True)
            ni = ni.squeeze(0).cpu().numpy()
            normals.append(ni)

        normals = np.concatenate(normals, axis=0)
        return normals

    def refine_mesh(self, mesh, occ_hat, c=None):
        ''' Refines the predicted mesh.

        Args:   
            mesh (trimesh object): predicted mesh
            occ_hat (tensor): predicted occupancy grid
            c (tensor): latent conditioned code c
        '''

        self.model.eval()

        # Some shorthands
        n_x, n_y, n_z = occ_hat.shape
        assert(n_x == n_y == n_z)
        # threshold = np.log(self.threshold) - np.log(1. - self.threshold)
        threshold = self.threshold

        # Vertex parameter
        v0 = torch.FloatTensor(mesh.vertices).to(self.device)
        v = torch.nn.Parameter(v0.clone())

        # Faces of mesh
        faces = torch.LongTensor(mesh.faces).to(self.device)

        # Start optimization
        optimizer = optim.RMSprop([v], lr=1e-4)

        for it_r in trange(self.refinement_step):
            optimizer.zero_grad()

            # Loss
            face_vertex = v[faces]
            eps = np.random.dirichlet((0.5, 0.5, 0.5), size=faces.shape[0])
            eps = torch.FloatTensor(eps).to(self.device)
            face_point = (face_vertex * eps[:, :, None]).sum(dim=1)

            face_v1 = face_vertex[:, 1, :] - face_vertex[:, 0, :]
            face_v2 = face_vertex[:, 2, :] - face_vertex[:, 1, :]
            face_normal = torch.cross(face_v1, face_v2)
            face_normal = face_normal / \
                (face_normal.norm(dim=1, keepdim=True) + 1e-10)
            face_value = torch.sigmoid(
                self.model.decode(face_point.unsqueeze(0), c).logits
            )
            normal_target = -autograd.grad(
                [face_value.sum()], [face_point], create_graph=True)[0]

            normal_target = \
                normal_target / \
                (normal_target.norm(dim=1, keepdim=True) + 1e-10)
            loss_target = (face_value - threshold).pow(2).mean()
            loss_normal = \
                (face_normal - normal_target).pow(2).sum(dim=1).mean()

            loss = loss_target + 0.01 * loss_normal

            # Update
            loss.backward()
            optimizer.step()

        mesh.vertices = v.data.cpu().numpy()

        return mesh