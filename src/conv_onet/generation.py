import torch
import numpy as np
import trimesh
from src.common import (
    make_3d_grid, normalize_coord, add_key, coord2index, 
    RFUniverseCamera, R_from_PYR, norm_pc_1, pc_cam_to_world,
    chamfer_distance, EarthMoverDistance, compute_iou
)
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
        height = h
        
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

            values = self.eval_points(pointsf, c, c_img_all, **kwargs).cpu().numpy()
            
            
        else:
            with torch.no_grad():
                c = self.model.encode_inputs(inputs)
            
            values = self.eval_points(pointsf, c, **kwargs).cpu().numpy()

            
        
        value_grid = values.reshape(nx, nx, nx)
        
        vertices, faces, normals, _ = measure.marching_cubes(value_grid, gradient_direction='ascent')
        vertices -= np.array([nx/2, nx/2, nx/2], dtype=np.float32)
        vertices *= 1.1/nx
        mesh = trimesh.Trimesh(vertices, faces)

        np.random.shuffle(vertices)
        vertices = vertices[:2048]
        
        vertices = np.ascontiguousarray(vertices, dtype=np.float32)
        
        print(type(points_obj), points_obj.shape)
        cd = chamfer_distance(points_obj.to(self.device), torch.FloatTensor([vertices]).to(self.device), use_kdtree=False)
        emd = EarthMoverDistance(np.array(points_obj[0]), vertices)
        
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
                
                    pc_world_all = pc_cam_to_world(pc_depth_all, rot=cam_rot[batch, t_idx, :]+[-np.pi/2, 0, np.pi/2], trans=cam_pos[batch, t_idx, :])
                    pc_world_l[batch, t_idx] = norm_pc_1(pc_world_all, pc_ply[batch].detach().cpu().numpy())
        
            return pc_world_l, data_name


        

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