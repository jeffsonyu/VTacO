import os
from tqdm import trange
import torch
from torch.nn import functional as F
from torch import distributions as dist
from src.common import (
    RFUniverseCamera, R_from_PYR, norm_pc_1, pc_cam_to_world, 
    chamfer_distance, compute_iou, make_3d_grid, add_key
)
from src.utils import visualize as vis
from src.inferencing import BaseInference
import numpy as np
from scipy.spatial import distance
from skimage import measure
import trimesh


depth_origin = np.loadtxt("./data/VTacO_mesh/depth_origin.txt")
w = 240
h = 320

class Inferencer(BaseInference):
    ''' Trainer object for the Occupancy Network.

    Args:
        model (nn.Module): Occupancy Network model
        optimizer (optimizer): pytorch optimizer object
        device (device): pytorch device
        input_type (str): input type
        vis_dir (str): visualization directory
        threshold (float): threshold value
        eval_sample (bool): whether to evaluate samples

    '''

    def __init__(self, model, optimizer, generator, device=None, input_type='pointcloud',
                 vis_dir=None, threshold=0.5, eval_sample=False, num_sample=2048, 
                 with_img=False, with_contact=False, train_tactile=False, encode_t2d=False):
        self.model = model
        self.optimizer = optimizer
        self.generator = generator
        self.device = device
        self.input_type = input_type
        self.vis_dir = vis_dir
        self.threshold = threshold
        self.eval_sample = eval_sample
        self.num_sample = num_sample
        self.with_img = with_img
        self.with_contact = with_contact
        self.train_tactile = train_tactile
        self.encode_t2d = encode_t2d
        
        self.resolution0 = self.generator.resolution0
        self.padding = self.generator.padding

        if vis_dir is not None and not os.path.exists(vis_dir):
            os.makedirs(vis_dir)

    def inference_step(self, data_vis_class):
        ''' Performs a training step.

        Args:
            data (dict): data dictionary
        '''
        self.model.eval()
        
        if not self.encode_t2d:
            if self.with_img:
                mesh_list_obj, mesh_list_hand = self.inference_img(data_vis_class)

            return mesh_list_obj, mesh_list_hand
        
        else:
            if self.with_img:
                mesh_list_obj, mesh_list_hand = self.inference_img_t2d(data_vis_class)
                
            return mesh_list_obj, mesh_list_hand

    def inference(self, data_vis_class):
        return None
            
    def inference_img(self, data_vis_class):
        ''' Inference one object.

        Args:
            data (dict): data dictionary
        '''
        self.model.eval()
        device = self.device
    
        box_size = 1 + self.padding
        nx = self.resolution0 * 4
        
        pointsf = box_size * make_3d_grid(
            (-0.5,)*3, (0.5,)*3, (nx,)*3
        )
        kwargs = {}
        
        mesh_list_obj = []
        mesh_list_hand = []

        for data_idx, data_vis in enumerate(data_vis_class):
            # print(data_vis['name'], data_vis['touch_id'])
            data = data_vis['data']
            inputs = data.get('inputs', torch.empty(1, 0)).to(device)
            inputs = add_key(inputs, data.get('inputs.ind'), 'points', 'index', device=device)
            imgs = data.get('inputs.img').to(device)
            touch_success = data.get('inputs.touch_success').to(device)
            pc_ply = data.get('inputs.pc_ply').to(device)
            mano_gt = data.get('points.mano').to(device)
            kwargs = {}
            
            wrist_rot_euler = data.get('points.wrist').to(device)
            wrist_rot_euler = wrist_rot_euler.squeeze().detach().cpu().numpy()
            
            wrist_pos = mano_gt.squeeze().detach().cpu().numpy()[:3]
            tips_idx = [4, 8, 12, 16, 20]
            
            with torch.no_grad():
                p = pointsf
                B = 1
                N, D = p.size()
                c_hand = self.model.encode_hand_inputs(inputs)

                c_img = self.model.encode_img_inputs(imgs)
                mano_param = c_hand['mano_param'].cpu().detach().numpy().squeeze()
                verts = c_hand['mano_verts'].cpu().detach().numpy().squeeze()
                faces = c_hand['mano_faces'].cpu().detach().numpy().squeeze()
                mano_joints = c_hand['mano_joints'].cpu().detach().numpy()
                # wrist_pos, wrist_rotvec = mano_param[0, :3].cpu().detach().numpy(), mano_param[0, 3:6].cpu().detach().numpy()
                
                verts = verts - np.array([0.11, 0.005, 0], dtype=np.float32)
                verts = np.linalg.inv(R_from_PYR(np.array([-np.pi/2, np.pi/2, 0]))) @ verts.T
                verts = np.linalg.inv(R_from_PYR(np.array(wrist_rot_euler))) @ verts
                verts = verts.T + wrist_pos
                verts = norm_pc_1(verts, pc_ply.squeeze().detach().cpu().numpy())
                
                mesh_hand = trimesh.Trimesh(verts, faces)
                mesh_list_hand.append(mesh_hand)
        
                tips_pos = mano_joints[0, tips_idx]
                tips_pos = tips_pos - np.array([0.11, 0.005, 0], dtype=np.float32)
                tips_pos = np.linalg.inv(R_from_PYR(np.array([-np.pi/2, np.pi/2, 0]))) @ tips_pos.T
                tips_pos = np.linalg.inv(R_from_PYR(np.array(wrist_rot_euler))) @ tips_pos
                tips_pos_b = tips_pos.T + wrist_pos
                
                tips_pos_b = norm_pc_1(tips_pos_b, pc_ply.squeeze().detach().cpu().numpy())
                
                
                p_new = p.cpu().detach().numpy()
                    
                p_new_b = p_new
                dist_p_tips = distance.cdist(p_new_b, tips_pos_b)
                
                if data_idx == 0:
                    
                    # if hand and object are separated
                    c = self.model.encode_inputs(inputs)
                    
                    c_img_all = torch.zeros(B, N, c_img.size()[2], requires_grad=True).to(device)
                    
                for finger in range(5):
                    # if touch successful, cat the local feature to the query points
                    if touch_success[0, finger]:
                        tips_points_idx_1 = np.where((np.min(dist_p_tips, 1) < 0.05) & (np.argmin(dist_p_tips, 1) == finger))[0]
                        # np.savetxt("/newssd1/home/zhenjun/conv_pcnet/out/inference/akb_3plane_img_1/test_{}_{}.txt".format(data_idx, finger), pointsf[tips_points_idx_1])
                        c_img_all[0, tips_points_idx_1, :] = c_img[0, finger, :]

                    
                values = self.generator.eval_points(pointsf, c, c_img_all, **kwargs).cpu().numpy()

                value_grid = values.reshape(nx, nx, nx)
                
                vertices, obj_faces, normals, _ = measure.marching_cubes_lewiner(value_grid,
                                                                                #  level = self.threshold,
                                                                                 gradient_direction='ascent')
                vertices -= np.array([nx/2, nx/2, nx/2], dtype=np.float32)
                vertices *= 1.1/nx
                mesh = trimesh.Trimesh(vertices, obj_faces)
                
                mesh_list_obj.append(mesh)

        return mesh_list_obj, mesh_list_hand
    
    def inference_img_t2d(self, data_vis_class):
        ''' Inference one object.

        Args:
            data (dict): data dictionary
        '''
        self.model.eval()
        device = self.device
    
        box_size = 1 + self.padding
        nx = self.resolution0 * 4
        
        pointsf = box_size * make_3d_grid(
            (-0.5,)*3, (0.5,)*3, (nx,)*3
        )
        kwargs = {}
        
        mesh_list_obj = []
        mesh_list_hand = []

        for data_idx, data_vis in enumerate(data_vis_class):
            # print(data_vis['name'], data_vis['touch_id'])
            data = data_vis['data']
            inputs = data.get('inputs', torch.empty(1, 0)).to(device)
            inputs = add_key(inputs, data.get('inputs.ind'), 'points', 'index', device=device)
            imgs = data.get('inputs.img').to(device)
            depths = data.get('inputs.depth').to(device)
            touch_success = data.get('inputs.touch_success').to(device)
            pc_ply = data.get('inputs.pc_ply').to(device).squeeze().detach().cpu().numpy()
            mano_gt = data.get('points.mano').to(device)
            kwargs = {}
            
            wrist_rot_euler = data.get('points.wrist').to(device)
            wrist_rot_euler = wrist_rot_euler.squeeze().detach().cpu().numpy()
            
            wrist_pos = mano_gt.squeeze().detach().cpu().numpy()[:3]
            tips_idx = [4, 8, 12, 16, 20]
            
            cam_pos = data.get('points.cam_pos').to(device).reshape(1, -1)
            cam_rot = data.get('points.cam_rot').to(device).reshape(1, -1)
            cam_info = torch.cat((cam_pos, cam_rot), dim=1)
            
            cam_pos_d = cam_pos.cpu().detach().numpy().reshape(1, 5, 3)
            cam_rot_d = cam_rot.cpu().detach().numpy().reshape(1, 5, 3)
            
            width = w
            height = h
            near_plane = 0.017
            far_plane = 0.022
            fov = 60
            cam_unity = RFUniverseCamera(width, height, near_plane, far_plane, fov)
            
            with torch.no_grad():
                p = pointsf
                B = 1
                N, D = p.size()
                
                pred_depth, c_hand = self.model.encode_t2d(inputs, imgs)
                pred_d_detach = pred_depth.cpu().detach().numpy()
                
                digit_param = c_hand['mano_param']
                cam_p = digit_param[:, :15].reshape(B, 5, 3)
                cam_r = digit_param[:, 15:].reshape(B, 5, 3)

                c_hand = self.model.encode_hand_inputs(inputs)
                
                mano_param = c_hand['mano_param'].cpu().detach().numpy().squeeze()
                verts = c_hand['mano_verts'].cpu().detach().numpy().squeeze()
                faces = c_hand['mano_faces'].cpu().detach().numpy().squeeze()
                mano_joints = c_hand['mano_joints'].cpu().detach().numpy()
                # wrist_pos, wrist_rotvec = mano_param[0, :3].cpu().detach().numpy(), mano_param[0, 3:6].cpu().detach().numpy()
                
                verts = verts - np.array([0.11, 0.005, 0], dtype=np.float32)
                verts = np.linalg.inv(R_from_PYR(np.array([-np.pi/2, np.pi/2, 0]))) @ verts.T
                verts = np.linalg.inv(R_from_PYR(np.array(wrist_rot_euler))) @ verts
                verts = verts.T + wrist_pos
                verts = norm_pc_1(verts, pc_ply)
                
                mesh_hand = trimesh.Trimesh(verts, faces)
                mesh_list_hand.append(mesh_hand)
                
                
                c_img = self.model.encode_img_inputs(imgs)
                
                p_new = p.cpu().detach().numpy()
                    
                p_new_b = p_new

                
                if data_idx == 0:
                    
                    # if hand and object are separated
                    c = self.model.encode_inputs(inputs)
                    
                    c_img_all = torch.zeros(B, N, c_img.size()[2], requires_grad=True).to(device)
                    
                for t_idx in range(5):
                    # if touch successful, cat the local feature to the query points
                    if touch_success[0, t_idx]:
                        depth = pred_d_detach[0, t_idx].reshape(h, w)
                        depth = depth*0.005 + 0.017
                        depth = depths.squeeze().cpu().numpy()[t_idx].reshape(h, w)
                                
                        depth_diff = depth.reshape(60000) - depth_origin
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

                values = self.generator.eval_points(pointsf, c, c_img_all, **kwargs).cpu().numpy()
                value_grid = values.reshape(nx, nx, nx)
                
                vertices, obj_faces, normals, _ = measure.marching_cubes_lewiner(value_grid,
                                                                                #  level = self.threshold,
                                                                                 gradient_direction='ascent')
                vertices -= np.array([nx/2, nx/2, nx/2], dtype=np.float32)
                vertices *= 1.1/nx
                mesh = trimesh.Trimesh(vertices, obj_faces)
                
                mesh_list_obj.append(mesh)

        return mesh_list_obj, mesh_list_hand