import os
import torch
from torch.nn import functional as F
from torch import distributions as dist
from src.common import (
    RFUniverseCamera, R_from_PYR, norm_pc_1, pc_cam_to_world, 
    chamfer_distance, compute_iou, make_3d_grid, add_key, hand_joint_error
)
from src.utils import visualize as vis
from src.training import BaseTrainer
import numpy as np
from scipy.spatial import distance
import igl
import time
import trimesh

depth_origin = np.loadtxt("./data/VTacO_mesh/depth_origin.txt")
w = 240
h = 320

class Trainer(BaseTrainer):
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

    def __init__(self, model, optimizer, device=None, input_type='pointcloud',
                 vis_dir=None, threshold=0.5, eval_sample=False, num_sample=2048, 
                 with_img=False, with_contact=False, train_tactile=False, 
                 encode_t2d=False, pretrained_t2d=True):
        self.model = model
        self.optimizer = optimizer
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
        self.pretrained_t2d = pretrained_t2d
        
        self.transformer = True
        self.sequence = True

        if vis_dir is not None and not os.path.exists(vis_dir):
            os.makedirs(vis_dir)

    def train_step(self, data, vf_dict):
        ''' Performs a training step.

        Args:
            data (dict): data dictionary
        '''
        self.model.train()
        self.optimizer.zero_grad()
        
        
        if self.transformer and not self.sequence:
            loss, loss_mano, loss_pc = self.compute_loss_transformer(data, vf_dict)
            return loss.item(), loss_mano.item(), loss_pc.item()
        elif self.transformer and self.sequence:
            loss, loss_mano, loss_pc = self.compute_loss_transformer_sequence(data, vf_dict)
            return loss.item(), loss_mano.item(), loss_pc.item()

        if not self.train_tactile:
            if not self.encode_t2d:
                if self.with_img:
                    loss, loss_mano, loss_pc = self.compute_loss_img(data)
                else:
                    loss, loss_mano, loss_pc = self.compute_loss(data)
                    
                loss.backward()
                self.optimizer.step()

                return loss.item(), loss_mano.item(), loss_pc.item()
            else:
                if self.with_img:
                    loss, loss_mano, loss_pc = self.compute_loss_t2d_img(data, vf_dict)
                else:
                    loss, loss_mano, loss_pc = self.compute_loss_t2d(data, vf_dict)
                
                loss.backward()
                self.optimizer.step()
                
                return loss.item(), loss_mano.item(), loss_pc.item()
        
        if self.train_tactile:
            loss, loss_depth, loss_digit = self.compute_loss_tactile(data)
            loss.backward()
            self.optimizer.step()
            
            if loss_digit is not None:
                return loss.item(), loss_depth.item(), loss_digit.item()
            else:
                return loss.item(), loss_depth.item(), None
            

    def eval_step(self, data, vf_dict):
        ''' Performs an evaluation step.

        Args:
            data (dict): data dictionary
        '''
        self.model.eval()

        device = self.device
        threshold = self.threshold
        eval_dict = {}

        data_name = data.get('points.name')
        points = data.get('points').to(device)
        occ = data.get('points.occ').to(device)

        inputs = data.get('inputs', torch.empty(points.size(0), 0)).to(device)
        pc_ply = data.get('inputs.pc_ply').to(device)
        imgs = data.get('inputs.img').to(device)
        depths = data.get('inputs.depth').to(device)
        
        touch_success = data.get('inputs.touch_success').to(device)
        mano_gt = data.get('points.mano').to(device)
        pc_hand = data.get('points.pc_hand').to(device)
        voxels_occ = data.get('voxels')
        
        wrist_rot_euler = data.get('points.wrist').to(device)
        wrist_rot_euler = wrist_rot_euler.squeeze().detach().cpu().numpy()
        
        wrist_pos = mano_gt.squeeze().detach().cpu().numpy()[:3]

        points_iou = data.get('points_iou').to(device)
        # occ_iou = data.get('points_iou.occ').to(device).cpu().numpy()
        
        batch_size = points.size(0)

        kwargs = {}
        
        # add pre-computed index
        inputs = add_key(inputs, data.get('inputs.ind'), 'points', 'index', device=device)
        # add pre-computed normalized coordinates
        points = add_key(points, data.get('points.normalized'), 'p', 'p_n', device=device)
        points_iou = add_key(points_iou, data.get('points_iou.normalized'), 'p', 'p_n', device=device)
        
        cam_pos = data.get('points.cam_pos').to(device).reshape(points.size(0), -1)
        cam_rot = data.get('points.cam_rot').to(device).reshape(points.size(0), -1)
        cam_info = torch.cat((cam_pos, cam_rot), dim=1)
                
        vertices_l = []
        faces_l = []
        for data_name_b in data_name:
            v, f = vf_dict[data_name_b]['v'], vf_dict[data_name_b]['f']
            vertices_l.append(v)
            faces_l.append(f)
        
        if self.transformer and not self.sequence:
            device = self.device
            p = data.get('points')
            inputs = data.get('inputs', torch.empty(p.size(0), 0)).to(device)
            
            occ, p, cam_pos, c, c_img = self.get_data_feature(data)
            
            kwargs = {}
            logits = self.model.decode_img(p, cam_pos, c, c_img, **kwargs).logits
            iou = compute_iou(occ, logits.cpu().numpy(), threshold)
            eval_dict['iou'] = iou[0]
            
        elif self.transformer and self.sequence:
            device = self.device
            p = data[2].get('points')
            inputs = data[2].get('inputs', torch.empty(p.size(0), 0)).to(device)
            
            _, p_front_global_2, p_local_front_2, c_front_2, c_img_front_2 = self.get_data_feature(data[-1])
            _, p_front_global_1, p_local_front_1, c_front_1, c_img_front_1 = self.get_data_feature(data[-2])
            occ, p, p_local, c, c_img = self.get_data_feature(data[2])
            _, p_back_global_1, p_local_back_1, c_back_1, c_img_back_1 = self.get_data_feature(data[1])
            _, p_back_global_2, p_local_back_2, c_back_2, c_img_back_2 = self.get_data_feature(data[0])

            p_front_global = p_front_global_1
            p_back_global = p_back_global_1
            
            c_front_1_fuse = self.model.decoder.fuse_pointcloud(p_front_global_1, p_local_front_1, c_front_1, c_img_front_1)
            c_front_2_fuse = self.model.decoder.fuse_pointcloud(p_front_global_2, p_local_front_2, c_front_2, c_img_front_2)
            c_front_fuse = self.model.decoder.fuse_f(p_front_global_1, p_front_global_2, c_front_1_fuse, c_front_2_fuse)
            
            c_back_1_fuse = self.model.decoder.fuse_pointcloud(p_back_global_1, p_local_back_1, c_back_1, c_img_back_1)
            c_back_2_fuse = self.model.decoder.fuse_pointcloud(p_back_global_2, p_local_back_2, c_back_2, c_img_back_2)
            c_back_fuse = self.model.decoder.fuse_f(p_back_global_1, p_back_global_2, c_back_1_fuse, c_back_2_fuse)

            c_fuse = self.model.decoder.fuse_f(p_front_global, p_back_global, c_front_fuse, c_back_fuse)
            c_fuse_now = self.model.decoder.fuse_pointcloud(p, p_local, c, c_img)
            
            kwargs = {}
            logits = self.model.decoder.forward_fuse(p, p_front_global, c_fuse_now, c_fuse, **kwargs)
            logits = dist.Bernoulli(logits=logits).logits
            iou = compute_iou(occ, logits.cpu().numpy(), threshold)
            eval_dict['iou'] = iou[0]
            
        if not self.train_tactile:
            with torch.no_grad():
                
                wrist_pos_zero = torch.zeros(1, 3).to(device)
                fea_m_full = torch.cat((wrist_pos_zero, mano_gt[:, 6:]), 1).to(device)
                hand_gt = self.model.encode_hand_mano(fea_m_full)
                joints_gt = hand_gt['mano_joints']
            
            if not self.encode_t2d:
            # Compute iou
                with torch.no_grad():
                    occ_iou = data.get('points_iou.occ').to(device).cpu().numpy()
                    c_hand = self.model.encode_hand_inputs(inputs)
                    pc_pred = c_hand['mano_verts']
                    joints_pred = c_hand['mano_joints']
                    
                    if self.with_img is False:
                        p_out = self.model(points_iou, inputs, imgs,
                                        sample=self.eval_sample, **kwargs)
                        
                    else:
                
                        p = points_iou
                        B, N, D = p.size()
                        
                        # if hand and object are separated
                        c = self.model.encode_inputs(inputs)
                        c_hand = self.model.encode_hand_inputs(inputs)

                        c_img = self.model.encode_img_inputs(imgs)
                        c_img_all = torch.zeros(B, N, c_img.size()[2], requires_grad=True).to(device)

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
                        
                        tips_pos_b = norm_pc_1(tips_pos_b, pc_ply.squeeze().detach().cpu().numpy())
                        
                        p_new = p.cpu().detach().numpy()
                        
                        # for batch in range(B):
                        batch = 0
                        p_new_b = p_new[batch]
                        # print(p_new_b.shape, tips_pos_b.shape)
                        dist_p_tips = distance.cdist(p_new_b, tips_pos_b)
                        
                        for finger in range(5):
                            # if touch successful, cat the local feature to the query points
                            if touch_success[batch, finger]:
                                tips_points_idx_1 = np.where((np.min(dist_p_tips, 1) < 0.05) & (np.argmin(dist_p_tips, 1) == finger))[0]

                                c_img_all[batch, tips_points_idx_1, :] = c_img[batch, finger, :]

                        kwargs = {}
                        # General points
                        p_out = self.model.decode_img(p, c, c_img_all, **kwargs)
                        

                    # occ_iou_np = (occ_iou >= 0.5).cpu().numpy()
                    # occ_iou_hat_np = (p_out.probs >= threshold).cpu().numpy()

                    # iou = compute_iou(occ_iou_np, occ_iou_hat_np, 0.5).mean()

                    # iou = compute_iou(occ_iou, p_out.logits.cpu().numpy(), threshold).mean()
                    # eval_dict['iou'] = iou
            
            else:
                with torch.no_grad():
                    p = data.get('points').to(device)
                    B, N, D = p.size()
                    num_sample = self.num_sample
                    
                    cam_pos = data.get('points.cam_pos').to(device).reshape(p.size(0), -1)
                    cam_rot = data.get('points.cam_rot').to(device).reshape(p.size(0), -1)
                    cam_info = torch.cat((cam_pos, cam_rot), dim=1)
                    
                    wrist_rot_euler = data.get('points.wrist').to(device)
                    wrist_rot_euler = wrist_rot_euler.squeeze().detach().cpu().numpy()
                    
                    cam_pos_d = cam_pos.cpu().detach().numpy().reshape(B, 5, 3)
                    cam_rot_d = cam_rot.cpu().detach().numpy().reshape(B, 5, 3)

                    


                    
                    width = w
                    height = h
                    near_plane = 0.019
                    far_plane = 0.022
                    fov = 60
                    cam_unity = RFUniverseCamera(width, height, near_plane, far_plane, fov)

                    pred_depth, c_hand = self.model.encode_t2d(inputs, imgs)
                    digit_param = c_hand['mano_param']
                    cam_p = digit_param[:, :15].reshape(B, 5, 3)
                    cam_r = digit_param[:, 15:].reshape(B, 5, 3)
                    
                    c = self.model.encode_inputs(inputs)
                    c_hand = self.model.encode_hand_inputs(inputs)
                    mano_param = c_hand['mano_param']
                    joints_pred = c_hand['mano_joints']
                    pc_pred = c_hand['mano_verts']
                    
                    pred_d_detach = pred_depth.cpu().detach().numpy()
                    p_sample = np.zeros((B, num_sample, 3))
                    occ_new = np.zeros((B, num_sample))
        
                    if self.with_img is False:
                        for batch in range(B):
                            p_b = p[batch].detach().cpu().numpy()
                            pc_world_l = []
                            for t_idx in range(5):
                                if touch_success[batch, t_idx]:
                                    depth = pred_d_detach[batch, t_idx].reshape(h, w)
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
                                    
                                    pc_world_all = pc_cam_to_world(pc_depth_new, rot=cam_rot_d[batch, t_idx]+[-np.pi/2, 0, np.pi/2], trans=cam_pos_d[batch, t_idx])
                                    pc_world_l.append(norm_pc_1(pc_world_all, pc_ply.squeeze().detach().cpu().numpy()))
                                                    
                            pc_world_l = np.array(pc_world_l, dtype=np.float32)
                            pc_world_l = pc_world_l.reshape(pc_world_l.shape[0]*pc_world_l.shape[1], -1)
                            p_indice = np.random.randint(p_b.shape[0], size=num_sample-pc_world_l.shape[0])
                            p_b_all = np.concatenate([pc_world_l, p_b[p_indice]])
                            

                            p_sample[batch] = p_b_all
                            occ_b = igl.fast_winding_number_for_meshes(vertices_l[batch], faces_l[batch], p_b_all)
                            occ_new[batch] = occ_b
                        
                        p_sample = torch.tensor(p_sample, dtype=torch.float32, requires_grad=True).to(device)
                        kwargs = {}
                        p_out = self.model.decode(p_sample, c, **kwargs)
                        occ_iou = occ_new
                        
                    else:
                        c_img = self.model.encode_img_inputs(imgs)
        
                        c_img_all = torch.ones(B, num_sample, c_img.size()[2], requires_grad=True).to(device)
            
                        
                        for batch in range(B):
                            idx_f_img = 0
                            p_b = p[batch].detach().cpu().numpy()
                            pc_world_l = []
                            for t_idx in range(5):
                                if touch_success[batch, t_idx]:
                                    depth = pred_d_detach[batch, t_idx].reshape(h, w)
                                    depth = depth*0.005 + 0.019
                                    depth = depths.squeeze().cpu().numpy()[t_idx].reshape(h, w)
                                    
                                    depth_diff = depth.reshape(w * h) - depth_origin
                                    idx_points = np.where(abs(depth_diff)>0.0001)
                                    if idx_points[0].shape[0] == 0: continue
                                    
                                    _, pc_depth_all = cam_unity.depth_2_camera_pointcloud(depth)
                                    pc_depth_new = pc_depth_all[idx_points]
                                    
                                    if pc_depth_new.shape[0] > 128:
                                        pc_world_indice = np.random.randint(pc_depth_new.shape[0], size=128)
                                        pc_depth_new = pc_depth_new[pc_world_indice]

                                    # pc_world = pc_cam_to_world(pc_depth, rot=cam_r[batch, t_idx]+[-90, 0, 90], trans=cam_p[batch, t_idx])
                                    pc_world_all = pc_cam_to_world(pc_depth_new, rot=cam_rot_d[batch, t_idx]+[-np.pi/2, 0, np.pi/2], trans=cam_pos_d[batch, t_idx])
                                    pc_world_l.append(norm_pc_1(pc_world_all, pc_ply.squeeze().detach().cpu().numpy()))
                                    
                                    c_img_all[batch, idx_f_img:idx_f_img + pc_depth_new.shape[0], :] = c_img[batch, t_idx, :]
                                    
                                    idx_f_img = pc_depth_new.shape[0] + idx_f_img
                                    # np.savetxt("/newssd1/home/zhenjun/conv_pcnet/out/inference/akb_grid_img_d_7/test_{}.txt".format(t_idx), pc_world_all)
                            
                            pc_world_final = []
                            for pc_world in pc_world_l:
                                pc_world_final += list(pc_world)
                            pc_world_l = np.array(pc_world_final, dtype=np.float32)
                            p_indice = np.random.randint(p_b.shape[0], size=num_sample-pc_world_l.shape[0])
                            p_b_all = np.concatenate([pc_world_l, p_b[p_indice]])

                            p_sample[batch] = p_b_all
                            occ_b = igl.fast_winding_number_for_meshes(vertices_l[batch], faces_l[batch], p_b_all)
                            occ_new[batch] = occ_b
                        
                        p_sample = torch.tensor(p_sample, dtype=torch.float32, requires_grad=True).to(device)
                        kwargs = {}
                        p_out = self.model.decode_img(p_sample, c, c_img_all, **kwargs)
                        # print(c_img_all.max(), c_img_all.min(), c_img_all.mean())

                        
                        occ_iou = occ_new

                    # iou = compute_iou(occ_new, p_out.logits.cpu().numpy(), threshold).mean()
                    # eval_dict['iou'] = iou
                    
                
            # Estimate voxel iou
            if voxels_occ is not None:
                voxels_occ = voxels_occ.to(device)
                points_voxels = make_3d_grid(
                    (-0.5 + 1/64,) * 3, (0.5 - 1/64,) * 3, voxels_occ.shape[1:])
                points_voxels = points_voxels.expand(
                    batch_size, *points_voxels.size())
                points_voxels = points_voxels.to(device)
                with torch.no_grad():
                    p_out = self.model(points_voxels, inputs,
                                       sample=self.eval_sample, **kwargs)

                voxels_occ_np = (voxels_occ >= 0.5).cpu().numpy()
                occ_hat_np = (p_out.probs >= threshold).cpu().numpy()
                iou_voxels = compute_iou(voxels_occ_np, occ_hat_np).mean()

                eval_dict['iou_voxels'] = iou_voxels
            
            
            # chamfer_dis = chamfer_distance(pc_hand.cpu(), pc_pred.cpu(), use_kdtree=False)
            # eval_dict['chamfer_distance'] = chamfer_dis
            
            # hand_joints_error = hand_joint_error(joints_gt.cpu(), joints_pred.cpu())
            # eval_dict['hand_joints_error'] = hand_joints_error
            
            # verts = pc_pred.squeeze().detach().cpu().numpy() - np.array([0.11, 0.005, 0], dtype=np.float32)
            # verts = np.dot(verts, np.linalg.inv(R_from_PYR(np.array([-np.pi/2, np.pi/2, 0]))).T)
            # verts = np.dot(verts, np.linalg.inv(R_from_PYR(np.array(wrist_rot_euler))).T)
            # verts = verts + wrist_pos
            
            # pc_pred_occ = norm_pc_1(verts, pc_ply.squeeze().detach().cpu().numpy())
            
            # occ_hand = igl.fast_winding_number_for_meshes(vertices_l[0], faces_l[0], pc_pred_occ.astype(np.float32))
            # verts_contain = pc_pred_occ[np.where(occ_hand>0.5)[0]]
            # if verts_contain.shape[0] == 0:
            #     max_depth = 0
            #     eval_dict['penetration_depth'] = max_depth
                
            # else:
            #     obj_mesh_temp = trimesh.Trimesh(vertices_l[0], faces_l[0])
            #     try:
            #         point_closest, pene_depth, _ = trimesh.proximity.closest_point(obj_mesh_temp, verts_contain)
            #         max_depth = pene_depth.max()*np.max(np.sqrt(np.sum(pc_ply.squeeze().detach().cpu().numpy() ** 2, axis=1)))
            #     except ValueError:
            #         max_depth = 0
            #     eval_dict['penetration_depth'] = max_depth
            
            iou = compute_iou(occ_iou, p_out.logits.cpu().numpy(), threshold)
            eval_dict['iou'] = iou[0]
        
        else:
            with torch.no_grad():
                
                
                depths = (depths-torch.min(depths))/(torch.max(depths)-torch.min(depths))
            
                pred_depth = self.model.encode_img_inputs(imgs)
                # pred_depth = torch.zeros(1, 5, 60000).to(device)
                # for f in range(5):
                #     pred_depth[:, f, :] = self.model.encode_img_inputs_test(imgs[:, f, :, :, :])
                
                loss_depth = F.l1_loss(pred_depth, depths)
                
                if self.model.encoder_hand is not None:
                    
                    c_hand = self.model.encode_hand_inputs(inputs)
                    digit_param = c_hand['mano_param']
                
                    loss_digit = F.mse_loss(digit_param, cam_info)
                
                    loss = loss_depth + loss_digit
                
                else:
                    loss = loss_depth
            
            eval_dict['loss'] = loss.item()
            eval_dict['loss_depth'] = loss_depth.item()
                
        return eval_dict

    def compute_loss(self, data):
        ''' Computes the loss.

        Args:
            data (dict): data dictionary
        '''
        device = self.device
        p = data.get('points').to(device)
        occ = data.get('points.occ').to(device)
        mano = data.get('points.mano').to(device)
        pc_hand = data.get('points.pc_hand').to(device)
        inputs = data.get('inputs', torch.empty(p.size(0), 0)).to(device)

        
        if 'pointcloud_crop' in data.keys():
            # add pre-computed index
            inputs = add_key(inputs, data.get('inputs.ind'), 'points', 'index', device=device)
            inputs['mask'] = data.get('inputs.mask').to(device)
            # add pre-computed normalized coordinates
            p = add_key(p, data.get('points.normalized'), 'p', 'p_n', device=device)

        # if hand and object are separated
        c = self.model.encode_inputs(inputs)
        c_hand = self.model.encode_hand_inputs(inputs)
        mano_param = c_hand['mano_param']
        mano_pc = c_hand['mano_verts']
        # c_hand is parameters of hand
        # loss on c_hand with manolayer
        
        # if we have tactile here
        # pose of tactile: p1 -> p
        # feature of tactile: c1 + c

        kwargs = {}
        # General points
        logits = self.model.decode(p, c, **kwargs).logits
        # loss_i = F.binary_cross_entropy_with_logits(logits, occ, reduction='none')
        
        loss_l1 = F.l1_loss(logits, occ)
        loss_mano = F.mse_loss(mano_param, mano)
        loss_pc = F.mse_loss(mano_pc, pc_hand)

        # loss = loss_i.sum(-1).mean() + loss_mano + loss_pc
        
        loss = loss_l1 + loss_mano + loss_pc

        return loss, loss_mano, loss_pc
    
    def compute_loss_img(self, data):
        ''' Computes the loss with tactile signals.

        Args:
            data (dict): data dictionary
        '''
        device = self.device
        p = data.get('points').to(device)
        occ = data.get('points.occ').to(device)
        mano = data.get('points.mano').to(device)
        pc_hand = data.get('points.pc_hand').to(device)
        
        inputs = data.get('inputs', torch.empty(p.size(0), 0)).to(device)
        pc_ply = data.get('inputs.pc_ply').to(device)
        imgs = data.get('inputs.img').to(device)
        touch_success = data.get('inputs.touch_success').to(device)
        
        wrist_rot_euler = data.get('points.wrist').to(device)
        wrist_rot_euler = wrist_rot_euler.detach().cpu().numpy()
        
        
        B, N, D = p.size()
        num_sample = self.num_sample

        if 'pointcloud_crop' in data.keys():
            # add pre-computed index
            inputs = add_key(inputs, data.get('inputs.ind'), 'points', 'index', device=device)
            inputs['mask'] = data.get('inputs.mask').to(device)
            # add pre-computed normalized coordinates
            p = add_key(p, data.get('points.normalized'), 'p', 'p_n', device=device)

        # if hand and object are separated
        c = self.model.encode_inputs(inputs)
        c_hand = self.model.encode_hand_inputs(inputs)

        c_img = self.model.encode_img_inputs(imgs)
        c_nocontact = torch.zeros(32, requires_grad=True).to(device)
        c_img_all = torch.zeros(B, num_sample, c_img.size()[2], requires_grad=True).to(device)

        mano_param = c_hand['mano_param']
        mano_pc = c_hand['mano_verts']
        mano_joints = c_hand['mano_joints']
        tips_idx = [4, 8, 12, 16, 20]
        tips_pos = mano_joints.cpu().detach().numpy()[:, tips_idx]


        for batch in range(B):
            wrist_pos, wrist_rotvec = mano_param[batch, :3].cpu().detach().numpy(), mano_param[batch, 3:6].cpu().detach().numpy()
            wrist_pos = mano.detach().cpu().numpy()[batch, :3]
            
            tips_pos_b = tips_pos[batch] - np.array([0.11, 0.005, 0], dtype=np.float32)
            tips_pos_b = np.linalg.inv(R_from_PYR(np.array([-np.pi/2, np.pi/2, 0]))) @ tips_pos_b.T
            tips_pos_b = np.linalg.inv(R_from_PYR(np.array(wrist_rot_euler[batch]))) @ tips_pos_b
            tips_pos_b = tips_pos_b.T + wrist_pos

            tips_pos_b = norm_pc_1(tips_pos_b, pc_ply.detach().cpu().numpy()[batch])
            
            tips_pos[batch] = tips_pos_b
        # np.savetxt("/newssd1/home/zhenjun/conv_pcnet/data/ShapeNet_test/02691156/003crackerbox/test_joints.txt", tips_pos_b)
        
        p_new = p.cpu().detach().numpy()
        
        num_sample_tips = []
        tips_points_idx_all = []
        for batch in range(B):
            num_sample_tips_b = 0
            tips_points_idx_b = []
            feature_start_cat = 0
            
            p_new_b, tips_pos_b = p_new[batch], tips_pos[batch]
            
            dist_p_tips = distance.cdist(p_new_b, tips_pos_b)
            
            for finger in range(5):
                # if touch successful, cat the local feature to the query points
                if touch_success[batch, finger]:
                    tips_points_idx = np.where((np.min(dist_p_tips, 1) < 0.05) & (np.argmin(dist_p_tips, 1) == finger))[0]
                    if tips_points_idx.shape[0] > 512:
                        tips_points_idx = tips_points_idx[np.random.choice(tips_points_idx.shape[0], 512)]

                    c_img_all[batch, feature_start_cat:feature_start_cat+len(tips_points_idx), :] = c_img[batch, finger, :]
                    
                    num_sample_tips_b += len(tips_points_idx)
                    tips_points_idx_b += list(tips_points_idx)
                    
                    feature_start_cat += len(tips_points_idx)

            num_sample_tips.append(num_sample_tips_b)
            tips_points_idx_all.append(tips_points_idx_b)
        
        
        p_sample = np.zeros((B, num_sample, 3))
        sample_all = np.arange(N)
        occ_new = torch.zeros((B, num_sample)).to(device)
        
        
        for batch in range(B):
            occ_new[batch, :len(tips_points_idx_all[batch])] = occ[batch, tips_points_idx_all[batch]]
            p_sample[batch, :len(tips_points_idx_all[batch]), :] = p_new[batch, tips_points_idx_all[batch], :]
            
            sample_rest = sample_all[~np.in1d(sample_all, tips_points_idx_all[batch])]

            indices_sample = np.random.randint(len(sample_rest), size=num_sample-num_sample_tips[batch])

            p_sample_rest = p_new[batch, indices_sample, :]
            p_sample[batch, num_sample_tips[batch]:, :] = p_sample_rest
            
            occ_new[batch, len(tips_points_idx_all[batch]):] = occ[batch, indices_sample]
            
            # np.savetxt("/newssd1/home/zhenjun/conv_pcnet/data/ShapeNet_test/02691156/003crackerbox/test_sample.txt", p_sample[0])
        
        
        p_sample = torch.tensor(p_sample, dtype=torch.float32, requires_grad=True).to(device)

        kwargs = {}
        logits = self.model.decode_img(p_sample, c, c_img_all, **kwargs).logits
    
    
        loss_l1 = F.l1_loss(logits, occ_new)
        loss_mano = F.mse_loss(mano_param, mano)
        loss_pc = F.mse_loss(mano_pc, pc_hand)
        
        loss = loss_l1 + loss_mano + loss_pc

        return loss, loss_mano, loss_pc


    def get_data_feature(self, data):
        device = self.device
        p = data.get('points')
        occ = data.get('points.occ')
        # mano = data.get('points.mano').to(device)
        # pc_hand = data.get('points.pc_hand').to(device)
        
        inputs = data.get('inputs', torch.empty(p.size(0), 0)).to(device)
        # pc_ply = data.get('inputs.pc_ply').to(device)
        imgs = data.get('inputs.img').to(device)
        # touch_success = data.get('inputs.touch_success').to(device)
        cam_pos = data.get('points.cam_pos').to(device).reshape(p.size(0), -1)
        cam_rot = data.get('points.cam_rot').to(device).reshape(p.size(0), -1)
        
        B, N, D = p.size()
        num_sample = self.num_sample
        random_indice = np.random.randint(0, N, size=(B, num_sample))
        p = p[np.arange(B)[:, None], random_indice]
        
        occ = occ[np.arange(B)[:, None], random_indice]

        # if hand and object are separated
        c = self.model.encode_inputs(inputs)

        c_img = self.model.encode_img_inputs(imgs)
        
        return occ.to(device), p.to(device), cam_pos, c, c_img
        
    def compute_loss_transformer(self, data, vf_dict):
        ''' Computes the loss with tactile signals.

        Args:
            data (dict): data dictionary
        '''
        device = self.device
        p = data.get('points')
        inputs = data.get('inputs', torch.empty(p.size(0), 0)).to(device)
        mano = data.get('points.mano').to(device)
        pc_hand = data.get('points.pc_hand').to(device)
        
        occ, p, cam_pos, c, c_img = self.get_data_feature(data)
        c_hand = self.model.encode_hand_inputs(inputs)
        
        mano_param = c_hand['mano_param']
        mano_pc = c_hand['mano_verts']
        mano_joints = c_hand['mano_joints']
        

        kwargs = {}
        logits = self.model.decode_img(p, cam_pos, c, c_img).logits
    
    
        loss_l1 = F.l1_loss(logits, occ)
        loss_mano = F.mse_loss(mano_param, mano)
        loss_pc = F.mse_loss(mano_pc, pc_hand)
        
        loss = loss_l1 + loss_mano + loss_pc

        return loss, loss_mano, loss_pc
    
    def compute_loss_transformer_sequence(self, data, vf_dict):
        device = self.device
        p = data[2].get('points')
        mano = data[2].get('points.mano').to(device)
        pc_hand = data[2].get('points.pc_hand').to(device)
        inputs = data[2].get('inputs', torch.empty(p.size(0), 0)).to(device)
        
        _, p_front_global_2, p_local_front_2, c_front_2, c_img_front_2 = self.get_data_feature(data[-1])
        _, p_front_global_1, p_local_front_1, c_front_1, c_img_front_1 = self.get_data_feature(data[-2])
        occ, p, p_local, c, c_img = self.get_data_feature(data[2])
        _, p_back_global_1, p_local_back_1, c_back_1, c_img_back_1 = self.get_data_feature(data[1])
        _, p_back_global_2, p_local_back_2, c_back_2, c_img_back_2 = self.get_data_feature(data[0])

        p_front_global = p_front_global_1
        p_back_global = p_back_global_1
        
        c_front_1_fuse = self.model.decoder.fuse_pointcloud(p_front_global_1, p_local_front_1, c_front_1, c_img_front_1)
        c_front_2_fuse = self.model.decoder.fuse_pointcloud(p_front_global_2, p_local_front_2, c_front_2, c_img_front_2)
        c_front_fuse = self.model.decoder.fuse_f(p_front_global_1, p_front_global_2, c_front_1_fuse, c_front_2_fuse)
        
        c_back_1_fuse = self.model.decoder.fuse_pointcloud(p_back_global_1, p_local_back_1, c_back_1, c_img_back_1)
        c_back_2_fuse = self.model.decoder.fuse_pointcloud(p_back_global_2, p_local_back_2, c_back_2, c_img_back_2)
        c_back_fuse = self.model.decoder.fuse_f(p_back_global_1, p_back_global_2, c_back_1_fuse, c_back_2_fuse)

        c_fuse = self.model.decoder.fuse_f(p_front_global, p_back_global, c_front_fuse, c_back_fuse)
        c_fuse_now = self.model.decoder.fuse_pointcloud(p, p_local, c, c_img)
        
        c_hand = self.model.encode_hand_inputs(inputs)
        mano_param = c_hand['mano_param']
        mano_pc = c_hand['mano_verts']
        
        kwargs = {}
        logits = self.model.decoder.forward_fuse(p, p_front_global, c_fuse_now, c_fuse)
        logits = dist.Bernoulli(logits=logits).logits
        
        loss_l1 = F.l1_loss(logits, occ)
        loss_mano = F.mse_loss(mano_param, mano)
        loss_pc = F.mse_loss(mano_pc, pc_hand)
        
        loss = loss_l1 + loss_mano + loss_pc

        return loss, loss_mano, loss_pc

        
    def compute_loss_t2d(self, data, vf_dict):
        device = self.device
        p = data.get('points').to(device)
        B, N, D = p.size()
        num_sample = self.num_sample
        
        
        occ = data.get('points.occ').to(device).detach().cpu().numpy()
        mano = data.get('points.mano').to(device)
        pc_hand = data.get('points.pc_hand').to(device)
        
        inputs = data.get('inputs', torch.empty(p.size(0), 0)).to(device)
        pc_ply = data.get('inputs.pc_ply').to(device).detach().cpu().numpy()
        imgs = data.get('inputs.img').to(device)
        
        depths = data.get('inputs.depth').to(device)
        depths = (depths-torch.min(depths))/(torch.max(depths)-torch.min(depths))
        
        touch_success = data.get('inputs.touch_success').to(device)
        cam_pos = data.get('points.cam_pos').to(device).reshape(p.size(0), -1)
        cam_rot = data.get('points.cam_rot').to(device).reshape(p.size(0), -1)
        cam_info = torch.cat((cam_pos, cam_rot), dim=1)
        
        wrist_rot_euler = data.get('points.wrist').to(device)
        wrist_rot_euler = wrist_rot_euler.detach().cpu().numpy()
        
        cam_pos_d = cam_pos.cpu().detach().numpy().reshape(B, 5, 3)
        cam_rot_d = cam_rot.cpu().detach().numpy().reshape(B, 5, 3)
        
        data_name = data.get('points.name')
        
        vertices_l = []
        faces_l = []

        for data_name_b in data_name:
            v, f = vf_dict[data_name_b]['v'], vf_dict[data_name_b]['f']
            vertices_l.append(v)
            faces_l.append(f)

        width = w
        height = h
        near_plane = 0.019
        far_plane = 0.022
        fov = 60
        cam_unity = RFUniverseCamera(width, height, near_plane, far_plane, fov)
        
        
        pred_depth, c_hand_d = self.model.encode_t2d(inputs, imgs)
        digit_param = c_hand_d['mano_param']
        cam_p = digit_param[:, :15].reshape(B, 5, 3)
        cam_r = digit_param[:, 15:].reshape(B, 5, 3)
        
        pred_d_detach = pred_depth.cpu().detach().numpy()
        
        p_sample = np.zeros((B, num_sample, 3))
        occ_new = np.zeros((B, num_sample))
        
        for batch in range(B):
            
            p_b = p[batch].detach().cpu().numpy()
            pc_world_l = []
            for t_idx in range(5):
                if touch_success[batch, t_idx]:
                    depth = pred_d_detach[batch, t_idx].reshape(h, w)
                    depth = depth*0.005 + 0.019
                    depth = depths[batch].cpu().numpy()[t_idx].reshape(h, w)
                    
                    depth_diff = depth.reshape(w * h) - depth_origin
                    idx_points = np.where(abs(depth_diff)>0.0001)
                    if idx_points[0].shape[0] == 0: continue
                    
                    _, pc_depth_all = cam_unity.depth_2_camera_pointcloud(depth)
                    pc_depth_new = pc_depth_all[idx_points]
                    
                    if pc_depth_new.shape[0] > 128:
                        pc_world_indice = np.random.randint(pc_depth_new.shape[0], size=128)
                        pc_depth_new = pc_depth_new[pc_world_indice]
                        
                    # pc_world = pc_cam_to_world(pc_depth, rot=cam_r[batch, t_idx]+[-90, 0, 90], trans=cam_p[batch, t_idx])
                    pc_world_all = pc_cam_to_world(pc_depth_new, rot=cam_rot_d[batch, t_idx]+[-np.pi/2, 0, np.pi/2], trans=cam_pos_d[batch, t_idx])
                    pc_world_l.append(norm_pc_1(pc_world_all, pc_ply[batch]))
                 
            pc_world_final = []
            for pc_world in pc_world_l:
                pc_world_final += list(pc_world)
            pc_world_l = np.array(pc_world_final, dtype=np.float32)
            p_indice = np.random.randint(p_b.shape[0], size=num_sample-pc_world_l.shape[0])
            p_b_all = np.concatenate([pc_world_l, p_b[p_indice]])
            

            p_sample[batch] = p_b_all
            # occ_b_pc = igl.fast_winding_number_for_meshes(vertices_l[batch], faces_l[batch], pc_world_l)
            # occ_b_sample = occ[batch][p_indice]
            # occ_b = np.hstack([occ_b_pc, occ_b_sample])
            
            occ_b = igl.fast_winding_number_for_meshes(vertices_l[batch], faces_l[batch], p_b_all)

            occ_new[batch] = occ_b
            # np.savetxt("/newssd1/home/zhenjun/conv_pcnet/out/inference/akb_3plane_img_1/test_{}.txt".format(batch), p_b_all)
            

        p_sample = torch.tensor(p_sample, dtype=torch.float32, requires_grad=True).to(device)
        occ_new = torch.tensor(occ_new, dtype=torch.float32, requires_grad=True).to(device)

        
        
        c = self.model.encode_inputs(inputs)
        c_hand = self.model.encode_hand_inputs(inputs)
        mano_param = c_hand['mano_param']
        mano_pc = c_hand['mano_verts']
        
        kwargs = {}
        logits = self.model.decode(p_sample, c, **kwargs).logits
        
        loss_l1 = F.l1_loss(logits, occ_new)
        loss_mano = F.mse_loss(mano_param, mano)
        loss_pc = F.mse_loss(mano_pc, pc_hand)

        
        loss = loss_l1 + loss_mano + loss_pc
        
        if not self.pretrained_t2d:
            loss_depth = F.l1_loss(pred_depth, depths)
            loss_digit = F.mse_loss(digit_param, cam_info)
            loss = loss + loss_depth + loss_digit

        return loss, loss_mano, loss_pc
        
        
    
    def compute_loss_t2d_img(self, data, vf_dict):
        device = self.device
        p = data.get('points').to(device)
        B, N, D = p.size()
        num_sample = self.num_sample
        
        
        occ = data.get('points.occ').to(device).detach().cpu().numpy()
        mano = data.get('points.mano').to(device)
        pc_hand = data.get('points.pc_hand').to(device)
        
        inputs = data.get('inputs', torch.empty(p.size(0), 0)).to(device)
        pc_ply = data.get('inputs.pc_ply').to(device).detach().cpu().numpy()
        imgs = data.get('inputs.img').to(device)
        
        depths = data.get('inputs.depth').to(device)
        
        
        touch_success = data.get('inputs.touch_success').to(device)
        cam_pos = data.get('points.cam_pos').to(device).reshape(p.size(0), -1)
        cam_rot = data.get('points.cam_rot').to(device).reshape(p.size(0), -1)
        cam_info = torch.cat((cam_pos, cam_rot), dim=1)
        
        cam_pos_d = cam_pos.cpu().detach().numpy().reshape(B, 5, 3)
        cam_rot_d = cam_rot.cpu().detach().numpy().reshape(B, 5, 3)
        
        data_name = data.get('points.name')
        
        vertices_l = []
        faces_l = []
        
        for data_name_b in data_name:
            v, f = vf_dict[data_name_b]['v'], vf_dict[data_name_b]['f']
            vertices_l.append(v)
            faces_l.append(f)

        
        width = w
        height = h
        near_plane = 0.019
        far_plane = 0.022
        fov = 60
        cam_unity = RFUniverseCamera(width, height, near_plane, far_plane, fov)

        
        pred_depth, c_hand = self.model.encode_t2d(inputs, imgs)
        digit_param = c_hand['mano_param']
        cam_p = digit_param[:, :15].reshape(B, 5, 3)
        cam_r = digit_param[:, 15:].reshape(B, 5, 3)
        
        pred_d_detach = pred_depth.cpu().detach().numpy()
        
        p_sample = np.zeros((B, num_sample, 3))
        occ_new = np.zeros((B, num_sample))
        
        c_img = self.model.encode_img_inputs(imgs)
        
        c_img_all = torch.ones(B, num_sample, c_img.size()[2], requires_grad=True).to(device)
        
        
        for batch in range(B):
            idx_f_img = 0
            p_b = p[batch].detach().cpu().numpy()
            pc_world_l = []
            for t_idx in range(5):
                if touch_success[batch, t_idx]:
                    depth = pred_d_detach[batch, t_idx].reshape(h, w)
                    depth = depth*0.005 + 0.019
                    depth = depths[batch].cpu().numpy()[t_idx].reshape(h, w)
                    depth_diff = depth.reshape(w * h) - depth_origin
                    idx_points = np.where(abs(depth_diff)>0.0001)
                    if idx_points[0].shape[0] == 0: continue
                    
                    _, pc_depth_all = cam_unity.depth_2_camera_pointcloud(depth)
                    pc_depth_new = pc_depth_all[idx_points]
                    
                    
                    if pc_depth_new.shape[0] > 128:
                        pc_world_indice = np.random.randint(pc_depth_new.shape[0], size=128)
                        pc_depth_new = pc_depth_new[pc_world_indice]
                        
                    pc_world_all = pc_cam_to_world(pc_depth_new, rot=cam_rot_d[batch, t_idx]+[-np.pi/2, 0, np.pi/2], trans=cam_pos_d[batch, t_idx])
                    pc_world_l.append(norm_pc_1(pc_world_all, pc_ply[batch]))
                    
                    c_img_all[batch, idx_f_img:idx_f_img + pc_depth_new.shape[0], :] = c_img[batch, t_idx, :]
                    idx_f_img = pc_depth_new.shape[0] + idx_f_img
                    
            
            pc_world_final = []
            for pc_world in pc_world_l:
                pc_world_final += list(pc_world)
            pc_world_l = np.array(pc_world_final, dtype=np.float32)

            p_indice = np.random.randint(p_b.shape[0], size=num_sample-pc_world_l.shape[0])
            p_b_all = np.concatenate([pc_world_l, p_b[p_indice]])
            # np.savetxt("/newssd1/home/zhenjun/conv_pcnet/out/inference/akb_grid_img_d_7/test_{}.txt".format(t_idx), p_b_all)

            p_sample[batch] = p_b_all

            # occ_b_pc = igl.fast_winding_number_for_meshes(vertices_l[batch], faces_l[batch], pc_world_l)
            # # print(occ_b_pc.max(), occ_b_pc.min(), occ_b_pc.mean())

            # occ_b_sample = occ[batch][p_indice]
            # occ_b = np.hstack([occ_b_pc, occ_b_sample])
            
            occ_b = igl.fast_winding_number_for_meshes(vertices_l[batch], faces_l[batch], p_b_all)

            
            # np.savetxt("/newssd1/home/zhenjun/conv_pcnet/out/inference/akb_3plane_img_1/test_{}.txt".format(batch), p_sample_b)
            occ_new[batch] = occ_b
        
        p_sample = torch.tensor(p_sample, dtype=torch.float32, requires_grad=True).to(device)
        occ_new = torch.tensor(occ_new, dtype=torch.float32, requires_grad=True).to(device)


        c = self.model.encode_inputs(inputs)
        c_hand = self.model.encode_hand_inputs(inputs)
        mano_param = c_hand['mano_param']
        mano_pc = c_hand['mano_verts']
        
        
        kwargs = {}
        logits = self.model.decode_img(p_sample, c, c_img_all, **kwargs).logits
        # print(c_img_all.max(), c_img_all.min(), c_img_all.mean())
        
        loss_l1 = F.l1_loss(logits, occ_new)
        loss_mano = F.mse_loss(mano_param, mano)
        loss_pc = F.mse_loss(mano_pc, pc_hand)

        loss = loss_l1 + loss_mano + loss_pc
        
        if not self.pretrained_t2d:
            depths = (depths-torch.min(depths))/(torch.max(depths)-torch.min(depths))
            loss_depth = F.l1_loss(pred_depth, depths)
            loss_digit = F.mse_loss(digit_param, cam_info)
            loss = loss + loss_depth + loss_digit
        
        return loss, loss_mano, loss_pc
        
    def compute_loss_contact(self, data):
        ''' Computes the loss with the contact result.

        Args:
            data (dict): data dictionary
        '''
        device = self.device
        p = data.get('points').to(device)
        occ = data.get('points.occ').to(device)
        contact = data.get('points.contact').to(device)
        mano = data.get('points.mano').to(device)
        pc_hand = data.get('points.pc_hand').to(device)
        inputs = data.get('inputs', torch.empty(p.size(0), 0)).to(device)

        
        if 'pointcloud_crop' in data.keys():
            # add pre-computed index
            inputs = add_key(inputs, data.get('inputs.ind'), 'points', 'index', device=device)
            inputs['mask'] = data.get('inputs.mask').to(device)
            # add pre-computed normalized coordinates
            p = add_key(p, data.get('points.normalized'), 'p', 'p_n', device=device)

        # if hand and object are separated
        c = self.model.encode_inputs(inputs)
        c_hand = self.model.encode_hand_inputs(inputs)
        mano_param = c_hand['mano_param']
        mano_pc = c_hand['mano_verts']
        # c_hand is parameters of hand
        # loss on c_hand with manolayer
        
        # if we have tactile here
        # pose of tactile: p1 -> p
        # feature of tactile: c1 + c

        kwargs = {}
        # General points
        p_r, pred_contact = self.model.decode_contact(p, c, **kwargs)
        # loss_i = F.binary_cross_entropy_with_logits(logits, occ, reduction='none')
        logits = p_r.logits
        
        
        loss_l1 = F.l1_loss(logits, occ)
        loss_contact = F.binary_cross_entropy_with_logits(pred_contact, contact, reduction='mean')
        
        loss_mano = F.mse_loss(mano_param, mano)
        loss_pc = F.mse_loss(mano_pc, pc_hand)

        # loss = loss_i.sum(-1).mean() + loss_mano + loss_pc
        
        loss = loss_contact + loss_l1 + loss_mano + loss_pc
        # loss = loss_l1 + loss_mano + loss_pc

        return loss, loss_mano, loss_pc, loss_contact

    def compute_loss_tactile(self, data):
        ''' Computes the loss from the light.

        Args:
            data (dict): data dictionary
        '''
        device = self.device
        p = data.get('points')
        inputs = data.get('inputs', torch.empty(p.size(0), 0)).to(device)
        imgs = data.get('inputs.img').to(device)
        depths = data.get('inputs.depth').to(device)
        cam_pos = data.get('points.cam_pos').to(device).reshape(p.size(0), -1)
        cam_rot = data.get('points.cam_rot').to(device).reshape(p.size(0), -1)
        cam_info = torch.cat((cam_pos, cam_rot), dim=1)
        
        depths = (depths-torch.min(depths))/(torch.max(depths)-torch.min(depths))
        
        B, N, D = p.size()
        pred_depth = self.model.encode_img_inputs(imgs)
        # pred_depth = torch.zeros(B, 5, 60000, requires_grad=True).to(device)
        # for f in range(5):
        #     pred_depth[:, f, :] = self.model.encode_img_inputs_test(imgs[:, f, :, :, :])
        
        loss_depth = F.l1_loss(pred_depth, depths)
        
        if self.model.encoder_hand is not None:
            c_hand = self.model.encode_hand_inputs(inputs)
            digit_param = c_hand['mano_param']
        
            loss_digit = F.mse_loss(digit_param, cam_info)
        
            loss = loss_depth + loss_digit
            return loss, loss_depth, loss_digit
        
        else:
            loss = loss_depth
            return loss, loss_depth, None
        
    
    def test_tactile(self, imgs):
        ''' Predict the depth from RGB image.

        Args:
            data (dict): data dictionary
        '''
        
        pred_depth_list = []
        for img in imgs:
            pred_depth = self.model.encode_img_inputs(img)
            pred_depth_list.append(pred_depth.cpu().detach().numpy())
            
        return pred_depth_list
