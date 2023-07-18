# import multiprocessing
import torch
from pykdtree.kdtree import KDTree
import numpy as np
import math
import pybullet as p
from scipy.spatial import distance
from scipy.optimize import linear_sum_assignment


def compute_iou(occ1, occ2, threshold):
    ''' Computes the Intersection over Union (IoU) value for two sets of
    occupancy values.

    Args:
        occ1 (tensor): first set of occupancy values
        occ2 (tensor): second set of occupancy values
    '''
    occ1 = np.asarray(occ1)
    occ2 = np.asarray(occ2)

    # Put all data in second dimension
    # Also works for 1-dimensional data
    if occ1.ndim >= 2:
        occ1 = occ1.reshape(occ1.shape[0], -1)
    if occ2.ndim >= 2:
        occ2 = occ2.reshape(occ2.shape[0], -1)

    # Convert to boolean values
    # occ1 = (occ1 >= threshold)
    # occ2 = (occ2 >= threshold)
    
    threshold = np.mean(occ2)
    occ1 = (occ1 >= threshold)
    occ2 = (occ2 >= threshold)

    # Compute IOU
    area_union = (occ1 | occ2).astype(np.float32).sum(axis=-1)
    area_intersect = (occ1 & occ2).astype(np.float32).sum(axis=-1)

    iou = (area_intersect / area_union)

    return iou

def EarthMoverDistance(points1, points2):
    d = distance.cdist(points1, points2)

    assignment = linear_sum_assignment(d)
    emd = d[assignment].sum() / len(d)

    return emd


def chamfer_distance(points1, points2, use_kdtree=True, give_id=False):
    ''' Returns the chamfer distance for the sets of points.

    Args:
        points1 (numpy array): first point set
        points2 (numpy array): second point set
        use_kdtree (bool): whether to use a kdtree
        give_id (bool): whether to return the IDs of nearest points
    '''
    if use_kdtree:
        return chamfer_distance_kdtree(points1, points2, give_id=give_id)
    else:
        return chamfer_distance_naive(points1, points2)


def chamfer_distance_naive(points1, points2):
    ''' Naive implementation of the Chamfer distance.

    Args:
        points1 (numpy array): first point set
        points2 (numpy array): second point set    
    '''
    if points2.size()[1] < 2048:
        points1 = points1[:, :points2.size()[1], :]

    assert(points1.size() == points2.size())
    batch_size, T, _ = points1.size()

    points1 = points1.view(batch_size, T, 1, 3)
    points2 = points2.view(batch_size, 1, T, 3)

    distances = (points1 - points2).pow(2).sum(-1)

    chamfer1 = distances.min(dim=1)[0].mean(dim=1)
    chamfer2 = distances.min(dim=2)[0].mean(dim=1)

    chamfer = chamfer1 + chamfer2
    return chamfer


def chamfer_distance_kdtree(points1, points2, give_id=False):
    ''' KD-tree based implementation of the Chamfer distance.

    Args:
        points1 (torch array): first point set
        points2 (torch array): second point set
        give_id (bool): whether to return the IDs of the nearest points
    '''
    # Points have size batch_size x T x 3
    batch_size = points1.size(0)

    # First convert points to numpy
    points1_np = points1.detach().cpu().numpy()
    points2_np = points2.detach().cpu().numpy()

    # Get list of nearest neighbors indieces
    idx_nn_12, _ = get_nearest_neighbors_indices_batch(points1_np, points2_np)
    idx_nn_12 = torch.LongTensor(idx_nn_12).to(points1.device)
    # Expands it as batch_size x 1 x 3
    idx_nn_12_expand = idx_nn_12.view(batch_size, -1, 1).expand_as(points1)

    # Get list of nearest neighbors indieces
    idx_nn_21, _ = get_nearest_neighbors_indices_batch(points2_np, points1_np)
    idx_nn_21 = torch.LongTensor(idx_nn_21).to(points1.device)
    # Expands it as batch_size x T x 3
    idx_nn_21_expand = idx_nn_21.view(batch_size, -1, 1).expand_as(points2)

    # Compute nearest neighbors in points2 to points in points1
    # points_12[i, j, k] = points2[i, idx_nn_12_expand[i, j, k], k]
    points_12 = torch.gather(points2, dim=1, index=idx_nn_12_expand)

    # Compute nearest neighbors in points1 to points in points2
    # points_21[i, j, k] = points2[i, idx_nn_21_expand[i, j, k], k]
    points_21 = torch.gather(points1, dim=1, index=idx_nn_21_expand)

    # Compute chamfer distance
    chamfer1 = (points1 - points_12).pow(2).sum(2).mean(1)
    chamfer2 = (points2 - points_21).pow(2).sum(2).mean(1)

    # Take sum
    chamfer = chamfer1 + chamfer2

    # If required, also return nearest neighbors
    if give_id:
        return chamfer1, chamfer2, idx_nn_12, idx_nn_21

    return chamfer

def hand_joint_error(joints_gt, joints_pred):
    ''' implementation of the hand_joint_error.

    Args:
        joints_gt (numpy array): gt of joints
        joints_pred (numpy array): predicted joints  
    '''
    j_gt = joints_gt.detach().cpu().squeeze().numpy()
    j_pred = joints_pred.detach().cpu().squeeze().numpy()
    
    joints_error = np.mean(np.linalg.norm(j_gt - j_pred, axis=1))
    
    return joints_error
    
    

def get_nearest_neighbors_indices_batch(points_src, points_tgt, k=1):
    ''' Returns the nearest neighbors for point sets batchwise.

    Args:
        points_src (numpy array): source points
        points_tgt (numpy array): target points
        k (int): number of nearest neighbors to return
    '''
    indices = []
    distances = []

    for (p1, p2) in zip(points_src, points_tgt):
        kdtree = KDTree(p2)
        dist, idx = kdtree.query(p1, k=k)
        indices.append(idx)
        distances.append(dist)

    return indices, distances


def make_3d_grid(bb_min, bb_max, shape):
    ''' Makes a 3D grid.

    Args:
        bb_min (tuple): bounding box minimum
        bb_max (tuple): bounding box maximum
        shape (tuple): output shape
    '''
    size = shape[0] * shape[1] * shape[2]

    pxs = torch.linspace(bb_min[0], bb_max[0], shape[0])
    pys = torch.linspace(bb_min[1], bb_max[1], shape[1])
    pzs = torch.linspace(bb_min[2], bb_max[2], shape[2])

    pxs = pxs.view(-1, 1, 1).expand(*shape).contiguous().view(size)
    pys = pys.view(1, -1, 1).expand(*shape).contiguous().view(size)
    pzs = pzs.view(1, 1, -1).expand(*shape).contiguous().view(size)
    p = torch.stack([pxs, pys, pzs], dim=1)

    return p


def transform_points(points, transform):
    ''' Transforms points with regard to passed camera information.

    Args:
        points (tensor): points tensor
        transform (tensor): transformation matrices
    '''
    assert(points.size(2) == 3)
    assert(transform.size(1) == 3)
    assert(points.size(0) == transform.size(0))

    if transform.size(2) == 4:
        R = transform[:, :, :3]
        t = transform[:, :, 3:]
        points_out = points @ R.transpose(1, 2) + t.transpose(1, 2)
    elif transform.size(2) == 3:
        K = transform
        points_out = points @ K.transpose(1, 2)

    return points_out


def b_inv(b_mat):
    ''' Performs batch matrix inversion.

    Arguments:
        b_mat: the batch of matrices that should be inverted
    '''

    eye = b_mat.new_ones(b_mat.size(-1)).diag().expand_as(b_mat)
    b_inv, _ = torch.gesv(eye, b_mat)
    return b_inv

def project_to_camera(points, transform):
    ''' Projects points to the camera plane.

    Args:
        points (tensor): points tensor
        transform (tensor): transformation matrices
    '''
    p_camera = transform_points(points, transform)
    p_camera = p_camera[..., :2] / p_camera[..., 2:]
    return p_camera


def fix_Rt_camera(Rt, loc, scale):
    ''' Fixes Rt camera matrix.

    Args:
        Rt (tensor): Rt camera matrix
        loc (tensor): location
        scale (float): scale
    '''
    # Rt is B x 3 x 4
    # loc is B x 3 and scale is B
    batch_size = Rt.size(0)
    R = Rt[:, :, :3]
    t = Rt[:, :, 3:]

    scale = scale.view(batch_size, 1, 1)
    R_new = R * scale
    t_new = t + R @ loc.unsqueeze(2)

    Rt_new = torch.cat([R_new, t_new], dim=2)

    assert(Rt_new.size() == (batch_size, 3, 4))
    return Rt_new

def normalize_coordinate(p, padding=0.1, plane='xz'):
    ''' Normalize coordinate to [0, 1] for unit cube experiments

    Args:
        p (tensor): point
        padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
        plane (str): plane feature type, ['xz', 'xy', 'yz']
    '''
    if plane == 'xz':
        xy = p[:, :, [0, 2]]
    elif plane =='xy':
        xy = p[:, :, [0, 1]]
    else:
        xy = p[:, :, [1, 2]]

    xy_new = xy / (1 + padding + 10e-6) # (-0.5, 0.5)
    xy_new = xy_new + 0.5 # range (0, 1)

    # f there are outliers out of the range
    if xy_new.max() >= 1:
        xy_new[xy_new >= 1] = 1 - 10e-6
    if xy_new.min() < 0:
        xy_new[xy_new < 0] = 0.0
    return xy_new

def normalize_3d_coordinate(p, padding=0.1):
    ''' Normalize coordinate to [0, 1] for unit cube experiments.
        Corresponds to our 3D model

    Args:
        p (tensor): point
        padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
    '''
    
    p_nor = p / (1 + padding + 10e-4) # (-0.5, 0.5)
    p_nor = p_nor + 0.5 # range (0, 1)
    # f there are outliers out of the range
    if p_nor.max() >= 1:
        p_nor[p_nor >= 1] = 1 - 10e-4
    if p_nor.min() < 0:
        p_nor[p_nor < 0] = 0.0
    return p_nor

def normalize_coord(p, vol_range, plane='xz'):
    ''' Normalize coordinate to [0, 1] for sliding-window experiments

    Args:
        p (tensor): point
        vol_range (numpy array): volume boundary
        plane (str): feature type, ['xz', 'xy', 'yz'] - canonical planes; ['grid'] - grid volume
    '''
    p[:, 0] = (p[:, 0] - vol_range[0][0]) / (vol_range[1][0] - vol_range[0][0])
    p[:, 1] = (p[:, 1] - vol_range[0][1]) / (vol_range[1][1] - vol_range[0][1])
    p[:, 2] = (p[:, 2] - vol_range[0][2]) / (vol_range[1][2] - vol_range[0][2])
    
    if plane == 'xz':
        x = p[:, [0, 2]]
    elif plane =='xy':
        x = p[:, [0, 1]]
    elif plane =='yz':
        x = p[:, [1, 2]]
    else:
        x = p    
    return x

def coordinate2index(x, reso, coord_type='2d'):
    ''' Normalize coordinate to [0, 1] for unit cube experiments.
        Corresponds to our 3D model

    Args:
        x (tensor): coordinate
        reso (int): defined resolution
        coord_type (str): coordinate type
    '''
    x = (x * reso).long()
    if coord_type == '2d': # plane
        index = x[:, :, 0] + reso * x[:, :, 1]
    elif coord_type == '3d': # grid
        index = x[:, :, 0] + reso * (x[:, :, 1] + reso * x[:, :, 2])
    index = index[:, None, :]
    return index

def coord2index(p, vol_range, reso=None, plane='xz'):
    ''' Normalize coordinate to [0, 1] for sliding-window experiments.
        Corresponds to our 3D model

    Args:
        p (tensor): points
        vol_range (numpy array): volume boundary
        reso (int): defined resolution
        plane (str): feature type, ['xz', 'xy', 'yz'] - canonical planes; ['grid'] - grid volume
    '''
    # normalize to [0, 1]
    x = normalize_coord(p, vol_range, plane=plane)
    
    if isinstance(x, np.ndarray):
        x = np.floor(x * reso).astype(int)
    else: #* pytorch tensor
        x = (x * reso).long()

    if x.shape[1] == 2:
        index = x[:, 0] + reso * x[:, 1]
        index[index > reso**2] = reso**2
    elif x.shape[1] == 3:
        index = x[:, 0] + reso * (x[:, 1] + reso * x[:, 2])
        index[index > reso**3] = reso**3
    
    return index[None]

def update_reso(reso, depth):
    ''' Update the defined resolution so that UNet can process.

    Args:
        reso (int): defined resolution
        depth (int): U-Net number of layers
    '''
    base = 2**(int(depth) - 1)
    if ~(reso / base).is_integer(): # when this is not integer, U-Net dimension error
        for i in range(base):
            if ((reso + i) / base).is_integer():
                reso = reso + i
                break    
    return reso

def decide_total_volume_range(query_vol_metric, recep_field, unit_size, unet_depth):
    ''' Update the defined resolution so that UNet can process.

    Args:
        query_vol_metric (numpy array): query volume size
        recep_field (int): defined the receptive field for U-Net
        unit_size (float): the defined voxel size
        unet_depth (int): U-Net number of layers
    '''
    reso = query_vol_metric / unit_size + recep_field - 1
    reso = update_reso(int(reso), unet_depth) # make sure input reso can be processed by UNet
    input_vol_metric = reso * unit_size
    p_c = np.array([0.0, 0.0, 0.0]).astype(np.float32)
    lb_input_vol, ub_input_vol = p_c - input_vol_metric/2, p_c + input_vol_metric/2
    lb_query_vol, ub_query_vol = p_c - query_vol_metric/2, p_c + query_vol_metric/2
    input_vol = [lb_input_vol, ub_input_vol]
    query_vol = [lb_query_vol, ub_query_vol]

    # handle the case when resolution is too large
    if reso > 10000:
        reso = 1
    
    return input_vol, query_vol, reso

def add_key(base, new, base_name, new_name, device=None):
    ''' Add new keys to the given input

    Args:
        base (tensor): inputs
        new (tensor): new info for the inputs
        base_name (str): name for the input
        new_name (str): name for the new info
        device (device): pytorch device
    '''
    if (new is not None) and (isinstance(new, dict)):
        if device is not None:
            for key in new.keys():
                new[key] = new[key].to(device)
        base = {base_name: base,
                new_name: new}
    return base

class map2local(object):
    ''' Add new keys to the given input

    Args:
        s (float): the defined voxel size
        pos_encoding (str): method for the positional encoding, linear|sin_cos
    '''
    def __init__(self, s, pos_encoding='linear'):
        super().__init__()
        self.s = s
        self.pe = positional_encoding(basis_function=pos_encoding)

    def __call__(self, p):
        p = torch.remainder(p, self.s) / self.s # always possitive
        # p = torch.fmod(p, self.s) / self.s # same sign as input p!
        p = self.pe(p)
        return p

class positional_encoding(object):
    ''' Positional Encoding (presented in NeRF)

    Args:
        basis_function (str): basis function
    '''
    def __init__(self, basis_function='sin_cos'):
        super().__init__()
        self.func = basis_function

        L = 10
        freq_bands = 2.**(np.linspace(0, L-1, L))
        self.freq_bands = freq_bands * math.pi

    def __call__(self, p):
        if self.func == 'sin_cos':
            out = []
            p = 2.0 * p - 1.0 # chagne to the range [-1, 1]
            for freq in self.freq_bands:
                out.append(torch.sin(freq * p))
                out.append(torch.cos(freq * p))
            p = torch.cat(out, dim=2)
        return p

class RFUniverseCamera:
    """Summary of Camera Class here.

        This class is a camera class.
        Github: https://github.com/ElectronicElephant/pybullet_ur5_robotiq/blob/main/utilities.py
                https://github.com/bulletphysics/bullet3/issues/1616

    Attributes:
        * @param near   Number Distance to the near clipping plane along the -Z axis
        * @param far    Number Distance to the far clipping plane along the -Z axis
    """
    def __init__(
            self,
            width,
            height,
            near_plane,
            far_plane,
            fov=90
    ):
        self.width, self.height = width, height
        self.aspect = self.width / self.height
        self.near, self.far = near_plane, far_plane
        self.fov = fov
        self.projection_matrix = p.computeProjectionMatrixFOV(self.fov, self.aspect, self.near, self.far)
        # pybullet requires the fov here to be in degrees.

        # rot_matrix = p.getMatrixFromQuaternion(cam_orn)
        # self.cam_rot_matrix = np.array(rot_matrix).reshape(3, 3)
        # Initial vectors
        # init_camera_vector = (1, 0, 0)  # z-axis
        # init_up_vector = (0, -1, 0)  # y-axis
        # Rotated vectors

        # self.camera_vector = self.cam_rot_matrix.dot(init_camera_vector)
        # self.up_vector = self.cam_rot_matrix.dot(init_up_vector)

        # self.view_matrix = p.computeViewMatrix(cam_pos, cam_pos + 0.1 * self.camera_vector, self.up_vector)

        # self._view_matrix = np.array(self.view_matrix).reshape((4, 4), order='F')
        self._projection_matrix = np.array(self.projection_matrix).reshape((4, 4), order='F')
        # order='F' means column first for pybullet magical configuration
        # self.tran_pix_world = np.linalg.inv(self._projection_matrix @ self._view_matrix)

        # https://stackoverflow.com/questions/60430958/understanding-the-view-and-projection-matrix-from-pybullet
        h = self.height
        # h = self.width
        self.f = h / (2 * math.tan(math.radians(self.fov / 2)))    # the equation need (fov / 2) in radius
        k = np.zeros((3, 3))
        self.intrinsic_matrix = np.array([[self.f, 0, self.width / 2],
                                          [0, self.f, self.height / 2],
                                          [0, 0, 1]])

    def depth_image_2_depth(self, depth_img: np.ndarray):
        """ Convert a 3-channel depth image to a 1-channel depth matrix
            Input:
                depth_img: numpy.ndarray
                    Depth image in shape (H, W, 3)
                    Height and width must match the initialization of this camera.

            Output:
                depth: numpy.ndarray
                    Converted depth matrix in shape (H, W)
        """
        assert depth_img.shape[0] == self.height and \
               depth_img.shape[1] == self.width and \
               depth_img.shape[2] == 3

        image_depth_out = (
                depth_img[:, :, 0]
                + depth_img[:, :, 1] / np.float32(256)
                + depth_img[:, :, 2] / np.float32(256 ** 2)
        )
        depth = image_depth_out * (self.far - self.near) / 255.0

        return self.far - depth
        # return self.near + depth

    def depth_2_camera_pointcloud(self, depth):
        """ 
        Generate point cloud using depth image only.
        author: GraspNet baseline

        Input:
            depth: [numpy.ndarray, (H,W), numpy.float32]
                depth image
            camera: [CameraInfo]
                camera intrinsics
            organized: bool
                whether to keep the cloud in image shape (H,W,3)

        Output:
            cloud: [numpy.ndarray, (H,W,3)/(H*W,3), numpy.float32]
                generated cloud, (H,W,3) for organized=True, (H*W,3) for organized=False
        """
        xmap = np.arange(self.width)
        ymap = np.arange(self.height)
        xmap, ymap = np.meshgrid(xmap, ymap)  # 0~999

        fx = fy = self.f
        cx = self.width / 2
        cy = self.height / 2
        points_z = depth
        points_x = (xmap - cx) * points_z / fx
        points_y = (ymap - cy) * points_z / fy
        # cloud = np.stack([points_x, points_y, points_z], axis=-1)
        cloud = np.stack([points_z, -points_x, -points_y], axis=-1)
        cloud = cloud.reshape([-1, 3])

        idx_none = np.where(cloud[:, 0] > self.far-0.0005)
        new_cloud = np.delete(cloud, idx_none, axis=0)

        # return cloud
        return new_cloud, cloud


def R_from_PYR(wrist_rot):
    roll, pitch, yaw = wrist_rot
    R_roll = np.array([[np.cos(roll), -np.sin(roll), 0],
                       [np.sin(roll), np.cos(roll), 0],
                       [0, 0, 1]])

    R_pitch = np.array([[1, 0, 0],
                        [0, np.cos(pitch), np.sin(pitch)],
                        [0, -np.sin(pitch), np.cos(pitch)]])

    R_yaw = np.array([[np.cos(yaw), 0, -np.sin(yaw)],
                      [0, 1, 0],
                      [np.sin(yaw), 0, np.cos(yaw)]])
    return R_pitch @ R_yaw @ R_roll

def norm_pc_1(pc, pc_obj):
    centroid = np.mean(pc_obj, axis=0)
    pc = pc - centroid
    pc_obj = pc_obj - centroid
    m = np.max(np.sqrt(np.sum(pc_obj ** 2, axis=1)))
    pc_normalized = pc / (2*m)
    return pc_normalized

def pc_cam_to_world(pc, rot, trans=[0, 0, 0]):
    # pc shape (n, 3)
    # extrinsic shape (4, 4)
    extrinsic = np.zeros((4, 4))
    extrinsic[:3, 3] = trans
    extrinsic[3, 3] = 1
    degree_x, degree_y, degree_z = rot
    rot_x = np.array([[np.cos(degree_x), 0, np.sin(degree_x)],
                      [0, 1, 0],
                      [-np.sin(degree_x), 0, np.cos(degree_x)]])
    
    rot_y = np.array([[np.cos(degree_y), -np.sin(degree_y), 0],
                      [np.sin(degree_y), np.cos(degree_y), 0],
                      [0, 0, 1]])

    rot_z = np.array([[0, 0, 1],
                      [np.cos(degree_z), np.sin(degree_z), 0],
                      [-np.sin(degree_z), np.cos(degree_z), 0]])

    extrinsic[:3, :3] = rot_z @ rot_x @ rot_y 

    extr_inv = np.linalg.inv(extrinsic)
    R = extr_inv[:3, :3]
    # T = extr_inv[:3, 3]
    # R = extrinsic[:3, :3]
    T = extrinsic[:3, 3]
    pc = (R @ pc.T).T + T
    return pc
