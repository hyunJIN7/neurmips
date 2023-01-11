import os
import numpy as np
import torch
from torch.utils.data import Dataset
from pytorch3d.renderer import PerspectiveCameras
from .utils_camera import get_ndc_grid
from .utils import get_image_tensors, get_depth_and_confidence, random_sample_points

"""
replica data format
"""
DEPTH_WIDTH = 256
DEPTH_HEIGHT = 192


def get_camera_intrinsic(path):
    with open(path, 'r') as file:
        lines = file.readlines()
        lines = lines[0].split(' ')  # fx 0 cx 0 fy cy 0 0 1
        W, H = DEPTH_WIDTH, DEPTH_HEIGHT  # int(lines[2]), int(lines[3])
        fx, fy = float(lines[0]), float(lines[4])
        px, py = float(lines[2]), float(lines[5])
    return [W, H, fx, fy, px, py]

class ScannerDataset(Dataset):
    def __init__(
            self,
            folder: str,
            sample_rate:float=0.1,
            read_points: bool = False,
            batch_points: int = 10000
            # read_points: bool = False,
            # sample_rate: float = 0.1,
            # batch_points: int = 10000
    ):
        pose_fname = os.path.join(folder, 'poses.txt')
        with open(pose_fname, "r") as f:
            cam_pose_lines = f.readlines()
        R, T = [], []
        # t1 = np.array([[1,0,0],[0,1,0],[0,0,-1]])  # 1
        # t1 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])  # 2
        t1 = np.array([[-1,0,0],[0,1,0],[0,0,-1]])  # 3
        # t1 = np.array([[1, 0, 0],
        #                [0, 0, 1],
        #                [0, -1, 0]])   #4
        for line in cam_pose_lines:
            line_data_list = line.split(' ')
            if len(line_data_list) == 0:
                continue
            line_data_list = np.array(line_data_list, dtype=float)
            pose_raw = np.reshape(line_data_list, (4, 4))
            r_raw = pose_raw[:3, :3] @ t1
            R.append(r_raw)
            T.append(pose_raw[:3, 3])
        R, T = torch.tensor(np.array(R), dtype=torch.float32), torch.tensor(np.array(T), dtype=torch.float32)
        self.R = R  # (n,3,3)
        self.T = T  # (n,3)

        intrinsic = get_camera_intrinsic(os.path.join(folder, 'intrinsic.txt'))
        self.intrinsic = intrinsic
        W, H, fx, fy, px, py = intrinsic
        self.focal_length = ((fx, fy),)
        self.principal_point = ((px, py),)

        #TODO : check  intrinsic : ndc to screen
        #
        # self.focal_length = ((focal_length, focal_length),)
        # self.principal_point = ((0, 0),)
        # intrinsic = [512, 512, focal_length, focal_length, 0, 0]
        # self.intrinsic = ndc_to_screen(intrinsic)  # W, H, fx, fy, px, py

        cameras = []
        for i in range(R.size(0)):
            cam = PerspectiveCameras(
                focal_length=self.focal_length,
                principal_point=self.principal_point,
                R=R[i][None],
                T=T[i][None]
            )
            cameras.append(cam)
        self.cameras = cameras

        images = get_image_tensors(os.path.join(folder, 'images'))
        self.images = images

        depth, confi = get_depth_and_confidence(folder) # (N, h, w)
        self.depths, self.confidences = depth, confi

        self.dense_points = None
        self.have_points = read_points
        self.batch_points = batch_points
        if read_points:
            points = []
            point_fname = os.path.join(folder, 'point_color.txt')  # X,Y,Z,R,G,B
            with open(point_fname, "r") as f:
                point_lines = f.readlines()
            for line in point_lines:
                line_data_list = line.split(',')
                if len(line_data_list) == 0:
                    continue
                line_data_list = np.array(line_data_list, dtype=float)
                points.append(line_data_list[:3])

            dense_points = torch.tensor(np.array(points), dtype=torch.float32)
            dense_points = random_sample_points(dense_points, sample_rate)
            self.dense_points = dense_points

    def get_camera_centers(self):
        R, T = self.R, self.T
        centers = torch.bmm(R, -T.unsqueeze(-1)).squeeze(-1)
        return centers

    def __len__(self):
        return len(self.cameras)

    def __getitem__(self, index):
        if self.have_points:
            dense_n = self.dense_points.size(0)
            sample_idx = torch.rand(self.batch_points)
            sample_idx = (sample_idx * dense_n).long()
            points = self.dense_points[sample_idx]
        else:
            points = torch.zeros(self.batch_points, 3)

        data = {
            'camera': self.cameras[index],
            'color': self.images[index],
            'depth': self.depths[index],
            'points': points,
            'confidence': self.confidences[index],
        }
        return data


def unproject_depth_points(depth, camera):
    '''
    Unproject depth points into world coordinates
    '''
    # print(camera.get_full_projection_transform().get_matrix())
    size = list(depth.size())
    ndc_grid = get_ndc_grid(size).to(depth.device)  # (h, w)
    ndc_grid[..., -1] = depth
    xy_depth = ndc_grid.view(1, -1, 3)
    points = camera.unproject_points(xy_depth)[0]
    return points


def dataset_to_depthpoints(dataset, point_num=None):
    '''
    Unproject all depth points within dataset into world coordinates
    Args
        dataset
    Return
        points: (point_num, 3)
    '''
    points_all = []
    for i in range(len(dataset)):
        data = dataset[i]
        depth = data['depth']
        camera = data['camera']
        points = unproject_depth_points(depth, camera)
        points_all.append(points)

    points = torch.cat(points_all, dim=0)
    if point_num is not None:
        sample_idx = torch.randperm(points.size(0))[:point_num]
        points = points[sample_idx]
    return points
