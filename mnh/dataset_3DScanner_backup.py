import os
import argparse
import vedo
import numpy as np
import torch
from torch.utils.data import Dataset
from pytorch3d.renderer import PerspectiveCameras
from .utils import random_sample_points, get_image_tensors, get_depth_and_confidence

"""
tat data format
"""
DEPTH_WIDTH = 256
DEPTH_HEIGHT = 192


def get_camera_intrinsic(path):
    # with open(pose_fname, "r") as f:
    #     cam_pose_lines = f.readlines()
    # R, T = [], []
    # for line in cam_pose_lines:
    #     line_data_list = line.split(' ')

    with open(path, 'r') as file:
        lines = file.readlines()
        lines = lines[0].split(' ')  # fx 0 cx 0 fy cy 0 0 1
        W, H = DEPTH_WIDTH, DEPTH_HEIGHT  # int(lines[2]), int(lines[3])
        fx, fy = float(lines[0]), float(lines[4])
        px, py = float(lines[2]), float(lines[5])
    return [W, H, fx, fy, px, py]


def screen_to_ndc(intrinsic):
    W, H, fx, fy, px, py = intrinsic
    # convert from screen space to NDC space
    half_w, half_h = W / 2, H / 2
    fx_new = fx / half_w
    fy_new = fy / half_h
    px_new = -(px - half_w) / half_w
    py_new = -(py - half_h) / half_h
    return [W, H, fx_new, fy_new, px_new, py_new]


class ScannerDataset(Dataset):
    def __init__(
            self,
            folder: str,
            read_points: bool = False,
            sample_rate: float = 0.1,
            batch_points: int = 10000
    ):

        pose_fname = os.path.join(folder, 'poses.txt')
        with open(pose_fname, "r") as f:
            cam_pose_lines = f.readlines()
        R, T = [], []
        # t1 = np.array([[1,0,0],[0,1,0],[0,0,-1]])  # 1
        t1 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])  # 2
        # t1 = np.array([[-1,0,0],[0,1,0],[0,0,-1]])  # 3
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

        # R = torch.from_numpy(np.array(R)).float()
        # T = torch.from_numpy(np.array(T)).float()
        R, T = torch.tensor(np.array(R), dtype=torch.float32), torch.tensor(np.array(T), dtype=torch.float32)
        self.R = R  # (n,3,3)
        self.T = T  # (n,3)

        intrinsic = get_camera_intrinsic(os.path.join(folder, 'intrinsic.txt'))
        self.intrinsic = intrinsic

        # convert to NDC coordinates
        intrinsic = screen_to_ndc(intrinsic)
        W, H, fx, fy, px, py = intrinsic

        self.focal_length = ((fx, fy),)
        self.principal_point = ((px, py),)
        cameras = []
        for i in range(R.size(0)):
            cam = PerspectiveCameras(
                focal_length=((fx, fy),),
                principal_point=((px, py),),
                R=R[i][None],
                T=T[i][None],
                # image_size=((H,W),),
                # in_ndc=True
            )
            cameras.append(cam)
        self.cameras = cameras

        images = get_image_tensors(os.path.join(folder, 'images'))
        self.images = images

        depth, confi = get_depth_and_confidence(folder)
        self.depth, self.confidence = depth, confi

        self.sparse_points = None
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
            'points': points,
            'depth': self.depth[index],
            'confidence': self.confidence[index],
        }
        return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-folder')
    args = parser.parse_args()

    dataset = ScannerDataset(args.folder, read_points=True, sample_rate=0.2)
    data = dataset[0]
    points = data['points']
    points_dense = dataset.dense_points
    print('dense points: {}'.format(points_dense.size()))
    points = vedo.Points(points_dense)
    vedo.show(points, axes=1)


if __name__ == '__main__':
    main()
